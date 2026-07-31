#!/usr/bin/env python3
"""
line_following.py — DonkeyCar Track 2, Mission 1: follow the yellow hashed line.


Written from scratch. Replaces donkeycar/parts/line_follower.py, which fails on
this track for two reasons:


 1. Its HSV threshold (H 0-50, S>50, V>50) matches the warm night lighting:
    the concrete, orange cones, and wall reflections all read as "yellow".
    Worse, HSV saturation of the tape swings from ~160 (night, warm lights)
    to ~87 (daylight) — no fixed HSV range covers both. This detector works
    in CIELAB instead, and (since the 2026-07-23 rework) primarily on LOCAL
    CONTRAST in the b-channel rather than absolute values: measured on this
    track, the tape reads b~161 in sun but only b~141 in building shadow,
    while the pavement under it reads 129 and 120 respectively. No absolute
    threshold covers both, but the tape is always +12..+30 yellower than
    the pavement immediately around it, in every lighting. So the mask is
    "b exceeds the local (median-blurred) background by LF_B_CONTRAST",
    OR'd with the old absolute test (b>=143) which is kept for night duty
    where contrast is lower (night tape b 145-149 vs concrete <=139).
 2. It samples ONE thin horizontal slice of the image. The center line is
    DASHED, so the slice regularly lands in a gap between dashes, detection
    drops out, and the car drifts off with stale steering.


This detector instead scans a tall region (the bottom part of the frame) split
into horizontal bands, finds the line centroid in each band, and fits a line
through them. That gives both lateral offset AND line heading, works across
dash gaps, and falls back to the WHITE boundary lines when the yellow is lost:
the white lines are solid (no dash gaps), so while the yellow is tracked we
learn the yellow-to-white distance on each side, and on a yellow dropout we
keep steering from the white lines instead of coasting blind. Only if neither
yellow nor white is usable does the car coast on last-known steering and then
stop.


--- HOW TO RUN ON THE CAR (drop-in, no framework changes) -------------------
1. Copy this file into your car directory (e.g. ~/mycar/line_following.py).
2. In ~/mycar/myconfig.py set:
      CV_CONTROLLER_MODULE = "line_following"
      CV_CONTROLLER_CLASS  = "LineFollower"
3. Drive as usual:  python manage.py drive   (cv_control template)
  Switch the web UI to "Full Auto (a)" or "Auto Steer (s)" mode to engage
  the follower (there is no "Local Pilot" mode in this UI).


--- HOW TO TEST OFF THE CAR (no hardware needed) ----------------------------
  python line_following.py test <image_or_video_or_folder> [--out overlay_dir]
This runs the exact same pipeline on saved frames and writes overlay images
showing the mask, detected centroids, fitted line, and steering output.
(Frames are downscaled to LF_PROC_WIDTH inside run(), exactly as on the car.)


--- HOW TO CALIBRATE THE CAMERA (LF_TARGET_X, LF_HEADING_BIAS) --------------
If the overlay shows the line sitting ON the white target tick but the car
still drives physically beside the tape, the camera is mounted off-center or
yawed — image center isn't over the car's centerline. Park the car with its
centerline EXACTLY over the yellow line, pointing STRAIGHT along it, save
one camera frame, then:
  python line_following.py calibrate <frame.jpg>
It prints the LF_TARGET_X and LF_HEADING_BIAS lines to paste into
myconfig.py (the second cancels the camera-yaw slope the fit reads even
when the car is perfectly aligned).
-----------------------------------------------------------------------------
"""


import logging
import time
from collections import deque


import cv2
import numpy as np


logger = logging.getLogger(__name__)




# =============================================================================
# Tunable defaults. Every value can be overridden in myconfig.py; the part
# reads them from cfg with these as fallbacks, so it runs even with a bare cfg.
# =============================================================================
DEFAULTS = dict(
   # --- color detection (CIELAB, local-contrast based) ---
   # The tape is detected where EITHER of these fires (both are additionally
   # gated by the a-channel, lightness, and hue guards below):
   #   contrast path: b exceeds the local background (median blur over a
   #     window wider than a dash) by LF_B_CONTRAST, and b >= LF_LAB_B_FLOOR.
   #     Handles sun (tape b~161/pav 129), building shadow (tape b~141/pav
   #     120 — below the old fixed 143 threshold, which is why the car used
   #     to drop the line crossing into shadow), and washed-out sun where
   #     only a few pixels clear an absolute bar but the whole dash clears
   #     the local one.
   #   absolute path: b >= LF_LAB_B_MIN. Kept for night driving, where warm
   #     lights lift the pavement's b and shrink the contrast margin (night
   #     tape b 145-149 vs concrete <=139, i.e. only +6..+10 contrast).
   LF_B_CONTRAST=10,       # min b-channel rise over local background
   LF_LAB_B_FLOOR=133,     # absolute b floor for the contrast path; keeps
                           # gray-on-gray noise contrast from reading yellow
                           # (sunlit pavement measures b<=130, shadow tape 141)
   LF_BG_B_CLAMP=124,      # floor for the background estimate itself. The
                           # BLUE tape patches on the track (b~98) drag the
                           # local median down, which would make the plain
                           # gray pavement NEXT to them read as "+contrast
                           # yellow" and steer the car at the tape border.
                           # Clamping the background to at-least-nearly-
                           # neutral kills that; shadow pavement (b~118-120)
                           # is barely affected (clamp raises the bar for a
                           # shadow dash to b>=134; measured shadow tape 141).
   LF_LAB_B_MIN=143,       # absolute-path threshold (the pre-rework value)
   LF_LAB_A_MAX=142,       # max redness; rejects cones/flowers
   LF_LAB_L_MIN=40,        # min lightness; rejects near-black noise


   # Hue guard: kills VEGETATION, which passes the LAB yellowness test.
   # Measured from drive video: sunlit grass/plants are H 34-46, the tape is
   # H 18-29 in all lighting (sun/shadow/night — shadow tape measured H 24
   # even though shadow pavement goes blue, H~122). OpenCV hue, 0-179.
   LF_HUE_MIN=10,
   LF_HUE_MAX=32,


   # --- region of interest ---
   LF_ROI_TOP=0.62,        # ignore everything above this fraction of the image.
                           # The camera is mounted high/level: the horizon sits
                           # ~55% down the frame, planters/bushes right above
                           # it. Only the bottom ~38% is reliably track.
   LF_NUM_BANDS=8,         # horizontal bands the ROI is split into
   LF_MIN_BAND_PIXELS=6,   # a band needs at least this many mask pixels to count
                           # (sized for 160x120; use ~12+ at 320x240)
   LF_MIN_BANDS=2,         # need centroids in at least this many bands for a fix
   LF_MIN_DASHES=2,        # ...and at least this many SEPARATE blobs in the
                           # mask to ACQUIRE the line. The line is dashed, so
                           # a real sighting is normally multiple dashes,
                           # while a single connected blob passing all the
                           # other filters is usually a shirt logo, dead
                           # leaf, or piece of furniture (over half the
                           # false "tracking" hits on carried-around footage
                           # were single blobs). But a single blob IS
                           # trusted while already tracking — a close-up
                           # dash can be the only one in view — provided it
                           # appears near the last known line position:
   LF_SINGLE_DASH_GRACE=30,   # ...within this many frames of the last fix
   LF_SINGLE_DASH_MAX_JUMP=0.15,  # ...and within this fraction of image
                                  # width of the last known x

   # Minimum vertical spread (fraction of roi_h) the kept fit points must
   # span before the SLOPE is trusted for heading. heading = a * roi_h /
   # (w/2): with only LF_MIN_BANDS=2 points and a narrow y-gap between
   # them, a couple pixels of centroid noise swings "a" wildly -- session
   # 26-07-25_51 (first daytime run) pegged the raw heading to +1.0 then
   # -1.0 within half a second while genuinely still tracking the dash,
   # driving the real car's steering back and forth until it wandered
   # into the planter. Position (x_near) isn't gated by this: it's a
   # bounded near-field average, not a raw slope, so it stays usable even
   # when the fit is too ill-conditioned to trust for heading.
   # (Restored 2026-07-28: this fix was lost in the 07-26 code pull.)
   LF_MIN_HEADING_SPREAD_FRAC=0.35,

   # --- shape filter (rejects color-alike blobs: bushes, leaves, furniture) ---
   # Color alone (LAB/hue above) also matches shrubs, dry leaves, and patio
   # furniture, which happen to share the tape's yellowness. Measured on this
   # track's data: those false positives are big blobs clipped against the
   # TOP of the ROI (they're background objects, not on the ground), whereas
   # a real dash is on the ground and shrinks toward the horizon the farther
   # away it is. So a genuine dash's height is always small relative to its
   # own distance from the ROI top; a bush/chair/person blob is not. Measured
   # across day footage: real dashes stay under height/distance ~0.33;
   # foliage/furniture/people start at ~0.36 and go up to ~0.85. Any
   # connected mask blob whose (bbox height / distance-from-ROI-top) exceeds
   # this ratio is dropped before banding.
   LF_BLOB_MAX_HEIGHT_RATIO=0.4,

   # A second, independent false-positive source: single stray pixels of
   # color-alike debris (a dead leaf, a mulch chip, a grass tip at the
   # track's edge) that are far too small and isolated to be a dash. These
   # don't get caught by the height-ratio check above because they're small
   # in every dimension, not tall-for-their-distance. But because each band
   # centroid is a moment average over its ENTIRE row width, even a single
   # ~30px speck sitting far to the side of the real dashes can drag that
   # band's centroid there and swing the whole fitted line's slope, since
   # least-squares fit has no way to know that speck isn't part of the line.
   # Expressed as a fraction of ROI area so it holds across resolutions;
   # measured contaminating specks were <0.0001 of ROI area, real dash
   # fragments (even distant, partial ones) were >0.0001.
   LF_BLOB_MIN_AREA_FRAC=0.0001,

   # Size of the ellipse used for the mask OPEN+CLOSE above. 0 = the
   # historical rule (5px at >=240 wide, else 3). MEASURED TRADE-OFF, not a
   # free win -- 2026-07-30 on session 26-07-30_89, whose complaint was
   # "not picking up any line":
   #   3px: on-track detection 59% -> 75% of driving frames (+56, -2), and
   #        the 4.5s parked tail 177393-177482 goes 0/90 -> 59/90 frames
   #        (5px OPEN erases 100% of a real, thin, obliquely-viewed line
   #        that passes the colour test on 176-572 px of the ROI).
   #   cost: it also stops suppressing yellow CLUTTER. On the carried
   #        off-track block 169565-170036 (the second RC car, yellow
   #        chassis, parked on pavement) detection goes 55% -> 73%: 30
   #        extra false frames. The clean negative control -- session 88
   #        facing the recycling bin with no yellow anywhere in frame --
   #        stays clean, 0/79 -> 1/79.
   # So 3 buys real sensitivity where the line is thin and pays for it in
   # yellow-clutter rejection. Left at the historical value pending a
   # real drive; flip it in myconfig.py to try the trade.
   LF_MORPH_KERNEL=0,

   # --- pavement barrier (keeps the car from following debris off the track) ---
   # The track is a paved lane bounded on each side by a solid WHITE line;
   # outside those lines is a tan gravel planter bed full of grass and dry
   # leaves. The yellow dashes we follow are ALWAYS painted on the grey
   # pavement between the white lines. The color+shape filters above still
   # let through leaf/mulch/grass-tip debris lying in the gravel bed, and
   # when the ROI fills with that bed the fit gets dragged off into the
   # bushes.
   #
   # What separates cleanly is the immediate BACKGROUND around each
   # candidate dash. A real dash sits on pavement: its surroundings are
   # colour-neutral (CIELAB b ~120-135 depending on light, never tan) and
   # nearly as bright as the dash itself, since dash and pavement share the
   # same illumination (measured: shadow pavement L=91 under a shadow dash
   # L=112; sun pavement 253 under a sun dash 241). A debris blob sits in
   # the gravel/grass: its surroundings are tan (b>=137 in sun) or dark
   # green (a fraction of the blob's own brightness). So we sample a window
   # around each blob and drop it unless enough of the background reads as
   # pavement. The brightness test is RELATIVE to the blob (background L
   # must be at least LF_PAVEMENT_L_RATIO of the blob's L) so it works in
   # shadow — the old fixed L>160 test rejected every dash the moment it
   # crossed into building shadow, which was the main "loses the line going
   # from brightness into shadow" failure.
   LF_REQUIRE_PAVEMENT=True,      # set False to disable the pavement barrier
   LF_PAVEMENT_L_RATIO=0.5,       # bg must be >= this fraction of blob brightness
                                  # (real dashes: bg/blob 0.8-1.05; leaf on dark
                                  # grass ~0.3)
   LF_PAVEMENT_B_MAX=136,         # pavement is neutral; above this = tan gravel
   LF_PAVEMENT_A_DEV=10,          # pavement a-channel stays within this of neutral(128)
   LF_PAVEMENT_BG_MIN_FRAC=0.5,   # min fraction of a blob's background that must
                                  # be pavement to keep it (real>=0.80, gravel<0.40)


   # --- white boundary lines (solid; used as fallback and for lane offset) ---
   # The white lines are brighter than the pavement right next to them
   # (near-field: L 255 vs 213 in sun) but at distance in full sun the
   # pavement washes out to nearly paint-white, so like the yellow they are
   # detected by LOCAL contrast: L above the median-blurred background,
   # with near-neutral chroma (relative to the local background, so the
   # blue cast of shadow doesn't disqualify them; the blue tape patches on
   # the track fail the b-deviation test and are excluded).
   # While the yellow is tracked, the detector learns the horizontal
   # distance from the yellow line to the nearest white line on each side
   # (EMA). When the yellow drops out (dash gap, deep shadow, worn paint)
   # but white lines are visible, it keeps steering from the white lines
   # plus those learned offsets — the whites are solid, so this fallback
   # has no gaps — with status "white-guided". Set LF_WHITE_ENABLED=False
   # to disable all of it.
   LF_WHITE_ENABLED=True,
   LF_WHITE_L_CONTRAST=18,  # min L rise over local background
   LF_WHITE_B_DEV=8,        # max |b - local background b| (colour-neutral)
   LF_WHITE_A_DEV=8,        # max |a - 128|
   LF_WHITE_MIN_BANDS=2,    # bands needed to trust a white line fit
   # The paint isn't the only thing that's locally bright: sun glints on
   # pavement aggregate, wall trim, and clothing all pass the contrast
   # test as small speckles. A painted line is a long connected streak, so
   # white blobs must clear both a minimum area and a minimum extent
   # (largest bbox side, as a fraction of ROI height) to count.
   LF_WHITE_MIN_AREA_FRAC=0.001,
   LF_WHITE_MIN_EXTENT=0.2,
   # Painted boundary lines are THIN: a few px of paint per image row
   # (thicker near the car, but never a solid slab). The bright-neutral
   # blobs that defeat the area/extent tests -- kiosk panels, wall-base
   # trim, elevator surrounds, signage (2026-07-28 footage: the white
   # mask lit up the recycling kiosk and the stone trim at the elevator
   # bank) -- are WIDE: tens of px per row. Drop any white blob whose
   # median per-row thickness exceeds this fraction of image width.
   LF_WHITE_MAX_THICK_FRAC=0.06,
   # ...and real boundary lines run from near the horizon DOWN toward
   # the car, while wall bases / kiosk feet / building trim live only in
   # the top band of the ROI (they sit at their wall's ground contact,
   # which is near the horizon). Drop white blobs whose BOTTOM never
   # reaches this fraction of the ROI height.
   LF_WHITE_MIN_BOTTOM_FRAC=0.3,
   # Light-GREY clutter (2026-07-29 footage: hexagonal plaza pavers, a
   # lighter repair patch mid-pavement, aluminium wall trim, stone
   # planter walls) passes the local-contrast test just like paint does,
   # and the survivors of the area/extent/thickness gates were getting
   # the white line drawn on them. Two structural tests, both
   # illumination-relative so they hold across sun/shadow:
   # 1) a painted line is many times LONGER than it is thick per row --
   #    grey patches and paver tops are blobs, about as wide as tall.
   #    Drop blobs whose extent is under this multiple of their median
   #    per-row thickness.
   LF_WHITE_MIN_ASPECT=3.2,
   # 2) paint is brighter than the pavement on BOTH horizontal sides of
   #    it; the bright strip that the median-blur background makes along
   #    the BORDER of a large bright region (plaza edge, wall base, the
   #    lit side of a patch) has its own parent region on one side.
   #    Require the blob's row centres to out-bright both flanks by this
   #    many L counts (a side with too few in-frame samples -- line at
   #    the frame edge -- is skipped, never fails).
   LF_WHITE_SIDE_CONTRAST=10,
   # Max age (sec) a _last_wl/_last_wr "last selected white cluster" value
   # may be reused for continuity/estimation before it's treated as unknown.
   # Session 26-07-25_52: the right boundary was last genuinely re-detected
   # at idx 77644 (car mid-turn, cx sitting near the right frame edge), then
   # went undetected (out of frame / occluded past a planter) for 8+
   # seconds while _last_wr just sat there unchanged -- nothing ever
   # invalidated it. _estimate_from_whites kept computing est_r from that
   # stale, now-geometrically-meaningless x every single frame, dragging
   # x_f/err toward it and steering the real car harder and harder toward
   # a recycling-bin enclosure until the run had to be stopped. A real
   # boundary sighting doesn't vanish for seconds and reappear valid;
   # expire it instead of trusting it forever.
   # (Restored 2026-07-28: lost in the 07-26 code pull.)
   LF_WHITE_STALE_SEC=1.5,
   # Max plausible frame-to-frame move for the continuity-picked side
   # cluster in _split_whites, as a fraction of image width. See the
   # comment at the call site: without this, continuity still accepts
   # whichever candidate is LEAST bad even when every candidate is really
   # just noise (paint fragments/debris), and picking a "best of bad
   # options" every frame random-walks the position estimate just as
   # badly as picking blind. Below this, no detection is reported.
   # (Restored 2026-07-28: lost in the 07-26 code pull.)
   LF_WHITE_JUMP_MAX_FRAC=0.15,
   # White guidance/tracking may only TAKE OVER steering after the yellow
   # has been continuously lost for this long. Added 2026-07-28: with the
   # white fallback engaging on the very first not-found frame, every
   # ordinary dash gap or single rejected fix flipped the steering source
   # to the white chain for a frame or two, and each flip re-anchored the
   # estimate on a different signal -- the whites should only ever steer
   # when the yellow is genuinely gone, not between two dashes. During
   # the dwell the ordinary coasting path holds the last steering, which
   # is what a dash gap needs anyway.
   LF_WHITE_TAKEOVER_SEC=0.2,
   # --- near-white tracker (primary reference for lane-offset driving) ---
   # Session 38 (-0.75, the "still gets lost on left-lane turns" runs)
   # settled the architecture: in the three episodes that ended in a
   # stop, the near white line was DETECTED in 92-97% of frames, but the
   # estimate chain (side-split against the last yellow position, then
   # sanity gates) accepted almost none of it — the yellow reference the
   # chain leans on is exactly what a blind bend doesn't have. So for
   # |lane offset| >= 0.25 the near boundary gets its own persistent
   # tracker: every frame its cluster is matched against the TRACKER'S
   # OWN previous position (gyro-compensated; falling back to where the
   # boundary belongs in ego frame), and its position AND heading are
   # filtered exactly like the yellow's. When the yellow drops out the
   # controller steers directly on this tracker — with LIVE heading
   # feedback through the whole arc, not a frozen snapshot. The solid
   # white has no dash gaps: if it's in view, this tracks it.
   LF_WHITE_TRACK=True,        # enable the tracker (lane-offset only)
   LF_WHITE_TRACK_JUMP=0.12,   # per-frame match allowance, fraction of
                               # image width (plus the gyro-predicted
                               # shift, added automatically)
   LF_WHITE_TRACK_FRESH=0.35,  # seconds; tracker older than this stops
                               # steering (falls back to the estimate
                               # chain / coast)
   # The single-shot white-guided estimate chain (_estimate_from_whites,
   # below) is only sanity-checked against its OWN last accepted output
   # (see the "line cannot teleport" gate at its call site) -- a purely
   # self-referential check, so a signal with any sustained directional
   # bias can still walk arbitrarily far over many frames, one small
   # step at a time, none of which individually looks implausible.
   # LF_WHITE_TRACK's persistent tracker (above) computes the same
   # physical quantity -- the lane's near white boundary -- through a
   # completely different, independent path (temporal/gyro-predicted
   # matching rather than a fresh per-frame side-split), so requiring
   # the two to agree catches exactly the drift the self-referential
   # gate cannot. Expressed as a fraction of the (calibrated or
   # learned) lane width rather than a pixel count or anything specific
   # to one camera/track, so it generalizes across cameras and lane
   # geometries. Only applied when the tracker is fresh enough to trust
   # (LF_WHITE_TRACK_FRESH); skipped entirely if the tracker has no
   # estimate at all (LF_WHITE_TRACK disabled, or nothing seen yet) so
   # this is purely additive.
   LF_WHITE_GUIDE_DISAGREE_MAX_FRAC=0.5,

   # In a bend, a straight fit through the CURVING white line projects to
   # the frame bottom along the chord, systematically toward the inside
   # of the curve — session 37 measured the near-white 40-70px low in a
   # right-hander, so the white-based yellow estimate landed left of the
   # lane target and the car "steered back" out of its own turn. The
   # frozen heading is a curvature proxy, so shift the estimate by this
   # fraction of the image width per unit of (heading - bias). 0 disables.
   LF_WHITE_CHORD_COMP=0.18,

   # White guidance bridges stretches without yellow: dash gaps, worn or
   # deeply shadowed paint — and, in lane-offset driving, whole TURNS
   # (riding -0.75, the yellow regularly exits the frame for the entire
   # arc of a bend while the near white boundary stays in view; session
   # 35/36 footage). It engages for at most this many seconds after the
   # last confirmed yellow fix, then the normal coast/stop logic takes
   # over. Unbounded, a picked-up car staring at any bright edge would
   # keep "driving" forever; 6s covers the longest bend at cornering
   # speed while still going passive within a couple of car lengths if
   # the scene is genuinely gone.
   LF_WHITE_GUIDE_SEC=6.0,
   # Gates on the yellow-to-white geometry, all fractions of image width.
   # MIN/MAX bound which white CLUSTERS can be the lane boundary at all
   # (mixed-row offsets); BOT_MIN..MAX bound the LEARNED samples, which
   # are bottom-projected and therefore larger — the true bottom-row lane
   # widths measure ~106px left / ~128px right (0.28/0.33) on this track.
   # The junk these exclude: white glare within a few px of the yellow,
   # the second parallel line on the right (~150-170px mixed-row, farther
   # still at the bottom), and neighboring-track paint. Cluster selection
   # (nearest valid line wins) is the main defense; these are backstops.
   LF_LANE_DIST_MIN_FRAC=0.08,
   LF_LANE_DIST_MAX_FRAC=0.45,
   LF_LANE_DIST_BOT_MIN_FRAC=0.15,
   # MAX_FRAC's ceiling (172.8px) leaves the left lane's pinned width
   # (LF_LANE_WIDTH_PX left side, 170px, measured from a real left-lane
   # vantage) only 2.8px of margin, vs. the right lane's 140px having
   # 32.8px. This module's own chord-distortion note above measures
   # in-turn offsets reaching ~177px even when straight-line is ~106px,
   # so a real, in-lane left-boundary cluster mid-turn was routinely
   # landing just past the ceiling and getting dropped as "implausible"
   # -- verified by instrumenting _split_whites against recorded
   # footage: the nearest real left-side cluster was rejected by this
   # ceiling in 347/971 frames, over a third, with 103 of those between
   # 174-220px (plausible real geometry, not a stray far line). Exactly
   # the frames where yellow is also likely lost (mid turn) are the
   # ones this fallback exists for. Widening MAX_FRAC itself measurably
   # changes center-line (LF_LANE_OFFSET=0.0) white-fallback behavior
   # too, since _split_whites runs unconditionally -- so instead this
   # ONLY widens the ceiling while lane-offset driving is actually
   # engaged (|_lane_applied| >= 0.25, same threshold used everywhere
   # else lane-offset-specific behavior gates on), leaving center-line
   # driving byte-identical. 211px gives the left lane comparable
   # proportional headroom above its chord-distorted reference to what
   # the right lane already had at the original ceiling.
   LF_LANE_DIST_MAX_FRAC_LANE=0.55,
   # Center-line white-guided estimate: how far apart (as a fraction of
   # image width) the left- and right-side yellow estimates are allowed
   # to disagree before _estimate_from_whites stops averaging them and
   # trusts only the one closer to the last known yellow position. See
   # the comment at the call site for why -- one side reading the
   # ADJACENT lane's boundary is otherwise invisible to the per-side
   # range gates. (Restored 2026-07-28: lost in the 07-26 code pull.)
   LF_CENTER_DISAGREE_MAX_FRAC=0.25,
   # Lane widths are only LEARNED while driving straight-ish (|corrected
   # heading| below this): mid-turn the bottom projections are chord-vs-arc
   # distorted (measured 177/102px vs 106/128 straight) and those samples
   # dragged the target sideways exactly in corners — the car crossed the
   # center line into the wrong lane on turns because of it. Frozen-in-turn
   # estimates stay at their straight-section values instead.
   LF_LANE_LEARN_MAX_HEADING=0.20,
   # ...and only from white clusters whose lowest sighting reaches this
   # fraction of the ROI height: clusters seen solely near the ROI top
   # anchor their bottom projection on extrapolation through clutter.
   LF_LANE_LEARN_NEAR_FRAC=0.3,
   # The lane width in bottom-row pixels, if you know it. The width is a
   # TRACK CONSTANT, and the live estimate of it is the weakest link in
   # lane driving: white-cluster bottom projections scatter 85-170px
   # frame to frame (sparse centroids, extrapolation), so the learned
   # median lands ±20px differently per run — visible as a different ride
   # position each session, and it starves entirely when the car STARTS
   # inside a lane (the near boundary exits the frame bottom before it's
   # ever seen). Set this to pin the offset geometry; measured ~140px on
   # this track (sessions 12-15, several gate configurations agree).
   # None = fall back to live learning (dist_left/dist_right).
   LF_LANE_WIDTH_PX=None,
   # Same-side white detections are clustered by their offset from the
   # yellow fit; x-gaps larger than this (fraction of image width) split
   # clusters, and only the cluster NEAREST the yellow is used. Averaging
   # everything on a side (the old behavior) blended the true boundary
   # with the second parallel line into a distance matching neither.
   LF_WHITE_CLUSTER_FRAC=0.06,


   # --- steering control (PID on normalized lateral error) ---
   # error: -1 = line at left edge, 0 = line at target, +1 = line at right edge
   # donkeycar steering: -1 = full left, +1 = full right
   #
   # Retuned 2026-07-23 against the afternoon tub sessions: the car held the
   # line but SWERVED constantly on straights. Replaying those sessions
   # showed why: the detector's lateral reading jumps ~20px frame to frame
   # (the dashes marching through the bands wiggle the fit, and projecting
   # to the frame bottom scales slope jitter by the ROI height), and the
   # old controller passed that noise straight to the servo — the raw
   # one-frame D term alone contributed more steering than every real
   # signal combined (std 0.66 vs P's 0.44), and the heading term regularly
   # saturated its whole +/-0.9 range. So the measurements are now low-pass
   # filtered before the PID sees them (LF_X_FILTER_TC,
   # LF_HEADING_FILTER_TC), the derivative gets its own filter
   # (LF_D_FILTER_TC), and the command is slew-limited (LF_STEER_SLEW).
   # Closed-loop sim (bicycle model, 100-150ms latency, measured noise):
   # steering std on straights 0.64 -> 0.17 at equal-or-better lateral
   # tracking, stable across latency/servo-lag/gain perturbations —
   # whereas heavier filtering (x TC 0.15s+) with a tight slew limit went
   # UNSTABLE: the heading term and D are the loop's phase lead, they must
   # stay fast. Don't raise the TCs to chase more calm.
   LF_STEER_KP=1.2,        # proportional gain on lateral offset. Lowered
                           # 1.6 -> 1.2 on 2026-07-24: the image-x the P
                           # term chases mixes true lateral offset (233px/m)
                           # with instantaneous camera yaw (279px/rad), and
                           # with the measured ~230ms loop delay the yaw
                           # share turns surplus P gain into the ~1Hz jerk.
   LF_STEER_KD=0.30,       # derivative gain (per second), computed on the
                           # filtered error and then low-passed again
   LF_X_FILTER_TC=0.12,    # seconds; EMA time constant on the lateral
                           # measurement (~2.4 frames at 20 Hz)
   LF_D_FILTER_TC=0.08,    # seconds; EMA on the derivative term
   LF_HEADING_FILTER_TC=0.06,  # seconds; EMA on the heading measurement.
                           # Deliberately light — see the retune note above.
   LF_STEER_SLEW=6.0,      # max steering change per second (full lock to
                           # full lock in ~1/3 s). Protects the servo and
                           # kills single-frame spikes; sim shows anything
                           # much tighter induces a limit cycle.
   LF_STEER_KI=0.2,        # integral gain (per second). This is what removes
                           # the constant "tracks the line but rides beside
                           # it" offset: any persistent bias (servo trim not
                           # quite centered, drivetrain pull, slight camera
                           # yaw) leaves a P controller with a steady-state
                           # error of bias/KP — the car follows the line
                           # PARALLEL, offset to one side, forever. The
                           # integral winds up until the residual error is
                           # zero. It is a TRIM estimator, nothing more:
                           # since 2026-07-24 it only integrates while
                           # tracking NEAR-STRAIGHT with a SMALL error (see
                           # LF_STEER_I_H_GATE / LF_STEER_I_ERR_GATE). The
                           # old always-on ki=0.4 wound to its cap inside
                           # every curve (a curve needs sustained steering
                           # the heading term didn't fully supply) and then
                           # unwound over seconds on the exit, dragging the
                           # car across the line — replays of sessions 17-18
                           # show i railed at ±0.35 for whole curve+exit
                           # stretches, and the closed-loop sim reproduces
                           # the resulting S-swerve on every straight that
                           # follows a bend. Held (applied but frozen)
                           # during white-guided / coasting; reset on stop.
   LF_STEER_I_MAX=0.20,    # cap on the integral term's steering contribution
                           # (anti-windup: also frozen while output saturated).
                           # 0.20 covers realistic servo/drivetrain trim; the
                           # old 0.35 was a third of full lock — as a stale
                           # post-curve residue it alone steered the car
                           # across the yellow.
   LF_STEER_I_H_GATE=0.25, # |corrected heading| must be below this for the
                           # integrator to update (trim is only observable
                           # while driving straight; in curves the sustained
                           # error is curvature, not trim)
   LF_STEER_I_ERR_GATE=0.25,  # ...and |error| below this (a big error means
                           # we're actively converging on the line/lane —
                           # integrating the transient just adds overshoot)
   LF_HEADING_GAIN=0.3,    # extra steering from the line's slope. NOTE
                           # (2026-07-24, identified from footage): this is
                           # NOT a damping term. A small camera yaw
                           # translates the line in the image without
                           # changing its slope (first-order), so the
                           # measured heading carries almost no yaw signal
                           # — regression against the true yaw (phase-
                           # correlated far-field strip) gives h_f ≈
                           # 0.35*err + 0.10*yaw: it is mostly REDUNDANT
                           # POSITION feedback plus, in bends, the genuine
                           # curvature lean (the useful part: curve feed-
                           # forward). So this gain buys turn-in and pays
                           # for it with extra position-loop gain, which
                           # the ~230ms loop delay punishes. 0.4 is the
                           # identified-plant optimum; 1.6 (tried earlier
                           # on the wrong belief it damps) made the weave
                           # worse. Real damping comes from LF_GYRO_GAIN.
   LF_HEADING_CLIP=0.7,    # cap on |heading| before the gain. The raw
                           # heading hit +/-1.0 routinely; beyond ~0.7 it's
                           # a fit artifact (near-horizontal line), not a
                           # steer-harder signal.
   LF_HEADING_BIAS=0.0,    # subtracted from the measured heading: residual
                           # camera yaw AFTER the vanishing-point
                           # convergence correction (see detect()). The
                           # clean straight-frame residual on this camera
                           # measures +0.06; before the correction existed
                           # this read +0.12 because riding position leaked
                           # into it. Measure it:
                           #   python line_following.py calibrate <frame>
                           # (same parked-on-the-line frame as LF_TARGET_X)
   LF_VP_Y_FRAC=0.6,      # vanishing-point row as a fraction of image
                           # height, for the convergence correction: ground
                           # lines parallel to the car converge here, so a
                           # line's expected image slope grows with its
                           # lateral offset by 1/(h*(1-this)) per px. The
                           # horizon sits ~55% down this camera's frame
                           # (session fit implied 59%).
   LF_ERR_DEADBAND=0.0,    # soft deadband on the normalized error: errors
                           # smaller than this are ignored (subtracted, so
                           # response stays continuous). 0 = tightest
                           # centering over the line; raise to ~0.05 only if
                           # the car weaves on straights.
   LF_TARGET_X=None,       # where the line should sit in the frame, in pixels
                           # (LF_PROC_WIDTH coords). None = image center. If
                           # the camera is mounted off-center or yawed, image
                           # center is NOT over the car's centerline, and the
                           # car will hold a physical offset from the line
                           # even with zero control error (the overlay shows
                           # the line ON the white target tick, yet the car
                           # sits beside the tape). To calibrate: park the car
                           # so its centerline is EXACTLY over the yellow
                           # line, save one camera frame, then run
                           #   python line_following.py calibrate <frame>
                           # and paste the printed LF_TARGET_X into myconfig.

   # --- vision gyro (yaw-rate damping) ---
   # Identified from session-21 footage (2026-07-24): the loop's real
   # problem is that NOTHING in the frame measures the car's yaw. A small
   # camera yaw just TRANSLATES the line in the image (first-order, the
   # slope doesn't change), so the "lateral error" the PID chases is
   # mostly instantaneous yaw (279px per rad vs 233px per meter of true
   # offset — in a weave the yaw part dominates), and the heading term is
   # ~0.35*err redundant position feedback, NOT damping (regression of
   # h_f against the true yaw extracted from the footage: yaw coefficient
   # +0.10, i.e. nil). With the measured 200-250ms loop delay (steering ->
   # visible yaw-rate cross-correlation peaks at 4-5 frames) the loop is
   # a stiff spring with no damper: it limit-cycles at ~0.9Hz — the
   # "jerky swerve about once a second".
   #
   # The fix measures yaw rate directly: phase-correlate a strip of the
   # FAR BACKGROUND (above the horizon, rows 0.35-0.50 of the frame)
   # between consecutive frames. Distant scenery has no parallax, so its
   # horizontal shift is pure camera yaw — a vision gyro (~1ms/frame at
   # 384px). Steering gets -LF_GYRO_GAIN * yaw_rate: true rate damping,
   # which is exactly what a delayed position loop needs. Set gain 0 to
   # disable (e.g. if the horizon strip is full of moving people).
   # --- slow curve feed-forward ---
   # Measured on the 2026-07-24 15:50 lap sessions (bottom-row dash pixels,
   # which the controller cannot fake): on straights the car sits dead on
   # the line (dash at x=188-193 of 192), but in every bend it runs
   # 40-50px (~0.2m) toward the OUTSIDE — right of the line in left turns,
   # left in right turns. That's textbook steady-state error: holding a
   # bend needs ~1.65*kappa of steering, the heading term at kh=0.4
   # supplies ~0.5*kappa, the trim integrator is deliberately frozen in
   # curves, so P must carry the rest — and P only pushes when there IS an
   # error. This term supplies the missing steady steering: the SLOW
   # component of the heading lean (EMA over LF_CURVE_TC) is the sustained
   # curve signature — boosting it acts like an in-curve integrator that
   # is bounded (proportional to the measured lean) and decays with the
   # lean itself on exit, so it cannot reproduce the old ki windup-unwind
   # S-swerve. The fast component (turn-in, noise) stays at kh.
   # FIELD-TUNED VALUES (2026-07-24 evening, confirmed working on track —
   # note the sign is NEGATIVE, opposite what the sim predicted; on the
   # real car the un-countered slow lean was cutting the car INTO the
   # curve/to the right, so the working gain opposes the lean):
   #   LF_LANE_OFFSET  0.0 (center line)  -> LF_CURVE_GAIN = -0.8
   #   LF_LANE_OFFSET -0.75 (left lane)   -> LF_CURVE_GAIN = -0.8
   #   LF_LANE_OFFSET +0.5 (right lane)   -> LF_CURVE_GAIN = -0.2
   # Change this value together with the offset. Applies only while
   # TRACKING; blind (white-guided) turns exclude this term on purpose —
   # a negative gain fed the frozen mid-turn lean would steer against
   # completing the turn.
   LF_CURVE_GAIN=-0.5,     # steering per unit of slow heading lean
   LF_CURVE_TC=0.7,        # seconds; EMA defining "sustained". Shorter =
                           # quicker to full curve hold but more of the
                           # heading noise leaks into this high-gain path.
   # A PEGGED heading (|hterm| at ~the clip) means the line is sweeping
   # out of the frame: the bend is real, immediate, and about to blind
   # the car — whatever the curve-gain trim is doing on moderate bends,
   # refusing to steer WITH a pegged heading guarantees losing the line.
   # Session 38, sharp right-hander at the building entrance: heading
   # +1.0, error positive, and the field-tuned NEGATIVE curve gain
   # cancelled the whole heading response — net command ~0, car went
   # straight into the breezeway. When |hterm| >= 0.9*LF_HEADING_CLIP,
   # the combined heading response (kh*hterm + curve_gain*h_slow) is
   # floored at this magnitude, same sign as the lean. Moderate bends
   # are untouched.
   LF_SHARP_BEND_MIN=0.5,
   # Same floor, but for the WHITE-GUIDED (blind, estimate-only) branch
   # specifically, as a fraction of LF_HEADING_CLIP. Session 26-07-24_44,
   # same right-hander as session 38 above: yellow dropped out early in
   # the bend, before the heading reading had ramped up past 0.9*h_clip,
   # so the general floor above never engaged and the frozen hterm (0.38,
   # 0.47 in two separate approaches) drove a weak, non-committal turn —
   # the car went straight into the furniture past the corner both times.
   # Lower only for this branch: tracking and white-tracking already
   # negotiate this bend fine when yellow/white are actually in view, so
   # their trigger point (0.9) is untouched -- only the blind estimate
   # chain, which has already lost its main sensor and needs to commit
   # harder on less evidence, engages the floor earlier.
   # (Restored 2026-07-28: lost in the 07-26 code pull; myconfig.py was
   # still setting it to 0.5 the whole time, silently ignored.)
   LF_WHITE_GUIDE_BEND_FRAC=0.9,

   LF_GYRO_GAIN=0.3,       # steering per rad/s of yaw (sim-tuned on the
                           # identified plant; the win is biggest exactly
                           # where the car was worst: at speed and at high
                           # latency, straight-line weave rms 10 -> 4cm)
   LF_GYRO_FILTER_TC=0.05, # seconds; light EMA on the measured yaw rate
   LF_GYRO_WASHOUT_TC=1.0, # seconds; the damper acts on yaw rate MINUS its
                           # own slow average (a washout, as in aircraft yaw
                           # dampers). A steady curve is a constant yaw rate
                           # — un-washed, the damper steers against the turn
                           # for its whole duration; washed out, it only
                           # resists CHANGES in yaw rate (the 0.9Hz weave)
                           # and lets a held arc pass. 0 disables washout.
   LF_GYRO_STRIP=(0.35, 0.50),  # frame-height fractions of the strip; keep
                           # it ABOVE the horizon (~0.55 on this camera) so
                           # only zero-parallax background is in it
   LF_CAM_HFOV_DEG=69.0,   # camera horizontal field of view, for the
                           # px-shift -> yaw-angle conversion (OAK-D RGB)

   # --- temporal consistency gate on the yellow fix ---
   # Observed on the 2026-07-24 lap sessions (34 events in 3 laps): for one
   # or two frames the fit latches onto color-alike clutter (dead leaves,
   # pavement patches, shoes) mixed in with sparse real dashes — the fitted
   # line swings diagonal (heading pegs at +/-1) and the bottom projection
   # jumps 50-140px, and the car twitches toward the phantom before the
   # next clean frame corrects it. The line cannot physically do that
   # between frames at 20Hz: the car's own yaw moves it at most
   # f*yaw_rate*dt (~15px/frame per rad/s, and we MEASURE yaw rate with
   # the vision gyro) and lateral motion adds a few px more. So a new fix
   # is only accepted if it lands near where the last accepted fix
   # predicts; otherwise the frame is treated as a dropout (the existing
   # coast/white-guided logic rides through it, holding steering). To
   # avoid locking out a genuinely new position forever (e.g. the gate
   # engaged on a phantom), the allowance grows with each consecutive
   # rejection and after LF_TEMPORAL_MAX_REJECT rejections the fix is
   # accepted unconditionally (filters reseed).
   LF_TEMPORAL_JUMP=0.09,     # base allowed |x jump| per frame, fraction of
                              # image width (~35px; measured noise is ~10px,
                              # real yaw shift is compensated separately)
   LF_TEMPORAL_H_JUMP=0.45,   # allowed heading change per frame; real yaw
                              # changes heading < ~0.2/frame even at full
                              # lock, phantom fits swing 0.5-2.0
   LF_TEMPORAL_MAX_REJECT=6,  # accept unconditionally after this many
                              # consecutive rejections (0.3s at 20Hz)
   # Passing the jump gate above only means a fix isn't wildly implausible
   # relative to the OLD reference -- after a real gap (a multi-second
   # coast/white-guided run, common mid-turn) that reference is stale
   # enough that the gate widens a lot (see the staleness growth above),
   # so "not implausible" stops meaning much. A one-off misfit can still
   # slip through, and on real footage the position then kept walking
   # frame to frame (each step individually small enough to pass) rather
   # than snapping back -- the same "several plausible steps compound
   # into an implausible walk" shape this project has hit before.
   # Require this many CONSECUTIVE reacquired fixes to agree with EACH
   # OTHER (a tight, ordinary frame-to-frame budget, not the widened
   # one) before trusting any of them; until confirmed, ride the
   # coast/white-guided path exactly like an ordinary dropout. Only
   # engages when reacquiring after a real gap -- continuous tracking
   # (last fix < 0.5s old) is unaffected and commits immediately, same
   # as before. Set to 1 to disable (old behavior: trust the first
   # fix that passes the jump gate).
   LF_TEMPORAL_REACQUIRE_CONFIRM=3,

   # --- lane offset (future: drive IN the left or right lane) ---
   # 0.0 rides directly on top of the yellow line. +1.0 aims the car at the
   # nearest white line to the RIGHT of the yellow, -1.0 at the white line
   # to the LEFT; +0.5 is the middle of the right lane. Uses the
   # yellow-to-white distances learned from the white-line detector, so it
   # only engages once those have been observed (falls back to riding the
   # yellow until then).
   # NOTE: set this in myconfig.py, not here — myconfig.py overrides this
   # file, so editing the default does nothing once myconfig defines it
   # (that is exactly what happened on 2026-07-23: DEFAULTS said 0.75 while
   # myconfig said 0.0, and every "different offset" run actually drove 0.0).
   # The effective value is logged at startup; check it there.
   LF_LANE_OFFSET=0.0,     # 0.0 = on yellow, +1.0 = right white, -1.0 = left white
   LF_LANE_RAMP_SEC=1.5,   # seconds to ramp the offset in/out once the
                           # needed lane distance is known — stepping the
                           # target sideways by ~80px in one frame would
                           # command a swerve.
   # Reinstated 2026-07-26 after right-lane driving (LF_LANE_OFFSET=0.5)
   # kept losing the lane at the same building corner across FOUR separate
   # incidents (sessions 26-07-26_58 and _59): a "genuine" fresh yellow fit
   # (found=True, not a white-line estimate) lands somewhere geometrically
   # impossible for this lane -- e.g. idx 93371 jumped the detected yellow
   # position from x=65 to x=407 in a single frame, past the edge of a
   # 384px-wide image, immediately pegging steering to full lock. Checks
   # against lane geometry (independent of how long it's been since the
   # last fix), via _lane_pos: 0.0 = on the yellow, 1.0 = on the outer
   # white. +/-0.5 lane-widths of slack past either boundary: verified
   # against the 26-07-26 sessions this catches all four incidents without
   # clipping legitimate driving anywhere else in those sessions.
   # (Restored 2026-07-28: lost in the 07-26 code pull.)
   LF_LANE_IMPLAUSIBLE_MIN=-0.5,
   LF_LANE_IMPLAUSIBLE_MAX=1.5,
   # ESCAPE HATCH (2026-07-29, session 26-07-29_88). As restored above this
   # gate could reject FOREVER: it leaves found=False, so _filt_time and
   # last_x never update, the reacquire gap grows without bound, and the
   # yellow is locked out for the rest of the run. That is exactly what
   # session 88 did once the car crossed to the WRONG SIDE of the yellow
   # while lane-offset driving -- _lane_pos then reads ~-0.8, which is
   # "impossible" to this gate but is simply where the car actually was.
   # Measured over that session's 5 driving windows: 68 rejected frames,
   # 68/68 on the MIN (wrong-side) bound and 0 on the MAX bound, and 12 of
   # the 26 re-acquisition attempts died here (only 2 ever reached
   # tracking). Frames 174136-174171 are the clean example: the yellow was
   # detected on 36/36 frames for 1.8s, passed the temporal gate on 36/36,
   # and was never once driven on.
   # So: after this many CONSECUTIVE implausible fixes that AGREE WITH EACH
   # OTHER (ordinary frame-to-frame budget -- LF_TEMPORAL_JUMP /
   # LF_TEMPORAL_H_JUMP), force-accept and reseed the filters, exactly like
   # LF_TEMPORAL_MAX_REJECT does for the jump gate. Self-consistency is the
   # discriminator the original gate lacked: the glitches it was built for
   # (idx 93371's x=65 -> x=407 single-frame teleport) do not hold still,
   # while a real line the car has driven past sits exactly where it is,
   # frame after frame. 0 disables the escape (old lock-out behaviour).
   LF_LANE_IMPLAUSIBLE_MAX_REJECT=6,
   # Forget the streak if the last implausible sighting is older than this
   # -- two agreeing fixes seconds apart are not evidence of anything.
   LF_LANE_IMPLAUSIBLE_STALE_SEC=0.5,
   # Reinstated alongside the gate above: separately, real (not glitched)
   # sustained curves at this same corner drift the car toward the yellow
   # because LF_MIN_HEADING_SPREAD_FRAC's guard stays engaged for the
   # whole turn (chronically narrow-spread fits from this camera angle),
   # leaving only the laggy position term to correct. Rather than loosen
   # that guard (shared with plain line-following, validated to stop wild
   # heading swings), the position gain ramps up to
   # LF_LANE_BOUNDARY_KP_BOOST times normal as the car closes within
   # LF_LANE_BOUNDARY_MARGIN lane-widths of either edge. No-op mid-lane or
   # outside lane-following.
   LF_LANE_BOUNDARY_MARGIN=0.3,
   LF_LANE_BOUNDARY_KP_BOOST=1.6,
   # A real on-car test at the black-bumper corner (26-07-26_60) showed
   # the kp boost above often can't actually act: frame-by-frame steering
   # deltas through the whole boundary crossing were pegged at EXACTLY
   # LF_STEER_SLEW (confirmed: dst/dt = -6.0 across 6 consecutive frame
   # pairs there), meaning the SLEW CAP, not the computed steering demand,
   # was what limited the correction. Ramps LF_STEER_SLEW up to this
   # multiple as lane_pos closes within LF_LANE_BOUNDARY_MARGIN of an
   # edge, same trigger as the kp boost. LF_STEER_SLEW itself is left
   # alone -- it's tuned to kill noise-driven jitter on straights; this
   # only raises the cap once boundary proximity has already ruled that
   # out.
   LF_LANE_BOUNDARY_SLEW_BOOST=2.5,


   # --- throttle ---
   LF_THROTTLE_MAX=0.30,   # on straights
   LF_THROTTLE_MIN=0.16,   # in hard corrections
   LF_THROTTLE_LOST=0.12,  # while coasting on a lost line
   LF_THROTTLE_MIN_LANE_OFFSET=0.18,  # floor used INSTEAD of
                           # LF_THROTTLE_MIN specifically while steering
                           # is saturating because of a large POSITIONAL
                           # error (abs(error) > LF_STEER_I_ERR_GATE), as
                           # opposed to genuine track curvature. Added
                           # 2026-07-27. `slow = min(1.0, max(abs(steering),
                           # abs(hterm)/h_clip))` can't tell "steering hard
                           # because the track is genuinely bending"
                           # (hterm-driven -- a real curve, where slowing
                           # down is correct and this floor does NOT
                           # apply) from "steering hard because the
                           # lateral target jumped a lot" (error-driven --
                           # e.g. obstacle_avoidance.py's lane_offset
                           # commit, or simply a large x_f error already
                           # present at a maneuver's start) -- it floors
                           # throttle to LF_THROTTLE_MIN identically
                           # either way. In two real obstacle-avoidance
                           # tests that collapsed throttle to the floor
                           # within ~3 frames (~0.17s, LF_STEER_SLEW's
                           # full-lock time) of a large lateral error
                           # appearing -- well before the car had any
                           # chance to actually close that error -- and
                           # kept it floored for as long as the error
                           # stayed large, starving the exact maneuver
                           # (obstacle_avoidance.py's swerve or lane
                           # change) that needs real speed to translate a
                           # steering angle into lateral displacement.
                           # Reuses LF_STEER_I_ERR_GATE as the same
                           # "error still large, not yet converged"
                           # threshold the integral term already gates on
                           # (see LF_STEER_KI), rather than inventing a
                           # second one. Set above LF_THROTTLE_MIN but
                           # still comfortably below LF_THROTTLE_MAX --
                           # more speed budget for the maneuver to
                           # actually complete, not a green light to
                           # barrel through at full speed while still
                           # this far off target.


   # --- lost-line recovery ---
   # On loss, steer from the white lines if they're visible (see
   # LF_WHITE_ENABLED above). Failing that, hold the last known steering
   # (don't commit to a blind turn — that just traces a circle) and ease
   # off the throttle until either the line reappears or the grace period
   # runs out, then stop.
   LF_LOST_STOP_SEC=2.0,   # give up and stop after this many seconds lost
   # Before actually stopping, keep crawling: hold the same steering at
   # the lost-coast throttle for up to this many EXTRA seconds, but only
   # while the white detector is still returning centroids (see
   # LF_LOST_RECOVERY_WHITE_AGE). Added 2026-07-29 (user request): the
   # final stop of session 26-07-29_86 parked the car for good with the
   # real white boundary drawn in the overlay on every frame -- the
   # white-use gates (side split / jump gates, judged against a stale
   # yellow reference) rejected it, the 2s clock ran out, and a stopped
   # car's view never changes, so it never came back. Crawling on in the
   # direction already being steered keeps the scene evolving so the
   # yellow/white gates get fresh looks. The white-pts condition is what
   # keeps this from being the blind circle the comment above warns
   # about: no boundary in sight at all -> stop exactly as before.
   LF_LOST_RECOVERY_SEC=3.0,
   LF_LOST_RECOVERY_WHITE_AGE=1.0,  # "still returning centroids" = raw
                           # white_pts seen at most this many seconds ago

   # --- stall detection + throttle kick (2026-07-29) ---
   # A commanded throttle at/near the floor can sit below the motor's
   # static-friction threshold: the car freezes, the camera scene stops
   # changing, and since every state transition in this controller is
   # driven by what the camera sees, a stalled car is a deadlocked car
   # (session 26-07-29_86 PM: 4 stalls, ~37s total, user had to push).
   # The watch compares consecutive downscaled grayscale frames; when the
   # scene stays static while throttle is commanded positive, it ramps an
   # additive boost until the car actually moves, then bleeds it off.
   LF_STALL_DETECT=True,
   LF_STALL_MOTION=3.0,    # mean abs gray diff (96x54) below this = static.
                           # Measured 07-29 PM: parked 0.9-1.4, stalled with
                           # motor humming 1.7-2.8 (p90 3.6), normal driving
                           # p50 6.6. A false "stalled" while crawling only
                           # adds a bounded, self-releasing nudge.
   LF_STALL_SEC=0.8,       # static-while-powered this long = stalled
   LF_STALL_KICK=0.02,     # boost added per step while still stalled
   LF_STALL_STEP_SEC=0.25, # seconds between boost steps
   LF_STALL_KICK_MAX=0.12, # boost ceiling (total throttle also capped at
                           # max(LF_THROTTLE_MAX, law output))

   # --- searching re-anchor (2026-07-29) ---
   # While "searching (line lost)" the stale yellow reference (2s+ old
   # after the uncontrolled coast) is dropped so the white tracker can
   # cold-start and re-engage -- session 26-07-29_86 idx 166349-166448
   # parked with the real boundary detected on every single frame purely
   # because every gate kept measuring candidates against that stale
   # reference. Dropping it unconditionally, however, lets the tracker
   # latch onto sun-dapple fields (verified in replay at 164586: the car
   # would dance on shadows instead of parking), so the drop is gated on
   # the raw white centroids actually LOOKING like one line: a thin,
   # elongated point cloud, sustained for a few consecutive frames.
   # Thresholds measured on 07-29 PM footage (w=384): real-boundary
   # frames rms p90=4.0px / cloud aspect p10=12.7; dapple rms p50=10.8 /
   # aspect p10=5.0. Gate rms<=0.013w + aspect>=6 keeps 97% of
   # real-line frames, rejects 74% of dapple frames per-frame (and
   # dapple rarely sustains 3 consecutive passes).
   LF_REANCHOR_RMS_FRAC=0.013,   # max perpendicular rms of the centroid
                                 # cloud, fraction of frame width
   LF_REANCHOR_MIN_ASPECT=6.0,   # min sqrt-eigenvalue ratio of the cloud
   LF_REANCHOR_FRAMES=3,         # consecutive coherent frames required
   # rms+aspect alone are degenerate on SPARSE clouds: two dapple blobs at
   # opposite ends of the frame define a line exactly, so they score a
   # perfect rms and a huge aspect ratio. Replay of 164636 (car off-track
   # facing the green garage door, sun-dapple on the pavement) re-anchored
   # on exactly that and drove on where the old code parked. Requiring the
   # centroids to be strung along the line with no huge void closes the
   # degenerate case. Separation is PARTIAL, not clean -- measured over
   # labelled 07-29 PM windows (real boundary n=281 frames: gap p50=42px
   # p90=127; dapple n=112: p10=50 p50=110, w=384), 0.15w keeps 68% of
   # real-boundary frames per frame and rejects 86% of dapple frames. It
   # earns its keep through LF_REANCHOR_FRAMES: a real boundary clears
   # three in a row within the search window, a dapple field rarely does.
   LF_REANCHOR_MAX_GAP_FRAC=0.15,  # max x-gap between neighbouring
                                   # centroids, fraction of frame width

   # --- misc ---
   LF_PROC_WIDTH=384,      # downscale input frames to this width before any
                           # processing (None = native). The OAK-D part
                           # streams full 1920x1080 video regardless of
                           # IMAGE_W/IMAGE_H, which is ~25x more pixels than
                           # detection needs; 384x216 runs the whole
                           # pipeline comfortably at 20 Hz on the Pi. The
                           # overlay/web image is the downscaled one too.
   OVERLAY_IMAGE=True,     # draw diagnostics on the image sent to the web UI
)




def _h_agrees(h_new, h_ref, tol):
   """Do two heading readings agree within tol?

   A heading of None means "not measured this frame" (detect() reports it
   when its slope fit is too ill-conditioned to trust -- see the
   LF_MIN_HEADING_SPREAD_FRAC branch there), NOT "measured as zero". An
   unmeasured heading is no evidence either way, so it cannot disprove
   agreement; whatever check calls this still has to satisfy its position
   test on its own.
   """
   if h_new is None or h_ref is None:
       return True
   return abs(h_new - h_ref) <= tol


def _cfg(cfg, name):
   """Read a value from cfg, falling back to DEFAULTS."""
   return getattr(cfg, name, DEFAULTS[name]) if cfg is not None else DEFAULTS[name]




class LineFollower:
   """
   DonkeyCar part.
     input:  'cam/image_array'  (RGB numpy array)
     output: 'pilot/steering', 'pilot/throttle', 'cv/image_array'


   Signature matches the cv_control template's add_cv_controller(), which
   constructs the class as LineFollower(pid, cfg). The pid argument is
   accepted for compatibility but ignored — control is a self-contained PID
   loop so this file has no simple_pid dependency.
   """


   def __init__(self, pid=None, cfg=None):
       self.b_contrast = int(_cfg(cfg, 'LF_B_CONTRAST'))
       self.b_floor = int(_cfg(cfg, 'LF_LAB_B_FLOOR'))
       self.bg_b_clamp = int(_cfg(cfg, 'LF_BG_B_CLAMP'))
       self.b_min = int(_cfg(cfg, 'LF_LAB_B_MIN'))
       self.a_max = int(_cfg(cfg, 'LF_LAB_A_MAX'))
       self.l_min = int(_cfg(cfg, 'LF_LAB_L_MIN'))
       self.hue_min = int(_cfg(cfg, 'LF_HUE_MIN'))
       self.hue_max = int(_cfg(cfg, 'LF_HUE_MAX'))


       self.roi_top = float(_cfg(cfg, 'LF_ROI_TOP'))
       self.num_bands = int(_cfg(cfg, 'LF_NUM_BANDS'))
       self.min_band_px = int(_cfg(cfg, 'LF_MIN_BAND_PIXELS'))
       self.min_bands = int(_cfg(cfg, 'LF_MIN_BANDS'))
       self.min_dashes = int(_cfg(cfg, 'LF_MIN_DASHES'))
       self.single_dash_grace = int(_cfg(cfg, 'LF_SINGLE_DASH_GRACE'))
       self.single_dash_max_jump = float(_cfg(cfg, 'LF_SINGLE_DASH_MAX_JUMP'))
       self.min_heading_spread_frac = float(_cfg(cfg, 'LF_MIN_HEADING_SPREAD_FRAC'))
       self.max_blob_height_ratio = float(_cfg(cfg, 'LF_BLOB_MAX_HEIGHT_RATIO'))
       self.min_blob_area_frac = float(_cfg(cfg, 'LF_BLOB_MIN_AREA_FRAC'))
       self.morph_kernel = int(_cfg(cfg, 'LF_MORPH_KERNEL'))

       self.require_pavement = bool(_cfg(cfg, 'LF_REQUIRE_PAVEMENT'))
       self.pav_l_ratio = float(_cfg(cfg, 'LF_PAVEMENT_L_RATIO'))
       self.pav_b_max = int(_cfg(cfg, 'LF_PAVEMENT_B_MAX'))
       self.pav_a_dev = int(_cfg(cfg, 'LF_PAVEMENT_A_DEV'))
       self.pav_bg_min_frac = float(_cfg(cfg, 'LF_PAVEMENT_BG_MIN_FRAC'))

       self.white_enabled = bool(_cfg(cfg, 'LF_WHITE_ENABLED'))
       self.white_l_contrast = int(_cfg(cfg, 'LF_WHITE_L_CONTRAST'))
       self.white_b_dev = int(_cfg(cfg, 'LF_WHITE_B_DEV'))
       self.white_a_dev = int(_cfg(cfg, 'LF_WHITE_A_DEV'))
       self.white_min_bands = int(_cfg(cfg, 'LF_WHITE_MIN_BANDS'))
       self.white_min_area_frac = float(_cfg(cfg, 'LF_WHITE_MIN_AREA_FRAC'))
       self.white_min_extent = float(_cfg(cfg, 'LF_WHITE_MIN_EXTENT'))
       self.white_max_thick_frac = float(_cfg(cfg, 'LF_WHITE_MAX_THICK_FRAC'))
       self.white_min_bottom_frac = float(_cfg(cfg, 'LF_WHITE_MIN_BOTTOM_FRAC'))
       self.white_min_aspect = float(_cfg(cfg, 'LF_WHITE_MIN_ASPECT'))
       self.white_side_contrast = int(_cfg(cfg, 'LF_WHITE_SIDE_CONTRAST'))
       self.white_stale_sec = float(_cfg(cfg, 'LF_WHITE_STALE_SEC'))
       self.white_jump_max_frac = float(_cfg(cfg, 'LF_WHITE_JUMP_MAX_FRAC'))
       self.white_takeover_sec = float(_cfg(cfg, 'LF_WHITE_TAKEOVER_SEC'))
       self.center_disagree_max_frac = float(
           _cfg(cfg, 'LF_CENTER_DISAGREE_MAX_FRAC'))
       self.white_guide_bend_frac = float(_cfg(cfg, 'LF_WHITE_GUIDE_BEND_FRAC'))
       self.lane_implausible_min = float(_cfg(cfg, 'LF_LANE_IMPLAUSIBLE_MIN'))
       self.lane_implausible_max = float(_cfg(cfg, 'LF_LANE_IMPLAUSIBLE_MAX'))
       self.lane_implausible_max_reject = int(
           _cfg(cfg, 'LF_LANE_IMPLAUSIBLE_MAX_REJECT'))
       self.lane_implausible_stale_sec = float(
           _cfg(cfg, 'LF_LANE_IMPLAUSIBLE_STALE_SEC'))
       self.lane_boundary_margin = float(_cfg(cfg, 'LF_LANE_BOUNDARY_MARGIN'))
       self.lane_boundary_kp_boost = float(
           _cfg(cfg, 'LF_LANE_BOUNDARY_KP_BOOST'))
       self.lane_boundary_slew_boost = float(
           _cfg(cfg, 'LF_LANE_BOUNDARY_SLEW_BOOST'))
       self.white_guide_sec = float(_cfg(cfg, 'LF_WHITE_GUIDE_SEC'))
       self.chord_comp = float(_cfg(cfg, 'LF_WHITE_CHORD_COMP'))
       self.white_track = bool(_cfg(cfg, 'LF_WHITE_TRACK'))
       self.white_track_jump = float(_cfg(cfg, 'LF_WHITE_TRACK_JUMP'))
       self.white_track_fresh = float(_cfg(cfg, 'LF_WHITE_TRACK_FRESH'))
       self.white_guide_disagree_max_frac = float(
           _cfg(cfg, 'LF_WHITE_GUIDE_DISAGREE_MAX_FRAC'))
       self.lane_dist_min_frac = float(_cfg(cfg, 'LF_LANE_DIST_MIN_FRAC'))
       self.lane_dist_max_frac = float(_cfg(cfg, 'LF_LANE_DIST_MAX_FRAC'))
       self.lane_dist_max_frac_lane = float(
           _cfg(cfg, 'LF_LANE_DIST_MAX_FRAC_LANE'))
       self.lane_dist_bot_min_frac = float(_cfg(cfg, 'LF_LANE_DIST_BOT_MIN_FRAC'))
       self.learn_max_heading = float(_cfg(cfg, 'LF_LANE_LEARN_MAX_HEADING'))
       self.learn_near_frac = float(_cfg(cfg, 'LF_LANE_LEARN_NEAR_FRAC'))
       # LF_LANE_WIDTH_PX: None, a single number, or a (left, right) pair.
       # The px-per-meter mapping is NOT symmetric on this camera (measured
       # 2026-07-24: the left lane spans ~170px at the bottom row from the
       # left-lane vantage, the right ~140) — one shared constant made
       # -0.5 hug the yellow while +0.5 sat correctly.
       lw = _cfg(cfg, 'LF_LANE_WIDTH_PX')
       if lw is None:
           self.lane_width_l = self.lane_width_r = None
       elif np.isscalar(lw):
           self.lane_width_l = self.lane_width_r = float(lw)
       else:
           self.lane_width_l, self.lane_width_r = float(lw[0]), float(lw[1])
       self.lane_width_px = lw   # kept for the startup log
       self.vp_y_frac = float(_cfg(cfg, 'LF_VP_Y_FRAC'))


       self.kp = float(_cfg(cfg, 'LF_STEER_KP'))
       self.kd = float(_cfg(cfg, 'LF_STEER_KD'))
       self.ki = float(_cfg(cfg, 'LF_STEER_KI'))
       self.i_max = float(_cfg(cfg, 'LF_STEER_I_MAX'))
       self.i_h_gate = float(_cfg(cfg, 'LF_STEER_I_H_GATE'))
       self.i_err_gate = float(_cfg(cfg, 'LF_STEER_I_ERR_GATE'))
       self.curve_gain = float(_cfg(cfg, 'LF_CURVE_GAIN'))
       self.curve_tc = float(_cfg(cfg, 'LF_CURVE_TC'))
       self.sharp_bend_min = float(_cfg(cfg, 'LF_SHARP_BEND_MIN'))
       self.gyro_gain = float(_cfg(cfg, 'LF_GYRO_GAIN'))
       self.gyro_tc = float(_cfg(cfg, 'LF_GYRO_FILTER_TC'))
       self.gyro_washout_tc = float(_cfg(cfg, 'LF_GYRO_WASHOUT_TC'))
       self.gyro_strip = tuple(_cfg(cfg, 'LF_GYRO_STRIP'))
       self.cam_hfov = float(_cfg(cfg, 'LF_CAM_HFOV_DEG'))
       self.temporal_jump = float(_cfg(cfg, 'LF_TEMPORAL_JUMP'))
       self.temporal_h_jump = float(_cfg(cfg, 'LF_TEMPORAL_H_JUMP'))
       self.temporal_max_reject = int(_cfg(cfg, 'LF_TEMPORAL_MAX_REJECT'))
       self.temporal_reacquire_confirm = int(
           _cfg(cfg, 'LF_TEMPORAL_REACQUIRE_CONFIRM'))
       self.kh = float(_cfg(cfg, 'LF_HEADING_GAIN'))
       self.h_clip = float(_cfg(cfg, 'LF_HEADING_CLIP'))
       self.h_bias = float(_cfg(cfg, 'LF_HEADING_BIAS'))
       self.x_tc = float(_cfg(cfg, 'LF_X_FILTER_TC'))
       self.d_tc = float(_cfg(cfg, 'LF_D_FILTER_TC'))
       self.h_tc = float(_cfg(cfg, 'LF_HEADING_FILTER_TC'))
       self.steer_slew = float(_cfg(cfg, 'LF_STEER_SLEW'))
       self.deadband = float(_cfg(cfg, 'LF_ERR_DEADBAND'))
       self.target_x = _cfg(cfg, 'LF_TARGET_X')
       self.lane_offset = float(_cfg(cfg, 'LF_LANE_OFFSET'))
       self.lane_ramp_sec = float(_cfg(cfg, 'LF_LANE_RAMP_SEC'))
       self.white_cluster_frac = float(_cfg(cfg, 'LF_WHITE_CLUSTER_FRAC'))


       self.th_max = float(_cfg(cfg, 'LF_THROTTLE_MAX'))
       self.th_min = float(_cfg(cfg, 'LF_THROTTLE_MIN'))
       self.th_min_lane_offset = float(_cfg(cfg, 'LF_THROTTLE_MIN_LANE_OFFSET'))
       self.th_lost = float(_cfg(cfg, 'LF_THROTTLE_LOST'))


       self.lost_stop_sec = float(_cfg(cfg, 'LF_LOST_STOP_SEC'))
       self.lost_recovery_sec = float(_cfg(cfg, 'LF_LOST_RECOVERY_SEC'))
       self.lost_recovery_white_age = float(
           _cfg(cfg, 'LF_LOST_RECOVERY_WHITE_AGE'))

       self.stall_detect = bool(_cfg(cfg, 'LF_STALL_DETECT'))
       self.stall_motion = float(_cfg(cfg, 'LF_STALL_MOTION'))
       self.stall_sec = float(_cfg(cfg, 'LF_STALL_SEC'))
       self.stall_kick = float(_cfg(cfg, 'LF_STALL_KICK'))
       self.stall_step_sec = float(_cfg(cfg, 'LF_STALL_STEP_SEC'))
       self.stall_kick_max = float(_cfg(cfg, 'LF_STALL_KICK_MAX'))
       self._stall_prev = None     # previous downscaled gray frame
       self._stall_since = None    # when the scene went static under power
       self._stall_boost = 0.0     # current additive throttle boost
       self._stall_step_t = 0.0    # last boost-step time

       self.reanchor_rms_frac = float(_cfg(cfg, 'LF_REANCHOR_RMS_FRAC'))
       self.reanchor_min_aspect = float(_cfg(cfg, 'LF_REANCHOR_MIN_ASPECT'))
       self.reanchor_frames = int(_cfg(cfg, 'LF_REANCHOR_FRAMES'))
       self.reanchor_max_gap_frac = float(
           _cfg(cfg, 'LF_REANCHOR_MAX_GAP_FRAC'))
       self._reanchor_streak = 0   # consecutive coherent searching frames
       self._reanchor_t = None     # last searching-frame evaluation time


       self.proc_width = _cfg(cfg, 'LF_PROC_WIDTH')
       self.overlay = bool(_cfg(cfg, 'OVERLAY_IMAGE'))


       # state
       self.steering = 0.0
       self.throttle = 0.0
       self.frames_since_fix = 10 ** 9
       self.prev_error = None
       self.i_term = 0.0        # integral term's steering contribution
       self.lost_frames = 0
       self.lost_since = None
       self.status = "init"
       self.last_x = None       # last known yellow-line x (px, full-frame)
       self.dist_left = None    # learned yellow -> nearest-left-white distance (px)
       self.dist_right = None   # learned yellow -> nearest-right-white distance (px)
       self._last_wl = None      # bottom-x of the white cluster actually
       self._last_wr = None      # SELECTED last frame, per side -- lets
                                 # _split_whites prefer continuity over
                                 # nearest-to-reference (see its comment)
       self._last_wl_t = None    # time.time() each was last genuinely
       self._last_wr_t = None    # RE-DETECTED (not just reused) -- lets
                                 # the estimate expire after going
                                 # undetected for LF_WHITE_STALE_SEC
       self.lane_pos = None     # last _lane_pos (0=yellow, 1=outer white);
                                # None outside lane-offset driving
       self._dl_buf = deque(maxlen=40)   # recent accepted distance samples;
       self._dr_buf = deque(maxlen=40)   # dist_* = median (robust to junk)
       self.x_f = None          # filtered lateral measurement (px)
       self.h_f = None          # filtered heading measurement
       self.d_f = 0.0           # filtered derivative term (steering units)
       self.last_err = 0.0      # last normalized error (HUD)
       self.last_heading = None # last raw heading (calibrate + HUD)
       self._t_prev = None      # last run() wall time, for loop dt
       self._filt_time = None   # last time the x/heading filters updated
       self._lane_applied = 0.0 # lane offset actually in force (ramped)
       self.yaw_rate_f = 0.0    # filtered vision-gyro yaw rate, rad/s (+ = right)
       self._gyro_slow = 0.0    # washout state: slow average of yaw_rate_f
       self._suspect_streak = 0 # consecutive fixes rejected by the temporal gate
       self._reacquire_streak = 0  # consecutive post-gap fixes agreeing
                                    # with each other (see
                                    # LF_TEMPORAL_REACQUIRE_CONFIRM)
       self._reacquire_x = None
       self._reacquire_heading = None
       self._implausible_streak = 0  # consecutive AGREEING fixes rejected by
                                     # the lane-geometry gate (see
                                     # LF_LANE_IMPLAUSIBLE_MAX_REJECT)
       self._implausible_x = None
       self._implausible_heading = None
       self._implausible_t = None
       self._implausible_committed = False  # latched past the gate; see
                                            # the gate in _run_frame
       self.last_npts = 0       # telemetry: raw detector output this frame,
       self.last_nblob = 0      # before any gate (see detect())
       self.last_nwhite = 0
       self.last_raw_found = False
       self.h_slow = 0.0        # sustained heading lean (curve feed-forward)
       self._last_guided = 0.0  # wall time of the last white-guided steer
       self.white_x_f = None    # near-white tracker: filtered bottom x
       self.white_h_f = None    # near-white tracker: filtered heading
       self._white_seen = None  # wall time the tracker last matched a cluster
       self._white_pts_seen = None  # wall time _detect_white last returned ANY
                                    # centroid (raw sighting, no gates — feeds
                                    # the lost-recovery crawl condition only)
       self._gyro_prev = None   # previous far-field strip (float32 gray)
       self._gyro_win = None    # Hanning window for phaseCorrelate

       logger.info(
           "LineFollower up: kp=%.2f kd=%.2f ki=%.2f kh=%.2f "
           "lane_offset=%+.2f lane_width=%s target_x=%s roi_top=%.2f "
           "proc_width=%s (values come from myconfig.py when set there — "
           "DEFAULTS edits do not apply if myconfig defines the key)",
           self.kp, self.kd, self.ki, self.kh, self.lane_offset,
           self.lane_width_px, self.target_x, self.roi_top, self.proc_width)


       # morphology kernel, sized on first frame (cleans mask speckle).
       # Must scale with resolution: at 160x120 the distant dashes are only
       # 2-4 px wide and a 5x5 opening erases them entirely.
       self._kernel = None
       self._kernel_small = None


   # ------------------------------------------------------------------ #
   # detection                                                          #
   # ------------------------------------------------------------------ #
   def _filter_by_shape(self, mask):
       """
       Drop mask blobs that are the wrong shape to be a ground-plane dash.

       A dash is small when far from the camera (near the top of the ROI)
       and grows as it approaches the bottom. Background clutter that
       happens to match the tape's color (bushes, dry leaves, patio
       furniture, a person's shirt) sits behind/above the track and gets
       clipped by the ROI's top edge instead of shrinking with distance, so
       it looks tall for how close to the ROI top it is. Reject any blob
       whose bounding-box height, relative to its distance from the ROI
       top, exceeds what a real dash ever measures.
       """
       n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
       min_area = self.min_blob_area_frac * mask.shape[0] * mask.shape[1]
       out = mask
       for i in range(1, n):
           area = stats[i, cv2.CC_STAT_AREA]
           bh = stats[i, cv2.CC_STAT_HEIGHT]
           top = stats[i, cv2.CC_STAT_TOP]
           cy = top + bh / 2.0
           reject = (area < min_area) or (bh / max(cy, 1.0) > self.max_blob_height_ratio)
           if reject:
               if out is mask:
                   out = mask.copy()
               out[labels == i] = 0
       return out

   def _filter_on_pavement(self, mask, L, A, B):
       """
       Drop mask blobs that are NOT sitting on the paved track.

       A genuine dash is painted on grey pavement, so the area immediately
       around it is colour-neutral and (sharing the dash's illumination)
       nearly as bright as the dash itself. Debris that shares the tape's
       colour (dead leaves, mulch, grass tips) lies in the tan gravel bed
       or green planter OUTSIDE the white boundary lines, so its
       surroundings are tan (yellowish b-channel) or dark green. For each
       blob we sample a window around it and keep it only if enough of that
       background reads as pavement. See LF_REQUIRE_PAVEMENT for the full
       rationale; the brightness test is relative to the blob so it holds
       in shadow.
       """
       # colour-neutral background (no yellow/tan tint, not green/red)
       neutral = ((B < self.pav_b_max)
                  & (np.abs(A.astype(np.int16) - 128) < self.pav_a_dev))
       rh, rw = mask.shape
       n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
       out = mask
       for i in range(1, n):
           cx, cy = int(cent[i][0]), int(cent[i][1])
           # window sized from the blob itself, so even a blob larger than
           # any fixed window is judged against its real surroundings (a
           # yellow logo on a shirt used to fill a fixed window completely,
           # leave no background pixels to judge, and pass by default)
           bw = stats[i, cv2.CC_STAT_WIDTH]
           bh = stats[i, cv2.CC_STAT_HEIGHT]
           m = max(6, int(round(rw * 0.023)), int(round(0.75 * max(bw, bh))))
           x0, x1 = max(0, cx - m), min(rw, cx + m)
           y0, y1 = max(0, cy - m), min(rh, cy + m)
           bg = mask[y0:y1, x0:x1] == 0          # background = non-mask pixels in window
           n_bg = int(bg.sum())
           if n_bg < 20:
               continue                          # too little background to judge; keep
           blob_l = float(np.median(L[labels == i]))
           bright = L[y0:y1, x0:x1] >= self.pav_l_ratio * blob_l
           pav_frac = float((neutral[y0:y1, x0:x1] & bright)[bg].mean())
           if pav_frac < self.pav_bg_min_frac:
               if out is mask:
                   out = mask.copy()
               out[labels == i] = 0
       return out

   def _band_centroids(self, mask, split=False):
       """
       Split a mask into horizontal bands and return per-band centroids as
       (cx, cy, weight) tuples. With split=True, connected components in a
       band are returned as separate centroids (used for the white lines,
       where the left and right line cross the same band); otherwise the
       band is averaged into one centroid (used for the single yellow line).
       """
       roi_h = mask.shape[0]
       band_h = max(1, roi_h // self.num_bands)
       pts = []
       for b in range(self.num_bands):
           ys, ye = b * band_h, min((b + 1) * band_h, roi_h)
           band = mask[ys:ye]
           if not split:
               n = cv2.countNonZero(band)
               if n < self.min_band_px:
                   continue
               m = cv2.moments(band, binaryImage=True)
               pts.append((m['m10'] / m['m00'], ys + m['m01'] / m['m00'], n))
           else:
               nc, _, stats, cent = cv2.connectedComponentsWithStats(band, connectivity=8)
               for i in range(1, nc):
                   n = stats[i, cv2.CC_STAT_AREA]
                   if n < self.min_band_px:
                       continue
                   pts.append((cent[i][0], ys + cent[i][1], n))
       return pts

   def _detect_white(self, Lc, Ac, Bc_abs, bgB, L):
       """
       Mask the solid white boundary lines by local contrast: brighter than
       the surrounding pavement, chroma close to the local background's
       (so shadow's blue cast doesn't disqualify them; the blue tape on the
       track deviates in b and is excluded). Returns per-band centroids,
       possibly several per band (left and right line).

       L is the raw (unblurred, un-differenced) lightness channel — the
       flank test below compares a blob against pixels a few px to its
       left/right, which only means anything in absolute values.
       """
       Bdev = np.abs(Bc_abs.astype(np.int16) - bgB.astype(np.int16))
       white = ((Lc >= self.white_l_contrast)
                & (Bdev <= self.white_b_dev)
                & (np.abs(Ac.astype(np.int16) - 128) <= self.white_a_dev)
                ).astype(np.uint8) * 255
       # light cleanup only: the far line is 2px wide and a 5x5 open erases it
       white = cv2.morphologyEx(white, cv2.MORPH_OPEN, self._kernel_small)
       # structure filter: keep only long connected streaks (see
       # LF_WHITE_MIN_AREA_FRAC) — glints, trim, and clothing highlights
       # pass the contrast test but only as small speckles
       rh, rw = white.shape
       min_area = self.white_min_area_frac * rh * rw
       min_ext = self.white_min_extent * rh
       max_thick = self.white_max_thick_frac * rw
       min_bottom = self.white_min_bottom_frac * rh
       n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
       for i in range(1, n):
           bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
           ext = max(bw, bh)
           # wide-and-flat blobs are pavement seams / crosswalk bands /
           # sun-shadow boundaries lying ACROSS the track (session 13
           # showed one spanning the whole ROI top) — a lane boundary
           # never presents as a near-horizontal full-width stripe
           seam = bw >= 0.5 * rw and bh <= 0.15 * rh
           # wall bases / kiosk feet / building trim pass the contrast
           # test but live only in the TOP band of the ROI (their ground
           # contact is near the horizon); a real boundary line runs
           # down toward the car (see LF_WHITE_MIN_BOTTOM_FRAC)
           high = (stats[i, cv2.CC_STAT_TOP] + bh) < min_bottom
           if stats[i, cv2.CC_STAT_AREA] < min_area or ext < min_ext \
                   or seam or high:
               white[labels == i] = 0
               continue
           # painted lines are THIN per image row; kiosk panels, signage
           # and elevator trim are solid slabs (LF_WHITE_MAX_THICK_FRAC)
           ys_i, xs_i = np.nonzero(labels == i)
           row_w = np.bincount(ys_i)
           med_w = float(np.median(row_w[row_w > 0]))
           if med_w > max_thick:
               white[labels == i] = 0
               continue
           # ...and much LONGER than they are thick: the PCA axis ratio
           # of the blob's pixel cloud must clear LF_WHITE_MIN_ASPECT.
           # Light-grey plaza pavers / pavement patches / stone walls
           # (2026-07-29 footage) pass every test above but present as
           # 2-D regions (ratio ~1-3); every real stroke of the boundary
           # line measured across today's footage — including the worn,
           # fragmented, near-horizontal pieces at idx 163360 that a
           # bbox/row-width version of this test wrongly killed — reads
           # 4.5-22 at any angle.
           cov = np.cov(np.stack((xs_i, ys_i)).astype(np.float64))
           ev = np.linalg.eigvalsh(cov)
           if float(np.sqrt(ev[1] / max(ev[0], 1e-6))) < self.white_min_aspect:
               white[labels == i] = 0
               continue
           if bw > bh:      # near-horizontal: slice per column, flank in y
               prim, sec, n_prim, n_sec = xs_i, ys_i, rw, rh
           else:            # near-vertical: slice per row, flank in x
               prim, sec, n_prim, n_sec = ys_i, xs_i, rh, rw
           thick = np.bincount(prim)
           med_t = float(np.median(thick[thick > 0]))
           # ...and brighter than the pavement on BOTH flanks of its
           # THIN dimension. The border strip of a large bright region
           # (plaza edge, aluminium wall trim, the lit face of a planter
           # wall) is exactly as bright/neutral/thin as paint, but its
           # parent region sits on one flank (LF_WHITE_SIDE_CONTRAST).
           # Flanks must be sampled ACROSS the stroke, not along it: a
           # near-vertical line is judged left/right per row, a
           # near-horizontal one (the boundary sweeping across the frame
           # bottom in a sharp bend) above/below per column — sampling
           # a horizontal line at its row ends lands on more line/worn
           # paint and killed the REAL boundary at idx 163360. Compare L
           # at each slice's centre against L just outside the slice's
           # span on both flanks; a flank is judged only where the
           # sample lands in-frame (a line hugging the frame edge keeps
           # its one judgeable flank). Each flank is sampled at TWO
           # distances — just outside the stroke, and again about a
           # stroke-width farther out: plaza pavers pass the near sample
           # (it lands in the dark grout joint between pavers) but the
           # far sample lands on the NEXT bright paver and fails; real
           # paint just sees more pavement at both distances. The
           # nearest other line on this track is ≥ a lane-distance away
           # (LF_LANE_DIST_MIN_FRAC), far beyond both samples.
           slices = np.unique(prim)
           if len(slices) > 48:                     # cap per-blob cost
               slices = slices[:: len(slices) // 48 + 1]
           smin = np.full(n_prim, n_sec, dtype=np.int32)
           smax = np.full(n_prim, -1, dtype=np.int32)
           np.minimum.at(smin, prim, sec)
           np.maximum.at(smax, prim, sec)
           gap = max(3, int(round(med_t)))
           gap2 = gap + max(4, gap)
           mid = (smin[slices] + smax[slices]) // 2
           if bw > bh:
               cL = L[mid, slices].astype(np.int16)
           else:
               cL = L[slices, mid].astype(np.int16)
           for sf in (smin[slices] - gap, smin[slices] - gap2,
                      smax[slices] + gap, smax[slices] + gap2):
               ok = (sf >= 0) & (sf < n_sec)
               if int(ok.sum()) < max(3, int(0.3 * len(slices))):
                   continue                         # not judgeable; keep
               if bw > bh:
                   flank = L[sf[ok], slices[ok]].astype(np.int16)
               else:
                   flank = L[slices[ok], sf[ok]].astype(np.int16)
               if float(np.median(cL[ok] - flank)) < self.white_side_contrast:
                   white[labels == i] = 0
                   break
       return white, self._band_centroids(white, split=True)

   def detect(self, rgb_img):
       """
       Find the hashed line in the bottom part of the frame.


       Returns (found, error_px, heading, debug) where
         error_px : x position of the line near the car (px, full-frame
                    coords), or None when not found. run() turns this into
                    the normalized steering error against the target.
         heading  : slope of the line, in normalized-x per ROI-height;
                    >0 means the line leans right as it goes away from the car.
         debug    : dict for the overlay, incl. white-line centroids.
       """
       h, w = rgb_img.shape[:2]
       y0 = int(h * self.roi_top)
       roi = rgb_img[y0:, :]


       lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
       L, A, B = cv2.split(lab)
       hue = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)[:, :, 0]

       # Local background estimate: median blur with a window wider than a
       # dash (a dash fills <30% of it, so the median ignores the dash).
       # This is what makes detection survive the sun/shadow boundary: each
       # pixel is compared against pavement in ITS OWN illumination.
       k_bg = (w // 6) | 1
       # bgB_raw feeds the WHITE detector; bgB (clamped) feeds the YELLOW
       # contrast path. The clamp exists solely to stop the blue tape
       # patches from dragging the yellow's background down (see
       # LF_BG_B_CLAMP) -- but the white detector's "chroma close to the
       # local background" test was being fed the same clamped field, so
       # in shadow (real pavement b ~118-120, below the clamp) the white
       # lines were judged against a background that doesn't exist and
       # the b-deviation test mis-fired both ways. Two estimates of the
       # same quantity for two different purposes: keep them separate.
       bgB_raw = cv2.medianBlur(B, k_bg)
       bgB = np.maximum(bgB_raw, self.bg_b_clamp)
       bgL = cv2.medianBlur(L, k_bg)
       Bc = B.astype(np.int16) - bgB.astype(np.int16)
       Lc = L.astype(np.int16) - bgL.astype(np.int16)

       yellow = ((Bc >= self.b_contrast) & (B >= self.b_floor)) | (B >= self.b_min)
       mask = (yellow & (A <= self.a_max) & (L >= self.l_min)
               & (hue >= self.hue_min) & (hue <= self.hue_max)
               ).astype(np.uint8) * 255


       if self._kernel is None:
           k = self.morph_kernel or (5 if w >= 240 else 3)
           self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
           self._kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
       mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
       mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
       mask = self._filter_by_shape(mask)
       if self.require_pavement:
           mask = self._filter_on_pavement(mask, L, A, B)


       roi_h = roi.shape[0]
       pts = self._band_centroids(mask)


       debug = dict(mask=mask, roi_y0=y0, pts=pts, w=w, h=h, white_pts=[])

       if self.white_enabled:
           wmask, wpts = self._detect_white(Lc, A, B, bgB_raw, L)
           debug['white_mask'] = wmask
           debug['white_pts'] = wpts


       # Observational only, for the telemetry log (lf_telemetry.py): what
       # the DETECTOR saw this frame, before any gate touched it. Telling
       # "the detector saw nothing" apart from "a gate threw a good fix
       # away" is the single most useful distinction when triaging a lost
       # line, and it cannot be reconstructed after the fact -- three
       # separate root causes so far (morphology erasing thin line, the
       # lane-implausibility lockout, temporal-gate rejections) all look
       # identical in the tub, which records only steering and throttle.
       self.last_npts = len(pts)
       self.last_nwhite = len(debug['white_pts'])
       self.last_nblob = 0
       self.last_raw_found = False

       self.frames_since_fix += 1

       if len(pts) < self.min_bands:
           return False, None, 0.0, debug


       pts_a = np.array(pts, dtype=np.float64)
       cxs, cys, ws = pts_a[:, 0], pts_a[:, 1], pts_a[:, 2]


       # Weighted least-squares fit x = a*y + b through the band centroids.
       # Weight by pixel count AND by closeness to the car (bottom of ROI):
       # near dashes matter more for lateral error.
       wgt = ws * (0.5 + cys / roi_h)
       a, bfit = np.polyfit(cys, cxs, 1, w=np.sqrt(wgt))


       # One pass of outlier rejection: vegetation/gravel at the frame edge
       # can hijack a band's centroid. Drop bands far from the fitted line
       # and refit with the rest.
       if len(pts) > self.min_bands:
           resid = np.abs(cxs - (a * cys + bfit))
           keep = resid < max(0.12 * w, 1.5 * np.median(resid) + 1)
           if keep.sum() >= self.min_bands and keep.sum() < len(pts):
               cxs, cys, ws, wgt = cxs[keep], cys[keep], ws[keep], wgt[keep]
               a, bfit = np.polyfit(cys, cxs, 1, w=np.sqrt(wgt))
               debug['pts'] = [p for p, k in zip(pts, keep) if k]


       # Lateral position: project the fit to the BOTTOM of the frame —
       # that's where the car is. The nearest detected dash can sit well up
       # the ROI (the gap between dashes), so an average of near centroids
       # reads the x of a point AHEAD of the car; steering then centers
       # that point, not the car, and on any slanted line (curves, camera
       # yaw) that is a systematic lateral offset. The centroid average
       # bounds the extrapolation, which blows up when the line runs
       # nearly horizontal in the frame (sharp curves). The bound is a
       # CLAMP, not a switch: the old either/or between two estimators up
       # to 57px apart toggled frame to frame (69%/31% in the afternoon
       # sessions) and was itself a large noise source.
       order = np.argsort(cys)[::-1]          # nearest (largest y) first
       near = order[:3]
       x_near = float(np.average(cxs[near], weights=ws[near]))
       x_bottom = float(a * roi_h + bfit)
       x_near += float(np.clip(x_bottom - x_near, -0.15 * w, 0.15 * w))

       # A real sighting of a DASHED line is multiple separate dashes; see
       # LF_MIN_DASHES. A single blob only counts while already tracking,
       # and only near where the line was last seen. In a turn the line
       # sweeps laterally fast and dashes leave the frame edge (61% of
       # loss episodes happened at |heading|>0.3), so the allowed jump
       # widens with the current heading — a lone dash at the frame edge
       # keeps the fix alive through the corner.
       n_blobs = cv2.connectedComponents(mask, connectivity=8)[0] - 1
       self.last_nblob = int(n_blobs)
       if n_blobs < self.min_dashes:
           recent = self.frames_since_fix <= self.single_dash_grace
           allow = self.single_dash_max_jump * w
           if self.h_f is not None:
               allow *= 1.0 + min(abs(self.h_f - self.h_bias), 1.0)
           near_last = (self.last_x is not None
                        and abs(x_near - self.last_x) <= allow)
           if not (recent and near_last):
               return False, None, 0.0, debug


       # heading: slope, normalized. a is px-x per px-y; y grows downward, so
       # a > 0 means the line moves right toward the car => it leans LEFT
       # ahead of the car => steer left. Negate to get "lean of the road ahead".
       #
       # Convergence correction: the image slope of a ground line depends
       # on WHERE the line is laterally — every line parallel to the car
       # converges at the vanishing point, so a line ridden at an offset
       # (lane driving) leans in the image even when the car is perfectly
       # aligned with it. Riding ~60px beside the yellow faked a ~0.35
       # heading, and the controller "fixed" that lean by drifting back
       # toward the line — the lane offset never held its width (the
       # "hugs the dashed line" bug). Verified on sessions 12-15: heading
       # vs lateral position slope ~-0.006/px, exactly the vanishing-point
       # prediction, implied horizon 59% down the frame (measured ~55%).
       # So subtract the slope a parallel line WOULD have at this lateral
       # position: a_exp = (x - vp_x) / (rows between horizon and bottom).
       vp_x = (w / 2.0) if self.target_x is None else float(self.target_x)
       a_exp = (x_near - vp_x) / max(h * (1.0 - self.vp_y_frac), 1.0)
       # Below LF_MIN_HEADING_SPREAD_FRAC of the ROI, "a" is a slope fit
       # through points bunched in a narrow y-band -- a couple pixels of
       # centroid noise swings it wildly (see LF_MIN_HEADING_SPREAD_FRAC).
       # Report NOT-MEASURED rather than propagate a fit this
       # ill-conditioned; x_near (position) is unaffected and still used.
       #
       # This used to return 0.0, which is a LIE that reads as a
       # measurement: on session 26-07-30_91 the guard fired on 37% of all
       # detections, so the reported heading toggled between "0.0" and its
       # real value (pegged at the +/-1.0 clip on 17% of frames) 239 times
       # -- and the temporal gate below read each toggle as the line
       # teleporting and threw away the WHOLE fix, good position included.
       # 245 of that run's 306 gate rejections were heading-triggered, half
       # of them on exactly this 0.0 placeholder, against a previous
       # heading sitting at the clip 88% of the time. None also stops the
       # fake zero being folded into h_f (which _update_filters already
       # skips for None) and thus out of the curve feed-forward.
       if (cys.max() - cys.min()) < self.min_heading_spread_frac * roi_h:
           heading = None
       else:
           heading = float(np.clip(-(a - a_exp) * roi_h / (w / 2.0), -1.0, 1.0))


       self.frames_since_fix = 0
       self.last_raw_found = True
       debug['fit'] = (a, bfit)
       debug['x_near'] = x_near
       return True, x_near, heading, debug


   # ------------------------------------------------------------------ #
   # white-line fallback helpers                                        #
   # ------------------------------------------------------------------ #
   def _cluster_bottom_x(self, pts, roi_h, w):
       """
       Project a white cluster's centroids to the ROI bottom, the same way
       the yellow is projected. Lane distances must compare line positions
       AT THE SAME image row: band centroids sit mid-ROI where perspective
       has already narrowed the lane, so the raw centroid average
       understated the distances by ~30% (learned 76/88px vs a true
       106/128 at the bottom row on sessions 12-15) — and the commanded
       lane offset came out equally short. Falls back to the near-centroid
       average, which also clamps the fit's extrapolation exactly like the
       yellow's projection.
       """
       arr = np.array(pts, dtype=np.float64)
       cxs, cys, ws = arr[:, 0], arr[:, 1], arr[:, 2]
       order = np.argsort(cys)[::-1]
       sel = order[:3]
       near = float(np.average(cxs[sel], weights=ws[sel]))
       if len(pts) < 2 or np.ptp(cys) < 1e-6:
           return near
       wgt = ws * (0.5 + cys / roi_h)
       a, b = np.polyfit(cys, cxs, 1, w=np.sqrt(wgt))
       xb = float(a * roi_h + b)
       return near + float(np.clip(xb - near, -0.15 * w, 0.15 * w))

   def _split_whites(self, white_pts, yellow_fit, w, roi_h, now):
       """
       Group white-line band centroids into distinct LINES on each side of
       the yellow (using the yellow fit when available, else the last known
       yellow x) and return, per side, the bottom-projected x of the line
       closest to the yellow plus a flag saying whether that line was seen
       in the near field (bottom half of the ROI) — only near-field
       sightings are allowed to teach lane widths.

       The track has more white than the two lane boundaries: a second
       parallel line to the right, markings, and neighboring-track paint.
       All of it lands in white_pts. The old average-everything-per-side
       blended those lines into a distance matching none of them (replayed
       learned distances wandered 66..166px vs a true ~85/~108). So the
       per-side centroids are clustered by their x-offset from the yellow
       fit (gaps over LF_WHITE_CLUSTER_FRAC split clusters), clusters whose
       median offset is outside the plausible lane-distance range are
       dropped, and among what's left, continuity wins: the cluster nearest
       to whichever line we actually picked LAST frame is preferred over
       whichever is nearest the yellow reference this instant. Without this,
       a diverging line (a branch/curb peeling off the real boundary) can
       be geometrically nearest the yellow reference for a frame or two
       right where the real boundary curves away, and the picked line
       silently swaps to it. (Continuity + stale expiry restored
       2026-07-28: lost in the 07-26 code pull.)
       """
       off = []                       # (dx from yellow, cx, cy, weight)
       for cx, cy, n in white_pts:
           if yellow_fit is not None:
               a, b = yellow_fit
               ref = a * cy + b
           elif self.last_x is not None:
               ref = self.last_x
           else:
               continue
           off.append((cx - ref, cx, cy, n))

       # See LF_LANE_DIST_MAX_FRAC_LANE: the plain ceiling leaves the left
       # lane's real, chord-distorted geometry almost no margin. Only
       # widen it while lane-offset driving is actually engaged, so
       # center-line (LF_LANE_OFFSET=0.0) white-fallback behavior is
       # untouched.
       max_frac = (self.lane_dist_max_frac_lane
                   if abs(self._lane_applied) >= 0.25
                   else self.lane_dist_max_frac)
       lo, hi = self.lane_dist_min_frac * w, max_frac * w
       gap = self.white_cluster_frac * w

       def near_x(side, last_val):
           if len(side) < self.white_min_bands:
               return None, []
           side = sorted(side)                     # by dx
           clusters, cur = [], [side[0]]
           for p in side[1:]:
               if p[0] - cur[-1][0] > gap:
                   clusters.append(cur)
                   cur = [p]
               else:
                   cur.append(p)
           clusters.append(cur)
           candidates = []
           for cl in clusters:
               if len(cl) < self.white_min_bands:
                   continue
               d = abs(float(np.median([p[0] for p in cl])))
               if not (lo < d < hi):
                   continue
               pts = [(p[1], p[2], p[3]) for p in cl]
               candidates.append((d, self._cluster_bottom_x(pts, roi_h, w), pts))
           if not candidates:
               return None, False
           if last_val is not None:
               # continuity: stick with whichever cluster is closest to the
               # line we were actually tracking last frame
               _, bx, pts = min(candidates, key=lambda c: abs(c[1] - last_val))
               # ...but only if even the BEST-available candidate is still
               # plausibly the same physical line. Session 26-07-24_47,
               # right at the crossing: the left side lost all detection
               # (wl froze, correctly, on nothing to pick from) while the
               # right side kept finding SOMETHING every frame -- just
               # fragments of paint/debris, not a real line -- and
               # continuity happily picked the least-bad fragment each
               # time even though it jumped 50-150px frame to frame. That
               # noise random-walked the position estimate to saturation
               # and the car timed out. A real line can't jump this far in
               # one frame; if nothing qualifies, report no detection
               # (None) rather than tracking noise.
               if abs(bx - last_val) > self.white_jump_max_frac * w:
                   return None, False
           else:
               # cold start: no prior line to be continuous with, fall back
               # to nearest-the-yellow-reference (the old behavior)
               _, bx, pts = min(candidates, key=lambda c: c[0])
           # a cluster observed only far up the ROI anchors its bottom
           # projection poorly — extrapolation from upper-row clutter is
           # how session 13 learned dist_l=153 (curb + pavement-seam
           # pixels). Such a fix may still bridge a dropout, but teaching
           # lane widths takes a sighting reaching LF_LANE_LEARN_NEAR_FRAC
           # of the ROI.
           near_ok = max(p[1] for p in pts) >= self.learn_near_frac * roi_h
           return bx, near_ok

       wl_last = (None if self._last_wl_t is not None
                  and now - self._last_wl_t > self.white_stale_sec
                  else self._last_wl)
       wr_last = (None if self._last_wr_t is not None
                  and now - self._last_wr_t > self.white_stale_sec
                  else self._last_wr)
       wl, wl_ok = near_x([p for p in off if p[0] < 0], wl_last)
       wr, wr_ok = near_x([p for p in off if p[0] >= 0], wr_last)
       if wl is not None:
           self._last_wl, self._last_wl_t = wl, now
       elif wl_last is None:
           self._last_wl = None    # expired and nothing fresh -- forget it
       if wr is not None:
           self._last_wr, self._last_wr_t = wr, now
       elif wr_last is None:
           self._last_wr = None
       return wl, wr, wl_ok, wr_ok

   def _learn_lane(self, x_yellow, wl, wr, w):
       """Learn the yellow-to-white distance on each side while tracking.
       Both inputs are bottom-projected, so this is the lane width AT THE
       CAR — the row where the steering error lives. Samples outside the
       plausible range (LF_LANE_DIST_BOT_MIN_FRAC..LF_LANE_DIST_MAX_FRAC)
       are misclassified speckles or a neighboring track's line — skip
       them. The estimate is the median of a ring of recent samples: an
       EMA let every junk sample through (each nudged it 10%, and
       stretches of junk walked it anywhere); a median needs a majority of
       the window to be wrong before it moves, yet still tracks slow real
       change. Requiring several samples before publishing keeps the lane
       offset from engaging off one glimpse. The caller additionally
       freezes learning while turning: mid-turn the projected geometry is
       chord-vs-arc distorted (measured 177/102 vs 106/128 straight), and
       letting those samples in dragged the target 30-50px toward the
       yellow exactly in corners — which is what pushed the car across
       the center line into the wrong lane on turns."""
       lo, hi = self.lane_dist_bot_min_frac * w, self.lane_dist_max_frac * w
       if wl is not None and lo < x_yellow - wl < hi:
           self._dl_buf.append(x_yellow - wl)
           if len(self._dl_buf) >= 8:
               self.dist_left = float(np.median(self._dl_buf))
       if wr is not None and lo < wr - x_yellow < hi:
           self._dr_buf.append(wr - x_yellow)
           if len(self._dr_buf) >= 8:
               self.dist_right = float(np.median(self._dr_buf))

   def _estimate_from_whites(self, wl, wr, w):
       """
       Estimate where the yellow line is from the white lines and the lane
       width (calibrated LF_LANE_WIDTH_PX, else the learned per-side
       distances). Returns x estimate or None.
       """
       dl = self.lane_width_l if self.lane_width_l is not None else self.dist_left
       dr = self.lane_width_r if self.lane_width_r is not None else self.dist_right
       est_l = wl + float(dl) if wl is not None and dl is not None else None
       est_r = wr - float(dr) if wr is not None and dr is not None else None
       # When lane-offset driving, trust only the NEAR boundary (the one
       # the car rides beside): it is close, solid, and unambiguous. The
       # far side of a stale yellow reference is where misclassified
       # clusters live (second parallel line, the other lane's boundary),
       # and averaging a wrong hypothesis in drags the estimate a half
       # lane sideways exactly during blind turns.
       if self._lane_applied <= -0.25:
           est = est_l if est_l is not None else est_r
       elif self._lane_applied >= 0.25:
           est = est_r if est_r is not None else est_l
       elif est_l is not None and est_r is not None:
           # Center-line driving had no version of the guard above: at a
           # wide-open crossing (session 26-07-24_49) one side can lock
           # onto the ADJACENT lane's boundary across the gap -- both
           # est_l/est_r still individually look like plausible in-range
           # white lines, so nothing upstream rejects it, and blindly
           # averaging a genuine near-boundary estimate with a
           # neighboring-lane one pulls the target off center toward
           # whichever lane the bad side belongs to (reported as "hugs
           # the inner part of the lane, starts seeing the other side").
           # Two genuine estimates of the SAME yellow line should roughly
           # agree; a wide split means one of them isn't measuring it.
           # Trust whichever is closer to the (gyro-dead-reckoned) last
           # known yellow instead of splitting the difference.
           # (Restored 2026-07-28: lost in the 07-26 code pull.)
           anchor = self.last_x
           if (anchor is not None
                   and abs(est_l - est_r) > self.center_disagree_max_frac * w):
               est = est_l if abs(est_l - anchor) <= abs(est_r - anchor) \
                   else est_r
           else:
               est = 0.5 * (est_l + est_r)
       else:
           est = est_l if est_l is not None else est_r
       return None if est is None else float(est)


   # ------------------------------------------------------------------ #
   # vision gyro                                                        #
   # ------------------------------------------------------------------ #
   def _measure_yaw_rate(self, rgb_img, dt):
       """
       Camera yaw rate from the frame-to-frame horizontal shift of the
       far background (see LF_GYRO_GAIN). Positive = yawing right.
       Returns 0.0 when there is nothing trackable (first frame, heavy
       blur, an occluder filling the strip): zero damping is the safe
       degradation, the PID still runs.
       """
       h, w = rgb_img.shape[:2]
       y0, y1 = int(self.gyro_strip[0] * h), int(self.gyro_strip[1] * h)
       strip = cv2.cvtColor(rgb_img[y0:y1], cv2.COLOR_RGB2GRAY).astype(np.float32)
       prev, self._gyro_prev = self._gyro_prev, strip
       if prev is None or prev.shape != strip.shape:
           self._gyro_win = cv2.createHanningWindow(
               (strip.shape[1], strip.shape[0]), cv2.CV_32F)
           return 0.0
       (dx, _dy), resp = cv2.phaseCorrelate(prev, strip, self._gyro_win)
       if resp < 0.1:
           return 0.0   # no dominant shift peak — strip content unusable
       # scene shifts LEFT when the car yaws RIGHT; f = w/2 / tan(HFOV/2)
       f = (w / 2.0) / np.tan(np.radians(self.cam_hfov) / 2.0)
       return float(np.clip(-dx / f / max(dt, 1e-3), -4.0, 4.0))

   def _ego_near_boundary(self, white_pts, w, roi_h):
       """
       Identify the lane's NEAR white boundary without any yellow
       reference, from where it SHOULD be relative to the car.

       _split_whites sides its clusters off the last yellow position;
       when the car has been blind through most of a bend that reference
       is stale garbage and every real cluster gets rejected (session
       37: reference dead-reckoned past x=700, whites in view, zero
       clusters accepted for 2s, car stopped mid-track). But while
       lane-offset driving we know where the near boundary belongs in
       EGO coordinates — riding -0.75 of a 170px lane puts the left
       white ~42px left of the car's own column — so cluster the white
       centroids by x alone and take the cluster whose bottom projection
       lands nearest that expectation. Returns (wl, wr) with only the
       near side filled, or (None, None).
       """
       if abs(self._lane_applied) < 0.25 or not white_pts:
           return None, None
       if self._lane_applied < 0:
           width = self.lane_width_l if self.lane_width_l is not None \
               else self.dist_left
       else:
           width = self.lane_width_r if self.lane_width_r is not None \
               else self.dist_right
       if width is None:
           return None, None
       target_yellow = (w / 2.0 if self.target_x is None
                        else float(self.target_x)) \
           - self._lane_applied * float(width)
       if self._lane_applied < 0:
           expect = target_yellow - float(width)
       else:
           expect = target_yellow + float(width)
       pts = sorted(white_pts)                       # by cx
       gap = self.white_cluster_frac * w
       clusters, cur = [], [pts[0]]
       for p in pts[1:]:
           if p[0] - cur[-1][0] > gap:
               clusters.append(cur)
               cur = [p]
           else:
               cur.append(p)
       clusters.append(cur)
       best, best_d = None, 0.2 * w   # must land within 0.2w of expectation
       for cl in clusters:
           if len(cl) < self.white_min_bands:
               continue
           bx = self._cluster_bottom_x(cl, roi_h, w)
           if abs(bx - expect) < best_d:
               best, best_d = bx, abs(bx - expect)
       if best is None:
           return None, None
       return (best, None) if self._lane_applied < 0 else (None, best)

   def _near_lane_width(self):
       """Pinned/learned width of the lane the current offset rides in,
       or None. Negative offsets ride left of the yellow."""
       if self._lane_applied < 0:
           return self.lane_width_l if self.lane_width_l is not None \
               else self.dist_left
       return self.lane_width_r if self.lane_width_r is not None \
           else self.dist_right

   def _update_white_tracker(self, white_pts, w, h, roi_h, now, dt):
       """
       Persistent tracker on the lane's NEAR white boundary (see
       LF_WHITE_TRACK). Matches a cluster to the tracker's own
       gyro-predicted previous position (ego-frame expectation when the
       tracker is cold), fits position and heading exactly like the
       yellow, and EMAs both. Runs every frame while lane-offset
       driving, tracked or not, so it is already warm when the yellow
       drops out mid-bend.
       """
       if (not self.white_track or abs(self._lane_applied) < 0.25
               or not white_pts):
           return
       width = self._near_lane_width()
       if width is None:
           return
       target = (w / 2.0) if self.target_x is None else float(self.target_x)
       target_yellow = target - self._lane_applied * float(width)
       expect = target_yellow - float(width) if self._lane_applied < 0 \
           else target_yellow + float(width)
       f_px = (w / 2.0) / np.tan(np.radians(self.cam_hfov) / 2.0)
       fresh = (self._white_seen is not None
                and now - self._white_seen <= 0.5)
       ref = self.white_x_f if fresh and self.white_x_f is not None else expect
       pred = ref - f_px * self.yaw_rate_f * dt
       allow = self.white_track_jump * w + abs(f_px * self.yaw_rate_f * dt)
       if not fresh:
           # Re-acquiring cold, not continuing a warm track: LF_WHITE_
           # TRACK_JUMP is sized for frame-to-frame continuity (how far a
           # real boundary can move between two consecutive frames), not
           # for how far the SITTING "expect" estimate can be from the
           # real boundary after the tracker has been stale a while.
           # Once stale, ref/pred stay pinned at the static ego-frame
           # "expect" (only nudged by one frame's gyro shift per call,
           # never compounded across the stale stretch), while a
           # sustained turn keeps moving the real boundary in the image
           # -- so the gap to close only grows, and the tight jump
           # tolerance never bridges it. Verified on real footage: a
           # sustained turn left the tracker stale (white_x_f frozen)
           # for 2.5+ seconds while a plain, ungated per-frame side-split
           # of the SAME points kept finding the boundary every single
           # frame just fine -- the boundary was never actually hard to
           # see, only hard to re-acquire against a tolerance sized for
           # tracking, not searching. Reuses the lane-distance bound
           # already trusted for "is this cluster plausibly the lane
           # boundary at all" (LF_LANE_DIST_MAX_FRAC_LANE -- this
           # function already only runs while lane-offset driving is
           # engaged, see the guard above) instead of a new constant --
           # a generic, geometry-scaled search radius, not anything
           # specific to one camera or track.
           allow = max(allow, self.lane_dist_max_frac_lane * w)

       pts = sorted(white_pts)
       gap = self.white_cluster_frac * w
       clusters, cur = [], [pts[0]]
       for p in pts[1:]:
           if p[0] - cur[-1][0] > gap:
               clusters.append(cur)
               cur = [p]
           else:
               cur.append(p)
       clusters.append(cur)
       best, best_d = None, allow
       for cl in clusters:
           if len(cl) < self.white_min_bands:
               continue
           bx = self._cluster_bottom_x(cl, roi_h, w)
           if abs(bx - pred) < best_d:
               best, best_d = cl, abs(bx - pred)
       if best is None:
           return
       bx = self._cluster_bottom_x(best, roi_h, w)
       # Sanity-check the match against lane geometry before committing:
       # the jump gate above only rejects deviation from this tracker's
       # OWN prior prediction, so once it locks onto a wrong-but-smoothly-
       # moving feature (a concrete edge, an obstacle boundary) it can't
       # self-correct -- session 26-07-26_61 walked white_x_f from 105px
       # to 21px over 7 frames, each step individually within the jump
       # gate, dragging the derived car-side position ~90px past the edge
       # of frame while every match kept "fresh"-resetting staleness.
       # Reuse the same implausible-lane-position bounds already validated
       # for the tracking branch's found-gate: reject here too, before the
       # bad fix ever reaches white_x_f, rather than only downstream of it.
       # (Restored 2026-07-28: lost in the 07-26 code pull.)
       virtual_check = bx - width if self._lane_applied > 0 else bx + width
       pos_check = self._lane_pos(virtual_check, w)
       if pos_check is not None and not (
               self.lane_implausible_min <= pos_check
               <= self.lane_implausible_max):
           return
       # Heading of the white fit — the live replacement for the frozen
       # yellow heading in bends. The vanishing-point correction is
       # anchored at the white's EXPECTED in-lane position, NOT its
       # instantaneous bottom projection: in a sharp bend the projection
       # collapses along the chord (bx read 130px low in session 38's
       # entrance right-hander), and anchoring there over-subtracts so
       # hard it FLIPS the lean's sign — the tracker then steered away
       # from the bend. Anchored at the steady-state position the two
       # are identical while driving in-lane, and a swept-away white
       # reads as real lean, which is exactly the steer-with-it signal.
       heading = None
       arr = np.array(best, dtype=np.float64)
       if len(best) >= 2 and np.ptp(arr[:, 1]) > 1e-6:
           wgt = arr[:, 2] * (0.5 + arr[:, 1] / roi_h)
           a, _b = np.polyfit(arr[:, 1], arr[:, 0], 1, w=np.sqrt(wgt))
           vp_x = (w / 2.0) if self.target_x is None else float(self.target_x)
           a_exp = (expect - vp_x) / max(h * (1.0 - self.vp_y_frac), 1.0)
           heading = float(np.clip(-(a - a_exp) * roi_h / (w / 2.0), -1.0, 1.0))
       stale = self._white_seen is None or now - self._white_seen > 0.5
       if stale or self.white_x_f is None:
           self.white_x_f = float(bx)
           self.white_h_f = heading
       else:
           self.white_x_f += self._alpha(dt, self.x_tc) * (bx - self.white_x_f)
           if heading is not None:
               if self.white_h_f is None:
                   self.white_h_f = heading
               else:
                   self.white_h_f += self._alpha(dt, self.h_tc) \
                       * (heading - self.white_h_f)
       self._white_seen = now

   # ------------------------------------------------------------------ #
   # control                                                            #
   # ------------------------------------------------------------------ #
   def _lane_pos(self, x_line, w):
       """
       Where x_line (a candidate yellow position/estimate, in image px)
       sits between the two lane boundaries, in lane-widths: 0.0 = ON the
       yellow centerline, 1.0 = ON the outer white line, values outside
       [0, 1] mean past one boundary or the other -- REGARDLESS of which
       side is being driven, since the sign is normalized to the active
       side. Reuses the same calibrated/learned yellow-to-white distance
       the offset geometry already uses (LF_LANE_WIDTH_PX / dist_left /
       dist_right) rather than re-deriving lane geometry.

       Returns None when lane-following isn't engaged (|_lane_applied| <
       0.25 -- riding the yellow itself has no "lane" to leave) or the
       needed distance isn't known yet.
       """
       if abs(self._lane_applied) < 0.25:
           return None
       center = (w / 2.0) if self.target_x is None else float(self.target_x)
       if self._lane_applied > 0:
           dist = self.lane_width_r if self.lane_width_r is not None \
               else self.dist_right
           sign = 1.0
       else:
           dist = self.lane_width_l if self.lane_width_l is not None \
               else self.dist_left
           sign = -1.0
       if dist is None or dist <= 0:
           return None
       return sign * (center - float(x_line)) / float(dist)

   def _boundary_urgency(self):
       """(kp_eff, slew_mult) ramped up as self.lane_pos closes within
       LF_LANE_BOUNDARY_MARGIN lane-widths of either lane edge; the plain
       (kp, 1.0) mid-lane or outside lane-offset driving."""
       kp_eff, slew_mult = self.kp, 1.0
       if self.lane_pos is not None:
           margin = min(self.lane_pos, 1.0 - self.lane_pos)
           if margin < self.lane_boundary_margin:
               frac = float(np.clip(
                   (self.lane_boundary_margin - margin)
                   / self.lane_boundary_margin, 0.0, 1.0))
               kp_eff = self.kp * (
                   1.0 + (self.lane_boundary_kp_boost - 1.0) * frac)
               slew_mult = 1.0 + (
                   self.lane_boundary_slew_boost - 1.0) * frac
       return kp_eff, slew_mult

   def _steer_error(self, x_line, w):
       """Normalized steering error for a line at x_line, with lane offset
       and soft deadband applied."""
       target = (w / 2.0) if self.target_x is None else float(self.target_x)
       # The offset shifts where the yellow should SIT in the frame, opposite
       # to where the car goes: to drive over the RIGHT lane (+offset) the
       # yellow must appear LEFT of center by that fraction of the lane
       # width (LF_LANE_WIDTH_PX when calibrated, else the learned per-side
       # distance). _lane_applied is the RAMPED offset maintained in run(),
       # so engaging doesn't step the target sideways.
       if self._lane_applied > 0:
           dist = self.lane_width_r if self.lane_width_r is not None \
               else self.dist_right
           if dist is not None:
               target -= self._lane_applied * float(dist)
       elif self._lane_applied < 0:
           dist = self.lane_width_l if self.lane_width_l is not None \
               else self.dist_left
           if dist is not None:
               target -= self._lane_applied * float(dist)
       err = (x_line - target) / (w / 2.0)
       if self.deadband > 0:
           err = np.sign(err) * max(0.0, abs(err) - self.deadband)
       return float(np.clip(err, -1.0, 1.0)), target

   def _alpha(self, dt, tc):
       """EMA coefficient for time constant tc at time step dt."""
       return 1.0 if tc <= 0 else 1.0 - float(np.exp(-dt / tc))

   def _heading_cmd(self, hterm, dt, update_slow=True, bend_frac=0.9):
       """
       Combined heading steering response: kh * hterm plus the slow
       curve feed-forward (see LF_CURVE_GAIN), floored on sharp leans
       (see LF_SHARP_BEND_MIN). With update_slow=False the curve state
       is frozen and excluded — used while steering on a FROZEN heading
       snapshot (white-guided), where the field-tuned negative curve
       gain would push against completing the turn. bend_frac sets how
       large |hterm| must be (as a fraction of LF_HEADING_CLIP) before
       the sharp-bend floor engages; see LF_WHITE_GUIDE_BEND_FRAC.
       """
       if update_slow:
           tc = self.curve_tc
           if hterm * self.h_slow < 0 or abs(hterm) < 0.5 * abs(self.h_slow):
               tc = self.curve_tc / 5.0
           self.h_slow += self._alpha(dt, tc) * (hterm - self.h_slow)
           cmd = self.kh * hterm + self.curve_gain * self.h_slow
       else:
           cmd = self.kh * hterm
       # The field-tuned negative curve gain may DAMP the heading
       # response to zero on a sustained lean, but it must never invert
       # it: session 39 crashed mid-bend when the heading reading dipped
       # to +0.35 (car locally aligned with the line it sees, bend still
       # going) — below the sharp-bend floor — and kc*h_slow flipped the
       # net response to steer-left inside a right-hander. Below 0.15
       # the lean's sign is noise and the clamp stays out of the way.
       if abs(hterm) >= 0.15:
           s = 1.0 if hterm > 0 else -1.0
           cmd = s * max(s * cmd, 0.0)
       if abs(hterm) >= bend_frac * self.h_clip and self.sharp_bend_min > 0:
           s = 1.0 if hterm > 0 else -1.0
           cmd = s * max(s * cmd, self.sharp_bend_min)
       return float(cmd)

   def _update_filters(self, x, heading, now, dt):
       """EMA the raw measurements before the PID sees them (see the
       LF_STEER_KP retune note). After a gap with no updates the filter
       state is stale — reseed instead of dragging the estimate over."""
       stale = self._filt_time is None or (now - self._filt_time) > 0.5
       self._filt_time = now
       if stale or self.x_f is None:
           self.x_f = float(x)
           self.d_f = 0.0
           self.prev_error = None
       else:
           self.x_f += self._alpha(dt, self.x_tc) * (x - self.x_f)
       if heading is not None:
           if stale or self.h_f is None:
               self.h_f = float(heading)
           else:
               self.h_f += self._alpha(dt, self.h_tc) * (heading - self.h_f)

   def _apply_steering(self, steer, dt, slew_mult=1.0):
       """
       Clip and slew-limit (LF_STEER_SLEW) the steering command. slew_mult
       raises the allowed rate above the normal cap -- see
       LF_LANE_BOUNDARY_SLEW_BOOST: session 26-07-26_60 showed the
       computed steering command pegged at EXACTLY LF_STEER_SLEW
       (-6.0/s, confirmed frame by frame) through the entire boundary
       crossing near the black-bumper corner -- the slew cap, not
       insufficient kp, was what actually limited how fast the car could
       correct. LF_STEER_SLEW itself stays as tuned (it exists to kill
       noise-driven jitter on straights, and loosening it globally
       reintroduced the 1Hz swerve this was tuned to fix); this only
       raises the cap when boundary proximity has already established
       we're not chasing noise.
       """
       steer = float(np.clip(steer, -1.0, 1.0))
       lim = self.steer_slew * slew_mult * dt
       self.steering = float(np.clip(steer, self.steering - lim,
                                     self.steering + lim))

   def run(self, cam_img):
       steering, throttle, out_img = self._run_frame(cam_img)
       if self.stall_detect and cam_img is not None:
           throttle = self._stall_watch(cam_img, throttle)
       return steering, throttle, out_img

   def _stall_watch(self, cam_img, throttle):
       """Detect a physically stalled car and kick the throttle until it
       moves again (see the LF_STALL_* block in DEFAULTS).

       Runs on the raw input frame, outside the control law: every state
       in _run_frame trusts the camera to eventually show something new,
       which is false the moment the car stops moving with throttle
       still commanded -- the law happily holds a sub-stall command
       forever. Uses its own tiny grayscale diff rather than the vision
       gyro: the gyro strip only measures yaw, and a stalled car can
       still read yaw jitter from people walking through the far
       background."""
       small = cv2.resize(cv2.cvtColor(cam_img, cv2.COLOR_RGB2GRAY),
                          (96, 54), interpolation=cv2.INTER_AREA) \
           .astype(np.int16)
       prev, self._stall_prev = self._stall_prev, small
       if prev is None:
           return throttle
       motion = float(np.abs(small - prev).mean())
       now = time.time()
       if throttle <= 0.01:
           # intentional stop (or coast-to-zero): never fight it
           self._stall_since = None
           self._stall_boost = 0.0
           return throttle
       if motion > self.stall_motion:
           # moving: bleed any boost off quickly rather than dropping it
           # in one frame, so the car doesn't lurch-stop-lurch across a
           # marginal patch (thick seams, tiny debris)
           self._stall_since = None
           self._stall_boost = max(0.0, self._stall_boost - self.stall_kick)
           return min(throttle + self._stall_boost, 1.0)
       if self._stall_since is None:
           self._stall_since = now
           self._stall_step_t = now
       elif (now - self._stall_since >= self.stall_sec
               and now - self._stall_step_t >= self.stall_step_sec):
           self._stall_step_t = now
           self._stall_boost = min(self._stall_boost + self.stall_kick,
                                   self.stall_kick_max)
       if self._stall_boost > 0.0:
           self.status += " +kick"
       return min(throttle + self._stall_boost,
                  max(self.th_max, throttle))

   def _white_cloud_coherent(self, white_pts, w, now):
       """True once the raw white centroids have formed a single thin,
       elongated (line-like) cloud for LF_REANCHOR_FRAMES consecutive
       frames. Only consulted while searching — see the caller. A real
       boundary's centroids hug one axis (perpendicular rms ~2px at
       w=384) no matter how the line curves through the frame; dapple
       and clutter scatter in 2D, and the few streak frames that do look
       thin rarely survive the consecutive-frames requirement. The
       neighbour-gap bound (LF_REANCHOR_MAX_GAP_FRAC) is what stops a
       two-blob cloud from passing the thinness test trivially."""
       ok = False
       if white_pts and len(white_pts) >= 3:
           P = np.array([(c[0], c[1]) for c in white_pts], float)
           n = np.array([c[2] for c in white_pts], float)
           mu = (P * n[:, None]).sum(0) / n.sum()
           Q = P - mu
           cov = (Q * n[:, None]).T @ Q / max(n.sum(), 1e-9)
           ev, V = np.linalg.eigh(cov)
           ratio = np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
           perp = np.array([-V[1, 1], V[0, 1]])
           rms = float(np.sqrt(np.average((Q @ perp) ** 2, weights=n)))
           xs = np.sort(P[:, 0])
           gap = float(np.max(np.diff(xs))) if len(xs) > 1 else 0.0
           ok = (rms <= self.reanchor_rms_frac * w
                 and ratio >= self.reanchor_min_aspect
                 and gap <= self.reanchor_max_gap_frac * w)
       cont = (self._reanchor_t is not None
               and now - self._reanchor_t <= 0.15)
       self._reanchor_t = now
       if ok:
           self._reanchor_streak = self._reanchor_streak + 1 if cont else 1
       else:
           self._reanchor_streak = 0
       return self._reanchor_streak >= self.reanchor_frames

   def _run_frame(self, cam_img):
       if cam_img is None:
           return 0.0, 0.0, None

       # work at LF_PROC_WIDTH; everything downstream (detection, learned
       # lane distances, overlay) lives in these downscaled coordinates
       if self.proc_width and cam_img.shape[1] > self.proc_width:
           pw = int(self.proc_width)
           ph = int(round(cam_img.shape[0] * pw / cam_img.shape[1]))
           cam_img = cv2.resize(cam_img, (pw, ph), interpolation=cv2.INTER_AREA)


       now = time.time()
       dt = 0.05 if self._t_prev is None else \
           float(np.clip(now - self._t_prev, 1e-3, 0.2))
       self._t_prev = now
       h, w = cam_img.shape[:2]

       # vision gyro: yaw-rate damping signal (see LF_GYRO_GAIN). The
       # _sim_yaw_rate hook lets the closed-loop sim inject its model's
       # yaw rate while detect() is stubbed out.
       if self.gyro_gain != 0.0:
           yr = getattr(self, '_sim_yaw_rate', None)
           if yr is None:
               yr = self._measure_yaw_rate(cam_img, dt)
           self.yaw_rate_f += self._alpha(dt, self.gyro_tc) * (yr - self.yaw_rate_f)
           if self.gyro_washout_tc > 0:
               self._gyro_slow += self._alpha(dt, self.gyro_washout_tc) \
                   * (self.yaw_rate_f - self._gyro_slow)
           else:
               self._gyro_slow = 0.0
       damp = -self.gyro_gain * (self.yaw_rate_f - self._gyro_slow)

       found, x_meas, heading, debug = self.detect(cam_img)

       # Temporal consistency gate (see LF_TEMPORAL_JUMP): a fix that
       # teleports relative to the last ACCEPTED fix — beyond what the
       # measured yaw rate explains — is clutter wearing the line's
       # colors, not the line. Treat the frame as a dropout; the coast /
       # white-guided path rides through it. The allowance widens each
       # consecutive rejection, and after LF_TEMPORAL_MAX_REJECT the fix
       # is accepted unconditionally (with filters reseeded) so a stale
       # last position can never lock the detector out for good.
       #
       # This used to only run at all within 0.5s of the last accepted
       # fix, skipping the check entirely afterward -- so the very FIRST
       # fix reacquired after a longer blind stretch (a multi-second
       # coast/white-guided run, common mid-turn) was accepted with NO
       # plausibility check whatsoever. Verified on real footage: after
       # a 2+ second coast, the first few reacquired fixes swung the
       # error from -1.0 to +0.75 within half a second and were let
       # straight through. Now the gate keeps running past 0.5s, but
       # its tolerance grows with how stale the reference is (same
       # "grow, capped" shape already used for the white-guided
       # chain's own jump gate) rather than either staying at the tight
       # frame-to-frame budget (which a long real gap can legitimately
       # exceed) or being skipped outright (which is how an implausible
       # reacquisition got through unchecked).
       # The heading half of this check is conditional on a heading having
       # actually been MEASURED this frame and last (detect() returns None
       # when its fit is too ill-conditioned to trust). The position check
       # is not: it stands on its own, so a frame with no usable heading
       # still gets its x vetted rather than skipping the gate entirely --
       # which is what dropping `last_heading is not None` from this
       # condition buys.
       if (found and self.last_x is not None
               and self._filt_time is not None):
           n = self._suspect_streak
           f_px = (w / 2.0) / np.tan(np.radians(self.cam_hfov) / 2.0)
           gap = now - self._filt_time
           pred_x = self.last_x - f_px * self.yaw_rate_f * gap
           stale_grow = min(1.0 + 2.0 * max(gap - 0.5, 0.0), 2.5)
           bad_x = abs(x_meas - pred_x) \
               > self.temporal_jump * w * (1 + n) * stale_grow
           bad_h = (heading is not None and self.last_heading is not None
                    and abs(heading - self.last_heading)
                        > self.temporal_h_jump * (1 + n) * stale_grow)
           forced = n >= self.temporal_max_reject
           if (bad_x or bad_h) and not forced:
               self._suspect_streak = n + 1
               self._reacquire_streak = 0
               found = False
           elif forced:
               # Escape hatch: stop fighting and TAKE the measurement.
               # This deliberately bypasses the reacquire debounce below.
               # It used to fall through into it, and the two gates then
               # deadlocked: the jump gate's (1+n) widening lets exactly
               # one fix through every ~4 frames, the debounce needs 3
               # CONSECUTIVE ones and zeroes its streak on every reject,
               # so it could never be satisfied -- while `_suspect_streak`
               # reset to 0 on each of those passes, so this escape was
               # never reached either. Session 26-07-30_92 sat in that
               # loop for 1.7s coming out of a shadow with the line
               # detected on 26 of 34 frames and x_f frozen the whole
               # time (sus cycling 1,2,3,0,1,2,3,0...). The reference
               # cannot unstick itself, so something has to force it.
               self._filt_time = None       # long fight: reseed filters
               self._suspect_streak = 0
               self._reacquire_streak = 0
               self._reacquire_x = self._reacquire_heading = None
           else:
               # Debounce a reacquisition after a real gap (see
               # LF_TEMPORAL_REACQUIRE_CONFIRM): passing the jump gate
               # above only means this fix isn't wildly implausible
               # relative to the OLD, possibly-stale reference above --
               # not that it's the start of a real, stable lock rather
               # than a one-off misfit. Require it to agree with the
               # NEXT fix too (a tight, ordinary frame-to-frame budget,
               # not the widened one) before committing to either;
               # until confirmed, ride through on the coast/white-guided
               # path exactly like an ordinary dropout. Continuous
               # tracking (gap <= 0.5s) is unaffected and commits
               # immediately, same as always.
               # An unmeasured heading (None, either side) cannot DISPROVE
               # agreement -- position still has to agree on its own.
               if gap > 0.5 and self.temporal_reacquire_confirm > 1:
                   if (self._reacquire_x is not None
                           and abs(x_meas - self._reacquire_x)
                               <= self.temporal_jump * w
                           and _h_agrees(heading, self._reacquire_heading,
                                         self.temporal_h_jump)):
                       self._reacquire_streak += 1
                   else:
                       self._reacquire_streak = 1
                   self._reacquire_x = x_meas
                   self._reacquire_heading = heading
                   if self._reacquire_streak < self.temporal_reacquire_confirm:
                       found = False
                       # WITHHELD, not accepted -- so the suspect streak
                       # keeps running. Zeroing it here (which is what
                       # falling through to the shared reset did) is what
                       # made the escape above unreachable: every fix the
                       # debounce swallowed looked to the jump gate like a
                       # successful commit. Now a debounce that cannot
                       # converge still counts toward the escape, so the
                       # worst case is bounded at LF_TEMPORAL_MAX_REJECT
                       # frames instead of forever.
                       self._suspect_streak = n + 1
                   else:
                       self._reacquire_streak = 0
                       self._suspect_streak = 0
               else:
                   self._reacquire_streak = 0
                   self._suspect_streak = 0
       elif found:
           self._suspect_streak = 0
           self._reacquire_streak = 0
       else:
           self._reacquire_streak = 0

       # While the yellow is unseen (dropout, off-frame in a lane-offset
       # turn, temporal-gate rejection), dead-reckon its image position
       # with the vision gyro: the reference that the white-line side
       # split and the guidance sanity gate compare against must keep
       # turning with the car, or a long blind curve walks the true line
       # straight out of the gate — session 36 lost the yellow for 2s in
       # a bend and coasted to a stop with a valid white in view because
       # the frozen reference no longer matched the scene.
       # Clamped to 1.5 frame-widths: session 37 integrated the reference
       # out to x=765 during a spin — and the white-line side split then
       # rejected every REAL white cluster against that phantom reference
       # for 2 straight seconds (distance bounds), locking guidance out.
       if not found and self.last_x is not None and self.gyro_gain != 0.0:
           f_px = (w / 2.0) / np.tan(np.radians(self.cam_hfov) / 2.0)
           self.last_x = float(np.clip(
               self.last_x - f_px * self.yaw_rate_f * dt, -0.5 * w, 1.5 * w))

       # a fit that failed the temporal gate is a phantom — never hand it
       # to the white-line side split as the yellow reference
       yellow_fit = debug.get('fit') if found else None
       wl = wr = None
       wl_ok = wr_ok = False
       roi_h = h - debug['roi_y0']
       if self.white_enabled and debug['white_pts']:
           self._white_pts_seen = now   # raw sighting; lost-recovery crawl
           wl, wr, wl_ok, wr_ok = self._split_whites(
               debug['white_pts'], yellow_fit, w, roi_h, now)
           debug['wl'], debug['wr'] = wl, wr
           # keep the near-white tracker warm every frame (lane-offset
           # driving) so it can take over the instant the yellow drops
           self._update_white_tracker(debug['white_pts'], w, h, roi_h, now, dt)

       # Ramp the lane offset toward its setpoint once the lane width it
       # needs is available (immediately when LF_LANE_WIDTH_PX is set,
       # else once the needed side has been learned).
       want = 0.0
       if self.lane_offset > 0 and (self.lane_width_r is not None
                                    or self.dist_right is not None):
           want = self.lane_offset
       elif self.lane_offset < 0 and (self.lane_width_l is not None
                                      or self.dist_left is not None):
           want = self.lane_offset
       step = dt / max(self.lane_ramp_sec, 1e-3)
       self._lane_applied += float(np.clip(want - self._lane_applied,
                                           -step, step))

       # Lane-geometry implausibility gate (see LF_LANE_IMPLAUSIBLE_MIN/
       # MAX): a "genuine" fresh yellow fit that implies the car is more
       # than half a lane-width past a boundary isn't describing this lane
       # at all -- reject it like any other bad fix. Independent of the
       # temporal-jump gate above, which forced-accepts after
       # LF_TEMPORAL_MAX_REJECT rejections -- exactly the state coming out
       # of a coast/white-tracking stretch, which is when every one of the
       # incidents that motivated this gate happened.
       # (Restored 2026-07-28: lost in the 07-26 code pull.)
       if found:
           pos_check = self._lane_pos(x_meas, w)
           if pos_check is not None and not (
                   self.lane_implausible_min <= pos_check
                   <= self.lane_implausible_max):
               # ...but never forever: see LF_LANE_IMPLAUSIBLE_MAX_REJECT.
               # A fix that keeps landing in the SAME "impossible" place is
               # not a glitch, it is the car reporting where it really is
               # (typically having crossed to the far side of the yellow),
               # and the old unconditional reject locked the yellow out for
               # the rest of the run.
               agrees = (
                   self._implausible_x is not None
                   and self._implausible_t is not None
                   and now - self._implausible_t
                       <= self.lane_implausible_stale_sec
                   and abs(x_meas - self._implausible_x)
                       <= self.temporal_jump * w
                   and _h_agrees(heading, self._implausible_heading,
                                 self.temporal_h_jump))
               self._implausible_streak = min(
                   self._implausible_streak + 1 if agrees else 1,
                   max(self.lane_implausible_max_reject, 1))
               self._implausible_x = x_meas
               self._implausible_heading = heading
               self._implausible_t = now
               if (self.lane_implausible_max_reject > 0
                       and self._implausible_streak
                           >= self.lane_implausible_max_reject):
                   # LATCH, not a one-frame pardon. Zeroing the streak
                   # here instead re-arms the gate immediately and only
                   # lets one frame in six through -- measured on session
                   # 88, that recovers 9 frames where staying committed
                   # recovers the whole episode. Once the evidence says
                   # the car really is over there, keep driving on the
                   # yellow until it either comes back into plausible
                   # geometry (the else branch below) or the fixes stop
                   # agreeing with each other (streak drops to 1 and the
                   # gate blocks again while it rebuilds).
                   if not self._implausible_committed:
                       self._implausible_committed = True
                       # Reseed once, on the first accept: the filters
                       # hold geometry from before the departure (same
                       # as the temporal gate's own forced accept), and
                       # the reacquire debounce would otherwise re-block
                       # the very next frame. Doing this every frame
                       # would keep the filters permanently unfiltered.
                       self._filt_time = None
                       self._suspect_streak = 0
                       self._reacquire_streak = 0
               else:
                   found = False
           else:
               self._implausible_streak = 0
               self._implausible_committed = False

       if found:
           # learn lane widths on straight-ish frames only (see _learn_lane),
           # and only from clusters observed in the near field (wl_ok/wr_ok)
           if (self.h_f is None
                   or abs(self.h_f - self.h_bias) <= self.learn_max_heading):
               self._learn_lane(x_meas, wl if wl_ok else None,
                                wr if wr_ok else None, w)
           self.last_x = x_meas
           # Keep the last MEASURED heading when this frame's fit was too
           # ill-conditioned to produce one -- overwriting it with None
           # would disarm the jump gate's heading half on the next frame,
           # and h_f (which _update_filters likewise leaves alone) would
           # then be compared against nothing.
           if heading is not None:
               self.last_heading = heading
           self._update_filters(x_meas, heading, now, dt)
           error, target = self._steer_error(self.x_f, w)
           self.last_err = error
           debug['target'] = target
           # Where this sits between the two boundaries (0=yellow, 1=outer
           # white; see _lane_pos). A real, fresh yellow fit is trusted
           # and DRIVEN ON even if it reports outside [0, 1] -- rejecting
           # a genuine detection here would be worse than acting on it —
           # but it's recorded for visibility (overlay/replay).
           self.lane_pos = self._lane_pos(self.x_f, w)
           debug['lane_pos'] = self.lane_pos
           # Boundary-proximity urgency (see LF_LANE_BOUNDARY_MARGIN/
           # LF_LANE_BOUNDARY_KP_BOOST/LF_LANE_BOUNDARY_SLEW_BOOST): ramp
           # the position gain AND the steering slew cap up as the car
           # closes in on either edge of the lane. The slew boost matters
           # as much as the gain boost -- a bigger computed command does
           # nothing once the slew cap is already the binding constraint.
           # No-op mid-lane or outside lane-following (lane_pos is None).
           kp_eff, slew_mult = self._boundary_urgency()


           # PID + heading feedback, on the FILTERED measurements.
           # h_f stays None until a heading has actually been measured --
           # detect() now reports None rather than a fake 0.0 for an
           # ill-conditioned fit, so a run can legitimately reach here
           # having tracked position for several frames with no heading
           # yet. No heading information means no heading contribution;
           # position still drives.
           hterm = 0.0 if self.h_f is None else float(np.clip(
               self.h_f - self.h_bias, -self.h_clip, self.h_clip))
           if self.prev_error is not None:
               # derivative of the filtered error, then low-passed again —
               # the raw one-frame difference at 20 Hz was the largest
               # single steering contributor, and it was all noise
               raw_d = (error - self.prev_error) / dt
               self.d_f += self._alpha(dt, self.d_tc) * (raw_d - self.d_f)
               # The integrator is a TRIM estimator: it updates only while
               # driving near-straight (trim is unobservable in a curve —
               # the sustained error there is curvature) with a small error
               # (a big one is a transient we're still converging out of),
               # with dt clamped so a frame hiccup can't spike the sum, and
               # frozen while the output is already saturated toward the
               # error (classic anti-windup). Letting it wind in curves is
               # what S-swerved every post-curve straight and pushed the
               # car back across the yellow when lane-offset driving.
               if ((abs(self.steering) < 1.0
                        or np.sign(self.steering) != np.sign(error))
                       and abs(hterm) <= self.i_h_gate
                       and abs(error) <= self.i_err_gate):
                   self.i_term = float(np.clip(
                       self.i_term + self.ki * error * min(dt, 0.2),
                       -self.i_max, self.i_max))
           self.prev_error = error


           # heading response = kh*hterm + slow curve feed-forward, with
           # the sharp-bend floor; see _heading_cmd / LF_SHARP_BEND_MIN
           steer = (kp_eff * error + self.kd * self.d_f + self.i_term
                    + self._heading_cmd(hterm, dt) + damp)
           self._apply_steering(steer, dt, slew_mult)


           # Slow down proportionally to how hard we're steering — and to
           # how hard the line ahead is BENDING: heading rises at a curve's
           # entry before the steering has wound up, so keying the throttle
           # off it too sheds speed going in, not halfway around.
           slow = min(1.0, max(abs(self.steering), abs(hterm) / max(self.h_clip, 1e-6)))
           # See LF_THROTTLE_MIN_LANE_OFFSET: this can't tell "steering
           # hard because the track is genuinely bending" from "steering
           # hard because the lateral target jumped a lot" -- use the
           # higher floor in the latter case (large POSITIONAL error, the
           # same "not yet converged" condition LF_STEER_KI already gates
           # on) so an active maneuver isn't starved of the speed it
           # needs to actually close that error.
           th_floor = (self.th_min_lane_offset if abs(error) > self.i_err_gate
                       else self.th_min)
           self.throttle = self.th_max - (self.th_max - th_floor) * slow
           self.status = "tracking"
           self.lost_frames = 0
           self.lost_since = None


       else:
           if self.lost_since is None:
               self.lost_since = now
           # PRIMARY blind-bend path for lane-offset driving: steer
           # directly on the live near-white tracker. Unlike the estimate
           # chain below it needs no yellow reference at all, and its
           # heading is measured fresh every frame — the car keeps real
           # heading feedback through the whole arc instead of holding a
           # frozen snapshot. (Session 38: whites were detected in >92%
           # of the frames of every episode that ended in a stop; what
           # failed was inference built on the stale yellow, not vision.)
           track_w = self._near_lane_width()
           white_tracking_ok = False
           # The takeover dwell (LF_WHITE_TAKEOVER_SEC) keeps the whites
           # from steering during ordinary dash gaps / single rejected
           # fixes: the yellow must be CONTINUOUSLY lost for the dwell
           # before any white path takes over. Coasting (steering held)
           # covers the dwell frames, which is what a dash gap needs.
           if (self.white_track and abs(self._lane_applied) >= 0.25
                   and track_w is not None
                   and self._white_seen is not None
                   and now - self._white_seen <= self.white_track_fresh
                   and now - self.lost_since >= self.white_takeover_sec
                   and now - self.lost_since <= self.white_guide_sec):
               virtual = self.white_x_f + float(track_w) \
                   if self._lane_applied < 0 \
                   else self.white_x_f - float(track_w)
               hterm = 0.0
               if self.white_h_f is not None:
                   hterm = float(np.clip(self.white_h_f - self.h_bias,
                                         -self.h_clip, self.h_clip))
               # same chord-bias compensation as the estimate chain —
               # the white's bottom projection leans into the bend
               virtual += self.chord_comp * w * hterm
               # Sanity-gate against the last known line position, same
               # jump tolerance the estimate-from-whites chain below
               # already uses ("the line cannot teleport" — see that
               # chain's own comment). Added 2026-07-25: this PRIMARY
               # path was missing the check the SECONDARY path already
               # had. Real footage from a visually cluttered section (two
               # candidate white lines in view at once, near a wall)
               # showed white_x_f able to drift steadily off the real
               # boundary over roughly half a second — each single
               # frame-to-frame step was small enough to pass
               # _update_white_tracker's OWN jump gate (which only checks
               # against its own last prediction, no independent anchor),
               # but the cumulative drift pulled the estimate 100px+ off
               # the real line and fed straight into steering. Falls
               # through to the estimate-from-whites chain below (or
               # ultimately coasting) instead of trusting an estimate
               # that has drifted away from reality.
               if self.last_x is not None:
                   grow = min(1.0 + 2.0 * (now - self.lost_since), 2.5)
                   allow = min(self.single_dash_max_jump * w * grow, 0.33 * w)
                   white_tracking_ok = abs(virtual - self.last_x) <= allow
               else:
                   white_tracking_ok = True
           if white_tracking_ok:
               self.last_x = float(virtual)
               self._update_filters(virtual, None, now, dt)
               error, target = self._steer_error(self.x_f, w)
               self.last_err = error
               debug['target'] = target
               debug['x_est'] = virtual
               # Boundary-proximity urgency, same as the tracking branch:
               # session 26-07-26_60 (real on-car test, right-lane) showed
               # boundary crossings happen exactly here -- lane_pos
               # reached 1.03-1.08 (past the outer white) specifically
               # while status was "white-tracking", climbing there over
               # ~5 frames with heading near-neutral (so not a case the
               # p_scale fade below was already covering).
               self.lane_pos = self._lane_pos(virtual, w)
               debug['lane_pos'] = self.lane_pos
               kp_eff, slew_mult = self._boundary_urgency()
               self.prev_error = None
               self.d_f = 0.0
               # The sharper the lean, the more the white runs sideways
               # through the frame and the less its bottom projection
               # means: at pegged lean the projection collapses along the
               # chord (session 38 read the white 130px left of reality
               # in the entrance right-hander, and full-gain P steered
               # hard AWAY from the bend). Fade the position term out
               # with lean and let the live heading carry the arc.
               p_scale = 1.0 - 0.85 * min(abs(hterm) / max(self.h_clip, 1e-6), 1.0)
               steer = (p_scale * kp_eff * error + self.i_term
                        + self._heading_cmd(hterm, dt) + damp)
               self._apply_steering(steer, dt, slew_mult)
               slow = min(1.0, max(abs(self.steering),
                                   abs(hterm) / max(self.h_clip, 1e-6)))
               # see LF_THROTTLE_MIN_LANE_OFFSET on the tracking path above
               th_floor = (self.th_min_lane_offset if abs(error) > self.i_err_gate
                           else self.th_min)
               self.throttle = self.th_max - (self.th_max - th_floor) * slow
               self.status = "white-tracking"
               self.lost_frames = 0
               self._last_guided = now
               out_img = self._overlay(cam_img, debug, found, 0.0) \
                   if self.overlay else cam_img
               return self.steering, self.throttle, out_img
           x_est = None
           if (self.white_enabled
                   and now - self.lost_since >= self.white_takeover_sec
                   and now - self.lost_since <= self.white_guide_sec):
               if wl is None and wr is None and debug['white_pts']:
                   # reference-based side split found nothing usable —
                   # fall back to ego-frame identification of the lane's
                   # near boundary (lane-offset driving only)
                   wl, wr = self._ego_near_boundary(
                       debug['white_pts'], w, h - debug['roi_y0'])
               x_est = self._estimate_from_whites(wl, wr, w)
               if x_est is not None and self.h_f is not None:
                   # chord-distortion compensation (LF_WHITE_CHORD_COMP):
                   # in a bend the white's straight-line bottom projection
                   # is biased toward the curve's inside
                   x_est += self.chord_comp * w * float(np.clip(
                       self.h_f - self.h_bias, -self.h_clip, self.h_clip))
           # Sanity-gate the white estimate against the last known line
           # position: _split_whites sides its clusters off the (possibly
           # stale or misfit) yellow reference, and one wrong-side call
           # turns "left white + lane width" into an estimate a full lane
           # out — session 18 replay shows x_est=-64 from exactly that,
           # which commanded a hard-left excursion until the stop timer
           # fired. The line cannot teleport: a guided estimate must land
           # near where the (gyro-dead-reckoned) yellow reference is, with
           # the allowance growing the longer we've been blind but never
           # past a third of the frame. NOT widened by heading like the
           # single-dash gate — the misfits that poison the side split
           # happen precisely mid-turn, so a heading widening opens the
           # gate exactly when it must hold. Deliberately NO absolute
           # in-frame bound: in a lane-offset turn the yellow genuinely
           # leaves the frame, and an off-frame estimate (wl + lane width
           # > image width) is the CORRECT virtual target — an earlier
           # in-frame check here silently discarded 2s of valid white
           # guidance in session 36 and let the car coast to a stop.
           if x_est is not None and self.last_x is not None:
               grow = min(1.0 + 2.0 * (now - self.lost_since), 2.5)
               allow = min(self.single_dash_max_jump * w * grow, 0.33 * w)
               if abs(x_est - self.last_x) > allow:
                   x_est = None
           # Cross-check against LF_WHITE_TRACK's persistent tracker (see
           # LF_WHITE_GUIDE_DISAGREE_MAX_FRAC): the gate above only
           # catches teleports relative to this chain's OWN last output,
           # so a consistent bias (a misidentified second line, chronic
           # clutter on one side) can walk the estimate arbitrarily far
           # over many frames -- an independent second opinion is the
           # only thing that catches that. Skipped whenever the tracker
           # itself has nothing fresh to say, so this never penalizes a
           # track/config that doesn't use the persistent tracker.
           #
           # Freshness bound is LF_WHITE_GUIDE_SEC (how long white-guided
           # itself is trusted at all), NOT LF_WHITE_TRACK_FRESH (the
           # much stricter bar for letting the tracker DRIVE steering
           # directly) -- a stale-for-direct-steering tracker estimate is
           # still a meaningful second opinion for a sanity check, and
           # using the tighter bound left a gap of its own: in a long
           # blind stretch last_x keeps drifting via gyro dead-reckoning
           # (see the dead-reckoning update above, in the "not found"
           # branch) even while the tracker goes stale, so the tighter
           # bound expired this cross-check at the exact moment the
           # self-referential gate above was also being defeated by that
           # same drift.
           if (x_est is not None and track_w is not None
                   and self.white_x_f is not None and self._white_seen is not None
                   and now - self._white_seen <= self.white_guide_sec):
               tracker_est = self.white_x_f + float(track_w) \
                   if self._lane_applied < 0 \
                   else self.white_x_f - float(track_w)
               if abs(x_est - tracker_est) \
                       > self.white_guide_disagree_max_frac * float(track_w):
                   x_est = None
           if x_est is not None:
               # Yellow gone (dash gap, worn paint, deep shadow) but the
               # solid white lines are visible: steer from them. Only for
               # LF_WHITE_GUIDE_SEC after the last yellow fix — this
               # bridges gaps, it must not drive the car indefinitely on
               # white edges alone. The lost clock keeps running so the
               # coast/stop logic takes over when the window expires. No D
               # term across the mode switch — the error source changed.
               # The estimate still feeds the x filter, so the handback to
               # yellow tracking is seamless on both ends.
               # Follow the estimate with a rate limit instead of jumping
               # to it: mid-turn white bottom-projections scatter +/-100px
               # frame to frame (sparse centroids, extrapolation), and a
               # reference that teleports with each sample whipsaws the
               # steering — session 37's guided stretches swung the
               # command full-range at 2-3Hz on exactly that scatter.
               if self.last_x is not None:
                   step = 0.08 * w
                   x_est = float(self.last_x
                                 + np.clip(x_est - self.last_x, -step, step))
               self.last_x = x_est
               self._update_filters(x_est, None, now, dt)
               error, target = self._steer_error(self.x_f, w)
               self.last_err = error
               debug['target'] = target
               debug['x_est'] = x_est
               self.prev_error = None
               self.d_f = 0.0
               # keep applying the learned trim (i_term) but don't update it
               # from white-line estimates — the error source changed.
               # ALSO keep the last tracked heading feed-forward (h_f is
               # frozen while the yellow is lost): 61% of yellow dropouts
               # happen mid-turn, and dropping the heading term here
               # straightened the car exactly when it needed to hold its
               # arc — it then re-found the line having drifted a lane over.
               hterm = 0.0
               if self.h_f is not None:
                   hterm = float(np.clip(self.h_f - self.h_bias,
                                         -self.h_clip, self.h_clip))
               # update_slow=False: h_slow frozen and the curve term
               # excluded — fed the frozen mid-turn lean, the field-tuned
               # negative curve gain would steer against completing the
               # turn. kh*hterm (plus the sharp-bend floor) holds the arc.
               # Position de-weighted with lean like the tracker branch:
               # these estimates come from the same chord-biased bottom
               # projections.
               p_scale = 1.0 - 0.85 * min(abs(hterm) / max(self.h_clip, 1e-6), 1.0)
               steer = (p_scale * self.kp * error + self.i_term
                        + self._heading_cmd(hterm, dt, update_slow=False,
                                            bend_frac=self.white_guide_bend_frac)
                        + damp)
               self._apply_steering(steer, dt)
               self.throttle = self.th_min
               self.status = "white-guided"
               self.lost_frames = 0
               self._last_guided = now
           else:
               self.lost_frames += 1
               self.prev_error = None


               # The give-up clock measures time since the car was last
               # STEERED BY A MEASUREMENT — yellow fix or white guidance —
               # not since the last yellow. It used to run from the first
               # yellow loss only: in session 36 the car was actively
               # white-guiding through a blind bend for 2s, then a single
               # scattered (gate-rejected) estimate tripped an instant
               # hard stop because the yellow-loss clock had already
               # expired underneath the working guidance.
               guided = getattr(self, '_last_guided', 0.0)
               blind = min(now - self.lost_since, now - guided)
               if blind > self.lost_stop_sec and \
                       blind <= self.lost_stop_sec + self.lost_recovery_sec \
                       and self._white_pts_seen is not None \
                       and now - self._white_pts_seen \
                           <= self.lost_recovery_white_age:
                   # Grace period over, but the white detector is still
                   # seeing SOMETHING (see LF_LOST_RECOVERY_SEC): instead
                   # of parking for good with a boundary in plain view —
                   # a stopped car's view never changes, so a stop here
                   # is final — keep crawling in the direction already
                   # being steered at the lost-coast throttle. Every
                   # normal path (yellow reacquire, white-tracking,
                   # white-guided) keeps getting evaluated first each
                   # frame; the moment one accepts, it takes over. If the
                   # whites drop out of view, or the budget runs out, the
                   # stop below fires exactly as before.
                   self.throttle = self.th_lost
                   self.status = "searching (line lost)"
                   # Drop the stale yellow reference while searching.
                   # By now the car has been unguided for LF_LOST_STOP_SEC
                   # (2s+ of uncontrolled coasting) — wherever the line
                   # was back then says nothing about where it is now,
                   # yet every re-entry path (white-tracking's jump gate,
                   # the estimate chain's side split) still measures
                   # candidates against it and rejects them. Session
                   # 26-07-29_86 idx 166349-166448: raw whites sat ON the
                   # real boundary every frame of the search, wl/wr never
                   # resolved, car parked with the line in plain view.
                   # With last_x gone those gates fall back to their own
                   # cold-start logic (the white tracker's stale-
                   # reacquire radius, the ego-frame boundary fallback),
                   # which is exactly what "searching" should mean.
                   # Gated on the whites forming a coherent line-like
                   # cloud (see LF_REANCHOR_*) so a dapple field can't
                   # seed the tracker.
                   if self._white_cloud_coherent(debug['white_pts'], w,
                                                 now):
                       self.last_x = None
               elif blind > self.lost_stop_sec:
                   # lost for too long — stop rather than wander off the track
                   self.steering, self.throttle = 0.0, 0.0
                   self.i_term = 0.0   # relearn trim on the next acquisition
                   self.x_f = self.h_f = None   # stale after a stop; reseed
                   self.d_f = 0.0
                   self.h_slow = 0.0
                   self.white_x_f = self.white_h_f = None
                   self._white_seen = None
                   self._white_pts_seen = None
                   self._last_wl = self._last_wr = None
                   self._last_wl_t = self._last_wr_t = None
                   self.lane_pos = None
                   self._lane_applied = 0.0     # re-ramp the lane offset too
                   self._reacquire_streak = 0
                   self._reacquire_x = self._reacquire_heading = None
                   self.status = "stopped (line lost)"
               else:
                   # dropout (dash gap or momentary occlusion): hold the last
                   # known steering and ease off throttle. Do NOT commit to an
                   # active "search" turn here — a turn applied blind just
                   # traces a circle for the entire grace window before the
                   # car gives up, which is exactly the runaway this caused.
                   self.throttle = max(self.th_lost, self.throttle * 0.9)
                   self.status = "coasting"


       out_img = self._overlay(cam_img, debug, found, heading) \
           if self.overlay else cam_img
       return self.steering, self.throttle, out_img


   # keep parity with parts that call shutdown
   def shutdown(self):
       pass


   # ------------------------------------------------------------------ #
   # diagnostics overlay                                                #
   # ------------------------------------------------------------------ #
   def _overlay(self, cam_img, debug, found, heading):
       img = cam_img.copy()
       h, w = img.shape[:2]
       y0 = debug['roi_y0']


       # tint detected yellow mask green, white-line mask blue
       region = img[y0:, :]
       if 'white_mask' in debug:
           region[debug['white_mask'] > 0] = (0, 128, 255)
       region[debug['mask'] > 0] = (0, 255, 0)


       # ROI top line
       cv2.line(img, (0, y0), (w, y0), (255, 255, 0), 1)


       # band centroids (yellow: red dots; white: blue dots)
       for cx, cy, _ in debug['white_pts']:
           cv2.circle(img, (int(cx), int(y0 + cy)), 2, (0, 0, 255), -1)
       for cx, cy, _ in debug['pts']:
           cv2.circle(img, (int(cx), int(y0 + cy)), 3, (255, 0, 0), -1)


       # fitted line + target + white-guided estimate
       if found and 'fit' in debug:
           a, b = debug['fit']
           roi_h = h - y0
           p1 = (int(b), y0)
           p2 = (int(a * roi_h + b), h - 1)
           cv2.line(img, p1, p2, (255, 0, 255), 2)
       if 'x_est' in debug:
           cv2.line(img, (int(debug['x_est']), y0), (int(debug['x_est']), h - 1),
                    (0, 200, 255), 2)
       if 'target' in debug:
           cv2.line(img, (int(debug['target']), h - 12),
                    (int(debug['target']), h - 1), (255, 255, 255), 2)


       lane = ""
       if self.dist_left is not None or self.dist_right is not None:
           dl = f"{self.dist_left:.0f}" if self.dist_left is not None else "?"
           dr = f"{self.dist_right:.0f}" if self.dist_right is not None else "?"
           lane = f" lane:{dl}/{dr}"
       if self._lane_applied != 0.0:
           lane += f" off:{self._lane_applied:+.2f}"
       for i, s in enumerate([
           f"st:{self.steering:+.2f} th:{self.throttle:.2f} i:{self.i_term:+.2f}",
           # heading is None when this frame's fit was too ill-conditioned
           # to measure one -- show that, don't print a fake 0.00
           f"e:{self.last_err:+.2f} "
           f"hdg:{'--' if heading is None else format(heading, '+.2f')}{lane}",
           self.status,
       ]):
           cv2.putText(img, s, (4, 12 + 12 * i), cv2.FONT_HERSHEY_SIMPLEX,
                       0.38, (255, 255, 255), 1, cv2.LINE_AA)
       return img




# =============================================================================
# Offline test harness — run the pipeline on saved images/videos, no car needed
# =============================================================================
def _test(paths, out_dir="lf_out"):
   import os
   import glob as globmod


   os.makedirs(out_dir, exist_ok=True)
   files = []
   for p in paths:
       if os.path.isdir(p):
           for ext in ('*.jpg', '*.jpeg', '*.png'):
               files += sorted(globmod.glob(os.path.join(p, ext)))
       else:
           files.append(p)


   def prep(bgr):
       return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

   lf = LineFollower()
   for f in files:
       if f.lower().endswith(('.mp4', '.avi', '.mov')):
           cap = cv2.VideoCapture(f)
           i = 0
           while True:
               ok, frame = cap.read()
               if not ok:
                   break
               st, th, ov = lf.run(prep(frame))
               cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.jpg"),
                           cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
               print(f"{f}[{i}] steering={st:+.3f} throttle={th:.3f} ({lf.status})")
               i += 1
           cap.release()
       else:
           bgr = cv2.imread(f)
           if bgr is None:
               print(f"skip {f}: unreadable")
               continue
           st, th, ov = lf.run(prep(bgr))
           out = os.path.join(out_dir, os.path.basename(f))
           cv2.imwrite(out, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
           print(f"{f}: steering={st:+.3f} throttle={th:.3f} ({lf.status}) -> {out}")




def _calibrate(path):
   """
   Measure LF_TARGET_X and LF_HEADING_BIAS from a frame taken with the car
   parked so its centerline sits EXACTLY over the yellow line, pointing
   straight along it. Whatever x the line reads in that frame is, by
   construction, where the line should sit whenever the car is centered on
   it — regardless of how the camera is mounted. Likewise, whatever
   heading the fit reads while physically aligned is pure camera yaw /
   perspective bias, which would otherwise be a constant steering push
   the integrator has to fight (the 2026-07-23 sessions measured +0.12).
   """
   bgr = cv2.imread(path)
   if bgr is None:
       print(f"unreadable: {path}")
       return
   lf = LineFollower()
   lf.min_dashes = 1   # a parked close-up may show only one dash; that's fine
                       # here — the operator is looking at the frame anyway
   lf.run(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
   if lf.last_x is None:
       print("no yellow line detected in this frame — repark so at least one "
             "dash is clearly visible in the bottom part of the frame, or "
             "check the frame isn't over/under-exposed")
       return
   w = int(lf.proc_width) if lf.proc_width else bgr.shape[1]
   off = lf.last_x - w / 2.0
   print(f"line at x={lf.last_x:.1f}, image center {w / 2.0:.1f} "
         f"(camera offset {off:+.1f}px in {w}-wide coords)")
   print(f"add to myconfig.py:  LF_TARGET_X = {lf.last_x:.0f}")
   if lf.last_heading is None:
       print("heading NOT measurable in this frame — the visible dashes are "
             "bunched into too narrow a band for the slope fit (see "
             "LF_MIN_HEADING_SPREAD_FRAC). LF_TARGET_X above is still "
             "valid; repark with dashes spread up the ROI to read the bias")
       return
   print(f"measured heading {lf.last_heading:+.3f} "
         f"(only valid if the car was parked pointing straight along the line)")
   print(f"add to myconfig.py:  LF_HEADING_BIAS = {lf.last_heading:.2f}")




if __name__ == '__main__':
   import sys
   if len(sys.argv) >= 3 and sys.argv[1] == 'calibrate':
       _calibrate(sys.argv[2])
   elif len(sys.argv) >= 3 and sys.argv[1] == 'test':
       args = sys.argv[2:]
       out = "lf_out"
       if "--out" in args:
           i = args.index("--out")
           out = args[i + 1]
           args = args[:i] + args[i + 2:]
       _test(args, out)
   else:
       print(__doc__)
