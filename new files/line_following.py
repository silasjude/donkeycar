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
-----------------------------------------------------------------------------
"""


import logging
import time


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
   # White guidance only bridges GAPS in the yellow (dash gaps, a worn or
   # deeply shadowed stretch): it engages for at most this many seconds
   # after the last confirmed yellow fix, then the normal coast/stop logic
   # takes over. Unbounded, a picked-up car staring at any bright edge
   # would keep "driving" forever.
   LF_WHITE_GUIDE_SEC=3.0,
   # Learned yellow-to-white distances are only trusted inside this range
   # (fraction of image width); outside it the sample is a misclassified
   # speck or a line from a neighboring track.
   LF_LANE_DIST_MIN_FRAC=0.08,
   LF_LANE_DIST_MAX_FRAC=0.45,


   # --- steering control (PD on normalized lateral error) ---
   # error: -1 = line at left edge, 0 = line at target, +1 = line at right edge
   # donkeycar steering: -1 = full left, +1 = full right
   LF_STEER_KP=2.4,        # proportional gain on lateral offset
   LF_STEER_KD=0.35,       # derivative gain (per second)
   LF_HEADING_GAIN=0.9,    # extra steering from the line's slope (lookahead)
   LF_ERR_DEADBAND=0.0,    # soft deadband on the normalized error: errors
                           # smaller than this are ignored (subtracted, so
                           # response stays continuous). 0 = tightest
                           # centering over the line; raise to ~0.05 only if
                           # the car weaves on straights.
   LF_TARGET_X=None,       # where the line should sit in the frame, in pixels.
                           # None = image center. If your camera is mounted
                           # off-center, set this.

   # --- lane offset (future: drive IN the left or right lane) ---
   # 0.0 rides directly on top of the yellow line. +1.0 aims the car at the
   # nearest white line to the RIGHT of the yellow, -1.0 at the white line
   # to the LEFT; +0.5 is the middle of the right lane. Uses the
   # yellow-to-white distances learned from the white-line detector, so it
   # only engages once those have been observed (falls back to riding the
   # yellow until then).
   LF_LANE_OFFSET=0.0,


   # --- throttle ---
   LF_THROTTLE_MAX=0.30,   # on straights
   LF_THROTTLE_MIN=0.16,   # in hard corrections
   LF_THROTTLE_LOST=0.12,  # while coasting on a lost line


   # --- lost-line recovery ---
   # On loss, steer from the white lines if they're visible (see
   # LF_WHITE_ENABLED above). Failing that, hold the last known steering
   # (don't commit to a blind turn — that just traces a circle) and ease
   # off the throttle until either the line reappears or the grace period
   # runs out, then stop.
   LF_LOST_STOP_SEC=2.0,   # give up and stop after this many seconds lost


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
   accepted for compatibility but ignored — control is a self-contained PD
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
       self.max_blob_height_ratio = float(_cfg(cfg, 'LF_BLOB_MAX_HEIGHT_RATIO'))
       self.min_blob_area_frac = float(_cfg(cfg, 'LF_BLOB_MIN_AREA_FRAC'))

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
       self.white_guide_sec = float(_cfg(cfg, 'LF_WHITE_GUIDE_SEC'))
       self.lane_dist_min_frac = float(_cfg(cfg, 'LF_LANE_DIST_MIN_FRAC'))
       self.lane_dist_max_frac = float(_cfg(cfg, 'LF_LANE_DIST_MAX_FRAC'))


       self.kp = float(_cfg(cfg, 'LF_STEER_KP'))
       self.kd = float(_cfg(cfg, 'LF_STEER_KD'))
       self.kh = float(_cfg(cfg, 'LF_HEADING_GAIN'))
       self.deadband = float(_cfg(cfg, 'LF_ERR_DEADBAND'))
       self.target_x = _cfg(cfg, 'LF_TARGET_X')
       self.lane_offset = float(_cfg(cfg, 'LF_LANE_OFFSET'))


       self.th_max = float(_cfg(cfg, 'LF_THROTTLE_MAX'))
       self.th_min = float(_cfg(cfg, 'LF_THROTTLE_MIN'))
       self.th_lost = float(_cfg(cfg, 'LF_THROTTLE_LOST'))


       self.lost_stop_sec = float(_cfg(cfg, 'LF_LOST_STOP_SEC'))


       self.proc_width = _cfg(cfg, 'LF_PROC_WIDTH')
       self.overlay = bool(_cfg(cfg, 'OVERLAY_IMAGE'))


       # state
       self.steering = 0.0
       self.throttle = 0.0
       self.frames_since_fix = 10 ** 9
       self.prev_error = None
       self.prev_time = None
       self.lost_frames = 0
       self.lost_since = None
       self.status = "init"
       self.last_x = None       # last known yellow-line x (px, full-frame)
       self.dist_left = None    # learned yellow -> nearest-left-white distance (px, EMA)
       self.dist_right = None   # learned yellow -> nearest-right-white distance (px, EMA)


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

   def _detect_white(self, Lc, Ac, Bc_abs, bgB):
       """
       Mask the solid white boundary lines by local contrast: brighter than
       the surrounding pavement, chroma close to the local background's
       (so shadow's blue cast doesn't disqualify them; the blue tape on the
       track deviates in b and is excluded). Returns per-band centroids,
       possibly several per band (left and right line).
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
       n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
       for i in range(1, n):
           ext = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
           if stats[i, cv2.CC_STAT_AREA] < min_area or ext < min_ext:
               white[labels == i] = 0
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
       bgB = np.maximum(cv2.medianBlur(B, k_bg), self.bg_b_clamp)
       bgL = cv2.medianBlur(L, k_bg)
       Bc = B.astype(np.int16) - bgB.astype(np.int16)
       Lc = L.astype(np.int16) - bgL.astype(np.int16)

       yellow = ((Bc >= self.b_contrast) & (B >= self.b_floor)) | (B >= self.b_min)
       mask = (yellow & (A <= self.a_max) & (L >= self.l_min)
               & (hue >= self.hue_min) & (hue <= self.hue_max)
               ).astype(np.uint8) * 255


       if self._kernel is None:
           k = 5 if w >= 240 else 3
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
           wmask, wpts = self._detect_white(Lc, A, B, bgB)
           debug['white_mask'] = wmask
           debug['white_pts'] = wpts


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


       # Lateral position comes from the NEAREST detected dashes (bottom-most
       # centroids), not from extrapolating the fit to the frame bottom —
       # extrapolation blows up when the line runs nearly horizontal in the
       # frame (sharp curves).
       order = np.argsort(cys)[::-1]          # nearest (largest y) first
       near = order[:3]
       x_near = float(np.average(cxs[near], weights=ws[near]))

       # A real sighting of a DASHED line is multiple separate dashes; see
       # LF_MIN_DASHES. A single blob only counts while already tracking,
       # and only near where the line was last seen.
       n_blobs = cv2.connectedComponents(mask, connectivity=8)[0] - 1
       if n_blobs < self.min_dashes:
           recent = self.frames_since_fix <= self.single_dash_grace
           near_last = (self.last_x is not None
                        and abs(x_near - self.last_x) <= self.single_dash_max_jump * w)
           if not (recent and near_last):
               return False, None, 0.0, debug


       # heading: slope, normalized. a is px-x per px-y; y grows downward, so
       # a > 0 means the line moves right toward the car => it leans LEFT
       # ahead of the car => steer left. Negate to get "lean of the road ahead".
       heading = float(np.clip(-a * roi_h / (w / 2.0), -1.0, 1.0))


       self.frames_since_fix = 0
       debug['fit'] = (a, bfit)
       debug['x_near'] = x_near
       return True, x_near, heading, debug


   # ------------------------------------------------------------------ #
   # white-line fallback helpers                                        #
   # ------------------------------------------------------------------ #
   def _split_whites(self, white_pts, yellow_fit):
       """
       Split white-line band centroids into left/right of the yellow line
       (using the yellow fit when available, else the last known yellow x)
       and return the nearest-to-car x of each side plus per-side fits.
       """
       left, right = [], []
       for cx, cy, n in white_pts:
           if yellow_fit is not None:
               a, b = yellow_fit
               ref = a * cy + b
           elif self.last_x is not None:
               ref = self.last_x
           else:
               ref = None
           if ref is None:
               continue
           (left if cx < ref else right).append((cx, cy, n))

       def near_x(side):
           if len(side) < self.white_min_bands:
               return None
           arr = np.array(side, dtype=np.float64)
           # nearest-to-car centroids dominate, same reasoning as the yellow
           order = np.argsort(arr[:, 1])[::-1]
           sel = arr[order[:3]]
           return float(np.average(sel[:, 0], weights=sel[:, 2]))

       return near_x(left), near_x(right), left, right

   def _learn_lane(self, x_yellow, wl, wr, w):
       """EMA the yellow-to-white distance on each side while tracking.
       Samples outside the plausible range (LF_LANE_DIST_*_FRAC) are
       misclassified speckles or a neighboring track's line — skip them."""
       lo, hi = self.lane_dist_min_frac * w, self.lane_dist_max_frac * w
       if wl is not None and lo < x_yellow - wl < hi:
           d = x_yellow - wl
           self.dist_left = d if self.dist_left is None else 0.9 * self.dist_left + 0.1 * d
       if wr is not None and lo < wr - x_yellow < hi:
           d = wr - x_yellow
           self.dist_right = d if self.dist_right is None else 0.9 * self.dist_right + 0.1 * d

   def _estimate_from_whites(self, wl, wr):
       """
       Estimate where the yellow line is from the white lines and the
       learned per-side distances. Returns x estimate or None.
       """
       est = []
       if wl is not None and self.dist_left is not None:
           est.append(wl + self.dist_left)
       if wr is not None and self.dist_right is not None:
           est.append(wr - self.dist_right)
       if not est:
           return None
       return float(np.mean(est))


   # ------------------------------------------------------------------ #
   # control                                                            #
   # ------------------------------------------------------------------ #
   def _steer_error(self, x_line, w):
       """Normalized steering error for a line at x_line, with lane offset
       and soft deadband applied."""
       target = (w / 2.0) if self.target_x is None else float(self.target_x)
       # The offset shifts where the yellow should SIT in the frame, opposite
       # to where the car goes: to drive over the RIGHT lane (+offset) the
       # yellow must appear LEFT of center by that fraction of the learned
       # yellow-to-white distance.
       if self.lane_offset > 0 and self.dist_right is not None:
           target -= self.lane_offset * self.dist_right
       elif self.lane_offset < 0 and self.dist_left is not None:
           target -= self.lane_offset * self.dist_left
       err = (x_line - target) / (w / 2.0)
       if self.deadband > 0:
           err = np.sign(err) * max(0.0, abs(err) - self.deadband)
       return float(np.clip(err, -1.0, 1.0)), target

   def run(self, cam_img):
       if cam_img is None:
           return 0.0, 0.0, None

       # work at LF_PROC_WIDTH; everything downstream (detection, learned
       # lane distances, overlay) lives in these downscaled coordinates
       if self.proc_width and cam_img.shape[1] > self.proc_width:
           pw = int(self.proc_width)
           ph = int(round(cam_img.shape[0] * pw / cam_img.shape[1]))
           cam_img = cv2.resize(cam_img, (pw, ph), interpolation=cv2.INTER_AREA)


       now = time.time()
       h, w = cam_img.shape[:2]
       found, x_near, heading, debug = self.detect(cam_img)

       yellow_fit = debug.get('fit')
       wl = wr = None
       if self.white_enabled and debug['white_pts']:
           wl, wr, _, _ = self._split_whites(debug['white_pts'], yellow_fit)
           debug['wl'], debug['wr'] = wl, wr

       if found:
           self._learn_lane(x_near, wl, wr, w)
           self.last_x = x_near
           error, target = self._steer_error(x_near, w)
           debug['target'] = target


           # PD + heading feed-forward
           d_err = 0.0
           if self.prev_error is not None and self.prev_time is not None:
               dt = max(now - self.prev_time, 1e-3)
               d_err = (error - self.prev_error) / dt
           self.prev_error, self.prev_time = error, now


           steer = self.kp * error + self.kd * d_err + self.kh * heading
           self.steering = float(np.clip(steer, -1.0, 1.0))


           # slow down proportionally to how hard we're steering
           self.throttle = self.th_max - (self.th_max - self.th_min) * abs(self.steering)
           self.status = "tracking"
           self.lost_frames = 0
           self.lost_since = None


       else:
           if self.lost_since is None:
               self.lost_since = now
           x_est = None
           if self.white_enabled and now - self.lost_since <= self.white_guide_sec:
               x_est = self._estimate_from_whites(wl, wr)
           if x_est is not None:
               # Yellow gone (dash gap, worn paint, deep shadow) but the
               # solid white lines are visible: steer from them. Only for
               # LF_WHITE_GUIDE_SEC after the last yellow fix — this
               # bridges gaps, it must not drive the car indefinitely on
               # white edges alone. The lost clock keeps running so the
               # coast/stop logic takes over when the window expires. No D
               # term across the mode switch — the error source changed.
               self.last_x = x_est
               error, target = self._steer_error(x_est, w)
               debug['target'] = target
               debug['x_est'] = x_est
               self.prev_error = None
               steer = self.kp * error
               self.steering = float(np.clip(steer, -1.0, 1.0))
               self.throttle = self.th_min
               self.status = "white-guided"
               self.lost_frames = 0
           else:
               self.lost_frames += 1
               self.prev_error = None


               if now - self.lost_since > self.lost_stop_sec:
                   # lost for too long — stop rather than wander off the track
                   self.steering, self.throttle = 0.0, 0.0
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
       for i, s in enumerate([
           f"st:{self.steering:+.2f} th:{self.throttle:.2f}",
           f"hdg:{heading:+.2f}{lane}",
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




if __name__ == '__main__':
   import sys
   if len(sys.argv) >= 3 and sys.argv[1] == 'test':
       args = sys.argv[2:]
       out = "lf_out"
       if "--out" in args:
           i = args.index("--out")
           out = args[i + 1]
           args = args[:i] + args[i + 2:]
       _test(args, out)
   else:
       print(__doc__)
