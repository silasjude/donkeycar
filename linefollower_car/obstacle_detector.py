#!/usr/bin/env python3
"""
obstacle_detector.py — DonkeyCar Track 2, Mission 3: obstacle maneuvering,
sensing half. MONOCULAR VERSION.

Rewritten 2026-07-26 to drop the OAK-D's stereo depth stream entirely.
Depth caused two repeated "Undervoltage detected!" crashes (dmesg) the
moment the VESC's motor-engage current spike landed on top of RGB+depth's
own draw -- a real combined power-budget limit, not something a config
change (lower depth FPS, non-blocking reads) fully solved. RGB alone has
never once caused a problem in any test on this car. This file now
estimates obstacle presence/side/distance from 'cam/image_array' ONLY --
no 'cam/depth_array' input, no dependence on OAKD_DEPTH being on at all.

--- HOW THIS WORKS: MONOCULAR GROUND-PLANE RANGING ---------------------
The camera is fixed-mount, looking down at a flat track -- the same
geometry line_following.py already leans on for its ROI/vanishing-point
math (LF_VP_Y_FRAC). For an object resting ON the ground, the image ROW
where its BASE touches the ground maps to a real-world distance: farther
ground points sit higher in the frame (nearer the horizon), closer ones
lower. The model used is

    1 / distance_m = OBSTACLE_MONO_SLOPE * (row - horizon_row)

pinned so row <= horizon_row means "at or above the horizon" (not on the
visible ground plane / not actionable). This needs exactly one number,
OBSTACLE_MONO_SLOPE, and it MUST be calibrated for real distances to mean
anything -- see "HOW TO CALIBRATE" below. Until then, DEFAULTS ships a
placeholder slope (assuming the bottom row of frame is ~0.3m away) purely
so the pipeline runs sensibly out of the box: presence/side detection and
relative near/far ordering are correct regardless (the model is
monotonic in row by construction), only the absolute meters -- and
therefore exactly where OBSTACLE_SLOW_DIST_M/STOP_DIST_M trigger -- are
approximate until calibrated.

Detection itself is color/shape based, NOT a classifier: a candidate
obstacle is any sufficiently large blob whose color meaningfully departs
from the local pavement background (the same "local contrast vs.
median-blurred background" technique line_following.py uses for the
yellow dash and white lines), EXCLUDING anything that looks like the
tracked yellow dash or white boundary paint itself -- the detection
corridor is centered ON the yellow line by construction, so without this
exclusion the car's own tracked lane markings would register as
obstacles on every frame. The safety decision (present/side/distance) is
still deliberately generic, not cone-specific -- it should catch any
solid foreign object sitting on the track, cone or otherwise. A separate
hue check adds a "how cone-like is this" hint for telemetry only.

Known limitations of going monocular (real, not hidden):
  - An obstacle whose color and lightness closely match the pavement
    (camouflaged, in the statistical sense) can be missed entirely --
    no stereo/motion cue backs this up. Nothing currently mitigates this.
  - Distance depends on the object's base actually being visible on the
    ground and the ground actually being flat between here and there;
    an occluded base or a slope/bump breaks the model.
  - (2026-07-27) This HAS now been validated against several real test
    drives and their recorded footage -- presence/side detection and
    near-vs-far ordering behave as designed. Absolute distances remain
    placeholder-scaled until OBSTACLE_MONO_SLOPE is calibrated (below).

--- HOW TO CALIBRATE (OBSTACLE_MONO_SLOPE) -------------------------------
Park (or hold) a real reference object (e.g. the same cone you'll test
avoidance with) at one known, tape-measured distance from the car,
pointing the camera at it normally, and save one frame. Then:
  python3 obstacle_detector.py calibrate <frame.jpg> <distance_m>
It runs the same detection pipeline on that frame, picks the largest
qualifying blob in the ROI as the reference object, and prints the
OBSTACLE_MONO_SLOPE line to paste into myconfig.py. Several reference
shots at different distances average out row-quantization error -- pass
them all at once and the CLI averages for you:
  python3 obstacle_detector.py calibrate f1.jpg 0.5 f2.jpg 1.0 f3.jpg 2.0
One pair is enough to get real numbers instead of the placeholder.

--- HOW TO TEST OFF THE CAR -----------------------------------------------
  python3 obstacle_detector.py selftest
Synthetic-image checks: presence/side/near-vs-far ordering, that the
tracked yellow dash / white lines never register as obstacles, and that
small color speckle noise doesn't false-trigger. No camera or car needed.
-----------------------------------------------------------------------------
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Tunable defaults. Every value can be overridden in myconfig.py; the part
# reads them from cfg with these as fallbacks, so it runs even with a bare cfg.
# =============================================================================
DEFAULTS = dict(
   # --- corridor geometry (cam/image_array coords, i.e. cfg.IMAGE_W x
   # cfg.IMAGE_H -- the OAK-D preview already outputs this size directly) ---
   OBSTACLE_ROI_TOP=0.45,        # start scanning a bit above
                                  # line_following's own LF_ROI_TOP
                                  # (0.5-0.62 typically) so an obstacle is
                                  # seen before the car is right on top of
                                  # where LineFollower's ROI begins
   OBSTACLE_LANE_HALF_WIDTH_PX=90,   # per-side corridor band width used
                                  # when the live LineFollower lane
                                  # widths aren't known (no learned/
                                  # pinned LF_LANE_WIDTH_PX available)
   # Fraction of each side's OWN lane width the corridor band spans,
   # applied per side (left band uses lane_width_l, right band uses
   # lane_width_r). Added 2026-07-27 evening after a real drive: a
   # full-size cone dead in the car's own path was RANGED continuously
   # from 17.9m in (idx 111872-111905, monotonic 17.9->0.95m) but
   # `present` stayed False until 0.77m because its centroid (cx
   # 215-240) sat 10-60px outside the old corridor edge -- the old
   # half-width was mean(lane widths)*0.55 = 85px, but the car rides at
   # +0.5 offset and its own lane physically spans the FULL
   # lane_width_r (140px, pinned in LF_LANE_WIDTH_PX) from the yellow
   # line to the white boundary. 1.0 = each band covers exactly its
   # side's whole lane, which is the corridor's documented intent
   # ("own lane" band / "opposing lane" band). The 0.55*mean factor
   # predated the per-side pinned widths and was never revisited.
   OBSTACLE_CORRIDOR_LANE_FRAC=1.0,
   OBSTACLE_ADJACENT_GAP_PX=25,  # skip this many px past the corridor
                                  # edge before the "adjacent lane" scan
                                  # band starts, so the yellow/white paint
                                  # itself isn't mistaken for an obstacle
                                  # sitting at the lane boundary
   # Corridor coverage while the wired LineFollower has NO lateral fix at
   # all yet (x_f AND last_x both None -- i.e. the first frames after
   # process start, before its first accepted yellow fix). Measured on
   # all three real recorded session starts (2026-07-27, idx 108545 /
   # 107772 / 107001): the first cone's blob fully qualifies on the very
   # FIRST frame (cx=22-72px, 0.75-0.85m) but the un-seeded corridor
   # defaults to frame center 192±85 and can't see it -- at idx 108545
   # the cone sat at cx=22 with x_f later converging to -21, so no
   # centered corridor of ANY reasonable width covers it; only a
   # full-width scan does. During this bootstrap window the detector
   # reports present/distance with side=None ("something ahead, lateral
   # position unresolved" -- obstacle_avoidance.py's fail-safe holds
   # lane and slows on that combination, it never guesses a direction),
   # because classifying left-vs-right against an arbitrary assumed
   # center would be a guess: at idx 108545 the cone (cx=22) was
   # OWN-lane (right of the line at -21) but sits far LEFT of frame
   # center -- a frame-center side call would have committed a swerve
   # toward the cone.
   OBSTACLE_BOOTSTRAP_HALF_WIDTH_FRAC=0.5,  # half-width as a fraction of
                                  # image width; 0.5 = full frame

   # --- monocular ground-plane ranging (see module docstring) ---
   OBSTACLE_HORIZON_Y_FRAC=0.6,  # matches line_following.py's own
                                  # LF_VP_Y_FRAC vanishing-point row;
                                  # override independently if that ever
                                  # needs to diverge
   # PLACEHOLDER pending real calibration (see module docstring/
   # `calibrate` CLI): assumes the bottom row of the frame is ~0.3m away.
   # Presence/side detection and near-vs-far ordering are correct
   # regardless of this value (the model is monotonic in row by
   # construction); only the absolute meters -- and therefore exactly
   # where OBSTACLE_SLOW_DIST_M/STOP_DIST_M trigger -- depend on it being
   # right.
   OBSTACLE_MONO_SLOPE=0.04,

   # --- foreign-object color mask ---
   # Pavement is close to color-NEUTRAL in LAB (a,b both ~128, chroma
   # ~0), regardless of how bright or dark the lighting is -- measured on
   # real gray pavement, and consistent with why line_following.py's own
   # hue-guard/a-channel filters treat LAB chroma as lighting-invariant.
   # A real foreign object (cone, car, anything else solid, verified
   # against a handful of sample colors) reads chroma ~70-100, cleanly
   # separated from pavement's ~0.
   #
   # This is deliberately an ABSOLUTE threshold, not local-background-
   # relative: an earlier local-median-blur version self-contaminated on
   # any object big enough to dominate its own blur window -- exactly
   # the case that matters most (a close, large obstacle almost filling
   # the corridor), since the "local background" ended up sampling the
   # object itself and reading near-zero deviation right where the
   # object was. Absolute chroma has no such failure mode. It also
   # deliberately does not try to name what the object is.
   OBSTACLE_FOREIGN_CHROMA=30,   # min LAB chroma (distance from neutral
                                  # a=b=128) to count as "colorful enough
                                  # to be a real object", not pavement
   OBSTACLE_MIN_BLOB_AREA_FRAC=0.003,  # min blob size, fraction of the
                                      # scanned ROI area -- much bigger
                                      # than line_following's own dash
                                      # threshold (LF_BLOB_MIN_AREA_FRAC
                                      # =0.0001): an obstacle close enough
                                      # to matter should be a large blob,
                                      # not a sliver
   # Found from the first real test drive: a permanent painted ground
   # marking (unrelated to the cone) on this track is colorful enough to
   # pass the chroma gate, and fragments into several small blobs that
   # nonetheless individually or collectively read as a persistent
   # close, flickering "obstacle" -- the exact recurring distance values
   # in that session's log matched this marking, not the cone (confirmed
   # by replaying the recorded frames). What actually separates a flat
   # painted line from a real solid object here isn't area, it's height:
   # the marking's fragments were 1-6px tall; the cone's own blob (same
   # frame) was 37px tall. This is the same principle line_following.py
   # already uses in the other direction (LF_BLOB_MAX_HEIGHT_RATIO: a
   # real ground dash is short for its distance, a background object is
   # tall) -- here a real solid obstacle close enough to matter should
   # have real vertical extent, a flat ground marking should not.
   OBSTACLE_MIN_BLOB_HEIGHT_PX=15,

   # Exclusion bands: pixels matching these are NEVER foreign-object
   # candidates, no matter how they score above, because the detection
   # corridor is centered ON the yellow line by construction -- without
   # this, the car's own tracked lane markings would read as obstacles
   # every frame. Deliberately generous/inclusive (wider than
   # line_following.py's own LF_HUE_MIN/MAX=10/32) -- occasionally
   # excluding a real orange-yellow cone at the boundary is an accepted
   # trade for never flagging the tracked line itself.
   OBSTACLE_EXCLUDE_HUE_MIN=8,
   OBSTACLE_EXCLUDE_HUE_MAX=36,
   OBSTACLE_EXCLUDE_WHITE_L_MIN=170,   # bright + near-neutral color =
   OBSTACLE_EXCLUDE_WHITE_CHROMA_MAX=15,  # paint-white signature

   # Found on the first two real test drives: this track also has a
   # permanent BLUE painted ground marking (unrelated to the cone,
   # purpose unknown -- a lane/zone marker of some kind). Sampled
   # directly from recorded frames: HSV hue clusters at 108-111, with
   # chroma often sitting right at OBSTACLE_FOREIGN_CHROMA's threshold
   # (21-35) -- close enough to the gate that per-frame JPEG/lighting
   # noise flipped it in and out of detection, which is what the
   # min-blob-height fix didn't fully catch (a flat strip viewed at a
   # close, grazing angle can still appear tall in the image -- height
   # alone can't distinguish "close and thin" from "close and solid").
   # Excluding its known hue directly is the more reliable fix, the same
   # way yellow/white lane paint is excluded above.
   OBSTACLE_EXCLUDE_BLUE_HUE_MIN=95,
   OBSTACLE_EXCLUDE_BLUE_HUE_MAX=130,

   # --- decision thresholds (meters) ---
   OBSTACLE_SLOW_DIST_M=2.2,     # start reacting (slow, and swerve/plan)
   OBSTACLE_STOP_DIST_M=0.6,     # throttle floor reached here -- see
                                  # obstacle_avoidance.py's speed ramp

   # --- color cone hint (telemetry / future VLM hook only; NOT used for
   # the present/side/distance decision above) ---
   OBSTACLE_CONE_HUE_MIN=0,
   OBSTACLE_CONE_HUE_MAX=14,
   OBSTACLE_CONE_SAT_MIN=110,
   OBSTACLE_CONE_VAL_MIN=90,

   OVERLAY=True,
)


def _cfg(cfg, name):
   """Read a value from cfg, falling back to DEFAULTS."""
   return getattr(cfg, name, DEFAULTS[name]) if cfg is not None else DEFAULTS[name]


class ObstacleDetector:
   """
   DonkeyCar part.
     input:  'cam/image_array' (RGB, cfg.IMAGE_W x IMAGE_H) -- ONLY. No
             depth input; see module docstring for why.
     output: 'obstacle/info' — dict, see _empty_info() for the schema
             (unchanged from the depth version, so obstacle_avoidance.py
             and manage_line.py's wiring don't need to change)

   line_follower, if given, is a reference to the running LineFollower
   instance (read-only here). When set, the corridor is centered on
   line_follower.x_f (its filtered lateral measurement) and sized from
   its learned lane width, so it tracks the actual lane instead of
   assuming the camera is perfectly
   centered on it.
   """

   def __init__(self, cfg=None, line_follower=None):
       self.roi_top = float(_cfg(cfg, 'OBSTACLE_ROI_TOP'))
       self.half_width_default = float(_cfg(cfg, 'OBSTACLE_LANE_HALF_WIDTH_PX'))
       self.adjacent_gap = float(_cfg(cfg, 'OBSTACLE_ADJACENT_GAP_PX'))
       self.corridor_lane_frac = float(_cfg(cfg, 'OBSTACLE_CORRIDOR_LANE_FRAC'))
       self.bootstrap_half_width_frac = float(
           _cfg(cfg, 'OBSTACLE_BOOTSTRAP_HALF_WIDTH_FRAC'))

       self.horizon_y_frac = float(_cfg(cfg, 'OBSTACLE_HORIZON_Y_FRAC'))
       self.mono_slope = float(_cfg(cfg, 'OBSTACLE_MONO_SLOPE'))

       self.chroma_thresh = float(_cfg(cfg, 'OBSTACLE_FOREIGN_CHROMA'))
       self.min_blob_area_frac = float(_cfg(cfg, 'OBSTACLE_MIN_BLOB_AREA_FRAC'))
       self.min_blob_height_px = float(_cfg(cfg, 'OBSTACLE_MIN_BLOB_HEIGHT_PX'))
       self.exclude_hue_min = int(_cfg(cfg, 'OBSTACLE_EXCLUDE_HUE_MIN'))
       self.exclude_hue_max = int(_cfg(cfg, 'OBSTACLE_EXCLUDE_HUE_MAX'))
       self.exclude_white_l_min = float(_cfg(cfg, 'OBSTACLE_EXCLUDE_WHITE_L_MIN'))
       self.exclude_white_chroma_max = float(_cfg(cfg, 'OBSTACLE_EXCLUDE_WHITE_CHROMA_MAX'))
       self.exclude_blue_hue_min = int(_cfg(cfg, 'OBSTACLE_EXCLUDE_BLUE_HUE_MIN'))
       self.exclude_blue_hue_max = int(_cfg(cfg, 'OBSTACLE_EXCLUDE_BLUE_HUE_MAX'))

       self.slow_dist_m = float(_cfg(cfg, 'OBSTACLE_SLOW_DIST_M'))
       self.stop_dist_m = float(_cfg(cfg, 'OBSTACLE_STOP_DIST_M'))
       self.hue_min = int(_cfg(cfg, 'OBSTACLE_CONE_HUE_MIN'))
       self.hue_max = int(_cfg(cfg, 'OBSTACLE_CONE_HUE_MAX'))
       self.sat_min = int(_cfg(cfg, 'OBSTACLE_CONE_SAT_MIN'))
       self.val_min = int(_cfg(cfg, 'OBSTACLE_CONE_VAL_MIN'))
       self.overlay = bool(_cfg(cfg, 'OVERLAY'))
       self.lf = line_follower

       self._warned_uncalibrated = False
       logger.info(
           "ObstacleDetector (monocular) up: roi_top=%.2f half_width=%.0fpx "
           "horizon_y_frac=%.2f mono_slope=%.4f slow=%.1fm stop=%.1fm "
           "min_blob_area_frac=%.4f min_blob_height_px=%.0f",
           self.roi_top, self.half_width_default, self.horizon_y_frac,
           self.mono_slope, self.slow_dist_m, self.stop_dist_m,
           self.min_blob_area_frac, self.min_blob_height_px)
       logger.warning(
           "ObstacleDetector: OBSTACLE_MONO_SLOPE is a PLACEHOLDER until "
           "calibrated (python3 obstacle_detector.py calibrate <frame> "
           "<distance_m>) -- presence/side detection and near-vs-far "
           "ordering are correct either way, but absolute distances (and "
           "therefore exactly when OBSTACLE_SLOW_DIST_M/STOP_DIST_M "
           "trigger) are only approximate until then.")

   # ------------------------------------------------------------------ #
   def _empty_info(self, status):
       return dict(
           present=False, distance_m=None, side=None,
           left_blocked=False, right_blocked=False,
           left_min_m=None, right_min_m=None,
           left_adjacent_clear=None, right_adjacent_clear=None,
           cone_score=0.0, status=status, corridor_locked=None,
       )

   def _corridor(self, w):
       """Return (center_x, width_left, width_right, locked) in
       image-pixel coords for this frame, tracking the live lane when a
       LineFollower reference is available and has current lane info.

       width_left/width_right are PER-SIDE band widths: the left
       (opposing-lane) band spans [cx - width_left, cx) and the right
       (own-lane) band [cx, cx + width_right), each sized from its own
       side's lane width x OBSTACLE_CORRIDOR_LANE_FRAC. They were one
       symmetric mean(lane widths)*0.55 half-width until 2026-07-27 --
       see OBSTACLE_CORRIDOR_LANE_FRAC's DEFAULTS comment for the real
       drive where that left a cone dead in the car's own path
       undetected until 0.77m despite being ranged from 17.9m in.

       Uses x_f (LineFollower's FILTERED lateral measurement -- what it
       actually steers on), not last_x (the raw, unfiltered one). Found
       from a real test replay: last_x jumps 50-150px frame to frame from
       ordinary detection noise (the same jitter line_following.py's own
       tuning notes describe at length), and once a close obstacle sits
       near that jittering boundary, it flips between "left"/"right"
       almost every tick -- the commander then reverses between
       lane_change_left and swerve_right continuously and never
       completes either maneuver. Switching to x_f cut status transitions
       across a 400-frame replay of that exact session from constant
       flip-flopping to 9, with 70+ stable frames between changes --
       enough for LF_LANE_RAMP_SEC's ramp to actually finish. Falls back
       to last_x if x_f isn't populated yet (post-stop reseed: x_f is
       cleared but last_x keeps the last real position).

       locked=False means a LineFollower IS wired but has never produced
       any lateral fix at all yet (x_f AND last_x both None -- the first
       frames after process start, or indefinitely if it never locks on).
       There is no lane position to center on, so the caller must not
       classify left-vs-right from the returned center -- see
       OBSTACLE_BOOTSTRAP_HALF_WIDTH_FRAC's comment for the real
       recorded frames where a frame-center side call would have
       committed a swerve TOWARD the cone. With no LineFollower wired at
       all (standalone/CLI/selftest use) the camera is assumed centered
       on the lane, as before, and the corridor counts as locked."""
       cx = w / 2.0
       w_left = w_right = self.half_width_default
       locked = True
       if self.lf is not None:
           lx = getattr(self.lf, 'x_f', None)
           if lx is None:
               lx = getattr(self.lf, 'last_x', None)
           if lx is not None:
               cx = float(lx)
           else:
               locked = False
               w_left = w_right = w * self.bootstrap_half_width_frac
           if locked:
               lw_l = getattr(self.lf, 'lane_width_l', None)
               lw_r = getattr(self.lf, 'lane_width_r', None)
               # each band from its own side's width; fall back to the
               # other side's, then to the configured default
               if lw_l or lw_r:
                   w_left = float(lw_l or lw_r) * self.corridor_lane_frac
                   w_right = float(lw_r or lw_l) * self.corridor_lane_frac
       return cx, w_left, w_right, locked

   def _row_to_distance_m(self, row):
       """Monocular ground-plane distance from image row. Monotonic in
       row by construction (closer -> lower row -> larger 1/distance),
       regardless of whether OBSTACLE_MONO_SLOPE has been calibrated."""
       delta = row - self.horizon_y_frac * self._h
       if delta <= 0:
           return None  # at/above the horizon -- not on the visible ground
       return 1.0 / (self.mono_slope * delta)

   def _foreign_mask(self, roi_rgb):
       """Boolean mask of pixels that look like a real foreign object --
       ABSOLUTE LAB chroma far from neutral (see OBSTACLE_FOREIGN_CHROMA
       for why this is absolute, not local-background-relative) --
       excluding anything matching the tracked yellow dash, white
       boundary paint, or this track's blue ground marking.
       """
       lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
       L = lab[:, :, 0].astype(np.float32)
       A = lab[:, :, 1].astype(np.float32)
       B = lab[:, :, 2].astype(np.float32)

       chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
       foreign = chroma > self.chroma_thresh

       hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
       hue = hsv[:, :, 0]
       yellow_like = (hue >= self.exclude_hue_min) & (hue <= self.exclude_hue_max)
       white_like = (L > self.exclude_white_l_min) & (chroma < self.exclude_white_chroma_max)
       blue_like = (hue >= self.exclude_blue_hue_min) & (hue <= self.exclude_blue_hue_max)

       foreign &= ~yellow_like
       foreign &= ~white_like
       foreign &= ~blue_like
       return foreign.astype(np.uint8)

   def _find_blobs(self, mask, roi_y0, min_area):
       """Connected foreign-object blobs. Returns list of
       (bottom_row_full_image, centroid_x_full_image, area, bbox) sorted
       by nothing in particular -- caller buckets by position."""
       num, _labels, stats, centroids = cv2.connectedComponentsWithStats(
           mask, connectivity=8)
       blobs = []
       for i in range(1, num):  # 0 is background
           area = stats[i, cv2.CC_STAT_AREA]
           h = stats[i, cv2.CC_STAT_HEIGHT]
           if area < min_area or h < self.min_blob_height_px:
               continue
           x0 = stats[i, cv2.CC_STAT_LEFT]
           y0 = stats[i, cv2.CC_STAT_TOP]
           w = stats[i, cv2.CC_STAT_WIDTH]
           bottom_row = roi_y0 + y0 + h - 1
           cx = float(centroids[i][0])
           blobs.append(dict(bottom_row=bottom_row, cx=cx, area=int(area),
                              bbox=(x0, y0, w, h)))
       return blobs

   def _nearest(self, band_blobs):
       """(distance_m, blob) of the closest (largest bottom_row)
       qualifying blob in a band, or (None, None)."""
       if not band_blobs:
           return None, None
       best = max(band_blobs, key=lambda b: b['bottom_row'])
       return self._row_to_distance_m(best['bottom_row']), best

   def _cone_score_for_blob(self, roi_rgb, blob):
       x0, y0, w, h = blob['bbox']
       if w <= 0 or h <= 0:
           return 0.0
       patch = roi_rgb[y0:y0 + h, x0:x0 + w, :]
       if patch.size == 0:
           return 0.0
       hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
       mask = cv2.inRange(
           hsv,
           (self.hue_min, self.sat_min, self.val_min),
           (self.hue_max, 255, 255))
       return float(np.count_nonzero(mask)) / float(mask.size)

   # ------------------------------------------------------------------ #
   def run(self, cam_img):
       if cam_img is None:
           return self._empty_info("no_image")

       h, w = cam_img.shape[:2]
       self._h = h  # used by _row_to_distance_m
       roi_y0 = int(self.roi_top * h)
       roi_rgb = cam_img[roi_y0:, :, :]
       if roi_rgb.shape[0] < 4:
           return self._empty_info("roi_too_small")

       mask = self._foreign_mask(roi_rgb)
       min_area = max(1, int(self.min_blob_area_frac * roi_rgb.shape[0] * roi_rgb.shape[1]))
       blobs = self._find_blobs(mask, roi_y0, min_area)

       cx, w_left, w_right, locked = self._corridor(w)

       if not locked:
           # Bootstrap window: the wired LineFollower has never produced
           # a lateral fix, so there is no lane position to bucket blobs
           # against. Scan one wide band and report presence/distance
           # with side=None -- obstacle_avoidance.py's fail-safe branch
           # holds lane and slows on that combination rather than
           # guessing a direction. See OBSTACLE_BOOTSTRAP_HALF_WIDTH_FRAC
           # in DEFAULTS for the recorded frames this covers (a cone at
           # cx=22 on the very first frame, 0.85m out).
           dist, blob = self._nearest(
               [b for b in blobs if cx - w_left <= b['cx'] < cx + w_right])
           present = dist is not None and dist <= self.slow_dist_m
           cone_score = (self._cone_score_for_blob(roi_rgb, blob)
                         if blob is not None else 0.0)
           return dict(
               present=present, distance_m=dist, side=None,
               left_blocked=False, right_blocked=False,
               left_min_m=None, right_min_m=None,
               left_adjacent_clear=None, right_adjacent_clear=None,
               cone_score=cone_score,
               status='obstacle_unlocalized' if present else 'clear',
               corridor_locked=False,
           )

       bands = dict(left=[], right=[], left_adj=[], right_adj=[])
       for b in blobs:
           x = b['cx']
           if cx - w_left <= x < cx:
               bands['left'].append(b)
           elif cx <= x < cx + w_right:
               bands['right'].append(b)
           elif cx - 2 * w_left - self.adjacent_gap <= x < cx - w_left - self.adjacent_gap:
               bands['left_adj'].append(b)
           elif cx + w_right + self.adjacent_gap <= x < cx + 2 * w_right + self.adjacent_gap:
               bands['right_adj'].append(b)

       left_min, left_blob = self._nearest(bands['left'])
       right_min, right_blob = self._nearest(bands['right'])
       left_adj_min, _ = self._nearest(bands['left_adj'])
       right_adj_min, _ = self._nearest(bands['right_adj'])

       left_blocked = left_min is not None and left_min <= self.slow_dist_m
       right_blocked = right_min is not None and right_min <= self.slow_dist_m
       candidates = [d for d in (left_min, right_min) if d is not None]
       distance_m = min(candidates) if candidates else None
       present = left_blocked or right_blocked

       side = None
       if left_blocked and right_blocked:
           side = 'both'
       elif left_blocked:
           side = 'left'
       elif right_blocked:
           side = 'right'

       cone_score = 0.0
       if side in ('left', 'both') and left_blob is not None:
           cone_score = max(cone_score, self._cone_score_for_blob(roi_rgb, left_blob))
       if side in ('right', 'both') and right_blob is not None:
           cone_score = max(cone_score, self._cone_score_for_blob(roi_rgb, right_blob))

       status = 'clear'
       if side == 'both':
           status = 'obstacle_both'
       elif side:
           status = f'obstacle_{side}'

       return dict(
           present=present, distance_m=distance_m, side=side,
           left_blocked=left_blocked, right_blocked=right_blocked,
           left_min_m=left_min, right_min_m=right_min,
           left_adjacent_clear=(left_adj_min is None or left_adj_min > self.slow_dist_m),
           right_adjacent_clear=(right_adj_min is None or right_adj_min > self.slow_dist_m),
           cone_score=cone_score, status=status, corridor_locked=True,
       )

   def shutdown(self):
       pass


# ============================================================================
# calibration CLI -- no camera/car needed, works from a saved frame
# ============================================================================
def _calibrate_one(det, path, distance_m):
   """Slope from one (frame, measured distance) pair, or None with a
   printed reason."""
   bgr = cv2.imread(path)
   if bgr is None:
       print(f"{path}: can't read image file")
       return None
   img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
   h, w = img.shape[:2]
   det._h = h
   roi_y0 = int(det.roi_top * h)
   roi_rgb = img[roi_y0:, :, :]
   mask = det._foreign_mask(roi_rgb)
   min_area = max(1, int(det.min_blob_area_frac * roi_rgb.shape[0] * roi_rgb.shape[1]))
   blobs = det._find_blobs(mask, roi_y0, min_area)
   if not blobs:
       print(f"{path}: no qualifying blob found in the ROI -- check the "
             "reference object is in frame, colorful enough to pass "
             "OBSTACLE_FOREIGN_CHROMA, and not being excluded as "
             "yellow/white paint (a very yellow-orange object can land in "
             "that excluded band -- see the module docstring).")
       return None
   ref = max(blobs, key=lambda b: b['area'])
   delta = ref['bottom_row'] - det.horizon_y_frac * h
   if delta <= 0:
       print(f"{path}: reference blob's base is at/above the assumed "
             "horizon row -- can't calibrate from this frame (check "
             "OBSTACLE_HORIZON_Y_FRAC or move the reference object "
             "further into frame).")
       return None
   slope = (1.0 / float(distance_m)) / delta
   print(f"{path}: bottom_row={ref['bottom_row']} area={ref['area']}px "
         f"at {distance_m}m -> slope {slope:.5f}")
   return slope


def _calibrate(pairs):
   """pairs: list of (frame_path, distance_m). Averages the per-pair
   slopes -- several reference distances beat one (row quantization)."""
   det = ObstacleDetector()
   slopes = [s for path, dist in pairs
             if (s := _calibrate_one(det, path, dist)) is not None]
   if not slopes:
       print("No usable reference frames -- nothing to calibrate from.")
       return
   avg = float(np.mean(slopes))
   if len(slopes) > 1:
       print(f"{len(slopes)} usable references, slope spread "
             f"{min(slopes):.5f}..{max(slopes):.5f}")
   print(f"add to myconfig.py:  OBSTACLE_MONO_SLOPE = {avg:.5f}")


# ============================================================================
# offline self-test -- no camera/car needed
# ============================================================================
def _make_synthetic(w=384, h=216, roi_top=0.45, obstacle_side=None,
                     bottom_row=200, cx=None, half_w=90,
                     color=(40, 180, 60), patch_h=25, patch_w_frac=0.6):
   """Neutral gray 'pavement' with an optional colored patch whose BOTTOM
   sits at `bottom_row` (controls the estimated distance). Default color
   is a saturated green (LAB chroma ~78, HSV hue ~64) -- deliberately far
   outside the yellow/white/blue exclusion bands (blue was added after a
   real track's blue ground marking turned out to sit right at hue~109),
   so these tests exercise generic foreign-object detection without
   tripping over the real, separate ambiguity that a genuinely
   orange-yellow object can share hue with the track tape (see the
   cone_score test below for that case)."""
   img = np.full((h, w, 3), 128, dtype=np.uint8)
   cx = cx if cx is not None else w / 2.0

   def paint(x0, x1):
       x0, x1 = int(max(0, x0)), int(min(w, x1))
       y0, y1 = max(0, bottom_row - patch_h), min(h, bottom_row)
       img[y0:y1, x0:x1] = color

   pw = half_w * patch_w_frac
   if obstacle_side in ('left', 'both'):
       paint(cx - half_w, cx - half_w + pw)
   if obstacle_side in ('right', 'both'):
       paint(cx, cx + pw)
   return img


def _paint_yellow_dash(img, cx, row, w=30, h=8):
   """Cone-orange-adjacent but within the excluded yellow-hue band --
   simulates the tracked dash so it can be confirmed NOT to trigger."""
   x0, x1 = int(cx - w // 2), int(cx + w // 2)
   y0, y1 = int(row - h // 2), int(row + h // 2)
   img[y0:y1, x0:x1] = (210, 190, 40)  # yellow-ish RGB
   return img


def _check(name, cond, info):
   print(f"[{'PASS' if cond else 'FAIL'}] {name}  ->  {info}")
   return cond


def _selftest():
   ok = True
   det = ObstacleDetector()

   img = _make_synthetic(obstacle_side=None)
   info = det.run(img)
   ok &= _check("clear track reports not present", not info['present'], info)

   img = _make_synthetic(obstacle_side='left', bottom_row=210)
   info = det.run(img)
   ok &= _check("left obstacle -> present, side=left",
                info['present'] and info['side'] == 'left', info)
   ok &= _check("right side reported clear", not info['right_blocked'], info)

   img = _make_synthetic(obstacle_side='right', bottom_row=205)
   info = det.run(img)
   ok &= _check("right obstacle -> present, side=right",
                info['present'] and info['side'] == 'right', info)

   img = _make_synthetic(obstacle_side='both', bottom_row=215)
   info = det.run(img)
   ok &= _check("full-width obstacle -> side=both", info['side'] == 'both', info)

   # near-vs-far ordering must hold regardless of calibration
   near_img = _make_synthetic(obstacle_side='left', bottom_row=215)
   far_img = _make_synthetic(obstacle_side='left', bottom_row=140)
   near_info = det.run(near_img)
   far_info = det.run(far_img)
   ok &= _check("row nearer the bottom of frame -> smaller distance_m",
                (near_info['distance_m'] is not None
                 and far_info['distance_m'] is not None
                 and near_info['distance_m'] < far_info['distance_m']),
                (near_info['distance_m'], far_info['distance_m']))

   # the tracked yellow dash must NEVER register as an obstacle -- the
   # corridor is centered on it by construction
   img = _make_synthetic(obstacle_side=None)
   cx = det._corridor(img.shape[1])[0]
   img = _paint_yellow_dash(img, cx, row=190)
   info = det.run(img)
   ok &= _check("tracked yellow dash never registers as an obstacle",
                not info['present'], info)

   # small color speckle noise shouldn't trigger (below min blob area)
   img = _make_synthetic(obstacle_side=None)
   img[150:153, 150:153] = (40, 180, 60)  # tiny 3x3 speck, same green as
                                           # the main test color -- isolates
                                           # the size gate specifically
   info = det.run(img)
   ok &= _check("tiny color speck doesn't trigger present",
                not info['present'], info)

   # the track's real blue ground marking (sampled directly from a
   # recorded frame: RGB~(85,115,165), HSV hue~109) must never register,
   # even as a big, tall, otherwise-qualifying blob -- this is what the
   # min-height gate alone didn't catch (a flat strip at a close, grazing
   # angle can still look tall in the image)
   img = _make_synthetic(obstacle_side='left', bottom_row=210,
                         color=(85, 115, 165))
   info = det.run(img)
   ok &= _check("real blue ground-marking color never registers as an obstacle",
                not info['present'], info)

   # a wide but FLAT painted ground marking (found on the real track: a
   # blue marking unrelated to any obstacle passed the color/area gates
   # but was only 1-6px tall) must be rejected even though it's plenty
   # wide/big in area -- this is the min-height gate's whole reason to
   # exist
   img = _make_synthetic(obstacle_side=None)
   cx, half_w = det._corridor(img.shape[1])[:2]
   x0, x1 = int(cx - half_w), int(cx)
   img[208:212, x0:x1] = (40, 180, 60)  # 4px tall, wide -- big area, short
   info = det.run(img)
   ok &= _check("wide-but-flat ground marking doesn't trigger present",
                not info['present'], info)

   # no image -> fails safe
   info = det.run(None)
   ok &= _check("no image -> not present, status=no_image",
                not info['present'] and info['status'] == 'no_image', info)

   # --- bootstrap corridor (wired LineFollower, no lateral fix yet) ---
   # Mirrors the real recorded session starts (2026-07-27, idx 108545:
   # cone blob at cx=22 on the very first frame, far outside any
   # centered corridor). A wired-but-never-locked LineFollower must NOT
   # leave the detector blind: presence/distance must be reported from
   # the full-width bootstrap scan, with side=None (no guessing which
   # side of an unknown lane the object is on).
   class _UnlockedLF:
       x_f = None
       last_x = None
       lane_width_l = None
       lane_width_r = None

   det_boot = ObstacleDetector(line_follower=_UnlockedLF())
   img = _make_synthetic(obstacle_side='right', bottom_row=210, cx=30,
                         half_w=30)  # patch at x~30-48: frame-edge cone
   info = det_boot.run(img)
   ok &= _check("bootstrap: frame-edge obstacle detected with side=None",
                info['present'] and info['side'] is None
                and info['distance_m'] is not None
                and info['status'] == 'obstacle_unlocalized'
                and info['corridor_locked'] is False, info)

   img = _make_synthetic(obstacle_side=None)
   info = det_boot.run(img)
   ok &= _check("bootstrap: clear frame stays not-present",
                not info['present'] and info['status'] == 'clear', info)

   # exclusion bands still apply during bootstrap -- a yellow dash or the
   # track's blue marking at the frame edge must not read as an obstacle
   img = _make_synthetic(obstacle_side=None)
   img = _paint_yellow_dash(img, 40, row=190)
   info = det_boot.run(img)
   ok &= _check("bootstrap: yellow dash still never registers",
                not info['present'], info)
   img = _make_synthetic(obstacle_side='right', bottom_row=210, cx=30,
                         half_w=30, color=(85, 115, 165))
   info = det_boot.run(img)
   ok &= _check("bootstrap: blue ground-marking color still never registers",
                not info['present'], info)

   # once the SAME wired LineFollower produces a fix, the corridor locks
   # onto it and side classification resumes
   lf_locked = _UnlockedLF()
   lf_locked.x_f = 20.0
   det_boot2 = ObstacleDetector(line_follower=lf_locked)
   img = _make_synthetic(obstacle_side='right', bottom_row=210, cx=30,
                         half_w=30)  # same frame-edge patch, cx~39 > x_f=20
   info = det_boot2.run(img)
   ok &= _check("locked corridor on the same frame classifies side again",
                info['present'] and info['side'] == 'right'
                and info['corridor_locked'] is True, info)

   # --- per-side corridor band widths (2026-07-27 evening) ---
   # With pinned asymmetric lane widths (LF_LANE_WIDTH_PX=(170,140)),
   # each band must span ITS side's full lane, not the old symmetric
   # mean*0.55=85px. Mirrors the real drive where a cone at ~cx+103,
   # dead in the car's own (right) lane, went undetected until 0.77m:
   # in-band under per-side widths, outside the old 85px half-width.
   class _WidthLF:
       x_f = 100.0
       last_x = 100.0
       lane_width_l = 170.0
       lane_width_r = 140.0

   det_w = ObstacleDetector(line_follower=_WidthLF())
   img = _make_synthetic(obstacle_side=None)
   img[185:210, 205:235] = (40, 180, 60)  # cx~220 = x_f+120: outside the
                                           # old 85px band, inside the
                                           # 140px own-lane band
   info = det_w.run(img)
   ok &= _check("own-lane obstacle beyond the old 85px half-width now detected, side=right",
                info['present'] and info['side'] == 'right', info)

   img = _make_synthetic(obstacle_side=None)
   img[185:210, 220:250] = (40, 180, 60)
   lf_far = _WidthLF()
   lf_far.x_f = lf_far.last_x = 60.0   # blob cx~234 = x_f+174: beyond the
                                        # 140px right band + 25px gap ->
                                        # right_adjacent, NOT own-lane
   det_w2 = ObstacleDetector(line_follower=lf_far)
   info = det_w2.run(img)
   ok &= _check("object past the own lane + gap lands in right-adjacent, not present",
                not info['present'] and info['right_adjacent_clear'] is not None
                and not info['right_adjacent_clear'], info)

   img = _make_synthetic(obstacle_side=None)
   img[185:210, 95:125] = (40, 180, 60)  # cx~110 = x_f-160: inside the
                                          # 170px opposing-lane band,
                                          # outside the old 85px one
   lf_l = _WidthLF()
   lf_l.x_f = lf_l.last_x = 270.0
   det_w3 = ObstacleDetector(line_follower=lf_l)
   info = det_w3.run(img)
   ok &= _check("opposing-lane obstacle beyond the old half-width now detected, side=left",
                info['present'] and info['side'] == 'left', info)

   # cone-colored obstacle should score a meaningful cone_score. Uses a
   # distinctly RED-orange (HSV hue~6, chroma~97) rather than the
   # yellower orange a real cone might be -- that yellower shade would
   # land inside the yellow-exclusion band and never register at all,
   # which is a real, disclosed ambiguity (see module docstring), not
   # something this test is trying to paper over.
   img = _make_synthetic(obstacle_side='left', bottom_row=210,
                         color=(255, 60, 10))
   info = det.run(img)
   ok &= _check("cone-colored obstacle scores cone_score > 0.3",
                info['cone_score'] > 0.3, info)

   print("ALL PASS" if ok else "SOME FAILED")
   return ok


if __name__ == '__main__':
   import sys
   if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
       ok = _selftest()
       sys.exit(0 if ok else 1)
   elif len(sys.argv) > 1 and sys.argv[1] == 'calibrate':
       rest = sys.argv[2:]
       if not rest or len(rest) % 2 != 0:
           print("usage: obstacle_detector.py calibrate <frame.jpg> "
                 "<distance_m> [<frame.jpg> <distance_m> ...]")
           sys.exit(1)
       _calibrate([(rest[i], float(rest[i + 1]))
                   for i in range(0, len(rest), 2)])
   else:
       print(__doc__)
