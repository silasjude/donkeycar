#!/usr/bin/env python3
"""
line_follower_multiband.py — multi-band CV line follower.

An alternative to donkeycar/parts/line_follower.py's single-slice HSV
detector, aimed at dashed/segmented lines and lighting that isn't stable
enough for one fixed HSV range. Three differences:

  1. Detection runs in CIELAB instead of HSV. A line's color is identified by
     LAB b-channel (yellowness/blueness) and a-channel (greenness/redness)
     thresholds plus an HSV hue guard, all configurable — see LF_LAB_* /
     LF_HUE_* below. LAB tends to separate a taped line from background
     (pavement, vegetation, cones) more robustly across day/shadow/artificial
     lighting than HSV alone, whose saturation channel in particular can swing
     widely between daylight and warm artificial light. The bundled defaults
     were tuned for a yellow dashed line; expect to recalibrate LF_LAB_*/
     LF_HUE_* for a different line color or surface using the offline test
     harness below.
  2. It samples a tall region (bottom LF_ROI_TOP..1.0 of the frame) split into
     horizontal bands, finds the line centroid in each band, and fits a line
     through them, instead of one thin horizontal slice. This tolerates
     dashed/segmented lines (a single-slice detector regularly lands in a gap
     between dashes) and additionally yields line heading, not just lateral
     offset.
  3. While the dashed line is tracked, it also learns the pixel spacing to
     each white lane-boundary line, per image band (so perspective is
     handled). If the dash briefly drops out (a gap, glare) but BOTH white
     boundaries are visible, it steers on their measured midpoint, corrected
     for that learned spacing rather than assumed symmetry — the dash isn't
     necessarily equidistant from both boundaries. If only one boundary is
     visible, it falls back further to a single-side projection off the
     learned spacing; that's less trustworthy (an over-time-averaged guess,
     not a direct measurement), so it's time-limited (LF_WHITE_MAX_SEC).

On losing the line entirely it coasts on last-known steering for a grace
period, then stops — by design it never blind-turns, since a turn applied
blind just traces a circle for the rest of the grace window.

--- HOW TO USE ---------------------------------------------------------------
In myconfig.py, point the cv_control template at this part:
    CV_CONTROLLER_MODULE = "donkeycar.parts.line_follower_multiband"
    CV_CONTROLLER_CLASS  = "LineFollowerMultiBand"
Then drive as usual:  python manage.py drive   (cv_control template)
Switch the web UI to "Local Pilot (d)" mode to engage the follower. Every
tunable below can be overridden from myconfig.py; the part runs on the
built-in defaults even with a bare/minimal cfg.

--- HOW TO TEST OFF THE CAR (no hardware needed) ----------------------------
   python line_follower_multiband.py test <image_or_video_or_folder> [--out overlay_dir]
This runs the exact same pipeline on saved frames and writes overlay images
showing the mask, detected centroids, fitted line, and steering output. Use
this to (re)calibrate LF_LAB_*/LF_HUE_* against photos of your own line/track
before trusting it on hardware.
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
    # --- color detection (CIELAB) ---
    # Detection runs in LAB space, not HSV, because HSV saturation of a taped
    # line tends to swing a lot between daylight and warm artificial light,
    # while LAB b-channel (yellowness) stays comparatively stable. Values
    # below are a starting point tuned for a yellow line on light pavement;
    # recalibrate against your own track photos with the offline test
    # harness (see module docstring) before trusting them on hardware.
    LF_LAB_B_MIN=143,       # min yellowness; raise if background leaks through
    LF_LAB_A_MAX=142,       # max redness; rejects orange/red obstacles
    LF_LAB_L_MIN=40,        # min lightness; rejects near-black noise

    # Hue guard: an extra check to reject false positives (e.g. vegetation)
    # that can pass the LAB yellowness test. OpenCV hue, 0-179. Defaults
    # below target yellow; widen/shift for a different line color.
    LF_HUE_MIN=10,
    LF_HUE_MAX=32,

    # Saturation floor: HSV hue is undefined/noisy for near-gray pixels (a
    # small RGB channel imbalance from compression or sensor noise can swing
    # hue anywhere), so bright achromatic surfaces -- white boundary tape,
    # washed-out pavement -- can pass the hue gate above by pure noise. This
    # rejects them before hue is trusted. Keep it well below the true line
    # color's saturation but comfortably above white/gray's; unlike the hue
    # and LAB values, this doesn't need retuning per lighting condition since
    # it's separating "has real color" from "is essentially gray," not
    # measuring the color itself.
    LF_HSV_S_MIN=35,

    # --- white boundary lines (used when the dash briefly drops out) ---
    # While the dash is tracked, the follower LEARNS the pixel spacing
    # between it and each white boundary line, per image band. See point 3
    # in the module docstring for how the two fallback tiers built on this
    # differ in how much they're trusted.
    LF_WHITE_S_MAX=80,        # white tape: low saturation
    LF_WHITE_V_FLOOR=110,     # min brightness floor for white
    LF_WHITE_MIN_BAND_PIXELS=4,
    LF_LANE_OFFSET_INIT=0.55, # initial guess: white lines sit this far from
                              # the dash, as a fraction of half image width
    LF_LANE_OFFSET_ALPHA=0.08,# learning rate of the spacing estimate (EMA)
    LF_LANE_GATE=0.30,        # search window for a white line around its
                              # expected position (fraction of half width)
    LF_MAX_FIT_RMS=0.05,      # max weighted RMS residual of a white-line-
                              # based fit, as a fraction of image width.
                              # Rejects a scattered, non-collinear "fit" —
                              # concrete patches and glare aren't collinear
                              # the way a real boundary line is.
    LF_WHITE_MAX_SEC=1.5,     # max continuous time steering on the SINGLE-
                              # SIDE white fallback before treating the line
                              # as lost (does not apply to the two-line
                              # lane-center tier — see docstring point 3).
                              # With several white lines on a course, the
                              # single-side estimate can latch onto the
                              # wrong one; this bounds the damage.

    # --- region of interest ---
    LF_ROI_TOP=0.62,        # ignore everything above this fraction of the image.
                            # Tune to your camera's mount height/angle: this
                            # should cut off the horizon and anything above
                            # the track surface (background, plants, walls).
    LF_NUM_BANDS=8,         # horizontal bands the ROI is split into
    LF_MIN_BAND_PIXELS=6,   # a band needs at least this many mask pixels to count
                            # (sized for 160x120; scale up roughly with pixel
                            # area for other resolutions, e.g. ~12+ at 320x240)
    LF_MIN_BANDS=2,         # need centroids in at least this many bands for a fix.
                            # Raising this trades line-loss sensitivity for
                            # heading stability: a 2-point line fit is very
                            # sensitive to single-band noise and can saturate
                            # heading output on a bad read.

    # --- steering control (PD on normalized lateral error) ---
    # error: -1 = line at left edge, 0 = line at target, +1 = line at right edge
    # donkeycar steering: -1 = full left, +1 = full right
    LF_STEER_KP=2.4,        # proportional gain on lateral offset
    LF_STEER_KD=0.35,       # derivative gain (per second)
    LF_HEADING_GAIN=0.9,    # extra steering from the line's slope (lookahead)
    LF_TARGET_X=None,       # where the line should sit in the frame, in pixels.
                            # None = image center. If your camera is mounted
                            # off-center, set this.
    LF_STEERING_TRIM=0.0,   # constant added to every steering command, in
                            # [-1, 1]. Compensates for a chassis/steering
                            # mechanical bias (e.g. the car visibly pulls
                            # right at a commanded-straight steering value)
                            # that isn't fixed by your drivetrain's own
                            # calibration. Some actuators expose a trim of
                            # their own (e.g. VESC_STEERING_OFFSET) that
                            # applies to every drive mode and is the better
                            # fix when available; this is a fallback for
                            # setups that don't have one (plain PWM steering
                            # only calibrates its left/right endpoints, not a
                            # center) or a residual correction on top. Sign
                            # matches steering: negative nudges left.

    # --- throttle ---
    LF_THROTTLE_MAX=0.30,   # on straights
    LF_THROTTLE_MIN=0.16,   # in hard corrections
    LF_THROTTLE_LOST=0.12,  # while coasting on a lost line

    # --- lost-line recovery ---
    # On loss, hold the last known steering (don't commit to a blind turn —
    # that just traces a circle) and ease off the throttle until either the
    # line reappears or the grace period runs out, then stop.
    LF_LOST_STOP_SEC=2.0,   # give up and stop after this many seconds lost

    # --- misc ---
    OVERLAY_IMAGE=True,     # draw diagnostics on the image sent to the web UI
)


def _cfg(cfg, name):
    """Read a value from cfg, falling back to DEFAULTS."""
    return getattr(cfg, name, DEFAULTS[name]) if cfg is not None else DEFAULTS[name]


class LineFollowerMultiBand:
    """
    DonkeyCar part.
      input:  'cam/image_array'  (RGB numpy array)
      output: 'pilot/steering', 'pilot/throttle', 'cv/image_array'

    Signature matches the cv_control template's add_cv_controller(), which
    constructs the class as LineFollowerMultiBand(pid, cfg). The pid argument
    is accepted for compatibility but ignored — control is a self-contained PD
    loop so this file has no simple_pid dependency.
    """

    def __init__(self, pid=None, cfg=None):
        self.b_min = int(_cfg(cfg, 'LF_LAB_B_MIN'))
        self.a_max = int(_cfg(cfg, 'LF_LAB_A_MAX'))
        self.l_min = int(_cfg(cfg, 'LF_LAB_L_MIN'))
        self.hue_min = int(_cfg(cfg, 'LF_HUE_MIN'))
        self.hue_max = int(_cfg(cfg, 'LF_HUE_MAX'))
        self.sat_min = int(_cfg(cfg, 'LF_HSV_S_MIN'))

        self.white_s_max = int(_cfg(cfg, 'LF_WHITE_S_MAX'))
        self.white_v_floor = int(_cfg(cfg, 'LF_WHITE_V_FLOOR'))
        self.white_min_px = int(_cfg(cfg, 'LF_WHITE_MIN_BAND_PIXELS'))
        self.lane_offset_init = float(_cfg(cfg, 'LF_LANE_OFFSET_INIT'))
        self.lane_alpha = float(_cfg(cfg, 'LF_LANE_OFFSET_ALPHA'))
        self.lane_gate = float(_cfg(cfg, 'LF_LANE_GATE'))
        self.max_fit_rms = float(_cfg(cfg, 'LF_MAX_FIT_RMS'))
        self.white_max_sec = float(_cfg(cfg, 'LF_WHITE_MAX_SEC'))

        self.roi_top = float(_cfg(cfg, 'LF_ROI_TOP'))
        self.num_bands = int(_cfg(cfg, 'LF_NUM_BANDS'))
        self.min_band_px = int(_cfg(cfg, 'LF_MIN_BAND_PIXELS'))
        self.min_bands = int(_cfg(cfg, 'LF_MIN_BANDS'))

        self.kp = float(_cfg(cfg, 'LF_STEER_KP'))
        self.kd = float(_cfg(cfg, 'LF_STEER_KD'))
        self.kh = float(_cfg(cfg, 'LF_HEADING_GAIN'))
        self.target_x = _cfg(cfg, 'LF_TARGET_X')
        self.steering_trim = float(_cfg(cfg, 'LF_STEERING_TRIM'))

        self.th_max = float(_cfg(cfg, 'LF_THROTTLE_MAX'))
        self.th_min = float(_cfg(cfg, 'LF_THROTTLE_MIN'))
        self.th_lost = float(_cfg(cfg, 'LF_THROTTLE_LOST'))

        self.lost_stop_sec = float(_cfg(cfg, 'LF_LOST_STOP_SEC'))

        self.overlay = bool(_cfg(cfg, 'OVERLAY_IMAGE'))

        # state
        self.steering = 0.0
        self.throttle = 0.0
        self.prev_error = None
        self.prev_time = None
        self.lost_frames = 0
        self.lost_since = None
        self.status = "init"
        self.mode = "none"          # 'yellow' | 'lane-center' | 'white-L' | 'white-R' | 'none'

        # learned yellow->white spacing per band (pixels); filled lazily
        self.off_left = None
        self.off_right = None
        self.offsets_learned = False  # becomes True once yellow+white have
                                      # been seen together (fallback requires)
        self.learned_l = None       # per-band flags: offset actually learned
        self.learned_r = None
        self.white_side = None      # which white line the single-side
                                    # fallback is holding ('L'/'R'); sticky
                                    # to avoid flip-flopping
        self.white_since = None     # when the single-side fallback took over
        self.consec_yellow = 0

        # morphology kernel, sized on first frame (cleans mask speckle).
        # Must scale with resolution: at 160x120 the distant dashes are only
        # 2-4 px wide and a 5x5 opening erases them entirely.
        self._kernel = None

    # ------------------------------------------------------------------ #
    # detection                                                          #
    # ------------------------------------------------------------------ #
    def detect(self, rgb_img):
        """
        Find the hashed line in the bottom part of the frame.

        Returns (found, error, heading, debug) where
          error   : normalized lateral offset in [-1, 1]; >0 means the line is
                    to the RIGHT of the target x, so the car steers right.
          heading : slope of the line, in normalized-x per ROI-height;
                    >0 means the line leans right as it goes away from the car.
          debug   : dict for the overlay.
        """
        h, w = rgb_img.shape[:2]
        y0 = int(h * self.roi_top)
        roi = rgb_img[y0:, :]

        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hue, sat = hsv[:, :, 0], hsv[:, :, 1]
        # sat gate must come before hue is trusted: hue is near-meaningless
        # for low-saturation (white/gray) pixels, so without it a bright
        # achromatic surface can pass the hue window on pure noise.
        mask = ((B >= self.b_min) & (A <= self.a_max) & (L >= self.l_min)
                & (sat >= self.sat_min)
                & (hue >= self.hue_min) & (hue <= self.hue_max)
                ).astype(np.uint8) * 255

        if self._kernel is None:
            k = 5 if w >= 240 else 3
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        roi_h = roi.shape[0]
        band_h = max(1, roi_h // self.num_bands)
        target = (w / 2.0) if self.target_x is None else float(self.target_x)
        half_w = w / 2.0

        # lazy init of per-band lane-spacing estimates
        if self.off_left is None:
            self.off_left = np.full(self.num_bands, self.lane_offset_init * half_w)
            self.off_right = np.full(self.num_bands, self.lane_offset_init * half_w)
            self.learned_l = np.zeros(self.num_bands, dtype=bool)
            self.learned_r = np.zeros(self.num_bands, dtype=bool)

        # white mask ingredients (computed per band with adaptive brightness,
        # so sun and shadow don't need one global threshold)
        S_ch, V_ch, H_ch = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
        # blue-ish pixels can't be "white" (some tracks have light-blue tape
        # squares that are bright and fairly unsaturated)
        not_blue = ~((H_ch >= 90) & (H_ch <= 135) & (S_ch >= 35))

        pts = []          # yellow: (cx, cy, weight) per band
        yellow_band = {}  # band index -> yellow cx
        white_pts = {}    # band index -> {'L': (cx, cy, n), 'R': (...)}
        gate = self.lane_gate * half_w

        for b in range(self.num_bands):
            ys, ye = b * band_h, min((b + 1) * band_h, roi_h)
            band = mask[ys:ye]
            n = cv2.countNonZero(band)
            if n >= self.min_band_px:
                m = cv2.moments(band, binaryImage=True)
                cx = m['m10'] / m['m00']
                cy = ys + m['m01'] / m['m00']
                pts.append((cx, cy, n))
                yellow_band[b] = cx

            # --- white boundary detection in this band ---
            Vb, Sb = V_ch[ys:ye], S_ch[ys:ye]
            v_thr = max(self.white_v_floor, np.percentile(Vb, 85) + 15)
            wm = (Sb <= self.white_s_max) & (Vb >= v_thr) & not_blue[ys:ye]
            cen = yellow_band.get(b, target)
            cy_band = (ys + ye) / 2.0
            for side, sign in (('L', -1), ('R', 1)):
                exp_x = cen + sign * (self.off_left[b] if side == 'L'
                                      else self.off_right[b])
                x0 = int(max(0, exp_x - gate))
                x1 = int(min(w, exp_x + gate + 1))
                if x1 - x0 < 3:
                    continue
                win = wm[:, x0:x1]
                cnt = int(win.sum())
                if cnt < self.white_min_px:
                    continue
                cols = np.where(win.any(axis=0))[0]
                wcx = x0 + float(np.average(
                    cols, weights=win[:, cols].sum(axis=0)))
                white_pts.setdefault(b, {})[side] = (wcx, cy_band, cnt)

        debug = dict(mask=mask, roi_y0=y0, pts=pts, w=w, h=h,
                     white_pts=white_pts)

        if len(pts) < self.min_bands:
            return self._via_white(white_pts, target, half_w, roi_h, w, debug)

        pts_a = np.array(pts, dtype=np.float64)
        cxs, cys, ws = pts_a[:, 0], pts_a[:, 1], pts_a[:, 2]

        # Weighted least-squares fit x = a*y + b through the band centroids.
        # Weight by pixel count AND by closeness to the car (bottom of ROI):
        # near dashes matter more for lateral error.
        wgt = ws * (0.5 + cys / roi_h)
        a, bfit = np.polyfit(cys, cxs, 1, w=np.sqrt(wgt))

        # One pass of outlier rejection: vegetation/gravel/glare at the frame
        # edge can hijack a band's centroid. Drop bands far from the fitted
        # line and refit with the rest. Gated on "enough points to fit
        # meaningfully" (3, since 2 points have zero residual to judge by),
        # NOT on min_bands -- min_bands is a separate, user-tunable "how many
        # bands make a valid detection" safety threshold, and coupling the two
        # meant an outlier landing in an exactly-min_bands detection (a common
        # case) could never be rejected in the first place.
        if len(pts) > 2:
            resid = np.abs(cxs - (a * cys + bfit))
            keep = resid < max(0.12 * w, 1.5 * np.median(resid) + 1)
            # Floor is 2 (the geometric minimum for a line fit), not
            # min_bands: the raw detection already cleared min_bands above
            # before we got here, so that safety bar has done its job.
            # Re-applying it here would mean a detection that's exactly at
            # min_bands can never have even one point rejected -- which
            # defeats outlier rejection precisely when it matters most (a
            # borderline detection is the likeliest to contain a bad point).
            if keep.sum() >= 2 and keep.sum() < len(pts):
                cxs, cys, ws, wgt = cxs[keep], cys[keep], ws[keep], wgt[keep]
                a, bfit = np.polyfit(cys, cxs, 1, w=np.sqrt(wgt))
                debug['pts'] = [p for p, k in zip(pts, keep) if k]

        # Lateral error comes from the NEAREST detected dashes (bottom-most
        # centroids), not from extrapolating the fit to the frame bottom —
        # extrapolation blows up when the line runs nearly horizontal in the
        # frame (sharp curves).
        order = np.argsort(cys)[::-1]          # nearest (largest y) first
        near = order[:3]
        x_near = float(np.average(cxs[near], weights=ws[near]))
        error = (x_near - target) / (w / 2.0)
        error = float(np.clip(error, -1.0, 1.0))

        # heading: slope, normalized. a is px-x per px-y; y grows downward, so
        # a > 0 means the line moves right toward the car => it leans LEFT
        # ahead of the car => steer left. Negate to get "lean of the road ahead".
        heading = float(np.clip(-a * roi_h / (w / 2.0), -1.0, 1.0))

        debug['fit'] = (a, bfit)
        debug['x_near'] = x_near
        debug['target'] = target

        # LEARN the yellow->white spacing per band for the fallback tiers.
        # Use the fitted yellow line to get yellow's x at each white
        # detection's row (the dashes don't land in every band, so same-band
        # co-occurrence would almost never happen otherwise).
        for b, sides in white_pts.items():
            for side, (wcx, wcy, _) in sides.items():
                ycx_fit = a * wcy + bfit
                if side == 'L':
                    d = ycx_fit - wcx
                    if d > 2:
                        self.off_left[b] += self.lane_alpha * (d - self.off_left[b])
                        self.learned_l[b] = True
                        self.offsets_learned = True
                else:
                    d = wcx - ycx_fit
                    if d > 2:
                        self.off_right[b] += self.lane_alpha * (d - self.off_right[b])
                        self.learned_r[b] = True
                        self.offsets_learned = True

        self.mode = "yellow"
        return True, error, heading, debug

    def _via_white(self, white_pts, target, half_w, roi_h, w, debug):
        '''
        The dash isn't usable this frame. Try real two-line lane centering
        first (both boundary lines directly measured -> trustworthy enough
        to drive on, not just bridge a gap), and only fall back to the
        single-side learned-offset estimate if just one line is visible.
        '''
        result = self._lane_center(white_pts, target, half_w, roi_h, w, debug)
        if result[0]:
            return result
        return self._white_fallback(white_pts, target, half_w, roi_h, w, debug)

    def _lane_center(self, white_pts, target, half_w, roi_h, w, debug):
        '''
        Both white boundary lines detected in the same band. The lane center
        is NOT their raw geometric midpoint -- the dash isn't necessarily
        equidistant from both boundaries. Instead, project each white
        detection back to the dash's position using the ALREADY-LEARNED
        per-band spacing (self.off_left/off_right -- the same spacing the
        single-side fallback below uses), then average those two independent
        estimates. Needs no per-frame commitment to one side, so it's
        trusted to steer on its own rather than just bridge gaps -- it just
        also needs the spacing to have actually been learned (same
        requirement as the single-side fallback).
        '''
        if not self.offsets_learned:
            return False, 0.0, 0.0, debug

        idx = np.arange(self.num_bands)
        def eff(raw, learned):
            if learned.any():
                return np.interp(idx, idx[learned], raw[learned])
            return raw
        eff_l = eff(self.off_left, self.learned_l)
        eff_r = eff(self.off_right, self.learned_r)

        mids = []
        for b, sides in white_pts.items():
            if 'L' not in sides or 'R' not in sides:
                continue
            lx, ly, ln = sides['L']
            rx, ry, rn = sides['R']
            if rx <= lx:
                continue  # sanity: the right detection must sit right of left
            est_from_l = lx + eff_l[b]
            est_from_r = rx - eff_r[b]
            cx = (est_from_l * ln + est_from_r * rn) / (ln + rn)
            mids.append((cx, (ly + ry) / 2.0, min(ln, rn)))

        if len(mids) < self.min_bands:
            return False, 0.0, 0.0, debug

        mids_a = np.array(mids, dtype=np.float64)
        mxs, mys, mws = mids_a[:, 0], mids_a[:, 1], mids_a[:, 2]

        a, bfit = np.polyfit(mys, mxs, 1, w=np.sqrt(mws))
        # the lane center must be a real line too: reject a scattered "fit"
        resid = mxs - (a * mys + bfit)
        rms = float(np.sqrt(np.average(resid ** 2, weights=mws)))
        if rms > self.max_fit_rms * w:
            return False, 0.0, 0.0, debug

        order = np.argsort(mys)[::-1]          # nearest (largest y) first
        near = order[:3]
        x_near = float(np.average(mxs[near], weights=mws[near]))
        error = float(np.clip((x_near - target) / half_w, -1.0, 1.0))
        heading = float(np.clip(-a * roi_h / half_w, -1.0, 1.0))

        self.white_side = None   # not holding a single side
        debug['fit'] = (a, bfit)
        debug['x_near'] = x_near
        debug['target'] = target
        debug['lane_mids'] = mids
        self.mode = "lane-center"
        return True, error, heading, debug

    def _white_fallback(self, white_pts, target, half_w, roi_h, w, debug):
        '''
        Only ONE white boundary line visible: hold the learned distance from
        it. Using a single side (sticky across frames) is crucial: blending
        both sides flip-flops the estimate whenever a band's gate catches
        the wrong line, which whipsaws the steering. Last resort -- less
        trustworthy than lane-center above, so run() time-limits how long
        this can keep steering (LF_WHITE_MAX_SEC).
        '''
        # never engage on the initial guess: the spacing must have actually
        # been learned from seeing yellow + white together
        if not self.offsets_learned:
            return False, 0.0, 0.0, debug

        idx = np.arange(self.num_bands)
        def eff(raw, learned):
            if learned.any():
                return np.interp(idx, idx[learned], raw[learned])
            return raw
        eff_l = eff(self.off_left, self.learned_l)
        eff_r = eff(self.off_right, self.learned_r)

        by_side = {'L': [], 'R': []}
        for b, sides in white_pts.items():
            for s, (wcx, wcy, n) in sides.items():
                by_side[s].append((b, wcx, wcy, n))
        weight = {s: sum(p[3] for p in v) for s, v in by_side.items()}

        # sticky side choice: keep holding the same line while it's visible
        side = self.white_side
        if side is None or weight[side] < self.white_min_px:
            side = 'L' if weight['L'] >= weight['R'] else 'R'
        # a real boundary line spans multiple bands; one or two isolated
        # bright patches on pale concrete must never fabricate a lane
        if weight[side] < self.white_min_px or len(by_side[side]) < 2:
            return False, 0.0, 0.0, debug

        est = []
        for b, wcx, wcy, n in by_side[side]:
            ecx = wcx + eff_l[b] if side == 'L' else wcx - eff_r[b]
            est.append((ecx, wcy, n))
        est_a = np.array(est, dtype=np.float64)
        exs, eys, ews = est_a[:, 0], est_a[:, 1], est_a[:, 2]

        heading = 0.0
        if len(est) >= 2:
            a, bfit = np.polyfit(eys, exs, 1, w=np.sqrt(ews))
            resid = exs - (a * eys + bfit)
            rms = float(np.sqrt(np.average(resid ** 2, weights=ews)))
            if rms > self.max_fit_rms * w:
                return False, 0.0, 0.0, debug
            heading = float(np.clip(-a * roi_h / half_w, -1.0, 1.0))
            debug['fit'] = (a, bfit)

        order = np.argsort(eys)[::-1]
        near = order[:3]
        x_near = float(np.average(exs[near], weights=ews[near]))
        error = float(np.clip((x_near - target) / half_w, -1.0, 1.0))

        self.white_side = side
        debug['x_near'] = x_near
        debug['target'] = target
        debug['est'] = est
        self.mode = f"white-{side}"
        return True, error, heading, debug

    # ------------------------------------------------------------------ #
    # control                                                            #
    # ------------------------------------------------------------------ #
    def run(self, cam_img):
        if cam_img is None:
            return 0.0, 0.0, None

        now = time.time()
        found, error, heading, debug = self.detect(cam_img)

        # The single-side white fallback is a bridge, not a driving mode: it
        # may steer for at most LF_WHITE_MAX_SEC continuously. With more than
        # one white line in view it can latch onto the wrong one; time-
        # limiting it prevents endless circling. Real lane-centering (both
        # lines) is NOT time-limited here -- it's a direct measurement each
        # frame, not a projection that goes stale. The budget only resets
        # after the dash has been tracked for 3 CONSECUTIVE frames -- a
        # single-frame debris blip that reads as "yellow" must not re-arm it.
        if found and self.mode == "yellow":
            self.consec_yellow += 1
            if self.consec_yellow >= 3:
                self.white_since = None
        else:
            self.consec_yellow = 0
        if found and self.mode.startswith("white"):
            if self.white_since is None:
                self.white_since = now
            if now - self.white_since > self.white_max_sec:
                found = False   # treat as lost -> coast/stop

        if found:
            self.lost_frames = 0
            self.lost_since = None

            # PD + heading feed-forward. Skip the derivative on a mode
            # change (yellow <-> lane-center <-> white-L/R): the error
            # source shifted, so the difference isn't a real rate and would
            # kick the steering.
            d_err = 0.0
            if self.prev_error is not None and self.prev_time is not None \
                    and self.mode == getattr(self, '_prev_mode', None):
                dt = max(now - self.prev_time, 1e-3)
                d_err = (error - self.prev_error) / dt
            self.prev_error, self.prev_time = error, now
            self._prev_mode = self.mode

            steer = self.kp * error + self.kd * d_err + self.kh * heading + self.steering_trim
            self.steering = float(np.clip(steer, -1.0, 1.0))

            # slow down proportionally to how hard we're steering
            self.throttle = self.th_max - (self.th_max - self.th_min) * abs(self.steering)
            self.status = f"tracking ({self.mode})"

        else:
            self.lost_frames += 1
            self.prev_error = None
            if self.lost_since is None:
                self.lost_since = now

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

        out_img = self._overlay(cam_img, debug, found, error, heading) \
            if self.overlay else cam_img
        return self.steering, self.throttle, out_img

    # keep parity with parts that call shutdown
    def shutdown(self):
        pass

    # ------------------------------------------------------------------ #
    # diagnostics overlay                                                #
    # ------------------------------------------------------------------ #
    def _overlay(self, cam_img, debug, found, error, heading):
        img = cam_img.copy()
        h, w = img.shape[:2]
        y0 = debug['roi_y0']

        # tint detected mask green
        mask = debug['mask']
        region = img[y0:, :]
        region[mask > 0] = (0, 255, 0)

        # ROI top line
        cv2.line(img, (0, y0), (w, y0), (255, 255, 0), 1)

        # band centroids (dash = red dots)
        for cx, cy, _ in debug['pts']:
            cv2.circle(img, (int(cx), int(y0 + cy)), 3, (255, 0, 0), -1)

        # white boundary detections = cyan dots
        for b, sides in debug.get('white_pts', {}).items():
            for side, (wcx, wcy, _) in sides.items():
                cv2.circle(img, (int(wcx), int(y0 + wcy)), 3, (0, 255, 255), -1)

        # virtual center estimates (single-side white fallback) = orange dots
        for ecx, ecy, _ in debug.get('est', []):
            cv2.circle(img, (int(ecx), int(y0 + ecy)), 3, (255, 128, 0), -1)

        # measured lane-center midpoints (both white lines) = yellow dots
        for mcx, mcy, _ in debug.get('lane_mids', []):
            cv2.circle(img, (int(mcx), int(y0 + mcy)), 3, (255, 255, 0), -1)

        # fitted line + target
        if found and 'fit' in debug:
            a, b = debug['fit']
            roi_h = h - y0
            p1 = (int(b), y0)
            p2 = (int(a * roi_h + b), h - 1)
            cv2.line(img, p1, p2, (255, 0, 255), 2)
            cv2.line(img, (int(debug['target']), h - 12),
                     (int(debug['target']), h - 1), (255, 255, 255), 2)

        for i, s in enumerate([
            f"st:{self.steering:+.2f} th:{self.throttle:.2f}",
            f"err:{error:+.2f} hdg:{heading:+.2f}",
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

    lf = LineFollowerMultiBand()
    for f in files:
        if f.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(f)
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st, th, ov = lf.run(rgb)
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
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            st, th, ov = lf.run(rgb)
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
