#!/usr/bin/env python3
"""
line_following.py — DonkeyCar Track 2, Mission 1: follow the yellow hashed line.

Written from scratch. Replaces donkeycar/parts/line_follower.py, which fails on
this track for two reasons:

  1. Its HSV threshold (H 0-50, S>50, V>50) matches the warm night lighting:
     the concrete, orange cones, and wall reflections all read as "yellow".
     Worse, HSV saturation of the tape swings from ~160 (night, warm lights)
     to ~87 (daylight) — no fixed HSV range covers both. This detector works
     in CIELAB instead: measured across day/sun/shadow/night photos of this
     track, the tape's b-channel (yellowness) stays at ~145-149 while
     concrete <=139, white tape ~135, blue tape ~98; the a-channel rejects
     the orange cones.
  2. It samples ONE thin horizontal slice of the image. The center line is
     DASHED, so the slice regularly lands in a gap between dashes, detection
     drops out, and the car drifts off with stale steering.

This detector instead scans a tall region (the bottom ~55% of the frame) split
into horizontal bands, finds the line centroid in each band, and fits a line
through them. That gives both lateral offset AND line heading, works across
dash gaps, and adds a search/recovery behavior when the line is lost.

--- HOW TO RUN ON THE CAR (drop-in, no framework changes) -------------------
1. Copy this file into your car directory (e.g. ~/mycar/line_following.py).
2. In ~/mycar/myconfig.py set:
       CV_CONTROLLER_MODULE = "line_following"
       CV_CONTROLLER_CLASS  = "LineFollower"
3. Drive as usual:  python manage.py drive   (cv_control template)
   Switch the web UI to "Local Pilot (d)" mode to engage the follower.

--- HOW TO TEST OFF THE CAR (no hardware needed) ----------------------------
   python line_following.py test <image_or_video_or_folder> [--out overlay_dir]
This runs the exact same pipeline on saved frames and writes overlay images
showing the mask, detected centroids, fitted line, and steering output.
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
    # Detection runs in LAB space, not HSV. Measured from track photos in
    # sunlight, shadow, AND at night, the tape's b-channel (yellowness) is
    # stable at ~145-149 while concrete is <=139, white tape ~135, blue tape
    # ~98. HSV saturation is NOT stable (daytime tape drops to S~87). The
    # a-channel guard rejects the orange cones (a~164 vs tape ~125).
    LF_LAB_B_MIN=143,       # min yellowness; raise if concrete leaks through
    LF_LAB_A_MAX=142,       # max redness; rejects cones/flowers
    LF_LAB_L_MIN=40,        # min lightness; rejects near-black noise

    # Hue guard: kills VEGETATION, which passes the LAB yellowness test.
    # Measured from drive video: sunlit grass/plants are H 34-46, the tape is
    # H 18-28 in all lighting (day/shadow/night). OpenCV hue, 0-179.
    LF_HUE_MIN=10,
    LF_HUE_MAX=32,

    # --- white boundary lines (fallback when the yellow line is lost) ---
    # While the yellow line is tracked, the follower LEARNS the pixel spacing
    # between it and each white boundary line, per image band (so perspective
    # is handled). When the yellow drops out (dash gap, hairpin), it steers
    # on a virtual center computed from whichever white line it still sees.
    LF_WHITE_S_MAX=80,        # white tape: low saturation
    LF_WHITE_V_FLOOR=110,     # min brightness floor for white
    LF_WHITE_MIN_BAND_PIXELS=4,
    LF_LANE_OFFSET_INIT=0.55, # initial guess: white lines sit this far from
                              # the yellow, as a fraction of half image width
    LF_LANE_OFFSET_ALPHA=0.08,# learning rate of the spacing estimate (EMA)
    LF_LANE_GATE=0.30,        # search window for a white line around its
                              # expected position (fraction of half width)
    LF_WHITE_MAX_SEC=1.5,     # max continuous time steering on the white
                              # fallback before treating the line as lost.
                              # The fallback bridges dash gaps and corners —
                              # it must NOT drive forever, because with
                              # several white lines on the course it can
                              # latch onto the wrong one and circle.

    # --- region of interest ---
    LF_ROI_TOP=0.62,        # ignore everything above this fraction of the image.
                            # The camera is mounted high/level: the horizon sits
                            # ~55% down the frame, planters/bushes right above
                            # it. Only the bottom ~38% is reliably track.
    LF_NUM_BANDS=8,         # horizontal bands the ROI is split into
    LF_MIN_BAND_PIXELS=6,   # a band needs at least this many mask pixels to count
                            # (sized for 160x120; use ~12+ at 320x240)
    LF_MIN_BANDS=2,         # need centroids in at least this many bands for a fix
    LF_MIN_MASK_FRAC=0.0022,# total yellow pixels must be at least this fraction
                            # of the ROI area (rejects scattered debris: dead
                            # leaves and petals are tape-colored, but sparse)
    LF_MAX_FIT_RMS=0.05,    # max weighted RMS residual of the line fit, as a
                            # fraction of image width. Real dashes are
                            # collinear (residual ~1-2 px); scattered junk
                            # is not. Detections above this are rejected.
    LF_MAX_JUMP=0.20,       # max per-frame jump of the line position, as a
                            # fraction of image width. At 20 Hz the real line
                            # moves smoothly; a detection that teleports is
                            # junk or a different track segment's line.
    LF_JUMP_ESCAPE=8,       # after this many consecutive jump-rejections,
                            # accept the new position (the line really did
                            # move, e.g. reacquired after a hairpin)

    # --- steering control (PD on normalized lateral error) ---
    # error: -1 = line at left edge, 0 = line at target, +1 = line at right edge
    # donkeycar steering: -1 = full left, +1 = full right
    LF_STEER_KP=2.4,        # proportional gain on lateral offset
    LF_STEER_KD=0.35,       # derivative gain (per second)
    LF_HEADING_GAIN=0.9,    # extra steering from the line's slope (lookahead)
    LF_ERR_DEADBAND=0.08,   # |error| below this = "on the line": go straight
                            # instead of constantly micro-correcting
    LF_STEER_SMOOTH=0.55,   # 0..1, fraction of new steering applied per frame
                            # (lower = smoother/lazier, 1.0 = no smoothing)

    # --- startup acquisition ---
    LF_ACQUIRE_FRAMES=10,   # consecutive detections required before the car
                            # starts moving (at 20 Hz, 10 = half a second).
                            # The car sits still until it has a stable lock,
                            # and returns to this state after a lost-line stop.
    LF_TARGET_X=None,       # where the line should sit in the frame, in pixels.
                            # None = image center. If your camera is mounted
                            # off-center, set this.

    # --- throttle ---
    LF_THROTTLE_MAX=0.30,   # on straights
    LF_THROTTLE_MIN=0.16,   # in hard corrections
    LF_THROTTLE_LOST=0.12,  # while searching for a lost line

    # --- lost-line recovery ---
    LF_LOST_GRACE=5,        # frames to coast on last steering before searching
    LF_SEARCH_STEER=0.85,   # steering magnitude while searching (toward the
                            # side the line was last seen)
    LF_LOST_STOP_SEC=2.0,   # give up and stop after this many seconds lost

    # --- misc ---
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
        self.b_min = int(_cfg(cfg, 'LF_LAB_B_MIN'))
        self.a_max = int(_cfg(cfg, 'LF_LAB_A_MAX'))
        self.l_min = int(_cfg(cfg, 'LF_LAB_L_MIN'))
        self.hue_min = int(_cfg(cfg, 'LF_HUE_MIN'))
        self.hue_max = int(_cfg(cfg, 'LF_HUE_MAX'))

        self.white_s_max = int(_cfg(cfg, 'LF_WHITE_S_MAX'))
        self.white_v_floor = int(_cfg(cfg, 'LF_WHITE_V_FLOOR'))
        self.white_min_px = int(_cfg(cfg, 'LF_WHITE_MIN_BAND_PIXELS'))
        self.lane_offset_init = float(_cfg(cfg, 'LF_LANE_OFFSET_INIT'))
        self.lane_alpha = float(_cfg(cfg, 'LF_LANE_OFFSET_ALPHA'))
        self.lane_gate = float(_cfg(cfg, 'LF_LANE_GATE'))
        self.white_max_sec = float(_cfg(cfg, 'LF_WHITE_MAX_SEC'))

        self.roi_top = float(_cfg(cfg, 'LF_ROI_TOP'))
        self.num_bands = int(_cfg(cfg, 'LF_NUM_BANDS'))
        self.min_band_px = int(_cfg(cfg, 'LF_MIN_BAND_PIXELS'))
        self.min_bands = int(_cfg(cfg, 'LF_MIN_BANDS'))
        self.min_mask_frac = float(_cfg(cfg, 'LF_MIN_MASK_FRAC'))
        self.max_fit_rms = float(_cfg(cfg, 'LF_MAX_FIT_RMS'))
        self.max_jump = float(_cfg(cfg, 'LF_MAX_JUMP'))
        self.jump_escape = int(_cfg(cfg, 'LF_JUMP_ESCAPE'))
        self.jump_rejects = 0

        self.kp = float(_cfg(cfg, 'LF_STEER_KP'))
        self.kd = float(_cfg(cfg, 'LF_STEER_KD'))
        self.kh = float(_cfg(cfg, 'LF_HEADING_GAIN'))
        self.deadband = float(_cfg(cfg, 'LF_ERR_DEADBAND'))
        self.smooth = float(_cfg(cfg, 'LF_STEER_SMOOTH'))
        self.acquire_frames = int(_cfg(cfg, 'LF_ACQUIRE_FRAMES'))
        self.target_x = _cfg(cfg, 'LF_TARGET_X')

        self.th_max = float(_cfg(cfg, 'LF_THROTTLE_MAX'))
        self.th_min = float(_cfg(cfg, 'LF_THROTTLE_MIN'))
        self.th_lost = float(_cfg(cfg, 'LF_THROTTLE_LOST'))

        self.lost_grace = int(_cfg(cfg, 'LF_LOST_GRACE'))
        self.search_steer = float(_cfg(cfg, 'LF_SEARCH_STEER'))
        self.lost_stop_sec = float(_cfg(cfg, 'LF_LOST_STOP_SEC'))

        self.overlay = bool(_cfg(cfg, 'OVERLAY_IMAGE'))

        # state
        self.steering = 0.0
        self.throttle = 0.0
        self.reason = ""            # why the last detection was rejected
        self.prev_error = None
        self.prev_time = None
        self.lost_frames = 0
        self.lost_since = None
        self.last_steer_sign = 0.0  # direction we were steering at last lock;
                                    # drives the search direction (handles turns)
        self.acquired = 0           # consecutive detections while acquiring
        self.moving = False         # False until acquisition completes
        self.status = "init"
        self.mode = "none"          # 'yellow' | 'white' | 'none'
        self.center_px = None       # last known line x at frame bottom
        # learned yellow->white spacing per band (pixels); filled lazily
        self.off_left = None
        self.off_right = None
        self.offsets_learned = False  # becomes True once yellow+white have
                                      # been seen together (fallback requires)
        self.white_since = None     # when the white fallback took over
        self.white_side = None      # which white line the fallback is holding
                                    # ('L'/'R'); sticky to avoid flip-flopping
        self.learned_l = None       # per-band flags: offset actually learned
        self.learned_r = None

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
        hue = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)[:, :, 0]
        mask = ((B >= self.b_min) & (A <= self.a_max) & (L >= self.l_min)
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
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        S_ch, V_ch = hsv_roi[:, :, 1], hsv_roi[:, :, 2]
        H_ch = hsv_roi[:, :, 0]
        # blue-ish pixels can't be "white" (the track has light-blue tape
        # squares that are bright and fairly unsaturated)
        not_blue = ~((H_ch >= 90) & (H_ch <= 135) & (S_ch >= 35))

        pts = []          # yellow: (cx, cy, weight) per band
        yellow_band = {}  # band index -> yellow cx
        white_pts = {}    # band index -> {'L': (cx, cy, n), 'R': (...)}
        gate = self.lane_gate * half_w
        exp_center = self.center_px if self.center_px is not None else target

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
                yellow_band[b] = cx

            # --- white boundary detection in this band ---
            Vb, Sb = V_ch[ys:ye], S_ch[ys:ye]
            v_thr = max(self.white_v_floor, np.percentile(Vb, 85) + 15)
            wm = (Sb <= self.white_s_max) & (Vb >= v_thr) & not_blue[ys:ye]
            cen = yellow_band.get(b, exp_center)
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

        # structural gate 1: enough total evidence (scattered tape-colored
        # debris — dead leaves, petals — gives isolated specks, not this much)
        total_yellow = sum(p[2] for p in pts)
        if len(pts) < self.min_bands:
            self.reason = f"bands {len(pts)}<{self.min_bands}"
            return self._white_fallback(white_pts, target, half_w, roi_h,
                                        w, debug)
        if total_yellow < self.min_mask_frac * w * roi_h:
            self.reason = (f"sparse {int(total_yellow)}px"
                           f"<{int(self.min_mask_frac * w * roi_h)}")
            return self._white_fallback(white_pts, target, half_w, roi_h,
                                        w, debug)

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

        # structural gate 2: the surviving centroids must actually be
        # COLLINEAR. Real dashes fit a line within a pixel or two; random
        # debris does not — reject rather than steer on garbage.
        resid = cxs - (a * cys + bfit)
        rms = float(np.sqrt(np.average(resid ** 2, weights=wgt)))
        if rms > self.max_fit_rms * w:
            self.reason = f"not collinear rms={rms:.1f}"
            return self._white_fallback(white_pts, target, half_w, roi_h,
                                        w, debug)

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
        # Scale trust by how many bands support the fit: a slope from just 2
        # points is unreliable and shouldn't saturate the steering.
        trust = min(1.0, (len(cxs) - 1) / 3.0)
        heading = float(np.clip(-a * roi_h / (w / 2.0), -1.0, 1.0)) * trust

        debug['fit'] = (a, bfit)
        debug['x_near'] = x_near
        debug['target'] = target

        # structural gate 3: temporal consistency
        if not self._accept_center(x_near, w):
            self.reason = (f"jump {int(abs(x_near - self.center_px))}px"
                           f" ({self.jump_rejects}/{self.jump_escape})")
            return self._white_fallback(white_pts, target, half_w, roi_h,
                                        w, debug)
        self.reason = ""

        # remember where the line is, and LEARN the yellow->white spacing per
        # band for the fallback. Use the fitted yellow line to get yellow's x
        # at each white detection's row (the dashes don't land in every band,
        # so same-band co-occurrence would almost never happen).
        self.center_px = float(np.clip(x_near, 0.08 * w, 0.92 * w))
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
        self.white_side = None   # tracking yellow again: unpin the fallback side

        self.mode = "yellow"
        return True, error, heading, debug

    def _accept_center(self, x_near, w):
        '''
        Temporal consistency: at 20 Hz the real line moves a few pixels per
        frame. A detection that jumps across the frame is debris or another
        track segment's line — reject it. After LF_JUMP_ESCAPE consecutive
        rejections, accept (the line genuinely moved, e.g. post-hairpin).
        '''
        if not self.moving or self.center_px is None:
            self.jump_rejects = 0
            return True
        if abs(x_near - self.center_px) <= self.max_jump * w \
                or self.jump_rejects >= self.jump_escape:
            self.jump_rejects = 0
            return True
        self.jump_rejects += 1
        return False

    def _white_fallback(self, white_pts, target, half_w, roi_h, w, debug):
        '''
        No yellow line visible: hold the learned distance from ONE white
        boundary line. Using a single side (sticky across frames) is crucial:
        blending both sides flip-flops the estimate whenever a band's gate
        catches the wrong line, which whipsaws the steering.
        '''
        # never engage on the initial guess: the spacing must have actually
        # been learned from seeing yellow + white together
        if not self.offsets_learned:
            self.mode = "none"
            return False, 0.0, 0.0, debug

        # spacing for every band: learned where available, interpolated
        # across bands elsewhere (spacing varies smoothly with perspective)
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
            self.mode = "none"
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
            # the boundary must be a line too: reject a scattered "fit"
            resid = exs - (a * eys + bfit)
            rms = float(np.sqrt(np.average(resid ** 2, weights=ews)))
            if rms > self.max_fit_rms * w:
                self.mode = "none"
                return False, 0.0, 0.0, debug
            trust = min(1.0, (len(est) - 1) / 3.0)
            heading = float(np.clip(-a * roi_h / half_w, -1.0, 1.0)) * trust
            debug['fit'] = (a, bfit)

        order = np.argsort(eys)[::-1]
        near = order[:3]
        x_near = float(np.average(exs[near], weights=ews[near]))
        error = float(np.clip((x_near - target) / half_w, -1.0, 1.0))

        # temporal consistency applies to the fallback too
        if not self._accept_center(x_near, w):
            self.mode = "none"
            return False, 0.0, 0.0, debug

        self.center_px = float(np.clip(x_near, 0.08 * w, 0.92 * w))
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

        # The white fallback is a bridge, not a driving mode: it may steer
        # for at most LF_WHITE_MAX_SEC continuously. With several white lines
        # on the course it can latch onto the wrong one; time-limiting it
        # prevents endless circling. The budget only resets after the yellow
        # line has been tracked for 3 CONSECUTIVE frames — a single-frame
        # debris blip that reads as "yellow" must not re-arm the fallback.
        if found and self.mode == "yellow":
            self.consec_yellow = getattr(self, 'consec_yellow', 0) + 1
            if self.consec_yellow >= 3:
                self.white_since = None
        else:
            self.consec_yellow = 0
        if found and self.mode.startswith("white"):
            if self.white_since is None:
                self.white_since = now
            if now - self.white_since > self.white_max_sec:
                found = False   # treat as lost -> coast/search/stop

        if found:
            self.lost_frames = 0
            self.lost_since = None

            # --- startup acquisition: sit still until the lock is stable ---
            if not self.moving:
                self.acquired += 1
                if self.acquired < self.acquire_frames:
                    self.steering, self.throttle = 0.0, 0.0
                    self.status = f"acquiring {self.acquired}/{self.acquire_frames}"
                    out_img = self._overlay(cam_img, debug, found, error, heading) \
                        if self.overlay else cam_img
                    return self.steering, self.throttle, out_img
                self.moving = True

            # --- deadband: when basically on the line, just go straight ---
            if abs(error) < self.deadband:
                error = 0.0

            # PD + heading feed-forward. Skip the derivative on a mode change
            # (yellow <-> white): the error source shifted, so the difference
            # is not a real rate and would kick the steering.
            d_err = 0.0
            if self.prev_error is not None and self.prev_time is not None \
                    and self.mode == getattr(self, '_prev_mode', None):
                dt = max(now - self.prev_time, 1e-3)
                d_err = (error - self.prev_error) / dt
            self.prev_error, self.prev_time = error, now
            self._prev_mode = self.mode

            steer = self.kp * error + self.kd * d_err + self.kh * heading
            steer = float(np.clip(steer, -1.0, 1.0))
            # low-pass filter: smooth, non-jerky steering
            self.steering = float(np.clip(
                self.steering + self.smooth * (steer - self.steering), -1.0, 1.0))
            if self.steering != 0:
                self.last_steer_sign = 1.0 if self.steering > 0 else -1.0

            # slow down for how hard we're steering AND for the curve we can
            # see coming (heading), so corners are entered slower
            effort = min(1.0, abs(self.steering) + 0.6 * abs(heading))
            if self.mode.startswith("white"):
                # slightly cautious while steering on the fallback estimate
                effort = min(1.0, effort + 0.25)
            self.throttle = self.th_max - (self.th_max - self.th_min) * effort
            self.status = f"tracking ({self.mode})"

        else:
            self.lost_frames += 1
            self.prev_error = None
            # a single missed frame (dash gap) shouldn't zero the acquisition
            # counter, just set it back a little
            self.acquired = max(0, self.acquired - 2)
            if self.lost_since is None:
                self.lost_since = now

            if not self.moving:
                # never had (or lost) the lock while stationary: stay put
                self.steering, self.throttle = 0.0, 0.0
                self.status = "waiting for line"
            elif now - self.lost_since > self.lost_stop_sec:
                # lost for too long — stop and go back to acquisition, so the
                # car resumes by itself once it sees the line again
                self.steering, self.throttle = 0.0, 0.0
                self.moving = False
                self.status = "stopped (line lost) - waiting"
            elif self.lost_frames <= self.lost_grace:
                # brief dropout (dash gap): hold course, ease off throttle
                self.throttle = max(self.th_lost, self.throttle * 0.9)
                self.status = "coasting"
            else:
                # keep turning the way we were already steering: in a corner
                # the line exits the frame on the inside, so continuing the
                # turn is what re-finds it (searching by the line's last
                # lateral position fails in hairpins)
                side = self.last_steer_sign if self.last_steer_sign != 0 else 1.0
                self.steering = float(self.search_steer * side)
                self.throttle = self.th_lost
                self.status = "searching"

        # log state transitions to the console so the terminal running
        # manage_line.py shows what the controller is doing
        if self.status != getattr(self, '_last_logged_status', None):
            extra = f" | {self.reason}" if self.reason else ""
            logger.info(f"LineFollower: {self.status} "
                        f"(st={self.steering:+.2f} th={self.throttle:.2f})"
                        f"{extra}")
            self._last_logged_status = self.status

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

        # band centroids (yellow line = red dots)
        for cx, cy, _ in debug['pts']:
            cv2.circle(img, (int(cx), int(y0 + cy)), 3, (255, 0, 0), -1)

        # white boundary detections = cyan dots
        for b, sides in debug.get('white_pts', {}).items():
            for side, (wcx, wcy, _) in sides.items():
                cv2.circle(img, (int(wcx), int(y0 + wcy)), 3, (0, 255, 255), -1)

        # virtual center estimates (white fallback) = orange dots
        for ecx, ecy, _ in debug.get('est', []):
            cv2.circle(img, (int(ecx), int(y0 + ecy)), 3, (255, 128, 0), -1)

        # fitted line + target
        if found and 'fit' in debug:
            a, b = debug['fit']
            roi_h = h - y0
            p1 = (int(b), y0)
            p2 = (int(a * roi_h + b), h - 1)
            cv2.line(img, p1, p2, (255, 0, 255), 2)
            cv2.line(img, (int(debug['target']), h - 12),
                     (int(debug['target']), h - 1), (255, 255, 255), 2)

        # State indicator: a bold colored bar across the top, readable even
        # when the 160x120 stream is scaled up in the web UI.
        #   green = driving, yellow = acquiring lock, red = stopped/waiting
        if self.status.startswith(("tracking", "coasting", "searching")):
            bar = (0, 200, 0)
        elif self.status.startswith("acquiring"):
            bar = (255, 200, 0)
        else:
            bar = (255, 0, 0)
        cv2.rectangle(img, (0, 0), (w - 1, max(4, h // 30)), bar, -1)

        scale = max(0.38, w / 420.0)
        lines = [
            f"st:{self.steering:+.2f} th:{self.throttle:.2f}",
            f"err:{error:+.2f} hdg:{heading:+.2f}",
            self.status,
        ]
        if self.reason:
            lines.append(self.reason)
        for i, s in enumerate(lines):
            y = int(h * 0.12 + i * (16 * scale + 7))
            cv2.putText(img, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), 1, cv2.LINE_AA)
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

    lf = LineFollower()
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
