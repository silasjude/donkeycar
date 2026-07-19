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
dash gaps, and coasts on last-known steering (then stops) when the line is lost.

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

    # --- region of interest ---
    LF_ROI_TOP=0.62,        # ignore everything above this fraction of the image.
                            # The camera is mounted high/level: the horizon sits
                            # ~55% down the frame, planters/bushes right above
                            # it. Only the bottom ~38% is reliably track.
    LF_NUM_BANDS=8,         # horizontal bands the ROI is split into
    LF_MIN_BAND_PIXELS=6,   # a band needs at least this many mask pixels to count
                            # (sized for 160x120; use ~12+ at 320x240)
    LF_MIN_BANDS=2,         # need centroids in at least this many bands for a fix

    # --- steering control (PD on normalized lateral error) ---
    # error: -1 = line at left edge, 0 = line at target, +1 = line at right edge
    # donkeycar steering: -1 = full left, +1 = full right
    LF_STEER_KP=2.4,        # proportional gain on lateral offset
    LF_STEER_KD=0.35,       # derivative gain (per second)
    LF_HEADING_GAIN=0.9,    # extra steering from the line's slope (lookahead)
    LF_TARGET_X=None,       # where the line should sit in the frame, in pixels.
                            # None = image center. If your camera is mounted
                            # off-center, set this.

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

        self.roi_top = float(_cfg(cfg, 'LF_ROI_TOP'))
        self.num_bands = int(_cfg(cfg, 'LF_NUM_BANDS'))
        self.min_band_px = int(_cfg(cfg, 'LF_MIN_BAND_PIXELS'))
        self.min_bands = int(_cfg(cfg, 'LF_MIN_BANDS'))

        self.kp = float(_cfg(cfg, 'LF_STEER_KP'))
        self.kd = float(_cfg(cfg, 'LF_STEER_KD'))
        self.kh = float(_cfg(cfg, 'LF_HEADING_GAIN'))
        self.target_x = _cfg(cfg, 'LF_TARGET_X')

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

        pts = []      # (cx, cy, weight) per band with enough line pixels
        for b in range(self.num_bands):
            ys, ye = b * band_h, min((b + 1) * band_h, roi_h)
            band = mask[ys:ye]
            n = cv2.countNonZero(band)
            if n < self.min_band_px:
                continue
            m = cv2.moments(band, binaryImage=True)
            cx = m['m10'] / m['m00']
            cy = ys + m['m01'] / m['m00']
            pts.append((cx, cy, n))

        debug = dict(mask=mask, roi_y0=y0, pts=pts, w=w, h=h)

        if len(pts) < self.min_bands:
            return False, 0.0, 0.0, debug

        pts_a = np.array(pts, dtype=np.float64)
        cxs, cys, ws = pts_a[:, 0], pts_a[:, 1], pts_a[:, 2]

        target = (w / 2.0) if self.target_x is None else float(self.target_x)

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
        return True, error, heading, debug

    # ------------------------------------------------------------------ #
    # control                                                            #
    # ------------------------------------------------------------------ #
    def run(self, cam_img):
        if cam_img is None:
            return 0.0, 0.0, None

        now = time.time()
        found, error, heading, debug = self.detect(cam_img)

        if found:
            self.lost_frames = 0
            self.lost_since = None

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

        # band centroids
        for cx, cy, _ in debug['pts']:
            cv2.circle(img, (int(cx), int(y0 + cy)), 3, (255, 0, 0), -1)

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