#!/usr/bin/env python3
"""
line_following.py — MINIMAL yellow-dash follower for DonkeyCar (Mission 1).

Deliberately simple. One job: keep the yellow hashed line under the car.
  1. Bottom part of the frame -> CIELAB + hue threshold -> yellow mask
     (thresholds measured from this track's tape in daylight, shadow, night)
  2. Split into horizontal bands, centroid per band, fit a line
  3. Steer with PD toward the line; slow down while turning
  4. Line gone: hold course briefly, then stop

Drop-in for the cv_control template (constructed as LineFollower(pid, cfg);
the pid argument is ignored). In myconfig.py:
    CV_CONTROLLER_MODULE = "line_following"
    CV_CONTROLLER_CLASS  = "LineFollower"

Offline test:  python line_following.py test <images_or_folder>
"""

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULTS = dict(
    # color of the yellow tape (calibrated from track photos day/shadow/night)
    LF_LAB_B_MIN=143,       # LAB yellowness: tape ~145-149, concrete <=139
    LF_LAB_A_MAX=142,       # rejects orange cones / red flowers
    LF_LAB_L_MIN=40,        # rejects near-black
    LF_HUE_MIN=10,          # rejects green vegetation (grass is H 34-46,
    LF_HUE_MAX=32,          #  tape is H 18-28 in all lighting)

    # where to look
    LF_ROI_TOP=0.50,        # ignore the top half of the image; the overlay
                            # draws this as a yellow horizontal line — it
                            # should sit just above the visible track
    LF_NUM_BANDS=8,
    LF_MIN_BAND_PIXELS=6,   # per-band evidence (for 160x120 camera)
    LF_MIN_BANDS=2,         # bands needed for a valid detection

    # steering: steer = KP*err + KH*heading, low-pass filtered
    LF_STEER_KP=2.0,
    LF_HEADING_GAIN=0.7,
    LF_ERR_DEADBAND=0.08,   # |err| below this = drive straight
    LF_STEER_SMOOTH=0.5,    # 0..1 fraction of new value applied per frame
    LF_TARGET_X=None,       # px where the line sits when centered;
                            # None = image center. Calibrate: park on the
                            # line, read err from overlay, add err*half_width

    # throttle
    LF_THROTTLE_MAX=0.38,   # straights (VESC needed ~0.3 to move)
    LF_THROTTLE_MIN=0.30,   # while turning hard
    LF_THROTTLE_LOST=0.26,  # while coasting after losing the line

    # losing the line
    LF_LOST_GRACE=6,        # frames to hold course (dash gaps)
    LF_LOST_STOP_SEC=2.0,   # then stop after this long without a line

    OVERLAY_IMAGE=True,
)


def _cfg(cfg, name):
    return getattr(cfg, name, DEFAULTS[name]) if cfg is not None else DEFAULTS[name]


class LineFollower:

    def __init__(self, pid=None, cfg=None):
        g = lambda n: _cfg(cfg, n)
        self.b_min, self.a_max, self.l_min = g('LF_LAB_B_MIN'), g('LF_LAB_A_MAX'), g('LF_LAB_L_MIN')
        self.hue_min, self.hue_max = g('LF_HUE_MIN'), g('LF_HUE_MAX')
        self.roi_top = g('LF_ROI_TOP')
        self.num_bands = g('LF_NUM_BANDS')
        self.min_band_px = g('LF_MIN_BAND_PIXELS')
        self.min_bands = g('LF_MIN_BANDS')
        self.kp, self.kh = g('LF_STEER_KP'), g('LF_HEADING_GAIN')
        self.deadband = g('LF_ERR_DEADBAND')
        self.smooth = g('LF_STEER_SMOOTH')
        self.target_x = g('LF_TARGET_X')
        self.th_max, self.th_min = g('LF_THROTTLE_MAX'), g('LF_THROTTLE_MIN')
        self.th_lost = g('LF_THROTTLE_LOST')
        self.lost_grace = g('LF_LOST_GRACE')
        self.lost_stop_sec = g('LF_LOST_STOP_SEC')
        self.overlay = g('OVERLAY_IMAGE')

        self.steering = 0.0
        self.throttle = 0.0
        self.lost_frames = 0
        self.lost_since = None
        self.status = "init"
        self._kernel = None

    # ------------------------------------------------------------------ #
    def detect(self, rgb):
        """Returns (found, error, heading, debug)."""
        h, w = rgb.shape[:2]
        y0 = int(h * self.roi_top)
        roi = rgb[y0:]
        roi_h = roi.shape[0]

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

        band_h = max(1, roi_h // self.num_bands)
        pts = []
        for b in range(self.num_bands):
            band = mask[b * band_h: min((b + 1) * band_h, roi_h)]
            n = cv2.countNonZero(band)
            if n < self.min_band_px:
                continue
            m = cv2.moments(band, binaryImage=True)
            pts.append((m['m10'] / m['m00'],
                        b * band_h + m['m01'] / m['m00'], n))

        debug = dict(mask=mask, y0=y0, pts=pts)
        if len(pts) < self.min_bands:
            return False, 0.0, 0.0, debug

        p = np.array(pts)
        cxs, cys, ws = p[:, 0], p[:, 1], p[:, 2]
        wgt = np.sqrt(ws * (0.5 + cys / roi_h))       # near dashes count more
        a, bf = np.polyfit(cys, cxs, 1, w=wgt)

        target = (w / 2.0) if self.target_x is None else float(self.target_x)
        # lateral error from the nearest dashes; line direction from the fit
        near = np.argsort(cys)[::-1][:3]
        x_near = np.average(cxs[near], weights=ws[near])
        error = float(np.clip((x_near - target) / (w / 2.0), -1.0, 1.0))
        trust = min(1.0, (len(pts) - 1) / 3.0)        # don't trust a 2-point slope
        heading = float(np.clip(-a * roi_h / (w / 2.0), -1.0, 1.0)) * trust

        debug.update(fit=(a, bf), target=target)
        return True, error, heading, debug

    # ------------------------------------------------------------------ #
    def run(self, cam_img):
        if cam_img is None:
            return 0.0, 0.0, None
        now = time.time()
        found, error, heading, debug = self.detect(cam_img)

        if found:
            self.lost_frames, self.lost_since = 0, None
            err = 0.0 if abs(error) < self.deadband else error
            steer = float(np.clip(self.kp * err + self.kh * heading, -1, 1))
            self.steering = float(np.clip(
                self.steering + self.smooth * (steer - self.steering), -1, 1))
            self.throttle = self.th_max - (self.th_max - self.th_min) * abs(self.steering)
            self.status = "tracking"
        else:
            self.lost_frames += 1
            if self.lost_since is None:
                self.lost_since = now
            if now - self.lost_since > self.lost_stop_sec:
                self.steering, self.throttle = 0.0, 0.0
                self.status = "stopped: no line"
            elif self.lost_frames <= self.lost_grace:
                self.throttle = max(self.th_lost, self.throttle * 0.9)
                self.status = "coasting"
            else:
                # keep turning the way we were turning; corners exit that way
                self.throttle = self.th_lost
                self.status = "searching"

        if self.status != getattr(self, '_logged', None):
            logger.info(f"LineFollower: {self.status} "
                        f"st={self.steering:+.2f} th={self.throttle:.2f}")
            self._logged = self.status

        img = self._draw(cam_img, debug, error, heading) if self.overlay else cam_img
        return self.steering, self.throttle, img

    def shutdown(self):
        pass

    # ------------------------------------------------------------------ #
    def _draw(self, cam_img, debug, error, heading):
        img = cam_img.copy()
        h, w = img.shape[:2]
        y0 = debug['y0']
        img[y0:][debug['mask'] > 0] = (0, 255, 0)
        cv2.line(img, (0, y0), (w, y0), (255, 255, 0), 1)
        bar = (0, 200, 0) if self.status == "tracking" else \
              (255, 200, 0) if self.status in ("coasting", "searching") else (255, 0, 0)
        cv2.rectangle(img, (0, 0), (w - 1, max(4, h // 30)), bar, -1)
        for cx, cy, _ in debug['pts']:
            cv2.circle(img, (int(cx), int(y0 + cy)), 3, (255, 0, 0), -1)
        if 'fit' in debug and self.status == "tracking":
            a, bf = debug['fit']
            cv2.line(img, (int(bf), y0), (int(a * (h - y0) + bf), h - 1),
                     (255, 0, 255), 2)
            cv2.line(img, (int(debug['target']), h - 12),
                     (int(debug['target']), h - 1), (255, 255, 255), 2)
        scale = max(0.38, w / 420.0)
        for i, s in enumerate([f"st:{self.steering:+.2f} th:{self.throttle:.2f}",
                               f"err:{error:+.2f} hdg:{heading:+.2f}",
                               self.status]):
            y = int(h * 0.12 + i * (16 * scale + 7))
            cv2.putText(img, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return img


# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    import sys, os, glob
    if len(sys.argv) >= 3 and sys.argv[1] == 'test':
        files = []
        for pth in sys.argv[2:]:
            files += sorted(glob.glob(os.path.join(pth, '*.[jp][pn]g'))) \
                if os.path.isdir(pth) else [pth]
        os.makedirs('lf_out', exist_ok=True)
        lf = LineFollower()
        for f in files:
            bgr = cv2.imread(f)
            if bgr is None:
                continue
            st, th, ov = lf.run(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            out = os.path.join('lf_out', os.path.basename(f))
            cv2.imwrite(out, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            print(f"{f}: steering={st:+.3f} throttle={th:.3f} ({lf.status}) -> {out}")
    else:
        print(__doc__)
