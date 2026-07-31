#!/usr/bin/env python3
"""Closed-loop sim v2 — plant IDENTIFIED from session 26-07-24_21 footage.

Differences from sim.py (which failed to predict the field behavior):
  - x measurement carries the camera-yaw shift: x = 192 - PPM*e - F*psi.
    In a weave, yaw dominates the image motion (F*psi >> PPM*e), so the
    "position error" the controller chases is mostly instantaneous yaw.
  - heading measurement is the EMPIRICAL model from regression on the real
    yaw trace (phase-correlated far-field strip): h = 0.35*err + 0.10*psi
    + 0.19 + curve lean. It is NOT a yaw damper (geometry: small camera
    yaw translates the line image without changing its slope).
  - total loop delay 4-5 frames (measured: corr(steering -> yaw rate)
    peaks at 200-250 ms).
  - optional vision-gyro: measured yaw rate (delayed, noisy) available to
    the controller if the module supports LF_GYRO_GAIN.
"""
import importlib.util
import numpy as np

MYCAR = '/home/pi/mycar'
PPM = 233.0
F_YAW = 279.0        # px of image shift per rad of camera yaw
DELTA_MAX = 0.40
VESC_SCALE = 0.5
WHEELBASE = 0.33
V_TC = 0.4
SERVO_TC = 0.10
DT = 0.05
X_NOISE = 10.0       # measured (residual vs 5-frame median)
H_NOISE = 0.05
YAWRATE_NOISE = 0.12 # rad/s, vision-gyro measurement noise
TRIM = 0.03
H_BIAS_TRUE = 0.19   # measured const in h_f at err=0, psi=0
H_ERR_COUP = 0.35    # measured: h_f vs instantaneous err
H_PSI_COUP = 0.10    # measured: h_f vs yaw beyond err's content
H_CURVE = 1.25       # h per (1/m) curvature: h~0.5 at R2.5 (matches field)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Cfg:
    pass


def make_cfg(overrides):
    cfg = Cfg()
    my = load_module('myconfig', MYCAR + '/myconfig.py')
    for k in dir(my):
        if k.startswith(('LF_', 'OVERLAY')):
            setattr(cfg, k, getattr(my, k))
    cfg.OVERLAY_IMAGE = False
    cfg.LF_WHITE_ENABLED = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_sim(module_path, overrides, scenario, seed=0, T=25.0,
            kv=7.0, lat=4, x_noise=X_NOISE, f_yaw=F_YAW,
            h_err_coup=H_ERR_COUP, h_psi_coup=H_PSI_COUP):
    rng = np.random.default_rng(seed)
    lf_mod = load_module('lf_sim', module_path)
    clock = {'t': 0.0}
    lf_mod.time = type('T', (), {'time': staticmethod(lambda: clock['t'])})
    cfg = make_cfg(overrides)
    lf = lf_mod.LineFollower(cfg=cfg)
    lf.overlay = False

    e, psi, v, delta = 0.05, 0.0, 0.0, 0.0
    meas_q = []
    log = []
    img = np.zeros((216, 384, 3), np.uint8)
    psi_prev = 0.0
    for k in range(int(T / DT)):
        t = k * DT
        clock['t'] = t
        kappa = scenario['kappa'](t)
        if 'offset_at' in scenario and t >= scenario['offset_at'][0]:
            lf.lane_offset = scenario['offset_at'][1]

        x_img = 192.0 - PPM * e - f_yaw * psi + rng.normal(0, x_noise)
        err_inst = (x_img - 192.0) / 192.0
        h_meas = (h_err_coup * err_inst + h_psi_coup * psi + H_BIAS_TRUE
                  + H_CURVE * kappa + rng.normal(0, H_NOISE))
        h_meas = float(np.clip(h_meas, -1, 1))
        yr_meas = (psi - psi_prev) / DT + rng.normal(0, YAWRATE_NOISE)
        psi_prev = psi
        meas_q.append((x_img, h_meas, yr_meas))
        x_m, h_m, yr_m = meas_q[0] if len(meas_q) <= lat else meas_q[-1 - lat]

        def fake_detect(_img, _x=x_m, _h=h_m):
            lf.frames_since_fix = 0
            return True, _x, _h, dict(mask=None, roi_y0=108, pts=[], w=384,
                                      h=216, white_pts=[], fit=None,
                                      yaw_rate=None)
        lf.detect = fake_detect
        if hasattr(lf, 'yaw_rate_f'):        # vision-gyro-aware module
            lf._sim_yaw_rate = yr_m
        st, th, _ = lf.run(img)

        delta_cmd = st * VESC_SCALE * DELTA_MAX + TRIM
        delta += min(DT / SERVO_TC, 1.0) * (delta_cmd - delta)
        v += (DT / V_TC) * (th * kv - v)
        psi += DT * (v * np.tan(delta) / WHEELBASE - v * kappa)
        e += DT * v * np.sin(psi)
        log.append((t, e, psi, v, st, th, lf.i_term))
    return np.array(log)


def metrics(log, settle=5.0, target_e=0.0):
    t, e, psi, v, st, th, i = log.T
    m = t >= settle
    de = e[m] - target_e
    s = st[m] - st[m].mean()
    zc = int(np.sum(np.sign(s[1:]) != np.sign(s[:-1])))
    return dict(lat_rms=float(np.sqrt((de ** 2).mean())),
                lat_max=float(np.abs(de).max()),
                lat_mean=float(de.mean()),
                st_std=float(st[m].std()),
                osc_hz=zc / 2.0 / (t[m][-1] - t[m][0]),
                v_mean=float(v[m].mean()))


def scen_straight():
    return dict(kappa=lambda t: 0.0)


def scen_lane():
    return dict(kappa=lambda t: 0.0, offset_at=(3.0, -0.5))


def scen_curve(R=2.5):
    return dict(kappa=lambda t: (1.0 / R) if 5.0 <= t < 15.0 else
                (-1.0 / R if 15.0 <= t < 25.0 else 0.0))


def evaluate(overrides, module=MYCAR + '/line_following.py', seeds=3, **kw):
    out = {}
    for name, sfn, extra, tgt in [
            ('str4', scen_straight, dict(), 0.0),
            ('str3', scen_straight, dict(LF_THROTTLE_MAX=0.3), 0.0),
            ('lane', scen_lane, dict(), -0.3),
            ('crv', scen_curve, dict(), 0.0)]:
        o = dict(overrides); o.update(extra)
        ms = [metrics(run_sim(module, o, sfn(), seed=s, **kw), target_e=tgt)
              for s in range(seeds)]
        out[name] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    return out


def show(tag, r):
    print(f"{tag:40s} str4 rms={r['str4']['lat_rms']*100:5.1f} sd={r['str4']['st_std']:.2f} "
          f"f={r['str4']['osc_hz']:.1f}Hz off={r['str4']['lat_mean']*100:+4.1f} | "
          f"str3 rms={r['str3']['lat_rms']*100:4.1f} | "
          f"lane rms={r['lane']['lat_rms']*100:4.1f} mx={r['lane']['lat_max']*100:4.1f} | "
          f"crv rms={r['crv']['lat_rms']*100:5.1f} mx={r['crv']['lat_max']*100:5.1f}")


if __name__ == '__main__':
    show('identified plant, current gains', evaluate({}))
    show(' .. lat=5', evaluate({}, lat=5))
    show(' .. kv=9', evaluate({}, kv=9))
