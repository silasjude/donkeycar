#!/usr/bin/env python3
"""Closed-loop bicycle-model sim driving the REAL LineFollower controller.

detect() is monkeypatched with a synthetic measurement generator; everything
downstream (filters, PID, integrator, slew, throttle coupling, lane ramp)
is the actual code under test.

World model:
  e     : car lateral offset RIGHT of the yellow line, meters
  psi   : car heading right of track direction, rad
  v     : speed m/s, first-order lag toward throttle*KV
  delta : wheel angle, rad = cmd * VESC_SCALE * DELTA_MAX, servo first-order lag
  image : line x in frame = 192 - PPM*e (+ lane offset handled by controller)
  measurement latency: LAT_FRAMES of loop delay
"""
import argparse, importlib.util, sys
import numpy as np

MYCAR = '/home/pi/mycar'

PPM = 233.0          # px per meter at frame bottom (140px lane ~ 0.6 m)
DELTA_MAX = 0.40     # rad, full servo lock at the wheels
VESC_SCALE = 0.5     # VESC_STEERING_SCALE
WHEELBASE = 0.33
KV = 7.0             # m/s per throttle unit
V_TC = 0.4           # s, speed lag
SERVO_TC = 0.06      # s, steering actuator lag
C_PSI = 1.8          # heading measurement per rad of yaw
K_VP = -0.0007       # residual VP-correction error, heading per px offset
D_LOOK = 0.8         # m, effective lookahead of the heading fit (curvature ff)
DT = 0.05            # 20 Hz
X_NOISE = 8.0        # px, white noise on lateral measurement
H_NOISE = 0.05       # heading measurement noise
TRIM = 0.03          # rad, constant wheel-angle trim error (integrator's job)
H_BIAS_TRUE = 0.06   # true camera-yaw heading bias (cfg subtracts its own)


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
            kv=KV, c_psi=C_PSI, lat_frames=2, x_noise=X_NOISE):
    rng = np.random.default_rng(seed)
    lf_mod = load_module('lf_sim', module_path)
    clock = {'t': 0.0}
    lf_mod.time = type('T', (), {'time': staticmethod(lambda: clock['t'])})
    cfg = make_cfg(overrides)
    lf = lf_mod.LineFollower(cfg=cfg)
    lf.overlay = False

    # scenario: list of (t_start, kappa, lane_offset_active)
    e, psi, v, delta = 0.05, 0.0, 0.0, 0.0
    meas_q = []
    log = []
    n = int(T / DT)
    img = np.zeros((216, 384, 3), np.uint8)
    w2 = 192.0
    for k in range(n):
        t = k * DT
        clock['t'] = t
        kappa = scenario['kappa'](t)
        # engage lane offset mid-run if requested
        if 'offset_at' in scenario and t >= scenario['offset_at'][0]:
            lf.lane_offset = scenario['offset_at'][1]

        # --- synth measurement (true state NOW, delivered late) ---
        x_img = 192.0 - PPM * e + rng.normal(0, x_noise)
        h_meas = (-c_psi * psi + K_VP * (192.0 - PPM * e - 192.0)
                  + H_BIAS_TRUE + c_psi * kappa * D_LOOK / 2.0
                  + rng.normal(0, H_NOISE))
        h_meas = float(np.clip(h_meas, -1, 1))
        meas_q.append((True, x_img, h_meas))
        found, x_m, h_m = meas_q[0] if len(meas_q) <= lat_frames \
            else meas_q[-1 - lat_frames]

        def fake_detect(_img, _f=found, _x=x_m, _h=h_m):
            lf.frames_since_fix = 0
            return _f, _x, _h, dict(mask=None, roi_y0=108, pts=[], w=384,
                                    h=216, white_pts=[], fit=None)
        lf.detect = fake_detect
        st, th, _ = lf.run(img)

        # --- plant ---
        delta_cmd = st * VESC_SCALE * DELTA_MAX + TRIM
        delta += (DT / SERVO_TC) * (delta_cmd - delta) if SERVO_TC > DT else 0
        if SERVO_TC <= DT:
            delta = delta_cmd
        v += (DT / V_TC) * (th * kv - v)
        psi += DT * (v * np.tan(delta) / WHEELBASE - v * kappa)
        e += DT * v * np.sin(psi)
        log.append((t, e, psi, v, st, th, lf.i_term))
    return np.array(log)


def metrics(log, settle=5.0, target_e=0.0):
    t, e, psi, v, st, th, i = log.T
    m = t >= settle
    de = e[m] - target_e
    dst = np.diff(st[m])
    # zero crossings of lateral error after settle = weave count
    sgn = np.sign(de[np.abs(de) > 0.01])
    crossings = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
    return dict(lat_rms=float(np.sqrt((de ** 2).mean())),
                lat_max=float(np.abs(de).max()),
                st_std=float(st[m].std()),
                st_act=float(np.abs(dst).mean() / DT),  # steering speed avg
                cross=crossings,
                v_mean=float(v[m].mean()))


def scen_straight():
    return dict(kappa=lambda t: 0.0)


def scen_lane():
    return dict(kappa=lambda t: 0.0, offset_at=(3.0, -0.5))


def scen_curve(R=2.5):
    return dict(kappa=lambda t: (1.0 / R) if 5.0 <= t < 15.0 else
                (-1.0 / R if 15.0 <= t < 25.0 else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', default=MYCAR + '/line_following.py')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--trace', default=None)
    args = ap.parse_args()
    overrides = {}
    for kv_ in getattr(args, 'set'):
        k, v = kv_.split('=')
        overrides[k] = eval(v)

    scens = [
        ('straight th0.4', scen_straight, dict(), 0.0),
        ('straight th0.2', scen_straight, dict(LF_THROTTLE_MAX=0.2), 0.0),
        ('lane -0.5 th0.4', scen_lane, dict(), -0.3),
        ('curve R2.5 th0.4', scen_curve, dict(), 0.0),
    ]
    for name, sfn, extra, tgt in scens:
        o = dict(overrides); o.update(extra)
        ms = [metrics(run_sim(args.module, o, sfn(), seed=s), target_e=tgt)
              for s in range(args.seeds)]
        agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        print(f"{name:18s} lat_rms={agg['lat_rms']*100:5.1f}cm "
              f"lat_max={agg['lat_max']*100:5.1f}cm st_std={agg['st_std']:.2f} "
              f"act={agg['st_act']:.2f}/s cross={agg['cross']:.0f} "
              f"v={agg['v_mean']:.1f}m/s")
        if args.trace:
            log = run_sim(args.module, o, sfn(), seed=0)
            np.savetxt(f"{args.trace}_{name.replace(' ', '_')}.csv", log,
                       delimiter=',', header='t,e,psi,v,st,th,i', comments='')


if __name__ == '__main__':
    main()
