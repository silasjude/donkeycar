#!/usr/bin/env python3
"""Sensitivity + gain grid search over the closed-loop sim."""
import itertools, sys
import numpy as np
import sim as S

MOD = '/home/pi/mycar/line_following.py'


def evaluate(overrides, kv=S.KV, c_psi=S.C_PSI, lat=2, xn=S.X_NOISE,
             seeds=3, T=25.0):
    out = {}
    for name, sfn, extra, tgt in [
            ('str4', S.scen_straight, dict(), 0.0),
            ('str2', S.scen_straight, dict(LF_THROTTLE_MAX=0.2), 0.0),
            ('lane', S.scen_lane, dict(), -0.3),
            ('crv', S.scen_curve, dict(), 0.0)]:
        o = dict(overrides); o.update(extra)
        ms = [S.metrics(S.run_sim(MOD, o, sfn(), seed=s, kv=kv, c_psi=c_psi,
                                  lat_frames=lat, x_noise=xn, T=T),
                        target_e=tgt) for s in range(seeds)]
        out[name] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    return out


def show(tag, r):
    print(f"{tag:42s} "
          f"str4 rms={r['str4']['lat_rms']*100:4.1f} mx={r['str4']['lat_max']*100:4.1f} sd={r['str4']['st_std']:.2f} | "
          f"lane rms={r['lane']['lat_rms']*100:4.1f} mx={r['lane']['lat_max']*100:4.1f} | "
          f"crv rms={r['crv']['lat_rms']*100:5.1f} mx={r['crv']['lat_max']*100:5.1f} sd={r['crv']['st_std']:.2f}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sens'
    if mode == 'sens':
        base = {}
        show('baseline', evaluate(base))
        show('lat=4 (200ms)', evaluate(base, lat=4))
        show('c_psi=1.0', evaluate(base, c_psi=1.0))
        show('c_psi=2.6', evaluate(base, c_psi=2.6))
        show('kv=11 (fast car)', evaluate(base, kv=11))
        show('x_noise=20', evaluate(base, xn=20))
        show('lat4+cpsi1.0+kv11', evaluate(base, lat=4, c_psi=1.0, kv=11))
    elif mode == 'grid':
        rows = []
        for kp, kh, ki, imax in itertools.product(
                [1.2, 1.6], [0.8, 1.2, 1.6], [0.15, 0.4], [0.2, 0.35]):
            o = dict(LF_STEER_KP=kp, LF_HEADING_GAIN=kh, LF_STEER_KI=ki,
                     LF_STEER_I_MAX=imax)
            r = evaluate(o)
            score = (r['str4']['lat_rms'] + r['lane']['lat_rms']
                     + 0.5 * r['crv']['lat_rms']
                     + 0.3 * r['crv']['lat_max']
                     + 0.05 * (r['str4']['st_std'] + r['lane']['st_std']))
            rows.append((score, f'kp={kp} kh={kh} ki={ki} imax={imax}', r))
        rows.sort(key=lambda x: x[0])
        for sc, tag, r in rows[:10]:
            show(f'{sc*100:5.1f} {tag}', r)
