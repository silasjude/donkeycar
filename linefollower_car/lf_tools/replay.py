#!/usr/bin/env python3
"""Replay a tub session through LineFollower with recorded timestamps.

Usage: replay.py <session_id> [--offset F] [--csv out.csv] [--frames a:b]
"""
import argparse, importlib.util, json, sys, time

import cv2
import numpy as np

MYCAR = '/home/pi/mycar'
IMG = MYCAR + '/data_line/images'

sys.path.insert(0, MYCAR)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def load_session(sid):
    recs = []
    import glob
    for cat in sorted(glob.glob(MYCAR + '/data_line/catalog_*.catalog'),
                      key=lambda p: int(p.split('_')[-1].split('.')[0])):
        for l in open(cat):
            r = json.loads(l)
            if r['_session_id'] == sid:
                recs.append(r)
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--offset', type=float, default=None)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--frames', default=None)
    ap.add_argument('--set', action='append', default=[],
                    help='config override, e.g. LF_ROI_TOP=0.62')
    ap.add_argument('--module', default=MYCAR + '/line_following.py')
    args = ap.parse_args()

    cfg = load_module('myconfig', MYCAR + '/myconfig.py')
    if args.offset is not None:
        cfg.LF_LANE_OFFSET = args.offset
    for kv in getattr(args, 'set'):
        k, v = kv.split('=')
        setattr(cfg, k, eval(v))

    lf_mod = load_module('line_following', args.module)

    recs = load_session(args.session)
    if args.frames:
        a, b = args.frames.split(':')
        recs = recs[int(a):int(b) if b else None]
    if not recs:
        print('no records'); return

    # fake clock: LineFollower calls time.time() once per run()
    clock = {'t': recs[0]['_timestamp_ms'] / 1000.0}
    lf_mod.time = type('T', (), {'time': staticmethod(lambda: clock['t'])})

    lf = lf_mod.LineFollower(cfg=cfg)
    # _last_wl/_last_wr don't exist on this LineFollower (that was the old,
    # now-replaced line_following.py) -- _split_whites's wl/wr are locals,
    # not stored on self. Wrap it so the CSV can report real values instead
    # of a silent nan for every row.
    lf._dbg_wl = lf._dbg_wr = None
    _orig_split_whites = lf._split_whites
    def _split_whites_dbg(*a, **kw):
        wl, wr, wl_ok, wr_ok = _orig_split_whites(*a, **kw)
        lf._dbg_wl, lf._dbg_wr = wl, wr
        return wl, wr, wl_ok, wr_ok
    lf._split_whites = _split_whites_dbg
    rows = []
    for r in recs:
        clock['t'] = r['_timestamp_ms'] / 1000.0
        bgr = cv2.imread(f"{IMG}/{r['_index']}_cam_image_array_.jpg")
        if bgr is None:
            continue
        lf._dbg_wl = lf._dbg_wr = None
        st, th, _ = lf.run(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        hterm = 0.0 if lf.h_f is None else float(np.clip(
            lf.h_f - lf.h_bias, -lf.h_clip, lf.h_clip))
        rows.append(dict(
            idx=r['_index'], t=clock['t'], rec_st=r['steering'],
            rec_th=r['throttle'], st=st, th=th, status=lf.status,
            x_raw=lf.last_x if lf.last_x is not None else np.nan,
            x_f=lf.x_f if lf.x_f is not None else np.nan,
            h_f=lf.h_f if lf.h_f is not None else np.nan,
            err=lf.last_err, p=lf.kp * lf.last_err, d=lf.kd * lf.d_f,
            i=lf.i_term, h=lf.kh * hterm,
            lane=lf._lane_applied,
            lane_pos=lf.lane_pos if getattr(lf, 'lane_pos', None) is not None
                else np.nan,
            wl=lf._dbg_wl if lf._dbg_wl is not None else np.nan,
            wr=lf._dbg_wr if lf._dbg_wr is not None else np.nan,
            dl=lf.dist_left or np.nan, dr=lf.dist_right or np.nan))
    import csv
    out = args.csv or '/dev/stdout'
    with open(out, 'w', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    # summary
    a = {k: np.array([r[k] for r in rows if isinstance(r[k], float)])
         for k in ('rec_st', 'st', 'err', 'p', 'd', 'i', 'h', 'h_f')}
    trk = np.array([r['status'] == 'tracking' for r in rows])
    print(f"# frames={len(rows)} tracking={trk.mean()*100:.0f}% "
          f"statuses={ {s: sum(1 for r in rows if r['status']==s) for s in set(r['status'] for r in rows)} }",
          file=sys.stderr)
    print(f"# rec_st: mean|.|={np.abs(a['rec_st']).mean():.3f} std={a['rec_st'].std():.3f}", file=sys.stderr)
    print(f"# sim st: mean|.|={np.abs(a['st']).mean():.3f} std={a['st'].std():.3f} "
          f"corr(rec,sim)={np.corrcoef(a['rec_st'], a['st'])[0,1]:.3f}", file=sys.stderr)
    for k in ('p', 'd', 'i', 'h'):
        print(f"#   {k}: mean={a[k].mean():+.3f} std={a[k].std():.3f}", file=sys.stderr)

if __name__ == '__main__':
    main()
