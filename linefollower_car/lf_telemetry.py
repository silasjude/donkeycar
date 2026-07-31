"""Per-frame line-follower telemetry log.

WHY THIS EXISTS
---------------
The tub records three things: the camera frame, the steering that reached
the drivetrain, and the throttle. Everything else about a run -- which lane
offset was live, what curve gain was paired with it, whether the detector
saw the line at all, which gate rejected it -- is thrown away the moment
the process exits.

That gap has cost real time. The dashboard writes lane_offset / curve_gain
/ th_min live and nothing recorded them, so reconstructing a run meant
replaying it at half a dozen candidate offsets and picking the one whose
steering correlated best with the recording -- an inference, not a
measurement, and it disagreed across windows of the SAME run (session
26-07-30_90 replays at -0.5 in its first window and clearly not in its
second). Worse, three genuinely different faults -- the morphology open
erasing thin line, the lane-implausibility lockout, and temporal-gate
rejections -- are indistinguishable in the tub: all three look like
"steering went somewhere odd and the car left the track".

So: one JSONL line per recorded frame, alongside the tub, never touching
the tub's schema (adding fields there would break the existing 178k-record
dataset). Join to the tub on `idx` (the tub record count) or `t`.

Cost is a dict build and a short line of json per frame at 20 Hz.
"""
import json
import os
import time


class LineTelemetry:
    """Vehicle part: snapshot the LineFollower's live state each frame.

    Reads the follower object directly rather than taking the values
    through the memory bus -- most of what matters (which gate fired,
    what the detector saw) is never published to the bus at all.
    """

    def __init__(self, line_follower, path, commander=None):
        self.lf = line_follower
        self.commander = commander
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, 'a', buffering=1)   # line-buffered
        self._n = 0
        print(f"LineTelemetry: logging to {path}")

    def run(self, steering=None, throttle=None, num_records=None, mode=None,
            obstacle_info=None, obstacle_plan=None):
        lf = self.lf
        if lf is None:
            return
        cmd = self.commander
        info = obstacle_info or {}
        try:
            rec = dict(
                t=round(time.time(), 3),
                idx=num_records,
                n=self._n,
                # --- what was commanded, and by WHOM. Without the mode a
                # manually-driven stretch is indistinguishable from an
                # autopilot one that simply steered oddly, and the two
                # demand opposite conclusions.
                mode=mode,
                st=_r(steering), th=_r(throttle),
                # --- live tuning (the values that were unrecoverable) ---
                off=_r(getattr(lf, 'lane_offset', None)),
                off_applied=_r(getattr(lf, '_lane_applied', None)),
                cg=_r(getattr(lf, 'curve_gain', None)),
                thmin=_r(getattr(lf, 'th_min', None)),
                thmax=_r(getattr(lf, 'th_max', None)),
                base_off=_r(getattr(cmd, 'base_offset', None)) if cmd else None,
                # --- what the DETECTOR saw, before any gate ---
                npts=getattr(lf, 'last_npts', None),
                nblob=getattr(lf, 'last_nblob', None),
                nwhite=getattr(lf, 'last_nwhite', None),
                raw=int(bool(getattr(lf, 'last_raw_found', False))),
                # --- what the CONTROLLER concluded ---
                status=getattr(lf, 'status', None),
                x_f=_r(getattr(lf, 'x_f', None)),
                h_f=_r(getattr(lf, 'h_f', None)),
                err=_r(getattr(lf, 'last_err', None)),
                pos=_r(getattr(lf, 'lane_pos', None)),
                wx=_r(getattr(lf, 'white_x_f', None)),
                yaw=_r(getattr(lf, 'yaw_rate_f', None)),
                # --- which gate is currently fighting the detector ---
                sus=getattr(lf, '_suspect_streak', None),
                reacq=getattr(lf, '_reacquire_streak', None),
                impl=getattr(lf, '_implausible_streak', None),
                impl_ok=int(bool(getattr(lf, '_implausible_committed', False))),
                fsf=getattr(lf, 'frames_since_fix', None),
                # --- obstacle stack (all None when it is disabled) ---
                # `present` flickers badly on this detector -- it dropped
                # to False for >1s at a stretch with the SAME cone still
                # there, which is why OBSTACLE_CLEAR_DEBOUNCE_SEC is 4.0.
                # Logging present ALONGSIDE the commitment state is the
                # only way to tell a genuine clearance from a detection
                # gap after the fact, and to measure how much of the
                # return-to-lane delay is debounce vs LF_LANE_RAMP_SEC.
                obs=int(bool(info.get('present'))) if info else None,
                obs_side=info.get('side'),
                obs_m=_r(info.get('distance_m'), 2),
                obs_status=info.get('status'),
                plan=(obstacle_plan.get('status')
                      if isinstance(obstacle_plan, dict) else obstacle_plan),
                avoid_dir=getattr(cmd, '_avoid_dir', None) if cmd else None,
                avoid_kind=getattr(cmd, '_avoid_kind', None) if cmd else None,
            )
            self._f.write(json.dumps(rec) + '\n')
            self._n += 1
        except Exception as e:      # telemetry must never stop the car
            if self._n >= 0:
                print(f"LineTelemetry: disabled after error: {e}")
                self._n = -1

    def shutdown(self):
        try:
            self._f.close()
        except Exception:
            pass


def _r(v, nd=3):
    """Round floats, pass None through, stringify anything unexpected."""
    if v is None:
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return str(v)
