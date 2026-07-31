#!/usr/bin/env python3
"""
obstacle_avoidance.py — DonkeyCar Track 2, Mission 3: obstacle maneuvering,
decision half.

Consumes 'obstacle/info' (from obstacle_detector.py's ObstacleDetector) and
turns it into a target lane position, throttle scale, and status for the
HUD. Deliberately does NOT reimplement any of line_following.py's steering,
smoothing, or lane-change machinery — LineFollower already has all of that
(LF_LANE_OFFSET / LF_CURVE_GAIN, ramped via LF_LANE_RAMP_SEC, plus the
white-line lane-boundary tracking used for blind bends). This file's whole
job is to drive those existing knobs from obstacle state:

  LaneOffsetCommander  — runs BEFORE the LineFollower part each tick.
                          Writes line_follower.lane_offset / .curve_gain
                          directly (plain instance attributes LineFollower
                          re-reads every frame — see run() there) so
                          LineFollower's own ramp smooths the transition.
                          This is also the natural hook for a future VLM:
                          anything that can set lane_offset/curve_gain on
                          the same schedule slots in here.
  ThrottleLimiter       — runs AFTER the LineFollower part. Scales
                          'pilot/throttle' down (or to a full stop) based
                          on the nearest obstacle's distance, independent
                          of whatever LineFollower's own throttle logic
                          (which knows nothing about obstacles) computed.
  ObstacleOverlay       — runs last. Burns a small status readout onto
                          'cv/image_array' (already LineFollower's own
                          overlay by this point) for the web/FPV UI.

Two behaviors worth knowing (see myconfig.py's OBSTACLE_* block):

  OBSTACLE_SWERVE_ENABLED     default True  — a modest steer WITHIN the
    current lane, never crossing the yellow centerline. This is the
    fallback whenever a lane change isn't available (disabled, or the
    needed adjacent lane isn't clear) but a real obstacle is in view.
  OBSTACLE_LANE_CHANGE_ENABLED — a full lane change (using the same
    offset/curve-gain pairing already field-tuned for Mission 2 lane
    following) when the adjacent lane reads clear AND the obstacle is
    still at least OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M away (see that
    config's comment — two real tests were detected already too close
    for a full lane change to physically complete before contact;
    inside this distance the swerve nudge is used instead, same as if
    lane change were disabled outright). Engages for an obstacle in the
    car's OWN lane (side='right', since base_offset rides right of the
    tracked yellow line) or one spanning the whole lane (side='both');
    an obstacle in the opposing lane alone (side='left') doesn't block
    the car, so it only gets the swerve nudge, never a lane change.
    Started disabled after the monocular rewrite until swerve/stop
    behavior was confirmed solid on real footage; enable once that's
    true for your track. The offset/curve-gain target itself also EASES
    in with distance (_eased_offset/_offset_scale_for) rather than
    snapping straight to the full value the instant a lane change is
    chosen — the same distance-based shape _speed_scale_for already
    used for throttle, just applied to the steering target too.

Whenever no safe maneuver exists (lane change disabled, too close to
safely start one, or the detector reports the adjacent lane is ALSO
occupied — e.g. an oncoming car in the other lane while a cone blocks
this one) the car holds its lane and relies purely on the throttle ramp
down to a stop. It never guesses.

LaneOffsetCommander also COMMITS to a lane-change direction once chosen,
rather than re-deciding it every tick (see OBSTACLE_CLEAR_DEBOUNCE_SEC).
Found from a real test replay: even reading LineFollower's filtered x_f
(not the raw, noisier last_x), the corridor-side reading can still flip
for a frame near a close obstacle, and re-picking the direction from
scratch each tick meant the car reversed mid-lane-change and never
finished one. Once committed, it holds that direction regardless of any
single frame's reading, and only releases back to the base lane once the
obstacle has read "not present" for a sustained stretch.

--- HOW TO TEST OFF THE CAR -------------------------------------------------
  python3 obstacle_avoidance.py selftest
Exercises the pure decision function and the stateful commander/limiter
against synthetic obstacle/info dicts. No camera, depth, or car needed.
-----------------------------------------------------------------------------
"""

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


DEFAULTS = dict(
   OBSTACLE_SWERVE_ENABLED=True,
   OBSTACLE_SWERVE_OFFSET=0.35,        # in-lane steer amount, same units
                                        # as LineFollower's LF_LANE_OFFSET
                                        # (-1..1); modest on purpose — this
                                        # only needs to clear a cone-width
                                        # obstacle, not reach the far
                                        # boundary

   OBSTACLE_LANE_CHANGE_ENABLED=False, # see module docstring — off until
                                        # validated on real depth footage
   # Mirrors the field-tuned pairing already documented in myconfig.py
   # above LF_LANE_OFFSET: offset and curve_gain are tuned TOGETHER, so
   # changing one without the other reproduces the "rides beside the
   # line" / "corners too wide" failures that pairing note describes.
   OBSTACLE_LANE_CHANGE_OFFSET_LEFT=-0.75,
   OBSTACLE_LANE_CHANGE_OFFSET_RIGHT=0.5,
   OBSTACLE_LANE_CHANGE_CURVE_GAIN_LEFT=-0.8,
   OBSTACLE_LANE_CHANGE_CURVE_GAIN_RIGHT=-0.2,
   OBSTACLE_CURVE_GAIN_RAMP_SEC=1.5,   # matches LF_LANE_RAMP_SEC; curve_gain
                                        # has no ramp of its own in
                                        # line_following.py (it's read
                                        # straight off self.curve_gain), so
                                        # this file ramps it to avoid a
                                        # step-change in the curve
                                        # feed-forward term

   OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M=1.0,  # a FRESH (not yet committed)
                                        # decision only chooses a full lane
                                        # change when the obstacle is at
                                        # least this far away; closer than
                                        # this it falls back to the smaller
                                        # in-lane swerve instead. Added
                                        # 2026-07-27 after two real tests
                                        # both crashed the same way: the
                                        # obstacle was first detected
                                        # already at ~0.75-0.85m (well
                                        # inside this threshold), and
                                        # line_following.py's own steering/
                                        # throttle laws (see its
                                        # LF_THROTTLE_MIN_LANE_OFFSET note)
                                        # pegged full-lock steering and
                                        # floored throttle within ~0.17s of
                                        # commit either way -- there simply
                                        # wasn't enough real distance left
                                        # for the -0.75 lane change to
                                        # complete before contact. A swerve
                                        # needs about half the target shift
                                        # (OBSTACLE_SWERVE_OFFSET=0.35 vs
                                        # 0.75) and can complete in less
                                        # distance/time, at the cost of not
                                        # fully leaving the lane. Once
                                        # committed, OBSTACLE_LANE_CHANGE_
                                        # MIN_DISTANCE_M no longer applies --
                                        # see LaneOffsetCommander's
                                        # commitment layer -- so a lane
                                        # change already underway is not
                                        # aborted just because distance
                                        # dropped below this.
   OBSTACLE_SLOW_DIST_M=2.2,           # kept equal to obstacle_detector.py's
                                        # own default so "present" and "the
                                        # throttle starts easing off" agree;
                                        # override both together if tuned
   OBSTACLE_STOP_DIST_M=0.6,
   OBSTACLE_MIN_SPEED_SCALE=0.0,       # 0.0 = full stop at/inside
                                        # OBSTACLE_STOP_DIST_M. Deliberately
                                        # not a "creep" speed: this is the
                                        # last line of defense when no
                                        # maneuver is safe. Applies to
                                        # swerve/blocked_holding, NOT a
                                        # committed lane change -- see
                                        # OBSTACLE_LANE_CHANGE_MIN_SPEED_SCALE.
   OBSTACLE_LANE_CHANGE_MIN_SPEED_SCALE=1.0,  # throttle floor while
                                        # ACTIVELY executing a swerve OR a
                                        # committed lane change (name is a
                                        # holdover from when only lane
                                        # change used it). Found from two
                                        # real tests: the regular
                                        # distance-based floor (above)
                                        # applies to the exact signal that
                                        # triggered the maneuver in the
                                        # first place. For a lane change it
                                        # collapsed throttle to ~0.01 while
                                        # the car's actual lateral offset
                                        # had only covered a third of the
                                        # distance to its target -- it sat
                                        # nearly stationary, still pointed
                                        # at the obstacle, and drove into
                                        # it. For a swerve against a
                                        # stationary obstacle it was worse:
                                        # throttle collapsed to ~0.006 and
                                        # never recovered, and since the
                                        # car never moved the scene never
                                        # changed either -- a permanent
                                        # deadlock. Either maneuver needs
                                        # the car MOVING to turn a steering
                                        # angle into actual displacement;
                                        # 1.0 (no reduction at all -- trust
                                        # LineFollower's own turn-based
                                        # throttle easing) is the safest
                                        # starting point given that. Lower
                                        # only after confirming on real
                                        # footage that maneuvers reliably
                                        # complete first.
   OBSTACLE_SPEED_SCALE_TC=0.2,        # seconds; EMA smoothing on the
                                        # speed-scale ramp so one noisy
                                        # depth reading can't yank the
                                        # throttle
   OBSTACLE_CLEAR_DEBOUNCE_SEC=1.0,    # seconds an obstacle must be gone
                                        # (present=False) before releasing
                                        # a committed lane-change back to
                                        # the base lane -- see
                                        # LaneOffsetCommander's commitment
                                        # layer. One clear-looking frame
                                        # isn't enough; a single frame
                                        # right up against the obstacle
                                        # can misfire "clear" from the
                                        # same corridor-side noise that
                                        # motivated committing in the
                                        # first place.

   # --- EARLY RELEASE ON A NEW OBSTACLE (the two-cone weave) ------------
   # The debounce above releases on TIME, which is the wrong question when
   # a SECOND cone appears while the first is still being avoided. Two
   # real two-cone runs (2026-07-30, idx 187478-187629 and 187630-187763)
   # show the failure precisely: the car commits left around cone 1, cone
   # 2 then appears on the LEFT at 4.63m, and because the commitment
   # ignores this frame's direction suggestion the eased offset is
   # recomputed from the NEW cone's closing distance while still pointed
   # LEFT -- so the car steers HARDER left, into it, and parks at 0.29m
   # with the right lane wide open. A timer cannot fix that; the release
   # has to be event-driven.
   #
   # A detection cannot be the object we are already committed to if it
   # sits much FARTHER away than that object ever got. Both conditions are
   # required, and both are measured from the runs above:
   #   PASSED_M : the committed object must have closed to at least this
   #              range first (cone 1 reaches 0.29-0.35m). Without it the
   #              rule fires mid-pass at 0.67m while cone 1 is still the
   #              thing to avoid -- observed at idx 187497.
   #   JUMP_M   : the new detection must be at least this much farther
   #              than the committed object's closest approach. Distance
   #              is quantised by image row, so at range it wobbles
   #              3.38<->4.63m on its own; 1.0m fires on that noise, 1.5m
   #              does not.
   # Validated over all five recorded cone runs: exactly ONE trigger in
   # each two-cone run (0.35m -> 4.63m left, and 0.34m -> 4.63m left) and
   # ZERO in all three single-cone runs -- no spurious weaving.
   OBSTACLE_NEW_OBJECT_PASSED_M=0.6,
   OBSTACLE_NEW_OBJECT_JUMP_M=1.5,

   OVERLAY=True,
)


def _cfg(cfg, name):
   return getattr(cfg, name, DEFAULTS[name]) if cfg is not None else DEFAULTS[name]


def _params_from_cfg(cfg):
   return dict(
       swerve_enabled=bool(_cfg(cfg, 'OBSTACLE_SWERVE_ENABLED')),
       swerve_offset=float(_cfg(cfg, 'OBSTACLE_SWERVE_OFFSET')),
       lane_change_enabled=bool(_cfg(cfg, 'OBSTACLE_LANE_CHANGE_ENABLED')),
       lane_change_offset_left=float(_cfg(cfg, 'OBSTACLE_LANE_CHANGE_OFFSET_LEFT')),
       lane_change_offset_right=float(_cfg(cfg, 'OBSTACLE_LANE_CHANGE_OFFSET_RIGHT')),
       lane_change_curve_gain_left=float(_cfg(cfg, 'OBSTACLE_LANE_CHANGE_CURVE_GAIN_LEFT')),
       lane_change_curve_gain_right=float(_cfg(cfg, 'OBSTACLE_LANE_CHANGE_CURVE_GAIN_RIGHT')),
       slow_dist_m=float(_cfg(cfg, 'OBSTACLE_SLOW_DIST_M')),
       stop_dist_m=float(_cfg(cfg, 'OBSTACLE_STOP_DIST_M')),
       min_speed_scale=float(_cfg(cfg, 'OBSTACLE_MIN_SPEED_SCALE')),
       lane_change_min_speed_scale=float(_cfg(cfg, 'OBSTACLE_LANE_CHANGE_MIN_SPEED_SCALE')),
       lane_change_min_distance_m=float(_cfg(cfg, 'OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M')),
   )


def _speed_scale_for(distance_m, min_scale, stop_dist_m, slow_dist_m):
   """Distance -> throttle scale, floored at min_scale. Factored out so
   both the fresh decision below and LaneOffsetCommander's commitment
   override (which must recompute this with the SAME floor a committed
   lane change uses, not whatever a same-tick swerve/hold suggestion
   would have used) share one formula."""
   if distance_m is None:
       return 1.0
   if distance_m <= stop_dist_m:
       return min_scale
   elif distance_m < slow_dist_m:
       frac = (distance_m - stop_dist_m) / max(slow_dist_m - stop_dist_m, 1e-6)
       return min_scale + frac * (1.0 - min_scale)
   else:
       return 1.0


def _offset_scale_for(distance_m, stop_dist_m, slow_dist_m):
   """0..1 fraction of the way from base_offset to a lane-change target,
   by distance -- 0 at/beyond slow_dist_m, 1 at/inside stop_dist_m,
   linear between. Used so lane_offset/curve_gain EASE toward their full
   commit value as the obstacle closes, instead of snapping to it the
   instant a lane change is chosen (see OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M
   for why this was added: it doesn't undo the underlying problem those
   real tests hit -- see that config's comment -- but it's the correct
   general shape regardless, mirroring how speed_scale is already eased
   by the same two distances rather than stepped in _speed_scale_for
   above). Shares the same stop/slow_dist_m breakpoints as
   _speed_scale_for so both signals converge to "fully committed" at
   exactly the same distance, not two different ones."""
   if distance_m is None:
       return 1.0
   if distance_m <= stop_dist_m:
       return 1.0
   elif distance_m < slow_dist_m:
       return 1.0 - (distance_m - stop_dist_m) / max(slow_dist_m - stop_dist_m, 1e-6)
   else:
       return 0.0


def _eased_offset(distance_m, p, base_offset, base_curve_gain, full_offset, full_curve_gain):
   """Interpolate base_offset/base_curve_gain toward the full lane-change
   target by _offset_scale_for's distance fraction. Factored out so the
   fresh decision and the commitment override (which must recompute this
   with the SAME shape a committed lane change uses, not some other
   value) share one formula -- same reasoning as _speed_scale_for."""
   scale = _offset_scale_for(distance_m, p['stop_dist_m'], p['slow_dist_m'])
   offset = base_offset + (full_offset - base_offset) * scale
   curve_gain = base_curve_gain + (full_curve_gain - base_curve_gain) * scale
   return offset, curve_gain


def _target_from_info(info, p, base_offset, base_curve_gain):
   """Pure function: obstacle info + params -> instantaneous target
   (offset, curve_gain, speed_scale, status, lane_change_active). No
   smoothing or state here, so this is directly unit-testable;
   LaneOffsetCommander.run() applies the ramping on top each tick.
   """
   if not info or not info.get('present') or info.get('distance_m') is None:
       return base_offset, base_curve_gain, 1.0, 'clear', False

   distance_m = info['distance_m']
   side = info.get('side')
   hold_speed = _speed_scale_for(distance_m, p['min_speed_scale'],
                                 p['stop_dist_m'], p['slow_dist_m'])
   # ANY active steering maneuver (swerve or lane change) needs the car
   # actually MOVING to translate a steering angle into real lateral
   # displacement -- scaling throttle down by distance-to-obstacle, the
   # exact signal that triggered the maneuver, starves it of the speed it
   # needs. Found twice, from two real tests: a lane change with the old
   # hold_speed floor left the car sitting nearly stationary, still
   # pointed at the cone, until it drove into it; a swerve with the same
   # floor against a persistent (stationary) obstacle went further --
   # throttle collapsed to ~0.006 and never recovered, and since the car
   # never moved the scene never changed either, so the trigger never
   # cleared. A permanent deadlock, not just a slow crawl. Only the
   # genuine "no safe maneuver, just stop" path (blocked_holding below)
   # keeps the harsh hold_speed floor -- there's nothing else for it to
   # do there, so braking hard is actually correct.
   #
   # Named lc_speed for its original lane-change-only history; it's the
   # floor for both now. Config knob is still OBSTACLE_LANE_CHANGE_
   # MIN_SPEED_SCALE for the same reason -- see myconfig.py.
   lc_speed = _speed_scale_for(distance_m, p['lane_change_min_speed_scale'],
                               p['stop_dist_m'], p['slow_dist_m'])

   # side='right' means the obstacle is in the band between the tracked
   # yellow line and cx+half_w -- i.e. the CAR'S OWN lane (base_offset
   # rides at +0.5, to the right of yellow), the "cone directly in my
   # path" case. side='left' means something is in the band on the
   # OTHER side of yellow -- the opposing lane, not blocking the car at
   # all; nudging right (deeper into the own lane) is the proportionate
   # response there, not a full lane change away from a lane the car
   # isn't even in.
   if side == 'right':
       # OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M: only START a fresh lane
       # change when there's still enough distance for one to plausibly
       # complete -- see that config's comment for the two real crashes
       # that motivated it. A lane change already committed to (see
       # LaneOffsetCommander) is NOT re-gated by this on later frames,
       # only the initial choice is.
       if (p['lane_change_enabled'] and bool(info.get('left_adjacent_clear'))
               and distance_m >= p['lane_change_min_distance_m']):
           offset, cg = _eased_offset(distance_m, p, base_offset, base_curve_gain,
                                      p['lane_change_offset_left'],
                                      p['lane_change_curve_gain_left'])
           return offset, cg, lc_speed, 'lane_change_left', True
       if p['swerve_enabled']:
           # lc_speed, not hold_speed -- see the note above lc_speed's
           # definition. A swerve is just as much an active maneuver as a
           # lane change: it needs the car actually moving to translate
           # steering into lateral displacement. Found from a real test
           # with two cones on the track: an obstacle sitting in the
           # OPPOSING lane (side='left', so this branch never fires, but
           # the mirror case below did) triggered swerve_right with the
           # old hold_speed floor, throttle collapsed to ~0.006 and never
           # recovered, and since the car wasn't moving the scene never
           # changed either -- a permanent deadlock, not a slow crawl.
           return -p['swerve_offset'], base_curve_gain, lc_speed, 'swerve_left', False
       return base_offset, base_curve_gain, hold_speed, 'blocked_holding', False

   if side == 'left':
       if p['swerve_enabled']:
           return p['swerve_offset'], base_curve_gain, lc_speed, 'swerve_right', False
       return base_offset, base_curve_gain, hold_speed, 'blocked_holding', False

   if side == 'both':
       can_right = bool(info.get('right_adjacent_clear'))
       can_left = bool(info.get('left_adjacent_clear'))
       close_enough = distance_m >= p['lane_change_min_distance_m']
       if p['lane_change_enabled'] and can_right and close_enough:
           offset, cg = _eased_offset(distance_m, p, base_offset, base_curve_gain,
                                      p['lane_change_offset_right'],
                                      p['lane_change_curve_gain_right'])
           return offset, cg, lc_speed, 'lane_change_right', True
       if p['lane_change_enabled'] and can_left and close_enough:
           offset, cg = _eased_offset(distance_m, p, base_offset, base_curve_gain,
                                      p['lane_change_offset_left'],
                                      p['lane_change_curve_gain_left'])
           return offset, cg, lc_speed, 'lane_change_left', True
       # No safe maneuver: lane-change disabled, too close to safely start
       # one, or the adjacent lane is ALSO occupied (e.g. an oncoming car)
       # -- hold lane and rely on the speed ramp above. Never guess a
       # direction here.
       return base_offset, base_curve_gain, hold_speed, 'blocked_holding', False

   # side is None but present: a LEGITIMATE state since 2026-07-27 --
   # obstacle_detector.py's bootstrap corridor (wired LineFollower with
   # no lateral fix yet, e.g. the first frames after process start)
   # reports "something ahead, lateral position unresolved" exactly this
   # way (status='obstacle_unlocalized'). With no lane reference there's
   # no defensible left-vs-right call, so hold lane and slow -- the same
   # never-guess fail-safe as blocked_holding. On the real recorded
   # session starts this lasts 1-3 frames before the follower's first
   # accepted fix localizes the obstacle and a normal maneuver commits.
   return base_offset, base_curve_gain, hold_speed, 'blocked_holding', False


class LaneOffsetCommander:
   """
   DonkeyCar part. Must be added to the Vehicle BEFORE the LineFollower
   part (see manage_line.py) so its writes to line_follower.lane_offset/
   .curve_gain land before LineFollower.run() reads them this tick.

     input:  'obstacle/info'
     output: 'obstacle/plan' — dict consumed by ThrottleLimiter/ObstacleOverlay
   """

   def __init__(self, line_follower, cfg=None):
       self.lf = line_follower
       # Whatever myconfig.py already configured (0.0 for plain line
       # following, or a nonzero LF_LANE_OFFSET if Mission 2 lane following
       # is the base mode) is where we return to once the obstacle clears
       # -- this file only ever deviates from it, never redefines it.
       self.base_offset = line_follower.lane_offset
       self.base_curve_gain = line_follower.curve_gain
       self.params = _params_from_cfg(cfg)
       self.curve_gain_ramp_sec = float(_cfg(cfg, 'OBSTACLE_CURVE_GAIN_RAMP_SEC'))
       self.speed_scale_tc = float(_cfg(cfg, 'OBSTACLE_SPEED_SCALE_TC'))
       self.clear_debounce_sec = float(_cfg(cfg, 'OBSTACLE_CLEAR_DEBOUNCE_SEC'))

       self._cg_f = self.base_curve_gain
       self._speed_scale_f = 1.0
       self._t_prev = None
       self._last_status = 'clear'
       # Commitment state: which maneuver KIND ('lane_change'/'swerve')
       # and direction ('left'/'right') is currently locked in, or None
       # for both. Found from a real test replay: even with x_f
       # (filtered) driving the corridor, the side reading can still
       # flip for a frame or two right up against a close obstacle --
       # and re-deciding the maneuver from scratch every tick meant the
       # car kept reversing mid-maneuver instead of ever finishing one.
       # Once a kind+direction is chosen, hold it regardless of what any
       # single frame's side reading says, and only release back to the
       # base lane once the obstacle has been genuinely gone for
       # OBSTACLE_CLEAR_DEBOUNCE_SEC -- not just one clear-looking frame.
       # _avoid_kind was added 2026-07-27 alongside OBSTACLE_LANE_CHANGE_
       # MIN_DISTANCE_M: once swerve became the common case (obstacles
       # are routinely detected too close for a lane change), the same
       # side-flicker problem that motivated committing lane changes
       # applies to swerve too -- previously swerve was rare enough in
       # practice that this gap never got exercised.
       self._avoid_dir = None
       self._avoid_kind = None
       self._last_present_t = None
       # Closest approach of the object the current commitment is for --
       # the reference the new-object test compares against. Reset with
       # every commitment; see OBSTACLE_NEW_OBJECT_JUMP_M.
       self._min_dist_seen = None
       self.new_object_passed_m = float(_cfg(cfg, 'OBSTACLE_NEW_OBJECT_PASSED_M'))
       self.new_object_jump_m = float(_cfg(cfg, 'OBSTACLE_NEW_OBJECT_JUMP_M'))

       logger.info(
           "LaneOffsetCommander up: base_offset=%+.2f base_curve_gain=%.2f "
           "swerve=%s(%.2f) lane_change=%s slow=%.1fm stop=%.1fm",
           self.base_offset, self.base_curve_gain, self.params['swerve_enabled'],
           self.params['swerve_offset'], self.params['lane_change_enabled'],
           self.params['slow_dist_m'], self.params['stop_dist_m'])

   def run(self, info):
       now = time.time()
       dt = 0.05 if self._t_prev is None else float(np.clip(now - self._t_prev, 1e-3, 0.5))
       self._t_prev = now

       offset, cg_target, speed_target, status, lane_change = _target_from_info(
           info, self.params, self.base_offset, self.base_curve_gain)

       # --- commitment layer (see __init__ for why) ---
       present = bool(info.get('present')) if info else False
       if present:
           self._last_present_t = now

       # EARLY RELEASE ON A NEW OBSTACLE (see OBSTACLE_NEW_OBJECT_JUMP_M).
       # Runs BEFORE the commit block below, so the fresh suggestion this
       # frame already carries can commit a new direction immediately --
       # a weave cannot afford to spend the clear-debounce first. Only
       # the distance is consulted, never the side: the side reading
       # flickers left/right at close range (that flicker is why the
       # commitment layer exists), so deciding "new object" from it would
       # reintroduce exactly the reversals this layer prevents.
       dist_now = info.get('distance_m') if (info and present) else None
       if self._avoid_dir is not None and dist_now is not None:
           if (self._min_dist_seen is not None
                   and self._min_dist_seen <= self.new_object_passed_m
                   and dist_now > self._min_dist_seen + self.new_object_jump_m):
               logger.info(
                   "ObstacleAvoidance: NEW obstacle at %.2fm (committed one "
                   "closed to %.2fm) -- releasing %s %s early to re-decide",
                   dist_now, self._min_dist_seen, self._avoid_kind,
                   self._avoid_dir)
               self._avoid_dir = None
               self._avoid_kind = None
               self._min_dist_seen = None
           else:
               self._min_dist_seen = min(self._min_dist_seen, dist_now) \
                   if self._min_dist_seen is not None else dist_now

       if status in ('lane_change_left', 'lane_change_right', 'swerve_left', 'swerve_right'):
           new_kind = 'lane_change' if lane_change else 'swerve'
           new_dir = 'left' if status.endswith('left') else 'right'
           if self._avoid_dir is None:
               logger.info("ObstacleAvoidance: committing to %s %s", new_kind, new_dir)
               self._avoid_dir = new_dir
               self._avoid_kind = new_kind
               # Seed the new commitment's closest-approach reference from
               # THIS object, so the new-object test above measures against
               # the thing we just committed to rather than the last one.
               self._min_dist_seen = dist_now
           # else: already committed -- this frame's kind/direction
           # suggestion is ignored, not acted on, until released. Never
           # upgrades a committed swerve to a lane change mid-maneuver
           # (or the reverse) even if a later frame's distance would now
           # qualify differently -- switching maneuver KIND partway
           # through is exactly the kind of reversal this layer exists
           # to prevent.

       if self._avoid_dir is not None:
           gone_long_enough = (self._last_present_t is None
                               or now - self._last_present_t >= self.clear_debounce_sec)
           if gone_long_enough:
               logger.info("ObstacleAvoidance: releasing %s commitment "
                           "(%s), returning to base lane", self._avoid_kind, self._avoid_dir)
               self._avoid_dir = None
               self._avoid_kind = None
               self._min_dist_seen = None
           else:
               # This tick's raw suggestion might have been a different
               # status (if the noisy side reading flipped, or distance
               # crossed OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M) -- since
               # we're overriding to the committed maneuver, its offset/
               # speed floor must match IT, not whatever this tick's raw
               # suggestion would have used, or the maneuver gets starved
               # of throttle (or snapped to the wrong offset) on exactly
               # the ticks the commitment layer exists to protect.
               distance_m = (info or {}).get('distance_m')
               if self._avoid_kind == 'lane_change':
                   # Recomputed with _eased_offset -- the SAME shape a
                   # fresh decision at this distance would use, not a
                   # step to the full value -- so a committed lane change
                   # keeps easing in as distance closes after commit,
                   # same as it would have pre-commit.
                   full_offset = (self.params['lane_change_offset_left']
                                  if self._avoid_dir == 'left'
                                  else self.params['lane_change_offset_right'])
                   full_cg = (self.params['lane_change_curve_gain_left']
                              if self._avoid_dir == 'left'
                              else self.params['lane_change_curve_gain_right'])
                   offset, cg_target = _eased_offset(
                       distance_m, self.params, self.base_offset, self.base_curve_gain,
                       full_offset, full_cg)
                   status = f'lane_change_{self._avoid_dir}'
                   lane_change = True
               else:
                   # swerve doesn't ease by distance (never did, even
                   # pre-commit -- it's already a small, constant in-lane
                   # nudge, not a distance-scaled target like a lane
                   # change)
                   offset = (-self.params['swerve_offset'] if self._avoid_dir == 'left'
                             else self.params['swerve_offset'])
                   cg_target = self.base_curve_gain
                   status = f'swerve_{self._avoid_dir}'
                   lane_change = False
               speed_target = _speed_scale_for(
                   distance_m, self.params['lane_change_min_speed_scale'],
                   self.params['stop_dist_m'], self.params['slow_dist_m'])

       cg_step = dt / max(self.curve_gain_ramp_sec, 1e-3)
       self._cg_f += float(np.clip(cg_target - self._cg_f, -cg_step, cg_step))

       alpha = 1.0 - float(np.exp(-dt / max(self.speed_scale_tc, 1e-3)))
       self._speed_scale_f += alpha * (speed_target - self._speed_scale_f)

       # Write through to the live LineFollower. lane_offset is a plain
       # attribute LineFollower re-reads every frame and ramps internally
       # via LF_LANE_RAMP_SEC (_lane_applied) -- no second ramp needed here.
       self.lf.lane_offset = offset
       self.lf.curve_gain = self._cg_f

       if status != self._last_status:
           logger.info(
               "ObstacleAvoidance: %s -> %s (side=%s dist=%s offset->%+.2f)",
               self._last_status, status,
               (info or {}).get('side'), (info or {}).get('distance_m'), offset)
           self._last_status = status

       return dict(
           status=status, target_offset=offset, curve_gain=self._cg_f,
           speed_scale=float(np.clip(self._speed_scale_f, 0.0, 1.0)),
           lane_change=lane_change,
       )

   def shutdown(self):
       pass


class ThrottleLimiter:
   """
   DonkeyCar part. Must be added AFTER the LineFollower part (reads/
   overwrites 'pilot/throttle', which LineFollower just produced).

     inputs: 'pilot/throttle', 'obstacle/plan'
     output: 'pilot/throttle' (overwritten)
   """

   def run(self, throttle, plan):
       if throttle is None:
           return 0.0
       scale = float(plan.get('speed_scale', 1.0)) if plan else 1.0
       return float(throttle) * scale

   def shutdown(self):
       pass


class ObstacleOverlay:
   """
   DonkeyCar part. Add last, after LineFollower and ThrottleLimiter.
   Burns a one-line status readout onto the image already shown in the
   web/FPV UI, without touching line_following.py's own _overlay().

     inputs: 'cv/image_array', 'obstacle/info', 'obstacle/plan'
     output: 'cv/image_array' (overwritten)
   """

   def __init__(self, cfg=None):
       self.enabled = bool(_cfg(cfg, 'OVERLAY'))

   def run(self, img, info, plan):
       if not self.enabled or img is None:
           return img
       out = np.array(img, copy=True)
       info = info or {}
       plan = plan or {}
       status = plan.get('status', info.get('status', 'n/a'))
       dist = info.get('distance_m')
       dist_s = f"{dist:.2f}m" if dist is not None else "--"
       color = (0, 200, 0)
       if status not in ('clear', 'depth_unavailable', 'no_image'):
           color = (255, 0, 0) if status == 'blocked_holding' else (255, 165, 0)
       cv2.putText(out, f"OBS:{status} d={dist_s}", (6, out.shape[0] - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
       return out

   def shutdown(self):
       pass


# ============================================================================
# offline self-test -- no camera/depth/car needed
# ============================================================================
class _FakeLF:
   def __init__(self, lane_offset=0.0, curve_gain=-0.5):
       self.lane_offset = lane_offset
       self.curve_gain = curve_gain
       self.last_x = None
       self.lane_width_l = None
       self.lane_width_r = None


def _fake_cfg(**overrides):
   vals = dict(
       OBSTACLE_SWERVE_ENABLED=True, OBSTACLE_SWERVE_OFFSET=0.35,
       OBSTACLE_LANE_CHANGE_ENABLED=False,
       OBSTACLE_LANE_CHANGE_OFFSET_LEFT=-0.75, OBSTACLE_LANE_CHANGE_OFFSET_RIGHT=0.5,
       OBSTACLE_LANE_CHANGE_CURVE_GAIN_LEFT=-0.8, OBSTACLE_LANE_CHANGE_CURVE_GAIN_RIGHT=-0.2,
       OBSTACLE_SLOW_DIST_M=2.2, OBSTACLE_STOP_DIST_M=0.6, OBSTACLE_MIN_SPEED_SCALE=0.0,
       OBSTACLE_LANE_CHANGE_MIN_SPEED_SCALE=1.0, OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M=1.0,
       OBSTACLE_CURVE_GAIN_RAMP_SEC=1.5, OBSTACLE_SPEED_SCALE_TC=0.2,
       OBSTACLE_CLEAR_DEBOUNCE_SEC=1.0,
       OVERLAY=True,
   )
   vals.update(overrides)
   return type('cfg', (), vals)


def _check(name, cond, info):
   print(f"[{'PASS' if cond else 'FAIL'}] {name}  ->  {info}")
   return cond


def _selftest():
   ok = True
   p = _params_from_cfg(_fake_cfg())

   off, cg, sp, status, lc = _target_from_info(None, p, 0.0, -0.5)
   ok &= _check("no info -> clear/base offset/full speed",
                status == 'clear' and off == 0.0 and sp == 1.0, (off, sp, status))

   info = dict(present=True, distance_m=1.0, side='left',
               left_adjacent_clear=True, right_adjacent_clear=True)
   off, cg, sp, status, lc = _target_from_info(info, p, 0.0, -0.5)
   ok &= _check("left obstacle -> swerve toward positive (right) offset",
                off > 0 and status == 'swerve_right', (off, status))

   info['side'] = 'right'
   off, cg, sp, status, lc = _target_from_info(info, p, 0.0, -0.5)
   ok &= _check("right obstacle -> swerve toward negative (left) offset",
                off < 0 and status == 'swerve_left', (off, status))

   # a swerve is an active maneuver too (see lc_speed's docstring for the
   # real deadlock this caused: side='left', persistent stationary
   # obstacle, throttle collapsed to ~0.006 and never recovered) -- it
   # must use the maneuver floor, NOT the harsh hold floor, even very
   # close.
   info = dict(present=True, distance_m=0.3, side='left',
               left_adjacent_clear=True, right_adjacent_clear=True)
   off, cg, sp, status, lc = _target_from_info(info, p, 0.0, -0.5)
   ok &= _check("swerve inside stop_dist_m keeps the maneuver speed floor, not the harsh one",
                sp == p['lane_change_min_speed_scale'], sp)

   # contrast: blocked_holding (no maneuver possible at all) is the ONLY
   # case that should still hit the harsh floor close up
   info_blocked = dict(present=True, distance_m=0.3, side='both',
                       left_adjacent_clear=False, right_adjacent_clear=False)
   off, cg, sp, status, lc = _target_from_info(info_blocked, p, 0.0, -0.5)
   ok &= _check("blocked_holding (genuinely no safe maneuver) still uses the harsh floor",
                status == 'blocked_holding' and sp == p['min_speed_scale'], (status, sp))

   # side='right' (car's OWN lane, since base_offset rides right of the
   # tracked yellow line) with lane_change enabled + left clear + far
   # enough away -> a full lane change, not just the small in-lane
   # swerve. distance_m=1.0 sits between OBSTACLE_LANE_CHANGE_MIN_
   # DISTANCE_M (1.0, so the gate just barely allows it) and stop_dist_m
   # (0.6), so the offset is only PARTIALLY eased in yet -- not equal to
   # the raw full target. See the dedicated easing tests below for that
   # shape in isolation.
   p_lc = _params_from_cfg(_fake_cfg(OBSTACLE_LANE_CHANGE_ENABLED=True))
   info = dict(present=True, distance_m=1.0, side='right',
               left_adjacent_clear=True, right_adjacent_clear=True)
   off, cg, sp, status, lc = _target_from_info(info, p_lc, 0.0, -0.5)
   expected_off, _ = _eased_offset(1.0, p_lc, 0.0, -0.5,
                                   p_lc['lane_change_offset_left'],
                                   p_lc['lane_change_curve_gain_left'])
   ok &= _check("own-lane obstacle + lane_change enabled + left clear + far enough -> lane change, eased",
                lc and status == 'lane_change_left' and off == expected_off
                and p_lc['lane_change_offset_left'] < off < 0.0, (off, expected_off, status, lc))

   # OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M: too close for a FRESH lane
   # change to plausibly complete -> falls back to swerve, same as if
   # lane_change were disabled outright. Added 2026-07-27 after two real
   # tests were both detected already this close (~0.75-0.85m) and
   # crashed anyway -- see that config's comment.
   info_close = dict(present=True, distance_m=0.1, side='right',
                     left_adjacent_clear=True, right_adjacent_clear=True)
   off, cg, sp, status, lc = _target_from_info(info_close, p_lc, 0.0, -0.5)
   ok &= _check("too close for a fresh lane change -> falls back to swerve instead",
                not lc and status == 'swerve_left', (off, status, lc))
   # a swerve is an active maneuver too (see lc_speed's docstring for the
   # real deadlock this caused) -- it must use the maneuver floor, NOT
   # the harsh hold floor, even very close. True whether lane_change is
   # enabled-but-gated-out-by-distance (p_lc here) or disabled outright
   # (p, checked right after) -- same fallback either way.
   ok &= _check("...and that swerve fallback still keeps the maneuver speed floor",
                sp == p_lc['lane_change_min_speed_scale'], sp)
   off, cg, sp, status, lc = _target_from_info(info_close, p, 0.0, -0.5)
   ok &= _check("swerve fallback at the same close distance also keeps the maneuver floor",
                status == 'swerve_left' and sp == p['lane_change_min_speed_scale'], (status, sp))

   # _offset_scale_for/_eased_offset in isolation: 0 at/beyond slow_dist_m
   # (no offset yet), 1 at/inside stop_dist_m (fully committed), linear
   # between, and monotonic -- closer must never mean LESS committed.
   far_off, _ = _eased_offset(2.2, p_lc, 0.0, -0.5,
                              p_lc['lane_change_offset_left'],
                              p_lc['lane_change_curve_gain_left'])
   mid_off, _ = _eased_offset(1.4, p_lc, 0.0, -0.5,
                              p_lc['lane_change_offset_left'],
                              p_lc['lane_change_curve_gain_left'])
   near_off, _ = _eased_offset(0.6, p_lc, 0.0, -0.5,
                               p_lc['lane_change_offset_left'],
                               p_lc['lane_change_curve_gain_left'])
   ok &= _check("offset eases toward the full target as distance closes, not a step",
                far_off == 0.0 and far_off > mid_off > near_off
                and near_off == p_lc['lane_change_offset_left'],
                (far_off, mid_off, near_off))

   # same obstacle, but the left lane is ALSO occupied (e.g. oncoming
   # car) -- must fall back to the swerve, never force the lane change
   info = dict(present=True, distance_m=1.0, side='right',
               left_adjacent_clear=False, right_adjacent_clear=True)
   off, cg, sp, status, lc = _target_from_info(info, p_lc, 0.0, -0.5)
   ok &= _check("own-lane obstacle + lane_change enabled but left NOT clear -> falls back to swerve",
                not lc and status == 'swerve_left', (off, status, lc))

   # side='left' (opposing lane, not blocking the car at all) must NEVER
   # trigger a lane change even with lane_change enabled -- there's
   # nothing in the car's own path to change away from
   info = dict(present=True, distance_m=1.0, side='left',
               left_adjacent_clear=True, right_adjacent_clear=True)
   off, cg, sp, status, lc = _target_from_info(info, p_lc, 0.0, -0.5)
   ok &= _check("opposing-lane-only obstacle never triggers a lane change",
                not lc and status == 'swerve_right', (off, status, lc))

   info = dict(present=True, distance_m=1.0, side='both',
               left_adjacent_clear=True, right_adjacent_clear=True)
   off, cg, sp, status, lc = _target_from_info(info, p, 0.0, -0.5)
   ok &= _check("full-width block, lane_change disabled -> holds lane",
                off == 0.0 and not lc and status == 'blocked_holding', (off, status, lc))

   p2 = _params_from_cfg(_fake_cfg(OBSTACLE_LANE_CHANGE_ENABLED=True))
   info = dict(present=True, distance_m=1.0, side='both',
               left_adjacent_clear=False, right_adjacent_clear=True)
   off, cg, sp, status, lc = _target_from_info(info, p2, 0.0, -0.5)
   expected_off2, _ = _eased_offset(1.0, p2, 0.0, -0.5,
                                    p2['lane_change_offset_right'],
                                    p2['lane_change_curve_gain_right'])
   ok &= _check("full block + right clear + lane_change enabled -> changes right, eased",
                lc and status == 'lane_change_right' and off == expected_off2, (off, status, lc))

   # same, but too close for a fresh lane change -> holds lane rather
   # than starting one (side='both' has no swerve fallback -- there's
   # nothing to swerve toward when the WHOLE lane reads blocked)
   info_close_both = dict(present=True, distance_m=0.1, side='both',
                          left_adjacent_clear=False, right_adjacent_clear=True)
   off, cg, sp, status, lc = _target_from_info(info_close_both, p2, 0.0, -0.5)
   ok &= _check("full block, too close for a fresh lane change -> holds lane",
                not lc and off == 0.0 and status == 'blocked_holding', (off, status, lc))

   info = dict(present=True, distance_m=1.0, side='both',
               left_adjacent_clear=False, right_adjacent_clear=False)
   off, cg, sp, status, lc = _target_from_info(info, p2, 0.0, -0.5)
   ok &= _check("BOTH sides blocked (e.g. oncoming car) -> never swerves blind",
                not lc and off == 0.0 and status == 'blocked_holding', (off, status, lc))

   # end-to-end: commander mutates a fake LineFollower and actually ramps
   lf = _FakeLF(lane_offset=0.0, curve_gain=-0.5)
   commander = LaneOffsetCommander(lf, _fake_cfg())
   info = dict(present=True, distance_m=1.0, side='left',
               left_adjacent_clear=True, right_adjacent_clear=True)
   plan = commander.run(info)
   ok &= _check("commander writes lane_offset onto the live LineFollower",
                lf.lane_offset > 0, lf.lane_offset)

   # speed_scale EMA ramping: use blocked_holding (genuinely no safe
   # maneuver), the one case that still uses the harsh floor -- a swerve
   # or lane change now deliberately does NOT ramp down at this distance
   # (that's the fix above), so they're not a useful case to prove the
   # EMA itself still works.
   lf2b = _FakeLF(lane_offset=0.0, curve_gain=-0.5)
   commander2b = LaneOffsetCommander(lf2b, _fake_cfg())
   commander2b._t_prev = time.time() - 1.0  # simulate elapsed time so the
                                             # EMA has actually moved, not
                                             # just been nudged by a near-
                                             # zero dt between back-to-back
                                             # calls
   info_blocked2 = dict(present=True, distance_m=1.0, side='both',
                        left_adjacent_clear=False, right_adjacent_clear=False)
   plan2b = commander2b.run(info_blocked2)
   ok &= _check("speed_scale ramps meaningfully toward target, not stuck at 1.0",
                plan2b['speed_scale'] < 0.9, plan2b['speed_scale'])

   # commitment layer: a noisy side reading that flips right/left/right
   # every tick (exactly what a real close obstacle did against the
   # jittery corridor reference) must NOT make the commander reverse
   # direction mid-maneuver -- it should lock onto the first direction
   # and hold it.
   lf2 = _FakeLF(lane_offset=0.5, curve_gain=-0.8)
   p_lc_cfg = _fake_cfg(OBSTACLE_LANE_CHANGE_ENABLED=True)
   commander2 = LaneOffsetCommander(lf2, p_lc_cfg)
   flicker_sides = ['right', 'left', 'right', 'right', 'left', 'right']
   offsets_seen = []
   for side in flicker_sides:
       info = dict(present=True, distance_m=1.0, side=side,
                   left_adjacent_clear=True, right_adjacent_clear=True)
       commander2.run(info)
       offsets_seen.append(lf2.lane_offset)
   ok &= _check("commitment layer holds ONE direction despite a flickering side reading",
                all(o == offsets_seen[0] for o in offsets_seen), offsets_seen)

   # same flicker test, but for the SWERVE commitment added 2026-07-27:
   # lane_change disabled (p, not p_lc_cfg) so every frame's fresh
   # suggestion is a swerve -- confirms swerve direction locks in and
   # holds too, not just lane-change direction. Before _avoid_kind was
   # added, swerve had no commitment layer at all and would have
   # reversed with the side reading every tick.
   lf2c = _FakeLF(lane_offset=0.0, curve_gain=-0.5)
   commander2c = LaneOffsetCommander(lf2c, _fake_cfg())
   offsets_seen_c = []
   for side in flicker_sides:
       info = dict(present=True, distance_m=1.0, side=side,
                   left_adjacent_clear=True, right_adjacent_clear=True)
       commander2c.run(info)
       offsets_seen_c.append(lf2c.lane_offset)
   ok &= _check("swerve commitment also holds ONE direction despite a flickering side reading",
                commander2c._avoid_kind == 'swerve'
                and all(o == offsets_seen_c[0] for o in offsets_seen_c), offsets_seen_c)

   # once genuinely clear for long enough, it releases back to the base lane
   commander2._last_present_t = time.time() - 10.0  # force the debounce to elapse
   plan2 = commander2.run(dict(present=False, distance_m=None, side=None,
                               left_adjacent_clear=True, right_adjacent_clear=True))
   ok &= _check("commitment releases back to base lane once clear long enough",
                lf2.lane_offset == commander2.base_offset, lf2.lane_offset)

   # a COMMITTED lane change must keep easing by the CURRENT distance as
   # it closes, not freeze at whatever it was on the frame it committed
   # -- this is the actual fix for the compounding half of the real
   # crashes (see OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M's comment): commit
   # while still relatively far (partial offset), then confirm the
   # offset grows MORE negative (toward the full target) as distance
   # closes on later frames, all while `side` is held fixed so only
   # distance is driving the change.
   lf3 = _FakeLF(lane_offset=0.0, curve_gain=-0.5)
   commander3 = LaneOffsetCommander(lf3, _fake_cfg(OBSTACLE_LANE_CHANGE_ENABLED=True))
   commander3.run(dict(present=True, distance_m=1.4, side='right',
                       left_adjacent_clear=True, right_adjacent_clear=True))
   offset_far = lf3.lane_offset
   commander3.run(dict(present=True, distance_m=0.7, side='right',
                       left_adjacent_clear=True, right_adjacent_clear=True))
   offset_near = lf3.lane_offset
   ok &= _check("committed lane change keeps easing in as distance closes, doesn't freeze at commit-time value",
                commander3._avoid_dir == 'left' and offset_far > offset_near,
                (offset_far, offset_near))

   # bootstrap/unlocalized obstacle (present with side=None, from
   # obstacle_detector.py's bootstrap corridor): must hold lane + slow
   # (never guess a direction), must NOT create a maneuver commitment,
   # and a normal maneuver must still commit cleanly the moment the
   # obstacle gets localized on a later frame -- the real session starts
   # do exactly this 1-3 frames in.
   info_unloc = dict(present=True, distance_m=0.85, side=None,
                     left_adjacent_clear=None, right_adjacent_clear=None)
   off, cg, sp, status, lc = _target_from_info(info_unloc, p, 0.0, -0.5)
   ok &= _check("unlocalized obstacle (side=None) -> holds lane, harsh floor, no maneuver",
                status == 'blocked_holding' and off == 0.0 and not lc
                and sp == _speed_scale_for(0.85, p['min_speed_scale'],
                                           p['stop_dist_m'], p['slow_dist_m']),
                (off, sp, status))
   lf4 = _FakeLF(lane_offset=0.0, curve_gain=-0.5)
   commander4 = LaneOffsetCommander(lf4, _fake_cfg())
   commander4.run(info_unloc)
   ok &= _check("unlocalized obstacle creates NO commitment",
                commander4._avoid_dir is None and commander4._avoid_kind is None,
                (commander4._avoid_dir, commander4._avoid_kind))
   commander4.run(dict(present=True, distance_m=0.85, side='right',
                       left_adjacent_clear=True, right_adjacent_clear=True))
   ok &= _check("...and the first localized frame after it still commits normally",
                commander4._avoid_kind == 'swerve' and commander4._avoid_dir == 'left',
                (commander4._avoid_kind, commander4._avoid_dir))

   tl = ThrottleLimiter()
   scaled = tl.run(0.3, dict(speed_scale=0.5))
   ok &= _check("ThrottleLimiter scales throttle by speed_scale",
                abs(scaled - 0.15) < 1e-6, scaled)
   scaled_none = tl.run(0.3, None)
   ok &= _check("ThrottleLimiter passes throttle through when plan is None",
                abs(scaled_none - 0.3) < 1e-6, scaled_none)
   ok &= _check("ThrottleLimiter returns 0 for a None throttle",
                tl.run(None, dict(speed_scale=1.0)) == 0.0, None)

   print("ALL PASS" if ok else "SOME FAILED")
   return ok


if __name__ == '__main__':
   import sys
   if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
       ok = _selftest()
       sys.exit(0 if ok else 1)
   else:
       print(__doc__)
