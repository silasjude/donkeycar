"""
Custom web dashboard for the line-follow + obstacle-maneuvering car.

Bolts onto DonkeyCar's stock LocalWebController (port 8887) WITHOUT
touching the donkeycar library: tornado's Application.add_handlers()
inserts new rules ahead of the app's original wildcard group, so a route
defined here shadows the stock one of the same name while every route it
does NOT define (/drive, /wsDrive, /static, ...) keeps working
unchanged. The classic UI stays reachable at /drive.

Nothing here is required by the car: remove the attach_dashboard() call
and the vehicle drives exactly as before, on the stock UI.


ROUTES
------
GET  /, /dashboard  Serves dashboard.html from this directory (re-read
      per request, so editing the page only needs a browser refresh, not
      a restart of the drive process).

GET  /api/tuning    Live lane_offset / curve_gain / throttle limits, the
      myconfig.py values to reset back to, the field-tuned LANE_PRESETS,
      and which preset (if any) the live pair currently sits on.

POST /api/tuning    Live-set any of lane_offset / curve_gain /
      throttle_min / throttle_max. Writes BOTH the LineFollower attribute
      AND LaneOffsetCommander's base_offset / base_curve_gain: the
      commander re-writes lf.lane_offset/.curve_gain every pilot frame
      from its base values, so updating only the follower would be
      silently overwritten one tick later. Changes are runtime-only --
      myconfig.py is NOT modified; note values you like and copy them
      there by hand.

GET  /video         Lower-latency MJPEG stream that shadows the stock
      one. See VideoAPI for the measurements; the library's handler is
      left untouched and takes over again if this route is dropped.


VEHICLE PART
------------
DashboardTelemetry pushes live follower/obstacle state to all /wsDrive
websocket clients a few times a second under a single 'telemetry' key
(the stock vehicle.html ignores unknown keys, so both UIs coexist). Add
it near the END of the part list, after DriveMode, so 'steering' and
'throttle' are this tick's final values rather than last tick's.


WIRING (see manage_line.py)
---------------------------
    from dashboard import attach_dashboard, DashboardTelemetry

    web_ctr = next((e['part'] for e in V.parts
                    if isinstance(e['part'], LocalWebController)), None)
    if web_ctr is not None:
        attach_dashboard(web_ctr, line_follower, obstacle_commander, cfg)
        ...
        V.add(DashboardTelemetry(web_ctr, line_follower),
              inputs=['user/mode', 'steering', 'throttle', 'recording',
                      'obstacle/info', 'obstacle/plan'])

attach_dashboard() must run after the CV controller is built (it needs
the live LineFollower, and the LaneOffsetCommander when obstacle
avoidance is on) and before V.start() spins up the web server thread.

Start/stop needs no server code here: the page drives the existing
/wsDrive protocol ('drive_mode': 'local'/'user'), same as the classic
page's mode dropdown.
"""

import json
import logging
import os
import time

import tornado.escape
import tornado.gen
import tornado.iostream
import tornado.web

try:
    from donkeycar.utils import arr_to_binary
except Exception:       # keeps this module importable off the car
    arr_to_binary = None

logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_HTML = os.path.join(_THIS_DIR, 'dashboard.html')

# Server-side safety clamps (the UI sliders use narrower ranges).
LANE_OFFSET_LIMITS = (-1.5, 1.5)
CURVE_GAIN_LIMITS = (-3.0, 3.0)
THROTTLE_LIMITS = (0.0, 1.0)
# The slowdown law bottoms out AT th_min whenever the heading term pegs
# (common in bends), so a th_min below the motor's stall threshold parks
# the car mid-corner: 2026-07-29 PM session, th_min~0.05 -> 4 stalls,
# ~37s frozen while "tracking". 0.10 still stalls occasionally but the
# follower's stall-kick recovers it; anything lower is unrecoverable by
# sliders alone.
THROTTLE_MIN_FLOOR = 0.10


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


# Field-tuned lane presets (2026-07-24 evening, on track; the sim had the
# SIGN wrong, so these numbers are measurements, not predictions). Each is
# an offset/curve-gain PAIR that was driven and tuned TOGETHER -- the
# offset on its own is not "the setting for that lane", the pair is.
#
# NOTE ON THE SCALE: lane_offset is in lane-widths out from the yellow
# centerline, and +/-1.0 puts the car's target ON the outer white boundary
# line, not in the lane (see line_following.py's LF_LANE_OFFSET and
# _lane_pos: "0.0 = on the yellow, 1.0 = on the outer white"). So "+1 =
# right lane" was never true -- the lane is driven at about half to
# three-quarters of that, which is what these measured values are.
LANE_PRESETS = (
    dict(key='left', label='left lane', lane_offset=-0.75, curve_gain=-0.8),
    dict(key='center', label='center line', lane_offset=0.0, curve_gain=-0.8),
    dict(key='right', label='right lane', lane_offset=0.5, curve_gain=-0.2),
)

# Moving the offset slider without moving the gain leaves the curve
# feed-forward mismatched: -0.8 at +0.5 is 4x too much anti-curve and
# under-turns every bend, which is a prime suspect for session 26-07-29_88
# drifting out of its lane until the yellow left the frame. The dashboard
# writes the two independently and nothing recorded either, so this warns
# instead of silently "helping" -- the pairing is the driver's call.
# (-0.5 is not itself a field measurement; it sits between -0.75 and 0.0,
# which were BOTH measured at -0.8, so it inherits that.)
CURVE_GAIN_PAIRINGS = tuple(
    (p['lane_offset'], p['curve_gain']) for p in LANE_PRESETS) + ((-0.5, -0.8),)
PAIRING_OFFSET_TOL = 0.15   # how close the offset must be to a known one
PAIRING_GAIN_TOL = 0.05     # gain mismatch worth mentioning

# Tighter than the pairing tolerances above: that one asks "is this gain
# plausible for this offset", this one asks "is the car sitting exactly on
# a measured preset", which is what the UI badges as optimal.
PRESET_OFFSET_TOL = 0.02
PRESET_GAIN_TOL = 0.02


def _pairing_warning(offset, curve_gain):
    """Message when the live offset/curve-gain pair is not a field-tuned
    combination, else None. Only speaks up about offsets we have actually
    measured a gain for."""
    for known_off, known_gain in CURVE_GAIN_PAIRINGS:
        if abs(offset - known_off) <= PAIRING_OFFSET_TOL:
            if abs(curve_gain - known_gain) > PAIRING_GAIN_TOL:
                return ('lane offset %+.2f was field-tuned with curve gain '
                        '%+.2f, not %+.2f' % (offset, known_gain, curve_gain))
            return None
    return None


def _preset_match(offset, curve_gain):
    """Key of the field-tuned preset the live pair is sitting on, else None.

    Both halves must match: an offset of +0.50 carrying a -0.80 gain is a
    mismatched pair, not "the right lane", and badging it as optimal would
    hide exactly the mismatch _pairing_warning exists to catch.
    """
    for p in LANE_PRESETS:
        if (abs(offset - p['lane_offset']) <= PRESET_OFFSET_TOL
                and abs(curve_gain - p['curve_gain']) <= PRESET_GAIN_TOL):
            return p['key']
    return None


class DashboardHandler(tornado.web.RequestHandler):
    """Serves dashboard.html. Read from disk per request so edits show up
    on refresh without restarting the drive process."""

    def get(self):
        self.set_header('Content-Type', 'text/html; charset=utf-8')
        with open(DASHBOARD_HTML, 'r', encoding='utf-8') as f:
            self.write(f.read())


# Ceiling on the stream rate, as a backstop only -- the de-duplication
# below already pins the rate to the camera (CAMERA_FRAMERATE =
# DRIVE_LOOP_HZ = 20). Keep it comfortably ABOVE that: setting it AT 20
# measured 18.9 fps rather than 20, because loop jitter puts some frames
# barely under the 50 ms bar and the cap drops a real one. Only a camera
# faster than this should ever be throttled here.
VIDEO_MAX_FPS = 40.0
VIDEO_POLL_SEC = 0.01   # how often to check for a new frame while idle


class VideoAPI(tornado.web.RequestHandler):
    """MJPEG stream that puts each camera frame on the wire at most once.

    The stock handler (donkeycar/parts/web_controller/web.py, VideoAPI)
    polls img_arr every 5 ms -- 200 Hz -- and re-encodes and re-sends it
    whether or not it changed. The camera runs at CAMERA_FRAMERATE
    (= DRIVE_LOOP_HZ = 20), so most of what it sends is byte-identical
    duplicates. MJPEG inside an <img> has no way to skip stale frames --
    the browser decodes and displays every one, in order -- so that
    backlog IS the lag the dashboard was showing.

    Two changes, both aimed at latency rather than throughput:

      - de-duplicate. Only encode and send when img_arr is a genuinely
        new array, so the wire rate follows the camera instead of the
        poll loop. This alone is the bulk of the win.
      - never queue. Awaiting the previous frame's flush before looking
        for the next one means a slow link costs frame RATE, not growing
        DELAY: whatever arrived while we were blocked is skipped and the
        client gets the newest frame, not the oldest unsent one.

    Measured on this Pi against a simulated 20 Hz camera at 384x216,
    8 s per sample over loopback:

        stock   163.3 fps   2133 KiB/s   (16.7 Mbit/s)
        this     19.8 fps    258 KiB/s   ( 2.0 Mbit/s)

    -- 8.3x less data, and the stream now tracks the camera exactly
    (80 distinct frames in 4 s, none duplicated, none dropped).

    Encoding deliberately stays on donkeycar.utils.arr_to_binary (PIL).
    The obvious "speed it up with cv2" is a pessimisation here:
    benchmarked at 384x216 on this Pi, PIL takes 0.64 ms/frame versus
    1.22 ms for cv2.imencode at equivalent quality. Encoding was never
    the bottleneck -- duplicate transmission was.
    """

    async def get(self):
        self.set_header(
            'Content-type',
            'multipart/x-mixed-replace;boundary=--boundarydonotcross')
        boundary = '--boundarydonotcross\n'
        min_interval = 1.0 / VIDEO_MAX_FPS
        # Holding a reference to the frame we last sent (rather than its
        # id()) is what makes the identity test exact: a released array
        # could otherwise be replaced by a new one at the same address and
        # read as "unchanged".
        last_img = None
        last_sent = 0.0
        while True:
            img = getattr(self.application, 'img_arr', None)
            now = time.time()
            # The vehicle loop rebinds img_arr to a new array each tick and
            # never mutates one in place, so identity is a valid "is this
            # frame new" test and costs nothing.
            if (img is None or img is last_img
                    or now - last_sent < min_interval):
                await tornado.gen.sleep(VIDEO_POLL_SEC)
                continue
            try:
                jpg = arr_to_binary(img)
            except Exception as e:
                logger.warning('Dashboard video: encode failed', exc_info=e)
                await tornado.gen.sleep(VIDEO_POLL_SEC)
                continue
            last_img, last_sent = img, now
            self.write(boundary)
            self.write('Content-type: image/jpeg\r\n')
            self.write('Content-length: %s\r\n\r\n' % len(jpg))
            self.write(jpg)
            try:
                await self.flush()
            except tornado.iostream.StreamClosedError:
                return          # client navigated away; end the request


class TuningAPI(tornado.web.RequestHandler):
    """GET current base tuning values; POST live updates.

    Runs on the tornado thread while the vehicle loop reads the same
    attributes -- these are single float attribute assignments (atomic
    under the GIL), same write pattern LaneOffsetCommander itself uses.
    """

    def _base(self):
        """The (offset, curve_gain) the car returns to between maneuvers.

        These live on the LaneOffsetCommander when obstacle avoidance is
        wired -- it overwrites the follower's own attributes every pilot
        frame, so the follower's values are the CURRENT ones (mid-swerve,
        mid-lane-change), not the ones the driver set.
        """
        d = self.application.dashboard
        lf, cmd = d['lf'], d['commander']
        if cmd:
            return cmd.base_offset, cmd.base_curve_gain
        return lf.lane_offset, lf.curve_gain

    def get(self):
        d = self.application.dashboard
        lf, cmd, cfg = d['lf'], d['commander'], d['cfg']
        offset, curve_gain = self._base()
        self.write(dict(
            lane_offset=offset,
            curve_gain=curve_gain,
            throttle_min=lf.th_min,
            throttle_max=lf.th_max,
            config_lane_offset=float(getattr(cfg, 'LF_LANE_OFFSET', 0.0)),
            config_curve_gain=float(getattr(cfg, 'LF_CURVE_GAIN', 0.0)),
            config_throttle_min=float(getattr(cfg, 'LF_THROTTLE_MIN', 0.1)),
            config_throttle_max=float(getattr(cfg, 'LF_THROTTLE_MAX', 0.3)),
            obstacle_avoidance=cmd is not None,
            mode=self.application.mode,
            pairing_warning=_pairing_warning(offset, curve_gain),
            presets=[dict(p) for p in LANE_PRESETS],
            preset=_preset_match(offset, curve_gain),
        ))

    def post(self):
        d = self.application.dashboard
        lf, cmd = d['lf'], d['commander']
        data = tornado.escape.json_decode(self.request.body)
        applied = {}
        if data.get('lane_offset') is not None:
            v = _clamp(data['lane_offset'], *LANE_OFFSET_LIMITS)
            lf.lane_offset = v
            if cmd:
                cmd.base_offset = v
            applied['lane_offset'] = v
        if data.get('curve_gain') is not None:
            v = _clamp(data['curve_gain'], *CURVE_GAIN_LIMITS)
            lf.curve_gain = v
            if cmd:
                cmd.base_curve_gain = v
                # keep the commander's own curve-gain ramp state in sync
                # so it doesn't slowly ramp back from the OLD value
                cmd._cg_f = v
            applied['curve_gain'] = v
        # th_min/th_max are read every frame by the throttle law -- no
        # commander sync needed (obstacle avoidance only multiplies the
        # resulting throttle). Cross-clamped so min <= max always holds.
        if data.get('throttle_min') is not None:
            v = _clamp(data['throttle_min'], THROTTLE_MIN_FLOOR,
                       THROTTLE_LIMITS[1])
            v = min(v, lf.th_max)
            lf.th_min = v
            applied['throttle_min'] = v
        if data.get('throttle_max') is not None:
            v = _clamp(data['throttle_max'], *THROTTLE_LIMITS)
            v = max(v, lf.th_min)
            lf.th_max = v
            applied['throttle_max'] = v
        if applied:
            logger.info('Dashboard tuning applied: %s (runtime only -- '
                        'myconfig.py unchanged)', applied)
        # Re-check the pairing and the preset match on every write, not
        # just when the offset moved: fixing a mismatch means moving the
        # OTHER slider, so both need to be re-reported either way.
        offset, curve_gain = self._base()
        warning = _pairing_warning(offset, curve_gain)
        if warning and applied:
            logger.warning('Dashboard tuning pairing: %s', warning)
        applied['pairing_warning'] = warning
        applied['preset'] = _preset_match(offset, curve_gain)
        self.write(applied)


def attach_dashboard(web_ctr, line_follower, commander, cfg):
    """Wire the dashboard routes onto an existing LocalWebController.

    Call after the CV controller is built (needs the live LineFollower
    and, when obstacle avoidance is on, the LaneOffsetCommander) and
    before V.start() spins up the web server thread.

    `commander` may be None -- with obstacle avoidance off, the follower
    holds the base tuning values itself.
    """
    web_ctr.dashboard = dict(lf=line_follower, commander=commander, cfg=cfg)
    routes = [
        (r'/', DashboardHandler),
        (r'/dashboard', DashboardHandler),
        (r'/api/tuning', TuningAPI),
    ]
    # Only shadow /video if the encoder is actually importable -- falling
    # back to the stock (laggy) stream beats serving a broken one.
    if arr_to_binary is not None:
        routes.append((r'/video', VideoAPI))
    else:
        logger.warning('Dashboard: donkeycar.utils.arr_to_binary unavailable, '
                       'leaving the stock /video stream in place')
    web_ctr.add_handlers(r'.*', routes)
    logger.info('Dashboard attached at / (classic UI still at /drive)')


class DashboardTelemetry:
    """Vehicle part: pushes live state to websocket clients ~5x/sec.

    inputs: 'user/mode', 'steering', 'throttle', 'recording',
            'obstacle/info', 'obstacle/plan'
    (final steering/throttle -- add this part AFTER DriveMode. obstacle
    keys are None until the pilot first runs; Memory.get returns None
    for missing keys.)
    """

    def __init__(self, web_ctr, line_follower, hz=5.0):
        self.web = web_ctr
        self.lf = line_follower
        self.interval = 1.0 / hz
        self._last_t = 0.0

    def run(self, mode, steering, throttle, recording, info, plan):
        now = time.time()
        if now - self._last_t < self.interval or self.web.loop is None \
                or not self.web.wsclients:
            return
        self._last_t = now
        lf = self.lf
        t = dict(
            mode=mode,
            recording=bool(recording),
            steering=None if steering is None else round(float(steering), 3),
            throttle=None if throttle is None else round(float(throttle), 3),
            lf_status=getattr(lf, 'status', None),
            x_f=None if getattr(lf, 'x_f', None) is None else round(float(lf.x_f), 1),
            lane_offset=round(float(lf.lane_offset), 3),
            lane_applied=round(float(getattr(lf, '_lane_applied', 0.0)), 3),
            curve_gain=round(float(lf.curve_gain), 3),
        )
        if info:
            t['obstacle'] = dict(
                present=bool(info.get('present')),
                side=info.get('side'),
                distance_m=(None if info.get('distance_m') is None
                            else round(float(info['distance_m']), 2)),
                status=info.get('status'),
            )
        if plan:
            t['plan'] = dict(
                status=plan.get('status'),
                speed_scale=(None if plan.get('speed_scale') is None
                             else round(float(plan['speed_scale']), 2)),
            )
        # update_wsclients must run on the tornado IOLoop's own thread
        self.web.loop.add_callback(self.web.update_wsclients,
                                   {'telemetry': t})

    def shutdown(self):
        pass
