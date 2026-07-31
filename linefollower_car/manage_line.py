#!/usr/bin/env python3
"""

Scripts to drive on autopilot using computer vision

Usage:
    manage.py (drive) [--js] [--log=INFO] [--camera=(single|stereo)] [--myconfig=<filename>]


Options:
    -h --help          Show this screen.
    --js               Use physical joystick.
    --myconfig=filename     Specify myconfig file to use.
                            [default: myconfig.py]
"""
import logging
import os

from docopt import docopt
from simple_pid import PID

import donkeycar as dk
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.templates.complete import add_odometry, add_camera, \
    add_user_controller, add_drivetrain, add_simulator, add_imu, DriveMode, \
    UserPilotCondition, ToggleRecording
from donkeycar.parts.logger import LoggerPart
from donkeycar.parts.transform import Lambda
from donkeycar.parts.explode import ExplodeDict
from donkeycar.parts.controller import JoystickController
from donkeycar.parts.web_controller.web import LocalWebController

from dashboard import attach_dashboard, DashboardTelemetry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def drive(cfg, use_joystick=False, camera_type='single', meta=[]):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    # Use a separate data folder from manage.py's tub -- this script records
    # a different set of inputs (steering/throttle vs. user/angle/user/throttle/
    # user/mode), and reusing the same folder trips the tub schema-mismatch
    # assertion in datastore_v2.py.
    cfg.DATA_PATH = os.path.join(cfg.CAR_PATH, 'data_line')

    #Initialize car
    V = dk.vehicle.Vehicle()

    #
    # if we are using the simulator, set it up
    #
    add_simulator(V, cfg)

    #
    # setup primary camera
    #
    add_camera(V, cfg, camera_type)

    #
    # add the user input controller(s)
    # - this will add the web controller
    # - it will optionally add any configured 'joystick' controller
    #
    has_input_controller = hasattr(cfg, "CONTROLLER_TYPE") and cfg.CONTROLLER_TYPE != "mock"
    ctr = add_user_controller(V, cfg, use_joystick, input_image = 'ui/image_array')

    #
    # explode the web buttons into their own key/values in memory
    #
    V.add(ExplodeDict(V.mem, "web/"), inputs=['web/buttons'])

    #
    # track user vs autopilot condition
    #
    V.add(UserPilotCondition(show_pilot_image=getattr(cfg, 'OVERLAY_IMAGE', False)),
          inputs=['user/mode', "cam/image_array", "cv/image_array"],
          outputs=['run_user', "run_pilot", "ui/image_array"])

    #
    # PID controller to be used with cv_controller
    #
    pid = PID(Kp=cfg.PID_P, Ki=cfg.PID_I, Kd=cfg.PID_D)
    def dec_pid_d():
        pid.Kd -= cfg.PID_D_DELTA
        logging.info("pid: d- %f" % pid.Kd)

    def inc_pid_d():
        pid.Kd += cfg.PID_D_DELTA
        logging.info("pid: d+ %f" % pid.Kd)

    def dec_pid_p():
        pid.Kp -= cfg.PID_P_DELTA
        logging.info("pid: p- %f" % pid.Kp)

    def inc_pid_p():
        pid.Kp += cfg.PID_P_DELTA
        logging.info("pid: p+ %f" % pid.Kp)

    #
    # Computer Vision Controller
    #
    line_follower, obstacle_commander = add_cv_controller(
                      V, cfg, pid,
                      cfg.CV_CONTROLLER_MODULE,
                      cfg.CV_CONTROLLER_CLASS,
                      cfg.CV_CONTROLLER_INPUTS,
                      cfg.CV_CONTROLLER_OUTPUTS,
                      cfg.CV_CONTROLLER_CONDITION)

    #
    # Custom web dashboard (dashboard.py / dashboard.html): live
    # lane_offset / curve_gain tuning + start/stop, served at "/" on the
    # same web controller port. add_user_controller returns the joystick
    # when one is configured, so find the web controller from V.parts.
    #
    web_ctr = ctr if isinstance(ctr, LocalWebController) else next(
        (e['part'] for e in V.parts if isinstance(e['part'], LocalWebController)),
        None)
    if web_ctr is not None:
        attach_dashboard(web_ctr, line_follower, obstacle_commander, cfg)

    recording_control = ToggleRecording(cfg.AUTO_RECORD_ON_THROTTLE, cfg.RECORD_DURING_AI)
    V.add(recording_control, inputs=['user/mode', "recording"], outputs=["recording"])


    #
    # Add buttons for handling various user actions
    # The button names are in configuration.
    # They may refer to game controller (joystick) buttons OR web ui buttons
    #
    # There are 5 programmable webui buttons, "web/w1" to "web/w5"
    # adding a button handler for a webui button
    # is just adding a part with a run_condition set to
    # the button's name, so it runs when button is pressed.
    #
    have_joystick = ctr is not None and isinstance(ctr, JoystickController)

    # button to toggle recording
    if cfg.TOGGLE_RECORDING_BTN:
        print(f"Toggle recording button is {cfg.TOGGLE_RECORDING_BTN}")
        if cfg.TOGGLE_RECORDING_BTN.startswith("web/w"):
            V.add(Lambda(lambda: recording_control.toggle_recording()), run_condition=cfg.TOGGLE_RECORDING_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.TOGGLE_RECORDING_BTN, recording_control.toggle_recording)

    # Buttons to tune PID constants
    if cfg.DEC_PID_P_BTN and cfg.PID_P_DELTA:
        print(f"Decrement PID P button is {cfg.DEC_PID_P_BTN}")
        if cfg.DEC_PID_P_BTN.startswith("web/w"):
            V.add(Lambda(lambda: dec_pid_p()), run_condition=cfg.DEC_PID_P_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.DEC_PID_P_BTN, dec_pid_p)
    if cfg.INC_PID_P_BTN and cfg.PID_P_DELTA:
        print(f"Increment PID P button is {cfg.INC_PID_P_BTN}")
        if cfg.INC_PID_P_BTN.startswith("web/w"):
            V.add(Lambda(lambda: inc_pid_p()), run_condition=cfg.INC_PID_P_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.INC_PID_P_BTN, inc_pid_p)
    if cfg.DEC_PID_D_BTN and cfg.PID_D_DELTA:
        print(f"Decrement PID D button is {cfg.DEC_PID_D_BTN}")
        if cfg.DEC_PID_D_BTN.startswith("web/w"):
            V.add(Lambda(lambda: dec_pid_d()), run_condition=cfg.DEC_PID_D_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.DEC_PID_D_BTN, dec_pid_d)
    if cfg.INC_PID_D_BTN and cfg.PID_D_DELTA:
        print(f"Increment PID D button is {cfg.INC_PID_D_BTN}")
        if cfg.INC_PID_D_BTN.startswith("web/w"):
            V.add(Lambda(lambda: inc_pid_d()), run_condition=cfg.INC_PID_D_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.INC_PID_D_BTN, inc_pid_d)

    #
    # Decide what inputs should change the car's steering and throttle
    # based on the choice of user or autopilot drive mode
    #
    V.add(DriveMode(cfg.AI_THROTTLE_MULT),
          inputs=['user/mode', 'user/steering', 'user/throttle',
                  'pilot/steering', 'pilot/throttle'],
          outputs=['steering', 'throttle'])

    #
    # push live telemetry to dashboard websocket clients. Added AFTER
    # DriveMode so 'steering'/'throttle' are this tick's final values;
    # obstacle keys read as None until the pilot first runs.
    #
    if web_ctr is not None:
        V.add(DashboardTelemetry(web_ctr, line_follower),
              inputs=['user/mode', 'steering', 'throttle', 'recording',
                      'obstacle/info', 'obstacle/plan'])


    #
    # Setup drivetrain
    #
    add_drivetrain(V, cfg)


    #
    # OLED display setup
    #
    if cfg.USE_SSD1306_128_32:
        from donkeycar.parts.oled import OLEDPart
        auto_record_on_throttle = cfg.USE_JOYSTICK_AS_DEFAULT and cfg.AUTO_RECORD_ON_THROTTLE
        oled_part = OLEDPart(cfg.SSD1306_128_32_I2C_ROTATION, cfg.SSD1306_RESOLUTION, auto_record_on_throttle)
        V.add(oled_part, inputs=['recording', 'tub/num_records', 'user/mode'], outputs=[], threaded=True)


    #
    # add tub to save data
    #
    inputs=['cam/image_array',
            'steering', 'throttle']

    types=['image_array',
           'float', 'float']

    #
    # Create data storage part
    #
    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    meta += getattr(cfg, 'METADATA', [])
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    #
    # Per-frame line-follower telemetry, written ALONGSIDE the tub (the
    # tub's schema is deliberately left alone -- see lf_telemetry.py for
    # why this exists and what it costs). Same run_condition as the tub so
    # the two line up frame for frame; join on `idx`.
    #
    if line_follower is not None and getattr(cfg, 'LF_TELEMETRY', True):
        from lf_telemetry import LineTelemetry
        import time as _time
        V.add(LineTelemetry(
                  line_follower,
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'logs',
                               _time.strftime('lf_%Y%m%d_%H%M%S.jsonl')),
                  commander=obstacle_commander),
              inputs=['steering', 'throttle', 'tub/num_records',
                      'user/mode', 'obstacle/info', 'obstacle/plan'],
              outputs=[], run_condition='recording')

    if cfg.DONKEY_GYM:
        print("You can now go to http://localhost:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    else:
        print("You can now go to <your hostname.local>:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    if has_input_controller:
        print("You can now move your controller to drive your car.")
        if isinstance(ctr, JoystickController):
            ctr.set_tub(tub_writer.tub)
            ctr.print_controls()

    #
    # run the vehicle
    #
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)


#
# Computer Vision Controller
#
def add_cv_controller(
        V, cfg, pid,
        module_name="donkeycar.parts.line_follower",
        class_name="LineFollower",
        inputs=['cam/image_array'],
        outputs=['pilot/steering', 'pilot/throttle', 'cv/image_array'],
        run_condition="run_pilot"):

        # __import__ the module
        module = __import__(module_name)

        # walk module path to get to module with class
        for attr in module_name.split('.')[1:]:
            module = getattr(module, attr)

        my_class = getattr(module, class_name)
        instance = my_class(pid, cfg)

        #
        # Obstacle maneuvering (Mission 3), wired AROUND the CV
        # controller rather than into it -- line_following.py stays
        # untouched. LaneOffsetCommander runs BEFORE instance so its
        # writes to instance.lane_offset/.curve_gain land before this
        # frame's run(); ThrottleLimiter/ObstacleOverlay run AFTER, once
        # pilot/throttle and cv/image_array actually exist to adjust.
        # Gated on hasattr(instance, 'lane_offset') so this stays a
        # no-op for any CV_CONTROLLER_CLASS that isn't LineFollower.
        #
        obstacle_on = (getattr(cfg, 'OBSTACLE_AVOIDANCE_ENABLED', False)
                       and hasattr(instance, 'lane_offset'))
        commander = None
        if obstacle_on:
            from obstacle_detector import ObstacleDetector
            from obstacle_avoidance import (
                LaneOffsetCommander, ThrottleLimiter, ObstacleOverlay)

            # RGB only -- see obstacle_detector.py's module docstring for
            # why depth was dropped (repeated power-budget crashes).
            V.add(ObstacleDetector(cfg, line_follower=instance),
                  inputs=['cam/image_array'],
                  outputs=['obstacle/info'],
                  run_condition=run_condition)
            commander = LaneOffsetCommander(instance, cfg)
            V.add(commander,
                  inputs=['obstacle/info'],
                  outputs=['obstacle/plan'],
                  run_condition=run_condition)

        # add instance of class to vehicle
        V.add(instance,
              inputs=inputs,
              outputs=outputs,
              run_condition=run_condition)

        if obstacle_on:
            V.add(ThrottleLimiter(),
                  inputs=['pilot/throttle', 'obstacle/plan'],
                  outputs=['pilot/throttle'],
                  run_condition=run_condition)
            V.add(ObstacleOverlay(cfg),
                  inputs=['cv/image_array', 'obstacle/info', 'obstacle/plan'],
                  outputs=['cv/image_array'],
                  run_condition=run_condition)

        return instance, commander


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    log_level = args['--log'] or "INFO"
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=numeric_level)

    if args['drive']:
        drive(cfg, use_joystick=args['--js'], camera_type=args['--camera'])