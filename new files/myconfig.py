"""
myconfig.py — setup for manage_line.py + LineFollowerMultiBand
(donkeycar.parts.line_follower_multiband), reset to madhav's tuning.
"""

# ── Computer Vision Controller ───────────────────────────────────────
CV_CONTROLLER_MODULE = "line_following"
CV_CONTROLLER_CLASS = "LineFollower"
CV_CONTROLLER_INPUTS = ['cam/image_array']
CV_CONTROLLER_OUTPUTS = ['pilot/steering', 'pilot/throttle', 'cv/image_array']
CV_CONTROLLER_CONDITION = "run_pilot"

# required by manage_line.py (builds a PID our follower ignores)
PID_P = -0.01
PID_I = 0.0
PID_D = -0.0001
PID_P_DELTA = 0.005
PID_D_DELTA = 0.00005
INC_PID_P_BTN = None
DEC_PID_P_BTN = None
INC_PID_D_BTN = None
DEC_PID_D_BTN = None

OVERLAY_IMAGE = True
# "option" is a PS4 DualShock button name and doesn't exist on the F710
# (CONTROLLER_TYPE below) -- LogitechJoystick's real button names are
# back/start/Logitech/A/B/X/Y/L1/R1/stick-presses (controller.py:719-731), so
# that binding was a dead entry. "B" is also the F710's built-in default
# recording-toggle button (LogitechJoystickController.init_trigger_maps()),
# so this just documents the button that already works.
TOGGLE_RECORDING_BTN = "B"

# Without this, ToggleRecording forces recording off whenever user/mode != 'user',
# so no frames are ever saved while the LineFollower is actually driving (mode
# 'local_angle'/'local_pilot') -- exactly the data needed to diagnose failures.
RECORD_DURING_AI = True

# ── LineFollower ─────────────────────────────────────────────────────
# Reset to madhav's tuning (his run performed better than our recalibrated
# sat-gate-era values below, which mostly coasted-then-stopped). Starting
# fresh from his numbers; re-tune from here based on new test footage.
#
# CALIBRATE ONCE after any camera change:
#   park the car dead-centered on the line pointing straight, read err
#   from the overlay, then set: LF_TARGET_X = 192 + err*192
#   (192 = center of the 384-wide processing image; the part downscales the
#   1080p camera frames to LF_PROC_WIDTH=384 internally, so target/overlay
#   coordinates are in that space). Commented out = use center.
# LF_TARGET_X = 192

# The overlay draws the ROI as a yellow horizontal line; it should sit just
# above the visible track. Adjust if the camera angle changes.
LF_ROI_TOP = 0.50

# Steering feel:
#   shakes fast left-right  -> lower LF_STEER_KP
#   weaves slowly on straight -> raise LF_ERR_DEADBAND (a deadband trades
#     centering accuracy for calm; keep it 0 to sit right on the line)
#   corners too wide         -> raise LF_STEER_KP
LF_STEER_KP = 2.0
LF_ERR_DEADBAND = 0.0  # was 0.08, but earlier line_following.py never
                       # implemented it; it does now, so 0.08 would add
                       # ~15px of ignored offset. 0 = tightest centering.

# Throttle (VESC needed ~0.30 to overcome friction):
LF_THROTTLE_MAX = 0.15
LF_THROTTLE_MIN = 0.10
LF_THROTTLE_LOST = LF_THROTTLE_MIN  # was 0.26 — HIGHER than LF_THROTTLE_MAX, so the car
                         # sped UP whenever it lost the line. Coasting blind
                         # should be the slowest state, not the fastest.

# Ride position: 0.0 = directly on top of the yellow line. Later, to drive
# in a lane instead: -1.0 aims at the white line left of the yellow, +1.0
# right of it, +/-0.5 = middle of that lane (uses the yellow-to-white
# distances the follower learns from the solid white boundary lines).
LF_LANE_OFFSET = 0.0

# ── Image ────────────────────────────────────────────────────────────
TRANSFORMATIONS = ['RESIZE']
RESIZE_WIDTH = 160
RESIZE_HEIGHT = 120

# ── Drive Loop ───────────────────────────────────────────────────────
DRIVE_LOOP_HZ = 20
MAX_LOOPS = None

# ── Camera ───────────────────────────────────────────────────────────
CAMERA_TYPE = "OAKD"
IMAGE_W = 384
IMAGE_H = 216
IMAGE_DEPTH = 3
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAMERA_INDEX = 0

# ── Drivetrain ───────────────────────────────────────────────────────
DRIVE_TRAIN_TYPE = "VESC"
VESC_MAX_SPEED_PERCENT = .35
VESC_SERIAL_PORT = "/dev/ttyACM0"
VESC_HAS_SENSOR = True
VESC_START_HEARTBEAT = True
VESC_BAUDRATE = 115200
VESC_TIMEOUT = 0.05
VESC_STEERING_SCALE = 0.5
# CALIBRATE ONCE: wheels off the ground, User mode, hands off controls;
# adjust by 0.01 until the front wheels point dead straight.
# (car naturally drifts right; 0.5 = no trim)
VESC_STEERING_OFFSET = 0.48

# ── Joystick ─────────────────────────────────────────────────────────
USE_JOYSTICK_AS_DEFAULT = True
JOYSTICK_MAX_THROTTLE = 0.35
JOYSTICK_STEERING_SCALE = 1.0
# False so the "option" button's manual recording toggle actually works in
# Local Pilot mode -- when True, JoystickController.on_throttle_changes() only
# ever sets recording=True while mode=='user', and toggle_manual_recording()
# becomes a no-op, so autopilot-mode frames can never be captured either way.
AUTO_RECORD_ON_THROTTLE = False
CONTROLLER_TYPE = 'F710'
USE_NETWORKED_JS = False
NETWORK_JS_SERVER_IP = None
JOYSTICK_DEADZONE = 0.01
JOYSTICK_THROTTLE_DIR = -1.0
JOYSTICK_DEVICE_FILE = "/dev/input/js0"

# ── Oak-D ────────────────────────────────────────────────────────────
OAKD_RGB = True
OAKD_DEPTH = False
OAKD_ID = None

# ── FPV ──────────────────────────────────────────────────────────────
USE_FPV = True
