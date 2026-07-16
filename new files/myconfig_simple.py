"""
myconfig.py — minimal setup for manage_line.py + the simple LineFollower.
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
TOGGLE_RECORDING_BTN = "option"

# ── LineFollower ─────────────────────────────────────────────────────
# CALIBRATE ONCE after any camera change:
#   park the car dead-centered on the line pointing straight, read err
#   from the overlay, then set: LF_TARGET_X = 80 + err*80
#   (80 = center of the 160-wide image). Commented out = use center.
# LF_TARGET_X = 80

# The overlay draws the ROI as a yellow horizontal line; it should sit just
# above the visible track. Adjust if the camera angle changes.
LF_ROI_TOP = 0.50

# Steering feel:
#   shakes fast left-right  -> lower LF_STEER_KP
#   weaves slowly on straight -> raise LF_ERR_DEADBAND
#   corners too wide         -> raise LF_STEER_KP
# LF_STEER_KP = 2.0
# LF_ERR_DEADBAND = 0.08

# Throttle (VESC needed ~0.30 to overcome friction):
# LF_THROTTLE_MAX = 0.38
# LF_THROTTLE_MIN = 0.30
# LF_THROTTLE_LOST = 0.26

# ── Image ────────────────────────────────────────────────────────────
TRANSFORMATIONS = ['RESIZE']
RESIZE_WIDTH = 160
RESIZE_HEIGHT = 120

# ── Drive Loop ───────────────────────────────────────────────────────
DRIVE_LOOP_HZ = 20
MAX_LOOPS = None

# ── Camera ───────────────────────────────────────────────────────────
CAMERA_TYPE = "OAKD"
IMAGE_W = 160
IMAGE_H = 120
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
AUTO_RECORD_ON_THROTTLE = True
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
