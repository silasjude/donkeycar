"""
myconfig.py — overrides for manage_line.py (cv_control template)
Line following via line_following.py (from-scratch CV follower).
"""

# ── Computer Vision Controller ───────────────────────────────────────
CV_CONTROLLER_MODULE = "line_following"
CV_CONTROLLER_CLASS = "LineFollower"
CV_CONTROLLER_INPUTS = ['cam/image_array']
CV_CONTROLLER_OUTPUTS = ['pilot/steering', 'pilot/throttle', 'cv/image_array']
CV_CONTROLLER_CONDITION = "run_pilot"

# manage_line.py constructs a simple_pid PID from these keys, so they must
# exist — but our LineFollower ignores that PID (it has its own PD loop).
PID_P = -0.01
PID_I = 0.0
PID_D = -0.0001
PID_P_DELTA = 0.005
PID_D_DELTA = 0.00005
INC_PID_P_BTN = None
DEC_PID_P_BTN = None
INC_PID_D_BTN = None
DEC_PID_D_BTN = None

OVERLAY_IMAGE = True            # detection overlay + status bar in web UI
TOGGLE_RECORDING_BTN = "option"

# ── LineFollower: steering (tuned to hold straights steady) ─────────
LF_STEER_KP = 2.0        # main gain. Wobbles left-right fast -> lower.
                         # Corners too wide / late -> raise.
LF_STEER_KD = 0.35       # damping. Small persistent wobble -> raise to 0.5.
LF_HEADING_GAIN = 0.7    # lookahead from line slope. Helps corners; too high
                         # causes weaving on straights.
LF_ERR_DEADBAND = 0.08   # "close enough = go straight" zone.
LF_STEER_SMOOTH = 0.5    # steering low-pass. Lower = smoother but lazier.
                         # Don't go below ~0.3 or hairpins get missed.

# Where the line should sit in the frame when driving dead-center (pixels;
# 80 = center of the 160-wide image). Calibrate: place the car perfectly on
# the line pointing straight, read err from the overlay, then set
# LF_TARGET_X = 80 + err*80 (e.g. err:+0.11 -> 89). Recheck after any
# camera remount.
LF_TARGET_X = 89

# ── LineFollower: throttle ───────────────────────────────────────────
# Scaled by VESC_MAX_SPEED_PERCENT (0.35): wheel duty = value * 0.35.
# Values below ~0.28 didn't overcome static friction on this car.
LF_THROTTLE_MAX = 0.38   # straights
LF_THROTTLE_MIN = 0.30   # hard corrections
LF_THROTTLE_LOST = 0.26  # while searching for a lost line

# ── LineFollower: detection / startup ────────────────────────────────
LF_MIN_BAND_PIXELS = 6   # sized for the 160x120 camera
LF_ACQUIRE_FRAMES = 10   # stable-lock frames required before moving (~0.5 s)

# ROI top: fraction of image height ignored. 0.62 was tuned for the ORIGINAL
# HIGH camera mount; the camera has been lowered since, so more of the frame
# is track now. Check the yellow horizontal line in the overlay: it should
# sit just above where track meets background. Raise/lower to match.
LF_ROI_TOP = 0.50

# Validation gates, relaxed for the lowered camera until recalibrated.
# If it ever chases debris again, tighten (raise MIN_MASK_FRAC, lower the
# other two). If yellow detection drops out on the line, relax further.
LF_MIN_MASK_FRAC = 0.0012
LF_MAX_FIT_RMS = 0.08
LF_MAX_JUMP = 0.35

# White-line fallback: DISABLED for now (0 = yellow-only + coast/search/stop).
# The video showed it fabricating lanes from pale concrete patches when the
# car was off-track. Get yellow-only laps working first; then re-enable with
# 0.8-1.5 to bridge dash gaps and hairpins.
LF_WHITE_MAX_SEC = 0

# LAB color thresholds (calibrated day/shadow/night). If dashes don't tint
# green in the overlay, lower LF_LAB_B_MIN toward 140.
# LF_LAB_B_MIN = 143
# LF_LAB_A_MAX = 142
# LF_LAB_L_MIN = 40

# ── Image Transformations ────────────────────────────────────────────
TRANSFORMATIONS = ['RESIZE']
RESIZE_WIDTH = 160
RESIZE_HEIGHT = 120

# ── Drive Loop Timing ────────────────────────────────────────────────
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
# Steering trim for the natural rightward drift. 0.5 = no trim. Test in
# MANUAL mode, hands off the stick, rolling forward: adjust by 0.01 steps
# until it tracks straight. If 0.48 makes the right-drift WORSE, go the
# other way (0.52).
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

# ── Oak-D Specific ───────────────────────────────────────────────────
OAKD_RGB = True
OAKD_DEPTH = False
OAKD_ID = None

# ── FPV Web Stream ───────────────────────────────────────────────────
USE_FPV = True
