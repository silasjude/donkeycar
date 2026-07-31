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
# 2026-07-28: camera remounted this morning (all sessions since ~13:40 are
# the new view). Measured from today's footage: the lane lines converge at
# y ~117-127 of 216 (vp ~0.55-0.58), and everything above ~0.56 is
# buildings/kiosks/people — with the old 0.50 the ROI top band was full of
# wall bases and signage that the white detector locked onto (confirmed on
# idx 140484/140580 overlays). 0.56 keeps a small margin above the track.
LF_ROI_TOP = 0.56
# Vanishing-point row for the heading convergence correction, same
# measurement as above (defaults to 0.6 in line_following.py).
LF_VP_Y_FRAC = 0.56

# Steering feel — retuned 2026-07-24 (second pass) after system-identifying
# the loop from session-21 footage:
#   * total steering->camera loop delay measured at 200-250ms;
#   * the image-x error mixes true offset (233px/m) with camera yaw
#     (279px/rad) — in a weave the controller mostly chases its own yaw;
#   * the "heading" slope measurement carries ~no yaw (it's redundant
#     position + curve lean), so nothing in the old loop actually damped —
#     hence the ~1Hz jerky swerve.
# The damper is now the vision gyro (LF_GYRO_GAIN in line_following.py):
# yaw rate measured by phase-correlating the far background between
# frames, washed out over 1s so held arcs aren't resisted. The integrator
# was already demoted to straight-line trim duty (gated off in curves).
# Tuning hints:
#   shakes fast left-right   -> lower LF_STEER_KP, then lower LF_STEER_SLEW
#   weaves slowly on straight -> raise LF_GYRO_GAIN toward 0.45
#   corners too wide          -> raise LF_HEADING_GAIN toward 0.8
#   rides beside line/lane center persistently -> re-measure LF_HEADING_BIAS
#     (park on the line pointing straight: python line_following.py
#     calibrate <frame>), then let the integrator trim the rest
LF_STEER_KP = 1.2      # identified-plant optimum (1.6 fed the 1Hz jerk)
LF_ERR_DEADBAND = 0.0  # 0 = tightest centering; the swerve fix is the
                       # filtering, not a deadband
# Camera-yaw slope bias. Re-measured 2026-07-24 by regressing the heading
# readout against the true yaw extracted from session-21 footage: the
# heading baseline at zero error/zero yaw is +0.19, not the 0.06 the
# session-12-15 estimate gave (that fit was contaminated by the closed-loop
# weave). The shortfall (0.13 x heading gain) was a constant steer-right
# push — why the car rode the RIGHT side of the line at offset 0.
# Re-measure after any camera change with:
#   python line_following.py calibrate <frame>
# (park centered on the line, pointing straight along it)
# 2026-07-28: the 0.19 above was measured on the OLD camera mount. On the
# new mount, the median raw heading across 222 near-center tracked frames
# of today's footage is -0.005 — i.e. no measurable yaw bias. Leaving 0.19
# in place was a constant steer push (one of the reasons the car "went off
# to the side" today). Re-run the parked calibration when convenient to
# confirm; expect a value near 0.
LF_HEADING_BIAS = 0.0

# Throttle (VESC needed ~0.30 to overcome friction):
LF_THROTTLE_MAX = 0.3
LF_THROTTLE_MIN = 0.15
LF_THROTTLE_LOST = LF_THROTTLE_MIN  # was 0.26 — HIGHER than LF_THROTTLE_MAX, so the car
                         # sped UP whenever it lost the line. Coasting blind
                         # should be the slowest state, not the fastest.

# Lost-line recovery crawl (added 2026-07-29, user request). Session
# 26-07-29_86's final stop (idx 163381) parked the car for good with the
# real white boundary line detected and drawn in the overlay on nearly
# every frame of the preceding coast — the white-USE gates (side split /
# jump gates, judged against a stale yellow reference) rejected it, the
# 2s LF_LOST_STOP_SEC clock expired, and a stopped car's view never
# changes, so it sat there for the remaining ~50s of the recording.
# After the coast grace period, IF the white detector is still returning
# centroids, keep crawling with the SAME held steering at
# LF_THROTTLE_LOST for up to this many extra seconds before the hard
# stop — the scene keeps evolving, so the yellow/white acceptance gates
# get fresh looks. No whites in sight → stops exactly as before (a blind
# crawl just traces a circle).
LF_LOST_RECOVERY_SEC = 3.0
LF_LOST_RECOVERY_WHITE_AGE = 1.0  # whites count as "in sight" this long

# Ride position: 0.0 = directly on top of the yellow line; -1.0 aims at the
# white line left of the yellow, +1.0 right of it, +/-0.5 = middle of that
# lane (uses the yellow-to-white distances the follower learns from the
# solid white boundary lines, and ramps in over LF_LANE_RAMP_SEC once
# they're learned).
#
# -0.5 = CENTER OF THE LEFT LANE. The fractions mean what they say now:
# the lane width is pinned by LF_LANE_WIDTH_PX below (live-learned widths
# were both ~30% short — measured mid-ROI where perspective narrows the
# lane — AND ±20px run-to-run; that's why ±0.75 still hugged the dashed
# line), widths are no longer trusted from turn-distorted samples (those
# dragged the target into the other lane mid-corner), and the
# vanishing-point heading fix stops the offset from decaying. If ±0.5 sits
# visibly off lane-center now, adjust LF_LANE_WIDTH_PX by ±10, not the
# offset fraction.
#
# THIS LINE IS THE ONE THAT COUNTS: myconfig.py overrides the DEFAULTS in
# line_following.py, so editing the value there does nothing while this one
# exists — that's why the 2026-07-23 "different offsets" runs all behaved
# the same (they all actually drove 0.0). The follower logs its effective
# value at startup; check that line when in doubt.
#
# PAIR THE OFFSET WITH LF_CURVE_GAIN (field-tuned 2026-07-24, confirmed
# on track; set in line_following.py DEFAULTS unless defined here):
#    0.0 (center line) -> LF_CURVE_GAIN = -0.8
#   -0.75 (left lane)  -> LF_CURVE_GAIN = -0.8
#   +0.5 (right lane)  -> LF_CURVE_GAIN = -0.2
#
# Set back to 0.0 (ride the yellow dash itself): the -0.75 left-lane-offset
# experiment was steering the car away from the yellow dash on purpose, which
# is why it kept riding the left side and losing the dash off-frame in turns
# (working as configured, just not what we want right now).
#
# 2026-07-28 (new camera): -0.75 is now GEOMETRICALLY OFF-FRAME — the new
# mount roughly doubled the bottom-row pixel scale (lane width ~255px
# measured vs the old 170), so -0.75 puts the yellow's bottom-row target
# at x ~383+ of a 384px image. The replay of tonight's 17:57 session shows
# the controller pinned at err=-1.0 / full-left for the entire run because
# of this. -0.5 (mid left lane) puts the target at x~320, same in-frame
# margin the old -0.75 had. Drop to 0.0 if you want the most robust ride
# while re-tuning.
# 2026-07-29 EVENING -- MEASURED REASON TO DO EXACTLY THAT, still -0.5
# here because which lane the car rides is a driving decision, not a bug
# fix. Session 26-07-29_87 crashed at the planter-corner right-hander on
# every lap, and the corner approach (idx 172745-172908) replays
# faithfully (corr recorded-vs-sim +0.93), so the trace is trustworthy:
# through that bend the yellow sits at x_f p50 = 350 while this offset's
# target is 192 + 0.5*255 = 319.5, so the position error is ~NULLED
# exactly where the car has to turn. Mean commanded steering over those
# 164 frames is +0.04 -- dead straight into the planter, which is the
# "it just goes straight" the user reported. The SAME frames at
# LF_LANE_OFFSET = 0.0 command mean +0.77 with 85% of frames hard-right
# (i.e. it takes the corner) and lose no yellow tracking (104 frames
# either way). +0.5 does turn (mean +0.46) but loses the line: yellow
# tracking drops 104 -> 36 frames as x_f falls to p50 151. Secondary
# term at the same spot: h_f reads -0.35..-0.88 (LEFT lean) through a
# RIGHT-hander at 172799-172829, actively pulling the wrong way.
#
# 2026-07-30 -- NOW +0.4 / -0.2, and this is no longer inference. Every
# good drive since the follower was fixed ran +0.4 with curve gain -0.2,
# set live from the dashboard, and lf_telemetry.py RECORDED it rather
# than it having to be reconstructed by replay sweep: session
# 26-07-30_91 logged off=+0.4 / cg=-0.2 constant across all 3230 frames
# (87% raw detection, 84% tracking after the heading fix), and _92 the
# same through the shadow-exit work the user then confirmed on track.
# The old -0.5/-0.8 pair above was a leftover that no successful run
# used; leaving it here meant the car STARTED in a configuration that
# has never worked and only reached a good one once two sliders were
# moved by hand -- which matters more now that obstacle avoidance is
# back on, since LaneOffsetCommander seeds its return-to base_offset /
# base_curve_gain from THESE values at construction.
# +0.4 keeps the pairing honest: the field-tuned partner for the
# right-lane offsets is -0.2, and dashboard.py's pairing guard treats
# anything within 0.15 of +0.5 as that case, so +0.4/-0.2 raises no
# warning. Changing the offset means changing BOTH lines together.
LF_LANE_OFFSET = 0.4
LF_CURVE_GAIN = -0.2

# 2026-07-29 (session 26-07-29_88): the lane-geometry implausibility gate
# used to reject a yellow fix forever once the car sat on the WRONG SIDE
# of the line -- the detection was there on 36/36 frames of a 1.8s stretch
# and never used, and 12 of that run's 26 re-acquisition attempts died in
# that gate. It now force-accepts after this many consecutive AGREEING
# implausible fixes (see the LF_LANE_IMPLAUSIBLE_MAX_REJECT block in
# line_following.py). Set to 0 to restore the old lock-out behaviour.
LF_LANE_IMPLAUSIBLE_MAX_REJECT = 6

# Yellow-to-white lane width in pixels AT THE FRAME BOTTOM (384-wide
# coords), as (LEFT, RIGHT). Measured 2026-07-24 from the offset-run
# footage: the px scale is NOT symmetric on this camera — from the
# left-lane vantage the left lane spans ~170px at the bottom row, while
# ~140 is right for the right lane (the +0.5 runs sat correctly on 140;
# the -0.5 runs hugged the yellow because 70px is only a third of the
# left lane's real 170px span). A single number is also accepted and
# used for both sides. Pinning these makes the ride position repeatable
# and lets the offset engage immediately even when the car starts
# already inside a lane. Comment out to fall back to live learning.
# 2026-07-28: re-measured for the NEW camera mount from today's footage.
# Two independent measurements agree the old (170, 140) is far too short
# now: (a) manual rowscan of straight frame 141500 extrapolates
# yellow->left-white ~277px / yellow->right-white ~300px at the bottom row;
# (b) replaying the fixed detector over the whole afternoon (idx
# 128000-139700) and collecting every accepted bottom-projected width
# sample gives left n=1459 median=253 (IQR 229-276) and right n=195
# median=194 (IQR 143-242 — sparse, the right boundary was rarely in
# clean view this afternoon). Pinning the well-measured left at its
# median and the right at a compromise of (a) and (b); re-measure the
# right side once there's a run with the right boundary steadily in view.
LF_LANE_WIDTH_PX = (255, 230)

# ── New-camera scale retune (2026-07-28) ────────────────────────────────
# The remount roughly doubled near-field pixel scale, so every constant
# expressed in fractions-of-image-width that encodes a PHYSICAL distance
# needs to roughly double with it. Bounds on which white clusters can be
# the lane boundary (fractions of the 384px width):
LF_LANE_DIST_MIN_FRAC = 0.15       # was 0.08 (31px) — glare within ~55px of
                                   # the yellow is not a boundary now
LF_LANE_DIST_MAX_FRAC = 0.90       # was 0.45 (173px) — the REAL boundary at
                                   # ~280-300px was being rejected as
                                   # "implausible" on every frame
LF_LANE_DIST_MAX_FRAC_LANE = 1.00  # in-turn chord distortion headroom while
                                   # lane-offset driving (was 0.55)
LF_LANE_DIST_BOT_MIN_FRAC = 0.35   # was 0.15 — learned-sample floor
# Frame-to-frame motion gates (fractions of image width): real per-frame
# motion in px also ~doubled.
LF_TEMPORAL_JUMP = 0.14            # was 0.09
LF_SINGLE_DASH_MAX_JUMP = 0.22     # was 0.15
LF_WHITE_TRACK_JUMP = 0.18         # was 0.12
LF_WHITE_JUMP_MAX_FRAC = 0.22      # was 0.15

# ── White detector: light-grey clutter rejection (added 2026-07-29) ──
# Sessions 26-07-29_85/_86: the white mask was lighting up light-GREY
# surfaces off the track — the hexagonal plaza pavers, a lighter repair
# patch mid-pavement (visible dead-centre in the very last frame of _86,
# idx 164383), aluminium wall trim and stone planter walls — and the car
# steered off chasing them the same way it once chased yellow-ish debris
# before LF_REQUIRE_PAVEMENT. Same family of fix, structural and
# illumination-relative (no absolute thresholds — those die at the
# sun/shadow boundary on this track):
LF_WHITE_MIN_ASPECT = 3.2     # min PCA axis ratio of a blob's pixel cloud
                              # (paint is a long thin streak at any angle:
                              # every real boundary stroke in the 07-29
                              # footage measures 4.5-22; paver/patch blobs
                              # read 1-3)
LF_WHITE_SIDE_CONTRAST = 10   # blob centres must out-bright BOTH flanks of
                              # the blob's thin dimension by this many L
                              # counts, sampled at two distances (the
                              # bright border strip of a big grey region
                              # has its parent region on one flank; pavers
                              # have a dark grout joint near but the NEXT
                              # bright paver farther out; paint has darker
                              # pavement at both distances on both flanks).
                              # Flanks sampled off-frame are skipped, so a
                              # line hugging the frame edge keeps its one
                              # judgeable flank.

# Session 26-07-24_44: at the same sharp right-hander session 38 already
# tuned LF_SHARP_BEND_MIN for, the car went straight into the furniture
# past the corner on 3 separate approaches. Replay showed why: the
# blind/white-guided branch only applies the sharp-bend floor once the
# frozen heading reading is past 0.9*LF_HEADING_CLIP (=0.63 here), but
# yellow dropped out early in the bend both times, freezing hterm at
# 0.38 and 0.47 -- never enough to trip the floor, so the car coasted
# through on a weak, non-committal turn command instead of committing to
# the corner. Lowered for the white-guided branch ONLY (see
# LF_WHITE_GUIDE_BEND_FRAC in line_following.py) -- live tracking and
# white-tracking already negotiate this bend fine when they can actually
# see yellow/white, so their 0.9 trigger is untouched. Retune down
# further (toward 0.3) if it still doesn't commit hard enough; back
# toward 0.9 if it starts over-committing on gentler bends elsewhere.
LF_WHITE_GUIDE_BEND_FRAC = 0.5

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
# Stable by-id path: the VESC re-enumerates as ttyACM1/2/... after any USB
# dropout (brownout, knocked cable), which breaks a hardcoded /dev/ttyACM0.
VESC_SERIAL_PORT = "/dev/serial/by-id/usb-STMicroelectronics_ChibiOS_RT_Virtual_COM_Port_304-if00"
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
OAKD_DEPTH = False    # 2026-07-26: staying off for good. Two attempts at
                       # running it (plain, then FPS-capped) both hit
                       # "Undervoltage detected!" (dmesg) the moment the
                       # motor engaged, on top of RGB+depth's own draw --
                       # a real combined power-budget limit, not a charge
                       # or config problem. RGB alone has never once
                       # caused an issue in any test. Obstacle avoidance
                       # is now monocular (see obstacle_detector.py) and
                       # no longer needs this at all -- don't re-enable
                       # unless the OAK-D gets its own dedicated/powered
                       # USB connection separate from the motor rail.
OAKD_ID = None

# ── Obstacle maneuvering (Mission 3) ────────────────────────────────────
# 2026-07-26: rewritten to be monocular -- estimates obstacle distance
# from the RGB image alone (ground-plane geometry: how low in the frame
# an object's base sits), no depth camera involved at all. See
# obstacle_detector.py's module docstring for how this works and its
# known limitations, and obstacle_avoidance.py for the decision layer
# (unchanged from before -- it only ever consumed 'obstacle/info', which
# has the same shape either way).
#
# OBSTACLE_MONO_SLOPE below is a PLACEHOLDER until calibrated -- run
#   python3 obstacle_detector.py calibrate <frame.jpg> <distance_m>
# with a real photo of the test cone at a known distance, then paste in
# the printed value. Presence/side detection work either way; only the
# absolute distances (and therefore exactly when OBSTACLE_SLOW_DIST_M/
# STOP_DIST_M trigger) depend on it being calibrated.
#
# Watch the FPV overlay's "OBS:" line and the console/log for
# "ObstacleAvoidance: ... -> ..." transitions the same way as before --
# a lane change is a bigger, more committed maneuver than a swerve, so
# keep a close eye/hand on the controller for the first few attempts.
# 2026-07-29 EVENING: DISABLED for line-follow testing -- NOT because a
# fault was found in the obstacle code. It confounds line-follow traces:
# during session 26-07-29_87 it fired swerves on 627 of 3225 frames
# (including a 149-frame swerve_right right over the planter corner, idx
# 172135-172283, where a pedestrian crosses frame ~172200), swapping the
# lane offset by +-0.35 with OBSTACLE_MONO_SLOPE still uncalibrated.
# A swerve was FIRST BLAMED for that session's repeated crash at the
# corner; replay disproved it -- the swerve drives the car hard RIGHT
# (|st| 0.95 with the stack vs 0.49 without) while the car actually went
# LEFT into the planter, and the crash reproduces with the stack off. The
# real cause was this file's LF_LANE_OFFSET (see the note there).
# RE-ENABLE after OBSTACLE_MONO_SLOPE is calibrated and/or swerves are
# gated during tracked bends. NOTE: while False the car will NOT stop for
# real obstacles either -- spot the car accordingly.
#
# 2026-07-30: RE-ENABLED at the user's request. It was switched off on
# 07-29 to isolate line-following debugging, and that paid off -- all
# three faults found since were in the follower's own gates (heading
# reported as 0.0 instead of "not measured", and the jump-gate/reacquire
# deadlock), none of them here. The follower now holds the line, so the
# stack goes back on. OBSTACLE_MONO_SLOPE is STILL UNCALIBRATED and the
# swerve-vs-lane-change decision is still open (see HANDOFF.md) -- what
# changed is that lf_telemetry.py now logs `off` (what the follower is
# actually using) alongside `base_off` (the commander's return-to value),
# so how often the commander moves the offset is finally measurable
# instead of being inferred.
OBSTACLE_AVOIDANCE_ENABLED = True

# monocular ranging -- see obstacle_detector.py's DEFAULTS for the full
# list and rationale.
OBSTACLE_HORIZON_Y_FRAC = 0.6   # matches line_following.py's own default
                                 # for LF_VP_Y_FRAC (not itself set in
                                 # this file, so it's using that same 0.6)
OBSTACLE_MONO_SLOPE = 0.04      # PLACEHOLDER -- see note above
# 2026-07-27: raised from 2.2. Time-accurate replay of a real crash (using
# the session's actual recorded timestamps, not just frame order) showed
# distance readings jump almost straight from "not present" to point-
# blank -- there was ONE clean mid-range reading (4.63m) during the whole
# approach, and it didn't count as "present" because it exceeded the old
# 2.2m threshold, so the car got essentially no lead time to execute the
# lane change before the next valid reading was already point-blank. This
# is a real limitation of the still-uncalibrated OBSTACLE_MONO_SLOPE (see
# above) compressing most of the useful warning range into a narrow band
# right near the bottom of frame. Raising the trigger radius (not the
# distance model itself) makes the system act on whatever early readings
# it does get instead of waiting for a threshold the model rarely reports
# in time. Lower this back down once OBSTACLE_MONO_SLOPE is genuinely
# calibrated and mid-range readings become reliable instead of sparse.
OBSTACLE_SLOW_DIST_M = 5.0      # start reacting (slow, and swerve/plan)
OBSTACLE_STOP_DIST_M = 0.6      # throttle floor reached here
OBSTACLE_MIN_BLOB_AREA_FRAC = 0.003  # raise if false positives show up on
                               # replay (stray color/lighting noise on
                               # the track surface); lower if a real
                               # cone is ever missed because its blob is
                               # too small
# 2026-07-26: added after the first real test drive found a permanent
# painted ground marking (unrelated to the cone) that was colorful
# enough to pass the chroma gate and kept flickering in and out of the
# detection corridor as a false "obstacle" -- its fragments were only
# 1-6px tall vs. the real cone's 37px in the same frame. This rejects
# anything too flat to be a real solid object, regardless of area.
OBSTACLE_MIN_BLOB_HEIGHT_PX = 15
# 2026-07-27: the height gate above didn't fully catch it -- a flat
# marking viewed at a close, grazing angle can still look tall in the
# image. This track also has a permanent BLUE ground marking; sampled
# directly from recorded frames its HSV hue clusters at 108-111. Excluded
# outright, same as the yellow dash and white lines are.
OBSTACLE_EXCLUDE_BLUE_HUE_MIN = 95
OBSTACLE_EXCLUDE_BLUE_HUE_MAX = 130
# 2026-07-27 (late evening): per-side corridor band widths. Found from
# the 17:41 real drive (idx 111872-111906): a full-size cone dead in the
# car's own path was RANGED monotonically from 17.9m all the way in, but
# `present` stayed False until 0.77m because the corridor's old
# SYMMETRIC half-width -- mean(LF_LANE_WIDTH_PX)*0.55 = 85px -- is
# narrower than the car's own lane. At +0.5 offset the own lane spans
# the full 140px (lane_width_r) from the yellow line to the white
# boundary, and this cone's centroid sat 10-60px outside the 85px edge
# for the entire approach. Each band now spans its own side's full lane
# width (left band x lane_width_l=170, right band x lane_width_r=140,
# scaled by this knob). At 1.0, replay shows that cone detected at
# 4.63m -> early lane change instead of a 0.77m last-moment swerve.
# Lower toward ~0.6 only if trackside objects (people/bins just inside
# the lane boundaries) start triggering spurious maneuvers.
OBSTACLE_CORRIDOR_LANE_FRAC = 1.0
# 2026-07-27 (evening): bootstrap corridor. On every real recorded
# session start the detection corridor (centered on LineFollower's x_f)
# was blind for the first 1-3 frames until the follower's first accepted
# fix -- replay of idx 108545/107772/107001 showed the first cone's blob
# fully qualifying on frame 1 (cx=22-72px, 0.75-0.85m) while the
# un-seeded corridor sat at frame center 192±85 and missed it; at
# 108545 the cone was at cx=22 with x_f later converging to -21, which
# no centered corridor of any width covers. Until the follower produces
# its first-ever lateral fix the detector now scans this fraction of the
# frame width to each side of center and reports present/distance with
# side=None (status 'obstacle_unlocalized'); obstacle_avoidance.py's
# never-guess fail-safe holds lane and slows on that, so an early
# detection can never commit a maneuver in a guessed direction. 0.5 =
# full frame. See obstacle_detector.py's DEFAULTS comment for the full
# evidence.
OBSTACLE_BOOTSTRAP_HALF_WIDTH_FRAC = 0.5

# decision layer -- see obstacle_avoidance.py's DEFAULTS
OBSTACLE_SWERVE_ENABLED = True   # in-lane steer around a single-side
                                  # obstacle; never crosses the yellow
OBSTACLE_SWERVE_OFFSET = 0.35
OBSTACLE_LANE_CHANGE_ENABLED = True  # 2026-07-27: swerve/stop confirmed
                                       # solid on real footage (car stopped
                                       # patiently in front of the cone
                                       # after the blue-tape false positive
                                       # was fixed) -- enabling the full
                                       # lane change now. An obstacle in
                                       # the car's own lane checks the
                                       # left lane is clear before
                                       # committing; if it isn't (e.g. an
                                       # oncoming car), it falls back to
                                       # the swerve/stop behavior already
                                       # verified, never forces the change.
OBSTACLE_MIN_SPEED_SCALE = 0.0   # full stop at/inside OBSTACLE_STOP_DIST_M
                                  # when no safe maneuver exists (e.g. an
                                  # oncoming car in the only clear-looking
                                  # adjacent lane) -- this never guesses
# 2026-07-27: after fixing every decision-layer bug I could find (side
# flip-flopping, throttle starvation, late/flaky detection, premature
# commitment release), a real test still drove dead center into the cone
# with zero visible sideways drift, even though the log confirms
# lane_offset was correctly committed to -0.75 and held for 6+ seconds.
# In that same log, throttle sat at its floor (0.1) during the hardest
# steering (-0.53 to -1.0) while the tracked reference (x_f) swung from
# 55px to 288px -- consistent with the car PIVOTING its heading at low
# speed rather than actually carving a turn and translating sideways.
# -0.75 is the Mission 2 steady-state lane-riding target (fine to ease
# into gradually); demanding that big a jump as an emergency maneuver may
# just be too sharp for this car to execute at the throttle it has
# available. Softened to -0.5 so less lateral travel is needed to clear
# the same obstacle. This is a hypothesis from visual+log evidence, not
# something a frame replay can verify (it's about real wheel-to-ground
# dynamics) -- if the car still doesn't visibly move sideways, the deeper
# issue is throttle/speed during the turn, not this offset value.
OBSTACLE_LANE_CHANGE_OFFSET_LEFT = -0.5
# 2026-07-27 (later same day): replayed session 26-07-27_75 (catalog_105/106)
# against the current code (swerve/throttle-floor fix already applied) --
# this time the car DID genuinely translate sideways at -0.5 (x_f climbed
# 104->156px over ~2s, throttle_final stayed healthy at 0.11-0.25 the whole
# pass, confirming the pivoting bug above is fixed). But two distance
# readings recorded mid-pass -- 0.53m at t=4.46s and 0.29m (the recurring
# near-field saturation floor) at t=7.97s, both well inside
# OBSTACLE_STOP_DIST_M=0.6 -- line up with the real-world report of the car
# clipping the first cone's side while driving by. Throttle was correctly
# NOT reduced here (that's the fixed, intended behavior for an active
# maneuver) so this is a pure lateral-clearance/geometry problem: -0.5
# isn't carrying the car far enough from the cone. Nudged to -0.6, still
# short of the pivot-prone -0.75 (which only pivoted because throttle was
# starved at the time; that throttle bug is now fixed, so -0.75 may in fact
# be safe too, but moving in smaller increments first). Like the entry
# above, this is a physical-clearance hypothesis a frame replay cannot
# fully confirm -- needs a real test drive.
OBSTACLE_LANE_CHANGE_OFFSET_LEFT = -0.6
# 2026-07-27 (still later): real test at -0.6 -- "same issue, starts
# changing lanes almost immediately, grazes the cone before continuing."
# Replayed session 26-07-27_76 (catalog_106/107) and got the smoking gun:
# the minimum distance reading during the pass was 0.5274m -- IDENTICAL
# (to 4 decimal places) to session 75's minimum at offset -0.5. A 0.1
# change in offset produced a ZERO measurable change in closest approach.
# So offset magnitude isn't the lever. Digging into WHY: printed the raw
# corridor/blob pixel positions frame-by-frame. lane_applied (the ramped
# offset) reaches its full -0.6 target in ~1.3s (matches LF_LANE_RAMP_SEC
# =1.5s), but the car's own tracked position (x_f, a proxy for real
# lateral progress) only crawls from ~82px to ~190px over the SAME ~4
# seconds it takes to draw level with the cone -- i.e. the steering
# TARGET arrives fast, but the car's REAL lateral translation is much
# slower, and line_following.py's own curvature-based throttle law
# (self.throttle = th_max - (th_max-th_min)*slow, where slow tracks
# abs(steering)) independently drops throttle to th_min for ~2.5s
# whenever steering pegs at +/-1.0 -- which it does here, immediately,
# because commanding a full -0.6 (or -0.75) jump from a +0.5 base creates
# a big instant P-error. This is a real vehicle-dynamics bottleneck, not
# a decision-layer bug, and not something further offset tuning fixes
# (confirmed by the -0.5->-0.6 null result). Bumping to -0.75 anyway:
# it's the one value in this codebase with a field-tuned pairing already
# validated for steady-state lane riding (LF_CURVE_GAIN=-0.8, see
# LF_CURVE_GAIN's table above) -- the ONLY reason -0.75 looked bad
# earlier was the throttle-starvation bug (since fixed). Small
# increments haven't moved the needle, so testing the full vetted value
# next instead of creeping by 0.05-0.1 at a time.
OBSTACLE_LANE_CHANGE_OFFSET_LEFT = -0.75
# 2026-07-27: even with x_f (filtered) driving the corridor, a real test
# replay showed the side reading can still flip for a frame right up
# against a close obstacle -- re-deciding the lane-change DIRECTION from
# scratch every tick meant the car reversed mid-maneuver and never
# finished one. LaneOffsetCommander now commits to a direction once
# chosen and holds it regardless of any single frame's reading, releasing
# back to the base lane only once the obstacle reads "not present" for
# this long.
#
# Raised from 1.0 the same day: a time-accurate replay (using the real
# recorded timestamps, not just frame order -- frame-order timing made
# the ramp look like it was progressing much slower than it really was)
# showed detection itself drops to "not present" for well over a second
# at a stretch even while the SAME obstacle is still there -- a real gap
# in an imperfect detector, not genuine clearance. At 1.0s that gap
# released the commitment and sent the car back into its original lane
# mid-lane-change, which is exactly what "starts changing lanes, then
# crashes anyway" looks like. 4.0s comfortably outlasts both the observed
# detection gap and LF_LANE_RAMP_SEC's ~1.5-2s to physically complete the
# ramp, so a committed maneuver gets seen through. Lower only once
# detection reliability at range has itself been improved (a symptom of
# the still-uncalibrated OBSTACLE_MONO_SLOPE and OBSTACLE_SLOW_DIST_M
# above being pushed wider than validated) -- until then, an
# occasionally-slow return to lane is a far smaller problem than
# abandoning a maneuver already in progress.
#
# 2026-07-30: LOWERED 4.0 -> 2.5 (user asked; the slow return was visible
# on track). This is backed by a measurement on the first cone drive with
# the fixed follower, session log lf_20260730_133948.jsonl / idx
# 185757-187281, replayed through the detector: 165 present-frames group
# into 5 cone encounters, and the WORST detection dropout *within* an
# encounter -- the failure this constant exists to survive -- is 1.20s
# (others 0.45s, 0.20s; only 3 dropouts total). 2.5s therefore keeps
# +1.30s of margin over the worst real gap, more than double it, while
# 1.0s would have broken one of the three.
#   Measured effect on the return: last present=True -> release was 4.0s,
#   plus LF_LANE_RAMP_SEC 1.5s to ramp back = ~5.5s tail. Now ~4.0s.
#   (Do NOT read the 7-27s "gaps" in a naive present=True bracketing as
#   dropouts -- those are the intervals BETWEEN different cones, and
#   releasing there is correct.)
# Caveat: one run, 5 encounters, 3 internal dropouts -- a small sample,
# and the old 4.0 was set from OLD-CAMERA sessions (pre-07-28 remount)
# whose geometry no longer applies. lf_telemetry.py now logs `obs` and
# `avoid_dir` per frame, so if a longer dropout does show up it will be
# visible directly instead of needing another replay reconstruction.
# The real fix is a release condition based on having PASSED the
# obstacle rather than on the detector going quiet -- see the notes on
# the two-cone slalom; this constant is a stopgap either way.
#
# 2026-07-30, second reduction: 2.5 -> 1.5. Measured on the single-cone
# run at idx 187282-187477 with 2.5 live: last sighting -> release took
# 2.61s, then LF_LANE_RAMP_SEC another 1.7s to ramp -0.75 back to +0.40
# (1.15 offset-units at 1/1.5 per second), so 4.3s total and still too
# slow on track. Across ALL FIVE recorded cone runs the worst detection
# dropout WITHIN an encounter is now only 0.40s (n=2, median 0.27s), so
# 1.5s keeps +1.10s of margin -- nearly 4x the worst real gap.
# Note the remaining tail is now RAMP-dominated, not debounce-dominated:
# ~1.5s wait + ~1.7s ramp. Cutting this further has diminishing returns;
# LF_LANE_RAMP_SEC is the thing to look at next, and that one trades
# against how sharply the car snaps back into lane.
# The event-driven early release (OBSTACLE_NEW_OBJECT_JUMP_M in
# obstacle_avoidance.py) is what actually handles a second cone -- this
# timer only governs the return to base lane when nothing new appears.
OBSTACLE_CLEAR_DEBOUNCE_SEC = 1.5

# 2026-07-27: the -0.75 test above wasn't actually a throttle-starvation
# problem after all -- re-checked with a time-accurate replay of the
# ACTUAL real footage (not a hypothetical config swap into old footage,
# see HANDOFF.md's methodology-trap warning) and found `x_f` starts BOTH
# real -0.75 recordings already ~65-71px off-center in the very first
# frame, before this lane-change offset has ramped in at all. That alone
# saturates line_following.py's steering law (LF_STEER_KP=1.2 * error)
# past +/-1.0, and LF_STEER_SLEW=6.0 walks it to a full peg in ~0.17s --
# independent of obstacle_avoidance.py entirely. Once pegged, the
# throttle law floors speed for as long as the error stays large, which
# starves out a -0.75 lane change (needs ~1.1s just to ramp in, per
# LF_LANE_RAMP_SEC) long before it can complete. Two changes together
# (both explicit values below, not just DEFAULTS, so they're visible
# here rather than buried):
#   1. Don't attempt a FRESH full lane change when the obstacle is
#      already this close -- fall back to the much smaller swerve
#      instead, which needs far less time/distance to complete. See
#      OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M's own comment in
#      obstacle_avoidance.py for the full reasoning.
#   2. line_following.py's throttle law no longer floors ALL THE WAY to
#      LF_THROTTLE_MIN while steering is saturated from a large
#      POSITIONAL error (as opposed to genuine track curvature, which is
#      untouched) -- see LF_THROTTLE_MIN_LANE_OFFSET's comment there.
# Neither of these is a full fix for what caused x_f to start off-center
# in the first place (unresolved -- see HANDOFF.md open question) but
# both should reduce how badly the car gets stuck reacting to it. Only a
# real test drive can confirm that -- replay can verify the new decision
# logic fires as intended against the recorded footage, but (per the
# same methodology-trap warning) it cannot show what the car would
# actually have done differently, since the camera frames themselves are
# fixed. NOT YET VALIDATED ON A REAL TEST.
OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M = 1.0
LF_THROTTLE_MIN_LANE_OFFSET = 0.18

# ── FPV ──────────────────────────────────────────────────────────────
USE_FPV = True
