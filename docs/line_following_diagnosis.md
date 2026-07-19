# Line-Following Diagnosis — Track 2, Mission 1

Summary of a debugging session investigating why the OAK-D + Raspberry Pi car
wasn't reliably following the yellow dashed center line or staying between the
white lane lines. Covers everything from getting the recording pipeline
working to a data-driven diagnosis of the actual failure and two confirmed
code fixes.

## Background

The car uses a custom CV part, `donkeycar/parts/line_following.py`
(`LineFollower`), driven via `manage_line.py` with `myconfig.py` pointing
`CV_CONTROLLER_MODULE`/`CV_CONTROLLER_CLASS` at it. It detects the yellow dash
in CIELAB color space (b-channel yellowness + a-channel to reject orange
cones) plus an HSV hue guard to reject vegetation, scans 8 horizontal bands in
the bottom portion of the frame, fits a line through the per-band centroids to
get lateral error + heading, and steers with a PD controller. On losing the
line it coasts on last-known steering for a grace period, then stops (by
design, it never blind-turns).

Reported symptoms: the car didn't reliably track the line or stay in-lane. At
least three different failure modes were observed across runs: growing
oscillation, tracking fine then suddenly going straight/unresponsive, and an
instant hard veer left or right from the start. A teammate got noticeably
better results at a different time of day, suggesting a lighting-robustness
problem, though the varied failure modes hinted at more than one cause.

## Getting to real data

No diagnostic footage existed initially. Getting a usable recording required
fixing three separate, independent recording blockers, all in
`~/mycar/myconfig.py` / controller behavior (car-local config, not part of
this repo):

1. `RECORD_DURING_AI` defaults to `False`, which unconditionally forces
   recording off whenever `user/mode != 'user'` — i.e. exactly when
   `LineFollower` is actually driving. Set `RECORD_DURING_AI = True`.
2. `AUTO_RECORD_ON_THROTTLE = True` gates the joystick controller's own
   recording state to `mode == 'user'` (`donkeycar/parts/controller.py`
   `on_throttle_changes()`), *and* makes the manual recording-toggle button a
   no-op (`toggle_manual_recording()`). Set `AUTO_RECORD_ON_THROTTLE = False`
   so manual toggling actually works in autopilot mode.
3. `TOGGLE_RECORDING_BTN = "option"` was configured for a controller type that
   doesn't have that button — `CONTROLLER_TYPE = "F710"` (Logitech), whose
   real button names are `back`/`start`/`Logitech`/`A`/`B`/`X`/`Y`/`L1`/`R1`
   plus stick-presses. `"option"` is PS4 DualShock naming and was a dead
   binding. `B` is the F710's actual default recording-toggle button.

With those fixed, a real recording session (a few manual laps + two autopilot
laps) produced 4,025 frames in `~/mycar/data_line/`.

## Data-driven diagnosis

Ran `LineFollower.detect()` offline against the ~1,900 kept (non-deleted)
frames from that recording. Quantified findings:

- **Detection rate: 64.8%** overall (35.2% of frames had no line fix at all).
- **15.2% of all frames had zero mask pixels** — not brief dash-gap dropouts
  (those would be 1-3 frames); many of these were contiguous runs of
  10-19 frames (0.5-1s at 20Hz), too long to be gaps between dashes.
- **Heading instability at low band counts**: when only the minimum 2 bands
  were found (32% of all detected frames), the fitted heading saturated to
  the ±1.0 clip in **65.7%** of those frames (mean |heading| 0.843). With
  `LF_HEADING_GAIN=0.9`, a saturated heading alone commands near-max steering
  regardless of actual lateral error — this is almost certainly the "instant
  hard veer" failure mode. A 2-point weighted line fit is extremely sensitive
  to single-band noise.

Visual inspection of the actual overlay frames surfaced two further, more
fundamental problems:

- **The detector locked onto the white boundary line instead of the yellow
  dash** in a "well-performing" (98% detection) segment — the white line was
  scoring as "yellow enough" under this lighting.
- **Direct pixel sampling on a zero-mask frame** showed the physically-yellow
  tape rendering as `RGB=(136,198,221)`, `HSV hue=98` — genuinely cyan, not a
  borderline miss. The OAK-D's auto white balance appears to shift color
  rendering dramatically under this track's mixed/artificial lighting, enough
  that no fixed LAB/hue threshold can track it. This plausibly explains the
  teammate's better results at a different time of day, the large dropout
  rate, and the false lock-on to the white line.

## Confirmed code bug: OAK-D never applied the configured resolution

Recorded frames were **1920x1080**, not the `IMAGE_W=160`/`IMAGE_H=120`
configured in `myconfig.py` — meaning `LineFollower` had been running on
frames ~12x/9x larger than every size-dependent constant in the file assumes
(band pixel thresholds, morphology kernel scale, etc. are explicitly
"sized for 160x120" per the file's own comments).

Root cause in `donkeycar/parts/oak_d.py`: `setup_rgb_camera()` calls
`cam_rgb.setPreviewSize(...)` but linked `cam_rgb.video.link(xout_rgb.input)`
— the DepthAI `.video` output ignores `setPreviewSize()` entirely and always
streams full sensor resolution; only `.preview` respects it. (Commit 8d4a707,
"Switch OAK-D RGB output from `video` to `preview` stream" (#1235), changed
the size-setter call from `setVideoSize` to `setPreviewSize` but never
actually updated the `.link()` call — the bug it claimed to fix was never
fully fixed.)

Separately, `donkeycar/templates/complete.py`'s `add_camera()` never passed
`cfg.IMAGE_W`/`cfg.IMAGE_H` into the `OakD(...)` constructor at all, so even a
correct `.preview` link would have defaulted to the class's 640x480 default
instead of the configured 160x120.

### Fixes applied (this commit)

- `donkeycar/parts/oak_d.py`: `cam_rgb.video.link(...)` →
  `cam_rgb.preview.link(...)`.
- `donkeycar/templates/complete.py`: `OakD(...)` now passes
  `width=cfg.IMAGE_W, height=cfg.IMAGE_H`.

These are unverified against real hardware (no camera access from this
environment) — needs an on-car test to confirm frames are now actually
160x120 before trusting any resolution-dependent tuning.

## Still open

The color/lighting robustness problem (yellow tape reading as cyan under
certain lighting; false lock-on to the white line) is unresolved and is a
separate, harder problem from the resolution bug. Worth re-running this same
offline diagnosis against fresh footage once the resolution fix is confirmed
on hardware, since fixing the scale mismatch may itself improve detection
stability (less noise-sensitive band counts) even before addressing color
robustness directly.
