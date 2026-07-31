# Autonomous Line-Following Car — Project Writeup

A complete account of the engineering work on a Donkeycar-based autonomous
RC car: what it had to do, every failure mode found and how each was
diagnosed, what was fixed, what deliberately wasn't, and the methodology
that made a car with no debugger attachable to it debuggable at all.

Written to be read standalone — for a presentation, a writeup, or as
preparation for talking through the project in detail. Every number in it
is a measurement from a real recorded session, not an estimate.

**Companion documents** (the round-by-round engineering logs this
summarises): [`line_following_diagnosis.md`](line_following_diagnosis.md),
[`line_following_lane_centering.md`](line_following_lane_centering.md),
[`line_following_mission2_lane_following.md`](line_following_mission2_lane_following.md),
and `linefollower_car/WORKLOG_20260730.md` on `main`.

---

## 1. The system

A 1/10-scale RC car running [Donkeycar](https://donkeycar.com), a Python
autonomous-driving framework, on a Raspberry Pi with an OAK-D depth
camera.

| layer | what it is |
|---|---|
| `manage_line.py` | vehicle assembly — wires camera → controller → drivetrain into Donkeycar's part pipeline, running at `DRIVE_LOOP_HZ = 20` |
| `line_following.py` | the CV controller: detects the line, computes steering + throttle |
| `obstacle_detector.py` / `obstacle_avoidance.py` | cone detection and swerve / lane-change planning |
| `lf_telemetry.py` | per-frame JSONL telemetry, joined to the recorded image tub by frame index |
| `dashboard.py` / `dashboard.html` | browser dashboard: live tuning, start/stop, camera feed |
| `lf_tools/replay.py` | offline replay harness — re-runs recorded frames through the real controller |

**The task.** The track is an outdoor pedestrian plaza marked with a
dashed yellow centreline and solid white boundary lines on each side, with
two side-by-side lanes.

- **Mission 1** — follow the yellow dashed centreline.
- **Mission 2** — drive *within one lane*, bounded by the yellow line on
  one side and the outer white line on the other, without crossing either.
- Throughout — detect and avoid obstacles (traffic cones), by swerving
  within the lane or changing lane entirely.

**Why this is harder than it sounds.** The centreline is *dashed*, so the
primary sensor signal disappears several times a second by design. The
track is outdoors, so lighting swings from full sun to deep building
shadow within a single lap. And the environment is a real plaza — concrete
planters, a recycling kiosk, benches, glass doors, pedestrians walking
through frame — none of which the detector may mistake for a lane marking.

---

## 2. The detector, in brief

Understanding two design decisions makes every failure below legible.

**Colour detection in CIELAB, on local contrast rather than absolute
values.** The stock Donkeycar line follower thresholds HSV, which fails
here: the tape's saturation swings from ~160 at night to ~87 in daylight,
and no fixed range covers both. Measured on this track, the tape reads
`b≈161` in sun but only `b≈141` in building shadow, while the pavement
beneath reads 129 and 120 respectively. No absolute threshold separates
them — but the tape is *always* +12..+30 yellower than the pavement
immediately around it, in every lighting condition. So the mask is "b
exceeds the local median-blurred background by `LF_B_CONTRAST`", OR'd with
an absolute test retained for night driving.

**Multi-band centroid fitting, not a single scanline.** The stock detector
samples one horizontal slice; since the line is dashed, that slice
regularly lands in a gap, detection drops, and the car drifts on stale
steering. This detector scans a tall region split into horizontal bands,
finds a centroid per band, and fits a line through them — yielding both
lateral offset *and* heading, and surviving dash gaps.

**The fallback ladder.** When yellow is lost, the controller degrades
through progressively weaker references rather than giving up:

```
yellow dash  →  two-white-line lane centre  →  single-side white line
             →  near-white tracker (blind bends)  →  coast  →  stop
```

Almost every serious failure in this project happened *inside this ladder*
— not in the primary detector, but in the machinery that decides what to
trust when the primary signal is gone.

---

## 3. Round 1 — Getting to real data, and a 12× resolution bug

**Symptom.** The car didn't reliably track the line. Three different
failure modes were reported: growing oscillation, tracking fine then
suddenly going straight, and an instant hard veer from the start. A
teammate got better results at a different time of day.

**First problem: there was no data.** Three independent, unrelated config
bugs each silently prevented recording during autopilot — exactly when
footage was needed:

1. `RECORD_DURING_AI` defaults to `False`, forcing recording off whenever
   the autopilot is driving.
2. `AUTO_RECORD_ON_THROTTLE = True` gates recording to manual mode *and*
   makes the manual toggle button a no-op.
3. `TOGGLE_RECORDING_BTN = "option"` — PS4 naming, on a Logitech F710
   controller that has no such button. A dead binding.

Fixing all three produced the project's first real dataset: 4,025 frames.

**Then, quantified diagnosis.** Replaying ~1,900 kept frames offline
through the real `detect()`:

- **Detection rate 64.8%** — 35.2% of frames had no line fix at all.
- **15.2% of frames had zero mask pixels**, many in contiguous runs of
  10–19 frames (0.5–1s) — far too long to be dash gaps.
- **Heading saturated in 65.7%** of the frames where only the minimum 2
  bands were found (itself 32% of detected frames), mean `|heading|`
  0.843. A 2-point fit is wildly noise-sensitive; with `LF_HEADING_GAIN
  = 0.9` a saturated heading commands near-max steering regardless of
  actual lateral error. **That is the "instant hard veer".**

**Then, the root cause nobody had suspected.** Recorded frames were
**1920×1080**, not the configured **160×120**. The detector had been
running on frames 12× wider and 9× taller than every size-dependent
constant in it assumes — band thresholds, morphology kernel scale, all
explicitly documented as "sized for 160x120".

Two independent bugs caused it:

- `oak_d.py` called `setPreviewSize()` but linked `cam_rgb.video` — the
  DepthAI `.video` output *ignores* `setPreviewSize()` entirely and always
  streams full sensor resolution. Only `.preview` respects it. An earlier
  upstream commit (#1235) had changed the size-setter but never updated
  the `.link()` call, so the bug it claimed to fix was never actually
  fixed.
- `complete.py` never passed `cfg.IMAGE_W`/`IMAGE_H` into the `OakD`
  constructor at all, so even a correct link would have used the class
  default.

**Also found, unresolved:** direct pixel sampling showed physically-yellow
tape rendering as `RGB=(136,198,221)`, `HSV hue=98` — genuinely **cyan**.
The OAK-D's auto white balance shifts colour rendering far enough under
this track's mixed lighting that no fixed threshold can track it. This
plausibly explains the teammate's better results at a different hour.

---

## 4. Round 2 — On-car testing contradicts offline replay

Round 1's resolution fix had been reasoned out from a recording, never run
on the car. Testing it on hardware found it was **incomplete in two ways**,
both of which crashed the camera outright:

- `setup_rgb_camera()` referenced `self.image_w`/`self.image_h`, which are
  never assigned anywhere (`__init__` sets `self.width`/`self.height`).
  `AttributeError` on every run.
- `_poll()` unconditionally fetched the `"depth"` queue whenever RGB *or*
  depth was enabled, but the pipeline only creates it when depth is on.
  Any RGB-only config — the common case for a line follower — crashed with
  `RuntimeError: Queue for stream name 'depth' doesn't exist`.

After both fixes, frames finally came back at 160×120 — confirmed by
checking `.shape` on *freshly recorded* frames, not by re-reading old ones.

**Then the round's real lesson.** A new two-white-line lane-centering tier
was added: when both boundaries are visible, project each back to where the
dash should be and average the two estimates. The first version used the
raw geometric midpoint, implicitly assuming the dash sits exactly halfway
between the boundaries. On this track it measurably doesn't.

On-car testing caught it: the car "drove nicely between the two white
lines" but visibly abandoned the dash. Replay explained it precisely — a
stretch tracking the dash at `error = -0.16` jumped to a stable-but-wrong
`error = -0.32` the instant the midpoint tier took over.

A revision projected each detection through the *learned* per-band offset
instead. Offline, it looked like a strict improvement: the error corrected
to `-0.27`.

**On a real drive it performed worse than the version it replaced.**

> **This is the single most important methodological finding in the
> project.** Offline replay against a fixed recording is genuinely useful —
> it correctly explained both of the driver's complaints — but it is
> **open-loop**. It replays the camera frames a real drive actually
> produced, regardless of what the code under test would have commanded. It
> can prove a fix changes a computed decision as intended; it can never
> prove the fix keeps the car on a different physical path. The gap is
> worst for anything involving a running EMA, whose state at drive time
> depends on the whole session history a partial replay won't reproduce.

That caveat governs every conclusion in this project, and is flagged again
wherever it materially limits what a fix could verify.

---

## 5. Round 3 — Daytime hardening and Mission 2

The detector, PID tuning and lane logic had only ever been exercised at
night. The first daytime test surfaced two new failure modes:

**Ill-conditioned slope fits.** Heading is a slope through as few as 2 band
centroids. With two points close together vertically, a couple of pixels of
noise — far likelier in patchy daylight shadow than uniform night light —
swings the fit wildly. Footage showed raw heading pegging **+1.0 to −1.0
within half a second** while genuinely tracking a real dash, steering the
car back and forth until it wandered into a planter bed. Fixed with
`LF_MIN_HEADING_SPREAD_FRAC`: when kept fit points are too vertically
bunched, heading reports neutral instead of the raw value. Position is
unaffected — it's a bounded near-field average, not a raw slope, so it
stays usable when the fit is too ill-conditioned for heading.

**Stale white-boundary reuse with no expiry.** The cached per-side white
cluster had no staleness check — once set, it would be reused indefinitely
if that side simply stopped being detected. One session's right boundary
went undetected for **8+ seconds** (out of frame past a planter) while the
stale value was fed into the position estimate every frame, dragging
steering harder and harder toward a recycling-bin enclosure until the run
had to be stopped. Fixed with `LF_WHITE_STALE_SEC`.

### Mission 2 lane following

Most of the geometry already existed. What was missing was a way to pin a
lane side, and any explicit check that a computed position doesn't imply
the car has left the lane.

- **`LF_LANE_SIDE`** — pins *both* halves of the field-tuned pairing
  (offset **and** curve gain) with one setting, a direct response to
  on-car iteration where keeping two settings in sync by hand kept going
  wrong. **This same failure recurs in Round 4, from the dashboard side.**
- **`_lane_pos()`** — the core metric. A position expressed as `0.0` = on
  the yellow, `1.0` = on the outer white. Outside `[0,1]` means past a
  boundary. Returns `None` when lane-following is off, making everything
  built on it a structural no-op for plain line-following — verified by
  replaying sessions before and after each change and confirming
  **byte-identical** steering, throttle and status output.
- **Soft boundary clamp** — rejects a white-guided estimate that lands past
  a boundary, but *only* when a boundary is seen fresh that frame, and
  deliberately *not* on the heavily-tuned near-white tracker used for blind
  bends. That tracker exists precisely for the no-yellow-reference case;
  gating it on an unvalidated check risked breaking the one thing it was
  built for.
- **Implausible-jump gate** — a real test found a frame where a *genuine*
  fresh yellow fit implied the car moved 1.5 lane-widths sideways in one
  1/20s frame. The existing temporal gate missed it because that compares
  against the previous frame, not against lane geometry.
- **Boundary-proximity urgency** — a test drive drifted to `lane_pos =
  −0.33`, a third of a lane-width into the oncoming lane, during a
  sustained curve. Root cause: the `LF_MIN_HEADING_SPREAD_FRAC` guard
  above stayed engaged for the *entire* ~2-second curve (this camera angle
  produces chronically narrow fits at that corner), leaving no curve
  feed-forward and only a laggy position term. Rather than loosen a guard
  that was itself a validated fix, the position gain now ramps up as
  `lane_pos` approaches an edge.

---

## 6. Round 4 — Corner failure analysis, dashboard, and upstreaming

Three workstreams. **This round changed no driving code at all** — the
diagnosis was delivered as evidence, deliberately separated from the fix.

### 6.1 Why the car left the track at the concrete block

**The symptom.** The car repeatedly failed at the first right-hand corner,
running into a concrete block with a black kick-strip at its base. Other
runs the same day also ended off-track.

**The data.** Six telemetry files, 12,261 recorded frames, ~19 minutes of
autopilot across six runs. Method: cross-reference the JSONL telemetry
against the recorded camera frames, then replay the failure window through
the *real* controller with the run's own live tuning values re-applied
per-frame — so the detector's own view of the scene is visible, not
inferred.

**The finding.** The car was tracking *cleanly* into the corner — yellow
found every frame, `err` within ±0.11, `lane_pos` 0.66–0.79, comfortably
mid-lane. Then three faults compounded:

**(a) A stale heading drives a safety floor the wrong way.** At frame
190862 the filtered heading `h_f` drops to **−0.724** and then *never
updates again* — frozen for the remaining **170 frames (8.5 seconds)**,
because nothing refreshes heading once yellow is gone.

`LF_HEADING_CLIP` is `0.7`, so the heading term sits *exactly pegged* at
−0.7. That trips the sharp-bend floor:

```python
if abs(hterm) >= bend_frac * self.h_clip and self.sharp_bend_min > 0:
    s = 1.0 if hterm > 0 else -1.0
    cmd = s * max(s * cmd, self.sharp_bend_min)
```

`|−0.7| ≥ 0.9 × 0.7 = 0.63` — so the floor fires and forces at least 0.5
of steering **in the sign of the lean: negative, i.e. LEFT, inside a
right-hand corner.**

That floor is *correct*. Its own comment explains why it exists: "refusing
to steer WITH a pegged heading guarantees losing the line." It was a
validated fix for cars going straight through sharp bends. But nothing
checks the pegged lean is still **fresh**, and here it is 8.5 seconds stale.

**(b) The white fallback reads a curve as sideways drift.** With yellow
gone, the white boundary sweeps across frame — because the *track turns*:

| frame | 190862 | 190868 | 190871 | 190874 | 190877 |
|---|---|---|---|---|---|
| `white_x_f` | 322 | 249 | 220 | 197 | 173 |
| `lane_pos` | 0.66 | 0.84 | 0.99 | 1.13 | **1.22** |
| `err` | 0.04 | −0.07 | −0.23 | −0.39 | −0.52 |
| `steer` | −0.03 | −0.21 | −0.32 | −0.36 | −0.46 |

The controller cannot distinguish "the boundary moved because the road
curves" from "the boundary moved because I drifted", so it reads a right
turn as rightward drift and steers **left** — past `lane_pos = 1.0`, i.e.
past the outer boundary entirely.

**(c) Freeze, then coast in.** At frame 190884 the white is lost too, and
`x_f`, `err`, `steer` and `h_f` all freeze together: **2.6 seconds coasting
at throttle 0.15 with the wheels held at half left lock**, then after a
brief reacquire a **second freeze of 1.8s**, driving straight at the block.

**(d) The endgame.** At frame 191016 the detector finally re-acquires and
accepts a **336-pixel jump** — dead-reckoned −184 to detected +245, on a
frame only **384 pixels wide** — saturating error to +1.0 and slamming full
right lock. By that frame the block already fills the camera view.

**The same signature across every failed run that day:**

| run / segment | longest blind | longest frozen steering | max accepted jump |
|---|---|---|---|
| 12:20 seg 2 | 9.8 s | 15.1 s | 239 px |
| 13:08 seg 1 | 1.5 s | 3.0 s | **442 px** |
| 13:08 seg 2 | 1.6 s | 3.0 s | **456 px** |
| 15:40 seg 1 | 4.4 s | 4.5 s (held at −0.93) | 33 px |
| 15:40 seg 5 | 3.3 s | 2.6 s | 336 px |

Jumps of 442 and 456 px are **larger than the frame is wide**. Both 13:08
segments died at the *same* planter on the *same* corner. These are the
temporal gate's deliberate forced-accept — a documented, intentional
mechanism ("a stale last position can never lock the detector out for
good") being exploited by a cluttered scene.

**Why this was subtle.** Every individual mechanism here is correct and
was a validated fix for a real earlier bug. The sharp-bend floor stops
under-turning. The forced-accept stops permanent lockout. The white
fallback bridges dash gaps. The failure is *emergent* — it only appears
where a stale input meets a committed-by-design response. That is why it
survived three prior rounds of debugging: nothing is individually wrong.

**Deliberately not fixed.** Diagnosis was delivered as evidence, with
candidate fixes ranked by strength of evidence, and *no driving code
changed*. Given Round 2's lesson — that an offline-verified "strict
improvement" drove worse on the real car — shipping a speculative fix to
`line_following.py` on offline evidence alone would repeat exactly the
mistake this project already paid for once.

### 6.2 The dashboard label that was actively dangerous

The live-tuning slider read `-1 left lane · 0 center · +1 right lane`.

That is **wrong**, and dangerously so. `lane_offset` is in lane-widths from
the yellow centreline, and `±1.0` puts the car's target **on the outer
white boundary paint** — not in the lane. The label invited exactly the
too-large offsets seen in the crash run, which drove at `+0.70` and spent
**129 of 516 frames past the outer boundary**.

Worse, offset and curve gain were presented as two independent sliders,
when they were field-tuned as *pairs* — the same
keep-two-settings-in-sync-by-hand problem `LF_LANE_SIDE` had already solved
on the config side in Round 3, reintroduced by the UI.

**Fixed** by replacing the label with the three measured pairs, served from
the server so UI and warning logic can't drift apart:

| preset | lane offset | curve gain |
|---|---|---|
| left lane | −0.75 | −0.80 |
| centre line | 0.00 | −0.80 |
| right lane | **+0.50** | **−0.20** |

The active pair is badged "these are the optimal settings for the *right
lane*"; anything else reads "custom pair — not one of the field-tuned
optima". Clicking a preset applies **both** values in one request, so the
car is never left running a mismatched pair mid-update.

### 6.3 The laggy video feed — and why the obvious fix was wrong

The dashboard's camera feed lagged badly. Root cause in Donkeycar's stock
MJPEG handler: it polls the current frame every 5 ms (**200 Hz**) and
re-encodes and re-sends it *whether or not it changed*. The camera runs at
20 Hz. Most of what went on the wire was byte-identical duplicates — and
MJPEG inside an `<img>` cannot skip stale frames, so the browser decodes
and displays every one, in order. **The backlog is the lag.**

The intuitive fix is "the JPEG encoder is slow, switch PIL for cv2."
Benchmarking first showed that would have been a **2× pessimisation**:

| encoder | per frame @ 384×216 |
|---|---|
| PIL (stock) | **0.64 ms** |
| `cv2.imencode` | 1.22 ms |

Encoding was never the bottleneck. The real fixes were structural:

- **De-duplicate** — only encode when the frame is genuinely new, so wire
  rate follows the camera, not the poll loop.
- **Never queue** — await the previous frame's flush before looking for the
  next, so a slow link costs frame *rate* rather than accumulating *delay*.
  The client always gets the newest frame, not the oldest unsent one.

| | frames/s | bandwidth |
|---|---|---|
| before | 163.3 | 2133 KiB/s (16.7 Mbit/s) |
| after | 19.9 | 258 KiB/s (2.0 Mbit/s) |

**8.3× less data**, and the stream now tracks the camera exactly.

A second measurement caught a subtler error: capping the stream at the
camera's 20 fps measured only **18.9 fps**, because loop jitter put some
frames barely under the interval and the cap dropped real ones. The cap was
raised to 40 as a pure backstop — de-duplication is what actually pins the
rate.

### 6.4 Upstreaming — and letting the guide change the deliverable

The duplicate-frame bug is not specific to this car; it affects every
Donkeycar user. Reading `CONTRIBUTING.md` before packaging it changed the
work in two ways:

**It requires unit tests** — listed both as a requirement and as a reason
PRs get rejected. Four were written: placeholder sent once, one camera
frame → one wire frame (the regression test), each new frame streamed
promptly, and de-duplication keyed on array *identity* not pixel equality
— because two captures of a static scene are different frames and both must
send, or the stream would stall whenever nothing moves.

**Writing the tests found a bug the benchmark could not.** The handler
parks forever on a disconnected client, because it only notices the
disconnect on its next write — which never comes if the camera has stopped.
Fixed with tornado's `on_connection_close()`. A throughput benchmark could
never have surfaced this; only asserting on *lifecycle* did.

**It also said the car code should not go upstream** — PRs are unlikely to
be accepted if they add features "not useful to a broad audience", with the
guidance to maintain those in a fork. That describes `line_following.py`
exactly: its thresholds are field measurements of specific tape under
specific lighting. So the work was split — a clean single-purpose branch
touching only the library and its tests, and the car code kept fork-side.

---

## 7. Results

| | outcome |
|---|---|
| Recording pipeline | 3 independent config bugs fixed; first real dataset obtained |
| OAK-D resolution | 12×/9× oversized frames fixed across 4 distinct bugs in 2 files |
| Detection robustness | heading-spread guard, white staleness expiry, jump gates |
| Mission 2 | lane-following built on `_lane_pos`, verified byte-identical no-op when off |
| Corner failure | root-caused to a stale heading driving a safety floor the wrong way |
| Dashboard tuning | wrong-and-dangerous label replaced with measured presets |
| Video latency | **8.3× less data**, stream tracks camera exactly |
| Upstream | single-purpose PR branch, 4 new tests, 13/13 web tests passing, PEP-8 clean |

---

## 8. Methodology — how you debug a car you can't attach a debugger to

The techniques that made this tractable, and the honest limits of each.

**1. Instrument what the recording throws away.** The image tub records the
camera frame, the steering that reached the drivetrain, and the throttle.
Everything else about *why* — which tuning was live, whether the detector
saw anything, which gate rejected the fix — was discarded at process exit.
Three genuinely different faults were therefore indistinguishable in the
tub: all three look like "steering went somewhere odd and the car left the
track". Adding one JSONL line per frame alongside the tub (never altering
its schema, which would break a 178k-record dataset) turned inference into
measurement.

**2. Replay through the real controller, not a model of it.** Re-running
recorded frames through the actual `LineFollower` — same class, same
config, same timestamps — means a conclusion is about the shipped code, not
a reimplementation that might differ in exactly the way that matters.

**3. Always look at the actual frames.** Telemetry says *what* the
controller concluded; only the image says *whether it was right*. The
cyan-rendering tape, the detector locking onto a white line, and the
concrete block itself were all found by looking at pixels.

**4. Measure before optimising.** Two intuitive fixes this round would have
made things worse: switching to cv2 (2× slower) and capping the frame rate
at the camera rate (dropped real frames). Both were caught by measuring
first.

**5. Know what your method cannot prove.** Offline replay is **open-loop**:
frames don't change in response to a different command. It can prove a
computed decision changed as intended; it can never prove the car takes a
different path. This is not a footnote — Round 2 shipped an
offline-verified "strict improvement" that drove *worse* on the real car.

**6. Separate diagnosis from fix when evidence is thin.** Round 4
deliberately shipped no driving-code change. Given (5), a speculative fix
on offline evidence would have repeated a mistake already paid for.

---

## 9. Talking points

Answers worth having ready, each anchored to something concrete above.

**"Tell me about a hard bug."** The corner failure (§6.1). Its value is
that *nothing was individually wrong* — a stale heading meets a
correct-by-design safety floor, and the emergent behaviour is the car
committing hard in the wrong direction. It survived three rounds of
debugging because every component in it was a validated fix for an earlier
real bug.

**"When did data change your mind?"** The cv2 benchmark (§6.3). The
intuition — "JPEG encoding on a Pi is slow, use the faster library" — was
confidently wrong; the measurement showed a 2× pessimisation, and the real
problem was that 90% of the traffic was duplicates.

**"A time you were wrong."** Round 2's learned-offset revision (§4).
Offline replay showed it strictly better; the real car drove worse. The
useful part is the resulting rule: know which of your tools are open-loop.

**"How do you handle a codebase you can break?"** Every Mission 2 mechanism
returns `None` when lane-following is off, and that was verified by
replaying real sessions before and after and confirming byte-identical
output. Non-negotiable when the failure mode is a physical crash.

**"Working within someone else's standards."** Reading `CONTRIBUTING.md`
changed the deliverable twice (§6.4): it mandated tests, which found a bug
benchmarking couldn't; and it told me the car-specific code shouldn't be
submitted at all, so the work was split rather than bundled.

**"Something you deliberately didn't do."** Retuning the temporal gate.
It's a carefully balanced mechanism with an explicit trade-off against
permanent lockout; retuning it blind from one section's replay risked
regressing recovery everywhere. Diagnosis, ranked candidates, no change.

---

## 10. Still open

Recorded honestly — these are known and unfixed.

- **Colour/lighting robustness** — the OAK-D's auto white balance can
  render yellow tape as cyan. Open since Round 1, the oldest unresolved
  issue.
- **The three Round 4 candidate fixes** — a stale-heading guard on the
  sharp-bend floor; blind-coast behaviour (freezing steering *and*
  continuing at throttle is what turns a lost line into a collision); and
  the temporal-gate forced-accept.
- **The plaza crossing** — a wide, visually ambiguous crossing with
  parking-stripe markings that has caused trouble across many sessions.
  One recent episode fits no known bug signature, in a window where replay
  actively *disagreed* with the recording (correlation went negative). No
  fix attempted — the evidence wasn't solid enough to name a cause without
  guessing.
- **The video fix has not been driven.** It is verified by unit tests and
  benchmarked over loopback, but has not been tested with a real browser
  over the car's own WiFi — which is the test that actually matters for a
  latency claim.
- **Tub corruption on hard power-off** — cutting power mid-drive loses
  buffered SD writes and can corrupt the datastore. Repaired twice with
  data preserved; the underlying cause is unaddressed.
