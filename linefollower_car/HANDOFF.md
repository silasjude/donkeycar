# Handoff: obstacle-maneuvering cone collisions (fixed and confirmed on
# two real test drives) — swerve-vs-lane-change behavior now documented,
# open design decision for the team below

## ⚠ READ FIRST — OBSTACLE AVOIDANCE IS CURRENTLY TURNED OFF (2026-07-29 evening)

`myconfig.py` now has **`OBSTACLE_AVOIDANCE_ENABLED = False`**, so
`manage_line.py` skips the whole stack (ObstacleDetector,
LaneOffsetCommander, ThrottleLimiter, ObstacleOverlay). **The car will
not stop or steer around real obstacles in this state — spot it
accordingly.** It was switched off to isolate line-following debugging,
not because a fault was found in this code. Nothing in
`obstacle_detector.py` / `obstacle_avoidance.py` was changed; flip the
flag back to `True` to restore everything below.

Context, since it is easy to misread the myconfig comment: the stack was
ON during test session `26-07-29_87` and fired swerves on 627 of 3225
frames, including a 149-frame `swerve_right` over the exact corner where
the car kept crashing (a pedestrian crosses frame around idx 172200).
That looked like the cause at first, and the myconfig comment was written
on that basis. Fuller replay disproved it: the swerve steers the car
**hard right** (|st| 0.95 with the stack vs 0.49 without) while the car
actually drove **left** into the planter, and the crash reproduces with
the stack disabled. The real cause was the line-follower's lane offset —
see the 2026-07-29 evening section of the line-follow notes. The stack
does still muddy line-follow traces (±0.35 lane-offset swaps on ~19% of
frames with `OBSTACLE_MONO_SLOPE` still uncalibrated), which is the
standing reason to leave it off during line tuning.

## TL;DR FOR TEAMMATES (as of 2026-07-27, late evening session)

- **NEW (6th update, evening): detector startup blindness fixed.** On
  every real recorded session start the detection corridor was blind for
  the first 1-3 frames (until LineFollower's first accepted fix seeds
  `x_f`) even though the cone's blob fully qualified from frame 1.
  `obstacle_detector.py` now runs a full-width "bootstrap" scan until
  the follower's first-ever lateral fix and reports the obstacle with
  `side=None` (never guesses a direction; the commander's existing
  fail-safe holds lane + slows). Validated by selftests (16/16 + 27/27)
  and a 3,398-frame old-vs-new replay over all four real cone sessions:
  bit-identical everywhere except exactly those blind startup frames.
  NOT yet validated by a new real drive. See the 6th update below.
- **The two real cone collisions from earlier today are fixed and
  confirmed.** Root cause was `line_following.py`'s throttle law
  flooring speed the instant steering saturates from a large lateral
  error — happened on EVERY test regardless of obstacle-avoidance
  config, because the car consistently starts each recording already
  significantly off-center. Fix: (1) `obstacle_avoidance.py` now falls
  back to a smaller in-lane swerve instead of attempting a full lane
  change when an obstacle is detected too close to complete one safely
  (`OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M`), and eases the offset in by
  distance instead of snapping to it; (2) `line_following.py`'s throttle
  floor is now higher specifically when steering saturation is
  positional-error-driven, not curvature-driven. **Two real test drives
  since this fix: zero collisions, one avoiding 2 cones, one avoiding
  ~4.** See the 4th update below for the fix details.
- **Open, unresolved design question for the team:** most cone
  encounters in real testing get the smaller in-lane SWERVE, not a full
  lane change, because they're detected too close (~0.7-0.85m) for a
  lane change to safely complete — this is the safety gate from the fix
  above working as designed. Investigated whether this is a fixable
  detection problem (it isn't — see the 5th update below) and it comes
  down to a real tradeoff: attempt lane changes closer-in (risks
  reproducing today's original crashes) vs. accept swerve-only for close
  cones (current behavior, safe, but not full lane changes). **Needs a
  team decision, not a code fix** — see the 5th update for the full
  evidence and options.
- **Separately, and lower priority per the user:** real lane-following
  instability (Mission 1/2, unrelated to obstacle avoidance) further
  down the track, in a visually cluttered section near a wall — one real
  bug found and fixed there, a second, bigger contributor found and NOT
  fixed. Full detail in `KNOWN_ISSUES.md`, not this file.
- Everything below this point is the full chronological investigation
  log, oldest content at the bottom. Read top-down for most-recent-first.

---

## UPDATE 2026-07-27 (6th update, evening): detector startup blindness
fixed — full-width "bootstrap" corridor until LineFollower's first
lateral fix, reported with side=None so no direction is ever guessed.
DEPLOYED to the files, NOT yet validated by a real drive (no drive
process was running this session — the next `manage_line.py drive`
picks it up automatically).

**What was wrong (measured, not theorized):** re-ran the time-accurate
replay against the actual first frames of ALL FOUR real cone recordings.
On every one, the first cone's blob already fully qualifies on the very
FIRST frame, but `_corridor()` had no `x_f`/`last_x` yet (the detector
part runs BEFORE LineFollower each tick, and `x_f` starts None until its
first accepted fix) and fell back to a corridor centered on frame center
192±85 — which missed the cone every time:

| Recording start | frame-1 cone blob | first frame detected (old) |
|---|---|---|
| idx 108545 (enc. A) | cx=22, 0.85m, area 2762 | 108548 (frame 4 — LF took 2 ticks to lock, `lf=coasting`) |
| idx 107772 (4th test) | cx=71.8, 0.796m | 107773 (frame 2) |
| idx 107001 (-0.75 test) | cx=71.1, 0.749m | 107002 (frame 2) |
| idx 106212 (-0.6 test) | 0.687m | 106213 (frame 2) |

The idx 108545 case kills any "just widen/re-center the default
corridor" idea: the cone sat at cx=22 and `x_f` itself converged to
**-21** (the tracked yellow line extrapolates past the left frame
edge) — no corridor centered anywhere sensible covers cx=22 without
being full-width. This is the same evidence trail as the 5th update's
"corridor hasn't initialized yet" finding, now fixed rather than just
documented.

**Why side must be None during bootstrap (not classified from frame
center):** at 108545 the cone (cx=22) is OWN-lane — right of the
tracked line at -21 — but far LEFT of frame center. A frame-center side
call would have read it as opposing-lane and committed `swerve_right`,
i.e. a swerve TOWARD the cone, and the commitment layer would then hold
that wrong direction for the whole encounter. `side=None` +
`present=True` instead lands in `_target_from_info()`'s existing
never-guess fail-safe (`blocked_holding`: hold lane, ease throttle), and
creates NO commitment, so the first localized frame afterwards commits
the normal (correct) maneuver.

**Changes, all live in the files now:**
1. `obstacle_detector.py` — `_corridor()` returns `(cx, half_w, locked)`;
   `locked=False` only when a LineFollower is wired but has never
   produced any lateral fix (`x_f` AND `last_x` both None — process
   start, or indefinitely if it never locks; post-stop reseed keeps
   `last_x` so that stays on the normal path). While unlocked, `run()`
   scans one full-width band (`OBSTACLE_BOOTSTRAP_HALF_WIDTH_FRAC=0.5`
   of frame width each side of center, myconfig-overridable) and
   returns `present`/`distance_m` with `side=None`,
   `status='obstacle_unlocalized'`, new key `corridor_locked=False`.
   All exclusion bands (yellow/white/blue) and the area/height gates
   apply unchanged during bootstrap. With NO LineFollower wired at all
   (selftest/calibrate CLI), behavior is exactly as before (centered
   corridor, side classified) — deliberate, so standalone use and the
   existing selftests keep their meaning.
2. `obstacle_avoidance.py` — no logic change needed: the
   present+side=None fail-safe branch already existed. Its comment no
   longer claims "shouldn't happen", and 3 new selftests pin the
   contract: holds lane + harsh floor, creates NO commitment, and the
   first localized frame after bootstrap still commits normally.
3. `myconfig.py` — `OBSTACLE_BOOTSTRAP_HALF_WIDTH_FRAC = 0.5` with a
   dated comment carrying the evidence summary.
4. `obstacle_detector.py` `calibrate` CLI now accepts multiple
   `<frame> <dist>` pairs and averages the slopes (backwards
   compatible), plus a friendly error for unreadable files — for
   whenever someone actually gets out there with the tape measure
   (option 3 of the 5th update's list; still not done, still needs
   physically measured reference distances).

**Validation (decision-logic only — the methodology trap section fully
applies, a real drive is still the only physical proof):**
- Selftests: detector 16/16 (5 new bootstrap tests), avoidance 27/27
  (3 new).
- Time-accurate old-vs-new replay over ALL FOUR real cone sessions
  (idx 106212-107000, 107001-107771, 107772-107926, 108545-110617 —
  3,398 frames with images; the last ~390 catalog records of the final
  session have no image files on disk, skipped identically by both
  runs): **bit-identical presence/side/distance/status on every frame
  except exactly the blind startup frames** (1+1+1+3 of them), where
  old=clear/blind, new=present at the correct distance with side=None
  → `blocked_holding`. Every documented encounter unchanged, including
  encounter C's full lane change at 3.91m (commits at idx 109514 in
  this replay) and both swerve encounters.
- The handover is seamless in replay: e.g. 108545-108547
  `obstacle_unlocalized`/`blocked_holding` (speed_scale easing down
  ~0.79→0.51), then 108548 commits `swerve_left` — the SAME frame the
  old code first noticed anything at all.

**Known, accepted behavior change at drive start:** with a cone already
in view at start, the first 1-3 frames now pre-slow (speed_scale dips
to ~0.5-0.8 for a few tenths of a second before the maneuver's 1.0
floor takes over and the EMA recovers). At a from-standstill launch
this is a slightly gentler start, not mid-maneuver starvation — but it
is a real difference from the two zero-collision drives' exact
throttle profile, so watch the first post-change drive for it.

**What this deliberately does NOT change:** the 5th update's open team
decision stands untouched. Bootstrap detection fires earlier in TIME
(frame 1) but at the same DISTANCE (~0.7-0.85m — the cone really is
that close at recording start), so close encounters still get the
swerve, not a lane change; the gate, offsets, and all thresholds are
unchanged. Replay artifacts for this session were kept off the project
dir (session scratchpad) and deleted after; backups of all three
touched files are in `backups/` as `*.pre-bootstrap-corridor.*`.

---

## UPDATE 2026-07-27 (5th update): swerve vs. lane-change — root cause is
starting distance, not a fixable detection threshold (open question for
the team, no code changed this round)

**Context:** after the 4th update's fix was deployed, the user ran two
more real test drives. First: 2 cones avoided, zero collisions. Second:
~4 cones avoided, zero collisions — but the user reported the car
"drove in the middle lane through them" instead of actually changing
lanes, and asked for cleaner (i.e., real lane-change) maneuvering,
explicitly deferring the lane-following issue (KNOWN_ISSUES.md) to
later.

**Replayed the second test** (session `26-07-27_77` continuing across
`catalog_108/109/110.catalog`, idx 108545-110617 — note a real ~30s gap
in the middle at idx 108675, the recording was paused, not a process
restart; same `_session_id` throughout) against the live (fixed) code.
Found 3 distinct obstacle encounters:

| Encounter | idx (commit) | Distance at commit | Result |
|---|---|---|---|
| A | 108548 | 0.85m | swerve (below the 1.0m gate) |
| B | 108979 | 0.85m | swerve (below the 1.0m gate) |
| C | 109514 | 3.91m | **full lane change** (comfortably above the gate) |

Pulled the actual frames for all three — confirmed visually there are
at least 3 separate cone clusters (3-4 cones each, same blue-tape-marker
style as every earlier test) spread across a long stretch of track, not
one continuous obstacle.

**Encounter C proves the lane-change mechanism itself works correctly
when given enough distance** — clean, full `-0.75` lane change, no
issue. The question is why A and B don't get the same treatment.

**Checked whether A/B's close detection (0.85m) is a fixable
blob-detection threshold problem — it is NOT.** Dumped the actual blob
geometry (not just the derived `obstacle/info`) at idx 108545-108549
(encounter A's first few frames, including the very first frame of the
whole recording): the cone's blob already fully qualifies against
`OBSTACLE_MIN_BLOB_AREA_FRAC`/`OBSTACLE_MIN_BLOB_HEIGHT_PX` (area~2761,
height~63, both comfortably above threshold) starting at the FIRST
frame of the recording, computing to the same 0.85m the whole time.
`present` doesn't flip `True` until idx 108548 not because the blob is
too small/far to detect, but because the detection corridor (centered
on the car's tracked lane position, `x_f`) hasn't initialized yet on
frame 1 and only catches up a few frames later. **Loosening the blob
thresholds would not create distance that isn't there** — the cone is
simply already ~0.85m away and essentially fully in view at the moment
each recording starts. That reflects either where the car is physically
positioned/pointed when the operator hits record, or genuinely where
that cone cluster sits relative to a turn earlier in the course — not a
software detection lag. (This resolves the ambiguity in the very first
open question raised in this file, all the way at the bottom: it's the
course-setup half of that either/or, not the threshold half.)

**No code was changed this round.** The remaining options, all with
real tradeoffs, for the team to decide:
1. **Lower `OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M`** (currently 1.0m) so
   lane changes get attempted at ~0.7-0.85m too. This is the EXACT
   distance/condition that caused BOTH of today's original real
   collisions (see the 3rd/4th updates below). Today's other fixes
   (higher throttle floor during positional-error saturation, offset
   easing by distance) should help, but that combination is
   **unvalidated at this close a range** — no real test has tried a full
   lane change this close since those fixes landed.
2. **Accept swerve-only for cones detected this close** (current
   behavior) — safe, zero collisions across two real tests, but not a
   "real" lane change; only clusters with a naturally longer sightline
   (like encounter C) get one.
3. **Calibrate `OBSTACLE_MONO_SLOPE`** (still an uncalibrated
   placeholder everywhere in this project — see `obstacle_detector.py`'s
   own module docstring) using its built-in `calibrate` CLI and a real
   measured reference distance. It's possible the "0.85m" reading
   understates true real-world distance and a calibrated value would
   naturally read farther, changing the gate outcome without changing
   any threshold. Untested — needs one real reference photo at a known
   physical distance, which needs the user/team physically present with
   a tape measure, not something resolvable from replay alone.
4. **Course/operational change**: start the recording/drive with more
   physical clearance before the first cone cluster, if that's
   consistent with the course's intended design. Not a code change at
   all.

Replay artifacts (`_replay_4cones.py`/`.jsonl`, `_blob_probe2.py`) were
cleaned up off the Pi after this investigation, per this file's own
convention.

---

## UPDATE 2026-07-27 (4th update): FIX IMPLEMENTED AND DEPLOYED, pending
a real test drive. User explicitly authorized touching `line_following.py`
for this (see the 3rd update below for why obstacle_avoidance.py alone
wasn't going to be enough). Two changes, both live on the Pi right now
(process restarted at 14:57 PDT, PID 5896, running the new code):

**obstacle_avoidance.py:**
1. `OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M=1.0` (new) — a FRESH (not yet
   committed) decision only starts a full lane change when the obstacle
   is at least this far away; closer than that it falls back to the
   smaller swerve instead, which needs far less time/distance to
   complete. Both real -0.75 crashes were detected already at ~0.75-0.85m,
   well inside this — so under current (still-uncalibrated) detection
   ranges, swerve is now effectively the default response, not lane
   change. A lane change already committed to is not aborted by this if
   distance later drops below it — only the initial choice is gated.
2. `lane_offset`/`curve_gain` now EASE toward the full lane-change
   target by distance (`_offset_scale_for`/`_eased_offset`, same shape
   `_speed_scale_for` already used for throttle) instead of snapping to
   it the instant a lane change is chosen — applies both to a fresh
   decision and to a committed lane change as distance keeps closing
   after commit.
3. The commitment/hysteresis layer (`_avoid_dir`) now also covers
   swerve, not just lane change (new `_avoid_kind` state). Needed
   because swerve became the common case per (1) above, and swerve
   previously had NO protection against the same per-frame side-flicker
   problem that originally motivated committing lane changes — this
   would have been a new, undocumented gap if left alone.
4. 24/24 `python3 obstacle_avoidance.py selftest` checks pass (up from
   22 — 2 new tests added for the min-distance gate and swerve
   commitment; several existing tests updated since they'd hardcoded
   the old "snap to full offset" behavior).

**line_following.py** (the hands-off file — touched only after explicit
user sign-off):
1. New `LF_THROTTLE_MIN_LANE_OFFSET=0.18` (vs `LF_THROTTLE_MIN=0.1`).
   The throttle law (`slow = min(1.0, max(abs(steering), abs(hterm)/
   h_clip))`) can't distinguish "steering hard because the track is
   genuinely curving" (hterm-driven — untouched, still floors to
   `LF_THROTTLE_MIN` as before) from "steering hard because the lateral
   target jumped a lot" (error-driven — e.g. an obstacle-avoidance
   commit, or simply a large `x_f` error already present when a
   maneuver starts). Now uses the higher floor specifically when
   `abs(error) > LF_STEER_I_ERR_GATE` (reusing the SAME "not yet
   converged" threshold `LF_STEER_KI`'s anti-windup gate already uses,
   rather than inventing a new one) — this is exactly the condition
   found in both real crashes: `x_f` was already ~0.6-0.7 normalized
   error off target from the very first frame of each recording,
   independent of anything obstacle_avoidance.py commanded.
2. No selftest exists for this file (none did before either) — validated
   by re-running the time-accurate replay methodology against BOTH real
   -0.75 recordings with the new code, in an isolated `_fixtest/`
   directory first (to avoid touching the live file before confirming
   the fix even loads/runs), then promoted to live only after
   confirming: throttle during previously-floored (0.100) windows is
   now consistently 0.18-0.29 instead, swerve status no longer flickers
   frame-to-frame despite noisy `side` readings, and normal
   curve-slowing behavior (large `hterm`, small `error`) is untouched by
   inspection of the gate condition. **This is decision-logic
   verification only** — see the "Critical methodology trap" section
   below, which fully applies here too: replaying the same fixed camera
   frames with different code can prove the new logic fires as intended,
   it CANNOT prove what the car will physically do differently in the
   real world. Only a new real test drive can do that.

**What this does NOT fix:** why `x_f` starts ~65-71px off-center in the
very first frame of these tests in the first place (see the 3rd update's
item 3, still open). Both changes above only make the car MORE ROBUST to
that large initial error once it happens — they don't address its root
cause. If a new real test still shows the car pointed noticeably
off-line before the obstacle course even starts, that's the next thing
to chase, separately from anything in this update.

**Cleanup:** `_fixtest/` and all `_verify_*`/`_obstacle_avoidance_new.py`
temp files removed from the Pi after validation, per this file's own
convention. The old drive process (PID 5276, running the pre-fix code)
was killed and replaced with a fresh one (PID 5896) so the fix is
actually live for the next test — output logged to
`/home/pi/mycar/_drive_20260727_new.log`.

---

Written 2026-07-27. Read this before touching `obstacle_detector.py` /
`obstacle_avoidance.py` / `myconfig.py` again — it captures evidence and
dead ends from three (now FOUR — see the 2026-07-27 second update, below
the first one) real test drives so you don't have to re-derive them.

## UPDATE 2026-07-27 (a 4th real test, same -0.75 config, same course):
user reported "it immediately ran into the first cone." Replayed it
(session `26-07-27_77`, appended to `catalog_107.catalog`, idx
107772-107926, process PID 5276 started 14:25 PDT — config unchanged,
still `OBSTACLE_LANE_CHANGE_OFFSET_LEFT=-0.75`). This **reproduces and
sharpens the same root-cause mechanism**, it doesn't contradict it:

- t=0.0s (idx 107772, first frame): car starts positioned at the
  canonical course start (visually confirmed — same two-cone framing as
  every previous test's first frame, both cones on their blue-tape
  markers).
- **t=0.048s (idx 107773, the SECOND frame of the whole recording):**
  obstacle already reads `present=True, dist=0.796m, side=right` and the
  commander instantly commits `avoid_dir=left`, jumping `lane_offset`
  from base (0.5) straight to -0.75.
- **t=0.148s (idx 107775, the 4th frame, ~0.15s in): steering is already
  pegged at exactly -1.00 and throttle is already at the LineFollower's
  exact floor (0.100).** This is faster than any prior test — the
  previous -0.75 test took ~0.2s to peg from a moving start; this one
  pegs from what looks like a standing start, 3 frames in.
- **t=0.0s-2.7s (idx 107772-107825): the recorded frames are visually
  static** — pulled and compared idx 107772 and idx 107790 side by side,
  they're indistinguishable (same cone sizes/positions, same
  background). `x_f` barely moves (64.7→65.2px) this whole time despite
  throttle nominally being 0.100, not 0. The car is not making visible
  progress while pegged.
- t=2.7s-4.6s (idx 107825-107852): `x_f` climbs rapidly (65px→305px) —
  the car finally does move/turn during this window, closing in on
  cone 1. Minimum distance during the pass: **0.31m** (idx 107852,
  t=4.161s) — closer than the previous -0.75 test's 0.38m and closer
  than the -0.5/-0.6 tests' 0.53m. Pulled this frame directly: cone 1 is
  dead-center, filling most of the frame, car aimed straight at it.
- t=4.6s onward: `dist`/`side` transition into the same oscillating,
  mostly-`present` pattern as the previous test's encounter with cone 2
  (values bouncing in the 1.0-2.0m band, `avoid_dir` never releasing).
  Recording ends at idx 107926 (t=7.86s — a much shorter recording than
  the previous ~39s test, consistent with the operator stopping it
  right after feeling/seeing the hit). **Last frame of the recording
  shows cone 1 filling the left of frame again, right against the
  camera** — same terminal signature as the previous test's stuck state,
  reached in a quarter of the time.

**This is the same steering-lock → throttle-floor mechanism described in
the root-cause hypothesis below, observed even more starkly: the
instant `lane_offset` jump from 0.5 to -0.75 on first detection pegs
steering and floors throttle within 3 frames (~0.15s) of the recording
starting, before the car has covered any real distance, and it stays
visibly motionless for ~2.7s before finally moving and passing cone 1 at
0.31m — the closest contact of any test so far.** This is evidence
AGAINST "the specific offset value is the fix" (a 4th data point, same
conclusion as the -0.5/-0.6/-0.75 comparison already documented below:
different offsets, same-magnitude graze) and evidence FOR treating the
root-cause hypothesis as the real target: something needs to change
about how abruptly `lane_offset` is commanded on commit, not which
number it's commanded to. No config was changed and no fix was applied
in this session — this is analysis only, so the car is still running
the same -0.75 config that just produced this 4th collision.

Replay/cleanup note: recreated the replay script again (parametrized to
this session's own idx start, 107772, learned from the catalog-boundary
gotcha already documented below) and deleted it plus its output off the
Pi afterward, per this file's own convention.

## UPDATE 2026-07-27 (3rd update): read `line_following.py`'s actual
steering/throttle code line-by-line to check whether the bug is really
where the "Root-cause hypothesis" section below says it is. **Partial
correction: the lane_offset ramp is NOT the primary driver of the
instant steering peg. It's rate-limited correctly and is doing its job.
The instant peg comes from `x_f` itself already sitting far off-center
in the very first frame of these recordings, before the ramp has moved
meaningfully at all** — which shifts where a real fix would need to
live. Read `_steer_error()` (line 1399), the ramp update (lines
1580-1592), the raw steer command (line 1639), `_apply_steering()`'s
slew limit (lines 1478-1483, `LF_STEER_SLEW=6.0` at line 360), and the
throttle law (lines 1648-1649) directly, and checked the arithmetic
against both replayed tests' actual frame-1 telemetry:

- **The ramp is genuinely rate-limited, not a step.** `step = dt /
  lane_ramp_sec` (line 1590) advances `_lane_applied` by at most
  `1/1.5 ≈ 0.667` offset-units per second — reaching the full -0.75
  target takes **~1.1 real seconds**, confirmed in myconfig.py's own
  comment ("reaches its full -0.6 target in ~1.3s, matches
  LF_LANE_RAMP_SEC"). In the first 3-4 frames (~0.15-0.2s) where
  steering was already observed pegged at -1.00 in both replayed tests,
  `_lane_applied` can only have moved by ~0.07-0.13 of the way from 0 —
  nowhere near -0.75. The "large instant P-error from jumping the
  offset" mechanism the hypothesis below describes isn't what's
  actually happening on frame 1-4; the ramp hadn't gotten anywhere yet.
- **What IS happening: `x_f` starts both recordings already far
  off-center, before the ramp has moved.** Frame-1 telemetry, both real
  tests: idx 107001 test → `x_f=70.9`px; idx 107772 test → `x_f=64.7`px
  — out of a 384px-wide frame (center=192, and even the STEADY-STATE
  lane-offset-driving target, `LF_LANE_OFFSET=0.5` base, sits around
  x≈122 once fully ramped). With `_lane_applied≈0` on frame 1,
  `_steer_error()`'s `target` is just frame-center (192) — so error ≈
  (65-192)/192 ≈ -0.66 to -0.69 **from the very first frame, before
  obstacle-avoidance has done anything.** `LF_STEER_KP=1.2` × that error
  alone ≈ -0.79 to -0.83 — already past saturation before `kd`/`ki`/
  heading terms are even added. (Confirmed `ki` isn't a factor here
  either: `LF_STEER_I_ERR_GATE=0.25`, line 1630, and this error is
  ~2.5-2.8x that gate, so the integral correctly stays frozen — this is
  a pure-P saturation.)
- **`LF_STEER_SLEW=6.0`** (comment: "full lock to full lock in ~1/3s")
  means ANY raw command this far past ±1.0 walks `self.steering` to the
  hard limit in about **0.167s** — 3-4 frames at ~20Hz — regardless of
  what produced the raw command. This is exactly the timing observed in
  both tests' telemetry, and it would happen identically from an
  off-center start under plain Mission-2 lane-offset driving, with NO
  obstacle avoidance involved at all.
- **The throttle law (line 1648-1649) couples directly to
  `abs(self.steering)`, not to the underlying cause or to whether
  progress is being made:** `slow = min(1.0, max(abs(self.steering),
  abs(hterm)/h_clip))`, `throttle = th_max - (th_max-th_min)*slow`. The
  instant `self.steering` reaches ±1.0 (~0.167s in, per above),
  throttle floors to `th_min=0.1` and STAYS floored for as long as
  `self.steering` stays near the peg — which in turn stays pegged for
  as long as the raw command (driven by however far `x_f` is from
  `target`) remains large. Once the -0.75 lane-offset ramp DOES start
  moving `target` further away over its ~1.1s climb, it keeps the error
  large for even longer, extending the pegged/floored window — so the
  ramp isn't blameless, it's just not the *initial* trigger the
  hypothesis below describes; it's more like fuel added to a fire `x_f`
  already started.

**Practical implication for where a fix belongs:** this reframes the
bug as less "obstacle_avoidance.py commands too abrupt an offset" and
more "line_following.py's throttle law floors speed on ANY large
steering error, however it arose, for as long as the error stays large
— and these test starts consistently begin with a large error already
present before obstacle avoidance ever engages." Two independent,
non-exclusive angles worth considering next, neither implemented:
  1. **On the obstacle_avoidance.py side** (no sign-off needed to touch
     this file): the original idea of ramping `lane_offset` in from
     farther away (open question 4 below) would still help the
     *compounding* part (keeping error large as the offset climbs), but
     per the above, won't fix the *initial* peg — that's already
     saturated from `x_f`'s starting position before obstacle_avoidance
     ever writes anything.
  2. **On the line_following.py side** (needs explicit sign-off per
     project convention — NOT done, this is analysis only): the
     throttle law's direct 1:1 coupling to `abs(self.steering)` is what
     turns a large error, from any source, into an immediate, total
     speed collapse. Something less punishing there (e.g. not flooring
     quite as hard, or reacting to how fast the error is closing rather
     than just its magnitude) would give the car more speed budget to
     actually execute a turn instead of crawling through it — but this
     is exactly the file the project's own convention says not to touch
     without the user's explicit go-ahead, and it's used by Mission
     1/2 as well as obstacle avoidance, so a change here has a much
     bigger blast radius than anything in obstacle_avoidance.py.
  3. Separately worth checking (not done): WHY does `x_f` start ~65-71px
     off in the very first frame of these tests, consistently across two
     independent real recordings? Could be the car's actual start-line
     placement/orientation (operator alignment before hitting record),
     or could be genuine course geometry (where the nearest yellow dash
     sits relative to the two cones). Hasn't been distinguished — would
     need either a frame from before the process restart (not available
     in either recording checked) or a description of how the car is
     physically positioned before each test.

## UPDATE 2026-07-27 (later same day): the course has TWO cones, not one
— open questions 1-3 below are now resolved by re-running the time-accurate
replay (recreated from scratch — the previous `_replay_*` scripts had
already been cleaned up, per this file's own convention) and cross-checking
its output against the actual recorded images (not just telemetry) at
`data_line/images/107xxx_cam_image_array_.jpg` from the -0.75 test
(session `26-07-27_76`, catalog_107). This is the single most important
correction to the mental model in this file: **every prior analysis here,
including the "root-cause hypothesis," was written as if there's one cone
on the course. There are two, spaced a few meters apart on the same
track segment.** This changes what the "12-second constant distance"
reading and the final "stuck" state actually are. No config was changed
and no new physical test was run — this is a re-analysis of the same
-0.75 recording described below, just with tighter frame-level evidence.

- **Q1 (is detection happening as early as possible)**: still open, same
  conclusion as before. The very first frame of the -0.75 test's own
  recording (idx 107002, 0.05s in) already shows cone 1 large and close
  in frame with cone 2 visible further down the track. That's the actual
  first frame the process captured after being restarted with the new
  config — there's no earlier footage in this recording to check
  against. Still can't distinguish "course is set up with the first cone
  close to the start line" from "detection could fire earlier" without a
  test where the car starts farther back.

- **Q2 (what is the ~12s constant `1.5243902439024384` reading,
  idx≈107180-107493) — RESOLVED: it's a real second cone, correctly
  detected, not a bug.** Dumped the actual winning blob geometry
  (bbox/cx/bottom_row/area) for frames across this window: there are
  reliably exactly two qualifying blobs every frame throughout —
  (a) a huge blob (area ~8000-10500px, bbox `(0, 0, ~78-97, 119)`, i.e.
  spanning the ENTIRE ROI height from its top edge to the frame's last
  row) hugging the far left of frame (cx~28-44) — this is cone 1, now
  pressed directly against the car/camera after the graze at t≈7.1-7.4s
  documented below. It's far enough left that it falls outside the live
  left/right corridor bands (centered on `x_f`≈275-290 in this window),
  so it does NOT drive the `side='right'` classification.
  (b) a small, stable blob (area ~1100-1150px, bbox roughly
  `(266-275, 0, 37-40, 49-50)`, bottom_row pinned at 145-146,
  cx~279-293) that DOES fall inside the corridor and drives
  `side='right'`. Visually confirmed directly from the recorded frames
  (e.g. `107146_cam_image_array_.jpg`, `107300_cam_image_array_.jpg`):
  this is a **second, separate cone standing further down the track**,
  visible in-frame from the very first frame of the recording onward,
  off to the side of cone 1. Its distance/position reads constant for
  12.7s not because of a detection glitch but because **the car is not
  making real forward progress toward it** — it's physically stuck near
  cone 1 (see Q3). This is a real obstacle, correctly and continuously
  detected; the bug (if there is one) is entirely on the
  vehicle-dynamics side, not detection.

- **Q3 (what happened at t≈25-27s, the "chaos" event) — RESOLVED, and
  it's worse than previously described: the car gets stuck TWICE in this
  one test, back to back, once on each cone.** Corrected/tightened
  timeline for the -0.75 test (re-anchored at idx 107001 = the process's
  actual first frame — see the new replay gotcha noted below; the
  original t-values already below were right, this just adds frame-level
  precision):
  - t≈7.1-7.4s: closest pass/graze of cone 1, distance bottoms at
    **0.38m** (slightly closer than the 0.407m first reported — same
    event, tighter replay).
  - t=7.6s: distance jumps to 5.68m — matches the original analysis's
    exact quoted number (good cross-check that this replay reproduces
    the same underlying event as before).
  - t≈9-12s: side flickers left/right frame to frame as the corridor
    (tied to noisy `x_f`) straddles the boundary between cone 1's
    proximity and cone 2 coming into view.
  - **t=12.05s-24.83s (12.7s, idx 107239-107493): dist and side are
    bit-for-bit identical every single frame** (`1.5243902439024384`,
    `'right'`) even though `x_f` is visibly drifting (~292→~275px) and
    steering/throttle are NOT pegged (steer -0.34 to -0.48, throttle
    ~0.2-0.24) — i.e. the car is actively steering and commanding real
    throttle this whole time but not advancing toward cone 2 at all.
    Frames through this whole window (`107180`, `107235`, `107300`,
    `107400`, `107480`, `107493`) all show cone 1 filling the left of
    frame in essentially the same position/size and cone 2 unchanged
    further down the track. **The car is wedged against/immediately
    beside cone 1 for the entire 12.7s stretch, unable to make forward
    progress, despite live steering/throttle commands that look
    superficially like normal driving.**
  - t≈24.9-25.13s (idx 107494-107500): cone 1 leaves the frame (view
    swings away from it — car breaks loose or rotates clear of it);
    `side`/`dist` briefly go `None`.
  - t=25.13-27.1s (idx 107500-107540): brief, apparently normal-looking
    recovery — `lf_status` flips to `white-tracking` transiently,
    steering/throttle vary normally (not pegged), and by idx 107540 the
    frame shows a clean wide shot of BOTH cones upright at their
    original blue-tape marker positions — composition nearly identical
    to the very first frame of the recording. This looked like a full
    recovery when first pulling that one frame. **It wasn't** — see
    next.
  - **t=27.13s-38.66s (idx 107543 through the LAST frame of the
    recording, ~11.5s, i.e. the entire remainder): steer pegs at exactly
    -1.00, throttle pegs at exactly the 0.100 floor again, `x_f`
    flatlines around 65px (a big jump left from the ~275-290 it held all
    through the cone-1 encounter), and `dist` locks onto another
    bit-identical constant, `0.7961783439490444`, for the rest of the
    recording.** Frames at idx 107600 and the final frame 107770
    (t=38.66s, literal end of the recording) are visually identical to
    each other — same static wide shot of both cones, both still
    upright, neither knocked over. The car is not crashed into a toppled
    cone; it's just stopped/stalled somewhere near cone 2 and never
    moves again for the rest of the recording. `avoid_dir` stays `left`
    (never releases) for the entire ~39s of this recording — because
    with two cones close together, at least one of them reads within
    `OBSTACLE_SLOW_DIST_M` (2.2m) `present=True` almost continuously, so
    the 4s clear-debounce never gets a long enough gap to fire even
    once.

  **Net correction to the root-cause section below: the steering-lock →
  throttle-floor mechanism isn't a one-time failure to complete a single
  maneuver, it's a state the car can fall into repeatedly against
  successive obstacles in the same run — with two cones close together
  on this course it did so twice, back to back, ending the recording
  stuck near the second cone rather than recovering after the first.**
  Neither cone was captured as toppled/moved from its base marker in the
  images — this reads as the drivetrain/steering genuinely unable to
  complete the commanded turn, not a violent collision.

  **New replay gotcha, same family as the session-id one already
  documented below:** the first record in `catalog_107.catalog` (idx
  107000) belongs to the PRIOR test — there's a ~925s timestamp gap
  between idx 107000 and 107001. This catalog's own drive actually
  starts at idx 107001 (confirmed against PID 4644's start time,
  13:58:42 PDT). Anchoring `t=0` on idx 107000 instead silently shifts
  the whole derived timeline by +925s. Same root cause as the
  session-id note — a session/catalog boundary doesn't cleanly line up
  with where a real drive starts/stops — it just resurfaced one catalog
  file later than where it was first found.

## Project context

UCSD RoboCar (DonkeyCar-based RC car) at `pi@ucsdrobocar-DSC-T1` (password
`master`). Working on obstacle maneuvering (avoid cones / oncoming cars
while staying on the course), one of three driving modes (line following,
lane following, obstacle maneuvering) that will eventually be orchestrated
by a VLM. This file covers obstacle maneuvering only.

## Architecture (all in `/home/pi/mycar/`)

- `line_following.py` — Mission 1/2 line/lane follower. Treated as
  fragile/hands-off (extensive prior tuning history) — **do not edit
  without explicit permission**, tune via its `myconfig.py`-exposed
  knobs instead. Exposes mutable instance attributes that
  `obstacle_avoidance.py` hooks into every frame without ever touching
  this file: `lane_offset` (target), `curve_gain`, `_lane_applied`
  (internally time-ramped toward `lane_offset` over `LF_LANE_RAMP_SEC`,
  default 1.5s), `x_f` (EMA-filtered lateral position, what it steers
  on), `last_x` (raw, noisy), `status`, `lane_width_l/_r`.
  - **Throttle law** (line 1649/1703):
    `self.throttle = self.th_max - (self.th_max - self.th_min) * slow`
    where `slow = min(1.0, max(abs(self.steering), abs(hterm)/h_clip))`.
    This means **whenever steering pegs at ±1.0, throttle independently
    drops to `th_min` (0.1)** — regardless of anything
    `obstacle_avoidance.py` does. This is the likely root mechanism
    behind the current bug (see below).
- `obstacle_detector.py` — monocular (no depth; OAK-D depth was
  abandoned 2026-07-26 after repeated undervoltage crashes, see
  `myconfig.py`'s `OAKD_DEPTH=False` comment) sensing. Ground-plane
  ranging (`distance = 1/(OBSTACLE_MONO_SLOPE * (row - horizon_row))`,
  monotonic but **not calibrated** — `OBSTACLE_MONO_SLOPE=0.04` is still
  a placeholder; `python3 obstacle_detector.py calibrate <frame> <dist>`
  exists but has never been run for real). Detection is absolute-LAB-
  chroma foreign-blob (not local-background-relative — that
  self-contaminated on large near objects), with hue exclusion bands for
  the track's yellow tape, white boundary, and a permanent blue ground
  marking. `_corridor(w)` centers the detection corridor on
  `self.lf.x_f` (falls back to `last_x`), sized from learned lane width
  or `OBSTACLE_LANE_HALF_WIDTH_PX=90` default.
- `obstacle_avoidance.py` — decision layer. `LaneOffsetCommander` reads
  `obstacle/info`, decides swerve / lane-change / stop, writes
  `lf.lane_offset` / `lf.curve_gain` directly, commits to a direction
  (`_avoid_dir`, 'left'/'right'/None) once chosen and holds it
  regardless of per-frame side flip-flop, releasing only after
  `OBSTACLE_CLEAR_DEBOUNCE_SEC` (myconfig: 4.0s) of continuous
  `present=False`. `ThrottleLimiter` multiplies throttle by
  `plan['speed_scale']`.
- `manage_line.py` — wires `ObstacleDetector → LaneOffsetCommander →
  LineFollower → ThrottleLimiter → ObstacleOverlay` in that `V.add()`
  order, gated on `cfg.OBSTACLE_AVOIDANCE_ENABLED`.
- `myconfig.py` — all tuning overrides, heavily dated/annotated. Treat
  every comment block as a mini changelog — read them before changing
  the same knob again.
- `KNOWN_ISSUES.md` (already on the Pi) — documents a **separate,
  confirmed-unrelated** bug: full-lock steering into a wall/planter well
  past the obstacle course, in `line_following.py`'s own tracking, not
  obstacle-avoidance code. Explicitly deferred, do not conflate with
  this file's bug.

## Selftest / replay tooling

- `python3 obstacle_detector.py selftest` / `python3 obstacle_avoidance.py
  selftest` — synthetic-image and synthetic-state unit tests, no camera
  needed. Both currently pass in full.
- **Time-accurate replay** is the only way to check real footage:
  monkey-patch `time.time` in both modules to return the recording's
  real `_timestamp_ms` values, so internally-computed `dt` (ramps, EMAs,
  debounce) matches real pacing. Naive frame-by-frame replay (no
  pacing) gives WRONG ramp-convergence conclusions — confirmed the hard
  way earlier in this project.
- DonkeyCar rolls catalog files periodically; a single drive session can
  span multiple `catalog_N.catalog` files under the same `_session_id`.
  Check adjacent catalogs, not just the newest file, or you'll silently
  replay only half a session.
- **A drive session's `_session_id` can stay the same across process
  restarts** (confirmed today — killing and relaunching
  `manage_line.py drive` did NOT create a new `_session_id`, it just
  kept appending to the same one with a large timestamp gap). Don't
  assume a new session id means a new drive, or vice versa — always
  check for large `_timestamp_ms` gaps between consecutive `_index`
  values to find where a real drive actually starts/stops.
- **Config is loaded once at process start.** If you push a
  `myconfig.py` change, any already-running `manage_line.py drive`
  process is still running the OLD values. Always check
  `ps aux | grep manage_line` and compare its START time against the
  timestamp of the frames you're about to trust, before believing a
  replay or a real test reflects the config you think it does. (Verify
  the actual epoch time of a frame with
  `date -d @<timestamp_ms/1000>`.)
- SSH pattern used throughout (interactive password, no key configured):
  ```
  expect -c '
  set timeout 30
  spawn ssh -o StrictHostKeyChecking=no pi@ucsdrobocar-DSC-T1 "<command>"
  expect "password:"
  send "master\r"
  expect eof
  '
  ```
  Same pattern with `spawn scp <local> pi@ucsdrobocar-DSC-T1:<remote>`
  for pushing files. **Gotcha:** `$(...)` inside the quoted ssh command
  gets eaten by tcl's own substitution — write a small `.sh`/`.py` file,
  scp it over, then `ssh ... "bash script.sh"` instead of trying to
  inline shell substitutions in the spawn string.
- The Pi's Python env is a venv at `/home/pi/env/bin/python3` — NOT the
  system `python3` (which lacks `cv2`). Always invoke replay scripts
  with the full venv path.

## ⚠️ Critical methodology trap (read this before replaying anything)

**Replay can verify decision logic against real footage. It CANNOT tell
you what would have physically happened with a config value that wasn't
actually live during that recording.** The camera frames are fixed —
already captured. Feeding a different `OBSTACLE_LANE_CHANGE_OFFSET_LEFT`
into a replay of the same session only changes what the code *decides*
this time; it can never change what the car *actually did*, because
that's baked into the recorded images. Concretely: replaying session
`26-07-27_76`'s first pass with -0.5, -0.6, and -0.75 hypothetically
injected all produced the **exact same** minimum distance
(`0.5274261603375526`, to 16 significant digits) and the **exact same**
`x_f` trajectory. That's not evidence the offset doesn't matter — it's
proof the replay was reusing the same frames regardless of the
hypothetical config. Only a genuinely different real test (or an
apples-to-apples comparison across two *different* real recordings that
each actually ran with their respective config, verified via the
process-restart-timing check above) is valid evidence.

## Bug history (fixed, chronological)

1. Blue-tape/ground-marking false positive → excluded by hue band.
2. Corridor jitter from using raw `last_x` instead of filtered `x_f`.
3. No commitment/hysteresis → added `_avoid_dir` commit state machine
   (user's own suggested fix: "outer lane then inner lane" = automatic
   lane change).
4. Lane-change throttle starvation → added
   `OBSTACLE_LANE_CHANGE_MIN_SPEED_SCALE` separate from the
   `blocked_holding` floor.
5. Detection triggering too late → raised `OBSTACLE_SLOW_DIST_M` to 5.0
   (though see the "still open" section — this may not have actually
   changed when detection *starts*, only when the *slow* branch logic
   nominally applies; the placeholder mono-slope makes 5m trigger
   basically immediately in practice).
6. Premature commitment release from flaky far-range detection → raised
   `OBSTACLE_CLEAR_DEBOUNCE_SEC` to 4.0.
7. Swerve using the harsh `blocked_holding` throttle floor instead of
   the lane-change floor → fixed, swerve now uses `lc_speed` too.
8. Pure pivoting instead of translating at offset -0.75 → diagnosed as
   throttle starvation (fixed by #7), softened offset to -0.5 as a
   belt-and-suspenders measure at the time.

All of the above are confirmed via selftest + time-accurate replay of
their respective real sessions.

## Current bug: still hits the first cone (UNRESOLVED)

Three real test drives, three different `OBSTACLE_LANE_CHANGE_OFFSET_LEFT`
values, three real collisions:

| Test | Offset | Real-world result |
|---|---|---|
| session `26-07-27_75` | -0.5 | "made it past the cone... issues further down" (later found unrelated, see KNOWN_ISSUES.md), but replay min distance during pass = 0.527m |
| session `26-07-27_76`, catalog_106 (idx 106212-107000) | -0.6 | "grazes the cone before continuing" — replay min distance = 0.527m (**identical** to -0.5 test) |
| session `26-07-27_76`, catalog_107 (idx 107001-107771) | -0.75 | "crashed into the first cone" — **worse**, see below |

The -0.75 test (most recent, most severe) is the one to start from. Full
timeline from time-accurate replay (config confirmed live: process PID
4644 started 13:58:xx PDT, matches the recording's first frame timestamp
13:58:42 PDT):

- t=0.00-0.05s: commits to `lane_change_left`, `dist=0.75m` already at
  first frame (as with every prior test — the cone is already close by
  the time this session's recording/detection window opens; unclear
  whether that's real course geometry or a late-detection artifact, see
  open questions).
- t=0.20-5.34s (**5+ seconds**): `steer` pegged at exactly -1.00 nearly
  continuously, `throttle_final` pinned at 0.100 (the LineFollower
  internal floor, not an obstacle_avoidance-side reduction — see
  throttle law above) the entire time. `x_f` barely moves: 71px → 180px
  over 5+ seconds. This is the longest full-lock stretch seen in any
  test so far (longer offset jump = longer time pegged).
- t=6.59-7.34s: `lf_status` flips to `white-tracking` briefly, `x_f`
  jumps erratically (215 → 266), `dist` bottoms at **0.407m** — this is
  the actual first-cone graze/impact point.
- t=7.59s: `dist` jumps to 5.68m (cone behind/beside now, or lost) —
  matches the "graze then continue" pattern from the -0.5/-0.6 tests.
- t=7.84s-24.63s (**~17 seconds**): `dist` hovers in the 1.2-2.6m range,
  `side` flip-flopping right/left continuously, and — notably —
  **`dist` reads the exact same value, `1.5243902439024384`, repeated
  for ~12 continuous seconds (t≈11.84 to t≈24.13)**. That's suspicious:
  either a real, roughly-stationary second obstacle, or a static
  track feature (like the already-known blue ground marking, or the
  track edge) getting persistently misclassified as an obstacle,
  preventing the 4s clear-debounce from ever firing. `avoid_dir` never
  releases back to `None` this entire time.
- t=25.13-27.13s: **chaos** — `x_f` swings wildly (262 → 221 → 250 →
  268 → 244 → 270 → **63**, a huge one-frame jump), `lf_status` flips
  `tracking`/`white-tracking`/`tracking` repeatedly, `dist` flickers
  between values and `None`. This looks like the point of an actual
  collision/getting-stuck event, not clean sensing.
- t=27.38s through the end of the recording at t=38.66s (**~11+
  seconds, the rest of the whole recording**): `steer` pegged at
  exactly -1.00, `throttle_final` pinned at exactly 0.100, `x_f`
  oscillating in a narrow band around 65px (not moving), `avoid_dir`
  still `left`, `lane_offset` still -0.75. The car never recovers for
  the rest of the recording. This reads like the car physically got
  stuck (wedged against the cone / off the track) and stayed pegged at
  full lock steering + minimum throttle until whoever was testing
  stopped it.

**Takeaway: -0.75 did not fix the graze, and additionally produced a
much longer full-lock/low-throttle phase up front and an apparent
terminal stuck state at the end that the -0.5/-0.6 tests didn't show.**
Bigger offset made the steering-lock/throttle-floor problem worse, not
better, and clearance at the actual pass point (0.407m) was not
meaningfully improved over the -0.5/-0.6 tests (0.527m) — if anything
slightly worse.

## Root-cause hypothesis (supported, not yet fixed)

Jumping `lane_offset`'s target instantly by a large amount (+0.5 → -0.6
or -0.75) creates a large instant P-error in `line_following.py`'s
steering law, which pegs `steering` at ±1.0 for several real seconds.
While pegged, `line_following.py`'s own curvature-based throttle law
(independent of anything `obstacle_avoidance.py` does) drops throttle to
`th_min`. The car is moving very slowly for the first several seconds of
the maneuver — exactly when it most needs to be translating laterally —
and only recovers speed once the steering error shrinks enough to unpeg.
Given the cone is already close (~0.7m per the placeholder distance
model) by the time detection/commit happens in every test so far, there
may simply not be enough real distance/time for this slow-start dynamic
to complete a full lateral translation before reaching the cone,
**regardless of how large the offset target is** — which would explain
why -0.5, -0.6, and -0.75 all produced similar-magnitude grazes: the
bottleneck isn't the target, it's the achievable rate of translation in
the available distance.

This is a genuine vehicle-dynamics/timing interaction between
`obstacle_avoidance.py`'s offset commands and `line_following.py`'s
internal throttle law — not a decision-layer logic bug (commit/release/
ramp/no-starvation all check out fine in replay).

## Open questions / where to pick up

1. **Is detection actually happening as early as it could?** STILL OPEN
   — see the 2026-07-27 update at the top of this file. Would need a
   test where the cone is deliberately placed farther back with room to
   spare, checking whether `obstacle/info.present` goes true earlier
   than the current ~0.7-0.85m mark.
2. ~~What is the ~12-second-long constant `1.5243902439024384`
   reading~~ **RESOLVED — see the 2026-07-27 update at the top of this
   file: it's a real second cone on the course, correctly detected. Not
   a bug.** This also means: **the course has two cones, not one** —
   re-read every prior test result in the table below with that in
   mind.
3. ~~What actually happened at t=25.13-27.13s~~ **RESOLVED — see the
   2026-07-27 update at the top: it's a brief, real recovery followed
   immediately by a SECOND steering-lock/throttle-floor episode, this
   time against/near the second cone, that never releases for the rest
   of the recording.**
4. **Should `lane_offset` be ramped in from farther away instead of
   commanded instantly on commit?** i.e. start easing the offset target
   before the car is close enough to need full avoidance, so the P-error
   (and therefore the steering-lock/throttle-floor duration) stays
   smaller when it matters most. This is speculative and untested.
5. **Is `_corridor()`'s use of live `lf.x_f` reliable during an active
   maneuver?** Partial investigation done today: printed per-frame
   corridor bounds vs. the tracked blob's pixel column during the -0.6
   pass. The corridor center tracked within ~5-25px of the near cone
   blob's own column for most of the approach, only diverging sharply
   in the last ~0.5s as the car passed beside it. Inconclusive whether
   this is contamination (x_f actually getting pulled toward the cone)
   or coincidence (both responding similarly to the same steering
   maneuver) — not resolved. **Caveat if you re-attempt this trace:** a
   debug script that mixed `print()` (stdout) and the modules'
   `logging` calls (stderr), redirected together via `> file 2>&1`,
   produced misleadingly reordered output — stderr flushes immediately,
   stdout buffers in blocks, so logger lines can appear to happen
   "before" print lines that were actually processed earlier. Route
   logging to stdout explicitly or flush after every print if you
   retry this.
6. Per project convention, **don't edit `line_following.py`** without
   explicit sign-off — it's treated as fragile/hands-off. If the
   throttle-law interaction (root-cause hypothesis above) really is the
   bottleneck, the fix likely needs to live on the
   `obstacle_avoidance.py` side (e.g., pre-ramping the offset, or
   triggering earlier) rather than changing the law itself, unless the
   user explicitly agrees to touch that file.
7. **NEW (added 2026-07-27, see update at top): confirm with whoever set
   up the course whether two cones spaced a few meters apart is the
   intended test layout**, and if so, whether `OBSTACLE_CLEAR_DEBOUNCE_SEC`
   (4.0s) makes sense for that spacing at all — with two cones this close,
   the debounce may structurally never get a long enough gap between them
   to fire, which means a committed lane-change direction can never
   release between cone 1 and cone 2 regardless of how the offset itself
   is tuned. That's a second, independent lever from the offset value
   this file's tests have been varying, and it hasn't been touched by
   any test so far.

## Current state as of this handoff

- `OBSTACLE_LANE_CHANGE_OFFSET_LEFT = -0.75` (myconfig.py, pushed and
  live) — **not verified to fix anything, may have made things worse**.
  Worth considering reverting to -0.5/-0.6 (equally bad on current
  evidence) while the real root cause above gets investigated, rather
  than continuing to guess at the offset value.
- `manage_line.py drive` process was left running (PID from the -0.75
  test, started 13:58 PDT) — check `ps aux | grep manage_line` for
  current state before the next test; restart after any further config
  changes.
- All temp `_replay_*.py` / `_replay_*_out.log` files on the Pi have
  been cleaned up. This handoff file itself is the persistent record —
  it lives at `/home/pi/mycar/HANDOFF.md`.

**2026-07-27 update:** re-verified the process is still the same one
(PID 4644, still running, still idling — no new test has been driven
since this file was first written) and `OBSTACLE_LANE_CHANGE_OFFSET_LEFT`
is still `-0.75` in `myconfig.py`. This session did a replay-only,
read-only re-analysis of the existing -0.75 recording (no config
changes, no new physical test, nothing pushed) — see the update at the
top of this file for what changed in the understanding of the bug. The
replay script and blob-probe script used for this are again cleaned up
off the Pi, per this file's own convention.
