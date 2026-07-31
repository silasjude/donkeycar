# Known issues

Bugs found during testing that are confirmed real but not yet fixed.
Each entry should have enough evidence (session/frame indices, symptoms)
to pick back up without re-diagnosing from scratch.

---

## Full-lock steering / lost line further down the track (Mission 1/2, NOT obstacle avoidance)

**Found:** 2026-07-27, session `26-07-27_73` (catalog_104, indices ~104480-104615).

**Symptom:** car drives into a stationary concrete planter/wall well past
the obstacle-avoidance cone encounter, on a stretch of track it hadn't
been tested on before (car got further down the track than prior runs
because the cone-avoidance fixes started working).

**Confirmed NOT related to obstacle avoidance.** Replaying the session
(`obstacle_detector.py` / `obstacle_avoidance.py` against the real
recorded frames, using the session's actual timestamps) shows the
avoidance system had already finished and fully released by t=24.15s
(`avoid_dir=None`, `lane_offset` back to the untouched base `+0.50`).
The crash happens at t=25.10s-30.13s+, by which point:
  - `dist=None` the whole time -- no obstacle detected, obstacle logic
    inactive
  - `lf.lane_offset` stays at the plain base value the whole time --
    obstacle avoidance never touches it during this window
  - `lf.status` (LineFollower's own state) alternates between
    `tracking` and `white-guided`
  - `steer` is pegged at full lock (1.00) repeatedly for seconds at a
    stretch

This is `line_following.py`'s own line/white-boundary tracking losing
the line and/or a real sharp turn in that section of track, not
anything obstacle-avoidance code writes to `lane_offset`/`curve_gain`.

**Not yet investigated:** what specifically causes the full-lock
steering in that stretch -- possible causes to check first: a genuinely
sharp turn beyond what's been tuned for, worn/harder-to-detect paint in
that section, or a white-boundary misdetection dragging `white_x_f` off
target. Start by pulling frames from indices 104480-104615 of
`data_line/catalog_104.catalog` (session `26-07-27_73`) and look at what
the camera actually saw in the few frames right before/during the
full-lock steering.

**Scope note:** this is `line_following.py` territory (Mission 1/2),
which has its own extensive tuning history and is treated as
fragile/hands-off elsewhere in this project -- fix it there, not by
routing around it from the obstacle-avoidance files.

---

## UPDATE 2026-07-27: second real occurrence, much richer evidence
(same bug, same location, now with visuals + full telemetry)

**Session:** `26-07-27_77`, idx 108100-108543 (spans
`catalog_107.catalog` into `catalog_108.catalog`). This is the FIRST
real test where the car made it past the two-cone obstacle section
without a collision (the obstacle-avoidance fix from earlier today —
see `HANDOFF.md` — worked), so it's also the first time it reached this
much further down the track. Replayed with the current (fixed) code,
using the same time-accurate methodology as `HANDOFF.md`, and visually
inspected the actual recorded frames (not just telemetry) — the pi's
`data_line/images/108xxx_cam_image_array_.jpg` files, several pulled and
viewed directly.

**Visually confirmed the physical cause of the "genuinely sharp turn"
half of the original hypothesis:** the same concrete planter/wall from
the first occurrence is directly in the car's path again, and this time
there's clear visual evidence of WHY it's hard to track — the white
boundary line runs immediately along the base of the planter, and the
track bends sharply around it. Further down (idx~108417+) there's a
THIRD obstacle zone (3 more cones, same blue-tape marker style as the
first two) with white boundary lines visible on BOTH sides of a wide
plaza-like section — two candidate white lines in view at once, which
is exactly the kind of scene the near-white tracker (`_split_whites` /
`white_x_f`) was never validated against having to disambiguate.

**Telemetry, precisely dated (t=0 at idx 107927, the process's actual
recording start):**
- t≈9.5s-23.7s (idx 108113-108396, ~13 seconds): `lane_offset=0.50`
  (plain base Mission-2 driving, obstacle avoidance fully released —
  confirms the original entry's "not obstacle avoidance" finding again)
  and the lateral error (`last_err`) stays SMALL almost the whole time
  (mostly within +/-0.25) — the car is not lost, it's genuinely tracking
  something. Yet `throttle` is pinned at exactly `LF_THROTTLE_MIN`
  (0.100, not the new `LF_THROTTLE_MIN_LANE_OFFSET` floor from today's
  fix — that only applies when `error` itself is large, and it isn't
  here) for nearly the entire 13 seconds. Since small error + floored
  throttle only happens via the OTHER branch of `slow = min(1.0,
  max(abs(steering), abs(hterm)/h_clip))`, this means `hterm` (the
  heading/slope reading) is pegged near `LF_HEADING_CLIP` almost
  continuously for 13 real seconds. `x_f`/`steer` hunt/oscillate
  throughout (x_f swings 80px-240px repeatedly, steer swings roughly
  -0.1 to +0.99) rather than settling — the car is crawling through a
  sustained bend, not smoothly arcing through it.
- t≈24.0-24.9s (idx 108402-108420): line tracking fails hard for the
  first time — `last_err` hits -1.00, `x_f` goes NEGATIVE (as low as
  -82.6, meaning the tracked reference is estimated off the LEFT EDGE of
  a 384px-wide frame), `lf_status` is `white-tracking` throughout. Brief
  recovery at 108420 (`tracking`, `err=+0.38`, `x_f=195.5`).
- t≈25.0-28.2s (idx 108423-108483): recovery does NOT hold — `x_f`
  swings wildly and repeatedly between deeply negative (-70) and
  strongly positive (376.5, itself past the right edge of the visible
  frame), `lf_status` flip-flops `tracking`/`white-tracking`/`coasting`
  many times, `last_err` swings between -1.00 and +0.82. This is the
  most chaotic stretch of the whole recording — genuinely unstable
  tracking, not just a single bad frame.
- t≈28.2s onward (idx 108486-108543, through the LAST frame of the
  recording): the third cone cluster gets detected
  (`obs_status='lane_change_left'`, `lane_offset` commanded to -0.75, a
  FULL lane change this time since detected far enough away —
  `distance_m=5.68` — to clear today's new
  `OBSTACLE_LANE_CHANGE_MIN_DISTANCE_M=1.0` gate). This lane change is
  attempted WHILE the underlying line tracking is still unstable from
  the section above (`x_f`/`err` still swinging, `lf_status` still
  flip-flopping `tracking`/`white-tracking`/`coasting`/`white-guided`).
  The recording ends abruptly at idx 108543 mid-lane-change
  (`lane_offset=-0.75`, `dist=None` on the very last frame) — reads like
  the operator manually stopped the test at this point, not a clean
  finish.

**Refined hypothesis (still not root-caused at the code level, but
narrower than before):** two compounding problems, not one — (1) a
sustained, tightly-curving section of track (confirmed by the actual
frames: a boundary line running right along a physical wall/planter)
that the heading-based throttle slowdown treats correctly in principle,
but the CAR clearly struggles to hold a stable line through for a long
stretch (13s of hunting, not a clean arc); and (2) once the line is
genuinely lost partway through, recovery is unreliable — the tracker
swings between wildly inconsistent estimates (including estimates
placed entirely outside the visible frame, both directions) for several
seconds rather than either reacquiring cleanly or coasting safely, and
this section also happens to be where TWO white boundary lines are
simultaneously visible in frame, a scene shape not mentioned anywhere in
this file's history of white-tracker tuning notes.

**Not yet investigated (updated from the original entry):**
1. Whether the near-white tracker (`_split_whites`, `white_x_f`) is
   specifically confusing the two simultaneously-visible white lines in
   the third obstacle zone — this seems like the most promising lead for
   the `x_f` estimates landing far outside the visible frame (a
   dead-reckoned/misattributed reference, not a real detected position).
2. Whether the sustained hterm-pegged slowdown (t≈9.5-23.7s) reflects a
   turn genuinely sharper than anything `LF_CURVE_GAIN`/`LF_SHARP_BEND_MIN`
   were tuned against, or a detection issue making a moderate turn read
   as sharper than it is.
3. Whether the THIRD obstacle zone's lane-change attempt compounding on
   top of already-unstable tracking (last bullet above) is a real
   contributing hazard worth addressing (e.g. not committing to a new
   lane change while `lf.status` has been unstable/flip-flopping
   recently) or just incidental timing in this one recording.

**Scope note (unchanged):** still `line_following.py` territory —
fix there, with explicit sign-off, not routed around from the
obstacle-avoidance files. Session/frame indices above are enough to
replay this exact stretch again without re-deriving the timeline.

---

## UPDATE 2026-07-27: one real bug fixed (with explicit sign-off), a
second, bigger one found and NOT fixed — instability persists

Built a deeper diagnostic replay (wraps `detect()` to capture its raw
per-frame output, not just `run()`'s return value) against the exact
same idx 108100-108543 stretch above.

**Fixed: the PRIMARY white-tracking path (`white-tracking` status,
`virtual = white_x_f +/- track_w`) had no sanity check against the
last known line position, unlike the SECONDARY estimate-from-whites
path a few lines below it, which already gates with `single_dash_max_jump`
("the line cannot teleport" — see that path's own comment).** Real
footage showed `white_x_f` able to drift steadily off the real boundary
over about half a second — each individual frame-to-frame step small
enough to pass `_update_white_tracker`'s own jump gate (which only
checks against its OWN last prediction, no independent anchor), but the
cumulative drift pulled the estimate 100px+ off the real line and fed
straight into steering. Added the same jump-tolerance check
(`single_dash_max_jump` growing allowance, same formula) to the primary
path too, so a drifted estimate now falls through to the secondary path
or coasting instead of being trusted outright. Confirmed via replay:
zero change to the calm, well-behaved sections of the drive (the
two-cone pass, byte-for-byte identical `x_f` trajectory); the
frame-to-frame estimate whiplash in the unstable section is gone (no
more single-frame 100px+ snaps from a stale white-tracker reseed).
Deployed live 2026-07-27.

**NOT fixed, and turned out to be the DOMINANT driver of the
instability, not the part above:** re-ran the same replay after the fix
and the worst-case `x_f` swing across the whole recording didn't
improve (peak frame-to-frame jump 320px before -> 329px after; overall
range -83..377 before -> -104..397 after) — blocking the bad
white-tracking path just meant the system fell back to the ORDINARY
`tracking` path more often (`white-tracking` frame count dropped
69->11, `tracking` rose 530->562), and that path is turning out to be
JUST AS capable of large jumps in this section. Concretely: at idx
108423-108432 (all `found=True`, ordinary `tracking` status, no
white-tracking involved at all), `last_x` swings 220.9 -> 256.1 -> 233.2
-> 337.0 -> 315.5 -> 309.7 -> 279.6 across consecutive ACCEPTED frames.

The likely mechanism: `line_following.py`'s temporal consistency gate
(`LF_TEMPORAL_JUMP=0.09`, `LF_TEMPORAL_MAX_REJECT=6`) is DESIGNED to
widen its jump tolerance by `(1+n)` on each consecutive rejection and
accept UNCONDITIONALLY after `LF_TEMPORAL_MAX_REJECT` rejections in a
row — a deliberate choice (see the code's own comment: "a stale last
position can never lock the detector out for good"). In a visually
cluttered section like this one (multiple cones, a nearby wall, the
raw yellow-hue detector apparently catching different candidate blobs
frame to frame), that growing tolerance and eventual forced-accept
looks like exactly what's letting genuinely wrong fixes through with
increasingly little scrutiny after just a few rejected frames. This
is a much bigger, more carefully-balanced mechanism than the one-line
gate-gap fixed above (explicit trade-off against permanently locking up
on a real occlusion elsewhere on the track) — did NOT touch it this
round. Retuning it blind from one section's replay risks regressing
recovery behavior everywhere else it's relied on; it needs either (a)
real-world validation on this specific section after the fix above is
tested, or (b) improving the raw yellow-detector's robustness against
this scene's specific clutter (multiple cones + wall + shadow) so the
temporal gate has fewer bad candidates to arbitrate between in the
first place.

**Also still not investigated:** the separate ~13s sustained
heading-pegged slow crawl (idx 108113-108396, see the update above this
one) — unclear whether that's a genuinely sharp turn beyond current
tuning or a related detection issue; not touched this round either.

**Net effect of this update:** one confirmed, narrow bug fixed and
deployed (validated safe, zero regression elsewhere). The overall
instability in this specific section of track is NOT resolved — a real
test drive through this exact stretch is the next useful data point,
both to see whether the fix helps at all in practice and to get fresh
footage of the (still unfixed) temporal-gate/raw-detection issue for
further diagnosis.

---

## Tub datastore corrupts on hard power-off (recurring: 2026-07-23, 2026-07-28)

**Symptom:** `manage_line.py` crashes at startup with
`json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`
from `datastore_v2.py _read_contents` while TubWriter reopens
`data_line/`.

**Cause:** cutting power to the Pi mid-drive loses buffered SD writes.
On 2026-07-28 this left `data_line/manifest.json` missing its 5th line
(catalog metadata), `catalog_139.catalog` truncated at exactly 128 KiB
(734 complete records + one cut mid-line), and `catalog_140.catalog`
existing but 0 bytes while its `.catalog_manifest` claimed 166 records.
First occurrence (2026-07-23) was handled by moving the whole tub aside
(`data_line_corrupt_20260723/`).

**Repair (2026-07-28, data preserved):** trimmed the partial record,
rebuilt `catalog_139.catalog_manifest` line_lengths from the actual
file, reconstructed manifest.json line 5
(`paths`/`current_index=139734`/`max_len=1000`/`deleted_indexes`), and
quarantined the empty catalog_140 pair. Originals + repair script in
`backups/tub_salvage_20260728/`. Verified: tub opens read-only with all
139734 records, and a writable open + append works (tested on a copy).

**Not fixed:** the underlying cause. Any hard power cut while recording
can do this again. Mitigations to consider: shut down / stop
`manage_line.py` before pulling power, or disable recording when tub
data isn't needed.

---

## Stale frozen heading drives the sharp-bend floor the WRONG WAY (first right-hand corner, concrete block with the black kick-strip)

**Found:** 2026-07-30, session `lf_20260730_154031.jsonl` autopilot
segment 5, idx 190505-191032. Full analysis and per-frame numbers in
`WORKLOG_20260730.md`. Live tuning during the run was lane offset
`+0.70`, curve gain `-0.20`.

**Symptom:** the car tracks the line cleanly into the corner, then
steers LEFT into a right-hand bend and coasts into the concrete block.

**New finding (not previously documented).** `h_f` drops to **-0.724**
at idx 190862 and then never updates again -- it is frozen at that value
for the remaining 170 frames (**8.5 seconds**), because nothing refreshes
the heading once the yellow is gone. `LF_HEADING_CLIP` is 0.7, so `hterm`
sits EXACTLY pegged at -0.7, which trips the sharp-bend floor at
`line_following.py:2199` (`|hterm| >= 0.9 * 0.7 = 0.63`). The floor then
forces the heading command to at least `LF_SHARP_BEND_MIN` (0.5) **in the
sign of the lean -- negative, i.e. LEFT -- inside a right-hander.**

The white-tracking branch (`line_following.py:2771`) calls `_heading_cmd`
with the default `bend_frac=0.9`, so it is this general floor that
engages, not the more permissive `LF_WHITE_GUIDE_BEND_FRAC` path.

The floor is doing what it was designed to do ("refusing to steer WITH a
pegged heading guarantees losing the line") -- but nothing checks the
pegged lean is still FRESH. Here it is an 8.5-second-old value from
before the line was lost.

**Two compounding faults alongside it:**

1. *The white fallback reads the corner as sideways drift.* From idx
   190868 `white_x_f` sweeps 322 -> 249 -> 220 -> 197 -> 173 because the
   TRACK TURNS, but the follower cannot distinguish that from lateral
   drift: `lane_pos` climbs 0.66 -> 0.84 -> 0.99 -> 1.13 -> **1.22**
   (past the outer white line), `err` saturates to -0.62, steering goes
   to -0.53 LEFT.
2. *Everything then freezes and the car coasts in.* At idx 190884 the
   white is lost too and `x_f` (-87.263), `err` (-0.616), `steer`
   (-0.531) and `h_f` all freeze together: **2.6 s of coasting at
   throttle 0.15 with the wheels held at half left lock**, then after a
   brief reacquire a SECOND freeze of 1.8 s at `steer +0.267`, driving
   straight at the block.

**The temporal-gate forced-accept is still the endgame**, as the existing
entries above describe: at idx 191016 a **336 px jump** is accepted
(dead-reckoned -184 -> detected +245, on a 384 px-wide frame), saturating
`err` to +1.0 and slamming full right lock -- by which frame the block
already fills the camera view. Same signature in the other runs that day:
12:20 seg2 239 px, 13:08 seg1 **442 px**, 13:08 seg2 **456 px**. Both
13:08 segments died at the same large planter on the same corner
(idx ~182559 and ~183727).

**Lane offset contributed but is not the cause.** `+0.70` instead of the
field-tuned `+0.50` put the car ~40% further out; over segment 5
`lane_pos` was past the outer white line on **129 of 516 frames**. The
freeze above would happen at any offset. (The dashboard now badges which
offset/curve-gain pairs are field-tuned, so this is visible before a run
rather than after -- see `WORKLOG_20260730.md` part 2a.)

**Not fixed.** Nothing in `line_following.py` or `myconfig.py` was
changed. Candidate fixes in the order the evidence supports them:
  1. **Stale-heading guard** -- the sharp-bend floor should not commit to
     a lean that has not been refreshed for N frames. Clearest evidence,
     most contained blast radius.
  2. **Blind-coast behaviour** -- freezing steering *and* continuing at
     throttle 0.15 for 2-4 s is what turns a lost line into a collision.
  3. **The temporal-gate forced-accept** -- already documented above as
     risky to retune blind; wants validation on this specific corner
     rather than a global tolerance change.
