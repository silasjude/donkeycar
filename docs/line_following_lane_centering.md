# Line-Following, Round 2: Two-White-Line Lane Centering

Follow-up to `line_following_diagnosis.md` (the OAK-D resolution bug and
initial detection diagnosis). This covers a separate session: restarting the
detector from a teammate's (madhav's) branch, re-verifying the resolution
fix on hardware, and adding real two-white-line lane centering — plus what
on-car feedback said about each step, since some of it contradicted what
offline replay predicted.

## Starting over from madhav's branch

The `toby` branch's own detector (this file, pre-this-session) had drifted
into a state that wasn't driving well on-car. Rather than keep debugging that
version blind, we restarted from a teammate's (madhav) branch, which had
reportedly performed better, and rebuilt from there — same overall algorithm
(CIELAB+hue detection, multi-band centroid fit, PD steering), but without
this file's saturation gate, decoupled-outlier-rejection fix, or steering
trim. Those three are genuinely valuable and are preserved in THIS version of
the file; the restart was about the detection/control logic, not discarding
prior bug fixes.

## Re-confirming the resolution bug, on hardware

`line_following_diagnosis.md` already identified that `oak_d.py` linked
`cam_rgb.video` instead of `cam_rgb.preview`, so `setPreviewSize()` had no
effect and the camera streamed full sensor resolution regardless of
`IMAGE_W`/`IMAGE_H`. That diagnosis was made by replaying an existing
recording, not by testing the fix on the car.

This session did test it on hardware, and found the fix was incomplete in
two ways:
- `setup_rgb_camera()` called `cam_rgb.setPreviewSize(self.image_w,
  self.image_h)` — `self.image_w`/`self.image_h` were never set anywhere
  (`__init__` only sets `self.width`/`self.height`). This crashed
  `OakD.__init__` with `AttributeError` on every real run, camera never
  started.
- `_poll()` unconditionally fetched the `"depth"` output queue whenever RGB
  *or* depth was enabled, but the pipeline only creates that queue when
  depth is actually enabled. Any RGB-only config (the common case for a line
  follower) crashed with `RuntimeError: Queue for stream name 'depth'
  doesn't exist`.

Both fixed (use `self.width`/`self.height`; fetch each queue only under its
own enable flag). After both fixes plus the original `.video`->`.preview`
change, a real recording's frames finally came back at the configured
160x120 instead of 1920x1080 — confirmed by checking `.shape` on freshly
recorded frames, not just re-reading old ones.

## Two-white-line lane centering: what changed and why

The existing single-side white-line fallback (used when the dash drops out)
holds the learned distance from *one* boundary line. It's fragile: replaying
a real recording turned up a frame where it projected the lane center far
enough off to command `steering=-0.93` in a spot that plainly wasn't a hard
turn.

Added a new tier, tried before the single-side fallback: when *both* white
boundary lines are detected in the same band, project each one back to
where the dash should be using the per-band spacing already being learned
while the dash is tracked, then average the two independent estimates. This
needs no per-frame commitment to a single side and isn't time-limited the
way the single-side fallback is (that one can only steer for
`LF_WHITE_MAX_SEC` before being treated as lost — this tier is a direct
measurement each frame, not a projection that goes stale).

Trust order is now: dash > two-line lane-center > single-side white line >
lost.

### The averaging detail that mattered

The first version of this feature used the raw geometric midpoint of the two
white detections — implicitly assuming the dash sits exactly halfway between
the two boundaries. On this track it measurably doesn't. On-car testing
caught this directly: the car "drove nicely between the two white lines,"
in the driver's words, but visibly abandoned the dash to do it. Offline
replay of that exact drive showed why — a stretch that tracked the dash
steadily at `error=-0.16` jumped to a stable-but-wrong `error=-0.32` the
instant the raw-midpoint lane-center tier took over.

A revision projected each white detection through the *learned* per-band
offset instead of averaging raw positions (described above). Offline replay
of the same stretch showed the error partially correct itself, to about
`-0.27` — better, not fully closed (the offset is a slow EMA and likely
hadn't converged yet that early in the session).

**On a real drive, that revision performed worse than the raw-midpoint
version it replaced**, despite looking like a strict improvement offline.
This is the headline lesson from this round: offline replay against a fixed
recording is a real, useful diagnostic tool (it did correctly explain both
of the driver's complaints), but it is not a substitute for on-car
verification, especially for anything involving a running EMA (the learned
offset's actual state at drive time depends on the whole session's history
up to that point, which a replay against a different or partial recording
won't reproduce). The raw-midpoint version was reverted back in as the base
for this branch; the learned-offset correction is worth revisiting, but only
with on-car A/B confirmation, not offline replay alone.

## Open, not yet addressed

- The single-side fallback (last resort) can still get stuck: one real
  stretch showed it pinned near its error clip for over a second before the
  dash was reacquired. Not touched this round.
- The color/lighting robustness problem from the original diagnosis (dash
  vs. white line color rendering shifting under this track's lighting) is
  still open.
- This track has two side-by-side lanes, each with its own dash and white
  boundaries (confirmed from a driver-supplied photo) — worth double-
  checking that the per-band white search window can't drift onto the
  neighboring lane's boundary under some perspectives; not specifically
  ruled out yet.
