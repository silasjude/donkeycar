# Line-Following, Round 3: Daytime Hardening + Mission 2 Lane Following

Follow-up to `line_following_diagnosis.md` (initial detection diagnosis, OAK-D
resolution bug) and `line_following_lane_centering.md` (restart from madhav's
branch, two-white-line lane centering). This covers a long iterative session
run against `~/mycar/line_following.py` (car-local, not this repo — see
below) across many real test drives: first hardening plain line-following
against a first-ever daytime test (this branch's prior work was all done at
night), then building Track 2 Mission 2 (lane following) on top of it.

Every fix below was found from a real recorded failure (tub footage replayed
offline through `lf_tools/replay.py`, a car-local script not in this repo,
against `donkeycar/parts/line_following.py`'s actual `LineFollower`), and
where noted, confirmed or contradicted by a subsequent real test drive.
Offline replay is open-loop — it replays the camera frames a real drive
actually produced, regardless of what the code under test would have
commanded, so it can prove a fix changes a computed decision as intended but
can never prove a fix keeps the car on a different physical path. That
caveat applies throughout; it's called out again wherever it materially
limits what a fix could actually verify.

## Daytime hardening

This branch's detector, PID tuning, and lane-centering logic had only ever
been exercised at night (artificial lighting, no strong shadows). The first
daytime test surfaced two failure modes specific to that new lighting:

**Wild heading swings from ill-conditioned slope fits.** The heading
measurement is a slope fit through as few as `LF_MIN_BANDS=2` band
centroids. With only two points and a narrow vertical gap between them, a
couple pixels of ordinary detection noise (more likely in patchy daylight
shadow than uniform night lighting) swings the fitted slope wildly. One
session's footage showed the raw heading pegging from +1.0 to -1.0 within
half a second while genuinely still tracking a real dash, driving the real
car's steering back and forth until it wandered into a planter bed. Fixed
with `LF_MIN_HEADING_SPREAD_FRAC`: below this fraction of the ROI height,
the kept fit points are too vertically bunched to trust the slope for
heading, so heading reports neutral (0.0) instead of the raw value.
Position isn't affected — it's a bounded near-field average, not a raw
slope, so it stays usable even when the fit is too ill-conditioned for
heading. (This guard's own cost shows up again under Mission 2 below.)

**Stale white-boundary reuse with no expiry.** `_last_wl`/`_last_wr` (the
white cluster actually picked last frame, kept for continuity) had no
staleness check — once set, `_split_whites` would keep reusing it
indefinitely if that side simply stopped being detected, with nothing to
invalidate it. One session's right boundary was last genuinely re-detected
mid-turn, then went undetected (out of frame, occluded past a planter) for
8+ seconds while the stale value kept getting fed into
`_estimate_from_whites`'s position estimate every frame, dragging the
car's steering harder and harder toward a recycling-bin enclosure until the
run had to be stopped. Fixed with `LF_WHITE_STALE_SEC`: each side now
tracks when it was last a *fresh* detection (`_last_wl_t`/`_last_wr_t`), and
`_split_whites` treats it as unknown (falls back to nearest-yellow-reference
selection, same as cold start) once it's older than that.

## Mission 2: lane following

The mission: instead of riding the yellow dash itself, drive within one lane
of the two-lane road, bounded by the yellow centerline on one side and the
outer white line on the other, without crossing either.

The existing lane-offset machinery (`LF_LANE_OFFSET`, `LF_LANE_WIDTH_PX`,
`dist_left`/`dist_right`, the persistent near-white tracker for blind bends)
already did most of the geometry — riding at a fixed offset toward one
white boundary was already supported. What was missing was (a) a clean way
to select and pin a lane side, and (b) any explicit check that a computed
position doesn't imply the car has actually left the lane.

**`LF_LANE_SIDE` convenience toggle.** Set to `"left"` or `"right"` and it
pins *both* halves of the already-field-tuned pairing documented above
`LF_LANE_OFFSET` — offset AND `LF_CURVE_GAIN` together (`"left"` ->
offset -0.5, curve gain -0.8; `"right"` -> offset +0.5, curve gain -0.2) —
overriding whatever those two settings say elsewhere in config. This was a
direct response to on-car iteration: it's one setting to flip instead of
two that have to be kept in sync by hand.

**`_lane_pos()`: the core boundary metric.** A candidate position (real
detection or an estimate) expressed as 0.0 = on the yellow, 1.0 = on the
outer white, using the same calibrated/learned lane width the offset math
already trusts. Values outside `[0, 1]` mean the car (or the estimate) is
past one boundary or the other. Returns `None` whenever lane-following
isn't engaged, so every mechanism built on it below is a structural no-op
for plain line-following — verified by replaying real recorded sessions
before and after each change and confirming byte-identical output
(steering, throttle, status) with lane-following off.

**Soft boundary clamp (`LF_LANE_SAFETY_MARGIN_FRAC`).** The lane-offset
geometry aims for a target position trusting a calibrated or learned lane
width; if that width is wrong, or the current estimate is bad, the computed
target can sit closer to a boundary than intended without reading as
unusual control error. When a boundary is seen *fresh this frame* (this
frame's own yellow fit, or a same-side white cluster with near-field
confidence), the white-guided position estimate is rejected outright if
`_lane_pos` says it's past a boundary beyond this margin — same rejection
idiom already used for every other implausibility gate in this file (jump
gates, temporal gate, disagreement gate). Deliberately narrow in scope at
first: only enforced on the general white-line estimate fallback
(`_estimate_from_whites`), not on the dedicated, heavily-tuned near-white
tracker used for blind bends — that tracker exists specifically for cases
with no yellow reference at all, and rejecting its output on an unvalidated
check risked breaking exactly the case it was built for. `lane_pos` is
recorded (HUD overlay, `debug`, and `lf_tools/replay.py`'s CSV output) at
every branch regardless of enforcement, for visibility during replay
analysis even where nothing is auto-rejected.

**Implausible one-frame jump gate (`LF_LANE_IMPLAUSIBLE_MIN`/`_MAX`).** A
real recorded lane-following test found one frame where a "genuine" fresh
yellow fit (i.e. `found=True`, not an estimate) implied the car had moved
1.5 lane-widths sideways in a single 1/20s frame — physically impossible.
The existing temporal-jump gate didn't catch it because it compares against
the *previous frame's* position, not against lane geometry. Added a second,
independent check: reject a fresh fit if `_lane_pos` puts it outside a
deliberately wide `[-1.0, 2.0]` lane-width range (wide enough that real,
if uncomfortably close, drift is still trusted — this only catches a fit
that isn't describing this lane at all). Confirmed via replay that the
specific frame that motivated this is now rejected and correctly falls
through to the near-white tracker instead of being acted on.

**Boundary-proximity urgency (`LF_LANE_BOUNDARY_MARGIN`/
`LF_LANE_BOUNDARY_KP_BOOST`).** A real lane-following test drive showed the
car drifting to `lane_pos = -0.33` (a third of a lane-width across the
yellow, into the oncoming lane) during a real, sustained curve at a
building corner. Root cause: the daytime `LF_MIN_HEADING_SPREAD_FRAC` guard
above stayed engaged for the entire ~2-second curve (this camera angle
produces chronically narrow-spread dash fits at that specific corner, not
just momentary noise), so the controller had no curve feed-forward for the
whole turn and was reduced to a laggy position-only correction that
couldn't catch up in time. Rather than loosen that guard — it's shared
with plain line-following and was itself a validated fix for a real
instability — the position term's gain now ramps up (to
`LF_LANE_BOUNDARY_KP_BOOST` times normal) as `lane_pos` closes in on either
edge, starting `LF_LANE_BOUNDARY_MARGIN` lane-widths out. Confirmed via
replay that the computed steering command is now meaningfully stronger
through that exact excursion (e.g. one frame went from a computed +0.62 to
full lock). **This is the clearest instance of the open-loop replay
limit in this whole session: because replay's camera frames don't change
in response to a different computed command, this cannot be verified to
actually keep the real car from crossing the boundary — only that the
command itself is now stronger. Needs a real test drive to confirm.**

## Known unresolved issue

A single wide, visually ambiguous plaza crossing (parking-stripe-style
white markings) has caused trouble across many sessions in both line- and
lane-following mode — a mix of confirmed, fixed bugs (several of the ones
above were found there) and, per the most recent test, at least one episode
that doesn't fit any of those bug signatures: real recorded steering
oscillating hard left/right repeatedly with no clear single cause, in a
window where offline replay actively disagreed with the real recording
(correlation went negative, far worse than anywhere else this project has
seen). No fix was attempted for that specific episode — the evidence wasn't
solid enough to name a cause without guessing, so none was invented.
