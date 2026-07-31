# linefollower_car

The car directory for our line-following / obstacle-avoiding Donkeycar.
This is a snapshot of what runs on the Pi at `~/mycar`.

## Why this is a folder here and not merged into `donkeycar/`

Donkeycar car directories are **not** part of the library by design —
`donkey createcar --path ~/mycar` generates one *outside* the package, and
it holds the config and the vehicle-assembly script for one specific car.
Dropping these files into `donkeycar/parts/` and `donkeycar/templates/`
would misrepresent them as library code, and would mean fixing up the
flat imports these files use for each other (`manage_line.py` does
`from dashboard import ...`, `dashboard.py` reads `dashboard.html` from
its own directory) purely to satisfy the move.

## Why this is not going upstream

Per the repo's own `CONTRIBUTING.md`, a PR is unlikely to be accepted if
it *"adds a feature that is not useful to a broad audience or is too
complex/complicated for the Donkeycar audience"*, and in that case the
guidance is to *"maintain the feature in your own fork"*.

That describes this code exactly. `line_following.py` is tuned against
one physical course — its thresholds are field measurements of specific
yellow tape under specific lighting (`LF_LAB_B_FLOOR=133` because *"sunlit
pavement measures b<=130, shadow tape 141"*), and its lane presets are
measured offsets for that track's lane widths. It is genuinely useful to
us and genuinely not useful to a broad audience.

**The one piece of this work that *is* upstream material** — a fix for
the MJPEG stream sending every camera frame ~8 times over — was separated
out and lives on the `fix/mjpeg-duplicate-frames` branch, touching only
`donkeycar/parts/web_controller/web.py` and its tests. See
`WORKLOG_20260730.md` §2b.

## Layout

| file | what it is |
|---|---|
| `manage_line.py` | vehicle assembly script (the `donkey` template this car drives with) |
| `myconfig.py` | this car's config overrides |
| `line_following.py` | the CV controller — yellow dashed-line follower with white-boundary fallback |
| `obstacle_detector.py` | cone/obstacle detection |
| `obstacle_avoidance.py` | swerve / lane-change planning around detected obstacles |
| `lf_telemetry.py` | per-frame JSONL telemetry (joins to the tub on `idx`) |
| `dashboard.py` | web dashboard: live tuning, lane presets, low-latency video |
| `dashboard.html` | the dashboard page (no build step, no CDN — it loads over the car's own WiFi) |
| `calibrate.py` | steering/throttle calibration helper |
| `lf_tools/` | offline replay and simulation tools |

## Documentation

| file | what it covers |
|---|---|
| `HANDOFF.md` | running notes — tuning history and what was tried |
| `KNOWN_ISSUES.md` | confirmed bugs that are **not** fixed, with session/frame indices |
| `WORKLOG_20260730.md` | 2026-07-30: corner-failure analysis + the dashboard changes |

## Running it

Copy the contents to the car directory on the Pi and drive as usual:

```bash
cd ~/mycar
python manage_line.py drive
```

The dashboard is then at `http://<car>:8887/`, with the stock Donkeycar
UI still at `/drive`.

## What is not in this snapshot

The tub (`data_line/`, ~9.2 GB of recorded frames), `logs/`, `models/`
and `backups/` are deliberately excluded — they are data, not source, and
the analysis in `WORKLOG_20260730.md` cites them by session and frame
index rather than shipping them.
