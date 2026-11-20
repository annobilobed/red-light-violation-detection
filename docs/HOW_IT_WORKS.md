# How it works

This document explains every component, in the order the data flows. File paths are relative to the repository root.

## 1. Configuration (`backend/app/pipeline/config.py`)

One `AnalysisConfig` describes **one camera view with one approach**:

| Part | Fields | Meaning |
|---|---|---|
| `signal` | `roi`, `layout`, `lamps` | Signal-head box. `vertical` splits it into thirds (red on top), `horizontal` into thirds (red on the left), and `custom` uses one box per lamp. |
| | `sat_min`, `val_min`, `min_score`, `margin` | HSV colour thresholds (section 3) |
| | `brightness_fallback`, `brightness_ratio` | Optional night fallback (off by default) |
| | `min_stable_frames`, `bridge_gap_s` | Temporal smoothing (section 3) |
| `crossing` | `stop_line`, `approach_point`, `approach_zone` | Geometry of the approach |
| | `line_margin_px`, `min_approach_frames`, `segment_tolerance`, `red_grace_s` | Crossing test and decision rule (section 4) |
| `detector` | `model`, `conf`, `iou`, `imgsz`, `device`, `tracker`, `vehicle_classes` | YOLO + ByteTrack |
| `evidence` | `pre_s`, `post_s`, `include_signal_uncertain`, `h264` | Clips |
| `frame_stride` | | Process every n-th frame |

`validate()` lists missing pieces (no stop line, no signal region, approach point on the line); an analysis will not start until it returns nothing.

## 2. Detection and tracking (`pipeline/detectors.py`)

`YoloTracker` calls `model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[...])` on every processed frame. Class IDs are looked up **by name** from the model (`car`, `motorcycle`, `bus`, `truck` for COCO). A fine-tuned model must use those names or set `vehicle_classes`. Boxes without a track ID are ignored. `reset()` clears ByteTrack state between videos, so IDs restart.

`ScriptedDetector` replays ground-truth boxes written by the synthetic video generator. It is used only for tests and the demo.

Verified here: YOLO11n + ByteTrack on CPU (4-core Xeon) ran at about 33 ms/frame on 1280×720 dashcam footage with persistent IDs (one car kept the same ID for 283 frames). YOLO11m ran at about 150 ms/frame. Low-resolution top-down footage (320×176) gave very few detections with the stock COCO model. This is one reason for fine-tuning on Cape Coast frames.

## 3. Signal-state reading (`pipeline/signal_state.py`)

**Per frame.** For each lamp box, convert to HSV and count the pixels that are bright (`V ≥ val_min`), saturated (`S ≥ sat_min`) and of that lamp's hue (OpenCV hue scale 0–180: red 0–10 and 160–180, amber 11–34, green 35–100). The lamp's **score** is that fraction. The lamp with the highest score is taken as lit only if its score is at least `min_score` and it beats the runner-up by at least `margin`. Otherwise the state is **`uncertain`**. The colour is checked inside each lamp's own position, so a red-looking amber lamp cannot be read as red.

Optional night fallback (`brightness_fallback`): when no lamp passes on colour (e.g. over-exposed white lamps), the lamp whose mean brightness is clearly higher than the others (`brightness_ratio`) is taken. This relies on lamp position, so it is off by default and should be validated on night footage before use.

**Over time** (offline, once the whole video has been read):

1. Colour runs shorter than `min_stable_frames` become `uncertain` (removes one-frame glitches).
2. `uncertain` gaps no longer than `bridge_gap_s`, with the **same** colour on both sides, take that colour. This handles LED flicker (red–dark–red is still red). A gap between **different** colours (e.g. amber→?→red) stays `uncertain`, because the change happened at an unknown moment inside it.

The result is a list of segments. A red segment's **onset** is the time of its first frame. `onset_exact` is false when the previous segment was `uncertain`; red then began at or before the recorded onset.

The Configure page's **Test signal reading** shows the single-frame scores for any frame, for tuning in daylight, dusk and night.

## 4. Stop-line crossing and the decision rule (`pipeline/crossing.py`)

**Anchor.** The bottom-centre of each vehicle box (where the vehicle meets the road).

**Signed distance** to the stop line, positive on the approach side (the side of `approach_point`).

**Arming.** A track must be seen at least `min_approach_frames` times more than `line_margin_px` on the approach side, and inside `approach_zone` if one is set. This guards against ID switches inside the junction creating crossings.

**Crossing.** An armed track crosses when its anchor is more than `line_margin_px` past the line. The crossing instant is interpolated linearly between the last sample before the line and the first sample at or past it. The interpolated crossing point must lie on the stop-line segment (± `segment_tolerance` of its length), so traffic in other lanes is not counted. Each track crosses at most once. Moving from the far side back to the approach side is never a crossing.

**Decision** (after smoothing), using the signal segment that contains the first frame at/after the line:

| Signal there | Rule | Outcome |
|---|---|---|
| `uncertain` | | `signal_uncertain`: listed for review, never flagged |
| red | crossing time − red onset > `red_grace_s` (default 0: strictly after) | `potential_violation` |
| red | crossed before red began (onset fell between the two frames) | `lawful` |
| green / amber | | `lawful` |

A vehicle that crossed on green or amber and is still in the junction when red comes on is lawful, because only the instant of crossing the stop line is judged.

Known limitation: if a vehicle's track is lost and a new ID starts after the line, no crossing is recorded (a miss, never a false alert).

## 5. The runner (`pipeline/runner.py`)

`analyse_video(video, config, tracker, out_dir)` returns plain data (no DB or web code):

1. **Pass 1**, every `frame_stride`-th frame: read the signal, run the tracker, update the crossing detector, store all detections.
2. Build the smoothed signal timeline and apply the decision rule to every crossing.
3. **Pass 2**: for each potential violation (and each `signal_uncertain` crossing), re-read the video around the crossing and render a snapshot and a clip.

Files written to `out_dir`: `result.json`, `tracks.npz` (all detections), `signal_readings.csv` (per-frame raw scores, raw state and smoothed state; used by `evaluate_signal.py`), and `evidence/`.

## 6. Evidence (`pipeline/evidence.py`, `pipeline/annotate.py`)

Clips cover `pre_s` before to `post_s` after the crossing at the original frame rate, with the stop line, lamp boxes, signal state, all tracked vehicles and the flagged vehicle highlighted. The banner reads "POTENTIAL red-light violation – for human review". Clips are written with OpenCV, then re-encoded to H.264/yuv420p with `ffmpeg` so browsers can play them. Without `ffmpeg` they stay MPEG-4 Part 2 and may not play in browsers; the dashboard warns about this.

## 7. Privacy (`pipeline/privacy.py`)

Used for the PDF's anonymised snapshots, the evidence ZIP and the accident demo. It only finds regions to **blur** and never reads plates:

- `standard`: OpenCV Haar faces (frontal and profile) and plate-like regions; for every tracked vehicle, the lower 35 % of the box (plate band) and a windscreen band (driver); for motorcycles, the top 40 % (rider's head).
- `strong`: additionally pixelates every vehicle box completely.

Blurring is applied to the raw frame **before** overlays, so labels stay readable. Automatic blurring can miss things, so a person must check files before publishing them. (OpenCV 5 removed the Haar cascades from the main package. The code then falls back to the box-based regions, and `/api/health` reports `haar_cascades: false`. The requirements pin OpenCV 4.x.)

## 8. Supplementary accident demonstration (`pipeline/accident.py`)

A person marks the approximate collision time. The module renders a contact sheet of frames from `before_s` to `after_s` around it, each labelled with its offset (t−2.0 s … t+2.0 s), and a clip with the banner "EXPLORATORY – … no claim about cause, speed or right of way". Vehicle tracks can optionally be drawn. It does not detect collisions and is not part of any evaluation.

## 9. Service, database and API (`service.py`, `db.py`, `main.py`)

SQLite tables: `videos` (with `location`, `session`, `is_cape_coast`, saved config), `analyses` (status, progress, stats), `events` (one row per crossing, with outcome, signal state, red onset, time since red, evidence paths and **review** fields), and `signal_segments`.

Analyses run one at a time in a background thread pool. Progress is written to the DB and polled by the dashboard. Jobs interrupted by a server restart are marked failed.

Main routes (all under `/api`, interactive docs at `/docs`): `videos` (upload, list, patch, delete, `frame?t=`, `config`, `signal-test`), `analyses` (start, list, cancel, delete, `signal-segments`, `summary`, `export.csv`, `export.pdf`, `export-evidence`), `events` (list with filters, `review`, `snapshot`, `clip`), `accident`, `files/…` (exports and accident outputs only, with path checks), `models`, `health`.

## 10. Reports (`reports.py`)

CSV columns include `cape_coast_footage`, `outcome`, signal state, red onset, time since red and review fields. The PDF states that alerts are potential violations, labels non–Cape Coast footage prominently, and reports the "share of reviewed alerts confirmed by reviewers" explicitly as **reviewer decisions, not system accuracy**. Accuracy is measured with the evaluation scripts against annotated ground truth.

## 11. Dashboard (`frontend/src/`)

React, bundled with esbuild into `backend/app/static/`. Pages: Videos, Configure (SVG overlay in video-pixel coordinates; drawing tools; signal test), Analyse, Review (keyboard review), Reports, Accident (exploratory). The persistent notice and the Cape Coast badges enforce the wording rules.

## 12. Synthetic test video (`tools/make_synthetic_video.py`)

A 60 s, 15 fps top-down junction with a scripted signal cycle (G 0–12, A 12–15, R 15–30, G 30–42, A 42–45, R 45–60 s), single-frame LED flicker, a 2 s occlusion of the signal, and 12 vehicles. Each vehicle covers one rule: lawful on green or amber; entered on amber and still in the junction at red; 0.1 s before red; 1, 3 and 9 s after red; stopped at the line and went on green; oncoming traffic off the stop line; crossing while the signal is hidden. It is test material only and never a result.
