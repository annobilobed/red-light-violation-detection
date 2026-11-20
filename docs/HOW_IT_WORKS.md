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

`YoloTracker` calls `model.track(frame, persist=True, tracker=..., classes=[...])` on every processed frame. Class IDs are looked up **by name** from the model (`car`, `motorcycle`, `bus`, `truck` for COCO). A fine-tuned model must use those names or set `vehicle_classes`. Boxes without a track ID are ignored. `reset()` clears tracker state between videos, so IDs restart. Models are loaded through `load_yolo()`, which switches off Ultralytics' anonymous usage analytics (on by default in Ultralytics).

**Trackers** (`detector.tracker`, chosen on the Analyse page):

| id | What it is |
|---|---|
| `bytetrack` (default) | ByteTrack: motion + box overlap, uses low-confidence boxes to keep tracks through partial occlusion |
| `botsort_static` | BoT-SORT with camera-motion compensation switched off (`pipeline/trackers/botsort_static.yaml`); for fixed cameras |
| `botsort_reid` | as above plus appearance re-identification from the detector's own features (no extra model) |
| `botsort` | stock BoT-SORT with camera-motion compensation (for moving cameras) |

Which tracker is best has to be measured on Cape Coast clips with `evaluation/compare_trackers.py`. On a public dashcam clip (300 frames, CPU), `bytetrack` and `botsort_static` produced the same 10 track IDs at 33 and 35 ms/frame, while `botsort_reid` fragmented into 17 IDs.

**Model formats.** `tools/export_model.py` exports a checkpoint to OpenVINO (or ONNX) into `backend/data/models/`, where the dashboard lists it. On a 4-core Xeon, YOLO11n took 29.0 ms/frame in PyTorch and 16.5 ms/frame in OpenVINO, with the same detections (162 of 163 boxes matched, mean IoU 0.98); the full pipeline ran at 32 vs 56 frames/s. ONNX Runtime was slower than PyTorch on that machine (39.6 ms/frame).

`ScriptedDetector` replays ground-truth boxes written by the synthetic video generator. It is used only for tests and the demo.

## 3. Signal-state reading (`pipeline/signal_state.py`, `pipeline/calibration.py`)

**Per frame, colour rules (`reader: hsv`, the default).** For each lamp box (thirds of the signal box, or custom boxes), convert to HSV and compute:
- the **colour score**: share of pixels that are bright (`V ≥ val_min`), saturated (`S ≥ sat_min`) and of that lamp's own hue (OpenCV hue 0–180: red 0–10 and 160–180, amber 11–34, green 35–100);
- the **bright-core share**: share of pixels with `V ≥ core_val`;
- the **peak**: 95th percentile of `V`.

Rules, in order (one shared, vectorised function, `decide_arrays`, is used by the reader and by calibration):
1. **Colour**: the best colour score is at least `min_score` and beats the runner-up by `margin`.
2. **Core + colour** (`core_agreement`, on by default): for over-exposed lamps whose centre is white. The lamp with the most very bright pixels (≥ `core_min`, ahead by `core_margin`) must also be the lamp with the highest colour score (≥ `weak_min`). Both cues must point at the same lamp.
3. **Brightness fallback** (off by default): position only, for night footage; validate before use.

**Safety guard** for every rule: the chosen lamp's peak must reach `peak_min` (130). Lit lamps are light sources; foliage or a green vehicle in front of the signal usually is not. This was added after the synthetic occlusion (dull green foliage over the signal) was read as a green lamp once the brightness threshold was lowered. Brightly sunlit green objects can still pass it, which is one reason for the classifier option below. Otherwise the state is **`uncertain`**: the reader never guesses.

**Defaults and real-lamp results.** The defaults were chosen on the training split (1187 crops) of a public, MIT-licensed set of real signal-head crops (US roads, *not* Cape Coast; `backend/tests/fixtures/real_lamps/README.md`). On its held-out split (297 crops): previous defaults read 68.4% of lamps; current defaults read 85.5%; both with **zero wrong colours**. Amber is weakest (3 of 9) because amber lamps are often over-exposed and the set has few of them.

**Calibration** (Configure page panel, `POST /api/videos/{id}/signal-calibrate`, `tools/calibrate_signal.py`). A person labels frames as red, amber or green. A grid of about 64,000 threshold combinations is scored on them, and the recommendation is the setting with the highest coverage that has **no red/other confusion** and no wrong colour (ties → stricter thresholds; if the current settings are already best, keep them). `peak_min` is not searched, because calibration frames rarely contain occlusions. Calibration numbers are measured on the labelled frames and are therefore optimistic; signal accuracy must be measured on other sessions with `evaluate_signal.py`. The labels are stored in `signal.samples`.

**Optional CNN classifier** (`reader: cnn | hybrid | agree`, `signal.classifier`). A YOLO11n-cls model with classes red, amber, green and **none** (unlit, hidden, not a signal) looks at the whole signal box. It commits only at confidence ≥ `classifier_conf` (0.9).
- `hybrid`: if rules and classifier give different colours → `uncertain`; if only one commits, its colour is used; a confident `none` from the classifier vetoes the rules.
- `agree`: both must give the same colour (safest).

Starter model, trained on the public crops above plus "none" crops from public road videos, measured on the held-out split (297 lamps + 350 non-lamp crops):

| Reader | Lamps read | Wrong colours | Non-lamp crops given a colour |
|---|---|---|---|
| hsv | 85.5% | 0 | 42 / 350 |
| cnn (0.9) | 98.3% | 0 | 3 / 350 |
| hybrid (0.9) | 98.7% | 0 | 5 / 350 |
| agree (0.9) | 85.2% | 0 | 1 / 350 |

Several of the remaining non-lamp errors come from green outlines burnt into one public road video. The starter model does **not** transfer to different-looking footage: on the synthetic video's drawn lamps it answers `none`, so hybrid/agree return `uncertain` (fail-safe, no false alerts). A classifier must therefore be trained on Cape Coast crops (`tools/export_signal_crops.py`, `evaluation/train_signal_classifier.py`) before it is used for results.

**Over time** (offline, once the whole video has been read):

1. Colour runs shorter than `min_stable_frames` become `uncertain` (removes one-frame glitches).
2. `uncertain` gaps no longer than `bridge_gap_s`, with the **same** colour on both sides, take that colour. This handles LED flicker (red–dark–red is still red). A gap between **different** colours (e.g. amber→?→red) stays `uncertain`, because the change happened at an unknown moment inside it.

The result is a list of segments. A red segment's **onset** is the time of its first frame. `onset_exact` is false when the previous segment was `uncertain`; red then began at or before the recorded onset.

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

Used for the PDF's anonymised snapshots, the evidence ZIP, the Review page's anonymised preview and the accident demo. It only finds regions to **blur** and never reads plates or identifies anyone. Layers are unioned, so one layer missing something is covered by another:

1. **Faces**: OpenCV's res10 SSD face detector (official OpenCV files, fetched and checksum-checked by `tools/download_models.py`), run on the whole frame and on overlapping tiles so small faces are found. Haar cascades are used if the model files are missing.
2. **People**: the COCO YOLO model's `person` class. The head region of every person box is blurred, and faces are searched in an upscaled crop of each box. This covers pedestrians, riders and roadside workers.
3. **Vehicle bands** from the tracked boxes: 10–50% of the height (windscreen and driver) and 40–100% (number plate); motorcycles also get the top half (rider). The plate band was widened after a real rear-view photo, with the plate at 47% of the car's height, stayed partly legible with the earlier band.
4. OpenCV's generic plate cascade (weak on Ghanaian plates; the bands are the main safeguard).

Blur strength is set from each region's longer side, so a thin band across a car still destroys plate characters. `strong` mode pixelates every vehicle and person box completely. Blurring is applied to the raw frame **before** overlays.

Checked on real photos (not Cape Coast): two faces in a close-up and the visible faces in a street scene were blurred, and a European rear plate became unreadable.

**Human sign-off.** Automatic blurring can miss things, so the Review page shows the anonymised still of each event and records "checked by … at …" (`PATCH /api/events/{id}/anon-check`). The evidence ZIP can include only signed-off events, and its README lists the check status of each event. A sign-off covers the still; clips must be watched before they are published.

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
