# Red-light violation detection: research prototype

Prototype for the paper *AI-Based Detection of Red-Light Violations at Urban Intersections Using Lightweight Computer Vision: A Case Study of Cape Coast, Ghana* (China–Africa Urban Development Forum, UCC, December 2026).

**Recorded traffic video → YOLO11 (nano; medium as baseline) + ByteTrack → signal state read by HSV colour rules from a region you mark → stop-line crossing test → *potential* red-light violations → human review in a dashboard → CSV/PDF reports and anonymised evidence.**

An automated alert is always a **potential** red-light violation until a person has reviewed it. The prototype does no number-plate recognition, driver identification, enforcement or live CCTV.

How every part works is explained in [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md). The evaluation protocol and file formats are in [`evaluation/README.md`](evaluation/README.md).

## Install

Python 3.10+ (tested on 3.13), and `ffmpeg` on the PATH so that evidence clips play in browsers.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

The first YOLO run downloads `yolo11n.pt` / `yolo11m.pt` automatically. For a CPU-only machine, you can install the smaller CPU build of PyTorch first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Run the dashboard

```bash
cd backend && uvicorn app.main:app --reload
```

Open http://localhost:8000 (the API docs are at http://localhost:8000/docs). Data (database, uploaded videos, evidence) goes to `backend/data/` or to `$RLV_DATA_DIR`.

To try everything without real footage, load the synthetic test video first (it is labelled as synthetic and not Cape Coast):

```bash
python tools/seed_demo.py --analyse
```

### Workflow

1. **Videos**: upload a recorded video. Give the location and session, and tick *recorded in Cape Coast* only for Cape Coast footage. Anything else is labelled "Not Cape Coast footage" everywhere.
2. **Configure**: on a frame of the video, draw the **stop line**, click the **approach side**, drag a box around the **signal head** for that approach, and optionally outline the **approach zone** (the lanes). Scrub to red, amber and green frames and press **Test signal reading** to check the HSV thresholds (per-lamp scores are shown). Save.
3. **Analyse**: choose YOLO11n (proposed) or YOLO11m (baseline), or a fine-tuned checkpoint placed in `backend/data/models/`. Analyses run in the background.
4. **Review**: each potential violation has a clip, a still at the crossing frame, the signal state, the red onset and the time since red. Mark it **Confirm / Reject / Unsure** (keys `c` `r` `u`; `j`/`k` to move). Crossings made while the signal was unreadable are listed separately and are never flagged.
5. **Reports**: summary, signal timeline, CSV, PDF (with no images, anonymised snapshots, or originals for internal use), and an **evidence ZIP** that blurs faces, plate regions and drivers before overlays are drawn.
6. **Accident footage (exploratory)**: a supplementary demonstration. You mark an approximate collision time and get a contact sheet and clip of the frames before and after it. It makes no claim about cause, speed or right of way.

## Command line

```bash
python tools/make_synthetic_video.py samples/synthetic_junction.mp4
```

```bash
python tools/analyse_cli.py samples/synthetic_junction.mp4 --config samples/synthetic_junction.config.json --out runs/syn
```

```bash
python tools/analyse_cli.py junction.mp4 --config junction.json --model yolo11n.pt --out runs/j1 --mot runs/j1/pred_mot.txt
```

The config JSON is what the dashboard saves; `GET /api/videos/{id}/config` returns it.

## Tests

```bash
cd backend && python -m pytest -q
```

The tests use a synthetic video in which every decision rule has a scripted scenario (`tools/make_synthetic_video.py`). They cover the FastAPI routes through `TestClient` and the evaluation metrics. A YOLO smoke test runs when `backend/yolo11n.pt` (or `$RLV_YOLO_WEIGHTS`) exists.

## Frontend development

The dashboard is React, bundled by esbuild into `backend/app/static/` (checked in, so Node is not needed to run the app).

```bash
cd frontend && npm install && npm run build
```

## Layout

```
backend/app/pipeline/   geometry, signal_state, crossing, detectors, evidence, annotate, privacy, accident, runner
backend/app/            service.py (jobs, exports), db.py (SQLite), reports.py (CSV/PDF), main.py (FastAPI)
backend/app/static/     built dashboard
backend/tests/          pytest suite
frontend/src/           React dashboard source
evaluation/             split, train, detector_val, tracking_check, evaluate_signal, evaluate_events,
                        agreement, compare_models, benchmark
tools/                  synthetic video generator, CLI runner, demo seeder
docs/HOW_IT_WORKS.md    component-by-component explanation
```

## Ground rules this code enforces

- An alert is recorded only when the vehicle crosses the stop line **after** the signal for its approach turned red. Vehicles that entered on green or amber and are still in the junction at red are lawful.
- If the signal cannot be read, the state is `uncertain` and no alert is raised. The reader never guesses a colour.
- Detection, tracking, signal-state recognition and end-to-end violation detection are evaluated separately, each with its own metrics.
- Results from footage not marked as Cape Coast carry a "Not Cape Coast footage" label in the dashboard, CSV, PDF and ZIP.
- Evidence meant for sharing is anonymised automatically (faces, plate regions and drivers are blurred). Plates are never read. A person must check anonymised files before publishing them.
