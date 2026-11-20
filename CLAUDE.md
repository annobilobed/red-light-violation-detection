# CLAUDE.md: context for continuing development

## What this project is
Research prototype for the paper *AI-Based Detection of Red-Light Violations at Urban Intersections Using Lightweight Computer Vision: A Case Study of Cape Coast, Ghana* (China–Africa Urban Development Forum, UCC, December 2026, Sub-theme 2).

Recorded traffic video → YOLO (nano: yolo11n; baseline: yolo11m) + ByteTrack → HSV signal reading from a user-marked region → stop-line crossing test → **potential red-light violation** events for human review → FastAPI + SQLite + React dashboard → CSV/PDF export. `README.md` covers usage; `docs/HOW_IT_WORKS.md` explains every component; `evaluation/README.md` gives the evaluation protocol.

## Ground rules (from the authors' project guide; do not break these)
- Build what the paper says was built, measure what it says was measured, report only what the evidence supports.
- An automated alert is always a "potential red-light violation", never a violation, until human review.
- A potential violation is recorded only when the vehicle crosses the stop line **after** the applicable signal turned red. A vehicle that entered on green or amber and is still in the junction at red is lawful.
- If the signal can't be read, the state is `uncertain`. Never guess a colour.
- Keep vehicle detection, tracking, signal-state recognition and end-to-end violation detection as separate tasks with separate metrics.
- Never present results from non-Cape Coast footage as Cape Coast results.
- No number-plate recognition, driver identification, enforcement decisions or live CCTV in this version.
- The accident-footage module (UCC East Gate) is a supplementary demonstration only; it must not claim crash causes.

## Status (20 Nov 2026)
Built and verified in this session (82 pytest tests pass: `cd backend && python -m pytest -q`):
- Pipeline `backend/app/pipeline/`: config, geometry, signal_state, crossing, detectors, evidence, annotate, privacy, accident, runner.
- `service.py`, `db.py`, `reports.py`, `main.py`. FastAPI was run with uvicorn and every route is covered by TestClient tests (`tests/test_api.py`).
- Dashboard `frontend/src/` (React → esbuild → `backend/app/static/`). Driven end to end in headless Chromium against the real uvicorn server: upload, drawing tools, signal test, YOLO analysis, review with keyboard, CSV/PDF/ZIP export, accident page. No console errors.
- `YoloTracker` ran with real yolo11n.pt / yolo11m.pt on public non–Cape Coast clips (dashcam and fixed-camera road): class filter, persistent IDs, `reset()`, about 33 ms/frame (n) and 150 ms/frame (m) on a 4-core CPU. These clips have no traffic signal, so they test tracking only.
- `evaluation/train.py`, `detector_val.py` and `benchmark.py` ran successfully (train/val smoke test on Ultralytics' coco8; the numbers are meaningless). All evaluation metrics have unit tests.
- Privacy (former item 4): anonymisation in `pipeline/privacy.py`, used by the PDF snapshots, the evidence ZIP and the accident demo.
- Accident demo (former item 5): `pipeline/accident.py` + Accident page.
- H.264 conversion verified (ffprobe reports h264/yuv420p).

Not verified:
- No real signalised-junction footage was available, so the HSV thresholds are untuned for real lamps (daylight/dusk/night). The defaults work on the synthetic video only.
- The Dockerfile has not been built.
- Haar cascades for faces/plates are generic OpenCV models; their recall on Cape Coast footage is unknown. The box-based blur regions are the main safeguard.

## Remaining work, in order
1. Signal reading on real Cape Coast footage: use Configure → Test signal reading on red/amber/green frames at each site and time of day; tune `sat_min`, `val_min`, `min_score`, `margin`; decide whether `brightness_fallback` is needed at night (validate it with `evaluate_signal.py` before using it).
2. Run yolo11n on real Cape Coast junction video; check missed/duplicated tracks at the stop line; tune `conf`, `line_margin_px`, `min_approach_frames`, approach zone.
3. Field data (the authors do the recording and annotation): permissions, recording, CVAT annotation, `agreement.py`, `split_sessions.py`, `train.py` (3 seeds × 2 models), then every evaluation script on the held-out test sessions (see `evaluation/README.md`).
4. Check anonymisation output on real footage before anything is shared.
5. Paper: fill Section 5 from the evaluation outputs and change the planned methods to past tense once the work is done (see the tense rules in the authors' guide). Never write expected outcomes as findings.

Nice to have: a Colab notebook, several approaches per camera, filter arrows for turning traffic, automatic re-calibration when the camera moves.

## Conventions
- Python 3.10+. Keep `runner.py` free of DB and web code (it returns plain data).
- OpenCV must stay 4.x (OpenCV 5 dropped `cv2.CascadeClassifier`); use `opencv-python`, not both it and `-headless`.
- Frontend rebuild: `cd frontend && npm install && npm run build` (writes `backend/app/static/app.js` and `app.css`). FrameCanvas `onChange` receives an updater function.
- Tests use the synthetic video and `ScriptedDetector`. When you add a decision rule, add a scenario to `tools/make_synthetic_video.py` and an assertion to `tests/test_end_to_end.py`.
- `tools/seed_demo.py --analyse` loads the synthetic video into a data dir for trying the dashboard without YOLO.
