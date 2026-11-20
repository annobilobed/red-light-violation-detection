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

## Status (20 Nov 2026, end of session 2)
113 pytest tests pass (`cd backend && python -m pytest -q`; one classifier test is skipped unless `RLV_SIGNAL_CLASSIFIER` points at a trained model).

Built and verified:
- Pipeline, service, DB, reports, FastAPI, React dashboard (driven end to end in headless Chromium against uvicorn, no console errors).
- YOLO11n/YOLO11m + ByteTrack verified on public non–Cape Coast clips. Tracker choice: `bytetrack` (default), `botsort_static`, `botsort_reid`, `botsort`; `evaluation/compare_trackers.py`.
- OpenVINO/ONNX export (`tools/export_model.py`); YOLO11n 29.0 → 16.5 ms/frame on a 4-core CPU, same detections.
- Signal reading: core+colour rule for over-exposed lamps, `peak_min` safety guard, defaults chosen on real MIT-licensed US crops (held-out coverage 85.5%, zero wrong colours), calibration from labelled frames (Configure page, API, `tools/calibrate_signal.py`). Real-lamp fixture: `backend/tests/fixtures/real_lamps/`.
- Optional CNN signal classifier (`reader: hybrid|agree|cnn`), training and crop-export tools. The starter model (trained on public crops) does not transfer to other footage: retrain on Cape Coast crops.
- Anonymisation: res10 DNN faces (`tools/download_models.py`), YOLO person heads, vehicle bands 10–100%, human sign-off on the Review page, ZIP filter for signed-off events.
- Ultralytics usage analytics disabled via `detectors.load_yolo()` and in the scripts.
- Docker: built and run with `--network none` here; health OK, 113 tests pass inside, YOLO analysis works. Build args `BASE_IMAGE`, `TORCH_INDEX`, optional CA secret.

Not verified (needs Cape Coast data):
- Signal thresholds and classifier on Cape Coast signals (day/dusk/night). The defaults come from US lamps.
- Tracker choice, detector accuracy, end-to-end precision/recall on Cape Coast footage.
- Anonymisation recall on Ghanaian plates and local scenes (the human sign-off stays mandatory).
- The CPU-only PyTorch Docker build (the CPU wheel index was blocked here; built with PyPI torch, 10.5 GB image).

## Remaining work, in order
1. Record Cape Coast footage; at each site calibrate signal reading on the Configure page (label frames per lighting condition).
2. Annotate (CVAT), `agreement.py`, `split_sessions.py`, `train.py` (3 seeds × 2 models), `detector_val.py`, `compare_models.py`.
3. `compare_trackers.py` on test clips to choose the tracker; `benchmark.py` for PyTorch vs OpenVINO.
4. Optional: export signal crops from TRAIN sessions, train the classifier, compare readers with `evaluate_signal.py` on TEST sessions.
5. `evaluate_signal.py` and `evaluate_events.py` on held-out test sessions; anonymise and sign off any material for publication.
6. Paper: Section 5 from the evaluation outputs only; methods in past tense once done. Never write expected outcomes as findings.

## Conventions
- Python 3.10+. Keep `runner.py` free of DB and web code (it returns plain data).
- Load YOLO models with `detectors.load_yolo()` (disables Ultralytics analytics).
- Signal decisions live in one function, `signal_state.decide_arrays`, used by both the reader and calibration; change it there only.
- OpenCV must stay 4.x (OpenCV 5 dropped `cv2.CascadeClassifier`); use `opencv-python`, not both it and `-headless`.
- Frontend rebuild: `cd frontend && npm install && npm run build` (writes `backend/app/static/app.js` and `app.css`). FrameCanvas `onChange` receives an updater function.
- Tests use the synthetic video and `ScriptedDetector`. When you add a decision rule, add a scenario to `tools/make_synthetic_video.py` and an assertion to `tests/test_end_to_end.py`.
- `tools/seed_demo.py --analyse` loads the synthetic video into a data dir for trying the dashboard without YOLO.
