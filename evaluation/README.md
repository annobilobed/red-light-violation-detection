# Evaluation protocol

Four tasks, measured **separately**, each on held-out **test sessions** of Cape Coast footage. Never pool non–Cape Coast footage into Cape Coast results.

| Task | Script | Ground truth | Metrics |
|---|---|---|---|
| Vehicle detection | `detector_val.py` | CVAT boxes (YOLO 1.1 export) | mAP@0.5, mAP@0.5:0.95, precision, recall (overall, per class) |
| Tracking | `tracking_check.py` | CVAT tracks (MOT 1.1 export) | MOTA, IDF1, ID switches, fragmentations |
| Signal-state recognition | `evaluate_signal.py` | signal intervals CSV | coverage, accuracy on covered frames, per-class P/R/F1, confusion matrix, red↔other errors |
| End-to-end potential violations | `evaluate_events.py` | violation events CSV | precision, recall, F1 with 95 % Wilson intervals; system alerts and reviewer-confirmed alerts |
| Speed | `benchmark.py` | – | ms/frame and FPS (detection, detection+tracking, full pipeline), with hardware recorded |
| Annotation quality | `agreement.py` | two annotators | Cohen's κ, percent agreement |
| Model comparison | `compare_models.py` | outputs above | mean ± SD across seeds, nano vs medium |

## Order of work

1. Annotate a sample twice and run `agreement.py`. Resolve disagreements before using labels.
2. `split_sessions.py`: split whole sessions into train/val/test (stratified by day/dusk/night with `--meta`). Fix the test sessions with `--test-sessions` once chosen, and do not change them after seeing results.
3. `train.py`: YOLO11n and YOLO11m, 3 seeds each, same settings.
4. `detector_val.py` on the test split for all 6 runs, then `compare_models.py`.
5. For each test video: run the analysis (dashboard or `tools/analyse_cli.py --mot`) with the chosen model, then:
   - `tracking_check.py` against MOT ground truth;
   - `evaluate_signal.py` with that analysis' `signal_readings.csv`;
   - export the events CSV, and after review export it again, then run `evaluate_events.py`.
6. `benchmark.py` on the target hardware.

## File formats

**Signal intervals** (`evaluate_signal.py --gt`, `agreement.py intervals`):
```
video,start_s,end_s,state
kotokuraba_s3.mp4,0.0,31.2,green
kotokuraba_s3.mp4,31.2,34.1,amber
kotokuraba_s3.mp4,34.1,70.0,red
```
`state` ∈ red / amber / green. Leave times you cannot judge out of the file; they are not scored.

**Violation events** (`evaluate_events.py --gt`):
```
video,track_hint,crossing_time_s,label
kotokuraba_s3.mp4,white taxi,41.8,potential_violation
```
Only rows labelled `potential_violation` (annotators judged that the vehicle crossed the stop line after red) are positives. A predicted alert matches if its crossing time is within `--tolerance` seconds (default 1.0) in the same video.

**Predictions** (`--pred`): the dashboard's CSV export (`/api/analyses/{id}/export.csv`), unchanged.

**MOT** (`tracking_check.py`): `frame,id,x,y,w,h[,conf,...]`, with 1-based frames. `tools/analyse_cli.py --mot` writes predictions in this format.

## Reporting rules

- Report every number with its sample size (frames, events, sessions) and, for events, the confidence interval.
- Report coverage alongside signal accuracy. A reader that says `uncertain` often can look accurate on the frames it does read.
- The "reviewed alerts confirmed" share in the dashboard is a count of reviewer decisions, not a system metric.
- Write results in Section 5 only from these output files. Do not write expected outcomes as findings.
