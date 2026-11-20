#!/usr/bin/env python3
"""Tune the HSV signal-reading thresholds from labelled frames or crops.

From a video (the usual case): label time intervals of the signal state in a
CSV (start_s,end_s,state, same format as evaluation ground truth) and give the
analysis config with the signal box already drawn:

  python tools/calibrate_signal.py --video junction.mp4 --config junction.json \
      --labels junction_signal_labels.csv --step 1.0 --write-config junction.calibrated.json

From signal-head crops (each image = one head, vertical layout, red on top),
in folders named red/ amber/ (or yellow/) green/:

  python tools/calibrate_signal.py --crops path/to/crops

Frames you calibrate on are not evaluation data: measure accuracy afterwards
on OTHER sessions with evaluation/evaluate_signal.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.calibration import apply, calibrate  # noqa: E402
from app.pipeline.config import AnalysisConfig, SignalConfig  # noqa: E402
from app.pipeline.evidence import probe_video  # noqa: E402

CROP_SIZE = (32, 64)


def samples_from_video(video: str, labels: str, step: float):
    meta = probe_video(video)
    with open(labels) as fh:
        rows = [r for r in csv.DictReader(fh) if r["state"].strip().lower() in ("red", "amber", "yellow", "green")]
    cap = cv2.VideoCapture(video)
    out = []
    for r in rows:
        t, end = float(r["start_s"]), float(r["end_s"])
        # Skip 0.5 s at both ends of each interval (transitions are hard to label exactly).
        t += 0.5
        while t < end - 0.5:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * meta["fps"])))
            ok, frame = cap.read()
            if ok:
                out.append((frame, r["state"]))
            t += step
    cap.release()
    return out


def samples_from_crops(folder: str):
    out = []
    for d in sorted(Path(folder).iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            im = cv2.imread(str(f))
            if im is not None:
                out.append((cv2.resize(im, CROP_SIZE, interpolation=cv2.INTER_AREA), d.name))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--crops")
    ap.add_argument("--config", help="AnalysisConfig JSON with the signal box (needed with --video)")
    ap.add_argument("--labels", help="CSV start_s,end_s,state (needed with --video)")
    ap.add_argument("--step", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--max-wrong-rate", type=float, default=0.0)
    ap.add_argument("--out", help="write the full result JSON here")
    ap.add_argument("--write-config", help="write the config with the recommended thresholds here")
    args = ap.parse_args(argv)

    if args.video:
        if not (args.config and args.labels):
            ap.error("--video needs --config and --labels")
        cfg = AnalysisConfig.from_dict(json.loads(Path(args.config).read_text()))
        samples = samples_from_video(args.video, args.labels, args.step)
        base = cfg.signal
    else:
        cfg = None
        samples = samples_from_crops(args.crops)
        base = SignalConfig(roi=[0, 0, *CROP_SIZE])
    print(f"{len(samples)} labelled samples", file=sys.stderr)
    res = calibrate(samples, base, max_wrong_rate=args.max_wrong_rate)
    for w in res["warnings"]:
        print("warning:", w)
    print(res["message"])
    if res["recommended"]:
        print("recommended:", json.dumps(res["recommended"]["settings"]))
        for c, v in res["recommended"]["metrics"]["per_class"].items():
            print(f"  {c:6s} read {v['read_correctly']}/{v['n']}  uncertain {v['uncertain']}  wrong {v['wrong']}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    if args.write_config:
        if cfg is None or not res["recommended"]:
            print("nothing to write (needs --video/--config and a recommendation)", file=sys.stderr)
            return 1
        cfg.signal = apply(cfg.signal, res["recommended"]["settings"])
        Path(args.write_config).write_text(json.dumps(cfg.to_dict(), indent=2))
        print(f"wrote {args.write_config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
