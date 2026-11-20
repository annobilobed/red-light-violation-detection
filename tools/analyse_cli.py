#!/usr/bin/env python3
"""Analyse one recorded video from the command line (no web server).

Examples
  python tools/analyse_cli.py samples/synthetic_junction.mp4 \
      --config samples/synthetic_junction.config.json --out runs/syn
  python tools/analyse_cli.py junction.mp4 --config junction.json --model yolo11n.pt --out runs/j1 \
      --mot runs/j1/pred_mot.txt

The config JSON is the same structure the dashboard saves (AnalysisConfig).
Prints a summary and writes result.json, tracks.npz, signal_readings.csv and
evidence/ to --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.config import AnalysisConfig  # noqa: E402
from app.pipeline.detectors import make_tracker  # noqa: E402
from app.pipeline.evidence import TrackStore  # noqa: E402
from app.pipeline.runner import analyse_video  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--config", required=True, help="AnalysisConfig JSON")
    ap.add_argument("--model", help="override detector model (yolo11n.pt, yolo11m.pt, path/to/best.pt, scripted)")
    ap.add_argument("--tracker", help="bytetrack | botsort_static | botsort_reid | botsort | path.yaml")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out", help="output directory (omit for a dry run without files)")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--mot", help="also write tracks in MOTChallenge format to this file")
    args = ap.parse_args(argv)

    cfg = AnalysisConfig.from_dict(json.loads(Path(args.config).read_text()))
    if args.model:
        cfg.detector.model = args.model
    if args.conf is not None:
        cfg.detector.conf = args.conf
    if args.tracker:
        cfg.detector.tracker = args.tracker
    if args.stride:
        cfg.frame_stride = args.stride
    if args.device:
        cfg.detector.device = args.device
    problems = cfg.validate()
    if problems:
        print("Configuration incomplete:", *problems, sep="\n  ", file=sys.stderr)
        return 2

    def progress(f, msg):
        print(f"\r{100 * f:5.1f}%  {msg:<30}", end="", file=sys.stderr, flush=True)

    tracker = make_tracker(cfg.detector, args.video)
    result = analyse_video(args.video, cfg, tracker, args.out, progress, render_evidence=not args.no_evidence)
    print(file=sys.stderr)
    if args.mot:
        if not result["tracks_path"]:
            print("--mot needs --out", file=sys.stderr)
        else:
            rows = TrackStore.load(result["tracks_path"]).to_mot_rows()
            with open(args.mot, "w") as fh:
                for r in rows:
                    fh.write(",".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + "\n")
    print(f"Video: {result['video']['name']}  tracker: {result['tracker']}")
    print(f"Timing: {result['timing']}")
    print(f"Counts: {result['counts']}")
    for c in result["crossings"]:
        if c["outcome"] != "lawful":
            extra = f"  {c['time_since_red_s']:.2f}s after red" if c["time_since_red_s"] is not None else ""
            print(f"  {c['crossing_time_s']:8.2f}s  track {c['track_id']:<5} {c['vehicle_class']:<11} {c['outcome']}{extra}")
    print("Alerts are POTENTIAL red-light violations for human review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
