#!/usr/bin/env python3
"""Vehicle DETECTION metrics on the held-out test split (separate task).

  python evaluation/detector_val.py --data datasets/capecoast/data.yaml \
      --weights runs/train/*/weights/best.pt --out results/detector

Writes one JSON per weights file: mAP@0.5, mAP@0.5:0.95, precision, recall,
overall and per class.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from _common import write_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device")
    ap.add_argument("--out", default="results/detector")
    args = ap.parse_args(argv)
    from ultralytics import YOLO, settings
    settings.update({"sync": False})  # no usage analytics
    for w in args.weights:
        model = YOLO(w)
        m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device, verbose=False,
                      plots=False)
        names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
        per_class = {}
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            per_class[names[int(c)]] = {"precision": float(p), "recall": float(r), "map50": float(ap50),
                                        "map50_95": float(ap)}
        run = Path(w).parent.parent.name if Path(w).parent.name == "weights" else Path(w).stem
        write_json(Path(args.out) / f"{run}.json", {
            "task": "vehicle_detection", "weights": w, "run": run, "split": args.split, "data": args.data,
            "map50": float(m.box.map50), "map50_95": float(m.box.map), "precision": float(m.box.mp),
            "recall": float(m.box.mr), "per_class": per_class,
            "speed_ms": dict(m.speed)})


if __name__ == "__main__":
    main()
