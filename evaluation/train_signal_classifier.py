#!/usr/bin/env python3
"""Train the optional CNN signal-state classifier (YOLO11n-cls).

Dataset: folders of signal-head crops (the region you draw on the Configure
page), one folder per class, split by SESSION (never by frame):
  <data>/train/{red,amber,green,none}/*.png
  <data>/val/{red,amber,green,none}/*.png
"none" = unlit, occluded, blurred, or not a signal. It lets the classifier say
it cannot tell. Build the folders from your footage with
tools/export_signal_crops.py.

  python evaluation/train_signal_classifier.py --data datasets/signal_crops --epochs 40
Output: runs/signal_cls/<name>/weights/best.pt. Copy it to backend/data/models/
signal_classifier.pt (or set signal.classifier in the config) to use it.

Hue augmentation is disabled: shifting hue would turn red into amber.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from _common import write_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n-cls.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--project", default="runs/signal_cls")
    ap.add_argument("--name", default="signal_cls")
    args = ap.parse_args(argv)
    classes = sorted(p.name for p in (Path(args.data) / "train").iterdir() if p.is_dir())
    if set(classes) != {"red", "amber", "green", "none"}:
        raise SystemExit(f"expected class folders red, amber, green, none; found {classes}")
    from ultralytics import YOLO, settings
    settings.update({"sync": False})  # no usage analytics
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=args.seed,
                deterministic=True, device=args.device, project=args.project, name=args.name, exist_ok=True,
                hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, fliplr=0.5, erasing=0.1, plots=False)
    best = Path(args.project) / args.name / "weights" / "best.pt"
    write_json(Path(args.project) / args.name / "manifest.json", {
        "task": "signal_classifier_training", "data": args.data, "epochs": args.epochs, "imgsz": args.imgsz,
        "seed": args.seed, "classes": classes, "weights": str(best)})
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
