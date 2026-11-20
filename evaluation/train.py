#!/usr/bin/env python3
"""Fine-tune the nano (proposed) and medium (baseline) YOLO models.

Default: yolo11n.pt and yolo11m.pt, 3 seeds each, same data and settings, so
the comparison reports mean ± SD across seeds.

  python evaluation/train.py --data datasets/capecoast/data.yaml --epochs 100
Outputs: runs/train/<model>_seed<k>/weights/best.pt and runs/train/manifest.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from _common import write_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11m.pt"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device")
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = ap.parse_args(argv)

    runs = []
    for model in args.models:
        for seed in args.seeds:
            name = f"{Path(model).stem}_seed{seed}"
            runs.append({"model": model, "seed": seed, "name": name,
                         "weights": str(Path(args.project) / name / "weights" / "best.pt")})
    if args.dry_run:
        for r in runs:
            print(r)
        return
    from ultralytics import YOLO, settings
    settings.update({"sync": False})  # no usage analytics
    for r in runs:
        print(f"=== training {r['name']}")
        YOLO(r["model"]).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                               seed=r["seed"], deterministic=True, patience=args.patience, device=args.device,
                               project=args.project, name=r["name"], exist_ok=True)
    write_json(Path(args.project) / "manifest.json", {"data": args.data, "epochs": args.epochs,
                                                      "imgsz": args.imgsz, "runs": runs})


if __name__ == "__main__":
    main()
