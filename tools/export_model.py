#!/usr/bin/env python3
"""Export a YOLO checkpoint to OpenVINO (fastest on Intel/AMD CPUs) or ONNX,
and put it where the dashboard lists it (backend/data/models/).

  python tools/export_model.py yolo11n.pt                      # OpenVINO FP32
  python tools/export_model.py runs/train/yolo11n_seed0/weights/best.pt --name capecoast_n
  python tools/export_model.py yolo11n.pt --format onnx

Measured on a 4-core Xeon CPU (1280x720 dashcam clip, 200 frames, YOLO11n +
ByteTrack): PyTorch 29.0 ms/frame, OpenVINO 16.5 ms/frame, ONNX Runtime
39.6 ms/frame; OpenVINO and ONNX found the same vehicles as PyTorch (162 of
163 boxes matched, mean IoU 0.98-0.99). Measure on your own hardware with
evaluation/benchmark.py, and re-run detector_val.py on the exported model
before reporting its accuracy.

--int8 (OpenVINO only) quantises with calibration images from --data; it is
faster still but changes accuracy, so validate it like a new model.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--format", choices=["openvino", "onnx"], default="openvino")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--data", help="dataset YAML for INT8 calibration")
    ap.add_argument("--name", help="name in the model list (default: weights file name)")
    ap.add_argument("--dest", default=os.environ.get("RLV_DATA_DIR", str(ROOT / "backend" / "data")))
    args = ap.parse_args(argv)
    if args.int8 and (args.format != "openvino" or not args.data):
        ap.error("--int8 needs --format openvino and --data")
    from ultralytics import YOLO, settings
    settings.update({"sync": False})  # no usage analytics
    out = Path(YOLO(args.weights).export(format=args.format, imgsz=args.imgsz, int8=args.int8,
                                         data=args.data, dynamic=False))
    name = args.name or Path(args.weights).stem
    dest_dir = Path(args.dest) / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if args.format == "openvino":
        dest = dest_dir / f"{name}{'_int8' if args.int8 else ''}_openvino_model"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
    else:
        dest = dest_dir / f"{name}.onnx"
        shutil.copy2(out, dest)
    print(f"exported to {dest}")
    print("It now appears in the dashboard's model list (Analyse page).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
