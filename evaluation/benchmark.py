#!/usr/bin/env python3
"""Processing SPEED of each model on a given machine (report the hardware!).

  python evaluation/benchmark.py --video clip.mp4 --models yolo11n.pt yolo11m.pt --frames 300 \
      --out results/benchmark.json [--config cfg.json]

Measures, after warm-up, per model: detection-only and detection+ByteTrack
milliseconds per frame and FPS. With --config, also the full pipeline
(signal reading + tracking + crossing test) via the runner, without evidence.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import _common  # noqa: F401
from _common import write_json


def read_frames(video, n):
    import cv2
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"cannot read {video}")
    return frames


def hardware():
    info = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        info["threads"] = torch.get_num_threads()
    except ImportError:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11m.pt"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device")
    ap.add_argument("--config", help="AnalysisConfig JSON for a full-pipeline timing")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args(argv)
    from ultralytics import YOLO
    frames = read_frames(args.video, args.frames + args.warmup)
    vehicle = None
    results = []
    for mpath in args.models:
        model = YOLO(mpath)
        names = model.names
        vehicle = [i for i, n in names.items() if n in ("car", "motorcycle", "bus", "truck")]
        kw = dict(imgsz=args.imgsz, device=args.device, classes=vehicle, verbose=False)
        for f in frames[:args.warmup]:
            model.predict(f, **kw)
        t0 = time.perf_counter()
        for f in frames[args.warmup:]:
            model.predict(f, **kw)
        det = (time.perf_counter() - t0) / len(frames[args.warmup:])
        model = YOLO(mpath)
        for f in frames[:args.warmup]:
            model.track(f, persist=True, tracker="bytetrack.yaml", **kw)
        t0 = time.perf_counter()
        for f in frames[args.warmup:]:
            model.track(f, persist=True, tracker="bytetrack.yaml", **kw)
        trk = (time.perf_counter() - t0) / len(frames[args.warmup:])
        row = {"model": mpath, "frames": len(frames) - args.warmup, "imgsz": args.imgsz,
               "detect_ms": round(1000 * det, 2), "detect_fps": round(1 / det, 2),
               "track_ms": round(1000 * trk, 2), "track_fps": round(1 / trk, 2)}
        if args.config:
            from app.pipeline.config import AnalysisConfig
            from app.pipeline.detectors import YoloTracker
            from app.pipeline.runner import analyse_video
            cfg = AnalysisConfig.from_dict(json.load(open(args.config)))
            cfg.detector.model = mpath
            cfg.detector.imgsz = args.imgsz
            cfg.detector.device = args.device
            r = analyse_video(args.video, cfg, YoloTracker(cfg.detector), None, render_evidence=False)
            row["pipeline_fps"] = r["timing"]["pipeline_fps"]
            row["pipeline_frames"] = r["timing"]["frames_processed"]
        print(row)
        results.append(row)
    hw = hardware()
    label = hw.get("gpu") or hw.get("cpu") or hw.get("processor") or "unknown"
    write_json(args.out, {"task": "speed_benchmark", "video": args.video, "device": args.device or "auto",
                          "device_label": label, "hardware": hw, "models": results})


if __name__ == "__main__":
    main()
