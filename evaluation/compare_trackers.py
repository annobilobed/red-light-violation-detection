#!/usr/bin/env python3
"""Compare trackers on the same clip against MOT ground truth.

  python evaluation/compare_trackers.py --video clip.mp4 --gt annot/clip_gt_mot.txt \
      --model runs/train/yolo11n_seed0/weights/best.pt --out results/trackers.json

Runs each tracker (default: bytetrack, botsort_static, botsort_reid) with the
same detector and reports MOTA, IDF1, ID switches, fragmentations and speed.
Use the tracker that does best on Cape Coast test clips; ID switches near the
stop line matter most, because they can create or lose a crossing.
"""
from __future__ import annotations

import argparse
import time

import _common  # noqa: F401
from _common import fmt, write_json
from metrics import read_mot, tracking_metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True, help="MOT ground truth (frames 1-based)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort_static", "botsort_reid"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device")
    ap.add_argument("--gt-frames-only", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    import cv2
    from app.pipeline.config import DetectorConfig
    from app.pipeline.detectors import YoloTracker
    gt = read_mot(args.gt)
    rows = []
    for name in args.trackers:
        tr = YoloTracker(DetectorConfig(model=args.model, tracker=name, conf=args.conf, imgsz=args.imgsz,
                                        device=args.device))
        tr.reset()
        cap = cv2.VideoCapture(args.video)
        pred, idx, t0 = {}, 0, time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = tr.track(frame, idx)
            if dets:
                pred[idx + 1] = [(d.track_id, d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1])
                                 for d in dets]
            idx += 1
        cap.release()
        ms = 1000 * (time.perf_counter() - t0) / max(1, idx)
        if args.gt_frames_only:
            pred = {f: v for f, v in pred.items() if f in gt}
        m = tracking_metrics(gt, pred)
        m.update({"tracker": name, "ms_per_frame": round(ms, 2)})
        rows.append(m)
        print(f"{name:15s} MOTA={fmt(m['mota'])} IDF1={fmt(m['idf1'])} IDsw={m['id_switches']} "
              f"frag={m['fragmentations']} {ms:.1f} ms/frame")
    if args.out:
        write_json(args.out, {"task": "tracker_comparison", "video": args.video, "gt": args.gt,
                              "model": args.model, "results": rows})


if __name__ == "__main__":
    main()
