#!/usr/bin/env python3
"""TRACKING metrics (separate task): MOTA, IDF1, ID switches, fragmentations.

  python tools/analyse_cli.py clip.mp4 --config cfg.json --out runs/c --mot runs/c/pred.txt
  python evaluation/tracking_check.py --gt annot/clip_gt_mot.txt --pred runs/c/pred.txt

Both files in MOTChallenge format: frame,id,x,y,w,h[,conf,...] (frames 1-based).
Export ground truth from CVAT as "MOT 1.1". Annotate every frame used (or set
the same --frame-stride for both). ID switches matter here: a switch at the
stop line can create or lose a crossing.
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401
from _common import fmt, write_json
from metrics import read_mot, tracking_metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--gt-frames-only", action="store_true",
                    help="score only frames that have ground truth (for sparsely annotated clips)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    gt, pred = read_mot(args.gt), read_mot(args.pred)
    if args.gt_frames_only:
        pred = {f: v for f, v in pred.items() if f in gt}
    m = tracking_metrics(gt, pred, args.iou)
    print(f"MOTA={fmt(m['mota'])} IDF1={fmt(m['idf1'])} IDsw={m['id_switches']} frag={m['fragmentations']} "
          f"FP={m['fp']} FN={m['fn']} gt_tracks={m['gt_tracks']} pred_tracks={m['pred_tracks']}")
    if args.out:
        write_json(args.out, {"task": "tracking", "gt": args.gt, "pred": args.pred, "iou": args.iou, **m})
    return m


if __name__ == "__main__":
    main()
