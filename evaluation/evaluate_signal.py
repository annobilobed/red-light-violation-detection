#!/usr/bin/env python3
"""SIGNAL-STATE recognition metrics (separate task).

  python evaluation/evaluate_signal.py --gt annot/signal_gt.csv \
      --pred backend/data/analyses/12/signal_readings.csv --video junction_s3.mp4 --condition dusk

--gt: CSV start_s,end_s,state[,video]  (rows filtered by --video if given)
--pred: signal_readings.csv written by every analysis (per processed frame)
Uses the smoothed state by default (--raw for single-frame readings).
Reports coverage (share of frames where a colour was given), accuracy on
covered frames, per-class P/R/F1, the confusion matrix, and the two
safety-relevant error counts (red read as green/amber; non-red read as red).
"""
from __future__ import annotations

import argparse
import csv

import _common  # noqa: F401
from _common import fmt, write_json
from metrics import signal_metrics


def read_gt(path, video=None):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if video and "video" in rows[0]:
        rows = [r for r in rows if r["video"] == video]
    return [(float(r["start_s"]), float(r["end_s"]), r["state"].strip()) for r in rows]


def read_pred(path, raw=False):
    times, states = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            times.append(float(r["time_s"]))
            states.append(r["raw_state"] if raw else r["smoothed_state"])
    return times, states


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--video")
    ap.add_argument("--condition", default="", help="label only, e.g. day/dusk/night")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    gt = read_gt(args.gt, args.video)
    times, states = read_pred(args.pred, args.raw)
    m = signal_metrics(gt, times, states)
    print(f"frames={m['frames']} coverage={fmt(m['coverage'], True)} "
          f"accuracy_on_covered={fmt(m['accuracy_on_covered'], True)} "
          f"red->green/amber={m['red_read_as_green_or_amber']} non-red->red={m['non_red_read_as_red']}")
    for c, v in m["per_class"].items():
        print(f"  {c:6s} P={fmt(v['precision'])} R={fmt(v['recall'])} F1={fmt(v['f1'])} n={v['support']}")
    if args.out:
        write_json(args.out, {"task": "signal_state", "gt": args.gt, "pred": args.pred, "video": args.video,
                              "condition": args.condition, "smoothed": not args.raw, **m})
    return m


if __name__ == "__main__":
    main()
