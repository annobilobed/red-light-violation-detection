#!/usr/bin/env python3
"""END-TO-END potential-violation detection metrics (separate task).

  python evaluation/evaluate_events.py --gt annot/events_gt.csv \
      --pred exports/analysis12_events.csv --tolerance 1.0

--gt: CSV video,crossing_time_s,label  (label: potential_violation for true
      red-light violations as judged by the annotators; other rows ignored)
--pred: the dashboard's events CSV export (columns video, crossing_time_s,
      outcome, review_status)
A prediction matches a ground-truth violation in the same video if their
crossing times differ by at most --tolerance seconds (one-to-one).

Reports, per video and pooled:
  * system: automated alerts (outcome = potential_violation) vs ground truth
  * after_review (if reviews exist): alerts confirmed by reviewers vs ground truth
Precision/recall come with 95% Wilson intervals; small samples are wide.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import _common  # noqa: F401
from _common import fmt, write_json
from metrics import match_events


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def evaluate(gt_rows, pred_rows, tolerance, pred_filter):
    gt_by = defaultdict(list)
    for r in gt_rows:
        if r["label"].strip() == "potential_violation":
            gt_by[r["video"]].append(float(r["crossing_time_s"]))
    pred_by = defaultdict(list)
    for r in pred_rows:
        if pred_filter(r):
            pred_by[r["video"]].append(float(r["crossing_time_s"]))
    videos = sorted(set(gt_by) | set(pred_by) | {r["video"] for r in gt_rows})
    per_video, tp, fp, fn = {}, 0, 0, 0
    for v in videos:
        m = match_events(gt_by.get(v, []), pred_by.get(v, []), tolerance)
        m.pop("matches")
        per_video[v] = m
        tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
    from metrics import wilson_interval
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if (prec == 0 or rec == 0) else None)
    return {"pooled": {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
                       "precision_ci95": wilson_interval(tp, tp + fp), "recall_ci95": wilson_interval(tp, tp + fn)},
            "per_video": per_video}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    gt, pred = load(args.gt), load(args.pred)
    res = {"system": evaluate(gt, pred, args.tolerance, lambda r: r["outcome"] == "potential_violation")}
    if any(r.get("review_status") in ("confirmed", "rejected") for r in pred):
        res["after_review"] = evaluate(gt, pred, args.tolerance, lambda r: r["outcome"] == "potential_violation"
                                       and r.get("review_status") == "confirmed")
    for k, v in res.items():
        p = v["pooled"]
        print(f"{k:12s} TP={p['tp']} FP={p['fp']} FN={p['fn']}  precision={fmt(p['precision'])} "
              f"recall={fmt(p['recall'])} F1={fmt(p['f1'])}")
    if args.out:
        write_json(args.out, {"task": "end_to_end_events", "gt": args.gt, "pred": args.pred,
                              "tolerance_s": args.tolerance, **res})
    return res


if __name__ == "__main__":
    main()
