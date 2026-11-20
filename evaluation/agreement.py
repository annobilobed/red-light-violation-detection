#!/usr/bin/env python3
"""Inter-annotator agreement (Cohen's kappa) before using labels as ground truth.

Item labels (e.g. each crossing: potential_violation / lawful):
  python evaluation/agreement.py items annotator_a.csv annotator_b.csv
  (CSV columns: item,label)
Signal-state intervals, compared every --step seconds:
  python evaluation/agreement.py intervals a_signal.csv b_signal.csv --step 0.1
  (CSV columns: start_s,end_s,state; a "video" column is allowed)
"""
from __future__ import annotations

import argparse
import csv

import _common  # noqa: F401
from _common import write_json
from metrics import cohen_kappa, state_at


def read_items(path):
    with open(path) as fh:
        return {r["item"]: r["label"].strip() for r in csv.DictReader(fh)}


def read_intervals(path):
    with open(path) as fh:
        return [(float(r["start_s"]), float(r["end_s"]), r["state"].strip()) for r in csv.DictReader(fh)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["items", "intervals"])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.mode == "items":
        a, b = read_items(args.a), read_items(args.b)
        common = sorted(set(a) & set(b))
        res = cohen_kappa([a[i] for i in common], [b[i] for i in common])
        res["only_in_a"] = sorted(set(a) - set(b))
        res["only_in_b"] = sorted(set(b) - set(a))
    else:
        a, b = read_intervals(args.a), read_intervals(args.b)
        end = min(max(x[1] for x in a), max(x[1] for x in b))
        la, lb = [], []
        t = 0.0
        while t < end:
            sa, sb = state_at(a, t), state_at(b, t)
            if sa is not None and sb is not None:
                la.append(sa)
                lb.append(sb)
            t += args.step
        res = cohen_kappa(la, lb)
        res["step_s"] = args.step
    print(f"n={res['n']}  agreement={res['percent_agreement']}  kappa={res['kappa']}")
    if args.out:
        write_json(args.out, {"task": f"agreement_{args.mode}", "a": args.a, "b": args.b, **res})
    return res


if __name__ == "__main__":
    main()
