"""Metric functions shared by the evaluation scripts (pure Python + numpy).

Each task has its own metrics and they are never mixed:
  * vehicle detection   -> Ultralytics val (mAP, P, R)          detector_val.py
  * tracking            -> MOTA, IDF1, ID switches, fragments    tracking_check.py
  * signal-state reading-> accuracy, coverage, per-class P/R/F1  evaluate_signal.py
  * end-to-end events   -> precision, recall, F1 of alerts       evaluate_events.py
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def confusion(truth: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(truth, pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


def per_class_prf(truth: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict:
    out = {}
    for l in labels:
        tp = sum(t == l and p == l for t, p in zip(truth, pred))
        fp = sum(t != l and p == l for t, p in zip(truth, pred))
        fn = sum(t == l and p != l for t, p in zip(truth, pred))
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
        out[l] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> dict:
    if len(a) != len(b):
        raise ValueError("annotations must be the same length")
    n = len(a)
    if n == 0:
        return {"n": 0, "percent_agreement": None, "kappa": None}
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labels)
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return {"n": n, "percent_agreement": po, "kappa": kappa, "labels": labels}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mean_sd(values: Iterable[float]) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {"n": 0, "mean": None, "sd": None}
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return {"n": len(v), "mean": m, "sd": sd}


# --------------------------------------------------------------------------
# Signal intervals
# --------------------------------------------------------------------------

def state_at(intervals: Sequence[tuple[float, float, str]], t: float) -> str | None:
    for a, b, s in intervals:
        if a <= t < b:
            return s
    return None


def signal_metrics(gt: Sequence[tuple[float, float, str]], pred_times: Sequence[float],
                   pred_states: Sequence[str]) -> dict:
    """Compare per-frame predictions against ground-truth intervals.

    Frames without ground truth are skipped. "uncertain" predictions count
    against coverage, not accuracy: accuracy_on_covered is computed only on
    frames where the system committed to a colour.
    """
    colours = ["red", "amber", "green"]
    truth, pred = [], []
    for t, p in zip(pred_times, pred_states):
        g = state_at(gt, t)
        if g in colours:
            truth.append(g)
            pred.append(p)
    n = len(truth)
    covered = [(t, p) for t, p in zip(truth, pred) if p != "uncertain"]
    correct = sum(t == p for t, p in covered)
    red_as_other = sum(t == "red" and p in ("green", "amber") for t, p in covered)
    other_as_red = sum(t != "red" and p == "red" for t, p in covered)
    return {
        "frames": n,
        "coverage": len(covered) / n if n else None,
        "accuracy_on_covered": correct / len(covered) if covered else None,
        "accuracy_counting_uncertain_as_wrong": correct / n if n else None,
        "per_class": per_class_prf(truth, pred, colours),
        "confusion_labels": colours + ["uncertain"],
        "confusion": confusion(truth, pred, colours + ["uncertain"]),
        # Safety-relevant errors, counted separately.
        "red_read_as_green_or_amber": red_as_other,
        "non_red_read_as_red": other_as_red,
    }


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

def match_events(gt_times: Sequence[float], pred_times: Sequence[float], tolerance: float) -> dict:
    """Greedy one-to-one matching by time, closest pairs first."""
    pairs = sorted(((abs(g - p), i, j) for i, g in enumerate(gt_times) for j, p in enumerate(pred_times)
                    if abs(g - p) <= tolerance))
    used_g, used_p, matches = set(), set(), []
    for d, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matches.append((i, j, d))
    tp = len(matches)
    fp = len(pred_times) - tp
    fn = len(gt_times) - tp
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if (prec == 0 or rec == 0) else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "precision_ci95": wilson_interval(tp, tp + fp), "recall_ci95": wilson_interval(tp, tp + fn),
            "matches": matches,
            "unmatched_gt": [i for i in range(len(gt_times)) if i not in used_g],
            "unmatched_pred": [j for j in range(len(pred_times)) if j not in used_p]}


# --------------------------------------------------------------------------
# Tracking (MOTChallenge-style)
# --------------------------------------------------------------------------

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,4) b: (m,4) as x,y,w,h."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / union, 0)


def _assign(cost: np.ndarray):
    from scipy.optimize import linear_sum_assignment
    return linear_sum_assignment(cost)


def tracking_metrics(gt: dict[int, list[tuple]], pred: dict[int, list[tuple]], iou_thr: float = 0.5) -> dict:
    """gt/pred: {frame: [(id, x, y, w, h), ...]}. Returns MOTA, IDF1, IDs, frags."""
    frames = sorted(set(gt) | set(pred))
    n_gt = sum(len(gt.get(f, [])) for f in frames)
    fp = fn = idsw = 0
    last_match: dict[int, int] = {}           # gt id -> last pred id
    tracked_prev: dict[int, bool] = {}        # gt id -> matched in its previous appearance
    frags = 0
    seen_matched: set[int] = set()
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    gt_counts: dict[int, int] = defaultdict(int)
    pred_counts: dict[int, int] = defaultdict(int)
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        for r in g:
            gt_counts[r[0]] += 1
        for r in p:
            pred_counts[r[0]] += 1
        ga = np.array([r[1:5] for r in g], dtype=float).reshape(-1, 4)
        pa = np.array([r[1:5] for r in p], dtype=float).reshape(-1, 4)
        ious = _iou_matrix(ga, pa)
        matched_g, matched_p = set(), set()
        if ious.size:
            rows, cols = _assign(1 - ious)
            for i, j in zip(rows, cols):
                if ious[i, j] >= iou_thr:
                    matched_g.add(i)
                    matched_p.add(j)
                    gid, pid = g[i][0], p[j][0]
                    pair_counts[(gid, pid)] += 1
                    if gid in last_match and last_match[gid] != pid:
                        idsw += 1
                    if gid in seen_matched and not tracked_prev.get(gid, True):
                        frags += 1
                    last_match[gid] = pid
                    seen_matched.add(gid)
        for i, r in enumerate(g):
            tracked_prev[r[0]] = i in matched_g
        fn += len(g) - len(matched_g)
        fp += len(p) - len(matched_p)
    mota = 1 - (fn + fp + idsw) / n_gt if n_gt else None
    # IDF1: global one-to-one assignment of gt ids to pred ids maximising IDTP.
    gids, pids = sorted(gt_counts), sorted(pred_counts)
    idtp = 0
    if gids and pids:
        m = np.zeros((len(gids), len(pids)))
        gi = {g: i for i, g in enumerate(gids)}
        pj = {p: j for j, p in enumerate(pids)}
        for (g, p), c in pair_counts.items():
            m[gi[g], pj[p]] = c
        rows, cols = _assign(-m)
        idtp = int(m[rows, cols].sum())
    total_gt, total_pred = sum(gt_counts.values()), sum(pred_counts.values())
    idf1 = 2 * idtp / (total_gt + total_pred) if total_gt + total_pred else None
    return {"frames": len(frames), "gt_boxes": n_gt, "fp": fp, "fn": fn, "id_switches": idsw,
            "fragmentations": frags, "mota": mota, "idf1": idf1,
            "gt_tracks": len(gids), "pred_tracks": len(pids)}


def read_mot(path) -> dict[int, list[tuple]]:
    out: dict[int, list[tuple]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            f, i, x, y, w, h = int(float(parts[0])), int(float(parts[1])), *map(float, parts[2:6])
            if len(parts) > 6 and float(parts[6]) == 0 and len(parts) >= 9:
                continue  # MOT gt "ignore" rows
            out[f].append((i, x, y, w, h))
    return dict(out)
