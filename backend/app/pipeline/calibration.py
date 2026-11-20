"""Automatic tuning of the HSV signal-reading thresholds from labelled frames.

The peak_min safety guard is kept as configured and is not searched:
calibration frames rarely include occlusions, so a search would remove it.

A person labels a handful of frames (or signal-head crops) as red, amber or
green. Every combination of thresholds in a grid is scored on those samples:

  * safety_errors  red read as amber/green, or amber/green read as red
  * wrong          any wrong colour (includes safety errors)
  * coverage       share of samples given a colour (not "uncertain")

The recommended setting is the one with the highest coverage among those with
no safety errors and a wrong-colour rate no higher than ``max_wrong_rate``
(default 0). Ties go to the stricter thresholds. If no setting qualifies, the
result says so and the current thresholds are kept. "uncertain" is never an
error: the reader may decline, it must not guess.

Calibration samples are not evaluation data. Measure accuracy afterwards on
other footage with evaluation/evaluate_signal.py.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, replace
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .config import SignalConfig
from .geometry import clamp_rect
from .signal_state import COLOURS, UNCERTAIN, _hue_mask, decide_arrays, lamp_rects

LABEL_ALIASES = {"yellow": "amber", "orange": "amber"}

DEFAULT_GRID = {
    "sat_min": [30, 50, 70, 90, 110, 130],
    "val_min": [80, 100, 120, 140, 160, 180, 200],
    "min_score": [0.02, 0.04, 0.06, 0.08, 0.12, 0.16],
    "margin": [0.01, 0.02, 0.04, 0.06, 0.1],
    # (core_agreement, core_val, core_min, core_margin, weak_min)
    "core": [(False, 200, 0.05, 0.03, 0.005)] + [
        (True, cv, cm, cg, wm) for cv in (200, 230) for cm in (0.05, 0.1) for cg in (0.03, 0.06)
        for wm in (0.005, 0.02)],
    # (brightness_fallback, brightness_ratio)
    "brightness_fallback": [(False, 1.6), (True, 1.6), (True, 2.0)],
}
SETTING_KEYS = ("sat_min", "val_min", "min_score", "margin", "core_agreement", "core_val", "core_min",
                "core_margin", "weak_min", "brightness_fallback", "brightness_ratio")


def normalise_label(label: str) -> str:
    label = label.strip().lower()
    return LABEL_ALIASES.get(label, label)


def _lamp_hsv(frame: np.ndarray, cfg: SignalConfig) -> dict[str, Optional[np.ndarray]]:
    h, w = frame.shape[:2]
    out = {}
    for c, (x, y, rw, rh) in lamp_rects(cfg).items():
        x, y, rw, rh = clamp_rect(x, y, rw, rh, w, h)
        out[c] = None if rw == 0 or rh == 0 else cv2.cvtColor(frame[y:y + rh, x:x + rw], cv2.COLOR_BGR2HSV)
    return out


def _metrics(truth: np.ndarray, state: np.ndarray) -> dict:
    n = len(truth)
    covered = state >= 0
    wrong = covered & (state != truth)
    red_t, red_p = truth == 0, state == 0
    safety = int(np.sum(covered & ((red_t & ~red_p) | (~red_t & red_p))))
    per_class = {}
    for i, c in enumerate(COLOURS):
        m = truth == i
        per_class[c] = {"n": int(m.sum()),
                        "read_correctly": int(np.sum(m & (state == i))),
                        "uncertain": int(np.sum(m & ~covered)),
                        "wrong": int(np.sum(m & wrong))}
    return {"samples": n, "coverage": float(covered.mean()) if n else 0.0,
            "wrong": int(wrong.sum()), "wrong_rate": float(wrong.sum() / n) if n else 0.0,
            "safety_errors": safety,
            "accuracy_on_covered": float(1 - wrong.sum() / covered.sum()) if covered.sum() else None,
            "per_class": per_class}


def evaluate_settings(samples: Sequence[tuple[np.ndarray, str]], cfg: SignalConfig) -> dict:
    """Metrics of one SignalConfig on labelled samples (uses read_signal)."""
    from .signal_state import read_signal
    truth, state = [], []
    for frame, label in samples:
        truth.append(COLOURS.index(normalise_label(label)))
        s = read_signal(frame, cfg).state
        state.append(-1 if s == UNCERTAIN else COLOURS.index(s))
    return _metrics(np.array(truth), np.array(state))


def calibrate(samples: Iterable[tuple[np.ndarray, str]], base: SignalConfig, grid: Optional[dict] = None,
              max_wrong_rate: float = 0.0, top: int = 5) -> dict:
    """Search thresholds on labelled samples.

    samples: (BGR frame or crop, label in red/amber/yellow/green). The lamp
    regions come from ``base`` (roi + layout), as on the Configure page.
    """
    grid = {**DEFAULT_GRID, **(grid or {})}
    labelled = []
    for frame, label in samples:
        label = normalise_label(label)
        if label not in COLOURS:
            raise ValueError(f"label must be red, amber or green, not {label!r}")
        labelled.append((frame, label))
    if not labelled:
        raise ValueError("no labelled samples")
    truth = np.array([COLOURS.index(l) for _, l in labelled])
    crops = [_lamp_hsv(f, base) for f, _ in labelled]
    n = len(crops)
    bright = np.zeros((n, 3))
    peak = np.zeros((n, 3))
    core_by_val = {cv: np.zeros((n, 3)) for cv in {c[1] for c in grid["core"]}}
    for i, c in enumerate(crops):
        for j, k in enumerate(COLOURS):
            if c.get(k) is not None:
                v = c[k][..., 2]
                bright[i, j] = float(v.mean())
                peak[i, j] = float(np.percentile(v, 95))
                for cv, arr in core_by_val.items():
                    arr[i, j] = float((v >= cv).mean())
    candidates = []
    for sat_min, val_min in itertools.product(grid["sat_min"], grid["val_min"]):
        scores = np.zeros((n, 3))
        for i, c in enumerate(crops):
            for j, k in enumerate(COLOURS):
                hsv = c.get(k)
                if hsv is not None:
                    lit = (hsv[..., 1] >= sat_min) & (hsv[..., 2] >= val_min) & _hue_mask(hsv[..., 0], k)
                    scores[i, j] = lit.mean()
        for min_score, margin, core, (fb, ratio) in itertools.product(
                grid["min_score"], grid["margin"], grid["core"], grid["brightness_fallback"]):
            ca, cv, cm, cg, wm = core
            settings = {"sat_min": sat_min, "val_min": val_min, "min_score": min_score, "margin": margin,
                        "core_agreement": ca, "core_val": cv, "core_min": cm, "core_margin": cg,
                        "weak_min": wm, "brightness_fallback": fb, "brightness_ratio": ratio}
            st, _ = decide_arrays(scores, core_by_val[cv], bright, peak, {**settings, "peak_min": base.peak_min})
            candidates.append((settings, _metrics(truth, st)))

    ok = [c for c in candidates if c[1]["safety_errors"] == 0 and c[1]["wrong_rate"] <= max_wrong_rate]
    # Highest coverage; then fewer wrong; then stricter (higher) thresholds,
    # no fallback preferred.
    # Prefer: higher coverage, fewer wrong, no position-only fallback, no core
    # rule, then stricter thresholds.
    key = lambda c: (-c[1]["coverage"], c[1]["wrong"], c[0]["brightness_fallback"],  # noqa: E731
                     c[0]["core_agreement"], -c[0]["min_score"], -c[0]["margin"], -c[0]["sat_min"],
                     -c[0]["val_min"], -c[0]["core_min"], -c[0]["core_margin"], -c[0]["weak_min"])
    ok.sort(key=key)
    current = evaluate_settings(labelled, base)
    counts = {c: int(np.sum(truth == i)) for i, c in enumerate(COLOURS)}
    warnings = []
    for c, n in counts.items():
        if n < 3:
            warnings.append(f"Only {n} {c} sample(s); label at least 3 (better 5+) frames of each colour.")
    result = {"samples": len(labelled), "label_counts": counts, "current": {
        "settings": {k: getattr(base, k) for k in SETTING_KEYS},
        "metrics": current}, "warnings": warnings, "searched": len(candidates)}
    if not ok:
        result["recommended"] = None
        result["message"] = ("No threshold setting reads these samples without confusing red with another "
                             "colour. Check the signal box and lamp layout, then label more frames.")
        return result
    best_settings, best_metrics = ok[0]
    if (current["safety_errors"] == 0 and current["wrong_rate"] <= max_wrong_rate
            and current["coverage"] >= best_metrics["coverage"]):
        result["recommended"] = {"settings": result["current"]["settings"], "metrics": current, "unchanged": True}
        result["alternatives"] = []
        result["message"] = (f"The current thresholds already read {100 * current['coverage']:.0f}% of the labelled "
                             "samples with no wrong colour; no setting does better. Keep them.")
        return result
    result["recommended"] = {"settings": best_settings, "metrics": best_metrics}
    result["alternatives"] = [{"settings": s, "metrics": m} for s, m in ok[1:top]]
    result["message"] = (f"Recommended thresholds read {100 * best_metrics['coverage']:.0f}% of the labelled "
                         f"samples with {best_metrics['wrong']} wrong colour(s) and no red/other confusion "
                         f"(current settings: {100 * current['coverage']:.0f}% read, {current['wrong']} wrong).")
    return result


def apply(base: SignalConfig, settings: dict) -> SignalConfig:
    return replace(base, **{k: settings[k] for k in SETTING_KEYS if k in settings})


def signal_config_dict(cfg: SignalConfig) -> dict:
    return asdict(cfg)
