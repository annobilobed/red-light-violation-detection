"""Traffic-signal state from a user-marked region, by HSV colour rules.

Each lamp (red, amber, green) has a known position inside the signal head.
For every lamp we measure the fraction of its pixels that are bright,
saturated and of that lamp's hue. The lamp with the clearly highest score is
taken as lit. If no lamp clearly wins, the state is "uncertain": the reader
never guesses a colour.

Temporal smoothing (``smooth_states``) is done offline over the whole video:
1. colour runs shorter than ``min_stable_frames`` become "uncertain";
2. "uncertain" gaps no longer than ``bridge_gap_s`` that have the SAME colour
   on both sides take that colour (for LED flicker). Gaps between different
   colours stay "uncertain".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import SignalConfig
from .geometry import clamp_rect

COLOURS = ("red", "amber", "green")
UNCERTAIN = "uncertain"
STATES = COLOURS + (UNCERTAIN,)

# OpenCV hue scale 0-180.
HUE_RANGES: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 10), (160, 180)],
    "amber": [(11, 34)],
    "green": [(35, 100)],
}


def lamp_rects(cfg: SignalConfig) -> dict[str, tuple[int, int, int, int]]:
    """Pixel rectangles (x, y, w, h) for each lamp."""
    if cfg.layout == "custom":
        lamps = cfg.lamps or {}
        return {c: tuple(int(v) for v in lamps[c]) for c in COLOURS if c in lamps}
    x, y, w, h = (int(v) for v in cfg.roi)
    rects = {}
    for i, c in enumerate(COLOURS):
        if cfg.layout == "horizontal":
            x0 = x + (w * i) // 3
            x1 = x + (w * (i + 1)) // 3
            rects[c] = (x0, y, x1 - x0, h)
        else:  # vertical, red on top
            y0 = y + (h * i) // 3
            y1 = y + (h * (i + 1)) // 3
            rects[c] = (x, y0, w, y1 - y0)
    return rects


def _hue_mask(hue: np.ndarray, colour: str) -> np.ndarray:
    mask = np.zeros(hue.shape, dtype=bool)
    for lo, hi in HUE_RANGES[colour]:
        mask |= (hue >= lo) & (hue <= hi)
    return mask


@dataclass
class SignalReading:
    state: str
    scores: dict[str, float]
    brightness: dict[str, float]
    method: str  # "colour", "brightness" or "none"

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "brightness": {k: round(v, 1) for k, v in self.brightness.items()},
            "method": self.method,
        }


def read_signal(frame_bgr: np.ndarray, cfg: SignalConfig) -> SignalReading:
    """Read the signal state from one frame."""
    height, width = frame_bgr.shape[:2]
    scores: dict[str, float] = {}
    brightness: dict[str, float] = {}
    for colour, (x, y, w, h) in lamp_rects(cfg).items():
        x, y, w, h = clamp_rect(x, y, w, h, width, height)
        if w == 0 or h == 0:
            scores[colour] = 0.0
            brightness[colour] = 0.0
            continue
        crop = frame_bgr[y:y + h, x:x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lit = (sat >= cfg.sat_min) & (val >= cfg.val_min) & _hue_mask(hue, colour)
        scores[colour] = float(lit.mean())
        brightness[colour] = float(val.mean())
    for c in COLOURS:
        scores.setdefault(c, 0.0)
        brightness.setdefault(c, 0.0)

    state = classify_scores(scores, cfg.min_score, cfg.margin)
    method = "colour" if state != UNCERTAIN else "none"
    if state == UNCERTAIN and cfg.brightness_fallback and max(scores.values()) < cfg.min_score:
        ranked = sorted(brightness.items(), key=lambda kv: kv[1], reverse=True)
        (best_c, best_v), (_, second_v) = ranked[0], ranked[1]
        if best_v >= cfg.val_min and best_v >= cfg.brightness_ratio * max(second_v, 1.0):
            state, method = best_c, "brightness"
    return SignalReading(state=state, scores=scores, brightness=brightness, method=method)


def classify_scores(scores: dict[str, float], min_score: float, margin: float) -> str:
    ranked = sorted(((scores.get(c, 0.0), c) for c in COLOURS), reverse=True)
    (best, best_c), (second, _) = ranked[0], ranked[1]
    if best >= min_score and best - second >= margin:
        return best_c
    return UNCERTAIN


# --------------------------------------------------------------------------
# Temporal smoothing and segments
# --------------------------------------------------------------------------

def _runs(states: Sequence[str]) -> list[list]:
    """[[state, start_index, end_index_inclusive], ...]"""
    runs: list[list] = []
    for i, s in enumerate(states):
        if runs and runs[-1][0] == s:
            runs[-1][2] = i
        else:
            runs.append([s, i, i])
    return runs


def smooth_states(states: Sequence[str], times: Sequence[float], cfg: SignalConfig) -> list[str]:
    out = list(states)
    if not out:
        return out
    # 1. Short colour runs -> uncertain.
    for s, a, b in _runs(out):
        if s != UNCERTAIN and (b - a + 1) < cfg.min_stable_frames:
            for i in range(a, b + 1):
                out[i] = UNCERTAIN
    # 2. Bridge short uncertain gaps between identical colours.
    if cfg.bridge_gap_s > 0:
        runs = _runs(out)
        for k in range(1, len(runs) - 1):
            s, a, b = runs[k]
            if s != UNCERTAIN:
                continue
            before, after = runs[k - 1][0], runs[k + 1][0]
            if before == after and before != UNCERTAIN:
                # Gap duration: from the last frame of the previous run to the
                # first frame of the next run.
                gap = times[b + 1] - times[a - 1]
                if gap <= cfg.bridge_gap_s:
                    for i in range(a, b + 1):
                        out[i] = before
    return out


@dataclass
class Segment:
    state: str
    start_s: float      # time of first frame in the segment
    end_s: float        # time of last frame in the segment
    start_frame: int
    end_frame: int
    # For colour segments: was the frame right before this segment a
    # different, readable colour? If it was "uncertain", the true onset may be
    # earlier than start_s (start_s is then an upper bound).
    onset_exact: bool = True

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "onset_exact": self.onset_exact,
        }


class SignalTimeline:
    """Smoothed signal states over the processed frames of one video."""

    def __init__(self, frames: Sequence[int], times: Sequence[float], raw: Sequence[str], cfg: SignalConfig):
        if not (len(frames) == len(times) == len(raw)):
            raise ValueError("frames, times and raw states must have equal length")
        self.frames = list(frames)
        self.times = list(times)
        self.raw = list(raw)
        self.smoothed = smooth_states(self.raw, self.times, cfg)
        self.segments: list[Segment] = []
        for s, a, b in _runs(self.smoothed):
            prev = self.segments[-1].state if self.segments else None
            self.segments.append(Segment(
                state=s,
                start_s=self.times[a], end_s=self.times[b],
                start_frame=self.frames[a], end_frame=self.frames[b],
                onset_exact=(prev is not None and prev != UNCERTAIN),
            ))
        self._index = {f: i for i, f in enumerate(self.frames)}

    def segment_at_frame(self, frame: int) -> Optional[Segment]:
        """Segment covering a processed frame (or the latest one before it)."""
        if not self.segments:
            return None
        lo, hi = 0, len(self.segments) - 1
        if frame < self.segments[0].start_frame:
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.segments[mid].start_frame <= frame:
                lo = mid
            else:
                hi = mid - 1
        return self.segments[lo]

    def state_at_frame(self, frame: int) -> str:
        seg = self.segment_at_frame(frame)
        return seg.state if seg else UNCERTAIN

    def summary(self) -> dict:
        total = len(self.smoothed) or 1
        counts = {s: self.smoothed.count(s) for s in STATES}
        raw_counts = {s: self.raw.count(s) for s in STATES}
        return {
            "frames": len(self.smoothed),
            "smoothed_fraction": {s: round(c / total, 4) for s, c in counts.items()},
            "raw_fraction": {s: round(c / total, 4) for s, c in raw_counts.items()},
        }
