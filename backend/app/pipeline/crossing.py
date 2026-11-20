"""Stop-line crossing detection and the red-light decision rule.

Crossing detection (online, per tracked vehicle):
  * The anchor is the bottom-centre of the vehicle box.
  * Its signed distance to the stop line is positive on the approach side.
  * A track is "armed" once it has been seen at least ``min_approach_frames``
    times more than ``line_margin_px`` on the approach side (and, if set,
    inside the approach zone).
  * An armed track crosses when its anchor is more than ``line_margin_px`` past
    the line. The crossing instant is interpolated between the last sample
    before the line and the first sample at or past it, and the crossing point
    must lie on the stop-line segment (within ``segment_tolerance``).
  * Each track crosses at most once. Movement away from the junction (from the
    far side back to the approach side) is never a crossing.

Decision rule (offline, once the signal timeline is known):
  * potential_violation: the signal at the first frame at/after the line is
    red AND the crossing instant is later than the red onset (+ red_grace_s).
  * signal_uncertain: the signal at that frame could not be read.
  * lawful: otherwise. A vehicle that crossed on green or amber and is still
    in the junction when red comes on is lawful, because only the instant of
    crossing the stop line is judged.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .config import CrossingConfig
from .geometry import (bottom_centre, lerp, point_in_polygon, projection_param,
                       signed_distance, zero_crossing_fraction, as_points)
from .signal_state import UNCERTAIN, SignalTimeline

POTENTIAL_VIOLATION = "potential_violation"
SIGNAL_UNCERTAIN = "signal_uncertain"
LAWFUL = "lawful"


@dataclass
class Detection:
    track_id: int
    cls: str
    conf: float
    box: tuple[float, float, float, float]


@dataclass
class Crossing:
    track_id: int
    vehicle_class: str
    crossing_time_s: float
    crossing_frame: int          # first processed frame at/after the line
    prev_frame: int              # last processed frame before the line
    anchor: tuple[float, float]  # interpolated crossing point
    det_conf: float
    # Filled in by decide():
    outcome: Optional[str] = None
    signal_state: Optional[str] = None
    red_onset_s: Optional[float] = None
    red_onset_exact: Optional[bool] = None
    time_since_red_s: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "vehicle_class": self.vehicle_class,
            "crossing_time_s": round(self.crossing_time_s, 3),
            "crossing_frame": self.crossing_frame,
            "prev_frame": self.prev_frame,
            "anchor": [round(self.anchor[0], 1), round(self.anchor[1], 1)],
            "det_conf": round(self.det_conf, 3),
            "outcome": self.outcome,
            "signal_state": self.signal_state,
            "red_onset_s": None if self.red_onset_s is None else round(self.red_onset_s, 3),
            "red_onset_exact": self.red_onset_exact,
            "time_since_red_s": None if self.time_since_red_s is None else round(self.time_since_red_s, 3),
        }


@dataclass
class _TrackState:
    samples: deque = field(default_factory=lambda: deque(maxlen=64))  # (frame, t, d, anchor)
    approach_count: int = 0
    classes: Counter = field(default_factory=Counter)
    confs: list = field(default_factory=list)
    crossed: bool = False
    last_frame: int = -1


class CrossingDetector:
    def __init__(self, cfg: CrossingConfig, max_track_age_frames: int = 300):
        self.cfg = cfg
        self.a = tuple(cfg.stop_line[0])
        self.b = tuple(cfg.stop_line[1])
        sign = signed_distance(tuple(cfg.approach_point), self.a, self.b)
        if sign == 0:
            raise ValueError("approach_point lies on the stop line")
        self.sign = 1.0 if sign > 0 else -1.0
        self.zone = as_points(cfg.approach_zone) if cfg.approach_zone else None
        self.tracks: dict[int, _TrackState] = {}
        self.crossings: list[Crossing] = []
        self.max_age = max_track_age_frames

    def distance(self, p) -> float:
        """Signed distance, positive on the approach side."""
        return self.sign * signed_distance(p, self.a, self.b)

    def update(self, frame_idx: int, t: float, detections: Iterable[Detection]) -> list[Crossing]:
        new: list[Crossing] = []
        m = self.cfg.line_margin_px
        for det in detections:
            st = self.tracks.setdefault(det.track_id, _TrackState())
            st.last_frame = frame_idx
            st.classes[det.cls] += 1
            st.confs.append(det.conf)
            if st.crossed:
                continue
            p = bottom_centre(det.box)
            d = self.distance(p)
            st.samples.append((frame_idx, t, d, p))
            if d > m:
                if self.zone is None or point_in_polygon(p, self.zone):
                    st.approach_count += 1
                continue
            if d < -m and st.approach_count >= self.cfg.min_approach_frames:
                c = self._make_crossing(det.track_id, st)
                if c is not None:
                    st.crossed = True
                    self.crossings.append(c)
                    new.append(c)
                else:
                    # Passed the line beside the segment: never count this track.
                    st.crossed = True
        # Forget stale tracks.
        stale = [tid for tid, st in self.tracks.items() if frame_idx - st.last_frame > self.max_age]
        for tid in stale:
            del self.tracks[tid]
        return new

    def _make_crossing(self, track_id: int, st: _TrackState) -> Optional[Crossing]:
        samples = list(st.samples)
        # Last sample on the approach side (d > 0), and the first sample after it.
        k = None
        for i in range(len(samples) - 1, -1, -1):
            if samples[i][2] > 0:
                k = i
                break
        if k is None or k == len(samples) - 1:
            return None
        f0, t0, d0, p0 = samples[k]
        f1, t1, d1, p1 = samples[k + 1]
        frac = zero_crossing_fraction(d0, d1)
        t_cross = lerp(t0, t1, frac)
        anchor = (lerp(p0[0], p1[0], frac), lerp(p0[1], p1[1], frac))
        u = projection_param(anchor, self.a, self.b)
        tol = self.cfg.segment_tolerance
        if u < -tol or u > 1 + tol:
            return None
        cls = st.classes.most_common(1)[0][0]
        conf = sum(st.confs) / len(st.confs)
        return Crossing(track_id=track_id, vehicle_class=cls, crossing_time_s=t_cross,
                        crossing_frame=f1, prev_frame=f0, anchor=anchor, det_conf=conf)


def decide(crossing: Crossing, timeline: SignalTimeline, cfg: CrossingConfig) -> Crossing:
    """Apply the red-light rule to one crossing (mutates and returns it)."""
    seg = timeline.segment_at_frame(crossing.crossing_frame)
    state = seg.state if seg else UNCERTAIN
    crossing.signal_state = state
    if state == UNCERTAIN or seg is None:
        crossing.outcome = SIGNAL_UNCERTAIN
        return crossing
    if state == "red":
        crossing.red_onset_s = seg.start_s
        crossing.red_onset_exact = seg.onset_exact
        since = crossing.crossing_time_s - seg.start_s
        crossing.time_since_red_s = since
        crossing.outcome = POTENTIAL_VIOLATION if since > cfg.red_grace_s else LAWFUL
        return crossing
    crossing.outcome = LAWFUL
    return crossing
