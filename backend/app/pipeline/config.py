"""Analysis configuration (plain dataclasses, JSON round-trippable).

One configuration describes one camera view with one approach: a stop line,
the side vehicles approach from, an optional approach zone, the signal head
that applies to that approach, and the detector settings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

# COCO class names used by the stock YOLO models. Fine-tuned models must use
# the same names (or set `vehicle_classes` accordingly).
DEFAULT_VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]


@dataclass
class SignalConfig:
    # Signal-head rectangle in pixels: x, y, w, h.
    roi: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    # "vertical" (red on top), "horizontal" (red on the left) or "custom".
    layout: str = "vertical"
    # Only for layout == "custom": {"red": [x,y,w,h], "amber": [...], "green": [...]}
    lamps: Optional[dict[str, list[int]]] = None
    # HSV thresholds (OpenCV scale: H 0-180, S and V 0-255).
    sat_min: int = 90
    val_min: int = 120
    # Minimum fraction of lit, correctly coloured pixels in a lamp to call it on.
    min_score: float = 0.08
    # The winning lamp must beat the runner-up by this much, else "uncertain".
    margin: float = 0.04
    # Optional night fallback: if no lamp passes on colour, call the lamp that is
    # clearly brighter than the others (by position). Off by default.
    brightness_fallback: bool = False
    brightness_ratio: float = 1.6
    # Temporal smoothing.
    min_stable_frames: int = 3      # shorter colour runs become "uncertain"
    bridge_gap_s: float = 0.5       # uncertain gaps this short, with the SAME
                                    # colour on both sides, take that colour


@dataclass
class CrossingConfig:
    # Stop line, two points [[x1, y1], [x2, y2]].
    stop_line: list[list[float]] = field(default_factory=lambda: [[0, 0], [1, 0]])
    # Any point on the side vehicles approach FROM (before the line).
    approach_point: list[float] = field(default_factory=lambda: [0, 1])
    # Optional polygon: only vehicles whose anchor was inside it before
    # crossing count (e.g. the approach lanes).
    approach_zone: Optional[list[list[float]]] = None
    # Hysteresis band around the line, pixels.
    line_margin_px: float = 3.0
    # A track must be seen this many times on the approach side before a
    # crossing is accepted (guards against ID switches inside the junction).
    min_approach_frames: int = 3
    # Allowed overshoot beyond the ends of the stop-line segment (fraction of
    # its length) when deciding whether the crossing point lies on the line.
    segment_tolerance: float = 0.05
    # A crossing counts as a potential violation only if it happened more
    # than this many seconds after the red onset. 0 = strictly after.
    red_grace_s: float = 0.0


@dataclass
class DetectorConfig:
    model: str = "yolo11n.pt"
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    device: Optional[str] = None
    tracker: str = "bytetrack.yaml"
    vehicle_classes: list[str] = field(default_factory=lambda: list(DEFAULT_VEHICLE_CLASSES))


@dataclass
class EvidenceConfig:
    pre_s: float = 3.0
    post_s: float = 3.0
    # Also render evidence for crossings made while the signal was unreadable.
    include_signal_uncertain: bool = True
    h264: bool = True


@dataclass
class AnalysisConfig:
    signal: SignalConfig = field(default_factory=SignalConfig)
    crossing: CrossingConfig = field(default_factory=CrossingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    frame_stride: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AnalysisConfig":
        data = data or {}

        def build(klass, values):
            values = values or {}
            known = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in values.items() if k in known})

        return cls(
            signal=build(SignalConfig, data.get("signal")),
            crossing=build(CrossingConfig, data.get("crossing")),
            detector=build(DetectorConfig, data.get("detector")),
            evidence=build(EvidenceConfig, data.get("evidence")),
            frame_stride=max(1, int(data.get("frame_stride", 1))),
        )

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty if usable)."""
        problems = []
        (x1, y1), (x2, y2) = self.crossing.stop_line
        if (x1, y1) == (x2, y2):
            problems.append("Stop line is not set (its two points are identical).")
        r = self.signal.roi
        if self.signal.layout != "custom" and (len(r) != 4 or r[2] <= 2 or r[3] <= 2):
            problems.append("Signal region is not set.")
        if self.signal.layout == "custom":
            lamps = self.signal.lamps or {}
            for c in ("red", "amber", "green"):
                if c not in lamps:
                    problems.append(f"Custom layout needs a {c} lamp rectangle.")
        if not problems:
            from .geometry import signed_distance
            d = signed_distance(tuple(self.crossing.approach_point), (x1, y1), (x2, y2))
            if abs(d) < 1:
                problems.append("Approach point lies on the stop line; place it on the approach side.")
        return problems
