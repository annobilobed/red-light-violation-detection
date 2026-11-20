"""Vehicle detection + tracking back-ends.

* ``YoloTracker``: Ultralytics YOLO (e.g. yolo11n.pt / yolo11m.pt or a
  fine-tuned checkpoint) with ByteTrack via ``model.track(..., persist=True)``.
* ``ScriptedDetector``: replays ground-truth boxes from a scenario JSON written
  by ``tools/make_synthetic_video.py``. Used by tests and the synthetic demo.

Every tracker exposes ``track(frame_bgr, frame_idx) -> list[Detection]`` and
``reset()``.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .config import DetectorConfig
from .crossing import Detection

log = logging.getLogger(__name__)

_TRACKER_DIR = Path(__file__).parent / "trackers"
# id -> (Ultralytics tracker config, label)
TRACKERS: dict[str, tuple[str, str]] = {
    "bytetrack": ("bytetrack.yaml", "ByteTrack (default)"),
    "botsort_static": (str(_TRACKER_DIR / "botsort_static.yaml"), "BoT-SORT, fixed camera (no motion compensation)"),
    "botsort_reid": (str(_TRACKER_DIR / "botsort_reid.yaml"), "BoT-SORT, fixed camera + appearance re-ID"),
    "botsort": ("botsort.yaml", "BoT-SORT, stock (with camera motion compensation)"),
}


def resolve_tracker(name: str) -> str:
    """Tracker id ('bytetrack', 'botsort_static', ...) or a .yaml path/name."""
    key = name.removesuffix(".yaml") if name in ("bytetrack.yaml", "botsort.yaml") else name
    if key in TRACKERS:
        return TRACKERS[key][0]
    if name.endswith(".yaml"):
        return name
    raise ValueError(f"unknown tracker {name!r}; choose one of {sorted(TRACKERS)} or a .yaml file")


class Tracker(Protocol):
    name: str

    def track(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]: ...

    def reset(self) -> None: ...


class YoloTracker:
    """Ultralytics YOLO + ByteTrack. Requires ``pip install ultralytics``."""

    def __init__(self, cfg: DetectorConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("Ultralytics is not installed: pip install ultralytics") from exc
        self.cfg = cfg
        self.model = YOLO(cfg.model)
        names = self.model.names  # {id: name}
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.names: dict[int, str] = {int(k): str(v) for k, v in names.items()}
        wanted = set(cfg.vehicle_classes)
        self.class_ids = [i for i, n in self.names.items() if n in wanted]
        missing = wanted - {self.names[i] for i in self.class_ids}
        if missing:
            log.warning("Model %s has no classes named %s", cfg.model, sorted(missing))
        if not self.class_ids:
            raise ValueError(
                f"Model {cfg.model} has none of the vehicle classes {sorted(wanted)}; "
                f"its classes are {sorted(set(self.names.values()))}")
        self.tracker_cfg = resolve_tracker(cfg.tracker)
        tname = Path(cfg.tracker).stem if cfg.tracker.endswith(".yaml") else cfg.tracker
        self.name = f"yolo:{Path(cfg.model).name}+{tname}"

    def reset(self) -> None:
        # Clears ByteTrack state so IDs restart for a new video.
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for t in predictor.trackers:
                t.reset()

    def track(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]:
        results = self.model.track(
            frame_bgr, persist=True, tracker=self.tracker_cfg, conf=self.cfg.conf,
            iou=self.cfg.iou, imgsz=self.cfg.imgsz, classes=self.class_ids,
            device=self.cfg.device, verbose=False)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []  # nothing tracked in this frame (untracked boxes are ignored)
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        out = []
        for b, tid, c, p in zip(xyxy, ids, cls, conf):
            out.append(Detection(track_id=int(tid), cls=self.names.get(int(c), str(c)),
                                 conf=float(p), box=tuple(float(v) for v in b)))
        return out


class ScriptedDetector:
    """Replays boxes from a scenario JSON (synthetic ground truth).

    Optional noise: ``jitter_px`` adds uniform box jitter, ``drop_rate`` drops
    detections at random (to imitate missed detections).
    """

    def __init__(self, scenario_path: str | Path, jitter_px: float = 0.0,
                 drop_rate: float = 0.0, seed: int = 0):
        data = json.loads(Path(scenario_path).read_text())
        self.frames: dict[int, list] = {int(k): v for k, v in data["frames"].items()}
        self.jitter = jitter_px
        self.drop = drop_rate
        self.seed = seed
        self.rng = random.Random(seed)
        self.name = "scripted"

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def track(self, frame_bgr: Optional[np.ndarray], frame_idx: int) -> list[Detection]:
        out = []
        for tid, cls, x1, y1, x2, y2 in self.frames.get(frame_idx, []):
            if self.drop and self.rng.random() < self.drop:
                continue
            if self.jitter:
                j = lambda: self.rng.uniform(-self.jitter, self.jitter)  # noqa: E731
                x1, y1, x2, y2 = x1 + j(), y1 + j(), x2 + j(), y2 + j()
            out.append(Detection(track_id=int(tid), cls=cls, conf=1.0,
                                 box=(float(x1), float(y1), float(x2), float(y2))))
        return out


def scenario_path_for(video_path: str | Path) -> Path:
    """Where the synthetic generator writes the scenario for a video."""
    p = Path(video_path)
    return p.with_name(p.stem + ".scenario.json")


def make_tracker(cfg: DetectorConfig, video_path: str | Path | None = None) -> Tracker:
    """Build a tracker from config. ``model == "scripted"`` replays the
    synthetic scenario that sits next to the video."""
    if cfg.model == "scripted":
        if video_path is None:
            raise ValueError("scripted detector needs the video path")
        sp = scenario_path_for(video_path)
        if not sp.exists():
            raise FileNotFoundError(f"No scenario file {sp.name} next to the video; "
                                    "the scripted detector only works on synthetic videos.")
        return ScriptedDetector(sp)
    return YoloTracker(cfg)
