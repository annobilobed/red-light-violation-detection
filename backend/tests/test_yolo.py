"""Smoke test of the real YOLO + ByteTrack back-end. Skipped unless
Ultralytics is installed and yolo11n.pt is available locally (set
RLV_YOLO_WEIGHTS or place the file in backend/)."""
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ultralytics")
WEIGHTS = Path(os.environ.get("RLV_YOLO_WEIGHTS", Path(__file__).resolve().parents[1] / "yolo11n.pt"))
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="yolo11n.pt not available")


def test_yolo_tracker_interface():
    from app.pipeline.config import DetectorConfig
    from app.pipeline.detectors import YoloTracker
    tr = YoloTracker(DetectorConfig(model=str(WEIGHTS)))
    assert sorted(tr.names[i] for i in tr.class_ids) == ["bus", "car", "motorcycle", "truck"]
    tr.reset()
    out = tr.track(np.zeros((360, 640, 3), np.uint8), 0)
    assert out == []


def test_unknown_classes_rejected():
    from app.pipeline.config import DetectorConfig
    from app.pipeline.detectors import YoloTracker
    with pytest.raises(ValueError):
        YoloTracker(DetectorConfig(model=str(WEIGHTS), vehicle_classes=["tuk-tuk"]))
