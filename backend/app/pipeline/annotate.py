"""Drawing overlays on frames for snapshots and evidence clips."""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from .config import AnalysisConfig
from .crossing import Detection
from .signal_state import lamp_rects

STATE_BGR = {
    "red": (40, 40, 230),
    "amber": (0, 170, 255),
    "green": (60, 190, 60),
    "uncertain": (160, 160, 160),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, org, colour, scale=0.5, thickness=1, bg=(0, 0, 0)):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - base // 2), FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, cfg: AnalysisConfig, detections: Iterable[Detection] = (),
                 signal_state: Optional[str] = None, highlight_track: Optional[int] = None,
                 banner: Optional[str] = None, time_s: Optional[float] = None) -> np.ndarray:
    img = frame.copy()
    (x1, y1), (x2, y2) = cfg.crossing.stop_line
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2, cv2.LINE_AA)
    if cfg.crossing.approach_zone:
        pts = np.array(cfg.crossing.approach_zone, dtype=np.int32)
        cv2.polylines(img, [pts], True, (200, 200, 0), 1, cv2.LINE_AA)
    colour = STATE_BGR.get(signal_state or "uncertain", STATE_BGR["uncertain"])
    for c, (x, y, w, h) in lamp_rects(cfg.signal).items():
        cv2.rectangle(img, (x, y), (x + w, y + h), STATE_BGR[c], 1)
    if signal_state:
        rx, ry, _, _ = cfg.signal.roi
        _label(img, f"signal: {signal_state}", (rx, max(14, ry - 4)), colour)
    for det in detections:
        bx1, by1, bx2, by2 = (int(v) for v in det.box)
        hl = highlight_track is not None and det.track_id == highlight_track
        col = (0, 0, 255) if hl else (255, 200, 0)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), col, 3 if hl else 1)
        _label(img, f"#{det.track_id} {det.cls}", (bx1, max(14, by1 - 2)), col, 0.45)
    if time_s is not None:
        _label(img, f"t={time_s:7.2f}s", (8, img.shape[0] - 8), (255, 255, 255))
    if banner:
        _label(img, banner, (8, 22), (255, 255, 255), 0.6, 1, bg=(0, 0, 160))
    return img
