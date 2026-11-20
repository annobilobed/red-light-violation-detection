"""Anonymisation of evidence before it is shared or published.

This module only finds regions to BLUR. It never reads number plates or
identifies anyone.

Regions blurred in "standard" mode:
  * faces found by OpenCV's Haar face detectors (frontal + profile);
  * plate-like regions found by OpenCV's Haar plate detector;
  * for every tracked vehicle, a band at the bottom of the box where plates
    usually are, and for cars/buses/trucks a band around the windscreen line
    (driver); for motorcycles the top of the box (rider's head) as well.
"strong" mode pixelates every vehicle box completely.

Automatic blurring can miss things. A person must check anonymised material
before it is published.
"""
from __future__ import annotations

import functools
import logging
from typing import Iterable

import cv2
import numpy as np

from .crossing import Detection

MODES = ("standard", "strong")
log = logging.getLogger(__name__)


def haar_available() -> bool:
    """OpenCV 5 moved Haar cascades out of the main package; we then fall back
    to the box-based regions only."""
    return hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")


@functools.lru_cache(maxsize=None)
def _cascade(name: str):
    if not haar_available():
        log.warning("OpenCV Haar cascades unavailable; blurring vehicle-box regions only")
        return None
    path = cv2.data.haarcascades + name
    c = cv2.CascadeClassifier(path)
    return None if c.empty() else c


def _detect(cascade_name: str, gray: np.ndarray, min_size: int) -> list[tuple[int, int, int, int]]:
    c = _cascade(cascade_name)
    if c is None:
        return []
    found = c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_size, min_size))
    return [tuple(int(v) for v in r) for r in found] if len(found) else []


def _blur(img: np.ndarray, x: int, y: int, w: int, h: int, pixelate: bool = False) -> None:
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = img[y0:y1, x0:x1]
    if pixelate:
        small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)),
                           interpolation=cv2.INTER_LINEAR)
        img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    else:
        k = max(3, ((min(x1 - x0, y1 - y0) // 3) | 1))
        img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)


def regions_to_blur(frame: np.ndarray, detections: Iterable[Detection] = (),
                    mode: str = "standard") -> list[tuple[int, int, int, int, bool]]:
    """Return (x, y, w, h, pixelate) rectangles."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    regions: list[tuple[int, int, int, int, bool]] = []
    dets = list(detections)
    if mode == "strong":
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            regions.append((x1, y1, x2 - x1, y2 - y1, True))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for r in _detect("haarcascade_frontalface_default.xml", gray, 12):
        regions.append((*r, False))
    for r in _detect("haarcascade_profileface.xml", gray, 12):
        regions.append((*r, False))
    for r in _detect("haarcascade_russian_plate_number.xml", gray, 10):
        regions.append((*r, False))
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        # Plate band (front or rear plate sits low on the vehicle).
        regions.append((x1, y1 + int(0.65 * h), w, int(0.35 * h) + 1, False))
        if d.cls == "motorcycle":
            regions.append((x1, y1, w, int(0.4 * h) + 1, False))   # rider's head
        else:
            regions.append((x1, y1 + int(0.15 * h), w, int(0.3 * h) + 1, False))  # windscreen/driver
    return regions


def anonymise_frame(frame: np.ndarray, detections: Iterable[Detection] = (),
                    mode: str = "standard") -> np.ndarray:
    out = frame.copy()
    for x, y, w, h, pix in regions_to_blur(frame, detections, mode):
        _blur(out, x, y, w, h, pixelate=pix)
    return out
