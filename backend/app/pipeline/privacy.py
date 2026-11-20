"""Anonymisation of evidence before it is shared or published.

This module only finds regions to BLUR. It never reads number plates or
identifies anyone.

Layers, unioned (see Anonymiser): DNN face detector (OpenCV res10 SSD, from
tools/download_models.py; Haar cascades if missing), people from the COCO
YOLO model (head region + face search in each person box), vehicle-box bands
(plate band, windscreen/driver, motorcycle rider's head) and OpenCV's plate
cascade. "strong" mode pixelates every vehicle and person box completely.

Automatic blurring can miss things. A person must check anonymised material
before it is published.
"""
from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

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
        # Kernel from the LARGER side: thin bands (a plate strip across a car)
        # must still be blurred hard enough to make characters unreadable.
        k = max(3, ((max(x1 - x0, y1 - y0) // 4) | 1))
        k = min(k, 151)
        img[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.GaussianBlur(roi, (k, k), 0), (k, k), 0)


MODELS_DIR = Path(os.environ.get("RLV_MODELS_DIR", Path(__file__).resolve().parents[2] / "models"))
FACE_PROTO = "face_res10_deploy.prototxt"
FACE_MODEL = "face_res10_300x300_ssd_iter_140000_fp16.caffemodel"


class FaceDetector:
    """OpenCV res10 SSD face detector (tools/download_models.py), with the
    Haar cascades as fallback when the model files are missing."""

    def __init__(self, models_dir: Path = MODELS_DIR, conf: float = 0.35):
        self.conf = conf
        self.net = None
        proto, model = Path(models_dir) / FACE_PROTO, Path(models_dir) / FACE_MODEL
        if proto.exists() and model.exists():
            self.net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
        self.name = "res10-ssd" if self.net is not None else ("haar" if haar_available() else "none")

    def detect(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        if img.size == 0 or min(img.shape[:2]) < 8:
            return []
        if self.net is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return (_detect("haarcascade_frontalface_default.xml", gray, 12)
                    + _detect("haarcascade_profileface.xml", gray, 12))
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        out = self.net.forward()
        faces = []
        for i in range(out.shape[2]):
            if float(out[0, 0, i, 2]) < self.conf:
                continue
            x1, y1, x2, y2 = (out[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 > 2 and y2 - y1 > 2:
                # Pad 20%: the SSD box is tight around the face.
                px, py = int(0.2 * (x2 - x1)), int(0.2 * (y2 - y1))
                faces.append((x1 - px, y1 - py, x2 - x1 + 2 * px, y2 - y1 + 2 * py))
        return faces

    def detect_tiled(self, frame: np.ndarray, tile: int = 400) -> list[tuple[int, int, int, int]]:
        """Run on overlapping tiles so small faces in large frames are found."""
        h, w = frame.shape[:2]
        found = self.detect(frame)
        if max(h, w) <= tile * 1.5:
            return found
        step = tile // 2
        for y in range(0, max(1, h - step), step):
            for x in range(0, max(1, w - step), step):
                for fx, fy, fw, fh in self.detect(frame[y:y + tile, x:x + tile]):
                    found.append((fx + x, fy + y, fw, fh))
        return found


class PersonDetector:
    """People (pedestrians, riders, roadside workers) from a COCO YOLO model,
    so their heads can be blurred. Uses the same weights as the vehicle
    detector (COCO class 'person')."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 960):
        from .detectors import load_yolo
        self.model = load_yolo(weights)
        names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.ids = [i for i, n in names.items() if n == "person"]
        self.conf, self.imgsz = conf, imgsz
        self.name = f"yolo:{Path(weights).name}"

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if not self.ids:
            return []
        r = self.model.predict(frame, classes=self.ids, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [tuple(int(v) for v in b) for b in r.boxes.xyxy.tolist()]


class Anonymiser:
    """Finds regions to blur. Never reads plates or identifies anyone.

    Layers (all unioned, so one layer missing something is covered by others):
      1. faces from the res10 DNN face detector (whole frame, tiled);
      2. people from YOLO: the head region of every person box is blurred
         (and faces are searched inside an upscaled crop of the box);
      3. vehicle-box bands from the tracker's boxes: plate band, windscreen
         band, motorcycle rider's head;
      4. OpenCV's plate cascade (generic; weak on Ghanaian plates);
      "strong" mode pixelates every vehicle and person box completely.
    """

    def __init__(self, person_weights: Optional[str] = None, models_dir: Path = MODELS_DIR):
        self.faces = FaceDetector(models_dir)
        self.persons: Optional[PersonDetector] = None
        if person_weights:
            try:
                self.persons = PersonDetector(person_weights)
            except Exception as exc:  # noqa: BLE001 - optional layer
                log.warning("person detector unavailable (%s); continuing without it", exc)

    def describe(self) -> dict:
        return {"faces": self.faces.name, "persons": self.persons.name if self.persons else None,
                "vehicle_bands": True, "plate_cascade": haar_available()}

    def regions(self, frame: np.ndarray, detections: Iterable[Detection] = (),
                mode: str = "standard") -> list[tuple[int, int, int, int, bool]]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        dets = list(detections)
        regions: list[tuple[int, int, int, int, bool]] = []
        for r in self.faces.detect_tiled(frame):
            regions.append((*r, False))
        for x1, y1, x2, y2 in (self.persons.detect(frame) if self.persons else []):
            w, h = x2 - x1, y2 - y1
            if mode == "strong":
                regions.append((x1, y1, w, h, True))
            pad = int(0.15 * w)
            regions.append((x1 - pad, y1 - pad, w + 2 * pad, int(0.3 * h) + pad, False))  # head
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size and h < 300:
                scale = 300 / max(1, h)
                up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                for fx, fy, fw, fh in self.faces.detect(up):
                    regions.append((x1 + int(fx / scale), y1 + int(fy / scale), int(fw / scale) + 1,
                                    int(fh / scale) + 1, False))
        regions += _vehicle_regions(frame, dets, mode)
        return regions

    def apply(self, frame: np.ndarray, detections: Iterable[Detection] = (), mode: str = "standard") -> np.ndarray:
        out = frame.copy()
        for x, y, w, h, pix in self.regions(frame, detections, mode):
            _blur(out, x, y, w, h, pixelate=pix)
        return out


@functools.lru_cache(maxsize=4)
def get_anonymiser(person_weights: Optional[str] = None) -> Anonymiser:
    return Anonymiser(person_weights)


def _vehicle_regions(frame: np.ndarray, dets: list[Detection], mode: str) -> list[tuple[int, int, int, int, bool]]:
    regions: list[tuple[int, int, int, int, bool]] = []
    if mode == "strong":
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            regions.append((x1, y1, x2 - x1, y2 - y1, True))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for r in _detect("haarcascade_russian_plate_number.xml", gray, 10):
        regions.append((*r, False))
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        # Plate band: depending on the viewpoint a front/rear plate sits
        # anywhere from ~40% to the bottom of the box (measured on a real rear
        # view: 47%). Overlaps the windscreen band, so 10%..100% is covered.
        regions.append((x1, y1 + int(0.40 * h), w, h - int(0.40 * h) + 1, False))
        if d.cls == "motorcycle":
            regions.append((x1, y1, w, int(0.5 * h) + 1, False))   # rider's head and body
        else:
            regions.append((x1, y1 + int(0.10 * h), w, int(0.40 * h) + 1, False))  # windscreen/driver
    return regions


def regions_to_blur(frame: np.ndarray, detections: Iterable[Detection] = (),
                    mode: str = "standard", person_weights: Optional[str] = None) -> list[tuple[int, int, int, int, bool]]:
    """Return (x, y, w, h, pixelate) rectangles."""
    return get_anonymiser(person_weights).regions(frame, detections, mode)


def anonymise_frame(frame: np.ndarray, detections: Iterable[Detection] = (),
                    mode: str = "standard", person_weights: Optional[str] = None) -> np.ndarray:
    return get_anonymiser(person_weights).apply(frame, detections, mode)
