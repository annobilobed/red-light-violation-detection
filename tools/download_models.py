#!/usr/bin/env python3
"""Download the optional models used for anonymisation into backend/models.

Only official OpenCV files are fetched, and each file is checked against a
pinned SHA-256 before use. The face detector is OpenCV's res10 SSD (Caffe
weights are a protobuf data file, not executable code).

  python tools/download_models.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "backend" / "models"
FILES = {
    "face_res10_deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
        "dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9"),
    "face_res10_300x300_ssd_iter_140000_fp16.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/"
        "res10_300x300_ssd_iter_140000_fp16.caffemodel",
        "510ffd2471bd81e3fcc88a5beb4eae4fb445ccf8333ebc54e7302b83f4158a76"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, (url, digest) in FILES.items():
        path = DEST / name
        if path.exists() and sha256(path) == digest:
            print(f"ok       {name}")
            continue
        print(f"download {name}")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256(tmp) != digest:
            tmp.unlink()
            print(f"CHECKSUM MISMATCH for {name}; file discarded", file=sys.stderr)
            ok = False
            continue
        tmp.replace(path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
