#!/usr/bin/env python3
"""Build a signal-classifier dataset (red / amber / green / none) from footage.

Per recording session, label the signal state over time in a CSV
(start_s,end_s,state with state red, amber, green or none; "none" = signal
off, hidden, or unreadable) and run, choosing the split PER SESSION:

  python tools/export_signal_crops.py --video s01.mp4 --config s01.json \
      --labels s01_signal.csv --out datasets/signal_crops --split train --step 0.5 --none-random 100
  python tools/export_signal_crops.py --video s07.mp4 ... --split test

The crop is the signal box drawn on the Configure page (config.signal).
--none-random adds that many random crops of the same size from elsewhere in
the frame as "none" examples (things that are not a lit signal).
--dark-copies adds dimmed copies of lamp crops as "none" (unlit heads).
--lamp-folder imports an existing folder of crops with red/ amber|yellow/
green/ subfolders (e.g. a public starter set) into --split.

Then: python evaluation/train_signal_classifier.py --data datasets/signal_crops
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.config import AnalysisConfig  # noqa: E402
from app.pipeline.evidence import probe_video  # noqa: E402
from app.pipeline.signal_state import lamp_rects  # noqa: E402

CLASSES = ("red", "amber", "green", "none")
ALIASES = {"yellow": "amber", "off": "none", "unknown": "none", "hidden": "none"}


def box_of(cfg) -> tuple[int, int, int, int]:
    rects = list(lamp_rects(cfg.signal).values())
    x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
    x1, y1 = max(r[0] + r[2] for r in rects), max(r[1] + r[3] for r in rects)
    return x0, y0, x1 - x0, y1 - y0


def save(out: Path, split: str, cls: str, name: str, img: np.ndarray) -> None:
    d = out / split / cls
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{name}.png"), img)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--video")
    ap.add_argument("--config")
    ap.add_argument("--labels")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--none-random", type=int, default=0)
    ap.add_argument("--dark-copies", type=float, default=0.0, help="fraction of lamp crops to add dimmed as none")
    ap.add_argument("--lamp-folder")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    rng = random.Random(args.seed)
    counts = {c: 0 for c in CLASSES}

    def add(cls, name, img):
        save(out, args.split, cls, name, img)
        counts[cls] += 1
        if cls != "none" and args.dark_copies and rng.random() < args.dark_copies:
            save(out, args.split, "none", f"dark_{name}", (img.astype(np.float32) * rng.uniform(0.15, 0.35)).astype(np.uint8))
            counts["none"] += 1

    if args.lamp_folder:
        for d in sorted(Path(args.lamp_folder).iterdir()):
            cls = ALIASES.get(d.name, d.name)
            if d.is_dir() and cls in CLASSES:
                for f in sorted(d.iterdir()):
                    img = cv2.imread(str(f))
                    if img is not None:
                        add(cls, f"{d.name}_{f.stem}", img)

    if args.video:
        if not (args.config and args.labels):
            ap.error("--video needs --config and --labels")
        cfg = AnalysisConfig.from_dict(json.loads(Path(args.config).read_text()))
        x, y, w, h = box_of(cfg)
        meta = probe_video(args.video)
        stem = Path(args.video).stem
        cap = cv2.VideoCapture(args.video)
        with open(args.labels) as fh:
            rows = list(csv.DictReader(fh))
        frames_for_none = []
        for r in rows:
            cls = ALIASES.get(r["state"].strip().lower(), r["state"].strip().lower())
            if cls not in CLASSES:
                continue
            t, end = float(r["start_s"]) + 0.3, float(r["end_s"]) - 0.3   # avoid transitions
            while t < end:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * meta["fps"])))
                ok, frame = cap.read()
                if ok:
                    add(cls, f"{stem}_{t:08.2f}", frame[y:y + h, x:x + w])
                    if len(frames_for_none) < 200:
                        frames_for_none.append(frame)
                t += args.step
        cap.release()
        H, W = meta["height"], meta["width"]
        for k in range(args.none_random if frames_for_none else 0):
            frame = rng.choice(frames_for_none)
            for _ in range(20):
                nx, ny = rng.randint(0, max(0, W - w - 1)), rng.randint(0, max(0, H - h - 1))
                if nx + w < x or nx > x + w or ny + h < y or ny > y + h:   # not overlapping the signal
                    save(out, args.split, "none", f"{stem}_rnd{k}", frame[ny:ny + h, nx:nx + w])
                    counts["none"] += 1
                    break
    print(f"added to {out / args.split}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
