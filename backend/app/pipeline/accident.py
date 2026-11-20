"""Supplementary accident-footage demonstration (EXPLORATORY ONLY).

Given a video and an approximate collision time marked by a person, produce:
  * a contact sheet: frames before and after the marked time, each labelled
    with its offset (t-2.0 s, ..., t+2.0 s);
  * a short clip of the same window with an "exploratory" banner;
  * optionally, vehicle tracks from a tracker drawn on both.

It makes NO claims about crash cause, speed, fault or right of way, and it is
not part of the red-light violation evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detectors import Tracker
from .evidence import ClipWriter, probe_video
from .privacy import anonymise_frame

DISCLAIMER = "EXPLORATORY - approximate collision time marked by a person; no claim about cause, speed or right of way"


def _put(img, text, org, scale=0.5, colour=(255, 255, 255), bg=(0, 0, 0)):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - base // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _draw_tracks(img, dets):
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        _put(img, f"#{d.track_id} {d.cls}", (x1, max(14, y1 - 2)), 0.45, (255, 200, 0))


def make_sequence(video_path: str | Path, collision_t: float, out_dir: str | Path,
                  before_s: float = 3.0, after_s: float = 3.0, n_tiles: int = 9,
                  tracker: Optional[Tracker] = None, anonymise: Optional[str] = None,
                  person_weights: Optional[str] = None,
                  stem: str = "accident") -> dict:
    meta = probe_video(video_path)
    fps = meta["fps"]
    duration = meta["duration_s"] or (meta["frames"] / fps)
    if not (0 <= collision_t <= duration):
        raise ValueError(f"collision time {collision_t:.2f}s is outside the video (0-{duration:.2f}s)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = max(0, int((collision_t - before_s) * fps))
    end = min(max(0, meta["frames"] - 1), int((collision_t + after_s) * fps))
    tile_frames = set(np.linspace(start, end, n_tiles).round().astype(int).tolist())
    collision_frame = int(round(collision_t * fps))
    tile_frames.add(min(max(collision_frame, start), end))

    if tracker is not None:
        tracker.reset()
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    clip_path = out_dir / f"{stem}_clip.mp4"
    writer = None
    tiles: list[tuple[int, np.ndarray]] = []
    idx = start
    while idx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        dets = tracker.track(frame, idx) if tracker is not None else []
        if anonymise:
            frame = anonymise_frame(frame, dets, anonymise, person_weights)
        img = frame.copy()
        if dets:
            _draw_tracks(img, dets)
        tile = img.copy()  # tiles get their own (larger) offset label, no banner
        offset = idx / fps - collision_t
        _put(img, f"t{offset:+.2f} s relative to marked collision time", (8, img.shape[0] - 10))
        _put(img, DISCLAIMER, (8, 22), 0.45, bg=(90, 40, 0))
        if writer is None:
            writer = ClipWriter(clip_path, fps, (img.shape[1], img.shape[0]))
        writer.write(img)
        if idx in tile_frames:
            tiles.append((idx, tile))
        idx += 1
    cap.release()
    if writer is not None:
        writer.close()
    if not tiles:
        raise ValueError("No frames could be read in the requested window")

    # Contact sheet.
    tw = 320
    th = int(tiles[0][1].shape[0] * tw / tiles[0][1].shape[1])
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    header = 40
    sheet = np.full((header + rows * th, cols * tw, 3), 30, np.uint8)
    _put(sheet, "Exploratory sequence around a marked collision time (not evidence of cause)", (6, 26), 0.5)
    for k, (fidx, img) in enumerate(tiles):
        small = cv2.resize(img, (tw, th))
        offset = fidx / fps - collision_t
        is_mark = fidx == min(max(collision_frame, start), end)
        _put(small, f"t{offset:+.2f} s" + ("  (marked)" if is_mark else ""), (4, th - 6), 0.45,
             bg=(0, 0, 160) if is_mark else (0, 0, 0))
        cv2.rectangle(small, (0, 0), (tw - 1, th - 1), (30, 30, 30), 2)  # tile border
        r, c = divmod(k, cols)
        sheet[header + r * th: header + (r + 1) * th, c * tw:(c + 1) * tw] = small
    sheet_path = out_dir / f"{stem}_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {
        "exploratory": True,
        "disclaimer": DISCLAIMER,
        "collision_time_s": collision_t,
        "window_s": [start / fps, end / fps],
        "contact_sheet": str(sheet_path),
        "clip": str(clip_path) if writer else None,
        "tiles": [{"frame": f, "offset_s": round(f / fps - collision_t, 3)} for f, _ in tiles],
        "tracker": tracker.name if tracker else None,
    }
