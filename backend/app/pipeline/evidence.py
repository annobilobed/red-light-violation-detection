"""Evidence: stored tracks, snapshots and short clips around each event.

Evidence is rendered in a second pass over the source video, after the signal
timeline and decisions are final. The same function renders anonymised
evidence for export (blurring happens before overlays are drawn).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .annotate import draw_overlay
from .config import AnalysisConfig
from .crossing import Crossing, Detection
from .privacy import anonymise_frame
from .signal_state import SignalTimeline

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Track storage
# --------------------------------------------------------------------------

class TrackStore:
    """All detections of one analysis, indexed by frame."""

    def __init__(self):
        self.by_frame: dict[int, list[Detection]] = defaultdict(list)

    def add(self, frame_idx: int, dets: list[Detection]) -> None:
        if dets:
            self.by_frame[frame_idx].extend(dets)

    def get(self, frame_idx: int) -> list[Detection]:
        return self.by_frame.get(frame_idx, [])

    def nearest(self, frame_idx: int, max_gap: int) -> list[Detection]:
        """Detections at the frame, or at the closest earlier processed frame
        (for clips rendered at full frame rate when analysis used a stride)."""
        for k in range(0, max_gap + 1):
            if (frame_idx - k) in self.by_frame:
                return self.by_frame[frame_idx - k]
        return []

    def save(self, path: str | Path) -> None:
        rows = [(f, d.track_id, d.cls, d.conf, *d.box)
                for f, dets in sorted(self.by_frame.items()) for d in dets]
        classes = sorted({r[2] for r in rows})
        cidx = {c: i for i, c in enumerate(classes)}
        arr = np.array([[r[0], r[1], cidx[r[2]], r[3], r[4], r[5], r[6], r[7]] for r in rows],
                       dtype=np.float64).reshape(-1, 8)
        np.savez_compressed(path, rows=arr, classes=np.array(classes, dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "TrackStore":
        data = np.load(path, allow_pickle=True)
        classes = list(data["classes"])
        store = cls()
        for r in data["rows"]:
            store.by_frame[int(r[0])].append(Detection(
                track_id=int(r[1]), cls=str(classes[int(r[2])]), conf=float(r[3]),
                box=(float(r[4]), float(r[5]), float(r[6]), float(r[7]))))
        return store

    def to_mot_rows(self) -> list[tuple]:
        """MOTChallenge rows: frame(1-based), id, x, y, w, h, conf, -1, -1, -1."""
        out = []
        for f, dets in sorted(self.by_frame.items()):
            for d in dets:
                x1, y1, x2, y2 = d.box
                out.append((f + 1, d.track_id, x1, y1, x2 - x1, y2 - y1, d.conf, -1, -1, -1))
        return out


# --------------------------------------------------------------------------
# Video helpers
# --------------------------------------------------------------------------

def probe_video(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or fps > 240:
        fps = 25.0  # some containers report nonsense; fall back to a common rate
    return {"fps": float(fps), "frames": frames, "width": width, "height": height,
            "duration_s": frames / fps if fps else 0.0}


def read_frame_at(path: str | Path, t: float) -> np.ndarray:
    meta = probe_video(path)
    idx = max(0, min(int(round(t * meta["fps"])), max(0, meta["frames"] - 1)))
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame at t={t:.2f}s")
    return frame


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def to_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264/yuv420p so browsers can play it. Returns success."""
    if not ffmpeg_available():
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-an", str(dst)]
    try:
        subprocess.run(cmd, check=True, timeout=600)
        return True
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("ffmpeg conversion failed: %s", exc)
        return False


class ClipWriter:
    """Writes frames to an .mp4; converts to H.264 on close when possible."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], h264: bool = True):
        self.path = Path(path)
        self.h264 = h264 and ffmpeg_available()
        self.tmp = self.path.with_suffix(".raw.mp4") if self.h264 else self.path
        self.writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {self.tmp}")
        self.encoded_h264 = False

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> Path:
        self.writer.release()
        if self.h264:
            self.encoded_h264 = to_h264(self.tmp, self.path)
            if self.encoded_h264:
                self.tmp.unlink(missing_ok=True)
            else:
                self.tmp.replace(self.path)
        return self.path


# --------------------------------------------------------------------------
# Rendering evidence for an event
# --------------------------------------------------------------------------

def banner_for(c: Crossing) -> str:
    if c.outcome == "potential_violation":
        return f"POTENTIAL red-light violation - for human review (#{c.track_id})"
    if c.outcome == "signal_uncertain":
        return f"Crossing while signal unreadable - for human review (#{c.track_id})"
    return f"Crossing (#{c.track_id})"


def render_event_evidence(video_path: str | Path, crossing: Crossing, tracks: TrackStore,
                          timeline: SignalTimeline, cfg: AnalysisConfig, out_dir: str | Path,
                          stem: str, anonymise: Optional[str] = None,
                          fps: Optional[float] = None) -> dict:
    """Write ``<stem>.jpg`` (crossing frame) and ``<stem>.mp4`` (clip).

    ``anonymise``: None, "standard" or "strong".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video_path)
    fps = fps or meta["fps"]
    start = max(0, int((crossing.crossing_time_s - cfg.evidence.pre_s) * fps))
    end = min(meta["frames"] - 1 if meta["frames"] else 10**9,
              int((crossing.crossing_time_s + cfg.evidence.post_s) * fps))
    banner = banner_for(crossing)
    stride = max(1, cfg.frame_stride)

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    snapshot_path = out_dir / f"{stem}.jpg"
    clip_path = out_dir / f"{stem}.mp4"
    writer: Optional[ClipWriter] = None
    wrote_snapshot = False
    idx = start
    while idx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        dets = tracks.nearest(idx, stride - 1)
        if anonymise:
            frame = anonymise_frame(frame, dets, anonymise)
        state = timeline.state_at_frame(idx)
        img = draw_overlay(frame, cfg, dets, state, crossing.track_id, banner, idx / fps)
        if writer is None:
            writer = ClipWriter(clip_path, fps, (img.shape[1], img.shape[0]), cfg.evidence.h264)
        writer.write(img)
        if idx == crossing.crossing_frame:
            cv2.imwrite(str(snapshot_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            wrote_snapshot = True
        idx += 1
    cap.release()
    h264 = False
    if writer is not None:
        writer.close()
        h264 = writer.encoded_h264
    if not wrote_snapshot:
        render_snapshot(video_path, crossing, tracks, timeline, cfg, snapshot_path, anonymise, fps)
    return {"snapshot": str(snapshot_path), "clip": str(clip_path) if writer else None,
            "clip_h264": h264, "clip_start_s": start / fps, "clip_end_s": idx / fps}


def render_snapshot(video_path: str | Path, crossing: Crossing, tracks: TrackStore,
                    timeline: SignalTimeline, cfg: AnalysisConfig, path: str | Path,
                    anonymise: Optional[str] = None, fps: Optional[float] = None) -> str:
    """Annotated still of the crossing frame (optionally anonymised)."""
    fps = fps or probe_video(video_path)["fps"]
    frame = read_frame_at(video_path, crossing.crossing_frame / fps)
    dets = tracks.get(crossing.crossing_frame)
    if anonymise:
        frame = anonymise_frame(frame, dets, anonymise)
    img = draw_overlay(frame, cfg, dets, timeline.state_at_frame(crossing.crossing_frame),
                       crossing.track_id, banner_for(crossing), crossing.crossing_frame / fps)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(path)
