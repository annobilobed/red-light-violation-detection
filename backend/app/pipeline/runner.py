"""End-to-end analysis of one recorded video. Returns plain data (no DB/web).

Pass 1 (every ``frame_stride``-th frame): read the signal, run the tracker,
update the crossing detector, store detections.
Then: smooth the signal timeline and apply the red-light rule to every
crossing.
Pass 2: render a snapshot and clip for every potential violation (and, if
configured, every crossing made while the signal was unreadable).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from .config import AnalysisConfig
from .crossing import (LAWFUL, POTENTIAL_VIOLATION, SIGNAL_UNCERTAIN, CrossingDetector, decide)
from .detectors import Tracker
from .evidence import TrackStore, probe_video, render_event_evidence
from .signal_state import SignalTimeline, read_signal

ProgressFn = Callable[[float, str], None]


class Cancelled(Exception):
    pass


def analyse_video(video_path: str | Path, cfg: AnalysisConfig, tracker: Tracker,
                  out_dir: Optional[str | Path] = None,
                  progress: Optional[ProgressFn] = None,
                  should_cancel: Optional[Callable[[], bool]] = None,
                  render_evidence: bool = True) -> dict:
    problems = cfg.validate()
    if problems:
        raise ValueError("Configuration incomplete: " + " ".join(problems))
    video_path = Path(video_path)
    meta = probe_video(video_path)
    fps = meta["fps"]
    stride = max(1, cfg.frame_stride)
    total = meta["frames"] or 1
    report = progress or (lambda f, m: None)

    tracker.reset()
    detector = CrossingDetector(cfg.crossing, max_track_age_frames=int(10 * fps))
    store = TrackStore()
    frames, times, raw, readings = [], [], [], []

    t_start = time.perf_counter()
    track_time = 0.0
    cap = cv2.VideoCapture(str(video_path))
    idx = -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = idx / fps
        reading = read_signal(frame, cfg.signal)
        t0 = time.perf_counter()
        dets = tracker.track(frame, idx)
        track_time += time.perf_counter() - t0
        store.add(idx, dets)
        detector.update(idx, t, dets)
        frames.append(idx)
        times.append(t)
        raw.append(reading.state)
        readings.append([idx, round(t, 3), reading.state] +
                        [round(reading.scores[c], 4) for c in ("red", "amber", "green")])
        if len(frames) % 25 == 0:
            report(min(0.9, 0.9 * idx / total), f"frame {idx}/{total}")
            if should_cancel and should_cancel():
                cap.release()
                raise Cancelled()
    cap.release()
    pass1_s = time.perf_counter() - t_start
    if not frames:
        raise ValueError("No frames could be read from the video.")

    timeline = SignalTimeline(frames, times, raw, cfg.signal)
    for c in detector.crossings:
        decide(c, timeline, cfg.crossing)

    crossings = sorted(detector.crossings, key=lambda c: c.crossing_time_s)
    result: dict = {
        "video": {**meta, "path": str(video_path), "name": video_path.name},
        "tracker": tracker.name,
        "config": cfg.to_dict(),
        "signal_segments": [s.to_dict() for s in timeline.segments],
        "signal_summary": timeline.summary(),
        "crossings": [],
        "counts": {
            POTENTIAL_VIOLATION: sum(c.outcome == POTENTIAL_VIOLATION for c in crossings),
            SIGNAL_UNCERTAIN: sum(c.outcome == SIGNAL_UNCERTAIN for c in crossings),
            LAWFUL: sum(c.outcome == LAWFUL for c in crossings),
        },
        "timing": {
            "frames_processed": len(frames),
            "pass1_s": round(pass1_s, 3),
            "pipeline_fps": round(len(frames) / pass1_s, 2) if pass1_s else None,
            "tracker_ms_per_frame": round(1000 * track_time / len(frames), 2),
        },
        "tracks_path": None,
        "signal_readings_path": None,
    }

    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        store.save(out / "tracks.npz")
        result["tracks_path"] = str(out / "tracks.npz")
        with open(out / "signal_readings.csv", "w") as fh:
            fh.write("frame,time_s,raw_state,score_red,score_amber,score_green,smoothed_state\n")
            for row, sm in zip(readings, timeline.smoothed):
                fh.write(",".join(str(v) for v in row) + f",{sm}\n")
        result["signal_readings_path"] = str(out / "signal_readings.csv")

    wanted = {POTENTIAL_VIOLATION}
    if cfg.evidence.include_signal_uncertain:
        wanted.add(SIGNAL_UNCERTAIN)
    to_render = [c for c in crossings if c.outcome in wanted]
    for n, c in enumerate(crossings):
        item = c.to_dict()
        item["evidence"] = None
        if out and render_evidence and c.outcome in wanted:
            k = to_render.index(c)
            report(0.9 + 0.1 * k / max(1, len(to_render)), f"evidence {k + 1}/{len(to_render)}")
            item["evidence"] = render_event_evidence(
                video_path, c, store, timeline, cfg, out / "evidence",
                stem=f"event_{n:04d}_track{c.track_id}", fps=fps)
        result["crossings"].append(item)

    result["timing"]["total_s"] = round(time.perf_counter() - t_start, 3)
    if out:
        (out / "result.json").write_text(json.dumps(result, indent=2))
    report(1.0, "done")
    return result


def load_timeline_from_result(result: dict, cfg: AnalysisConfig) -> SignalTimeline:
    """Rebuild a SignalTimeline from a stored signal_readings.csv."""
    path = result.get("signal_readings_path")
    frames, times, raw = [], [], []
    with open(path) as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split(",")
            frames.append(int(parts[0]))
            times.append(float(parts[1]))
            raw.append(parts[2])
    return SignalTimeline(frames, times, raw, cfg.signal)
