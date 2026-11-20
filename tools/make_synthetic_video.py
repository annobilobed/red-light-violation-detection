#!/usr/bin/env python3
"""Generate a synthetic signalised-approach video with known ground truth.

The video is NOT real footage and must never be reported as a result. It
exists to test the decision rules end to end.

Outputs (next to the .mp4):
  <stem>.scenario.json   per-frame boxes with track IDs (for ScriptedDetector)
                         + vehicle list + expected outcomes
  <stem>.config.json     an AnalysisConfig for this scene (model "scripted")
  <stem>.signal_gt.csv   ground-truth signal intervals (start_s,end_s,state)
  <stem>.events_gt.csv   ground-truth crossings with expected outcome

Scene: vehicles drive up the frame (approach side is below the stop line,
y = 300, across the two approach lanes x = 480..660). Oncoming traffic uses
the left half. The signal head is at x = 700..740, y = 60..170 (red on top).

Signal cycle: green 0-12 s, amber 12-15, red 15-30, green 30-42, amber
42-45, red 45-60. The signal is hidden 50.5-52.5 s (occlusion), and single
dark frames imitate LED flicker.

Scenarios (each decision rule has one):
  A  car    crosses 6.0 s on green                        -> lawful
  B  car    crosses 13.5 s on amber                       -> lawful
  C  car    crosses 14.6 s on amber, still in junction at red -> lawful
  D  car    crosses 18.0 s, 3 s after red                 -> potential violation
  E  moto   crosses 24.0 s on red                         -> potential violation
  F  car    stops at the line during red, crosses on green -> lawful
  H  bus    oncoming lane, passes y=300 off the stop line -> no crossing
  J  car    crosses 33.0 s on green                       -> lawful
  K  truck  crosses 40.0 s on green                       -> lawful
  M  car    crosses 44.9 s, 0.1 s before red              -> lawful
  I  bus    crosses 46.0 s, 1 s after red                 -> potential violation
  G  car    crosses 51.5 s while signal is hidden         -> signal_uncertain
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
STOP_Y = 300
LANES_UP = {1: 525, 2: 615}
LANE_DOWN = 390
SIGNAL_ROI = [700, 60, 40, 110]
SIZES = {"car": (56, 96), "motorcycle": (24, 52), "bus": (78, 190), "truck": (78, 150)}
COLOURS = {"car": (180, 90, 30), "motorcycle": (30, 30, 30), "bus": (0, 200, 240), "truck": (90, 90, 150)}

SIGNAL_SCHEDULE = [(0, 12, "green"), (12, 15, "amber"), (15, 30, "red"),
                   (30, 42, "green"), (42, 45, "amber"), (45, 60, "red")]
OCCLUSION = (50.5, 52.5)
FLICKER_TIMES = [5.0, 20.0, 35.0, 48.0]

# id, label, class, lane, kind, params, expected outcome
VEHICLES = [
    (1, "A", "car", 1, "through", {"tc": 6.0, "v": 140}, "lawful"),
    (2, "B", "car", 2, "through", {"tc": 13.5, "v": 140}, "lawful"),
    (3, "C", "car", 1, "through", {"tc": 14.6, "v": 60}, "lawful"),
    (4, "D", "car", 2, "through", {"tc": 18.0, "v": 150}, "potential_violation"),
    (5, "E", "motorcycle", 1, "through", {"tc": 24.0, "v": 180}, "potential_violation"),
    (6, "F", "car", 2, "stop_go", {"t_enter": 18.0, "v": 150, "stop_y": 318, "t_go": 30.5, "v2": 60}, "lawful"),
    (7, "H", "bus", 0, "oncoming", {"tc": 27.0, "v": 160}, None),
    (8, "J", "car", 1, "through", {"tc": 33.0, "v": 140}, "lawful"),
    (9, "K", "truck", 2, "through", {"tc": 40.0, "v": 120}, "lawful"),
    (10, "M", "car", 2, "through", {"tc": 44.9, "v": 150}, "lawful"),
    (11, "I", "bus", 1, "through", {"tc": 46.0, "v": 120}, "potential_violation"),
    (12, "G", "car", 2, "through", {"tc": 51.5, "v": 140}, "signal_uncertain"),
]


def signal_at(t: float) -> str:
    for a, b, s in SIGNAL_SCHEDULE:
        if a <= t < b:
            return s
    return SIGNAL_SCHEDULE[-1][2]


def bottom_y(kind: str, p: dict, t: float) -> float:
    if kind == "through":
        return STOP_Y + p["v"] * (p["tc"] - t)
    if kind == "oncoming":
        return STOP_Y - p["v"] * (p["tc"] - t)
    if kind == "stop_go":
        y = H + 200 - p["v"] * (t - p["t_enter"])
        if t < p["t_go"]:
            return max(y, p["stop_y"])
        return p["stop_y"] - p["v2"] * (t - p["t_go"])
    raise ValueError(kind)


def crossing_time(kind: str, p: dict) -> float | None:
    if kind == "through":
        return p["tc"]
    if kind == "stop_go":
        return p["t_go"] + (p["stop_y"] - STOP_Y) / p["v2"]
    return None


def box_at(v, t):
    vid, _, cls, lane, kind, p, _ = v
    if kind == "stop_go" and t < p["t_enter"]:
        return None
    w, h = SIZES[cls]
    yb = bottom_y(kind, p, t)
    x = LANE_DOWN if kind == "oncoming" else LANES_UP[lane]
    x1, x2 = x - w / 2, x + w / 2
    if kind == "oncoming":
        # Heading down: the box's leading edge is its bottom; same box shape.
        pass
    y1, y2 = yb - h, yb
    cy1, cy2 = max(0.0, y1), min(float(H), y2)
    if cy2 - cy1 < 12:
        return None
    return (round(x1, 1), round(cy1, 1), round(x2, 1), round(cy2, 1)), (x1, y1, x2, y2)


def draw_background() -> np.ndarray:
    img = np.full((H, W, 3), (70, 120, 70), np.uint8)
    cv2.rectangle(img, (300, 0), (660, H), (90, 90, 90), -1)          # main road
    cv2.rectangle(img, (0, 150), (W, 290), (90, 90, 90), -1)          # cross road
    for y in range(310, H, 40):                                        # lane marks
        cv2.line(img, (570, y), (570, y + 20), (220, 220, 220), 2)
    cv2.line(img, (480, 300), (480, H), (0, 200, 230), 2)              # centre line
    cv2.line(img, (480, 0), (480, 150), (0, 200, 230), 2)
    cv2.line(img, (480, STOP_Y), (660, STOP_Y), (255, 255, 255), 5)    # stop line
    cv2.rectangle(img, (716, 170), (724, 240), (40, 40, 40), -1)       # signal pole
    return img


def draw_signal(img: np.ndarray, state: str, t: float, dark: bool) -> None:
    x, y, w, h = SIGNAL_ROI
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), -1)
    lit = {"red": (30, 30, 255), "amber": (0, 190, 255), "green": (80, 230, 60)}
    dim = {"red": (40, 40, 70), "amber": (30, 55, 70), "green": (40, 65, 40)}
    for i, c in enumerate(("red", "amber", "green")):
        centre = (x + w // 2, y + 18 + i * 37)
        on = (c == state) and not dark
        cv2.circle(img, centre, 14, lit[c] if on else dim[c], -1, cv2.LINE_AA)
    if OCCLUSION[0] <= t < OCCLUSION[1]:
        cv2.rectangle(img, (x - 15, y - 10), (x + w + 15, y + h + 10), (60, 100, 60), -1)  # foliage


def draw_vehicle(img, cls, full_box):
    x1, y1, x2, y2 = (int(round(v)) for v in full_box)
    cv2.rectangle(img, (x1, y1), (x2, y2), COLOURS[cls], -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 1)
    if cls != "motorcycle":
        cv2.rectangle(img, (x1 + 6, y1 + 10), (x2 - 6, y1 + 10 + (y2 - y1) // 5), (60, 40, 30), -1)


def generate(out_path: str | Path, fps: float = 15.0, duration: float = 60.0, seed: int = 0,
             noise: float = 3.0) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bg = draw_background()
    n_frames = int(round(duration * fps))
    flicker_frames = {int(round(t * fps)) for t in FLICKER_TIMES}
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("cannot open video writer")
    frames: dict[str, list] = {}
    for idx in range(n_frames):
        t = idx / fps
        img = bg.copy()
        draw_signal(img, signal_at(t), t, idx in flicker_frames)
        boxes = []
        for v in VEHICLES:
            b = box_at(v, t)
            if b is None:
                continue
            visible, full = b
            draw_vehicle(img, v[2], full)
            boxes.append([v[0], v[2], *visible])
        frames[str(idx)] = boxes
        if noise:
            img = np.clip(img.astype(np.int16) + rng.normal(0, noise, img.shape).astype(np.int16),
                          0, 255).astype(np.uint8)
        writer.write(img)
    writer.release()

    vehicles = []
    for vid, label, cls, lane, kind, p, expected in VEHICLES:
        vehicles.append({"track_id": vid, "label": label, "class": cls, "lane": lane, "kind": kind,
                         "params": p, "crossing_time_s": crossing_time(kind, p),
                         "expected": expected})
    scenario = {"fps": fps, "width": W, "height": H, "frames": frames, "vehicles": vehicles,
                "signal_schedule": SIGNAL_SCHEDULE, "occlusion": OCCLUSION,
                "note": "SYNTHETIC test video - not real footage."}
    stem = out_path.with_suffix("")
    Path(f"{stem}.scenario.json").write_text(json.dumps(scenario))
    config = {
        "signal": {"roi": SIGNAL_ROI, "layout": "vertical"},
        "crossing": {"stop_line": [[480, STOP_Y], [660, STOP_Y]], "approach_point": [570, 420],
                     "approach_zone": [[480, 302], [660, 302], [660, 540], [480, 540]]},
        "detector": {"model": "scripted"},
        "evidence": {"pre_s": 2.0, "post_s": 2.0},
        "frame_stride": 1,
    }
    Path(f"{stem}.config.json").write_text(json.dumps(config, indent=2))
    with open(f"{stem}.signal_gt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "start_s", "end_s", "state"])
        for a, b, s in SIGNAL_SCHEDULE:
            if a >= duration:
                break
            # The occluded interval is still red in truth; evaluators decide
            # how to score "uncertain" against it.
            w.writerow([out_path.name, a, min(b, duration), s])
    with open(f"{stem}.events_gt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "track_hint", "crossing_time_s", "label"])
        for v in vehicles:
            if v["expected"]:
                w.writerow([out_path.name, v["label"], v["crossing_time_s"], v["expected"]])
    return scenario


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="samples/synthetic_junction.mp4")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    generate(args.out, args.fps, args.duration, args.seed)
    print(f"wrote {args.out} (+ scenario, config and ground-truth files)")


if __name__ == "__main__":
    sys.exit(main())
