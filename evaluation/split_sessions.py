#!/usr/bin/env python3
"""Split annotated recording SESSIONS (not frames) into train/val/test.

Frames from one session are highly correlated, so a frame-level random split
would leak test data into training. This script assigns whole sessions.

Input layout (e.g. CVAT export "YOLO 1.1", one folder per session):
  <src>/<session>/images/*.jpg
  <src>/<session>/labels/*.txt
Optional --meta CSV with columns: session,site,condition  (condition e.g.
day/dusk/night). Sessions are split per condition so each split covers all
conditions where possible. Use --test-sessions to fix the held-out sessions
explicitly (recommended once chosen: never re-split after seeing results).

Output: <out>/{train,val,test}/{images,labels} (symlinks), <out>/data.yaml,
<out>/split_manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401
from _common import write_json

CLASSES = ["car", "motorcycle", "bus", "truck"]


def assign_sessions(sessions: list[str], conditions: dict[str, str], ratios=(0.7, 0.15, 0.15),
                    seed: int = 0, fixed_test: list[str] | None = None) -> dict[str, str]:
    rng = random.Random(seed)
    out: dict[str, str] = {}
    fixed_test = set(fixed_test or [])
    for s in fixed_test:
        if s not in sessions:
            raise ValueError(f"fixed test session {s!r} not found")
        out[s] = "test"
    groups = defaultdict(list)
    for s in sorted(sessions):
        if s not in out:
            groups[conditions.get(s, "all")].append(s)
    for cond, items in sorted(groups.items()):
        rng.shuffle(items)
        n = len(items)
        n_test = 0 if fixed_test else max(1 if n >= 3 else 0, round(ratios[2] * n))
        n_val = max(1 if n - n_test >= 2 else 0, round(ratios[1] * n))
        for i, s in enumerate(items):
            out[s] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--meta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--test-sessions", nargs="*")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    args = ap.parse_args(argv)

    src, out = Path(args.src), Path(args.out)
    sessions = sorted(p.name for p in src.iterdir() if (p / "images").is_dir())
    if not sessions:
        raise SystemExit(f"No <session>/images folders in {src}")
    meta = {}
    if args.meta:
        with open(args.meta) as fh:
            meta = {r["session"]: r for r in csv.DictReader(fh)}
    conditions = {s: meta.get(s, {}).get("condition", "all") for s in sessions}
    split = assign_sessions(sessions, conditions, tuple(args.ratios), args.seed, args.test_sessions)

    counts = defaultdict(lambda: {"sessions": 0, "images": 0})
    for s, part in split.items():
        counts[part]["sessions"] += 1
        for img in sorted((src / s / "images").iterdir()):
            lab = src / s / "labels" / (img.stem + ".txt")
            for kind, f in (("images", img), ("labels", lab)):
                if not f.exists():
                    continue
                dst = out / part / kind / f"{s}__{f.name}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if args.copy:
                    shutil.copy2(f, dst)
                else:
                    os.symlink(f.resolve(), dst)
            counts[part]["images"] += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    write_json(out / "split_manifest.json", {"seed": args.seed, "split": split, "counts": counts,
                                             "conditions": conditions})
    for part in ("train", "val", "test"):
        print(f"{part:5s}: {counts[part]['sessions']} sessions, {counts[part]['images']} images")


if __name__ == "__main__":
    main()
