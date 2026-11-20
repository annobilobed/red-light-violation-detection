#!/usr/bin/env python3
"""Compare the nano and medium models across seeds: mean ± SD.

  python evaluation/compare_models.py --detector results/detector --benchmark results/benchmark.json \
      --out results/model_comparison.md

Detector JSONs come from detector_val.py (one per run, named <model>_seed<k>).
The benchmark JSON comes from benchmark.py.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401
from _common import write_json
from metrics import mean_sd

METRICS = ["map50", "map50_95", "precision", "recall"]


def group(results: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in results:
        by[re.sub(r"_seed\d+$", "", r["run"])].append(r)
    out = {}
    for model, rs in sorted(by.items()):
        out[model] = {"runs": [r["run"] for r in rs], **{m: mean_sd(r[m] for r in rs) for m in METRICS}}
    return out


def table(groups: dict, bench: dict | None) -> str:
    lines = ["| Model | Seeds | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Speed |",
             "|---|---|---|---|---|---|---|"]
    for model, g in groups.items():
        cells = [f"{g[m]['mean']:.3f} ± {g[m]['sd']:.3f}" if g[m]["mean"] is not None else "–" for m in METRICS]
        speed = "–"
        if bench:
            for b in bench.get("models", []):
                if Path(b["model"]).stem.startswith(model) or model.startswith(Path(b["model"]).stem):
                    speed = f"{b['track_fps']:.1f} FPS ({bench.get('device_label', '')})"
        lines.append(f"| {model} | {g['map50']['n']} | " + " | ".join(cells) + f" | {speed} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", required=True, help="folder of detector_val JSONs")
    ap.add_argument("--benchmark")
    ap.add_argument("--out", default="results/model_comparison.md")
    args = ap.parse_args(argv)
    results = [json.loads(p.read_text()) for p in sorted(Path(args.detector).glob("*.json"))]
    groups = group(results)
    bench = json.loads(Path(args.benchmark).read_text()) if args.benchmark else None
    md = table(groups, bench)
    print(md)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md + "\n")
    write_json(Path(args.out).with_suffix(".json"), {"task": "model_comparison", "groups": groups})


if __name__ == "__main__":
    main()
