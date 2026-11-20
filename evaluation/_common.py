"""Small helpers shared by the evaluation scripts."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "backend", ROOT, ROOT / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), **data}
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {path}")


def fmt(x, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"
