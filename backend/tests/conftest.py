import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from tools.make_synthetic_video import generate  # noqa: E402


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """The synthetic junction video plus its scenario and config."""
    d = tmp_path_factory.mktemp("synthetic")
    video = d / "synthetic_junction.mp4"
    scenario = generate(video)
    config = json.loads((d / "synthetic_junction.config.json").read_text())
    return {"video": video, "scenario": scenario, "config": config, "dir": d}
