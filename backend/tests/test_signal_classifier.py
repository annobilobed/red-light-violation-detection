"""Combining the colour rules with the CNN classifier."""
import os
from pathlib import Path

import cv2
import pytest

from app.pipeline.config import AnalysisConfig, SignalConfig
from app.pipeline.signal_state import combine

C = 0.9


@pytest.mark.parametrize("hsv,label,conf,expected", [
    ("red", "red", 0.95, "red"),            # agree
    ("red", "green", 0.95, "uncertain"),    # conflict -> never guess
    ("uncertain", "amber", 0.95, "amber"),  # classifier reads an over-exposed lamp
    ("uncertain", "amber", 0.50, "uncertain"),
    ("green", "none", 0.95, "uncertain"),   # confident "not a lit signal" vetoes the rules
    ("green", "none", 0.40, "green"),       # unsure classifier: keep the rules' reading
    ("green", "red", 0.40, "green"),
    ("uncertain", "none", 0.99, "uncertain"),
])
def test_hybrid(hsv, label, conf, expected):
    assert combine(hsv, label, conf, "hybrid", C)[0] == expected


def test_agree_and_cnn_modes():
    assert combine("red", "red", 0.95, "agree", C)[0] == "red"
    assert combine("uncertain", "red", 0.99, "agree", C)[0] == "uncertain"
    assert combine("red", "red", 0.85, "agree", C)[0] == "uncertain"
    assert combine("uncertain", "green", 0.95, "cnn", C)[0] == "green"
    assert combine("red", "none", 0.99, "cnn", C)[0] == "uncertain"


def test_cnn_reader_requires_classifier_file():
    cfg = AnalysisConfig.from_dict({"signal": {"roi": [0, 0, 30, 90], "reader": "hybrid"},
                                    "crossing": {"stop_line": [[0, 0], [10, 0]], "approach_point": [5, 5]}})
    assert any("classifier" in p for p in cfg.validate())
    cfg.signal.reader = "magic"
    assert any("hsv, cnn, hybrid or agree" in p for p in cfg.validate())


WEIGHTS = os.environ.get("RLV_SIGNAL_CLASSIFIER")


@pytest.mark.skipif(not (WEIGHTS and Path(WEIGHTS).exists()), reason="set RLV_SIGNAL_CLASSIFIER to a trained classifier")
def test_trained_classifier_on_real_lamps():
    from app.pipeline.calibration import evaluate_settings
    fix = Path(__file__).parent / "fixtures" / "real_lamps"
    samples = [(cv2.resize(cv2.imread(str(f)), (32, 64), interpolation=cv2.INTER_AREA), c)
               for c in ("red", "amber", "green") for f in sorted((fix / c).iterdir())]
    m = evaluate_settings(samples, SignalConfig(roi=[0, 0, 32, 64], reader="hybrid", classifier=WEIGHTS))
    assert m["safety_errors"] == 0 and m["coverage"] >= 0.9
