"""Signal reading on REAL lamp crops, the foliage guard, and calibration."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.pipeline.calibration import apply, calibrate, evaluate_settings
from app.pipeline.config import SignalConfig
from app.pipeline.signal_state import read_signal

FIX = Path(__file__).parent / "fixtures" / "real_lamps"
SIZE = (32, 64)


def real_samples():
    out = []
    for c in ("red", "amber", "green"):
        for f in sorted((FIX / c).iterdir()):
            out.append((cv2.resize(cv2.imread(str(f)), SIZE, interpolation=cv2.INTER_AREA), c))
    return out


@pytest.fixture(scope="module")
def real():
    s = real_samples()
    assert len(s) == 297
    return s


def test_defaults_on_real_lamps_never_confuse_colours(real):
    m = evaluate_settings(real, SignalConfig(roi=[0, 0, *SIZE]))
    assert m["wrong"] == 0 and m["safety_errors"] == 0
    assert m["coverage"] >= 0.8          # 85.5% when written
    assert m["per_class"]["red"]["read_correctly"] >= 150


def test_core_rule_reads_overexposed_lamps(real):
    plain = evaluate_settings(real, SignalConfig(roi=[0, 0, *SIZE], core_agreement=False))
    core = evaluate_settings(real, SignalConfig(roi=[0, 0, *SIZE]))
    assert core["coverage"] > plain["coverage"] and core["wrong"] == 0


def test_calibration_generalises_across_halves(real):
    a, b = real[0::2], real[1::2]
    res = calibrate(a, SignalConfig(roi=[0, 0, *SIZE]))
    assert res["recommended"] and res["recommended"]["metrics"]["safety_errors"] == 0
    held_out = evaluate_settings(b, apply(SignalConfig(roi=[0, 0, *SIZE]), res["recommended"]["settings"]))
    assert held_out["safety_errors"] == 0
    assert held_out["wrong"] <= 1
    assert held_out["coverage"] >= 0.8


def test_dull_green_occluder_is_uncertain():
    """Foliage over the signal must not be read as a green lamp."""
    img = np.zeros((90, 30, 3), np.uint8)
    img[:] = (60, 100, 60)  # BGR dull green over the whole head
    assert read_signal(img, SignalConfig(roi=[0, 0, 30, 90])).state == "uncertain"
    img[:] = (90, 200, 70)  # bright, saturated green (sunlit leaves)
    r = read_signal(img, SignalConfig(roi=[0, 0, 30, 90]))
    assert r.peak["green"] >= 130  # documented limitation: sunlit green objects can pass the guard


def test_calibration_rejects_bad_labels_and_warns():
    img = np.zeros((90, 30, 3), np.uint8)
    with pytest.raises(ValueError):
        calibrate([(img, "blue")], SignalConfig(roi=[0, 0, 30, 90]))
    with pytest.raises(ValueError):
        calibrate([], SignalConfig(roi=[0, 0, 30, 90]))
    res = calibrate([(img, "red")], SignalConfig(roi=[0, 0, 30, 90]))
    assert any("red" in w for w in res["warnings"])


def test_mislabelled_frames_get_no_recommendation():
    # Two frames labelled "red" actually show green. Every setting that reads
    # the clean lamps also reads these as green (a red/other confusion), so
    # calibration must refuse instead of recommending something.
    def head(lit):
        im = np.zeros((90, 30, 3), np.uint8)
        for i, (c, col) in enumerate((("red", (30, 30, 255)), ("amber", (0, 190, 255)), ("green", (80, 230, 60)))):
            cv2.circle(im, (15, 15 + 30 * i), 11, col if c == lit else (40, 40, 50), -1)
        return im
    samples = [(head("green"), "red")] * 2 + [(head("red"), "red")] * 3 + [(head("green"), "green")] * 3
    res = calibrate(samples, SignalConfig(roi=[0, 0, 30, 90]))
    assert res["recommended"] is None
    assert "Check the signal box" in res["message"]


def test_keeps_current_settings_when_nothing_is_better():
    def head(lit):
        im = np.zeros((90, 30, 3), np.uint8)
        for i, (c, col) in enumerate((("red", (30, 30, 255)), ("amber", (0, 190, 255)), ("green", (80, 230, 60)))):
            cv2.circle(im, (15, 15 + 30 * i), 11, col if c == lit else (40, 40, 50), -1)
        return im
    samples = [(head(c), c) for c in ("red", "amber", "green") for _ in range(3)]
    base = SignalConfig(roi=[0, 0, 30, 90])
    res = calibrate(samples, base)
    assert res["recommended"]["unchanged"] is True
    assert res["recommended"]["settings"]["sat_min"] == base.sat_min
