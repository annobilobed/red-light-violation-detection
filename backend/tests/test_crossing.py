import pytest

from app.pipeline.config import CrossingConfig, SignalConfig
from app.pipeline.crossing import (LAWFUL, POTENTIAL_VIOLATION, SIGNAL_UNCERTAIN, CrossingDetector,
                                   Detection, decide)
from app.pipeline.signal_state import SignalTimeline

FPS = 10.0


def cc(**kw):
    base = dict(stop_line=[[0, 100], [200, 100]], approach_point=[100, 200], min_approach_frames=3,
                line_margin_px=2)
    base.update(kw)
    return CrossingConfig(**base)


def drive(det, ys, x=100, tid=1, cls="car", start=0):
    """Feed one vehicle whose box bottom follows ys (one value per frame)."""
    out = []
    for i, y in enumerate(ys):
        f = start + i
        out += det.update(f, f / FPS, [Detection(tid, cls, 0.9, (x - 20, y - 40, x + 20, y))])
    return out


def test_crossing_detected_and_interpolated():
    det = CrossingDetector(cc())
    out = drive(det, [160, 140, 120, 104, 96, 80])
    assert len(out) == 1
    c = out[0]
    assert c.prev_frame == 3 and c.crossing_frame == 4
    assert c.crossing_time_s == pytest.approx(0.35)  # halfway between 104 and 96
    assert c.anchor[1] == pytest.approx(100)


def test_each_track_crosses_once():
    det = CrossingDetector(cc())
    drive(det, [160, 140, 120, 90, 120, 150, 120, 80])
    assert len(det.crossings) == 1


def test_wrong_direction_not_counted():
    det = CrossingDetector(cc())
    assert drive(det, [20, 40, 60, 80, 120, 140, 160]) == []


def test_needs_approach_frames():
    det = CrossingDetector(cc(min_approach_frames=3))
    assert drive(det, [110, 104, 90, 70]) == []  # only 2 approach samples


def test_jitter_around_line_not_a_crossing():
    det = CrossingDetector(cc(line_margin_px=3))
    assert drive(det, [140, 130, 120, 101, 99, 101, 99, 101]) == []


def test_off_segment_not_counted():
    det = CrossingDetector(cc())
    assert drive(det, [160, 140, 120, 90, 70], x=300) == []


def test_approach_zone():
    zone = [[0, 100], [200, 100], [200, 300], [0, 300]]
    det = CrossingDetector(cc(approach_zone=zone))
    assert len(drive(det, [160, 140, 120, 90], x=100)) == 1
    det = CrossingDetector(cc(approach_zone=[[0, 100], [50, 100], [50, 300], [0, 300]]))
    assert drive(det, [160, 140, 120, 90], x=100) == []


def test_majority_class():
    det = CrossingDetector(cc())
    seq = [160, 140, 120, 90]
    for i, y in enumerate(seq):
        cls = "truck" if i == 0 else "bus"
        det.update(i, i / FPS, [Detection(1, cls, 0.9, (80, y - 40, 120, y))])
    assert det.crossings[0].vehicle_class == "bus"


def test_approach_point_on_line_rejected():
    with pytest.raises(ValueError):
        CrossingDetector(cc(approach_point=[50, 100]))


# ------------------------------------------------------------------ decide

def timeline(states):
    n = len(states)
    return SignalTimeline(list(range(n)), [i / FPS for i in range(n)], states,
                          SignalConfig(min_stable_frames=1, bridge_gap_s=0))


def crossing_at(t_cross, frame):
    det = CrossingDetector(cc())
    from app.pipeline.crossing import Crossing
    return Crossing(1, "car", t_cross, frame, frame - 1, (100, 100), 0.9)


SIG = ["green"] * 10 + ["amber"] * 5 + ["red"] * 10 + ["uncertain"] * 5 + ["red"] * 5


def test_decide_green_amber_lawful():
    assert decide(crossing_at(0.55, 6), timeline(SIG), cc()).outcome == LAWFUL
    c = decide(crossing_at(1.25, 13), timeline(SIG), cc())
    assert (c.outcome, c.signal_state) == (LAWFUL, "amber")


def test_decide_after_red_is_potential_violation():
    c = decide(crossing_at(1.85, 19), timeline(SIG), cc())
    assert c.outcome == POTENTIAL_VIOLATION
    assert c.red_onset_s == pytest.approx(1.5)
    assert c.time_since_red_s == pytest.approx(0.35)
    assert c.red_onset_exact is True


def test_crossing_before_onset_inside_frame_gap_is_lawful():
    # Crossed at 1.47 s; first frame past the line (15, t=1.5) already shows red.
    c = decide(crossing_at(1.47, 15), timeline(SIG), cc())
    assert c.signal_state == "red" and c.outcome == LAWFUL


def test_grace_period():
    c = decide(crossing_at(1.6, 16), timeline(SIG), cc(red_grace_s=0.2))
    assert c.outcome == LAWFUL
    c = decide(crossing_at(1.8, 18), timeline(SIG), cc(red_grace_s=0.2))
    assert c.outcome == POTENTIAL_VIOLATION


def test_uncertain_signal_never_flagged():
    c = decide(crossing_at(2.75, 27), timeline(SIG), cc())
    assert c.outcome == SIGNAL_UNCERTAIN


def test_red_after_uncertain_has_inexact_onset():
    c = decide(crossing_at(3.25, 32), timeline(SIG), cc())
    assert c.outcome == POTENTIAL_VIOLATION and c.red_onset_exact is False
