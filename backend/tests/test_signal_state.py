import cv2
import numpy as np
import pytest

from app.pipeline.config import SignalConfig
from app.pipeline.signal_state import (SignalTimeline, classify_scores, lamp_rects, read_signal,
                                       smooth_states)

LIT = {"red": (30, 30, 255), "amber": (0, 190, 255), "green": (80, 230, 60)}
DIM = (40, 40, 50)


def head(state, layout="vertical"):
    img = np.zeros((200, 200, 3), np.uint8)
    for i, c in enumerate(("red", "amber", "green")):
        centre = (100, 40 + 50 * i) if layout == "vertical" else (40 + 50 * i, 100)
        cv2.circle(img, centre, 18, LIT[c] if c == state else DIM, -1)
    return img


def cfg(layout="vertical", **kw):
    roi = [80, 20, 40, 150] if layout == "vertical" else [20, 80, 150, 40]
    return SignalConfig(roi=roi, layout=layout, **kw)


@pytest.mark.parametrize("state", ["red", "amber", "green"])
@pytest.mark.parametrize("layout", ["vertical", "horizontal"])
def test_reads_each_lamp(state, layout):
    assert read_signal(head(state, layout), cfg(layout)).state == state


def test_all_dark_is_uncertain():
    r = read_signal(head(None), cfg())
    assert r.state == "uncertain"
    assert r.method == "none"


def test_two_lamps_lit_is_uncertain():
    img = head("red")
    cv2.circle(img, (100, 140), 18, LIT["green"], -1)
    assert read_signal(img, cfg()).state == "uncertain"


def test_roi_outside_frame_is_uncertain():
    c = SignalConfig(roi=[500, 500, 30, 90])
    assert read_signal(head("red"), c).state == "uncertain"


def test_brightness_fallback_only_when_enabled():
    img = head(None)
    cv2.circle(img, (100, 40), 18, (250, 250, 250), -1)  # over-exposed white red lamp
    assert read_signal(img, cfg()).state == "uncertain"
    r = read_signal(img, cfg(brightness_fallback=True))
    assert (r.state, r.method) == ("red", "brightness")


def test_custom_layout_rects():
    c = SignalConfig(layout="custom", lamps={"red": [1, 2, 3, 4], "amber": [5, 6, 7, 8], "green": [9, 9, 9, 9]})
    assert lamp_rects(c)["amber"] == (5, 6, 7, 8)


def test_classify_scores_margin():
    assert classify_scores({"red": 0.5, "amber": 0.0, "green": 0.0}, 0.08, 0.04) == "red"
    assert classify_scores({"red": 0.5, "amber": 0.48, "green": 0.0}, 0.08, 0.04) == "uncertain"
    assert classify_scores({"red": 0.05, "amber": 0.0, "green": 0.0}, 0.08, 0.04) == "uncertain"


def times(n, fps=10):
    return [i / fps for i in range(n)]


def test_short_runs_become_uncertain():
    s = ["red"] * 5 + ["green"] + ["red"] * 5
    out = smooth_states(s, times(len(s)), SignalConfig(min_stable_frames=3, bridge_gap_s=0))
    assert out[5] == "uncertain"


def test_bridge_same_colour_gap():
    s = ["red"] * 5 + ["uncertain"] * 2 + ["red"] * 5
    out = smooth_states(s, times(len(s)), SignalConfig(min_stable_frames=1, bridge_gap_s=0.5))
    assert out == ["red"] * 12


def test_never_bridge_between_different_colours():
    s = ["amber"] * 5 + ["uncertain"] * 2 + ["red"] * 5
    out = smooth_states(s, times(len(s)), SignalConfig(min_stable_frames=1, bridge_gap_s=5))
    assert out[5:7] == ["uncertain", "uncertain"]


def test_long_gap_not_bridged():
    s = ["red"] * 5 + ["uncertain"] * 10 + ["red"] * 5
    out = smooth_states(s, times(len(s)), SignalConfig(min_stable_frames=1, bridge_gap_s=0.5))
    assert out.count("uncertain") == 10


def test_timeline_segments_and_onset():
    s = ["green"] * 5 + ["amber"] * 3 + ["red"] * 6 + ["uncertain"] * 10 + ["red"] * 4
    frames = list(range(0, 2 * len(s), 2))  # stride 2
    tl = SignalTimeline(frames, times(len(s)), s, SignalConfig(min_stable_frames=1, bridge_gap_s=0))
    assert [g.state for g in tl.segments] == ["green", "amber", "red", "uncertain", "red"]
    red1, red2 = tl.segments[2], tl.segments[4]
    assert red1.onset_exact and not red2.onset_exact
    assert red1.start_s == pytest.approx(0.8)
    assert tl.state_at_frame(17) == "red"      # between processed frames 16 and 18
    assert tl.state_at_frame(-1) == "uncertain"
