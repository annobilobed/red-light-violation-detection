import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

from metrics import (cohen_kappa, match_events, mean_sd, per_class_prf, signal_metrics,  # noqa: E402
                     tracking_metrics, wilson_interval)
from split_sessions import assign_sessions  # noqa: E402


def test_kappa():
    assert cohen_kappa(list("aabb"), list("aabb"))["kappa"] == pytest.approx(1)
    r = cohen_kappa(list("aabb"), list("abab"))
    assert r["percent_agreement"] == 0.5 and r["kappa"] == pytest.approx(0)
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


def test_prf_and_ci():
    m = per_class_prf(["r", "r", "g"], ["r", "g", "g"], ["r", "g"])
    assert m["r"]["precision"] == 1 and m["r"]["recall"] == 0.5
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.8 < hi
    assert wilson_interval(0, 0) is None
    assert mean_sd([1, 2, 3])["sd"] == pytest.approx(1)


def test_match_events_one_to_one():
    m = match_events([10.0, 20.0, 30.0], [10.4, 19.0, 29.8, 50.0], tolerance=0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 2, 1)
    m = match_events([10.0], [10.1, 10.2], tolerance=1)  # two predictions, one truth
    assert (m["tp"], m["fp"]) == (1, 1)


def test_signal_metrics_coverage_and_safety_errors():
    gt = [(0, 1, "green"), (1, 2, "red")]
    times = [0.1, 0.5, 0.9, 1.1, 1.5, 1.9]
    pred = ["green", "uncertain", "green", "green", "red", "red"]
    m = signal_metrics(gt, times, pred)
    assert m["coverage"] == pytest.approx(5 / 6)
    assert m["accuracy_on_covered"] == pytest.approx(4 / 5)
    assert m["red_read_as_green_or_amber"] == 1 and m["non_red_read_as_red"] == 0


def test_tracking_metrics_id_switch():
    gt = {f: [(1, 10 * f, 0, 10, 10)] for f in range(1, 7)}
    pred = {f: [(7 if f <= 3 else 8, 10 * f, 0, 10, 10)] for f in range(1, 7)}
    m = tracking_metrics(gt, pred)
    assert m["id_switches"] == 1 and m["fp"] == 0 and m["fn"] == 0
    assert m["mota"] == pytest.approx(1 - 1 / 6)
    assert m["idf1"] == pytest.approx(0.5)


def test_tracking_metrics_fragmentation():
    gt = {f: [(1, 0, 0, 10, 10)] for f in range(1, 6)}
    pred = {f: [(5, 0, 0, 10, 10)] for f in (1, 2, 4, 5)}
    m = tracking_metrics(gt, pred)
    assert m["fn"] == 1 and m["fragmentations"] == 1 and m["id_switches"] == 0


def test_split_keeps_sessions_whole_and_fixed_test():
    sessions = [f"s{i}" for i in range(10)]
    cond = {s: ("day" if i < 6 else "night") for i, s in enumerate(sessions)}
    split = assign_sessions(sessions, cond, seed=1)
    assert set(split) == set(sessions)
    for c in ("day", "night"):
        assert any(split[s] == "test" for s in sessions if cond[s] == c)
    assert split == assign_sessions(sessions, cond, seed=1)  # deterministic
    fixed = assign_sessions(sessions, cond, fixed_test=["s0", "s9"])
    assert sorted(s for s, p in fixed.items() if p == "test") == ["s0", "s9"]
