"""Every decision rule has a scenario in tools/make_synthetic_video.py."""
import subprocess

import pytest

from app.pipeline.config import AnalysisConfig
from app.pipeline.detectors import ScriptedDetector, make_tracker, scenario_path_for
from app.pipeline.runner import analyse_video


def by_track(result):
    return {c["track_id"]: c for c in result["crossings"]}


@pytest.fixture(scope="module")
def run(synthetic, tmp_path_factory):
    cfg = AnalysisConfig.from_dict(synthetic["config"])
    out = tmp_path_factory.mktemp("e2e")
    tracker = make_tracker(cfg.detector, synthetic["video"])
    return analyse_video(synthetic["video"], cfg, tracker, out), out


def test_every_scenario_outcome(run, synthetic):
    result, _ = run
    got = by_track(result)
    for v in synthetic["scenario"]["vehicles"]:
        if v["expected"] is None:
            assert v["track_id"] not in got, f"{v['label']} should not cross"
        else:
            assert got[v["track_id"]]["outcome"] == v["expected"], v["label"]
            assert got[v["track_id"]]["crossing_time_s"] == pytest.approx(v["crossing_time_s"], abs=0.07)


def test_counts(run):
    result, _ = run
    assert result["counts"] == {"potential_violation": 3, "signal_uncertain": 1, "lawful": 7}


def test_entered_on_amber_still_in_junction_at_red_is_lawful(run):
    c = by_track(run[0])[3]   # scenario C
    assert c["signal_state"] == "amber" and c["outcome"] == "lawful"


def test_crossing_just_before_red_is_lawful(run):
    assert by_track(run[0])[10]["outcome"] == "lawful"   # scenario M, 0.1 s before red


def test_time_since_red(run):
    got = by_track(run[0])
    assert got[4]["time_since_red_s"] == pytest.approx(3.0, abs=0.07)
    assert got[11]["time_since_red_s"] == pytest.approx(1.0, abs=0.07)


def test_flicker_bridged_and_occlusion_uncertain(run):
    segs = run[0]["signal_segments"]
    states = [s["state"] for s in segs]
    assert states == ["green", "amber", "red", "green", "amber", "red", "uncertain", "red"]
    red = segs[2]
    assert red["start_s"] == pytest.approx(15.0) and red["onset_exact"]


def test_evidence_written_h264(run):
    result, out = run
    flagged = [c for c in result["crossings"] if c["outcome"] != "lawful"]
    assert len(flagged) == 4
    for c in flagged:
        ev = c["evidence"]
        assert ev and (out / "evidence").exists()
        from pathlib import Path
        assert Path(ev["snapshot"]).stat().st_size > 1000
        assert Path(ev["clip"]).stat().st_size > 1000
        if ev["clip_h264"]:
            codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=codec_name", "-of", "csv=p=0", ev["clip"]],
                                   capture_output=True, text=True).stdout.strip()
            assert codec == "h264"
    assert all(c["evidence"] is None for c in result["crossings"] if c["outcome"] == "lawful")
    assert (out / "tracks.npz").exists() and (out / "result.json").exists()


def test_frame_stride_two_same_outcomes(synthetic, tmp_path):
    cfg = AnalysisConfig.from_dict(synthetic["config"])
    cfg.frame_stride = 2
    r = analyse_video(synthetic["video"], cfg, make_tracker(cfg.detector, synthetic["video"]), tmp_path,
                      render_evidence=False)
    assert r["counts"] == {"potential_violation": 3, "signal_uncertain": 1, "lawful": 7}


def test_robust_to_box_jitter(synthetic, tmp_path):
    cfg = AnalysisConfig.from_dict(synthetic["config"])
    tracker = ScriptedDetector(scenario_path_for(synthetic["video"]), jitter_px=2.0, seed=3)
    r = analyse_video(synthetic["video"], cfg, tracker, None)
    assert r["counts"]["potential_violation"] == 3


def test_incomplete_config_rejected(synthetic):
    cfg = AnalysisConfig()
    with pytest.raises(ValueError, match="incomplete"):
        analyse_video(synthetic["video"], cfg, ScriptedDetector(scenario_path_for(synthetic["video"])))
