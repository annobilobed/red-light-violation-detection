import pytest

from app.pipeline.detectors import TRACKERS, resolve_tracker


def test_resolve_tracker_names_and_legacy_values():
    assert resolve_tracker("bytetrack") == "bytetrack.yaml"
    assert resolve_tracker("bytetrack.yaml") == "bytetrack.yaml"   # configs saved before the option
    assert resolve_tracker("botsort_static").endswith("botsort_static.yaml")
    assert resolve_tracker("custom/my.yaml") == "custom/my.yaml"
    with pytest.raises(ValueError):
        resolve_tracker("deepsort")


def test_shipped_tracker_files_exist():
    from pathlib import Path
    for key, (path, _) in TRACKERS.items():
        if "/" in path:
            assert Path(path).exists(), key
