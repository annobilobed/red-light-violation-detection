import csv
import io

from app import reports


def ev(i, outcome="potential_violation", status="pending", tsr=1.0, cls="car"):
    return {"id": i, "analysis_id": 1, "video_filename": "v.mp4", "video_location": "Test",
            "video_is_cape_coast": False, "track_id": i, "vehicle_class": cls, "outcome": outcome,
            "crossing_time_s": 10.0 + i, "crossing_frame": 100 + i, "signal_state": "red",
            "red_onset_s": 9.0, "red_onset_exact": True, "time_since_red_s": tsr, "det_conf": 0.8,
            "review_status": status, "reviewer": "", "review_note": "", "reviewed_at": None}


def test_csv_has_header_and_rows():
    text = reports.events_csv([ev(1), ev(2, "lawful")])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["outcome"] == "potential_violation"
    assert rows[0]["cape_coast_footage"] == "no"
    assert set(reports.CSV_FIELDS) == set(rows[0])


def test_summary_counts_and_reviewer_share():
    events = [ev(1, status="confirmed", tsr=0.5), ev(2, status="rejected", tsr=2.0), ev(3, tsr=3.0),
              ev(4, "lawful"), ev(5, "signal_uncertain")]
    s = reports.summarise(events, [{"state": "red", "start_s": 0, "end_s": 3},
                                   {"state": "green", "start_s": 3, "end_s": 4}])
    assert s["potential_violations"] == 3
    assert s["by_outcome"] == {"potential_violation": 3, "lawful": 1, "signal_uncertain": 1}
    assert s["reviewer_confirmed_share"] == 0.5
    assert s["time_since_red_s"] == {"min": 0.5, "median": 2.0, "max": 3.0}
    assert s["signal_time_share"]["red"] == 0.75


def test_summary_no_reviews():
    assert reports.summarise([ev(1)])["reviewer_confirmed_share"] is None


def test_pdf_renders_and_labels_non_cape_coast():
    analysis = {"id": 1, "video_filename": "v.mp4", "video_location": "Lab", "model": "scripted",
                "finished_at": None, "video_is_cape_coast": False}
    pdf = reports.events_pdf(analysis, [ev(1)], reports.summarise([ev(1)]))
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1500
