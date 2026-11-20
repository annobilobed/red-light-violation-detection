"""FastAPI routes, exercised through TestClient with the synthetic video."""
import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(tmp_path_factory.mktemp("data"))
    with TestClient(app) as c:
        c.svc = app.state.service
        yield c


@pytest.fixture(scope="module")
def video(client, synthetic):
    # Upload through the API path, then attach the scenario (as the CLI does).
    v = client.svc.add_video_from_path(synthetic["video"], location="Synthetic test scene",
                                       is_cape_coast=False, session="syn-1")
    return v


@pytest.fixture(scope="module")
def analysis(client, video):
    r = client.post("/api/analyses", json={"video_id": video["id"], "model": "scripted"})
    assert r.status_code == 201, r.text
    aid = r.json()["id"]
    for _ in range(600):
        a = client.get(f"/api/analyses/{aid}").json()
        if a["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.1)
    assert a["status"] == "done", a
    return a


def test_health_and_models(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    ids = [m["id"] for m in client.get("/api/models").json()]
    assert "yolo11n.pt" in ids and "yolo11m.pt" in ids


def test_upload_multipart(client, synthetic):
    data = synthetic["video"].read_bytes()
    r = client.post("/api/videos", files={"file": ("clip.mp4", io.BytesIO(data), "video/mp4")},
                    data={"location": "Kotokuraba junction", "is_cape_coast": "true", "session": "s2"})
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["is_cape_coast"] is True and v["frames"] == 900 and v["has_scenario"] is False
    assert "path" not in v
    r = client.patch(f"/api/videos/{v['id']}", json={"notes": "dusk"})
    assert r.json()["notes"] == "dusk"
    assert client.delete(f"/api/videos/{v['id']}").status_code == 204
    assert client.get(f"/api/videos/{v['id']}").status_code == 404


def test_upload_rejects_non_video(client):
    r = client.post("/api/videos", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")})
    assert r.status_code == 400
    r = client.post("/api/videos", files={"file": ("fake.mp4", io.BytesIO(b"not a video"), "video/mp4")})
    assert r.status_code == 400


def test_video_list_and_frame(client, video):
    assert any(v["id"] == video["id"] for v in client.get("/api/videos").json())
    r = client.get(f"/api/videos/{video['id']}/frame", params={"t": 18})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"


def test_config_roundtrip(client, video):
    cfg = client.get(f"/api/videos/{video['id']}/config").json()["config"]
    assert cfg["detector"]["model"] == "scripted"         # synthetic default
    cfg["signal"]["sat_min"] = 80
    r = client.put(f"/api/videos/{video['id']}/config", json={"config": cfg})
    assert r.status_code == 200 and r.json()["problems"] == []
    assert client.get(f"/api/videos/{video['id']}/config").json()["config"]["signal"]["sat_min"] == 80


def test_config_problems_reported(client, synthetic):
    v = client.svc.add_video_from_path(synthetic["video"], copy=True)
    cfg = client.get(f"/api/videos/{v['id']}/config").json()["config"]
    cfg["signal"]["roi"] = [0, 0, 0, 0]
    probs = client.put(f"/api/videos/{v['id']}/config", json={"config": cfg}).json()["problems"]
    assert any("Signal region" in p for p in probs)
    r = client.post("/api/analyses", json={"video_id": v["id"], "model": "scripted"})
    assert r.status_code == 400 and "incomplete" in r.json()["detail"]


def test_signal_test_endpoint(client, video):
    sig = {"roi": [700, 60, 40, 110], "layout": "vertical"}
    for t, expected in [(5.5, "green"), (13, "amber"), (20.5, "red"), (51, "uncertain")]:
        r = client.post(f"/api/videos/{video['id']}/signal-test", json={"t": t, "signal": sig}).json()
        assert r["state"] == expected, (t, r)
        assert set(r["lamp_rects"]) == {"red", "amber", "green"}


def test_signal_calibrate_endpoint(client, video):
    sig = {"roi": [700, 60, 40, 110], "layout": "vertical", "min_score": 0.95, "core_agreement": False}  # deliberately bad
    samples = [{"t": t, "state": s} for t, s in [(2, "green"), (6, "green"), (10, "green"), (12.5, "amber"),
                                                  (13.5, "amber"), (14.5, "amber"), (17, "red"), (22, "red"),
                                                  (27, "red")]]
    r = client.post(f"/api/videos/{video['id']}/signal-calibrate", json={"signal": sig, "samples": samples})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["current"]["metrics"]["coverage"] == 0
    rec = body["recommended"]
    assert rec["metrics"]["coverage"] == 1.0 and rec["metrics"]["safety_errors"] == 0
    bad = client.post(f"/api/videos/{video['id']}/signal-calibrate", json={"signal": sig, "samples": []})
    assert bad.status_code == 400
    # Samples are stored with the config.
    cfg = client.get(f"/api/videos/{video['id']}/config").json()["config"]
    cfg["signal"]["samples"] = samples
    client.put(f"/api/videos/{video['id']}/config", json={"config": cfg})
    assert len(client.get(f"/api/videos/{video['id']}/config").json()["config"]["signal"]["samples"]) == 9


def test_analysis_results(client, analysis):
    aid = analysis["id"]
    assert analysis["stats"]["counts"] == {"potential_violation": 3, "signal_uncertain": 1, "lawful": 7}
    segs = client.get(f"/api/analyses/{aid}/signal-segments").json()
    assert segs[2]["state"] == "red" and segs[2]["start_s"] == pytest.approx(15.0)
    pv = client.get("/api/events", params={"analysis_id": aid, "outcome": "potential_violation"}).json()
    assert [e["track_id"] for e in pv] == [4, 5, 11]
    assert all(e["has_snapshot"] and e["has_clip"] for e in pv)
    assert len(client.get("/api/events", params={"analysis_id": aid}).json()) == 11
    assert any(a["id"] == aid for a in client.get("/api/analyses").json())


def test_event_files(client, analysis):
    e = client.get("/api/events", params={"analysis_id": analysis["id"], "outcome": "potential_violation"}).json()[0]
    snap = client.get(f"/api/events/{e['id']}/snapshot")
    assert snap.status_code == 200 and snap.content[:2] == b"\xff\xd8"
    clip = client.get(f"/api/events/{e['id']}/clip")
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4"
    lawful = client.get("/api/events", params={"analysis_id": analysis["id"], "outcome": "lawful"}).json()[0]
    assert client.get(f"/api/events/{lawful['id']}/clip").status_code == 404


def test_review_flow_and_summary(client, analysis):
    aid = analysis["id"]
    pv = client.get("/api/events", params={"analysis_id": aid, "outcome": "potential_violation"}).json()
    r = client.patch(f"/api/events/{pv[0]['id']}/review", json={"status": "confirmed", "reviewer": "AB", "note": "clear"})
    assert r.status_code == 200 and r.json()["review_status"] == "confirmed" and r.json()["reviewed_at"]
    client.patch(f"/api/events/{pv[1]['id']}/review", json={"status": "rejected", "reviewer": "AB"})
    assert client.patch(f"/api/events/{pv[2]['id']}/review", json={"status": "guilty"}).status_code == 400
    s = client.get(f"/api/analyses/{aid}/summary").json()
    assert s["potential_violations"] == 3
    assert s["review_status"] == {"confirmed": 1, "rejected": 1, "pending": 1}
    assert s["reviewer_confirmed_share"] == 0.5
    confirmed = client.get("/api/events", params={"analysis_id": aid, "review_status": "confirmed"}).json()
    assert len(confirmed) == 1


def test_exports(client, analysis):
    aid = analysis["id"]
    csv_text = client.get(f"/api/analyses/{aid}/export.csv").text
    assert csv_text.startswith("event_id,") and csv_text.count("\n") == 12
    assert client.get("/api/events.csv", params={"outcome": "potential_violation"}).text.count("\n") == 4
    pdf = client.get(f"/api/analyses/{aid}/export.pdf", params={"snapshots": True, "anonymise": "standard"})
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    plain = client.get(f"/api/analyses/{aid}/export.pdf")
    assert plain.content.startswith(b"%PDF")


def test_evidence_zip_anonymised(client, analysis):
    r = client.post(f"/api/analyses/{analysis['id']}/export-evidence", json={"anonymise": "standard"})
    assert r.status_code == 200, r.text
    z = client.get(r.json()["url"])
    assert z.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert "events.csv" in names and "README.txt" in names
    assert sum(n.endswith(".mp4") for n in names) == 3 and sum(n.endswith(".jpg") for n in names) == 3
    assert all("standard" in n for n in names if n.endswith((".mp4", ".jpg")))
    bad = client.post(f"/api/analyses/{analysis['id']}/export-evidence", json={"anonymise": "magic"})
    assert bad.status_code == 400


def test_accident_demo(client, video):
    r = client.post("/api/accident", json={"video_id": video["id"], "collision_t": 20, "before_s": 2, "after_s": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exploratory"] is True and "no claim" in body["disclaimer"]
    assert client.get(body["contact_sheet_url"]).status_code == 200
    assert client.get(body["clip_url"]).status_code == 200
    assert client.post("/api/accident", json={"video_id": video["id"], "collision_t": 999}).status_code == 400


def test_file_route_blocks_traversal(client):
    assert client.get("/api/files/../app.db").status_code == 404
    assert client.get("/api/files/app.db").status_code == 404
    assert client.get("/api/files/accident/../app.db").status_code == 404


def test_cancel_and_delete(client, video):
    r = client.post("/api/analyses", json={"video_id": video["id"], "model": "scripted", "frame_stride": 1})
    aid = r.json()["id"]
    client.post(f"/api/analyses/{aid}/cancel")
    a = client.svc.wait(aid, timeout=120)
    assert a["status"] in ("cancelled", "done")
    assert client.delete(f"/api/analyses/{aid}").status_code == 204
    assert client.get(f"/api/analyses/{aid}").status_code == 404


def test_not_found(client):
    assert client.get("/api/videos/9999").status_code == 404
    assert client.get("/api/events/9999").status_code == 404
    assert client.post("/api/analyses", json={"video_id": 9999}).status_code == 404
