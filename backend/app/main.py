"""FastAPI application: REST API + the prebuilt React dashboard.

Run:  cd backend && uvicorn app.main:app --reload
Data directory: $RLV_DATA_DIR (default backend/data).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline.evidence import ffmpeg_available
from .pipeline.privacy import haar_available
from .service import BadRequest, NotFound, Service

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class VideoUpdate(BaseModel):
    location: Optional[str] = None
    is_cape_coast: Optional[bool] = None
    session: Optional[str] = None
    notes: Optional[str] = None


class SignalTest(BaseModel):
    t: float = 0.0
    signal: dict


class SignalCalibration(BaseModel):
    signal: dict
    samples: list[dict]
    max_wrong_rate: float = Field(0.0, ge=0, le=0.2)


class AnalysisRequest(BaseModel):
    video_id: int
    model: Optional[str] = None
    conf: Optional[float] = Field(None, ge=0.01, le=0.99)
    frame_stride: Optional[int] = Field(None, ge=1, le=30)
    device: Optional[str] = None
    imgsz: Optional[int] = Field(None, ge=160, le=1920)
    tracker: Optional[str] = None


class Review(BaseModel):
    status: str
    reviewer: str = ""
    note: str = ""


class AnonCheck(BaseModel):
    checked: bool
    reviewer: str = ""


class EvidenceExport(BaseModel):
    only_anon_checked: bool = False
    anonymise: Optional[str] = "standard"
    review_status: Optional[str] = None
    outcome: Optional[str] = "potential_violation"


class AccidentRequest(BaseModel):
    video_id: int
    collision_t: float = Field(..., ge=0)
    before_s: float = 3.0
    after_s: float = 3.0
    anonymise: Optional[str] = None
    model: Optional[str] = None


def create_app(data_dir: Optional[str | Path] = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("RLV_DATA_DIR") or DEFAULT_DATA_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.service.shutdown()

    app = FastAPI(title="Red-light violation detection prototype",
                  description="Research prototype. Alerts are POTENTIAL red-light violations for human review.",
                  version="1.0.0", lifespan=lifespan)
    svc = Service(data_dir)
    app.state.service = svc

    @app.exception_handler(NotFound)
    async def _nf(_: Request, exc: NotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(BadRequest)
    async def _br(_: Request, exc: BadRequest):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # ---------------- system ----------------
    @app.get("/api/health")
    def health():
        from .pipeline.privacy import FaceDetector
        return {"status": "ok", "ffmpeg": ffmpeg_available(), "haar_cascades": haar_available(),
                "face_detector": FaceDetector().name,
                "data_dir": str(svc.data)}

    @app.get("/api/models")
    def models():
        return svc.list_models()

    @app.get("/api/signal-classifiers")
    def signal_classifiers():
        return svc.list_signal_classifiers()

    @app.get("/api/trackers")
    def trackers():
        return svc.list_trackers()

    # ---------------- videos ----------------
    @app.post("/api/videos", status_code=201)
    def upload_video(file: UploadFile = File(...), location: str = Form(""), is_cape_coast: bool = Form(False),
                     session: str = Form(""), notes: str = Form("")):
        return svc.add_video_from_stream(file.file, file.filename or "video.mp4", location, is_cape_coast,
                                         session, notes)

    @app.get("/api/videos")
    def list_videos():
        return svc.list_videos()

    @app.get("/api/videos/{vid}")
    def get_video(vid: int):
        return svc.get_video(vid)

    @app.patch("/api/videos/{vid}")
    def update_video(vid: int, body: VideoUpdate):
        return svc.update_video(vid, **body.model_dump(exclude_none=True))

    @app.delete("/api/videos/{vid}", status_code=204)
    def delete_video(vid: int):
        svc.delete_video(vid)
        return Response(status_code=204)

    @app.get("/api/videos/{vid}/frame")
    def video_frame(vid: int, t: float = Query(0.0, ge=0)):
        return Response(svc.frame_jpeg(vid, t), media_type="image/jpeg")

    @app.get("/api/videos/{vid}/config")
    def get_config(vid: int):
        return {"config": svc.get_config(vid)}

    @app.put("/api/videos/{vid}/config")
    def put_config(vid: int, body: dict):
        return svc.save_config(vid, body.get("config", body))

    @app.post("/api/videos/{vid}/signal-test")
    def signal_test(vid: int, body: SignalTest):
        return svc.test_signal(vid, body.t, body.signal)

    @app.post("/api/videos/{vid}/signal-calibrate")
    def signal_calibrate(vid: int, body: SignalCalibration):
        return svc.calibrate_signal(vid, body.signal, body.samples, body.max_wrong_rate)

    # ---------------- analyses ----------------
    @app.post("/api/analyses", status_code=201)
    def start_analysis(body: AnalysisRequest):
        return svc.start_analysis(body.video_id, body.model, body.conf, body.frame_stride, body.device, body.imgsz,
                                  body.tracker)

    @app.get("/api/analyses")
    def list_analyses(video_id: Optional[int] = None):
        return svc.list_analyses(video_id)

    @app.get("/api/analyses/{aid}")
    def get_analysis(aid: int):
        return svc.get_analysis(aid)

    @app.delete("/api/analyses/{aid}", status_code=204)
    def delete_analysis(aid: int):
        svc.delete_analysis(aid)
        return Response(status_code=204)

    @app.post("/api/analyses/{aid}/cancel")
    def cancel_analysis(aid: int):
        return svc.cancel_analysis(aid)

    @app.get("/api/analyses/{aid}/signal-segments")
    def signal_segments(aid: int):
        return svc.segments(aid)

    @app.get("/api/analyses/{aid}/summary")
    def summary(aid: int):
        return svc.summary(aid)

    @app.get("/api/analyses/{aid}/export.csv")
    def export_csv(aid: int, outcome: Optional[str] = None):
        return PlainTextResponse(svc.export_csv(aid, outcome or None), media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="analysis{aid}_events.csv"'})

    @app.get("/api/analyses/{aid}/export.pdf")
    def export_pdf(aid: int, snapshots: bool = False, anonymise: Optional[str] = None,
                   outcome: Optional[str] = "potential_violation"):
        pdf = svc.export_pdf(aid, snapshots, anonymise or None, outcome or None)
        return Response(pdf, media_type="application/pdf", headers={
            "Content-Disposition": f'attachment; filename="analysis{aid}_report.pdf"'})

    @app.post("/api/analyses/{aid}/export-evidence")
    def export_evidence(aid: int, body: EvidenceExport):
        path = svc.export_evidence_zip(aid, body.anonymise or None, body.review_status or None,
                                       body.outcome or None, body.only_anon_checked)
        return {"url": svc.file_url(path), "filename": path.name}

    @app.get("/api/events.csv")
    def export_all_csv(outcome: Optional[str] = None):
        return PlainTextResponse(svc.export_csv(None, outcome or None), media_type="text/csv", headers={
            "Content-Disposition": 'attachment; filename="all_events.csv"'})

    # ---------------- events ----------------
    @app.get("/api/events")
    def list_events(analysis_id: Optional[int] = None, outcome: Optional[str] = None,
                    review_status: Optional[str] = None):
        return svc.list_events(analysis_id, outcome or None, review_status or None)

    @app.get("/api/events/{eid}")
    def get_event(eid: int):
        return svc.get_event(eid)

    @app.patch("/api/events/{eid}/review")
    def review(eid: int, body: Review):
        return svc.review_event(eid, body.status, body.reviewer, body.note)

    @app.get("/api/events/{eid}/snapshot")
    def snapshot(eid: int, anonymise: Optional[str] = None):
        if anonymise:
            return FileResponse(svc.anonymised_snapshot(eid, anonymise), media_type="image/jpeg")
        return FileResponse(svc.event_file(eid, "snapshot"), media_type="image/jpeg")

    @app.patch("/api/events/{eid}/anon-check")
    def anon_check(eid: int, body: AnonCheck):
        return svc.set_anon_checked(eid, body.checked, body.reviewer)

    @app.get("/api/events/{eid}/clip")
    def clip(eid: int):
        return FileResponse(svc.event_file(eid, "clip"), media_type="video/mp4")

    # ---------------- accident demo ----------------
    @app.post("/api/accident")
    def accident(body: AccidentRequest):
        return svc.accident_sequence(body.video_id, body.collision_t, body.before_s, body.after_s,
                                     body.anonymise or None, body.model or None)

    @app.get("/api/files/{rel:path}")
    def files(rel: str):
        p = svc.resolve_file(rel)
        return FileResponse(p, filename=p.name if p.suffix == ".zip" else None)

    # ---------------- dashboard ----------------
    if (STATIC_DIR / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(STATIC_DIR / "index.html")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            f = STATIC_DIR / path
            if f.is_file() and STATIC_DIR in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
