"""Application services used by the FastAPI routes (and by tests/CLI).

Owns the data directory layout:
  <data>/app.db
  <data>/videos/<id-prefix>_<name>
  <data>/analyses/<analysis_id>/{result.json, tracks.npz, signal_readings.csv, evidence/}
  <data>/exports/
  <data>/accident/
  <data>/models/            optional fine-tuned *.pt checkpoints
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO, Optional

import cv2

from . import reports
from .db import REVIEW_STATUSES, Database, now_iso
from .pipeline.accident import make_sequence
from .pipeline.calibration import calibrate as calibrate_thresholds
from .pipeline.config import AnalysisConfig, SignalConfig
from .pipeline.crossing import Crossing
from .pipeline.detectors import make_tracker, scenario_path_for
from .pipeline.evidence import (TrackStore, ffmpeg_available, probe_video, read_frame_at,
                                render_event_evidence, render_snapshot)
from .pipeline.privacy import MODES as ANON_MODES
from .pipeline.runner import Cancelled, analyse_video, load_timeline_from_result
from .pipeline.signal_state import lamp_rects, read_signal

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".webm"}
STOCK_MODELS = [
    {"id": "yolo11n.pt", "label": "YOLO11n (nano, proposed lightweight model)"},
    {"id": "yolo11m.pt", "label": "YOLO11m (medium, baseline)"},
]


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


def _safe_name(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120] or "video"


class Service:
    def __init__(self, data_dir: str | Path):
        self.data = Path(data_dir).resolve()
        for sub in ("videos", "analyses", "exports", "accident", "models"):
            (self.data / sub).mkdir(parents=True, exist_ok=True)
        self.db = Database(self.data / "app.db")
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
        self._cancel: dict[int, threading.Event] = {}
        self._futures: dict = {}
        # Jobs interrupted by a restart cannot resume.
        for a in self.db.list_analyses():
            if a["status"] in ("queued", "running"):
                self.db.update_analysis(a["id"], status="failed", error="Server restarted during analysis.",
                                        finished_at=now_iso())

    def shutdown(self) -> None:
        for ev in self._cancel.values():
            ev.set()
        self.pool.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------------
    # Videos
    # ------------------------------------------------------------------
    def add_video_from_stream(self, stream: BinaryIO, filename: str, location: str = "",
                              is_cape_coast: bool = False, session: str = "", notes: str = "") -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in VIDEO_EXTS:
            raise BadRequest(f"Unsupported file type '{ext}'. Use one of {sorted(VIDEO_EXTS)}.")
        dest = self.data / "videos" / f"{uuid.uuid4().hex[:8]}_{_safe_name(filename)}"
        with open(dest, "wb") as fh:
            shutil.copyfileobj(stream, fh, length=1024 * 1024)
        return self._register(dest, filename, location, is_cape_coast, session, notes)

    def add_video_from_path(self, path: str | Path, location: str = "", is_cape_coast: bool = False,
                            session: str = "", notes: str = "", copy: bool = True) -> dict:
        src = Path(path)
        if not src.exists():
            raise BadRequest(f"No such file: {src}")
        dest = src
        if copy:
            dest = self.data / "videos" / f"{uuid.uuid4().hex[:8]}_{_safe_name(src.name)}"
            shutil.copy2(src, dest)
            sp = scenario_path_for(src)
            if sp.exists():  # keep synthetic ground truth and config next to the copy
                shutil.copy2(sp, scenario_path_for(dest))
                syn_cfg = Path(str(src.with_suffix("")) + ".config.json")
                if syn_cfg.exists():
                    shutil.copy2(syn_cfg, Path(str(dest.with_suffix("")) + ".config.json"))
        return self._register(dest, src.name, location, is_cape_coast, session, notes)

    def _register(self, path: Path, filename: str, location, is_cape_coast, session, notes) -> dict:
        try:
            meta = probe_video(path)
        except ValueError as exc:
            path.unlink(missing_ok=True)
            raise BadRequest(f"Not a readable video: {exc}") from exc
        if meta["frames"] <= 0:
            path.unlink(missing_ok=True)
            raise BadRequest("The video has no frames.")
        vid = self.db.add_video(filename=filename, path=str(path), location=location,
                                is_cape_coast=is_cape_coast, session=session, notes=notes, **{
                                    k: meta[k] for k in ("fps", "frames", "width", "height", "duration_s")})
        return self.get_video(vid)

    def get_video(self, vid: int) -> dict:
        v = self.db.get_video(vid)
        if not v:
            raise NotFound(f"video {vid} not found")
        v["has_scenario"] = scenario_path_for(v["path"]).exists()
        v.pop("path", None)
        return v

    def _video_path(self, vid: int) -> Path:
        v = self.db.get_video(vid)
        if not v:
            raise NotFound(f"video {vid} not found")
        return Path(v["path"])

    def list_videos(self) -> list[dict]:
        out = []
        for v in self.db.list_videos():
            v["has_scenario"] = scenario_path_for(v["path"]).exists()
            v.pop("path", None)
            out.append(v)
        return out

    def update_video(self, vid: int, **fields) -> dict:
        self.get_video(vid)
        self.db.update_video_meta(vid, **fields)
        return self.get_video(vid)

    def delete_video(self, vid: int) -> None:
        path = self._video_path(vid)
        for a in self.db.list_analyses(vid):
            if a["status"] in ("queued", "running"):
                raise BadRequest("Cancel the running analysis of this video first.")
            if a.get("output_dir"):
                shutil.rmtree(a["output_dir"], ignore_errors=True)
        self.db.delete_video(vid)
        if self.data in path.resolve().parents:
            path.unlink(missing_ok=True)
            scenario_path_for(path).unlink(missing_ok=True)
            Path(str(path.with_suffix("")) + ".config.json").unlink(missing_ok=True)

    def frame_jpeg(self, vid: int, t: float) -> bytes:
        frame = read_frame_at(self._video_path(vid), max(0.0, t))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise BadRequest("Could not encode frame")
        return buf.tobytes()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def default_config(self, vid: int) -> dict:
        v = self.get_video(vid)
        w, h = v["width"], v["height"]
        cfg = AnalysisConfig()
        cfg.crossing.stop_line = [[w * 0.25, h * 0.6], [w * 0.75, h * 0.6]]
        cfg.crossing.approach_point = [w * 0.5, h * 0.85]
        cfg.signal.roi = [0, 0, 0, 0]
        return cfg.to_dict()

    def get_config(self, vid: int) -> dict:
        v = self.get_video(vid)
        cfg = v["config"]
        if cfg is None:
            # Synthetic videos ship a ready-made config.
            p = self._video_path(vid)
            syn = Path(str(p.with_suffix("")) + ".config.json")
            if syn.exists():
                return AnalysisConfig.from_dict(json.loads(syn.read_text())).to_dict()
            return self.default_config(vid)
        return AnalysisConfig.from_dict(cfg).to_dict()

    def save_config(self, vid: int, data: dict) -> dict:
        self.get_video(vid)
        try:
            cfg = AnalysisConfig.from_dict(data)
        except (TypeError, ValueError) as exc:
            raise BadRequest(f"Invalid configuration: {exc}") from exc
        self.db.set_video_config(vid, cfg.to_dict())
        return {"config": cfg.to_dict(), "problems": cfg.validate()}

    def test_signal(self, vid: int, t: float, signal: dict) -> dict:
        frame = read_frame_at(self._video_path(vid), max(0.0, t))
        try:
            scfg = AnalysisConfig.from_dict({"signal": signal}).signal
            rects = lamp_rects(scfg)
        except (TypeError, ValueError, KeyError) as exc:
            raise BadRequest(f"Invalid signal settings: {exc}") from exc
        reading = read_signal(frame, scfg)
        out = reading.to_dict()
        out["lamp_rects"] = {k: list(v) for k, v in rects.items()}
        out["thresholds"] = {"sat_min": scfg.sat_min, "val_min": scfg.val_min, "min_score": scfg.min_score,
                             "margin": scfg.margin}
        out["note"] = ("Single-frame reading, before temporal smoothing. 'uncertain' means no lamp clearly "
                       "passed the thresholds; the system never guesses a colour.")
        return out

    def calibrate_signal(self, vid: int, signal: dict, samples: list[dict], max_wrong_rate: float = 0.0) -> dict:
        """Search HSV thresholds that read the labelled frames best (see
        pipeline/calibration.py). Does not save anything."""
        path = self._video_path(vid)
        try:
            scfg = AnalysisConfig.from_dict({"signal": signal}).signal
            lamp_rects(scfg)
        except (TypeError, ValueError, KeyError) as exc:
            raise BadRequest(f"Invalid signal settings: {exc}") from exc
        if not samples:
            raise BadRequest("Label some frames first (at least 3 of each colour you can find).")
        frames = []
        for smp in samples:
            try:
                frames.append((read_frame_at(path, max(0.0, float(smp["t"]))), str(smp["state"])))
            except (KeyError, ValueError) as exc:
                raise BadRequest(f"Bad sample {smp}: {exc}") from exc
        try:
            return calibrate_thresholds(frames, scfg, max_wrong_rate=max_wrong_rate)
        except ValueError as exc:
            raise BadRequest(str(exc)) from exc

    def list_models(self) -> list[dict]:
        models = [dict(m) for m in STOCK_MODELS]
        for p in sorted((self.data / "models").glob("*.pt")):
            models.append({"id": str(p), "label": f"Fine-tuned: {p.name}"})
        models.append({"id": "scripted", "label": "Synthetic ground truth (test videos only)"})
        return models

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------
    def start_analysis(self, vid: int, model: Optional[str] = None, conf: Optional[float] = None,
                       frame_stride: Optional[int] = None, device: Optional[str] = None,
                       imgsz: Optional[int] = None) -> dict:
        path = self._video_path(vid)
        cfg = AnalysisConfig.from_dict(self.get_config(vid))
        if model:
            cfg.detector.model = model
        if conf is not None:
            cfg.detector.conf = float(conf)
        if frame_stride:
            cfg.frame_stride = max(1, int(frame_stride))
        if device:
            cfg.detector.device = device
        if imgsz:
            cfg.detector.imgsz = int(imgsz)
        problems = cfg.validate()
        if problems:
            raise BadRequest("Configuration incomplete: " + " ".join(problems))
        if cfg.detector.model == "scripted" and not scenario_path_for(path).exists():
            raise BadRequest("The synthetic ground-truth detector only works on synthetic test videos.")
        aid = self.db.add_analysis(vid, cfg.detector.model, cfg.to_dict())
        out_dir = self.data / "analyses" / str(aid)
        self.db.update_analysis(aid, output_dir=str(out_dir))
        cancel = threading.Event()
        self._cancel[aid] = cancel
        self._futures[aid] = self.pool.submit(self._run, aid, path, cfg, out_dir, cancel)
        return self.get_analysis(aid)

    def _run(self, aid: int, path: Path, cfg: AnalysisConfig, out_dir: Path, cancel: threading.Event) -> None:
        if cancel.is_set():
            self.db.update_analysis(aid, status="cancelled", finished_at=now_iso())
            return
        self.db.update_analysis(aid, status="running", started_at=now_iso(), message="loading model")
        last = [0.0]

        def progress(f: float, msg: str) -> None:
            if f - last[0] >= 0.01 or f >= 1.0:
                last[0] = f
                self.db.update_analysis(aid, progress=round(f, 4), message=msg)

        try:
            tracker = make_tracker(cfg.detector, path)
            result = analyse_video(path, cfg, tracker, out_dir, progress, cancel.is_set)
            v = self.db.get_analysis(aid)
            self.db.store_results(aid, v["video_id"], result)
            stats = {"counts": result["counts"], "timing": result["timing"],
                     "signal_summary": result["signal_summary"], "tracker": result["tracker"],
                     "ffmpeg": ffmpeg_available()}
            self.db.update_analysis(aid, status="done", progress=1.0, message="done", finished_at=now_iso(),
                                    stats=stats)
        except Cancelled:
            self.db.update_analysis(aid, status="cancelled", message="cancelled", finished_at=now_iso())
        except Exception as exc:  # noqa: BLE001 - report any failure to the user
            log.exception("analysis %s failed", aid)
            self.db.update_analysis(aid, status="failed", error=f"{type(exc).__name__}: {exc}",
                                    message=traceback.format_exc(limit=3)[-500:], finished_at=now_iso())
        finally:
            self._cancel.pop(aid, None)

    def wait(self, aid: int, timeout: Optional[float] = None) -> dict:
        fut = self._futures.get(aid)
        if fut is not None:
            fut.result(timeout=timeout)
        return self.get_analysis(aid)

    def cancel_analysis(self, aid: int) -> dict:
        a = self.get_analysis(aid)
        ev = self._cancel.get(aid)
        if ev is None or a["status"] not in ("queued", "running"):
            raise BadRequest(f"Analysis {aid} is not running (status: {a['status']}).")
        ev.set()
        return a

    def get_analysis(self, aid: int) -> dict:
        a = self.db.get_analysis(aid)
        if not a:
            raise NotFound(f"analysis {aid} not found")
        a.pop("output_dir", None)
        return a

    def list_analyses(self, video_id: Optional[int] = None) -> list[dict]:
        out = []
        for a in self.db.list_analyses(video_id):
            a.pop("output_dir", None)
            out.append(a)
        return out

    def delete_analysis(self, aid: int) -> None:
        a = self.db.get_analysis(aid)
        if not a:
            raise NotFound(f"analysis {aid} not found")
        if a["status"] in ("queued", "running"):
            raise BadRequest("Cancel the analysis first.")
        if a.get("output_dir"):
            shutil.rmtree(a["output_dir"], ignore_errors=True)
        self.db.execute("DELETE FROM analyses WHERE id=?", (aid,))

    def segments(self, aid: int) -> list[dict]:
        self.get_analysis(aid)
        return self.db.list_segments(aid)

    def summary(self, aid: int) -> dict:
        a = self.get_analysis(aid)
        events = self.db.list_events(aid)
        s = reports.summarise(events, self.db.list_segments(aid))
        s["analysis"] = a
        return s

    # ------------------------------------------------------------------
    # Events and review
    # ------------------------------------------------------------------
    def list_events(self, analysis_id: Optional[int] = None, outcome: Optional[str] = None,
                    review_status: Optional[str] = None) -> list[dict]:
        return [self._public_event(e) for e in self.db.list_events(analysis_id, outcome, review_status)]

    def _public_event(self, e: dict) -> dict:
        e = dict(e)
        e.pop("snapshot_path", None)
        e.pop("clip_path", None)
        return e

    def get_event(self, eid: int) -> dict:
        e = self.db.get_event(eid)
        if not e:
            raise NotFound(f"event {eid} not found")
        return self._public_event(e)

    def event_file(self, eid: int, kind: str) -> Path:
        e = self.db.get_event(eid)
        if not e:
            raise NotFound(f"event {eid} not found")
        p = e.get("snapshot_path" if kind == "snapshot" else "clip_path")
        if not p or not Path(p).exists():
            raise NotFound(f"no {kind} for event {eid}")
        return Path(p)

    def review_event(self, eid: int, status: str, reviewer: str = "", note: str = "") -> dict:
        self.get_event(eid)
        if status not in REVIEW_STATUSES:
            raise BadRequest(f"status must be one of {REVIEW_STATUSES}")
        self.db.review_event(eid, status, reviewer.strip()[:100], note.strip()[:2000])
        return self.get_event(eid)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def export_csv(self, analysis_id: Optional[int] = None, outcome: Optional[str] = None) -> str:
        if analysis_id is not None:
            self.get_analysis(analysis_id)
        return reports.events_csv(self.db.list_events(analysis_id, outcome))

    def _render_context(self, aid: int):
        a = self.db.get_analysis(aid)
        if not a:
            raise NotFound(f"analysis {aid} not found")
        if a["status"] != "done":
            raise BadRequest("Analysis has not finished.")
        out_dir = Path(a["output_dir"])
        result = json.loads((out_dir / "result.json").read_text())
        cfg = AnalysisConfig.from_dict(a["config"])
        timeline = load_timeline_from_result(result, cfg)
        tracks = TrackStore.load(out_dir / "tracks.npz")
        video = self._video_path(a["video_id"])
        return a, cfg, timeline, tracks, video, out_dir

    @staticmethod
    def _crossing_from_event(e: dict) -> Crossing:
        return Crossing(track_id=e["track_id"], vehicle_class=e["vehicle_class"],
                        crossing_time_s=e["crossing_time_s"], crossing_frame=e["crossing_frame"],
                        prev_frame=e["crossing_frame"], anchor=(0, 0), det_conf=e["det_conf"] or 0,
                        outcome=e["outcome"], signal_state=e["signal_state"])

    def export_pdf(self, aid: int, include_snapshots: bool = False, anonymise: Optional[str] = None,
                   outcome: Optional[str] = "potential_violation") -> bytes:
        a = self.get_analysis(aid)
        events = self.db.list_events(aid, outcome)
        summary = reports.summarise(self.db.list_events(aid), self.db.list_segments(aid))
        snapshots: dict[int, str] = {}
        if include_snapshots and events:
            if anonymise:
                if anonymise not in ANON_MODES:
                    raise BadRequest(f"anonymise must be one of {ANON_MODES}")
                a2, cfg, timeline, tracks, video, out_dir = self._render_context(aid)
                tmp = out_dir / "anonymised"
                tmp.mkdir(exist_ok=True)
                for e in events:
                    snapshots[e["id"]] = render_snapshot(video, self._crossing_from_event(e), tracks, timeline,
                                                         cfg, tmp / f"event_{e['id']}_{anonymise}.jpg", anonymise)
            else:
                snapshots = {e["id"]: e["snapshot_path"] for e in events
                             if e.get("snapshot_path") and Path(e["snapshot_path"]).exists()}
        return reports.events_pdf(a, events, summary, snapshots, anonymised=bool(anonymise))

    def export_evidence_zip(self, aid: int, anonymise: Optional[str] = "standard",
                            review_status: Optional[str] = None,
                            outcome: Optional[str] = "potential_violation") -> Path:
        """Zip of snapshots + clips (+ CSV). With ``anonymise`` set, faces,
        plate regions and drivers are blurred before overlays are drawn."""
        if anonymise is not None and anonymise not in ANON_MODES:
            raise BadRequest(f"anonymise must be one of {ANON_MODES} or empty")
        a, cfg, timeline, tracks, video, out_dir = self._render_context(aid)
        events = self.db.list_events(aid, outcome, review_status)
        tag = anonymise or "original"
        stamp = now_iso().replace(":", "").replace("-", "")[:15]
        zpath = self.data / "exports" / f"analysis{aid}_evidence_{tag}_{stamp}.zip"
        work = out_dir / f"export_{tag}"
        work.mkdir(exist_ok=True)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for e in events:
                stem = f"event{e['id']}_track{e['track_id']}_{tag}"
                if anonymise:
                    ev = render_event_evidence(video, self._crossing_from_event(e), tracks, timeline, cfg,
                                               work, stem, anonymise=anonymise)
                    snap, clip = ev["snapshot"], ev["clip"]
                else:
                    snap, clip = e.get("snapshot_path"), e.get("clip_path")
                if snap and Path(snap).exists():
                    z.write(snap, f"{stem}.jpg")
                if clip and Path(clip).exists():
                    z.write(clip, f"{stem}.mp4")
            z.writestr("events.csv", reports.events_csv(events))
            z.writestr("README.txt", (
                "Potential red-light violations exported for human review.\n"
                f"Analysis {aid}, video {a['video_filename']}"
                f" ({'Cape Coast footage' if a['video_is_cape_coast'] else 'NOT Cape Coast footage'}).\n"
                + (f"Anonymisation: {anonymise} (automatic; check every file before publishing).\n"
                   if anonymise else "NOT anonymised: do not share or publish.\n")))
        shutil.rmtree(work, ignore_errors=True)
        return zpath

    # ------------------------------------------------------------------
    # Supplementary accident-footage demonstration
    # ------------------------------------------------------------------
    def accident_sequence(self, vid: int, collision_t: float, before_s: float = 3.0, after_s: float = 3.0,
                          anonymise: Optional[str] = None, model: Optional[str] = None) -> dict:
        path = self._video_path(vid)
        if anonymise is not None and anonymise not in ANON_MODES:
            raise BadRequest(f"anonymise must be one of {ANON_MODES} or empty")
        if not (0 < before_s <= 30 and 0 < after_s <= 30):
            raise BadRequest("before/after must be between 0 and 30 seconds")
        tracker = None
        if model:
            cfg = AnalysisConfig.from_dict(self.get_config(vid)).detector
            cfg.model = model
            tracker = make_tracker(cfg, path)
        out = self.data / "accident" / f"video{vid}_{uuid.uuid4().hex[:6]}"
        try:
            res = make_sequence(path, collision_t, out, before_s, after_s, tracker=tracker, anonymise=anonymise)
        except ValueError as exc:
            raise BadRequest(str(exc)) from exc
        res["contact_sheet_url"] = self.file_url(res["contact_sheet"])
        res["clip_url"] = self.file_url(res["clip"]) if res["clip"] else None
        res["video_id"] = vid
        res["anonymised"] = anonymise
        return res

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------
    def file_url(self, path: str | Path) -> str:
        rel = Path(path).resolve().relative_to(self.data)
        return f"/api/files/{rel.as_posix()}"

    def resolve_file(self, rel: str) -> Path:
        p = (self.data / rel).resolve()
        allowed = [self.data / "accident", self.data / "exports"]
        if not any(base in p.parents for base in allowed) or not p.is_file():
            raise NotFound("file not found")
        return p
