"""SQLite storage. One file, created on first use."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    location TEXT NOT NULL DEFAULT '',
    is_cape_coast INTEGER NOT NULL DEFAULT 0,
    session TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    fps REAL, frames INTEGER, width INTEGER, height INTEGER, duration_s REAL,
    config_json TEXT
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    status TEXT NOT NULL,             -- queued, running, done, failed, cancelled
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT, finished_at TEXT,
    output_dir TEXT,
    error TEXT,
    stats_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    vehicle_class TEXT NOT NULL,
    outcome TEXT NOT NULL,            -- potential_violation, signal_uncertain, lawful
    crossing_time_s REAL NOT NULL,
    crossing_frame INTEGER NOT NULL,
    signal_state TEXT,
    red_onset_s REAL,
    red_onset_exact INTEGER,
    time_since_red_s REAL,
    det_conf REAL,
    snapshot_path TEXT,
    clip_path TEXT,
    data_json TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending',  -- pending, confirmed, rejected, unsure
    reviewer TEXT NOT NULL DEFAULT '',
    review_note TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT,
    anon_checked INTEGER NOT NULL DEFAULT 0,       -- a person checked the anonymised image
    anon_checked_by TEXT NOT NULL DEFAULT '',
    anon_checked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_analysis ON events(analysis_id);
CREATE TABLE IF NOT EXISTS signal_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    state TEXT NOT NULL,
    start_s REAL NOT NULL, end_s REAL NOT NULL,
    start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL,
    onset_exact INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_analysis ON signal_segments(analysis_id);
"""

REVIEW_STATUSES = ("pending", "confirmed", "rejected", "unsure")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.connect() as con:
            con.executescript(SCHEMA)
            # Upgrade databases created before the anonymisation check existed.
            cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
            for col, ddl in (("anon_checked", "INTEGER NOT NULL DEFAULT 0"),
                             ("anon_checked_by", "TEXT NOT NULL DEFAULT ''"), ("anon_checked_at", "TEXT")):
                if col not in cols:
                    con.execute(f"ALTER TABLE events ADD COLUMN {col} {ddl}")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            con = sqlite3.connect(self.path, timeout=30)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            try:
                yield con
                con.commit()
            finally:
                con.close()

    # ---- generic helpers -------------------------------------------------
    def one(self, sql: str, args: tuple = ()) -> Optional[dict]:
        with self.connect() as con:
            row = con.execute(sql, args).fetchone()
            return dict(row) if row else None

    def all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, args).fetchall()]

    def execute(self, sql: str, args: tuple = ()) -> int:
        with self.connect() as con:
            cur = con.execute(sql, args)
            return cur.lastrowid

    # ---- videos ----------------------------------------------------------
    def add_video(self, **v: Any) -> int:
        return self.execute(
            "INSERT INTO videos (filename, path, uploaded_at, location, is_cape_coast, session, notes,"
            " fps, frames, width, height, duration_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (v["filename"], v["path"], now_iso(), v.get("location", ""), int(bool(v.get("is_cape_coast"))),
             v.get("session", ""), v.get("notes", ""), v["fps"], v["frames"], v["width"], v["height"],
             v["duration_s"]))

    def get_video(self, vid: int) -> Optional[dict]:
        return _decode_video(self.one("SELECT * FROM videos WHERE id=?", (vid,)))

    def list_videos(self) -> list[dict]:
        return [_decode_video(r) for r in self.all("SELECT * FROM videos ORDER BY id DESC")]

    def set_video_config(self, vid: int, cfg: dict) -> None:
        self.execute("UPDATE videos SET config_json=? WHERE id=?", (json.dumps(cfg), vid))

    def update_video_meta(self, vid: int, **fields: Any) -> None:
        allowed = {"location", "is_cape_coast", "session", "notes"}
        sets = [(k, int(bool(v)) if k == "is_cape_coast" else v) for k, v in fields.items() if k in allowed]
        if sets:
            self.execute(f"UPDATE videos SET {', '.join(k + '=?' for k, _ in sets)} WHERE id=?",
                         tuple(v for _, v in sets) + (vid,))

    def delete_video(self, vid: int) -> None:
        self.execute("DELETE FROM videos WHERE id=?", (vid,))

    # ---- analyses --------------------------------------------------------
    def add_analysis(self, video_id: int, model: str, cfg: dict) -> int:
        return self.execute(
            "INSERT INTO analyses (video_id, status, model, config_json, created_at) VALUES (?,?,?,?,?)",
            (video_id, "queued", model, json.dumps(cfg), now_iso()))

    def update_analysis(self, aid: int, **fields: Any) -> None:
        if "stats" in fields:
            fields["stats_json"] = json.dumps(fields.pop("stats"))
        keys = list(fields)
        self.execute(f"UPDATE analyses SET {', '.join(k + '=?' for k in keys)} WHERE id=?",
                     tuple(fields[k] for k in keys) + (aid,))

    def get_analysis(self, aid: int) -> Optional[dict]:
        return _decode_analysis(self.one(
            "SELECT a.*, v.filename AS video_filename, v.location AS video_location,"
            " v.is_cape_coast AS video_is_cape_coast FROM analyses a JOIN videos v ON v.id=a.video_id"
            " WHERE a.id=?", (aid,)))

    def list_analyses(self, video_id: Optional[int] = None) -> list[dict]:
        sql = ("SELECT a.*, v.filename AS video_filename, v.location AS video_location,"
               " v.is_cape_coast AS video_is_cape_coast FROM analyses a JOIN videos v ON v.id=a.video_id")
        args: tuple = ()
        if video_id is not None:
            sql += " WHERE a.video_id=?"
            args = (video_id,)
        return [_decode_analysis(r) for r in self.all(sql + " ORDER BY a.id DESC", args)]

    # ---- results ---------------------------------------------------------
    def store_results(self, aid: int, video_id: int, result: dict) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM events WHERE analysis_id=?", (aid,))
            con.execute("DELETE FROM signal_segments WHERE analysis_id=?", (aid,))
            for s in result["signal_segments"]:
                con.execute(
                    "INSERT INTO signal_segments (analysis_id, state, start_s, end_s, start_frame, end_frame,"
                    " onset_exact) VALUES (?,?,?,?,?,?,?)",
                    (aid, s["state"], s["start_s"], s["end_s"], s["start_frame"], s["end_frame"],
                     int(s["onset_exact"])))
            for c in result["crossings"]:
                ev = c.get("evidence") or {}
                con.execute(
                    "INSERT INTO events (analysis_id, video_id, track_id, vehicle_class, outcome, crossing_time_s,"
                    " crossing_frame, signal_state, red_onset_s, red_onset_exact, time_since_red_s, det_conf,"
                    " snapshot_path, clip_path, data_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (aid, video_id, c["track_id"], c["vehicle_class"], c["outcome"], c["crossing_time_s"],
                     c["crossing_frame"], c["signal_state"], c["red_onset_s"],
                     None if c["red_onset_exact"] is None else int(c["red_onset_exact"]),
                     c["time_since_red_s"], c["det_conf"], ev.get("snapshot"), ev.get("clip"), json.dumps(c)))

    def list_events(self, analysis_id: Optional[int] = None, outcome: Optional[str] = None,
                    review_status: Optional[str] = None) -> list[dict]:
        sql = ("SELECT e.*, v.filename AS video_filename, v.location AS video_location,"
               " v.is_cape_coast AS video_is_cape_coast FROM events e JOIN videos v ON v.id=e.video_id WHERE 1=1")
        args: list = []
        if analysis_id is not None:
            sql += " AND e.analysis_id=?"
            args.append(analysis_id)
        if outcome:
            sql += " AND e.outcome=?"
            args.append(outcome)
        if review_status:
            sql += " AND e.review_status=?"
            args.append(review_status)
        return [_decode_event(r) for r in self.all(sql + " ORDER BY e.analysis_id, e.crossing_time_s", tuple(args))]

    def get_event(self, eid: int) -> Optional[dict]:
        return _decode_event(self.one(
            "SELECT e.*, v.filename AS video_filename, v.location AS video_location,"
            " v.is_cape_coast AS video_is_cape_coast FROM events e JOIN videos v ON v.id=e.video_id"
            " WHERE e.id=?", (eid,)))

    def review_event(self, eid: int, status: str, reviewer: str, note: str) -> None:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"review status must be one of {REVIEW_STATUSES}")
        self.execute("UPDATE events SET review_status=?, reviewer=?, review_note=?, reviewed_at=? WHERE id=?",
                     (status, reviewer, note, now_iso() if status != "pending" else None, eid))

    def set_anon_checked(self, eid: int, checked: bool, by: str) -> None:
        self.execute("UPDATE events SET anon_checked=?, anon_checked_by=?, anon_checked_at=? WHERE id=?",
                     (int(checked), by if checked else "", now_iso() if checked else None, eid))

    def list_segments(self, aid: int) -> list[dict]:
        rows = self.all("SELECT state, start_s, end_s, start_frame, end_frame, onset_exact FROM signal_segments"
                        " WHERE analysis_id=? ORDER BY start_frame", (aid,))
        for r in rows:
            r["onset_exact"] = bool(r["onset_exact"])
        return rows


def _decode_video(r: Optional[dict]) -> Optional[dict]:
    if r is None:
        return None
    r["is_cape_coast"] = bool(r["is_cape_coast"])
    r["config"] = json.loads(r.pop("config_json")) if r.get("config_json") else None
    return r


def _decode_analysis(r: Optional[dict]) -> Optional[dict]:
    if r is None:
        return None
    r["config"] = json.loads(r.pop("config_json"))
    r["stats"] = json.loads(r.pop("stats_json")) if r.get("stats_json") else None
    if "video_is_cape_coast" in r:
        r["video_is_cape_coast"] = bool(r["video_is_cape_coast"])
    return r


def _decode_event(r: Optional[dict]) -> Optional[dict]:
    if r is None:
        return None
    r.pop("data_json", None)
    if r.get("red_onset_exact") is not None:
        r["red_onset_exact"] = bool(r["red_onset_exact"])
    if "video_is_cape_coast" in r:
        r["video_is_cape_coast"] = bool(r["video_is_cape_coast"])
    if "anon_checked" in r:
        r["anon_checked"] = bool(r["anon_checked"])
    r["has_snapshot"] = bool(r.get("snapshot_path"))
    r["has_clip"] = bool(r.get("clip_path"))
    return r
