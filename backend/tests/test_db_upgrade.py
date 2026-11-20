import sqlite3

from app.db import Database


def test_old_database_gets_new_columns(tmp_path):
    p = tmp_path / "app.db"
    con = sqlite3.connect(p)
    con.executescript("""
    CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id INTEGER, video_id INTEGER,
      track_id INTEGER, vehicle_class TEXT, outcome TEXT, crossing_time_s REAL, crossing_frame INTEGER,
      signal_state TEXT, red_onset_s REAL, red_onset_exact INTEGER, time_since_red_s REAL, det_conf REAL,
      snapshot_path TEXT, clip_path TEXT, data_json TEXT, review_status TEXT NOT NULL DEFAULT 'pending',
      reviewer TEXT NOT NULL DEFAULT '', review_note TEXT NOT NULL DEFAULT '', reviewed_at TEXT);
    """)
    con.close()
    Database(p)
    cols = {r[1] for r in sqlite3.connect(p).execute("PRAGMA table_info(events)")}
    assert {"anon_checked", "anon_checked_by", "anon_checked_at"} <= cols
