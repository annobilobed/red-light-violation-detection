import { useEffect, useRef, useState } from "react";
import { api, fmtTime } from "../api.js";

export function useFetch(url, deps = []) {
  const [state, set] = useState({ data: null, error: null, loading: !!url });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (!url) {
      set({ data: null, error: null, loading: false });
      return;
    }
    let live = true;
    set((s) => ({ ...s, loading: true }));
    api
      .get(url)
      .then((data) => live && set({ data, error: null, loading: false }))
      .catch((error) => live && set({ data: null, error, loading: false }));
    return () => {
      live = false;
    };
  }, [url, nonce, ...deps]);
  return { ...state, reload: () => setNonce((n) => n + 1) };
}

export function useInterval(fn, ms, active = true) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => ref.current(), ms);
    return () => clearInterval(id);
  }, [ms, active]);
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="alert error">{String(error.message || error)}</div>;
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>;
}

export function VideoPicker({ videos, value, onChange, label = "Video" }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}>
        <option value="">Choose a video…</option>
        {videos.map((v) => (
          <option key={v.id} value={v.id}>
            #{v.id} {v.filename} {v.location ? `(${v.location})` : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

export function AnalysisPicker({ analyses, value, onChange }) {
  const done = analyses.filter((a) => a.status === "done");
  return (
    <label className="field">
      <span>Analysis</span>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}>
        <option value="">Choose a finished analysis…</option>
        {done.map((a) => (
          <option key={a.id} value={a.id}>
            #{a.id} · {a.video_filename} · {a.model} · {a.stats?.counts?.potential_violation ?? 0} potential
          </option>
        ))}
      </select>
    </label>
  );
}

export function SourceBadge({ isCapeCoast }) {
  return isCapeCoast ? (
    <span className="badge ok">Cape Coast footage</span>
  ) : (
    <span className="badge warn" title="Results from this footage must not be presented as Cape Coast results">
      Not Cape Coast footage
    </span>
  );
}

export function NotCapeCoastBanner({ isCapeCoast }) {
  if (isCapeCoast) return null;
  return (
    <div className="alert warn">
      This footage is <b>not</b> marked as Cape Coast footage. Do not present these results as Cape Coast results.
    </div>
  );
}

const SEG_COLOUR = { red: "#d93025", amber: "#f29900", green: "#1e8e3e", uncertain: "#9aa0a6" };

export function SignalTimeline({ segments, duration, events = [], cursor, onSeek }) {
  if (!segments?.length) return null;
  const total = duration || segments[segments.length - 1].end_s || 1;
  const pct = (t) => `${(100 * t) / total}%`;
  return (
    <div className="timeline" title="Smoothed signal state over time">
      <div
        className="timeline-bar"
        onClick={(e) => {
          if (!onSeek) return;
          const r = e.currentTarget.getBoundingClientRect();
          onSeek(((e.clientX - r.left) / r.width) * total);
        }}
      >
        {segments.map((s, i) => (
          <div
            key={i}
            className="seg"
            style={{ left: pct(s.start_s), width: pct(Math.max(s.end_s - s.start_s, total / 1000)), background: SEG_COLOUR[s.state] }}
            title={`${s.state}: ${fmtTime(s.start_s)} – ${fmtTime(s.end_s)}`}
          />
        ))}
        {events.map((e) => (
          <div
            key={e.id}
            className={`tick ${e.outcome}`}
            style={{ left: pct(e.crossing_time_s) }}
            title={`#${e.id} ${e.vehicle_class} at ${fmtTime(e.crossing_time_s)}`}
          />
        ))}
        {cursor !== undefined && cursor !== null && <div className="cursor" style={{ left: pct(cursor) }} />}
      </div>
      <div className="timeline-legend">
        {Object.entries(SEG_COLOUR).map(([k, c]) => (
          <span key={k}>
            <i style={{ background: c }} /> {k}
          </span>
        ))}
        <span>
          <i className="tick-legend potential_violation" /> potential violation
        </span>
        <span>
          <i className="tick-legend signal_uncertain" /> signal unreadable
        </span>
      </div>
    </div>
  );
}

export function Progress({ value }) {
  return (
    <div className="progress">
      <div style={{ width: `${Math.round(100 * (value || 0))}%` }} />
    </div>
  );
}
