import { useState } from "react";
import { api, fmtTime } from "../api.js";

const COLOURS = ["red", "amber", "green"];
const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(100 * x)}%`);

function MetricRow({ label, m }) {
  return (
    <tr>
      <th>{label}</th>
      <td>{pct(m.coverage)}</td>
      <td className={m.wrong ? "bad" : ""}>{m.wrong}</td>
      <td className={m.safety_errors ? "bad" : ""}>{m.safety_errors}</td>
      {COLOURS.map((c) => (
        <td key={c} className="mono">
          {m.per_class[c].read_correctly}/{m.per_class[c].n}
        </td>
      ))}
    </tr>
  );
}

/**
 * Label frames as red/amber/green and search thresholds that read them.
 * Samples live in config.signal.samples (saved with the configuration).
 */
export default function Calibration({ videoId, t, setT, config, setConfig, setSaved }) {
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const samples = config.signal.samples || [];

  const setSamples = (fn) => {
    setSaved(false);
    setResult(null);
    setConfig((c) => ({ ...c, signal: { ...c.signal, samples: fn(c.signal.samples || []) } }));
  };

  const add = (state) =>
    setSamples((s) => [...s.filter((x) => Math.abs(x.t - t) > 0.01), { t: Number(t.toFixed(2)), state }].sort((a, b) => a.t - b.t));

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const { samples: _s, ...signal } = config.signal;
      setResult(await api.post(`/api/videos/${videoId}/signal-calibrate`, { signal, samples }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const applyRecommended = () => {
    setSaved(false);
    setConfig((c) => ({ ...c, signal: { ...c.signal, ...result.recommended.settings } }));
    setResult(null);
  };

  const counts = Object.fromEntries(COLOURS.map((c) => [c, samples.filter((s) => s.state === c).length]));

  return (
    <section className="card">
      <h3>Calibrate thresholds from labelled frames</h3>
      <p className="muted small">
        Scrub to a frame where you can see the signal clearly and label it. Label at least 5 frames of each colour, spread over the video
        (and over day, dusk and night if the footage covers them). Then let the system find thresholds that read your frames without ever
        confusing red with another colour.
      </p>
      <div className="toolbar">
        {COLOURS.map((c) => (
          <button key={c} className={`label-btn ${c}`} onClick={() => add(c)}>
            Frame at {fmtTime(t)} is {c}
          </button>
        ))}
      </div>
      <div className="muted small">
        {samples.length} labelled · red {counts.red} · amber {counts.amber} · green {counts.green}
      </div>
      {samples.length > 0 && (
        <div className="chips">
          {samples.map((s) => (
            <span key={s.t} className={`chip ${s.state}`}>
              <button className="link" onClick={() => setT(s.t)} title="Go to this frame">
                {fmtTime(s.t)}
              </button>
              <button className="link" title="Remove" onClick={() => setSamples((all) => all.filter((x) => x.t !== s.t))}>
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <button className="wide" disabled={busy || samples.length === 0} onClick={run}>
        {busy ? "Searching…" : "Find best thresholds"}
      </button>
      {error && <div className="alert error">{String(error.message || error)}</div>}
      {result && (
        <div className="calib-result">
          {result.warnings.map((w) => (
            <div key={w} className="alert warn small">
              {w}
            </div>
          ))}
          <div className={`alert ${result.recommended ? "ok" : "warn"} small`}>{result.message}</div>
          <table className="facts calib">
            <thead>
              <tr>
                <th />
                <th>Read</th>
                <th>Wrong</th>
                <th>Red/other</th>
                {COLOURS.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <MetricRow label="Current" m={result.current.metrics} />
              {result.recommended && <MetricRow label="Recommended" m={result.recommended.metrics} />}
            </tbody>
          </table>
          {result.recommended && (
            <>
              <div className="mono small muted">
                {Object.entries(result.recommended.settings)
                  .map(([k, v]) => `${k}=${v}`)
                  .join("  ")}
              </div>
              <button className="primary wide" onClick={applyRecommended}>
                Apply recommended thresholds
              </button>
            </>
          )}
          <p className="muted small">
            These numbers are measured on the frames you labelled, so they are optimistic. Check accuracy on other sessions with{" "}
            <code>evaluation/evaluate_signal.py</code>.
          </p>
        </div>
      )}
    </section>
  );
}
