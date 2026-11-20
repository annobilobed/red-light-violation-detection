import { useState } from "react";
import { api } from "../api.js";
import { AnalysisPicker, Empty, ErrorBox, NotCapeCoastBanner, SignalTimeline, SourceBadge, useFetch } from "../components/common.jsx";

function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="muted small">{sub}</div>}
    </div>
  );
}

export default function Reports({ analyses, analysisId, setAnalysisId }) {
  const summary = useFetch(analysisId ? `/api/analyses/${analysisId}/summary` : null);
  const segments = useFetch(analysisId ? `/api/analyses/${analysisId}/signal-segments` : null);
  const events = useFetch(analysisId ? `/api/events?analysis_id=${analysisId}` : null);
  const [pdf, setPdf] = useState("none");
  const [zip, setZip] = useState({ anonymise: "standard", review_status: "", only_anon_checked: true });
  const [busy, setBusy] = useState(false);
  const [link, setLink] = useState(null);
  const [error, setError] = useState(null);

  const s = summary.data;
  const a = s?.analysis;

  const pdfUrl = () => {
    const p = new URLSearchParams();
    if (pdf !== "none") {
      p.set("snapshots", "true");
      if (pdf !== "original") p.set("anonymise", pdf);
    }
    return `/api/analyses/${analysisId}/export.pdf?${p}`;
  };

  const exportZip = async () => {
    if (!zip.anonymise && !confirm("Export WITHOUT anonymisation? Faces and number plates will be visible. Do not share or publish these files.")) return;
    setBusy(true);
    setError(null);
    setLink(null);
    try {
      setLink(await api.post(`/api/analyses/${analysisId}/export-evidence`, { anonymise: zip.anonymise || null, review_status: zip.review_status || null, only_anon_checked: !!zip.anonymise && zip.only_anon_checked }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="toolbar wrap">
        <AnalysisPicker analyses={analyses} value={analysisId} onChange={setAnalysisId} />
        {a && <SourceBadge isCapeCoast={a.video_is_cape_coast} />}
      </div>
      <ErrorBox error={error || summary.error} />
      {!analysisId ? (
        <Empty>Choose a finished analysis.</Empty>
      ) : s ? (
        <>
          <NotCapeCoastBanner isCapeCoast={a.video_is_cape_coast} />
          <div className="stats">
            <Stat label="Stop-line crossings" value={s.crossings} />
            <Stat label="Potential red-light violations" value={s.potential_violations} sub="automated alerts, not violations" />
            <Stat label="Crossings while signal unreadable" value={s.by_outcome.signal_uncertain || 0} sub="never flagged" />
            <Stat label="Confirmed by reviewers" value={s.review_status.confirmed || 0} sub={`${s.review_status.pending || 0} still pending`} />
            <Stat
              label="Reviewed alerts confirmed"
              value={s.reviewer_confirmed_share === null ? "–" : `${(100 * s.reviewer_confirmed_share).toFixed(0)}%`}
              sub="reviewer decisions, not system accuracy"
            />
          </div>
          <SignalTimeline segments={segments.data} events={(events.data || []).filter((e) => e.outcome !== "lawful")} />
          <div className="grid2 cards">
            <section className="card">
              <h3>Breakdown</h3>
              <table className="facts">
                <tbody>
                  {Object.entries(s.potential_violations_by_class).map(([k, v]) => (
                    <tr key={k}>
                      <th>Potential violations: {k}</th>
                      <td>{v}</td>
                    </tr>
                  ))}
                  {s.time_since_red_s && (
                    <tr>
                      <th>Crossed after red (min / median / max)</th>
                      <td className="mono">
                        {s.time_since_red_s.min.toFixed(2)} / {s.time_since_red_s.median.toFixed(2)} / {s.time_since_red_s.max.toFixed(2)} s
                      </td>
                    </tr>
                  )}
                  {s.signal_time_share &&
                    Object.entries(s.signal_time_share).map(([k, v]) => (
                      <tr key={k}>
                        <th>Signal time: {k}</th>
                        <td>{(100 * v).toFixed(1)}%</td>
                      </tr>
                    ))}
                  <tr>
                    <th>Detector / tracker</th>
                    <td className="mono">{a.stats?.tracker}</td>
                  </tr>
                  <tr>
                    <th>Processing speed</th>
                    <td className="mono">
                      {a.stats?.timing?.pipeline_fps} frames/s ({a.stats?.timing?.tracker_ms_per_frame} ms/frame detector+tracker)
                    </td>
                  </tr>
                </tbody>
              </table>
              <p className="muted small">
                These are counts on one video. Detection, tracking, signal-reading and end-to-end accuracy are measured separately against annotated ground truth
                with the scripts in <code>evaluation/</code>.
              </p>
            </section>
            <section className="card">
              <h3>Export</h3>
              <div className="export-row">
                <a className="button" href={`/api/analyses/${analysisId}/export.csv?outcome=potential_violation`}>
                  CSV: potential violations
                </a>
                <a className="button" href={`/api/analyses/${analysisId}/export.csv`}>
                  CSV: all crossings
                </a>
              </div>
              <div className="export-row">
                <label className="field">
                  <span>PDF images</span>
                  <select value={pdf} onChange={(e) => setPdf(e.target.value)}>
                    <option value="none">No images</option>
                    <option value="standard">Anonymised snapshots (blur faces, plates, drivers)</option>
                    <option value="strong">Anonymised snapshots (pixelate whole vehicles)</option>
                    <option value="original">Original snapshots (internal use only)</option>
                  </select>
                </label>
                <a className="button primary" href={pdfUrl()}>
                  PDF report
                </a>
              </div>
              <div className="export-row">
                <label className="field">
                  <span>Evidence package</span>
                  <select value={zip.anonymise} onChange={(e) => setZip({ ...zip, anonymise: e.target.value })}>
                    <option value="standard">Anonymised: blur faces, plates, drivers</option>
                    <option value="strong">Anonymised: pixelate whole vehicles</option>
                    <option value="">Not anonymised (internal use only)</option>
                  </select>
                </label>
                <label className="field">
                  <span>Include</span>
                  <select value={zip.review_status} onChange={(e) => setZip({ ...zip, review_status: e.target.value })}>
                    <option value="">All potential violations</option>
                    <option value="confirmed">Reviewer-confirmed only</option>
                  </select>
                </label>
                {zip.anonymise && (
                  <label className="check">
                    <input type="checkbox" checked={zip.only_anon_checked} onChange={(e) => setZip({ ...zip, only_anon_checked: e.target.checked })} />
                    Only events whose anonymised still a person signed off
                  </label>
                )}
                <button className="primary" disabled={busy} onClick={exportZip}>
                  {busy ? "Rendering…" : "Build ZIP"}
                </button>
              </div>
              {link && (
                <div className="alert ok">
                  Ready: <a href={link.url}>{link.filename}</a>
                </div>
              )}
              <p className="muted small">
                Automatic anonymisation can miss faces or plates. Check every image and clip before sharing or publishing.
              </p>
            </section>
          </div>
        </>
      ) : (
        <Empty>Loading…</Empty>
      )}
    </div>
  );
}
