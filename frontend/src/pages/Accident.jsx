import { useEffect, useState } from "react";
import { api, fmtTime } from "../api.js";
import { Empty, ErrorBox, VideoPicker } from "../components/common.jsx";

export default function Accident({ videos, videoId, setVideoId }) {
  const video = videos.find((v) => v.id === videoId);
  const [t, setT] = useState(0);
  const [mark, setMark] = useState(null);
  const [opts, setOpts] = useState({ before_s: 3, after_s: 3, anonymise: "standard", model: "" });
  const [models, setModels] = useState([]);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.get("/api/models").then((m) => setModels(m.filter((x) => x.id !== "scripted"))).catch(() => {});
  }, []);
  useEffect(() => {
    setT(0);
    setMark(null);
    setResult(null);
  }, [videoId]);

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(
        await api.post("/api/accident", {
          video_id: videoId,
          collision_t: mark,
          before_s: Number(opts.before_s),
          after_s: Number(opts.after_s),
          anonymise: opts.anonymise || null,
          model: opts.model || null,
        })
      );
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="alert warn">
        <b>Supplementary, exploratory demonstration.</b> This page shows the sequence of frames around a collision time that <i>you</i> mark. It does not
        detect collisions and makes no claim about cause, speed, fault or right of way. It is not part of the red-light violation evaluation.
      </div>
      <div className="toolbar">
        <VideoPicker videos={videos} value={videoId} onChange={setVideoId} label="Accident footage" />
      </div>
      <ErrorBox error={error} />
      {!video ? (
        <Empty>Upload the accident footage on the Videos page, then choose it here.</Empty>
      ) : (
        <div className="configure">
          <div className="configure-main">
            <div className="frame-wrap">
              <img src={`/api/videos/${videoId}/frame?t=${t.toFixed(2)}`} alt="frame" />
            </div>
            <div className="scrub">
              <input type="range" min={0} max={Math.max(0, video.duration_s - 0.05)} step={0.04} value={t} onChange={(e) => setT(Number(e.target.value))} />
              <span className="mono">{fmtTime(t)}</span>
              <button onClick={() => setMark(t)}>Mark approximate collision time</button>
            </div>
          </div>
          <div className="configure-side">
            <section className="card">
              <h3>Sequence</h3>
              <p>
                Marked time: <b className="mono">{mark === null ? "not set" : fmtTime(mark)}</b>
              </p>
              <div className="grid2">
                <label className="field small">
                  <span>Seconds before</span>
                  <input type="number" min={0.5} max={30} step={0.5} value={opts.before_s} onChange={(e) => setOpts({ ...opts, before_s: e.target.value })} />
                </label>
                <label className="field small">
                  <span>Seconds after</span>
                  <input type="number" min={0.5} max={30} step={0.5} value={opts.after_s} onChange={(e) => setOpts({ ...opts, after_s: e.target.value })} />
                </label>
              </div>
              <label className="field small">
                <span>Anonymise</span>
                <select value={opts.anonymise} onChange={(e) => setOpts({ ...opts, anonymise: e.target.value })}>
                  <option value="standard">Blur faces, plates, drivers</option>
                  <option value="strong">Pixelate whole vehicles</option>
                  <option value="">No (internal use only)</option>
                </select>
              </label>
              <label className="field small">
                <span>Draw vehicle tracks</span>
                <select value={opts.model} onChange={(e) => setOpts({ ...opts, model: e.target.value })}>
                  <option value="">No tracks</option>
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </label>
              <button className="primary wide" disabled={mark === null || busy} onClick={run}>
                {busy ? "Rendering…" : "Build sequence"}
              </button>
            </section>
          </div>
        </div>
      )}
      {result && (
        <section className="card">
          <h3>Exploratory sequence</h3>
          <p className="muted small">{result.disclaimer}</p>
          <img src={result.contact_sheet_url} className="clip" alt="contact sheet" />
          {result.clip_url && <video src={result.clip_url} controls className="clip" />}
          <div className="export-row">
            <a className="button" href={result.contact_sheet_url} download>
              Download contact sheet
            </a>
            {result.clip_url && (
              <a className="button" href={result.clip_url} download>
                Download clip
              </a>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
