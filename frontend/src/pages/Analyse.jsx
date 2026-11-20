import { useEffect, useState } from "react";
import { api } from "../api.js";
import { ErrorBox, Empty, Progress, VideoPicker, useInterval } from "../components/common.jsx";

export default function Analyse({ videos, videoId, setVideoId, analyses, reloadAnalyses, go }) {
  const video = videos.find((v) => v.id === videoId);
  const [models, setModels] = useState([]);
  const [trackers, setTrackers] = useState([]);
  const [opts, setOpts] = useState({ model: "", conf: "", frame_stride: "", device: "" });
  const [error, setError] = useState(null);
  const running = analyses.some((a) => a.status === "running" || a.status === "queued");

  useEffect(() => {
    api.get("/api/models").then(setModels).catch(() => {});
    api.get("/api/trackers").then(setTrackers).catch(() => {});
  }, []);

  useEffect(() => {
    if (!videoId) return;
    api
      .get(`/api/videos/${videoId}/config`)
      .then(({ config }) => setOpts({ model: config.detector.model, tracker: config.detector.tracker.replace(".yaml", ""), conf: config.detector.conf, frame_stride: config.frame_stride, device: config.detector.device || "" }))
      .catch(() => {});
  }, [videoId]);

  useInterval(reloadAnalyses, 1000, running);

  const start = async () => {
    setError(null);
    try {
      await api.post("/api/analyses", {
        video_id: videoId,
        model: opts.model || null,
        tracker: opts.model === "scripted" ? null : opts.tracker || null,
        conf: opts.conf === "" ? null : Number(opts.conf),
        frame_stride: opts.frame_stride === "" ? null : Number(opts.frame_stride),
        device: opts.device || null,
      });
      reloadAnalyses();
    } catch (e) {
      setError(e);
    }
  };

  const act = async (fn) => {
    setError(null);
    try {
      await fn();
      reloadAnalyses();
    } catch (e) {
      setError(e);
    }
  };

  const usable = models.filter((m) => m.id !== "scripted" || video?.has_scenario);
  const list = videoId ? analyses.filter((a) => a.video_id === videoId) : analyses;

  return (
    <div>
      <section className="card">
        <h3>Run an analysis</h3>
        <div className="toolbar wrap">
          <VideoPicker videos={videos} value={videoId} onChange={setVideoId} />
          <label className="field">
            <span>Detector</span>
            <select value={opts.model} onChange={(e) => setOpts({ ...opts, model: e.target.value })}>
              {usable.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Tracker</span>
            <select value={opts.tracker || "bytetrack"} disabled={opts.model === "scripted"} onChange={(e) => setOpts({ ...opts, tracker: e.target.value })}>
              {trackers.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field small">
            <span>Confidence</span>
            <input type="number" step={0.05} min={0.01} max={0.99} value={opts.conf} onChange={(e) => setOpts({ ...opts, conf: e.target.value })} />
          </label>
          <label className="field small">
            <span>Frame stride</span>
            <input type="number" min={1} max={30} value={opts.frame_stride} onChange={(e) => setOpts({ ...opts, frame_stride: e.target.value })} />
          </label>
          <label className="field small">
            <span>Device</span>
            <input placeholder="auto / cpu / 0" value={opts.device} onChange={(e) => setOpts({ ...opts, device: e.target.value })} />
          </label>
          <button className="primary" disabled={!videoId} onClick={start}>
            Start analysis
          </button>
        </div>
        <p className="muted small">
          The first run of a YOLO model downloads its weights. Analyses run one at a time in the background; you can leave this page.
        </p>
        <ErrorBox error={error} />
      </section>

      {list.length === 0 ? (
        <Empty>No analyses yet.</Empty>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>#</th>
              <th>Video</th>
              <th>Model</th>
              <th>Status</th>
              <th>Potential</th>
              <th>Unreadable</th>
              <th>Crossings</th>
              <th>Speed</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {list.map((a) => {
              const c = a.stats?.counts;
              return (
                <tr key={a.id}>
                  <td>{a.id}</td>
                  <td>{a.video_filename}</td>
                  <td className="mono">{a.model}</td>
                  <td>
                    <span className={`status ${a.status}`}>{a.status}</span>
                    {(a.status === "running" || a.status === "queued") && (
                      <>
                        <Progress value={a.progress} />
                        <span className="muted small">{a.message}</span>
                      </>
                    )}
                    {a.status === "failed" && <div className="alert error small">{a.error}</div>}
                  </td>
                  <td>{c ? c.potential_violation : "–"}</td>
                  <td>{c ? c.signal_uncertain : "–"}</td>
                  <td>{c ? c.potential_violation + c.signal_uncertain + c.lawful : "–"}</td>
                  <td className="mono small">
                    {a.stats?.timing ? `${a.stats.timing.pipeline_fps} fps · ${a.stats.timing.tracker_ms_per_frame} ms/frame detector` : "–"}
                  </td>
                  <td className="actions">
                    {a.status === "done" && <button onClick={() => go("review", null, a.id)}>Review</button>}
                    {a.status === "done" && <button onClick={() => go("reports", null, a.id)}>Report</button>}
                    {(a.status === "running" || a.status === "queued") && <button onClick={() => act(() => api.post(`/api/analyses/${a.id}/cancel`))}>Cancel</button>}
                    {!(a.status === "running" || a.status === "queued") && (
                      <button className="danger" onClick={() => confirm(`Delete analysis #${a.id} and its evidence?`) && act(() => api.del(`/api/analyses/${a.id}`))}>
                        Delete
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
