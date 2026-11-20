import { useState } from "react";
import { api, fmtTime } from "../api.js";
import { Empty, ErrorBox, Progress, SourceBadge } from "../components/common.jsx";

export default function Videos({ videos, reload, go }) {
  const [file, setFile] = useState(null);
  const [form, setForm] = useState({ location: "", is_cape_coast: false, session: "", notes: "" });
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);

  const upload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const v = await api.upload(file, form, setProgress);
      setFile(null);
      e.target.reset();
      setForm({ location: "", is_cape_coast: false, session: "", notes: "" });
      reload();
      go("configure", v.id);
    } catch (err) {
      setError(err);
    } finally {
      setProgress(null);
    }
  };

  const toggleCape = async (v) => {
    try {
      await api.patch(`/api/videos/${v.id}`, { is_cape_coast: !v.is_cape_coast });
      reload();
    } catch (err) {
      setError(err);
    }
  };

  const remove = async (v) => {
    if (!confirm(`Delete video #${v.id} (${v.filename}) and all its analyses and evidence?`)) return;
    try {
      await api.del(`/api/videos/${v.id}`);
      reload();
    } catch (err) {
      setError(err);
    }
  };

  return (
    <div>
      <section className="card">
        <h3>Add recorded video</h3>
        <form className="upload" onSubmit={upload}>
          <label className="field">
            <span>Video file</span>
            <input type="file" accept="video/*,.mp4,.avi,.mov,.mkv" onChange={(e) => setFile(e.target.files[0] || null)} required />
          </label>
          <label className="field">
            <span>Location / junction</span>
            <input value={form.location} placeholder="e.g. Kotokuraba junction, approach from Pedu" onChange={(e) => setForm({ ...form, location: e.target.value })} />
          </label>
          <label className="field">
            <span>Session</span>
            <input value={form.session} placeholder="e.g. 2026-11-03 morning peak" onChange={(e) => setForm({ ...form, session: e.target.value })} />
          </label>
          <label className="field">
            <span>Notes</span>
            <input value={form.notes} placeholder="lighting, weather, camera position…" onChange={(e) => setForm({ ...form, notes: e.target.value })} />
          </label>
          <label className="check">
            <input type="checkbox" checked={form.is_cape_coast} onChange={(e) => setForm({ ...form, is_cape_coast: e.target.checked })} />
            This footage was recorded in Cape Coast
          </label>
          <button className="primary" disabled={!file || progress !== null}>
            Upload
          </button>
        </form>
        {progress !== null && <Progress value={progress} />}
        <ErrorBox error={error} />
        <p className="muted small">Recorded video only. Live CCTV, number-plate recognition and driver identification are out of scope for this prototype.</p>
      </section>

      {videos.length === 0 ? (
        <Empty>No videos yet.</Empty>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>#</th>
              <th>File</th>
              <th>Location</th>
              <th>Source</th>
              <th>Session</th>
              <th>Length</th>
              <th>Size</th>
              <th>Configured</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {videos.map((v) => (
              <tr key={v.id}>
                <td>{v.id}</td>
                <td>
                  {v.filename}
                  {v.has_scenario && <span className="badge info">synthetic test</span>}
                </td>
                <td>{v.location || <span className="muted">–</span>}</td>
                <td>
                  <button className="link" onClick={() => toggleCape(v)} title="Click to change">
                    <SourceBadge isCapeCoast={v.is_cape_coast} />
                  </button>
                </td>
                <td>{v.session || <span className="muted">–</span>}</td>
                <td className="mono">{fmtTime(v.duration_s)}</td>
                <td className="mono">
                  {v.width}×{v.height} @ {v.fps.toFixed(1)}
                </td>
                <td>{v.config ? "yes" : <span className="muted">no</span>}</td>
                <td className="actions">
                  <button onClick={() => go("configure", v.id)}>Configure</button>
                  <button onClick={() => go("analyse", v.id)}>Analyse</button>
                  <button className="danger" onClick={() => remove(v)}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
