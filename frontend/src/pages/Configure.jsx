import { useEffect, useState } from "react";
import { api, fmtTime } from "../api.js";
import { ErrorBox, Empty, SourceBadge, VideoPicker } from "../components/common.jsx";
import FrameCanvas from "../components/FrameCanvas.jsx";
import Calibration from "../components/Calibration.jsx";

const TOOLS = [
  { id: "line", label: "Stop line", help: "Click the two ends of the stop line." },
  { id: "approach", label: "Approach side", help: "Click anywhere on the side vehicles approach from (before the line)." },
  { id: "signal", label: "Signal head", help: "Drag a box tightly around the signal head that applies to this approach." },
  { id: "zone", label: "Approach zone", help: "Optional. Click the corners of the approach lanes; only vehicles that were inside count." },
];

function Num({ label, value, onChange, step = 1, min, max, help }) {
  return (
    <label className="field small" title={help}>
      <span>{label}</span>
      <input type="number" value={value} step={step} min={min} max={max} onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))} />
    </label>
  );
}

export default function Configure({ videos, videoId, setVideoId }) {
  const video = videos.find((v) => v.id === videoId);
  const [config, setConfig] = useState(null);
  const [t, setT] = useState(0);
  const [tool, setTool] = useState(null);
  const [reading, setReading] = useState(null);
  const [problems, setProblems] = useState(null);
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState(false);
  const [models, setModels] = useState([]);

  useEffect(() => {
    api.get("/api/models").then(setModels).catch(() => {});
  }, []);

  useEffect(() => {
    setConfig(null);
    setReading(null);
    setProblems(null);
    setT(0);
    if (!videoId) return;
    api
      .get(`/api/videos/${videoId}/config`)
      .then((r) => setConfig(r.config))
      .catch(setError);
  }, [videoId]);

  const upd = (section, key, value) => {
    setSaved(false);
    setConfig((c) => ({ ...c, [section]: { ...c[section], [key]: value } }));
  };

  const testSignal = async () => {
    setError(null);
    try {
      setReading(await api.post(`/api/videos/${videoId}/signal-test`, { t, signal: config.signal }));
    } catch (e) {
      setError(e);
    }
  };

  const save = async () => {
    setError(null);
    try {
      const r = await api.put(`/api/videos/${videoId}/config`, { config });
      setConfig(r.config);
      setProblems(r.problems);
      setSaved(true);
    } catch (e) {
      setError(e);
    }
  };

  if (!videos.length) return <Empty>Upload a video first.</Empty>;

  const toolHelp = TOOLS.find((x) => x.id === tool)?.help || (tool?.startsWith("lamp:") ? `Drag a box around the ${tool.split(":")[1]} lamp.` : null);
  const usableModels = models.filter((m) => m.id !== "scripted" || video?.has_scenario);

  return (
    <div>
      <div className="toolbar">
        <VideoPicker videos={videos} value={videoId} onChange={setVideoId} />
        {video && <SourceBadge isCapeCoast={video.is_cape_coast} />}
      </div>
      <ErrorBox error={error} />
      {video && config && (
        <div className="configure">
          <div className="configure-main">
            <div className="toolbar">
              {TOOLS.map((x) => (
                <button key={x.id} className={tool === x.id ? "active" : ""} onClick={() => setTool(tool === x.id ? null : x.id)}>
                  {x.label}
                </button>
              ))}
              {config.crossing.approach_zone?.length > 0 && (
                <button className="ghost" onClick={() => upd("crossing", "approach_zone", null)}>
                  Clear zone
                </button>
              )}
              {tool && (
                <button className="ghost" onClick={() => setTool(null)}>
                  Done
                </button>
              )}
            </div>
            {toolHelp && <div className="hint">{toolHelp}</div>}
            <FrameCanvas
              src={`/api/videos/${videoId}/frame?t=${t.toFixed(2)}`}
              width={video.width}
              height={video.height}
              config={config}
              tool={tool}
              reading={reading}
              onChange={(c) => {
                setSaved(false);
                setConfig(c);
              }}
            />
            <div className="scrub">
              <input type="range" min={0} max={Math.max(0, video.duration_s - 0.05)} step={0.1} value={t} onChange={(e) => setT(Number(e.target.value))} />
              <span className="mono">{fmtTime(t)}</span>
            </div>
            <p className="muted">
              Pick frames where the signal is red, amber and green in turn and press <b>Test signal reading</b> on each, to check the
              thresholds before running an analysis. Check daylight, dusk and night footage separately.
            </p>
          </div>

          <div className="configure-side">
            <section className="card">
              <h3>Signal reading</h3>
              <label className="field small">
                <span>Lamp layout</span>
                <select value={config.signal.layout} onChange={(e) => upd("signal", "layout", e.target.value)}>
                  <option value="vertical">Vertical (red on top)</option>
                  <option value="horizontal">Horizontal (red on left)</option>
                  <option value="custom">Custom lamp boxes</option>
                </select>
              </label>
              {config.signal.layout === "custom" && (
                <div className="toolbar">
                  {["red", "amber", "green"].map((c) => (
                    <button key={c} className={tool === `lamp:${c}` ? "active" : ""} onClick={() => setTool(`lamp:${c}`)}>
                      {c} lamp
                    </button>
                  ))}
                </div>
              )}
              <div className="grid2">
                <Num label="Saturation min" value={config.signal.sat_min} min={0} max={255} onChange={(v) => upd("signal", "sat_min", v)} help="HSV saturation (0–255) a lit pixel must reach" />
                <Num label="Brightness min" value={config.signal.val_min} min={0} max={255} onChange={(v) => upd("signal", "val_min", v)} help="HSV value (0–255) a lit pixel must reach" />
                <Num label="Min lamp score" value={config.signal.min_score} step={0.01} min={0} max={1} onChange={(v) => upd("signal", "min_score", v)} help="Fraction of lit pixels needed to call a lamp on" />
                <Num label="Margin" value={config.signal.margin} step={0.01} min={0} max={1} onChange={(v) => upd("signal", "margin", v)} help="How far the best lamp must beat the runner-up" />
                <Num label="Stable frames" value={config.signal.min_stable_frames} min={1} max={60} onChange={(v) => upd("signal", "min_stable_frames", v)} help="Shorter colour runs become uncertain" />
                <Num label="Bridge gap (s)" value={config.signal.bridge_gap_s} step={0.1} min={0} max={5} onChange={(v) => upd("signal", "bridge_gap_s", v)} help="Unreadable gaps this short between the SAME colour take that colour (LED flicker)" />
              </div>
              <details>
                <summary>Over-exposed lamps and safety guard</summary>
                <label className="check">
                  <input type="checkbox" checked={!!config.signal.core_agreement} onChange={(e) => upd("signal", "core_agreement", e.target.checked)} />
                  Read over-exposed lamps when bright core and colour agree
                </label>
                <div className="grid2">
                  <Num label="Core brightness" value={config.signal.core_val} min={0} max={255} onChange={(v) => upd("signal", "core_val", v)} help="V at or above this counts as very bright" />
                  <Num label="Core min share" value={config.signal.core_min} step={0.01} min={0} max={1} onChange={(v) => upd("signal", "core_min", v)} />
                  <Num label="Core margin" value={config.signal.core_margin} step={0.01} min={0} max={1} onChange={(v) => upd("signal", "core_margin", v)} />
                  <Num label="Weak colour min" value={config.signal.weak_min} step={0.005} min={0} max={1} onChange={(v) => upd("signal", "weak_min", v)} />
                  <Num label="Peak brightness min" value={config.signal.peak_min} min={0} max={255} onChange={(v) => upd("signal", "peak_min", v)} help="Safety guard: a lit lamp's brightest pixels must reach this (stops foliage or green vehicles being read as a green lamp). Not changed by calibration." />
                </div>
              </details>
              <label className="check">
                <input type="checkbox" checked={!!config.signal.brightness_fallback} onChange={(e) => upd("signal", "brightness_fallback", e.target.checked)} />
                Night fallback: use lamp brightness by position if colour is washed out
              </label>
              <button onClick={testSignal} disabled={!config.signal.roi[2] && config.signal.layout !== "custom"}>
                Test signal reading at {fmtTime(t)}
              </button>
              {reading && (
                <div className="reading">
                  <div>
                    Reading: <span className={`state ${reading.state}`}>{reading.state}</span>
                    {reading.method !== "colour" && reading.method !== "none" && <span className="muted"> (by {reading.method})</span>}
                  </div>
                  {["red", "amber", "green"].map((c) => (
                    <div key={c} className="scorebar">
                      <span>{c}</span>
                      <div>
                        <i className={c} style={{ width: `${Math.min(100, 100 * reading.scores[c])}%` }} />
                        <b style={{ left: `${Math.min(100, 100 * config.signal.min_score)}%` }} title="min lamp score" />
                      </div>
                      <span className="mono" title={`bright core ${reading.core[c].toFixed(3)}, peak V ${reading.peak[c]}`}>
                        {reading.scores[c].toFixed(3)}
                      </span>
                    </div>
                  ))}
                  <p className="muted small">{reading.note}</p>
                </div>
              )}
            </section>

            <Calibration videoId={videoId} t={t} setT={setT} config={config} setConfig={setConfig} setSaved={setSaved} />

            <section className="card">
              <h3>Stop-line crossing</h3>
              <div className="grid2">
                <Num label="Line margin (px)" value={config.crossing.line_margin_px} step={0.5} min={0} onChange={(v) => upd("crossing", "line_margin_px", v)} help="Hysteresis band around the stop line" />
                <Num label="Approach frames" value={config.crossing.min_approach_frames} min={1} onChange={(v) => upd("crossing", "min_approach_frames", v)} help="Times a vehicle must be seen before the line" />
                <Num label="Red grace (s)" value={config.crossing.red_grace_s} step={0.1} min={0} onChange={(v) => upd("crossing", "red_grace_s", v)} help="0 = any crossing strictly after red onset is flagged" />
                <Num label="Line end tolerance" value={config.crossing.segment_tolerance} step={0.01} min={0} max={1} onChange={(v) => upd("crossing", "segment_tolerance", v)} help="Allowed overshoot past the ends of the stop line (fraction of its length)" />
              </div>
            </section>

            <section className="card">
              <h3>Detector and evidence</h3>
              <label className="field small">
                <span>Default model</span>
                <select value={config.detector.model} onChange={(e) => upd("detector", "model", e.target.value)}>
                  {usableModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                  {!usableModels.some((m) => m.id === config.detector.model) && <option value={config.detector.model}>{config.detector.model}</option>}
                </select>
              </label>
              <div className="grid2">
                <Num label="Confidence" value={config.detector.conf} step={0.05} min={0.01} max={0.99} onChange={(v) => upd("detector", "conf", v)} />
                <Num label="Image size" value={config.detector.imgsz} step={32} min={160} max={1920} onChange={(v) => upd("detector", "imgsz", v)} />
                <Num label="Frame stride" value={config.frame_stride} min={1} max={30} onChange={(v) => { setSaved(false); setConfig((c) => ({ ...c, frame_stride: v })); }} help="Process every n-th frame (1 = all)" />
                <Num label="Clip before (s)" value={config.evidence.pre_s} step={0.5} min={0} onChange={(v) => upd("evidence", "pre_s", v)} />
                <Num label="Clip after (s)" value={config.evidence.post_s} step={0.5} min={0} onChange={(v) => upd("evidence", "post_s", v)} />
              </div>
            </section>

            <button className="primary wide" onClick={save}>
              Save configuration
            </button>
            {saved && problems && problems.length === 0 && <div className="alert ok">Saved. Ready to analyse.</div>}
            {problems && problems.length > 0 && (
              <div className="alert warn">
                Saved, but not ready:
                <ul>
                  {problems.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
