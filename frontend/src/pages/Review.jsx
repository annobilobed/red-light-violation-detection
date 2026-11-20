import { useEffect, useMemo, useState } from "react";
import { api, fmtTime, OUTCOME_LABEL } from "../api.js";
import { AnalysisPicker, Empty, ErrorBox, NotCapeCoastBanner, SignalTimeline, useFetch } from "../components/common.jsx";

const STATUSES = [
  { id: "confirmed", label: "Confirm", key: "c" },
  { id: "rejected", label: "Reject", key: "r" },
  { id: "unsure", label: "Unsure", key: "u" },
  { id: "pending", label: "Reset", key: null },
];

export default function Review({ analyses, analysisId, setAnalysisId }) {
  const [outcome, setOutcome] = useState("potential_violation");
  const [status, setStatus] = useState("");
  const [selected, setSelected] = useState(null);
  const [reviewer, setReviewer] = useState(() => localStorage.getItem("reviewer") || "");
  const [note, setNote] = useState("");
  const [error, setError] = useState(null);

  const q = new URLSearchParams({ analysis_id: analysisId ?? "", outcome, review_status: status }).toString();
  const events = useFetch(analysisId ? `/api/events?${q}` : null);
  const segments = useFetch(analysisId ? `/api/analyses/${analysisId}/signal-segments` : null);
  const analysis = analyses.find((a) => a.id === analysisId);
  const list = events.data || [];
  const ev = list.find((e) => e.id === selected) || null;

  useEffect(() => {
    if (list.length && !list.some((e) => e.id === selected)) setSelected(list[0].id);
    if (!list.length) setSelected(null);
  }, [events.data]);

  useEffect(() => {
    setNote(ev?.review_note || "");
  }, [ev?.id]);

  useEffect(() => localStorage.setItem("reviewer", reviewer), [reviewer]);

  const idx = list.findIndex((e) => e.id === selected);
  const move = (d) => {
    const n = list[idx + d];
    if (n) setSelected(n.id);
  };

  const decide = async (s) => {
    if (!ev) return;
    setError(null);
    try {
      await api.patch(`/api/events/${ev.id}/review`, { status: s, reviewer, note });
      events.reload();
      if (s !== "pending") move(1);
    } catch (e) {
      setError(e);
    }
  };

  useEffect(() => {
    const onKey = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
      if (e.key === "j" || e.key === "ArrowDown") move(1);
      else if (e.key === "k" || e.key === "ArrowUp") move(-1);
      else {
        const s = STATUSES.find((x) => x.key === e.key);
        if (s) decide(s.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const counts = useMemo(() => {
    const c = {};
    list.forEach((e) => (c[e.review_status] = (c[e.review_status] || 0) + 1));
    return c;
  }, [list]);

  return (
    <div>
      <div className="toolbar wrap">
        <AnalysisPicker analyses={analyses} value={analysisId} onChange={setAnalysisId} />
        <label className="field">
          <span>Show</span>
          <select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
            <option value="potential_violation">Potential red-light violations</option>
            <option value="signal_uncertain">Crossings while signal unreadable</option>
            <option value="lawful">Crossings not flagged</option>
            <option value="">All crossings</option>
          </select>
        </label>
        <label className="field">
          <span>Review status</span>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">Any</option>
            <option value="pending">Pending</option>
            <option value="confirmed">Confirmed</option>
            <option value="rejected">Rejected</option>
            <option value="unsure">Unsure</option>
          </select>
        </label>
        <label className="field">
          <span>Reviewer</span>
          <input value={reviewer} placeholder="your initials" onChange={(e) => setReviewer(e.target.value)} />
        </label>
      </div>
      {analysis && <NotCapeCoastBanner isCapeCoast={analysis.video_is_cape_coast} />}
      <ErrorBox error={error || events.error} />
      {!analysisId ? (
        <Empty>Choose a finished analysis.</Empty>
      ) : (
        <>
          <SignalTimeline segments={segments.data} events={list.filter((e) => e.outcome !== "lawful")} cursor={ev?.crossing_time_s} />
          <div className="review">
            <div className="event-list">
              <div className="muted small">
                {list.length} shown · {counts.pending || 0} pending · {counts.confirmed || 0} confirmed · {counts.rejected || 0} rejected · {counts.unsure || 0} unsure
              </div>
              {list.length === 0 && <Empty>Nothing to show.</Empty>}
              {list.map((e) => (
                <button key={e.id} className={`event-item ${e.id === selected ? "selected" : ""}`} onClick={() => setSelected(e.id)}>
                  <span className="mono">{fmtTime(e.crossing_time_s)}</span>
                  <span>
                    #{e.track_id} {e.vehicle_class}
                  </span>
                  <span className={`review-tag ${e.review_status}`}>{e.review_status}</span>
                </button>
              ))}
            </div>
            <div className="event-detail">
              {ev ? (
                <>
                  <h3>
                    {OUTCOME_LABEL[ev.outcome]} <span className="muted">· event {ev.id}</span>
                  </h3>
                  {ev.has_clip ? (
                    <video key={ev.id} src={`/api/events/${ev.id}/clip`} controls autoPlay muted loop playsInline className="clip" />
                  ) : ev.has_snapshot ? (
                    <img src={`/api/events/${ev.id}/snapshot`} className="clip" alt="snapshot" />
                  ) : (
                    <Empty>No evidence is rendered for crossings that were not flagged.</Empty>
                  )}
                  {ev.has_clip && ev.has_snapshot && (
                    <details>
                      <summary>Still at the crossing frame</summary>
                      <img src={`/api/events/${ev.id}/snapshot`} className="clip" alt="snapshot" />
                    </details>
                  )}
                  <table className="facts">
                    <tbody>
                      <tr>
                        <th>Crossed stop line at</th>
                        <td className="mono">
                          {fmtTime(ev.crossing_time_s)} (frame {ev.crossing_frame})
                        </td>
                      </tr>
                      <tr>
                        <th>Signal at crossing</th>
                        <td>
                          <span className={`state ${ev.signal_state}`}>{ev.signal_state}</span>
                        </td>
                      </tr>
                      {ev.red_onset_s !== null && (
                        <tr>
                          <th>Red began</th>
                          <td className="mono">
                            {fmtTime(ev.red_onset_s)} {ev.red_onset_exact ? "" : "(or earlier: signal unreadable before this)"}
                          </td>
                        </tr>
                      )}
                      {ev.time_since_red_s !== null && (
                        <tr>
                          <th>Crossed after red by</th>
                          <td className="mono">{ev.time_since_red_s.toFixed(2)} s</td>
                        </tr>
                      )}
                      <tr>
                        <th>Vehicle</th>
                        <td>
                          track #{ev.track_id}, {ev.vehicle_class} (mean detection confidence {ev.det_conf?.toFixed(2)})
                        </td>
                      </tr>
                      <tr>
                        <th>Review</th>
                        <td>
                          <span className={`review-tag ${ev.review_status}`}>{ev.review_status}</span>
                          {ev.reviewer && ` by ${ev.reviewer}`} {ev.reviewed_at && <span className="muted">({ev.reviewed_at})</span>}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  <p className="muted small">
                    Check: did this vehicle cross the stop line after the signal for its approach turned red? Vehicles that entered on green or amber and are
                    still in the junction are lawful. Reject false detections, wrong signal readings and turning traffic governed by another signal.
                  </p>
                  <textarea placeholder="Reviewer note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
                  <div className="toolbar">
                    {STATUSES.map((s) => (
                      <button key={s.id} className={`review-btn ${s.id}`} onClick={() => decide(s.id)}>
                        {s.label}
                        {s.key && <kbd>{s.key}</kbd>}
                      </button>
                    ))}
                    <span className="muted small">
                      <kbd>j</kbd>/<kbd>k</kbd> next/previous
                    </span>
                  </div>
                </>
              ) : (
                <Empty>Select an event.</Empty>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
