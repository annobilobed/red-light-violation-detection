import "./styles.css";
import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api.js";
import { useFetch, useInterval } from "./components/common.jsx";
import Videos from "./pages/Videos.jsx";
import Configure from "./pages/Configure.jsx";
import Analyse from "./pages/Analyse.jsx";
import Review from "./pages/Review.jsx";
import Reports from "./pages/Reports.jsx";
import Accident from "./pages/Accident.jsx";

const TABS = [
  ["videos", "1. Videos"],
  ["configure", "2. Configure"],
  ["analyse", "3. Analyse"],
  ["review", "4. Review"],
  ["reports", "5. Reports"],
  ["accident", "Accident footage (exploratory)"],
];

function readHash() {
  const [tab, q] = window.location.hash.replace(/^#\/?/, "").split("?");
  const p = new URLSearchParams(q || "");
  return {
    tab: TABS.some(([t]) => t === tab) ? tab : "videos",
    video: p.get("video") ? Number(p.get("video")) : null,
    analysis: p.get("analysis") ? Number(p.get("analysis")) : null,
  };
}

function App() {
  const initial = readHash();
  const [tab, setTab] = useState(initial.tab);
  const [videoId, setVideoId] = useState(initial.video);
  const [analysisId, setAnalysisId] = useState(initial.analysis);
  const videos = useFetch("/api/videos");
  const analyses = useFetch("/api/analyses");
  const health = useFetch("/api/health");

  useEffect(() => {
    const p = new URLSearchParams();
    if (videoId) p.set("video", videoId);
    if (analysisId) p.set("analysis", analysisId);
    const h = `#/${tab}${p.toString() ? "?" + p : ""}`;
    if (window.location.hash !== h) window.history.replaceState(null, "", h);
  }, [tab, videoId, analysisId]);

  useEffect(() => {
    const onHash = () => {
      const h = readHash();
      setTab(h.tab);
      if (h.video) setVideoId(h.video);
      if (h.analysis) setAnalysisId(h.analysis);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Keep analysis statuses fresh while anything is running.
  const running = (analyses.data || []).some((a) => a.status === "running" || a.status === "queued");
  useInterval(analyses.reload, 2000, running && tab !== "analyse");

  const go = (t, vid, aid) => {
    if (vid) setVideoId(vid);
    if (aid) setAnalysisId(aid);
    setTab(t);
    if (t === "analyse" || t === "review" || t === "reports") analyses.reload();
    window.scrollTo(0, 0);
  };

  const vids = videos.data || [];
  const ans = analyses.data || [];
  const props = { videos: vids, videoId, setVideoId, analyses: ans, analysisId, setAnalysisId, go };

  return (
    <>
      <header>
        <div className="brand">
          <b>Red-light violation detection</b>
          <span>Research prototype · Cape Coast case study</span>
        </div>
        <nav>
          {TABS.map(([t, label]) => (
            <button key={t} className={tab === t ? "active" : ""} onClick={() => go(t)}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <div className="notice">
        Automated alerts are <b>potential</b> red-light violations for human review, never confirmed violations. No number-plate recognition, driver
        identification or enforcement decisions.
        {health.data && !health.data.ffmpeg && <b> ffmpeg not found: evidence clips may not play in the browser.</b>}
      </div>
      <main>
        {videos.error && <div className="alert error">Cannot reach the backend: {String(videos.error.message)}</div>}
        {tab === "videos" && <Videos {...props} reload={videos.reload} />}
        {tab === "configure" && <Configure {...props} />}
        {tab === "analyse" && <Analyse {...props} reloadAnalyses={analyses.reload} />}
        {tab === "review" && <Review {...props} />}
        {tab === "reports" && <Reports {...props} />}
        {tab === "accident" && <Accident {...props} />}
      </main>
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
