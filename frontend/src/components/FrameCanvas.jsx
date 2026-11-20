import { useEffect, useRef, useState } from "react";

// Lamp rectangles, mirroring backend lamp_rects().
export function lampRects(signal) {
  if (signal.layout === "custom") return signal.lamps || {};
  const [x, y, w, h] = signal.roi;
  const out = {};
  ["red", "amber", "green"].forEach((c, i) => {
    if (signal.layout === "horizontal") {
      const x0 = x + Math.floor((w * i) / 3);
      const x1 = x + Math.floor((w * (i + 1)) / 3);
      out[c] = [x0, y, x1 - x0, h];
    } else {
      const y0 = y + Math.floor((h * i) / 3);
      const y1 = y + Math.floor((h * (i + 1)) / 3);
      out[c] = [x, y0, w, y1 - y0];
    }
  });
  return out;
}

const LAMP_STROKE = { red: "#ff5252", amber: "#ffb300", green: "#4caf50" };

function normRect(a, b) {
  const x = Math.round(Math.min(a[0], b[0]));
  const y = Math.round(Math.min(a[1], b[1]));
  return [x, y, Math.round(Math.abs(a[0] - b[0])), Math.round(Math.abs(a[1] - b[1]))];
}

/**
 * Frame image with an SVG overlay in video pixel coordinates.
 * onChange receives a state-updater function (prev => next).
 * tool: null | "line" | "approach" | "signal" | "zone" | "lamp:red" | "lamp:amber" | "lamp:green"
 */
export default function FrameCanvas({ src, width, height, config, tool, onChange, reading }) {
  const svg = useRef(null);
  const [pending, setPending] = useState(null); // first point of a line / rect start
  const [hover, setHover] = useState(null);
  const [shown, setShown] = useState(0); // rendered width in CSS pixels

  useEffect(() => {
    if (!svg.current) return;
    const ro = new ResizeObserver(([e]) => setShown(e.contentRect.width));
    ro.observe(svg.current);
    return () => ro.disconnect();
  }, []);

  const toPx = (e) => {
    const r = svg.current.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * width, ((e.clientY - r.top) / r.height) * height];
  };

  // Functional update: rapid clicks must not overwrite each other.
  const set = (fn) =>
    onChange((prev) => {
      const next = structuredClone(prev);
      fn(next);
      return next;
    });

  const isRectTool = tool === "signal" || (tool && tool.startsWith("lamp:"));

  const onDown = (e) => {
    if (!tool) return;
    const p = toPx(e);
    if (isRectTool) {
      setPending(p);
      return;
    }
    if (tool === "line") {
      if (!pending) setPending(p);
      else {
        set((c) => (c.crossing.stop_line = [pending.map(Math.round), p.map(Math.round)]));
        setPending(null);
      }
    } else if (tool === "approach") {
      set((c) => (c.crossing.approach_point = p.map(Math.round)));
    } else if (tool === "zone") {
      set((c) => (c.crossing.approach_zone = [...(c.crossing.approach_zone || []), p.map(Math.round)]));
    }
  };

  const onUp = (e) => {
    if (!isRectTool || !pending) return;
    const rect = normRect(pending, toPx(e));
    setPending(null);
    if (rect[2] < 3 || rect[3] < 3) return;
    if (tool === "signal") set((c) => (c.signal.roi = rect));
    else {
      const colour = tool.split(":")[1];
      set((c) => {
        c.signal.lamps = { ...(c.signal.lamps || {}), [colour]: rect };
        c.signal.layout = "custom";
      });
    }
  };

  const { crossing, signal } = config;
  const [[x1, y1], [x2, y2]] = crossing.stop_line;
  const lamps = lampRects(signal);
  // Sizes are specified in screen pixels and converted to video pixels, so
  // overlays look the same for small and large videos.
  const k = shown ? width / shown : 1;
  const sw = 1.5 * k;
  const fs = 13 * k;

  // Arrow showing the direction of travel: from the approach side across the line.
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  let nx = -(y2 - y1);
  let ny = x2 - x1;
  const nlen = Math.hypot(nx, ny) || 1;
  nx /= nlen;
  ny /= nlen;
  const [ax, ay] = crossing.approach_point;
  if ((ax - mx) * nx + (ay - my) * ny < 0) {
    nx = -nx;
    ny = -ny;
  }
  const arrowLen = 40 * k;

  return (
    <div className="frame-wrap">
      {src ? <img src={src} alt="video frame" draggable={false} /> : <div className="frame-placeholder" />}
      <svg
        ref={svg}
        viewBox={`0 0 ${width} ${height}`}
        className={tool ? "drawing" : ""}
        onMouseDown={onDown}
        onMouseUp={onUp}
        onMouseMove={(e) => setHover(toPx(e))}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#00e5ff" />
          </marker>
        </defs>
        {crossing.approach_zone?.length > 0 && (
          <polygon
            points={crossing.approach_zone.map((p) => p.join(",")).join(" ")}
            fill="rgba(0,229,255,0.12)"
            stroke="#00e5ff"
            strokeWidth={sw}
            strokeDasharray="6 4"
          />
        )}
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#fff" strokeWidth={sw * 2.5} />
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#e53935" strokeWidth={sw} strokeDasharray="8 6" />
        <text x={x1} y={y1 - 6 * k} fill="#fff" fontSize={fs} className="svg-label">
          stop line
        </text>
        <line
          x1={mx + nx * arrowLen}
          y1={my + ny * arrowLen}
          x2={mx - nx * arrowLen * 0.4}
          y2={my - ny * arrowLen * 0.4}
          stroke="#00e5ff"
          strokeWidth={sw * 1.5}
          markerEnd="url(#arrow)"
        />
        <circle cx={ax} cy={ay} r={5 * k} fill="#00e5ff" stroke="#000" strokeWidth={sw / 2} />
        <text x={ax + 8 * k} y={ay + 4 * k} fill="#00e5ff" fontSize={fs} className="svg-label">
          approach side
        </text>
        {(signal.layout === "custom" || signal.roi[2] > 0) && (
          <>
            {signal.layout !== "custom" && (
              <rect x={signal.roi[0]} y={signal.roi[1]} width={signal.roi[2]} height={signal.roi[3]} fill="none" stroke="#fff" strokeWidth={sw} />
            )}
            {Object.entries(lamps).map(([c, r]) => (
              <rect
                key={c}
                x={r[0]}
                y={r[1]}
                width={r[2]}
                height={r[3]}
                fill={reading && reading.state === c ? "rgba(255,255,255,0.25)" : "none"}
                stroke={LAMP_STROKE[c]}
                strokeWidth={sw}
              />
            ))}
          </>
        )}
        {pending && hover && tool === "line" && (
          <line x1={pending[0]} y1={pending[1]} x2={hover[0]} y2={hover[1]} stroke="#ffeb3b" strokeWidth={sw} strokeDasharray="4 4" />
        )}
        {pending && hover && isRectTool && (() => {
          const r = normRect(pending, hover);
          return <rect x={r[0]} y={r[1]} width={r[2]} height={r[3]} fill="rgba(255,235,59,0.15)" stroke="#ffeb3b" strokeWidth={sw} />;
        })()}
      </svg>
      {hover && (
        <div className="coords">
          x {Math.round(hover[0])}, y {Math.round(hover[1])}
        </div>
      )}
    </div>
  );
}
