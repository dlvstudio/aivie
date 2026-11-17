import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { fetchTrace, type ScoreItem, type TracePayload, type TraceStep } from "./api";

const KIND_COLOR: Record<string, string> = {
  rule: "#e06c75",
  var: "#98c379",
  label: "#d19a66",
  position: "#c678dd",
  surface: "#61afef",
  special: "#abb2bf",
  concept: "#ff79c6",
  other: "#8b9cb3",
};

const EDGE_COLOR: Record<string, string> = {
  grammar: "rgba(91,159,212,0.8)",
  anchor: "rgba(224,108,117,0.9)",
  secondary: "rgba(139,156,179,0.5)",
  collocation: "rgba(152,195,121,0.9)",
  co_activation: "rgba(198,120,221,0.6)",
  position: "rgba(97,175,239,0.6)",
  reinforced: "rgba(229,192,123,0.7)",
  emergent: "rgba(139,156,179,0.4)",
  analogy: "rgba(78,201,176,0.75)", // §180 — loại suy khái niệm (teal, phân biệt)
};

const nodeRadius = (kind: string): number => {
  if (kind === "rule") return 7;
  if (kind === "var") return 6;
  if (kind === "concept") return 9;
  if (kind === "surface") return 5;
  return 4;
};

function ScoreBars({
  title,
  items,
  className,
  highlight,
}: {
  title: string;
  items: ScoreItem[];
  className?: string;
  highlight?: string | null;
}) {
  const max = items[0]?.score ?? 1;
  return (
    <div className="score-bars">
      <h3>{title}</h3>
      {items.map((item) => (
        <div className="bar-row" key={`${title}-${item.token}`}>
          <span>{item.token}</span>
          <div className="bar-track">
            <div
              className={`bar-fill ${className ?? ""}`}
              style={{
                width: `${Math.max(2, (item.score / max) * 100)}%`,
                opacity: highlight === item.token ? 1 : 0.85,
              }}
            />
          </div>
          <span>{item.score.toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}

function StepDetail({ step }: { step: TraceStep }) {
  return (
    <div className="step-detail">
      <h3 style={{ margin: "0 0 0.5rem" }}>
        Step {step.step} · w₀ = <code>{step.w0}</code>
      </h3>
      <p style={{ fontSize: "0.85rem", color: "var(--muted)" }}>
        phase: {step.phase}
        {step.frame_complete ? " · frame complete" : ""}
      </p>
      <div className="tags">
        {step.active_rules.map((r) => (
          <span key={r} className="tag rule">
            {r}
          </span>
        ))}
        {step.eligible_vars.map((v) => (
          <span key={v} className="tag var">
            {v}
          </span>
        ))}
      </div>
      <table className="decay-table">
        <thead>
          <tr>
            <th>History</th>
            <th>d_frame</th>
            <th>d_bind</th>
          </tr>
        </thead>
        <tbody>
          {step.decay.map((d) => (
            <tr key={d.token + String(d.d_frame)}>
              <td>{d.token}</td>
              <td>{d.d_frame}</td>
              <td>{d.d_bind}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {step.chosen && (
        <p style={{ marginTop: "0.75rem" }}>
          Chosen: <strong>{step.chosen}</strong>
        </p>
      )}
    </div>
  );
}

const DEFAULT_PROMPT = "Hỏi: Em học gì ? Đáp:";

export default function InferenceLab() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxLen, setMaxLen] = useState(8);
  const [trace, setTrace] = useState<TracePayload | null>(null);
  const [stepIdx, setStepIdx] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Play/pause autoplay states
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1200); // milliseconds per step

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ w: 400, h: 400 });

  // Update canvas size dynamically
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => {
      setSize({
        w: el.clientWidth || 400,
        h: el.clientHeight || 400,
      });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [trace, stepIdx]);

  const run = useCallback(async () => {
    setLoading(true);
    setIsPlaying(false);
    setError(null);
    try {
      const data = await fetchTrace(prompt, maxLen);
      setTrace(data);
      setStepIdx(0);
      // Automatically trigger playback of the generated trace!
      setIsPlaying(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [prompt, maxLen]);

  const step = trace?.steps[stepIdx];

  // Autoplay effect
  useEffect(() => {
    if (!isPlaying || !trace) return;
    const interval = setInterval(() => {
      setStepIdx((prev) => {
        if (prev < trace.steps.length - 1) {
          return prev + 1;
        } else {
          setIsPlaying(false);
          return prev;
        }
      });
    }, playSpeed);
    return () => clearInterval(interval);
  }, [isPlaying, trace, playSpeed]);

  // Map step_graph into stable d3 data
  const graphData = useMemo(() => {
    if (!step?.step_graph) return { nodes: [], links: [] };
    const nodes = step.step_graph.nodes.map((n) => ({ ...n }));
    const links = step.step_graph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      weight: e.weight,
      kind: e.kind,
    }));
    return { nodes, links };
  }, [step]);

  const paintNode = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const r = nodeRadius(node.kind || "other");
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = KIND_COLOR[node.kind] ?? "#8b9cb3";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.lineWidth = 1 / globalScale;
      ctx.stroke();

      const fontSize = Math.max(9 / globalScale, 3.5);
      ctx.font = `${fontSize}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#e7ecf3";
      const lbl = node.label.length > 20 ? node.label.slice(0, 18) + "…" : node.label;
      ctx.fillText(lbl, x, y + r + 1.5 / globalScale);
    },
    []
  );

  const paintLink = useCallback(
    (link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const start = link.source;
      const end = link.target;
      if (start.x == null || end.x == null) return;

      const w = link.weight;
      const kind = link.kind ?? "grammar";
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.strokeStyle = EDGE_COLOR[kind] ?? "rgba(139,156,179,0.45)";
      ctx.lineWidth = Math.max(0.6, Math.min(3, Math.abs(w) * 0.85)) / globalScale;
      ctx.stroke();

      const mx = (start.x + end.x) / 2;
      const my = (start.y + end.y) / 2;
      const label = w.toFixed(2);
      const pad = 2 / globalScale;
      const fontSize = Math.max(8 / globalScale, 3);
      ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(15,20,25,0.85)";
      ctx.fillRect(
        mx - tw / 2 - pad,
        my - fontSize / 2 - pad,
        tw + pad * 2,
        fontSize + pad * 2
      );
      ctx.fillStyle = "#e7ecf3";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, mx, my);
    },
    []
  );

  return (
    <div className="workspace inference-workspace">
      <aside className="panel panel-left">
        <section className="panel-block">
          <h2 className="panel-title">Prompt</h2>
          <textarea
            className="field-input"
            style={{ minHeight: 88, resize: "vertical" }}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <label className="field" style={{ marginTop: "0.65rem" }}>
            <span className="field-label">Max generate steps</span>
            <input
              className="field-input"
              type="number"
              min={1}
              max={32}
              value={maxLen}
              onChange={(e) => setMaxLen(Number(e.target.value))}
            />
          </label>
          <button
            type="button"
            className="btn-primary"
            onClick={run}
            disabled={loading}
            style={{ marginTop: "0.75rem" }}
          >
            {loading ? "Tracing…" : "Run trace"}
          </button>
          {error && <div className="error">{error}</div>}
          {trace && (
            <p className="panel-hint" style={{ marginTop: "1rem" }}>
              Generated: <strong>{trace.generated}</strong>
            </p>
          )}
        </section>

        <section className="panel-block">
          <h2 className="panel-title">Autoplay Speed</h2>
          <label className="field">
            <span className="field-label">Interval (ms)</span>
            <input
              className="field-input"
              type="number"
              step="100"
              min="200"
              max="5000"
              value={playSpeed}
              onChange={(e) => setPlaySpeed(Number(e.target.value))}
            />
          </label>
        </section>
      </aside>

      <section className="inference-main">
        {trace && step ? (
          <>
            <div className="step-nav">
              <button
                type="button"
                disabled={stepIdx <= 0}
                onClick={() => {
                  setIsPlaying(false);
                  setStepIdx((i) => i - 1);
                }}
              >
                ←
              </button>
              <span>
                Step {stepIdx + 1} / {trace.steps.length}
              </span>
              <button
                type="button"
                disabled={stepIdx >= trace.steps.length - 1}
                onClick={() => {
                  setIsPlaying(false);
                  setStepIdx((i) => i + 1);
                }}
              >
                →
              </button>

              <button
                type="button"
                style={{ marginLeft: "1rem", minWidth: 80 }}
                onClick={() => setIsPlaying(!isPlaying)}
              >
                {isPlaying ? "Pause ⏸" : "Play ▶"}
              </button>
            </div>

            <div className="inference-grid">
              {/* Column 1: Step Detail */}
              <StepDetail step={step} />

              {/* Column 2: Score Bars */}
              <div>
                <ScoreBars
                  title="Dual-context (base)"
                  items={step.scores_base_top}
                  highlight={step.chosen}
                />
                <ScoreBars
                  title="After matcher"
                  items={step.scores_matcher_top}
                  className="matcher"
                  highlight={step.chosen}
                />
                <ScoreBars
                  title="Final (generate)"
                  items={step.scores_final_top}
                  highlight={step.chosen}
                />
              </div>

              {/* Column 3: Step-level Spreading Activation Subgraph */}
              <div className="step-graph-container">
                <div className="step-graph-header">
                  <h3>Spreading Activation Subgraph</h3>
                </div>
                <div ref={wrapRef} className="step-graph-canvas-wrap">
                  {graphData.nodes.length > 0 ? (
                    <ForceGraph2D
                      width={size.w}
                      height={size.h}
                      graphData={graphData}
                      nodeId="id"
                      nodeLabel={(n: any) => n.label}
                      linkDirectionalArrowLength={4}
                      linkDirectionalArrowRelPos={0.92}
                      linkCurvature={0.1}
                      cooldownTicks={120}
                      d3AlphaDecay={0.022}
                      d3VelocityDecay={0.3}
                      nodeVal={(n: any) => Math.pow(nodeRadius(n.kind) + 8, 2)}
                      nodeCanvasObject={paintNode}
                      nodeCanvasObjectMode={() => "replace"}
                      linkCanvasObject={paintLink}
                      linkCanvasObjectMode={() => "replace"}
                      backgroundColor="#0f1419"
                    />
                  ) : (
                    <div className="graph-empty" style={{ border: "none" }}>
                      No active graph connections for this step.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="graph-empty">
            Nhập prompt QA (vd. sau <code>Đáp:</code>) và Run trace để xem
            active rules, eligible Var, và top-k scores từng bước.
          </div>
        )}
      </section>
    </div>
  );
}
