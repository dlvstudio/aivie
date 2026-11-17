import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";
import {
  fetchComplete,
  fetchParents,
  type CompletionGraph,
  type ParentRow,
  type V3Edge,
  type V3Kind,
  type V3Node,
} from "./api";

const MODE_LABEL = "A3 — sinh từng token (multi)";

const FACT_SKIP_LABEL: Record<string, string> = {
  limit: "đủ K fact",
  empty: "completion rỗng",
  entity: "thiếu entity",
  dedup: "trùng đáp",
};

const KIND_COLOR: Record<V3Kind, string> = {
  query: "#61afef", // token câu hỏi (lan xuôi)
  phrase: "#d19a66", // nơ-ron trung gian (cụm phân rã Re-Pair)
  cluster: "#e5c07b", // nơ-ron cụm ISG (seg_graph)
  parent: "#e06c75", // nơ-ron cha THẮNG
  parent_alt: "#8b9cb3", // cha ứng viên khác
  completion: "#98c379", // token đáp (lan ngược)
  hub: "#c678dd", // nơ-ron mẫu chung (họ mẫu / slot hub)
  fact: "#56b6c2", // fact ghép (composition nhiều ý)
  ar_token: "#e5c07b", // token sinh A3 (autoregressive)
  episode: "#e06cbe", // nơ-ron trừu tượng t1 (episode SFT — Trụ 7 T2)
};

const KIND_LABEL: Record<V3Kind, string> = {
  query: "Token câu hỏi",
  phrase: "Nơ-ron trung gian (cụm)",
  cluster: "Nơ-ron cụm ISG",
  parent: "Nơ-ron cha (thắng)",
  parent_alt: "Cha ứng viên",
  completion: "Đáp (dội ra)",
  hub: "Nơ-ron mẫu chung (họ mẫu)",
  fact: "Fact ghép (nhiều ý)",
  ar_token: "Token A3 (sinh từng bước)",
  episode: "Nơ-ron trừu tượng t1 (episode)",
};

/** Kiểu cạnh v3 — màu, nét, chú thích sidebar. */
interface LinkStyle {
  color: string;
  width: number;
  dash?: [number, number];
  label: string;
  hint: string;
  showWeight: boolean;
  showIndex: boolean;
}

const LINK_STYLE: Record<string, LinkStyle> = {
  activate: {
    color: "rgba(224,108,117,0.92)",
    width: 2.4,
    label: "Lan xuôi → cha thắng",
    hint: "Token rời → cha (hiếm) · index slot + IDF",
    showWeight: true,
    showIndex: true,
  },
  activate_alt: {
    color: "rgba(139,156,179,0.8)",
    width: 1.8,
    dash: [6, 4],
    label: "Lan xuôi → ứng viên",
    hint: "Token rời → cha ứng viên · index + IDF",
    showWeight: true,
    showIndex: true,
  },
  seg_slot: {
    color: "rgba(224,108,117,0.88)",
    width: 2.2,
    label: "Token slot → cha",
    hint: "Token rời là 1 slot trên chuỗi match · index + trọng số cạnh kề",
    showWeight: true,
    showIndex: true,
  },
  seg_cluster: {
    color: "rgba(209,154,102,0.95)",
    width: 2.2,
    label: "Cụm ISG → cha",
    hint: "Cụm → cha · index slot trên chuỗi match · power",
    showWeight: true,
    showIndex: true,
  },
  seg_compose: {
    color: "rgba(97,175,239,0.88)",
    width: 1.8,
    label: "Token → cụm ISG",
    hint: "Token con gộp vào cụm · index 1=trái 2=phải · power",
    showWeight: true,
    showIndex: true,
  },
  compose: {
    color: "rgba(97,175,239,0.85)",
    width: 1.8,
    label: "Token → cụm phrase",
    hint: "Token query → nơ-ron trung gian Re-Pair",
    showWeight: false,
    showIndex: false,
  },
  seg_h: {
    color: "rgba(152,195,121,0.95)",
    width: 2.6,
    label: "ISG ngang (có hướng)",
    hint: "Cạnh ngang trong cha · hiện index + trọng số",
    showWeight: true,
    showIndex: true,
  },
  complete: {
    color: "rgba(152,195,121,0.9)",
    width: 2.2,
    label: "Lan ngược (đáp)",
    hint: "Cha → token đáp (pattern completion)",
    showWeight: false,
    showIndex: false,
  },
  hub_slot: {
    color: "rgba(198,120,221,0.95)",
    width: 2.2,
    dash: [6, 3],
    label: "Hub · slot",
    hint: "Filler query → nơ-ron mẫu chung",
    showWeight: false,
    showIndex: false,
  },
  hub_frame: {
    color: "rgba(198,120,221,0.5)",
    width: 1.4,
    dash: [4, 4],
    label: "Hub · khung",
    hint: "Token khung → nơ-ron mẫu chung",
    showWeight: false,
    showIndex: false,
  },
  compose_fact: {
    color: "rgba(86,182,194,0.85)",
    width: 2.0,
    label: "Query → fact ghép",
    hint: "Lan xuôi vào fact khi gom nhiều ý",
    showWeight: false,
    showIndex: false,
  },
  compose_link: {
    color: "rgba(209,154,102,0.9)",
    width: 2.0,
    label: "Fact → fact",
    hint: "Nối các fact đã gom",
    showWeight: false,
    showIndex: false,
  },
  compose_out: {
    color: "rgba(152,195,121,0.85)",
    width: 2.0,
    label: "Fact → đáp ghép",
    hint: "Sinh phần đáp sau khi ghép fact",
    showWeight: false,
    showIndex: false,
  },
  ar_step: {
    color: "rgba(229,192,123,0.95)",
    width: 2.2,
    label: "A3 · bước sinh",
    hint: "Chuỗi token autoregressive",
    showWeight: false,
    showIndex: false,
  },
  essay_body: {
    color: "rgba(152,195,121,0.75)",
    width: 2.0,
    label: "Essay · thân bài",
    hint: "Cha t1 → token bài dài",
    showWeight: false,
    showIndex: false,
  },
  match: {
    color: "rgba(224,108,190,0.85)",
    width: 1.8,
    label: "Episode khớp",
    hint: "Query → episode SFT",
    showWeight: false,
    showIndex: false,
  },
  resonance: {
    color: "rgba(139,156,179,0.55)",
    width: 1.2,
    dash: [3, 4],
    label: "Cộng hưởng episode",
    hint: "Episode → fact cộng hưởng",
    showWeight: true,
    showIndex: false,
  },
  resonance_fact: {
    color: "rgba(86,182,194,0.65)",
    width: 1.4,
    dash: [3, 4],
    label: "Fact cộng hưởng",
    hint: "Fact phụ trong essay",
    showWeight: true,
    showIndex: false,
  },
};

const LINK_LEGEND_ORDER = [
  "seg_h",
  "seg_slot",
  "seg_cluster",
  "seg_compose",
  "compose",
  "activate",
  "activate_alt",
  "complete",
  "hub_slot",
  "hub_frame",
  "compose_fact",
  "compose_link",
  "compose_out",
  "ar_step",
  "essay_body",
  "match",
  "resonance",
  "resonance_fact",
] as const;

function formatLinkWeight(w: number): string {
  if (w >= 100) return String(Math.round(w));
  if (w >= 10) return w.toFixed(1);
  if (w >= 1) return w.toFixed(2);
  return w.toFixed(3);
}

function resolveLinkStyle(link: FGEdge): LinkStyle {
  return (
    LINK_STYLE[link.kind] ?? {
      color: "rgba(139,156,179,0.45)",
      width: 1.0,
      dash: [4, 4],
      label: link.kind,
      hint: "",
      showWeight: false,
      showIndex: false,
    }
  );
}

function formatMs(ms: number): string {
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.round(ms)} ms`;
}

function linkCaption(link: FGEdge, style: LinkStyle): string | null {
  const parts: string[] = [];
  if (style.showIndex && link.slot_index != null) {
    parts.push(`i${link.slot_index}`);
  }
  if (style.showWeight && link.weight != null) {
    const w = link.weight;
    if (w !== 1 || link.kind === "seg_h" || link.kind === "seg_cluster" || link.kind === "seg_compose" || link.kind === "seg_slot") {
      parts.push(`w${formatLinkWeight(w)}`);
    }
  }
  return parts.length ? parts.join(" ") : null;
}

interface FGNode extends V3Node {
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}
interface FGEdge {
  source: string | FGNode;
  target: string | FGNode;
  weight: number;
  kind: string;
  slot_index?: number;
  parent?: number;
}

function nodeRadius(kind: V3Kind): number {
  if (kind === "parent") return 15;
  if (kind === "hub") return 13;
  if (kind === "fact") return 11;
  if (kind === "cluster") return 10;
  if (kind === "ar_token") return 8;
  if (kind === "parent_alt") return 9;
  if (kind === "completion") return 10;
  return 9;
}

/** Tầng: query dưới (y>0), parent giữa (y≈0), completion trên (y<0). Ghim fx/fy để drag không làm co cụm. */
function seedLayered(nodes: FGNode[]): FGNode[] {
  const byKind: Record<string, FGNode[]> = {};
  nodes.forEach((n) => (byKind[n.kind] ??= []).push(n));
  const place = (arr: FGNode[], y: number, spread: number) =>
    arr.forEach((n, i) => {
      const t = arr.length > 1 ? i / (arr.length - 1) - 0.5 : 0;
      n.x = t * spread;
      n.y = y;
    });
  place(byKind.hub ?? [], 360, 300);
  place(byKind.query ?? [], 240, 520);
  place(
    [...(byKind.phrase ?? []), ...(byKind.cluster ?? [])],
    120,
    420,
  );
  place([...(byKind.parent ?? []), ...(byKind.parent_alt ?? []), ...(byKind.episode ?? [])], 0, 620);
  place(byKind.fact ?? [], -120, 640);
  place(byKind.ar_token ?? [], -280, 480);
  place(byKind.completion ?? [], -220, 520);
  // Ghim mọi node — chỉ node kéo đổi fx/fy; các node khác không bị force kéo co cụm.
  nodes.forEach((n) => {
    if (n.x != null && n.y != null) {
      n.fx = n.x;
      n.fy = n.y;
    }
  });
  return nodes;
}

export default function GraphExplorer() {
  const [prompt, setPrompt] = useState("Hỏi: Người chữa bệnh là ai Đáp:");
  const [maxLen, setMaxLen] = useState(48);
  const [arStepIdx, setArStepIdx] = useState(0);
  const [data, setData] = useState<CompletionGraph | null>(null);
  const [nodes, setNodes] = useState<FGNode[]>([]);
  const [links, setLinks] = useState<FGEdge[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [runTiming, setRunTiming] = useState<{ serverMs?: number; totalMs: number } | null>(
    null,
  );

  // Browse parents
  const [browseQuery, setBrowseQuery] = useState("");
  const [parents, setParents] = useState<ParentRow[]>([]);
  const [stats, setStats] = useState({ total: 0, qa: 0, statement: 0 });

  const wrapRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods<FGNode, FGEdge> | undefined>(undefined);
  const [size, setSize] = useState({ w: 900, h: 560 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () =>
      setSize({ w: Math.max(400, el.clientWidth), h: Math.max(420, el.clientHeight) });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const runComplete = useCallback(async () => {
    setLoading(true);
    setError(null);
    setRunTiming(null);
    const t0 = performance.now();
    try {
      const g = await fetchComplete(prompt, {
        topCandidates: 6,
        maxLen,
      });
      const totalMs = performance.now() - t0;
      setRunTiming({ serverMs: g.meta.elapsed_ms, totalMs });
      setData(g);
      setArStepIdx(0);
      setNodes(seedLayered(g.nodes.map((n) => ({ ...n }))));
      setLinks(
        g.edges.map((e: V3Edge) => ({
          source: e.source,
          target: e.target,
          weight: e.weight,
          kind: e.kind,
          slot_index: e.slot_index,
          parent: e.parent,
        }))
      );
      setTimeout(() => fgRef.current?.zoomToFit(500, 60), 400);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [prompt, maxLen]);

  const runBrowse = useCallback(async () => {
    try {
      const r = await fetchParents(browseQuery, 80);
      setParents(r.parents);
      setStats({ total: r.total, qa: r.qa, statement: r.statement });
    } catch (e) {
      setError(String(e));
    }
  }, [browseQuery]);

  useEffect(() => {
    runBrowse();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const graphData = useMemo(() => ({ nodes, links }), [nodes, links]);

  const presentLinkKinds = useMemo(() => {
    const kinds = new Set(links.map((l) => l.kind));
    return LINK_LEGEND_ORDER.filter((k) => kinds.has(k));
  }, [links]);

  const paintNode = useCallback(
    (node: FGNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const r = nodeRadius(node.kind);
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      if (node.kind === "hub") {
        // hình thoi cho nơ-ron mẫu chung — khác hẳn nơ-ron cha (tròn)
        ctx.beginPath();
        ctx.moveTo(x, y - r);
        ctx.lineTo(x + r, y);
        ctx.lineTo(x, y + r);
        ctx.lineTo(x - r, y);
        ctx.closePath();
        ctx.fillStyle = KIND_COLOR.hub;
        ctx.fill();
        // viền: filler đã nối (liền, xanh) vs filler lạ (đứt, cam)
        ctx.strokeStyle = node.slot_known ? "#98c379" : "#e5c07b";
        ctx.lineWidth = 2 / globalScale;
        if (!node.slot_known) ctx.setLineDash([3 / globalScale, 3 / globalScale]);
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = KIND_COLOR[node.kind] ?? "#8b9cb3";
        ctx.fill();
        if (node.kind === "parent") {
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 2 / globalScale;
          ctx.stroke();
        }
      }
      const fontSize = Math.max(11 / globalScale, 3);
      ctx.font = `${fontSize}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#e7ecf3";
      const isParent = node.kind === "parent" || node.kind === "parent_alt";
      let lbl = node.label;
      if ((isParent || node.kind === "hub" || node.kind === "fact") && lbl.length > 30)
        lbl = lbl.slice(0, 28) + "…";
      ctx.fillText(lbl, x, y + r + 2 / globalScale);
      if (node.kind === "hub" && node.slot) {
        ctx.fillStyle = node.slot_known ? "#98c379" : "#e5c07b";
        const tag = node.slot_known
          ? `slot: ${node.slot} ✓`
          : `slot: ${node.slot} (lạ)`;
        ctx.fillText(tag, x, y + r + 2 / globalScale + fontSize + 1);
      } else if (node.score != null) {
        ctx.fillStyle = "#d19a66";
        ctx.fillText(node.score.toFixed(2), x, y + r + 2 / globalScale + fontSize + 1);
      }
    },
    []
  );

  const paintLink = useCallback(
    (link: FGEdge, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const s = link.source as FGNode;
      const t = link.target as FGNode;
      if (s.x == null || s.y == null || t.x == null || t.y == null) return;

      const style = resolveLinkStyle(link);
      const caption = linkCaption(link, style);
      const dx = t.x - s.x;
      const dy = t.y - s.y;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len;
      const uy = dy / len;
      const sr = nodeRadius((s.kind as V3Kind) ?? "query");
      const tr = nodeRadius((t.kind as V3Kind) ?? "query");
      const arrowPad = 6 / globalScale;
      const x1 = s.x + ux * (sr + 2 / globalScale);
      const y1 = s.y + uy * (sr + 2 / globalScale);
      const x2 = t.x - ux * (tr + arrowPad);
      const y2 = t.y - uy * (tr + arrowPad);

      ctx.save();
      ctx.strokeStyle = style.color;
      ctx.fillStyle = style.color;
      ctx.lineWidth = style.width / globalScale;
      if (style.dash) {
        ctx.setLineDash(style.dash.map((d) => d / globalScale));
      }

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      const head = 7 / globalScale;
      const angle = Math.atan2(y2 - y1, x2 - x1);
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(
        x2 - head * Math.cos(angle - Math.PI / 7),
        y2 - head * Math.sin(angle - Math.PI / 7),
      );
      ctx.lineTo(
        x2 - head * Math.cos(angle + Math.PI / 7),
        y2 - head * Math.sin(angle + Math.PI / 7),
      );
      ctx.closePath();
      ctx.fill();

      ctx.setLineDash([]);

      if (caption && globalScale > 0.35) {
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        const nx = -uy;
        const ny = ux;
        const offset = 7 / globalScale;
        const lx = mx + nx * offset;
        const ly = my + ny * offset;
        const fontSize = Math.max(9 / globalScale, 7);
        ctx.font = `600 ${fontSize}px ui-monospace, monospace`;
        const tw = ctx.measureText(caption).width;
        const pad = 3 / globalScale;
        ctx.fillStyle = "rgba(15,20,25,0.88)";
        ctx.fillRect(lx - tw / 2 - pad, ly - fontSize - pad, tw + pad * 2, fontSize + pad * 2);
        ctx.fillStyle = style.color;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(caption, lx, ly - fontSize * 0.35);
      }
      ctx.restore();
    },
    []
  );

  return (
    <div className="workspace graph-workspace">
      <aside className="panel panel-left">
        <section className="panel-block">
          <h2 className="panel-title">Pattern Completion (v3)</h2>
          <p className="panel-hint">
            Lan xuôi: token → cụm trung gian → nơ-ron cha khớp. Lan ngược: cha thắng → đáp dội ra.
          </p>
          <textarea
            className="field-input"
            rows={2}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Hỏi: … Đáp:"
          />
          <label className="field" style={{ marginTop: "0.5rem" }}>
            <span className="field-label">Max bước A3</span>
            <input
              className="field-input"
              type="number"
              min={8}
              max={64}
              value={maxLen}
              onChange={(e) => setMaxLen(Number(e.target.value))}
            />
          </label>
          <button
            type="button"
            className="btn-primary"
            onClick={runComplete}
            disabled={loading}
          >
            {loading ? "Đang chạy…" : "Lan xuôi → ngược"}
          </button>
          {runTiming && (
            <div className="v3-timing">
              <span className="v3-timing-label">Thời gian</span>
              <strong>{formatMs(runTiming.serverMs ?? runTiming.totalMs)}</strong>
              {runTiming.serverMs != null && (
                <span className="muted">
                  {" "}
                  · tổng {formatMs(runTiming.totalMs)}
                </span>
              )}
            </div>
          )}
          {error && <div className="error">{error}</div>}
        </section>

        {data && (
          <section className="panel-block">
            <h2 className="panel-title panel-title-sm">Kết quả</h2>
            <div className="v3-result-meta">
              <div className="v3-mode-badge">{MODE_LABEL}</div>
              {runTiming && (
                <div className="v3-timing-badge" title="Thời gian xử lý completion_graph trên server">
                  ⏱ {formatMs(runTiming.serverMs ?? runTiming.totalMs)}
                  {runTiming.serverMs != null ? ` · tổng ${formatMs(runTiming.totalMs)}` : ""}
                </div>
              )}
            </div>
            <div className="v3-answer">
              <span className="v3-answer-label">Đáp sinh ra (A3 token-by-token)</span>
              <strong>{data.generated.split("Đáp:").pop()?.trim() || "—"}</strong>
            </div>
            {data.hub && (
              <div className="v3-hub">
                <span className="v3-hub-label">Nơ-ron mẫu chung (họ mẫu)</span>
                <code className="v3-hub-skeleton">{data.hub.skeleton}</code>
                <div className="v3-hub-slot">
                  slot ={" "}
                  <strong>{data.hub.slot}</strong>{" "}
                  <span className={`hub-badge ${data.hub.slot_known ? "known" : "novel"}`}>
                    {data.hub.slot_known ? "đã nối lúc train ✓" : "filler lạ → chú ý"}
                  </span>
                  {data.hub.guard === "reroute" && (
                    <span className="hub-badge known"> route → đúng thực thể</span>
                  )}
                  {data.hub.guard === "suppress" && (
                    <span className="hub-badge suppress"> chặn bịa (không đủ dữ kiện)</span>
                  )}
                </div>
                <div className="v3-hub-fillers">
                  {data.hub.n_fillers} filler:{" "}
                  {data.hub.fillers.map((f) => (
                    <span
                      key={f}
                      className={`hub-filler${f === data.hub!.slot ? " current" : ""}`}
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {data.anchor_groups && data.anchor_groups.length > 0 && (
              <div className="v3-hub">
                <span className="v3-hub-label">Cụm WH/frame kích thích + gom (anchor)</span>
                {data.anchor_groups.map((g) => (
                  <div key={g.cluster_idx} className="v3-hub-fillers">
                    <code className="v3-hub-skeleton">{g.cluster}</code>{" "}
                    <span className="hub-badge known">{g.n_members} cha gom · power {g.power}</span>{" "}
                    {g.fillers.map((f) => (
                      <span key={f} className="hub-filler">
                        {f}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            )}
            {data.fact_stimulus && (
              <div className="v3-compose v3-fact-stimulus">
                <span className="v3-compose-label">
                  Fact bồi · stimulus (TK_023)
                  {" · "}thực thể:{" "}
                  <strong>{data.fact_stimulus.entity.join(" ")}</strong>
                  {data.fact_stimulus.primary ? (
                    <> · neo: <strong>{data.fact_stimulus.primary}</strong></>
                  ) : null}
                  {" · "}
                  {data.fact_stimulus.eligible ? (
                    <span className="fact-stim-badge eligible">cổng WH mở</span>
                  ) : (
                    <span className="fact-stim-badge ineligible">cổng WH đóng</span>
                  )}
                </span>
                {data.fact_stimulus.facts.length > 0 ? (
                  <ol className="v3-compose-facts v3-fact-stimulus-list">
                    {data.fact_stimulus.facts.map((f) => (
                      <li
                        key={f.idx}
                        className={
                          f.picked
                            ? "fact-stim-picked"
                            : f.skip
                              ? "fact-stim-skipped"
                              : ""
                        }
                      >
                        <span className="fact-stim-rank">#{f.rank}</span>
                        <span className="fact-stim-score">{f.score.toFixed(3)}</span>
                        <span className="fact-stim-text">{f.text}</span>
                        {f.picked ? (
                          <span className="fact-stim-badge picked">ghép</span>
                        ) : f.skip ? (
                          <span className="fact-stim-badge skipped" title={f.skip}>
                            {FACT_SKIP_LABEL[f.skip] ?? f.skip}
                          </span>
                        ) : null}
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="v3-compose-hint muted">
                    {data.fact_stimulus.eligible
                      ? "Không có ứng viên fact trong pool facts_about."
                      : "Chưa đủ điều kiện cổng WH để xếp và bồi fact."}
                  </p>
                )}
              </div>
            )}
            {data.compose && data.compose.facts.length >= 2 && (
              <div className="v3-compose">
                <span className="v3-compose-label">
                  Gom fact (multi) · thực thể: <strong>{data.compose.focus.join(" ")}</strong>
                </span>
                {data.compose.answer && (
                  <p className="v3-compose-answer">
                    Ghép: <strong>{data.compose.answer}</strong>
                  </p>
                )}
                <ol className="v3-compose-facts">
                  {data.compose.facts.map((f) => (
                    <li key={f.idx}>{f.text}</li>
                  ))}
                </ol>
              </div>
            )}
            {data.autoregressive && (
              <div className="v3-ar">
                <span className="v3-ar-label">
                  A3 autoregressive · agenda ({data.autoregressive.multi ? "multi" : "gọn"})
                </span>
                <p className="v3-ar-agenda">
                  Ý cần phủ:{" "}
                  <code>{data.autoregressive.agenda.join(" · ") || "—"}</code>
                </p>
                {data.autoregressive.steps.length > 0 && (
                  <div className="v3-ar-steps">
                    <div className="v3-ar-step-nav">
                      <button
                        type="button"
                        disabled={arStepIdx <= 0}
                        onClick={() => setArStepIdx((i) => i - 1)}
                      >
                        ←
                      </button>
                      <span>
                        Bước {arStepIdx + 1}/{data.autoregressive.steps.length}
                      </span>
                      <button
                        type="button"
                        disabled={arStepIdx >= data.autoregressive.steps.length - 1}
                        onClick={() => setArStepIdx((i) => i + 1)}
                      >
                        →
                      </button>
                    </div>
                    {(() => {
                      const s = data.autoregressive!.steps[arStepIdx];
                      return (
                        <div className="v3-ar-step-detail">
                          <p>
                            w₀=<code>{s.w0}</code> → <strong>{s.chosen}</strong>
                          </p>
                          <p className="muted">
                            Agenda còn: {s.agenda_pending.join(", ") || "—"}
                          </p>
                          <ul className="v3-ar-top">
                            {s.top.slice(0, 6).map((t) => (
                              <li key={t.token}>
                                <span>{t.token}</span>
                                <span>{t.score.toFixed(3)}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            )}
            <h3 className="panel-title panel-title-sm">Nơ-ron cha ứng viên</h3>
            <ul className="cand-list">
              {data.candidates.map((c) => (
                <li
                  key={c.idx}
                  className={`cand-item${c.matched ? " is-matched" : ""}`}
                  title={c.answer}
                >
                  <span className="cand-score">{c.score.toFixed(2)}</span>
                  <span className="cand-q">{c.question || c.answer}</span>
                  <span className={`cand-badge ${c.is_qa ? "qa" : "stmt"}`}>
                    {c.is_qa ? "QA" : "câu"}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="panel-block panel-legend">
          <h2 className="panel-title panel-title-sm">Tầng nơ-ron</h2>
          <div className="legend legend-nodes">
            {(Object.keys(KIND_COLOR) as V3Kind[]).map((k) => (
              <span
                key={k}
                style={{ ["--dot" as string]: KIND_COLOR[k] }}
                className="legend-dot"
              >
                {KIND_LABEL[k]}
              </span>
            ))}
          </div>
          <h2 className="panel-title panel-title-sm">Loại cạnh (link)</h2>
          <p className="panel-hint panel-hint-tight">
            Mũi tên = hướng lan. Nhãn trên cạnh: <code>iN</code> = index slot ISG,{" "}
            <code>wX</code> = trọng số (IDF / power).
          </p>
          <div className="legend legend-links">
            {LINK_LEGEND_ORDER.map((kind) => {
              const st = LINK_STYLE[kind];
              if (!st) return null;
              const active = presentLinkKinds.includes(kind);
              return (
                <div
                  key={kind}
                  className={`legend-link${active ? " is-active" : ""}`}
                  title={st.hint}
                >
                  <span
                    className="legend-link-line"
                    style={{
                      borderTopColor: st.color,
                      borderTopWidth: `${Math.min(st.width, 3)}px`,
                      borderTopStyle: st.dash ? "dashed" : "solid",
                    }}
                  />
                  <span className="legend-link-text">
                    <strong>{st.label}</strong>
                    {st.showIndex || st.showWeight ? (
                      <span className="legend-link-meta">
                        {st.showIndex ? " · index" : ""}
                        {st.showWeight ? " · trọng số" : ""}
                      </span>
                    ) : null}
                    <span className="legend-link-hint">{st.hint}</span>
                  </span>
                </div>
              );
            })}
          </div>
        </section>

        <section className="panel-block">
          <h2 className="panel-title panel-title-sm">
            Duyệt nơ-ron cha ({stats.total}: {stats.qa} QA · {stats.statement} câu)
          </h2>
          <input
            type="search"
            className="field-input"
            placeholder="Tìm trong nội dung cha…"
            value={browseQuery}
            onChange={(e) => setBrowseQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runBrowse()}
          />
          <ul className="parent-browse">
            {parents.slice(0, 40).map((p) => (
              <li
                key={p.idx}
                className="parent-browse-item"
                onClick={() => setPrompt(`Hỏi: ${p.question} Đáp:`)}
                title={p.emit}
              >
                <span className={`cand-badge ${p.is_qa ? "qa" : "stmt"}`}>
                  {p.is_qa ? "QA" : "câu"}
                </span>
                <span className="pb-text">{p.emit}</span>
                {p.power > 1 && <span className="pb-power">×{p.power}</span>}
              </li>
            ))}
          </ul>
        </section>
      </aside>

      <div className="graph-main">
        <div className="graph-status-bar">
          <span>
            <strong>{nodes.length}</strong> nodes ·{" "}
            <strong>{links.length}</strong> edges · cha:{" "}
            <strong>{data?.meta.n_parents ?? 0}</strong>
            {runTiming && (
              <>
                {" "}
                · <strong>{formatMs(runTiming.serverMs ?? runTiming.totalMs)}</strong>
              </>
            )}
          </span>
          <span className="muted">
            {data?.compose
              ? `Multi · gom ${data.compose.facts.length} fact`
              : data?.matched != null
                ? `Cha #${data.matched} khớp`
                : "Không cha đủ ngưỡng · xem fact ghép hoặc W_seq"}
          </span>
        </div>
        <div ref={wrapRef} className="graph-canvas-wrap">
          {nodes.length > 0 ? (
            <ForceGraph2D
              ref={fgRef}
              width={size.w}
              height={size.h}
              graphData={graphData}
              nodeId="id"
              nodeLabel={(n) => (n as FGNode).label}
              linkDirectionalArrowLength={0}
              linkDirectionalArrowRelPos={1}
              linkCurvature={0.06}
              cooldownTicks={0}
              d3AlphaDecay={0.02}
              nodeCanvasObject={paintNode}
              nodeCanvasObjectMode={() => "replace"}
              linkCanvasObject={paintLink}
              linkCanvasObjectMode={() => "replace"}
              onNodeDrag={(n) => {
                const node = n as FGNode;
                node.fx = node.x;
                node.fy = node.y;
              }}
              onNodeDragEnd={(n) => {
                const node = n as FGNode;
                node.fx = node.x;
                node.fy = node.y;
              }}
              backgroundColor="#0f1419"
            />
          ) : (
            <div className="graph-empty">Nhập prompt và bấm “Lan xuôi → ngược”.</div>
          )}
        </div>
      </div>
    </div>
  );
}
