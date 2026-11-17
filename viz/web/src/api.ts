export type NodeKind = "rule" | "var" | "label" | "position" | "surface" | "special" | "concept" | "other";

export interface GraphNode {
  id: number;
  label: string;
  kind: NodeKind;
}

export interface GraphEdge {
  source: number;
  target: number;
  weight: number;
  kind?: string;
}

export interface GraphPayload {
  meta: {
    rules: string[];
    n_nodes: number;
    n_edges: number;
    min_weight?: number;
    hops?: number;
    checkpoint?: string;
  };
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ScoreItem {
  id: number;
  token: string;
  score: number;
}

export interface TraceStep {
  step: number;
  phase: string;
  w0: string;
  history: string[];
  decay: { token: string; d_frame: number; d_bind: number }[];
  active_rules: string[];
  eligible_vars: string[];
  frame_complete: boolean;
  scores_base_top: ScoreItem[];
  scores_matcher_top: ScoreItem[];
  scores_final_top: ScoreItem[];
  chosen: string | null;
  step_graph?: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
}

export interface TracePayload {
  prompt: string;
  steps: TraceStep[];
  generated: string;
}

const API = "/api";

export async function fetchRules(): Promise<string[]> {
  const res = await fetch(`${API}/rules`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.rules as string[];
}

export async function fetchGraph(
  rules: string[],
  minWeight = 0.15,
  hops = 2,
  mode: "structural" | "bfs" = "structural"
): Promise<GraphPayload> {
  const q = new URLSearchParams({
    rules: rules.join(","),
    min_weight: String(minWeight),
    hops: String(hops),
    mode,
  });
  const res = await fetch(`${API}/graph?${q}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchTrace(
  prompt: string,
  maxLen = 8,
  topK = 12
): Promise<TracePayload> {
  const res = await fetch(`${API}/trace`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, max_len: maxLen, top_k: topK }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchHealth(): Promise<{ ok: boolean; checkpoint: string }> {
  const res = await fetch(`${API}/health`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// --- v3: nơ-ron cha + pattern completion ---

export type GenerateMode = "autoregressive";

export type V3Kind =
  | "query"
  | "phrase"
  | "cluster"
  | "parent"
  | "parent_alt"
  | "completion"
  | "hub"
  | "fact"
  | "ar_token"
  | "episode";

export interface V3Node {
  id: string;
  label: string;
  kind: V3Kind;
  score?: number;
  is_qa?: boolean;
  answer?: string;
  index?: number;
  df?: number;
  slot?: string;
  slot_known?: boolean;
  fillers?: string[];
  n_fillers?: number;
}

export interface HubInfo {
  skeleton: string;
  slot: string;
  slot_known: boolean;
  fillers: string[];
  n_fillers: number;
  guard?: "reroute" | "suppress";
}

// TK_014 — anchor cụm WH/frame được kích thích + nhóm gom (X→"ở đâu", Y→"ở đâu")
export interface AnchorGroup {
  cluster: string;
  cluster_idx: number;
  power: number;
  n_members: number;
  fillers: string[];
}

export interface V3Edge {
  source: string;
  target: string;
  weight: number;
  kind: string;
  /** ISG ngang: vị trí slot trên chuỗi cha (TK_013). */
  slot_index?: number;
  /** ISG ngang: parent_idx gắn với cạnh. */
  parent?: number;
}

export interface V3Candidate {
  idx: number;
  score: number;
  rank: number;
  is_qa: boolean;
  question: string;
  answer: string;
  matched: boolean;
}

export interface ComposeInfo {
  focus: string[];
  facts: { idx: number; text: string }[];
  answer: string;
}

export interface FactStimulusItem {
  idx: number;
  text: string;
  score: number;
  rank: number;
  picked: boolean;
  skip: string | null;
}

export interface FactStimulusInfo {
  eligible: boolean;
  entity: string[];
  primary: string;
  facts: FactStimulusItem[];
}

export interface AutoregressiveStep {
  step: number;
  w0: string;
  chosen: string;
  agenda_pending: string[];
  top: { token: string; score: number }[];
}

export interface AutoregressiveInfo {
  multi: boolean;
  agenda: string[];
  answer: string;
  steps: AutoregressiveStep[];
}

export interface EssayInfo {
  matched: boolean;
  route?: "sft" | "unified" | "none";
  fallback?: "compose" | "hybrid";
  key?: string;
  title?: string;
  score?: number;
  n_members?: number;
  n_resonance?: number;
  answer: string;
}

export interface CompletionGraph {
  nodes: V3Node[];
  edges: V3Edge[];
  candidates: V3Candidate[];
  matched: number | null;
  completion: string;
  generated: string;
  hybrid_generated?: string;
  hub: HubInfo | null;
  anchor_groups?: AnchorGroup[];
  compose: ComposeInfo | null;
  fact_stimulus: FactStimulusInfo | null;
  autoregressive: AutoregressiveInfo | null;
  essay: EssayInfo | null;
  generate_mode: GenerateMode;
  meta: {
    n_parents: number;
    n_hubs?: number;
    q_content?: string[];
    min_score?: number;
    elapsed_ms?: number;
  };
}

export interface ParentRow {
  idx: number;
  is_qa: boolean;
  power: number;
  question: string;
  answer: string;
  emit: string;
}

export interface ParentsList {
  parents: ParentRow[];
  total: number;
  qa: number;
  statement: number;
  shown: number;
}

export interface CompleteOptions {
  topCandidates?: number;
  maxLen?: number;
  mode?: GenerateMode;
  multi?: boolean;
  maxFacts?: number;
}

export async function fetchComplete(
  prompt: string,
  options: CompleteOptions | number = 6
): Promise<CompletionGraph> {
  const opts: CompleteOptions =
    typeof options === "number" ? { topCandidates: options } : options;
  const res = await fetch(`${API}/v3/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      top_candidates: opts.topCandidates ?? 6,
      max_len: opts.maxLen ?? 48,
      mode: opts.mode ?? "autoregressive",
      multi: opts.multi ?? true,
      max_facts: opts.maxFacts ?? 3,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchParents(query = "", limit = 60): Promise<ParentsList> {
  const q = new URLSearchParams({ query, limit: String(limit) });
  const res = await fetch(`${API}/v3/parents?${q}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// --- embedding E (PPMI + SVD) ---

export interface EmbeddingTokenRow {
  id: number;
  token: string;
  norm?: number;
  parent_df?: number;
}

export interface EmbeddingInfo {
  available: boolean;
  vocab_size: number;
  dim: number;
  nonzero_rows?: number;
  norm_mean?: number;
  norm_max?: number;
  sample_tokens?: EmbeddingTokenRow[];
  message?: string;
}

export interface EmbeddingNeighbor {
  id: number;
  token: string;
  similarity: number;
}

export interface EmbeddingLookup {
  found: boolean;
  query: string;
  suggestions?: string[];
  id?: number;
  token?: string;
  dim?: number;
  norm?: number;
  vector_preview?: number[];
  vector_preview_dims?: number[];
  neighbors?: EmbeddingNeighbor[];
  parent_df?: number;
}

export interface EmbeddingCompare {
  found: boolean;
  token_a: string;
  token_b: string;
  id_a?: number;
  id_b?: number;
  similarity?: number;
  norm_a?: number;
  norm_b?: number;
}

export interface EmbeddingSearch {
  query: string;
  tokens: EmbeddingTokenRow[];
  shown: number;
}

export async function fetchEmbeddingInfo(): Promise<EmbeddingInfo> {
  const res = await fetch(`${API}/embedding/info`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchEmbeddingLookup(
  token: string,
  topk = 16,
  minSim = 0
): Promise<EmbeddingLookup> {
  const q = new URLSearchParams({
    token,
    topk: String(topk),
    min_sim: String(minSim),
  });
  const res = await fetch(`${API}/embedding/lookup?${q}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface AssocRelated {
  phrase: string;
  available: boolean;
  pairs: { phrase: string; pmi: number }[];
  n_total: number;
}

export async function fetchAssocRelated(phrase: string, topk = 10): Promise<AssocRelated> {
  const res = await fetch(`${API}/assoc/related?phrase=${encodeURIComponent(phrase)}&topk=${topk}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchEmbeddingCompare(a: string, b: string): Promise<EmbeddingCompare> {
  const q = new URLSearchParams({ a, b });
  const res = await fetch(`${API}/embedding/compare?${q}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchEmbeddingSearch(query: string, limit = 40): Promise<EmbeddingSearch> {
  const q = new URLSearchParams({ q: query, limit: String(limit) });
  const res = await fetch(`${API}/embedding/search?${q}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// --- TK_015/016: sense-neuron tách nghĩa (từ điển D1) ---
export interface SenseEntry {
  sense: string;
  id: number;
  pos: string[];
  neighbors: EmbeddingNeighbor[];
}

export interface EmbeddingSenses {
  word: string;
  token_found: boolean;
  token_blur?: EmbeddingNeighbor[];
  senses: SenseEntry[];
  n_senses: number;
}

export async function fetchEmbeddingSenses(word: string, topk = 8): Promise<EmbeddingSenses> {
  const q = new URLSearchParams({ word, topk: String(topk) });
  const res = await fetch(`${API}/embedding/senses?${q}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


// --- TK_020: cạnh ngang XA (FRAME_BRACKET) — PMI + power + gap-hist + provenance ---
export interface HLinkRow {
  closer: string;
  pmi: number;
  power?: number;
  mode_gap?: number | null;
  p_mode?: number;
  gaps?: Record<string, number>;
  examples?: string[];
}

export interface HLinks {
  found: boolean;
  token: string;
  id?: number;
  links?: HLinkRow[];
}

export async function fetchHLinks(token: string, topk = 12): Promise<HLinks> {
  const q = new URLSearchParams({ token, topk: String(topk) });
  const res = await fetch(`${API}/hlinks?${q}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
