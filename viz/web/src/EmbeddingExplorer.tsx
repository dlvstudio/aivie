import { useCallback, useEffect, useState } from "react";
import {
  fetchAssocRelated,
  type AssocRelated,
  fetchEmbeddingCompare,
  fetchEmbeddingInfo,
  fetchEmbeddingLookup,
  fetchEmbeddingSearch,
  fetchEmbeddingSenses,
  fetchHLinks,
  type EmbeddingCompare,
  type EmbeddingInfo,
  type EmbeddingLookup,
  type EmbeddingSenses,
  type HLinks,
  type EmbeddingTokenRow,
} from "./api";

function VectorBars({
  values,
  dims,
  title,
}: {
  values: number[];
  dims?: number[];
  title: string;
}) {
  const max = Math.max(...values.map(Math.abs), 1e-6);
  return (
    <div className="emb-vector">
      <h3 className="panel-title">{title}</h3>
      <div className="emb-vector-grid">
        {values.map((v, i) => (
          <div className="emb-dim" key={dims?.[i] ?? i}>
            <div
              className={`emb-dim-bar ${v < 0 ? "neg" : ""}`}
              style={{ height: `${Math.max(4, (Math.abs(v) / max) * 100)}%` }}
              title={`d${dims?.[i] ?? i}: ${v}`}
            />
            <span className="emb-dim-label">{dims?.[i] ?? i}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function NeighborTable({
  rows,
  onPick,
}: {
  rows: { id: number; token: string; similarity: number }[];
  onPick: (token: string) => void;
}) {
  const max = rows[0]?.similarity ?? 1;
  return (
    <table className="emb-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Token</th>
          <th>cos</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={r.id}>
            <td className="muted">{i + 1}</td>
            <td>
              <button type="button" className="emb-token-link" onClick={() => onPick(r.token)}>
                {r.token}
              </button>
            </td>
            <td>
              <div className="emb-sim-cell">
                <div className="emb-sim-track">
                  <div
                    className="emb-sim-fill"
                    style={{ width: `${Math.max(2, (r.similarity / max) * 100)}%` }}
                  />
                </div>
                <span>{r.similarity.toFixed(3)}</span>
              </div>
            </td>
            <td className="muted">id {r.id}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function EmbeddingExplorer() {
  const [info, setInfo] = useState<EmbeddingInfo | null>(null);
  const [token, setToken] = useState("Nội");
  const [lookup, setLookup] = useState<EmbeddingLookup | null>(null);
  const [compareA, setCompareA] = useState("thủ đô");
  const [compareB, setCompareB] = useState("Hà Nội");
  const [compare, setCompare] = useState<EmbeddingCompare | null>(null);
  const [assocQ, setAssocQ] = useState("thể dục");
  const [assoc, setAssoc] = useState<AssocRelated | null>(null);
  const [searchQ, setSearchQ] = useState("");
  const [searchHits, setSearchHits] = useState<EmbeddingTokenRow[]>([]);
  const [senseWord, setSenseWord] = useState("cân");
  const [senses, setSenses] = useState<EmbeddingSenses | null>(null);
  const [hlinkTok, setHlinkTok] = useState("nếu");
  const [hlinks, setHlinks] = useState<HLinks | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchEmbeddingInfo()
      .then(setInfo)
      .catch((e) => setError(String(e)));
  }, []);

  const runLookup = useCallback(async (t: string) => {
    const q = t.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    setToken(q);
    try {
      const res = await fetchEmbeddingLookup(q, 20, 0);
      setLookup(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const runCompare = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchEmbeddingCompare(compareA.trim(), compareB.trim());
      setCompare(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [compareA, compareB]);

  const runAssoc = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setAssoc(await fetchAssocRelated(assocQ.trim(), 10));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [assocQ]);

  const runSenses = useCallback(async (w: string) => {
    const q = w.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    setSenseWord(q);
    try {
      setSenses(await fetchEmbeddingSenses(q, 8));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const runHlinks = useCallback(async (t: string) => {
    const q = t.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    setHlinkTok(q);
    try {
      setHlinks(await fetchHLinks(q, 12));
      setSenses(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const runSearch = useCallback(async (q: string) => {
    setSearchQ(q);
    try {
      const res = await fetchEmbeddingSearch(q, 50);
      setSearchHits(res.tokens);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    if (info?.available) runLookup(token);
  }, [info?.available]); // eslint-disable-line react-hooks/exhaustive-deps

  const pickToken = useCallback(
    (t: string) => {
      setToken(t);
      runLookup(t);
    },
    [runLookup]
  );

  return (
    <div className="workspace embedding-workspace">
      <aside className="panel panel-left">
        <h2 className="panel-title">Ma trận E</h2>
        {info && !info.available && (
          <p className="error">{info.message ?? "Không có embedding."}</p>
        )}
        {info?.available && (
          <dl className="inspector-dl emb-meta">
            <dt>Kích thước</dt>
            <dd>
              {info.vocab_size} × {info.dim}
            </dd>
            <dt>Hàng khác 0</dt>
            <dd>{info.nonzero_rows}</dd>
            <dt>‖v‖ trung bình</dt>
            <dd>{info.norm_mean?.toFixed(4)}</dd>
            <dt>‖v‖ max</dt>
            <dd>{info.norm_max?.toFixed(4)}</dd>
          </dl>
        )}

        <div className="panel-block">
          <label className="field-label">Tra token</label>
          <div className="emb-inline">
            <input
              className="field-input"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runLookup(token)}
              placeholder="vd. Hà Nội"
            />
            <button
              type="button"
              className="btn-primary emb-btn-inline"
              disabled={loading || !info?.available}
              onClick={() => runLookup(token)}
            >
              Tra
            </button>
          </div>
        </div>

        <div className="panel-block">
          <label className="field-label">Tìm trong vocab</label>
          <input
            className="field-input"
            value={searchQ}
            onChange={(e) => runSearch(e.target.value)}
            placeholder="substring…"
          />
          <ul className="emb-search-list">
            {searchHits.map((h) => (
              <li key={h.id}>
                <button type="button" className="emb-token-link" onClick={() => pickToken(h.token)}>
                  {h.token}
                </button>
                {h.norm != null && <span className="muted">‖{h.norm.toFixed(2)}‖</span>}
              </li>
            ))}
          </ul>
        </div>

        <div className="panel-block">
          <h3 className="panel-title">Cạnh ngang xa (TK_020)</h3>
          <p className="muted" style={{ margin: "0 0 0.4rem", fontSize: "0.78rem" }}>
            opener → closer: PMI · power · KHOẢNG CÁCH (gap) · cha nguồn (provenance).
          </p>
          <div className="emb-inline">
            <input
              className="field-input"
              value={hlinkTok}
              onChange={(e) => setHlinkTok(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runHlinks(hlinkTok)}
              placeholder="vd. nếu, tuy, càng, vì"
            />
            <button
              type="button"
              className="btn-primary emb-btn-inline"
              disabled={loading}
              onClick={() => runHlinks(hlinkTok)}
            >
              Xem cạnh
            </button>
          </div>
        </div>

        <div className="panel-block">
          <h3 className="panel-title">Tách nghĩa (từ điển D1)</h3>
          <p className="muted" style={{ margin: "0 0 0.4rem", fontSize: "0.78rem" }}>
            Token đa nghĩa = 1 vector bị TRỘN; mỗi nơ-ron nghĩa có POS + cụm hàng xóm riêng.
          </p>
          <div className="emb-inline">
            <input
              className="field-input"
              value={senseWord}
              onChange={(e) => setSenseWord(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runSenses(senseWord)}
              placeholder="vd. cân, đá, đông"
            />
            <button
              type="button"
              className="btn-primary emb-btn-inline"
              disabled={loading || !info?.available}
              onClick={() => runSenses(senseWord)}
            >
              Xem nghĩa
            </button>
          </div>
        </div>

        <div className="panel-block">
          <h3 className="panel-title">So sánh cặp</h3>
          <input
            className="field-input"
            value={compareA}
            onChange={(e) => setCompareA(e.target.value)}
            placeholder="token A"
          />
          <input
            className="field-input"
            style={{ marginTop: "0.35rem" }}
            value={compareB}
            onChange={(e) => setCompareB(e.target.value)}
            placeholder="token B"
          />
          <button
            type="button"
            className="btn-primary"
            style={{ marginTop: "0.5rem" }}
            disabled={loading || !info?.available}
            onClick={runCompare}
          >
            Cosine
          </button>
          {compare?.found && (
            <p className="emb-compare-result">
              cos(<strong>{compare.token_a}</strong>, <strong>{compare.token_b}</strong>) ={" "}
              <span className="emb-sim-big">{compare.similarity?.toFixed(4)}</span>
            </p>
          )}
          {compare && !compare.found && (
            <p className="error">Không tìm thấy một hoặc cả hai token trong vocab.</p>
          )}
        </div>

        <div className="panel-block">
          <h3 className="panel-title">Liên tưởng cụm (đồng-episode)</h3>
          <input
            className="field-input"
            value={assocQ}
            onChange={(e) => setAssocQ(e.target.value)}
            placeholder="cụm 2 từ, vd: thể dục"
          />
          <button
            type="button"
            className="btn-primary"
            style={{ marginTop: "0.5rem" }}
            disabled={loading}
            onClick={runAssoc}
          >
            Tra liên tưởng
          </button>
          {assoc && !assoc.available && (
            <p className="error">Checkpoint chưa build phrase_assoc (TK_008).</p>
          )}
          {assoc?.available && (
            <>
              <p className="emb-compare-result">
                «{assoc.phrase}» · kho {assoc.n_total.toLocaleString()} cặp
              </p>
              {assoc.pairs.length === 0 ? (
                <p className="error">Không có liên tưởng (df cụm thấp — trần data).</p>
              ) : (
                <ul className="emb-assoc-list">
                  {assoc.pairs.map((r: { phrase: string; pmi: number }) => (
                    <li key={r.phrase}>
                      <strong>{r.phrase}</strong>{" "}
                      <span className="emb-sim-big">{r.pmi.toFixed(1)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>

        {error && <p className="error">{error}</p>}
      </aside>

      <main className="emb-main">
        {hlinks ? (
          <div className="senses-view">
            <div className="emb-header">
              <h2>
                Cạnh ngang: <code>{hlinks.token}</code>
                <span className="muted"> · {hlinks.links?.length ?? 0} closer</span>
              </h2>
              <button type="button" className="btn-ghost" onClick={() => setHlinks(null)}>
                × đóng
              </button>
            </div>
            {!hlinks.found && <p className="error">Không có token này trong vocab.</p>}
            <div className="sense-grid">
              {(hlinks.links ?? []).map((l) => (
                <div className="sense-card" key={l.closer}>
                  <div className="sense-card-head">
                    <strong>→ {l.closer}</strong>
                    <span className="tag pos-tag">PMI {l.pmi}</span>
                    {l.power != null && <span className="tag">power {l.power}</span>}
                    {l.mode_gap != null && (
                      <span className="tag surface">
                        gap {l.mode_gap} ({Math.round((l.p_mode ?? 0) * 100)}%)
                      </span>
                    )}
                  </div>
                  {l.gaps && (
                    <div className="sense-chips">
                      {Object.entries(l.gaps).map(([g, c]) => (
                        <span key={g} className="tag">
                          d{g}:{c}
                        </span>
                      ))}
                    </div>
                  )}
                  {(l.examples ?? []).map((ex, i) => (
                    <p key={i} className="muted" style={{ fontSize: "0.76rem", margin: "0.3rem 0 0" }}>
                      « {ex}… »
                    </p>
                  ))}
                </div>
              ))}
            </div>
          </div>
        ) : senses ? (
          <div className="senses-view">
            <div className="emb-header">
              <h2>
                Tách nghĩa: <code>{senses.word}</code>
                <span className="muted"> · {senses.n_senses} nơ-ron nghĩa</span>
              </h2>
              <button type="button" className="btn-ghost" onClick={() => setSenses(null)}>
                × đóng
              </button>
            </div>
            {senses.token_found && senses.token_blur && (
              <div className="sense-card sense-blur">
                <div className="sense-card-head">
                  <strong>token đơn «{senses.word}»</strong>
                  <span className="tag">BLUR — 1 vector trộn mọi nghĩa</span>
                </div>
                <div className="sense-chips">
                  {senses.token_blur.map((n) => (
                    <button
                      key={n.id}
                      type="button"
                      className="tag surface emb-token-link"
                      onClick={() => pickToken(n.token)}
                    >
                      {n.token} <span className="muted">{n.similarity.toFixed(2)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="sense-grid">
              {senses.senses.map((s) => (
                <div className="sense-card" key={s.id}>
                  <div className="sense-card-head">
                    <strong>{s.sense}</strong>
                    {s.pos.map((p) => (
                      <span key={p} className="tag pos-tag">
                        {p}
                      </span>
                    ))}
                  </div>
                  {s.neighbors.length ? (
                    <div className="sense-chips">
                      {s.neighbors.map((n) => (
                        <button
                          key={n.id}
                          type="button"
                          className="tag surface emb-token-link"
                          onClick={() => pickToken(n.token)}
                        >
                          {n.token} <span className="muted">{n.similarity.toFixed(2)}</span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="muted" style={{ fontSize: "0.78rem", margin: 0 }}>
                      (nghĩa thưa — từ điển ít ví dụ; sổ nợ D1′)
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : lookup?.found ? (
          <>
            <div className="emb-header">
              <h2>
                <code>{lookup.token}</code>
                <span className="muted"> id {lookup.id}</span>
              </h2>
              <span className="muted">
                ‖v‖ = {lookup.norm?.toFixed(4)}
                {lookup.parent_df != null && lookup.parent_df > 0 && (
                  <> · parent df {lookup.parent_df}</>
                )}
              </span>
            </div>
            {lookup.vector_preview && (
              <VectorBars
                title={`Vector (${lookup.vector_preview.length} chiều)`}
                values={lookup.vector_preview}
                dims={lookup.vector_preview_dims}
              />
            )}
            <h3 className="panel-title" style={{ margin: "1rem 0 0.5rem" }}>
              Hàng xóm cosine (top {lookup.neighbors?.length ?? 0})
            </h3>
            {lookup.neighbors && (
              <NeighborTable rows={lookup.neighbors} onPick={pickToken} />
            )}
          </>
        ) : lookup && !lookup.found ? (
          <div className="graph-empty">
            <p>
              Không có token <code>{lookup.query}</code> trong vocab.
            </p>
            {lookup.suggestions && lookup.suggestions.length > 0 && (
              <>
                <p className="muted">Gợi ý:</p>
                <div className="tags">
                  {lookup.suggestions.map((s) => (
                    <button
                      key={s}
                      type="button"
                      className="tag surface emb-token-link"
                      onClick={() => pickToken(s)}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="graph-empty">
            {info?.available
              ? "Nhập token để xem hàng embedding và hàng xóm."
              : "Checkpoint chưa có E — chạy train_v3 để build PPMI+SVD."}
          </div>
        )}
      </main>
    </div>
  );
}
