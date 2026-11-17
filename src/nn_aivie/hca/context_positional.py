"""
Ngữ cảnh có vị trí cho next-token (Hebbian).

q / micro không chỉ từ w0 — chồng chập λ^(T-1-i) trên cả history.
Luật 0: chỉ đọc A/E đã học, không hardcode từ vựng hub.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nn_aivie.core.edge_kinds import EdgeKind


def positional_weights(n: int, lam: float) -> np.ndarray:
    """Token cuối weight=1; lùi về trước ×λ. Shape (n,) theo thứ tự history."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    return lam ** np.arange(n - 1, -1, -1, dtype=np.float64)


def _is_structural_uncached(vocab, tid: int) -> bool:
    if not vocab.is_surface(tid):
        return True
    tok = vocab.token_of(tid).lstrip("Ġ")
    if not tok or tok.startswith("Ċ") or tok in ("\n", "\r"):
        return True
    if all(not ch.isalnum() for ch in tok):
        return True
    return False


def is_structural_token(vocab, tid: int) -> bool:
    """Punct / newline / non-surface — không nên neo W_seq (hub Ċ→Giải).

    §195 — CACHE mảng bool [vocab]: hàm thuần theo (vocab, tid), gọi 200k+ lần
    trong vòng sinh. Dựng một lần (32k token, ~ms) rồi index — bỏ 200k lần gọi."""
    tid = int(tid)
    cache = getattr(vocab, "_struct_cache", None)
    if cache is None:
        n = len(vocab)
        cache = np.fromiter(
            (_is_structural_uncached(vocab, i) for i in range(n)),
            dtype=bool, count=n,
        )
        try:
            vocab._struct_cache = cache  # noqa: SLF001
        except Exception:
            return _is_structural_uncached(vocab, tid)
    if 0 <= tid < len(cache):
        return bool(cache[tid])
    return _is_structural_uncached(vocab, tid)


def _add_seq_row(
    micro: np.ndarray,
    state,
    tid: int,
    weight: float,
    *,
    vocab_size: int,
    use_ppmi: bool,
) -> None:
    if weight <= 0:
        return
    tid = int(tid)
    if use_ppmi:
        from nn_aivie.hca.attention_qk import ensure_directed_ppmi

        row = ensure_directed_ppmi(state).getrow(tid)
        for v, c in zip(row.indices, row.data):
            if 0 <= int(v) < vocab_size and c > 0:
                micro[int(v)] += weight * float(c)
        return
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    for v, c in seq.row(tid).items():
        if 0 <= v < vocab_size and c > 0:
            micro[v] += weight * float(c)


def positional_wseq_micro(
    state,
    history_ids: list[int],
    *,
    vocab_size: int,
    lam: float = 0.85,
    ctx_len: int = 12,
    use_ppmi: bool = False,
    skip_structural: bool = False,
) -> np.ndarray:
    """micro[v] = Σ_i λ^{T-1-i} · seq(history_i → v); seq = PPMI hoặc W_seq thô."""
    micro = np.zeros(vocab_size, dtype=np.float64)
    if not history_ids:
        return micro
    ctx = history_ids[-ctx_len:]
    weights = positional_weights(len(ctx), lam)
    used = 0
    for tid, w in zip(ctx, weights):
        if skip_structural and is_structural_token(state.vocab, tid):
            continue
        _add_seq_row(
            micro, state, tid, w, vocab_size=vocab_size, use_ppmi=use_ppmi
        )
        used += 1
    # Cửa sổ toàn punct → neo từ content gần nhất trong history.
    if used == 0 and skip_structural:
        for tid in reversed(history_ids):
            if is_structural_token(state.vocab, tid):
                continue
            _add_seq_row(
                micro, state, tid, 1.0, vocab_size=vocab_size, use_ppmi=use_ppmi
            )
            break
    mmax = micro.max()
    if mmax > 0:
        micro /= mmax
    return micro


def positional_readout_base(
    state,
    history_ids: list[int],
    *,
    vocab_size: int,
    lam: float = 0.85,
    ctx_len: int = 12,
) -> np.ndarray:
    """base readout chồng chập theo vị trí (cùng λ)."""
    base = np.zeros(vocab_size, dtype=np.float64)
    if not history_ids:
        return base
    ctx = history_ids[-ctx_len:]
    weights = positional_weights(len(ctx), lam)
    for tid, w in zip(ctx, weights):
        if w <= 0:
            continue
        idx, val = _readout_arrays(state.adjacency, int(tid), vocab_size)
        if idx.size:
            base[idx] += w * val
    return base


_RO_ARR: dict = {}


def _readout_arrays(adj, tid: int, vocab_size: int):
    """TK_045 §12.36 — dạng MẢNG của `row_readout(tid)`, nhớ lại giữa các bước.

    `positional_readout_base` là hotspot #1 còn lại trên v12 sau §12.32–§12.34: profile M4 cho
    **0.531s/12 bước = 44ms mỗi token**, gần như toàn bộ là `tottime` ⇒ vòng Python tự thân.
    Vòng cũ duyệt `.items()` của dict readout: với token phổ biến trên 430k cha, dict đó rất lớn,
    và ta duyệt lại nó MỖI TOKEN × 12 vị trí ngữ cảnh.

    KHÔNG dùng `readout_csr_normalized` sẵn có: ma trận đó **chuẩn hoá theo hàng**, còn hàm này
    cần ĐẾM THÔ ⇒ dùng sẽ đổi hành vi âm thầm.

    Vô hiệu hoá: `row_readout` memo theo hàng và trả dict MỚI khi hàng bị ghi lại; ta giữ tham
    chiếu tới đúng dict đó và so `is` — khác thì dựng lại. Không phụ thuộc `id()` (xem §12.30).
    """
    key = (id(adj), int(tid))
    hit = _RO_ARR.get(key)
    row = adj.row_readout(int(tid))
    if hit is not None and hit[0] is row:
        return hit[1], hit[2]
    items = [(v, c) for v, c in row.items() if 0 <= v < vocab_size and c > 0]
    if items:
        idx = np.fromiter((v for v, _ in items), dtype=np.int64, count=len(items))
        val = np.fromiter((float(c) for _, c in items), dtype=np.float64, count=len(items))
    else:
        idx = np.zeros(0, dtype=np.int64)
        val = np.zeros(0, dtype=np.float64)
    _RO_ARR[key] = (row, idx, val)
    return idx, val


def positional_wseq_pool(
    state,
    history_ids: list[int],
    *,
    lam: float = 0.85,
    ctx_len: int = 12,
    k_per: int = 24,
    k_total: int = 96,
    use_ppmi: bool = False,
    skip_structural: bool = False,
) -> tuple[list[int], dict[int, float]]:
    """Gộp successor seq từ nhiều token history (có trọng số vị trí)."""
    scores: dict[int, float] = {}
    if not history_ids:
        return [], scores
    ctx = list(history_ids[-ctx_len:])
    weights = positional_weights(len(ctx), lam)
    pairs = list(zip(ctx, weights))
    if skip_structural:
        filt = [
            (tid, w)
            for tid, w in pairs
            if not is_structural_token(state.vocab, tid)
        ]
        if not filt:
            for tid in reversed(history_ids):
                if not is_structural_token(state.vocab, tid):
                    filt = [(tid, 1.0)]
                    break
        pairs = filt
    ppmi = None
    if use_ppmi:
        from nn_aivie.hca.attention_qk import ensure_directed_ppmi

        ppmi = ensure_directed_ppmi(state)
    seq = None if use_ppmi else state.adjacency.layer(EdgeKind.SEQUENTIAL)
    for tid, w in pairs:
        if w <= 0:
            continue
        tid = int(tid)
        if use_ppmi:
            row = ppmi.getrow(tid)
            items = sorted(
                ((int(j), float(c)) for j, c in zip(row.indices, row.data) if c > 0),
                key=lambda t: -t[1],
            )[:k_per]
        else:
            row = seq.row(tid)
            if not row:
                continue
            items = sorted(
                ((int(j), float(c)) for j, c in row.items() if c > 0),
                key=lambda t: -t[1],
            )[:k_per]
        for j, c in items:
            scores[j] = scores.get(j, 0.0) + w * c
    neighbors = sorted(scores, key=scores.get, reverse=True)[:k_total]
    return neighbors, scores


def positional_query(
    wq: np.ndarray,
    history_ids: list[int],
    *,
    lam: float = 0.85,
    ctx_len: int = 48,
) -> np.ndarray:
    """q = Σ_i λ^{T-1-i} Wq[w_i]  (chuẩn hoá L2)."""
    d = int(wq.shape[1])
    q = np.zeros(d, dtype=np.float64)
    if not history_ids:
        return q
    n = int(wq.shape[0])
    ctx = history_ids[-ctx_len:]
    weights = positional_weights(len(ctx), lam)
    for tid, w in zip(ctx, weights):
        tid = int(tid)
        if 0 <= tid < n and w > 0:
            q += w * wq[tid]
    norm = float(np.linalg.norm(q))
    if norm > 1e-12:
        q /= norm
    return q


def content_token_ids(history_ids: list[int], vocab, *, max_n: int = 32) -> list[int]:
    """Token surface trong history — neo ngữ cảnh (bỏ punct ngắn)."""
    out: list[int] = []
    seen: set[int] = set()
    for t in history_ids:
        t = int(t)
        if t in seen:
            continue
        if not vocab.is_surface(t):
            continue
        tok = vocab.token_of(t).lstrip("ĠĊ")
        if len(tok) < 2:
            continue
        if all(not ch.isalnum() for ch in tok):
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_n:
            break
    return out


def ensure_varlm(state, *, sidecar: str | Path | None = None, fallbacks: list[str | Path] | None = None):
    """Lazy-load VarOrderLM sidecar vào state.varlm (n-gram biến-bậc 1..4)."""
    from nn_aivie.core.varlm import VarOrderLM, varlm_sidecar_path

    if getattr(state, "varlm", None) is not None:
        return state.varlm
    candidates: list[Path] = []
    if sidecar:
        candidates.append(Path(sidecar))
    hint = getattr(state, "_varlm_sidecar", None)
    if hint:
        candidates.append(Path(hint))
    ckpt = getattr(state, "_checkpoint_path", None)
    if ckpt:
        candidates.append(varlm_sidecar_path(ckpt))
    for fb in fallbacks or ():
        candidates.append(Path(fb))
    candidates.append(Path("checkpoints/network_v11.varlm.pkl"))
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            try:
                state.varlm = VarOrderLM.load(p)
                return state.varlm
            except Exception:  # noqa: BLE001
                continue
    return None


def ngram_varlm_micro(
    state,
    history_ids: list[int],
    *,
    vocab_size: int,
) -> np.ndarray:
    """VarOrderLM (ctx 1..4): điểm next-token theo chuỗi khớp dài nhất — nhìn xa hơn bigram."""
    out = np.zeros(vocab_size, dtype=np.float64)
    varlm = getattr(state, "varlm", None)
    if varlm is None or not history_ids:
        return out
    ctx = history_ids
    if getattr(varlm, "use_clusters", False) and state.parents is not None:
        ctx = state.parents.seg.encode_tokens_with_clusters(history_ids)
    pp = varlm.scores(ctx[-varlm.max_ctx :])
    if not pp:
        return out
    pmax = max(pp.values()) or 1.0
    for v, p in pp.items():
        if 0 <= int(v) < vocab_size:
            out[int(v)] = float(p) / pmax
    return out


def order2_micro(
    state,
    history_ids: list[int],
    *,
    vocab_size: int,
    min_hi: float = 2.0,
) -> np.ndarray:
    """Trigram-cha (prev, cur)→next qua ParentStore.seq_next — bậc 2/3."""
    out = np.zeros(vocab_size, dtype=np.float64)
    if state.parents is None or len(history_ids) < 1:
        return out
    cur = int(history_ids[-1])
    prev = int(history_ids[-2]) if len(history_ids) >= 2 else None
    hi = state.parents.seq_next(prev, cur, min_hi=min_hi)
    if not hi:
        return out
    hmax = max(hi.values()) or 1.0
    for v, c in hi.items():
        if 0 <= int(v) < vocab_size:
            out[int(v)] = float(c) / hmax
    return out


def _suppress_structural_targets(out: np.ndarray, vocab) -> None:
    """Distant/neo không nên đẩy punct/newline lên đỉnh."""
    for i in range(int(out.shape[0])):
        if out[i] > 0 and is_structural_token(vocab, i):
            out[i] = 0.0
    mmax = float(out.max()) if out.size else 0.0
    if mmax > 0:
        out /= mmax


def distant_content_wseq_micro(
    state,
    content_ids: list[int],
    *,
    vocab_size: int,
) -> np.ndarray:
    """W_seq thô từ mọi content (legacy) — dễ kéo hub cái/số; ưu tiên ppmi/coact."""
    out = np.zeros(vocab_size, dtype=np.float64)
    if not content_ids:
        return out
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    for tid in content_ids:
        for v, c in seq.row(int(tid)).items():
            if 0 <= v < vocab_size and c > 0:
                out[v] += float(c)
    _suppress_structural_targets(out, state.vocab)
    return out


def distant_content_coact_micro(
    state,
    content_ids: list[int],
    *,
    vocab_size: int,
) -> np.ndarray:
    """CO_ACT từ content prompt — neo chủ đề/entity (đã lọc punct)."""
    out = np.zeros(vocab_size, dtype=np.float64)
    if not content_ids:
        return out
    co = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
    for tid in content_ids:
        for v, c in co.row(int(tid)).items():
            if 0 <= v < vocab_size and c > 0:
                out[v] += float(c)
    _suppress_structural_targets(out, state.vocab)
    return out


def distant_content_ppmi_micro(
    state,
    content_ids: list[int],
    *,
    vocab_size: int,
) -> np.ndarray:
    """PPMI(W_seq) từ content — collocates có hướng thay hub raw."""
    out = np.zeros(vocab_size, dtype=np.float64)
    if not content_ids:
        return out
    from nn_aivie.hca.attention_qk import ensure_directed_ppmi

    ppmi = ensure_directed_ppmi(state)
    for tid in content_ids:
        row = ppmi.getrow(int(tid))
        for v, c in zip(row.indices, row.data):
            if 0 <= int(v) < vocab_size and c > 0:
                out[int(v)] += float(c)
    _suppress_structural_targets(out, state.vocab)
    return out


def distant_content_e_micro(
    state,
    content_ids: list[int],
    *,
    vocab_size: int,
    top_k: int = 256,
) -> np.ndarray:
    """Cosine E tới content — chỉ top_k ứng viên gần nhất (không full-vocab)."""
    out = np.zeros(vocab_size, dtype=np.float64)
    if not content_ids or state.embedding is None:
        return out
    emb = state.embedding.matrix
    n = min(vocab_size, int(emb.shape[0]))
    C = emb[np.asarray(content_ids, dtype=np.int64)]
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    q = C.mean(axis=0)
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    sims = emb[:n] @ q
    k = min(top_k, n)
    idx = np.argpartition(-sims, k - 1)[:k]
    smax = float(sims[idx].max()) if k else 0.0
    if smax <= 0:
        return out
    for j in idx:
        s = float(sims[j])
        if s > 0:
            out[int(j)] = s / smax
    _suppress_structural_targets(out, state.vocab)
    return out


def distant_content_micro(
    state,
    content_ids: list[int],
    *,
    vocab_size: int,
    mode: str = "ppmi",
) -> np.ndarray:
    """Neo content xa theo mode: ppmi | coact | e | wseq."""
    m = (mode or "ppmi").lower()
    if m == "coact":
        return distant_content_coact_micro(
            state, content_ids, vocab_size=vocab_size
        )
    if m == "e":
        return distant_content_e_micro(state, content_ids, vocab_size=vocab_size)
    if m == "wseq":
        return distant_content_wseq_micro(
            state, content_ids, vocab_size=vocab_size
        )
    return distant_content_ppmi_micro(
        state, content_ids, vocab_size=vocab_size
    )


def apply_content_context_gate(
    scores: np.ndarray,
    state,
    content_ids: list[int],
    *,
    floor: float = 0.12,
    top_probe: int = 64,
) -> None:
    """Nhân score với (floor + (1-floor)·rel); rel = max cos(E) tới content prompt.

    Hạ hub không liên quan ngữ cảnh (vd template từ điển sau Ċ) — không hardcode tên.
    """
    if not content_ids or state.embedding is None:
        return
    emb = state.embedding.matrix
    n = min(int(scores.shape[0]), int(emb.shape[0]))
    finite = np.flatnonzero(np.isfinite(scores[:n]) & (scores[:n] > 0))
    if finite.size == 0:
        return
    if finite.size > top_probe:
        part = finite[np.argpartition(-scores[finite], top_probe - 1)[:top_probe]]
    else:
        part = finite
    C = emb[np.asarray(content_ids, dtype=np.int64)]
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    V = emb[part]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    rel = np.clip((V @ C.T).max(axis=1), 0.0, 1.0)
    scores[part] *= floor + (1.0 - floor) * rel
