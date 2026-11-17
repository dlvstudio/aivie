"""
W_know — succession chắt từ nơ-ron cha (+ Oja-style kìm hub).

Luật 0: chỉ đọc ParentStore đã học (emit/answer + power), không hardcode từ.
Khác W_seq toàn cục: chỉ bigram/trật tự trong chuỗi cha → ít power-hub `để/là`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from nn_aivie.core.state import GraphState
from nn_aivie.hca.objcache import cache_get, cache_put


_KNOW_CACHE: dict[int, csr_matrix] = {}


def build_w_know_csr(state: GraphState, *, oja_row: bool = True) -> csr_matrix:
    """CSR bigram từ chuỗi cha (power-weighted). Oja: chuẩn hoá L2 từng hàng."""
    ps = state.parents
    n = len(state.vocab)
    if ps is None:
        return csr_matrix((n, n), dtype=np.float64)
    ps._ensure_seq_index()  # noqa: SLF001
    bi = getattr(ps, "_seq_bi", {}) or {}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i, dest in bi.items():
        i = int(i)
        if i < 0 or i >= n:
            continue
        items = [(int(j), float(c)) for j, c in dest.items() if c > 0 and 0 <= int(j) < n]
        if not items:
            continue
        if oja_row:
            # Fixed-point Oja (‖w‖₂=1): cạnh tranh nội hàng — hub không nuốt cả hàng.
            norm = float(np.sqrt(sum(c * c for _, c in items))) or 1.0
            items = [(j, c / norm) for j, c in items]
        for j, c in items:
            rows.append(i)
            cols.append(j)
            vals.append(c)
    if not vals:
        return csr_matrix((n, n), dtype=np.float64)
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)


def ensure_w_know(state: GraphState, *, oja_row: bool = True) -> csr_matrix:
    cached = cache_get(_KNOW_CACHE, state)
    if cached is None:
        cached = cache_put(_KNOW_CACHE, state,
                           build_w_know_csr(state, oja_row=oja_row))
    return cached


def clear_w_know_cache() -> None:
    _KNOW_CACHE.clear()


def positional_know_micro(
    state: GraphState,
    history_ids: list[int],
    *,
    vocab_size: int,
    lam: float = 0.85,
    ctx_len: int = 12,
    oja_row: bool = True,
) -> np.ndarray:
    """Σ λ·W_know[history_i → ·] — fluency-from-parents, đã Oja theo hàng."""
    out = np.zeros(vocab_size, dtype=np.float64)
    if not history_ids or state.parents is None:
        return out
    from nn_aivie.hca.context_positional import (
        is_structural_token,
        positional_weights,
    )

    know = ensure_w_know(state, oja_row=oja_row)
    ctx = history_ids[-ctx_len:]
    weights = positional_weights(len(ctx), lam)
    used = 0
    for tid, w in zip(ctx, weights):
        if w <= 0 or is_structural_token(state.vocab, tid):
            continue
        row = know.getrow(int(tid))
        for j, c in zip(row.indices, row.data):
            if 0 <= int(j) < vocab_size and c > 0:
                out[int(j)] += w * float(c)
        used += 1
    if used == 0:
        for tid in reversed(history_ids):
            if is_structural_token(state.vocab, tid):
                continue
            row = know.getrow(int(tid))
            for j, c in zip(row.indices, row.data):
                if 0 <= int(j) < vocab_size and c > 0:
                    out[int(j)] += float(c)
            break
    mmax = float(out.max()) if out.size else 0.0
    if mmax > 0:
        out /= mmax
    return out


def parent_span_next_micro(
    state: GraphState,
    history_ids: list[int],
    *,
    vocab_size: int,
    top_n: int = 24,
    min_span: int = 3,
    max_span: int = 24,
) -> np.ndarray:
    """Next-token từ cha: khớp suffix history trong emit/answer → token kế (chắt fact).

    Dùng `_index` token→cha (không `rank_matches` full) — đủ nhanh mỗi bước AR.
    """
    out = np.zeros(vocab_size, dtype=np.float64)
    ps = state.parents
    if ps is None or len(history_ids) < min_span:
        return out
    hist = list(history_ids)
    # Ứng viên: giao posting của vài token cuối (ưu tiên token hiếm hơn = posting ngắn).
    tail = hist[-min(8, len(hist)) :]
    postings: list[list[int]] = []
    for t in tail:
        lst = ps._index.get(int(t))  # noqa: SLF001
        if lst:
            postings.append(lst)
    if not postings:
        return out
    postings.sort(key=len)
    cand = set(postings[0])
    for lst in postings[1:]:
        s = set(lst)
        inter = cand & s
        if len(inter) >= 3:
            cand = inter
        if len(cand) <= top_n:
            break
    if len(cand) > top_n * 4:
        # cắt theo độ dài posting token cuối
        last_list = ps._index.get(int(hist[-1]), [])  # noqa: SLF001
        cand = set(last_list[: top_n * 4]) & cand if last_list else set(list(cand)[: top_n * 4])
    max_l = min(len(hist), max_span)
    scored: list[tuple[int, float]] = []
    for pi in cand:
        rec = ps.parents[int(pi)]
        seq = list(
            rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids
        )
        if len(seq) < min_span + 1:
            continue
        best_next: int | None = None
        best_len = 0
        upper = min(max_l, len(seq) - 1)
        for L in range(upper, min_span - 1, -1):
            suf = hist[-L:]
            for start in range(len(seq) - L, -1, -1):
                if seq[start : start + L] == suf:
                    best_next = int(seq[start + L])
                    best_len = L
                    break
            if best_next is not None:
                break
        if best_next is None or not (0 <= best_next < vocab_size):
            continue
        w = float(max(1, rec.power)) * (best_len / float(upper or 1))
        scored.append((best_next, w))
    if not scored:
        return out
    scored.sort(key=lambda t: -t[1])
    for nxt, w in scored[:top_n]:
        out[nxt] += w
    mmax = float(out.max()) if out.size else 0.0
    if mmax > 0:
        out /= mmax
    return out


def oja_compete(scores: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Mềm hoá hub: y' = y / (1 + β y) — cùng tinh thần hạng tử −y² của Oja."""
    if beta <= 0:
        return scores
    s = np.maximum(scores, 0.0)
    return s / (1.0 + beta * s)


def apply_gidf_target_gate(
    scores: np.ndarray,
    state: GraphState,
    *,
    weight: float = 0.4,
    floor: float = 0.15,
) -> None:
    """Nhân điểm đích với gidf đã chuẩn hoá — hạ power-token (để/là/có)."""
    ps = state.parents
    if ps is None or weight <= 0:
        return
    n = min(int(scores.shape[0]), len(state.vocab))
    finite = np.flatnonzero(np.isfinite(scores[:n]) & (scores[:n] > 0))
    if finite.size == 0:
        return
    g = np.array([float(ps._gidf(int(i))) for i in finite], dtype=np.float64)  # noqa: SLF001
    gmax = float(g.max()) if g.size else 0.0
    if gmax <= 0:
        return
    g /= gmax
    scores[finite] *= floor + (1.0 - floor) * (weight * g + (1.0 - weight))


def save_w_know_sidecar(state: GraphState, path: str | Path, *, oja_row: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = ensure_w_know(state, oja_row=oja_row)
    np.savez_compressed(
        path,
        data=m.data,
        indices=m.indices,
        indptr=m.indptr,
        shape=np.asarray(m.shape, dtype=np.int64),
    )
    return path
