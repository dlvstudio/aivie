"""
TK_017 R2 / doc 03 (D2 lite) — Attention Q/K CÓ HƯỚNG chiết từ đồ thị, không backprop.

    M_seq (PPMI có hướng từ W_seq) ≈ U·Σ·Vᵀ   →   Wq = U√Σ,  Wk = V√Σ
    q_i · k_j ≈ PPMI(i→j)  — kể cả cặp (i,j) CHƯA từng đồng hiện (tổng quát hoá khuôn).

Đúng doc 03 §2: lấy CẢ HAI phía U (vai đứng-trước = query) và V (vai đứng-sau = key) của ma
trận CÓ HƯỚNG — khác E hiện tại (PPMI đối xứng, chỉ U → chỉ biết "giống nhau").

Vai trò trong R2: kênh micro của `_wseq_fallback` chỉ có bigram thô (unseen → 0) + readout
hub-heavy → word-salad `. , và` (đo TK_017 §12). q·k lấp unseen-transitions bằng khuôn chuyển
tiếp. Trung thực (doc 03 §4 + doc 07): type-level, lo "CÁCH nói" — không thay agenda "NÓI GÌ".

Luật 0 sạch: mọi thứ chiết từ A đã học. View offline (build lại được từ A), không train.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.hca.objcache import cache_get, cache_put


def _structural_mask(vocab, n: int) -> np.ndarray:
    """True = punct/newline/non-surface — loại khỏi PPMI (tránh Ċ→Giải/Ví)."""
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not vocab.is_surface(i):
            mask[i] = True
            continue
        tok = vocab.token_of(i).lstrip("Ġ")
        if not tok or tok.startswith("Ċ") or tok in ("\n", "\r"):
            mask[i] = True
        elif all(not ch.isalnum() for ch in tok):
            mask[i] = True
    return mask


def build_directed_ppmi(
    state: GraphState,
    *,
    smoothing: float = 0.75,
    drop_structural: bool = True,
) -> csr_matrix:
    """PPMI thưa CÓ HƯỚNG từ SEQUENTIAL. Mặc định bỏ cạnh chạm punct/newline."""
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    n = len(state.vocab)
    struct = _structural_mask(state.vocab, n) if drop_structural else None
    rows, cols, vals = [], [], []
    for i, j, w in seq.items():
        if w <= 0 or i >= n or j >= n:
            continue
        if struct is not None and (struct[i] or struct[j]):
            continue
        rows.append(i)
        cols.append(j)
        vals.append(w)
    if not vals:
        return csr_matrix((n, n), dtype=np.float64)
    c = coo_matrix((np.asarray(vals, float), (rows, cols)), shape=(n, n)).tocsr()
    total = float(c.sum())
    row_sum = np.asarray(c.sum(axis=1)).ravel()
    col_sum = np.asarray(c.sum(axis=0)).ravel()
    col_smooth = np.power(col_sum, smoothing)
    p_i = row_sum / total
    p_j = col_smooth / (col_smooth.sum() or 1.0)
    coo = c.tocoo()
    denom = p_i[coo.row] * p_j[coo.col]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((coo.data / total) / denom)
    mask = np.isfinite(pmi) & (denom > 0) & (pmi > 0)
    return csr_matrix((pmi[mask], (coo.row[mask], coo.col[mask])), shape=(n, n))


class DirectedQK:
    """Wq/Wk hai phía từ SVD ma trận có hướng. `scores(i)` = q_i · Kᵀ (≈ PPMI(i→·))."""

    def __init__(self, wq: np.ndarray, wk: np.ndarray) -> None:
        self.wq = wq  # (N, d) — vai đứng-trước
        self.wk = wk  # (N, d) — vai đứng-sau

    def scores(self, i: int) -> np.ndarray:
        if i < 0 or i >= self.wq.shape[0]:
            return np.zeros(self.wk.shape[0], dtype=np.float64)
        return self.wk @ self.wq[i]

    def top(self, vocab, token: str, k: int = 10) -> list[tuple[str, float]]:
        """Đọc tay: token nào hay ĐỨNG SAU `token` theo khuôn (kể cả chưa đồng hiện)."""
        tid = vocab._token_to_id.get(token)
        if tid is None:
            return []
        s = self.scores(tid)
        idx = np.argsort(-s)[: k * 2]
        out = []
        for j in idx:
            out.append((vocab.token_of(int(j)), round(float(s[j]), 2)))
            if len(out) >= k:
                break
        return out


def build_directed_qk(state: GraphState, *, dim: int = 96) -> DirectedQK:
    """SVD hai phía: Wq=U√Σ (hàng), Wk=V√Σ (cột). svds trả tăng dần → đảo giảm dần."""
    ppmi = build_directed_ppmi(state)
    n = ppmi.shape[0]
    k = min(dim, n - 1)
    if k < 1 or ppmi.nnz == 0:
        z = np.zeros((n, dim))
        return DirectedQK(z, z.copy())
    u, s, vt = svds(ppmi.asfptype(), k=k, which="LM")
    order = np.argsort(-s)
    u, s, vt = u[:, order], s[order], vt[order]
    root = np.sqrt(np.maximum(s, 0.0))
    wq = u * root
    wk = vt.T * root
    if wq.shape[1] < dim:  # pad giữ chiều
        wq = np.pad(wq, ((0, 0), (0, dim - wq.shape[1])))
        wk = np.pad(wk, ((0, 0), (0, dim - wk.shape[1])))
    return DirectedQK(wq, wk)


# --- lazy cache per-state (view offline; cùng pattern ensure_frame_brackets) ---
_qk_cache: dict[int, DirectedQK] = {}
_ppmi_cache: dict[int, csr_matrix] = {}


def ensure_directed_ppmi(state: GraphState, *, smoothing: float = 0.75) -> csr_matrix:
    """Cache PPMI có hướng từ W_seq — dùng scoring continuation (không chỉ SVD QK)."""
    cached = cache_get(_ppmi_cache, state)
    if cached is None:
        cached = cache_put(_ppmi_cache, state,
                           build_directed_ppmi(state, smoothing=smoothing))
    return cached


def ensure_directed_qk(state: GraphState, *, dim: int = 96, logger=None) -> DirectedQK:
    qk = cache_get(_qk_cache, state)
    if qk is None:
        if logger is not None:
            logger.log("build directed q·k (SVD W_seq)…")
        qk = cache_put(_qk_cache, state, build_directed_qk(state, dim=dim))
        if logger is not None:
            logger.log(f"✓ q·k built (N={qk.wq.shape[0]}, d={qk.wq.shape[1]})")
    return qk
