"""PHA 1 (§12.114) — embedding E dựng từ W_skip thay lớp CO_ACTIVATION.

Luận đề lộ trình: embedding là một VIEW của W_skip — `Σ_d λ^{d-1} (W_skip[d] + W_skipᵀ[d])` đối
xứng hoá → PPMI → SVD. Khác `core/embedding.build_ppmi` (nguồn = lớp CO_ACTIVATION) DUY NHẤT ở
NGUỒN; giữ nguyên công thức PPMI (context-distribution smoothing 0.75, Levy 2015) để so sạch.

Đặt ở tầng `hca` (không phải `core/embedding.py`) vì cần `iter_sentence_spans`/`_content_positions`
của `hca.sent_intra` — nhập `hca`→`core` được, `core`→`hca` thì vòng. `core.embedding` chỉ cho
mượn `truncated_svd` + `GraphEmbedding` (chiều `core`→`core`, an toàn).

Sạch Luật 0: đếm đồng-hiện theo khoảng cách + PPMI + SVD, đều closed-form, không backprop.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from nn_aivie.core.embedding import GraphEmbedding, truncated_svd
from nn_aivie.core.state import GraphState
from nn_aivie.hca.context_positional import is_structural_token
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans


def wskip_cooccur(
    state: GraphState,
    *,
    d_max: int = 8,
    lam: float = 0.85,
    content_only: bool = True,
) -> tuple[dict[tuple[int, int], float], dict[int, float], float]:
    """Đếm đồng-hiện ĐỐI XỨNG theo khoảng cách trong câu, giảm `λ^{d-1}`.

    Trả `(c, u, ut)`: c[(a,b)] = khối đồng-hiện; u[t] = tần suất token; ut = tổng.
    Đối xứng hoá (a→b và b→a cùng cộng) vì embedding lo TƯƠNG ĐỒNG, không lo thứ tự.
    """
    v = state.vocab
    n = len(v)
    struct = [is_structural_token(v, i) for i in range(n)]
    c: dict[tuple[int, int], float] = defaultdict(float)
    u: dict[int, float] = defaultdict(float)
    ut = 0.0
    for rec in state.parents.parents:
        sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        for sp in iter_sentence_spans(sq, v):
            tk = [int(t) for t in _content_positions(
                sp, v, content_only=content_only, struct_mask=struct)]
            for t in tk:
                u[t] += 1.0
                ut += 1.0
            L = len(tk)
            for i in range(L):
                for j in range(i + 1, min(L, i + 1 + d_max)):
                    w = lam ** (j - i - 1)
                    c[(tk[i], tk[j])] += w
                    c[(tk[j], tk[i])] += w
    return c, u, ut


def wskip_ppmi(
    state: GraphState,
    *,
    d_max: int = 8,
    lam: float = 0.85,
    content_only: bool = True,
    smoothing: float = 0.75,
) -> csr_matrix:
    """PPMI `[N×N]` từ đồng-hiện W_skip. Công thức khớp `core.embedding.build_ppmi`.

    Context-distribution smoothing: nâng P(context) lên luỹ thừa <1 để hạ thiên lệch hub.
    PPMI[i,j] = max(0, log P(i,j)/(P(i)·P_smooth(j))).
    """
    n = len(state.vocab)
    c, u, _ = wskip_cooccur(state, d_max=d_max, lam=lam, content_only=content_only)
    if not c:
        return csr_matrix((n, n), dtype=np.float64)
    r = np.fromiter((a for (a, _b) in c), dtype=np.int64, count=len(c))
    cc = np.fromiter((b for (_a, b) in c), dtype=np.int64, count=len(c))
    vv = np.fromiter(c.values(), dtype=np.float64, count=len(c))
    mat = coo_matrix((vv, (r, cc)), shape=(n, n)).tocsr()
    total = float(mat.sum())
    if total <= 0:
        return csr_matrix((n, n), dtype=np.float64)
    row_sum = np.asarray(mat.sum(axis=1)).ravel()
    col_sum = np.asarray(mat.sum(axis=0)).ravel()
    col_smooth = np.power(col_sum, smoothing)
    col_smooth_total = col_smooth.sum() or 1.0
    p_i = row_sum / total
    p_j = col_smooth / col_smooth_total
    co = mat.tocoo()
    denom = p_i[co.row] * p_j[co.col]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co.data / total) / denom)
    mask = np.isfinite(pmi) & (denom > 0) & (pmi > 0)
    return csr_matrix((pmi[mask], (co.row[mask], co.col[mask])), shape=(n, n))


def build_eskip(
    state: GraphState,
    *,
    dim: int = 96,
    d_max: int = 8,
    lam: float = 0.85,
    content_only: bool = True,
    smoothing: float = 0.75,
) -> GraphEmbedding:
    """`GraphEmbedding` (N×dim) từ W_skip — thả thẳng vào `state.embedding`."""
    ppmi = wskip_ppmi(state, d_max=d_max, lam=lam,
                      content_only=content_only, smoothing=smoothing)
    return GraphEmbedding(matrix=truncated_svd(ppmi, dim))
