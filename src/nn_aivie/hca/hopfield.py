"""TK_045 §13 Trụ D — TRUY HỒI HOPFIELD HIỆN ĐẠI trên `ParentBipartite.A`.

## Quan sát mở đầu

`ParentBipartite.A` **đã là** một bộ nhớ Hopfield: cột `pi` = vector token có trọng số
`sqrt(power·priority)·gidf` của cha `pi` — tức **mẫu lưu**. Ramsauer et al. 2020 chứng minh
attention ≡ **một bước truy hồi Hopfield hiện đại**:

    s = Aᵀq          # điểm khớp mẫu
    p = softmax(β·s) # phân bố trên mẫu
    q' = A p         # HOÀN NGUYÊN mẫu — truy hồi
    (lặp)

Dạng đóng, **không backprop** — cùng họ toán với PPMI+SVD dự án đã nhận (Luật 0 sạch).

## ⚠️ Một bước KHÔNG đổi thứ hạng — nói trước để không bán no-op

`softmax` **đơn điệu** ⇒ xếp theo `p` ≡ xếp theo `s = Aᵀq`. Vậy `steps=1` **không phải**
"Hopfield", nó chỉ là **cosine trên vector IDF**. Nó vẫn đáng đo, nhưng như một **đối chứng**
trả lời câu hỏi riêng: *đống cổng chỉnh tay trong `rank_matches` (distinct_ratio² · priority ·
power · dir_bonus · QA near-tie · two-tier) có thật sự thắng cosine trần không?*

Lợi ích **chỉ có thể** đến từ `steps ≥ 2`: `q' = A p` là **query đã được HOÀN NGUYÊN** — token
thiếu trong câu hỏi được các cha top điền vào, rồi khớp lại. Đó đúng là thứ §12.2
(*"hard-AND loại cha ĐÚNG"*) đang vật lộn bằng tay.

## ⚠️ Định kiến ÂM đã đo — ghi trước khi chạy

Dự án đã đo **nhiều lần** rằng "xa hơn / nhiều bước hơn" **làm mờ**: toán tử chuyển §12.118
(Spearman +0.06), tầm `d` §12.108, và bài học §0.W: *"Cặp gần-bất-ngờ > khả-đạt xa/nhiều-bước"*.
`steps≥2` đi **ngược** định kiến đó.

Lý do vẫn đáng đo: những phép trên **không có nhiệt độ**. Cái phân biệt Hopfield *hiện đại*
với Hopfield cổ điển đúng là `softmax(β·)` — β cao thì truy hồi **một** mẫu thay vì trộn.
**β chính là núm chống-mờ.** Nếu β cao mà vẫn mờ thì định kiến thắng và hướng này đóng —
đó là một kết quả sạch, không phải thất bại.

## Cắt ngọn = sắc, có chủ đích

`top_k` (mẫu giữ lại khi hoàn nguyên) và `top_m` (token giữ lại của `q'`) không chỉ để nhanh —
chúng là **một tầng làm sắc thứ hai** bên cạnh β. Ghi rõ vì chúng ảnh hưởng kết quả, không phải
chi tiết cài đặt vô hại.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from nn_aivie.hca.objcache import cache_get, cache_put

_MEM_CACHE: dict = {}

DEFAULT_BETA = 8.0
DEFAULT_STEPS = 2
DEFAULT_TOP_K = 64   # mẫu (cha) giữ khi hoàn nguyên
DEFAULT_TOP_M = 64   # token giữ của query đã hoàn nguyên


@dataclass
class HopfieldMemory:
    """`A` (n_token × P) từ `ParentBipartite`, kèm chuẩn cột để chấm bằng cosine."""

    A_csr: csr_matrix              # (n, P) — lấy HÀNG theo token query
    A_csc: csc_matrix = field(repr=False, default=None)  # (n, P) — lấy CỘT theo cha
    col_norm: np.ndarray = field(repr=False, default=None)

    @property
    def n_parents(self) -> int:
        return int(self.A_csr.shape[1])


def build_hopfield_memory(state) -> HopfieldMemory:
    from nn_aivie.hca.matrix_core import ensure_parent_bipartite

    pb = ensure_parent_bipartite(state)
    A = pb.A.tocsr()
    cn = np.sqrt(np.asarray(A.multiply(A).sum(axis=0)).ravel())
    cn[cn <= 0] = 1.0
    return HopfieldMemory(A_csr=A, A_csc=A.tocsc(), col_norm=cn)


def ensure_hopfield_memory(state) -> HopfieldMemory:
    got = cache_get(_MEM_CACHE, state)
    if got is not None:
        return got
    return cache_put(_MEM_CACHE, state, build_hopfield_memory(state))


def clear_hopfield_cache() -> None:
    _MEM_CACHE.clear()


def _match_scores(
    mem: HopfieldMemory,
    tok_ids: np.ndarray,
    tok_w: np.ndarray,
    *,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """`s = Aᵀq` chuẩn hoá cosine. Chỉ đụng HÀNG của token có trong q ⇒ thưa, rẻ."""
    P = mem.n_parents
    s = np.zeros(P, dtype=np.float64)
    if tok_ids.size == 0:
        return s
    n = mem.A_csr.shape[0]
    for t, w in zip(tok_ids, tok_w):
        t = int(t)
        if not (0 <= t < n) or w == 0.0:
            continue
        row = mem.A_csr.getrow(t)
        if row.nnz:
            s[row.indices] += float(w) * row.data
    qn = float(np.linalg.norm(tok_w)) or 1.0
    s /= mem.col_norm * qn
    if candidates is not None:
        mask = np.zeros(P, dtype=bool)
        mask[candidates] = True
        s[~mask] = 0.0
    return s


def _softmax_top(s: np.ndarray, beta: float, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """softmax(β·s) giới hạn top_k mẫu. Trả (chỉ số, xác suất)."""
    nz = np.flatnonzero(s > 0)
    if nz.size == 0:
        return nz, np.zeros(0, dtype=np.float64)
    if nz.size > top_k:
        nz = nz[np.argpartition(-s[nz], top_k - 1)[:top_k]]
    z = beta * s[nz]
    z -= z.max()
    p = np.exp(z)
    tot = p.sum()
    return nz, (p / tot if tot > 0 else p)


def retrieve(
    mem: HopfieldMemory,
    token_ids,
    token_weights=None,
    *,
    beta: float = DEFAULT_BETA,
    steps: int = DEFAULT_STEPS,
    top_k: int = DEFAULT_TOP_K,
    top_m: int = DEFAULT_TOP_M,
    top_n: int = 8,
    candidates=None,
) -> list[tuple[int, float]]:
    """Truy hồi cha bằng lặp Hopfield hiện đại. Trả `[(parent_idx, score)]` giảm dần.

    `steps=1` ⇒ cosine trần (softmax đơn điệu, không đổi thứ hạng) — dùng làm ĐỐI CHỨNG.
    `steps≥2` ⇒ query được hoàn nguyên `q' = A p` rồi khớp lại; đây là chỗ duy nhất
    cơ chế có thể ăn hơn cosine.
    """
    tok = np.asarray([int(t) for t in token_ids], dtype=np.int64)
    if tok.size == 0:
        return []
    w = (
        np.ones(tok.size, dtype=np.float64)
        if token_weights is None
        else np.asarray(token_weights, dtype=np.float64)
    )
    cand = None if candidates is None else np.asarray(list(candidates), dtype=np.int64)

    s = _match_scores(mem, tok, w, candidates=cand)
    for _ in range(max(0, int(steps) - 1)):
        idx, p = _softmax_top(s, float(beta), int(top_k))
        if idx.size == 0:
            break
        # q' = A p — hoàn nguyên query trong không gian token
        q_new = np.zeros(mem.A_csr.shape[0], dtype=np.float64)
        for pi, pw in zip(idx, p):
            col = mem.A_csc.getcol(int(pi))
            if col.nnz:
                q_new[col.indices] += float(pw) * col.data
        nzt = np.flatnonzero(q_new > 0)
        if nzt.size == 0:
            break
        if nzt.size > top_m:
            nzt = nzt[np.argpartition(-q_new[nzt], top_m - 1)[:top_m]]
        s = _match_scores(mem, nzt, q_new[nzt], candidates=cand)

    nz = np.flatnonzero(s > 0)
    if nz.size == 0:
        return []
    k = min(int(top_n), nz.size) if top_n and top_n > 0 else nz.size
    sel = nz[np.argsort(-s[nz])[:k]]
    return [(int(pi), float(s[pi])) for pi in sel]


def tiebreak_ranked(
    state,
    ranked: list[tuple[int, float]],
    token_ids,
    token_weights=None,
    *,
    beta: float = 32.0,
    steps: int = DEFAULT_STEPS,
    top_k: int = DEFAULT_TOP_K,
    top_m: int = DEFAULT_TOP_M,
) -> list[tuple[int, float]]:
    """Xếp lại **chỉ NHÓM HOÀ** của một bảng xếp hạng có sẵn, bằng điểm Hopfield.

    Đây là dạng dùng DUY NHẤT của Trụ D sống sót phép đo (P48, §13.9):

    * thay `rank_matches` bằng Hopfield ⇒ **−7.5…−10pp rank-1** (ÂM TÍNH — cổng chỉnh tay
      + đơn vị phân tầng mang thông tin mà `A` không có);
    * dùng Hopfield **phá thế hoà** ⇒ **+2.25pp rank-1, MRR +0.053, hoà 38.2%→0.0%** (mini).

    Thứ hạng theo điểm chính **BẤT BIẾN** — chỉ trong mỗi bậc điểm bằng nhau mới xếp lại ⇒
    62% query không hoà **không bị đụng theo cấu tạo**. Không phải "không thể tệ đi": trên
    query hoà, đoán sai vẫn đẩy cha đúng xuống. Lãi ròng là thứ phải ĐO, và đã đo.
    """
    if len(ranked) < 2 or beta <= 0:
        return ranked
    top = ranked[0][1]
    if sum(1 for _, sc in ranked if sc == top) < 2:
        return ranked  # không hoà đỉnh → khỏi tính Hopfield (rẻ)
    mem = ensure_hopfield_memory(state)
    hop = dict(retrieve(
        mem, token_ids, token_weights,
        beta=beta, steps=steps, top_k=top_k, top_m=top_m, top_n=0,
    ))
    out = sorted(ranked, key=lambda x: (-x[1], -hop.get(x[0], 0.0)))

    # ⚠️ Chỉ ĐỔI THỨ TỰ mà giữ nguyên điểm là KHÔNG ĐỦ: consumer nào sắp lại theo điểm
    # (hoặc đo hoà theo điểm) sẽ mất sạch hiệu lực — âm thầm. Bắt được khi nối thật:
    # `probe_hopfield_rank` báo `A_rank_matches` y hệt trước khi wire vì `_tie_stats`
    # đọc ĐIỂM chứ không đọc thứ tự.
    # ⇒ Ghi thứ tự vào chính điểm bằng epsilon nhân, đủ nhỏ để KHÔNG vượt bậc:
    #   s_k = s · (1 + ε·(m−k)/m),  ε = 1e-9  ⇒ xáo trong bậc, giữ nguyên giữa các bậc
    #   (an toàn khi khoảng cách tương đối giữa hai bậc > 1e-9 — đúng với coverage thực).
    eps = 1e-9
    res: list[tuple[int, float]] = []
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j][1] == out[i][1]:
            j += 1
        m = j - i
        for k in range(m):
            pi, sc = out[i + k]
            res.append((pi, float(sc) * (1.0 + eps * (m - k) / m) if m > 1 else float(sc)))
        i = j
    return res
