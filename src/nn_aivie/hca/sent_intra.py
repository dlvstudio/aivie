"""
TK_045 P7 / P0.2 — cạnh trong câu (L2R/R2L+d) và giữa câu (SENT_SUCC → W_agenda).

Gen vẫn trái→phải. R2L = neo/rescore cha (KHÔNG phải phân phối next).
L2R continuation chỉ sau mask(emit_remaining).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.hca.objcache import cache_get, cache_put

if TYPE_CHECKING:
    from nn_aivie.hca.matrix_core import ParentBipartite

_SENT_INTRA: dict[int, "SentIntraViews"] = {}


def _bound_of(state):
    """Tín hiệu biên học được của `state` (import cục bộ để tránh vòng import)."""
    from nn_aivie.hca.clause_layer import ensure_boundary_signal

    return ensure_boundary_signal(state)


def _is_whitespace_break(vocab, tid: int) -> bool:
    """Ngắt do KHOẢNG TRẮNG — xuống dòng / token rỗng.

    Đây **không** phải tri thức từ vựng: nó không nói token nào *có nghĩa* kết câu, nó chỉ
    đọc ký tự trắng. Vì thế nó **không** thuộc diện Luật 0 cấm (khác hẳn một danh sách dấu
    câu tiếng Việt). Giữ lại vì tín hiệu học KHÔNG bắt được nó: `ingest._with_eos` chèn EOS
    ở cuối ĐƠN VỊ chứ không sau xuống dòng, nên `Ċ`/`ĊĊ` có lift = 0 dù chúng là biên thật
    (đo mini: 192 lần xuất hiện).
    """
    tok = vocab.token_of(int(tid)).lstrip("Ġ")
    return (not tok) or tok.startswith("Ċ") or tok in ("\n", "\r")


# §13.54 — MỒI NGUỘI, chỉ dùng lúc `ingest`. Xem docstring `_is_boundary`: tín hiệu học được
# lấy từ `W_seq`, mà lúc ingest thì `W_seq` CHƯA CÓ ⇒ vòng luẩn quẩn thật, không phải lỗi đấu
# dây. Danh sách này vì thế KHÔNG chết được — nhưng nó bị giới hạn vào đúng một chỗ, và mọi
# quyết định lúc SINH đều đã chuyển sang tín hiệu học.
_BOOTSTRAP_PUNCT = (".", "!", "?", "…", ";", "。")


def _is_boundary(vocab, tid: int, bound=None, *, bootstrap: bool = False) -> bool:
    """Biên ĐƠN VỊ = **EOS-lift học được** (§0.4.2) ∪ ngắt khoảng trắng.

    ## Trả nợ Luật 0 (§13.54)

    Bản cũ hardcode `(".", "!", "?", "…", ";", "。")` — đúng thứ §0.4.2 cấm, và tầng mệnh đề
    (Trụ A) đang dựa vào nó qua `split_emit_to_sentence_seqs`. Nay dùng `BoundarySignal`,
    thứ **đếm được từ dữ liệu**: `lift[w] = P(EOS|w) / P(EOS) nền`.

    ## Đã đo trước khi đổi (mini) — vì sao đổi được mà không vỡ

    | | |
    |---|---:|
    | token loại lệch giữa hai định nghĩa | 17 / 32.002 |
    | theo **tần suất thật** | **0,219%** lần xuất hiện |

    `.` lift **18,49** · `!` **22,98** · `?` **6,58** ⇒ ba dấu chính khớp. Lệch nằm ở:
    - `;` (42 lần) và `…` (1 lần): lift **0** — corpus **không bao giờ** đặt chúng trước
      EOS, nên tín hiệu học nói chúng **không** kết đơn vị. Đó là **đúng hơn** danh sách
      cứng: `;` trong tiếng Việt là ngắt dưới-câu.
    - `。` (dấu câu Trung văn): **0 lần** trong corpus ⇒ mã chết, bỏ đi không mất gì.
    - xuống dòng: **vẫn giữ**, qua `_is_whitespace_break` — xem docstring ở đó.

    ## Vì sao danh sách cứng KHÔNG chết hẳn được — vòng luẩn quẩn ở `ingest`

    Tín hiệu học lấy từ `W_seq`. Nhưng `ingest` **tách câu trong lúc đang dựng** `W_seq` ⇒ ở
    thời điểm đó tín hiệu **chưa tồn tại**. Đây là vòng luẩn quẩn thật, không phải thiếu sót
    khi vá. ⇒ Danh sách chỉ còn sống ở **đúng một chỗ**: `bootstrap=True`, do `ingest` truyền
    **tường minh**. Mọi quyết định lúc **SINH** — thứ Luật 0 nhắm tới — đã dùng tín hiệu học.

    ⚠️ `bound=None` và `bootstrap=False` ⇒ **chỉ** còn ngắt khoảng trắng, KHÔNG âm thầm quay
    về danh sách cứng. Fallback im lặng chính là cách một mặc định bị đảo mà không ai thấy;
    thà mất tín hiệu **ồn ào** còn hơn giữ nợ **kín đáo**.
    """
    if _is_whitespace_break(vocab, tid):
        return True
    if bound is not None:
        return bool(bound.ends_unit(int(tid)))
    if bootstrap:
        return vocab.token_of(int(tid)).lstrip("Ġ") in _BOOTSTRAP_PUNCT
    return False


@dataclass
class SentIntraViews:
    """CSR decode + meta."""

    W_l2r: csr_matrix  # (N, N)
    W_r2l: csr_matrix  # (N, N)
    W_agenda: csr_matrix  # (N, N) SENT_SUCC token cuối→đầu câu sau
    n_sent_spans: int = 0
    n_l2r_pairs: int = 0
    n_r2l_pairs: int = 0
    n_agenda_edges: int = 0
    lam: float = 0.85
    d_max: int = 12
    content_only: bool = True


def _f_dist(d: int, lam: float) -> float:
    if d < 1:
        return 0.0
    return float(lam) ** (d - 1)


def iter_sentence_spans(
    seq: list[int] | tuple[int, ...],
    vocab,
    bound=None,
    *,
    bootstrap: bool = False,
) -> list[list[int]]:
    """Tách emit thành span câu theo biên ĐƠN VỊ **học được** (§13.54) ∪ ngắt trắng."""
    if not seq:
        return []
    spans: list[list[int]] = []
    cur: list[int] = []
    for t in seq:
        t = int(t)
        cur.append(t)
        if _is_boundary(vocab, t, bound, bootstrap=bootstrap):
            if cur:
                spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans if spans else [list(seq)]


def _content_positions(
    span: list[int],
    vocab,
    *,
    content_only: bool,
    struct_mask: np.ndarray | None,
    bound=None,
) -> list[int]:
    out: list[int] = []
    n = len(struct_mask) if struct_mask is not None else 0
    for t in span:
        t = int(t)
        if content_only:
            if struct_mask is not None and 0 <= t < n and struct_mask[t]:
                continue
            if _is_boundary(vocab, t, bound):
                continue
            if not vocab.is_surface(t):
                continue
        out.append(t)
    return out


def accumulate_span_pairs(
    tokens: list[int],
    *,
    lam: float,
    d_max: int,
    weight: float = 1.0,
    gidf_fn=None,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """∀ i<j với d=j-i ≤ d_max: L2R t_i→t_j và R2L t_j→t_i với f(d)."""
    l2r: list[tuple[int, int, float]] = []
    r2l: list[tuple[int, int, float]] = []
    L = len(tokens)
    for i in range(L):
        for j in range(i + 1, min(L, i + 1 + int(d_max))):
            d = j - i
            w = weight * _f_dist(d, lam)
            if gidf_fn is not None:
                w *= float(
                    np.sqrt(
                        max(gidf_fn(tokens[i]), 1e-6) * max(gidf_fn(tokens[j]), 1e-6)
                    )
                )
            if w <= 0:
                continue
            a, b = int(tokens[i]), int(tokens[j])
            l2r.append((a, b, w))
            r2l.append((b, a, w))
    return l2r, r2l


def _first_last_content(
    seq: list[int] | tuple[int, ...] | np.ndarray,
    vocab,
    struct_mask: np.ndarray | None,
    bound=None,
) -> tuple[int | None, int | None]:
    ids = [int(t) for t in seq]
    n = len(struct_mask) if struct_mask is not None else 0
    first = last = None
    for t in ids:
        if struct_mask is not None and 0 <= t < n and struct_mask[t]:
            continue
        if _is_boundary(vocab, t, bound):
            continue
        if not vocab.is_surface(t):
            continue
        if first is None:
            first = t
        last = t
    return first, last


def build_sent_intra_views(
    state: GraphState,
    *,
    lam: float = 0.85,
    d_max: int = 12,
    content_only: bool = True,
    use_gidf: bool = True,
    write_adjacency: bool = False,
) -> SentIntraViews:
    """Đếm L2R/R2L trong từng span câu + W_agenda từ episode / multi-span parent."""
    # import cục bộ: `clause_layer` là nơi giữ tín hiệu biên HỌC ĐƯỢC (§0.4.2), và import
    # ở mức module sẽ tạo vòng vì `matrix_core` (do `clause_layer` gọi) lại đọc file này.
    from nn_aivie.hca.clause_layer import ensure_boundary_signal
    from nn_aivie.hca.context_positional import is_structural_token

    n = len(state.vocab)
    vocab = state.vocab
    ps = state.parents
    empty = csr_matrix((n, n), dtype=np.float64)
    if ps is None or len(ps.parents) == 0:
        return SentIntraViews(
            empty, empty, empty, lam=lam, d_max=d_max, content_only=content_only
        )

    bd = ensure_boundary_signal(state)   # §13.54 — biên đơn vị đọc từ W_seq, không hardcode
    struct = np.array([is_structural_token(vocab, i) for i in range(n)], dtype=bool)
    gidf_fn = (lambda tid: float(ps._gidf(int(tid)))) if use_gidf else None  # noqa: SLF001

    l_r, l_c, l_v = [], [], []
    r_r, r_c, r_v = [], [], []
    n_spans = 0
    n_l2r = n_r2l = 0

    for rec in ps.parents:
        seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        pw = float(max(1, rec.power)) * float(getattr(rec, "priority", 1.0) or 1.0)
        for span in iter_sentence_spans(seq, vocab, bd):
            toks = _content_positions(
                span, vocab, content_only=content_only, struct_mask=struct, bound=bd
            )
            if len(toks) < 2:
                continue
            n_spans += 1
            l2r, r2l = accumulate_span_pairs(
                toks, lam=lam, d_max=d_max, weight=pw, gidf_fn=gidf_fn
            )
            for a, b, w in l2r:
                if 0 <= a < n and 0 <= b < n:
                    l_r.append(a)
                    l_c.append(b)
                    l_v.append(w)
                    n_l2r += 1
            for a, b, w in r2l:
                if 0 <= a < n and 0 <= b < n:
                    r_r.append(a)
                    r_c.append(b)
                    r_v.append(w)
                    n_r2l += 1

    W_l2r = (
        coo_matrix((l_v, (l_r, l_c)), shape=(n, n), dtype=np.float64).tocsr()
        if l_v
        else empty
    )
    W_r2l = (
        coo_matrix((r_v, (r_r, r_c)), shape=(n, n), dtype=np.float64).tocsr()
        if r_v
        else empty
    )

    a_r, a_c, a_v = [], [], []
    n_agenda = 0
    for ep in getattr(ps, "episodes", None) or []:
        members = [int(pi) for pi, _w in ep.members]
        for k in range(len(members) - 1):
            pi, pj = members[k], members[k + 1]
            if not (0 <= pi < len(ps.parents) and 0 <= pj < len(ps.parents)):
                continue
            seq_i = list(
                ps.parents[pi].answer_ids
                if (ps.parents[pi].is_qa and ps.parents[pi].answer_ids)
                else ps.parents[pi].emit_ids
            )
            seq_j = list(
                ps.parents[pj].answer_ids
                if (ps.parents[pj].is_qa and ps.parents[pj].answer_ids)
                else ps.parents[pj].emit_ids
            )
            spans_i = iter_sentence_spans(seq_i, vocab, bd)
            spans_j = iter_sentence_spans(seq_j, vocab, bd)
            if not spans_i or not spans_j:
                continue
            _f, last = _first_last_content(spans_i[-1], vocab, struct, bd)
            first, _l = _first_last_content(spans_j[0], vocab, struct, bd)
            if last is None or first is None:
                continue
            w = 1.0
            if gidf_fn is not None:
                w *= float(np.sqrt(max(gidf_fn(last), 1e-6) * max(gidf_fn(first), 1e-6)))
            a_r.append(int(last))
            a_c.append(int(first))
            a_v.append(w)
            n_agenda += 1

    if n_agenda == 0:
        for rec in ps.parents:
            seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
            spans = iter_sentence_spans(seq, vocab, bd)
            if len(spans) < 2:
                continue
            pw = float(max(1, rec.power))
            for k in range(len(spans) - 1):
                _f, last = _first_last_content(spans[k], vocab, struct, bd)
                first, _l = _first_last_content(spans[k + 1], vocab, struct, bd)
                if last is None or first is None:
                    continue
                w = pw
                if gidf_fn is not None:
                    w *= float(
                        np.sqrt(max(gidf_fn(last), 1e-6) * max(gidf_fn(first), 1e-6))
                    )
                a_r.append(int(last))
                a_c.append(int(first))
                a_v.append(w)
                n_agenda += 1

    W_agenda = (
        coo_matrix((a_v, (a_r, a_c)), shape=(n, n), dtype=np.float64).tocsr()
        if a_v
        else empty
    )

    if write_adjacency and hasattr(state, "adjacency"):
        from nn_aivie.core.adjacency import SparseAdjacency

        adj = state.adjacency
        for kind in (EdgeKind.SENT_L2R, EdgeKind.SENT_R2L, EdgeKind.SENT_SUCC):
            if kind not in adj._layers:  # noqa: SLF001
                adj._layers[kind] = SparseAdjacency()  # noqa: SLF001

        def _layer_empty(kind: EdgeKind) -> bool:
            layer = adj.layer(kind)
            return len(layer) == 0

        if _layer_empty(EdgeKind.SENT_L2R) and W_l2r.nnz:
            coo = W_l2r.tocoo()
            for i, j, w in zip(coo.row, coo.col, coo.data):
                adj.add(int(i), int(j), float(w), EdgeKind.SENT_L2R)
        if _layer_empty(EdgeKind.SENT_R2L) and W_r2l.nnz:
            coo = W_r2l.tocoo()
            for i, j, w in zip(coo.row, coo.col, coo.data):
                adj.add(int(i), int(j), float(w), EdgeKind.SENT_R2L)
        if _layer_empty(EdgeKind.SENT_SUCC) and W_agenda.nnz:
            coo = W_agenda.tocoo()
            for i, j, w in zip(coo.row, coo.col, coo.data):
                adj.add(int(i), int(j), float(w), EdgeKind.SENT_SUCC)

    return SentIntraViews(
        W_l2r=W_l2r,
        W_r2l=W_r2l,
        W_agenda=W_agenda,
        n_sent_spans=n_spans,
        n_l2r_pairs=n_l2r,
        n_r2l_pairs=n_r2l,
        n_agenda_edges=n_agenda,
        lam=lam,
        d_max=d_max,
        content_only=content_only,
    )


def ensure_sent_intra(
    state: GraphState,
    *,
    config=None,
    force: bool = False,
) -> SentIntraViews:
    cached = cache_get(_SENT_INTRA, state)
    lam = float(getattr(config, "ar_sent_lr_lam", 0.85)) if config else 0.85
    d_max = int(getattr(config, "ar_sent_lr_d_max", 12)) if config else 12
    content_only = (
        bool(getattr(config, "ar_sent_lr_content_only", True)) if config else True
    )
    if (
        force
        or cached is None
        or getattr(cached, "_v", 0) < 1
        or abs(cached.lam - lam) > 1e-9
        or cached.d_max != d_max
        or cached.content_only != content_only
    ):
        cached = build_sent_intra_views(
            state,
            lam=lam,
            d_max=d_max,
            content_only=content_only,
            use_gidf=bool(getattr(config, "ar_sent_lr_use_gidf", True))
            if config
            else True,
            write_adjacency=bool(getattr(config, "ar_sent_lr_write_adj", False))
            if config
            else False,
        )
        cached._v = 1  # type: ignore[attr-defined]
        cache_put(_SENT_INTRA, state, cached)
    return cached


def clear_sent_intra_cache() -> None:
    _SENT_INTRA.clear()
    _SENT_QK.clear()


def _csr_row_dense(mat: csr_matrix, i: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    if not (0 <= i < mat.shape[0]):
        return out
    row = mat.getrow(i)
    if row.nnz:
        out[row.indices] = row.data.astype(np.float64)
    return out


def r2l_anchor_scores(
    views: SentIntraViews,
    history_ids: list[int],
    *,
    vocab_size: int,
    M: int = 3,
    alpha_decay: float = 0.85,
) -> np.ndarray:
    """Neo từ đuôi: Σ_m α_m W_r2l[h_{k-m}] — conditioning, không dùng làm next."""
    anchors = np.zeros(vocab_size, dtype=np.float64)
    if not history_ids or views.W_r2l.nnz == 0:
        return anchors
    M = max(1, int(M))
    for m, tid in enumerate(reversed(history_ids[-M:])):
        row = _csr_row_dense(views.W_r2l, int(tid), vocab_size)
        anchors += (float(alpha_decay) ** m) * row
    return anchors


def emit_remaining_mask(
    bi: "ParentBipartite",
    parent_act: np.ndarray,
    history_ids: list[int],
    *,
    vocab_size: int,
    top_k: int = 8,
) -> np.ndarray:
    """Mask token còn lại trong emit của top cha (sau suffix match history)."""
    mask = np.zeros(vocab_size, dtype=np.float64)
    if bi.n_parents <= 0 or parent_act.size == 0:
        return mask
    k = min(int(top_k), bi.n_parents)
    if k <= 0:
        return mask
    if bi.n_parents > k:
        idx = np.argpartition(-parent_act, k - 1)[:k]
    else:
        idx = np.arange(bi.n_parents)
    hist = [int(t) for t in history_ids]
    for pi in idx:
        pi = int(pi)
        if parent_act[pi] <= 0 or pi >= len(bi.emits):
            continue
        emit = [int(t) for t in bi.emits[pi]]
        if not emit:
            continue
        best_pos = -1
        max_m = min(len(hist), len(emit))
        for m in range(max_m, 0, -1):
            target = hist[-m:]
            for s in range(0, len(emit) - m + 1):
                if emit[s : s + m] == target:
                    best_pos = s + m - 1
                    break
            if best_pos >= 0:
                break
        start = best_pos + 1 if best_pos >= 0 else 0
        for t in emit[start:]:
            if 0 <= t < vocab_size:
                mask[t] = max(mask[t], float(parent_act[pi]))
    return mask


def emit_membership(bi: "ParentBipartite"):
    """TK_045 §12.32 — ma trận thưa (token → cha PHÁT token), dựng MỘT lần, cache trên `bi`.

    `rescore_parents_with_anchors` trước đây hỏi "cha `pi` có phát token `tid` không" bằng hai
    vòng Python lồng nhau quét TOÀN BỘ cha, mỗi lần **dựng lại một set** từ `bi.emits[pi]`:
    tới `16 × n_parents` lần dựng set **cho mỗi token sinh ra**. Trên v12 (430.447 cha) là
    ~7,3 triệu lần/token — cProfile bắt được `<setcomp>` gọi **170.844 lần cho 2 bước sinh**
    trên mini, và 87% thời gian `generate()` nằm ở đây. Đây là lý do v12 sinh ~5 giây/token.

    Quan hệ đó là TĨNH theo checkpoint ⇒ dựng một lần rồi tra cột. KHÔNG dùng `bi.E` sẵn có vì
    `E` bỏ token cấu-trúc và có thêm cạnh succ ⇒ không đúng nghĩa "có mặt trong emit".
    """
    m = getattr(bi, "_emit_memb", None)
    if m is not None:
        return m
    rows: list[int] = []
    cols: list[int] = []
    n = bi.A.shape[0]
    for pi, e in enumerate(bi.emits):
        for t in {int(x) for x in e}:
            if 0 <= t < n:
                rows.append(t)
                cols.append(pi)
    m = coo_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                   shape=(n, max(1, bi.n_parents))).tocsr()
    bi._emit_memb = m  # type: ignore[attr-defined]
    return m


def rescore_parents_with_anchors(
    bi: "ParentBipartite",
    parent_act: np.ndarray,
    anchors: np.ndarray,
    *,
    hist_ids: set[int] | None = None,
) -> np.ndarray:
    """Tăng act cha có emit/match giao neo R2L."""
    act = parent_act.astype(np.float64).copy()
    if bi.n_parents <= 0 or anchors.max() <= 0:
        return act
    k = min(16, int(anchors.size))
    top_a = np.argpartition(-anchors, k - 1)[:k]
    top_a = [int(t) for t in top_a if anchors[t] > 0]
    if not top_a:
        return act
    memb = emit_membership(bi)  # §12.32: token → cha PHÁT token đó (CSR, dựng 1 lần)
    boost = np.ones(bi.n_parents, dtype=np.float64)
    for tid in top_a:
        hit = np.zeros(bi.n_parents, dtype=np.float64)
        if tid < bi.A.shape[0]:
            row = bi.A.getrow(tid)
            if row.nnz:
                hit[row.indices] = 1.0
        if tid < memb.shape[0]:
            ei = memb.getrow(tid).indices
            if ei.size:
                hit[ei] = np.maximum(hit[ei], 0.5)
        boost *= 0.35 + 0.65 * hit
    act = act * boost
    if hist_ids:
        ids = [t for t in (int(x) for x in hist_ids) if 0 <= t < memb.shape[0]]
        if ids:
            overlap = np.asarray(memb[ids, :].sum(axis=0)).ravel().astype(np.float64)
            act = act * (1.0 + 0.05 * overlap)
    return act


def sent_intra_decode_channels(
    state: GraphState,
    history_ids: list[int],
    parent_act: np.ndarray,
    *,
    config,
    vocab_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(anchors, cont_masked, agenda_next, parent_act')."""
    from nn_aivie.hca.matrix_core import ensure_parent_bipartite

    n = vocab_size if vocab_size is not None else len(state.vocab)
    z = np.zeros(n, dtype=np.float64)
    if not history_ids:
        return z, z, z, parent_act

    views = ensure_sent_intra(state, config=config)
    bi = ensure_parent_bipartite(state)
    M = int(getattr(config, "ar_sent_r2l_M", 3))
    anchors = r2l_anchor_scores(
        views,
        history_ids,
        vocab_size=n,
        M=M,
        alpha_decay=float(getattr(config, "ar_sent_r2l_alpha", 0.85)),
    )
    act2 = rescore_parents_with_anchors(
        bi,
        parent_act,
        anchors,
        hist_ids={int(t) for t in history_ids},
    )
    early_n = int(getattr(config, "ar_sent_l2r_early", 4))
    seeds: list[int] = []
    for t in history_ids[:early_n]:
        seeds.append(int(t))
    if anchors.max() > 0:
        ka = min(6, n)
        for tid in np.argpartition(-anchors, ka - 1)[:ka]:
            if anchors[int(tid)] > 0:
                seeds.append(int(tid))
    cont = z.copy()
    for tid in seeds:
        cont = cont + _csr_row_dense(views.W_l2r, tid, n)
    mask = emit_remaining_mask(
        bi,
        act2,
        history_ids,
        vocab_size=n,
        top_k=int(getattr(config, "ar_sent_l2r_parent_topk", 8)),
    )
    use_mask = bool(getattr(config, "ar_sent_l2r_parent_mask", True))
    if use_mask and mask.max() > 0:
        cont = cont * mask
    elif use_mask:
        # Mask rỗng = hết emit (hoặc không neo cha) → tắt cont, cho EOS/handoff.
        cont = z.copy()

    agenda = z.copy()
    w0 = int(history_ids[-1])
    if views.W_agenda.nnz:
        agenda = _csr_row_dense(views.W_agenda, w0, n)

    return anchors, cont, agenda, act2


_HANDOFF_BLOCK = frozenset(
    {
        "Nghĩa",
        "Giải",
        "Ví",
        "Loại",
        "Các",
        "Bài",
        "Khi",
        "Trong",
        "Theo",
        "Với",
        "Và",
        "Hoặc",
        "Nhưng",
        "Nếu",
        "Để",
        "Một",
        "Những",
        "Này",
        "Đó",
        "Lâm",
        "Đường",
        "Lưu",
        "Ch",
        ":",
        "1",
        "2",
        "3",
        "4",
        "5",
    }
)


def _surface_content_folds(vocab, ids: list[int], bound=None) -> set[str]:
    """Tập fold token nội dung (bỏ function/hub ngắn) — neo entity cho handoff."""
    out: set[str] = set()
    n_vocab = len(vocab)
    for t in ids:
        t = int(t)
        if t < 0 or t >= n_vocab:
            continue
        if not vocab.is_surface(t) or _is_boundary(vocab, t, bound):
            continue
        tok = vocab.token_of(t).lstrip("Ġ")
        if len(tok) < 2 or tok in _HANDOFF_BLOCK:
            continue
        # số thuần / 1 ký tự sau strip
        if tok.isdigit():
            continue
        out.add(vocab.fold(t) if hasattr(vocab, "fold") else tok.casefold())
    return out


def _episode_next_span(
    state: GraphState,
    history_ids: list[int],
    *,
    vocab_size: int,
    bi: "ParentBipartite",
    max_tokens: int = 28,
) -> list[int] | None:
    """P0.3: hist khớp member k → trả span content của member k+1 (không chỉ token đầu).

    Chỉ chèn first token khiến matrix lạc ("Năm" → EOS → Năm lặp). Emit cả câu kế
    (cắt EOS/biên đuôi, tối đa max_tokens).
    """
    ps = state.parents
    if ps is None or not getattr(ps, "episodes", None) or not history_ids:
        return None
    hist = [int(t) for t in history_ids]
    from nn_aivie.hca.clause_layer import ensure_boundary_signal

    _bd = ensure_boundary_signal(state)
    while hist and _is_boundary(state.vocab, hist[-1], _bd):
        hist.pop()
    if len(hist) < 3:
        return None

    from nn_aivie.hca.matrix_core import parent_coactivation_logits

    _bag, _succ, _conf, act = parent_coactivation_logits(
        state, hist, vocab_size=vocab_size, top_k_parents=64
    )
    if act.size == 0 or float(act.max()) <= 0:
        return None
    k_top = min(48, int(act.size))
    order = np.argpartition(-act, k_top - 1)[:k_top]
    order = order[np.argsort(-act[order])]
    cands = [int(pi) for pi in order if act[int(pi)] > 0]

    hist_fold = _surface_content_folds(state.vocab, hist)
    best_score = -1.0
    best_span: list[int] | None = None
    for pi in cands:
        if pi < 0 or pi >= len(bi.emits):
            continue
        emit = [int(t) for t in bi.emits[pi]]
        while emit and (
            not state.vocab.is_surface(emit[-1])
            or _is_boundary(state.vocab, emit[-1], _bound_of(state))
        ):
            emit.pop()
        if not emit:
            continue
        max_k = min(len(hist), len(emit), 16)
        matched = 0
        for k in range(max_k, 2, -1):
            if emit[-k:] == hist[-k:]:
                matched = k
                break
        if matched < 3:
            em_fold = _surface_content_folds(state.vocab, emit)
            ov = len(hist_fold & em_fold) if hist_fold and em_fold else 0
            if ov < max(4, len(hist_fold) // 2):
                continue
            matched = max(3, ov)
        for ei in ps.episodes_of(pi):
            if ei >= len(ps.episodes):
                continue
            members = [int(m) for m, _w in ps.episodes[ei].members]
            if pi not in members:
                continue
            ix = members.index(pi)
            if ix + 1 >= len(members):
                continue
            nxt_pi = members[ix + 1]
            if nxt_pi >= len(bi.emits):
                continue
            nxt_emit = [int(t) for t in bi.emits[nxt_pi]]
            # bỏ EOS / biên / non-surface đuôi
            while nxt_emit and (
                not state.vocab.is_surface(nxt_emit[-1])
                or _is_boundary(state.vocab, nxt_emit[-1], _bound_of(state))
            ):
                nxt_emit.pop()
            # bỏ leading newline / structural junk
            while nxt_emit and (
                not state.vocab.is_surface(nxt_emit[0])
                or state.vocab.token_of(nxt_emit[0]).lstrip("Ġ") in ("\n",)
            ):
                nxt_emit.pop(0)
            if not nxt_emit:
                continue
            first = int(nxt_emit[0])
            if not state.vocab.is_surface(first):
                continue
            tok = state.vocab.token_of(first).lstrip("Ġ")
            if tok in _HANDOFF_BLOCK:
                continue
            nxt_fold = _surface_content_folds(state.vocab, nxt_emit)
            overlap = len(hist_fold & nxt_fold) if hist_fold and nxt_fold else 0
            # §12.21 — ĐỪNG nới: câu sau mở đầu bằng ĐẠI TỪ mà history không có tiền ngữ
            # thì nối vào là đại-từ-lơ-lửng. A/B ngưỡng 6→4 trên mini-có-episode: đúng 1/8 ca
            # đổi, và đổi theo hướng XẤU — `Trận Bạch Đằng năm 1288.` + `Ông được nhân dân tôn
            # kính…` (`Ông` chẳng trỏ ai). 7 ca còn lại y hệt ⇒ nới chỉ có hại.
            if overlap < 1 and matched < 6:
                continue
            score = float(matched) * 10.0 + float(overlap) * 8.0 + float(act[pi])
            if score > best_score:
                best_score = score
                # surface content span, giữ dấu câu giữa câu
                span: list[int] = []
                for t in nxt_emit:
                    t = int(t)
                    if not (0 <= t < vocab_size):
                        continue
                    if not state.vocab.is_surface(t):
                        continue
                    raw = state.vocab.token_of(t)
                    if raw in ("\n", "Ċ", "ĊĊ") or raw.startswith("Ċ"):
                        break
                    span.append(t)
                    if len(span) >= int(max_tokens):
                        break
                if span:
                    best_span = span
    return best_span


def _episode_next_first_token(
    state: GraphState,
    history_ids: list[int],
    *,
    vocab_size: int,
    bi: "ParentBipartite",
) -> int | None:
    span = _episode_next_span(
        state, history_ids, vocab_size=vocab_size, bi=bi, max_tokens=1
    )
    return int(span[0]) if span else None


def agenda_handoff_span(
    state: GraphState,
    history_ids: list[int],
    *,
    config,
    eos_logit: float = 0.0,
    vocab_size: int | None = None,
) -> list[int] | None:
    """P0.3: trả span câu sau (episode). W_agenda fallback chỉ first-token nếu bật."""
    if not history_ids:
        return None
    if not bool(getattr(config, "ar_sent_agenda_handoff", True)):
        return None
    n = vocab_size if vocab_size is not None else len(state.vocab)
    max_tok = int(getattr(config, "ar_sent_agenda_handoff_span", 28))

    if state.parents is not None and getattr(state.parents, "episodes", None):
        from nn_aivie.hca.matrix_core import ensure_parent_bipartite

        bi = ensure_parent_bipartite(state)
        span = _episode_next_span(
            state, history_ids, vocab_size=n, bi=bi, max_tokens=max_tok
        )
        if span:
            return span

    # fallback: 1 token từ W_agenda (tắt mặc định)
    one = agenda_handoff_next(
        state,
        history_ids,
        config=config,
        eos_logit=eos_logit,
        vocab_size=n,
        _skip_episode=True,
    )
    return [int(one)] if one is not None else None


def agenda_handoff_next(
    state: GraphState,
    history_ids: list[int],
    *,
    config,
    eos_logit: float,
    vocab_size: int | None = None,
    _skip_episode: bool = False,
) -> int | None:
    """P0.3: EOS local ≠ hết bài — nối câu sau qua episode / W_agenda.

    Ưu tiên episode member kế (khớp đuôi emit). W_agenda là fallback (mặc định tắt —
    dễ hub "Các/Bài"); bật bằng ar_sent_agenda_handoff_w_agenda=True.
    Không so mass agenda với eos_logit tuyệt đối (thang khác nhau → handoff chết).
    """
    if not history_ids:
        return None
    if not bool(getattr(config, "ar_sent_agenda_handoff", True)):
        return None
    # ar_no_agenda chỉ tắt MZ/fact plan — vẫn cho handoff matrix (episode / W_agenda).
    n = vocab_size if vocab_size is not None else len(state.vocab)

    # --- (1) Episode successor (kênh chính cho câu dài) ---
    if (
        not _skip_episode
        and state.parents is not None
        and getattr(state.parents, "episodes", None)
    ):
        from nn_aivie.hca.matrix_core import ensure_parent_bipartite

        bi = ensure_parent_bipartite(state)
        ep_first = _episode_next_first_token(
            state, history_ids, vocab_size=n, bi=bi
        )
        if ep_first is not None:
            return ep_first

    # --- (2) W_agenda fallback (tắt mặc định — nhiễu hub) ---
    if not bool(getattr(config, "ar_sent_agenda_handoff_w_agenda", False)):
        return None
    views = ensure_sent_intra(state, config=config)
    if views.W_agenda.nnz == 0:
        return None
    w0 = int(history_ids[-1])
    row = _csr_row_dense(views.W_agenda, w0, n)
    if row.max() <= 0:
        return None
    from nn_aivie.core.vocab import SPECIAL_EOS

    eos_id = state.vocab._token_to_id.get(SPECIAL_EOS)  # noqa: SLF001
    if eos_id is not None and 0 <= int(eos_id) < n:
        row[int(eos_id)] = 0.0
    for j in range(min(n, row.shape[0])):
        if row[j] <= 0:
            continue
        if not state.vocab.is_surface(j):
            row[j] = 0.0
            continue
        tok = state.vocab.token_of(j).lstrip("Ġ")
        if tok in _HANDOFF_BLOCK:
            row[j] = 0.0
            continue
        if tok in (":",):
            row[j] *= 0.05

    # boost đầu câu kế nếu A[w0] trỏ episode
    boosted = False
    ps = state.parents
    boost = np.ones(n, dtype=np.float64)
    if ps is not None and getattr(ps, "episodes", None) and len(ps.episodes) > 0:
        from nn_aivie.hca.matrix_core import ensure_parent_bipartite

        bi = ensure_parent_bipartite(state)
        hit_pis: list[int] = []
        if bi.n_parents > 0 and 0 <= w0 < bi.A.shape[0]:
            arow = bi.A.getrow(w0)
            if arow.nnz:
                order = np.argsort(-arow.data)[:24]
                hit_pis = [int(arow.indices[i]) for i in order]
        for pi in hit_pis:
            for ei in ps.episodes_of(pi):
                if ei >= len(ps.episodes):
                    continue
                members = [int(m) for m, _w in ps.episodes[ei].members]
                if pi not in members:
                    continue
                ix = members.index(pi)
                if ix + 1 >= len(members):
                    continue
                nxt_pi = members[ix + 1]
                if nxt_pi >= len(bi.emits):
                    continue
                first, _ = _first_last_content(
                    bi.emits[nxt_pi], state.vocab, None, _bound_of(state))
                if first is not None and 0 <= first < n:
                    boost[first] *= 4.0
                    boosted = True
        row = row * boost

    if row.max() <= 0:
        return None
    j = int(np.argmax(row))
    mass = float(row[j])
    handoff_min = float(getattr(config, "ar_sent_agenda_handoff_min", 0.15))
    if mass < handoff_min:
        return None
    # Tập trung trong hàng agenda (không so eos_logit — thang khác).
    conc = mass / (float(row.sum()) + 1e-12)
    conc_min = float(getattr(config, "ar_sent_agenda_handoff_ratio", 0.12))
    if conc_min > 1.0:  # knob cũ 0.85 từng so eos
        conc_min = 0.12
    if not boosted and float(boost[j]) <= 1.0 and conc < conc_min:
        return None
    _ = eos_logit  # giữ chữ ký API
    return j


def ppmi_from_csr(
    mat: csr_matrix,
    *,
    smoothing: float = 0.75,
) -> csr_matrix:
    """PPMI có hướng từ CSR đếm (L2R hoặc R2L)."""
    if mat.nnz == 0:
        return mat.copy()
    c = mat.tocsr().astype(np.float64)
    total = float(c.sum())
    if total <= 0:
        return csr_matrix(c.shape, dtype=np.float64)
    row_sum = np.asarray(c.sum(axis=1)).ravel()
    col_sum = np.asarray(c.sum(axis=0)).ravel()
    col_smooth = np.power(np.maximum(col_sum, 0.0), smoothing)
    p_i = row_sum / total
    p_j = col_smooth / (col_smooth.sum() or 1.0)
    coo = c.tocoo()
    denom = p_i[coo.row] * p_j[coo.col]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((coo.data / total) / np.maximum(denom, 1e-30))
    mask = np.isfinite(pmi) & (denom > 0) & (pmi > 0)
    return csr_matrix(
        (pmi[mask], (coo.row[mask], coo.col[mask])), shape=c.shape, dtype=np.float64
    )


_SENT_QK: dict[tuple, object] = {}
_SENT_QK_SOURCE = "sent_ppmi_svd_v1"


def sent_qk_sidecar_path(state: GraphState) -> Path | None:
    hint = getattr(state, "_sent_qk_sidecar", None)
    if hint:
        return Path(hint)
    ckpt = getattr(state, "_checkpoint_path", None)
    if ckpt:
        return Path(ckpt).with_suffix(".sent_qk.npz")
    return None


def _svd_qk_from_csr(src: csr_matrix, dim: int):
    from nn_aivie.hca.attention_qk import DirectedQK
    from scipy.sparse.linalg import svds

    n = int(src.shape[0])
    if src.nnz == 0:
        z = np.zeros((n, dim), dtype=np.float64)
        return DirectedQK(z, z.copy()), 0
    ppmi = ppmi_from_csr(src)
    k = min(dim, max(1, n - 1))
    if k < 1 or ppmi.nnz == 0:
        z = np.zeros((n, dim), dtype=np.float64)
        return DirectedQK(z, z.copy()), int(ppmi.nnz)
    u, s, vt = svds(ppmi.asfptype(), k=k, which="LM")
    order = np.argsort(-s)
    u, s, vt = u[:, order], s[order], vt[order]
    root = np.sqrt(np.maximum(s, 0.0))
    wq = u * root
    wk = vt.T * root
    if wq.shape[1] < dim:
        wq = np.pad(wq, ((0, 0), (0, dim - wq.shape[1])))
        wk = np.pad(wk, ((0, 0), (0, dim - wk.shape[1])))
    return DirectedQK(wq.astype(np.float64), wk.astype(np.float64)), int(ppmi.nnz)


def save_sent_qk_sidecar(
    state: GraphState,
    path: str | Path | None = None,
    *,
    dim: int = 64,
    config=None,
    logger=None,
) -> Path:
    """Bake PPMI_l2r + PPMI_r2l → Q/K (offline, không nnz-cap)."""
    path = Path(path) if path else sent_qk_sidecar_path(state)
    if path is None:
        raise ValueError("no sidecar path (set state._checkpoint_path or pass path)")
    views = ensure_sent_intra(state, config=config, force=False)
    if logger:
        logger(
            f"sent_qk bake: N={views.W_l2r.shape[0]} "
            f"l2r_nnz={views.W_l2r.nnz} r2l_nnz={views.W_r2l.nnz} dim={dim}"
        )
    qk_l, nnz_l = _svd_qk_from_csr(views.W_l2r, dim)
    qk_r, nnz_r = _svd_qk_from_csr(views.W_r2l, dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        l2r_wq=qk_l.wq,
        l2r_wk=qk_l.wk,
        r2l_wq=qk_r.wq,
        r2l_wk=qk_r.wk,
        dim=np.int32(dim),
        n=np.int32(qk_l.wq.shape[0]),
        l2r_ppmi_nnz=np.int64(nnz_l),
        r2l_ppmi_nnz=np.int64(nnz_r),
        source=np.asarray(_SENT_QK_SOURCE),
    )
    cache_put(_SENT_QK, state, qk_l, ("l2r", int(dim)))
    cache_put(_SENT_QK, state, qk_r, ("r2l", int(dim)))
    if logger:
        logger(f"✓ sent_qk sidecar → {path}")
    return path


def load_sent_qk_sidecar(
    state: GraphState,
    path: str | Path,
    *,
    dim: int | None = None,
) -> bool:
    """Load cả L2R+R2L vào cache. False nếu stale/missing."""
    from nn_aivie.hca.attention_qk import DirectedQK

    path = Path(path)
    if not path.is_file():
        return False
    try:
        z = np.load(path, allow_pickle=False)
        src = str(z["source"]) if "source" in z.files else ""
        if src != _SENT_QK_SOURCE:
            return False
        n = int(z["n"])
        if n != len(state.vocab):
            return False
        d = int(z["dim"])
        if dim is not None and int(dim) != d:
            return False
        qk_l = DirectedQK(
            np.asarray(z["l2r_wq"], dtype=np.float64),
            np.asarray(z["l2r_wk"], dtype=np.float64),
        )
        qk_r = DirectedQK(
            np.asarray(z["r2l_wq"], dtype=np.float64),
            np.asarray(z["r2l_wk"], dtype=np.float64),
        )
        cache_put(_SENT_QK, state, qk_l, ("l2r", d))
        cache_put(_SENT_QK, state, qk_r, ("r2l", d))
        return True
    except Exception:
        return False


def ensure_sent_qk(
    state: GraphState,
    *,
    which: str = "l2r",
    dim: int = 64,
    config=None,
    allow_large: bool = False,
):
    """SVD Q/K từ PPMI(SENT_L2R|R2L). Ưu tiên sidecar; runtime lớn → fallback nếu thiếu bake."""
    from nn_aivie.hca.attention_qk import DirectedQK, ensure_directed_qk

    _sub = (which, int(dim))
    cached = cache_get(_SENT_QK, state, _sub)
    if cached is not None:
        return cached

    path = sent_qk_sidecar_path(state)
    if path is not None and path.is_file():
        if load_sent_qk_sidecar(state, path, dim=dim):
            hit = cache_get(_SENT_QK, state, _sub)
            if hit is not None:
                return hit

    views = ensure_sent_intra(state, config=config)
    src = views.W_l2r if which == "l2r" else views.W_r2l
    n = int(src.shape[0])
    nnz_cap = int(getattr(config, "ar_sent_qk_nnz_cap", 2_000_000) or 2_000_000)
    too_big = src.nnz > nnz_cap or n > 80_000
    if src.nnz == 0 or (too_big and not allow_large):
        if which == "l2r":
            qk = ensure_directed_qk(state, dim=dim)
        else:
            z = np.zeros((n, dim), dtype=np.float64)
            qk = DirectedQK(z, z.copy())
        cache_put(_SENT_QK, state, qk, _sub)
        return qk

    qk, _ = _svd_qk_from_csr(src, dim)
    cache_put(_SENT_QK, state, qk, _sub)
    # nếu vừa bake một phía online và path có → không auto-save (dùng script bake đủ cặp)
    return qk


def long_context_h_logits(
    state: GraphState,
    history_ids: list[int],
    anchors: np.ndarray,
    cont: np.ndarray,
    *,
    config=None,
) -> np.ndarray:
    """P6.A / P7.5: h = decay(E[hist]) + E[top anchors] + E[top cont]; score = E h."""
    n = len(state.vocab)
    out = np.zeros(n, dtype=np.float64)
    emb = getattr(state, "embedding", None)
    mat = getattr(emb, "matrix", None) if emb is not None else None
    if mat is None or mat.shape[0] < n or not history_ids:
        return out
    d = int(mat.shape[1])
    lam = float(getattr(config, "ar_ctx_lambda", 0.85) if config else 0.85)
    h = np.zeros(d, dtype=np.float64)
    # surface history (decay từ đuôi)
    w = 1.0
    for tid in reversed(history_ids):
        tid = int(tid)
        if 0 <= tid < n and state.vocab.is_surface(tid):
            h += w * np.asarray(mat[tid], dtype=np.float64)
            w *= lam
        if w < 1e-4:
            break
    k_a = int(getattr(config, "ar_matrix_h_anchor_k", 8) if config else 8)
    k_c = int(getattr(config, "ar_matrix_h_cont_k", 8) if config else 8)
    if anchors is not None and anchors.size == n and anchors.max() > 0:
        idx = np.argpartition(-anchors, min(k_a, n) - 1)[:k_a]
        for j in idx:
            if anchors[j] > 0:
                h += float(anchors[j]) * np.asarray(mat[int(j)], dtype=np.float64)
    if cont is not None and cont.size == n and cont.max() > 0:
        idx = np.argpartition(-cont, min(k_c, n) - 1)[:k_c]
        for j in idx:
            if cont[j] > 0:
                h += 0.75 * float(cont[j]) * np.asarray(mat[int(j)], dtype=np.float64)
    hn = float(np.linalg.norm(h))
    if hn < 1e-12:
        return out
    h /= hn
    # chấm mọi token: E[j] · h
    out = np.asarray(mat[:n] @ h, dtype=np.float64)
    out = np.maximum(out, 0.0)
    return out
