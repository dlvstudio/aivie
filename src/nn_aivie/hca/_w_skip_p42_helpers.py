"""P42/P43 helpers — EOS, SENT_SUCC, content gate, hist sau biên."""

from __future__ import annotations

import numpy as np

from nn_aivie.core.state import GraphState
from nn_aivie.hca.context_positional import is_structural_token


def _is_vote_content_token(vocab, tid: int) -> bool:
    """Token nội dung để vote — bỏ frame Hỏi:/Đáp:, Ġ trống, structural."""
    tid = int(tid)
    if not (0 <= tid) or not vocab.is_surface(tid) or is_structural_token(vocab, tid):
        return False
    from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL, MARK_SPACE

    tok = vocab.token_of(tid)
    if tok in (DAP_LABEL, HOI_LABEL, MARK_SPACE, "Ġ", "Ċ", ""):
        return False
    bare = tok.lstrip(MARK_SPACE).lstrip("Ċ")
    return bool(bare)


def hist_after_last_boundary(
    hist: list[int],
    vocab,
    *,
    pre_boundary_k: int = 2,
    bound=None,
) -> list[int]:
    """§12.141 — hist dùng để vote: chỉ sau biên câu cuối.

    Nếu token cuối là biên: lấy tối đa `pre_boundary_k` token nội dung ngay trước
    biên (để Long→Đây qua W_skip), không kéo cả câu cũ (là hub).
    Nếu đã có token nội dung sau biên: chỉ dùng đoạn sau biên.
    Nếu sau biên chỉ còn nhãn frame (`? Đáp:`) → vote từ mệnh đề trước biên
    (không để hist=`[Đáp:]` → phiếu ~0 → pesto/English).
    """
    def _is_boundary_b(v, t):
        # §13.54 — biên ĐƠN VỊ học được; `bound=None` ⇒ mồi nguội (đường gọi cũ, ví dụ
        # test đơn vị không dựng W_seq). Ở đường SINH thì `bound` luôn được truyền.
        return _is_boundary(v, t, bound, bootstrap=bound is None)

    if not hist:
        return list(hist)
    from nn_aivie.hca.sent_intra import _is_boundary

    last_b = None
    for i in range(len(hist) - 1, -1, -1):
        if _is_boundary_b(vocab, int(hist[i])):
            last_b = i
            break
    if last_b is None:
        return list(hist)
    after = list(hist[last_b + 1 :])
    content_after = [int(t) for t in after if _is_vote_content_token(vocab, int(t))]
    if content_after:
        return content_after
    # Sau biên không có nội dung thật (đứng trên biên, hoặc chỉ `Đáp:`/`Ġ`)
    if after:
        # Frame sau `?` — lấy cả mệnh đề trước biên (câu hỏi), không cắt k=2
        pre_full = [
            int(t) for t in hist[:last_b] if _is_vote_content_token(vocab, int(t))
        ]
        return pre_full if pre_full else list(hist[-1:])
    pre: list[int] = []
    for t in reversed(hist[:last_b]):
        tid = int(t)
        if not _is_vote_content_token(vocab, tid):
            continue
        pre.append(tid)
        if len(pre) >= max(0, int(pre_boundary_k)):
            break
    pre.reverse()
    return pre if pre else list(hist[-1:])


def eos_from_end_after(
    state: GraphState,
    hist: list[int],
    *,
    eos_id: int,
    n_vocab: int,
    beta: float = 1.2,
    ref_scale: float = 1.0,
    boundary_count: int = 0,
    episode_eos_boost: float = 1.8,
) -> float:
    """§12.141/P42 — điểm EOS từ end_after/cont_after (+ succ→EOS)."""
    if beta <= 0.0 or not hist or state.parents is None:
        return 0.0
    if not (0 <= int(eos_id) < int(n_vocab)):
        return 0.0
    from nn_aivie.hca.matrix_core import ensure_parent_bipartite

    bi = ensure_parent_bipartite(state)
    if bi.end_after.size == 0:
        return 0.0
    ea = bi.end_after
    ca = bi.cont_after if bi.cont_after.size == ea.shape[0] else None
    n_ea = int(ea.shape[0])
    w0 = int(hist[-1])
    s = 0.0
    if 0 <= w0 < n_ea:
        e = float(ea[w0])
        c = float(ca[w0]) if ca is not None else 0.0
        s += e
        if e + c > 0:
            s += e * (e / (e + c))
        for _pi, nxt, w in bi.succ.get(w0, ()):
            if int(nxt) == int(eos_id):
                s += 0.35 * float(w)
    ctx = np.zeros(n_ea, dtype=np.float64)
    for tid in hist[-8:]:
        tid = int(tid)
        if 0 <= tid < n_ea:
            ctx[tid] += 1.0
    s += 0.15 * float(ctx @ ea)
    if s <= 0.0:
        return 0.0
    sm = float(ref_scale) if ref_scale > 0 else 1.0
    score = float(beta) * (s / (s + sm))
    if int(boundary_count) >= 1 and episode_eos_boost > 1.0:
        score *= float(episode_eos_boost)
    return score


def sent_open_from_w_agenda(
    state: GraphState,
    hist: list[int],
    n_vocab: int,
    *,
    beta: float = 1.5,
) -> np.ndarray:
    """§12.141/P42 — sau biên câu: boost mở câu sau qua W_agenda (SENT_SUCC)."""
    out = np.zeros(int(n_vocab), dtype=np.float64)
    if beta <= 0.0 or not hist:
        return out
    from nn_aivie.hca.sent_intra import _is_boundary, ensure_sent_intra

    from nn_aivie.hca.sent_intra import _bound_of

    w0 = int(hist[-1])
    # §13.58 — biên ĐƠN VỊ học được; thiếu `bound` ⇒ boost W_agenda chết âm thầm.
    if not _is_boundary(state.vocab, w0, _bound_of(state)):
        return out
    try:
        views = ensure_sent_intra(state)
    except Exception:
        return out
    W = getattr(views, "W_agenda", None)
    if W is None or getattr(W, "nnz", 0) == 0:
        return out
    src = w0
    for t in reversed(hist[:-1]):
        tid = int(t)
        if 0 <= tid < n_vocab and state.vocab.is_surface(tid):
            if not is_structural_token(state.vocab, tid):
                src = tid
                break
    if not (0 <= src < W.shape[0]):
        return out
    row = W.getrow(src)
    if row is None or row.nnz == 0:
        if 0 <= w0 < W.shape[0]:
            row = W.getrow(w0)
        if row is None or row.nnz == 0:
            return out
    mx = float(row.data.max()) if row.nnz else 0.0
    if mx <= 0:
        return out
    for j, w in zip(row.indices, row.data):
        if 0 <= int(j) < n_vocab:
            out[int(j)] = float(beta) * float(w) / mx
    return out


def content_channel_gate(
    n_vocab: int,
    *,
    agenda_ids: list[int] | None = None,
    entity_ids: tuple[int, ...] | set[int] = (),
    prompt_ids: list[int] | None = None,
    floor: float = 0.2,
    vocab=None,
) -> np.ndarray:
    """§12.146/P43 — cổng nội dung mềm: agenda + entity + token prompt."""
    g = np.full(int(n_vocab), float(floor), dtype=np.float64)
    for t in agenda_ids or ():
        if 0 <= int(t) < n_vocab:
            g[int(t)] = 1.0
    for t in entity_ids or ():
        if 0 <= int(t) < n_vocab:
            g[int(t)] = 1.0
    if prompt_ids and vocab is not None:
        for t in prompt_ids:
            tid = int(t)
            if not (0 <= tid < n_vocab):
                continue
            if not vocab.is_surface(tid) or is_structural_token(vocab, tid):
                continue
            g[tid] = max(float(g[tid]), 1.0)
    return g


def count_boundaries(hist: list[int], vocab, bound=None) -> int:
    """§13.58 — đếm mốc biên trong hist.

    ⚠️ Hàm này từng gọi `_is_boundary_b`, một closure của `hist_after_last_boundary` — tức
    NameError ngay khi chạy. Suite 361 test KHÔNG bắt: không test nào đi qua đây. Lỗi chỉ
    lộ khi chạy `generate()` thật trên đường AR.
    """
    from nn_aivie.hca.sent_intra import _is_boundary

    return sum(
        1 for t in hist
        if _is_boundary(vocab, int(t), bound, bootstrap=bound is None)
    )


def sent_open_from_w_skip(
    state: GraphState,
    hist: list[int],
    n_vocab: int,
    *,
    beta: float = 1.5,
    d_max: int = 4,
    lam: float = 0.5,
    topk: int = 32,
) -> np.ndarray:
    """§12.141 — sau biên: boost đích W_skip từ token nội dung sát trước biên.

    Bổ sung SENT_SUCC: `Long→Đây` nằm trên W_skip xuyên câu (§12.143).
    """
    out = np.zeros(int(n_vocab), dtype=np.float64)
    if beta <= 0.0 or not hist:
        return out
    from nn_aivie.hca.sent_intra import _is_boundary
    from nn_aivie.hca.w_skip import ensure_w_skip

    from nn_aivie.hca.sent_intra import _bound_of

    _bd = _bound_of(state)   # §13.54 — đường SINH: dùng tín hiệu biên HỌC ĐƯỢC
    if not _is_boundary(state.vocab, int(hist[-1]), _bd):
        return out
    pre = hist_after_last_boundary(hist, state.vocab, pre_boundary_k=2, bound=_bd)
    if not pre:
        return out
    ws = ensure_w_skip(state, d_max=max(4, int(d_max)), min_count=2, content_only=False)
    for src in pre:
        src = int(src)
        if not (0 <= src < n_vocab):
            continue
        for d in range(1, int(d_max) + 1):
            if d > ws.d_max:
                break
            w_d = float(lam) ** (d - 1)
            for j, c in ws.row_topk(src, d, int(topk)):
                if 0 <= int(j) < n_vocab and not is_structural_token(
                    state.vocab, int(j)
                ):
                    out[int(j)] += float(beta) * w_d * float(c)
    mx = float(out.max()) if out.size else 0.0
    if mx > 0:
        out /= mx
        out *= float(beta)
    return out
