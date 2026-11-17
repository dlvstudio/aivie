"""Candidate-set AR scoring — tránh dense O(V) mỗi bước sinh (TK_036 perf)."""

from __future__ import annotations

import numpy as np

from nn_aivie.core.adjacency import TypedAdjacency
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.frame_bracket_gate import qualifying_opener
from nn_aivie.hca.readout import spreading_activation_read_sparse


def build_micro_dict(
    w0: int,
    out: list[int],
    state: GraphState,
    seq_layer,
    vocab_size: int,
    config: HCAConfig,
) -> dict[int, float]:
    micro: dict[int, float] = {}
    for v, w in seq_layer.row(w0).items():
        if v < vocab_size and w > 0:
            micro[v] = w
    if state.parents is not None and config.ar_seq_order2_weight > 0:
        prev = out[-2] if len(out) >= 2 else None
        hi = state.parents.seq_next(prev, w0, min_hi=config.ar_seq_order2_min)
        if hi:
            bmax = max(micro.values()) if micro else 1.0
            hmax = max(hi.values()) or 1.0
            for v, c in hi.items():
                if v < vocab_size:
                    micro[v] = micro.get(v, 0.0) + config.ar_seq_order2_weight * (c / hmax) * bmax
    varlm = getattr(state, "varlm", None)
    if varlm is not None and getattr(config, "ar_varlm_weight", 0.0) > 0:
        ctx = out
        if getattr(varlm, "use_clusters", False) and state.parents is not None:
            ctx = state.parents.seg.encode_tokens_with_clusters(out)
        pp = varlm.scores(ctx[-varlm.max_ctx :])
        if pp:
            bmax = max(micro.values()) if micro else 1.0
            pmax = max(pp.values()) or 1.0
            for v, p in pp.items():
                if v < vocab_size:
                    micro[v] = micro.get(v, 0.0) + config.ar_varlm_weight * (p / pmax) * bmax

    # ---- TK_042 G10: LM nơ-ron THAY kênh trôi chảy ----
    #
    # `micro` (W_seq bigram) CHÍNH LÀ mắt xích 6.4-bits. G7 đo trên CÙNG corpus + held-out:
    # W_seq 7.088 bits/token vs LM 5.594 ⇒ tốt hơn **1.494 bits/token** (§7.8).
    #
    # NỘI SUY, không NHÂN (§7.10d): Hadamard là AND — LM nói X, agenda nói Y ⇒ dìm cả hai.
    # `ar_lm_weight`: 0 = W_seq thuần (hành vi CŨ) · 1 = LM thuần · 0<w<1 = trộn.
    # Kênh `macro` (agenda) GIỮ NGUYÊN ⇒ đồ thị vẫn lái NỘI DUNG từ bên ngoài ⇒ LM KHÔNG cần
    # in-context learning (thứ 11.4M tham số không có — §7.10b).
    #
    # ⚠️ PHẢI đặt ở ĐÂY, không phải nhánh `else` của `generate_autoregressive`. Bản đầu chèn
    # vào nhánh else → `ar_candidate_scoring=True` (mặc định) nên đường THẬT chạy nhánh
    # sparse → **code LM CHƯA BAO GIỜ CHẠY**: ar_lm_weight 0.0/0.5/1.0 cho output GIỐNG HỆT
    # NHAU từng byte. Không crash, không test đỏ, vẫn ra chữ — tính năng chỉ đơn giản là
    # KHÔNG TỒN TẠI. Lớp lỗi im lặng TK_041 §5.2; bắt được vì thử w=1.0 (kỳ vọng phải đổi
    # NHIỀU) chứ không phải w=0.5 (đổi ít thì dễ tự lừa là "có tác dụng nhẹ").
    nlm = getattr(state, "neural_lm", None)
    w_lm = getattr(config, "ar_lm_weight", 0.0)
    if nlm is not None and w_lm > 0:
        p_lm = nlm.probs(out)  # đã chuẩn hoá max=1
        mmax = max(micro.values()) if micro else 1.0
        blended: dict[int, float] = {}
        for v, w in micro.items():
            blended[v] = (1.0 - w_lm) * (w / mmax)
        top = np.argpartition(-p_lm, config.ar_lm_topk)[: config.ar_lm_topk]
        for v in top:
            v = int(v)
            if v < vocab_size:
                blended[v] = blended.get(v, 0.0) + w_lm * float(p_lm[v])
        micro = blended
    return micro


def collect_ar_candidates(
    w0: int,
    out: list[int],
    pending: list[int],
    agenda_seq: list[int],
    entity_ids: list[int],
    open_brackets: list[tuple[int, int]],
    state: GraphState,
    adjacency: TypedAdjacency,
    seq_layer,
    vocab_size: int,
    eos_id: int,
    config: HCAConfig,
    *,
    prompt_len: int = 0,
) -> set[int]:
    c: set[int] = set()
    c.update(v for v in seq_layer.row(w0) if v < vocab_size)
    c.update(adjacency.readout_row_ids(w0, vocab_size))  # TK_011 D3: cached filtered ids

    for i in pending:
        t = agenda_seq[i]
        if t < vocab_size:
            c.add(t)

    fb = adjacency.layer(EdgeKind.FRAME_BRACKET)
    for oid, _ in open_brackets:
        c.update(v for v in fb.row(oid) if v < vocab_size)
    qual = qualifying_opener(state, out, min_gap=config.ar_frame_bracket_min_gap)
    if qual is not None:
        c.update(v for v in fb.row(qual) if v < vocab_size)

    ps = state.parents
    if ps is not None:
        left_idx = getattr(ps, "_skip_left_index", None) or {}
        if not left_idx and getattr(ps, "_skip_lookup", None):
            ps._rebuild_skip_left_index()
            left_idx = ps._skip_left_index
        for right_ids, _ in left_idx.get(w0, ()):
            if right_ids:
                c.add(right_ids[0])

    if pending:
        for i in pending:
            sid = agenda_seq[i]
            c.update(adjacency.readout_row_ids(sid, vocab_size))  # TK_011 D3

    if config.verify_enabled:
        seen: set[int] = set()
        for a in [w0, *entity_ids, *(agenda_seq[i] for i in pending[:4])]:
            if a in seen:
                continue
            seen.add(a)
            c.update(adjacency.learned_row_ids(a, vocab_size))  # TK_011 D3

    lo = max(prompt_len, len(out) - 6)
    c.update(tid for tid in out[lo:] if tid < vocab_size)

    if 0 <= eos_id < vocab_size:
        c.add(eos_id)
    return c


def assemble_ar_scores(
    micro: dict[int, float],
    macro: dict[int, float],
    candidates: set[int],
    vocab_size: int,
    config: HCAConfig,
) -> np.ndarray:
    scores = np.full(vocab_size, -np.inf, dtype=np.float64)
    mmax = max(micro.values()) if micro else 1.0
    gmax = max(macro.values()) if macro else 1.0
    for v in candidates:
        if v >= vocab_size:
            continue
        m = micro.get(v, 0.0)
        g = macro.get(v, 0.0)
        scores[v] = (config.ar_micro_floor + m / mmax) * (
            config.ar_macro_floor + config.ar_spread_weight * g / gmax
        )
    return scores


def macro_sparse(
    w0: int,
    pending: list[int],
    agenda_seq: list[int],
    adjacency: TypedAdjacency,
    config: HCAConfig,
    vocab_size: int,
) -> dict[int, float]:
    if not pending:
        return {}
    seeds = [(agenda_seq[i], 1.0) for i in pending]
    return spreading_activation_read_sparse(w0, seeds, adjacency, config, vocab_size)
