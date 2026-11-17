"""
TK_017 + TK_037 — position-targeted FRAME_BRACKET gate cho A3 readout.

Endpoints: token | ISG cluster | phrase (boost first token of multi-token closer).
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.frame_bracket_ref import (
    expand_endpoint_tokens,
    is_cluster_adj_id,
    is_phrase_adj_id,
)
from nn_aivie.core.seg_graph import CLUSTER_OFFSET
from nn_aivie.core.state import GraphState
from nn_aivie.hca.readout import multiply_scores_by_boost
from nn_aivie.training.frame_bracket import mine_frame_brackets
from nn_aivie.hca.objcache import cache_get, cache_put

if TYPE_CHECKING:
    from nn_aivie.hca.config import HCAConfig

# §12.30: `set[id]` cùng lỗi id-tái-dùng ⇒ tưởng ĐÃ mine ⇒ bỏ qua ÂM THẦM.
_mined_state_ids: dict = {}


def _runtime_mine_allowed(state: GraphState) -> bool:
    """Guard runtime auto-mine: checkpoint lớn phải bake offline, không mine lúc generate."""
    if os.getenv("AIVIE_RUNTIME_FB_FORCE_MINE", "0") == "1":
        return True
    ps = state.parents
    if ps is None:
        return False
    cap = int(os.getenv("AIVIE_RUNTIME_FB_MINE_MAX_PARENTS", "5000"))
    return len(ps.parents) <= cap


def _warn_mine_skipped(state: GraphState, logger=None) -> None:
    """Báo guard đã TẮT gate — phải thấy được kể cả khi caller không truyền logger.

    Mọi call site trong `generate_hebbian` / `viz.server` gọi `ensure_frame_brackets(state)`
    KHÔNG logger, nên `if logger is not None: logger.log(...)` là đường chết: gate tắt lặng
    lẽ trên mọi checkpoint >cap. Dùng `warnings.warn` để luôn có tín hiệu (Python in một lần
    mỗi call site — đủ báo, không spam mỗi token).
    """
    ps = state.parents
    n_par = len(ps.parents) if ps is not None else 0
    if n_par == 0:
        return  # không có cha → không có gì để mine; guard không làm mất tính năng nào
    cap = int(os.getenv("AIVIE_RUNTIME_FB_MINE_MAX_PARENTS", "5000"))
    ckpt = getattr(state, "_checkpoint_path", None) or "<checkpoint>"
    msg = (
        f"FRAME_BRACKET gate TẮT: checkpoint {n_par} cha > cap {cap} nên bỏ mine lúc runtime, "
        f"mà checkpoint chưa bake đủ (thiếu layer/hlink_meta/hlink_endpoints). Bracket gate "
        f"(TK_017/TK_037) sẽ KHÔNG hoạt động. Sửa: "
        f"PYTHONPATH=src python scripts/bake_checkpoint_indices.py {ckpt} --in-place"
    )
    if logger is not None:
        logger.log(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _hlink_endpoints(state: GraphState) -> dict:
    ps = state.parents
    if ps is None:
        return {}
    return getattr(ps, "hlink_endpoints", None) or {}


def _completed_cluster_opener(prefix: list[int], state: GraphState, min_gap: int) -> tuple[int, int] | None:
    """(cluster_adj_id, end_position) nếu suffix vừa hoàn thành cụm opener."""
    ps = state.parents
    if ps is None or len(prefix) < 2:
        return None
    seg = ps.seg
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    pair = (prefix[-2], prefix[-1])
    ci = seg.cluster_of_pair.get(pair)
    if ci is None:
        return None
    cid = seg.cluster_id(ci)
    if not fb.row(cid):
        return None
    end_pos = len(prefix) - 1
    if len(prefix) - end_pos < min_gap:
        return None
    return cid, end_pos


def _completed_phrase_opener(prefix: list[int], state: GraphState, min_gap: int) -> tuple[int, int] | None:
    endpoints = _hlink_endpoints(state)
    if not endpoints:
        return None
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    n = len(prefix)
    best: tuple[int, int, int] | None = None
    for eid, ep in endpoints.items():
        if ep.kind != "phrase" or not fb.row(eid):
            continue
        ph = ep.label_tokens()
        L = len(ph)
        if L < 2 or n < L:
            continue
        if tuple(prefix[-L:]) != ph:
            continue
        if n - (n - 1) < min_gap:
            continue
        if best is None or L > best[2]:
            best = (eid, n - 1, L)
    return (best[0], best[1]) if best else None


def ensure_frame_brackets(state: GraphState, *, logger=None) -> int:
    """Mine FRAME_BRACKET nếu layer rỗng HOẶC thiếu meta/endpoints (TK_037)."""
    from nn_aivie.core.adjacency import SparseAdjacency

    if cache_get(_mined_state_ids, state) is not None:
        return len(state.adjacency.layer(EdgeKind.FRAME_BRACKET))
    layer = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    ps = state.parents
    has_meta = ps is not None and bool(getattr(ps, "hlink_meta", None))
    has_eps = ps is not None and bool(getattr(ps, "hlink_endpoints", None))
    if len(layer) == 0 or not has_meta or not has_eps:
        if not _runtime_mine_allowed(state):
            _warn_mine_skipped(state, logger)
            cache_put(_mined_state_ids, state, True)
            return len(layer)
        if len(layer) > 0:
            state.adjacency._layers[EdgeKind.FRAME_BRACKET] = SparseAdjacency()
        mine_frame_brackets(state, logger=logger)
    cache_put(_mined_state_ids, state, True)
    return len(state.adjacency.layer(EdgeKind.FRAME_BRACKET))


def fb_boost_from_opener(state: GraphState, opener: int, vocab_size: int) -> np.ndarray:
    boost = np.zeros(vocab_size, dtype=np.float64)
    endpoints = _hlink_endpoints(state)
    seg = state.parents.seg if state.parents else None
    row = state.adjacency.layer(EdgeKind.FRAME_BRACKET).row(opener)
    for b, wv in row.items():
        for tid in expand_endpoint_tokens(b, endpoints, seg, vocab_size):
            if tid < vocab_size:
                boost[tid] = max(boost[tid], wv)
    return boost


def qualifying_opener(
    state: GraphState,
    prefix: list[int],
    *,
    min_gap: int = 2,
) -> int | None:
    """Opener gần nhất: cluster/phrase hoàn thành hoặc token có outgoing FB."""
    n = len(prefix)
    cl = _completed_cluster_opener(prefix, state, min_gap)
    if cl is not None:
        cid, end = cl
        if n - 1 - end >= min_gap - 1 or n - end >= min_gap:
            return cid
    ph = _completed_phrase_opener(prefix, state, min_gap)
    if ph is not None:
        return ph[0]
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    for j in range(n - 1, -1, -1):
        if n - j < min_gap:
            continue
        if fb.row(prefix[j]):
            return prefix[j]
    return None


def _opener_end_position(
    state: GraphState,
    prefix: list[int],
    opener: int,
) -> int:
    """Vị trí token cuối của opener (cluster/phrase) hoặc token opener."""
    n = len(prefix)
    endpoints = _hlink_endpoints(state)
    ep = endpoints.get(opener)
    if ep is not None and ep.kind in ("cluster", "phrase"):
        L = len(ep.label_tokens())
        if L and n >= L and tuple(prefix[-L:]) == ep.label_tokens():
            return n - 1
    for j in range(n - 1, -1, -1):
        if prefix[j] == opener or (
            is_cluster_adj_id(opener)
            and j >= 1
            and state.parents is not None
            and state.parents.seg.cluster_of_pair.get((prefix[j - 1], prefix[j])) is not None
            and state.parents.seg.cluster_id(
                state.parents.seg.cluster_of_pair[(prefix[j - 1], prefix[j])]
            ) == opener
        ):
            return j
    return n - 1


def update_open_brackets(
    open_brackets: list[tuple[int, int]],
    token_id: int,
    pos: int,
    state: GraphState,
    out: list[int] | None = None,
) -> None:
    """Đóng khung nếu token khớp closer; mở token/cluster/phrase opener."""
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    endpoints = _hlink_endpoints(state)
    seg = state.parents.seg if state.parents else None
    vs = len(state.vocab) if state.vocab else 0
    prefix = out if out is not None else []

    kept: list[tuple[int, int]] = []
    for oid, oidx in open_brackets:
        row = fb.row(oid)
        closed = False
        for b, wv in row.items():
            if wv <= 0:
                continue
            closer_toks = expand_endpoint_tokens(b, endpoints, seg, vs)
            if token_id in closer_toks:
                closed = True
                break
            if is_cluster_adj_id(b) and len(prefix) >= 2:
                ci = seg.cluster_of_pair.get((prefix[-2], prefix[-1])) if seg else None
                if ci is not None and seg.cluster_id(ci) == b:
                    closed = True
                    break
        if not closed:
            kept.append((oid, oidx))
    open_brackets[:] = kept

    if fb.row(token_id):
        open_brackets.append((token_id, pos))

    if seg is not None and len(prefix) >= 2:
        ci = seg.cluster_of_pair.get((prefix[-2], prefix[-1]))
        if ci is not None:
            cid = seg.cluster_id(ci)
            if fb.row(cid):
                open_brackets.append((cid, pos))

    if endpoints and len(prefix) >= 2:
        for eid, ep in endpoints.items():
            if ep.kind != "phrase" or not fb.row(eid):
                continue
            ph = ep.label_tokens()
            if ph and tuple(prefix[-len(ph) :]) == ph:
                open_brackets.append((eid, pos))


def opener_for_gate(
    state: GraphState,
    out: list[int],
    open_brackets: list[tuple[int, int]],
    *,
    min_gap: int = 2,
) -> int | None:
    n = len(out)
    for oid, oidx in reversed(open_brackets):
        if n - oidx >= min_gap:
            return oid
    return qualifying_opener(state, out, min_gap=min_gap)


def apply_frame_bracket_gate(
    scores: np.ndarray,
    state: GraphState,
    out: list[int],
    open_brackets: list[tuple[int, int]],
    config: HCAConfig,
    vocab_size: int,
    active_parents: frozenset[int] | None = None,
    gate_trace: dict | None = None,
) -> bool:
    w = config.ar_frame_bracket_weight
    if w <= 0:
        return False
    opener = None
    oidx = -1
    n = len(out)
    for oid, oi in reversed(open_brackets):
        if n - oi >= config.ar_frame_bracket_min_gap:
            opener, oidx = oid, oi
            break
    if opener is None:
        opener = qualifying_opener(state, out, min_gap=config.ar_frame_bracket_min_gap)
        if opener is not None:
            oidx = _opener_end_position(state, out, opener)
    if opener is None:
        return False
    d = n - oidx
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    meta = getattr(state.parents, "hlink_meta", None) or {}
    endpoints = _hlink_endpoints(state)
    seg = state.parents.seg if state.parents else None
    boost = np.zeros(vocab_size, dtype=np.float64)
    override_b = -1
    override_best = 0.0
    for b, pmi in fb.row(opener).items():
        closer_toks = expand_endpoint_tokens(b, endpoints, seg, vocab_size)
        if not closer_toks:
            continue
        m = meta.get((opener, b))
        val = pmi
        p_gap = 0.0
        ctx_hit = False
        if config.ar_gap_prior and m is not None:
            tot = m["power"] or 1
            p_gap = m["gaps"].get(d, 0) / tot
            if p_gap <= 0 and m["gaps"]:
                p_gap = max(m["gaps"].values()) / tot
            val = pmi * p_gap
        if (
            active_parents and m is not None
            and config.ar_hlink_ctx_weight > 0
            and any(pi in active_parents for pi, _pos, _g in m["prov"])
        ):
            ctx_hit = True
            val *= 1.0 + config.ar_hlink_ctx_weight
        t0 = closer_toks[0]
        if t0 < vocab_size:
            boost[t0] = max(boost[t0], val)
            if len(closer_toks) > 1:
                for tid in closer_toks[1:]:
                    if tid < vocab_size:
                        boost[tid] = max(boost[tid], val * 0.35)
        if (
            config.ar_gap_override_margin > 0
            and p_gap >= config.ar_gap_override_min_p
            and ctx_hit
            and val > override_best
        ):
            override_b = t0
            override_best = val
    bmax = boost.max()
    if bmax <= 0:
        return False
    floor = config.ar_frame_bracket_floor
    multiply_scores_by_boost(
        scores, boost, floor=floor, weight=w,
    )
    if override_b >= 0 and np.isfinite(scores[override_b]):
        finite = scores[np.isfinite(scores)]
        cur_max = float(finite.max()) if finite.size else 0.0
        scores[override_b] = max(float(scores[override_b]), cur_max) * (
            1.0 + config.ar_gap_override_margin
        )
        if gate_trace is not None:
            gate_trace["override"] = int(override_b)
    if gate_trace is not None:
        gate_trace.update({"opener": int(opener), "gap": int(d)})
    return True


def seed_open_brackets(
    state: GraphState,
    prefix: list[int],
    open_brackets: list[tuple[int, int]],
) -> None:
    """Khởi tạo khung mở từ prompt: token, cluster, phrase openers."""
    open_brackets.clear()
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    endpoints = _hlink_endpoints(state)
    n = len(prefix)
    for i, tid in enumerate(prefix):
        if fb.row(tid):
            open_brackets.append((tid, i))
    if state.parents is not None:
        seg = state.parents.seg
        for j in range(1, n):
            ci = seg.cluster_of_pair.get((prefix[j - 1], prefix[j]))
            if ci is None:
                continue
            cid = seg.cluster_id(ci)
            if fb.row(cid):
                open_brackets.append((cid, j))
        for eid, ep in endpoints.items():
            if ep.kind != "phrase" or not fb.row(eid):
                continue
            ph = ep.label_tokens()
            L = len(ph)
            if L and n >= L and tuple(prefix[-L:]) == ph:
                open_brackets.append((eid, n - 1))
