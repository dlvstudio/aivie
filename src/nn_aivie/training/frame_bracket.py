"""
TK_016 D3 + TK_037 — FRAME_BRACKET: cạnh có hướng giữa endpoint khung XA (gap slot ≥2).

TK_037: endpoint = token | ISG cluster | learned phrase (SlotRef), không chỉ token→token.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.frame_bracket_ref import (
    FrameEndpoint,
    endpoint_from_phrase,
    endpoint_from_slot,
    phrase_adj_id,
)
from nn_aivie.core.seg_graph import CLUSTER_OFFSET, SlotRef
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import is_surface_token
from nn_aivie.training.skip_span import build_phrase_by_start


def _is_frame_token(
    tid: int,
    vocab,
    df: Counter[int],
    func: set[int],
    *,
    min_df_opener: int,
) -> bool:
    if not is_surface_token(vocab.token_of(tid)):
        return False
    return tid in func or df.get(tid, 0) >= min_df_opener


def _is_frame_cluster(ci: int, seg, vocab, func: set[int], *, min_power: int) -> bool:
    if ci < 0 or ci >= len(seg.clusters):
        return False
    c = seg.clusters[ci]
    if int(getattr(c, "power", 0) or 0) < min_power:
        return False
    return c.left in func or c.right in func


def _build_frame_phrase_by_start(ps, vocab, df: Counter[int], func: set[int], *, min_df: int) -> dict[int, list[tuple[int, ...]]]:
    """Phrase 2–4 token làm endpoint khung."""
    raw = build_phrase_by_start(ps, min_count=min_df)
    out: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for start, phrases in raw.items():
        for ph in phrases:
            if not (2 <= len(ph) <= 4):
                continue
            if not any(_is_frame_token(t, vocab, df, func, min_df_opener=1) for t in ph):
                continue
            out[start].append(ph)
    return out


def _phrase_endpoints_in_chain(
    chain: tuple[SlotRef, ...],
    seg,
    frame_phrases: dict[int, list[tuple[int, ...]]],
    ps,
) -> list[tuple[int, FrameEndpoint]]:
    """(slot_index, endpoint) cho phrase khung khớp chuỗi token mở rộng."""
    exp: list[tuple[int, int]] = []
    for si, ref in enumerate(chain):
        for tid in seg.expand_slot(ref):
            exp.append((si, tid))
    hits: list[tuple[int, FrameEndpoint]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for i, (si, t0) in enumerate(exp):
        for ph in frame_phrases.get(t0, ()):
            L = len(ph)
            if i + L > len(exp):
                continue
            if tuple(t for _s, t in exp[i : i + L]) != ph:
                continue
            end_si = exp[i + L - 1][0]
            key = (si, ph)
            if key in seen:
                continue
            seen.add(key)
            pidx = ps._phrase_lookup.get(ph)
            if pidx is None:
                continue
            hits.append((si, endpoint_from_phrase(ph, pidx)))
    return hits


def _collect_frame_positions(
    rec,
    ps,
    vocab,
    seg,
    df: Counter[int],
    func: set[int],
    frame_phrases: dict[int, list[tuple[int, ...]]],
    *,
    min_df_opener: int,
    min_cluster_power: int,
) -> list[tuple[int, FrameEndpoint, int]]:
    """(slot_idx, endpoint, opener_token_pos_in_parent) — token pos cho prov/gate."""
    chain = rec.slot_chain
    if not chain:
        chain = tuple(
            SlotRef("token", t)
            for t in rec.source_ids
            if _is_frame_token(t, vocab, df, func, min_df_opener=min_df_opener)
        )
    positions: list[tuple[int, FrameEndpoint, int]] = []
    token_pos = 0
    for si, ref in enumerate(chain):
        opener_tok_pos = token_pos
        if ref.kind == "token":
            if _is_frame_token(ref.id, vocab, df, func, min_df_opener=min_df_opener):
                positions.append((si, FrameEndpoint("token", (ref.id,)), opener_tok_pos))
            token_pos += 1
        else:
            ci = ref.id - CLUSTER_OFFSET
            if _is_frame_cluster(ci, seg, vocab, func, min_power=min_cluster_power):
                ep = endpoint_from_slot(ref, seg)
                if ep is not None:
                    positions.append((si, ep, opener_tok_pos))
            token_pos += len(seg.expand_slot(ref))
    for si, ep in _phrase_endpoints_in_chain(chain, seg, frame_phrases, ps):
        tok_pos = 0
        for j, ref in enumerate(chain):
            if j == si:
                break
            tok_pos += len(seg.expand_slot(ref))
        positions.append((si, ep, tok_pos))
    best: dict[int, tuple[FrameEndpoint, int]] = {}
    pri = {"cluster": 2, "phrase": 1, "token": 0}
    for si, ep, tp in positions:
        cur = best.get(si)
        if cur is None or pri[ep.kind] > pri[cur[0].kind]:
            best[si] = (ep, tp)
    return [(si, ep, tp) for si, (ep, tp) in sorted(best.items())]


def mine_frame_brackets(
    state: GraphState,
    *,
    min_df: int = 25,
    max_df_frac: float = 0.4,
    min_df_opener: int = 2,
    min_gap: int = 2,
    max_gap: int = 20,
    min_count: int = 4,
    min_pmi: float = 1.3,
    dir_ratio: float = 2.0,
    max_prov_per_pair: int = 50,
    min_cluster_power: int = 3,
    logger=None,
) -> dict[str, int]:
    """Khai phá FRAME_BRACKET trên slot_chain (TK_037) + fallback token path.

    Đồng-hiện (power) = số parent có cặp (opener, closer); mỗi parent đếm tối đa 1 lần
    cho cùng cặp (gap ghi nhận = khoảng slot nhỏ nhất trong parent đó).
    """
    ps = state.parents
    vocab = state.vocab
    if ps is None:
        return {"edges": 0}
    seg = ps.seg

    def _log(msg: str) -> None:
        if logger is not None:
            logger.log(msg)

    df: Counter[int] = Counter()
    for rec in ps.parents:
        for t in set(rec.source_ids):
            if is_surface_token(vocab.token_of(t)):
                df[t] += 1
    n_par = len(ps.parents)
    df_hi = max(min_df + 1, int(max_df_frac * n_par))
    func = {t for t, d in df.items() if min_df <= d <= df_hi}
    _log(f"function tokens (df∈[{min_df},{df_hi}]) = {len(func)}")

    frame_phrases = _build_frame_phrase_by_start(ps, vocab, df, func, min_df=min_df_opener)

    pair: dict[tuple[int, int], int] = defaultdict(int)
    a_tot: Counter[int] = Counter()
    b_tot: Counter[int] = Counter()
    endpoints: dict[int, FrameEndpoint] = {}
    total = 0

    # Fix #6: KHÔNG giữ gaps/prov/edge_kinds cho cả ~6.5M cặp thô (spike RAM vài GB
    # lúc bake). Pass 1 chỉ đếm count; lọc survivor; Pass 2 gom gaps/prov CHỈ cho
    # survivor (số ít). edge_kinds suy từ endpoints (kind xác định bởi id range).
    # Byte-identical (verify chữ ký hlink_meta + FRAME_BRACKET edges). Giá: 2× scan.
    def _parent_pairs(rec) -> dict[tuple[int, int], tuple[int, int]]:
        """Cặp (opener→closer) đã dedup trong một cha (min-gap); populate endpoints."""
        fpos = _collect_frame_positions(
            rec, ps, vocab, seg, df, func, frame_phrases,
            min_df_opener=min_df_opener, min_cluster_power=min_cluster_power,
        )
        pp: dict[tuple[int, int], tuple[int, int]] = {}
        for x in range(len(fpos)):
            si, a_ep, a_tok = fpos[x]
            a_id = a_ep.adj_id()
            endpoints[a_id] = a_ep
            for y in range(x + 1, len(fpos)):
                gap = fpos[y][0] - si
                if gap < min_gap:
                    continue
                if gap > max_gap:
                    break
                _b_si, b_ep, _b_tok = fpos[y]
                b_id = b_ep.adj_id()
                endpoints[b_id] = b_ep
                key = (a_id, b_id)
                prev = pp.get(key)
                if prev is None or gap < prev[0]:
                    pp[key] = (gap, a_tok)
        return pp

    # Pass 1 — CHỈ đếm count + a_tot/b_tot (+ endpoints, total). Không gaps/prov.
    for k, rec in enumerate(ps.parents):
        pp = _parent_pairs(rec)
        parent_openers: set[int] = set()
        parent_closers: set[int] = set()
        for (a_id, b_id), (gap, a_tok) in pp.items():
            pair[(a_id, b_id)] += 1
            parent_openers.add(a_id)
            parent_closers.add(b_id)
            total += 1
        for a_id in parent_openers:
            a_tot[a_id] += 1
        for b_id in parent_closers:
            b_tot[b_id] += 1
        if logger is not None and (k + 1) % 20000 == 0:
            logger.progress(
                "scan parents (pass1)", k + 1, len(ps.parents), extra=f"pairs={len(pair)}"
            )
    _log(f"cặp thô = {len(pair)}, tổng đồng-hiện = {total}")

    # Lọc survivor (min_count + dir_ratio + PMI) — pass1 đủ dữ liệu; giữ thứ tự pair.items().
    survivors: dict[tuple[int, int], float] = {}
    for (a, b), c in pair.items():
        if c < min_count:
            continue
        bwd = pair.get((b, a), 0)
        if c < dir_ratio * max(bwd, 1):
            continue
        pmi = math.log((c * total) / (a_tot[a] * b_tot[b]))
        if pmi < min_pmi:
            continue
        survivors[(a, b)] = pmi
    _log(f"survivor = {len(survivors)}; pass2 gom gaps/prov …")

    # Pass 2 — gaps/prov CHỈ cho survivor (nhỏ). Cùng thứ tự cha → gaps/prov y hệt.
    gaps: dict[tuple[int, int], Counter] = defaultdict(Counter)
    prov: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for k, rec in enumerate(ps.parents):
        pp = _parent_pairs(rec)
        for key, (gap, a_tok) in pp.items():
            if key not in survivors:
                continue
            gaps[key][gap] += 1
            if len(prov[key]) < max_prov_per_pair:
                prov[key].append((k, a_tok, gap))
        if logger is not None and (k + 1) % 20000 == 0:
            logger.progress("scan parents (pass2)", k + 1, len(ps.parents))

    n_edges = 0
    meta: dict[tuple[int, int], dict] = {}
    from nn_aivie.core.adjacency import SparseAdjacency

    state.adjacency._layers[EdgeKind.FRAME_BRACKET] = SparseAdjacency()
    for (a, b), pmi in survivors.items():
        c = pair[(a, b)]
        state.adjacency.set(a, b, pmi, EdgeKind.FRAME_BRACKET)
        ep_a = endpoints.get(a)
        ep_b = endpoints.get(b)
        lk = ep_a.kind if ep_a is not None else "token"
        rk = ep_b.kind if ep_b is not None else "token"
        meta[(a, b)] = {
            "power": c,
            "gaps": dict(gaps[(a, b)]),
            "prov": prov[(a, b)],
            "left_kind": lk,
            "right_kind": rk,
        }
        n_edges += 1
    ps.hlink_meta = meta
    ps.hlink_endpoints = endpoints
    n_cluster = sum(1 for e in endpoints.values() if e.kind == "cluster")
    n_phrase = sum(1 for e in endpoints.values() if e.kind == "phrase")
    _log(
        f"✓ FRAME_BRACKET edges = {n_edges} "
        f"(endpoints: {len(endpoints)} token/cluster/phrase = "
        f"{len(endpoints)-n_cluster-n_phrase}/{n_cluster}/{n_phrase})"
    )
    return {
        "function_tokens": len(func),
        "pairs": len(pair),
        "edges": n_edges,
        "endpoints": len(endpoints),
        "cluster_endpoints": n_cluster,
        "phrase_endpoints": n_phrase,
    }


def top_brackets(state: GraphState, opener: str, k: int = 8) -> list[tuple[str, float]]:
    """Đọc tay: closer mạnh nhất của opener (token hoặc label cluster/phrase)."""
    from nn_aivie.core.frame_bracket_ref import decode_endpoint_label

    vocab = state.vocab
    ps = state.parents
    endpoints = getattr(ps, "hlink_endpoints", None) or {}
    aid = vocab._token_to_id.get(opener)
    if aid is None:
        return []
    row = state.adjacency.layer(EdgeKind.FRAME_BRACKET).row(aid)
    top = sorted(row.items(), key=lambda kv: -kv[1])[:k]
    return [
        (decode_endpoint_label(b, endpoints, vocab), round(w, 2))
        for b, w in top
    ]
