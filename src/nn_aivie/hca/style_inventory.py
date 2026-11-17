"""
Style inventory — lọc marker / cụm inject tại ranh giới (2026-07-23).

Boundary inject thô chèn `cốt` (domain noun). Inventory lọc Luật-0:
  1) **ưu tiên** n-gram thân styled corpus (label-conditioned tf×IDF) — tránh
     neighbor CO_ACT kiểu giải trí (`cốt/kịch`)
  2) fallback graph: W×IDF × boundary-fitness ÷ content-leak
  3) cụm 2-token: bigram thân corpus, hoặc W_seq giữa ứng viên graph

Không hardcode danh sách marker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.hca.style_corpus import (
    ranked_corpus_inventory,
    ranked_corpus_phrases,
    ranked_discourse_ngrams,
)
from nn_aivie.hca.style_readout import (
    _core,
    _is_artifact,
    style_boost_from_coact,
)


@dataclass(frozen=True)
class StylePhrase:
    tokens: tuple[int, ...]
    score: float


def _punct_boundary_ids(vocab) -> list[int]:
    """Token ranh giới diễn ngôn — thuộc-tính surface, không list tiếng Việt."""
    out: list[int] = []
    seen: set[int] = set()
    for s in (".", ",", "!", "?", ":", ";", "Ċ", "Ġ."):
        tid = vocab._token_to_id.get(s)  # noqa: SLF001
        if tid is not None and tid not in seen:
            out.append(tid)
            seen.add(tid)
    for tok, tid in vocab._token_to_id.items():  # noqa: SLF001
        core = _core(tok)
        if core in ".!?,;:" and tid not in seen:
            out.append(tid)
            seen.add(tid)
        if len(out) > 24:
            break
    return out


def boundary_fitness(state, token_id: int) -> float:
    """Σ W_seq(punct → token). Cao = hay mở mệnh đề sau dấu."""
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    total = 0.0
    for p in _punct_boundary_ids(state.vocab):
        w = seq.get(p, token_id)
        if w > 0:
            total += float(w)
    return total


def content_leak_score(
    state,
    token_id: int,
    control_ids: list[int],
) -> float:
    """Độ mạnh CO_ACT với control content (entity). Cao = hay là fact/domain, không phải tone."""
    if not control_ids:
        return 0.0
    co = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
    s = 0.0
    for c in control_ids:
        s += float(co.get(c, token_id)) + float(co.get(token_id, c))
    return s


def _default_controls(vocab) -> list[int]:
    out: list[int] = []
    for w in ("Trần Hưng Đạo", "Ngô Quyền", "Hà Nội", "Nhà Lý"):
        out.extend(vocab.encode_prompt(w))
    return list(dict.fromkeys(out))


def ranked_style_inventory(
    state,
    query_ids: list[int],
    *,
    top_k: int = 24,
    gidf_min: float = 4.5,
    c_min: float = 2.0,
    max_seeds: int = 2,
    min_boundary: float = 0.5,
) -> list[tuple[int, float]]:
    """[(token_id, score)] — ưu tiên boundary-opener có liên kết style, demote domain leak.

    Đo 2026-07-23: rank = style×log(1+bf) vẫn để `cốt/kịch` thắng (boost lớn, bf vừa).
    → Lọc ứng viên có bf≥min; giữ nửa trên theo bf; score = bf×log(1+style)/(1+leak).
    `khiếu` bf≈0.9 nên min mặc định 0.5 (không 1.0).
    """
    vocab, ps = state.vocab, state.parents
    if ps is None:
        return []
    boost, seeds = style_boost_from_coact(
        state, query_ids, len(vocab),
        gidf_min=gidf_min, c_min=c_min, top_k=max(top_k * 4, 64), max_seeds=max_seeds,
    )
    if not seeds or float(boost.max()) <= 0:
        return []
    controls = _default_controls(vocab)
    qset = set(query_ids)
    controls = [c for c in controls if c not in qset]

    pool: list[tuple[float, float, float, int]] = []  # bf, style, leak, j
    for j in np.where(boost > 0)[0]:
        j = int(j)
        tok = vocab.token_of(j)
        if vocab.cap_class(j) >= 1:
            continue
        if _is_artifact(tok):
            continue
        # bỏ token quá ngắn / mảnh BPE 1 ký tự
        if len(_core(tok)) <= 1:
            continue
        bf = boundary_fitness(state, j)
        if bf < min_boundary:
            continue
        leak = content_leak_score(state, j, controls)
        pool.append((bf, float(boost[j]), leak, j))
    if not pool:
        return []
    # nửa trên theo boundary-fitness
    pool.sort(key=lambda x: -x[0])
    keep_n = max(top_k, (len(pool) + 1) // 2)
    pool = pool[:keep_n]

    scored: list[tuple[float, int]] = []
    for bf, st, leak, j in pool:
        sc = bf * math.log(1.0 + st) / (1.0 + 0.08 * leak)
        if sc > 0:
            scored.append((sc, j))
    scored.sort(reverse=True)
    return [(j, sc) for sc, j in scored[:top_k]]


def ranked_style_phrases(
    state,
    query_ids: list[int],
    *,
    top_k: int = 12,
    gidf_min: float = 4.5,
    max_seeds: int = 2,
) -> list[StylePhrase]:
    """Cụm 2-token: u→v trên W_seq; ưu tiên u có boundary-fitness cao (mở sau dấu)."""
    inv = ranked_style_inventory(
        state, query_ids, top_k=40, gidf_min=gidf_min, max_seeds=max_seeds,
        min_boundary=0.5,
    )
    if len(inv) < 2:
        return []
    score_of = {j: sc for j, sc in inv}
    ids = [j for j, _ in inv]
    id_set = set(ids)
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    phrases: list[StylePhrase] = []
    for u in ids:
        bf_u = boundary_fitness(state, u)
        row = seq.row(u)
        for v, w in row.items():
            if v not in id_set or w <= 0 or u == v:
                continue
            if state.vocab.cap_class(u) >= 1 or state.vocab.cap_class(v) >= 1:
                continue
            # head của cụm phải mở được sau ranh giới
            if bf_u < 1.0:
                continue
            sc = (score_of[u] + 0.5 * score_of[v]) * math.log(1.0 + float(w)) * math.log(1.0 + bf_u)
            phrases.append(StylePhrase(tokens=(u, v), score=sc))
    phrases.sort(key=lambda p: -p.score)
    seen: set[tuple[str, str]] = set()
    out: list[StylePhrase] = []
    for p in phrases:
        key = (
            _core(state.vocab.token_of(p.tokens[0])).lower(),
            _core(state.vocab.token_of(p.tokens[1])).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= top_k:
            break
    return out


def style_inject_plan(
    state,
    query_ids: list[int],
    *,
    prefer_phrase: bool = True,
    max_items: int = 3,
    use_corpus: bool = True,
    corpus_paths: list[Path] | None = None,
    discourse_ngram: bool = True,
) -> list[tuple[int, ...]]:
    """Danh sách đơn vị inject.

    Ưu tiên discourse n-gram thân style (`ngơ ngác`…); rồi corpus uni; fallback graph.
    """
    plan: list[tuple[int, ...]] = []
    if use_corpus and discourse_ngram:
        for unit, _sc in ranked_discourse_ngrams(
            state, query_ids, top_k=max_items, paths=corpus_paths,
        ):
            plan.append(unit)
            if len(plan) >= max_items:
                return plan
        if plan:
            # đủ cụm discourse — không đổ thêm unigram topic (hệt/lướt)
            return plan

    if use_corpus:
        singles = ranked_corpus_inventory(
            state, query_ids, top_k=max_items * 2, paths=corpus_paths,
        )
        phrases = (
            ranked_corpus_phrases(
                state, query_ids, top_k=max_items, paths=corpus_paths, min_seed_df=2,
            )
            if prefer_phrase else []
        )
        if phrases:
            plan.append(phrases[0][0])
        used = {t for ph in plan for t in ph}
        for j, _sc in singles:
            if j in used:
                continue
            plan.append((j,))
            if len(plan) >= max_items:
                return plan
        for pair, _sc in phrases[1:]:
            if any(t in used for t in pair):
                continue
            plan.append(pair)
            used.update(pair)
            if len(plan) >= max_items:
                return plan
        if plan:
            return plan

    phrases = ranked_style_phrases(state, query_ids, top_k=max_items) if prefer_phrase else []
    singles = ranked_style_inventory(state, query_ids, top_k=max_items * 2)
    for p in phrases:
        plan.append(p.tokens)
        if len(plan) >= max_items:
            return plan
    used = {t for ph in plan for t in ph}
    for j, _sc in singles:
        if j in used:
            continue
        plan.append((j,))
        if len(plan) >= max_items:
            break
    return plan
