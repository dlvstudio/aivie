"""
a_style readout — kênh Hadamard thứ 5 từ CO_ACT (skeleton, 2026-07-23).

Nhãn HOW trong prompt (viết thường, gidf đủ) → neighbors CO_ACT xếp W×IDF
(C×gidf, sàn C) → boost vector. Không hardcode marker (Luật 0).

Caller: `generate_autoregressive` nhân `multiply_scores_by_boost` khi
`ar_style_weight > 0`. Default tắt.
"""

from __future__ import annotations

import string

import numpy as np

from nn_aivie.core.edge_kinds import EdgeKind

_PUNCT = set(string.punctuation + "…“”‘’—–·•")


def _core(tok: str) -> str:
    if tok.startswith("Ġ") or tok.startswith("Ċ"):
        tok = tok[1:]
    return tok


def _is_artifact(tok: str) -> bool:
    c = _core(tok)
    if not c or c.endswith(":"):
        return True
    return all(ch in _PUNCT or ch.isspace() for ch in c)


def style_seeds_from_prompt(
    query_ids: list[int],
    vocab,
    ps,
    *,
    gidf_min: float = 4.5,
    max_seeds: int = 2,
) -> list[int]:
    """Token HOW/style: surface, không viết-hoa (cap_class=0), gidf≥min.

    Entity viết-hoa để cho multizone/FACT; phần còn lại có thể mang nhãn
    «hài hước»/«buồn»… — tách head STYLE khỏi head FACT (RUNBOOK_STYLE).

    Siết (probe 2026-07-23): chỉ giữ top-`max_seeds` theo gidf — tránh
    «đoạn/văn/trận» át nhãn «hài hước» khi gộp mọi token thường.
    """
    cands: list[tuple[float, int]] = []
    seen: set[int] = set()
    for t in query_ids:
        if t in seen:
            continue
        if not vocab.is_surface(t):
            continue
        if vocab.cap_class(t) >= 1:
            continue
        g = ps._gidf(t)  # noqa: SLF001
        if g < gidf_min:
            continue
        tok = vocab.token_of(t)
        if _is_artifact(tok):
            continue
        seen.add(t)
        cands.append((g, t))
    cands.sort(reverse=True)
    return [t for _, t in cands[:max_seeds]]


def style_boost_from_coact(
    state,
    query_ids: list[int],
    vocab_size: int,
    *,
    gidf_min: float = 4.5,
    c_min: float = 2.0,
    top_k: int = 48,
    max_seeds: int = 2,
) -> tuple[np.ndarray, list[int]]:
    """Trả (boost[V], seed_ids). boost≥0; rỗng nếu không có seed/neighbor."""
    boost = np.zeros(vocab_size, dtype=np.float64)
    ps, vocab = state.parents, state.vocab
    if ps is None:
        return boost, []
    seeds = style_seeds_from_prompt(
        query_ids, vocab, ps, gidf_min=gidf_min, max_seeds=max_seeds,
    )
    if not seeds:
        return boost, []

    co = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
    seed_set = set(seeds)
    seed_cores = {_core(vocab.token_of(t)).lower() for t in seeds}
    agg: dict[int, float] = {}
    for s in seeds:
        for j, w in co.row(s).items():
            if j in seed_set or w <= 0 or j >= vocab_size:
                continue
            agg[j] = agg.get(j, 0.0) + float(w)

    scored: list[tuple[float, int]] = []
    for j, c in agg.items():
        if c < c_min:
            continue
        tok = vocab.token_of(j)
        if _is_artifact(tok):
            continue
        if tok in ("<EOS>", "<BOS>", "<PAD>") or _core(tok).startswith("<"):
            continue
        if _core(tok).lower() in seed_cores:
            continue
        g = ps._gidf(j)  # noqa: SLF001
        if g < gidf_min:
            continue
        scored.append((c * g, j))
    scored.sort(reverse=True)
    for sc, j in scored[:top_k]:
        boost[j] = sc
    return boost, seeds
