"""CSR matvec spreading — thay BFS dict per-hop (TK_036 perf phase 2)."""

from __future__ import annotations

import numpy as np

from nn_aivie.core.adjacency import TypedAdjacency
from nn_aivie.hca.config import HCAConfig


def _frontier_topk(
    frontier: dict[int, float],
    *,
    max_frontier: int,
    threshold: float,
) -> dict[int, float]:
    if not frontier:
        return {}
    items = sorted(frontier.items(), key=lambda kv: -kv[1])[:max_frontier]
    return {u: au for u, au in items if au >= threshold}


def _csr_hop(
    frontier: dict[int, float],
    r_norm,
    vocab_size: int,
    *,
    gamma: float,
    threshold: float,
    max_frontier: int,
) -> dict[int, float]:
    picked = _frontier_topk(
        frontier, max_frontier=max_frontier, threshold=threshold,
    )
    if not picked:
        return {}
    f = np.zeros(vocab_size, dtype=np.float64)
    for u, au in picked.items():
        if 0 <= u < vocab_size:
            f[u] = au
    nxt_vec = gamma * (r_norm.T @ f)
    nz = np.flatnonzero(nxt_vec >= threshold)
    return {int(v): float(nxt_vec[v]) for v in nz}


def spreading_ctx_csr(
    seeds: list[tuple[int, float]],
    adjacency: TypedAdjacency,
    config: HCAConfig,
    vocab_size: int,
) -> dict[int, float]:
    """Lan activation qua readout CSR; trả ctx dict (trước gate w0)."""
    gamma = config.spread_gamma
    hops = max(1, config.spread_hops)
    max_frontier = config.spread_max_frontier
    threshold = config.spread_threshold

    r_norm = adjacency.readout_csr_normalized(vocab_size)
    frontier: dict[int, float] = {}
    for sid, w in seeds:
        if 0 <= sid < vocab_size and w > 0:
            frontier[sid] = frontier.get(sid, 0.0) + w

    ctx: dict[int, float] = {}
    for _ in range(hops):
        if not frontier:
            break
        nxt = _csr_hop(
            frontier, r_norm, vocab_size,
            gamma=gamma, threshold=threshold, max_frontier=max_frontier,
        )
        for v, c in nxt.items():
            ctx[v] = ctx.get(v, 0.0) + c
        frontier = nxt
    return ctx
