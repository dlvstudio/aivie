"""B1 — batch G1 coactivation tương đương immediate path."""

from __future__ import annotations

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.training.g1_batch import (
    begin_g1_batch,
    co_window_starts,
    flush_g1_batch,
    g1_and_wseq,
)


def _adj_snapshot(state: GraphState) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for kind in (EdgeKind.CO_ACTIVATION, EdgeKind.SEQUENTIAL):
        for i, j, w in state.adjacency.layer(kind).items():
            out[(kind, i, j)] = w
    return out


def test_g1_batch_matches_immediate():
    texts = [
        "Hà Nội là thủ đô Việt Nam",
        "Thủ đô Hà Nội đặt ở đâu",
        "Con mèo ở đâu",
    ]
    eta = 0.05

    st1 = GraphState()
    for t in texts:
        ids = st1.vocab.encode_tokens(t.split())
        g1_and_wseq(st1, ids, eta)
    ref = _adj_snapshot(st1)

    st2 = GraphState()
    begin_g1_batch(st2)
    for t in texts:
        ids = st2.vocab.encode_tokens(t.split())
        g1_and_wseq(st2, ids, eta)
    flush_g1_batch(st2)
    got = _adj_snapshot(st2)

    assert got == ref


def test_co_window_starts_short_text_uses_all():
    assert co_window_starts(20, 12) == list(range(20))


def test_co_window_starts_long_text_capped():
    starts = co_window_starts(10_000, 12)
    assert len(starts) <= 600
    assert len(starts) < 10_000
    assert starts[0] == 0
