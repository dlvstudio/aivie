"""Tests cho lớp embedding suy-từ-đồ-thị (core/embedding.py)."""

from __future__ import annotations

import numpy as np

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.embedding import (
    GraphEmbedding,
    build_embedding,
    build_ppmi,
    concept_meanpool,
)
from nn_aivie.core.hebb import update_from_coactivation
from nn_aivie.core.state import GraphState


def _toy_state() -> GraphState:
    """Đồ thị nhỏ: 'chó' và 'mèo' chia sẻ ngữ cảnh 'ăn'; 'xe' tách biệt."""
    st = GraphState()
    v = st.vocab
    sents = [
        ["chó", "sủa"],
        ["chó", "ăn"],
        ["mèo", "kêu"],
        ["mèo", "ăn"],
        ["xe", "chạy"],
        ["xe", "chạy"],
    ]
    for s in sents:
        ids = v.encode_tokens(s)
        update_from_coactivation(st.adjacency, ids, eta=1.0)
    return st


def test_ppmi_nonnegative_and_shaped():
    st = _toy_state()
    ppmi = build_ppmi(st.adjacency, len(st.vocab))  # csr thưa
    assert ppmi.shape == (len(st.vocab), len(st.vocab))
    assert ppmi.data.size == 0 or ppmi.data.min() >= 0.0  # PPMI cắt âm về 0


def test_shared_context_tokens_are_closer():
    """'chó' và 'mèo' (chia sẻ 'ăn') gần nhau hơn 'chó' và 'xe' (không chung)."""
    st = _toy_state()
    emb = build_embedding(st.adjacency, st.vocab, dim=8, meanpool_concepts=False)
    v = st.vocab
    cho, meo, xe = (v._token_to_id[t] for t in ("chó", "mèo", "xe"))
    sim_cho_meo = emb.similarity(cho, meo)
    sim_cho_xe = emb.similarity(cho, xe)
    assert sim_cho_meo > sim_cho_xe


def test_nearest_excludes_self_and_respects_candidates():
    st = _toy_state()
    emb = build_embedding(st.adjacency, st.vocab, dim=8, meanpool_concepts=False)
    v = st.vocab
    cho = v._token_to_id["chó"]
    cands = [v._token_to_id[t] for t in ("mèo", "xe")]
    near = emb.nearest(cho, topk=5, candidate_ids=cands)
    ids = [j for j, _ in near]
    assert cho not in ids
    assert set(ids) <= set(cands)
    # 'mèo' (chung ngữ cảnh) phải xếp trên 'xe'
    assert near[0][0] == v._token_to_id["mèo"]


def test_zero_vector_node_returns_no_neighbors():
    st = _toy_state()
    emb = build_embedding(st.adjacency, st.vocab, dim=8, meanpool_concepts=False)
    isolated = st.vocab.id_of("hoàn_toàn_mới")  # node mới, không cạnh
    assert emb.nearest(isolated, topk=3) == []
    assert emb.similarity(isolated, 0) == 0.0


def test_concept_meanpool_averages_members():
    st = _toy_state()
    v = st.vocab
    cid = v.concept_id("animal")
    cho, meo = v._token_to_id["chó"], v._token_to_id["mèo"]
    # Thành viên surface trỏ vào concept (bottom-up CO_ACTIVATION)
    st.adjacency.set(cho, cid, 1.0, EdgeKind.CO_ACTIVATION)
    st.adjacency.set(meo, cid, 1.0, EdgeKind.CO_ACTIVATION)
    e = build_ppmi(st.adjacency, len(v))
    from nn_aivie.core.embedding import truncated_svd

    base = truncated_svd(e, 8)
    pooled = concept_meanpool(base, st.adjacency, v)
    expected = (base[cho] + base[meo]) / 2.0
    assert np.allclose(pooled[cid], expected)


def test_serialization_roundtrip():
    st = _toy_state()
    emb = build_embedding(st.adjacency, st.vocab, dim=8, meanpool_concepts=False)
    restored = GraphEmbedding.from_payload(emb.to_payload())
    assert restored.matrix.shape == emb.matrix.shape
    assert np.allclose(restored.matrix, emb.matrix, atol=1e-5)


def test_graphstate_persists_embedding(tmp_path):
    st = _toy_state()
    st.embedding = build_embedding(
        st.adjacency, st.vocab, dim=8, meanpool_concepts=False
    )
    p = tmp_path / "state.pkl"
    st.save(p)
    loaded = GraphState.load(p)
    assert loaded.embedding is not None
    assert np.allclose(loaded.embedding.matrix, st.embedding.matrix, atol=1e-5)
