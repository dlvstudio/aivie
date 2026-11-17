"""Tests for embedding export helpers (Graph UI)."""

from __future__ import annotations

import numpy as np
import pytest

from nn_aivie.core.embedding import GraphEmbedding
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.viz.export_embedding import (
    embedding_compare,
    embedding_info,
    embedding_lookup,
    embedding_search,
)


def _mini_state() -> GraphState:
    vocab = Vocabulary()
    for t in ("Hà", "Nội", "thủ", "đô", "Huế"):
        vocab.id_of(t)
    mat = np.random.randn(len(vocab), 8).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    emb = GraphEmbedding(mat)
    state = GraphState(vocab=vocab, embedding=emb)
    return state


def test_embedding_info_available():
    info = embedding_info(_mini_state())
    assert info["available"] is True
    assert info["dim"] == 8
    assert info["vocab_size"] == len(_mini_state().vocab)


def test_embedding_lookup_found():
    state = _mini_state()
    out = embedding_lookup(state, "Hà", topk=3)
    assert out["found"] is True
    assert out["token"] == "Hà"
    assert len(out["neighbors"]) <= 3


def test_embedding_lookup_unknown_no_new_vocab():
    state = _mini_state()
    n_before = len(state.vocab)
    out = embedding_lookup(state, "xyz_missing")
    assert out["found"] is False
    assert len(state.vocab) == n_before


def test_embedding_compare():
    state = _mini_state()
    out = embedding_compare(state, "Hà", "Nội")
    assert out["found"] is True
    assert "similarity" in out


def test_embedding_search_substring():
    state = _mini_state()
    hits = embedding_search(state, "ộ", limit=10)
    assert any(h["token"] == "Nội" for h in hits["tokens"])
