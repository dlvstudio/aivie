"""Tests for G1 passive corpus training."""

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.data.corpus import TextCorpus, load_manifest
from nn_aivie.data.registry import grade_manifest_path
from nn_aivie.data.types import CorpusRecord
from nn_aivie.training.g1_passive import _passive_sentence, run_g1_passive


def test_g1_passive_builds_edges():
    state = GraphState()
    manifest = load_manifest(grade_manifest_path(3))
    corpus = TextCorpus(manifest.for_phase("G1"), grade=3, split="train")
    stats = run_g1_passive(state, corpus, eta=0.05, max_records=5)
    assert stats["records"] == 5
    assert stats["sentence_updates"] == 5
    assert len(state.adjacency) > 0


def test_g1_sentence_bind_links_distant_tokens():
    state = GraphState()
    tokens = [f"t{i}" for i in range(10)]
    text = " ".join(tokens)
    stats = {"records": 0, "tokens": 0, "updates": 0, "sentence_updates": 0}
    _passive_sentence(
        state,
        CorpusRecord(text, grade=3, dataset_id="t", domain="language"),
        0.1,
        stats,
        window=8,
        sentence_bind=True,
    )
    ids = state.vocab.encode_tokens(tokens)
    assert state.adjacency.get(ids[0], ids[-1], EdgeKind.CO_ACTIVATION) > 0
    assert stats["sentence_updates"] == 1


def test_g1_sentence_bind_can_disable():
    state = GraphState()
    tokens = [f"t{i}" for i in range(10)]
    text = " ".join(tokens)
    stats = {"records": 0, "tokens": 0, "updates": 0, "sentence_updates": 0}
    _passive_sentence(
        state,
        CorpusRecord(text, grade=3, dataset_id="t", domain="language"),
        0.1,
        stats,
        window=8,
        sentence_bind=False,
    )
    ids = state.vocab.encode_tokens(tokens)
    assert state.adjacency.get(ids[0], ids[-1], EdgeKind.CO_ACTIVATION) == 0
    assert stats["sentence_updates"] == 0


