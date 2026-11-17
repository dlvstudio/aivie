"""Token hoa/thường là các neuron riêng trong train và inference."""

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.data.types import CorpusRecord
from nn_aivie.teaching.align import tokenize
from nn_aivie.training.g1_passive import _passive_sentence


def test_tokenize_preserves_case():
    assert tokenize("Em em") == ["Em", "em"]


def test_em_and_Em_are_distinct_neurons_after_g1():
    state = GraphState()
    stats = {"records": 0, "tokens": 0, "updates": 0}
    _passive_sentence(
        state,
        CorpusRecord("Em ơi", grade=3, dataset_id="t", domain="language"),
        0.1,
        stats,
    )
    _passive_sentence(
        state,
        CorpusRecord("em đi", grade=3, dataset_id="t", domain="language"),
        0.1,
        stats,
    )

    em_cap = state.vocab.id_of("Em")
    em_low = state.vocab.id_of("em")
    assert em_cap != em_low

    assert state.adjacency.get(em_cap, state.vocab.id_of("ơi"), EdgeKind.CO_ACTIVATION) > 0
    assert state.adjacency.get(em_low, state.vocab.id_of("đi"), EdgeKind.CO_ACTIVATION) > 0
    assert state.adjacency.get(em_cap, em_low, EdgeKind.CO_ACTIVATION) == 0
