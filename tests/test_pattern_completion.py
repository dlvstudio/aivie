"""Tests for R4 pattern-completion routing (TK_017)."""

from __future__ import annotations

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.generate_hebbian import _pattern_completion_tail, generate


def test_open_prompt_pattern_completion_tail():
    st = GraphState()
    ps = ParentStore()
    ids = st.vocab.encode_tokens
    ps.add_example(
        ids("Mùa hè em thích đi biển cùng gia đình .".split()),
        st.vocab, dap_id=None, priority=1.0,
    )
    st.parents = ps
    tail = _pattern_completion_tail(
        st, ids("Mùa hè em".split()), HCAConfig(), min_suffix_tokens=3,
    )
    assert tail is not None
    assert decode_surface(st, tail).startswith("thích")


def test_generate_open_prompt_uses_pattern_not_wseq_salad():
    st = GraphState()
    ps = ParentStore()
    ids = st.vocab.encode_tokens
    ps.add_example(
        ids("Mùa hè em thích đi biển cùng gia đình .".split()),
        st.vocab, dap_id=None, priority=1.0,
    )
    st.parents = ps
    out = decode_surface(st, generate(st, "Mùa hè em", max_len=24))
    assert "thích" in out
    assert "và các thành phố" not in out


def test_longest_suffix_min_k_blocks_single_token_false_match():
    from nn_aivie.hca.generate_hebbian import _longest_suffix_cut

    st = GraphState()
    ids = st.vocab.encode_tokens
    for t in "Gia đình em rất vui Mùa hè".split():
        st.vocab.id_of(t)
    fold = st.vocab.fold_group_id
    src = ids("Gia đình em rất vui .".split())
    q = ids("Mùa hè em".split())
    q_folds = [fold(t) for t in q]
    assert _longest_suffix_cut(src, q_folds, fold, min_k=3) == (-1, 0)
    assert _longest_suffix_cut(src, q_folds, fold, min_k=1)[0] >= 0
