"""Tests nạp dữ liệu thật (training/ingest.py) → nơ-ron cha + pattern completion."""

from __future__ import annotations

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.training.ingest import ingest_qa, ingest_statements


def _fresh() -> GraphState:
    st = GraphState()
    st.parents = ParentStore()
    return st


def test_ingest_qa_becomes_answerable_parent():
    st = _fresh()
    n0 = len(st.parents)
    ingest_qa(st, [("Hành tinh đỏ là gì", "Sao Hỏa là hành tinh đỏ.")])
    assert len(st.parents) == n0 + 1
    ids = generate(st, "Hỏi: Hành tinh đỏ là gì Đáp:", max_len=12)
    ans = decode_surface(st, ids).split("Đáp:")[-1].strip()
    assert "Sao" in ans and "Hỏa" in ans


def test_ingest_statement_enables_pattern_completion():
    """Câu phát biểu (không phải QA) → vẫn hoàn thành được câu hỏi (case con kiến)."""
    st = _fresh()
    ingest_statements(st, ["Cá voi là động vật có vú lớn nhất."])
    ids = generate(st, "Hỏi: Cá voi là gì Đáp:", max_len=12)
    ans = decode_surface(st, ids).split("Đáp:")[-1].strip()
    assert "Cá" in ans and "voi" in ans


def test_ingest_adds_wseq_edges():
    from nn_aivie.core.edge_kinds import EdgeKind

    st = _fresh()
    ingest_statements(st, ["Mặt trời mọc ở hướng đông."])
    assert len(st.adjacency.layer(EdgeKind.SEQUENTIAL)) > 0
