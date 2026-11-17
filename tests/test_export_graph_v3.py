"""Tests cho export_graph_v3 — cấu trúc cạnh phân tầng token → cụm → cha."""

from __future__ import annotations

from pathlib import Path

import pytest

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.viz.export_graph_v3 import completion_graph


def _edge_set(graph: dict) -> set[tuple[str, str, str]]:
    return {(e["source"], e["target"], e["kind"]) for e in graph["edges"]}


def _build_cong_dan_state() -> GraphState:
    st = GraphState()
    st.parents = ParentStore()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = st.parents
    examples = [
        "Hỏi: Công dân được bầu cử từ bao nhiêu tuổi Đáp: Công dân có quyền bầu cử từ 18 tuổi .",
        "Hỏi: Công dân được bầu cử tại địa phương Đáp: Công dân được bầu cử trực tiếp .",
    ]
    for text in examples:
        ids = st.vocab.encode_tokens(text.split())
        ps.add_example(ids, st.vocab, dap_id=dap)
    ps.decompose(st.vocab, min_frequency=2, max_len=5)
    return st


def test_completion_graph_includes_fact_stimulus():
    """TK_023: API completion_graph trả fact_stimulus (score + picked)."""
    st = _build_cong_dan_state()
    prompt = "Hỏi: Công dân được bầu cử từ bao nhiêu tuổi ? Đáp:"
    g = completion_graph(st, prompt)
    assert "fact_stimulus" in g
    fs = g["fact_stimulus"]
    if fs is not None:
        assert "eligible" in fs and "entity" in fs and "facts" in fs


def test_completion_graph_hierarchical_edges_synthetic():
    """Token thuộc cụm: q→phrase→parent; không q→parent trực tiếp (cha thắng)."""
    st = _build_cong_dan_state()
    prompt = "Hỏi: Công dân được bầu cử từ bao nhiêu tuổi ? Đáp:"
    g = completion_graph(st, prompt)
    assert g["matched"] is not None
    matched = g["matched"]
    edges = _edge_set(g)

    phrase_nodes = [n for n in g["nodes"] if n.get("kind") == "phrase"]
    assert phrase_nodes, "expected intermediate phrase nodes"

    vocab = st.vocab
    for word in ("được", "bầu", "cử", "Công", "dân"):
        qid = vocab.id_of(word)
        ph_targets = {e[1] for e in edges if e[0] == f"q:{qid}" and e[2] == "compose"}
        assert ph_targets, f"{word} should connect to phrase"
        direct = [
            e for e in edges
            if e[0] == f"q:{qid}" and e[2] in ("activate", "activate_alt")
        ]
        assert not direct, f"{word} must not connect directly to any parent"

    for ph in phrase_nodes:
        assert (ph["id"], f"p:{matched}", "activate") in edges

    bare_word = "tuổi"
    if vocab.id_of(bare_word) in {n["label"] for n in g["nodes"] if n.get("kind") == "query"}:
        qid = vocab.id_of(bare_word)
        assert (f"q:{qid}", f"p:{matched}", "activate") in edges


def test_completion_graph_shows_hub_for_family_query():
    """Câu hỏi thuộc họ mẫu → đồ thị có node hub + cạnh hub_frame/hub_slot + hub info."""
    st = GraphState()
    st.parents = ParentStore()
    dap = st.vocab.id_of(DAP_LABEL)
    for text in [
        "Hỏi: Thủ đô của Mỹ là gì Đáp: Thủ đô của Mỹ là Washington .",
        "Hỏi: Thủ đô của Pháp là gì Đáp: Thủ đô của Pháp là Paris .",
        "Hỏi: Thủ đô của Anh là gì Đáp: Thủ đô của Anh là London .",
        "Thủ đô của Ý là Rome .",
    ]:
        st.parents.add_example(st.vocab.encode_tokens(text.split()), st.vocab, dap_id=dap)
    st.parents.decompose(st.vocab, min_frequency=2, max_len=4)
    st.parents.induce_slot_hubs(min_distinct_fillers=2)
    st.parents.bind_slot_fillers()

    # filler đã nối lúc train (Ý)
    g = completion_graph(st, "Hỏi: Thủ đô của Ý là gì Đáp:")
    assert g["hub"] is not None
    assert g["hub"]["slot"] == "Ý" and g["hub"]["slot_known"] is True
    hub_nodes = [n for n in g["nodes"] if n.get("kind") == "hub"]
    assert len(hub_nodes) == 1
    kinds = {e["kind"] for e in g["edges"]}
    assert "hub_frame" in kinds and "hub_slot" in kinds

    # filler lạ (Đức) — vẫn nhận hub nhưng slot_known=False
    g2 = completion_graph(st, "Hỏi: Thủ đô của Đức là gì Đáp:")
    assert g2["hub"] is not None
    assert g2["hub"]["slot"] == "Đức" and g2["hub"]["slot_known"] is False

    # câu ngoài họ mẫu → không hub
    g3 = completion_graph(st, "Hỏi: Ai chữa bệnh Đáp:")
    assert g3["hub"] is None


def test_completion_graph_isg_cluster_edges_per_parent():
    """ISG: token thuộc cụm → cl→cha; không q→cha trực tiếp (kể cả candidate alt)."""
    st = GraphState()
    ps = ParentStore()
    ps.seg_min_frequency = 3
    dap = st.vocab.id_of(DAP_LABEL)
    for text in [
        "Hỏi: Hà Nội ở đâu Đáp: Hà Nội ở Việt Nam .",
        "Hỏi: Thủ đô Việt Nam ở đâu Đáp: Thủ đô Hà Nội ở Việt Nam .",
        "Hỏi: Con mèo ở đâu Đáp: Con mèo ở nhà .",
    ]:
        ps.add_example(st.vocab.encode_tokens(text.split()), st.vocab, dap_id=dap)
    st.parents = ps

    g = completion_graph(st, "Hỏi: Thủ đô Việt Nam ở đâu ? Đáp:", top_candidates=8)
    edges = _edge_set(g)
    vocab = st.vocab
    alt = next(c for c in g["candidates"] if c["question"] == "Thủ đô Việt Nam ở đâu")
    pidx = alt["idx"]
    ci = next(
        i
        for i, c in enumerate(ps.seg.clusters)
        if vocab.token_of(c.left) == "ở" and vocab.token_of(c.right) == "đâu"
    )
    assert (f"cl:{ci}", f"p:{pidx}", "seg_cluster") in edges
    for word in ("ở", "đâu"):
        qid = vocab.id_of(word)
        direct = [
            e for e in edges
            if e[0] == f"q:{qid}" and e[1] == f"p:{pidx}"
            and e[2] in ("activate", "activate_alt")
        ]
        assert not direct, f"{word} must route through cluster, not parent directly"
    assert any(
        e["kind"] == "seg_h" and e.get("parent") == pidx for e in g["edges"]
    ), "expected horizontal ISG links on alt parent"
    # index + trọng số trên cạnh cụm (có trong mạng, phải export)
    cl_edges = [
        e for e in g["edges"]
        if e.get("target") == f"p:{pidx}" and e["source"] == f"cl:{ci}" and e["kind"] == "seg_cluster"
    ]
    assert cl_edges and cl_edges[0].get("slot_index") is not None
    q_comp = [e for e in g["edges"] if e["target"] == f"cl:{ci}" and e["kind"] == "seg_compose"]
    assert len(q_comp) == 2
    assert {e["slot_index"] for e in q_comp} == {1, 2}


def test_completion_graph_autoregressive_mode():
    """A3 mode trả trace từng bước + node ar_token trên đồ thị."""
    st = _build_cong_dan_state()
    prompt = "Hỏi: Công dân được bầu cử từ bao nhiêu tuổi ? Đáp:"
    g = completion_graph(st, prompt, mode="autoregressive", max_len=24)
    assert g["generate_mode"] == "autoregressive"
    assert g["autoregressive"] is not None
    assert len(g["autoregressive"]["agenda"]) > 0
    assert len(g["autoregressive"]["steps"]) > 0
    ar_nodes = [n for n in g["nodes"] if n.get("kind") == "ar_token"]
    assert ar_nodes
    assert any(e["kind"] == "ar_step" for e in g["edges"])
    assert "Đáp:" in g["generated"] or g["autoregressive"]["answer"]


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("checkpoints/grade_v3_big_a2_t3_isg.pkl").exists(),
    reason="ISG checkpoint not available",
)
def test_standalone_token_routes_via_seg_h_not_direct_parent():
    """Token rời trên slot_chain (vd. ở giữa Hà Nội–Việt Nam) → seg_h, không activate thẳng."""
    st = GraphState.load("checkpoints/grade_v3_big_a2_t3_isg.pkl")
    prompt = "Hỏi: Thủ đô Hà Nội ở đâu ? Đáp:"
    g = completion_graph(st, prompt, top_candidates=8)
    o_id = st.vocab.id_of("ở")
    direct = [
        e for e in g["edges"]
        if e["source"] == f"q:{o_id}" and e["target"] == "p:57" and e["kind"] == "activate"
    ]
    assert not direct, "ở must not use IDF activate shortcut to parent p57"
    slot_e = [
        e for e in g["edges"]
        if e["source"] == f"q:{o_id}" and e["target"] == "p:57" and e["kind"] == "seg_slot"
    ]
    assert slot_e and slot_e[0].get("slot_index") == 3
    seg_h = [
        e for e in g["edges"]
        if e["source"] == f"q:{o_id}" and e["kind"] == "seg_h"
    ]
    assert seg_h and seg_h[0].get("slot_index") == 3


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("checkpoints/grade_v3_big.pkl").exists(),
    reason="grade_v3_big checkpoint not available",
)
def test_completion_graph_hierarchical_edges_big_checkpoint():
    st = GraphState.load("checkpoints/grade_v3_big.pkl")
    prompt = "Hỏi: Công dân được bầu cử từ bao nhiêu tuổi ? Đáp:"
    g = completion_graph(st, prompt)
    assert g["matched"] is not None
    matched = g["matched"]
    edges = _edge_set(g)
    vocab = st.vocab
    for word in ("Công", "dân", "bầu", "cử"):
        qid = vocab.id_of(word)
        assert any(
            e[0] == f"q:{qid}" and e[2] == "compose" for e in edges
        ), f"{word} should connect to phrase cluster"
        assert (f"q:{qid}", f"p:{matched}", "activate") not in edges
