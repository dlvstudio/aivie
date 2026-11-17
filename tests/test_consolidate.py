"""Tests for #3 emergent consolidation (Hebb tạo cạnh mới)."""

from nn_aivie.core.edge_kinds import EdgeKind, READOUT_EM_SCALE
from nn_aivie.core.state import GraphState
from nn_aivie.training.consolidate import consolidate_emergent


def _add_rf(state, src, tgt, w):
    state.adjacency.add(
        state.vocab._ensure(src),
        state.vocab._ensure(tgt),
        w,
        EdgeKind.REINFORCED,
    )


def test_consolidate_links_siblings_and_inherits_targets():
    """Hai nguồn đồng vai (chia sẻ đích) → kế thừa đích của nhau qua EMERGENT."""
    state = GraphState()
    # 'meo' và 'cho' cùng dẫn tới {chay, ngu, thu} → đồng vai.
    for tgt in ("chay", "ngu", "thu"):
        _add_rf(state, "meo", tgt, 0.5)
        _add_rf(state, "cho", tgt, 0.5)
    # mỗi node có một đích riêng để kiểm tra kế thừa.
    _add_rf(state, "meo", "cung", 0.6)  # chỉ mèo có
    _add_rf(state, "cho", "sua", 0.6)  # chỉ chó có

    stats = consolidate_emergent(state, min_shared=2, sim_min=0.2)
    assert stats["sibling_pairs"] >= 1
    assert stats["emergent_edges"] >= 1

    cho = state.vocab.id_of("cho")
    cung = state.vocab.id_of("cung")
    # chó kế thừa 'cung' (đích riêng của mèo) qua cạnh emergent.
    em = state.adjacency.get(cho, cung, EdgeKind.EMERGENT)
    assert em > 0.0
    # và emergent tham gia readout ở scale nhỏ.
    assert state.adjacency.get_readout(cho, cung) >= em * READOUT_EM_SCALE


def test_consolidate_skips_non_surface_sources():
    """Node cấu trúc/đặc biệt không sinh cạnh emergent làm nguồn."""
    state = GraphState()
    for tgt in ("a", "b", "c"):
        _add_rf(state, "Var:S", tgt, 0.5)
        _add_rf(state, "x", tgt, 0.5)
    consolidate_emergent(state, min_shared=2, sim_min=0.2)
    var_s = state.vocab.id_of("Var:S")
    out = state.adjacency.layer(EdgeKind.EMERGENT).row(var_s)
    assert out == {}


def test_consolidate_respects_similarity_threshold():
    """Tương đồng thấp (ít đích chung) → không tạo cặp anh em."""
    state = GraphState()
    _add_rf(state, "p", "t1", 0.5)
    _add_rf(state, "p", "t2", 0.5)
    _add_rf(state, "q", "t3", 0.5)
    _add_rf(state, "q", "t4", 0.5)
    stats = consolidate_emergent(state, min_shared=2, sim_min=0.5)
    assert stats["sibling_pairs"] == 0
    assert stats["emergent_edges"] == 0
