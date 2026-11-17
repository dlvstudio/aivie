"""Tests for TK_017 position-targeted FRAME_BRACKET gate."""

from __future__ import annotations

import numpy as np

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.frame_bracket_gate import apply_frame_bracket_gate, qualifying_opener, update_open_brackets


def _toy_state_with_bracket() -> GraphState:
    vocab = Vocabulary()
    neu = vocab.id_of
    for t in ("nếu", "thì", "mưa", "trời"):
        neu(t)
    state = GraphState(vocab=vocab)
    state.parents = __import__("nn_aivie.core.parent_store", fromlist=["ParentStore"]).ParentStore()
    a, b = neu("nếu"), neu("thì")
    state.adjacency.add(a, b, 2.5, EdgeKind.FRAME_BRACKET)
    return state


def test_qualifying_opener_requires_gap():
    st = _toy_state_with_bracket()
    neu = st.vocab.id_of
    n, t, m = neu("nếu"), neu("thì"), neu("mưa")
    assert qualifying_opener(st, [n], min_gap=2) is None
    assert qualifying_opener(st, [n, m, t], min_gap=2) == n


def test_apply_gate_boosts_closer():
    st = _toy_state_with_bracket()
    neu = st.vocab.id_of
    n, t, m = neu("nếu"), neu("thì"), neu("mưa")
    vs = len(st.vocab)
    cfg = HCAConfig(ar_frame_bracket_weight=0.5, ar_frame_bracket_floor=0.2)
    scores = np.zeros(vs, dtype=np.float64)
    scores[n] = 1.0
    scores[t] = 0.5
    scores[m] = 0.3
    open_br = [(n, 0)]
    fired = apply_frame_bracket_gate(scores, st, [n, m, t], open_br, cfg, vs)
    assert fired
    assert scores[t] > scores[m]


def test_update_open_brackets_closes_on_closer():
    st = _toy_state_with_bracket()
    neu = st.vocab.id_of
    n, t = neu("nếu"), neu("thì")
    open_br: list[tuple[int, int]] = [(n, 0)]
    update_open_brackets(open_br, t, 3, st)
    assert open_br == []


# --- R1 (TK_017 §11 F5): focus-gate trên plan fallback ---

def test_filter_plan_on_focus_drops_offtopic_segment():
    """Fact không chứa focus phân biệt nhất của query → segment bị loại → agenda rỗng."""
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.hca.generate_hebbian import _SegmentPlan, _filter_plan_on_focus

    vocab = Vocabulary()
    ids = vocab.encode_tokens
    state = GraphState(vocab=vocab)
    state.parents = ParentStore()
    # corpus: fact về "Mặt trời" (không nói gì về mưa) + 1 fact khác có "mưa" (để "mưa"
    # có gdf>0 — checkpoint thật luôn vậy; token gdf=0 bị W3 loại khỏi focus)
    fact = ids("Mặt trời cung cấp ánh sáng".split())
    state.parents.add_example(fact, vocab, dap_id=None)
    state.parents.add_example(ids("Mùa mưa kéo dài".split()), vocab, dap_id=None)
    # "trời" phổ biến hơn "mưa" (như corpus thật) → primary focus = "mưa"
    state.parents.add_example(ids("Bầu trời trong xanh".split()), vocab, dap_id=None)
    query = ids("nếu trời mưa".split())
    # plan fallback kéo fact lạc (chứa "trời" nhưng KHÔNG chứa "mưa" = focus phân biệt nhất)
    plan = _SegmentPlan([list(fact)], (), list(fact))
    out = _filter_plan_on_focus(state, list(query), plan)
    assert out.segments == [] and out.flat_agenda == []


def test_filter_plan_on_focus_keeps_covering_segment():
    """Case con-kiến: statement CHỨA thực thể focus → giữ nguyên plan."""
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.hca.generate_hebbian import _SegmentPlan, _filter_plan_on_focus

    vocab = Vocabulary()
    ids = vocab.encode_tokens
    state = GraphState(vocab=vocab)
    state.parents = ParentStore()
    fact = ids("Cá voi là động vật có vú".split())
    state.parents.add_example(fact, vocab, dap_id=None)
    query = ids("Cá voi là gì".split())
    plan = _SegmentPlan([list(fact)], (), list(fact))
    out = _filter_plan_on_focus(state, list(query), plan)
    assert out.segments == [list(fact)]


# --- TK_020: cạnh ngang = (PMI, POWER, GAP_HIST, PROVENANCE) + ctx-resonance ---

def _tk020_state():
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.training.frame_bracket import mine_frame_brackets

    vocab = Vocabulary()
    ids = vocab.encode_tokens
    st = GraphState(vocab=vocab)
    st.parents = ParentStore()
    st.parents.seg._defer_promotions = True
    for k in range(6):
        st.parents.add_example(ids(f"nếu trời mưa thì ta ở nhà s{k}".split()), vocab, dap_id=None)
    mine_frame_brackets(st, min_df=2, max_df_frac=1.0, min_count=2, min_pmi=0.0, dir_ratio=1.0)
    return st, ids


def test_mine_keeps_power_gap_provenance():
    st, ids = _tk020_state()
    n, t = st.vocab.id_of("nếu"), st.vocab.id_of("thì")
    meta = st.parents.hlink_meta.get((n, t))
    assert meta is not None
    assert meta["power"] == 6                 # power = số parent có cặp
    assert meta["gaps"] == {3: 6}             # khoảng cách là tham số cạnh
    assert len(meta["prov"]) == 6             # provenance: cha nguồn
    assert all(len(p) == 3 for p in meta["prov"])  # (parent_idx, pos, gap)


def test_mine_counts_pair_once_per_parent():
    """Nhiều opener trước một closer trong cùng parent → power += 1, không nhân theo opener."""
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.training.frame_bracket import mine_frame_brackets

    vocab = Vocabulary()
    ids = vocab.encode_tokens
    st = GraphState(vocab=vocab)
    st.parents = ParentStore()
    st.parents.seg._defer_promotions = True
    for k in range(3):
        st.parents.add_example(
            ids(f"nếu mưa . nếu gió . nếu nắng thì ở nhà s{k}".split()),
            vocab,
            dap_id=None,
        )
    mine_frame_brackets(st, min_df=2, max_df_frac=1.0, min_count=2, min_pmi=0.0, dir_ratio=1.0)
    n, t = st.vocab.id_of("nếu"), st.vocab.id_of("thì")
    meta = st.parents.hlink_meta[(n, t)]
    assert meta["power"] == 3
    assert sum(meta["gaps"].values()) == 3
    assert len(meta["prov"]) == 3


def test_gap_prior_and_ctx_resonance_boost():
    import numpy as np
    st, ids = _tk020_state()
    n, t = st.vocab.id_of("nếu"), st.vocab.id_of("thì")
    V = len(st.vocab)
    out = list(ids("nếu trời mưa".split()))  # opener tại 0 → d = 3 (đúng mode gap)
    ob = [(n, 0)]
    base = np.ones(V)
    cfg = HCAConfig(ar_gap_prior=True, ar_hlink_ctx_weight=1.0,
                    ar_frame_bracket_weight=0.5, ar_frame_bracket_floor=0.2)
    s1 = base.copy()
    fired = apply_frame_bracket_gate(s1, st, out, ob, cfg, V, active_parents=frozenset())
    assert fired and s1[t] > s1[st.vocab.id_of("trời")]  # closer được nâng tại đúng gap
    # ctx-resonance: cha provenance ĐANG kích hoạt → mạnh hơn nữa (chuẩn hoá theo bmax nên
    # so trên tỉ lệ với token thường)
    s2 = base.copy()
    apply_frame_bracket_gate(s2, st, out, ob, cfg, V, active_parents=frozenset({0}))
    assert s2[t] >= s1[t]


def test_hlink_meta_roundtrip():
    st, _ = _tk020_state()
    from nn_aivie.core.parent_store import ParentStore
    payload = st.parents.to_payload()
    ps2 = ParentStore.from_payload(payload)
    assert ps2.hlink_meta == st.parents.hlink_meta
    assert getattr(ps2, "hlink_endpoints", {}) == getattr(st.parents, "hlink_endpoints", {})


def _cluster_frame_state():
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.training.frame_bracket import mine_frame_brackets

    vocab = Vocabulary()
    ids = vocab.encode_tokens
    st = GraphState(vocab=vocab)
    st.parents = ParentStore()
    for k in range(8):
        st.parents.add_example(
            ids(f"Mặc dù trời mưa vẫn đi học s{k}".split()),
            vocab,
            dap_id=None,
        )
    st.parents.seg.finalize_promotions(st.parents.parents)
    mine_frame_brackets(
        st, min_df=2, max_df_frac=1.0, min_count=2, min_pmi=0.0,
        dir_ratio=1.0, min_gap=2, max_gap=10, min_cluster_power=2,
    )
    return st, vocab
    st, vocab = _cluster_frame_state()
    pair = st.parents.seg.cluster_of_pair.get(
        (vocab.id_of("Mặc"), vocab.id_of("dù"))
    )
    assert pair is not None
    cid = st.parents.seg.cluster_id(pair)
    still = vocab.id_of("vẫn")
    row = st.adjacency.layer(EdgeKind.FRAME_BRACKET).row(cid)
    assert still in row or len(row) > 0
    meta = st.parents.hlink_meta
    assert any(k[0] == cid for k in meta)
    assert getattr(st.parents, "hlink_endpoints", None)


def test_cluster_opener_gate_boosts_closer():
    st, vocab = _cluster_frame_state()
    mac = vocab.id_of("Mặc")
    du = vocab.id_of("dù")
    van = vocab.id_of("vẫn")
    pair = st.parents.seg.cluster_of_pair[(mac, du)]
    cid = st.parents.seg.cluster_id(pair)
    # ensure edge cid -> van
    if van not in st.adjacency.layer(EdgeKind.FRAME_BRACKET).row(cid):
        st.adjacency.set(cid, van, 2.5, EdgeKind.FRAME_BRACKET)
        st.parents.hlink_meta[(cid, van)] = {
            "power": 8, "gaps": {4: 8}, "prov": [(0, 0, 4)],
            "left_kind": "cluster", "right_kind": "token",
        }
    vs = len(vocab)
    cfg = HCAConfig(
        ar_frame_bracket_weight=1.0, ar_frame_bracket_floor=0.0, ar_gap_prior=False,
    )
    out = list(vocab.encode_tokens("Mặc dù trời mưa".split()))
    open_br = [(cid, 1)]
    scores = np.zeros(vs, dtype=np.float64)
    scores[van] = 0.2
    scores[vocab.id_of("mưa")] = 1.0
    fired = apply_frame_bracket_gate(scores, st, out, open_br, cfg, vs)
    assert fired
    assert scores[van] > scores[vocab.id_of("mưa")]


def test_gap_override_fires_only_with_ctx_and_peak():
    st = _toy_state_with_bracket()
    neu = st.vocab.id_of("nếu")
    thi = st.vocab.id_of("thì")
    st.parents.hlink_meta = {
        (neu, thi): {
            "power": 6,
            "gaps": {3: 6},
            "prov": [(0, 0, 3)],
            "left_kind": "token",
            "right_kind": "token",
        },
    }
    V = len(st.vocab)
    out = list(st.vocab.encode_tokens("nếu trời mưa".split()))
    ob = [(neu, 0)]
    cfg = HCAConfig()
    base = np.ones(V)
    base[st.vocab.id_of("trời")] = 100.0
    s1 = base.copy()
    tr1 = {}
    apply_frame_bracket_gate(s1, st, out, ob, cfg, V, active_parents=frozenset(), gate_trace=tr1)
    assert "override" not in tr1
    s2 = base.copy()
    tr2 = {}
    apply_frame_bracket_gate(s2, st, out, ob, cfg, V, active_parents=frozenset({0}), gate_trace=tr2)
    assert tr2.get("override") == thi and s2[thi] > s2[st.vocab.id_of("trời")]


def test_ar_trace_cluster_opener_no_crash():
    """TK_037: trace phải decode cluster/phrase opener — không gọi token_of trên adj id."""
    from pathlib import Path

    import pytest

    from nn_aivie.core.checkpoint_io import load_checkpoint
    from nn_aivie.hca.generate_hebbian import generate

    ckpt = Path(__file__).resolve().parents[1] / "checkpoints/grade_lessons_test.pkl"
    if not ckpt.exists():
        pytest.skip("grade_lessons_test.pkl not present")
    state, _ = load_checkpoint(ckpt)
    trace: list[dict] = []
    generate(
        state,
        "Hỏi: Mẹ làm gì ? Đáp:",
        max_len=16,
        config=HCAConfig(ar_frame_bracket_weight=0.5),
        multi=True,
        trace_out=trace,
    )
    assert trace
    for step in trace:
        gate = step.get("gate")
        if not gate or gate.get("opener") is None:
            continue
        assert isinstance(gate["opener"], str)


# --- TK_037: bake phải khớp điều kiện của runtime `ensure_frame_brackets` ---


def test_bake_remines_when_endpoints_missing():
    """Checkpoint bake TRƯỚC TK_037 (layer + meta, endpoints rỗng) → bake phải mine lại.

    Regression: bake chỉ hỏi `has_meta` còn `ensure_frame_brackets` hỏi thêm `has_eps`.
    Lệch điều kiện → checkpoint kẹt: bake báo "skipped", runtime bị guard checkpoint-lớn
    (`_runtime_mine_allowed`, >5000 cha) chặn → `hlink_endpoints` mãi rỗng → phrase opener
    chết âm thầm. Đo 2026-07-14: network_v10_histwiki{,_paragraph}.pkl đúng trạng thái này.
    """
    from nn_aivie.training.checkpoint_bake import bake_derived_indices

    st, _ids = _tk020_state()
    ps = st.parents
    n_edges = len(st.adjacency.layer(EdgeKind.FRAME_BRACKET))
    assert n_edges > 0 and ps.hlink_meta and ps.hlink_endpoints

    ps.hlink_endpoints = {}  # giả lập checkpoint pre-TK_037
    stats = bake_derived_indices(st, skip_spans=False, seq_index=False, frame_brackets=True)

    assert stats["frame_brackets"].get("skipped") is not True, "bake không được bỏ qua"
    assert ps.hlink_endpoints, "bake phải khôi phục hlink_endpoints"
    # re-mine reset layer trước khi mine → không trộn cạnh của hai lần mine
    assert len(st.adjacency.layer(EdgeKind.FRAME_BRACKET)) == n_edges


def test_bake_skips_when_complete_and_force_overrides():
    from nn_aivie.training.checkpoint_bake import bake_derived_indices

    st, _ids = _tk020_state()
    n_edges = len(st.adjacency.layer(EdgeKind.FRAME_BRACKET))

    stats = bake_derived_indices(st, skip_spans=False, seq_index=False, frame_brackets=True)
    assert stats["frame_brackets"] == {"skipped": True, "edges": n_edges}

    forced = bake_derived_indices(
        st, skip_spans=False, seq_index=False, frame_brackets=True, force_frame_brackets=True,
    )
    assert forced["frame_brackets"].get("skipped") is not True
    assert len(st.adjacency.layer(EdgeKind.FRAME_BRACKET)) == n_edges


def test_guard_warns_when_mine_skipped_without_logger(monkeypatch, recwarn):
    """Guard TẮT gate → phải có tín hiệu dù caller KHÔNG truyền logger.

    Regression: bản đầu chỉ `if logger is not None: logger.log(...)`, mà cả 6 call site
    (generate_hebbian ×5, viz/server/main.py) đều gọi `ensure_frame_brackets(state)` không
    logger → cảnh báo là code chết, gate tắt lặng lẽ trên mọi checkpoint >cap.
    """
    import warnings

    from nn_aivie.core.adjacency import SparseAdjacency
    from nn_aivie.hca import frame_bracket_gate as fbg

    st, _ids = _tk020_state()
    st.adjacency._layers[EdgeKind.FRAME_BRACKET] = SparseAdjacency()
    st.parents.hlink_meta = {}
    st.parents.hlink_endpoints = {}
    fbg._mined_state_ids.clear()
    monkeypatch.setenv("AIVIE_RUNTIME_FB_MINE_MAX_PARENTS", "1")  # 6 cha > 1 → guard bật

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = fbg.ensure_frame_brackets(st)  # KHÔNG logger — đúng như production

    assert n == 0, "guard phải chặn mine"
    msgs = [str(x.message) for x in caught if issubclass(x.category, RuntimeWarning)]
    assert msgs, "guard tắt gate mà không cảnh báo"
    assert "bake_checkpoint_indices" in msgs[0], "cảnh báo phải chỉ cách sửa"


def test_guard_silent_when_mine_allowed(monkeypatch):
    """Checkpoint nhỏ (<=cap) → mine bình thường, KHÔNG cảnh báo (tránh nhiễu)."""
    import warnings

    from nn_aivie.core.adjacency import SparseAdjacency
    from nn_aivie.hca import frame_bracket_gate as fbg

    st, _ids = _tk020_state()
    st.adjacency._layers[EdgeKind.FRAME_BRACKET] = SparseAdjacency()
    st.parents.hlink_meta = {}
    st.parents.hlink_endpoints = {}
    fbg._mined_state_ids.clear()
    monkeypatch.setenv("AIVIE_RUNTIME_FB_MINE_MAX_PARENTS", "5000")  # 6 cha <= cap

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = fbg.ensure_frame_brackets(st)

    assert n > 0, "checkpoint nhỏ phải được mine"
    assert not [x for x in caught if issubclass(x.category, RuntimeWarning)]


def test_guard_silent_when_no_parents():
    """State không có cha → không có gì để mine; guard KHÔNG được kêu (false positive).

    `_runtime_mine_allowed` trả False cả khi `ps is None`, gộp "không có gì để mine" với
    "quá lớn nên không mine" → cảnh báo vô nghĩa "checkpoint 0 cha > cap 5000" nổ ở mọi
    đường sinh không dùng nơ-ron cha (vd test_no_parent_falls_back_to_wseq_chain).
    """
    import warnings

    from nn_aivie.hca import frame_bracket_gate as fbg

    st = GraphState(vocab=Vocabulary())
    fbg._mined_state_ids.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fbg.ensure_frame_brackets(st)
    assert not [x for x in caught if issubclass(x.category, RuntimeWarning)]
