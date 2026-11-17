"""Tests TK_013 — Incremental Segment Graph (ISG)."""

from __future__ import annotations

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.seg_graph import CLUSTER_OFFSET, SegmentGraph
from nn_aivie.core.state import GraphState


def _ps() -> tuple[GraphState, ParentStore]:
    st = GraphState()
    ps = ParentStore()
    ps.seg_min_frequency = 3
    return st, ps


def _ingest(ps: ParentStore, st: GraphState, text: str) -> int:
    ids = st.vocab.encode_tokens(text.split())
    ps.add_example(ids, st.vocab, dap_id=None, priority=1.0)
    return len(ps.parents) - 1


def test_seg_graph_odau_three_parents():
    """Ví dụ 3 câu: ở→đâu đạt ngưỡng 3 → cụm + rewire."""
    st, ps = _ps()
    n1 = _ingest(ps, st, "Hà Nội ở đâu")
    n2 = _ingest(ps, st, "Thủ đô Hà Nội đặt ở đâu")
    n3 = _ingest(ps, st, "Con mèo ở đâu")

    assert len(ps.seg.clusters) == 1
    c = ps.seg.clusters[0]
    o_id = st.vocab.id_of("ở")
    d_id = st.vocab.id_of("đâu")
    assert c.left == o_id and c.right == d_id and c.power == 3

    # n1: Hà → Nội → c1
    chain1 = ps.parents[n1].slot_chain
    assert len(chain1) == 3
    assert chain1[2].kind == "cluster"
    assert chain1[2].id == CLUSTER_OFFSET
    links1 = ps.seg.parent_links(n1)
    assert links1[1].dst.kind == "cluster"
    assert links1[1].src.id == st.vocab.id_of("Nội")

    # n2: ... đặt → c1
    chain2 = ps.parents[n2].slot_chain
    dat_idx = next(i for i, s in enumerate(chain2) if s.kind == "token" and st.vocab.token_of(s.id) == "đặt")
    assert chain2[dat_idx + 1].kind == "cluster"

    # n3: mèo → c1
    chain3 = ps.parents[n3].slot_chain
    meo_idx = next(i for i, s in enumerate(chain3) if s.kind == "token" and st.vocab.token_of(s.id) == "mèo")
    assert chain3[meo_idx + 1].kind == "cluster"

    children = ps.seg.cluster_children(0)
    assert children == ((o_id, 1), (d_id, 2))


def test_seg_graph_multi_link_same_pair():
    """Bigram Hà→Nội có 2 link khác parent."""
    st, ps = _ps()
    _ingest(ps, st, "Hà Nội ở đâu")
    _ingest(ps, st, "Thủ đô Hà Nội đặt ở đâu")
    ha = st.vocab.id_of("Hà")
    noi = st.vocab.id_of("Nội")
    assert ps.seg.bigram_count[(ha, noi)] == 2
    links = [
        ln
        for ln in ps.seg.links
        if ln.src.kind == "token"
        and ln.dst.kind == "token"
        and ln.src.id == ha
        and ln.dst.id == noi
    ]
    parent_idxs = {ln.parent_idx for ln in links}
    assert len(parent_idxs) == 2


def test_rank_matches_cluster_odau():
    st, ps = _ps()
    _ingest(ps, st, "Hà Nội ở đâu")
    _ingest(ps, st, "Thủ đô Hà Nội đặt ở đâu")
    _ingest(ps, st, "Con mèo ở đâu")
    q = st.vocab.encode_tokens("Hồ Tây ở đâu".split())
    ranked = ps.rank_matches(q, st.vocab, embedding=None, top_n=3)
    assert ranked
    units = ps.query_match_units(tuple(q))
    assert units.phrase_units or "ở" in [st.vocab.token_of(t) for t in units.bare_tokens]


def test_checkpoint_seg_roundtrip():
    st, ps = _ps()
    _ingest(ps, st, "Hà Nội ở đâu")
    _ingest(ps, st, "Thủ đô Hà Nội đặt ở đâu")
    _ingest(ps, st, "Con mèo ở đâu")
    payload = ps.to_payload()
    ps2 = ParentStore.from_payload(payload)
    assert len(ps2.seg.clusters) == len(ps.seg.clusters)
    assert ps2.parents[0].slot_chain == ps.parents[0].slot_chain
    assert len(ps2.seg.links) == len(ps.seg.links)


def test_anchor_members_and_left_fillers():
    """Hướng 2 (TK_014): X→"ở đâu", Y→"ở đâu" ⇒ gom X,Y qua nơ-ron cụm chung."""
    st, ps = _ps()
    n1 = _ingest(ps, st, "Hà Nội ở đâu")
    n2 = _ingest(ps, st, "Thủ đô Hà Nội đặt ở đâu")
    n3 = _ingest(ps, st, "Con mèo ở đâu")
    ci = ps.seg.cluster_of_pair[(st.vocab.id_of("ở"), st.vocab.id_of("đâu"))]

    assert set(ps.anchor_members(ci)) == {n1, n2, n3}

    fillers = ps.anchor_left_fillers(ci)
    filler_toks = {
        st.vocab.token_of(ref.id) for ref, _pi in fillers if ref.kind == "token"
    }
    # thực thể X đứng ngay trước "ở đâu" trong mỗi member
    assert {"Nội", "đặt", "mèo"} <= filler_toks

    # query mang "ở đâu" → cụm được kích thích, độc lập gate scoring
    q = tuple(st.vocab.encode_tokens("Hồ Tây ở đâu".split()))
    assert ci in ps.query_anchor_clusters(q)


def test_match_common_clusters_admits_frequent_wh_cluster():
    """Hướng 1 (TK_014): cụm phổ biến (df cao) vào phrase_units khi cờ BẬT; bị gate khi TẮT."""
    st, ps = _ps()
    ps.seg_min_frequency = 9  # promote khi đủ 9 → power/df = 9 > phrase_max_df mặc định (8)
    # 9 cha chứa "ở đâu"
    for i in range(9):
        _ingest(ps, st, f"thucthe{i} ở đâu")
    ci = ps.seg.cluster_of_pair[(st.vocab.id_of("ở"), st.vocab.id_of("đâu"))]
    j = ps._phrase_lookup[(st.vocab.id_of("ở"), st.vocab.id_of("đâu"))]
    assert ps.phrase_df[j] > 8  # phổ biến → gate cũ loại

    q = tuple(st.vocab.encode_tokens("Hồ Tây ở đâu".split()))

    ps.match_common_clusters = False
    off = ps.query_match_units(q)
    assert j not in off.phrase_units
    assert st.vocab.id_of("đâu") in off.bare_tokens  # rơi về token rời

    ps.match_common_clusters = True
    on = ps.query_match_units(q)
    assert j in on.phrase_units  # được kích thích làm đơn vị match
    assert ci == ps.seg.cluster_of_pair[(st.vocab.id_of("ở"), st.vocab.id_of("đâu"))]


def test_rebuild_seg_graph_from_source_ids():
    st, ps = _ps()
    for t in ["Hà Nội ở đâu", "Thủ đô Hà Nội đặt ở đâu", "Con mèo ở đâu"]:
        _ingest(ps, st, t)
    n_clusters = len(ps.seg.clusters)
    ps.seg = SegmentGraph()
    for rec in ps.parents:
        rec.slot_chain = ()
    res = ps.rebuild_seg_graph(st.vocab)
    assert res["clusters"] == n_clusters
    assert ps.seg.clusters[0].power == 3


def test_seed_pairs_promote_without_min_frequency():
    """TK_033 S1: seed_pairs cho phép promote dù count < min_frequency."""
    st, ps = _ps()
    # Chỉ 1 câu ⇒ count=1 < min_frequency=3, nhưng nếu seed (ở,đâu) thì vẫn promote.
    n1 = _ingest(ps, st, "Hà Nội ở đâu")
    o_id = st.vocab.id_of("ở")
    d_id = st.vocab.id_of("đâu")
    assert ps.seg.bigram_count[(o_id, d_id)] == 1
    assert (o_id, d_id) not in ps.seg.cluster_of_pair

    ps.seg.seed_pairs.add((o_id, d_id))
    n_promoted = ps.seg.finalize_promotions(ps.parents)
    assert n_promoted >= 1
    ci = ps.seg.cluster_of_pair[(o_id, d_id)]
    assert ps.seg.clusters[ci].left == o_id and ps.seg.clusters[ci].right == d_id
    chain1 = ps.parents[n1].slot_chain
    assert any(s.kind == "cluster" and s.id == CLUSTER_OFFSET + ci for s in chain1)


def test_compositional_vi_du_mau_three_tokens():
    """TK_038: Ví dụ mẫu (3 token) → cl(Ví,dụ) + mẫu → span 3."""
    st, ps = _ps()
    for _ in range(5):
        _ingest(ps, st, "Ví dụ mẫu")
    vi = st.vocab.id_of("Ví")
    du = st.vocab.id_of("dụ")
    mau = st.vocab.id_of("mẫu")
    assert (vi, du) in ps.seg.cluster_of_pair
    ci2 = ps.seg.cluster_of_pair[(vi, du)]
    assert ps.seg.clusters[ci2].token_span() == (vi, du)
    cl_slot = ps.seg.cluster_slot(ci2)
    comp_key = ("cluster", cl_slot.id, "token", mau)
    assert comp_key in ps.seg.composite_of_pair, ps.seg.composite_count
    ci3 = ps.seg.composite_of_pair[comp_key]
    assert ps.seg.clusters[ci3].token_span() == (vi, du, mau)


def test_overlap_power_wins_strong_pair():
    """Rival yếu (ban,học) power=3; (học,sinh) count=5 > 3 → vẫn promote."""
    st, ps = _ps()
    ps.seg.begin_batch()
    for _ in range(3):
        _ingest(ps, st, "ban học")
    for _ in range(5):
        _ingest(ps, st, "học sinh")
    hoc = st.vocab.id_of("học")
    ban = st.vocab.id_of("ban")
    sinh = st.vocab.id_of("sinh")
    assert ps.seg.bigram_count[(ban, hoc)] == 3
    assert ps.seg.bigram_count[(hoc, sinh)] == 5
    # Promote rival yếu trước (giả lập ingest sớm), rồi finalize phần còn lại
    assert ps.seg.should_promote((ban, hoc))
    ps.seg.promote_pair(ban, hoc, ps.parents)
    assert (ban, hoc) in ps.seg.cluster_of_pair
    assert ps.seg.clusters[ps.seg.cluster_of_pair[(ban, hoc)]].power == 3
    # (học,sinh) đang overlap-blocked nhưng count thắng → vào heap + promote
    assert ps.seg._bigram_overlap_blocked((hoc, sinh))
    assert ps.seg._overlap_power_wins((hoc, sinh))
    n = ps.seg.finalize_promotions(ps.parents)
    assert n >= 1
    assert (hoc, sinh) in ps.seg.cluster_of_pair
    assert (ban, hoc) in ps.seg.cluster_of_pair  # không demote


def test_overlap_power_loses_to_stronger_rival():
    """Rival mạnh (ban,học) power=10; (học,sinh) count=5 → vẫn chặn TK_038."""
    st, ps = _ps()
    ps.seg.begin_batch()
    for _ in range(10):
        _ingest(ps, st, "ban học")
    for _ in range(5):
        _ingest(ps, st, "học sinh")
    hoc = st.vocab.id_of("học")
    ban = st.vocab.id_of("ban")
    sinh = st.vocab.id_of("sinh")
    assert ps.seg.bigram_count[(ban, hoc)] == 10
    assert ps.seg.bigram_count[(hoc, sinh)] == 5
    ps.seg.promote_pair(ban, hoc, ps.parents)
    assert ps.seg.clusters[ps.seg.cluster_of_pair[(ban, hoc)]].power == 10
    assert ps.seg._bigram_overlap_blocked((hoc, sinh))
    assert not ps.seg._overlap_power_wins((hoc, sinh))
    ps.seg.finalize_promotions(ps.parents)
    assert (hoc, sinh) not in ps.seg.cluster_of_pair
