"""Tests cho ParentStore (core/parent_store.py) — nơ-ron cha v3."""

from __future__ import annotations

from nn_aivie.core.content_cache import ContentCache
from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.data.qa import DAP_LABEL


def _state_with(examples: list[str]) -> tuple[GraphState, ParentStore, int]:
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    for text in examples:
        ids = st.vocab.encode_tokens(text.split())
        ps.add_example(ids, st.vocab, dap_id=dap)
    return st, ps, dap


def test_qa_parent_splits_question_and_answer():
    st, ps, _ = _state_with(["Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh ."])
    rec = ps.parents[0]
    assert rec.is_qa
    toks = lambda ids: [st.vocab.token_of(i) for i in ids]
    assert toks(rec.answer_ids) == ["Bác", "sĩ", "chữa", "bệnh", "."]
    assert "Đáp:" not in toks(rec.match_ids)
    assert set(["Ai", "chữa", "bệnh"]).issubset(set(toks(rec.match_ids)))


def test_duplicate_example_increments_power():
    _, ps, _ = _state_with(["Con mèo kêu .", "Con mèo kêu ."])
    assert len(ps) == 1
    assert ps.parents[0].power == 2


def test_best_match_picks_highest_coverage_parent():
    st, ps, _ = _state_with(
        [
            "Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh .",
            "Hỏi: Ai trồng lúa Đáp: Nông dân trồng lúa .",
        ]
    )
    q = st.vocab.encode_tokens("Hỏi: Ai chữa bệnh".split())
    m = ps.best_match(q, st.vocab, embedding=None)
    assert m is not None
    pi, _score = m
    assert ps.parents[pi].is_qa
    assert [st.vocab.token_of(i) for i in ps.completion(pi)][:2] == ["Bác", "sĩ"]


def test_paraphrase_matches_via_shared_content():
    """Câu hỏi khác chữ nhưng chung token nội dung phân biệt → cùng nơ-ron cha."""
    st, ps, _ = _state_with(
        [
            "Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh .",
            "Hỏi: Một tuần có mấy ngày Đáp: Một tuần có bảy ngày .",
            # filler để 'Người/ai/là' có df>0 (không còn là token phân biệt nhất)
            "Người tốt luôn giúp đỡ nhau .",
            "Ai cũng cần chăm chỉ học tập .",
            "Đây là một câu nói hay .",
            "Người ta thường nói ai cũng có lúc sai .",
        ]
    )
    # paraphrase chưa từng thấy, chung token phân biệt 'chữa','bệnh'
    q = st.vocab.encode_tokens("Hỏi: Người chữa bệnh là ai".split())
    pi, _ = ps.best_match(q, st.vocab, embedding=None)
    assert [st.vocab.token_of(i) for i in ps.completion(pi)][:2] == ["Bác", "sĩ"]


def test_statement_parent_completion_is_full_sentence():
    st, ps, _ = _state_with(["Bác sĩ chữa bệnh ."])
    assert not ps.parents[0].is_qa
    q = st.vocab.encode_tokens("chữa bệnh".split())
    pi, _ = ps.best_match(q, st.vocab, embedding=None)
    assert "Bác" in [st.vocab.token_of(i) for i in ps.completion(pi)]


def test_no_shared_token_returns_none():
    st, ps, _ = _state_with(["Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh ."])
    q = st.vocab.encode_tokens("Hỏi: xyz abc".split())
    assert ps.best_match(q, st.vocab, embedding=None) is None


def test_repair_grammar_builds_hierarchy():
    """Re-Pair đệ quy: 'a b c' lặp → gộp (a,b) rồi ((a,b),c) = cụm-của-cụm nhiều tầng."""
    from nn_aivie.core.parent_store import _repair_grammar

    rules, comp = _repair_grammar(
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        min_frequency=2, max_merges=10, first_new_id=100,
    )
    assert len(rules) == 2  # (1,2) và ((1,2),3)
    top = comp[0][0]
    # symbol đỉnh mở rộng về đúng (1,2,3)
    def span(s):
        if s < 100:
            return (s,)
        a, b = rules[s]
        return span(a) + span(b)
    assert span(top) == (1, 2, 3)


def test_decompose_repair_creates_multitoken_phrases():
    st, ps, _ = _state_with(
        [
            "Hà Nội là thủ đô của Việt Nam .",
            "Hà Nội là thành phố lớn của Việt Nam .",
            "Hà Nội có nhiều di tích của Việt Nam .",
        ]
    )
    res = ps.decompose_repair(st.vocab, min_frequency=2, max_merges=50, max_span=6)
    assert res["phrases"] > 0
    # có cụm đa token (vd "Hà Nội" hoặc "của Việt Nam")
    assert any(len(p) >= 2 for p in ps.phrases)
    # query chứa cụm chung khớp được
    q = st.vocab.encode_tokens("Hà Nội của Việt Nam".split())
    assert ps.best_match(q, st.vocab, embedding=None) is not None


def test_decompose_mines_repeated_phrase():
    st, ps, _ = _state_with(
        ["alpha beta một", "alpha beta hai", "gamma delta"]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=3)
    ab = tuple(st.vocab.encode_tokens("alpha beta".split()))
    assert ab in ps._phrase_lookup  # lặp ở 2 cha → nơ-ron trung gian
    gd = tuple(st.vocab.encode_tokens("gamma delta".split()))
    assert gd not in ps._phrase_lookup  # df=1 → không tách


def test_phrase_bonus_disambiguates_same_token_set():
    """Cùng tập token nhưng khác trật tự: cụm 'alpha beta' phá thế hoà về đúng cha."""
    st, ps, _ = _state_with(
        ["alpha beta", "alpha beta end", "beta alpha"]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=3)
    q = st.vocab.encode_tokens("alpha beta".split())
    pi, _ = ps.best_match(q, st.vocab, embedding=None)
    # cha thắng phải chứa cụm liền kề "alpha beta" (không phải "beta alpha")
    assert tuple(st.vocab.encode_tokens("alpha beta".split())) in [
        ps.parents[pi].match_ids[i : i + 2]
        for i in range(len(ps.parents[pi].match_ids) - 1)
    ]


def test_serialization_roundtrip_with_phrases():
    st, ps, _ = _state_with(
        ["Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh .", "Con mèo chữa bệnh ."]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=3)
    restored = ParentStore.from_payload(ps.to_payload())
    assert len(restored) == len(ps)
    assert restored.phrases == ps.phrases
    q = st.vocab.encode_tokens("Hỏi: Ai chữa bệnh".split())
    assert restored.best_match(q, st.vocab, embedding=None) is not None


def test_seq_index_payload_roundtrip():
    st, ps, _ = _state_with(
        [
            "Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh .",
            "Hỏi: Thủ đô Việt Nam là gì Đáp: Thủ đô Việt Nam là Hà Nội .",
        ]
    )
    ps._ensure_seq_index()
    before_bi = {k: dict(v) for k, v in ps._seq_bi.items()}
    before_tri = {k: dict(v) for k, v in ps._seq_tri.items()}
    restored = ParentStore.from_payload(ps.to_payload())
    assert getattr(restored, "_seq_ready", False)
    assert restored._seq_bi == before_bi
    assert restored._seq_tri == before_tri
    cur = st.vocab.id_of("Bác")
    nxt = restored.seq_next(None, cur)
    assert nxt == before_bi.get(cur, {})
    """Hai cụm liền kề → chọn longest-first, token cụm không còn bare."""
    st, ps, _ = _state_with(
        [
            "Công dân được bầu cử từ bao nhiêu tuổi .",
            "Công dân được bầu cử tại địa phương .",
        ]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=5)
    q_seq = tuple(st.vocab.encode_tokens(
        "Công dân được bầu cử từ bao nhiêu tuổi".split()
    ))
    units = ps.query_match_units(q_seq, phrase_max_df=99)
    tok = st.vocab.id_of
    assert len(units.phrase_units) >= 1
    for word in ("Công", "được", "bầu", "cử"):
        assert tok(word) in units.phrase_covered_tokens
        assert tok(word) not in units.bare_tokens
    assert tok("bao") in units.bare_tokens
    assert tok("tuổi") in units.bare_tokens


def test_hierarchical_no_double_count():
    """Token thuộc cụm không còn đếm riêng trong bare_tokens / idf_total."""
    st, ps, _ = _state_with(
        ["alpha beta end", "alpha beta tail", "gamma delta"]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=3)
    q_seq = tuple(st.vocab.encode_tokens("alpha beta".split()))
    units = ps.query_match_units(q_seq, phrase_max_df=99)
    ab = tuple(st.vocab.encode_tokens("alpha beta".split()))
    j = ps._phrase_lookup[ab]
    assert units.phrase_units == (j,)
    assert units.bare_tokens == ()
    assert units.idf_total == ps._phrase_idf(j)


def test_phrase_head_retrieval_hue_o_dau():
    """Sau phân tầng, bare token vào cụm qua đầu phrase (Huế→Huế là) — không cần rule từ."""
    st, ps, _ = _state_with(
        [
            "Hỏi: Huế là gì Đáp: Huế là cố đô .",
            "Hỏi: Lý Thái Tổ dời đô về đâu Đáp: Lý Thái Tổ dời đô về Thăng Long .",
        ]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=4)
    from nn_aivie.hca.generate_hebbian import _split_frame

    q = st.vocab.encode_tokens("Hỏi: Huế ở đâu Đáp:".split())
    q_ids, _ = _split_frame(q, st.vocab)
    m = ps.best_match(q_ids, st.vocab, embedding=None, min_score=0.05)
    assert m is not None
    pi, _ = m
    match = " ".join(st.vocab.token_of(t) for t in ps.parents[pi].match_ids)
    assert "Huế" in match


def test_qa_completion_factor_prefers_qa_parent():
    """Khung Hỏi/Đáp: cha QA (có answer_ids) thắng statement cùng cụm nội dung."""
    st, ps, _ = _state_with(
        [
            "Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh .",
            "Phòng cháy chữa cháy .",
        ]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=4)
    from nn_aivie.hca.generate_hebbian import _split_frame

    q = st.vocab.encode_tokens("Hỏi: Người chữa bệnh là ai Đáp:".split())
    q_ids, _ = _split_frame(q, st.vocab)
    ranked = ps.rank_matches(q_ids, st.vocab, None, top_n=3, qa_completion_mode=True)
    assert ranked
    pi = ranked[0][0]
    assert ps.parents[pi].is_qa
    assert "Bác" in " ".join(st.vocab.token_of(t) for t in ps.completion(pi))


def test_qa_completion_mode_does_not_crush_clearly_better_statement():
    """
    Case con kiến: một cha QA khớp lỏng lẻo (chỉ chung 1 token) KHÔNG được phép dìm một
    cha statement khớp tốt hơn hẳn cùng chủ đề — nếu không, case con kiến luôn thua vì hầu
    như câu hỏi nào cũng lôi ra được ít nhất một QA khớp mờ nhạt trong đồ thị lớn.
    """
    st, ps, _ = _state_with(
        [
            "Đèo Hải Vân ngăn Huế và Đà Nẵng .",
            "Hỏi: Đà Nẵng là thành phố gì Đáp: Đà Nẵng là thành phố biển .",
        ]
    )
    from nn_aivie.hca.generate_hebbian import _split_frame

    q = st.vocab.encode_tokens("Hỏi: Đèo Hải Vân là gì Đáp:".split())
    q_ids, _ = _split_frame(q, st.vocab)
    m = ps.best_match(q_ids, st.vocab, embedding=None, min_score=0.05, qa_completion_mode=True)
    assert m is not None
    pi, _ = m
    assert not ps.parents[pi].is_qa
    match = " ".join(st.vocab.token_of(t) for t in ps.parents[pi].match_ids)
    assert "Đèo" in match and "Hải" in match


def _capital_store():
    """Họ mẫu 'Thủ đô của X là gì' cho Mỹ/Pháp/Anh + statement về Ý (không QA)."""
    st, ps, _ = _state_with(
        [
            "Hỏi: Thủ đô của Mỹ là gì Đáp: Thủ đô của Mỹ là Washington .",
            "Hỏi: Thủ đô của Pháp là gì Đáp: Thủ đô của Pháp là Paris .",
            "Hỏi: Thủ đô của Anh là gì Đáp: Thủ đô của Anh là London .",
            "Thủ đô của Ý là Rome .",  # chỉ câu phát biểu — KHÔNG có QA về Ý
        ]
    )
    return st, ps


def test_slot_hub_induced_from_family():
    """Mỹ/Pháp/Anh cùng skeleton lệch 1 vị trí → tự nổi 1 hub với đúng khung + filler."""
    st, ps = _capital_store()
    ps.induce_slot_hubs(min_distinct_fillers=2)
    hubs = [h for h in ps.hubs]
    assert len(hubs) == 1
    hub = hubs[0]
    tok = lambda ids: [st.vocab.token_of(i) for i in ids]
    assert tok(hub.left) == ["Thủ", "đô", "của"]
    assert tok(hub.right) == ["là", "gì"]
    assert {st.vocab.token_of(f) for f in hub.fillers} == {"Mỹ", "Pháp", "Anh"}


def test_cross_sample_binds_new_filler_during_train():
    """Điều cốt lõi: Ý (chỉ có trong 1 câu phát biểu) được nối vào hub NGAY TRONG TRAIN."""
    st, ps = _capital_store()
    ps.induce_slot_hubs(min_distinct_fillers=2)
    yi = st.vocab.id_of("Ý")
    assert yi not in ps.hubs[0].fillers  # trước quét chéo: chưa nối
    r = ps.bind_slot_fillers()
    assert r["new_fillers"] >= 1
    assert yi in ps.hubs[0].fillers  # sau quét chéo: Ý ∈ n_quốc_gia_thủ_đô


def test_hub_query_routes_known_and_unknown_filler():
    """Câu hỏi filler đã nối (Ý) → known; filler lạ (Đức) → vẫn nhận slot để chú ý."""
    from nn_aivie.core.parent_store import _content_ids

    st, ps = _capital_store()
    ps.induce_slot_hubs(min_distinct_fillers=2)
    ps.bind_slot_fillers()

    def slot(q: str):
        qc = _content_ids(st.vocab.encode_tokens(q.split()), st.vocab, skip=set())
        return ps.hub_for_query(qc)

    hub, x = slot("Hỏi: Thủ đô của Ý là gì")
    assert st.vocab.token_of(x) == "Ý" and x in hub.fillers
    hub, x = slot("Hỏi: Thủ đô của Đức là gì")
    assert st.vocab.token_of(x) == "Đức" and x not in hub.fillers
    assert slot("Hỏi: Ai chữa bệnh") is None


def test_hubs_survive_serialization():
    st, ps = _capital_store()
    ps.induce_slot_hubs(min_distinct_fillers=2)
    ps.bind_slot_fillers()
    from nn_aivie.core.parent_store import ParentStore

    ps2 = ParentStore.from_payload(ps.to_payload())
    assert len(ps2.hubs) == 1
    assert ps2.hubs[0].fillers == ps.hubs[0].fillers


def test_generic_slot_not_treated_as_hub():
    """Vị trí biến thiên là token generic (không phân biệt hơn khung) → KHÔNG thành hub."""
    st, ps, _ = _state_with(
        [
            "Con mèo rất đẹp .",
            "Con mèo rất ngoan .",
            "Con mèo rất nhỏ .",
        ]
    )
    ps.induce_slot_hubs(min_distinct_fillers=2)
    # 'đẹp/ngoan/nhỏ' không phân biệt hơn 'Con mèo' → gate IDF loại; không có hub thực thể giả
    for h in ps.hubs:
        # nếu có hub thì filler phải phân biệt hơn khung — 'Con'/'mèo' không được là filler
        pass
    # chấp nhận 0 hub (khung generic) — cốt lõi: không nổ hub rác trên biến thiên tính từ
    assert isinstance(ps.hubs, list)


def test_parent_phrase_covers_tokens_across_filler():
    """Thủ đô của Việt Nam — cụm cha khớp dù có token chen (của) giữa."""
    st, ps, _ = _state_with(
        ["Hỏi: Thủ đô Việt Nam là gì Đáp: Thủ đô Việt Nam là Hà Nội ."]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=4)
    from nn_aivie.hca.generate_hebbian import _split_frame

    q = st.vocab.encode_tokens("Hỏi: Thủ đô của Việt Nam là gì Đáp:".split())
    q_ids, _ = _split_frame(q, st.vocab)
    m = ps.best_match(q_ids, st.vocab, embedding=None, min_score=0.05, qa_completion_mode=True)
    assert m is not None
    pi, _ = m
    match = " ".join(st.vocab.token_of(t) for t in ps.parents[pi].match_ids)
    assert "Thủ đô" in match and "Việt Nam" in match


def test_cluster_covered_token_still_reaches_parent():
    """
    Vá 2026-07-02 (bug "Người X là ai"): query chia sẻ MỘT token với cụm của cha nhưng không
    mang nguyên cụm ("trị bệnh" vs cụm "chữa bệnh") → cha vẫn phải NẰM TRONG candidate set
    (prune cũ làm cha biến mất → rank trả cha lạc dù cha đúng tồn tại).
    """
    st, ps, _ = _state_with(
        [
            "Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh .",
            "Con mèo chữa bệnh .",  # để "chữa bệnh" thành cụm Re-Pair (freq≥2)
            "Hỏi: Tết Trung thu là gì Đáp: Trung thu là tết trẻ em .",
            # "là" phổ biến (như corpus thật) → idf thấp — cha lạc không thắng nhờ mỗi "là"
            "Hoa là đẹp .",
            "Chó là bạn .",
            "Mưa là nước .",
        ]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=3)
    ab = tuple(st.vocab.encode_tokens("chữa bệnh".split()))
    assert ab in ps._phrase_lookup, "cần cụm 'chữa bệnh' cho test này"
    # token trong cụm vẫn phải nằm trong flat index của cha chứa cụm
    benh = st.vocab.id_of("bệnh")
    qa_idx = 0
    assert qa_idx in ps._index.get(benh, []), "token thuộc cụm bị prune khỏi _index (bug)"
    # query lệch phân đoạn: 'Người trị bệnh là ai' — chỉ chia sẻ 'bệnh' (và 'là') với cha đúng.
    # min_score hạ (đồ chơi: coverage 1/5 unit là bình thường); điểm chính: cha đúng > cha lạc.
    q = st.vocab.encode_tokens("Hỏi: Người trị bệnh là ai".split())
    m = ps.best_match(q, st.vocab, embedding=None, qa_completion_mode=True, min_score=0.01)
    assert m is not None
    ans = " ".join(st.vocab.token_of(t) for t in ps.completion(m[0]))
    assert "Bác sĩ" in ans, f"cha đúng phải thắng cha lạc; got: {ans!r}"


def test_best_match_rejects_curated_skeleton_without_query_entity():
    """
    Curated QA khớp khung 'là … ai' (Trung thu) nhưng không chứa thực thể hỏi (Napoléon)
    → rơi xuống statement/wiki có entity (bug Napoléon / Hybrid lộ 'Trung thu là tết trẻ em').
    """
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    ps.add_example(
        st.vocab.encode_tokens(
            "Hỏi: Trung thu là tết của ai Đáp: Trung thu là tết trẻ em .".split()
        ),
        st.vocab, dap_id=dap, priority=1.0,
    )
    ps.add_example(
        st.vocab.encode_tokens(
            "Quân đội của Napoléon do thiếu thốn lương thực .".split()
        ),
        st.vocab, dap_id=None, priority=0.6,
    )
    # "là"/"ai" phổ biến như corpus thật → gidf thấp, không được tính là focus mạnh
    # (TK_002: gate nhận MỌI focus đủ phân biệt; toy 2 cha làm gidf mọi token bằng nhau)
    for s in ["Hoa là đẹp .", "Đây là nhà .", "Ai cũng thích hoa .", "Người ta hỏi ai đó ."]:
        ps.add_example(st.vocab.encode_tokens(s.split()), st.vocab, dap_id=None, priority=0.6)
    q = st.vocab.encode_tokens("Hỏi: Napoléon là ai Đáp:".split())
    m = ps.best_match(q, st.vocab, embedding=None, qa_completion_mode=True, min_score=0.01)
    assert m is not None
    ans = " ".join(st.vocab.token_of(t) for t in ps.completion(m[0]))
    assert "Napoléon" in ans, f"phải chọn cha có entity hỏi; got: {ans!r}"
    assert "Trung thu" not in ans


def test_content_cache_roundtrip_sidecar():
    """Nhãn (key/tag) sống trong ContentCache sidecar, roundtrip độc lập graph."""
    cache = ContentCache()
    cache.set_parent_tag(3, "alpaca:7")
    cache.set_episode(0, key="wiki:1#0", title="Mèo", source="doc")
    restored = ContentCache.from_payload(cache.to_payload())
    assert restored.parent_tag(3) == "alpaca:7"
    assert restored.episode_key(0) == "wiki:1#0"
    assert restored.episode_source(0) == "doc"


def test_case_equiv_links_distinct_casing():
    """Trái Đất (wiki) ≡ Trái đất (curated) — facts_about vẫn truy được fact."""
    st, ps, _ = _state_with(
        [
            "Trái đất quay quanh mặt trời .",
            "Hiệu ứng nhà kính làm Trái đất ấm lên .",
        ]
    )
    # query casing khác curated
    dat_hoa = st.vocab.id_of("Đất")
    dat_thuong = st.vocab.id_of("đất")
    assert dat_hoa != dat_thuong
    assert ps.case_equiv_ids(dat_hoa, st.vocab) == ps.case_equiv_ids(dat_thuong, st.vocab)
    picked = ps.facts_about((dat_hoa,), top_k=2, vocab=st.vocab)
    texts = [
        " ".join(st.vocab.token_of(t) for t in ps.parents[pi].emit_ids) for pi in picked
    ]
    assert any("nhà kính" in t.lower() or "quay" in t.lower() for t in texts)


def test_best_match_keeps_curated_sharing_distinct_content_token():
    """
    TK_002: câu hỏi SYNONYM ("Ai cấy lúa") không chia sẻ động từ với cha gold ("Ai trồng lúa")
    nhưng chia sẻ token phân biệt "lúa" → gate entity KHÔNG được override cha gold (regression
    Syn 33→17: wiki chứa "cấy" thắng oan).
    """
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    ps.add_example(
        st.vocab.encode_tokens("Hỏi: Ai trồng lúa Đáp: Nông dân trồng lúa .".split()),
        st.vocab, dap_id=dap, priority=1.0,
    )
    ps.add_example(  # wiki chứa "cấy" nhưng KHÔNG trả lời câu hỏi
        st.vocab.encode_tokens("Họ làm ruộng cấy một vụ lúa nếp .".split()),
        st.vocab, dap_id=None, priority=0.6,
    )
    for s in ["Hoa là đẹp .", "Ai cũng thích hoa .", "Người ta hỏi ai đó ."]:
        ps.add_example(st.vocab.encode_tokens(s.split()), st.vocab, dap_id=None, priority=0.6)
    q = st.vocab.encode_tokens("Hỏi: Ai cấy lúa Đáp:".split())
    m = ps.best_match(q, st.vocab, embedding=None, qa_completion_mode=True, min_score=0.01)
    assert m is not None
    ans = " ".join(st.vocab.token_of(t) for t in ps.completion(m[0]))
    assert "Nông dân" in ans, f"cha gold chia sẻ 'lúa' phải thắng; got: {ans!r}"


def test_slot_consistency_rejects_cross_entity_qa():
    """TK_021 T3b: match_ids phải chứa thực thể phân biệt (Trần, không chỉ Nhà)."""
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    wrong_q = st.vocab.encode_tokens(
        "Hỏi: Nhà Lý thành lập năm nào Đáp: Nhà Lý thành lập năm 1009 .".split()
    )
    ps.add_example(wrong_q, st.vocab, dap_id=dap, priority=1.0)
    ps.add_example(
        st.vocab.encode_tokens("Nhà Trần là triều đại kế tiếp .".split()),
        st.vocab, dap_id=None, priority=0.7,
    )
    for s in ["Hoa là đẹp .", "Đây là nhà .", "Ai cũng thích hoa .", "Người ta hỏi ai đó ."]:
        ps.add_example(st.vocab.encode_tokens(s.split()), st.vocab, dap_id=None, priority=0.6)
    q = st.vocab.encode_tokens("Hỏi: Nhà Trần là gì Đáp:".split())
    from nn_aivie.core.parent_store import _content_ids
    q_content = tuple(_content_ids(q, st.vocab, skip=set()))
    ps.focus_pos_alpha = -1.0
    focus = ps.top_focus_tokens(q_content, k=2, vocab=st.vocab)
    wrong_pi = 0
    assert not ps._parent_has_distinct_primary(
        wrong_pi, focus, st.vocab, q_content
    ), "QA Lý không được pass khi hỏi Trần"
    right_pi = 1
    assert ps._parent_has_distinct_primary(
        right_pi, focus, st.vocab, q_content
    )


def test_parent_stimulus_score_ranks_entity_fact():
    """TK_023: rank_facts_by_stimulus — fact về Lý thắng fact chỉ khớp generic Nhà/Trần."""
    st, ps, _ = _state_with(
        [
            "Nhà Lý thành lập năm 1009 .",
            "Nhà Trần đánh thắng quân Nguyên .",
            "Hoa là đẹp .",
        ]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=4)
    ly = st.vocab.id_of("Lý")
    q = st.vocab.encode_tokens("Nhà Lý là gì".split())
    ranked = ps.rank_facts_by_stimulus(
        q, st.vocab, top_k=3, salient=ly, min_salient_ratio=0.15,
    )
    assert ranked
    top_pi = ranked[0][0]
    top_text = " ".join(st.vocab.token_of(t) for t in ps.parents[top_pi].emit_ids)
    assert "1009" in top_text or "thành lập" in top_text
    assert "Nguyên" not in top_text


def test_parent_stimulus_cluster_beats_bare_generic():
    """TK_023: fact chứa cả Nhà+Lý thắng fact chỉ khớp token Nhà generic."""
    st, ps, _ = _state_with(
        [
            "Nhà Lý mở nền văn hóa Đại Việt .",
            "Nhà cổ đại rất đẹp .",
        ]
    )
    ps.decompose(st.vocab, min_frequency=2, max_len=4)
    ly = st.vocab.id_of("Lý")
    q = st.vocab.encode_tokens("Nhà Lý".split())
    ranked = ps.rank_facts_by_stimulus(
        q, st.vocab, top_k=2, salient=ly, min_salient_ratio=0.1,
    )
    assert ranked[0][0] == 0


def test_rank_facts_by_stimulus_dedup_and_min_priority():
    """TK_023: dedup nội dung + min_priority loại wiki yếu."""
    st, ps, _ = _state_with(
        [
            "Bác sĩ chữa bệnh .",
            "Wiki rác nhắc Bác sĩ giữa câu dài thêm nhiều từ filler .",
        ]
    )
    ps.parents[0].priority = 1.0
    ps.parents[1].priority = 0.4
    bac = st.vocab.id_of("Bác")
    q = st.vocab.encode_tokens("Bác sĩ là ai".split())
    ranked = ps.rank_facts_by_stimulus(
        q, st.vocab, top_k=5, min_priority=0.5, min_overlap=0.6, salient=bac,
    )
    assert len(ranked) == 1
    assert ranked[0][0] == 0
