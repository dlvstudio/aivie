"""Tests cho lõi sinh v3 (hca/generate_hebbian.py) — nơ-ron cha + pattern completion."""

from __future__ import annotations

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import SPECIAL_EOS
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.generate_hebbian import generate, _knowledge_extra_facts_eligible, _split_frame
from nn_aivie.training.ingest import _g1_and_wseq


def _qa_state(examples: list[str]) -> GraphState:
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    for text in examples:
        ids = list(st.vocab.encode_tokens(text.split()))
        ps.add_example(ids, st.vocab, dap_id=dap)
        # Nạp W_seq + co-occurrence (như G1 lúc train) để đường sinh A3 hợp nhất chảy được;
        # trước đây helper chỉ dựng nơ-ron cha (đủ cho dội verbatim, KHÔNG đủ cho A3).
        _g1_and_wseq(st, ids, eta=1.0)
    st.parents = ps
    return st


def test_completion_emits_memorized_answer():
    # HỢP NHẤT P1: sinh A3 token-by-token (không dội verbatim) → tái hiện fact đã thuộc ĐÚNG
    # trật tự nội dung; dấu câu cuối do EOS đóng (có thể lược) — kiểm tra nội dung, không chuỗi.
    st = _qa_state(["Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh ."])
    ids = generate(st, "Hỏi: Ai chữa bệnh Đáp:", max_len=12)
    ans = decode_surface(st, ids).split("Đáp:")[-1].strip()
    assert ans.split()[:4] == ["Bác", "sĩ", "chữa", "bệnh"]


def test_paraphrase_completes_same_answer():
    st = _qa_state(
        [
            "Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh .",
            "Hỏi: Thủ đô Việt Nam là gì Đáp: Thủ đô Việt Nam là Hà Nội .",
            "Người tốt luôn giúp đỡ nhau .",
            "Ai cũng cần chăm chỉ học tập .",
            "Đây là một câu nói hay .",
            "Người ta thường nói ai cũng có lúc sai .",
        ]
    )
    ids = generate(st, "Hỏi: Người chữa bệnh là ai Đáp:", max_len=12)
    ans = decode_surface(st, ids).split("Đáp:")[-1].strip()
    assert "Bác" in ans and "sĩ" in ans


def _capital_family_state() -> GraphState:
    st = _qa_state(
        [
            "Hỏi: Thủ đô của Mỹ là gì Đáp: Thủ đô của Mỹ là Washington .",
            "Hỏi: Thủ đô của Pháp là gì Đáp: Thủ đô của Pháp là Paris .",
            "Hỏi: Thủ đô của Anh là gì Đáp: Thủ đô của Anh là London .",
            "Thủ đô của Ý là Rome .",  # statement về Ý — không có QA
        ]
    )
    st.parents.decompose(st.vocab, min_frequency=2, max_len=4)
    st.parents.induce_slot_hubs(min_distinct_fillers=2)
    st.parents.bind_slot_fillers()
    return st


def test_hub_routes_novel_filler_to_entity_statement():
    """Filler lạ CÓ statement đúng thực thể → route tới nó (case con kiến qua hub)."""
    st = _capital_family_state()
    ids = generate(st, "Hỏi: Thủ đô của Ý là gì Đáp:", max_len=16)
    ans = decode_surface(st, ids).split("Đáp:")[-1].strip()
    assert "Ý" in ans and "Rome" in ans


def test_hub_suppresses_cross_filler_hallucination():
    """Filler lạ KHÔNG có data đúng thực thể → không bịa fact của filler khác (đáp rỗng)."""
    st = _capital_family_state()
    ids = generate(st, "Hỏi: Thủ đô của Đức là gì Đáp:", max_len=16)
    ans = decode_surface(st, ids).split("Đáp:")[-1].strip()
    # KHÔNG được bịa Washington/Paris/London (fact của Mỹ/Pháp/Anh)
    assert "Washington" not in ans and "Paris" not in ans and "London" not in ans


def test_known_filler_still_answers_with_guard():
    """Filler đã biết (có QA) → guard không can thiệp, vẫn đáp đúng."""
    st = _capital_family_state()
    ids = generate(st, "Hỏi: Thủ đô của Pháp là gì Đáp:", max_len=16)
    ans = decode_surface(st, ids).split("Đáp:")[-1].strip()
    assert "Paris" in ans


def test_facts_about_dedups_and_ranks_by_entity():
    """facts_about: chỉ statement chứa thực thể trọng tâm, khử trùng lặp."""
    st = _qa_state(
        [
            "Hổ là loài ăn thịt lớn .",
            "Hổ là loài ăn thịt lớn nhất .",  # gần trùng → khử bớt
            "Hổ sống trong rừng rậm .",
        ]
    )
    ps = st.parents
    ho = st.vocab.id_of("Hổ")
    picked = ps.facts_about((ho,), top_k=5, min_overlap=0.6)
    # 3 câu nhưng 2 câu đầu gần trùng → chỉ giữ 1 → tối đa 2 fact phân biệt
    assert 1 <= len(picked) <= 2
    for pi in picked:
        assert ho in ps.parents[pi].emit_ids


def test_theme_neuron_ranks_subject_fact_over_mention():
    """
    v4 P1: nơ-ron chủ đề — fact mà thực thể đứng ĐẦU câu (câu VỀ nó) thắng fact chỉ NHẮC nó
    giữa câu, dù coverage IDF bằng nhau (chính pattern drift Napoléon/Trái Đất đo trên big_a2).
    """
    st = _qa_state(
        [
            "Cuộc chiến tranh chống Napoléon gây nhiều thiệt hại lớn .",  # NHẮC (giữa câu)
            "Napoléon trốn khỏi đảo Elba về Paris .",                     # VỀ (đầu câu)
        ]
    )
    ps = st.parents
    nap = st.vocab.id_of("Napoléon")

    neuron = ps.theme_neuron(nap)
    assert neuron is not None and len(neuron.facts) == 2
    top_pi, top_w = neuron.facts[0]
    assert nap == ps.parents[top_pi].emit_ids[0]  # fact xếp đầu: thực thể mở câu
    assert top_w == 1.0

    picked = ps.facts_about((nap,), top_k=1, theme_weight=1.0)
    assert "Elba" in " ".join(st.vocab.token_of(t) for t in ps.parents[picked[0]].emit_ids)


def test_autoregressive_reconstructs_answer_token_by_token():
    """
    A3: vòng sinh TỪNG TOKEN (không dội verbatim) — dựng lại đáp bám agenda, phủ hết ý.
    Dùng ingest để có W_seq (a_micro). Chuỗi W_seq rõ → tái dựng đúng nội dung.
    """
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.hca.generate_hebbian import generate_autoregressive
    from nn_aivie.training.ingest import ingest_qa

    st = GraphState()
    st.parents = ParentStore()
    ingest_qa(st, [("Ai chữa bệnh", "Bác sĩ chữa bệnh .")])
    ids = generate_autoregressive(st, "Hỏi: Ai chữa bệnh Đáp:", max_len=16)
    ans = decode_surface(st, ids).split("Đáp:")[-1]
    gen = set(st.vocab.encode_tokens(ans.split()))
    for w in ("Bác", "sĩ", "chữa", "bệnh"):
        assert st.vocab.id_of(w) in gen  # phủ đủ agenda (nội dung đáp)


def test_autoregressive_index_backbone_orders_words():
    """
    A3 phase-2: index backbone (docx §1) giữ ĐÚNG trật tự từ — token index kế tiếp thắng khi
    đạt được, thay cạnh vị trí R-slot (v3 không có). Tái dựng đúng thứ tự đáp đã học.
    """
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.hca.generate_hebbian import generate_autoregressive
    from nn_aivie.training.ingest import ingest_qa

    st = GraphState()
    st.parents = ParentStore()
    ingest_qa(st, [("Thủ đô Việt Nam là gì", "Thủ đô Việt Nam là Hà Nội .")])
    ids = generate_autoregressive(st, "Hỏi: Thủ đô Việt Nam là gì Đáp:", max_len=20)
    ans = decode_surface(st, ids).split("Đáp:")[-1].strip()
    # đúng trật tự (không đảo thành "Việt Nam là Thủ đô …")
    assert ans.startswith("Thủ đô Việt Nam")


def test_autoregressive_stops_after_agenda_covered():
    """A3 dừng ngay khi phủ đủ agenda — không sinh rác chạy tới max_len."""
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.hca.generate_hebbian import generate_autoregressive
    from nn_aivie.training.ingest import ingest_qa

    st = GraphState()
    st.parents = ParentStore()
    ingest_qa(st, [("Ai chữa bệnh", "Bác sĩ chữa bệnh .")])
    prompt_len = len(st.vocab.encode_tokens("Hỏi: Ai chữa bệnh Đáp:".split()))
    ids = generate_autoregressive(st, "Hỏi: Ai chữa bệnh Đáp:", max_len=40)
    # đáp ~4 token; không được phình gần max_len (đã dừng theo agenda)
    assert (len(ids) - prompt_len) <= 10


def test_autoregressive_empty_agenda_does_not_crash():
    """Không parent/agenda → trả về an toàn (không lỗi), không bịa dài."""
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.hca.generate_hebbian import generate_autoregressive

    st = GraphState()
    st.parents = ParentStore()
    ids = generate_autoregressive(st, "Hỏi: Xyz Abc Đáp:", max_len=12)
    assert isinstance(ids, list)


def test_no_parent_falls_back_to_wseq_chain():
    """Không nơ-ron cha → autoregressive W_seq (a→b→c→EOS)."""
    st = GraphState()
    v = st.vocab
    ids = {t: v.id_of(t) for t in ("Đáp:", "a", "b", "c")}
    eos = v.id_of(SPECIAL_EOS)
    chain = [ids["Đáp:"], ids["a"], ids["b"], ids["c"], eos]
    for p, q in zip(chain, chain[1:]):
        st.adjacency.set(p, q, 1.0, EdgeKind.GRAMMAR)
        st.adjacency.set(p, q, 1.0, EdgeKind.SEQUENTIAL)
    out = generate(st, "Đáp:", max_len=6)
    assert decode_surface(st, out) == "Đáp: a b c"


def test_ar_orders_agenda_via_wseq():
    """HỢP NHẤT P1: A3 sắp agenda token của cha theo trật tự W_seq (thay chế độ Markov cũ
    `hybrid_markov_answer` đã bỏ) — dấu câu cuối do EOS đóng, kiểm tra trật tự nội dung."""
    st = _qa_state(["Hỏi: q Đáp: a b c ."])
    v = st.vocab
    chain = [v.id_of("Đáp:"), v.id_of("a"), v.id_of("b"), v.id_of("c"), v.id_of(".")]
    for p, nxt in zip(chain, chain[1:]):
        st.adjacency.set(p, nxt, 1.0, EdgeKind.SEQUENTIAL)
    ids = generate(st, "Hỏi: q Đáp:", max_len=8)
    assert decode_surface(st, ids).split("Đáp:")[-1].strip().split()[:3] == ["a", "b", "c"]


def test_empty_prompt_returns_empty():
    st = _qa_state(["Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh ."])
    assert generate(st, "   ", max_len=4) == []


def test_unrelated_question_does_not_crash():
    st = _qa_state(["Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh ."])
    out = generate(st, "Hỏi: xyz abc Đáp:", max_len=6)
    assert isinstance(out, list)


def test_knowledge_extra_eligible_on_entity_presence():
    """TK_025: bồi fact khi query có THỰC THỂ phân biệt (không còn cổng WH-filler)."""
    from nn_aivie.core.parent_store import _content_ids
    from nn_aivie.hca.config import HCAConfig
    from nn_aivie.hca.generate_hebbian import _split_frame

    st = _qa_state(
        [
            "Hỏi: Mùa hè có gì Đáp: Mùa hè nóng .",
            "Hỏi: Quyền có gì Đáp: Quyền là quyền .",
            "Hỏi: Cân có gì Đáp: Cân là cân .",
        ]
    )
    ps = st.parents
    cfg = HCAConfig()
    q_ids, _ = _split_frame(
        st.vocab.encode_tokens("Hỏi: Mùa hè có gì Đáp:".split()), st.vocab
    )
    q_content = tuple(_content_ids(q_ids, st.vocab, skip=set()))
    assert _knowledge_extra_facts_eligible(ps, q_content, cfg)


def test_knowledge_extra_fact_plan_scores_and_picks():
    """TK_023: plan trả score + picked/skip khớp đường bồi fact."""
    from nn_aivie.hca.config import HCAConfig

    st = _qa_state(
        [
            "Hỏi: Mùa hè có gì Đáp: Mùa hè nóng .",
            "Hỏi: Mùa hè có bão không Đáp: Mùa hè có bão .",
            "Hỏi: Mùa hè có nắng Đáp: Mùa hè có nắng .",
            "Hỏi: Quyền có gì Đáp: Quyền là quyền .",
            "Hỏi: Cân có gì Đáp: Cân là cân .",
        ]
    )
    st.parents.decompose(st.vocab, min_frequency=2, max_len=4)
    cfg = HCAConfig(
        knowledge_extra_facts=2,
        wh_min_filler_diversity=5,
        knowledge_extra_wh_min_filler=3,
        knowledge_extra_require_cap_primary=False,  # test CƠ CHẾ plan; primary "hè" viết thường
    )
    prompt = "Hỏi: Mùa hè có gì Đáp:"
    prompt_ids = list(st.vocab.encode_tokens(prompt.split()))
    q_ids, _ = _split_frame(prompt_ids, st.vocab)
    plan_holder: dict = {}
    out = generate(st, prompt, max_len=48, config=cfg, fact_plan_out=plan_holder)
    plan = plan_holder
    assert plan is not None
    assert plan["eligible"]
    assert "Mùa" in plan["entity"] and "hè" in plan["entity"]
    assert plan["facts"]
    assert any(f["picked"] for f in plan["facts"])
    for f in plan["facts"]:
        assert f["score"] >= 0
        assert f["rank"] >= 1
        if f["picked"]:
            assert f["skip"] is None


def test_knowledge_extra_facts_cap_primary_gate():
    """Nhánh C: gate bồi theo cap primary. Cùng input primary VIẾT THƯỜNG ("hè"):
    gate ON → chặn bồi (chỉ crisp); gate OFF → cơ chế bồi chạy như cũ. Chống bồi-drift trên
    role/synonym query ("Người chữa bệnh"→"bệnh" thường) mà KHÔNG đụng entity tên-riêng."""
    from nn_aivie.hca.config import HCAConfig

    st = _qa_state(
        [
            "Hỏi: Mùa hè có gì Đáp: Mùa hè nóng .",
            "Hỏi: Mùa hè có bão không Đáp: Mùa hè có bão .",
            "Hỏi: Mùa hè có nắng Đáp: Mùa hè có nắng .",
            "Hỏi: Quyền có gì Đáp: Quyền là quyền .",
            "Hỏi: Cân có gì Đáp: Cân là cân .",
        ]
    )
    st.parents.decompose(st.vocab, min_frequency=2, max_len=4)
    prompt = "Hỏi: Mùa hè có gì Đáp:"
    base = dict(knowledge_extra_facts=2, wh_min_filler_diversity=5, knowledge_extra_wh_min_filler=3)

    # gate OFF → primary thường ("hè") vẫn bồi (cơ chế nguyên vẹn)
    hold_off: dict = {}
    generate(st, prompt, max_len=48,
                    config=HCAConfig(**base, knowledge_extra_require_cap_primary=False),
                    fact_plan_out=hold_off)
    assert hold_off.get("primary") == "hè"  # primary danh-từ-thường
    assert hold_off.get("facts"), "gate OFF: danh-từ-thường vẫn bồi"

    # gate ON (default) → primary thường bị chặn → không bồi (plan_out rỗng)
    hold_on: dict = {}
    generate(st, prompt, max_len=48, config=HCAConfig(**base), fact_plan_out=hold_on)
    assert not hold_on, "gate ON: danh-từ-thường KHÔNG bồi"


def test_knowledge_extra_facts_no_double_period():
    """Fact bồi có đầu `.` sau khi A3 đã đóng câu → không `. .`."""
    from nn_aivie.hca.config import HCAConfig

    st = _qa_state(
        [
            "Hỏi: Việt Nam là gì Đáp: Thủ đô Việt Nam là Hà Nội .",
            "Hỏi: Việt Nam là gì Đáp: . Miền Nam Việt Nam là một khái niệm .",
            "Hỏi: Việt Nam là gì Đáp: Từ năm 1945 khái niệm miền Nam được dùng .",
        ]
    )
    st.parents.decompose(st.vocab, min_frequency=2, max_len=6)
    cfg = HCAConfig(
        knowledge_extra_facts=1,
        knowledge_extra_require_cap_primary=False,
    )
    out = decode_surface(
        st,
        generate(
            st, "Hỏi: Việt Nam là gì Đáp:",
            max_len=64, config=cfg, multi=True,
        ),
    )
    assert ". ." not in out, out
    assert "Hà Nội" in out


def test_open_continuation_agenda_skips_prompt_prefix():
    """Open continuation: agenda theo index source_ids — không lặp lại prefix đã gõ."""
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    sent = "Người Minh dùng hỏa tiễn bắn voi đã rút lui ."
    ids = list(st.vocab.encode_tokens(sent.split()))
    ps.add_example(ids, st.vocab, dap_id=dap)
    _g1_and_wseq(st, ids, eta=1.0)
    st.parents = ps

    prompt = "Người Minh dùng hỏa tiễn bắn voi"
    out = decode_surface(st, generate(st, prompt, max_len=48))
    phrase = "Người Minh dùng hỏa tiễn bắn voi"
    assert out.count(phrase) <= 1, out
    assert "đã" in out or "rút" in out, out


def test_open_continuation_emits_trailing_period_not_echo():
    """Prompt = câu thiếu dấu chấm → agenda chỉ `.`, không fallback lặp cả câu."""
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    sent = "Người Minh dùng hỏa tiễn bắn voi ."
    ids = list(st.vocab.encode_tokens(sent.split()))
    ps.add_example(ids, st.vocab, dap_id=dap)
    _g1_and_wseq(st, ids, eta=1.0)
    st.parents = ps

    prompt = "Người Minh dùng hỏa tiễn bắn voi"
    out = decode_surface(st, generate(st, prompt, max_len=24))
    phrase = "Người Minh dùng hỏa tiễn bắn voi"
    assert out.count(phrase) <= 1, out
    assert out.rstrip().endswith("."), out


def test_ar_candidate_scoring_matches_dense():
    """Sparse candidate-set AR phải cho cùng output với dense legacy."""
    from nn_aivie.hca.config import HCAConfig
    from nn_aivie.hca.generate_hebbian import generate_autoregressive

    st = _qa_state(
        [
            "Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh .",
            "Hỏi: Thủ đô Việt Nam là gì Đáp: Thủ đô Việt Nam là Hà Nội .",
        ]
    )
    prompt = "Hỏi: Ai chữa bệnh Đáp:"
    sparse_cfg = HCAConfig(ar_candidate_scoring=True)
    dense_cfg = HCAConfig(ar_candidate_scoring=False)
    sparse_ids = generate_autoregressive(st, prompt, max_len=16, config=sparse_cfg, seed=0)
    dense_ids = generate_autoregressive(st, prompt, max_len=16, config=dense_cfg, seed=0)
    assert sparse_ids == dense_ids


def test_spreading_csr_matches_dict():
    """CSR matvec spreading phải khớp BFS dict (cùng semantics)."""
    import numpy as np

    from nn_aivie.core.edge_kinds import EdgeKind
    from nn_aivie.hca.config import HCAConfig
    from nn_aivie.hca.readout import (
        _spreading_ctx_dict,
        spreading_activation_read,
        spreading_activation_read_sparse,
    )
    from nn_aivie.hca.spreading_csr import spreading_ctx_csr

    st = _qa_state(["Hỏi: Ai chữa bệnh Đáp: Bác sĩ chữa bệnh ."])
    for a, b in zip(
        st.vocab.encode_tokens("Bác sĩ chữa bệnh".split()),
        st.vocab.encode_tokens("sĩ chữa bệnh .".split()),
    ):
        st.adjacency.add(a, b, 1.0, EdgeKind.CO_ACTIVATION)
    cfg = HCAConfig(spread_use_csr=True)
    cfg_dict = HCAConfig(spread_use_csr=False)
    vocab_size = len(st.vocab)
    seeds = [(st.vocab.id_of("Bác"), 1.0)]
    w0 = st.vocab.id_of("sĩ")
    ctx_csr = spreading_ctx_csr(seeds, st.adjacency, cfg, vocab_size)
    ctx_dict = _spreading_ctx_dict(seeds, st.adjacency, cfg_dict, vocab_size)
    for v in set(ctx_csr) | set(ctx_dict):
        assert abs(ctx_csr.get(v, 0.0) - ctx_dict.get(v, 0.0)) < 1e-9
    dense_csr = spreading_activation_read(
        w0, seeds, st.adjacency, cfg, vocab_size,
    )
    dense_dict = spreading_activation_read(
        w0, seeds, st.adjacency, cfg_dict, vocab_size,
    )
    nz = np.union1d(
        np.flatnonzero(dense_csr > 0),
        np.flatnonzero(dense_dict > 0),
    )
    np.testing.assert_allclose(dense_csr[nz], dense_dict[nz], rtol=1e-9, atol=1e-9)
    sparse_csr = spreading_activation_read_sparse(
        w0, seeds, st.adjacency, cfg, vocab_size,
    )
    sparse_dict = spreading_activation_read_sparse(
        w0, seeds, st.adjacency, cfg_dict, vocab_size,
    )
    assert sparse_csr.keys() == sparse_dict.keys()
    for k in sparse_csr:
        assert abs(sparse_csr[k] - sparse_dict[k]) < 1e-9
