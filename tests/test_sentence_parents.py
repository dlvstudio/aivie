"""TK_045 P0.1 sentence parent split + P0.3 agenda handoff."""

from __future__ import annotations

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.sentence_parents import (
    migrate_parents_to_sentences,
    split_emit_to_sentence_seqs,
)
from nn_aivie.hca.sent_intra import (
    agenda_handoff_next,
    clear_sent_intra_cache,
    ensure_sent_intra,
)
from nn_aivie.hca.generate_hebbian import _wseq_fallback


def test_split_emit_two_sentences():
    st = GraphState()
    ids = st.vocab.encode_tokens(
        ["Ngô", "Quyền", "lập", "nước", ".", "Chiến", "thắng", "Bạch", "Đằng", "."]
    )
    # §13.54 — `st` ở đây dựng TAY, chưa train nên `W_seq` rỗng ⇒ không có tín hiệu
    # EOS-lift. Đó đúng là tình huống `ingest` gặp (vòng luẩn quẩn: tách câu trong lúc
    # đang dựng W_seq), nên dùng MỒI NGUỘI và nói ra, thay vì để hàm đoán ngầm.
    sents = split_emit_to_sentence_seqs(ids, st.vocab, min_content=2, bootstrap=True)
    assert len(sents) == 2
    assert st.vocab.token_of(sents[0][0]) == "Ngô"
    assert st.vocab.token_of(sents[1][0]) == "Chiến"


def test_migrate_builds_episodes():
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    long_ids = st.vocab.encode_tokens(
        [
            "Ngô",
            "Quyền",
            "lập",
            "nhà",
            "Ngô",
            ".",
            "Ông",
            "thắng",
            "ở",
            "Bạch",
            "Đằng",
            ".",
            "Thời",
            "kỳ",
            "độc",
            "lập",
            "bắt",
            "đầu",
            ".",
        ]
    )
    # force mega
    assert len(long_ids) > 10
    ps.add_example(long_ids, st.vocab, dap_id=dap)
    st.parents = ps
    clear_sent_intra_cache()
    stats = migrate_parents_to_sentences(
        st, bootstrap=True, only_mega_gt=5, min_content=2, max_tokens=64, max_new_parents=100
    )
    assert stats.new_parents >= 3
    assert stats.episodes >= 1
    assert len(st.parents.episodes) >= 1
    views = ensure_sent_intra(st, force=True)
    assert views.n_agenda_edges >= 1


def test_agenda_handoff_picks_next_sentence_start():
    # §13.54 — fixture phải ĐƯỢC TRAIN. `agenda_handoff_next` là đường SINH, và biên đơn vị
    # nay đọc từ EOS-lift trên `W_seq`; fixture dựng tay bằng `ps.add_example` không tạo cạnh
    # nào nên tín hiệu rỗng và test đo một hệ không tồn tại. Đi qua `ingest_statements` để
    # có `W_seq` thật, đúng như lúc chạy.
    from nn_aivie.training.ingest import ingest_statements

    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, ["Em bé Lan .", "Lan đi học ."], split_punct=True)
    ps = st.parents
    from nn_aivie.core.parent_store import AbstractNeuron

    # PHẢI qua `_register_episode`: nó vừa append vừa dựng chỉ mục ngược `_parent_episodes`.
    # `episodes.append(...)` trần ⇒ `episodes_of(pi)` trả () ⇒ handoff câm (test đỏ từ 28f371a).
    ps._register_episode(AbstractNeuron(members=((0, 1.0), (1, 1.0))))
    st.parents = ps
    clear_sent_intra_cache()
    cfg = HCAConfig(
        ar_sent_agenda_handoff=True,
        ar_sent_agenda_handoff_ratio=0.0,
        ar_sent_agenda_handoff_min=0.0,
    )
    lan = st.vocab.id_of("Lan")
    # đứng ở cuối câu 1 (trước dấu) — dùng token nội dung cuối
    # emit s1 ends with . ; last content is Lan — but Lan also starts s2.
    # Use "bé" as w0 with agenda from episode: last content of s1 before punct.
    # First/last content: Em...Lan → agenda Lan→Lan (same) or nước-style.
    # Rebuild: s1 last content Lan, s2 first Lan — still ok if mass > 0
    hist = st.vocab.encode_tokens(["Em", "bé", "Lan"])
    nxt = agenda_handoff_next(st, list(hist), config=cfg, eos_logit=10.0)
    # may be Lan (start of s2) 
    assert nxt is not None
    assert st.vocab.is_surface(nxt)


def test_episode_handoff_continues_to_next_member():
    """Khớp đuôi câu 1 → đầu câu 2 khi câu sau CÓ nhắc lại thực thể (overlap ≥ 1).

    §12.21 — bản gốc (28f371a) đòi câu sau là `Ông thắng ở Bạch Đằng` (mở đầu bằng ĐẠI TỪ,
    overlap=0) và **đỏ từ ngày commit**. Đo trên mini-có-episode: nới ngưỡng cho ca overlap=0
    đi qua thì đúng 1/8 ca đổi và đổi theo hướng XẤU — sinh ra `Trận Bạch Đằng năm 1288.` +
    `Ông được nhân dân tôn kính…`, `Ông` chẳng trỏ ai. ⇒ kỳ vọng cũ SAI, luật `overlap ≥ 1`
    là chủ ý: chặn đại-từ-lơ-lửng. Test nay khoá đúng hợp đồng đó (xem test kế cho vế âm).
    """
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    s1 = st.vocab.encode_tokens(["Ngô", "Quyền", "lập", "nước", "."])
    s2 = st.vocab.encode_tokens(["Ngô", "Quyền", "thắng", "ở", "Bạch", "Đằng", "."])
    ps.add_example(s1, st.vocab, dap_id=dap)
    ps.add_example(s2, st.vocab, dap_id=dap)
    from nn_aivie.core.parent_store import AbstractNeuron

    # PHẢI qua `_register_episode`: nó vừa append vừa dựng chỉ mục ngược `_parent_episodes`.
    # `episodes.append(...)` trần ⇒ `episodes_of(pi)` trả () ⇒ handoff câm (test đỏ từ 28f371a).
    ps._register_episode(AbstractNeuron(members=((0, 1.0), (1, 1.0))))
    st.parents = ps
    clear_sent_intra_cache()
    cfg = HCAConfig(ar_sent_agenda_handoff=True, ar_no_agenda=False)
    hist = st.vocab.encode_tokens(["Ngô", "Quyền", "lập", "nước"])
    nxt = agenda_handoff_next(st, list(hist), config=cfg, eos_logit=99.0)
    assert nxt is not None
    assert st.vocab.token_of(nxt).lstrip("Ġ") == "Ngô"


def test_ar_no_agenda_khong_tat_handoff_con_flag_rieng_thi_co():
    """§12.21 — công tắc thật của handoff là `ar_sent_agenda_handoff`, KHÔNG phải `ar_no_agenda`.

    Nửa sau của test gốc khẳng định `ar_no_agenda=True` (+ `eos_logit` thấp) làm handoff im —
    trái chính chú thích trong `sent_intra.agenda_handoff_next`: *"ar_no_agenda chỉ tắt MZ/fact
    plan — vẫn cho handoff matrix"* và *"không so mass agenda với eos_logit tuyệt đối"*. Khoá
    lại hợp đồng đúng để lần sau khỏi ai sửa nhầm hàm cho vừa test.
    """
    from nn_aivie.core.parent_store import AbstractNeuron

    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    # câu 1 đủ dài để matched ≥ 3 (điều kiện bên trong _episode_next_span)
    ps.add_example(st.vocab.encode_tokens(["Em", "bé", "Lan", "đi", "học", "."]), st.vocab, dap_id=dap)
    ps.add_example(st.vocab.encode_tokens(["Lan", "đi", "học", "ở", "trường", "."]), st.vocab, dap_id=dap)
    ps._register_episode(AbstractNeuron(members=((0, 1.0), (1, 1.0))))
    st.parents = ps
    clear_sent_intra_cache()
    # hist = đuôi câu 1 (3 token cuối) → matched = 3 ≥ 3
    hist = list(st.vocab.encode_tokens(["Em", "bé", "Lan", "đi", "học"]))

    on = HCAConfig(ar_no_agenda=True, ar_sent_agenda_handoff=True)
    assert agenda_handoff_next(st, hist, config=on, eos_logit=1.0) is not None
    off = HCAConfig(ar_no_agenda=False, ar_sent_agenda_handoff=False)
    assert agenda_handoff_next(st, hist, config=off, eos_logit=1.0) is None


def test_sharp_entropy_gate_argmax_on_flat():
    import numpy as np

    from nn_aivie.hca.generate_hebbian import _sample, logits_topk_entropy

    flat = np.ones(32, dtype=np.float64)
    assert logits_topk_entropy(flat, 12) > 2.0
    cfg = HCAConfig(
        hybrid_temperature=0.9,
        hybrid_top_k=12,
        ar_sharp_gate=True,
        ar_sharp_entropy_max=1.5,
        ar_sharp_fallback_argmax=True,
    )
    rng = np.random.default_rng(0)
    assert _sample(flat, cfg, rng) == 0
    peaked = np.full(32, -10.0)
    peaked[7] = 5.0
    assert logits_topk_entropy(peaked, 12) < 1.0


def test_wseq_fallback_stops_clean_on_no_agenda():
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    s1 = st.vocab.encode_tokens(
        ["Ngô", "Quyền", "là", "vua", "sáng", "lập", "."]
    )
    s2 = st.vocab.encode_tokens(["Nghĩa", "2", "Giải", "thích", "."])
    ps.add_example(s1, st.vocab, dap_id=dap)
    ps.add_example(s2, st.vocab, dap_id=dap)
    from nn_aivie.core.parent_store import AbstractNeuron

    # PHẢI qua `_register_episode`: nó vừa append vừa dựng chỉ mục ngược `_parent_episodes`.
    # `episodes.append(...)` trần ⇒ `episodes_of(pi)` trả () ⇒ handoff câm (test đỏ từ 28f371a).
    ps._register_episode(AbstractNeuron(members=((0, 1.0), (1, 1.0))))
    st.parents = ps
    clear_sent_intra_cache()
    cfg = HCAConfig(
        ar_matrix_core=True,
        ar_no_agenda=True,
        ar_sent_intra=True,
        ar_sent_agenda_handoff=True,
        ar_matrix_qk_weight=0.0,
        ar_matrix_stack_weight=0.0,
        ar_matrix_know_weight=0.0,
        ar_matrix_readout_weight=0.0,
        ar_matrix_r2l_ffn_weight=0.0,
        ar_matrix_h_weight=0.0,
        hybrid_temperature=0.0,
    )
    from nn_aivie.core.vocab import SPECIAL_EOS
    import numpy as np

    eos = st.vocab.id_of(SPECIAL_EOS)
    rng = np.random.default_rng(0)
    gen = st.vocab.encode_tokens(["Ngô", "Quyền", "là", "vua", "sáng", "lập"])
    out = _wseq_fallback(st, list(gen), eos, max_steps=12, config=cfg, rng=rng)
    toks = [st.vocab.token_of(t).lstrip("Ġ") for t in out]
    assert "Nghĩa" not in toks
    assert "Giải" not in toks


def test_episode_handoff_chan_dai_tu_lo_lung():
    """§12.21 vế ÂM — câu sau mở đầu bằng đại từ, KHÔNG nhắc lại thực thể nào (overlap=0)
    ⇒ handoff phải IM. Đo mini-có-episode: cho qua thì sinh `Trận Bạch Đằng năm 1288. Ông
    được nhân dân tôn kính…` — `Ông` không có tiền ngữ. Đây là chủ ý, không phải thiếu sót."""
    from nn_aivie.core.parent_store import AbstractNeuron

    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    ps.add_example(st.vocab.encode_tokens(["Ngô", "Quyền", "lập", "nước", "."]),
                   st.vocab, dap_id=dap)
    ps.add_example(st.vocab.encode_tokens(["Ông", "thắng", "ở", "Bạch", "Đằng", "."]),
                   st.vocab, dap_id=dap)
    ps._register_episode(AbstractNeuron(members=((0, 1.0), (1, 1.0))))
    st.parents = ps
    clear_sent_intra_cache()
    cfg = HCAConfig(ar_sent_agenda_handoff=True, ar_no_agenda=False)
    hist = st.vocab.encode_tokens(["Ngô", "Quyền", "lập", "nước"])
    assert agenda_handoff_next(st, list(hist), config=cfg, eos_logit=99.0) is None
