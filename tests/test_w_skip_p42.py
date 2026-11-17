"""P42/P43 — EOS end_after + content gate + hist sau biên."""

from __future__ import annotations

from nn_aivie.core.parent_store import ParentRecord, ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import SPECIAL_EOS, Vocabulary
from nn_aivie.hca.w_skip import (
    clear_w_skip_cache,
    content_channel_gate,
    eos_from_end_after,
    hist_after_last_boundary,
)


def test_content_channel_gate_floor_and_agenda():
    v = Vocabulary()
    for w in ("a", "b", "c"):
        v.id_of(w)
    g = content_channel_gate(
        len(v), agenda_ids=[1], entity_ids=(0,), prompt_ids=[2],
        floor=0.2, vocab=v,
    )
    assert g[0] == 1.0 and g[1] == 1.0 and g[2] == 1.0


def test_hist_after_last_boundary_cuts_old_sentence():
    v = Vocabulary()
    for w in ("là", "Long", ".", "Đây"):
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in ("là", "Long", ".", "Đây")}
    h = hist_after_last_boundary(
        [id_["là"], id_["Long"], id_["."]], v, pre_boundary_k=1,
    )
    assert h == [id_["Long"]]
    h2 = hist_after_last_boundary(
        [id_["là"], id_["Long"], id_["."], id_["Đây"]], v, pre_boundary_k=2,
    )
    assert h2 == [id_["Đây"]]


def test_hist_after_question_skips_dap_frame():
    """§12.149 — sau `? Đáp:` không vote từ một mình Đáp:."""
    from nn_aivie.core.tokenizer import DAP_LABEL

    v = Vocabulary()
    for w in ("Nhà", "Lý", "đâu", "?", DAP_LABEL):
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in ("Nhà", "Lý", "đâu", "?", DAP_LABEL)}
    h = hist_after_last_boundary(
        [id_["Nhà"], id_["Lý"], id_["đâu"], id_["?"], id_[DAP_LABEL]],
        v,
        pre_boundary_k=2,
    )
    assert id_[DAP_LABEL] not in h
    assert id_["Nhà"] in h and id_["đâu"] in h


def test_eos_from_end_after_positive_when_mass():
    # §13.55 — fixture phải ĐƯỢC TRAIN. `end_after` nay nhận biên câu bằng EOS-lift đọc từ
    # `W_seq`; dựng `ParentRecord` trần thì không có cạnh nào ⇒ tín hiệu rỗng ⇒ test đo một
    # hệ không tồn tại lúc chạy. Cùng lỗi fixture đã sửa ở `test_sentence_parents` (§13.54).
    from nn_aivie.core.parent_store import ParentStore as _PS
    from nn_aivie.training.ingest import ingest_statements

    st = GraphState()
    st.parents = _PS()
    ingest_statements(st, ["A B ."], split_punct=True)
    v = st.vocab
    eos = v.id_of(SPECIAL_EOS)
    id_ = {w: v.id_of(w) for w in ("A", "B", ".")}
    clear_w_skip_cache()
    s = eos_from_end_after(
        st, [id_["A"], id_["B"]], eos_id=eos, n_vocab=len(v), beta=1.2, ref_scale=1.0,
    )
    assert s > 0.0
    s0 = eos_from_end_after(
        st, [id_["A"]], eos_id=eos, n_vocab=len(v), beta=1.2, ref_scale=1.0,
    )
    assert s >= s0
