"""build_parents — nơ-ron cha từ corpus thật."""

from __future__ import annotations

from pathlib import Path

import pytest

from nn_aivie.core.state import GraphState

# `checkpoints/*.pkl` nằm trong `.gitignore` ⇒ fixture này KHÔNG BAO GIỜ có trong
# một bản clone sạch, nên test đỏ trên mọi máy chưa tự dựng checkpoint. Nó đã đỏ
# suốt cả phiên 2026-08-17 và làm nhiễu mọi lần chạy: mỗi lần phải đọc lại để
# xác nhận «vẫn là lỗi môi trường có sẵn».
#
# Bỏ qua có LÝ DO, đúng quy ước sẵn có của dự án (`test_copy_number.py`,
# `test_frame_bracket_gate.py`). Có checkpoint thì test chạy y như cũ — không hạ
# thấp phạm vi kiểm, chỉ thôi báo đỏ về một tệp không thể có mặt.
_CK = Path("checkpoints/grade_v3.pkl")


@pytest.mark.skipif(not _CK.exists(), reason=f"thiếu {_CK} (checkpoints/ bị gitignore)")
def test_build_parents_includes_qa_val_shard():
    """qa_corpus val (Ai chữa bệnh, …) phải thành cha QA — không chỉ shard train."""
    from scripts.train_v3 import REAL_KNOWLEDGE, build_parents

    st = GraphState.load(str(_CK))
    before = len(st.parents) if st.parents else 0
    ps = build_parents(st, [3], ["qa_corpus"])
    found = False
    for rec in ps.parents:
        if not rec.is_qa:
            continue
        match = " ".join(st.vocab.token_of(t) for t in rec.match_ids)
        if "chữa" in match and "bệnh" in match:
            found = True
            ans = " ".join(st.vocab.token_of(t) for t in rec.answer_ids)
            assert "Bác" in ans
            break
    assert found, "expected QA parent from qa val shard (Ai chữa bệnh)"
    assert len(ps) > 7  # train 58 + val 7 grade-3 qa rows
