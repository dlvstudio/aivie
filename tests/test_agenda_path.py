"""TK_045 §13.57 — phủ ĐƯỜNG AGENDA, chỗ suite 357 test không chạm tới.

## Vì sao file này tồn tại

§13.56 thay danh sách dấu câu chép tay bằng tín hiệu học ở đường agenda. Trong lúc vá, **hai
lỗi im lặng** xảy ra và chỉ một cái bị suite bắt:

1. đổi tên `_key_ok` → `_key_ok_id` bỏ sót bốn chỗ gọi ⇒ `NameError`. Suite bắt được —
   **nhưng chỉ vì tên đổi**. Giữ nguyên tên thì hành vi đã đổi mà 357 test vẫn xanh.
2. nối `_ends_u` xong mà **chưa caller nào truyền `bound`** ⇒ mốc kết đơn vị **biến mất** khỏi
   đường agenda, và suite **vẫn xanh 357**. Chỉ phát hiện nhờ soát tay danh sách caller.

Lỗi (2) là bằng chứng cứng: suite không phủ đường này **ở mức hành vi**. Ba test dưới khoá
đúng ba thứ đã suýt trôi.
"""

from __future__ import annotations

import re
from pathlib import Path

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.hca.generate_hebbian import (
    _agenda_from_parent_index,
    _is_agenda_content_token,
)
from nn_aivie.hca.sent_intra import _bound_of
from nn_aivie.training.ingest import ingest_statements

_QA = "Hỏi: Ai lập nước ? Đáp: Ngô Quyền lập nước . Ông thắng Bạch Đằng ."


def _trained_qa() -> GraphState:
    """Fixture PHẢI qua `ingest_statements`.

    Dựng `ParentRecord` bằng tay thì không có cạnh nào ⇒ `W_seq` rỗng ⇒ mọi thứ đọc-từ-dữ-liệu
    đều câm, và test sẽ đo một hệ **không tồn tại lúc chạy**. Đúng lỗi hai fixture khác đã mắc
    (§13.54, §13.55) và không ai thấy, vì hồi đó danh sách dấu câu cứng vẫn chạy trên state rỗng.
    """
    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, [_QA], split_punct=True)
    return st


def test_agenda_khong_ro_token_cua_cau_thu_hai_khi_co_bound():
    """Bất biến hành vi: có tín hiệu biên ⇒ agenda dừng trong câu ĐẦU của đáp.

    Đây chính là thứ lỗi im lặng (2) đã tắt: `_ends_u` luôn False ⇒ không cắt ở mốc kết đơn
    vị ⇒ agenda kéo cả câu hai sang. Suite cũ không có test nào thấy.
    """
    st = _trained_qa()
    v = st.vocab
    assert st.parents.parents[0].is_qa, "fixture không tạo được cha QA — test vô nghĩa"

    emit = list(v.encode_tokens(["Hỏi:", "Ai", "lập", "nước", "?", "Đáp:"]))
    ag, _ = _agenda_from_parent_index(
        st.parents, 0, emit, v, bound=_bound_of(st))
    got = {v.token_of(int(t)) for t in ag}

    cau_hai = {"Ông", "thắng", "Bạch", "Đằng"}
    assert not (got & cau_hai), (
        f"agenda rò token của câu THỨ HAI: {sorted(got & cau_hai)} — "
        "mốc kết đơn vị không còn tác dụng"
    )


def test_khong_co_bound_thi_HONG_ON_AO_chu_khong_am_tham_dung_dau_cau():
    """Thiết kế cố ý: `bound=None` ⇒ KHÔNG quay về so ký tự `"."`.

    Test này khoá **hành vi hỏng**, không phải hành vi đúng — để không ai "tiện tay" thêm
    fallback im lặng vào `_ends_u`. Có fallback thì hai nhánh sẽ giống nhau và test đỏ.
    """
    st = _trained_qa()
    v = st.vocab
    emit = list(v.encode_tokens(["Hỏi:", "Ai", "lập", "nước", "?", "Đáp:"]))

    co = {v.token_of(int(t)) for t in
          _agenda_from_parent_index(st.parents, 0, emit, v, bound=_bound_of(st))[0]}
    khong = {v.token_of(int(t)) for t in
             _agenda_from_parent_index(st.parents, 0, emit, v, bound=None)[0]}

    assert co != khong, (
        "bỏ `bound` mà agenda KHÔNG đổi ⇒ hoặc tín hiệu biên vô tác dụng, hoặc ai đó đã "
        "thêm fallback về danh sách dấu câu"
    )


def test_loc_MOI_dau_cau_ke_ca_ngoai_danh_sach_cu():
    """Bản cũ liệt kê tay 11 dấu; bản mới xét LỚP KÝ TỰ nên phủ cả cái không ai nghĩ tới."""
    st = _trained_qa()
    v = st.vocab
    # không có dấu nào dưới đây nằm trong danh sách cũ
    # (".", ",", "!", "?", ":", ";", "*", "(", ")", "…", "。")
    for ch in ("—", "«", "»", "–", "·", "‽", "§"):
        assert not _is_agenda_content_token(v, v.id_of(ch)), f"{ch!r} lọt qua bộ lọc nội dung"
    for w in ("Quyền", "nước", "Bạch"):
        assert _is_agenda_content_token(v, v.id_of(w)), f"{w!r} bị loại nhầm"


def test_moi_duong_goi_agenda_deu_truyen_bound():
    """Bất biến ĐẤU DÂY — test duy nhất bắt được lỗi im lặng (2).

    Lỗi đó không đổi API, không ném lỗi, không đổi kết quả của bất kỳ test nào đang có: nó
    chỉ **tắt** một tín hiệu. Kiểu lỗi này chỉ chặn được bằng cách khoá chính chỗ đấu dây.
    """
    src = (Path(__file__).resolve().parent.parent
           / "src/nn_aivie/hca/generate_hebbian.py").read_text(encoding="utf-8")
    names = ("_agenda_from_parent_index", "_agenda_ids_from_parent_index",
             "_pick_parent_agenda")
    thieu: list[str] = []
    for name in names:
        for m in re.finditer(rf"\b{name}\(", src):
            if src[:m.start()].rstrip().endswith("def"):
                continue                      # định nghĩa, không phải lời gọi
            # cắt tới dấu đóng ngoặc cân bằng
            i, depth = m.end() - 1, 0
            while i < len(src):
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            call = src[m.start(): i + 1]
            if "bound=" not in call:
                thieu.append(call.splitlines()[0][:80])
    assert not thieu, (
        "đường gọi agenda KHÔNG truyền `bound` ⇒ mốc kết đơn vị tắt âm thầm:\n  "
        + "\n  ".join(thieu)
    )
