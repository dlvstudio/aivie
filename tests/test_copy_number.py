"""TK_045 §12.5 v2 + §12.12 — chép cụm số verbatim từ cha ĐANG KHOÁ, rồi ĐÓNG số.

Vì sao cần: mạng không có cơ quan số (TK_041 `1288`⊥`1287`) nên sinh từng chữ-số làm mất
giá trị (`1010`→`1 0 1 0`). Chép nguyên từ emit của cha khoá là đường duy nhất đúng.
§12.12 vá tiếp: chép xong mà bước kế còn đẻ chữ-số dính-liền → `1010`+`9`=`10109` (đo v12 §12.11).
"""

from __future__ import annotations

import numpy as np
import pytest

from nn_aivie.core.tokenizer import BPETokenizer
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.matrix_core import digit_run_from_locked, inner_digit_ids

_BPE = "checkpoints/bpe_v1.json"


@pytest.fixture(scope="module")
def vocab():
    import os

    if not os.path.exists(_BPE):
        pytest.skip("thiếu checkpoints/bpe_v1.json")
    return Vocabulary(BPETokenizer.load(_BPE))


class _Bi:
    def __init__(self, emits):
        self.emits = emits


def test_inner_digit_ids_loai_tru_nhanh_co_space(vocab):
    """`Ġ1` = số MỚI sau dấu cách ⇒ KHÔNG chặn; `1` trần = nối vào số đang viết ⇒ chặn."""
    ids = set(int(t) for t in inner_digit_ids(vocab))
    assert ids, "phải bắt được chữ-số trần"
    for tid in ids:
        s = vocab.token_of(tid)
        assert s.isdigit() and not s.startswith(("Ġ", "Ċ"))
    for s in ("Ġ1", "Ġ0", "Ġ9"):
        tid = vocab.encode_text(s.replace("Ġ", " "))[0]
        assert int(tid) not in ids, f"{s} là số mới, không được chặn"


def test_digit_run_chep_nguyen_cum_so(vocab):
    """emit `… Ġnăm Ġ1 0 1 0 .` + w0=`Ġnăm`, nxt=`Ġ1` → run 4 token = 1010 (dừng ở `.`)."""
    seq = np.asarray(list(vocab.encode_text("Thăng Long năm 1010.")), dtype=np.int64)
    bi = _Bi([seq])
    nam, one = int(seq[2]), int(seq[3])
    run = digit_run_from_locked(bi, vocab, 0, nam, one)
    assert [vocab.token_of(t) for t in run] == ["Ġ1", "0", "1", "0"]


def test_digit_run_bo_qua_khi_lock_pi_am(vocab):
    """Không khoá được cha (lock_pi<0) ⇒ TUYỆT ĐỐI không chép — v1 chép từ succ[w0] ra `18110`."""
    seq = np.asarray(list(vocab.encode_text("Thăng Long năm 1010.")), dtype=np.int64)
    bi = _Bi([seq])
    assert digit_run_from_locked(bi, vocab, -1, int(seq[2]), int(seq[3])) == []


def test_chu_so_khong_phai_hu_tu(vocab):
    """TK_045 §12.13 — `_is_path_light` từng coi MỌI token 1 ký tự là hư-từ ⇒ nuốt sạch chữ-số.

    Hệ quả: chuỗi content của "…năm 1288" ≡ "…năm" ⇒ path khớp lại đúng chỗ vừa sinh, trỏ về
    `Ġ1` mãi ⇒ lặp câu + salad chữ-số (đo mini). Chữ-số là CONTENT."""
    from nn_aivie.hca.matrix_core import _is_path_light

    gidf = lambda _t: 0.0  # noqa: E731 — ép nhánh "hư-từ" bật hết cỡ
    for s in ("1", "8", "1288"):
        tid = int(vocab.encode_text(s)[0])
        assert not _is_path_light(vocab, tid, gidf=gidf), f"{s!r} là content, không phải hư-từ"


# ── §12.14 chép-lùi: số của cha ĐANG KÍCH khi path không khoá nổi ───────────────
def _act_on(pi: int, n: int = 3):
    a = np.zeros(n, dtype=np.float64)
    a[pi] = 1.0
    return a


def _emit(vocab, text):
    return np.asarray(list(vocab.encode_text(text)), dtype=np.int64)


def test_back_copy_chep_so_dung_truoc_phan_khop(vocab):
    """Cha "Năm 938, Ngô Quyền đánh bại quân Nam Hán…" — số đứng TRƯỚC ⇒ path bó tay, act thì không."""
    from nn_aivie.hca.matrix_core import digit_run_from_active

    bi = _Bi([_emit(vocab, "Năm 938, Ngô Quyền đánh bại quân Nam Hán trên sông Bạch Đằng.")])
    hist = list(vocab.encode_prompt("Ngô Quyền đánh bại quân Nam Hán năm"))
    nxt = int(vocab.encode_text("9")[0])
    run = digit_run_from_active(bi, vocab, _act_on(0, 1), hist, nxt)
    assert "".join(vocab.token_of(t).lstrip("Ġ") for t in run) == "938"


def test_back_copy_khong_no_khi_nxt_khong_phai_chu_so(vocab):
    """Mạng đang định viết CHỮ → đừng chen số vào (đo: `ly` mất "Thăng Long năm")."""
    from nn_aivie.hca.matrix_core import digit_run_from_active

    bi = _Bi([_emit(vocab, "Năm 1010, nhà Lý dời đô từ Hoa Lư về Thăng Long.")])
    hist = list(vocab.encode_prompt("Nhà Lý dời đô về"))
    nxt = int(vocab.encode_text(" Thăng")[0])
    assert digit_run_from_active(bi, vocab, _act_on(0, 1), hist, nxt) == []


def test_back_copy_khong_bia_khi_cha_khac_su_viec(vocab):
    """Cổng phủ-100%: hỏi `quân Tống` mà cha nói `quân Nam Hán` ⇒ KHÔNG được chép 938.

    Đây chính là cổng v1 (§12.5) thiếu → chép nhầm cha ra `18110`."""
    from nn_aivie.hca.matrix_core import digit_run_from_active

    bi = _Bi([_emit(vocab, "Năm 938, Ngô Quyền đánh bại quân Nam Hán trên sông Bạch Đằng.")])
    hist = list(vocab.encode_prompt("Ngô Quyền đánh bại quân Tống năm"))
    nxt = int(vocab.encode_text("9")[0])
    assert digit_run_from_active(bi, vocab, _act_on(0, 1), hist, nxt) == []


def test_back_copy_khong_chep_khi_cha_co_NHIEU_so(vocab):
    """Hai con số trong emit ⇒ không biết chép cái nào ⇒ im lặng, đừng đoán."""
    from nn_aivie.hca.matrix_core import digit_run_from_active

    bi = _Bi([_emit(vocab, "Năm 938, Ngô Quyền đánh bại quân Nam Hán; năm 981 Lê Hoàn thắng.")])
    hist = list(vocab.encode_prompt("Ngô Quyền đánh bại quân Nam Hán năm"))
    nxt = int(vocab.encode_text("9")[0])
    assert digit_run_from_active(bi, vocab, _act_on(0, 1), hist, nxt) == []


# ── §12.27 chép span ĐÁP của cha QA đang kích ──────────────────────────────────
def test_qa_answer_copy_chep_dap_khi_cau_hoi_khop(vocab):
    """Cha QA: `emit = answer_ids` nên suffix-match history↔emit KHÔNG BAO GIỜ fire.

    Đo mini-có-QA: hỏi `2 cộng 3 bằng mấy?` → retrieval trúng tuyệt đối (act=0.798) mà
    `path.max=0`, `lock=-1` ⇒ trường logit phẳng ⇒ đáp 1 token. Chép span đáp là đường duy nhất."""
    import numpy as np

    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.core.state import GraphState
    from nn_aivie.data.qa import DAP_LABEL
    from nn_aivie.hca.matrix_core import answer_span_from_active

    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    ps.add_example(st.vocab.encode_tokens(
        "Hoa Lư nằm ở tỉnh nào ? Đáp: Hoa Lư nằm ở tỉnh Ninh Bình .".replace(
            "Đáp:", DAP_LABEL).split()), st.vocab, dap_id=dap)
    st.parents = ps
    act = np.array([1.0])
    hist = list(st.vocab.encode_tokens("Hoa Lư nằm ở tỉnh nào ?".split())) + [dap]
    span = answer_span_from_active(st, act, hist, min_cover=0.8)
    assert span, "cha QA khớp câu hỏi mà không chép được đáp"
    assert "Ninh" in " ".join(st.vocab.token_of(t).lstrip("Ġ") for t in span)


def test_qa_answer_copy_chan_khi_cau_hoi_KHAC(vocab):
    """Cổng phủ `match_ids`: hỏi `Nhà Lý dời đô về đâu?` thì cha act đỉnh trên mini là
    "Nhà mình sắp đón Tết, dọn dẹp từ ĐÂU?" — khớp theo HÌNH DẠNG câu hỏi. `match_ids` của nó
    không nằm trong history ⇒ phải CHẶN, đừng chép đáp về dọn tủ quần áo."""
    import numpy as np

    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.core.state import GraphState
    from nn_aivie.data.qa import DAP_LABEL
    from nn_aivie.hca.matrix_core import answer_span_from_active

    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    ps.add_example(st.vocab.encode_tokens(
        ["Nhà", "mình", "sắp", "đón", "Tết", "dọn", "dẹp", "từ", "đâu", "?", DAP_LABEL,
         "Bắt", "đầu", "tủ", "quần", "áo", "."]), st.vocab, dap_id=dap)
    st.parents = ps
    hist = list(st.vocab.encode_tokens("Nhà Lý dời đô về đâu ?".split())) + [dap]
    assert answer_span_from_active(st, np.array([1.0]), hist, min_cover=0.8) == []


def test_cache_theo_doi_tuong_tu_choi_ban_cua_CHU_KHAC(vocab):
    """§12.29/§12.30 — `id()` chỉ duy nhất khi đối tượng còn SỐNG. State bị thu hồi rồi state
    mới trúng cùng địa chỉ sẽ nhận đồ dựng cho state CŨ. Bắt được vì suite chạy 3 lần ra
    284/283+1/284 trên CÙNG cây mã (`vocab.fold(7) IndexError`). Crash là trường hợp MAY —
    thường id vẫn hợp lệ nên nó truy hồi sai ÂM THẦM.

    `objcache` giữ id làm khoá nhưng so `is` qua weakref ⇒ đúng tuyệt đối, không heuristic
    kích thước (hai state khác nhau hoàn toàn có thể cùng `len(vocab)`)."""
    from nn_aivie.core.state import GraphState
    from nn_aivie.hca.objcache import cache_get, cache_put

    a, b = GraphState(), GraphState()
    box: dict = {}
    cache_put(box, a, "của-A")
    assert cache_get(box, a) == "của-A"
    assert cache_get(box, b) is None, "chủ khác phải MISS, không được dùng nhầm"

    # id tái dùng: nhét bản của A vào ĐÚNG khoá id(b) → vẫn phải từ chối
    box[id(b)] = box[id(a)]
    assert cache_get(box, b) is None
    assert cache_get(box, a) == "của-A", "chủ đúng vẫn phải HIT"


def test_bipartite_dung_lai_khi_chu_khac(vocab):
    """Hệ quả trên `ensure_parent_bipartite`: khoá trùng id nhưng chủ khác ⇒ dựng lại."""
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.core.state import GraphState
    from nn_aivie.data.qa import DAP_LABEL
    from nn_aivie.hca.matrix_core import _PARENT_BI, ensure_parent_bipartite

    def _mk(words):
        st = GraphState()
        ps = ParentStore()
        ps.add_example(st.vocab.encode_tokens(words), st.vocab,
                       dap_id=st.vocab.id_of(DAP_LABEL))
        st.parents = ps
        return st

    st1 = _mk("Ngô Quyền lập nước .".split())
    st2 = _mk("Lê Lợi khởi nghĩa ở Lam Sơn Thanh Hóa năm .".split())
    bi1 = ensure_parent_bipartite(st1)
    _PARENT_BI[id(st2)] = _PARENT_BI[id(st1)]      # ← bản của st1, sai vocab
    bi2 = ensure_parent_bipartite(st2)
    assert bi2 is not bi1
    assert bi2.A.shape[0] == len(st2.vocab)
