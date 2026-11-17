"""TK_045 §12.15 — cổng TIỀN-ĐỀ: tổ hợp token đặc-hiệu của prompt chưa cha nào chứng thực.

`Nhà Trần dời đô về Thăng Long năm` → mạng trả `1010` (fact của nhà **Lý**). Cha thắng thiếu
đúng một token gidf-cao (`Trần`) — nhưng `Trận Bạch Đằng năm` cũng thiếu đúng một (`Trận`) mà
lại ĐÚNG, nên "thiếu token" một mình không phân biệt được. Thứ phân biệt là **giao các cha**
của mọi token gidf-cao: rỗng = trí nhớ chưa từng thấy tổ hợp ⇒ mọi cha chỉ khớp một PHẦN.
"""

from __future__ import annotations

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.hca.matrix_core import _PARENT_BI, unattested_premise


def _state(sentences: list[str]) -> GraphState:
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    for s in sentences:
        ps.add_example(st.vocab.encode_tokens(s.split()), st.vocab, dap_id=dap)
    st.parents = ps
    _PARENT_BI.pop(id(st), None)  # cache bipartite theo id(state)
    return st


# gidf là IDF HỌC ĐƯỢC nên thang phụ thuộc cỡ corpus: 5 câu đồ chơi → max ~2.1 (corpus thật
# ~4–7). Test truyền `gidf_hi=2.0` cho khớp thang, không phải nới lỏng cổng.
GIDF_HI = 2.0

CORPUS = [
    "Năm 1010 nhà Lý dời đô từ Hoa Lư về Thăng Long .",
    "Năm 938 Ngô Quyền đánh bại quân Nam Hán trên sông Bạch Đằng .",
    "Năm 981 Lê Hoàn đánh bại quân Tống trên sông Bạch Đằng .",
    "Lê Lợi khởi nghĩa ở Lam Sơn .",
    "Quân dân nhà Trần đánh tan quân Nguyên trên sông Bạch Đằng .",
]


def test_tien_de_that_khong_no():
    """Tổ hợp CÓ trong trí nhớ ⇒ cổng im, sinh bình thường."""
    st = _state(CORPUS)
    ids = st.vocab.encode_tokens("nhà Lý dời đô về Thăng Long".split())
    assert unattested_premise(st, list(ids), gidf_hi=GIDF_HI) is False


def test_tien_de_hoan_thuc_the_thi_no():
    """`nhà Trần` + `dời đô` + `Thăng Long` chưa cha nào chứng thực ⇒ ĐỪNG trả năm của nhà Lý."""
    st = _state(CORPUS)
    ids = st.vocab.encode_tokens("nhà Trần dời đô về Thăng Long".split())
    assert unattested_premise(st, list(ids), gidf_hi=GIDF_HI) is True


def test_qua_it_token_dac_hieu_thi_khong_ket_luan():
    """1 token đặc-hiệu thì giao chỉ là "cha của token đó" — không đủ cơ sở kết luận bịa."""
    st = _state(CORPUS)
    ids = st.vocab.encode_tokens(["Lý"])
    assert unattested_premise(st, list(ids), gidf_hi=GIDF_HI) is False


def test_prompt_rong_khong_no():
    st = _state(CORPUS)
    assert unattested_premise(st, [], gidf_hi=GIDF_HI) is False


def test_last_k_chi_xet_menh_de_dang_viet():
    """Prompt DÀI gom quá nhiều token đặc-hiệu ⇒ giao rỗng ⇒ nổ bừa (đo mini: 3/4 prompt HỢP LỆ).

    `last_k` cắt về mệnh đề đang viết. Ở đây tiền đề ĐÚNG (`nhà Lý dời đô … Thăng Long`) bị
    ghép thêm một câu khác cùng corpus — không giới hạn thì nổ, giới hạn thì im."""
    st = _state(CORPUS)
    long_ids = st.vocab.encode_tokens(
        "Lê Lợi khởi nghĩa ở Lam Sơn . nhà Lý dời đô về Thăng Long".split()
    )
    assert unattested_premise(st, list(long_ids), gidf_hi=GIDF_HI, last_k=0) is True
    assert unattested_premise(st, list(long_ids), gidf_hi=GIDF_HI, last_k=4) is False


def test_nhan_frame_khong_duoc_lam_rong_giao():
    """§12.18 — `Hỏi:`/`Đáp:` KHÔNG bị `is_structural_token` bắt, mà `A` lại không có chúng
    (df=0) ⇒ mọi prompt QA làm giao rỗng NGAY token đầu ⇒ cổng nổ 100% bất kể nội dung.
    Đây chính là lỗi làm hỏng toàn bộ số QA ở §12.17."""
    from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL
    from nn_aivie.hca.context_positional import is_structural_token

    st = _state(CORPUS)
    hoi, dap = st.vocab.id_of(HOI_LABEL), st.vocab.id_of(DAP_LABEL)
    # tiền đề của bug: nhãn frame KHÔNG phải "structural" theo hàm hiện có
    assert not is_structural_token(st.vocab, hoi)
    plain = list(st.vocab.encode_tokens("nhà Lý dời đô về Thăng Long".split()))
    framed = [hoi] + plain + [dap]
    assert unattested_premise(st, plain, gidf_hi=GIDF_HI, last_k=0) is False
    assert unattested_premise(st, framed, gidf_hi=GIDF_HI, last_k=0) is False


def test_cong_bit_luon_prompt_GHEP_nhieu_fact():
    """§12.20 — giới hạn BẢN CHẤT, không phải bug: luật "không cha nào chứa ĐỦ token đặc-hiệu
    ⇒ im" chính là "từ chối mọi thứ cần hơn MỘT trí nhớ". Prompt so sánh 2 thực thể trải trên
    2 cha ⇒ cổng nổ y như prompt BỊA. Vì thế guard phải OFF mặc định (composition, TK_024)."""
    st = _state(CORPUS)
    ghep = st.vocab.encode_tokens("so sánh nhà Lý và nhà Trần".split())   # 2 fact CÓ THẬT
    bia = st.vocab.encode_tokens("nhà Trần dời đô về Thăng Long".split())  # tiền đề BỊA
    assert unattested_premise(st, list(ghep), gidf_hi=GIDF_HI, last_k=0) is True
    assert unattested_premise(st, list(bia), gidf_hi=GIDF_HI, last_k=0) is True


# ── §12.28 QA routing: từ-để-hỏi không được làm neo hard-AND ───────────────────
def test_emit_match_ratio_tach_tu_de_hoi_khoi_tu_noi_dung():
    """Tín hiệu HỌC ĐƯỢC, không bảng từ: (số cha PHÁT token)/(số cha KHỚP bằng token).

    Token chỉ dùng để KHỚP mà không bao giờ được PHÁT = chỉ mang hình thức câu hỏi.
    Đo mini-có-QA: gì .12 · mấy .15 · nào .22 · đâu .43 ‖ Lý 1.02 · dời 1.10 · Thăng 1.17."""
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.core.state import GraphState
    from nn_aivie.data.qa import DAP_LABEL
    from nn_aivie.hca.matrix_core import _PARENT_BI, emit_match_ratio, ensure_parent_bipartite

    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    # `đâu` CHỈ xuất hiện ở phần hỏi; `Thăng`/`Long` xuất hiện ở phần đáp.
    for q, a in [("Nhà Lý dời đô về đâu", "nhà Lý dời đô về Thăng Long"),
                 ("Kinh đô nhà Lý ở đâu", "kinh đô nhà Lý là Thăng Long")]:
        ps.add_example(st.vocab.encode_tokens(q.split() + [DAP_LABEL] + a.split()),
                       st.vocab, dap_id=dap)
    st.parents = ps
    _PARENT_BI.pop(id(st), None)
    er = emit_match_ratio(ensure_parent_bipartite(st))
    dau = st.vocab.id_of("đâu")
    thang = st.vocab.id_of("Thăng")
    assert er[dau] < er[thang], "từ-để-hỏi phải có E/A THẤP hơn từ nội dung"
    assert er[dau] < 0.9 <= er[thang]


def test_qa_wh_slot_ban_uu_tien_cha_noi_dung():
    """§12.42: prompt kết Đáp: → cấm WH-slot khỏi ctx/AND ⇒ cha nội dung thắng WH-shape.

    `emit_match_ratio` trên v12 ≈1.0 cho cả đâu/Lý nên không tách được; luật vị trí+độ dài
    (cuối ≤3, đầu ≤2, không chữ-số) thay thế.
    """
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.core.state import GraphState
    from nn_aivie.core.tokenizer import DAP_LABEL
    from nn_aivie.hca.matrix_core import (
        _PARENT_BI,
        answer_span_from_active,
        ensure_parent_bipartite,
        parent_coactivation_logits,
    )

    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    ps.add_example(
        st.vocab.encode_tokens("Nhà ở đâu".split() + [DAP_LABEL] + "nhà ở đây".split()),
        st.vocab, dap_id=dap,
    )
    ps.add_example(
        st.vocab.encode_tokens(
            "Nhà Lý dời đô về đâu".split() + [DAP_LABEL]
            + "nhà Lý dời đô về Thăng Long".split()
        ),
        st.vocab, dap_id=dap,
    )
    st.parents = ps
    _PARENT_BI.pop(id(st), None)
    ensure_parent_bipartite(st)

    hist = list(st.vocab.encode_prompt("Hỏi: Nhà Lý dời đô về đâu? Đáp:"))
    _b, _s, _c, act = parent_coactivation_logits(
        st, hist, vocab_size=len(st.vocab), and_form_min=0.9, top_k_parents=8,
    )
    assert act.max() > 0
    span = answer_span_from_active(st, act, hist, form_min=0.9, statement=True)
    text = " ".join(st.vocab.token_of(t).lstrip("Ġ") for t in span)
    assert "Thăng" in text or "Long" in text, f"WH-ban phải chép cha dời đô; span={text!r}"


def test_left_pmi_khong_gop_hoa_thuong_thi_tach_duoc_loai_tu():
    """§12.46 — PMI(prev, thực-thể) trên emit, KHÔNG gộp hoa/thường.

    §12.45③: gộp fold thì `nhà lý` (3.4) THUA `quản lý` (5.4) vì fold trộn danh-từ-riêng `Lý`
    với từ thường `lý`. Tách theo id thì `Lý`(cap=1) → `nhà`/`vua`, `lý`(cap=0) → `quản`/`xử`."""
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.core.state import GraphState
    from nn_aivie.data.qa import DAP_LABEL
    from nn_aivie.hca.matrix_core import _PARENT_BI, left_pmi

    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    # Mỗi cụm phải lặp ≥3 lần: `min_count=1` để lọt token hiếm-ngẫu-nhiên (đo: `do` xuất hiện
    # đúng 1 lần trước `Lý` đã chiếm PMI cao nhất) — chính lý do hàm có ngưỡng đếm.
    for s in ["nhà Lý dời đô về Thăng Long .", "nhà Lý lập kinh đô mới .",
              "nhà Lý xây nhiều chùa lớn .", "vua nhà Lý trị vì lâu dài .",
              "quản lý đất đai theo luật .", "quản lý nhân sự trong triều .",
              "quản lý kho lương của quân ."]:
        ps.add_example(st.vocab.encode_tokens(s.split()), st.vocab, dap_id=dap)
    st.parents = ps
    _PARENT_BI.pop(id(st), None)

    hoa = st.vocab.id_of("Lý")     # danh từ riêng
    thuong = st.vocab.id_of("lý")  # từ thường
    assert hoa != thuong, "fixture phải giữ hai token KHÁC nhau"
    top_hoa = max(left_pmi(st, hoa, min_count=3).items(), key=lambda kv: kv[1])[0]
    top_thuong = max(left_pmi(st, thuong, min_count=3).items(), key=lambda kv: kv[1])[0]
    assert st.vocab.token_of(top_hoa).lstrip("Ġ") in {"nhà", "vua"}
    assert st.vocab.token_of(top_thuong).lstrip("Ġ") == "quản"
