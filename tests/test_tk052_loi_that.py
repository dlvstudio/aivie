"""TK_052 — SÁU LỖI THẬT của phiên 2026-08-15, mỗi lỗi một test canh.

Cả sáu đều là lỗi ĐANG CHẠY trong sản phẩm, sống từ 20 tới 40 mục mà không test
nào chạm tới. Không có lưới này thì chúng quay lại lặng lẽ — đúng cách chúng đã
sống lâu như thế.

    1. `hub_chain_proposals` gọi ① mà KHÔNG truyền `kenh_thuoc_tinh`  (vòng 2)
    2. `_liet_ke` đọc định nghĩa giống–loài thành LIỆT KÊ, phát cạnh NGƯỢC (v6)
    3. bề mặt span WS ghép bằng dấu cách ⇒ từ bị BPE băm mất cạnh        (v7–9)
    4. `ids_theo_be_mat` phải thấy id ẢO của span                        (v7)
    5. `_ghep_bm` guard phải hỏi TỪ VỰNG, không hỏi dãy đang xét         (v9)
    6. gộp mảnh phải VÔ HIỆU với vocab không đánh dấu ranh giới          (v10)
"""
from __future__ import annotations

import inspect

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca import generate_hebbian as G
from nn_aivie.hca import hub_spread as H


def _ck(cau: list[str]) -> GraphState:
    """Checkpoint tí hon; token mở đầu từ mang MARK_SPACE như tokenizer thật."""
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()
    for s in cau:
        ids = [v.id_of(MARK_SPACE + t) for t in s.split()] + [v.id_of("<EOS>")]
        st.parents.add_example(ids, v, dap_id=None, defer_index=True)
    st.parents.rebuild_hierarchical_index()
    return st


# ── 1. kênh thuộc tính phải được đường sinh GỌI ──────────────────────────
def test_LOI1_duong_sinh_phai_truyen_kenh_thuoc_tinh() -> None:
    """§53 dựng `via_attr` cho bước ①, nhưng `hub_chain_proposals` quên truyền
    `kenh_thuoc_tinh=True` ⇒ kênh nằm im ~20 mục. Đo được: thu hồi 27,6% →
    44,1% khi vá (TK_052 vòng 2)."""
    src = inspect.getsource(G.hub_chain_proposals)
    i = src.index("score_hubs_from")
    assert "kenh_thuoc_tinh=True" in src[i:i + 400], (
        "bước ① của đường sinh KHÔNG bật kênh thuộc tính — §53 nằm im")


# ── 2. `_liet_ke` không được nuốt định nghĩa giống–loài ──────────────────
def test_LOI2_dinh_nghia_giong_loai_khong_bi_doc_thanh_liet_ke() -> None:
    """«cắt là chim ăn thịt , nhỏ hơn diều hâu , cánh dài và nhọn» có ĐÚNG hình
    dạng bề mặt của khuôn liệt kê §29, và trước vòng 6 nó phát NGƯỢC CHIỀU
    `chim → cắt`. Mục liệt kê thật phải là DANH TỪ."""
    # Khu biệt thật gồm TÍNH TỪ («nhỏ hơn diều hâu, cánh dài và nhọn, bay rất
    # nhanh»), khác hẳn liệt kê thật gồm DANH TỪ («NBC, CBS, ABC và Fox»).
    st = _ck([
        "cắt là chim nhỏ , nhanh , mạnh và dữ",
        "chim là động vật",
    ])
    pm = {"cắt": "danh từ", "chim": "danh từ", "động vật": "danh từ",
          "nhỏ": "tính từ", "nhanh": "tính từ", "mạnh": "tính từ",
          "dữ": "tính từ"}
    idx = H.build_hub_spread_index(
        st, d_max=6, min_count=0.5, use_isg=False, pos_map=pm,
        head_rule="trai_du_dac_thu", head_rule_trai="phai_nhat",
        label_gidf_q=0.0, head_gidf_q=0.0, label_indeg_q=0.0)
    v = st.vocab
    nhan_chim = {H._surf(v, d) for i in (H.resolve_surface_ids(v, "chim") or [])
                 for d in idx.via_pivot.get(int(i), {})}
    assert "cắt" not in nhan_chim, (
        f"cạnh NGƯỢC CHIỀU `chim → cắt` đã quay lại: {sorted(nhan_chim)}")


# ── 3+5. bề mặt span ghép theo MARK_SPACE, guard hỏi TỪ VỰNG ─────────────
def test_LOI3_be_mat_span_ghep_theo_mark_space() -> None:
    """`sóc` bị BPE tách thành `s`+`óc`; span WS gộp đúng nhưng bề mặt ghép
    bằng dấu cách thành `"s óc"` ⇒ `resolve` bằng `sóc` không thấy cạnh."""
    v = Vocabulary()
    a = v.id_of(MARK_SPACE + "s")      # mở đầu từ
    b = v.id_of("óc")                  # mảnh tiếp, KHÔNG mở đầu
    c = v.id_of(MARK_SPACE + "là")
    assert H._ghep_bm(v, [a, b]) == "sóc"
    assert H._ghep_bm(v, [a, b, c]) == "sóc là"


def test_LOI5_guard_hoi_TU_VUNG_khong_hoi_day_dang_xet() -> None:
    """Từ đứng ĐẦU DÒNG thì mảnh đầu cũng không mang `MARK_SPACE`. Guard cũ
    kiểm «dãy này có dấu không» nên rơi vào nhánh ghép-bằng-dấu-cách và cho
    `"s óc"` — vòng 9 mất ba lần đoán vì lỗi này."""
    v = Vocabulary()
    v.id_of(MARK_SPACE + "là")         # từ vựng CÓ đánh dấu ranh giới
    a, b = v.id_of("s"), v.id_of("óc")  # nhưng dãy này thì KHÔNG
    assert H._ghep_bm(v, [a, b]) == "sóc", (
        "guard đang hỏi DÃY thay vì TỪ VỰNG")


# ── 4. tra id phải thấy id ẢO của span ───────────────────────────────────
def test_LOI4_ids_theo_be_mat_thay_id_ao() -> None:
    """`resolve_surface_ids` chỉ trả id TOKEN THẬT; cạnh của từ bị băm nằm trên
    id ẢO của span. Không gộp hai nguồn thì tra bằng bề mặt luôn rỗng."""
    st = _ck(["chó là thú", "mèo là thú"])
    idx = H.build_hub_spread_index(
        st, d_max=6, min_count=0.5, use_isg=False,
        head_rule="trai_du_dac_thu", head_rule_trai="phai_nhat",
        label_gidf_q=0.0, head_gidf_q=0.0, label_indeg_q=0.0)
    idx.label_surf[999_999] = "chó"
    idx.label_by_surf.setdefault("chó", []).append(999_999)
    ids = H.ids_theo_be_mat(idx, st.vocab, "chó")
    assert 999_999 in ids, "id ẢO của span không được tra ra"
    assert set(H.resolve_surface_ids(st.vocab, "chó") or []) <= set(ids)


# ── 6. gộp mảnh VÔ HIỆU với vocab không đánh dấu ranh giới ───────────────
def test_LOI6_gop_manh_vo_hieu_khi_vocab_khong_danh_dau() -> None:
    """Vocab fixture không có `MARK_SPACE` ⇒ không có khái niệm «mảnh của một
    từ», gộp là vô nghĩa mà vẫn đổi hành vi ① đủ để CA THỬ VÀNG mất `khỉ`
    (2 test đỏ ở vòng 9). Bật/tắt cờ phải cho KẾT QUẢ Y HỆT."""
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()
    for s in ("chó là thú", "mèo là thú", "khỉ là thú"):
        st.parents.add_example(
            [v.id_of(t) for t in s.split()] + [v.id_of("<EOS>")],
            v, dap_id=None, defer_index=True)
    st.parents.rebuild_hierarchical_index()
    assert not H._vocab_co_mark(v)
    ra = []
    for g in (False, True):
        idx = H.build_hub_spread_index(
            st, d_max=6, min_count=0.5, use_isg=False, gop_manh_tu=g,
            head_rule="trai_du_dac_thu", head_rule_trai="phai_nhat",
            label_gidf_q=0.0, head_gidf_q=0.0, label_indeg_q=0.0)
        ra.append(sum(len(d) for d in idx.via_pivot.values()))
    assert ra[0] == ra[1], f"gộp mảnh KHÔNG trung tính trên vocab không dấu: {ra}"


# ── 7. hai chỗ ép EOS phải NHẬN tín hiệu biên học được ───────────────────
def test_LOI7_ep_eos_phai_truyen_bound() -> None:
    """§81 — `_w_skip_primary_loop` tính `_bd_gen = _bound_of(state)` ngay đầu
    hàm và truyền nó cho ba chỗ khác, nhưng HAI chỗ ép EOS gọi `_is_boundary`
    TRẦN. Gọi trần ⇒ `bound=None, bootstrap=False`, và theo đúng docstring của
    `_is_boundary` khi ấy nó chỉ còn ngắt-khoảng-trắng: mọi dấu câu trả False.

    Trong khi `count_boundaries` mặc định `bootstrap=(bound is None)` nên NÓ
    đếm được dấu chấm ⇒ vế `n_b_gen >= 1` đúng, vế `_is_boundary(...)` luôn sai
    ⇒ luật «đứng trên biên sau ≥1 câu → EOS» CHƯA BAO GIỜ nổ.

    Đo được (§81.1, `network_v12_sent_dn.pkl`): «chó là …đi săn .» bước 14 phát
    token `Ġ.`, trần=False · bound-đúng=True · count_boundaries=1 — đáng ra dừng
    mà vẫn đẻ tiếp `vào năm 1 công tấn`. Vá xong: 4/20 prompt hết đuôi rác, 0 ca
    xấu đi.
    """
    src = inspect.getsource(G._w_skip_primary_loop)
    assert "_bd_gen = _bound_of(state)" in src, "mất tay cầm tín hiệu biên"
    goi = [d for d in ("_is_b_eos(", "_is_b_g(") if d in src]
    assert len(goi) == 1, f"§157 go ep cung ⇒ con dung 1; thay {goi}"
    for d in goi:
        i = src.index(d)
        doan = src[i:i + 120]
        assert "_bd_gen" in doan, (
            f"`{d}` gọi TRẦN — dấu câu sẽ không bao giờ được nhận là biên: {doan!r}")


# ── 8. cache `_vocab_co_mark` không được khoá bằng id() ──────────────────
def test_LOI8_cache_co_mark_khong_khoa_bang_id() -> None:
    """§83.2 — `_CO_MARK` từng là `dict[int, bool]` khoá bằng `id(vocab)`, không
    giữ tham chiếu và không bao giờ dọn. Vocab cũ bị thu gom ⇒ CPython tái dùng
    địa chỉ ⇒ vocab MỚI nhận câu trả lời CŨ, và `_ghep_bm` rẽ nhầm nhánh đúng
    như TK_052 vòng 7–9 (`"s óc"`, mất cạnh `sóc → thú`).

    Đã hiện hình thật: `test_LOI5_...` XANH khi chạy riêng, ĐỎ khi chạy cả bộ,
    kể cả khi đã stash mọi thay đổi khác — thuần do thứ tự chạy + gc.

    Test này ép đúng kịch bản: tạo vocab CÓ dấu, thả cho gc thu, rồi tạo vocab
    KHÔNG dấu và hỏi lại. Khoá bằng `id()` sẽ trả `True` sai.
    """
    import gc as _gc

    dia_chi = []
    for _ in range(40):
        v_co = Vocabulary()
        v_co.id_of(MARK_SPACE + "là")
        assert H._vocab_co_mark(v_co) is True
        dia_chi.append(id(v_co))
        del v_co
        _gc.collect()

        v_khong = Vocabulary()
        v_khong.id_of("là")
        assert H._vocab_co_mark(v_khong) is False, (
            "vocab KHÔNG dấu nhận câu trả lời của vocab CŨ — cache khoá bằng id()")
        del v_khong
        _gc.collect()


# ── 9. cache chỉ mục chuỗi không được khoá bằng id() trần ────────────────
def test_LOI9_cache_hub_chain_khong_khoa_bang_id_tran() -> None:
    """§83.2 — `_HUB_CHAIN_CACHE` từng khoá bằng `id(state)` trần. `id()` chỉ
    duy nhất TRONG THỜI GIAN đối tượng còn sống, nên một `GraphState` bị thu hồi
    rồi state khác trùng địa chỉ sẽ được phát chỉ mục loại suy của CHECKPOINT
    CŨ — sai lặng, ngay trên đường chuỗi §79.

    `hca/objcache.py` sinh ra đúng để chặn việc này (id + weakref + so `is`).
    Test ép kịch bản: state CÓ cạnh `chó→thú`, thả cho gc, rồi state RỖNG hỏi
    lại. Khoá bằng `id()` trần sẽ phát nhầm chỉ mục cũ.
    """
    import gc as _gc

    from nn_aivie.hca.config import HCAConfig

    cfg = HCAConfig()
    cfg.hub_chain_boost = 1.0
    for _ in range(40):
        st = _ck(["chó là thú", "mèo là thú", "khỉ là thú"])
        G.hub_chain_proposals(st, [st.vocab.id_of(MARK_SPACE + "chó")], cfg)
        del st
        _gc.collect()

        rong = _ck(["a b c"])
        de = G.hub_chain_proposals(
            rong, [rong.vocab.id_of(MARK_SPACE + "a")], cfg)
        surf = {H._surf(rong.vocab, int(t)) for t in de}
        assert "thú" not in surf, (
            f"chỉ mục của state CŨ rò sang state MỚI: {sorted(surf)}")
        del rong, de
        _gc.collect()


# ── 10. hai kênh bậc-2 phải TẮT theo mặc định (§83/§84 âm tính) ──────────
def test_LOI10_kenh_bac2_mac_dinh_TAT() -> None:
    """§83/§84 — cả kênh bậc-2 toàn cục lẫn bản cha-đang-hoạt đều ÂM TÍNH: chúng
    đổi TRI THỨC lấy TRÔI CHẢY. Đo trên `network_v12_sent_dn.pkl`, 20 prompt,
    thước TRUNG THÀNH (tỉ lệ từ sinh ra có trong câu nguồn thật):

        tắt              78,9%      ← không mức nào vượt
        toàn cục w=1     53,8%
        cha hoạt w=1     67,0%  (top_n=20) · tốt nhất 74,7% ở top_n=3

    Mã vẫn giữ (có cờ, có tài liệu) để tái lập được, nhưng mặc định PHẢI tắt —
    bật ngầm là lặng lẽ đánh đổi tri thức lấy văn trôi chảy.
    """
    from nn_aivie.hca.config import HCAConfig

    cfg = HCAConfig()
    assert cfg.ar_w_skip_order2_weight == 0.0, (
        "kênh bậc-2 bị bật ngầm — §83.3 đo được nó xoá tri thức")
    assert cfg.ar_w_skip_order2_cha_hoat is False


# ── 11. lọc đích via_pivot phải TẮT theo mặc định (§93 âm tính) ──────────
def test_LOI11_loc_dich_mac_dinh_TAT() -> None:
    """§93 — bộ lọc đích `A là B` đúng và vô hại nhưng VÔ ÍCH: bỏ 61 674/223 988
    cạnh (danh từ 39,1% → 53,9%), cạnh vàng còn nguyên, mà thu hồi KHÔNG đổi
    (đối chứng hoán vị vẫn thắng) và đầu ra sinh giống TỪNG BYTE.

    Giữ mã để tái lập, nhưng mặc định phải TẮT: bật một bộ lọc không đo được
    lợi ích mà lại đổi đồ thị cho MỌI tầng tiêu thụ khác là rủi ro không đối giá.
    """
    from nn_aivie.hca.config import HCAConfig

    assert HCAConfig().hub_chain_loc_dich is False


# ── 12. idf gộp ranh giới: lỗi THẬT, nhưng mặc định TẮT (§96) ────────────
def test_LOI12_idf_gop_ranh_gioi_mac_dinh_TAT() -> None:
    """§96 — token ĐẦU DÒNG (`con`) và GIỮA CÂU (` con`) là hai id với hai `df`
    riêng, mà từ đầu tiên của MỌI prompt luôn ở dạng đầu dòng ⇒ idf vống:

        là 14,0 vs 8,3 · chó 14,0 vs 10,7 · con 13,3 vs 10,7

    Vá ở tầng đọc (`_df_gop`) chạy đúng — `con` 13,3 → 10,7 — nhưng KHÔNG đổi
    hạng truy hồi (3 → 3) và KHÔNG đổi một byte đầu ra nào. Lỗi thật, bản vá
    đúng, tác dụng bằng 0 ⇒ giữ mã, mặc định TẮT (cùng lý do §93/`LOI11`).
    """
    from nn_aivie.core.parent_store import ParentStore

    ps = ParentStore()
    assert getattr(ps, "idf_gop_ranh_gioi", False) is False
    assert hasattr(ps, "_df_gop")


# ── 13. lùi về từ chính: ÂM TÍNH, mặc định TẮT (§99) ─────────────────────
def test_LOI13_lui_ve_tu_chinh_mac_dinh_TAT() -> None:
    """§99 — lùi về từ chính khi cụm chủ ngữ không có mục định nghĩa. Ý tưởng
    đến từ tương quan 10/10 ở §98, nhưng đo đầu-cuối thì ÂM TÍNH: 7/8 ca hỏng
    không đổi, ca thứ 8 đổi sang rác khác, và `xe đạp là` (ĐANG ĐÚNG) bị lùi
    nhầm về `xe` nên HỎNG — phép kiểm cả-cụm có âm tính giả.

    Giữ mã để tái lập; mặc định TẮT.
    """
    from nn_aivie.hca.config import HCAConfig

    assert HCAConfig().lui_ve_tu_chinh is False


# ── 14. cửa sổ thứ tự agenda (§112) ──────────────────────────────────────
def test_LOI14_cua_so_thu_tu_agenda() -> None:
    """§111/§112 — agenda LÀ câu nguồn đúng thứ tự; cửa sổ cho phép mỗi bước
    chọn bất kỳ trong N mục kế tiếp, nên N=4 sinh ra `hoạt động rất rộng` →
    `hoạt · rất · động`. Quét 4/3/2/1 với thước đã sửa: 3 qua cổng (TRUNG THÀNH
    79,3 → 81,5 · TRẬT TỰ 89,3 → 89,8).

    Canh để không ai nới lại lên 4 mà không đo.
    """
    from nn_aivie.hca.config import HCAConfig

    # §151 — DỪNG THEO ĐỘ TIN bật mặc định 0,15. Đo 6 hạt/20 prompt:
    # TRUNG THÀNH 86,1±1,3 → 88,3±0,3 · TRẬT TỰ 97,2±1,8 → 99,3±0,0 ·
    # LẶP 9,9±2,4 → 9,4±0,5. Phương sai SỤP vì cơ chế cắt đúng vùng mạng không
    # chắc chắn — vùng mà lấy mẫu quyết định kết quả.
    # §153 — hạ 0,15 → 0,05: 0,15 cắt «muối là» thành «sản phẩm axit» (SAI).
    assert HCAConfig().ar_w_skip_do_tin_min == 0.05

    # §113 — sửa từ 3 xuống 1: TRUNG THÀNH mù với THỨ TỰ (chấm câu rối và câu
    # đúng ngang nhau, còn phạt câu đúng khi nó lệch ở đuôi). Chọn theo TRẬT TỰ
    # (89,8 → 93,8) và LẶP (14,5 → 10,8).
    assert HCAConfig().ar_w_skip_agenda_order_window == 1
