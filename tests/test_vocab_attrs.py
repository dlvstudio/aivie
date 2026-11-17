"""TK_012 Phase 2 — thuộc-tính-token precompute (cap_class/fold/fold_group_id)."""

from __future__ import annotations

from nn_aivie.core.vocab import Vocabulary


def test_punct_is_an_ordinary_token():
    """Dấu câu KHÔNG còn thuộc tính riêng — nó là token như mọi token (Luật 0).

    `is_punct`/`PUNCT_TOKENS` xoá 2026-07-15. Chúng tạo một vòng luẩn quẩn: dấu câu bị loại
    khỏi `match_ids` → `_gdf['.']`=0 → `_gidf('.')`=7.17 = CAO NHẤT corpus (gấp 6× 'Việt'
    =1.17) → mạng coi dấu chấm là token phân biệt nhất đời — rồi ~24 filter `is_punct` rải
    khắp inference tồn tại để CHE hậu quả đó. Đếm dấu câu vào thống kê thì '.' có df≈n →
    gidf≈1.10 → tự chìm, không cần danh sách nào nói hộ.
    """
    v = Vocabulary()
    assert not hasattr(v, "is_punct")


def test_each_punct_is_its_own_fold_group():
    """Mỗi dấu câu là MỘT fold-group riêng — nhưng `Việt.` vẫn ≡ `việt`.

    Bug cũ (ẩn vì `is_punct` lọc trước khi tới đây): `FOLD_STRIP` strip sạch dấu câu → mọi
    dấu câu fold về chuỗi RỖNG → CÙNG một fold-group, tức với mạng `.` `,` `?` `:` là MỘT
    token. Gỡ `is_punct` thì fold `''` tràn vào `_query_content_folds` →
    `_longest_suffix_cut` khớp trúng mọi dấu câu trong source → cắt hết câu → agenda RỖNG.
    """
    v = Vocabulary()
    assert v.fold_group_id(v.id_of(".")) != v.fold_group_id(v.id_of("?"))
    assert v.fold_group_id(v.id_of(",")) != v.fold_group_id(v.id_of(":"))
    assert v.fold_group_id(v.id_of("Việt.")) == v.fold_group_id(v.id_of("việt"))
    assert v.fold_group_id(v.id_of("(học)")) == v.fold_group_id(v.id_of("học"))


def test_cap_class():
    v = Vocabulary()
    assert v.cap_class(v.id_of("mèo")) == 0
    assert v.cap_class(v.id_of("Mèo")) == 1
    assert v.cap_class(v.id_of("USA")) == 2
    assert v.cap_class(v.id_of("123")) == 0  # không có chữ cái


def test_fold_and_group_equivalence():
    """'Phở'≡'phở'≡'phở.' cùng fold-group; token khác nhóm khác id."""
    v = Vocabulary()
    a = v.id_of("Phở")
    b = v.id_of("phở")
    c = v.id_of("phở.")
    d = v.id_of("bún")
    assert v.fold(a) == "phở" == v.fold(b)
    assert v.fold_group_id(a) == v.fold_group_id(b) == v.fold_group_id(c)
    assert v.fold_group_id(a) != v.fold_group_id(d)


def test_attrs_survive_reload_order():
    """Vocab rebuild theo thứ tự id_of khi load → thuộc tính recompute nhất quán."""
    v = Vocabulary()
    for t in ["Mèo", "rất", "nhanh", ".", "?"]:
        v.id_of(t)
    v2 = Vocabulary()
    for t in v._id_to_token:
        v2.id_of(t)
    for i in range(len(v)):
        assert v.cap_class(i) == v2.cap_class(i)
        assert v.fold(i) == v2.fold(i)
