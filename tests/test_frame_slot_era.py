"""Unit: `eras_in_text` khớp theo ÂM TIẾT, không phải chuỗi con.

Bộ lọc niên đại nằm trên đường truy hồi ship của `compose_2cha`
(`rank_compose_parents`), nên dương tính giả ở đây loại nhầm cha thật.
"""
from __future__ import annotations

from nn_aivie.hca.frame_slot_gen import eras_in_text


def test_khong_khop_chuoi_con_trong_am_tiet() -> None:
    """`Lê` ⊄ `Lên`, `Hồ` ⊄ `Hồng`, `Ngô` ⊄ `Ngôn` — lỗi khớp chuỗi con."""
    assert eras_in_text("Lên đường từ sớm") == set()
    assert eras_in_text("sông Hồng chảy qua") == set()
    assert eras_in_text("Ngôn ngữ học") == set()
    assert eras_in_text("Nguyên nhân chính") == {"Nguyên"}


def test_khop_am_tiet_dung() -> None:
    assert "Trần" in eras_in_text("nhà Trần đánh giặc")
    assert "Lý" in eras_in_text("thời nhà Lý")
    # dấu câu ở biên không chặn khớp
    assert "Lý" in eras_in_text("đời nhà Lý, quân dân một lòng")
    assert "Trần" in eras_in_text("(nhà Trần)")


def test_khop_niem_dai_nhieu_am_tiet() -> None:
    assert "Tây Sơn" in eras_in_text("phong trào Tây Sơn nổi dậy")
    assert "Thăng Long" in eras_in_text("dời đô về Thăng Long")
    assert "Đại Việt" in eras_in_text("nước Đại Việt ta")
    # phải liền nhau mới tính
    assert "Tây Sơn" not in eras_in_text("phía Tây có núi Sơn")


def test_phan_biet_hoa_thuong() -> None:
    """`hồ` danh từ chung không phải triều `Hồ`; `đường` không phải `Đường`."""
    assert eras_in_text("mặt hồ phẳng lặng") == set()
    assert eras_in_text("con đường làng") == set()


def test_rong_va_bien() -> None:
    assert eras_in_text("") == set()
    assert eras_in_text("   ") == set()
    assert eras_in_text("...") == set()


def test_marker_bpe_co_dau_noi_khoang_trang() -> None:
    """`eras_of_rep` truyền chuỗi còn `Ġ` — phải tách được âm tiết."""
    assert "Trần" in eras_in_text("nhàĠTrần")
    assert eras_in_text("ĠLên") == set()
