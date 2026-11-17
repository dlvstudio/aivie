"""Tests cổng chất lượng câu tiếng Việt (data/quality.py)."""

from __future__ import annotations

from nn_aivie.data.quality import (
    is_quality_vi_sentence,
    split_sentences,
    vi_diacritic_ratio,
)


def test_accepts_clean_vietnamese_facts():
    for s in [
        "Bác sĩ chữa bệnh cho mọi người.",
        "Hà Nội là thủ đô của nước Việt Nam.",
        "Mặt trời mọc ở hướng đông.",
    ]:
        assert is_quality_vi_sentence(s)


def test_rejects_noise():
    for s in [
        "Champions League bảng B vòng 3",          # không kết câu, không dấu
        "The cat is on the table.",                # tiếng Anh
        "Xem thêm | Sửa | [[Liên kết]]",            # markup
        "2,86% 0,33% S&P 500 index hôm nay.",      # số tràn
        "ngủ.",                                      # quá ngắn
        "https://example.com là trang web.",        # URL
        "Trần Hưng Đạo (chữ Nho : 陳興道 ; ?).",   # dòng từ điển / chú giải
    ]:
        assert not is_quality_vi_sentence(s)


def test_split_sentences():
    out = split_sentences("Hà Nội là thủ đô. Sông Hồng chảy qua! Có mấy cầu?")
    assert out == ["Hà Nội là thủ đô.", "Sông Hồng chảy qua!", "Có mấy cầu?"]


def test_diacritic_ratio_distinguishes_languages():
    assert vi_diacritic_ratio("Việt Nam tươi đẹp") > 0.1
    assert vi_diacritic_ratio("hello world today") == 0.0
