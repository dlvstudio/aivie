"""Thước `copy` của eval_discourse_prompts — phân biệt SINH THẬT với DỘI VERBATIM.

Vì sao cần: `overlap` (thước cũ) = % token PROMPT có trong output, mà `generate()` luôn
trả prompt + phần sinh → overlap LUÔN = 1.00 dù chép nguyên xi hay sinh thật (đo
2026-07-15 trên network_v10: 129/129 dòng = 1.00). Nó không thể phát hiện việc mạng đang
recite corpus — đúng thứ Trụ 4 lấy làm mục tiêu đo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from eval_discourse_prompts import CorpusIndex, _copy_frac, _stats  # noqa: E402

from nn_aivie.teaching.align import tokenize_punct  # noqa: E402

_SRC = "Khúc Thừa Dụ nhân đó nổi dậy chiếm giữ thành Đại La rồi tự xưng là Tiết độ sứ ."


def _index(*texts: str) -> CorpusIndex:
    return CorpusIndex([tokenize_punct(t) for t in texts])


def test_copy_detects_verbatim():
    assert _copy_frac(_SRC, _index(_SRC)) == 1.0


def test_copy_low_for_novel_text():
    novel = "Con mèo màu trắng thích ngủ trên ghế và ăn cá mỗi ngày"
    assert _copy_frac(novel, _index(_SRC)) < 0.35


def test_copy_partial_between():
    mixed = " ".join(_SRC.split()[:10]) + " rồi con mèo trắng nhảy qua hàng rào xanh"
    c = _copy_frac(mixed, _index(_SRC))
    assert 0.35 < c < 1.0


def test_copy_survives_dropped_punctuation():
    """Recite ĐÚNG chuỗi từ nhưng MẤT dấu câu vẫn phải = chép sạch.

    Regression thật (đo 2026-07-15, `--no-verbatim-nets` trên network_v10): A3 đọc lại
    nguyên đoạn corpus nhưng không sinh nổi `.`/`,`. Bản n-gram-liền của thước lệch pha ở
    mọi chỗ có dấu → báo copy=0.29 trong khi sự thật là 1.00, tức KHEN mạng sinh thật
    đúng lúc nó đang recite. LCS chỉ mất 1 vị trí mỗi dấu rơi → vẫn bắt được.
    """
    stripped = _SRC.replace(",", "").replace(".", "")
    assert _copy_frac(stripped, _index(_SRC)) == 1.0


def test_copy_survives_any_dropped_token():
    """Không riêng dấu câu — LCS bền với MỌI token rơi, nên thước không cần biết
    token nào 'được phép rơi'. Đó là lý do nó không cần danh sách dấu câu (Luật 0)."""
    toks = _SRC.split()
    dropped = " ".join(toks[:5] + toks[7:])  # rơi 2 token nội dung giữa câu
    assert _copy_frac(dropped, _index(_SRC)) == 1.0


def test_copy_needs_order_not_just_bag():
    """Đảo thứ tự → LCS tụt. Nếu chỉ đếm túi từ thì câu đảo vẫn 1.00 (thước hỏng)."""
    rev = " ".join(reversed(_SRC.split()))
    assert _copy_frac(rev, _index(_SRC)) < 0.5


def test_stats_copy_ignores_prompt():
    """copy chỉ tính PHẦN SINH — prompt tự nó là verbatim, tính vào thì luôn dương giả."""
    prompt = " ".join(_SRC.split()[:8])
    idx = _index(_SRC)
    out_novel = prompt + " con mèo trắng nhảy qua hàng rào xanh mỗi buổi sáng"
    out_copied = _SRC
    assert _stats(prompt, out_novel, idx)["copy"] < _stats(prompt, out_copied, idx)["copy"]


def test_no_hardcoded_punctuation_in_metric():
    """Thước KHÔNG được mang danh sách dấu câu viết tay (Luật 0).

    Lịch sử: bản n-gram phải lọc PUNCT_TOKENS mới khớp được (A3 recite nhưng bỏ dấu câu)
    → thước tự mang tri thức ngôn ngữ đi chấm mạng. LCS không cần biết token nào là dấu
    câu, nên không có danh sách nào để lệch khỏi định nghĩa của vocab.
    """
    import inspect

    import eval_discourse_prompts as m

    src = inspect.getsource(m._copy_frac) + inspect.getsource(m._lcs_len)
    for ch in (".", ",", "!", "?", ";", "…"):
        assert f'"{ch}"' not in src, f"thước hardcode dấu câu {ch!r}"
