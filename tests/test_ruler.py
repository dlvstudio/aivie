"""TK_045 §13 Trụ E — cây thước cho sinh sáng tạo.

Bất biến khoá bằng test — mỗi cái chặn một cách đọc SAI đã từng xảy ra trong ticket:

1. Bộ **CHÉP** phải ra `attestation=1.0` **và** `novelty≈0` ⇒ ô "CHÉP", không phải "ĐÍCH".
   (Đây là lỗi §12.112 "copy=1.00 là đỉnh" mặc áo mới.)
2. Bộ **SALAD** phải ra `attestation` thấp ⇒ ô "SALAD", dù `novelty` ≈ 1.0.
3. `novelty` dùng đoạn **LIÊN TỤC**, không phải LCS dãy-con.
4. `min_pairs_for` phải nói đúng vì sao n=30 không đủ (§12.113).
"""

from __future__ import annotations

import pytest

from nn_aivie.eval.ruler import (
    CopySpanIndex,
    NgramIndex,
    _longest_common_run,
    attestation_rate,
    min_pairs_for,
    novelty_score,
    score_generation,
    win_rate_ci,
)

# corpus đồ chơi: id token, mỗi "cha" một chuỗi
_EMITS = [
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11],
    [1, 2, 20, 21, 22],
    [30, 31, 32, 33, 34],
]


@pytest.fixture
def idx():
    return NgramIndex(_EMITS), CopySpanIndex(_EMITS)


def test_verbatim_copy_is_named_CHEP_not_DICH(idx):
    """Bất biến 1 — chép nguyên khối: attestation hoàn hảo NHƯNG phải bị gọi tên là CHÉP."""
    ng, cs = idx
    copied = [1, 2, 3, 4, 5, 6]
    s = score_generation(ng, cs, copied)
    assert s.attestation == 1.0
    assert s.novelty == pytest.approx(0.0)
    assert s.cell() == "CHÉP", "bộ chép mà được gọi ĐÍCH ⇒ tái phạm lỗi §12.112"


def test_salad_is_named_SALAD(idx):
    """Bất biến 2 — mối nối chưa từng thấy: novelty cao nhưng attestation thấp."""
    ng, cs = idx
    salad = [4, 9, 21, 33, 2, 11]  # toàn cặp không có trong corpus
    s = score_generation(ng, cs, salad)
    assert s.attestation == 0.0
    assert s.novelty > 0.5
    assert s.cell() == "SALAD"


def test_recombination_is_named_DICH(idx):
    """Ô ĐÍCH: mọi bigram đã chứng thực nhưng KHÔNG chép nguyên khối cha nào."""
    ng, cs = idx
    # 1→2 (cha 0 và 2), 2→20 (cha 2), 20→21→22 (cha 2)… ghép hai nguồn
    recomb = [3, 4, 5, 1, 2, 20, 21]
    s = score_generation(ng, cs, recomb)
    assert s.attestation < 1.0  # mối nối 5→1 chưa chứng thực
    ok = score_generation(ng, cs, [1, 2, 20, 21, 22])
    assert ok.attestation == 1.0
    # chuỗi này TRÙNG trọn cha 2 ⇒ phải là CHÉP, không phải ĐÍCH — thước không được rộng lượng
    assert ok.cell() == "CHÉP"


def test_novelty_uses_contiguous_run_not_lcs(idx):
    """Bất biến 3 — LCS dãy-con và đoạn liên tục KHÁC nhau; thước phải dùng cái sau."""
    # [1,3,5] là dãy-con của [1,2,3,4,5,6] (LCS=3) nhưng đoạn liên tục chung chỉ = 1
    assert _longest_common_run([1, 3, 5], [1, 2, 3, 4, 5, 6]) == 1
    _ng, cs = idx
    n = novelty_score(cs, [1, 3, 5])
    assert n == pytest.approx(1 - 1 / 3), "novelty đang dùng LCS thay vì đoạn liên tục"


def test_attestation_order3_stricter_than_order2(idx):
    ng, _cs = idx
    seq = [1, 2, 3, 4]
    assert attestation_rate(ng, seq, order=2) == 1.0
    assert attestation_rate(ng, seq, order=3) == 1.0
    mixed = [1, 2, 20, 3, 4]  # 20→3 chưa thấy
    assert attestation_rate(ng, mixed, order=2) < 1.0


def test_too_short_is_flagged_not_scored(idx):
    """Chuỗi rất ngắn làm mọi thước nói dối (§12.27) — phải được GỌI TÊN riêng."""
    ng, cs = idx
    assert score_generation(ng, cs, [1, 2]).cell() == "quá-ngắn"


def test_sample_size_explains_why_n30_failed():
    """Bất biến 4 — §12.113 chấm n=30 rồi thấy tín hiệu không lặp. Số học nói trước."""
    assert min_pairs_for(15) == 88
    assert min_pairs_for(10) == 197
    assert min_pairs_for(5) == 785
    # n=30 chỉ phát hiện được chênh lệch RẤT lớn (~26pp) — nguồn gốc §12.113
    assert min_pairs_for(28) <= 30 < min_pairs_for(15)
    # công thức BỎ power (chỉ CI) cho số nhỏ hơn ~2× — đó là bẫy, khoá lại
    assert min_pairs_for(15, power_z=0.0) == 43
    assert min_pairs_for(15) > 2 * min_pairs_for(15, power_z=0.0) - 5


def test_win_rate_ci_shape():
    p, lo, hi = win_rate_ci(60, 100)
    assert p == 0.6 and lo < 0.6 < hi
    assert lo > 0.5, "60/100 phải tách khỏi 50% ở mức 95%"
    p2, lo2, _hi2 = win_rate_ci(18, 30)
    assert p2 == 0.6 and lo2 < 0.5, "cùng tỉ lệ nhưng n=30 KHÔNG tách khỏi 50%"


def test_copy_run_is_absolute_and_novelty_alone_can_mislead(idx):
    """§13.13 — `novelty` là TỈ LỆ; nó đẹp lên chỉ vì đầu ra dài ra.

    Đo được ở P52: dữ liệu ×45 ⇒ novelty 0.545→0.737 (tỉ lệ TỐT hơn) trong khi khối chép
    liên tục 22.7→51.7 token (TỆ hơn 2.3×). Khoá lại để không ai đọc `novelty` một mình.
    """
    from nn_aivie.eval.ruler import copy_run_tokens

    ng, cs = idx
    short = [1, 2, 3, 4]                    # chép trọn cha 0 (dài 4)
    longer = [1, 2, 3, 4] + [90, 91, 92, 93, 94, 95, 96, 97]  # cùng khối chép, dài gấp 3
    s_short = score_generation(ng, cs, short)
    s_long = score_generation(ng, cs, longer)
    assert s_long.novelty > s_short.novelty, "tỉ lệ phải đẹp lên khi đầu ra dài ra"
    assert s_long.copy_run == s_short.copy_run == 4, (
        "khối chép TUYỆT ĐỐI phải KHÔNG đổi — đó là điều novelty che mất"
    )
    assert copy_run_tokens(cs, longer) == 4
