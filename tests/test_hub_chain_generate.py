"""TK_046 §22 — chuỗi ba bước NỐI VÀO ĐƯỜNG SINH.

Suốt chín vòng `hub_spread` được đo bằng thước TRUY HỒI (top-k + chấm nhị phân)
trong khi thiết kế của tác giả là cơ chế SINH: prompt «nghịch ngợm như loài» →
phân bố token kế, `khỉ` thấp hơn `chó`/`mèo` nhưng **khác 0 và được NÂNG lên**.

Đo trên ca thử vàng + nhiễu thật, hạng của `khỉ`:

    cha=895    h479  → h9     (53×)
    cha=3069   h1560 → h22    (71×)
    cha=10287  không đổi — bước ① đã trôi sang bạn ghép từ (TK_046 §21.4)

⚠️ `hub_chain_boost=0` là MẶC ĐỊNH. 428 test cũ vẫn xanh sau khi nối, tức là
KHÔNG test nào chạy nhánh bật — đó là lý do file này tồn tại.
"""
from __future__ import annotations

import numpy as np
import pytest

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import hub_chain_proposals
from nn_aivie.hca.hub_spread import _surf


def _state_vang() -> GraphState:
    """Đúng 5 câu của tác giả — `khỉ` chưa BAO GIỜ đứng cạnh `nghịch`."""
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str) -> None:
        st.parents.add_example(
            [v.id_of(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    for c in ("chó", "mèo", "khỉ"):
        add(f"loài {c} là động vật")
    for c in ("chó", "mèo"):
        add(f"nghịch ngợm như loài {c}")
    st.parents.rebuild_hierarchical_index()
    return st


def _cfg(boost: float) -> HCAConfig:
    c = HCAConfig()
    c.hub_chain_boost = boost
    return c


def test_MAC_DINH_BAT_va_tat_thi_khong_de_cu() -> None:
    """§79 — chuỗi nay BẬT mặc định (`boost = 2.0`).

    Đo đầu-cuối trên ca sạch (`scripts/thuoc_sinh_dau_cuoi.py`), chấm PHÂN BỐ
    CUỐI sau khi bơm — 55 ca lớp CỤ THỂ:

    Đo qua ĐƯỜNG PRODUCTION (`generate()`), 55 ca lớp CỤ THỂ, token kế đúng lớp:

        boost 0 → 74,5%   ·   boost 1 → 76,4%   ·   boost 2 → 76,4%

    Chọn 1.0 vì 2.0 rối đuôi câu. Chỗ nó thật sự đóng góp là prompt LOẠI SUY:
    «nghịch ngợm như loài» TẮT → rác · BẬT → `thú` (§79).

    Vẫn canh chiều ngược: đặt 0 thì KHÔNG đề cử gì (caller cũ không đổi hành vi).
    """
    assert HCAConfig().hub_chain_boost == 1.0


def test_chuoi_de_cu_KHI_chua_tung_ke_moi() -> None:
    """`khỉ` phải có trong đề cử dù `nghịch` chưa bao giờ đứng cạnh nó."""
    st = _state_vang()
    v = st.vocab
    ctx = list(v.encode_prompt("nghịch ngợm như loài"))
    ra = hub_chain_proposals(st, ctx, _cfg(1.0))
    mat = {_surf(v, t) for t in ra}
    assert "khỉ" in mat, f"chuỗi không đề cử được `khỉ`: {sorted(mat)}"


def test_khi_KHONG_ke_nghich_trong_W_skip() -> None:
    """Chốt tiền đề: nếu `khỉ` vốn đã kề `nghịch` thì bài toán là TRA CỨU,
    không phải sáng tạo — test trên sẽ vô nghĩa."""
    from nn_aivie.hca import hub_spread as H

    st = _state_vang()
    idx = H.build_hub_spread_index(st, d_max=6, min_count=0.5, use_isg=False)
    for p in H.resolve_surface_ids(st.vocab, "nghịch"):
        for dst in idx.fwd.get(p, {}):
            assert _surf(st.vocab, dst) != "khỉ", "fixture hỏng: khỉ đã kề nghịch"


def test_boost_NANG_diem_khi() -> None:
    """Đề cử phải thật sự nâng điểm, không chỉ có mặt trong danh sách."""
    st = _state_vang()
    v = st.vocab
    ctx = list(v.encode_prompt("nghịch ngợm như loài"))
    ra = hub_chain_proposals(st, ctx, _cfg(1.0))
    khi = [t for t in ra if _surf(v, t) == "khỉ"]
    assert khi and ra[khi[0]] > 0.0

    base = np.full(len(v), 0.01, dtype=np.float64)
    moi = base.copy()
    for t, w in ra.items():
        moi[t] = moi[t] * (1.0 + 1.0 * w)
    assert moi[khi[0]] > base[khi[0]]


@pytest.mark.parametrize("boost", [0.5, 1.0, 4.0])
def test_de_cu_khong_doi_theo_boost(boost: float) -> None:
    """`boost` chỉ chỉnh ĐỘ MẠNH; tập đề cử do đồ thị quyết, phải bất biến."""
    st = _state_vang()
    ctx = list(st.vocab.encode_prompt("nghịch ngợm như loài"))
    assert set(hub_chain_proposals(st, ctx, _cfg(boost))) == set(
        hub_chain_proposals(st, ctx, _cfg(1.0)))


def test_ctx_rong_khong_no() -> None:
    assert hub_chain_proposals(_state_vang(), [], _cfg(1.0)) == {}


def test_chu_ky_ngu_canh_duoc_hoc(  ) -> None:
    """§43 — nhãn phải học được CHỮ KÝ NGỮ CẢNH từ câu sinh ra cạnh của nó.

    `via_pivot` xưa nay chỉ giữ (chủ ngữ, nhãn, pivot) và vứt phần còn lại của
    câu. §42 đo được: 8 tín hiệu suy-từ-đồ-thị đều ~0 ở bước ②; ngữ cảnh là thứ
    bị vứt, nên nó mới thêm được thông tin.
    """
    from nn_aivie.hca.hub_spread import build_hub_spread_index

    st = _state_vang()
    idx = build_hub_spread_index(st, d_max=6, min_count=0.5, use_isg=False)
    assert idx.label_ctx, "không học được chữ ký ngữ cảnh nào"
    # nhãn `động`/`vật` phải mang ngữ cảnh từ câu «loài X là động vật»
    got = {_surf(st.vocab, t) for t in idx.label_ctx}
    assert {"động", "vật"} & got, f"nhãn lớp không có chữ ký: {sorted(got)}"


def test_MAC_DINH_rank_la_phieu_x_ctx() -> None:
    """Khoá mặc định — 12 test channel ở §19 đều truyền tham số tường minh nên
    không khoá được mặc định; bẫy đó đã dính một lần (TK_046 §19)."""
    assert HCAConfig().hub_chain_rank == "phieu_x_ctx"


def test_MAC_DINH_hai_loc_phan_vi_DA_TAT() -> None:
    """§55 — hai bộ lọc phân vị (§24) cắt vào CẠNH VÀNG khi corpus lớn.

    cha=10287: `gidf_q=0,2 indeg_q=0,8` ⇒ via_pivot[khỉ] RỖNG; tắt đi thì
    `[(động, 0,95)]`. Tắt cả hai đưa ngưỡng nhiễu 3 069 → ≥20 316 cha và
    classrich ②top1 15,2% → 19,4%.

    Phân vị được dùng để TRÁNH bẫy ngưỡng-hằng-không-chuyển-quy-mô, nhưng nó
    mắc đúng bẫy đó ở dạng khác — phân bố dịch thì cùng phân vị cắt thứ khác.
    """
    # §56 ĐÍNH CHÍNH §55: lọc BẬT lại, nhưng chỉ áp cho BỂ ③ (`via_pivot_sach`),
    # bể ② lấy tất cả. §55 tắt toàn cục ⇒ ②top1 và ngưỡng nhiễu tốt lên nhưng
    # thước sinh tầng ② ĐÚNG hỏng nặng (0,290 → 0,995).
    # §57 — tác giả chọn ƯU TIÊN NĂNG LỰC: lọc TẮT. Đánh đổi đã đo ở §56
    # (sinh · ② ĐÚNG 0,290 → 0,995) và được chấp nhận có chủ ý.
    c = HCAConfig()
    assert c.hub_chain_indeg_q == 0.0
    assert c.hub_chain_gidf_q == 0.0


def test_MAC_DINH_chu_ngu_la_phai_nhat() -> None:
    """§58 ĐÍNH CHÍNH §54/§57 — `phai_nhat_danh_tu` bị TRỘI ở MỌI thước.

    §54 thêm `phai_nhat_danh_tu` để sửa `phổ biến → biến`. Nhưng đó là triệu
    chứng của HAI BỘ LỌC PHÂN VỊ; tắt lọc (§55) thì `phai_nhat` cho nhãn `động`
    đúng ở mọi mức nhiễu và tốt hơn hẳn:

        luật                ngưỡng   classrich phủ   ②top1
        phai_nhat           20 316   72,4%           46,6%
        phai_nhat_danh_tu   20 316   37,3%           30,4%

    Hai bản vá cho CÙNG một nguyên nhân; giữ bản tốt hơn.
    """
    assert HCAConfig().hub_chain_head_rule_trai == "phai_nhat"
