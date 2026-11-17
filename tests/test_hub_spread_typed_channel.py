"""Kênh TYPED (`A·là·B`) phải bất biến theo nhiễu đồng hiện.

Bệnh gốc làm `hub_spread` ĐẠT trên synthetic mà CÂM trên ckpt thật
(TK_046 §18.H): hai kênh khác THANG ĐO bị cộng vào một điểm.

- typed = đếm câu định nghĩa → **không** tăng theo cỡ corpus
- cooc  = đồng hiện W_skip   → tăng tuyến tính theo cỡ corpus

Một câu đời thường lặp đủ nhiều (`chó nhà hàng xóm sủa to`) chôn
`chó là động vật` dưới nhiều bậc độ lớn; hệ số nhân hằng nào cũng thua.
"""
from __future__ import annotations

import pytest

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.hub_spread import (
    _FUNC,
    _surf,
    _surf_ix,
    build_hub_spread_index,
    resolve_surface_ids,
    reverse_peers,
    score_hubs_from,
)

# Ví dụ gốc của tác giả: nhãn `động vật` nổi lên từ ba câu định nghĩa.
_DINH_NGHIA = [
    "loài chó là động vật",
    "loài mèo là động vật",
    "loài khỉ là động vật",
]
_THUOC_TINH = [
    "nghịch ngợm như loài chó",
    "nghịch ngợm như loài mèo",
]
# Câu đời thường — hợp lệ, nhưng dìm cạnh định nghĩa nếu cộng chung thang đo.
_NHIEU = [
    "chó nhà hàng xóm sủa to",
    "mèo hay nằm trên mái nhà",
    "thời tiết hôm nay rất đẹp",
    "máy tính xử lý dữ liệu nhanh",
]


def _state(n_nhieu: int) -> GraphState:
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str) -> None:
        st.parents.add_example(
            [v.id_of(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    for s in _DINH_NGHIA + _THUOC_TINH:
        for _ in range(3):
            add(s)
    for i in range(n_nhieu):
        add(_NHIEU[i % len(_NHIEU)])
    st.parents.rebuild_hierarchical_index()
    return st


def _hubs(st: GraphState, channel: str, topk: int = 5):
    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    seeds = resolve_surface_ids(st.vocab, "chó")
    assert seeds
    return idx, seeds, score_hubs_from(
        st, idx, seeds, topk=topk, gidf_min=0.0, ban_func=True, channel=channel,
    )


@pytest.mark.parametrize("n_nhieu", [0, 200, 2000])
def test_typed_first_giu_nhan_bat_ke_nhieu(n_nhieu: int) -> None:
    """`động` (từ `chó là động vật`) phải đứng đầu ở MỌI mức nhiễu."""
    st = _state(n_nhieu)
    _idx, _seeds, hubs = _hubs(st, "typed_first")
    assert hubs, "phải có hub"
    assert hubs[0].typed is True
    # neo cụm ISG ⇒ nhãn nguyên khối `động vật`, không phải mảnh `động`
    assert hubs[0].surface == "động vật"


@pytest.mark.parametrize("n_nhieu", [0, 200, 2000])
def test_diem_typed_khong_doi_theo_nhieu(n_nhieu: int) -> None:
    """Đếm câu định nghĩa không phụ thuộc lượng văn bản không liên quan."""
    st = _state(n_nhieu)
    _idx, _seeds, hubs = _hubs(st, "typed")
    dong = [h for h in hubs if h.surface == "động vật"]
    assert dong, "kênh typed phải thấy `động vật`"
    assert dong[0].weight == pytest.approx(3.0)


def test_cooc_BI_nhieu_lan_at() -> None:
    """Đối chứng: kênh đồng hiện thuần ĐỔI ngôi khi thêm nhiễu.

    Đây là bằng chứng bệnh có thật — nếu test này hết đỏ nghĩa là fixture
    không còn tái lập được hiện tượng, phải sửa fixture chứ không phải mừng.
    """
    _i, _s, sach = _hubs(_state(0), "cooc")
    _i, _s, ban = _hubs(_state(2000), "cooc")
    # `động`/`vật` cùng thuộc cụm nhãn nên hoà điểm — chấp nhận cả hai.
    nhan = {"động", "vật", "động vật"}
    assert sach[0].surface in nhan
    assert ban[0].surface not in nhan
    assert ban[0].weight > sach[0].weight * 10


def test_mixed_la_hanh_vi_CU_va_van_thua() -> None:
    """`mixed` (cộng điểm, boost 8×) giữ để A/B — và nó vẫn thua ở nhiễu cao."""
    _i, _s, hubs = _hubs(_state(2000), "mixed")
    assert hubs[0].surface != "động"


def test_typed_first_van_ra_duoc_khi_KHONG_co_canh_typed() -> None:
    """Không có câu `X là Y` thì rơi về đồng hiện, không trả rỗng."""
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()
    # `chó` phải có HẬU TỐ để tồn tại cạnh đồng hiện, và không câu nào có
    # pivot `là`/`như` bắt đầu từ `chó` ⇒ kênh typed rỗng đúng ý test.
    for _ in range(3):
        for s in ("loài chó chạy rất nhanh", "loài chó sủa rất to"):
            st.parents.add_example(
                [v.id_of(t) for t in (s + " <EOS>").split()],
                v, dap_id=None, defer_index=True,
            )
    st.parents.rebuild_hierarchical_index()
    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    seeds = resolve_surface_ids(v, "chó")
    hubs = score_hubs_from(
        st, idx, seeds, topk=5, gidf_min=0.0, ban_func=True, channel="typed_first",
    )
    assert hubs
    assert all(h.typed is False for h in hubs)


def test_buoc_ba_lan_nguoc_ra_thanh_vien_chua_thay() -> None:
    """Bước sáng tạo: từ nhãn lan ngược phải ra `khỉ` — chưa từng đứng cạnh `nghịch`."""
    st = _state(2000)
    idx, seeds, hubs = _hubs(st, "typed_first")
    peers = reverse_peers(st, idx, [hubs[0].tid], exclude=set(seeds), topk=10)
    surfs = {p.surface for p in peers}
    assert "khỉ" in surfs, f"mất bước sáng tạo; peers={sorted(surfs)}"


def test_prefer_via_pivot_van_tuong_thich_nguoc() -> None:
    st = _state(0)
    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    seeds = resolve_surface_ids(st.vocab, "chó")
    cu = score_hubs_from(
        st, idx, seeds, topk=5, gidf_min=0.0, ban_func=True, prefer_via_pivot=True,
    )
    assert all(h.typed is False for h in cu)  # cờ cũ ⇒ chế độ `mixed`


def test_MAC_DINH_phai_la_typed_first() -> None:
    """Khoá MẶC ĐỊNH, không chỉ khoá chế độ truyền tay.

    `spread_analogy` gọi `score_hubs_from` không truyền `channel`; nếu mặc định
    tụt về `mixed` thì bệnh quay lại mà mọi test truyền-tường-minh vẫn xanh.
    """
    st = _state(2000)
    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    seeds = resolve_surface_ids(st.vocab, "chó")
    hubs = score_hubs_from(st, idx, seeds, topk=5, gidf_min=0.0, ban_func=True)
    assert hubs[0].typed is True
    assert hubs[0].surface == "động vật"


def _state_co_mot(n_nhieu: int = 0) -> GraphState:
    """Corpus viết kiểu `X là MỘT động vật` — dạng phổ biến ở v12."""
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str) -> None:
        st.parents.add_example(
            [v.id_of(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    for s in ("loài chó là một động vật",
              "loài mèo là một động vật",
              "loài khỉ là một động vật"):
        for _ in range(3):
            add(s)
    for s in _THUOC_TINH:
        for _ in range(3):
            add(s)
    # `một` phải PHỔ BIẾN TOÀN CORPUS thì gidf mới dìm được nó — đúng như thực
    # tế (trên v12 `một` có bậc-vào 2846). Không có phần này thì `một`, `động`,
    # `vật` cùng df ⇒ gidf hoà ⇒ chọn trung tâm bất khả.
    for s in ("đây là một ngày đẹp trời",
              "tôi mua một quyển sách mới",
              "có một chiếc xe ngoài sân",
              "chỉ một lần thôi nhé bạn"):
        for _ in range(6):
            add(s)
    for i in range(n_nhieu):
        add(_NHIEU[i % len(_NHIEU)])
    st.parents.rebuild_hierarchical_index()
    return st


def test_nhan_khong_duoc_la_MOT() -> None:
    """`X là một động vật` ⇒ nhãn phải là `động`, KHÔNG phải `một`.

    Bệnh đo được trên v12: 11/15 nhãn mạnh nhất là hư từ, `một` đứng đầu
    (bậc-vào 2846) — vì bản cũ lấy token LIỀN KỀ sau pivot.
    """
    st = _state_co_mot()
    seeds = resolve_surface_ids(st.vocab, "chó")

    cu = build_hub_spread_index(
        st, d_max=3, min_count=2, head_lookahead=1, use_isg=False)
    h_cu = score_hubs_from(
        st, cu, seeds, topk=3, gidf_min=0.0, ban_func=False, channel="typed")
    assert h_cu and h_cu[0].surface == "một", "fixture không tái lập được bệnh"

    moi = build_hub_spread_index(st, d_max=3, min_count=2)  # lookahead mặc định
    h_moi = score_hubs_from(
        st, moi, seeds, topk=3, gidf_min=0.0, ban_func=False, channel="typed")
    assert h_moi and h_moi[0].surface == "động vật", (
        "neo cụm ISG phải cho nhãn NGUYÊN KHỐI, không phải mảnh")


def test_buoc3_typed_ra_THANH_VIEN_khong_ra_hang_xom() -> None:
    """`thú`→thành viên phải là con vật, không phải từ hay đứng gần `thú`.

    Trên v12 bản cũ: `mèo`→hub `thú` (đúng) → thành viên
    ['hứng','yêu','thích',…] = đồng hiện, không phải "ai được gọi là thú".
    """
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str) -> None:
        st.parents.add_example(
            [v.id_of(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    for _ in range(3):
        for s in ("chó là thú", "mèo là thú", "khỉ là thú"):
            add(s)
    # nhiễu: từ hay ĐỨNG TRƯỚC `thú` nhưng chưa bao giờ «là thú»
    for _ in range(300):
        add("câu chuyện thú vị lắm")
    st.parents.rebuild_hierarchical_index()

    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    hub = resolve_surface_ids(v, "thú")
    assert hub

    cu = [p.surface for p in reverse_peers(
        st, idx, hub, topk=6, channel="mixed")]
    moi = [p.surface for p in reverse_peers(
        st, idx, hub, topk=6, channel="typed_first")]

    con_vat = {"chó", "mèo", "khỉ"}
    assert not (set(cu[:3]) <= con_vat), f"fixture không tái lập bệnh: {cu}"
    assert set(moi[:3]) == con_vat, f"bước 3 vẫn lẫn đồng hiện: {moi}"
    assert all(p.typed for p in reverse_peers(
        st, idx, hub, topk=3, channel="typed_first"))


def test_MAC_DINH_reverse_peers_cung_phai_typed_first() -> None:
    """Khoá MẶC ĐỊNH của BƯỚC 3 — không chỉ chế độ truyền tay.

    `spread_analogy` gọi `reverse_peers` không truyền `channel`; mặc định tụt về
    `mixed` là bệnh quay lại mà mọi test truyền-tường-minh vẫn xanh.
    """
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str) -> None:
        st.parents.add_example(
            [v.id_of(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    for _ in range(3):
        for s in ("chó là thú", "mèo là thú", "khỉ là thú"):
            add(s)
    for _ in range(300):
        add("câu chuyện thú vị lắm")
    st.parents.rebuild_hierarchical_index()

    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    hub = resolve_surface_ids(v, "thú")
    peers = reverse_peers(st, idx, hub, topk=3)  # KHÔNG truyền channel
    assert {p.surface for p in peers} == {"chó", "mèo", "khỉ"}
    assert all(p.typed for p in peers)


def test_neo_nhan_bang_cum_ISG_nguyen_khoi() -> None:
    """Nhãn phải là CỤM, không phải mảnh — `tình thế` ≠ `tình`.

    Ca v12 thật (`out/hub3_blind_v12_samples.txt` H002): câu train
    «vũng lầy là một tình thế khó khăn» cho cạnh `lầy → tình`. Cắt `tình thế`
    thành `tình` không chỉ MẤT thông tin — nó nhảy sang chùm nghĩa khác và
    đông hơn (`tình yêu` 469, `tình hình` 613 vs `tình thế` 287 trong corpus).
    """
    st = _state_co_mot()
    seeds = resolve_surface_ids(st.vocab, "chó")

    cu = build_hub_spread_index(
        st, d_max=3, min_count=2, head_lookahead=1, use_isg=False)
    h_cu = score_hubs_from(
        st, cu, seeds, topk=1, gidf_min=0.0, ban_func=False, channel="typed")
    assert h_cu[0].surface == "một", "fixture không tái lập được bệnh"

    moi = build_hub_spread_index(st, d_max=3, min_count=2)
    h_moi = score_hubs_from(
        st, moi, seeds, topk=1, gidf_min=0.0, ban_func=False, channel="typed")
    assert h_moi[0].surface == "động vật"
    assert " " in h_moi[0].surface, "nhãn phải đa token"
    assert h_moi[0].tid >= len(st.vocab), "cụm phải là id ảo"
    assert moi.label_span[h_moi[0].tid] == tuple(
        resolve_surface_ids(st.vocab, "động")[:1]
        + resolve_surface_ids(st.vocab, "vật")[:1])


def test_gidf_cum_khong_dung_power() -> None:
    """gidf cụm phải cùng MẪU SỐ với gidf token (số cha), không phải `power`.

    `power` đếm LƯỢT xuất hiện; `_gdf` đếm SỐ CHA. Trộn hai thứ làm cụm bị dìm
    giả tạo — đo được: `động vật` 1,18 < `động` 1,41 dù cùng phân bố.
    """
    st = _state(0)
    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    vid = next(k for k, s in idx.label_surf.items() if s == "động vật")
    from nn_aivie.hca.hub_spread import _gidf
    parts = [_gidf(st, t) for t in idx.label_span[vid]]
    assert idx.label_gidf[vid] == pytest.approx(max(parts))
    assert idx.label_gidf[vid] >= max(parts) - 1e-9


def test_quet_chong_lan_khong_an_mat_cum_dung() -> None:
    """Quét tham lam `một động` rồi nhảy qua sẽ giấu mất `động vật`."""
    st = _state_co_mot()
    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    surfs = set(idx.label_surf.values())
    assert "động vật" in surfs
    seeds = resolve_surface_ids(st.vocab, "chó")
    h = score_hubs_from(
        st, idx, seeds, topk=3, gidf_min=0.0, ban_func=False, channel="typed")
    assert h[0].surface == "động vật", f"bị `một động` ăn mất: {[x.surface for x in h]}"


def test_channel_la_thi_no() -> None:
    st = _state(0)
    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    with pytest.raises(ValueError, match="channel"):
        score_hubs_from(st, idx, [0], channel="khong_ton_tai")


def test_CHUOI_BA_BUOC_day_du_ra_duoc_khi() -> None:
    """Ví dụ gốc của tác giả, chạy trọn ba bước, ở mức nhiễu 2000.

    ① «nghịch» → {loài, chó, mèo}   thuộc tính chọn tập hạt giống
    ② tập      → «động vật»          nhãn CHUNG (bỏ phiếu ≥2), cụm ISG nguyên khối
    ③ nhãn     → «khỉ»               MỚI: chưa từng đứng cạnh «nghịch»

    Vòng đo `sample_hub_step3_blind.py` BỎ MẤT bước ① (xuất phát từ một seed
    đơn lẻ) nên kết quả 10,7% của nó không kiểm định thiết kế này.

    Hai bẫy đã vấp, test khoá luôn:
    - `d_max` phải ≥4: trong «nghịch ngợm như loài chó» thì `chó` ở d=4.
    - `ngợm` (nửa kia của `nghịch ngợm`) phải bị loại khỏi hạt giống.
    """
    st = _state(2000)
    v = st.vocab
    idx = build_hub_spread_index(st, d_max=6, min_count=2)

    cung: dict[int, set[int]] = {}
    for c in st.parents.seg.clusters:
        sp = tuple(int(t) for t in c.token_span())
        for t in sp:
            cung.setdefault(t, set()).update(sp)

    p = resolve_surface_ids(v, "nghịch")[0]
    ban = cung.get(p, set()) | {p}
    nb = sorted(idx.fwd.get(p, {}).items(), key=lambda kv: -kv[1][0])
    hat = [
        int(t) for t, _w in nb
        if (s := _surf(v, t)) and len(s) > 1
        and s not in _FUNC and int(t) not in ban
    ][:6]
    assert "ngợm" not in [_surf(v, t) for t in hat], "nửa kia của từ lọt vào"
    assert {"chó", "mèo"} <= {_surf(v, t) for t in hat}

    phieu: dict[int, float] = {}
    for t in hat:
        for h in score_hubs_from(
            st, idx, [t], topk=3, gidf_min=0.0, ban_func=True,
            channel="typed_first",
        ):
            phieu[h.tid] = phieu.get(h.tid, 0.0) + 1.0
    hub = max(phieu.items(), key=lambda kv: (kv[1], kv[0]))
    assert hub[1] >= 2, "nhãn phải được ≥2 hạt giống cùng bầu"
    assert _surf_ix(idx, v, hub[0]) == "động vật"

    peers = reverse_peers(
        st, idx, [hub[0]], exclude=set(hat), topk=10, channel="typed_first")
    moi = {x.surface for x in peers if x.tid not in set(hat)}
    assert "khỉ" in moi, f"mất bước sáng tạo; MỚI={sorted(moi)}"


def _state_liet_ke() -> GraphState:
    """Câu THẬT từ corpus v12 — khuôn liệt kê `<LỚP> là A, B và C`."""
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str) -> None:
        st.parents.add_example(
            [v.id_of(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    for _ in range(4):
        add("Bốn đài truyền hình lớn nhất tại Hoa Kỳ là NBC , CBS , ABC và Fox")
        add("Ba loài thú phổ biến ở Việt Nam là chó , mèo và trâu")
    # hút-điểm `là vậy` từ truyện mạng (54 lần thật trong conv_stories)
    for _ in range(40):
        add("thì ra là vậy mấy đại thế lực đều có")
    st.parents.rebuild_hierarchical_index()
    return st


def test_khuon_liet_ke_doc_dung_chieu() -> None:
    """`<LỚP> là A, B, C` — lớp bên TRÁI. Khuôn `A là B` đọc NGƯỢC.

    Ca v12 thật (C008): «Bốn đài truyền hình lớn nhất tại Hoa Kỳ là NBC, CBS,
    ABC và Fox» sinh ra cạnh thành-viên→thành-viên (`NBC → Fox`) rồi lấy `Fox`
    làm nhãn. Khuôn liệt kê đếm được 554 lần trong corpus.
    """
    st = _state_liet_ke()
    v = st.vocab
    idx = build_hub_spread_index(st, d_max=6, min_count=2)

    def _lop(w: str) -> set[str]:
        out: set[str] = set()
        for t in resolve_surface_ids(v, w):
            for k in idx.via_pivot.get(t, {}):
                out.add(_surf_ix(idx, v, k))
        return out

    # bốn đài phải trỏ về CÙNG một nút lớp
    lops = [_lop(w) for w in ("NBC", "CBS", "ABC", "Fox")]
    assert all(l for l in lops), f"thiếu cạnh: {lops}"
    chung = set.intersection(*lops)
    assert len(chung) == 1, f"không hội tụ về một lớp: {lops}"
    # và KHÔNG được trỏ vào nhau (đó là bệnh đọc ngược)
    assert not (chung & {"NBC", "CBS", "ABC", "Fox"})

    # bước ③ từ lớp phải trả lại đủ bốn thành viên
    lop_id = [k for k, s in idx.label_surf.items() if s in chung]
    lop_id = lop_id or resolve_surface_ids(v, next(iter(chung)))
    peers = {p.surface for p in reverse_peers(
        st, idx, lop_id[:1], topk=8, channel="typed_first")}
    assert {"NBC", "CBS", "ABC", "Fox"} <= peers, f"peers={sorted(peers)}"


def test_hut_diem_hoi_chi_bi_loai_bang_PHAN_VI() -> None:
    """`là vậy` (hồi chỉ) phải rụng — bằng phân vị gidf, KHÔNG bằng danh sách từ.

    Đo v12: nhãn `vậy` hút 8/22 dòng và cho y hệt một danh sách MỚI.
    Ngưỡng HẰNG không dùng được (đã dính bẫy `bậc-vào >=3` bão hoà ở v12),
    nên lọc theo phân vị của chính phân bố gidf nhãn.
    """
    st = _state_liet_ke()
    idx = build_hub_spread_index(st, d_max=6, min_count=2)
    moi_nhan = {
        _surf_ix(idx, st.vocab, k)
        for src in idx.via_pivot for k in idx.via_pivot[src]
    }
    assert "vậy" not in moi_nhan, f"hút-điểm còn sống: {sorted(moi_nhan)}"


def test_tat_liet_ke_thi_benh_QUAY_LAI() -> None:
    """Đối chứng: `use_liet_ke=False` ⇒ bốn đài KHÔNG hội tụ về một lớp."""
    st = _state_liet_ke()
    v = st.vocab
    idx = build_hub_spread_index(
        st, d_max=6, min_count=2, use_liet_ke=False)
    lops = []
    for w in ("NBC", "CBS", "ABC", "Fox"):
        s: set[str] = set()
        for t in resolve_surface_ids(v, w):
            for k in idx.via_pivot.get(t, {}):
                s.add(_surf_ix(idx, v, k))
        lops.append(s)
    chung = set.intersection(*lops) if all(lops) else set()
    assert len(chung) != 1 or (chung & {"NBC", "CBS", "ABC", "Fox"}), (
        f"fixture không tái lập bệnh đọc ngược: {lops}")


class _FakeWS:
    """WS-CRF giả: gộp theo bảng cho trước (không cần model thật trong test)."""

    def __init__(self, groups: dict[str, int]) -> None:
        self.groups = groups

    def tag(self, syl: list[str]) -> list[str]:
        out, prev = [], None
        for s in syl:
            g = self.groups.get(s)
            out.append("I-W" if (g is not None and g == prev and out) else "B-W")
            prev = g
        return out


def test_ws_span_bo_moc_khung_va_tia_hu_tu() -> None:
    """Ba ca M4 lộ ra ở bake v12 (`252b750`).

    - `Hỏi: Tàu` — mốc khung. `is_structural_token` KHÔNG bắt (is_surface=True)
      và `_FUNC` cũng không; dấu hiệu thật là kết bằng `:`.
    - `tại Gia Lý Học Viện` — giới từ dẫn đầu. `tại` KHÔNG có trong `_FUNC`
      (đã đo), nên phải nhận bằng gidf rồi TỈA, không bỏ cả span.
    - `đảng Quốc xã` — span tốt, phải GIỮ NGUYÊN.
    """
    from nn_aivie.hca.hub_spread import ws_word_spans

    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    # `ws_word_spans` ánh xạ token→âm tiết qua MARK_SPACE (đáy BPE). Vocab
    # mức-từ không có marker ⇒ gộp cả câu thành một âm tiết. Mô phỏng đáy thật.
    def tid(w: str) -> int:
        return v.id_of(w if w == "<EOS>" else MARK_SPACE + w)

    def add(s: str) -> None:
        st.parents.add_example(
            [tid(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    # `tại` phải PHỔ BIẾN để gidf dìm nó (đúng thực tế: tại 4,47 « Gia 6,06)
    for _ in range(3):
        add("Hỏi: Tàu là gì")
        add("trường tại Gia Lý Học Viện rất lớn")
        add("đảng Quốc xã lên nắm quyền")
    # ParentStore KHỬ TRÙNG LẶP (lặp một câu 20 lần = 1 cha, power=20) ⇒ phải
    # dùng câu KHÁC NHAU thì df(`tại`) mới lên và gidf mới dìm được nó. Ngưỡng
    # 1.0 đã kiểm trên ckpt thật: tại 4,47 « Gia 6,06 (chênh 1,59).
    for i in range(24):
        add(f"nhà tại nơi số {i} rất đẹp")
    st.parents.rebuild_hierarchical_index()

    grp = {"Hỏi:": 1, "Tàu": 1,               # mốc khung dính từ
           "tại": 2, "Gia": 2, "Lý": 2,       # giới từ dẫn đầu
           "đảng": 3, "Quốc": 3, "xã": 3}     # span tốt
    spans = ws_word_spans(st, _FakeWS(grp), only_pivot_parents=False)
    got = {" ".join(_surf(v, t) for t in x) for x in spans}

    assert not any("Hỏi:" in x for x in got), f"mốc khung lọt: {got}"
    assert "tại Gia Lý" not in got, f"chưa tỉa hư từ dẫn đầu: {got}"
    assert "đảng Quốc xã" in got, f"span tốt bị mất: {got}"


def test_thanh_KHONG_con_la_pivot() -> None:
    """`thành` là âm tiết của `thành phố`/`thành công`/`hoàn thành`…

    Đếm corpus v12: trở thành 3 516 (đúng) vs thành phố 2 262 · thành công
    1 995 · hoàn thành 1 466 · thành lập 1 368 · hình thành 1 297 (sai)
    ⇒ ~8 400 sai / 3 500 đúng. Mỗi «thành phố» bị khai thác như triple phân
    loại — nguồn hút-điểm `phố` (10/28 dòng vòng 3 v12).
    """
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def tid(w: str) -> int:
        return v.id_of(w if w == "<EOS>" else MARK_SPACE + w)

    for i in range(8):
        st.parents.add_example(
            [tid(t) for t in f"nền kinh tế thành phố số {i} phục hồi <EOS>".split()],
            v, dap_id=None, defer_index=True)
    st.parents.rebuild_hierarchical_index()
    idx = build_hub_spread_index(st, d_max=6, min_count=2)
    assert not idx.via_pivot, f"`thành` vẫn sinh cạnh: {idx.via_pivot}"


def test_nhan_KHONG_duoc_chua_pivot() -> None:
    """Span cụm có thể trùm cả `là` ⇒ lớp thành `trọng khác là`.

    Tái lập trên câu thật «Các thành phố quan trọng khác LÀ Göteborg, …».
    """
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def tid(w: str) -> int:
        return v.id_of(w if w in ("<EOS>", ",") else MARK_SPACE + w)

    for i in range(6):
        st.parents.add_example(
            [tid(t) for t in (
                f"Các thành phố quan trọng khác là G{i} , M{i} , U{i} và O{i} <EOS>"
            ).split()], v, dap_id=None, defer_index=True)
    st.parents.rebuild_hierarchical_index()
    idx = build_hub_spread_index(st, d_max=6, min_count=2)
    nhan = {_surf_ix(idx, v, k) for s in idx.via_pivot for k in idx.via_pivot[s]}
    assert not [x for x in nhan if "là" in x.split()], f"nhãn dính pivot: {nhan}"


def test_ws_spans_THAY_THE_khong_tron_voi_ISG() -> None:
    """Trộn = ISG vẫn cạnh tranh và thắng bằng span rác cắt ngang từ.

    Vòng 3 v12 đã so nhầm (WS+ISG) với (ISG) — không phải (WS) với (ISG).
    """
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def tid(w: str) -> int:
        return v.id_of(w if w in ("<EOS>", ",") else MARK_SPACE + w)

    for i in range(6):
        st.parents.add_example(
            [tid(t) for t in (
                f"Các thành phố quan trọng khác là G{i} , M{i} , U{i} và O{i} <EOS>"
            ).split()], v, dap_id=None, defer_index=True)
    st.parents.rebuild_hierarchical_index()

    ws = {(tid("thành"), tid("phố")), (tid("quan"), tid("trọng"))}
    idx = build_hub_spread_index(st, d_max=6, min_count=2, ws_spans=ws)
    nhan = {_surf_ix(idx, v, k) for s in idx.via_pivot for k in idx.via_pivot[s]}
    # `phố quan` là cụm ISG cắt ngang `thành phố`/`quan trọng` — không được có
    assert "phố quan" not in nhan, f"ISG còn lẫn vào: {nhan}"


# ── head_rule: chọn TRUNG TÂM hay BỔ NGỮ sau pivot ───────────────────────────

def _state_bo_ngu() -> GraphState:
    """`X là quốc gia duy nhất …` — bổ ngữ hiếm hơn đầu lớp ⇒ max-gidf lấy nhầm.

    Bệnh đo được trên `classrich` (2026-08-12): `via_pivot` chỉ phủ 18,1% cạnh
    lớp có thật (144/797) vì `Cuba là quốc gia duy nhất` cho
    `via_pivot[Cuba] = ['duy nhất']`.
    """
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str) -> None:
        st.parents.add_example(
            [v.id_of(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    for s in ("Cuba là quốc gia duy nhất",
              "Canada là quốc gia rộng lớn",
              "Angkor là quốc gia cổ xưa"):
        for _ in range(3):
            add(s)
    # `quốc gia` phải PHỔ BIẾN hơn bổ ngữ thì mới tái lập được bệnh — đúng như
    # corpus thật (đầu lớp luôn thường gặp hơn bổ ngữ của nó).
    for s in ("mỗi quốc gia có luật riêng",
              "nhiều quốc gia đã ký kết",
              "hai quốc gia giáp biên giới"):
        for _ in range(6):
            add(s)
    st.parents.rebuild_hierarchical_index()
    return st


def _nhan_dau(st: GraphState, rule: str) -> str:
    idx = build_hub_spread_index(
        st, d_max=3, min_count=2, use_isg=False, head_rule=rule)
    h = score_hubs_from(
        st, idx, resolve_surface_ids(st.vocab, "Cuba"),
        topk=3, gidf_min=0.0, ban_func=False, channel="typed")
    return h[0].surface if h else ""


def test_max_gidf_lay_nham_BO_NGU() -> None:
    """Bằng chứng bệnh có thật. Test này hết đỏ = fixture hỏng, phải sửa fixture."""
    assert _nhan_dau(_state_bo_ngu(), "max_gidf") == "duy", (
        "fixture không tái lập được bệnh bổ-ngữ-thắng-đầu-lớp")


def test_trai_du_dac_thu_lay_dung_DAU_LOP() -> None:
    """Luật thắng phép đo phủ cạnh lớp: 28,6% vs 18,1% của `max_gidf`."""
    assert _nhan_dau(_state_bo_ngu(), "trai_du_dac_thu") == "quốc"


def test_head_rule_la_phai_bao_loi() -> None:
    import pytest
    with pytest.raises(ValueError, match="head_rule"):
        build_hub_spread_index(_state_bo_ngu(), head_rule="khong_co_that")


# ── head_rule_trai + rank="phieu": điểm gãy 895 → 3069 cha ───────────────────

def _state_luong_tu() -> GraphState:
    """`loài chó là động vật` + nhiễu — LƯỢNG TỪ `loài` cướp chỗ THÀNH VIÊN.

    Bệnh đo được bằng đường cong nhiễu (`probe_chain3_noise_curve.py`):

        cha=895   via_pivot[chó] = [('vật', 0.95)]   ✓
        cha=3069  via_pivot[chó] = []                ✗
                  via_pivot[loài] = [('vật', 1.90)]

    gidf `loài` vs `chó` đổi vị trí khi corpus lớn ⇒ chủ ngữ chuyển sang lượng từ.
    """
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str, lap: int = 1) -> None:
        for _ in range(lap):
            st.parents.add_example(
                [v.id_of(t) for t in (s + " <EOS>").split()],
                v, dap_id=None, defer_index=True,
            )

    for c in ("chó", "mèo", "khỉ"):
        add(f"loài {c} là động vật", 3)
    # `chó` xuất hiện thêm ở nhiều ngữ cảnh khác ⇒ gidf tụt xuống dưới `loài`.
    for s in ("con chó chạy ngoài sân", "chó sủa suốt đêm qua",
              "nhà bên nuôi chó dữ", "chó con mới sinh hôm qua"):
        add(s, 4)
    st.parents.rebuild_hierarchical_index()
    return st


def _via(st: GraphState, rule: str, tu: str) -> list[str]:
    idx = build_hub_spread_index(
        st, d_max=6, min_count=0.5, use_isg=False, head_rule_trai=rule)
    out: list[str] = []
    for i in resolve_surface_ids(st.vocab, tu):
        for dst in idx.via_pivot.get(i, {}):
            out.append(_surf_ix(idx, st.vocab, dst))
    return out


def test_max_gidf_trai_DANH_MAT_canh_vang() -> None:
    """Bằng chứng bệnh. Hết đỏ = fixture hỏng, phải sửa fixture chứ không mừng."""
    st = _state_luong_tu()
    assert not ({"động", "vật"} & set(_via(st, "max_gidf", "chó"))), (
        "fixture không tái lập được bệnh lượng-từ-cướp-chỗ")


def test_phai_nhat_PHUC_HOI_canh_vang() -> None:
    """`phai_nhat` XÁC ĐỊNH — không phụ thuộc phân vị nên khoá được ở fixture.

    Trên dữ liệu thật `phai_du_dac_thu` mới là luật thắng (nó chặn thêm
    `chính là`/`cũng là`); ở fixture tí hon phân vị không hiệu chỉnh nổi, nên
    đừng khoá nó ở đây — bằng chứng của nó là đường cong nhiễu, TK_046 §21.
    """
    st = _state_luong_tu()
    # Không neo cụm ⇒ nhãn là MẢNH của `động vật`; cả hai mảnh đều hợp lệ.
    for tu in ("chó", "mèo", "khỉ"):
        assert {"động", "vật"} & set(_via(st, "phai_nhat", tu)), (
            f"mất cạnh vàng: {tu}")


def test_rank_phieu_chon_DONG_THUAN_khong_chon_diem() -> None:
    """3 hạt giống bầu `vật` phải thắng 1 hạt giống bầu cạnh nặng gấp đôi."""
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str, lap: int = 1) -> None:
        for _ in range(lap):
            st.parents.add_example(
                [v.id_of(t) for t in (s + " <EOS>").split()],
                v, dap_id=None, defer_index=True,
            )

    for c in ("chó", "mèo", "khỉ"):
        add(f"{c} là động vật", 1)
    add("mèo là đặc biệt lắm", 4)          # MỘT hạt giống, cạnh NẶNG hơn
    st.parents.rebuild_hierarchical_index()

    idx = build_hub_spread_index(st, d_max=6, min_count=0.5, use_isg=False)
    seeds = [resolve_surface_ids(v, x)[0] for x in ("chó", "mèo", "khỉ")]
    theo_diem = score_hubs_from(
        st, idx, seeds, topk=5, gidf_min=0.0, ban_func=False, channel="typed")
    assert theo_diem and theo_diem[0].surface == "đặc", (
        "fixture không tái lập được bệnh điểm-thắng-đồng-thuận")

    theo_phieu = score_hubs_from(
        st, idx, seeds, topk=5, gidf_min=0.0, ban_func=False,
        channel="typed", rank="phieu")
    assert theo_phieu and theo_phieu[0].surface in ("động", "vật")


def test_rank_va_head_rule_trai_la_phai_bao_loi() -> None:
    import pytest
    st = _state_luong_tu()
    with pytest.raises(ValueError, match="head_rule_trai"):
        build_hub_spread_index(st, head_rule_trai="khong_co_that")
    idx = build_hub_spread_index(st, use_isg=False)
    with pytest.raises(ValueError, match="rank"):
        score_hubs_from(st, idx, [1], rank="khong_co_that")


def test_buoc3_KE_NGU_CANH_go_hoa_ABC() -> None:
    """§60 — mọi thành viên nặng BẰNG NHAU thì thứ tự rơi về ABC.

    Đo trên toy 99 362 cha: 62/62 thành viên typed của nhãn `động` đều nặng
    đúng 0,95 (mỗi câu định nghĩa xuất hiện một lần), nên `khỉ` (K) thua
    `Am`/`An`/`Biên` và rơi khỏi `topk`. Cạnh vàng KHÔNG mất — thiếu THƯỚC.

    Fixture tái lập đúng thế hoà đó: ba thành viên cùng một câu định nghĩa,
    tên riêng đứng trước theo ABC. `Am` KHÔNG đứng sau `loài` bao giờ.
    """
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str) -> None:
        st.parents.add_example(
            [v.id_of(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    for s in ("loài khỉ là thú", "Am là thú", "Biên là thú"):
        add(s)
    st.parents.rebuild_hierarchical_index()

    # đúng luật đang chốt (§58) — vế trái lấy `khỉ`, không lấy `loài`
    idx = build_hub_spread_index(
        st, d_max=3, min_count=0.5, head_rule="trai_du_dac_thu",
        head_rule_trai="phai_nhat")
    hub = resolve_surface_ids(v, "thú")
    assert hub
    ctx = list(v.encode_prompt("nghịch ngợm như loài"))

    cu = [p.surface for p in reverse_peers(st, idx, hub, topk=6)]
    assert cu.index("Am") < cu.index("khỉ"), (
        f"fixture không tái lập thế hoà-ABC: {cu}")

    moi = [p.surface for p in reverse_peers(
        st, idx, hub, topk=6, ctx_ids=ctx, ke_ctx=2.0)]
    assert moi[0] == "khỉ", f"kề ngữ cảnh không gỡ được hoà: {moi}"
    assert set(moi) == set(cu), "kề ngữ cảnh chỉ được XẾP LẠI, không đổi bể"


def test_buoc3_ke_ngu_canh_chi_lay_token_CUOI() -> None:
    """§60 — KỀ phải lấy ô sắp điền, KHÔNG lấy cả ngữ cảnh.

    THUỘC TÍNH cũng nằm trong ngữ cảnh. Nâng thành viên từng đứng cạnh thuộc
    tính là giết đúng thứ cơ chế sinh ra: `khỉ` đáng giá CHÍNH VÌ nó chưa bao
    giờ đứng cạnh `nghịch ngợm`. Ở đây `Am` đứng cạnh thuộc tính, `khỉ` đứng
    sau từ cuối của prompt — thắng phải thuộc về `khỉ`.
    """
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()

    def add(s: str) -> None:
        st.parents.add_example(
            [v.id_of(t) for t in (s + " <EOS>").split()],
            v, dap_id=None, defer_index=True,
        )

    for s in ("loài khỉ là thú", "Am là thú", "nghịch ngợm Am rất lạ"):
        add(s)
    st.parents.rebuild_hierarchical_index()

    idx = build_hub_spread_index(
        st, d_max=3, min_count=0.5, head_rule="trai_du_dac_thu",
        head_rule_trai="phai_nhat")
    hub = resolve_surface_ids(v, "thú")
    moi = [p.surface for p in reverse_peers(
        st, idx, hub, topk=6,
        ctx_ids=list(v.encode_prompt("nghịch ngợm như loài")), ke_ctx=2.0)]
    assert moi[0] == "khỉ", f"KỀ đang ăn cả ngữ cảnh, không chỉ ô cuối: {moi}"
