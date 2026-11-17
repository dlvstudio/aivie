"""Unit: `analogy_la` (TK_046 P98G) — điền `X là ?` qua cầu thuộc_tính.

Fixture tổng hợp, không cần checkpoint. Ngưỡng lọc hub (`prop_gidf_min`,
`pred_top_ban`, `min_strength`) được nới vì chúng chỉnh theo corpus v12; test
này khoá **cơ chế** (cầu → donor → điền) và **cổng support**, không khoá số
tuning.
"""
from __future__ import annotations

import pytest

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.analogy_bridge_gen import (
    AnalogyBridgeConfig,
    hu_tu_tu_do_thi,
    build_analogy_bridge_index,
    fill_predicate,
    generate_analogy_bridge,
    parse_la_prompt,
    resolve_head_phrase,
    surface,
)

# Nới hub-filter: corpus tí hon thì mọi pred đều nằm trong top-40.
_CFG = AnalogyBridgeConfig(prop_gidf_min=0.0, min_strength=0.0, pred_top_ban=0)

_CAU_NOI = [
    "con khỉ nghịch ngợm <EOS>",
    "con chó nghịch ngợm <EOS>",
    "con mèo nghịch ngợm <EOS>",
]
_CAU_LA = [
    "con chó là thú <EOS>",
    "con mèo là thú <EOS>",
]


def _state(cau: list[str], *, lap: int = 2) -> GraphState:
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()
    for s in cau:
        for _ in range(lap):
            st.parents.add_example(
                [v.id_of(t) for t in s.split()], v, dap_id=None, defer_index=True
            )
    st.parents.rebuild_hierarchical_index()
    return st


def test_dien_qua_cau_thuoc_tinh_la_sang_tao() -> None:
    """`khỉ` chưa từng «là thú»; hai bạn cùng cầu `nghịch` thì suy ra được."""
    st = _state(_CAU_NOI + _CAU_LA)
    v = st.vocab
    idx = build_analogy_bridge_index(st, _CFG)
    a, _ = resolve_head_phrase(
        v, "khỉ", idx.props, len(v), compound_surf=idx.compound_surf
    )
    assert a is not None
    pred, meta = fill_predicate(st, idx, a, _CFG)
    assert pred is not None
    assert surface(v, pred, idx.compound_surf) == "thú"
    assert meta["stop"] == "analogy_la"
    assert set(meta["donors"]) == {"chó", "mèo"}
    # `la_ac` = head → pred đã thấy trong corpus. `khỉ` không có pred nào,
    # nên «khỉ là thú» là điền SUY RA, không phải chép.
    assert pred not in idx.la_ac.get(a, set())
    assert {surface(v, h, idx.compound_surf) for h in idx.la_ac} == {"chó", "mèo"}


def test_mot_donor_thi_khong_dien() -> None:
    """`min_support=2` là cổng giữ FP=0 (P98D/E) — phải khoá bằng test."""
    st = _state(_CAU_NOI + ["con chó là thú <EOS>"])
    v = st.vocab
    idx = build_analogy_bridge_index(st, _CFG)
    a, _ = resolve_head_phrase(
        v, "khỉ", idx.props, len(v), compound_surf=idx.compound_surf
    )
    assert a is not None
    pred, meta = fill_predicate(st, idx, a, _CFG)
    assert pred is None
    assert meta["stop"] == "khong_dien"


def test_generate_them_la_khi_de_chua_co() -> None:
    st = _state(_CAU_NOI + _CAU_LA)
    v = st.vocab
    idx = build_analogy_bridge_index(st, _CFG)

    ids_co_la, _ = generate_analogy_bridge(st, "khỉ là", index=idx, cfg=_CFG)
    assert [surface(v, t) for t in ids_co_la] == ["thú"]

    ids_khong, _ = generate_analogy_bridge(st, "khỉ", index=idx, cfg=_CFG)
    assert [surface(v, t) for t in ids_khong] == ["là", "thú"]


def test_la_emit_chon_theo_tan_suat_khong_theo_id() -> None:
    """Biến thể `là` phát ra = df cao nhất, không phải `min(la_ids)`."""
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()
    id_hiem = v.id_of("là")  # id nhỏ hơn, hiếm hơn
    id_pho_bien = v.id_of(MARK_SPACE + "là")
    assert id_hiem < id_pho_bien
    for toks in (
        ["con", "chó", MARK_SPACE + "là", "thú", "<EOS>"],
        ["con", "mèo", MARK_SPACE + "là", "thú", "<EOS>"],
        ["anh", "là", "người", "<EOS>"],
    ):
        st.parents.add_example(
            [v.id_of(t) for t in toks], v, dap_id=None, defer_index=True
        )
    st.parents.rebuild_hierarchical_index()

    idx = build_analogy_bridge_index(st, _CFG)
    assert idx.la_ids == {id_hiem, id_pho_bien}
    assert idx.la_emit_id == id_pho_bien
    assert idx.la_emit_id != min(idx.la_ids)


def test_thieu_gdf_thi_no_ngay_khong_am_tham_ve_1() -> None:
    """Thiếu `_gdf` ⇒ gidf im lặng = 1.0 cho mọi cầu — phải fail nhanh."""
    st = _state(_CAU_LA, lap=1)
    del st.parents._gdf  # noqa: SLF001
    with pytest.raises(RuntimeError, match="_gdf"):
        build_analogy_bridge_index(st, _CFG)


def test_meta_khong_con_truong_lop_chet() -> None:
    """`lop` từng luôn rỗng sau khi gỡ `lop_trai` — đã bỏ hẳn."""
    st = _state(_CAU_NOI + _CAU_LA)
    idx = build_analogy_bridge_index(st, _CFG)
    a, _ = resolve_head_phrase(
        st.vocab, "khỉ", idx.props, len(st.vocab), compound_surf=idx.compound_surf
    )
    _pred, meta = fill_predicate(st, idx, a, _CFG)
    assert "lop" not in meta


def test_hu_tu_gidf_mac_dinh_TAT() -> None:
    """Mặc định phải là bảng tay: gidf cấm cả danh từ riêng tần suất cao.

    Đo trên `mini_sent_ep_qa.pkl`: bật `hu_tu_tu_gidf` thì `Nguyễn`
    (gidf 4,09 < 4,5) bị cấm làm predicate ⇒ smoke ship `anh là → Hồ` chết.
    Xem docstring `hu_tu_tu_do_thi`.
    """
    assert AnalogyBridgeConfig().hu_tu_tu_gidf is False


def test_hu_tu_tu_do_thi_bat_duoc_hu_tu() -> None:
    """Cơ chế dẫn xuất vẫn phải chạy được (dùng cho A/B)."""
    st = _state(_CAU_NOI + _CAU_LA, lap=3)
    hu = hu_tu_tu_do_thi(st, gidf_max=99.0)
    assert "con" in hu  # ngưỡng rất cao ⇒ bắt tất
    assert hu_tu_tu_do_thi(st, gidf_max=0.0) == frozenset()


@pytest.mark.parametrize(
    ("de", "head", "co_la"),
    [
        ("anh là", "anh", True),
        ("anh là.", "anh", True),
        ("anh  là", "anh", True),
        ("học sinh là", "học sinh", True),
        ("anh", "anh", False),
        ("anh?", "anh", False),
        ("", "", False),
    ],
)
def test_parse_la_prompt(de: str, head: str, co_la: bool) -> None:
    assert parse_la_prompt(de) == (head, co_la)
