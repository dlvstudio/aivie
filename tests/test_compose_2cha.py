"""Unit: `compose_2cha` (TK_051 P203) — dual retrieve + bản lề.

Khoá hai thứ trên đường ship: bộ lọc niên đại không loại nhầm cha đúng đề,
và `ent_keep` / `join_score` đúng hợp đồng.
"""
from __future__ import annotations

import numpy as np

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.compose_2cha import ent_keep, join_score, rank_compose_parents
from nn_aivie.hca.frame_slot_gen import eras_in_text
from nn_aivie.hca.w_skip import ensure_w_skip


def _state(cau: list[str], *, lap: int = 1) -> GraphState:
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


def _enc(st: GraphState, s: str) -> list[int]:
    return [st.vocab.id_of(t) for t in s.split()]


def _emit(st: GraphState, pi: int) -> str:
    return " ".join(st.vocab.token_of(t) for t in st.parents.parents[pi].emit_ids)


def test_khong_loai_nham_cha_vi_nien_dai_gia() -> None:
    """`sông Hồng` từng bị đọc thành triều `Hồ` → cha đúng đề bị loại oan."""
    cau = [
        "nhà Trần đánh giặc Nguyên Mông <EOS>",
        "nhà cửa dựng bên sông Hồng <EOS>",
    ]
    st = _state(cau)
    keep = np.ones(len(st.vocab), dtype=bool)
    prompt = _enc(st, "nhà Trần")

    # Tiền đề của ca: cha #1 KHÔNG chứa niên đại nào. Code cũ khớp chuỗi con
    # đọc `Hồng` thành triều `Hồ` ⇒ {Hồ} ∩ {Trần} = ∅ ⇒ loại oan cha đúng đề.
    assert eras_in_text("nhà cửa dựng bên sông Hồng") == set()

    got = rank_compose_parents(st, prompt, keep, top_n=16, max_cands=12)
    emits = [_emit(st, pi) for pi in got]
    assert any("sông Hồng" in e for e in emits)


def test_loc_nien_dai_van_loai_trieu_khac() -> None:
    """Cùng đề `nhà`, khác triều → vẫn phải loại (bộ lọc còn tác dụng)."""
    cau = [
        "nhà Trần đánh giặc Nguyên Mông <EOS>",
        "nhà Lý dời đô về Thăng Long <EOS>",
    ]
    st = _state(cau)
    keep = np.ones(len(st.vocab), dtype=bool)
    prompt = _enc(st, "nhà Trần")

    got = rank_compose_parents(st, prompt, keep, top_n=16, max_cands=12)
    emits = [_emit(st, pi) for pi in got]
    assert not any("nhà Lý" in e for e in emits)


def test_bo_qua_cha_nguon_de() -> None:
    cau = [
        "nhà Trần đánh giặc Nguyên Mông <EOS>",
        "nhà Trần đóng đô bên sông Hồng <EOS>",
    ]
    st = _state(cau)
    keep = np.ones(len(st.vocab), dtype=bool)
    prompt = _enc(st, "nhà Trần")

    got = rank_compose_parents(st, prompt, keep, skip_parent=0, top_n=16, max_cands=12)
    assert 0 not in got


def test_ent_keep_hop_dong() -> None:
    st = _state(["nhà Trần đánh giặc <EOS>"])
    keep = np.ones(len(st.vocab), dtype=bool)
    de = _enc(st, "nhà Trần")
    assert ent_keep(de, _enc(st, "nhà Trần đánh giặc"), keep) == 1.0
    assert ent_keep(de, _enc(st, "nhà đánh giặc"), keep) == 0.5
    assert ent_keep(de, _enc(st, "đánh giặc"), keep) == 0.0
    # đề không có token keep nào → quy ước 1.0
    keep_rong = np.zeros(len(st.vocab), dtype=bool)
    assert ent_keep(de, _enc(st, "đánh giặc"), keep_rong) == 1.0


def test_join_score_duong_khi_co_ban_le_da_thay() -> None:
    """Bản lề = cặp (cuối A, đầu B) từng đồng hiện trong W_skip."""
    cau = ["nhà Trần đánh giặc <EOS>", "đánh giặc Nguyên Mông <EOS>"]
    st = _state(cau, lap=2)
    keep = np.ones(len(st.vocab), dtype=bool)
    ws = ensure_w_skip(st, d_max=8, min_count=1, content_only=False)

    a = _enc(st, "nhà Trần đánh")
    b = _enc(st, "giặc Nguyên Mông")
    assert join_score(st, ws, a, b, keep, d_max=8, width=3) > 0.0

    # emit rỗng → không có bản lề
    assert join_score(st, ws, a, [], keep, d_max=8, width=3) == 0.0
