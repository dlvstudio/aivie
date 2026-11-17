"""TK_045 §13 Trụ B — thay khe.

Bất biến khoá bằng test — cái quan trọng nhất là (2): bảo chứng attestation phải giữ,
nếu không thì Trụ B chỉ đổi CHÉP thành SALAD, tệ hơn chỗ xuất phát.
"""

from __future__ import annotations

import random

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.eval.ruler import NgramIndex, attestation_rate, novelty_score
from nn_aivie.eval.ruler import CopySpanIndex
from nn_aivie.hca.slot_swap import (
    build_slot_index,
    clear_slot_cache,
    ensure_slot_index,
    swap_slots,
)
from nn_aivie.training.ingest import ingest_statements

_SENTS = [
    "Con mèo nhỏ nằm trên chiếc ghế gỗ .",
    "Con chó nhỏ nằm trên chiếc ghế mây .",
    "Con mèo lớn ngồi trên chiếc bàn gỗ .",
    "Con chó lớn ngồi trên chiếc bàn mây .",
    "Con chim nhỏ đậu trên chiếc ghế gỗ .",
]


def _trained() -> GraphState:
    clear_slot_cache()
    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, _SENTS, split_punct=True)
    return st


def test_fillers_are_paradigm_class_from_slot_not_embedding():
    """Khe (Con, _, nhỏ) phải cho {mèo, chó, chim} — lớp mẫu ĐẾM ĐƯỢC, không cần embedding."""
    st = _trained()
    idx = build_slot_index(st)
    a, b = st.vocab.id_of("Con"), st.vocab.id_of("nhỏ")
    fill = idx.fillers(a, b, order=2, max_gidf=99.0)
    surf = {st.vocab.token_of(x).lstrip("Ġ") for x in fill}
    assert {"mèo", "chó", "chim"} <= surf, surf


def test_swap_preserves_attestation_by_construction():
    """BẤT BIẾN QUAN TRỌNG NHẤT — thay khe KHÔNG được hạ attestation.

    Nếu test này đỏ thì Trụ B đang biến CHÉP thành SALAD, tức tệ hơn chỗ xuất phát.
    """
    st = _trained()
    idx = build_slot_index(st)
    emits = [r.emit_ids for r in st.parents.parents]
    ng = NgramIndex(emits, orders=(2, 3))
    for seed in range(12):
        src = list(st.parents.parents[seed % len(st.parents.parents)].emit_ids)
        out, n = swap_slots(idx, src, order=2, max_gidf=99.0, rng=random.Random(seed))
        if n == 0:
            continue
        assert attestation_rate(ng, out, order=2) == 1.0, (
            f"thay khe làm gãy bigram: {[st.vocab.token_of(t) for t in out]}"
        )


def test_swap_raises_novelty():
    """Thay khe phải CẮT được đoạn chép liên tục — lý do tồn tại của Trụ B (§13.11)."""
    st = _trained()
    idx = build_slot_index(st)
    emits = [r.emit_ids for r in st.parents.parents]
    cs = CopySpanIndex(emits)
    src = list(st.parents.parents[0].emit_ids)
    before = novelty_score(cs, src)
    assert before == 0.0, "chuỗi gốc phải là chép hoàn toàn"
    best = before
    for seed in range(12):
        out, n = swap_slots(idx, src, order=2, max_gidf=99.0, rng=random.Random(seed))
        if n:
            best = max(best, novelty_score(cs, out))
    assert best > before, "không lần thay nào cắt được khối chép"


def test_gidf_valve_protects_high_idf_tokens():
    """Van fact — token gidf CAO (thực thể/số) không được vào khe."""
    st = _trained()
    idx = build_slot_index(st)
    src = list(st.parents.parents[0].emit_ids)
    out, _n = swap_slots(idx, src, order=2, max_gidf=0.0, rng=random.Random(0))
    assert out == src, "max_gidf=0 phải cấm mọi thay thế"


def test_order3_is_stricter_than_order2():
    st = _trained()
    idx = build_slot_index(st)
    a, b = st.vocab.id_of("Con"), st.vocab.id_of("nhỏ")
    f2 = idx.fillers(a, b, order=2, max_gidf=99.0)
    f3 = idx.fillers(a, b, order=3, max_gidf=99.0)
    assert f3 <= f2, "trigram phải là tập CON của bigram"


def test_min_gap_prevents_adjacent_swaps():
    """Thay hai vị trí liền kề làm mối nối GIỮA chúng chưa được chứng thực → gãy bảo chứng."""
    st = _trained()
    idx = build_slot_index(st)
    src = list(st.parents.parents[0].emit_ids)
    for seed in range(8):
        out, n = swap_slots(
            idx, src, order=2, max_gidf=99.0, rng=random.Random(seed), min_gap=2,
        )
        diff = [i for i, (x, y) in enumerate(zip(src, out)) if x != y]
        assert all(
            b - a >= 2 for a, b in zip(diff, diff[1:])
        ), f"có hai thay thế liền kề: {diff}"
        assert n == len(diff)


def test_ensure_slot_index_caches():
    st = _trained()
    assert ensure_slot_index(st) is ensure_slot_index(st)
    assert ensure_slot_index(st, with_trigrams=False) is not ensure_slot_index(st)


def test_slot_index_source_is_emit_ids_not_wseq():
    """REGRESSION §13.12 — nguồn "đã chứng thực" phải là `emit_ids`, KHÔNG phải W_seq.

    Bản đầu dựng succ/pred từ `EdgeKind.SEQUENTIAL`. Đo trên mini: **60.6%** bigram W_seq
    KHÔNG có trong emit của cha ⇒ cổng sinh và thước dùng HAI định nghĩa khác nhau ⇒ tuyên
    bố "attestation giữ theo cấu tạo" sai (0.979→0.882). Test đồ chơi cũ KHÔNG bắt được vì
    trên toy hai nguồn trùng khít — nên test này phải cố tình làm chúng LỆCH.
    """
    from nn_aivie.core.edge_kinds import EdgeKind

    st = _trained()
    a, b = st.vocab.id_of("Con"), st.vocab.id_of("nhỏ")
    ghost = st.vocab.id_of("ghost_token_khong_co_trong_emit")
    # chỉ có trên W_seq, KHÔNG có trong emit_ids của cha nào
    st.adjacency.add(a, ghost, 5.0, EdgeKind.SEQUENTIAL)
    st.adjacency.add(ghost, b, 5.0, EdgeKind.SEQUENTIAL)

    idx = build_slot_index(st)
    fill = idx.fillers(a, b, order=2, max_gidf=99.0)
    assert ghost not in fill, (
        "filler đến từ W_seq — cổng sinh đang dùng nguồn KHÁC với thước"
    )


def test_digit_valve_protects_numbers():
    """REGRESSION §13.12 — `max_gidf` KHÔNG bảo vệ được số (chữ số là atom phổ biến,
    gidf thấp). Van `has_digit` phải chặn cả token số LẪN láng giềng của nó."""
    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, [
        "Vua lên ngôi năm 1010 tại kinh đô .",
        "Vua lên ngôi năm 1225 tại kinh đô .",
        "Vua lên ngôi năm 1400 tại kinh đô .",
    ], split_punct=True)
    idx = build_slot_index(st)
    src = list(st.parents.parents[0].emit_ids)
    digits = {i for i, t in enumerate(src)
              if any(ch.isdigit() for ch in st.vocab.token_of(t))}
    assert digits, "fixture không có token số — test vô nghĩa"
    for seed in range(15):
        out, _n = swap_slots(idx, src, order=2, max_gidf=99.0, rng=random.Random(seed))
        for i in digits:
            assert out[i] == src[i], (
                f"token SỐ bị thay: {st.vocab.token_of(src[i])} → {st.vocab.token_of(out[i])}"
            )
