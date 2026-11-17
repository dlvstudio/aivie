"""TK_045 §13 Trụ D — truy hồi Hopfield hiện đại trên `ParentBipartite.A`.

Bất biến khoá bằng test — mỗi cái ứng một phát hiện của P48 (§13.9):

1. **`steps=1` KHÔNG đổi thứ hạng so với cosine trần** — softmax đơn điệu. Khoá để không ai
   tưởng `steps=1` là "đã bật Hopfield".
2. **β là núm chống mờ** — β thấp làm nhoè phân bố, β cao giữ sắc.
3. **Phá thế hoà GIỮ NGUYÊN bậc điểm chính** — 62% query không hoà không được phép đụng.
4. Không hoà đỉnh ⇒ trả nguyên xi (đường rẻ).
"""

from __future__ import annotations

import numpy as np

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.hca.hopfield import (
    _match_scores,
    _softmax_top,
    clear_hopfield_cache,
    ensure_hopfield_memory,
    retrieve,
    tiebreak_ranked,
)
from nn_aivie.training.ingest import ingest_statements

_SENTS = [
    "Phở là món ăn nổi tiếng của Việt Nam .",
    "Hà Nội là thủ đô của Việt Nam .",
    "Huế là cố đô nằm ở miền Trung .",
    "Sài Gòn là đô thị lớn nhất miền Nam .",
    "Đà Lạt là thành phố cao nguyên mát mẻ .",
]


def _trained() -> GraphState:
    clear_hopfield_cache()
    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, _SENTS, split_punct=True)
    return st


def _q(st, *words) -> list[int]:
    return [st.vocab.id_of(w) for w in words]


def test_memory_shape_matches_bipartite():
    st = _trained()
    mem = ensure_hopfield_memory(st)
    assert mem.A_csr.shape == (len(st.vocab), len(st.parents.parents))
    assert mem.A_csr.nnz > 0
    assert np.all(mem.col_norm > 0)


def test_step1_ranking_equals_plain_cosine():
    """Bất biến 1 — softmax đơn điệu ⇒ steps=1 chỉ là cosine, KHÔNG phải Hopfield.

    Nếu test này đỏ thì `retrieve` đã lén làm gì đó ở steps=1 và mọi kết luận "đối chứng
    B_hop_s1" của P48 mất hiệu lực.
    """
    st = _trained()
    mem = ensure_hopfield_memory(st)
    q = _q(st, "Việt", "Nam")
    w = [1.0, 1.0]
    raw = _match_scores(mem, np.asarray(q), np.asarray(w))
    want = [int(i) for i in np.argsort(-raw) if raw[i] > 0]
    for beta in (1.0, 8.0, 64.0):
        got = [pi for pi, _ in retrieve(mem, q, w, beta=beta, steps=1, top_n=0)]
        assert got == want, f"steps=1 đổi thứ hạng ở β={beta}"


def test_beta_controls_sharpness():
    """Bất biến 2 — β cao ⇒ phân bố sắc hơn (entropy thấp hơn)."""
    s = np.array([0.0, 0.9, 0.8, 0.5, 0.3])

    def _ent(beta):
        _idx, p = _softmax_top(s, beta, top_k=8)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    assert _ent(32.0) < _ent(1.0), "β cao phải làm SẮC — nếu không, tuyên bố 'β là núm chống mờ' sai"


def test_tiebreak_preserves_primary_order():
    """Bất biến 3 — bậc điểm chính BẤT BIẾN; chỉ trong bậc mới xếp lại."""
    st = _trained()
    q = _q(st, "Việt", "Nam")
    ranked = [(0, 1.0), (1, 1.0), (2, 0.5), (3, 0.5), (4, 0.2)]
    out = tiebreak_ranked(st, ranked, q, [1.0, 1.0], beta=32.0)
    assert {pi for pi, _ in out} == {pi for pi, _ in ranked}, "mất/thêm ứng viên"
    assert {out[0][0], out[1][0]} == {0, 1}, "phần tử nhảy khỏi bậc của nó"
    assert {out[2][0], out[3][0]} == {2, 3}
    # điểm chỉ được nhiễu EPSILON, không được vượt bậc
    scores = [sc for _, sc in out]
    assert scores == sorted(scores, reverse=True), "điểm không còn giảm dần"
    for got, want in zip(scores, [1.0, 1.0, 0.5, 0.5, 0.2]):
        assert abs(got - want) < 1e-6, f"điểm lệch quá epsilon: {got} vs {want}"
    # và epsilon PHẢI phá được hoà (nếu không thì consumer sắp-theo-điểm mất hiệu lực)
    assert scores[0] != scores[1], "hoà vẫn còn ⇒ tie-break vô hình với consumer theo điểm"


def test_tiebreak_noop_without_top_tie():
    """Bất biến 4 — không hoà đỉnh ⇒ trả NGUYÊN XI (không tính Hopfield)."""
    st = _trained()
    q = _q(st, "Việt", "Nam")
    ranked = [(0, 1.0), (1, 0.9), (2, 0.5)]
    assert tiebreak_ranked(st, ranked, q, [1.0, 1.0], beta=32.0) == ranked
    assert tiebreak_ranked(st, ranked, q, [1.0, 1.0], beta=0.0) == ranked  # beta=0 = tắt


def test_retrieve_finds_source_parent():
    """Truy hồi cơ bản: query lấy từ một cha phải kéo chính cha đó lên top."""
    st = _trained()
    mem = ensure_hopfield_memory(st)
    ranked = retrieve(mem, _q(st, "Huế", "cố", "đô"), beta=32.0, steps=2, top_n=3)
    assert ranked, "không truy hồi được gì"
    emit = st.parents.parents[ranked[0][0]].emit_ids
    assert st.vocab.id_of("Huế") in emit


def test_rank_matches_uses_tiebreak_by_default():
    """§13.15 — phá-thế-hoà BẬT mặc định trong `rank_matches`, qua backref state.

    Kiểm hai điều: (a) backref được đặt tự động khi gán `state.parents`;
    (b) `hopfield_tiebreak_beta=0` tắt được (đường lùi khi cần A/B).
    """
    st = _trained()
    v = st.vocab
    q = _q(st, "Việt", "Nam")

    assert getattr(st.parents, "_owner_ref", None) is not None, "backref chưa được đặt"
    assert st.parents._owner_ref() is st, "backref trỏ sai state"

    on = st.parents.rank_matches(q, v, None, fuzzy_topk=0, top_n=0)
    st.parents.hopfield_tiebreak_beta = 0.0
    off = st.parents.rank_matches(q, v, None, fuzzy_topk=0, top_n=0)
    assert {pi for pi, _ in on} == {pi for pi, _ in off}, "tie-break làm MẤT/THÊM ứng viên"
    # điểm chính bất biến ⇒ multiset điểm phải y hệt, chỉ THỨ TỰ trong bậc đổi
    assert sorted(s for _, s in on) == sorted(s for _, s in off), "bậc điểm chính bị đổi"


def test_backref_is_weak_not_strong():
    """weakref, không phải tham chiếu cứng — nếu không thì state↔parents thành chu trình
    giữ sống lẫn nhau, đúng lớp lỗi cache-theo-id(state) §12.29/§12.30."""
    import gc
    import weakref

    st = _trained()
    ps = st.parents
    ref = weakref.ref(st)
    del st
    gc.collect()
    assert ref() is None, "GraphState không được thu hồi ⇒ backref đang giữ CỨNG"
    assert ps._owner_ref() is None, "backref phải chết theo chủ"
