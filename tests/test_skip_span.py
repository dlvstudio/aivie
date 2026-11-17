"""TK_036 — skip span mining, retrieval units, AR gate."""

import pytest

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.hca.readout import decode_surface
from nn_aivie.core.skip_span_constants import SKIP_MAX_GAP, SKIP_MIN_GAP, skip_right_tuple
from nn_aivie.training.skip_span import mine_skip_spans


def _g1_and_wseq(st: GraphState, ids: list[int], *, eta: float = 1.0) -> None:
    from nn_aivie.core.edge_kinds import EdgeKind

    for a, b in zip(ids, ids[1:]):
        st.adjacency.add(a, b, eta, EdgeKind.SEQUENTIAL)


# Hai họ câu cùng khung, khác thực thể (phở↔Việt Nam / sushi↔Nhật Bản). Miner quét
# `slot_chain` (không phải token thô) nên fixture phải thoả hai điều kiện của TK_036 S3:
#   1. chain ≥ min_gap+2 slot — câu lặp y hệt bị `add_example` gộp còn MỘT cha và ISG gom
#      cụm nuốt gần hết câu → chain quá ngắn, `scan_parent` thoát ngay.
#   2. ≥2 endpoint phải cạnh tranh — nếu cả corpus chỉ có một endpoint thì b_tot == total
#      nên pmi = log(c/a_tot) = 0 với MỌI trái, không span nào qua `min_pmi`.
# Nhờ đó cặp phân biệt (phở→Việt Nam) sống, cặp generic (là→Việt Nam) bị PMI loại.
_SKIP_SENTS = [
    "phở là món ăn rất phổ biến ở Việt Nam hiện nay .",
    "phở là món ngon nổi tiếng của Việt Nam từ lâu .",
    "phở là món truyền thống lâu đời tại Việt Nam xưa kia .",
    "sushi là món ăn rất phổ biến ở Nhật Bản hiện nay .",
    "sushi là món ngon nổi tiếng của Nhật Bản từ lâu .",
    "sushi là món truyền thống lâu đời tại Nhật Bản xưa kia .",
]

_SKIP_QUERY = "phở là món ăn rất phổ biến ở Việt Nam hiện nay"


def _skip_state(*, wseq: bool = True) -> GraphState:
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    for i, sent in enumerate(_SKIP_SENTS):
        ids = list(st.vocab.encode_tokens(sent.split()))
        ps.add_example(ids, st.vocab, dap_id=dap, priority=0.5 + i * 0.01)
        if wseq:
            _g1_and_wseq(st, ids, eta=1.0)
    st.parents = ps
    return st


def _vn(st: GraphState) -> tuple[int, int]:
    return (st.vocab.id_of("Việt"), st.vocab.id_of("Nam"))


def test_mine_skip_span_promotes_pair():
    st = _skip_state()
    ps = st.parents
    stats = mine_skip_spans(st, min_count=1, min_pmi=0.1, promote_power=2)
    assert stats["spans"] >= 1
    pho = st.vocab.id_of("phở")
    assert (pho, _vn(st)) in ps._skip_lookup
    span = ps.skip_spans[ps._skip_lookup[(pho, _vn(st))]]
    assert span["power"] == 3  # 3 cha họ phở đóng góp, mỗi cha đúng 1 lần
    assert span["promoted"] is True


def test_mine_skip_span_pmi_rejects_generic_left():
    """`là` đứng trước CẢ hai endpoint → pmi=0 → không thành span; chỉ trái phân biệt sống."""
    st = _skip_state()
    ps = st.parents
    mine_skip_spans(st, min_count=1, min_pmi=0.1, promote_power=2)
    la = st.vocab.id_of("là")
    nb = (st.vocab.id_of("Nhật"), st.vocab.id_of("Bản"))
    pho = st.vocab.id_of("phở")
    assert (la, _vn(st)) not in ps._skip_lookup
    assert (pho, nb) not in ps._skip_lookup  # không lẫn chéo họ
    assert (st.vocab.id_of("sushi"), nb) in ps._skip_lookup


def test_mine_skip_span_token_to_phrase():
    """Đầu phải là CỤM nhiều token (Việt Nam), không phải token đơn."""
    st = _skip_state(wseq=False)
    ps = st.parents
    mine_skip_spans(st, min_count=1, min_pmi=0.1)
    pho = st.vocab.id_of("phở")
    assert (pho, _vn(st)) in ps._skip_lookup
    assert all(len(s["right"]) >= 2 for s in ps.skip_spans)


def test_skip_left_index_matches_naive_scan():
    st = _skip_state(wseq=False)
    ps = st.parents
    mine_skip_spans(st, min_count=1, min_pmi=0.1)
    q = tuple(st.vocab.encode_tokens(_SKIP_QUERY.split()))
    assert ps._skip_left_index
    pho = st.vocab.id_of("phở")
    assert pho in ps._skip_left_index
    got = set(ps._skip_spans_in(q))
    # naive O(|lookup|) reference — chỉ dùng trong test
    naive: set[tuple[int, int, int, int]] = set()
    for (left, right), si in ps._skip_lookup.items():
        right = skip_right_tuple(right)
        L = len(right)
        for i, a in enumerate(q):
            if a != left:
                continue
            for j in range(i + SKIP_MIN_GAP + 1, min(len(q), i + SKIP_MAX_GAP + 2)):
                if j + L <= len(q) and tuple(q[j : j + L]) == right:
                    naive.add((i, j + L, si, j - i - 1))
    assert got == naive


def test_query_match_units_covers_query():
    """Query phân rã thành unit (cụm/skip) — KHÔNG khẳng định skip_units sống.

    Assert cũ là `units.skip_units or units.phrase_units`: vế `or` khiến nó luôn xanh nhờ
    phrase_units, che mất việc skip_units chưa bao giờ bật. Xem test xfail bên dưới.
    """
    st = _skip_state()
    ps = st.parents
    mine_skip_spans(st, min_count=1, min_pmi=0.1)
    q = tuple(st.vocab.encode_tokens(_SKIP_QUERY.split()))
    units = ps.query_match_units(q)
    assert units.phrase_units
    assert units.phrase_covered_tokens


@pytest.mark.xfail(
    strict=True,
    reason="TK_036 — ÂM TÍNH ĐÃ ĐO, KHÔNG PHẢI SỔ NỢ CẦN SỬA. skip_units chết thật "
    "(phrase_units giành used_pos trước ở parent_store.py:1024; đầu-phải skip span luôn là "
    "cụm nên luôn trùng vị trí → skip_units bật 0/250 query trên grade_lessons_test.pkl dù "
    "đã bake 53 span). NHƯNG sửa cho sống thì retrieval TỤT: rank-1 76.3%→70.0%, MRR "
    "0.853→0.815 (additive, cùng ckpt); quét trọng số w=0/0.05/0.10/1.0 → không w nào có "
    "lợi, tốt nhất là trung tính khi w≈0. Cơ chế: skip span là bằng chứng cấp HỌ (phở→Việt "
    "Nam có ở CẢ 3 cha họ phở) → nâng đều cả họ, loãng bằng chứng riêng của cha đúng. "
    "Giữ xfail này để ghi nhận kết quả âm tính — ĐỪNG 'sửa' cho xanh. Giá trị của skip span "
    "nằm ở AR gate (hca/skip_span_gate.py dùng _skip_spans_in trực tiếp, KHÔNG dính lỗi này), "
    "không ở retrieval. Muốn lật kết luận cần eval CÓ NHÃN (self-retrieval thiên vị chống skip).\n\n"
    "BA CHI TIẾT DỄ VẤP KHI ĐỌC LẠI:\n"
    "1. Guard README §0.5.12 KHÔNG chứng minh được gì ở đây — grade_v3.pkl mine ra 0 span (có "
    "trước ISG/TK_013; miner quét slot_chain) nên QA/Para/Syn BẤT BIẾN dù sửa gì. Thấy 'guard "
    "không đổi' KHÔNG có nghĩa là an toàn: code path không hề chạy.\n"
    "2. Gate `any(p in used_pos for p in range(s, e))` phủ CẢ GAP, không chỉ hai đầu — nên cụm "
    "nằm giữa gap cũng giết span. Chặt hơn mô tả 'chỉ trùng đầu-phải'.\n"
    "3. Đây là BẤT KHẢ ĐẠT THEO CẤU TRÚC, không phải 'thứ tự ưu tiên sai': skip_ledger."
    "_right_at_slot trả None khi len(right) < 2, nên đầu-phải LUÔN là cụm ≥2 token → luôn đã bị "
    "phrase_units chiếm. Không có thứ tự nào cứu được; chỉ additive (bỏ tranh used_pos) mới sống "
    "— và additive chính là bản đo ra −6.3pp ở trên.",
)
def test_query_match_units_detect_skip():
    st = _skip_state()
    ps = st.parents
    mine_skip_spans(st, min_count=1, min_pmi=0.1)
    q = tuple(st.vocab.encode_tokens(_SKIP_QUERY.split()))
    # span (phở → Việt Nam) CÓ trong _skip_spans_in(q) nhưng bị lọc khỏi skip_units
    assert ps._skip_spans_in(q), "tiền đề: skip span khớp query"
    units = ps.query_match_units(q)
    assert units.skip_units


def test_skip_span_gate_does_not_break_generate():
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    sent = "phở là món ăn rất phổ biến ở Việt Nam ."
    ids = list(st.vocab.encode_tokens(sent.split()))
    ps.add_example(ids, st.vocab, dap_id=dap)
    _g1_and_wseq(st, ids, eta=1.0)
    st.parents = ps
    mine_skip_spans(st, min_count=1, min_pmi=0.1)
    cfg = HCAConfig(ar_skip_span_weight=0.35)
    out = decode_surface(st, generate(st, "phở là", max_len=24, config=cfg))
    assert out.startswith("phở")
    assert "phở" in out


def test_skip_spans_sidecar_roundtrip(tmp_path):
    st = _skip_state(wseq=False)
    ps = st.parents
    mine_skip_spans(st, min_count=1, min_pmi=0.1)
    n_spans = len(ps.skip_spans)
    assert n_spans >= 1

    ckpt = tmp_path / "toy.pkl"
    ckpt.write_bytes(b"stub")
    from nn_aivie.core.skip_span_cache import (
        load_skip_spans_sidecar,
        save_skip_spans_sidecar,
    )

    save_skip_spans_sidecar(ps, ckpt)
    ps2 = ParentStore()
    for p in ps.parents:
        ps2.parents.append(p)
    ps2.skip_spans = []
    assert load_skip_spans_sidecar(ps2, ckpt)
    assert len(ps2.skip_spans) == n_spans
    assert ps2._skip_lookup == ps._skip_lookup


def test_bake_derived_embeds_skip_in_payload():
    st = _skip_state(wseq=False)
    ps = st.parents
    mine_skip_spans(st, min_count=1, min_pmi=0.1)
    from nn_aivie.training.checkpoint_bake import bake_derived_indices

    bake_derived_indices(st, skip_spans=True, seq_index=True, frame_brackets=False)
    assert len(ps.skip_spans) >= 1
    payload = ps.to_payload()
    assert payload.get("skip_spans")
    assert payload.get("seq_bi") is not None
    restored = ParentStore.from_payload(payload)
    assert len(restored.skip_spans) == len(ps.skip_spans)
    assert getattr(restored, "_seq_ready", False)
