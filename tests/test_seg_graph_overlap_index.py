"""Chỉ mục nghịch đảo chồng biên (TK_038 power-win) ≡ quét tuyến tính.

`_bigram_overlap_blocked` / `_overlap_rival_power` từng quét toàn bộ
`cluster_of_pair` mỗi lần gọi. Test khoá **tương đương ngữ nghĩa** với bản
quét, và khoá việc chỉ mục là view dẫn xuất (không vào payload, dựng lại
được sau `from_payload`).
"""
from __future__ import annotations

import itertools

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.seg_graph import SegmentGraph
from nn_aivie.core.state import GraphState


def _naive_blocked(sg: SegmentGraph, pair: tuple[int, int]) -> bool:
    a, b = pair
    return any(y == a or x == b for x, y in sg.cluster_of_pair)


def _naive_rival(sg: SegmentGraph, pair: tuple[int, int]) -> int:
    a, b = pair
    rival = 0
    for (x, y), ci in sg.cluster_of_pair.items():
        if y == a or x == b:
            rival = max(rival, int(sg.clusters[ci].power))
    return rival


def _built() -> tuple[GraphState, ParentStore]:
    st = GraphState()
    ps = ParentStore()
    ps.seg_min_frequency = 2
    cau = [
        "Ví dụ mẫu về câu",
        "Ví dụ mẫu thứ hai",
        "một ví dụ mẫu khác",
        "Hà Nội ở đâu",
        "Thủ đô Hà Nội ở đâu",
        "con mèo ở đâu rồi",
        "học sinh giỏi toán",
        "học sinh giỏi văn",
    ]
    for s in cau:
        ps.add_example(st.vocab.encode_tokens(s.split()), st.vocab, dap_id=None, priority=1.0)
    return st, ps


def test_tuong_duong_voi_quet_tuyen_tinh() -> None:
    st, ps = _built()
    sg = ps.seg
    assert sg.cluster_of_pair, "fixture phải sinh được ít nhất một cụm"

    toks = sorted({t for pair in sg.cluster_of_pair for t in pair})
    toks += [st.vocab.id_of("câu"), st.vocab.id_of("rồi")]
    for pair in itertools.product(toks, repeat=2):
        assert sg._bigram_overlap_blocked(pair) == _naive_blocked(sg, pair)  # noqa: SLF001
        assert sg._overlap_rival_power(pair) == _naive_rival(sg, pair)  # noqa: SLF001


def test_chi_muc_dong_bo_sau_promote_moi() -> None:
    """Cụm mới promote phải hiện ra ngay trong chỉ mục."""
    st, ps = _built()
    sg = ps.seg
    a, b = st.vocab.id_of("hoàn"), st.vocab.id_of("toàn")
    assert not sg._bigram_overlap_blocked((b, a))  # noqa: SLF001

    sg.promote_pair(a, b, ps.parents)
    # cụm (a,b): chặn cặp kết thúc ở a, và cặp bắt đầu ở b
    assert sg._bigram_overlap_blocked((b, st.vocab.id_of("câu")))  # noqa: SLF001
    assert sg._bigram_overlap_blocked((st.vocab.id_of("câu"), a))  # noqa: SLF001
    assert sg._overlap_rival_power((b, st.vocab.id_of("câu"))) == _naive_rival(  # noqa: SLF001
        sg, (b, st.vocab.id_of("câu"))
    )


def test_power_doc_song_theo_cluster() -> None:
    """Chỉ mục giữ id cụm, không giữ power — power đổi thì đọc ra giá trị mới."""
    st, ps = _built()
    sg = ps.seg
    (_x, y), ci = next(iter(sg.cluster_of_pair.items()))
    pair = (y, st.vocab.id_of("câu"))
    truoc = sg._overlap_rival_power(pair)  # noqa: SLF001
    # nâng đủ lớn để cụm này chắc chắn thành max, bất kể fixture
    sg.clusters[ci].power += 10_000
    sau = sg._overlap_rival_power(pair)  # noqa: SLF001
    assert sau == sg.clusters[ci].power
    assert sau == _naive_rival(sg, pair)
    assert sau > truoc


def test_chi_muc_khong_vao_payload_va_dung_lai_duoc() -> None:
    st, ps = _built()
    sg = ps.seg
    sg._ensure_overlap_index()  # noqa: SLF001

    payload = sg.to_payload()
    assert not any(str(k).startswith("_ov_") for k in payload)

    lai = SegmentGraph.from_payload(payload)
    # sau from_payload chưa có chỉ mục → phải dựng lười, và cho cùng kết quả
    assert getattr(lai, "_ov_by_right", None) is None
    for pair in sg.cluster_of_pair:
        assert lai._bigram_overlap_blocked(pair) == _naive_blocked(lai, pair)  # noqa: SLF001
        assert lai._overlap_rival_power(pair) == _naive_rival(lai, pair)  # noqa: SLF001
