"""Mốc RANH GIỚI ĐƠN VỊ học từ EOS, không đoán từ danh sách dấu câu (Luật 0)."""

from __future__ import annotations

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import SPECIAL_EOS
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import _ends_unit
from nn_aivie.training.ingest import ingest_qa, ingest_statements

_SENTS = [
    "Phở là món ăn nổi tiếng ở Việt Nam .",
    "Hà Nội là thủ đô của Việt Nam .",
    "Việt Nam là quốc gia ở Đông Nam Á .",
]


def _trained() -> GraphState:
    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, _SENTS, split_punct=True)
    return st


def test_ingest_appends_eos():
    st = _trained()
    rec = st.parents.parents[0]
    assert st.vocab.token_of(rec.source_ids[-1]) == SPECIAL_EOS


def test_eos_not_in_match_ids():
    """EOS KHÔNG được vào match_ids — nếu vào thì nó nhiễu retrieval (mọi cha đều có nó)."""
    st = _trained()
    eos = st.vocab.id_of(SPECIAL_EOS)
    assert all(eos not in rec.match_ids for rec in st.parents.parents)


def test_wseq_learns_end_marker():
    st = _trained()
    seq = st.adjacency.layer(EdgeKind.SEQUENTIAL)
    eos = st.vocab.id_of(SPECIAL_EOS)
    dot = st.vocab.id_of(".")
    assert seq.get(dot, eos) > 0, "'.' phải học được cạnh → EOS"


def test_boundary_separates_end_marker_from_content():
    """Tín hiệu P(→EOS) tách ranh giới khỏi nội dung — thứ 2 tín hiệu khác KHÔNG làm được.

    Đã thử và thất bại (TK_012 §9.5): entropy W_seq ('là'=5.45 > ','=4.31) và tỉ lệ
    đứng-cuối-chuỗi (','=':'='Đáp:'=0.000, ngang từ nội dung). Đồ thị không mã hoá "kết
    thúc đơn vị" cho tới khi dữ liệu train ĐÁNH DẤU nó bằng EOS.
    """
    st = _trained()
    cfg = HCAConfig()
    assert _ends_unit(st, st.vocab.id_of("."), cfg)
    for t in ("là", "Việt", "Nam", "ăn"):
        assert not _ends_unit(st, st.vocab.id_of(t), cfg), f"{t!r} không phải mốc kết"


def test_boundary_off_on_legacy_checkpoint():
    """Đồ thị KHÔNG có cạnh →EOS (train trước 2026-07-15) → False, lùi an toàn.

    Quan trọng: checkpoint cũ phải không tệ hơn hành vi hiện tại, không phải bật bừa.
    """
    st = GraphState()
    st.parents = ParentStore()
    ids = list(st.vocab.encode_tokens("Phở là món ăn ngon .".split()))
    for a, b in zip(ids, ids[1:]):
        st.adjacency.add(a, b, 1.0, EdgeKind.SEQUENTIAL)  # W_seq KHÔNG có →EOS
    assert not _ends_unit(st, st.vocab.id_of("."), HCAConfig())


def test_boundary_knob_off():
    st = _trained()
    from dataclasses import replace

    cfg = replace(HCAConfig(), ar_eos_boundary_min_lift=0.0)
    assert not _ends_unit(st, st.vocab.id_of("."), cfg)


def test_boundary_survives_long_units():
    """LIFT phải bất biến với ĐỘ DÀI ĐƠN VỊ — đây là lý do không dùng ngưỡng tuyệt đối.

    Regression thật (verify network_v10, 2026-07-16 → FAIL): bản đầu dùng ngưỡng tuyệt đối
    `P(EOS|w0) >= 0.25`, hiệu chỉnh trên corpus đồ chơi 1-câu/đơn-vị (P=1.000). Corpus thật
    là đoạn 2-4 câu + bài dài → '.' xuất hiện ~17 lần/đơn vị, chỉ 1 lần trước EOS → P=0.057
    → ends_unit=False → at_boundary vẫn mù dù đồ thị ĐÃ học mốc.

    P đổi 6× giữa hai corpus dưới đây (1.000 → 0.167); lift gần như không đổi (6.0 → 7.0).
    """
    long_units = [
        "Phở là món ăn ngon . Hà Nội là thủ đô của Việt Nam . Việt Nam nằm ở Đông Nam Á . "
        "Người dân rất thân thiện . Ẩm thực nơi đây phong phú . Du khách đến rất đông ."
    ] * 3
    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, long_units, split_punct=True)
    cfg = HCAConfig()
    assert _ends_unit(st, st.vocab.id_of("."), cfg)
    for t in ("là", "Việt", "Nam", "ngon"):
        assert not _ends_unit(st, st.vocab.id_of(t), cfg)


def test_qa_answer_learns_end_marker():
    """Đáp KẾT THÚC ở đâu cũng phải học — REINFORCED gồm cả `→ EOS`."""
    st = GraphState()
    st.parents = ParentStore()
    ingest_qa(st, [("Thủ đô Việt Nam là gì", "Thủ đô Việt Nam là Hà Nội .")], split_punct=True)
    rec = st.parents.parents[0]
    assert st.vocab.token_of(rec.source_ids[-1]) == SPECIAL_EOS
    rf = st.adjacency.layer(EdgeKind.REINFORCED)
    assert rf.get(st.vocab.id_of("."), st.vocab.id_of(SPECIAL_EOS)) > 0


def test_train_pipeline_appends_eos(tmp_path):
    """Đường `train.py` THẬT (pipeline.ingest_text_file) cũng phải nối EOS.

    Bẫy đã vấp: `_with_eos` ban đầu chỉ thêm vào `ingest.py` (ingest_statements/ingest_qa →
    train_v3.py). Nhưng `train.py` → `pipeline.run_manifest` KHÔNG dùng ingest.py — nó gọi
    thẳng `g1_and_wseq` + `add_example`. Nếu không sửa cả hai, mẻ train chính chạy hàng giờ
    xong vẫn không có mốc kết, và không ai biết cho tới khi đo.
    """
    import json

    from nn_aivie.training.dataset_spec import DatasetSpec
    from nn_aivie.training.pipeline import ingest_text_file

    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in _SENTS), encoding="utf-8")
    st = GraphState()
    st.parents = ParentStore()
    ingest_text_file(st, p, DatasetSpec(path=str(p), priority=0.9))

    rec = st.parents.parents[0]
    assert st.vocab.token_of(rec.source_ids[-1]) == SPECIAL_EOS
    assert st.vocab.id_of(SPECIAL_EOS) not in rec.match_ids
    assert _ends_unit(st, st.vocab.id_of("."), HCAConfig())
    assert not _ends_unit(st, st.vocab.id_of("là"), HCAConfig())
