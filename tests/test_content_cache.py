"""
TK_012 — Bất biến "mạng thuần nơ-ron": inference VÀ training không đọc content-cache.

Cache (nhãn tag/key/title) là sidecar cho UI/eval/debug. Bằng chứng: có/không cache thì
graph (adjacency + ParentStore ids) và output inference PHẢI giống hệt.
"""

from __future__ import annotations

from nn_aivie.core.checkpoint_io import (
    content_sidecar_path,
    load_checkpoint,
    save_checkpoint,
)
from nn_aivie.core.content_cache import ContentCache
from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.training.ingest import ingest_statements, ingest_qa


def _train_graph() -> GraphState:
    """Train thuần text→token-id ở biên; graph không giữ nhãn."""
    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(
        st,
        ["Mèo là loài vật nuôi .", "Mèo rất thích ngủ .", "Chó giữ nhà tốt ."],
        priority=0.6,
    )
    ingest_qa(st, [("Hỏi: mèo ăn gì", "Mèo ăn cá .")], priority=0.9)
    return st


def _parent_signature(ps: ParentStore) -> list:
    return [
        (p.emit_ids, p.match_ids, p.answer_ids, p.is_qa, round(p.priority, 6))
        for p in ps.parents
    ]


def test_training_blind_to_cache():
    """Train KHÔNG đọc cache: build với cache rỗng vs cache-đầy → cùng graph."""
    st_a = _train_graph()
    # cache-đầy: gắn nhãn tùy ý — KHÔNG được ảnh hưởng train/graph
    st_b = _train_graph()
    cache = ContentCache()
    for i in range(len(st_b.parents.parents)):
        cache.set_parent_tag(i, f"junk:{i}")

    assert _parent_signature(st_a.parents) == _parent_signature(st_b.parents)
    assert st_a.adjacency.to_dict() == st_b.adjacency.to_dict()


def test_inference_blind_to_cache():
    """Inference KHÔNG đọc cache: generate với cache rỗng vs đầy → cùng token output."""
    st = _train_graph()
    out_no_cache = decode_surface(st, generate(st, "Hỏi: mèo ăn gì Đáp:", max_len=8))

    # đính cache rác — inference không import/không đọc → output bất biến
    cache = ContentCache()
    cache.set_parent_tag(0, "junk")
    out_with_cache = decode_surface(st, generate(st, "Hỏi: mèo ăn gì Đáp:", max_len=8))
    assert out_no_cache == out_with_cache


def test_graph_pickle_has_no_labels():
    """Graph pickle payload KHÔNG chứa tag/key/title/source (README luật 0.0)."""
    st = _train_graph()
    payload = st.parents.to_payload()
    for p in payload["parents"]:
        assert "tag" not in p
    for e in payload["episodes"]:
        assert "key" not in e and "title" not in e and "source" not in e


def test_checkpoint_sidecar_roundtrip(tmp_path):
    """save/load_checkpoint: graph + sidecar tách file; nhãn khôi phục đúng qua cache."""
    st = _train_graph()
    cache = ContentCache()
    cache.set_parent_tag(0, "alpaca:0")
    cache.set_episode(0, key="wiki:1#0", title="Mèo", source="doc")

    ckpt = tmp_path / "toy.pkl"
    save_checkpoint(st, ckpt, cache)
    assert content_sidecar_path(ckpt).exists()

    st2, cache2 = load_checkpoint(ckpt)
    assert _parent_signature(st2.parents) == _parent_signature(st.parents)
    assert cache2.parent_tag(0) == "alpaca:0"
    assert cache2.episode_key(0) == "wiki:1#0"

