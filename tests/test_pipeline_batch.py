"""Batch ingest pipeline (TK_027) — defer promote/index đến cuối file."""

from __future__ import annotations

import tempfile
from pathlib import Path

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.seg_graph import CLUSTER_OFFSET
from nn_aivie.core.state import GraphState
from nn_aivie.training.dataset_spec import DatasetSpec
from nn_aivie.training.pipeline import ingest_text_file


def _ps() -> tuple[GraphState, ParentStore]:
    st = GraphState()
    ps = ParentStore()
    ps.seg_min_frequency = 3
    ps.ensure_skip_ledger(min_count=3, min_pmi=0.0)
    st.parents = ps
    return st, ps


def test_finalize_batch_ingest_promotes_clusters():
    st, ps = _ps()
    ps.begin_ingest_batch()
    for _ in range(5):
        ids = st.vocab.encode_tokens("Ví dụ mẫu".split())
        ps.add_example(ids, st.vocab, dap_id=None, defer_index=True)
    ps.finalize_batch_ingest(st.vocab)
    assert len(ps.seg.clusters) >= 1
    assert any(
        s.kind == "cluster"
        for rec in ps.parents
        for s in rec.slot_chain
    )


def test_ingest_text_file_batch_mode(tmp_path: Path):
    corpus = tmp_path / "tiny.jsonl"
    corpus.write_text(
        '"Ví dụ mẫu"\n"Ví dụ mẫu"\n"Ví dụ mẫu"\n"Ví dụ mẫu"\n"Ví dụ mẫu"\n',
        encoding="utf-8",
    )
    st, ps = _ps()
    spec = DatasetSpec(path=str(corpus), priority=1.0, eta=0.05, split_punct=True)
    stats = ingest_text_file(st, corpus, spec)
    assert stats["texts"] == 5
    assert len(ps.parents) == 1
    assert ps.parents[0].power == 5
    assert len(ps.seg.clusters) >= 1
    cl_slots = [
        s for rec in ps.parents for s in rec.slot_chain if s.kind == "cluster"
    ]
    assert cl_slots


def test_batch_matches_rebuild_seg_graph():
    """Batch ingest cuối file ≡ replay ISG từ source_ids."""
    st, ps = _ps()
    texts = ["Hà Nội ở đâu", "Thủ đô Hà Nội đặt ở đâu", "Con mèo ở đâu"]
    ps.begin_ingest_batch()
    for t in texts:
        ids = st.vocab.encode_tokens(t.split())
        ps.add_example(ids, st.vocab, dap_id=None, defer_index=True)
    ps.finalize_batch_ingest(st.vocab)
    n_parents = len(ps.parents)
    n_clusters = len(ps.seg.clusters)
    chains_batch = [tuple(rec.slot_chain) for rec in ps.parents]

    ps.rebuild_seg_graph(st.vocab)
    assert len(ps.parents) == n_parents
    assert len(ps.seg.clusters) == n_clusters
    chains_rebuild = [tuple(rec.slot_chain) for rec in ps.parents]
    assert chains_batch == chains_rebuild


def test_incremental_sync_skips_unchanged_parents():
    """Finalize batch: cha cũ không rewire — không cần sync lại."""
    st, ps = _ps()
    ps.begin_ingest_batch()
    ids_a = st.vocab.encode_tokens("Hà Nội là thủ đô".split())
    ps.add_example(ids_a, st.vocab, dap_id=None, defer_index=True)
    ps.finalize_batch_ingest(st.vocab)
    snap = [
        (rec.match_ids, rec.phrase_ids, tuple(rec.slot_chain))
        for rec in ps.parents
    ]

    ps.begin_ingest_batch()
    ids_b = st.vocab.encode_tokens("Con mèo ở đâu".split())
    ps.add_example(ids_b, st.vocab, dap_id=None, defer_index=True)
    ps.finalize_batch_ingest(st.vocab)

    assert len(ps.parents) == 2
    assert (ps.parents[0].match_ids, ps.parents[0].phrase_ids, tuple(ps.parents[0].slot_chain)) == snap[0]
