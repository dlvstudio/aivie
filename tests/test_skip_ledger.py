"""Test incremental skip ledger + rewire compress (TK_036 S3)."""

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.training.ingest import ingest_statements


def test_skip_ledger_incremental_on_ingest():
    st = GraphState()
    ps = ParentStore()
    st.parents = ps
    ps.seg_min_frequency = 2
    ps.ensure_skip_ledger(min_count=2, min_pmi=0.0)
    sents = [
        "phở Việt Nam rất nổi tiếng .",
        "phở Việt Nam rất ngon .",
        "phở Việt Nam rất phổ biến .",
        "ẩm thực Việt Nam rất đa dạng .",
    ]
    ingest_statements(st, sents, eta=1.0, priority=0.9)
    ledger = ps.skip_ledger
    assert ledger is not None
    assert ledger._synced
    pho = st.vocab.id_of("phở")
    vn = (st.vocab.id_of("Việt"), st.vocab.id_of("Nam"))
    assert ps.skip_spans or ledger.pair.get((pho, vn), 0) > 0


def test_rewire_refreshes_skip_ledger():
    """Khi ISG gom Việt+Nam → cluster, flush ledger thấy Phở→VN (slot gap=1)."""
    st = GraphState()
    ps = ParentStore()
    st.parents = ps
    ps.seg_min_frequency = 2
    ps.ensure_skip_ledger(min_count=2, min_pmi=0.0)
    # Câu đa dạng: tránh promote phở+Việt (chỉ Việt+Nam đủ freq)
    sents = [
        "phở Việt Nam rất nổi tiếng .",
        "phở Việt Nam rất ngon .",
        "phở Việt Nam rất phổ biến .",
        "ẩm thực Việt Nam rất đa dạng .",
    ]
    ingest_statements(st, sents, eta=1.0, priority=0.9)
    pho = st.vocab.id_of("phở")
    assert any(c.left == st.vocab.id_of("Việt") for c in ps.seg.clusters)
    assert any(
        a == pho and ps.skip_ledger.pair.get((a, right), 0) > 0
        for (a, right) in ps.skip_ledger.pair
    )


def test_skip_power_counts_pair_once_per_parent():
    """Cặp lặp nhiều lần trong 1 cha (như 'Ví dụ' lặp trong bài) → power = số cha, không bội."""
    import nn_aivie.training.skip_span as ss

    st = GraphState()
    ps = ParentStore()
    st.parents = ps
    ps.seg_min_frequency = 99  # giữ raw content, không promote cụm
    ps.ensure_skip_ledger(min_count=2, min_pmi=-9.0, dir_ratio=1.0)
    # "báo"/"mưa to" lặp 3 lần trong mỗi câu → scan sinh key trùng trong cùng cha.
    sents = [
        "báo mưa to báo mưa to báo mưa to xong roi alpha .",
        "báo mưa to báo mưa to báo mưa to xong roi beta .",
    ]
    ingest_statements(st, sents, eta=1.0, priority=0.9)
    ledger = ps.skip_ledger
    assert len(ps.parents) == 2

    # xác nhận scan THỰC SỰ sinh key trùng trong 1 cha (điều kiện gây bội trước đây)
    pbs = ss.build_phrase_by_start(ps, min_count=2)
    key = None
    for pi, p in enumerate(ps.parents):
        scanned = ledger.scan_parent(
            pi, p.slot_chain, ps.seg, st.vocab, ps._gdf, len(ps.parents), pbs
        )
        from collections import Counter

        keyc = Counter(k for k, _g, _pos in scanned)
        dupes = [k for k, c in keyc.items() if c > 1]
        assert dupes, f"parent {pi} không có key trùng — test không kiểm được"
        key = key or dupes[0]

    # power của cặp trùng = số cha (2), KHÔNG phải tổng số lần scan (4+)
    assert ledger.pair[key] == 2
    assert len(ledger.prov[key]) == 2  # mỗi cha 1 provenance


def ledger_hits_pho_vn(ps, pho, vn) -> bool:
    if (pho, vn) in ps._skip_lookup:
        return True
    ledger = ps.skip_ledger
    return bool(ledger and ledger.pair.get((pho, vn), 0) > 0)


def _pho_vn_sents():
    return [
        "phở Việt Nam rất nổi tiếng .",
        "phở Việt Nam rất ngon .",
        "phở Việt Nam rất phổ biến .",
        "ẩm thực Việt Nam rất đa dạng .",
    ]


def _skip_snapshot(ps):
    ledger = ps.skip_ledger
    return {
        "pair": dict(ledger.pair) if ledger else {},
        "spans": len(ps.skip_spans or []),
        "skip_ids": [tuple(rec.skip_span_ids) for rec in ps.parents],
    }


def test_incremental_flush_matches_single_batch():
    """Một file ingest (một flush) — snapshot skip ledger ổn định."""
    st = GraphState()
    ps = ParentStore()
    st.parents = ps
    ps.seg_min_frequency = 2
    ps.ensure_skip_ledger(min_count=2, min_pmi=0.0)
    ingest_statements(st, _pho_vn_sents(), eta=1.0, priority=0.9)
    got = _skip_snapshot(ps)
    assert got["pair"]
    assert len(ps.parents) == 4


def test_phrase_by_start_built_once_per_flush(monkeypatch):
    """B2: build_phrase_by_start chỉ gọi 1 lần mỗi flush (không N lần theo số cha)."""
    import nn_aivie.training.skip_span as ss

    calls = {"n": 0}
    orig = ss.build_phrase_by_start

    def counted(ps, **kw):
        calls["n"] += 1
        return orig(ps, **kw)

    monkeypatch.setattr(ss, "build_phrase_by_start", counted)
    st = GraphState()
    ps = ParentStore()
    st.parents = ps
    ps.seg_min_frequency = 2
    ps.ensure_skip_ledger(min_count=2, min_pmi=0.0)
    ingest_statements(st, _pho_vn_sents(), eta=1.0, priority=0.9)
    assert calls["n"] == 1
