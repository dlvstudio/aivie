"""TK_035 — đơn vị đoạn 2–4 câu."""

from __future__ import annotations

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.data.paragraph_units import (
    PARAGRAPH_POOL_PRIORITY,
    PARAGRAPH_PRIORITY_BAND,
    count_punctuation_stats,
    is_quality_paragraph_unit,
    iter_sentence_windows,
    join_sentences,
)
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import (
    _conv_pool_tail,
    _is_paragraph_parent,
    _paragraph_pool_indices,
)


def test_join_sentences_keeps_punctuation():
    sents = [
        "Nhà Lý thành lập năm 1009.",
        "Lý Thái Tổ dời đô về Thăng Long, mở nền văn hóa mới!",
    ]
    unit = join_sentences(sents)
    assert "1009." in unit
    assert "," in unit
    assert "!" in unit


def test_iter_sentence_windows_2_to_4():
    sents = [f"Câu {i} là một câu tiếng Việt đủ dài." for i in range(5)]
    windows = list(iter_sentence_windows(sents, min_sent=2, max_sent=4))
    assert len(windows) >= 6
    assert all(is_quality_paragraph_unit(w) for w in windows[:3])


def test_is_quality_paragraph_unit_rejects_single_sentence():
    one = "Đây là một câu đủ dài để qua quality gate."
    assert not is_quality_paragraph_unit(one)


def test_paragraph_pool_priority_band():
    st = GraphState()
    ps = ParentStore()
    ids = st.vocab.encode_tokens
    short = ids("Nếu trời mưa thì em ở nhà .".split())
    long = ids(
        "Nếu trời mưa thì em ở nhà . "
        "Gia đình em rất vui , chúng em cùng chơi !".split()
    )
    ps.add_example(short, st.vocab, dap_id=None, priority=0.3)
    ps.add_example(long, st.vocab, dap_id=None, priority=PARAGRAPH_POOL_PRIORITY)
    ps.rebuild_hierarchical_index()
    st.parents = ps
    pool = _paragraph_pool_indices(ps)
    assert len(pool) == 1
    pi = next(iter(pool))
    assert _is_paragraph_parent(ps, pi)
    lo, hi = PARAGRAPH_PRIORITY_BAND
    assert lo <= ps.parents[pi].priority <= hi


def test_conv_pool_prefers_longer_paragraph_tail():
    st = GraphState()
    ps = ParentStore()
    ids = st.vocab.encode_tokens
    q = ids("Nếu trời mưa".split())
    short = ids("Nếu trời mưa thì em ở nhà .".split())
    long = ids(
        "Nếu trời mưa thì em ở nhà . "
        "Gia đình em rất vui , chúng em cùng chơi trò chơi !".split()
    )
    ps.add_example(short, st.vocab, dap_id=None, priority=0.3)
    ps.add_example(long, st.vocab, dap_id=None, priority=PARAGRAPH_POOL_PRIORITY)
    ps.rebuild_hierarchical_index()
    st.parents = ps
    cfg = HCAConfig(paragraph_prefer_long_tail=True, conv_pool_min_score=0.0)
    tail = _conv_pool_tail(st, q, cfg)
    assert tail is not None
    assert len(tail) > 6
    text = " ".join(st.vocab.token_of(t) for t in tail)
    assert "," in text or "!" in text


def test_count_punctuation_stats():
    units = [
        "A là x. B là y, và C là z!",
        "Một câu. Hai câu?",
    ]
    st = count_punctuation_stats(units)
    assert st["units"] == 2
    assert st["comma"] >= 1
    assert st["exclaim"] >= 1
    assert st["multi_sent"] == 2
