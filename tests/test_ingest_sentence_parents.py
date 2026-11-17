"""TK_045 — ingest-time sentence parents + episodes."""

from __future__ import annotations

import json
from pathlib import Path

from nn_aivie.core.state import GraphState
from nn_aivie.training.dataset_spec import DatasetSpec
from nn_aivie.training.pipeline import ingest_text_file, run_manifest


def test_ingest_splits_multi_sentence(tmp_path: Path):
    corpus = tmp_path / "toy.jsonl"
    # one article, two sentences
    line = json.dumps(
        "Ngô Quyền lập nước. Ông thắng ở Bạch Đằng.",
        ensure_ascii=False,
    )
    corpus.write_text(line + "\n", encoding="utf-8")
    st = GraphState()
    spec = DatasetSpec(path=str(corpus), priority=0.9)
    groups: list = []
    stats = ingest_text_file(
        st,
        corpus,
        spec,
        sentence_parents=True,
        episode_groups=groups,
        defer_skip=True,
    )
    assert stats["split_docs"] == 1
    assert stats["sent_parents"] >= 2
    assert len(st.parents.parents) >= 2
    # build episodes
    made, matched = st.parents.build_episodes_from_provenance(groups, min_members=2)
    assert made >= 1
    assert matched >= 2
    assert len(st.parents.episodes) >= 1
    members = [m for m, _ in st.parents.episodes[0].members]
    assert len(members) >= 2


def test_qa_not_split(tmp_path: Path):
    corpus = tmp_path / "qa.jsonl"
    line = json.dumps(
        "Hỏi: Ngô Quyền là ai Đáp: Vua sáng lập nhà Ngô. Ông thắng Bạch Đằng.",
        ensure_ascii=False,
    )
    corpus.write_text(line + "\n", encoding="utf-8")
    st = GraphState()
    spec = DatasetSpec(path=str(corpus), priority=0.95)
    stats = ingest_text_file(
        st, corpus, spec, sentence_parents=True, episode_groups=[], defer_skip=True
    )
    assert stats["split_docs"] == 0
    assert len(st.parents.parents) == 1
