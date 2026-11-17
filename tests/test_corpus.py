"""Tests for parquet corpus (nanochat-style shards)."""

from pathlib import Path

import pyarrow.parquet as pq

from nn_aivie.data.corpus import TextCorpus, load_manifest
from nn_aivie.data.parquet_io import list_parquet_shards, shard_paths_for_split
from nn_aivie.data.registry import grade_manifest_path


def test_grade_03_parquet_shards_exist():
    manifest = load_manifest(grade_manifest_path(3))
    general = next(d for d in manifest.datasets if d.id == "general")
    assert general.data_dir is not None
    shards = list_parquet_shards(general.data_dir)
    assert len(shards) >= 1
    assert shards[-1].name.endswith(".parquet")


def test_nanochat_val_split_convention():
    manifest = load_manifest(grade_manifest_path(3))
    general = next(d for d in manifest.datasets if d.id == "general")
    train_paths = shard_paths_for_split(general.data_dir, "train")
    val_paths = shard_paths_for_split(general.data_dir, "val")
    assert len(train_paths) >= 1
    if len(list_parquet_shards(general.data_dir)) > 1:
        assert val_paths == [list_parquet_shards(general.data_dir)[-1]]


def test_text_corpus_parquet_load():
    manifest = load_manifest(grade_manifest_path(3))
    corpus = TextCorpus(manifest.for_phase("G1"), grade=3, split="train")
    n = 0
    for rec in corpus:
        assert rec.text
        n += 1
        if n >= 3:
            break
    assert n == 3


def test_parquet_has_text_column():
    manifest = load_manifest(grade_manifest_path(3))
    general = next(d for d in manifest.datasets if d.id == "general")
    pf = pq.ParquetFile(shard_paths_for_split(general.data_dir, "train")[0])
    assert "text" in pf.schema.names
