"""Tests text_corpus — một chuỗi JSON mỗi dòng."""

import json
import tempfile
from pathlib import Path

import pytest

from nn_aivie.training.text_corpus import iter_texts, write_text, write_texts


def test_iter_texts_one_string_per_line():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.jsonl"
        write_texts(p, ["câu một .", "Hỏi: ai ? Đáp: em ."])
        assert list(iter_texts(p)) == ["câu một .", "Hỏi: ai ? Đáp: em ."]


def test_iter_texts_newline_inside_string():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "para.jsonl"
        para = "Đoạn một .\n\nĐoạn hai ."
        p.write_text(json.dumps(para, ensure_ascii=False) + "\n", encoding="utf-8")
        assert list(iter_texts(p)) == [para]


def test_reject_array_line():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.jsonl"
        p.write_text('["a", "b"]\n', encoding="utf-8")
        with pytest.raises(ValueError, match="mảng"):
            list(iter_texts(p))
