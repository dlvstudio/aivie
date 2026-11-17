"""
Corpus thống nhất — jsonl = danh sách text; mỗi **dòng** là **một** chuỗi JSON.

Ví dụ file `corpus.jsonl`:

  "phở Việt Nam rất ngon ."
  "Hỏi: Phở là gì ? Đáp: Phở là món ăn truyền thống ."
  "Đoạn dài nhiều câu . Câu hai . Câu ba ."

Một dòng = một đơn vị train (câu, đoạn, bài, khối Q&A — tùy nguồn đã ghép sẵn).
Không dùng mảng trong mảng: không phải ["a","b"] trên một dòng.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def iter_texts(path: Path | str) -> Iterator[str]:
    """Đọc jsonl — mỗi dòng một JSON string."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: JSON không hợp lệ — {e}") from e
            if isinstance(row, list):
                raise ValueError(
                    f"{p}:{lineno}: cần một chuỗi mỗi dòng, không phải mảng — "
                    "tách thành nhiều dòng jsonl"
                )
            text = str(row).strip()
            if text:
                yield text


def load_all_texts(path: Path | str) -> list[str]:
    return list(iter_texts(path))


def write_text(path: Path | str, text: str) -> None:
    """Ghi một dòng = một chuỗi JSON."""
    t = text.strip()
    if not t:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")


def write_texts(path: Path | str, texts: Iterator[str] | list[str]) -> int:
    """Ghi nhiều dòng — mỗi text một dòng (một lần mở file)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a", encoding="utf-8") as f:
        for text in texts:
            t = text.strip()
            if not t:
                continue
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            n += 1
    return n
