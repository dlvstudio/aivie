from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaggedSentence:
    tokens: list[str]
    tags: list[str]  # B-W/I-W


def read_conll_tagged_file(path: str | Path) -> list[TaggedSentence]:
    """Đọc file CoNLL-style (UTS_WTK): mỗi dòng 'token<TAB>B-W|I-W', câu cách nhau dòng trống."""
    path = Path(path)
    sents: list[TaggedSentence] = []
    toks: list[str] = []
    tags: list[str] = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            if toks:
                sents.append(TaggedSentence(tokens=toks, tags=tags))
                toks, tags = [], []
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.rsplit(None, 1)
            if len(parts) < 2:
                continue
        tag = parts[-1].replace("_", "-")  # chấp nhận cả B_W/I_W
        if tag not in ("B-W", "I-W"):
            continue
        toks.append(parts[0])
        tags.append(tag)
    if toks:
        sents.append(TaggedSentence(tokens=toks, tags=tags))
    return sents


def read_tagged_file(path: str | Path) -> list[TaggedSentence]:
    """Auto-detect: có TAB → CoNLL (UTS_WTK), ngược lại → underscore (VLSP-style)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return read_conll_tagged_file(path) if "\t" in line else read_underscore_tagged_file(path)
    return []


def read_underscore_tagged_file(path: str | Path) -> list[TaggedSentence]:
    """Đọc file mỗi dòng là câu, word nối bằng '_' (VLSP-style)."""
    path = Path(path)
    sents: list[TaggedSentence] = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        toks: list[str] = []
        tags: list[str] = []
        for item in line.split():
            syls = [s for s in item.split("_") if s]
            for i, s in enumerate(syls):
                toks.append(s)
                tags.append("B-W" if i == 0 else "I-W")
        if toks:
            sents.append(TaggedSentence(tokens=toks, tags=tags))
    return sents

