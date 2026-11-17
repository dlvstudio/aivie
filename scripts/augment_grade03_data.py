#!/usr/bin/env python3
"""Bổ sung corpus grade_03 từ HF + giữ JSONL hiện có."""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1] / "data" / "grade_03"


def load_jsonl(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line)["text"])
    return out


def save_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    unique = []
    for t in texts:
        t = t.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        unique.append(t)
    path.write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in unique) + "\n",
        encoding="utf-8",
    )


def word_count(s: str) -> int:
    return len(s.split())


def fetch_general(max_rows: int = 400) -> list[str]:
    ds = load_dataset("thanhdath/vietnamese-sentences", split="train", streaming=True)
    out = []
    for row in ds:
        text = row["text"].strip()
        wc = word_count(text)
        if 3 <= wc <= 12 and not re.search(r"http|www\.|@", text):
            out.append(text)
        if len(out) >= max_rows:
            break
    return out


def fetch_copula(max_rows: int = 200) -> list[str]:
    """Câu có ' là ' — phù hợp pattern copula."""
    ds = load_dataset("thanhdath/vietnamese-sentences", split="train", streaming=True)
    out = []
    for row in ds:
        text = row["text"].strip()
        if " là " not in text:
            continue
        wc = word_count(text)
        if 4 <= wc <= 14 and not re.search(r"http|www\.|@", text):
            out.append(text)
        if len(out) >= max_rows:
            break
    return out


def main() -> None:
    general_train = ROOT / "general" / "train.jsonl"
    general_val = ROOT / "general" / "val.jsonl"
    copula_train = ROOT / "copula" / "train.jsonl"
    copula_val = ROOT / "copula" / "val.jsonl"

    print("Fetching HF general sentences...")
    hf_general = fetch_general(400)
    print(f"  got {len(hf_general)} general")

    print("Fetching HF copula-like sentences...")
    hf_copula = fetch_copula(200)
    print(f"  got {len(hf_copula)} copula")

    merged_general = load_jsonl(general_train) + hf_general
    merged_copula = load_jsonl(copula_train) + hf_copula

    # val: giữ cũ + lấy mẫu từ cuối HF batch
    val_general = load_jsonl(general_val) + hf_general[-20:]
    val_copula = load_jsonl(copula_val) + hf_copula[-15:]

    save_jsonl(general_train, merged_general)
    save_jsonl(general_val, val_general)
    save_jsonl(copula_train, merged_copula)
    save_jsonl(copula_val, val_copula)

    print(f"general train: {len(load_jsonl(general_train))}")
    print(f"general val:   {len(load_jsonl(general_val))}")
    print(f"copula train:  {len(load_jsonl(copula_train))}")
    print(f"copula val:    {len(load_jsonl(copula_val))}")


if __name__ == "__main__":
    main()
