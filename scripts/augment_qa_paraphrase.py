#!/usr/bin/env python3
"""
Augment gold QA bằng **paraphrase câu hỏi nhất quán theo chủ thể**.

Mỗi fact (định nghĩa / màu / hành động …) của một chủ thể được hỏi bằng nhiều
cách diễn đạt, tất cả ánh xạ về **một đáp canonical**. Mục tiêu:

1. Phủ paraphrase: fingerprint của `là gì` / `là con gì` / `là cái gì` đều có gold
   → khái quát hóa không còn phụ thuộc một chữ ký câu hỏi duy nhất.
2. Củng cố đường dẫn **đặc thù chủ thể** để áp đảo bleed chéo (mèo↔chó).
3. Khử xung đột: một chuỗi câu hỏi chỉ giữ một đáp (ưu tiên canonical từ bảng fact).

Đây là **dữ liệu dạy** (teaching data), không phải luật suy luận trong code.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Bảng fact: chủ thể -> {loại fact: đáp canonical}. Đáp luôn nhắc lại chủ thể để
# ràng buộc chủ thể–đáp mạnh (chống bleed chéo).
FACTS: dict[str, dict[str, str]] = {
    "Con mèo": {
        "def": "Con mèo là thú cưng.",
        "color": "Con mèo màu trắng.",
        "act": "Con mèo ngủ.",
    },
    "Con chó": {
        "def": "Con chó là thú cưng.",
        "color": "Con chó màu nâu.",
        "act": "Con chó sủa.",
    },
    "Con gà": {
        "def": "Con gà là loài chim.",
        "color": "Con gà màu vàng.",
        "act": "Con gà gáy.",
    },
    "Con cá": {
        "def": "Con cá là động vật.",
        "color": "Con cá màu bạc.",
        "act": "Con cá bơi.",
    },
    "Con chim": {
        "def": "Con chim là động vật.",
        "color": "Con chim màu nâu.",
        "act": "Con chim bay.",
    },
    "Hoa hồng": {
        "def": "Hoa hồng là loài hoa.",
        "color": "Hoa hồng màu đỏ.",
    },
    "Quả táo": {
        "def": "Quả táo là trái cây.",
        "color": "Quả táo màu đỏ.",
    },
}

# Mẫu paraphrase cho từng loại fact. {s} = chủ thể.
PARAPHRASE: dict[str, list[str]] = {
    "def": [
        "{s} là gì ?",
        "{s} là con gì ?",
        "{s} là cái gì ?",
        "{s} thuộc loại gì ?",
    ],
    "color": [
        "{s} màu gì ?",
        "{s} có màu gì ?",
        "{s} màu sắc gì ?",
    ],
    "act": [
        "{s} làm gì ?",
        "{s} đang làm gì ?",
    ],
}


def generate_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for subj, facts in FACTS.items():
        for ftype, answer in facts.items():
            for templ in PARAPHRASE.get(ftype, []):
                pairs.append((templ.format(s=subj), answer))
    return pairs


def merge_grade(grade: int) -> dict[str, int]:
    qa_dir = ROOT / "data" / f"grade_{grade:02d}" / "qa"
    train_path = qa_dir / "train.jsonl"

    rows: list[dict] = []
    if train_path.is_file():
        for line in train_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    canonical = dict(generate_pairs())

    # Khử xung đột: với câu hỏi có trong bảng canonical, ép về đáp canonical;
    # giữ nguyên các câu hỏi khác. Bỏ trùng (question,answer).
    out: list[dict] = []
    seen_q_canon: set[str] = set()
    seen_pair: set[tuple[str, str]] = set()
    overridden = 0
    for r in rows:
        q = r.get("question")
        a = r.get("answer")
        if q is None or a is None:
            out.append(r)
            continue
        if q in canonical:
            # đáp canonical sẽ được thêm ở dưới; bỏ qua bản cũ xung đột.
            if a != canonical[q]:
                overridden += 1
            continue
        if (q, a) in seen_pair:
            continue
        seen_pair.add((q, a))
        out.append(r)

    added = 0
    for q, a in generate_pairs():
        if q in seen_q_canon:
            continue
        seen_q_canon.add(q)
        out.append({"question": q, "answer": a})
        added += 1

    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"added": added, "overridden": overridden, "total": len(out)}


def main() -> None:
    for grade in (3, 4, 5, 6, 7):
        qa = ROOT / "data" / f"grade_{grade:02d}" / "qa" / "train.jsonl"
        if not qa.is_file():
            continue
        stats = merge_grade(grade)
        print(
            f"grade_{grade:02d}: +{stats['added']} paraphrase, "
            f"{stats['overridden']} conflicts resolved, total={stats['total']}"
        )


if __name__ == "__main__":
    main()
