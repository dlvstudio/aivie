#!/usr/bin/env python3
"""Sync vi_QA_hoi_dap lesson examples from qa/train.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from nn_aivie.data.qa import render_qa

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "grade_03" / "qa" / "train.jsonl"
LESSON = ROOT / "lessons" / "vi_QA_hoi_dap.lesson.yaml"

HEADER = """lesson_id: vi_grade3_qa
grade: 3
repeat: 3
eta_teach: 0.5
w_init: 2.0

pattern_id: vi_QA_hoi_dap
slots:
  - const: "Hỏi:"
  - var: Q
  - const: "Đáp:"
  - var: A

bind:
  Q: câu hỏi
  A: câu trả lời

examples:
"""


def main() -> int:
    short = [
        "Hỏi: ai Đáp: Lan",
        "Hỏi: gì Đáp: toán",
        "Hỏi: sao Đáp: vui",
    ]
    seen: set[str] = set(short)
    examples: list[str] = list(short)
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            text, _ = render_qa(str(obj["question"]), str(obj["answer"]))
            if text not in seen:
                seen.add(text)
                examples.append(text)
    body = HEADER + "\n".join(f'  - "{ex}"' for ex in examples) + "\n"
    LESSON.write_text(body, encoding="utf-8")
    print(f"Wrote {len(examples)} examples -> {LESSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
