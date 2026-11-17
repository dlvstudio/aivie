"""QA cases từ val jsonl (TK_019: phần eval fingerprint v2 đã xóa — eval nay ở eval/v4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nn_aivie.data.qa import render_qa

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QA_VAL_PATH = _REPO_ROOT / "data" / "grade_03" / "qa" / "val.jsonl"


@dataclass(frozen=True)
class QACase:
    """Một dòng val QA — không hardcode must_contain / must_not_contain."""

    prompt: str
    question: str
    gold_answer: str
    expected_fingerprint: str


def load_qa_cases(
    path: Path | str | None = None,
    *,
    limit: int | None = None,
) -> tuple[QACase, ...]:
    """Nạp case từ val jsonl (question + gold answer)."""
    p = Path(path) if path is not None else DEFAULT_QA_VAL_PATH
    if not p.exists():
        return ()
    cases: list[QACase] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        prompt, _ = render_qa(question, "")
        cases.append(
            QACase(
                prompt=prompt,
                question=question,
                gold_answer=answer,
                expected_fingerprint="",  # TK_019: fingerprint v2 đã bỏ
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return tuple(cases)
