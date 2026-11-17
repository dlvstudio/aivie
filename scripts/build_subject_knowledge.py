#!/usr/bin/env python3
"""Sinh corpus kiến thức theo môn (toán, địa lý, …) cho lớp 3–7."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nn_aivie.data.subject_knowledge import SUBJECT_META, all_grades, grade_subjects

VAL_RATIO = 0.12
MIN_VAL = 2
LESSONS_DIR = ROOT / "lessons"


def _split_train_val(items: list[str]) -> tuple[list[str], list[str]]:
    if len(items) <= MIN_VAL + 2:
        return items, items[-1:]
    n_val = max(MIN_VAL, int(len(items) * VAL_RATIO))
    return items[:-n_val], items[-n_val:]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _merge_qa(existing: Path, new_rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    if existing.exists():
        for line in existing.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row.get("question", ""), row.get("answer", ""))
            if key not in seen:
                seen.add(key)
                out.append(row)
    for row in new_rows:
        key = (row["question"], row["answer"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def _dedup(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _is_copula(sentence: str) -> bool:
    return " là " in f" {sentence} "


def _language_corpora(
    subjects: dict[str, "object"],
) -> tuple[list[str], list[str], list[str]]:
    """Dẫn xuất corpora ngôn ngữ TỪ kiến thức thực (không dùng câu mẫu đồ chơi).

    - curated: hợp nhất toàn bộ câu kiến thức (ngữ pháp học từ fact thật).
    - copula:  câu định nghĩa "X là Y" (củng cố pattern S + là + N/A bằng fact).
    - compound: ghép hai fact copula cùng môn bằng " và " (câu ghép thật).
    """
    curated: list[str] = []
    copula: list[str] = []
    compound: list[str] = []
    for bundle in subjects.values():
        sents = list(bundle.sentences)
        curated.extend(sents)
        cop = [s for s in sents if _is_copula(s)]
        copula.extend(cop)
        for i in range(0, len(cop) - 1, 2):
            compound.append(f"{cop[i]} và {cop[i + 1]}")
    return _dedup(curated), _dedup(copula), _dedup(compound)


def _yaml_q(s: str) -> str:
    """Bọc chuỗi ví dụ thành scalar YAML an toàn (double-quoted, JSON-compatible)."""
    return json.dumps(s, ensure_ascii=False)


def _qa_lesson_examples(subjects: dict[str, "object"]) -> list[str]:
    """Ví dụ Hỏi/Đáp dẫn xuất TỪ qa_pairs của chính lớp (không dùng fact lớp khác)."""
    out: list[str] = []
    for bundle in subjects.values():
        for q, a in bundle.qa_pairs:
            out.append(f"Hỏi: {q} ? Đáp: {a}")
    return _dedup(out)


def _compound_lesson_examples(subjects: dict[str, "object"]) -> list[str]:
    """Câu ghép thật: định nghĩa + đại từ "nó" + vế cuối, chỉ ghép fact CÙNG chủ thể.

    Không bịa: vế giữa/cuối lấy từ câu kiến thức khác cùng nhắc tới chủ thể của định
    nghĩa. Lớp nào không đủ fact liên quan thì sinh ít/không có (pack sẽ bỏ qua).
    """
    out: list[str] = []
    for bundle in subjects.values():
        sents = [s.rstrip(".") for s in bundle.sentences]
        for q, a in bundle.qa_pairs:
            ans = a.rstrip(".")
            if " là " not in ans:
                continue
            subj = ans.split(" là ", 1)[0].strip()
            if not subj or " " not in subj and len(subj) < 2:
                continue
            related = [s for s in sents if subj in s and s != ans]
            if len(related) < 2:
                continue
            mid = related[0]
            if mid.startswith(subj):
                mid = "nó" + mid[len(subj):]
            last = related[1]
            out.append(f"Hỏi: {subj} là gì ? Đáp: {ans} , {mid} , {last}.")
    return _dedup(out)


def _write_lesson(
    path: Path,
    *,
    lesson_id: str,
    grade: int,
    pattern_id: str,
    slots_bind: str,
    examples: list[str],
    eta_teach: float = 0.25,
) -> None:
    header = (
        f"lesson_id: {lesson_id}\n"
        f"grade: {grade}\n"
        "repeat: 3\n"
        f"eta_teach: {eta_teach}\n"
        "w_init: 2.0\n\n"
        f"pattern_id: {pattern_id}\n"
        f"{slots_bind}\n"
        "# Tự sinh từ kiến thức CHÍNH lớp này (scripts/build_subject_knowledge.py).\n"
        "# Không dùng fact lớp khác → chống bleed chéo lớp. Đừng sửa tay.\n"
        "examples:\n"
    )
    body = "\n".join(f"  - {_yaml_q(e)}" for e in examples) + "\n"
    path.write_text(header + body, encoding="utf-8")


_QA_SLOTS_BIND = (
    "slots:\n"
    '  - const: "Hỏi:"\n'
    "  - var: Q\n"
    '  - const: "Đáp:"\n'
    "  - var: A\n\n"
    "bind:\n"
    "  Q: câu hỏi\n"
    "  A: câu trả lời\n"
)

_COMPOUND_SLOTS_BIND = (
    "slots:\n"
    '  - const: "Hỏi:"\n'
    "  - var: Q\n"
    '  - const: "Đáp:"\n'
    "  - var: A1\n"
    '  - const: ","\n'
    "  - var: P\n"
    "  - gap: s+\n"
    "    min: 0\n"
    '  - const: ","\n'
    "  - var: T\n\n"
    "bind:\n"
    "  Q: câu hỏi\n"
    "  A1: vế đầu\n"
    "  P: đại từ\n"
    "  T: vế cuối\n"
)


def _write_grade_lessons(grade: int, subjects: dict[str, "object"]) -> dict[str, str]:
    """Sinh lesson QA + compound theo lớp; trả map filename để cập nhật pack."""
    refs: dict[str, str] = {}
    qa_examples = _qa_lesson_examples(subjects)
    if qa_examples:
        fname = f"vi_QA_hoi_dap.grade{grade:02d}.lesson.yaml"
        _write_lesson(
            LESSONS_DIR / fname,
            lesson_id=f"vi_grade{grade}_qa",
            grade=grade,
            pattern_id="vi_QA_hoi_dap",
            slots_bind=_QA_SLOTS_BIND,
            examples=qa_examples,
        )
        refs["vi_QA_hoi_dap"] = fname

    comp_examples = _compound_lesson_examples(subjects)
    if len(comp_examples) >= 2:
        fname = f"vi_QA_compound.grade{grade:02d}.lesson.yaml"
        _write_lesson(
            LESSONS_DIR / fname,
            lesson_id=f"vi_grade{grade}_qa_compound",
            grade=grade,
            pattern_id="vi_QA_compound",
            slots_bind=_COMPOUND_SLOTS_BIND,
            examples=comp_examples,
            eta_teach=0.25,
        )
        refs["vi_QA_compound"] = fname
    return refs


def _update_pack(grade: int, refs: dict[str, str]) -> None:
    """Trỏ pack sang lesson QA theo lớp; bỏ compound nếu lớp không sinh được."""
    pack = LESSONS_DIR / f"grade_{grade:02d}_pack.yaml"
    if not pack.exists():
        return
    lines = pack.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    emitted: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped in (
            "- vi_QA_hoi_dap.lesson.yaml",
            f"- vi_QA_hoi_dap.grade{grade:02d}.lesson.yaml",
        ):
            if "vi_QA_hoi_dap" in refs:
                out.append(f"  - {refs['vi_QA_hoi_dap']}")
                emitted.add("vi_QA_hoi_dap")
            continue
        if stripped in (
            "- vi_QA_compound.lesson.yaml",
            f"- vi_QA_compound.grade{grade:02d}.lesson.yaml",
        ):
            if "vi_QA_compound" in refs:
                out.append(f"  - {refs['vi_QA_compound']}")
                emitted.add("vi_QA_compound")
            continue
        out.append(line)
    # Pack chưa từng tham chiếu lesson QA → thêm vào cuối (vd grade 5/6 thiếu).
    for key in ("vi_QA_hoi_dap", "vi_QA_compound"):
        if key in refs and key not in emitted:
            out.append(f"  - {refs[key]}")
    while out and not out[-1].strip():
        out.pop()
    pack.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_grade(grade: int, *, merge_qa: bool = False) -> dict[str, int]:
    grade_dir = ROOT / "data" / f"grade_{grade:02d}"
    subjects = grade_subjects(grade)
    stats: dict[str, int] = {}
    all_qa: list[dict] = []

    for subj_key, bundle in subjects.items():
        ds_id, domain, folder = SUBJECT_META[subj_key]
        sents = list(bundle.sentences)
        train_s, val_s = _split_train_val(sents)
        sub_dir = grade_dir / folder
        _write_jsonl(sub_dir / "train.jsonl", [{"text": t} for t in train_s])
        _write_jsonl(sub_dir / "val.jsonl", [{"text": t} for t in val_s])
        stats[f"{folder}/train"] = len(train_s)

        for q, a in bundle.qa_pairs:
            all_qa.append({"question": q, "answer": a})

    # Ngôn ngữ corpora (curated/copula/compound) — dẫn xuất từ kiến thức thực,
    # ghi đè câu mẫu đồ chơi cũ. Ngữ pháp học từ fact thật.
    curated, copula, compound = _language_corpora(subjects)
    for folder, items in (("curated", curated), ("copula", copula), ("compound", compound)):
        if not items:
            continue
        train_s, val_s = _split_train_val(items)
        sub_dir = grade_dir / folder
        _write_jsonl(sub_dir / "train.jsonl", [{"text": t} for t in train_s])
        _write_jsonl(sub_dir / "val.jsonl", [{"text": t} for t in val_s])
        stats[f"{folder}/train"] = len(train_s)

    qa_dir = grade_dir / "qa"
    if all_qa:
        train_qa, val_qa = _split_train_val(all_qa)
        if merge_qa and (qa_dir / "train.jsonl").exists():
            train_qa = _merge_qa(qa_dir / "train.jsonl", train_qa)
            val_qa = _merge_qa(qa_dir / "val.jsonl", val_qa) if (qa_dir / "val.jsonl").exists() else val_qa
        _write_jsonl(qa_dir / "train.jsonl", train_qa)
        _write_jsonl(qa_dir / "val.jsonl", val_qa)
        stats["qa/train"] = len(train_qa)

    # Lesson QA/compound theo lớp (chống bleed chéo lớp) + cập nhật pack.
    refs = _write_grade_lessons(grade, subjects)
    _update_pack(grade, refs)
    stats["lessons"] = len(refs)

    return stats


def repackage(grades: list[int]) -> None:
    for g in grades:
        subprocess.run(
            [sys.executable, "-m", "nn_aivie.data", "grade", "--grade", str(g)],
            cwd=ROOT,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build subject knowledge JSONL for grades 3–7")
    parser.add_argument(
        "--grade",
        type=int,
        nargs="+",
        default=list(all_grades()),
        help="Grade(s) to build (default: 3 4 5 6 7)",
    )
    parser.add_argument("--no-repackage", action="store_true")
    parser.add_argument(
        "--merge-qa",
        action="store_true",
        help="Gộp QA môn học vào qa/train.jsonl hiện có (mặc định: ghi đè, bỏ câu mẫu cũ)",
    )
    args = parser.parse_args()

    for g in sorted(set(args.grade)):
        stats = build_grade(g, merge_qa=args.merge_qa)
        total = sum(stats.values())
        print(f"grade_{g:02d}: {total} rows ({stats})")

    if not args.no_repackage:
        repackage(sorted(set(args.grade)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
