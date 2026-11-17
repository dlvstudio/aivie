"""
Biên soạn text_corpus — mỗi dòng jsonl = một JSON string (một đơn vị train).

- Từ điển: một dòng / lemma (mọi nghĩa + ví dụ gộp)
- Wiki / sử / doc: một dòng / bài (doc_id + title)
- Lớp 3–7: một dòng / bài có nhãn Môn, Lớp, Tiết, Bài
- QA: file riêng qa_supervised.jsonl
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

from nn_aivie.data.qa import render_qa
from nn_aivie.data.subject_knowledge import SUBJECT_META, all_grades, grade_subjects
from nn_aivie.training.text_corpus import write_texts

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "text_corpus"
LESSONS_DIR = REPO_ROOT / "lessons"

SUBJECT_LABEL: dict[str, str] = {
    "toan": "Toán",
    "dia_ly": "Địa lý",
    "lich_su": "Lịch sử",
    "khoa_hoc": "Khoa học",
    "xa_hoi": "Đạo đức và Xã hội",
    "ngon_ngu": "Tiếng Việt",
}

SENTENCES_PER_BAI = 8
BAI_PER_TIET = 3
MIN_QA_SENTENCES = 5

# curated = hợp nhất fact mọi môn → KHÔNG dùng cho Tiếng Việt
LANG_DATA_DIRS = ("general", "compound", "copula", "dialogue")

LEGACY_CORPUS_FILES = (
    "dictionary_examples.jsonl",
    "conv_sentences.jsonl",
    "essays.jsonl",
    "histwiki_para.jsonl",
    "history_vi.jsonl",
    "doc_add.jsonl",
    "hf_anaphora.jsonl",
    "hf_statements.jsonl",
    "grade_3_7.jsonl",
    "histwiki_articles.jsonl",
    "history_vi_articles.jsonl",
    "_smoke.jsonl",
)

PATTERN_TITLES: dict[str, str] = {
    "vi_copula_S_la_NA": "Câu định nghĩa: Ai/Cái gì là gì",
    "vi_S_V": "Câu đơn: Chủ ngữ + Động từ",
    "vi_S_khong_V": "Câu phủ định: không",
    "vi_S_dang_V": "Câu tiếp diễn: đang",
    "vi_S_rat_A": "Câu mô tả: rất + tính từ",
    "vi_N_gap_A": "Câu mô tả danh từ",
    "vi_S_o_N": "Giới từ: ở",
    "vi_S_co_N": "Câu sở hữu: có",
    "vi_S_se_V": "Tương lai: sẽ",
    "vi_S_da_V": "Quá khứ: đã",
    "vi_S_V_A": "Câu bổ ngữ",
    "vi_S_V_va_S_V": "Câu ghép: và",
    "vi_S_V_nhung_S_khong_V": "Câu ghép: nhưng ... không",
    "vi_S_V_roi_S_V": "Câu ghép: rồi",
    "vi_S_V_nen_S_V": "Câu ghép: nên",
    "vi_S_V_hoac_S_V": "Câu ghép: hoặc",
    "vi_copula_va_copula": "Câu ghép định nghĩa kép",
    "vi_QA_hoi_dap": "Hỏi và đáp",
    "vi_chat_hoi_thoai": "Hội thoại",
    "vi_S_A_hon_S": "So sánh hơn",
    "vi_S_A_nhat": "So sánh nhất",
    "vi_S_bi_V": "Bị động",
    "vi_S_duoc_V": "Được",
    "vi_khi_S_V_S_V": "Mệnh đề khi",
    "vi_S_V_de_V": "Mục đích: để",
    "vi_neu_S_V_thi_S_V": "Nếu ... thì",
    "vi_mac_du_S_V_van_V": "Mặc dù ... vẫn",
    "vi_S_vua_V_vua_V": "Vừa ... vừa",
    "vi_S_V_ma_S_V": "Mệnh đề quan hệ: mà",
    "vi_vi_S_V_nen_S_V": "Vì ... nên",
    "vi_QA_compound": "Hỏi đáp câu ghép",
}

_MATH_SENT = re.compile(
    r"^\d+\s*(cộng|trừ|nhân|chia)\b|^\d+.*\bbằng\s+\d|"
    r"phương trình|delta|nghiệm|sin\s|cos\s|tan\s|parabol|đa thức",
    re.IGNORECASE,
)
_PRECISE_Q = re.compile(
    r"bằng mấy|bằng gì|tính thế nào|delta|sin |cos |tan |có mấy |bao nhiêu",
    re.IGNORECASE,
)


def _fresh_out(name: str) -> Path:
    p = OUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


def _delete_legacy_corpus() -> int:
    n = 0
    for name in LEGACY_CORPUS_FILES:
        p = OUT_DIR / name
        if p.exists():
            p.unlink()
            n += 1
    return n


def _sentence_end(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s[-1] not in ".!?":
        return s + "."
    return s


def _read_jsonl_texts(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        t = str(row.get("text") or "").strip()
        if t:
            out.append(t)
    return out


def _group_jsonl_by_article(path: Path, *, id_key: str = "doc_id") -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(row.get(id_key) or row.get("title") or "unknown")
        title = str(row.get("title") or doc_id)
        key = f"{title}::{doc_id}"
        groups[key].append(text)
    return groups


def _chunk(lst: list[str], size: int) -> list[list[str]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _is_math_or_fact_noise(s: str) -> bool:
    return bool(_MATH_SENT.search(s.strip()))


def _quality_lang_sentence(s: str, grade: int) -> bool:
    s = s.strip()
    if not s or _is_math_or_fact_noise(s):
        return False
    words = s.split()
    max_words = {3: 10, 4: 12, 5: 14, 6: 16, 7: 18}.get(grade, 14)
    if len(words) < 2 or len(words) > max_words:
        return False
    if re.search(r"http|www\.|\.com\b|%|\$|@|#", s, re.IGNORECASE):
        return False
    bad = (
        r"Trump|Swift|Brexit|WTI|Brent|ma túy|HIV|Google|UBND|triệu|tỷ|"
        r"Bão |dầu |kinh tế|chính phủ|nghị viện|quốc hội|đình công|"
        r"Taylor|Hemingway|phiên bản|ảnh:|clip |bão số"
    )
    if re.search(bad, s, re.IGNORECASE):
        return False
    if grade <= 5 and re.search(r"\d{4}|\d+[,\.]\d+", s):
        return False
    return True


def _grade_subject_sentences(grade: int, subject_key: str) -> list[str]:
    """Chỉ parquet môn học — không trùng curated."""
    seen: set[str] = set()
    out: list[str] = []
    meta = SUBJECT_META.get(subject_key)
    if not meta:
        return out
    _, _, data_dir = meta
    jp = REPO_ROOT / f"data/grade_{grade:02d}" / data_dir / "train.jsonl"
    for t in _read_jsonl_texts(jp):
        if t not in seen:
            seen.add(t)
            out.append(t)
    bundle = grade_subjects(grade).get(subject_key)
    if bundle:
        for s in bundle.sentences:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _lang_dir_sentences(grade: int, data_dir: str) -> list[str]:
    jp = REPO_ROOT / f"data/grade_{grade:02d}" / data_dir / "train.jsonl"
    if data_dir == "dialogue":
        out: list[str] = []
        if not jp.exists():
            return out
        for line in jp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            turns = row.get("turns") or []
            parts: list[str] = []
            for turn in turns:
                if isinstance(turn, dict):
                    u = str(turn.get("user") or "").strip()
                    a = str(turn.get("assistant") or "").strip()
                    if u:
                        parts.append(f"{u}?")
                    if a:
                        parts.append(a)
                else:
                    parts.append(str(turn).strip())
            if parts:
                out.append(" ".join(_sentence_end(p) for p in parts))
        return out
    texts = _read_jsonl_texts(jp)
    if data_dir == "general":
        return [t for t in texts if _quality_lang_sentence(t, grade)]
    return [t for t in texts if not _is_math_or_fact_noise(t)]


def _lesson_pack_paths(grade: int) -> list[Path]:
    pack = LESSONS_DIR / f"grade_{grade:02d}_pack.yaml"
    if not pack.exists():
        return []
    data = yaml.safe_load(pack.read_text(encoding="utf-8")) or {}
    paths: list[Path] = []
    for entry in data.get("lessons") or []:
        name = str(entry).strip().lstrip("- ")
        p = LESSONS_DIR / name
        if p.exists():
            paths.append(p)
    return paths


def _load_lesson(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _pattern_title(pattern_id: str) -> str:
    return PATTERN_TITLES.get(pattern_id, pattern_id.replace("_", " "))


def _format_lesson(
    label: str,
    grade: int,
    ti: int,
    ti_title: str,
    bi: int,
    body_lines: list[str],
    *,
    intro: str = "",
) -> str:
    parts = [f"{label}:", f"Lớp {grade}:", f"Tiết {ti}: {ti_title}", f"Bài {bi}:"]
    if intro:
        parts.append(intro)
    parts.extend(body_lines)
    return "\n".join(parts)


def _subject_tiet_intro(subject_key: str, grade: int, ti: int) -> str:
    label = SUBJECT_LABEL.get(subject_key, subject_key)
    intros = {
        "toan": "Tiết Toán giúp em củng cố tư duy số học, hình học và đo lường.",
        "dia_ly": "Tiết Địa lý giúp em hiểu đất nước, địa hình và con người Việt Nam.",
        "lich_su": "Tiết Lịch sử giúp em biết nguồn cội, truyền thống và các sự kiện quan trọng.",
        "khoa_hoc": "Tiết Khoa học giúp em khám phá tự nhiên và cơ thể con người.",
        "xa_hoi": "Tiết Đạo đức và Xã hội giúp em sống có trách nhiệm với gia đình và cộng đồng.",
    }
    base = intros.get(subject_key, f"Tiết {label} tổng hợp kiến thức cốt lõi.")
    return f"{base} Đây là phần {ti} trong chương trình lớp {grade}."


def normalize_grade_lessons(out_name: str = "grade_lessons.jsonl") -> int:
    """Một dòng / bài — nhãn Môn, Lớp, Tiết, Bài."""
    path = _fresh_out(out_name)
    texts: list[str] = []

    for grade in all_grades():
        # --- Môn học (toán, sử, địa lý, ...) ---
        for subj in SUBJECT_META:
            label = SUBJECT_LABEL[subj]
            sentences = _grade_subject_sentences(grade, subj)
            if not sentences:
                continue
            bais = _chunk(sentences, SENTENCES_PER_BAI)
            tiets = _chunk(bais, BAI_PER_TIET)
            for ti, bai_group in enumerate(tiets, 1):
                ti_title = bai_group[0][0] if bai_group and bai_group[0] else label
                if len(ti_title) > 72:
                    ti_title = ti_title[:69] + "..."
                intro = _subject_tiet_intro(subj, grade, ti)
                for bi, sents in enumerate(bai_group, 1):
                    body = [_sentence_end(s) for s in sents]
                    if bi == 1:
                        body = [intro, *body]
                    if bi == len(bai_group):
                        body.append(
                            f"Em hãy đọc kỹ và ghi nhớ các ý chính của bài {bi} môn {label}."
                        )
                    texts.append(_format_lesson(label, grade, ti, ti_title, bi, body))

        # --- Tiếng Việt: lesson pack + general/compound/copula/dialogue ---
        lang_label = SUBJECT_LABEL["ngon_ngu"]
        ti_idx = 0

        for lesson_path in _lesson_pack_paths(grade):
            lesson = _load_lesson(lesson_path)
            pattern = str(lesson.get("pattern_id") or lesson_path.stem)
            if pattern.startswith("vi_QA"):
                continue  # QA → qa_supervised
            examples = [str(e).strip() for e in (lesson.get("examples") or []) if str(e).strip()]
            if not examples:
                continue
            ti_idx += 1
            ti_title = _pattern_title(pattern)
            ex_bais = _chunk(examples, 5)
            for bi, ex_group in enumerate(ex_bais, 1):
                body = [f"Ví dụ mẫu {j}: {_sentence_end(e)}" for j, e in enumerate(ex_group, 1)]
                if bi == 1:
                    body.insert(
                        0,
                        f"Tiết này luyện mẫu câu «{ti_title}». Em đọc từng ví dụ và nhận ra cấu trúc.",
                    )
                texts.append(_format_lesson(lang_label, grade, ti_idx, ti_title, bi, body))

        for data_dir, ti_name in (
            ("general", "Câu tiếng Việt trong văn cảnh"),
        ):
            sents = _lang_dir_sentences(grade, data_dir)
            if not sents:
                continue
            ti_idx += 1
            bais = _chunk(sents, SENTENCES_PER_BAI)
            for bi, group in enumerate(bais, 1):
                body = [_sentence_end(s) for s in group]
                if bi == 1:
                    body.insert(0, f"Tiết «{ti_name}»: em đọc các câu mẫu và cảm nhận cách dùng từ.")
                texts.append(_format_lesson(lang_label, grade, ti_idx, ti_name, bi, body))

        dialogues = _lang_dir_sentences(grade, "dialogue")
        if dialogues:
            ti_idx += 1
            for bi, d in enumerate(dialogues, 1):
                body = [
                    "Tiết hội thoại: em luyện đặt câu hỏi và trả lời ngắn gọn.",
                    _sentence_end(d),
                ]
                texts.append(
                    _format_lesson(lang_label, grade, ti_idx, "Hội thoại hằng ngày", bi, body)
                )

    return write_texts(path, texts)


def normalize_dictionary(out_name: str = "dictionary.jsonl") -> int:
    raw = REPO_ROOT / "data" / "dict" / "vi_dict_raw.jsonl"
    path = _fresh_out(out_name)
    if not raw.exists():
        return 0
    by_word: dict[str, list[dict]] = defaultdict(list)
    for line in raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        w = str(row.get("word") or "").strip()
        if w:
            by_word[w].append(row)
    texts: list[str] = []
    for word, senses in sorted(by_word.items()):
        parts = [f"Từ: {word}"]
        for i, s in enumerate(senses, 1):
            pos = str(s.get("part_of_speech") or "").strip()
            meaning = str(s.get("meaning") or "").strip()
            example = str(s.get("example") or "").strip()
            if len(senses) > 1:
                parts.append(f"Nghĩa {i}:")
            if pos:
                parts.append(f"Loại từ: {pos}")
            if meaning:
                parts.append(f"Giải nghĩa: {meaning}")
            if example:
                parts.append(f"Ví dụ: {example}")
        texts.append("\n".join(parts))
    return write_texts(path, texts)


def normalize_articles(
    src: Path,
    out_name: str,
    *,
    header_prefix: str = "",
) -> int:
    path = _fresh_out(out_name)
    groups = _group_jsonl_by_article(src)
    texts: list[str] = []
    for key, sents in groups.items():
        title = key.split("::", 1)[0]
        body = "\n".join(_sentence_end(s) for s in sents)
        if header_prefix:
            texts.append(f"{header_prefix}\nTiêu đề: {title}\n\n{body}")
        else:
            texts.append(f"Tiêu đề: {title}\n\n{body}")
    return write_texts(path, texts)


def normalize_merged_history(out_name: str = "history_articles.jsonl") -> int:
    """Gộp history_vi + history_wiki → một file, nhiều bài sử hơn."""
    rv = REPO_ROOT / "data" / "real_vi"
    path = _fresh_out(out_name)
    merged: dict[str, list[str]] = {}
    for src in (rv / "history_vi.jsonl", rv / "history_wiki_para.jsonl"):
        for key, sents in _group_jsonl_by_article(src).items():
            title = key.split("::", 1)[0]
            if title not in merged:
                merged[title] = []
            for s in sents:
                if s not in merged[title]:
                    merged[title].append(s)
    texts: list[str] = []
    for title in sorted(merged):
        body = "\n".join(_sentence_end(s) for s in merged[title])
        texts.append(f"Lịch sử:\nTiêu đề: {title}\n\n{body}")
    return write_texts(path, texts)


def normalize_subject_articles(out_name: str = "subject_articles.jsonl") -> int:
    """Bài tổng hợp theo môn/lớp — format giống history_vi."""
    path = _fresh_out(out_name)
    texts: list[str] = []
    for grade in all_grades():
        for subj in SUBJECT_META:
            label = SUBJECT_LABEL[subj]
            sents = _grade_subject_sentences(grade, subj)
            if len(sents) < 4:
                continue
            paras = _chunk(sents, 4)
            body_parts: list[str] = []
            for pi, para in enumerate(paras, 1):
                block = " ".join(_sentence_end(s) for s in para)
                body_parts.append(block)
            body = "\n\n".join(body_parts)
            texts.append(
                f"{label}:\nLớp {grade}:\nTiêu đề: {label} lớp {grade} — bài tổng hợp\n\n{body}"
            )
    return write_texts(path, texts)


def _related_sentence(cand: str, question: str, answer: str) -> bool:
    qtok = set(re.findall(r"\w+", question.lower(), flags=re.UNICODE))
    atok = set(re.findall(r"\w+", answer.lower(), flags=re.UNICODE))
    ctok = set(re.findall(r"\w+", cand.lower(), flags=re.UNICODE))
    key = qtok | atok
    key -= {"là", "gì", "mấy", "nào", "có", "không", "bao", "nhiêu", "thế", "ai", "đâu", "em", "hãy"}
    if len(key) < 2:
        return False
    return len(ctok & key) >= 2


def _is_precise_qa(question: str, answer: str) -> bool:
    if _PRECISE_Q.search(question):
        return True
    al = answer.strip()
    if _is_math_or_fact_noise(al):
        return True
    if len(al.split()) <= 12 and re.search(r"\d", al):
        return True
    return False


def _expand_answer(question: str, answer: str, pool: list[str], min_sents: int = MIN_QA_SENTENCES) -> str:
    if _is_precise_qa(question, answer):
        return _sentence_end(answer)
    parts: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = _sentence_end(s.strip())
        if s and s not in seen:
            seen.add(s)
            parts.append(s)

    add(answer)
    for s in pool:
        if len(parts) >= min_sents:
            break
        if _related_sentence(s, question, answer):
            add(s)
    return " ".join(parts)


def _collect_qa_pairs() -> list[tuple[str, str, list[str], str]]:
    """(question, short_answer, context_pool, subject_key)"""
    out: list[tuple[str, str, list[str], str]] = []
    seen: set[tuple[str, str]] = set()

    def add(q: str, a: str, pool: list[str], subj: str = "") -> None:
        q, a = q.strip(), a.strip()
        if not q or not a:
            return
        key = (q.casefold(), a.casefold())
        if key in seen:
            return
        seen.add(key)
        out.append((q, a, pool, subj))

    for grade in all_grades():
        for subj, bundle in grade_subjects(grade).items():
            label = SUBJECT_LABEL.get(subj, subj)
            pool = list(bundle.sentences)
            for q, a in bundle.qa_pairs:
                add(q, a, pool, subj)
            if subj == "toan":
                continue
            answered = {a.casefold() for _, a in bundle.qa_pairs}
            for s in bundle.sentences:
                if s.casefold() in answered or _is_math_or_fact_noise(s):
                    continue
                hook = s if len(s) <= 72 else s[:69] + "..."
                q = f"Môn {label} lớp {grade}: em hãy giải thích thêm về {hook}"
                add(q, s, pool, subj)
        qdir = REPO_ROOT / f"data/grade_{grade:02d}" / "qa"
        for fname in ("train.jsonl", "val.jsonl"):
            fp = qdir / fname
            if not fp.exists():
                continue
            for line in fp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("question") and row.get("answer"):
                    subj_pool = _grade_subject_sentences(grade, "toan")
                    subj = "toan" if _is_precise_qa(row["question"], row["answer"]) else ""
                    pool = subj_pool if subj == "toan" else []
                    if not pool:
                        for sk in SUBJECT_META:
                            pool.extend(_grade_subject_sentences(grade, sk))
                    add(row["question"], row["answer"], pool, subj)

    essays = REPO_ROOT / "data/real_vi/essays.jsonl"
    if essays.exists():
        for line in essays.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            inst = str(row.get("instruction") or "").strip()
            sents = row.get("sentences") or []
            if inst and sents:
                ans = " ".join(_sentence_end(str(s)) for s in sents)
                pool = [str(s) for s in sents]
                add(inst, ans, pool, "")

    return out


def normalize_qa_supervised(out_name: str = "qa_supervised.jsonl") -> int:
    path = _fresh_out(out_name)
    texts: list[str] = []
    for q, a, pool, _subj in _collect_qa_pairs():
        final_a = _expand_answer(q, a, pool, MIN_QA_SENTENCES)
        prompt, _ = render_qa(q, final_a)
        texts.append(prompt)
    return write_texts(path, texts)


def normalize_conv_stories(out_name: str = "conv_stories.jsonl") -> int:
    path = _fresh_out(out_name)
    texts: list[str] = []
    stories = REPO_ROOT / "data/conv/stories_raw.jsonl"
    if stories.exists():
        for line in stories.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if text:
                texts.append(f"Truyện:\n{text}")
    chat = REPO_ROOT / "data/conv/chat_raw.jsonl"
    if chat.exists():
        for line in chat.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            turns = [str(t).strip() for t in (row.get("turns") or []) if str(t).strip()]
            if turns:
                body = "\n".join(turns)
                texts.append(f"Hội thoại:\n{body}")
    return write_texts(path, texts)


def normalize_paragraph_units(
    doc_paths: list[Path],
    conv_paths: list[Path],
    *,
    out_name: str = "paragraph_units.jsonl",
    min_sent: int = 2,
    max_sent: int = 4,
    max_units: int = 25_000,
) -> int:
    from nn_aivie.data.paragraph_units import (
        iter_units_from_conv_jsonl,
        iter_units_from_doc_jsonl,
    )

    path = _fresh_out(out_name)
    seen: set[str] = set()
    texts: list[str] = []
    for p in doc_paths:
        if not p.exists():
            continue
        for unit in iter_units_from_doc_jsonl(p, min_sent=min_sent, max_sent=max_sent, dedup=False):
            key = unit.casefold()
            if key in seen:
                continue
            seen.add(key)
            texts.append(unit)
            if len(texts) >= max_units:
                return write_texts(path, texts)
    for p in conv_paths:
        if not p.exists():
            continue
        for unit in iter_units_from_conv_jsonl(p, min_sent=min_sent, max_sent=max_sent, dedup=False):
            key = unit.casefold()
            if key in seen:
                continue
            seen.add(key)
            texts.append(unit)
            if len(texts) >= max_units:
                return write_texts(path, texts)
    return write_texts(path, texts)


def normalize_all() -> dict[str, int]:
    stats: dict[str, int] = {}
    stats["legacy_deleted"] = _delete_legacy_corpus()
    rv = REPO_ROOT / "data" / "real_vi"
    stats["dictionary"] = normalize_dictionary()
    stats["history_articles"] = normalize_merged_history()
    stats["subject_articles"] = normalize_subject_articles()
    stats["doc_add_articles"] = normalize_articles(
        rv / "statements_doc_add.jsonl", "doc_add_articles.jsonl",
    )
    stats["grade_lessons"] = normalize_grade_lessons()
    stats["qa_supervised"] = normalize_qa_supervised()
    stats["conv_stories"] = normalize_conv_stories()
    stats["paragraph_units"] = normalize_paragraph_units(
        [rv / "statements_doc.jsonl", rv / "statements_doc_add.jsonl",
         rv / "history_wiki_para.jsonl", rv / "history_vi.jsonl"],
        [REPO_ROOT / "data/conv/stories_raw.jsonl", REPO_ROOT / "data/conv/chat_raw.jsonl"],
    )
    return stats
