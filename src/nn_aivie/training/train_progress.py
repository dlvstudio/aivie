"""Log tiến trình train theo từng đơn vị corpus (dataset / bài / nơ-ron cha)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.core.vocab import Vocabulary


def count_corpus_lines(path: Path | str) -> int:
    n = 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def text_label(text: str, *, max_len: int = 72) -> str:
    """Nhãn ngắn cho một dòng corpus — ưu tiên Môn/Lớp/Bài hoặc Hỏi:."""
    raw = text.strip()
    if not raw:
        return "(rỗng)"
    first = raw.split("\n", 1)[0].strip()
    for pat in (
        r"^(Hỏi:\s*.+?)(?:\s*Đáp:|$)",
        r"^((?:Toán|Lịch sử|Địa lý|Khoa học|Đạo đức[^:\n]*|Tiếng Việt):.+)$",
        r"^(Tiêu đề:\s*.+)$",
        r"^(Bài\s*\d+:.+)$",
        r"^(Tiết\s*\d+:.+)$",
    ):
        m = re.match(pat, first, re.I)
        if m:
            first = m.group(1).strip()
            break
    first = re.sub(r"\s+", " ", first)
    if len(first) > max_len:
        return first[: max_len - 1] + "…"
    return first


def parent_item_stats(
    ps: ParentStore,
    pi: int,
    vocab: Vocabulary,
) -> dict[str, Any]:
    """Thống kê một nơ-ron cha vừa thêm (hoặc trùng)."""
    rec = ps.parents[pi]
    n_token_slots = sum(1 for s in rec.slot_chain if s.kind == "token")
    n_cluster_slots = sum(1 for s in rec.slot_chain if s.kind == "cluster")
    clusters = ps.seg.clusters_in_match_span(rec) if rec.slot_chain else ()
    return {
        "parent": pi,
        "power": rec.power,
        "qa": rec.is_qa,
        "tokens": len(rec.source_ids),
        "match": len(rec.match_ids),
        "slots": len(rec.slot_chain),
        "token_slots": n_token_slots,
        "cluster_slots": n_cluster_slots,
        "clusters_in_match": len(clusters),
        "phrases": len(rec.phrase_ids),
        "skip_on_parent": len(getattr(rec, "skip_span_ids", ()) or ()),
    }


def global_graph_stats(ps: ParentStore, vocab: Vocabulary) -> dict[str, int]:
    return {
        "parents": len(ps.parents),
        "clusters": len(ps.seg.clusters),
        "vocab": len(vocab),
        "skip_spans": len(getattr(ps, "skip_spans", None) or []),
    }


def format_item_log(
    dataset: str,
    index: int,
    total: int,
    label: str,
    item: dict[str, Any],
    global_stats: dict[str, int],
    *,
    duplicate: bool = False,
) -> str:
    dup = " [trùng, power+1]" if duplicate else ""
    return (
        f"[{dataset} {index}/{total}] cha#{item['parent']}{dup} "
        f"tok={item['tokens']} slot={item['slots']} "
        f"(cụm_slot={item['cluster_slots']} cụm_match={item['clusters_in_match']} "
        f"skip={item.get('skip_on_parent', 0)}) "
        f"| graph: cha={global_stats['parents']} cụm={global_stats['clusters']} "
        f"skip={global_stats['skip_spans']} vocab={global_stats['vocab']} — {label}"
    )


def append_progress_jsonl(path: Path | str, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
