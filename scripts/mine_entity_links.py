#!/usr/bin/env python3
"""
TK_024 nhánh C / B1 — MINE ENTITY_LINK candidates từ đồng-hiện mức ĐOẠN (measure-first).

Không wire runtime. Chỉ ĐO nguyên liệu: nếu build lớp cạnh entity↔entity (đồng-đoạn, PMI+floor)
thì được bao nhiêu cạnh "tốt", tỉ lệ drift (đồng-tự / off-topic) bao nhiêu, và với vài entity
truy vấn (Lý, Trần, phở…) associate nổi lên có ĐÚNG loại tác giả muốn không.

Nguồn: data/real_vi/statements_doc.jsonl (đã có doc_id+para) → gộp thành đoạn.
Chạy: python scripts/mine_entity_links.py [--min-co 2] [--min-df 3] [--top 40]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "data/real_vi/statements_doc.jsonl"
OUT = ROOT / "out/tk024_entity_links.log"

# stop-cap: từ hay viết hoa đầu câu nhưng KHÔNG phải tên riêng (tiếng Việt)
STOP_CAP = {
    "Tôi", "Bạn", "Nó", "Họ", "Chúng", "Các", "Những", "Một", "Này", "Đó", "Khi", "Nếu",
    "Tuy", "Vì", "Do", "Và", "Nhưng", "Ông", "Bà", "Anh", "Chị", "Em", "Người", "Sau",
    "Trước", "Trong", "Ngoài", "Theo", "Với", "Từ", "Đây", "Vào", "Ở", "Có", "Là", "Được",
    "Cách", "Cả", "Nhiều", "Mọi", "Mỗi", "Hai", "Ba", "Bốn", "Năm", "Sáu", "Bảy", "Tám",
}
_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def _entity_spans(tokens: list[str]) -> list[str]:
    """Maximal run of capitalized tokens (bỏ stop-cap ở đầu run)."""
    spans: list[str] = []
    run: list[str] = []
    for tok in tokens:
        cap = bool(tok) and tok[0].isupper() and not tok.isupper()  # Title-ish, bỏ ALLCAPS acronym-noise
        if cap and not (not run and tok in STOP_CAP):
            run.append(tok)
        else:
            if len(run) >= 1:
                spans.append(" ".join(run))
            run = []
    if run:
        spans.append(" ".join(run))
    # chỉ giữ span có ít nhất 1 token >3 ký tự (bỏ "Kỳ", "Bộ" lẻ nhiễu) HOẶC đa token
    return [s for s in spans if len(s.split()) >= 2 or any(len(w) > 3 for w in s.split())]


def _load_paragraphs(docs: list[Path]) -> list[list[str]]:
    """Gộp câu cùng (doc_id, para) → danh sách entity-span mỗi đoạn (gộp nhiều file)."""
    by_para: dict[tuple[str, int], list[str]] = defaultdict(list)
    for doc in docs:
        if not doc.exists():
            continue
        with doc.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                key = (f"{doc.name}:{o.get('doc_id', '')}", int(o.get("para", 0)))
                by_para[key].extend(_entity_spans(_tokens(o.get("text", "") or "")))
    return [ents for ents in by_para.values() if ents]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-co", type=int, default=2, help="sàn số đoạn đồng-hiện của một cặp")
    ap.add_argument("--min-df", type=int, default=3, help="sàn số đoạn chứa mỗi entity")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--query", action="append", default=["Lý", "Trần", "Văn Miếu", "Thăng Long", "Lý Thường Kiệt"],
                    help="entity truy vấn để đọc associate")
    ap.add_argument("--doc", action="append", default=None,
                    help="jsonl path (repeatable); mặc định statements_doc.jsonl")
    args = ap.parse_args()

    docs = [ROOT / d for d in args.doc] if args.doc else [DOC]
    paras = _load_paragraphs(docs)
    n_para = len(paras)

    ent_df: Counter[str] = Counter()          # #đoạn chứa entity
    pair_co: Counter[tuple[str, str]] = Counter()  # #đoạn chứa cả 2
    for ents in paras:
        uniq = sorted(set(ents))
        for e in uniq:
            ent_df[e] += 1
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair_co[(uniq[i], uniq[j])] += 1

    total_pairs = sum(pair_co.values()) or 1

    # PMI: log( p(a,b) / (p(a) p(b)) ), dùng df làm proxy count
    def pmi(a: str, b: str, co: int) -> float:
        pa = ent_df[a] / n_para
        pb = ent_df[b] / n_para
        pab = co / n_para
        return math.log(pab / (pa * pb) + 1e-12)

    cands: list[tuple[float, int, str, str]] = []
    for (a, b), co in pair_co.items():
        if co < args.min_co:
            continue
        if ent_df[a] < args.min_df or ent_df[b] < args.min_df:
            continue
        cands.append((pmi(a, b, co), co, a, b))
    cands.sort(reverse=True)

    lines: list[str] = []

    def w(m: str = "") -> None:
        print(m)
        lines.append(m)

    w(f"paragraphs={n_para}  distinct_entities={len(ent_df)}  "
      f"pair_types={len(pair_co)}  candidate_links(min_co={args.min_co},min_df={args.min_df})={len(cands)}")
    w("")
    w(f"=== TOP {args.top} ENTITY_LINK theo PMI (co≥{args.min_co}, df≥{args.min_df}) ===")
    for score, co, a, b in cands[: args.top]:
        w(f"  {score:5.2f}  co={co:<3d} df=({ent_df[a]},{ent_df[b]})  {a!r} ↔ {b!r}")

    w("")
    w("=== ASSOCIATE theo entity truy vấn (đọc tay: đúng loại 'liên-quan không chung token'?) ===")
    for q in args.query:
        neigh: list[tuple[float, int, str]] = []
        for (a, b), co in pair_co.items():
            other = None
            if a == q:
                other = b
            elif b == q:
                other = a
            if other is None or co < args.min_co:
                continue
            if ent_df[other] < args.min_df:
                continue
            neigh.append((pmi(q, other, co), co, other))
        neigh.sort(reverse=True)
        top = ", ".join(f"{o}(co{co})" for _, co, o in neigh[:12]) or "(none)"
        w(f"  {q!r} (df={ent_df.get(q,0)}) → {top}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", "utf-8")
    print(f"\n✓ wrote {OUT}")


if __name__ == "__main__":
    main()
