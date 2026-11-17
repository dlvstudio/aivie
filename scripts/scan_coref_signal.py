#!/usr/bin/env python3
"""Đo tín hiệu binding trong CORPUS THẬT: lặp-tên vs đại-từ/danh-xưng đồng-chỉ (P3.1 pivot).

Heuristic PHÂN TÍCH (không phải runtime — Luật 0 nói về inference graph, không phải script đo):
  - thực-thể ứng viên = cụm ≥2 từ Hoa liền nhau (Nguyễn Du, Biển Đông) — tín hiệu mạnh, ít nhiễu.
  - lặp-tên = cụm đó xuất hiện ≥2 lần trong 1 record.
  - đại-từ/danh-xưng đồng-chỉ = record chứa marker trong tập (đo loại nào trội)."""
import json, re, sys
from collections import Counter
from pathlib import Path

# Resolve corpus path relative to repository root so script is portable across machines.
ROOT = (Path(__file__).resolve().parents[1] / "data" / "text_corpus").resolve()
COREF_MARKERS = ["ông ấy", "bà ấy", "anh ấy", "chị ấy", "cô ấy", "ông ta", "bà ta",
                 "vị này", "người này", "nơi này", "thành phố này", "đất nước này", "tổ chức này",
                 "ông", "bà", "nó", "họ", "hắn", "ngài", "đây", "này", "đó"]
# multi-word markers ưu tiên khớp trước; "này"/"đó"/"ông"… đơn lẻ đếm riêng (nhiễu hơn)
MULTI = [m for m in COREF_MARKERS if " " in m]
SINGLE = {"ông", "bà", "nó", "họ", "hắn", "ngài", "đây"}   # "này"/"đó" quá nhiễu → đo riêng

CAP_SPAN = re.compile(r"(?:[A-ZÀ-Ỹ][^\s]*\s+){1,}[A-ZÀ-Ỹ][^\s]*")  # ≥2 từ Hoa liền


def entities(text):
    # bỏ token đầu MỖI câu (Hoa do đầu câu) để giảm nhiễu
    out = []
    for sent in re.split(r"[.\n?!]", text):
        toks = sent.split()
        if len(toks) < 2:
            continue
        body = " ".join(toks[1:])  # bỏ từ đầu câu
        out += [m.group(0).strip() for m in CAP_SPAN.finditer(body)]
    return [e for e in out if len(e) >= 4]


def scan(path, limit):
    n = rep = demo_this = pron_multi = pron_single = 0
    marker_hits = Counter()
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec if isinstance(rec, str) else str(rec.get("text", rec))
            n += 1
            low = " " + text.lower() + " "
            ents = entities(text)
            c = Counter(ents)
            if any(v >= 2 for v in c.values()):
                rep += 1
            hit_multi = False
            for m in MULTI:
                if " " + m + " " in low:
                    marker_hits[m] += 1; hit_multi = True
            if hit_multi:
                pron_multi += 1
            if re.search(r"\b(này|đó)\b", low):
                demo_this += 1
            for m in SINGLE:
                if " " + m + " " in low:
                    marker_hits[m] += 1
            if any(" " + m + " " in low for m in SINGLE):
                pron_single += 1
    return n, rep, demo_this, pron_multi, pron_single, marker_hits


def pct(a, b):
    return f"{100*a/b:.1f}%" if b else "—"


print("═══ tín hiệu binding trong corpus (mẫu) — lặp-tên vs đồng-chỉ ═══")
tot = Counter()
for name, lim in [("history_articles", 20000), ("conv_stories", 8000),
                  ("paragraph_units", 20000), ("viwiki", 20000), ("subject_articles", 20000)]:
    p = ROOT / f"{name}.jsonl"
    if not p.exists():
        continue
    n, rep, demo, pm, ps_, mk = scan(p, lim)
    print(f"\n[{name}] n={n}")
    print(f"   lặp-tên ≥2× (cụm Hoa)      : {pct(rep,n)}")
    print(f"   danh-xưng đa-từ (ông ấy…)  : {pct(pm,n)}")
    print(f"   đại-từ đơn (ông/bà/nó/họ…) : {pct(ps_,n)}")
    print(f"   chỉ-định (này/đó)          : {pct(demo,n)}")
    print(f"   marker top: {', '.join(f'{k}×{v}' for k,v in mk.most_common(8))}")
    tot.update(mk)
print(f"\n→ TỔNG marker top: {', '.join(f'{k}×{v}' for k,v in tot.most_common(12))}")
