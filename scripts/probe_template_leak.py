#!/usr/bin/env python
"""P67 (TK_045 §13.34) — đếm **RÒ RỈ KHUÔN MẪU** trong corpus train v12.

**Vì sao.** §13.33: giám khảo mù vòng bảy phát hiện một kiểu hỏng **chưa từng ghi** — đầu ra
chứa dấu vết **khuôn mẫu soạn sẵn**: chỗ trống `[chủ đề]` của mẫu thư (G054), gạch đầu dòng
`*` (G076), danh sách `- Bạn có…` chen giữa văn xuôi (G001). Đây **không** phải lỗi tầng sinh:
mạng chỉ phát lại thứ nó đã đọc. ⇒ đo ở **corpus**, không đo ở mạng.

**Đo cái gì.** Tỷ lệ **đơn vị sẽ thành cha** có mang dấu khuôn mẫu. v12 dùng
`--sentence-parents` ⇒ đơn vị ≈ **câu**. Script tách câu bằng regex trên văn bản thô, nên con
số là **xấp xỉ** của mức cha thật (bộ tách thật chạy trên token id sau BPE). Ghi rõ để không
ai trích nó như số chính xác.

**Không sửa gì.** Script chỉ đếm và in mẫu. Lọc hay không là quyết định sau, cần số trước.

Chạy:
  PYTHONPATH=src python scripts/probe_template_leak.py
  PYTHONPATH=src python scripts/probe_template_leak.py --manifest config/train_v12_sent.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# --- các lớp dấu khuôn mẫu -------------------------------------------------------------
# Mỗi lớp: (tên, regex, ghi chú vì sao coi là khuôn mẫu)
PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # `[chủ đề]`, `[tên người nhận]` — chỗ trống của mẫu thư. LOẠI TRỪ `[1]`, `[12]` (chú thích
    # wiki) vì đó là nhiễu trích dẫn, bệnh khác.
    ("cho_trong_ngoac_vuong", re.compile(r"\[[^\]\d][^\]]{0,39}\]"),
     "chỗ trống mẫu thư `[chủ đề]` — G054"),
    ("chu_thich_so", re.compile(r"\[\d{1,3}\]"),
     "chú thích wiki `[12]` — KHÁC lớp trên, tách ra để không trộn"),
    ("gach_dau_dong_sao", re.compile(r"(?m)^\s*\*+\s|\*\*"),
     "gạch đầu dòng / in đậm markdown `*` — G076"),
    ("gach_dau_dong_ngang", re.compile(r"(?m)^\s*[-–—]\s+\S"),
     "danh sách `- Bạn có…` — G001"),
    ("danh_sach_so", re.compile(r"(?m)^\s*\d{1,2}[.)]\s+\S"),
     "danh sách đánh số `1. `"),
    ("tieu_de_markdown", re.compile(r"(?m)^\s*#{1,6}\s+\S"), "tiêu đề markdown `## `"),
    ("ngoac_nhon", re.compile(r"\{\{?[^}]{1,40}\}?\}"), "biến mẫu `{{tên}}`"),
    ("the_html", re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,9}\s*/?>"), "thẻ HTML còn sót"),
    ("cho_trong_gach_duoi", re.compile(r"_{3,}"), "chỗ trống `____` của phiếu bài tập"),
]

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def units_of(text: str) -> list[str]:
    """Xấp xỉ đơn vị-thành-cha: tách câu theo dấu kết + xuống dòng."""
    return [s for s in (p.strip() for p in _SENT_SPLIT.split(text)) if s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/train_v12_sent.yaml")
    ap.add_argument("--examples", type=int, default=3, help="số mẫu in cho mỗi lớp")
    ap.add_argument("--out", default="out/probe_template_leak.json")
    args = ap.parse_args()

    import yaml

    man = yaml.safe_load((_ROOT / args.manifest).read_text(encoding="utf-8"))
    datasets = [d["path"] for d in man["datasets"]]

    per_ds: dict[str, dict] = {}
    tot_units = 0
    tot_hit = 0
    cls_total: Counter[str] = Counter()
    samples: dict[str, list[str]] = {n: [] for n, _, _ in PATTERNS}

    for rel in datasets:
        p = _ROOT / rel
        if not p.exists():
            print(f"⚠️ thiếu {rel} — bỏ qua")
            continue
        n_units = 0
        n_hit = 0
        cls: Counter[str] = Counter()
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(text, dict):  # phòng khi định dạng khác
                    text = text.get("text") or text.get("content") or ""
                if not isinstance(text, str):
                    continue
                for u in units_of(text):
                    n_units += 1
                    hit = False
                    for name, rx, _ in PATTERNS:
                        if rx.search(u):
                            cls[name] += 1
                            hit = True
                            if len(samples[name]) < args.examples:
                                samples[name].append(f"[{Path(rel).stem}] {u[:150]}")
                    if hit:
                        n_hit += 1
        per_ds[rel] = {"units": n_units, "hit": n_hit,
                       "rate": (n_hit / n_units if n_units else 0.0), "classes": dict(cls)}
        cls_total.update(cls)
        tot_units += n_units
        tot_hit += n_hit
        print(f"{Path(rel).stem:<22} {n_units:>8,} đơn vị · dính {n_hit:>7,} "
              f"({n_hit / n_units if n_units else 0:6.2%})")

    print(f"\nTỔNG {tot_units:,} đơn vị · dính {tot_hit:,} "
          f"({tot_hit / tot_units if tot_units else 0:.2%})")

    print("\nTheo LỚP dấu (một đơn vị có thể dính nhiều lớp):")
    for name, _, why in PATTERNS:
        c = cls_total.get(name, 0)
        print(f"  {name:<24} {c:>8,} ({c / tot_units if tot_units else 0:6.2%})  {why}")

    print("\nMẫu:")
    for name, _, _ in PATTERNS:
        for s in samples[name][: args.examples]:
            print(f"  [{name}] {s}")

    out = {"manifest": args.manifest, "units": tot_units, "hit": tot_hit,
           "rate": (tot_hit / tot_units if tot_units else 0.0),
           "by_class": dict(cls_total), "by_dataset": per_ds,
           "samples": {k: v for k, v in samples.items() if v},
           "caveat": "đơn vị = câu tách bằng regex trên văn bản thô ⇒ XẤP XỈ mức cha thật"}
    op = _ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
