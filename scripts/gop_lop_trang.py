"""§94 — GỘP ba nguồn phân loại thành DANH SÁCH LỚP TRẮNG (`lop_trang`).

§93.3 định vị lại chỗ hỏng: ② KHÔNG hỏng vì đồ thị nhiễu (bỏ 61 674 cạnh rác,
mọi chỉ số đứng im) mà vì **khái niệm «lớp» của tầng xếp hạng** — nó chuộng nhãn
TẦN SUẤT CAO, và nhãn tần suất cao trong tiếng Việt chính là nhãn chung chung
(`đồ`, `phần`, `loài`, `loại`). Một DANH SÁCH LỚP TRẮNG ghi đè được sở thích đó.

Chỉ cần tập TÊN LỚP hợp lệ, không cần ánh xạ thành viên — nên ba nguồn ở §90/§91
gộp được dù mỗi nguồn thiếu một thuộc tính khác nhau:

    A. giống từ điển (`out/cay_phan_loai.json`)  — phủ tốt, nhưng đầu cụm là hư
       danh: `phần` 202 · `cái` 174 · `từ` 137
    B. Wikidata `P31/P279` (`out/cay_wikidata.json`) — cạnh đúng kiểu, sai NGHĨA
    C. VietNet WN-LMF (`/tmp/vn_*.xml`) — đúng nghĩa, nhưng nhỏ

BIỂU QUYẾT: một nhãn vào danh sách trắng nếu ≥`--phieu` nguồn coi nó là LỚP.
Đó chính là cách loại `đồ`/`phần`/`cái`: chỉ nguồn A gọi chúng là lớp; B và C
không bao giờ trả về chúng như một lớp.

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ (a) danh sách trắng KHÔNG được chứa `đồ`, `phần`, `tên`, `cái`, `loại`,   ║
║     `loài` — sáu nhãn mà §89/§93 bắt được đứng đầu đề cử, và              ║
║ (b) PHẢI chứa `chim`, `thú`, `cây`, `kim loại` — bốn lớp vàng.            ║
║ Trượt (a) ⇒ biểu quyết không lọc được gì, dừng.                           ║
║ Trượt (b) ⇒ lọc quá tay, dừng.                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/gop_lop_trang.py
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

XAU = ["đồ", "phần", "tên", "cái", "loại", "loài"]
VANG = ["chim", "thú", "cây", "kim loại"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phieu", type=int, default=2)
    ap.add_argument("--out", default="out/lop_trang_gop.json")
    a = ap.parse_args()

    ng: dict[str, set[str]] = {}

    # A — giống từ điển: nhãn là LỚP nếu có ≥3 con (đã lọc lúc dựng)
    p = Path("out/cay_phan_loai.json")
    ng["tudien"] = set(json.loads(p.read_text("utf-8"))["lop"]) if p.exists() else set()

    # B — Wikidata: mọi lớp trả về từ P31/P279
    p = Path("out/cay_wikidata.json")
    ng["wikidata"] = ({c.lower().strip() for cs in json.loads(p.read_text("utf-8")).values()
                       for c in cs} if p.exists() else set())

    # C — VietNet: đích của cạnh hypernym, lấy LEMMA của synset đích
    vn: set[str] = set()
    for f in ("/tmp/vn_animal.xml", "/tmp/vn_food.xml"):
        if not Path(f).exists():
            continue
        r = ET.parse(f).getroot()
        s2w: dict[str, set[str]] = {}
        for e in r.iter("LexicalEntry"):
            lm = e.find("Lemma")
            if lm is None:
                continue
            w = lm.get("writtenForm", "").strip().lower()
            for s in e.iter("Sense"):
                s2w.setdefault(s.get("synset"), set()).add(w)
        for s in r.iter("Synset"):
            for rel in s.iter("SynsetRelation"):
                if rel.get("relType") == "hypernym":
                    vn |= s2w.get(rel.get("target"), set())
    ng["vietnet"] = vn

    for k, v in ng.items():
        print(f"###   {k:<10} {len(v):>5} nhãn")

    dem: Counter = Counter()
    for v in ng.values():
        for x in v:
            if x:
                dem[x] += 1
    trang = sorted(x for x, k in dem.items() if k >= a.phieu)

    print(f"\n### biểu quyết ≥{a.phieu} nguồn ⇒ {len(trang)} nhãn vào danh sách trắng\n")
    print("   CỔNG (a) sáu nhãn chung chung phải VẮNG:")
    for x in XAU:
        print(f"      {x:<10} {'✗ CÓ MẶT' if x in trang else 'vắng ✓'}"
              f"   (phiếu={dem.get(x, 0)})")
    print("   CỔNG (b) bốn lớp vàng phải CÓ:")
    for x in VANG:
        print(f"      {x:<10} {'có ✓' if x in trang else '✗ THIẾU'}"
              f"   (phiếu={dem.get(x, 0)})")
    print(f"\n   mẫu: {trang[:24]}")
    Path(a.out).write_text(json.dumps(trang, ensure_ascii=False), encoding="utf-8")
    print(f"\n### → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
