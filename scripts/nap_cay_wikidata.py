"""§90 — nạp CÂY PHÂN LOẠI THẬT từ Wikidata (`P31` thực thể / `P279` lớp con).

Vì sao không dùng quan hệ GIỐNG của từ điển (§90.1 đã dựng và BÁC):
định nghĩa tiếng Việt hay mở đầu bằng danh từ GIỮ CHỖ — `cái…`, `phần của…`,
`đồ…`, `từ dùng để…` — nên đầu cụm là hư danh chứ không phải lớp. Cây dựng ra có
đỉnh bảng đúng bằng loại nhãn §89 chê: `phần` 202 · `cái` 174 · `từ` 137 ·
`người` 124 · `vật` 112. Lọc kích thước không cứu được: `cây` (102, tốt) và `vật`
(112, xấu) không phân biệt nổi bằng số thành viên. Và bao đóng bắc cầu rò qua từ
đa nghĩa: `chì → KIM → vật → cái` (`kim` = cái kim, không phải kim loại).

Wikidata có cạnh CÓ KIỂU nên không mắc lỗi đó:

    chì → kim loại yếu, nguyên tố hóa học      (từ điển cho `kim` ✗)
    sắt → kim loại  ·  cò → chim, diệc  ·  voi → lớp Thú  ·  cam → quả có múi

Chỉ gửi đi một DANH SÁCH DANH TỪ tiếng Việt phổ thông (lấy từ khoá corpus và cây
từ điển), không có dữ liệu người dùng.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/nap_cay_wikidata.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

T0 = time.time()
EP = "https://query.wikidata.org/sparql"
# meta-lớp của Wikidata, không phải lớp ngữ nghĩa
BO = {"đơn vị phân loại", "tên phổ thông",
      "sinh vật được biết đến bởi một tên chung cụ thể"}


def hoi(tu: list[str]) -> list[tuple[str, str]]:
    vals = " ".join(f'"{t}"@vi' for t in tu)
    q = (f'SELECT ?l ?cl WHERE {{ VALUES ?l {{ {vals} }} '
         f'?i rdfs:label ?l . ?i wdt:P31|wdt:P279 ?c . '
         f'?c rdfs:label ?cl . FILTER(lang(?cl)="vi") }}')
    p = subprocess.run(
        ["curl", "-s", "--max-time", "90", "-G", EP,
         "--data-urlencode", f"query={q}",
         "-H", "Accept: application/sparql-results+json",
         "-A", "nn-aivie-research/1.0 (local research)"],
        capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return []
    return [(x["l"]["value"], x["cl"]["value"])
            for x in d.get("results", {}).get("bindings", [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=90, help="số từ mỗi lô")
    ap.add_argument("--out", default="out/cay_wikidata.json")
    a = ap.parse_args()

    tu: set[str] = set()
    p = Path("out/cay_phan_loai.json")
    if p.exists():
        for c, ms in json.loads(p.read_text("utf-8"))["lop"].items():
            tu.add(c)
            tu.update(ms)
    k = Path("out/loai_suy_that_khoa.json")
    if k.exists():
        for b in json.loads(k.read_text("utf-8")):
            tu.update(b["gieo"])
            tu.update(b["khi"])
    tu = {t for t in tu if t and " " not in t and '"' not in t}
    ds = sorted(tu)
    print(f"\n### hỏi Wikidata cho {len(ds)} từ, lô {a.lo}\n")

    cha: dict[str, set[str]] = defaultdict(set)
    for i in range(0, len(ds), a.lo):
        lo = ds[i:i + a.lo]
        r = hoi(lo)
        for w, c in r:
            if c not in BO:
                cha[w].add(c)
        print(f"   lô {i//a.lo + 1}: {len(lo)} từ → {len(r)} cặp"
              f" · tích luỹ {len(cha)} từ có lớp", flush=True)
        time.sleep(1.0)

    Path(a.out).write_text(
        json.dumps({w: sorted(s) for w, s in cha.items()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n### {len(cha)}/{len(ds)} từ có lớp Wikidata → {a.out}")
    for w in sorted(cha)[:14]:
        print(f"   {w:>12} → {sorted(cha[w])[:5]}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
