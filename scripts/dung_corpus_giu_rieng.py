"""§146 — Dựng corpus GIỮ RIÊNG để kiểm CHUYỂN GIAO thật.

§145 chứng minh phép thử chuyển giao hiện tại VÔ HIỆU THEO CẤU TẠO: bảng nhãn
lớp (§128) trích TỪ CHÍNH `dictionary.jsonl`, mà tệp đó nằm trong corpus huấn
luyện ⇒ mọi từ có nhãn lớp đều đã được dạy định nghĩa ⇒ không tồn tại tập giữ
riêng, nên mọi «thành công» đều giải thích được bằng GHI NHỚ.

Script này TÁCH một nhóm từ ra: xoá định nghĩa của chúng khỏi CẢ HAI nguồn
(`dictionary.jsonl` + `dinh_nghia.jsonl`), giữ nguyên anh em cùng lớp. Sau khi
train trên corpus đã tách và nướng câu lớp (§144), prompt `<từ giữ riêng> là`
chỉ có thể trả lời đúng NẾU tri thức chuyển giao qua kết nối lớp.

Tiêu chí chọn từ giữ riêng:
  · có định nghĩa trong `dictionary.jsonl` (để biết ĐÁP ÁN mà chấm)
  · lớp có ≥ 3 thành viên, và ≥ 2 anh em VẪN Ở LẠI corpus (nguồn để suy)
  · không phải từ ghép nhiều chữ (tránh nhiễu BPE)

    PYTHONPATH=src python scripts/dung_corpus_giu_rieng.py --n 60
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

DICT = Path("data/text_corpus/dictionary.jsonl")
DN = Path("data/text_corpus/dinh_nghia.jsonl")
BANG = Path("data/dict/nhan_lop.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="số từ giữ riêng")
    ap.add_argument("--hau-to", default="_ho", help="hậu tố tệp corpus mới")
    ap.add_argument("--ra-khoa", default="out/giu_rieng.json")
    a = ap.parse_args()

    bang = json.loads(BANG.read_text("utf-8"))
    lop2tu: dict[str, list[str]] = defaultdict(list)
    for w, ls in bang.items():
        for x in ls:
            p = x.split()
            if len(p) >= 2:
                lop2tu[" ".join(p[:2])].append(w)
                break

    rng = random.Random(0)
    giu: dict[str, dict] = {}
    for lop, tv in sorted(lop2tu.items()):
        tv = sorted(set(tv))
        if len(tv) < 3:
            continue
        ung = [w for w in tv if " " not in w and len(w) > 1]
        if len(ung) < 3:
            continue
        w = rng.choice(ung)                      # giữ riêng ĐÚNG MỘT từ mỗi lớp
        giu[w] = {"lop": lop, "anh_em": [x for x in ung if x != w][:6]}
        if len(giu) >= a.n:
            break
    print(f"### giữ riêng {len(giu)} từ (mỗi lớp một từ, lớp ≥3 thành viên)")
    for w, d in list(giu.items())[:6]:
        print(f"    {w:<14} lớp «{d['lop']}»  anh em còn lại: {d['anh_em'][:4]}")

    # ── lọc dictionary.jsonl
    n_bo = 0
    ra = DICT.with_name(DICT.stem + a.hau_to + DICT.suffix)
    with ra.open("w", encoding="utf-8") as f:
        for ln in DICT.open(encoding="utf-8"):
            try:
                t = json.loads(ln)
            except json.JSONDecodeError:
                continue
            m = re.search(r"^Từ:\s*(.+)$", t, re.M) if isinstance(t, str) else None
            if m and m.group(1).strip().lower() in giu:
                n_bo += 1
                continue
            f.write(ln)
    print(f"### dictionary: bỏ {n_bo} mục → {ra}")

    # ── lọc dinh_nghia.jsonl
    n_bo2 = 0
    ra2 = DN.with_name(DN.stem + a.hau_to + DN.suffix)
    with ra2.open("w", encoding="utf-8") as f:
        for ln in DN.open(encoding="utf-8"):
            try:
                t = json.loads(ln)
            except json.JSONDecodeError:
                continue
            m = re.match(r"^\s*([\wÀ-ỹ]+)\s+là\s", t.strip()) if isinstance(t, str) else None
            if m and m.group(1).lower() in giu:
                n_bo2 += 1
                continue
            f.write(ln)
    print(f"### dinh_nghia: bỏ {n_bo2} câu → {ra2}")

    Path(a.ra_khoa).parent.mkdir(parents=True, exist_ok=True)
    Path(a.ra_khoa).write_text(json.dumps(giu, ensure_ascii=False, indent=1), "utf-8")
    print(f"### khoá → {a.ra_khoa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
