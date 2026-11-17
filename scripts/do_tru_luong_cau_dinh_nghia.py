"""Trữ lượng câu ĐỊNH NGHĨA trong `viwiki` — và bao nhiêu LỚP THẬT rút ra được.

§64 chốt: nút thắt cuối không phải cơ chế mà là MẬT ĐỘ — câu định nghĩa dạng
vàng chỉ chiếm 2,1% trên `classrich`. Nhưng `classrich` được dựng từ 4 nguồn
nhỏ (~16 MB) và **loại hẳn `viwiki` 687 MB**, trong khi câu mở đầu bài
Wikipedia tiếng Việt VỐN LÀ câu định nghĩa theo quy ước biên tập:

    grep viwiki:  là một loài 71 267 · là một xã 33 581 · thuộc họ 29 834
                  thuộc chi 20 615 · là một thành phố 10 523
    classrich  :  là loài 164 · là động vật 54

Script này KHÔNG train — nó chỉ đọc văn bản và đếm xem nếu lọc theo khuôn thì
được bao nhiêu cặp (thành viên → lớp) và bao nhiêu lớp có ≥3 thành viên, để so
với 949 cặp / 168 lớp hiện có.

    PYTHONHASHSEED=0 python scripts/do_tru_luong_cau_dinh_nghia.py [số dòng]
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict

T0 = time.time()

# KHUÔN — mỗi khuôn: (tên, regex có 2 nhóm: chủ ngữ, lớp)
# Chủ ngữ = đầu câu tới pivot; lớp = ngữ đoạn ngay sau, cắt ở dấu câu/liên từ.
_DUNG = r"[^,.;:()\[\]«»\"]{1,60}"
KHUON: list[tuple[str, re.Pattern[str]]] = [
    ("là một loài", re.compile(rf"^({_DUNG}?) là một loài ({_DUNG})")),
    ("là loài", re.compile(rf"^({_DUNG}?) là loài ({_DUNG})")),
    ("là một loại", re.compile(rf"^({_DUNG}?) là một loại ({_DUNG})")),
    ("là một <lớp>", re.compile(rf"^({_DUNG}?) là một ({_DUNG})")),
    ("thuộc họ/chi/bộ", re.compile(rf"^({_DUNG}?) .{{0,40}}?thuộc (?:họ|chi|bộ|lớp) ({_DUNG})")),
]
# `là một <lớp>` bắt cả `là một trong…` — loại bằng chính khuôn, không danh sách
BO = re.compile(r"^(trong|những|số|phần|người|điều|nơi|cách|vài|ít|nhiều)\b")


def _tu_dien() -> set[str]:
    """Từ điển tiếng Việt — để tách thành viên TIẾNG VIỆT khỏi tên Latin."""
    tu: set[str] = set()
    with open("data/dict/vi_dict_raw.jsonl", encoding="utf-8") as f:
        for dong in f:
            try:
                d = json.loads(dong)
            except Exception:  # noqa: BLE001
                continue
            w = (d.get("word") or d.get("tu") or "").strip().lower()
            if w:
                tu.add(w)
    return tu


def main() -> int:
    gioi_han = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    tu = _tu_dien()
    print(f"  từ điển {len(tu)} từ")
    lop_sach: dict[str, set[str]] = defaultdict(set)
    lop: dict[str, set[str]] = defaultdict(set)
    dem_khuon: dict[str, int] = defaultdict(int)
    n_cau = n_khop = 0
    with open("data/text_corpus/viwiki.jsonl", encoding="utf-8") as f:
        for i, dong in enumerate(f):
            if i >= gioi_han:
                break
            try:
                doc = json.loads(dong)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(doc, dict):
                doc = doc.get("text") or doc.get("content") or ""
            # CHỈ câu đầu mỗi đoạn — câu định nghĩa của Wikipedia nằm ở đó
            for para in str(doc).split("\n"):
                cau = para.strip().split(". ")[0].strip()
                if not (20 < len(cau) < 300):
                    continue
                n_cau += 1
                for ten, rx in KHUON:
                    m = rx.match(cau)
                    if not m:
                        continue
                    tv, c = m.group(1).strip(), m.group(2).strip()
                    if BO.match(c) or len(tv) < 2 or len(c) < 2:
                        continue
                    # LỚP = 2 từ đầu ngữ đoạn (đầu ngữ đoạn tiếng Việt ở TRƯỚC)
                    c = " ".join(c.split()[:2])
                    if ten in ("là một loài", "là loài"):
                        c = "loài " + c.split()[0]
                    elif ten == "thuộc họ/chi/bộ":
                        c = "họ " + c.split()[0]
                    tv3 = " ".join(tv.split()[-3:])
                    lop[c].add(tv3)
                    # DÙNG ĐƯỢC = thành viên là TỪ TIẾNG VIỆT có trong từ điển
                    # (không phải tên khoa học Latin của bài stub tự sinh) và
                    # đầu lớp cũng vậy. Đây mới là phần chuỗi ba bước dùng nổi:
                    # `khỉ` phải là từ xuất hiện trong văn xuôi thường.
                    tvl = tv.split()[-1].lower()
                    if (tvl in tu and c.split()[0].lower() in tu
                            and len(tv.split()) <= 3):
                        lop_sach[c].add(tv3)
                    dem_khuon[ten] += 1
                    n_khop += 1
                    break
    print(f"\n### {gioi_han} dòng viwiki · {n_cau} câu đầu đoạn · {n_khop} khớp khuôn\n")
    for ten, n in sorted(dem_khuon.items(), key=lambda kv: -kv[1]):
        print(f"  {ten:<18} {n:>7}")
    big = {c: m for c, m in lop.items() if len(m) >= 3}
    print(f"\n  cặp (thành viên → lớp) : {sum(len(m) for m in lop.values()):>7}"
          f"      (classrich hiện có: 949)")
    print(f"  lớp có ≥3 thành viên   : {len(big):>7}"
          f"      (classrich hiện có: 168)")
    bigs = {c: m for c, m in lop_sach.items() if len(m) >= 3}
    print(f"\n  ── phần DÙNG ĐƯỢC (thành viên là từ tiếng Việt) ──")
    print(f"  cặp  : {sum(len(m) for m in lop_sach.values()):>7}"
          f"      lớp ≥3 tv: {len(bigs)}")
    print("  12 lớp DÙNG ĐƯỢC lớn nhất:")
    for c, m in sorted(bigs.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"     {c:<22} {len(m):>5} tv  vd {sorted(m)[:4]}")
    sv = {c: m for c, m in bigs.items()
          if c.split()[0] in ("loài", "họ", "chi", "cây", "con")}
    print(f"\n  ── lớp SINH VẬT dùng được (đúng kiểu ca vàng) ──")
    print(f"  {len(sv)} lớp · {sum(len(m) for m in sv.values())} cặp")
    for c, m in sorted(sv.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"     {c:<22} {len(m):>5} tv  vd {sorted(m)[:4]}")
    print("\n  12 lớp lớn nhất (thô):")
    for c, m in sorted(big.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"     {c:<22} {len(m):>5} tv  vd {sorted(m)[:3]}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
