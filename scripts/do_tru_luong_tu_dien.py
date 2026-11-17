"""Trữ lượng lớp trong TỪ ĐIỂN — nguồn `classrich` cố ý loại bỏ.

viwiki cho 194× câu định nghĩa thô, nhưng phần DÙNG ĐƯỢC (thành viên là từ
tiếng Việt, không phải tên Latin của bài stub) chỉ còn 906 cặp kiểu sinh vật —
vì Wikipedia tiếng Việt viết về `Abrocoma bennettii`, không viết về `chó`.

Từ điển thì ngược lại: 26 244 mục, toàn TỪ THƯỜNG, và lời giải nghĩa có cấu
trúc GIỐNG–LOÀI: «a: *nông cụ* gồm hai lưỡi cắt…» ⇒ `a → nông cụ`. Đúng quan
hệ mà `via_pivot` cần, chỉ thiếu chữ `là`.

Script này đếm trữ lượng trước khi quyết có dựng corpus mới không.

    PYTHONHASHSEED=0 python scripts/do_tru_luong_tu_dien.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict

T0 = time.time()
_TU = re.compile(r"^Từ:\s*(.+?)\s*$", re.M)
_LOAI = re.compile(r"^Loại từ:\s*(.+?)\s*$", re.M)
_NGHIA = re.compile(r"^Giải nghĩa:\s*(.+?)\s*$", re.M)
# Bỏ phần dẫn: «(khẩu ngữ)», «(cũ)», «(id)»…
_DAN = re.compile(r"^\s*\([^)]{1,30}\)\s*")


def main() -> int:
    lop: dict[str, set[str]] = defaultdict(set)
    n = n_dt = n_lay = 0
    tra: dict[str, str] = {}
    with open("data/text_corpus/dictionary.jsonl", encoding="utf-8") as f:
        for dong in f:
            try:
                t = json.loads(dong)
            except Exception:  # noqa: BLE001
                continue
            t = str(t)
            n += 1
            mt = _TU.search(t)
            ml = _LOAI.search(t)
            mn = _NGHIA.search(t)
            if not (mt and ml and mn):
                continue
            if "danh từ" not in ml.group(1):
                continue
            n_dt += 1
            tu = mt.group(1).strip()
            nghia = _DAN.sub("", mn.group(1).strip())
            # ĐẦU ngữ đoạn tiếng Việt nằm TRƯỚC: lấy 1–2 từ đầu lời giải nghĩa,
            # cắt ở dấu câu hoặc ở từ nối.
            dau = re.split(r"[,:;(]| mà | để | dùng | có | gồm | là ", nghia)[0]
            c = " ".join(dau.split()[:2]).strip().lower()
            if not c or len(c) < 2 or not c.replace(" ", "").isalpha():
                continue
            lop[c].add(tu)
            tra[tu] = c
            n_lay += 1
    print(f"\n### từ điển: {n} mục · {n_dt} danh từ · {n_lay} rút được lớp\n")
    big = {c: m for c, m in lop.items() if len(m) >= 3}
    print(f"  cặp (thành viên → lớp): {n_lay:>6}      (classrich 949 · viwiki dùng được 15 591)")
    print(f"  lớp có ≥3 thành viên  : {len(big):>6}      (classrich 168 · viwiki dùng được 513)")
    print("\n  15 lớp lớn nhất:")
    for c, m in sorted(big.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"     {c:<18} {len(m):>4} tv  vd {sorted(m)[:5]}")
    print("\n  BA CON VẬT của ca vàng:")
    for w in ("chó", "mèo", "khỉ", "hổ", "gấu", "voi"):
        print(f"     {w:<6} → {tra.get(w)!r}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
