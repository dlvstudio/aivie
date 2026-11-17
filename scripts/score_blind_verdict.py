#!/usr/bin/env python
"""Lật khoá + đếm phiếu chấm mù — MỘT nguồn duy nhất cho mọi vòng.

Sinh ra sau khi đếm tay **sáu vòng** và mắc lỗi ở vòng thứ sáu. Ba cái bẫy đã thật sự xảy
ra, script này đóng cả ba:

1. **Đếm theo VỊ TRÍ thay vì theo CẤU HÌNH.** `X`/`Y` được trộn ngẫu nhiên **từng cặp**, nên
   tổng X-vs-Y **vô nghĩa**. Đã xảy ra **ba** lần: hai giám khảo tự kết luận "không phân biệt
   được" (thực tế 92% và 85.6%), và **một lần chính tôi** — ngay sau khi vừa viết ra cái bẫy đó.
2. **So chuỗi phiếu hoà một cách ngây thơ.** Một vòng ghi `"hoà"` (có dấu), các vòng khác ghi
   `"hoa"`. `v == 'hoa'` ⇒ 13 phiếu hoà rơi xuống nhánh else và thành **thắng cho bên phải**,
   âm thầm. Nay chuẩn hoá NFC và coi **mọi** giá trị ngoài `X`/`Y` là hoà.
3. **Không kiểm hiện vật vị trí.** Một bên có thể "thắng" chỉ vì nó hay nằm bên trái. Script
   luôn in thắng-thua **tách theo vị trí**; nếu chỉ thắng ở một phía thì kết quả không tin được.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/score_blind_verdict.py \\
      --key out/join_key_v2.json --verdict out/join_verdict_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.eval.ruler import min_pairs_for, win_rate_ci  # noqa: E402


def normalize(v) -> str:
    """`X` / `Y` / `TIE`. Mọi thứ không phải X,Y đều là hoà — không so chuỗi cứng."""
    s = unicodedata.normalize("NFC", str(v)).strip().upper()
    return s if s in ("X", "Y") else "TIE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--verdict", required=True)
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    raw = json.loads(Path(args.verdict).read_text(encoding="utf-8"))
    ver = raw.get("verdicts", raw)

    missing = set(key) - set(ver)
    extra = set(ver) - set(key)
    if missing or extra:
        print(f"⚠️ khoá/verdict LỆCH — thiếu {len(missing)} · thừa {len(extra)}")

    wins = Counter()
    by_pos = Counter()      # (bên nào ở TRÁI, ai thắng)
    left_count = Counter()
    raw_pos = Counter()
    for pid, v in ver.items():
        if pid not in key:
            continue
        n = normalize(v)
        raw_pos[n] += 1
        k = key[pid]
        left_count[k["left"]] += 1
        if n == "TIE":
            wins["hoà"] += 1
            continue
        w = k["left"] if n == "X" else k["right"]
        wins[w] += 1
        by_pos[(k["left"], w)] += 1

    cfgs = sorted(c for c in wins if c != "hoà")
    print(f"=== {Path(args.verdict).name} ===")
    print(f"  thô theo VỊ TRÍ (VÔ NGHĨA — chỉ in để đối chiếu): {dict(raw_pos)}")
    print(f"\n  theo CẤU HÌNH:")
    for c in cfgs:
        print(f"    {c:16s} {wins[c]:3d}")
    print(f"    {'hoà':16s} {wins['hoà']:3d}")

    if len(cfgs) != 2:
        print(f"\n⚠️ tìm thấy {len(cfgs)} cấu hình, cần đúng 2 để tính CI.")
        return
    a, b = cfgs
    na, nb = wins[a], wins[b]
    tot = na + nb
    p, lo, hi = win_rate_ci(na, tot)
    sep = lo > 0.5 or hi < 0.5
    print(f"\n  cân bằng vị trí: {a}-trái {left_count[a]} · {b}-trái {left_count[b]}")
    print(f"  **{a} thắng {na}/{tot} = {p:.1%}**  CI95 [{lo:.1%}, {hi:.1%}]")
    print(f"  tách khỏi 50%? {'CÓ' if sep else 'KHÔNG — không phân biệt được'}")
    print(f"  (n={tot} phân định · cần {min_pairs_for(15)} cho +15pp @ power 80%)")

    print(f"\n  kiểm hiện vật VỊ TRÍ (phải thắng ở CẢ HAI phía mới tin được):")
    ok_both = True
    for side in (a, b):
        wa, wb = by_pos[(side, a)], by_pos[(side, b)]
        print(f"    khi {side:16s} ở TRÁI: {a} {wa:3d} · {b} {wb:3d}")
        if sep:
            winner = a if p > 0.5 else b
            if by_pos[(side, winner)] <= by_pos[(side, a if winner == b else b)]:
                ok_both = False
    if sep and not ok_both:
        print("    ⚠️ KHÔNG thắng ở cả hai phía ⇒ nghi hiện vật vị trí, ĐỪNG kết luận.")
    elif sep:
        print("    ✓ thắng ở cả hai phía — không phải hiện vật vị trí.")


if __name__ == "__main__":
    main()
