"""§90 — dựng CÂY PHÂN LOẠI thật từ quan hệ GIỐNG của từ điển.

§89 định vị nút thắt: ① chạy (33 974 cạnh `via_attr`, `khoẻ → voi` có thật) nhưng
② lấy phải NHÃN LỚP CHUNG CHUNG (`chậm như ___` → `đồ, phần, tên, bệnh`), rồi ③
nở lớp chung chung đó thành nhiễu. §75/§76 đã kết luận không lọc được nhãn chung
chung bằng sáu tín hiệu nội tại lẫn nguồn từ điển ⇒ phải nạp cây phân loại.

NGUỒN: chính từ điển. Mục `X` có nghĩa mở đầu bằng một CỤM DANH TỪ ⇒ đó là GIỐNG
của X. Đây là dữ liệu NGOÀI đã curated (cùng tiền lệ `pos_map`/`ws_spans`, Luật
0 cho phép ở bake), không phải tín hiệu nội tại của đồ thị.

ĐO TRƯỚC (§90.1): 1 978 cạnh giống trực tiếp. Cạnh ĐỘ SÂU 1 sạch:

    voi → thú → động vật    ✓        cắt → chim → động vật   ✓
    cam → cây → thực vật    ✓

nhưng BAO ĐÓNG BẮC CẦU rò qua từ đa nghĩa:

    chì → KIM → vật → cái → giống   ✗  (`kim` = cái kim, không phải kim loại)
    thúng → ĐỒ → người → động vật   ✗  (nghĩa lóng của `đồ`)

⇒ `động vật` phình thành 277 thành viên gồm `album`, `amygdala`, `bar`.

VÌ THẾ: chặn độ sâu, và chỉ leo qua nút thật sự là GIỐNG — nút có ≥`--con-toi-thieu`
con trực tiếp. Nút chỉ có một hai con là đầu cụm tình cờ, không phải lớp.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/dung_cay_phan_loai.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_aivie.hca.hub_spread import load_pos_map  # noqa: E402

T0 = time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="data/dict/vi_dict_raw.jsonl")
    ap.add_argument("--out", default="out/cay_phan_loai.json")
    ap.add_argument("--sau", type=int, default=2, help="độ sâu leo tối đa")
    ap.add_argument("--con-toi-thieu", type=int, default=3,
                    help="nút phải có ≥N con trực tiếp mới được coi là GIỐNG")
    a = ap.parse_args()

    pos = {k.lower(): str(v).lower() for k, v in load_pos_map(a.pos).items()}
    dt = lambda w: pos.get(w, "").startswith("danh")   # noqa: E731

    giong: dict[str, str] = {}
    for ln in Path(a.pos).open(encoding="utf-8"):
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        w = str(d.get("word", "")).strip().lower()
        m = str(d.get("meaning", "")).strip().lower()
        if not w or " " in w or not dt(w) or not m:
            continue
        ws = re.split(r"[,;:.]", m)[0].split()
        g = next((" ".join(ws[:k]) for k in (3, 2, 1)
                  if " ".join(ws[:k]) in pos and dt(" ".join(ws[:k]))), None)
        if g and g != w:
            giong.setdefault(w, g)

    con = Counter(giong.values())
    la_giong = {g for g, k in con.items() if k >= a.con_toi_thieu}

    # leo, CHẶN ĐỘ SÂU và chỉ qua nút thật sự là GIỐNG
    tv: dict[str, set[str]] = defaultdict(set)
    for x in giong:
        cur, sau = x, 0
        while sau < a.sau and cur in giong:
            g = giong[cur]
            if g not in la_giong:
                break
            tv[g].add(x)
            cur, sau = g, sau + 1

    lop = {c: sorted(s) for c, s in tv.items() if len(s) >= a.con_toi_thieu}
    Path(a.out).write_text(
        json.dumps({"lop": lop, "n_canh_giong": len(giong)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    Path("out/lop_trang.json").write_text(
        json.dumps(sorted(lop), ensure_ascii=False), encoding="utf-8")

    print(f"\n### {len(giong)} cạnh giống · {len(la_giong)} nút là GIỐNG"
          f" (≥{a.con_toi_thieu} con) · sâu ≤{a.sau}")
    print(f"### {len(lop)} lớp giữ lại → {a.out} · danh sách → out/lop_trang.json\n")
    for c, s in sorted(lop.items(), key=lambda kv: -len(kv[1]))[:16]:
        print(f"   {c:>12}: {len(s):>3} tv  ·  {s[:8]}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
