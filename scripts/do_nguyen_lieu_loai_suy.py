"""§88 ĐO NGUYÊN LIỆU — dựng corpus loại suy từ nguồn nào?

Kiểm `data/text_corpus/so_sanh_that.jsonl` (TK_052 vòng 4) cho thấy nó KHÔNG
phải corpus so sánh: 11 553 câu chỉ CHỨA chữ `như`, mà `như` ở đó gần như đều
nghĩa «chẳng hạn» (liệt kê), không phải so sánh thuộc tính:

    …các nước châu Âu khác NHƯ Pháp, Thuỵ Sĩ, Đức…       ← liệt kê
    …gần NHƯ tương tự…                                     ← trạng từ mức độ

Train nó còn NGUY HIỂM: bơm đúng khuôn liệt kê mà TK_052 vòng 6 phải vá để khỏi
phát cạnh NGƯỢC. Và `loai_suy.jsonl` thì tổng hợp, vô nghĩa (`ngặt như loài
biển`), lặp 4 lần — không train.

Vòng này ĐẾM nguyên liệu thật của ba nguồn ứng viên, trước khi dựng bất cứ gì:

  N1  thành ngữ so sánh CÓ SẴN trong từ điển  — mục có dạng «A như B»
  N2  `<tính từ> như <danh từ>` lọc bằng TỪ LOẠI thật (`vi_dict_raw.jsonl`),
      quét trên mọi nguồn — chứ không chỉ khớp chữ `như` như TK_052 đã làm
  N3  THUỘC TÍNH rút từ chính câu định nghĩa (`X là <lớp>, <thuộc tính>…`)
      — đây là công thức §80: viết lại tri thức THẬT thành hình dạng cơ chế đọc

╔═ CỔNG CHỐT TRƯỚC KHI CHẠY ═══════════════════════════════════════════════╗
║ Nguồn nào dùng được phải có ĐỦ CẢ HAI:                                   ║
║   (a) ≥300 thuộc tính có ≥2 VẬT MANG khác nhau (điều kiện cần của ②), và ║
║   (b) vật mang là DANH TỪ thật theo từ điển từ loại — không phải chuỗi   ║
║       bắt được bằng regex.                                               ║
║ Nguồn nào trượt (a) thì loại, không dựng. Nếu CẢ BA trượt ⇒ báo lại,     ║
║ không dựng corpus tổng hợp thay thế (đó là tự lừa: TK_052 đã làm rồi và  ║
║ nó chỉ chứng minh cơ chế, không tạo tri thức).                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/do_nguyen_lieu_loai_suy.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

T0 = time.time()
GOC = Path("data/text_corpus")
POS = "data/dict/vi_dict_raw.jsonl"


def nap_pos() -> dict[str, str]:
    """DÙNG LOADER CỦA DỰ ÁN. Bản đầu tự đoán tên trường (`tu`/`tu_loai`) và nạp
    được 0 mục ⇒ bộ lọc từ loại vứt sạch, cả ba nguồn ra 0 — một phép đo VÔ HIỆU
    trông y như kết quả âm tính thật. Trường thật là `word`/`part_of_speech`."""
    from nn_aivie.hca.hub_spread import load_pos_map
    return {k.lower(): str(v).lower() for k, v in load_pos_map(POS).items()}


def cau(path: Path):
    if not path.exists():
        return
    for ln in path.open(encoding="utf-8"):
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        t = d if isinstance(d, str) else (
            d.get("text") or d.get("content") or "" if isinstance(d, dict) else "")
        if t:
            yield str(t)


def main() -> int:
    pos = nap_pos()
    print(f"\n### từ điển từ loại: {len(pos)} mục")
    la_tt = lambda w: pos.get(w, "").startswith("tính")   # noqa: E731
    la_dt = lambda w: pos.get(w, "").startswith("danh")   # noqa: E731

    # ── N1 + N2: khuôn «A như B» trên MỌI nguồn, lọc bằng TỪ LOẠI
    RX = re.compile(r"(\w+(?:\s+\w+)?)\s+như\s+(\w+(?:\s+\w+)?)", re.UNICODE)
    mang: dict[str, set[str]] = defaultdict(set)
    tho = loc = 0
    for f in sorted(GOC.glob("*.jsonl")):
        if f.name in ("loai_suy.jsonl",):      # tổng hợp — bỏ
            continue
        for t in cau(f):
            for m in RX.finditer(t.lower()):
                tho += 1
                a, b = m.group(1).strip(), m.group(2).strip()
                a1, b1 = a.split()[-1], b.split()[0]
                if la_tt(a1) and la_dt(b1):
                    loc += 1
                    mang[a1].add(b1)
    du = {a: b for a, b in mang.items() if len(b) >= 2}
    print(f"\n── N1+N2  «A như B» trên mọi nguồn")
    print(f"     khớp THÔ (chỉ chữ `như`)          : {tho:>7}")
    print(f"     còn lại sau lọc TÍNH TỪ như DANH TỪ: {loc:>7}")
    print(f"     thuộc tính có ≥2 vật mang          : {len(du):>7}"
          f"   ← cổng (a) cần ≥300")
    for a, b in sorted(du.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"        {a:>12} → {sorted(b)[:6]}")

    # ── N3: thuộc tính rút từ CÂU ĐỊNH NGHĨA (công thức §80)
    RXD = re.compile(r"^(.+?)\s+là\s+(.+)$")
    mang3: dict[str, set[str]] = defaultdict(set)
    n_dn = 0
    for t in cau(GOC / "dinh_nghia.jsonl"):
        m = RXD.match(t.strip())
        if not m:
            continue
        n_dn += 1
        chu = m.group(1).strip().lower()
        if " " in chu or not la_dt(chu):
            continue
        for ve in re.split(r"[,;]", m.group(2)):
            ws = ve.strip().lower().split()
            if not ws:
                continue
            # tính từ ĐỨNG CUỐI mệnh đề mô tả: «mõm tròn», «đuôi xù»
            if la_tt(ws[-1]):
                mang3[ws[-1]].add(chu)
    du3 = {a: b for a, b in mang3.items() if len(b) >= 2}
    print(f"\n── N3  thuộc tính rút từ CÂU ĐỊNH NGHĨA ({n_dn} câu «X là …»)")
    print(f"     thuộc tính có ≥2 vật mang          : {len(du3):>7}"
          f"   ← cổng (a) cần ≥300")
    for a, b in sorted(du3.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"        {a:>12} → {sorted(b)[:6]}")

    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
