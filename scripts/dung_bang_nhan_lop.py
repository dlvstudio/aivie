"""§128 — Bake bảng NHÃN LỚP từ KHUÔN ĐỊNH NGHĨA trong từ điển.

§127 đo: `via_pivot` (học thống kê `A là B` trên toàn văn bản) cho nhãn lớp đúng
**4,1%** trên 1 718 từ — vì mẫu bắt trúng câu hỏi («Nước là gì?» ⇒ nhãn `gì`) và
mệnh đề tu từ («chó là bạn»). Lọc đích (`loc_dich`) chỉ nâng lên 4,8%, đúng 1 SE
⇒ không cứu được.

Nhưng KHUÔN định nghĩa là dữ liệu CÓ CẤU TRÚC nằm sẵn trong corpus:

    Từ: a
    Loại từ: danh từ
    Giải nghĩa: NÔNG CỤ gồm hai lưỡi cắt tra vào cán dài…
                └── loài gần = nhãn lớp

Luật 0: từ điển là dữ liệu NGOÀI dùng ở BAKE (cùng tiền lệ `pos_map`, `ws_spans`);
runtime chỉ đọc bảng đã bake. KHÔNG có danh sách từ viết tay ở đây — từ hạn định
(`một`, `các`, `những`) bị loại bằng IDF tính từ chính corpus, không bằng bảng tay.

    PYTHONPATH=src python scripts/dung_bang_nhan_lop.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

NGUON = Path("data/text_corpus/dictionary.jsonl")
RA = Path("data/dict/nhan_lop.json")
TU = re.compile(r"[\wÀ-ỹ]+")


def _muc(t: str) -> tuple[str, str, str] | None:
    """`Từ:` / `Loại từ:` / `Giải nghĩa:` → (từ, từ loại, giải nghĩa)."""
    m_t = re.search(r"^Từ:\s*(.+)$", t, re.M)
    m_l = re.search(r"^Loại từ:\s*(.+)$", t, re.M)
    m_g = re.search(r"^Giải nghĩa:\s*(.+)$", t, re.M)
    if not (m_t and m_g):
        return None
    return (m_t.group(1).strip().lower(),
            (m_l.group(1).strip().lower() if m_l else ""),
            m_g.group(1).strip())


def main() -> int:
    muc: list[tuple[str, str, str]] = []
    for ln in NGUON.open(encoding="utf-8"):
        try:
            t = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(t, str) and (x := _muc(t)):
            muc.append(x)
    print(f"### {len(muc)} mục đọc được từ {NGUON}")

    # IDF trên chính tập giải nghĩa — để loại từ hạn định mà KHÔNG viết bảng tay.
    df: Counter[str] = Counter()
    for _, _, g in muc:
        df.update(set(w.lower() for w in TU.findall(g)))
    N = max(1, len(muc))

    def idf(w: str) -> float:
        return math.log(N / (1 + df.get(w, 0)))

    # Ngưỡng «từ hạn định» = phân vị 15 của IDF trên tập từ xuất hiện ở đầu lời
    # giải nghĩa. Dữ liệu tự quyết, không có danh sách tay.
    _dau_ws: list[float] = []
    for _, _l, _g in muc:
        if "danh từ" not in _l:
            continue
        _w = TU.findall(re.sub(r"\([^)]*\)", " ", _g).split(":")[0])
        if _w:
            _dau_ws.append(idf(_w[0].lower()))
    _dau_ws.sort()
    nguong = _dau_ws[max(0, int(0.15 * len(_dau_ws)))] if _dau_ws else 0.0
    print(f"### ngưỡng hạn định (phân vị 15 IDF) = {nguong:.3f}")

    bang: dict[str, list[str]] = {}
    for tu, loai, g in muc:
        if "danh từ" not in loai:          # nhãn lớp chân lý 80% là danh từ (§40)
            continue
        g = re.sub(r"\([^)]*\)", " ", g)   # bỏ chú thích «(khẩu ngữ)»
        g = g.split(":")[0]                # bỏ phần ví dụ sau dấu hai chấm
        ws = [w.lower() for w in TU.findall(g)][:6]
        if not ws:
            continue
        # LOÀI GẦN nằm ở VỊ TRÍ ĐẦU của khuôn «loài gần + khác biệt», KHÔNG
        # phải ở độ hiếm: bản đầu dùng `max(idf)` và ra `sắt → màu xám` (xám
        # hiếm hơn kim) — sai, vì từ hiếm nhất thường là phần KHÁC BIỆT.
        # IDF chỉ dùng để BỎ TỪ HẠN ĐỊNH (`một`, `các`, `những` — IDF thấp),
        # rồi lấy từ ĐẦU TIÊN còn lại. Ngưỡng lấy từ phân vị của chính corpus,
        # không phải hằng số viết tay.
        i = 0
        while i < len(ws) - 1 and idf(ws[i]) < nguong:
            i += 1
        dau = ws[i]
        # §131 — CỬA SỔ tới 4 TỪ. Bản §128 chỉ lấy 2 từ nên cắt cụt nhãn:
        # `thú gặm` (đúng: `thú gặm nhấm`), `cây lương` (`cây lương thực`),
        # `loài linh` (`loài linh trưởng`). Đo trên chỉ mục: nhịp dài tới 6 từ,
        # và nhịp 3 TỪ là PHỔ BIẾN NHẤT (28 240 / 79 601) — `thú gặm nhấm` và
        # `động vật có vú` đều CÓ nhịp sẵn. Tức nút không thiếu, bảng cắt cụt.
        # Xếp NGẮN trước để `[:3]` giữ nguyên cách chấm §128; bản dài nối sau
        # cho chỗ dùng cần khớp nhịp (§129 ưu tiên nhãn ghép).
        nhan = [dau]
        for _k in (2, 3, 4):
            if i + _k - 1 < len(ws):
                nhan.append(" ".join(ws[i:i + _k]))
        if i > 0:
            nhan.append(f"{ws[i - 1]} {dau}")
        bang.setdefault(tu, [])
        for x in nhan:
            if x not in bang[tu]:
                bang[tu].append(x)

    RA.parent.mkdir(parents=True, exist_ok=True)
    RA.write_text(json.dumps(bang, ensure_ascii=False, indent=0), "utf-8")
    print(f"### {len(bang)} từ có nhãn lớp → {RA}")
    for w in ("chó", "mèo", "hổ", "sóc", "sắt", "đồng", "lúa", "muối", "khỉ"):
        print(f"   {w:<6} → {bang.get(w)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
