"""Dựng corpus GIÀU CÂU ĐỊNH NGHĨA — trả lời nút thắt §61.3/§64.

Đo được (§65):

    nguồn                     cặp (tv → lớp)   lớp ≥3 tv   thành viên là gì
    classrich hiện nay                   949         168   hỗn hợp
    viwiki, khuôn định nghĩa         183 690       1 905   84% tên Latin (stub)
    viwiki, phần dùng được            15 591         513   phần lớn tên xã/phường
    từ điển, giải nghĩa                9 178         643   TỪ THƯỜNG (chó·mèo·khỉ)

Từ điển là nguồn duy nhất mà thành viên là từ đời thường — đúng thứ đường sinh
cần — nhưng lời giải nghĩa KHÔNG có chữ `là` nên `via_pivot` không thấy nó.
Script này viết lại thành câu định nghĩa thật, và gộp với câu định nghĩa lọc
từ viwiki.

    PYTHONHASHSEED=0 python scripts/dung_corpus_dinh_nghia.py \
        --out data/text_corpus/dinh_nghia.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

T0 = time.time()
_TU = re.compile(r"^Từ:\s*(.+?)\s*$", re.M)
_LOAI = re.compile(r"^Loại từ:\s*(.+?)\s*$", re.M)
_NGHIA = re.compile(r"^Giải nghĩa:\s*(.+?)\s*$", re.M)
_VIDU = re.compile(r"^Ví dụ:\s*(.+?)\s*$", re.M)
_DAN = re.compile(r"^\s*\([^)]{1,30}\)\s*")
_DUNG = r"[^,.;:()\[\]«»\"]{1,60}"
_KHUON = [
    re.compile(rf"^{_DUNG}? là một loài {_DUNG}"),
    re.compile(rf"^{_DUNG}? là loài {_DUNG}"),
    re.compile(rf"^{_DUNG}? là một loại {_DUNG}"),
    re.compile(rf"^{_DUNG}? .{{0,40}}?thuộc (?:họ|chi|bộ|lớp) {_DUNG}"),
]
# Bài stub sinh tự động: chủ ngữ là TÊN KHOA HỌC Latin. Giữ chúng thì corpus
# phình bằng thứ không bao giờ xuất hiện trong văn xuôi (§65).
_LATIN = re.compile(r"^[A-Z][a-z]+ [a-z]{3,}\b")
_DAU = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
_TU_AZ = re.compile(r"[A-Za-zÀ-ỹ]{2,}")


def _thuan_viet(cau: str, tu_dien: set[str]) -> bool:
    """Câu có phải TIẾNG VIỆT không — không lẫn tên khoa học.

    §65.3: lọc chủ ngữ Latin CHƯA ĐỦ. «… là một loài bướm đêm thuộc họ
    Crambidae» có chủ ngữ tiếng Việt nhưng vẫn bơm `Cram b idae` vào từ vựng,
    và BPE băm nó thành mảnh — ③ trả về `ensis`/`idae`/`ates` thay vì con vật.
    Giữ câu chỉ khi ĐA SỐ từ của nó có dấu tiếng Việt hoặc nằm trong từ điển.
    """
    ws = _TU_AZ.findall(cau)
    if not ws:
        return False
    # ⚠️ NGƯỠNG TỈ LỆ KHÔNG ĂN THUA: «… là một loài bướm đêm thuộc họ
    # Crambidae» có 10/11 từ tiếng Việt ⇒ qua mọi ngưỡng, mà vẫn bơm
    # `Cram b idae` vào từ vựng. Phải chặn TUYỆT ĐỐI: một từ KHÔNG DẤU ≥5 ký
    # tự và không có trong từ điển thì cả câu bị bỏ.
    for w in ws:
        wl = w.lower()
        if any(c in _DAU for c in wl) or wl in tu_dien:
            continue
        if len(wl) >= 5:
            return False
    return True


def _cau_tu_dien(t: str) -> list[str]:
    """Mỗi mục từ điển → câu «X là <giải nghĩa>.» + câu ví dụ."""
    mt, ml, mn = _TU.search(t), _LOAI.search(t), _NGHIA.search(t)
    if not (mt and ml and mn) or "danh từ" not in ml.group(1):
        return []
    tu = mt.group(1).strip()
    nghia = _DAN.sub("", mn.group(1).strip())
    nghia = nghia.split(":")[0].strip()          # bỏ phần dẫn ví dụ sau dấu :
    if not tu or len(nghia) < 6 or len(nghia) > 200:
        return []
    ra = [f"{tu} là {nghia} ."]
    mv = _VIDU.search(t)
    if mv:
        vd = mv.group(1).strip()
        if 10 < len(vd) < 200:
            ra.append(vd if vd.endswith((".", "!", "?")) else vd + " .")
    return ra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/text_corpus/dinh_nghia.jsonl")
    ap.add_argument("--max-wiki", type=int, default=60000,
                    help="trần số câu lấy từ viwiki (giữ corpus train được)")
    a = ap.parse_args()

    tu_dien: set[str] = set()
    with open("data/dict/vi_dict_raw.jsonl", encoding="utf-8") as f:
        for dong in f:
            try:
                w = (json.loads(dong).get("word") or "").strip().lower()
            except Exception:  # noqa: BLE001
                continue
            if w:
                tu_dien.add(w)
    ra: list[str] = []
    n_td = 0
    with open("data/text_corpus/dictionary.jsonl", encoding="utf-8") as f:
        for dong in f:
            try:
                t = str(json.loads(dong))
            except Exception:  # noqa: BLE001
                continue
            cs = _cau_tu_dien(t)
            ra.extend(cs)
            n_td += len(cs)
    print(f"  từ điển  → {n_td} câu")

    n_wiki = n_bo_latin = 0
    with open("data/text_corpus/viwiki.jsonl", encoding="utf-8") as f:
        for dong in f:
            if n_wiki >= a.max_wiki:
                break
            try:
                doc = json.loads(dong)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(doc, dict):
                doc = doc.get("text") or doc.get("content") or ""
            for para in str(doc).split("\n"):
                cau = para.strip().split(". ")[0].strip()
                if not (20 < len(cau) < 300):
                    continue
                if not any(rx.match(cau) for rx in _KHUON):
                    continue
                # chủ ngữ ĐẠI TỪ («Đây là một loài rắn…») không cho thành viên
                if re.match(r"^(Đây|Nó|Chúng|Loài này|Cả hai)\b", cau):
                    n_bo_latin += 1
                    continue
                if _LATIN.match(cau) or not _thuan_viet(cau, tu_dien):
                    n_bo_latin += 1
                    continue
                ra.append(cau if cau.endswith(".") else cau + " .")
                n_wiki += 1
                if n_wiki >= a.max_wiki:
                    break
    print(f"  viwiki   → {n_wiki} câu  (bỏ {n_bo_latin} câu chủ ngữ TÊN LATIN)")

    p = Path(a.out)
    with p.open("w", encoding="utf-8") as f:
        for c in ra:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n  ✓ {p} · {len(ra)} câu · {p.stat().st_size/1e6:.1f} MB")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
