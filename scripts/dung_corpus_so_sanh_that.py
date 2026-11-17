"""TK_052 vòng 4 — corpus SO SÁNH lấy từ CÂU THẬT, không tổng hợp.

Vòng 3 chứng minh cơ chế làm được trên corpus tổng hợp (70,3%). Vòng 4 hỏi câu
cuối: **văn bản tiếng Việt THẬT có đủ khuôn đó để cơ chế chạy không?**

Đo được (`do_thanh_ngu_so_sanh.py`): 34 273 khuôn «tính từ như danh từ» trong
các nguồn đang có · 1 138 thuộc tính có ≥2 vật mang · **279 thuộc tính có ≥2
vật mang CÙNG MỘT LỚP** — đúng điều kiện cần của bước ②.

Script này rút CÂU THẬT chứa các so sánh đó, và dựng khoá (A, lớp, m3) trong
đó m3 là thành viên cùng lớp mà A CHƯA từng so sánh tới.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/dung_corpus_so_sanh_that.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

T0 = time.time()
NGUON = [
    "data/text_corpus/conv_stories.jsonl",
    "data/text_corpus/dictionary.jsonl",
    "data/text_corpus/paragraph_units.jsonl",
    "data/text_corpus/history_articles.jsonl",
    "data/text_corpus/subject_articles.jsonl",
    "data/text_corpus/doc_add_articles.jsonl",
    "data/text_corpus/viwiki.jsonl",
]
RX = re.compile(r"([A-Za-zÀ-ỹ]+) (?:như|tựa) ([A-Za-zÀ-ỹ]+)")


def main() -> int:
    mang = {a: set(b) for a, b in
            json.load(open("out/thanh_ngu_so_sanh.json")).items()}
    lop = json.load(open("out/chan_ly_v2_classrich2.json"))["lop"]
    st, _ = load_checkpoint("checkpoints/classrich2.pkl")
    v = st.vocab
    tv2lop: dict[str, set[str]] = defaultdict(set)
    for c, ms in lop.items():
        for m in ms:
            tv2lop[m.lower()].add(c)

    # KHOÁ: A có ≥2 vật mang cùng lớp C, và C còn thành viên m3 mà A chưa chạm
    khoa = []
    for a, b in mang.items():
        if len(b) < 2:
            continue
        dem: dict[str, list[str]] = defaultdict(list)
        for x in b:
            for c in tv2lop.get(x, ()):
                dem[c].append(x)
        for c, vm in dem.items():
            if len(vm) < 2:
                continue
            con = [m for m in lop[c]
                   if m.lower() not in b and H.resolve_surface_ids(v, m)]
            if not con:
                continue
            khoa.append({"attr": a, "lop": c, "mang": sorted(vm)[:4],
                         "m3": con[0]})
            break
    print(f"\n  {len(khoa)} bộ dựng được từ CÂU THẬT")

    # rút CÂU THẬT chứa so sánh của các thuộc tính đó
    can = {k["attr"] for k in khoa}
    cau: list[str] = []
    for p in NGUON:
        q = Path(p)
        if not q.exists():
            continue
        with q.open(encoding="utf-8") as f:
            for dong in f:
                try:
                    doc = json.loads(dong)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(doc, dict):
                    doc = doc.get("text") or doc.get("content") or ""
                for para in str(doc).split("\n"):
                    for s in para.split(". "):
                        s = s.strip()
                        if not (15 < len(s) < 250):
                            continue
                        m = RX.search(s)
                        if m and m.group(1).lower() in can:
                            cau.append(s if s.endswith(".") else s + " .")
    # mỗi dòng GỘP hai câu để pipeline dựng cha câu (bài học vòng 1)
    ra = [f"{cau[i]} {cau[i + 1] if i + 1 < len(cau) else ''}".strip()
          for i in range(0, len(cau), 2)]
    Path("data/text_corpus/so_sanh_that.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ra),
        encoding="utf-8")
    Path("out/so_sanh_that_khoa.json").write_text(
        json.dumps(khoa, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {len(cau)} câu thật → {len(ra)} dòng "
          f"→ data/text_corpus/so_sanh_that.jsonl")
    print("  3 bộ đầu:")
    for k in khoa[:3]:
        print(f"     A={k['attr']!r} lớp={k['lop']!r} mang={k['mang']} "
              f"GIỮ LẠI={k['m3']!r}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
