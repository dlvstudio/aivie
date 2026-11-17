"""TK_052 vòng 4 — dữ liệu THẬT có khuôn SO SÁNH ở quy mô không?

Vòng 3 chứng minh cơ chế làm được (thu hồi 70,3%, đối chứng 0,0%) trên corpus
TỔNG HỢP. Điều kiện nêu ra: cần văn bản viết theo lối SO SÁNH («nhanh như cắt»
· «đen như mực» · «hiền như bụt»), thứ mà văn xuôi bách khoa không có.

Trước khi dựng corpus, đếm xem các nguồn ĐANG CÓ chứa bao nhiêu khuôn đó, và
— quan trọng hơn — bao nhiêu THUỘC TÍNH có ≥2 vật mang (điều kiện cần của ②,
đo được 0,0% trên bách khoa ở §66).

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/do_thanh_ngu_so_sanh.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

T0 = time.time()
NGUON = [
    "data/text_corpus/conv_stories.jsonl",
    "data/text_corpus/dictionary.jsonl",
    "data/text_corpus/paragraph_units.jsonl",
    "data/text_corpus/history_articles.jsonl",
    "data/text_corpus/subject_articles.jsonl",
    "data/text_corpus/grade_lessons.jsonl",
    "data/text_corpus/doc_add_articles.jsonl",
    "data/text_corpus/viwiki.jsonl",
]
# «<tính từ> như <danh từ>» — bắt cả cụm hai từ ở mỗi vế
RX = re.compile(r"([A-Za-zÀ-ỹ]+(?: [A-Za-zÀ-ỹ]+)?) (?:như|tựa|giống như) "
                r"([A-Za-zÀ-ỹ]+(?: [A-Za-zÀ-ỹ]+)?)")


def main() -> int:
    pm = H.load_pos_map("data/dict/vi_dict_raw.jsonl")
    pm = {k.replace(MARK_SPACE, "").strip().lower(): x for k, x in pm.items()}
    tt = {w for w, p in pm.items() if "tính từ" in p}
    dt = {w for w, p in pm.items() if "danh từ" in p}
    print(f"\n  từ điển: {len(tt)} tính từ · {len(dt)} danh từ\n")

    mang: dict[str, set[str]] = defaultdict(set)
    dem_nguon: Counter[str] = Counter()
    n_dong = 0
    for p in NGUON:
        q = Path(p)
        if not q.exists():
            continue
        with q.open(encoding="utf-8") as f:
            for dong in f:
                n_dong += 1
                try:
                    doc = json.loads(dong)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(doc, dict):
                    doc = doc.get("text") or doc.get("content") or ""
                for m in RX.finditer(str(doc)):
                    a, b = m.group(1).strip().lower(), m.group(2).strip().lower()
                    a1, b1 = a.split()[-1], b.split()[0]
                    # vế trái là TÍNH TỪ, vế phải là DANH TỪ (đầu ngữ đoạn)
                    if (a in tt or a1 in tt) and (b in dt or b1 in dt):
                        mang[a if a in tt else a1].add(b if b in dt else b1)
                        dem_nguon[q.name] += 1
    print(f"  quét {n_dong} dòng · khuôn «tính từ như danh từ»:")
    for k, n in dem_nguon.most_common():
        print(f"     {k:<28} {n:>7}")
    hai = {a: b for a, b in mang.items() if len(b) >= 2}
    ba = {a: b for a, b in mang.items() if len(b) >= 3}
    print(f"\n  thuộc tính có ≥1 vật mang: {len(mang)}")
    print(f"  thuộc tính có ≥2 vật mang: {len(hai)}"
          f"  ({100*len(hai)/max(1,len(mang)):.1f}%)   ← điều kiện cần của ②")
    print(f"  thuộc tính có ≥3 vật mang: {len(ba)}")
    print("\n  15 thuộc tính nhiều vật mang nhất:")
    for a, b in sorted(mang.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"     {a:<14} {len(b):>4} : {sorted(b)[:7]}")
    Path("out/thanh_ngu_so_sanh.json").write_text(
        json.dumps({a: sorted(b) for a, b in mang.items()},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  → out/thanh_ngu_so_sanh.json")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
