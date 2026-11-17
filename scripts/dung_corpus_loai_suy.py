"""Dựng corpus CÓ CHỦ ĐÍCH đúng cấu trúc ca vàng — TK_052.

Ca thử vàng có 5 câu, nên nó chỉ cho thấy cơ chế chạy ở n=1. Corpus này dựng
HÀNG TRĂM bộ cùng cấu trúc, từ lớp THẬT của chân lý v2 và tính từ THẬT của từ
điển, để phân định H1 (cơ chế sai) khỏi H2 (dữ liệu tự nhiên không có hình
dạng đó).

Mỗi bộ:

    <A> như loài <m1> .        ← thuộc tính A chạm m1 và m2
    <A> như loài <m2> .
    <m1> là <C> .  <m2> là <C> .  <m3> là <C> .
    m3 KHÔNG BAO GIỜ đứng cùng A          ← «con khỉ» của bộ đó

Danh sách (A, C, m3) ghi ra JSON để chấm.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/dung_corpus_loai_suy.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

T0 = time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chan-ly", default="out/chan_ly_v2_classrich2.json")
    ap.add_argument("--out", default="data/text_corpus/loai_suy.jsonl")
    ap.add_argument("--khoa", default="out/loai_suy_khoa.json")
    ap.add_argument("--vocab-tu", default="checkpoints/classrich2.pkl",
                    help="ckpt lấy TỪ VỰNG — thuộc tính/thành viên phải TỒN "
                         "TẠI trong đó, nếu không BPE băm thành mảnh")
    ap.add_argument("--nhac-lai", type=int, default=3,
                    help="mỗi câu lặp mấy lần (mật độ so với nhiễu)")
    a = ap.parse_args()
    rnd = random.Random(0)

    # ⚠️ TỪ VỰNG LÀ RÀNG BUỘC CỨNG. Bản đầu chọn tính từ hiếm từ từ điển và
    # 95% trong số đó có tần suất 0 trong corpus — BPE băm chúng thành mảnh,
    # `resolve_surface_ids` trả rỗng, và KHÔNG cạnh nào dựng được. Phép thử đo
    # ra 6,0% và tôi suýt kết luận «cơ chế sai» từ một corpus mà cơ chế không
    # đọc nổi.
    from nn_aivie.core.checkpoint_io import load_checkpoint as _lc
    _st, _ = _lc(a.vocab_tu)
    _v = _st.vocab
    def _co(w: str) -> bool:
        return bool(H.resolve_surface_ids(_v, w))

    lop = json.load(open(a.chan_ly))["lop"]
    pm = H.load_pos_map("data/dict/vi_dict_raw.jsonl")
    pm = {k.replace(MARK_SPACE, "").strip().lower(): x for k, x in pm.items()}
    # THUỘC TÍNH = tính từ THẬT của từ điển, một từ, không dấu cách, đủ dài để
    # hiếm. Mỗi lớp lấy MỘT thuộc tính RIÊNG — đúng cấu trúc `nghịch ngợm`:
    # thuộc tính chỉ chạm vài vật mang, và chúng cùng lớp.
    tt = sorted({w for w, p in pm.items()
                 if "tính từ" in p and " " not in w and 4 <= len(w) <= 12
                 and w.isalpha() and w.islower() and _co(w)})
    rnd.shuffle(tt)

    cau: list[str] = []
    khoa: list[dict] = []
    i_tt = 0
    for c, ms in sorted(lop.items(), key=lambda kv: -len(kv[1])):
        # THÀNH VIÊN phải là DANH TỪ THƯỜNG của từ điển. Không siết thì lớp
        # `loại` cho [ATM, Bia, Biên bản] và phép thử về CƠ CHẾ bị trộn lẫn
        # với rác dữ liệu — đúng bệnh đã mắc ở §61 (sai tầng đơn vị).
        ms = [m for m in ms
              if 1 <= len(m.split()) <= 2 and m.replace(" ", "").isalpha()
              and m[:1].islower() and "danh từ" in pm.get(m.lower(), "")
              and _co(m)]
        if len(ms) < 3 or i_tt >= len(tt):
            continue
        attr = tt[i_tt]
        i_tt += 1
        m1, m2, m3 = ms[0], ms[1], ms[2]
        # ⚠️ MỘT DÒNG = MỘT ĐOẠN NHIỀU CÂU. Pipeline chỉ dựng CHA CÂU khi dòng
        # tách được ≥2 câu (`if len(sents) >= 2`); corpus bản đầu mỗi dòng một
        # câu ngắn nên 129/145 bộ BỊ BỎ NGAY KHI TRAIN — phép thử đo ra 6,0%
        # và tôi suýt kết luận «cơ chế sai» từ dữ liệu chưa từng vào đồ thị.
        for _ in range(a.nhac_lai):
            cau.append(f"{attr} như loài {m1} . {attr} như loài {m2} . "
                       f"loài {m1} là {c} . loài {m2} là {c} . "
                       f"loài {m3} là {c} .")
        khoa.append({"attr": attr, "lop": c, "m1": m1, "m2": m2, "m3": m3})

    Path(a.out).write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cau),
        encoding="utf-8")
    Path(a.khoa).write_text(json.dumps(khoa, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"\n  {len(khoa)} bộ · {len(cau)} câu → {a.out}")
    print(f"  khoá → {a.khoa}")
    print(f"  3 bộ đầu:")
    for k in khoa[:3]:
        print(f"     A={k['attr']!r} lớp={k['lop']!r} "
              f"chạm=[{k['m1']}, {k['m2']}] · GIỮ LẠI={k['m3']!r}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
