"""Trên dữ liệu thật có bao nhiêu LỚP THẬT để ca «con khỉ» chạy?

Ca thử vàng cần một lớp PHÂN LOẠI (`động vật` — chó/mèo/khỉ cùng là thành viên,
và thành viên chia nhau thuộc tính). Manifest `train_classrich.yaml` đã cảnh
báo câu phân loại CHUNG CHUNG nhiều gấp ~9 lần câu CỤ THỂ. Script này đếm trên
chính checkpoint đang đo, và tách hai loại bằng một thước khách quan:

  • lớp CỤ THỂ  — thành viên là danh từ chung có trong từ điển (chó · mèo · khỉ)
  • lớp CHUNG   — thành viên là tên riêng / số / mảnh, hoặc head là `khoảng`,
                  `một trong`, `nơi`, `điều`… (không phân loại được gì)

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/dem_lop_that_vs_chung_chung.py
"""
from __future__ import annotations

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

_DET = frozenset({"một", "các", "những", "mọi", "mỗi"})
T0 = time.time()


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classrich.pkl"
    state, _ = load_checkpoint(ck)
    v, ps = state.vocab, state.parents
    n = len(v)
    raw = [str(v.token_of(i)) for i in range(n)]
    S = [r.replace(MARK_SPACE, "").strip() for r in raw]
    FUNC = {i for i in range(n)
            if S[i].lower() in H._FUNC or len(S[i]) <= 1
            or MARK_SPACE not in raw[i] or S[i].startswith("<")}
    LA = {i for i in range(n) if S[i] == "là"}
    DET = {i for i in range(n) if S[i].lower() in _DET}

    pm = H.load_pos_map("data/dict/vi_dict_raw.jsonl")
    tu = {k.replace(MARK_SPACE, "").strip() for k in pm}
    print(f"\n### {ck} · {len(ps.parents)} cha · từ điển {len(tu)} từ\n")

    cls: dict[str, set[str]] = defaultdict(set)
    for rec in ps.parents:
        seq = list(rec.emit_ids)
        for k, t in enumerate(seq):
            if t not in LA or k == 0 or k + 1 >= len(seq):
                continue
            mem = seq[k - 1]
            if mem in FUNC or mem in DET:
                continue
            j = k + 1
            while j < len(seq) and seq[j] in DET:
                j += 1
            if j >= len(seq) or seq[j] in FUNC:
                continue
            head = S[seq[j]]
            if j + 1 < len(seq) and seq[j + 1] not in FUNC:
                head += " " + S[seq[j + 1]]
            cls[head].add(S[mem])

    big = {c: m for c, m in cls.items() if len(m) >= 3}
    print(f"lớp có ≥3 thành viên: {len(big)}")

    def la_tu_thuong(s: str) -> bool:
        """danh từ chung có trong từ điển, không phải tên riêng/số/mảnh."""
        return (s in tu and s[:1].islower() and s.isalpha())

    ro, mo = {}, {}
    for c, m in big.items():
        tv = sum(1 for x in m if la_tu_thuong(x))
        (ro if (tv >= 3 and tv / len(m) >= 0.5) else mo)[c] = (tv, len(m))

    print(f"  ⤷ lớp CỤ THỂ (≥3 thành viên là danh từ chung, ≥50%): "
          f"**{len(ro)}** = {100*len(ro)/max(1,len(big)):.1f}%")
    print(f"  ⤷ lớp CHUNG CHUNG / tên riêng                     : {len(mo)}\n")

    print("  10 lớp LỚN NHẤT (mọi loại) — thứ cơ chế hay gặp nhất:")
    for c, m in sorted(big.items(), key=lambda kv: -len(kv[1]))[:10]:
        tv = ro.get(c, mo.get(c))[0]
        loai = "CỤ THỂ" if c in ro else "chung "
        print(f"     {loai} {c:<16} {len(m):>4} thành viên (danh từ chung {tv})"
              f"  vd {sorted(m)[:4]}")

    print("\n  10 lớp CỤ THỂ lớn nhất — thứ ca «con khỉ» chạy được:")
    for c, (tv, tot) in sorted(ro.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"     {c:<16} {tot:>4} thành viên  vd "
              f"{[x for x in sorted(big[c]) if la_tu_thuong(x)][:5]}")

    for ten in ("loài", "động vật", "kim loại"):
        m = cls.get(ten, set())
        print(f"\n  `{ten}`: {len(m)} thành viên  {sorted(m)[:8]}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
