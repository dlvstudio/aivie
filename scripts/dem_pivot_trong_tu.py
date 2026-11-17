"""Bao nhiêu lần PIVOT nằm LỌT TRONG một từ (span WS)?

§61.4: cạnh rác nặng nhất của toy 148 875 đến từ «bao GỒM» — `gồm` được tính
là pivot phân loại, và luật vế trái lấy `bao`, tức MẢNH CỦA CHÍNH CỤM PIVOT.

Dự án đã bỏ `thành` khỏi pivot vì đúng bệnh đó (`thành phố` · `hoàn thành`),
nhưng bỏ bằng DANH SÁCH TỪ. Nay đã có span WS-CRF nên kiểm được bằng CẤU TRÚC:
pivot nào nằm trong một từ ≥2 token thì không phải pivot.

Script này đếm trước khi vá — và in mẫu để đọc tay.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/dem_pivot_trong_tu.py \
        checkpoints/classrich.pkl checkpoints/classrich.ws_spans.json
"""
from __future__ import annotations

import gc
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

T0 = time.time()
PIV = {"là", "như", "gồm", "gọi"}


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classrich.pkl"
    sp = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/classrich.ws_spans.json"
    state, _ = load_checkpoint(ck)
    v, ps = state.vocab, state.parents
    S = [str(v.token_of(i)).replace(MARK_SPACE, "").strip() for i in range(len(v))]
    span = {tuple(int(x) for x in t) for t in (H.load_ws_spans(sp) or ())}
    dai = max((len(t) for t in span), default=1)
    print(f"\n### {ck} · {len(ps.parents)} cha · {len(span)} span\n")

    tong: Counter[str] = Counter()
    trong: Counter[str] = Counter()
    mau: dict[str, list[str]] = {}
    for rec in ps.parents:
        seq = [int(t) for t in rec.emit_ids]
        # đánh dấu vị trí nào bị một từ ≥2 token phủ
        phu = [None] * len(seq)
        i = 0
        while i < len(seq):
            hit = 0
            for L in range(min(dai, len(seq) - i), 1, -1):
                if tuple(seq[i:i + L]) in span:
                    hit = L
                    break
            if hit:
                tu = " ".join(S[t] for t in seq[i:i + hit])
                for k in range(i, i + hit):
                    phu[k] = tu
                i += hit
            else:
                i += 1
        for k, t in enumerate(seq):
            s = S[t]
            if s not in PIV:
                continue
            tong[s] += 1
            if phu[k] is not None:
                trong[s] += 1
                mau.setdefault(s, [])
                if len(mau[s]) < 6 and phu[k] not in mau[s]:
                    mau[s].append(phu[k])

    print(f"  {'pivot':>6}  {'tổng':>7}  {'LỌT TRONG TỪ':>13}  {'%':>6}  từ chứa nó")
    for s in sorted(tong, key=lambda x: -tong[x]):
        p = 100 * trong[s] / max(1, tong[s])
        print(f"  {s:>6}  {tong[s]:>7}  {trong[s]:>13}  {p:>5.1f}%  {mau.get(s, [])}")
    print(f"\n  TỔNG: {sum(trong.values())}/{sum(tong.values())} lần pivot"
          f" = {100*sum(trong.values())/max(1,sum(tong.values())):.1f}% là MẢNH TỪ")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
