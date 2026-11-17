#!/usr/bin/env python3
"""ĐO thông tin theo KHOẢNG CÁCH và theo CHIỀU — TK_045 §12.80. **Chỉ đo.**

    PYTHONPATH=src python -u scripts/probe_distance_info.py \
        --checkpoint checkpoints/mini_sent_ep_qa.pkl \
        --log-file out/dist_info_mini.log

## Vì sao

Tác giả đề xuất: dựng ma trận (N×N) **có hướng, có giảm theo khoảng cách** rồi PPMI+SVD ra
embedding — đường ống ấy đã có sẵn (`ensure_directed_ppmi` → `ensure_directed_qk`), chỉ đang được
nuôi bằng `W_seq` (**d=1**). Còn `E` hiện tại chiết từ `CO_ACTIVATION`, vốn CÓ giảm theo khoảng
cách (cửa sổ trượt bước 1 ⇒ trọng số ∝ (W−d)) nhưng **ĐỐI XỨNG** — nên `E` về cấu trúc không thể
mã hoá "a trước b", và `cos(zone,fact)` vì thế chỉ đo được "nói về cùng thực thể".

Chỗ trống là góc **xa VÀ có hướng**. Trước khi dựng, đo xem góc ấy có gì không.

## Hai câu hỏi, hai cột

**(1) HƯỚNG có thêm thông tin không?** Với mỗi cặp không thứ tự {a,b} ở khoảng cách d, tính
lệch chiều `|c_ab − c_ba| / (c_ab + c_ba)`. Gần 0 = hai chiều ngang nhau ⇒ hướng vô nghĩa.

**BẮT BUỘC có đối chứng.** Với đếm nhỏ, lệch chiều xuất hiện do NGẪU NHIÊN: cặp tổng n chia đôi
theo nhị thức có kỳ vọng lệch ≈ √(2/πn) — n=4 đã cho 0,40. Nên so với **cùng dữ liệu đã xáo trật
tự trong câu**: phần vượt lên trên đối chứng mới là hướng THẬT.

**(2) TẦM xa tới đâu?** PMI trung bình ở mỗi d. Rơi về 0 nhanh ⇒ không có phụ thuộc dài để khai
thác, và `d_max` lớn chỉ thêm nhiễu.

## Cổng — CHỐT TRƯỚC KHI CHẠY

  G1 tỉnh táo   lệch chiều VƯỢT đối chứng ≥ 0,10 ở d=1. Không đạt ⇒ phép đo hỏng (kề nhau thì
                bắt buộc phải có hướng: "dời đô" ≫ "đô dời").
  G2 hướng có TẦM  phần vượt đối chứng phải còn ≥ 0,05 tới d ≥ 8. Không đạt ⇒ hướng chỉ là hiện
                tượng của d=1, mà `W_seq` đã bắt trọn ⇒ **ma trận mới không thêm gì**.
  G3 tầm xa     PMI ở d=8 phải còn ≥ 25% PMI ở d=1. Không đạt ⇒ đặt `d_max` lớn là vô ích.

G2 là cổng quyết định. Trượt G2 ⇒ chốt âm cho cả hướng "ma trận có hướng tầm xa", và tiết kiệm
được việc dựng/nuôi một ma trận (N×N) cỡ v12.
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--log-file", default="out/dist_info.log")
    ap.add_argument("--d-max", type=int, default=24)
    ap.add_argument("--min-count", type=int, default=4,
                    help="cặp có tổng đếm < ngưỡng thì bỏ (lệch chiều của chúng gần như thuần "
                         "nhiễu; đối chứng vẫn giữ để so công bằng)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logf = (_ROOT / args.log_file).open("w", encoding="utf-8")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)
    log(f"[{time.time()-t0:.0f}s] ✓ {len(ps):,} cha")

    struct = [is_structural_token(v, i) for i in range(n)]
    rng = random.Random(args.seed)

    def collect(shuffle: bool):
        """cnt[d][(a,b)] và unigram — `shuffle` xáo trật tự TRONG câu (đối chứng)."""
        cnt = [defaultdict(int) for _ in range(args.d_max + 1)]
        uni: dict[int, int] = defaultdict(int)
        tot = 0
        for rec in ps.parents:
            seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
            for span in iter_sentence_spans(seq, v):
                toks = [int(t) for t in _content_positions(
                    span, v, content_only=True, struct_mask=struct)]
                if len(toks) < 2:
                    continue
                if shuffle:
                    toks = toks[:]
                    rng.shuffle(toks)
                for t in toks:
                    uni[t] += 1
                    tot += 1
                L = len(toks)
                for i in range(L):
                    for j in range(i + 1, min(L, i + 1 + args.d_max)):
                        cnt[j - i][(toks[i], toks[j])] += 1
        return cnt, uni, max(1, tot)

    def stats(cnt, uni, tot):
        """(lệch chiều có trọng số, PMI trung bình có trọng số, số cặp) cho từng d."""
        out = []
        for d in range(1, args.d_max + 1):
            c = cnt[d]
            nd = sum(c.values())
            if nd == 0:
                out.append((float("nan"), float("nan"), 0))
                continue
            seen = set()
            num = den = 0.0
            pmi_w = pmi_n = 0.0
            for (a, b), cab in c.items():
                key = (a, b) if a <= b else (b, a)
                if key not in seen:
                    seen.add(key)
                    cba = c.get((b, a), 0)
                    s = cab + cba
                    if s >= args.min_count:
                        num += abs(cab - cba)
                        den += s
                if cab >= args.min_count:
                    p = cab / nd
                    q = (uni[a] / tot) * (uni[b] / tot)
                    if q > 0 and p > 0:
                        pmi_w += cab * math.log(p / q)
                        pmi_n += cab
            out.append((num / den if den > 0 else float("nan"),
                        pmi_w / pmi_n if pmi_n > 0 else float("nan"),
                        len(c)))
        return out

    log(f"[{time.time()-t0:.0f}s] đếm THẬT …")
    real = stats(*collect(False))
    log(f"[{time.time()-t0:.0f}s] đếm ĐỐI CHỨNG (xáo trật tự trong câu) …")
    ctrl = stats(*collect(True))
    log(f"[{time.time()-t0:.0f}s] ✓")

    log("\n" + "=" * 78)
    log(f"{'d':>3}{'lệch chiều':>12}{'đối chứng':>11}{'VƯỢT':>9}{'PMI':>9}{'%PMI(d1)':>10}"
        f"{'#cặp':>10}")
    log("=" * 78)
    pmi1 = real[0][1] if real and real[0][1] == real[0][1] else float("nan")
    excess = []
    for d in range(1, args.d_max + 1):
        sk, pmi, npair = real[d - 1]
        sk0 = ctrl[d - 1][0]
        ex = sk - sk0 if (sk == sk and sk0 == sk0) else float("nan")
        excess.append(ex)
        rel = (pmi / pmi1 * 100.0) if (pmi == pmi and pmi1 == pmi1 and pmi1 != 0) else float("nan")
        log(f"{d:>3}{sk:>12.3f}{sk0:>11.3f}{ex:>9.3f}{pmi:>9.3f}{rel:>9.1f}%{npair:>10,}")
    log("=" * 78)

    g1 = excess[0] == excess[0] and excess[0] >= 0.10
    d8 = excess[7] if len(excess) > 7 else float("nan")
    g2 = d8 == d8 and d8 >= 0.05
    p8 = real[7][1] if len(real) > 7 else float("nan")
    g3 = (p8 == p8 and pmi1 == pmi1 and pmi1 != 0 and p8 / pmi1 >= 0.25)
    log(f"G1 tỉnh táo (vượt ≥0.10 ở d=1) : {'ĐẠT' if g1 else 'TRƯỢT'} ({excess[0]:+.3f})")
    log(f"G2 hướng CÓ TẦM (vượt ≥0.05 ở d=8): {'ĐẠT' if g2 else 'TRƯỢT'} ({d8:+.3f})  ← cổng quyết định")
    log(f"G3 PMI ở d=8 ≥25% của d=1      : {'ĐẠT' if g3 else 'TRƯỢT'} "
        f"({(p8/pmi1*100 if pmi1 else float('nan')):.1f}%)")
    verdict = ("ĐÁNG dựng ma trận có hướng tầm xa" if (g1 and g2)
               else "KHÔNG đáng dựng — hướng chỉ là hiện tượng của d nhỏ, `W_seq` đã bắt trọn")
    log(f"\n⇒ {verdict}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
