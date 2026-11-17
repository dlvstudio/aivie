#!/usr/bin/env python
"""P91 (TK_045 §13.66) — **SỐ KHUNG CHUNG** có bão hoà như "≥2 thực thể" không?

§13.65 bác ràng buộc "≥2 thực thể ủng hộ" bằng đúng một dòng: **94,8%** ca thoả ⇒ bộ lọc
**không lọc gì**. Cái phân biệt được `khỉ` với `bàn` trên 7 câu là **sự khan hiếm** của corpus
đồ chơi, không phải bản chất tiêu chí.

Câu hỏi vòng này: **số KHUNG chung** có chịu chung số phận không?

    khỉ ~ chó : chung {nghịch ngợm, con, ăn, chạy, …}   ← nhiều khung ⇒ CÙNG LỚP
    bàn ~ chó : chung {màu nâu}                          ← một khung ⇒ TRÙNG HỢP

Nếu ở quy mô thật **mọi cặp** đều chung nhiều khung thì tiêu chí này cũng vô hiệu, và **cả
hướng tenxơ-tách-lát chết ngay tại đây** — biết sớm, không phải xây gì.

## Định nghĩa (thuần đếm)

**Khung** của một token `a` = tập token **kề trước** hoặc **kề sau** `a` trong `W_seq`, lọc
rác bằng `gidf` (§13.20: rác ở 4,0–4,2). Đây là "vai" thô nhất mà đồ thị hiện có: cùng đứng
cạnh một token nghĩa là cùng một chỗ trong câu.

**Số khung chung** của `(a, b)` = `|khung(a) ∩ khung(b)|`.

## Ba nhóm so sánh

| nhóm | định nghĩa |
|---|---|
| `cung_lop` | cặp **đồng hiện với ≥3 khung giống nhau** — xấp xỉ "bạn cùng lớp" |
| `dong_hien_1` | cặp chỉ đồng hiện trong **đúng một** cha — xấp xỉ "trùng hợp" |
| `ngau_nhien` | hai token nội dung bốc ngẫu nhiên — sàn |

⚠️ Nhóm `cung_lop` được định nghĩa **bằng chính đại lượng đang đo** ⇒ **tự quy chiếu**, không
được dùng làm bằng chứng. Nó chỉ để xem **trần** của thang đo. Bằng chứng thật nằm ở chỗ
`dong_hien_1` có **tách khỏi** `ngau_nhien` không — vì đó là hai nhóm định nghĩa **độc lập**
với thang đo.

## CỔNG G-frames (nêu TRƯỚC)

1. **≥300 cặp** mỗi nhóm.
2. `dong_hien_1` trung vị **≥ 2×** `ngau_nhien` — thang đo phải phân biệt được cặp có quan hệ
   thật với cặp vô can.
3. **KHÔNG bão hoà**: tỷ lệ cặp `ngau_nhien` có ≥2 khung chung phải **< 50%**. Vượt 50% thì
   tiêu chí lặp lại đúng số phận "≥2 thực thể" (94,8%) và **hướng này đóng**.

## Dự đoán ghi TRƯỚC (Claude)

- (3) là mục tôi lo nhất và **thật sự không chắc**. `W_seq` trên 5.664 cha khá dày; token phổ
  biến sẽ kề với rất nhiều thứ. Nếu bão hoà, nó bão hoà **vì cùng lý do** với §13.65.
- (2) **đạt** — cặp đồng hiện thật phải chung nhiều khung hơn cặp ngẫu nhiên.
- Nếu (3) trượt ⇒ ghi đóng: **mọi tiêu chí đếm-cái-chung đều bão hoà trên đồ thị đồng hiện
  dày**, và đó mới là lý do gốc khiến ba cách hiện thực trước đều thua `E`.

Chạy: `PYTHONPATH=src python scripts/probe_shared_frames.py`
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.adjacency import EdgeKind  # noqa: E402
from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_MIN = 300
GATE_VS_RANDOM = 2.0
GATE_SATURATE_MAX = 0.50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_shared_frames.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_shared_frames")
    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)

    keep = np.zeros(n, dtype=bool)
    for t in range(n):
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            if float(ps._gidf(t)) >= args.gidf_min:  # noqa: SLF001
                keep[t] = True
        except Exception:
            continue

    # --- KHUNG: token kề trước / kề sau trên W_seq ----------------------------------------
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    frames: dict[int, set[int]] = defaultdict(set)
    for i, j, w in seq.items():
        if w <= 0:
            continue
        i, j = int(i), int(j)
        if 0 <= i < n and 0 <= j < n and keep[i] and keep[j]:
            frames[i].add(j)      # j đứng SAU i
            frames[j].add(i)      # i đứng TRƯỚC j
    logger.log(f"token có khung: {len(frames):,} ({time.time()-t0:.0f}s)")

    # --- cặp đồng hiện trong ĐÚNG MỘT cha -------------------------------------------------
    pair_par: dict[tuple[int, int], int] = defaultdict(int)
    for rec in ps.parents:
        toks = sorted({int(t) for t in rec.emit_ids if 0 <= int(t) < n and keep[int(t)]})
        if len(toks) > 40:
            continue
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                pair_par[(toks[i], toks[j])] += 1

    rng = random.Random(args.seed)
    ids = [t for t in frames if len(frames[t]) >= 2]

    def shared(a: int, b: int) -> int:
        return len(frames.get(a, set()) & frames.get(b, set()))

    once = [p for p, c in pair_par.items() if c == 1]
    rng.shuffle(once)
    g_once = [shared(a, b) for a, b in once[: args.n_pairs]]

    g_rand: list[int] = []
    while len(g_rand) < args.n_pairs and ids:
        a, b = rng.choice(ids), rng.choice(ids)
        if a != b:
            g_rand.append(shared(a, b))

    # tự quy chiếu — chỉ để xem TRẦN, không dùng làm bằng chứng
    g_cls: list[int] = []
    tries = 0
    while len(g_cls) < args.n_pairs and tries < 200000 and ids:
        a, b = rng.choice(ids), rng.choice(ids)
        tries += 1
        if a != b and shared(a, b) >= 3:
            g_cls.append(shared(a, b))

    def summ(xs: list[int]) -> dict:
        if not xs:
            return {}
        s = sorted(xs)
        return {"n": len(s), "trung_vi": statistics.median(s),
                "p25": s[len(s) // 4], "p75": s[3 * len(s) // 4],
                "ty_le_ge2": sum(1 for x in s if x >= 2) / len(s)}

    R = {"dong_hien_1": summ(g_once), "ngau_nhien": summ(g_rand), "cung_lop": summ(g_cls)}
    print(f"\n{'nhóm':<14}{'n':>6}{'trung vị':>10}{'p75':>7}{'≥2 khung chung':>17}")
    for k in ("ngau_nhien", "dong_hien_1", "cung_lop"):
        r = R[k]
        if r:
            print(f"{k:<14}{r['n']:>6}{r['trung_vi']:>10.1f}{r['p75']:>7.0f}"
                  f"{r['ty_le_ge2']:>16.1%}")

    med_r = max(R["ngau_nhien"].get("trung_vi", 0), 1e-9)
    ratio = R["dong_hien_1"].get("trung_vi", 0) / med_r
    sat = R["ngau_nhien"].get("ty_le_ge2", 1.0)
    g1 = min(len(g_once), len(g_rand)) >= GATE_MIN
    g2 = ratio >= GATE_VS_RANDOM
    g3 = sat < GATE_SATURATE_MAX
    print(f"\ndong_hien_1/ngau_nhien (trung vị) = {ratio:.2f}× · "
          f"ngẫu nhiên có ≥2 khung chung: {sat:.1%}")
    print(f"CỔNG G-frames: (1) {'✅' if g1 else '❌'} · (2) {'✅' if g2 else '❌'} · "
          f"(3) {'✅' if g3 else '❌'} ⇒ {'ĐẠT' if (g1 and g2 and g3) else 'TRƯỢT'}")

    (_ROOT / args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "ket_qua": R, "ty_so": ratio, "bao_hoa": sat,
         "gate_pass": bool(g1 and g2 and g3), "seconds": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
