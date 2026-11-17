#!/usr/bin/env python3
"""W_skip có phải KÊNH TRÔI CHẢY tốt hơn `W_seq` không — TK_045 §12.87. **Chỉ đo.**

    PYTHONPATH=src python -u scripts/probe_next_token.py \
        --checkpoint checkpoints/mini_sent_ep_qa.pkl \
        --log-file out/next_token_mini.log

## Vì sao phép đo này khác §12.84/§12.85

P19 hỏi: *"giao top-k của các token prompt có chứa gold không"* — đó là câu hỏi **CHỌN NỘI DUNG**
(kiến thức). W_skip trả lời tệ (gold∈∩ 0,307 ‖ d=1 0,639) vì đếm thô ⇒ hub chiếm chỗ giao.

Tác giả nhắm việc khác: **sinh mượt, tự nhiên**. Đó là câu hỏi **DỰ ĐOÁN TOKEN KẾ**, và nó chưa
ai đo. Hai việc này khác nhau tận gốc — một kênh có thể tồi ở việc chọn *nói gì* mà tốt ở việc
giữ *câu trôi chảy*. `W_seq` chính là ví dụ: nó là kênh trôi chảy, không ai dùng nó để chọn fact.

Và `W_seq` **là order-1** — `varlm.py` ghi: *"nhìn 1 token trước → mất trí nhớ ngữ cảnh sau mỗi
từ → văn rời"*. W_skip đúng là bản mở rộng order-D của nó. Nên câu hỏi tự nhiên: **mở rộng ấy có
dự đoán token kế tốt hơn không?**

## Ba khác biệt so với bản P19

1. **PPMI thay vì đếm/chuẩn-hoá-hàng.** `row_score_map` cho `P(b|a,d)` — vẫn ưu ái hub vì hub có
   `P(b)` biên lớn. PPMI chia thêm cho `P(b)` ⇒ giết hub. Đây đúng thứ M4 chẩn ra là nguyên nhân.
2. **Cộng MỀM trên toàn cửa sổ**, không top-k, không giao, không hard-gate. Hard-gate trên tập
   giao bẩn chính là thứ đẩy sinh vào hub (§12.85e2).
3. **Dùng cả cửa sổ** (`d_max`), không phải `ctx_max=2`.

## Chống rò

Đếm trên nửa A, chấm trên nửa B — tách theo CÂU, như §12.83.

## Cổng — CHỐT TRƯỚC KHI CHẠY

  G1 tỉnh táo   `d1` top-10 ≥ 0,20. Thấp hơn ⇒ phép đo hỏng (bigram phải đoán được chút gì).
  G2 QUYẾT ĐỊNH  `skip_ppmi` HOẶC `d1+skip` phải hơn `d1` **≥ +0,05 top-10**.
  G3 chẩn đoán   `skip_ppmi` phải hơn `skip_raw` — xác nhận hub đúng là nguyên nhân của P19.

G2 trượt ⇒ W_skip không phải kênh trôi chảy tốt hơn, và cả hướng "tận dụng W_skip cho sinh" đóng.
G2 đạt ⇒ đáng nối vào `matrix_core` như kênh MỀM (không hard-gate) rồi A/B sinh.
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
    ap.add_argument("--log-file", default="out/next_token.log")
    ap.add_argument("--d-max", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.85)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=6000)
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
    sents_a: list[list[int]] = []
    sents_b: list[list[int]] = []
    k = 0
    for rec in ps.parents:
        seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        for span in iter_sentence_spans(seq, v):
            toks = [int(t) for t in _content_positions(
                span, v, content_only=True, struct_mask=struct)]
            if len(toks) >= 3:
                (sents_a if (k & 1) == 0 else sents_b).append(toks)
                k += 1
    log(f"  câu: nửa A {len(sents_a):,} · nửa B {len(sents_b):,}")

    # ---- đếm trên nửa A -----------------------------------------------------------------------
    cnt = [defaultdict(lambda: defaultdict(float)) for _ in range(args.d_max + 1)]
    nd = [0.0] * (args.d_max + 1)
    uni: dict[int, float] = defaultdict(float)
    utot = 0.0
    for toks in sents_a:
        L = len(toks)
        for t in toks:
            uni[t] += 1.0
            utot += 1.0
        for i in range(L):
            for j in range(i + 1, min(L, i + 1 + args.d_max)):
                d = j - i
                cnt[d][toks[i]][toks[j]] += 1.0
                nd[d] += 1.0
    log(f"[{time.time()-t0:.0f}s] ✓ đếm A")

    # ---- ba bảng điểm: P(b|a,d) · PPMI(a→b|d) · d=1 -------------------------------------------
    prob = [dict() for _ in range(args.d_max + 1)]
    ppmi = [dict() for _ in range(args.d_max + 1)]
    for d in range(1, args.d_max + 1):
        for a, row in cnt[d].items():
            s = sum(w for w in row.values() if w >= args.min_count)
            if s <= 0:
                continue
            pr, pm = {}, {}
            for b, w in row.items():
                if w < args.min_count:
                    continue
                pr[b] = w / s
                # PPMI: chia thêm cho P(b) biên ⇒ hub bị triệt (nguyên nhân P19, §12.85e1)
                q = (uni[a] / utot) * (uni[b] / utot)
                p_ = (w / nd[d]) if nd[d] > 0 else 0.0
                if q > 0 and p_ > 0:
                    x = math.log(p_ / q)
                    if x > 0:
                        pm[b] = x
            if pr:
                prob[d][a] = pr
            if pm:
                ppmi[d][a] = pm
    log(f"[{time.time()-t0:.0f}s] ✓ dựng bảng điểm")

    # ---- chấm trên nửa B ----------------------------------------------------------------------
    rng = random.Random(args.seed)
    pos = []
    for si, toks in enumerate(sents_b):
        for t in range(1, len(toks)):
            pos.append((si, t))
    rng.shuffle(pos)
    pos = pos[: args.n_test]
    log(f"  chấm {len(pos):,} vị trí")

    def score(tbl, toks, t, dmax):
        out: dict[int, float] = defaultdict(float)
        for i in range(1, min(dmax, t) + 1):
            src = toks[t - i]
            row = tbl[i].get(src)
            if not row:
                continue
            w = args.lam ** (i - 1)
            for b, x in row.items():
                out[b] += w * x
        return out

    # §12.87 SỬA THƯỚC: bản đầu đo `d1` bằng PPMI nhưng `skip_raw` bằng xác suất chuẩn-hoá-hàng
    # ⇒ so táo với cam, chênh lệch có thể là PPMI-vs-prob chứ không phải d1-vs-skip. Giờ mỗi
    # thang có ĐỦ CẶP baseline/skip của nó.
    names = ("d1_prob", "skip_prob", "d1_ppmi", "skip_ppmi", "d1p+skip_p")
    hit1 = {k_: 0 for k_ in names}
    hit10 = {k_: 0 for k_ in names}
    mrr = {k_: 0.0 for k_ in names}
    nn = 0
    for si, t in pos:
        toks = sents_b[si]
        gold = toks[t]
        sc = {
            "d1_prob": score(prob, toks, t, 1),
            "skip_prob": score(prob, toks, t, args.d_max),
            "d1_ppmi": score(ppmi, toks, t, 1),
            "skip_ppmi": score(ppmi, toks, t, args.d_max),
        }
        comb = defaultdict(float, dict(sc["skip_prob"]))
        for b, x in sc["d1_prob"].items():
            comb[b] += 2.0 * x      # d=1 nặng gấp đôi — tín hiệu mạnh nhất (§12.82c 82/18)
        sc["d1p+skip_p"] = comb
        nn += 1
        for k_ in names:
            s_ = sc[k_]
            if not s_:
                continue
            g = s_.get(gold)
            if g is None:
                continue
            rank = 1 + sum(1 for x in s_.values() if x > g)
            hit1[k_] += int(rank == 1)
            hit10[k_] += int(rank <= 10)
            mrr[k_] += 1.0 / rank

    log("\n" + "=" * 60)
    log(f"{'kênh':<12}{'top1':>10}{'top10':>10}{'MRR':>10}   (n={nn:,})")
    log("=" * 60)
    for k_ in names:
        log(f"{k_:<12}{hit1[k_]/nn:>10.3f}{hit10[k_]/nn:>10.3f}{mrr[k_]/nn:>10.3f}")
    log("=" * 60)

    b10 = hit10["d1_prob"] / nn      # baseline CÙNG THANG với skip_prob
    best = max(hit10["skip_prob"], hit10["d1p+skip_p"]) / nn
    g1 = b10 >= 0.20
    g2 = (best - b10) >= 0.05
    g3 = hit10["skip_ppmi"] > hit10["skip_prob"]
    log(f"G1 tỉnh táo (d1_prob top10 ≥0.20): {'ĐẠT' if g1 else 'TRƯỢT'} ({b10:.3f})")
    log(f"G2 W_skip hơn d1 ≥+0.05 top10 : {'ĐẠT' if g2 else 'TRƯỢT'} ({best-b10:+.3f})  ← quyết định")
    log(f"G3 PPMI hơn đếm thô           : {'ĐẠT' if g3 else 'TRƯỢT'} "
        f"({hit10['skip_ppmi']/nn:.3f} ‖ {hit10['skip_prob']/nn:.3f})")
    verdict = ("ĐÁNG nối W_skip vào matrix_core như kênh MỀM rồi A/B sinh"
               if (g1 and g2) else
               "W_skip KHÔNG phải kênh trôi chảy tốt hơn — đóng hướng dùng nó cho sinh")
    log(f"\n⇒ {verdict}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
