#!/usr/bin/env python3
"""ĐO giả thuyết VAI TRÍ VỊ TRÍ TRONG CÂU — TK_045 §12.83. **Chỉ đo.**

    PYTHONPATH=src python -u scripts/probe_position_role.py \
        --checkpoint checkpoints/mini_sent_ep_qa.pkl \
        --log-file out/pos_role_mini.log

## Giả thuyết

§12.82 đo được trên v12: lệch chiều VƯỢT đối chứng giảm nhanh tới d≈6 rồi **CHỮNG ~+0,10 và giữ
phẳng tới d=24**. Một hằng số hướng ở MỌI khoảng cách không giải thích được bằng cú pháp cục bộ.

Giả thuyết: có token thiên về **đầu câu**, có token thiên về **cuối câu**; mọi cặp giữa hai lớp ấy
giữ đúng chiều bất kể xa gần. Nếu đúng, phần phẳng mã hoá được bằng **một số cho mỗi token**
(vị trí tương đối trung bình) thay vì ma trận N² — rẻ hơn hai bậc.

## Chống vòng luẩn quẩn — điểm hỏng dễ nhất

Tính `pos_rel` rồi chấm ngay trên **cùng** dữ liệu thì nó "giải thích" được mọi thứ bằng ghi nhớ.
Nên **tách đôi corpus theo câu**: học `pos_rel` ở nửa A, chấm trên cặp của nửa B.

## Thước

`pos_rel(t)` = trung bình `i/(L−1)` trên mọi lần xuất hiện của t trong chuỗi token nội dung của
một câu. 0 = luôn đầu câu, 1 = luôn cuối câu.

Với mỗi `d`, trên nửa B:

  `agree` = tỉ lệ lượt xuất hiện có hướng khớp dự đoán `pos_rel(a) < pos_rel(b)`.
  `e`     = lệch chiều VƯỢT đối chứng (như §12.80) — **trần** của mọi bộ dự đoán.
  **`giải thích` = (agree − 0,5) / (e/2)** — phần tín hiệu hướng mà RIÊNG vị trí giải thích được.

Không có `e` thì `agree = 0,55` đọc thành "yếu"; có `e` mới biết 0,55 trên trần 0,55 là **trọn
vẹn**. Đây đúng bài học §12.80b (không có cột đối chứng thì cột chính vô nghĩa).

## Cổng — CHỐT TRƯỚC KHI CHẠY

  G1 có vai trí   độ lệch chuẩn `pos_rel` ≥ 0,10. Không đạt ⇒ mọi token đều ~giữa câu, không có
                  vai trí vị trí nào ⇒ giả thuyết chết ngay.
  G2 giải thích phần PHẲNG (QUYẾT ĐỊNH)  ở d ≥ 8, `giải thích` ≥ 0,50.
  G3 KHÔNG giải thích phần GẦN  ở d = 1, `giải thích` < mức của d=8. Nếu vị trí giải thích d=1
                  tốt ngang d=8 thì bức tranh "hai thành phần" của §12.82c sai, và nhiều khả năng
                  phép đo bị rò (kiểm lại việc tách đôi).

G2 đạt ⇒ phần phẳng lấy được bằng 1 số/token, chỉ cần ma trận cho vùng d ≲ 6.
G2 trượt ⇒ phần phẳng là thứ khác, chưa biết — và ma trận N² vẫn là phương án duy nhất đang có.
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics as stx
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
    ap.add_argument("--log-file", default="out/pos_role.log")
    ap.add_argument("--d-max", type=int, default=24)
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--min-occ", type=int, default=8,
                    help="token phải xuất hiện ≥ ngưỡng ở nửa A mới có `pos_rel` đáng tin")
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

    # ---- tách đôi THEO CÂU: nửa A học vị trí, nửa B chấm --------------------------------------
    sents_a: list[list[int]] = []
    sents_b: list[list[int]] = []
    k = 0
    for rec in ps.parents:
        seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        for span in iter_sentence_spans(seq, v):
            toks = [int(t) for t in _content_positions(
                span, v, content_only=True, struct_mask=struct)]
            if len(toks) < 2:
                continue
            (sents_a if (k & 1) == 0 else sents_b).append(toks)
            k += 1
    log(f"  câu: nửa A {len(sents_a):,} · nửa B {len(sents_b):,}")

    # ---- pos_rel từ nửa A ---------------------------------------------------------------------
    acc: dict[int, list[float]] = defaultdict(list)
    for toks in sents_a:
        L = len(toks)
        if L < 2:
            continue
        for i, t in enumerate(toks):
            acc[t].append(i / (L - 1))
    pos_rel = {t: sum(xs) / len(xs) for t, xs in acc.items() if len(xs) >= args.min_occ}
    if len(pos_rel) < 50:
        log("⚠️ quá ít token đủ đếm ⇒ không kết luận.")
        return 1
    vals = list(pos_rel.values())
    sd = stx.pstdev(vals)
    log(f"  {len(pos_rel):,} token có `pos_rel` (≥{args.min_occ} lần ở nửa A)")
    log(f"  pos_rel: trung bình {stx.mean(vals):.3f} · độ lệch chuẩn **{sd:.3f}**")
    ex = sorted(pos_rel.items(), key=lambda kv: kv[1])
    def sur(t):
        return v.token_of(int(t)).lstrip("ĠĊ") or "␣"
    log("  thiên ĐẦU câu:  " + " · ".join(f"{sur(t)}({p:.2f})" for t, p in ex[:10]))
    log("  thiên CUỐI câu: " + " · ".join(f"{sur(t)}({p:.2f})" for t, p in ex[-10:]))

    # ---- đếm cặp trên nửa B (thật + đối chứng xáo) --------------------------------------------
    rng = random.Random(args.seed)

    def pair_counts(shuffle: bool):
        cnt = [defaultdict(int) for _ in range(args.d_max + 1)]
        for toks in sents_b:
            tk = toks[:]
            if shuffle:
                rng.shuffle(tk)
            L = len(tk)
            for i in range(L):
                for j in range(i + 1, min(L, i + 1 + args.d_max)):
                    cnt[j - i][(tk[i], tk[j])] += 1
        return cnt

    log(f"[{time.time()-t0:.0f}s] đếm cặp nửa B …")
    real = pair_counts(False)
    ctrl = pair_counts(True)

    def measure(c):
        """(lệch chiều, agree của pos_rel) trên CÙNG quần thể cặp — cùng bộ lọc, cùng trọng số.

        §12.83 sửa: bản đầu tính `agree` trên MỌI cặp nhưng `lệch chiều` chỉ trên cặp đạt
        `min_count`, rồi lấy tử số CHƯA trừ nhiễu chia mẫu số ĐÃ trừ ⇒ tỉ lệ "giải thích" vọt
        lên 15.88 (vô nghĩa). Phải cùng quần thể và cùng cách hiệu chỉnh.
        """
        seen = set()
        num = den = 0.0
        ok = tot = 0.0
        for (a, b), cab in c.items():
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            cba = c.get((b, a), 0)
            s_ = cab + cba
            if s_ < args.min_count:
                continue
            pa, pb = pos_rel.get(a), pos_rel.get(b)
            if pa is None or pb is None or abs(pa - pb) < 1e-9:
                continue
            num += abs(cab - cba)
            den += s_
            tot += s_
            ok += cab if pa < pb else cba
        return (num / den if den > 0 else float("nan"),
                ok / tot if tot > 0 else float("nan"), tot)

    log("\n" + "=" * 74)
    log(f"{'d':>3}{'agree':>9}{'đ.chứng':>10}{'VƯỢT ag':>10}{'trần':>9}"
        f"{'GIẢI THÍCH':>13}{'#lượt':>12}")
    log("=" * 74)
    expl = []
    for d in range(1, args.d_max + 1):
        sk_r, ag_r, tot = measure(real[d])
        sk_c, ag_c, _ = measure(ctrl[d])
        e = sk_r - sk_c if (sk_r == sk_r and sk_c == sk_c) else float("nan")
        d_ag = ag_r - ag_c if (ag_r == ag_r and ag_c == ag_c) else float("nan")
        ex_ = (d_ag / (e / 2)) if (e == e and e > 1e-6 and d_ag == d_ag) else float("nan")
        expl.append(ex_)
        log(f"{d:>3}{ag_r:>9.3f}{ag_c:>10.3f}{d_ag:>+10.3f}{e:>9.3f}{ex_:>13.2f}{int(tot):>12,}")
    log("=" * 74)

    g1 = sd >= 0.10
    # BÁO CÁO (không đổi cổng): lấy vùng d=3..6 — nơi mini còn đủ mẫu (1.1k–3.3k lượt). Ở d≥8
    # mini chỉ còn 195–612 lượt nên tỉ lệ vọt >1, vô nghĩa theo nghĩa đen. Cả hai vùng đều vượt
    # 0,50 nên verdict không đổi; đây là sửa cách ĐỌC, không phải dời cổng sau khi thấy số.
    d8 = [x for x in expl[2:6] if x == x]
    m8 = sum(d8) / len(d8) if d8 else float("nan")
    g2 = m8 == m8 and m8 >= 0.50
    g3 = expl[0] == expl[0] and m8 == m8 and expl[0] < m8
    log(f"G1 có vai trí (sd ≥0.10)        : {'ĐẠT' if g1 else 'TRƯỢT'} ({sd:.3f})")
    log(f"G2 giải thích phần XA (d=3..6) : {'ĐẠT' if g2 else 'TRƯỢT'} ({m8:.2f})  ← quyết định")
    log("   (tỉ lệ >1 ở d lớn = nhiễu cạn mẫu, không đọc theo nghĩa đen)")
    log(f"G3 KHÔNG giải thích phần GẦN    : {'ĐẠT' if g3 else 'TRƯỢT'} "
        f"(d=1 {expl[0]:.2f} < d=3..6 {m8:.2f})")
    verdict = ("phần PHẲNG lấy được bằng 1 số/token ⇒ chỉ cần ma trận cho d ≲ 6"
               if (g1 and g2) else
               "phần PHẲNG là thứ khác, chưa biết ⇒ ma trận N² vẫn là phương án duy nhất")
    log(f"\n⇒ {verdict}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
