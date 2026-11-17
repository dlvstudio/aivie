#!/usr/bin/env python3
"""THƯỚC TRÔI CHẢY — bước ⓪ của hướng soạn câu. TK_045 §12.103. **Chỉ đo.**

    PYTHONPATH=src python -u scripts/probe_fluency_ruler.py \
        --checkpoint checkpoints/mini_sent_ep_qa.pkl \
        --log-file out/fluency_mini.log

## Vì sao thước này phải có TRƯỚC bộ soạn

Bộ đo hiện có (`eval_gen_broad`): `empty` · `degen` · `lạc` · `digit⎵` · `loop4` · `copy` ·
`n_facts` · `stop` · `n_tok`. **Không cột nào đo ngữ pháp.** Mà §12.102b vừa đo được: trên v12,
kho mảnh có 22 triệu phần tử nên **câu đã XÁO TRẬT TỰ cũng phủ 61,7%**, và 9,9% câu xáo đạt ≥0,8.

⇒ "Lắp được từ mảnh đã nhớ" là ràng buộc **rất lỏng**. Một bộ soạn có thể ra **văn què** mà mọi
cột hiện có **không báo động** — đúng như `lạc` từng mù với trôi đề cho tới §12.73.

## Thước

`logP/token` = trung bình `log P(t_i | t_{i-1})` trên chuỗi, với làm trơn lùi về unigram:

    P(b|a) = (c(a,b) + α·P(b)) / (c(a) + α)

Đếm trên **nửa A**, chấm trên **nửa B** — chống rò như §12.83. Dùng **chuỗi emit đầy đủ** (kể cả
dấu câu và từ chức năng), **không** lọc content-only: ngữ pháp sống chính ở đám đó.

## HƯỚNG MONG MUỐN — chỗ dễ hiểu sai nhất

**KHÔNG phải "càng cao càng tốt".** Văn chép nguyên si đạt điểm rất cao mà chẳng soạn gì; hiện
`copy ≈ 1,0` nên rất có thể **văn sinh > văn corpus**. Nếu vậy đó là **chữ ký của việc chép**, một
phát hiện, không phải thành tích.

Cách dùng đúng: **mức corpus là ĐÍCH**. Khi thêm bộ soạn, điểm phải **giữ gần mức corpus**, không
được trôi về phía xáo. Thước này là **lan can**, không phải hàm mục tiêu.

## Cổng — CHỐT TRƯỚC KHI CHẠY

  G1 tỉnh táo    `corpus(B)` − `xáo(B)` ≥ **0,5** nat/token. Không đạt ⇒ thước không phát hiện nổi
                 việc đảo trật tự ⇒ vô dụng, dừng.
  G2 thứ tự nền  `corpus` > `xáo` > `unigram ngẫu nhiên`. Sai thứ tự ⇒ thước hỏng.
  G3 hiệu chỉnh  ghi `văn sinh` nằm đâu trong ba mức trên. Không phải cổng đậu/rớt — là **mốc**
                 để sau này biết bộ soạn có kéo tụt hay không.
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import statistics as stx
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

PROMPTS = [
    "Nhà Lý dời đô về", "Ngô Quyền đánh bại quân Nam Hán năm", "Hoa Lư là kinh đô của",
    "Trần Hưng Đạo là", "Chiến thắng Bạch Đằng năm", "Hỏi: Nhà Lý dời đô về đâu? Đáp:",
    "Hỏi: Ai đánh bại quân Nam Hán trên sông Bạch Đằng? Đáp:", "Kinh đô đầu tiên của nước ta là",
    "Lê Lợi khởi nghĩa ở", "Nhà Trần thay nhà Lý năm", "Sông Hồng chảy qua",
    "Thủ đô Việt Nam là",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--log-file", default="out/fluency.log")
    ap.add_argument("--alpha", type=float, default=1.0, help="làm trơn lùi unigram")
    ap.add_argument("--n-test", type=int, default=1200)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--d-max", type=int, default=8, help="§12.103f ngữ cảnh xa cho thước")
    ap.add_argument("--lam", type=float, default=0.85)
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
    log(f"[{time.time()-t0:.0f}s] ✓ {len(ps):,} cha")

    # ---- tách đôi theo CHA; dùng chuỗi emit ĐẦY ĐỦ (ngữ pháp sống ở dấu câu + từ chức năng) --
    seqs_a: list[list[int]] = []
    seqs_b: list[list[int]] = []
    for i, rec in enumerate(ps.parents):
        e = [int(x) for x in (getattr(rec, "emit_ids", ()) or ())]
        if len(e) < 6:
            continue
        (seqs_a if (i & 1) == 0 else seqs_b).append(e)
    log(f"  cha: nửa A {len(seqs_a):,} · nửa B {len(seqs_b):,}")

    # ---- mô hình bigram + unigram từ nửa A -------------------------------------------------
    bi: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    ctx: dict[int, float] = defaultdict(float)
    uni: dict[int, float] = defaultdict(float)
    utot = 0.0
    for e in seqs_a:
        for t in e:
            uni[t] += 1.0
            utot += 1.0
        for a, b in zip(e, e[1:]):
            bi[a][b] += 1.0
            ctx[a] += 1.0
    log(f"[{time.time()-t0:.0f}s] ✓ {len(bi):,} ngữ cảnh · {int(utot):,} token")

    # §12.103f — NGỮ CẢNH XA. Bigram trượt nghiệm thu G4b vì mối ghép chỉ phá MỘT bigram, mà
    # văn thật vốn có bigram hiếm hợp lệ (vị trí tệ nhất của corpus đã −14,17). Mối ghép phá vỡ
    # nhất quán TẦM XA: nhìn từ 8 token trước thì nó vi phạm ~8 ràng buộc, không phải 1.
    # Đây cũng là chỗ W_skip có việc thật — làm THƯỚC PHÁT HIỆN, không phải kênh sinh.
    sk: list[dict[int, dict[int, float]]] = [dict() for _ in range(args.d_max + 1)]
    skc: list[dict[int, float]] = [dict() for _ in range(args.d_max + 1)]
    for e in seqs_a:
        L = len(e)
        for i in range(L):
            for d in range(1, min(args.d_max, L - i - 1) + 1):
                x, y = e[i], e[i + d]
                sk[d].setdefault(x, {})
                sk[d][x][y] = sk[d][x].get(y, 0.0) + 1.0
                skc[d][x] = skc[d].get(x, 0.0) + 1.0
    log(f"[{time.time()-t0:.0f}s] ✓ bảng skip d≤{args.d_max}")

    a_ = float(args.alpha)

    def logp_pos_skip(seq: list[int]) -> list[float]:
        """logP từng vị trí với ngữ cảnh XA: trung bình có trọng λ^(d−1) trên d=1..d_max."""
        out = []
        for i in range(1, len(seq)):
            y = seq[i]
            pu = uni.get(y, 0.0) / utot if utot > 0 else 0.0
            if pu <= 0:
                pu = 1.0 / max(1.0, utot)
            num = den = 0.0
            for d in range(1, min(args.d_max, i) + 1):
                x = seq[i - d]
                p = (sk[d].get(x, {}).get(y, 0.0) + a_ * pu) / (skc[d].get(x, 0.0) + a_)
                wd = args.lam ** (d - 1)
                num += wd * math.log(max(p, 1e-300))
                den += wd
            out.append(num / den if den > 0 else math.log(pu))
        return out

    def logp_pos(seq: list[int]) -> list[float]:
        """logP TỪNG VỊ TRÍ. §12.103e: điểm TRUNG BÌNH pha loãng lỗi cục bộ — ghép 2 nửa câu chỉ
        tạo MỘT bigram sai, chia cho ~20 token thì gần như tàng hình (đo được: −0,295 so với
        −0,710 của việc đảo 1 cặp kề). Hình phạt phải đọc CỤC BỘ."""
        out = []
        for x, y in zip(seq, seq[1:]):
            pu = uni.get(y, 0.0) / utot if utot > 0 else 0.0
            if pu <= 0:
                pu = 1.0 / max(1.0, utot)
            p = (bi.get(x, {}).get(y, 0.0) + a_ * pu) / (ctx.get(x, 0.0) + a_)
            out.append(math.log(max(p, 1e-300)))
        return out

    def logp(seq: list[int]) -> float:
        """log P/token với làm trơn lùi unigram. Bỏ token đầu (không có ngữ cảnh)."""
        if len(seq) < 2:
            return float("nan")
        tot = 0.0
        n = 0
        for x, y in zip(seq, seq[1:]):
            pu = uni.get(y, 0.0) / utot if utot > 0 else 0.0
            if pu <= 0:
                pu = 1.0 / max(1.0, utot)          # token chưa từng thấy
            c_ab = bi.get(x, {}).get(y, 0.0)
            c_a = ctx.get(x, 0.0)
            p = (c_ab + a_ * pu) / (c_a + a_)
            tot += math.log(max(p, 1e-300))
            n += 1
        return tot / n

    rng = random.Random(args.seed)
    tgt = seqs_b[:]
    rng.shuffle(tgt)
    tgt = tgt[: args.n_test]

    uni_ids = list(uni.keys())
    uni_w = [uni[t] for t in uni_ids]

    def worst(seq: list[int]) -> float:
        """Vị trí TỆ NHẤT theo bigram — thước cục bộ, không pha loãng theo độ dài."""
        xs = logp_pos(seq)
        return min(xs) if xs else float("nan")

    def worst_skip(seq: list[int]) -> float:
        xs = logp_pos_skip(seq)
        return min(xs) if xs else float("nan")

    lv: dict[str, list[float]] = defaultdict(list)
    wo: dict[str, list[float]] = defaultdict(list)
    wk: dict[str, list[float]] = defaultdict(list)
    for e in tgt:
        lv["corpus(B)"].append(logp(e))
        wo["corpus(B)"].append(worst(e))
        wk["corpus(B)"].append(worst_skip(e))
        sh = e[:]
        rng.shuffle(sh)
        lv["xáo(B)"].append(logp(sh))
        wo["xáo(B)"].append(worst(sh))
        wk["xáo(B)"].append(worst_skip(sh))
        rnd = rng.choices(uni_ids, weights=uni_w, k=len(e))
        lv["unigram ngẫu nhiên"].append(logp(rnd))
        # §12.103d — XÁO THÔ không đủ để nghiệm thu thước. Bộ soạn sẽ gây lỗi TINH VI hơn:
        # một mối ghép sai, hoặc đảo hai token kề. Nếu thước không thấy hai thứ này thì nó là
        # lan can quá thưa cho đúng việc nó sinh ra để làm.
        if len(e) >= 5:
            sw = e[:]
            j = rng.randrange(len(sw) - 1)
            sw[j], sw[j + 1] = sw[j + 1], sw[j]          # đảo ĐÚNG MỘT cặp kề
            lv["đảo 1 cặp kề"].append(logp(sw))
        wo["đảo 1 cặp kề"].append(worst(sw))
        wk["đảo 1 cặp kề"].append(worst_skip(sw))
        other = tgt[rng.randrange(len(tgt))]
        if len(other) >= 6 and len(e) >= 6:
            h = len(e) // 2
            spl = e[:h] + other[len(other) // 2:]        # GHÉP nửa câu này + nửa câu khác
            lv["ghép 2 nửa câu"].append(logp(spl))
        wo["ghép 2 nửa câu"].append(worst(spl))
        wk["ghép 2 nửa câu"].append(worst_skip(spl))

    # ---- văn SINH từ đường sinh hiện tại ---------------------------------------------------
    log(f"[{time.time()-t0:.0f}s] sinh {len(PROMPTS)} prompt …")
    cfg = HCAConfig(ar_matrix_core=True, ar_style_payoff=False, hybrid_temperature=0.0,
                    ar_matrix_stack_weight=0.0, ar_no_agenda=True,
                    ar_sent_agenda_handoff=True)
    gen_txt = []
    for p_ in PROMPTS:
        try:
            o = [int(t) for t in generate(state, p_, max_len=args.max_len, config=cfg, seed=42)]
        except Exception as e_:
            log(f"  ⚠️ {p_!r}: {e_}")
            continue
        plen = len(v.encode_prompt(p_))
        cont = o[plen:]
        if len(cont) >= 3:
            lv["văn sinh"].append(logp(cont))
            wo["văn sinh"].append(worst(cont))
            wk["văn sinh"].append(worst_skip(cont))
            gen_txt.append(decode_surface(state, cont)[:88])
    log(f"[{time.time()-t0:.0f}s] ✓ {len(lv['văn sinh'])} mẫu sinh chấm được")

    order = ["corpus(B)", "văn sinh", "đảo 1 cặp kề", "ghép 2 nửa câu", "xáo(B)",
             "unigram ngẫu nhiên"]
    log("\n" + "=" * 62)
    log(f"{'mức':<22}{'logP/token':>13}{'tệ nhất bi':>13}{'tệ nhất SKIP':>15}{'n':>7}")
    log("=" * 62)
    m, w, wsk = {}, {}, {}
    for k in order:
        xs = [x for x in lv[k] if x == x]
        if not xs:
            continue
        m[k] = stx.mean(xs)
        ws = [x for x in wo.get(k, []) if x == x]
        w[k] = stx.mean(ws) if ws else float("nan")
        ks = [x for x in wk.get(k, []) if x == x]
        wsk[k] = stx.mean(ks) if ks else float("nan")
        log(f"{k:<22}{m[k]:>13.3f}{w[k]:>13.3f}{wsk[k]:>15.3f}{len(xs):>7}")
    log("=" * 62)

    g1 = ("corpus(B)" in m and "xáo(B)" in m and (m["corpus(B)"] - m["xáo(B)"]) >= 0.5)
    g2 = ("corpus(B)" in m and "xáo(B)" in m and "unigram ngẫu nhiên" in m
          and m["corpus(B)"] > m["xáo(B)"] > m["unigram ngẫu nhiên"])
    gap = m.get("corpus(B)", float("nan")) - m.get("xáo(B)", float("nan"))
    log(f"G1 corpus − xáo ≥ 0.5 nat/token : {'ĐẠT' if g1 else 'TRƯỢT'} ({gap:+.3f})")
    log(f"G2 corpus > xáo > unigram       : {'ĐẠT' if g2 else 'TRƯỢT'}")
    if "văn sinh" in m and "corpus(B)" in m:
        d = m["văn sinh"] - m["corpus(B)"]
        log(f"G3 mốc `văn sinh` so corpus     : {d:+.3f}  "
            f"({'CAO HƠN corpus ⇒ chữ ký CHÉP' if d > 0 else 'thấp hơn corpus'})")
    # nghiệm thu cho ĐÚNG việc: thước phải thấy mối ghép sai, không chỉ thấy xáo thô
    if "ghép 2 nửa câu" in m and "corpus(B)" in m:
        dg = m["corpus(B)"] - m["ghép 2 nửa câu"]
        dw = w["corpus(B)"] - w["ghép 2 nửa câu"]
        log(f"G4 thấy MỐI GHÉP SAI — TRUNG BÌNH (≥0.3): {'ĐẠT' if dg >= 0.3 else 'TRƯỢT'} "
            f"({dg:+.3f})")
        dk = wsk["corpus(B)"] - wsk["ghép 2 nửa câu"]
        log(f"G4b MỐI GHÉP — tệ nhất BIGRAM (≥1.0): {'ĐẠT' if dw >= 1.0 else 'TRƯỢT'} ({dw:+.3f})")
        log(f"G4c MỐI GHÉP — tệ nhất SKIP   (≥1.0): {'ĐẠT' if dk >= 1.0 else 'TRƯỢT'} ({dk:+.3f})"
            "  ← nghiệm thu cho đúng việc")
    if "đảo 1 cặp kề" in m and "corpus(B)" in m:
        ds = m["corpus(B)"] - m["đảo 1 cặp kề"]
        log(f"G5 thấy ĐẢO 1 CẶP KỀ (≥0.1)     : {'ĐẠT' if ds >= 0.1 else 'TRƯỢT'} ({ds:+.3f})")

    log("\nHƯỚNG MONG MUỐN: mức corpus là ĐÍCH, không phải cực đại. Bộ soạn sau này phải GIỮ gần")
    log("mức corpus, không được trôi về phía `xáo`. Thước này là LAN CAN, không phải hàm mục tiêu.")
    if gen_txt:
        log("\nmẫu sinh đã chấm:")
        for t_ in gen_txt[:6]:
            log(f"  {t_!r}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
