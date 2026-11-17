#!/usr/bin/env python
"""P52 (TK_045 §13 Trụ F) — tầng mệnh đề CO GIÃN theo dữ liệu thế nào?

§0.W khẳng định *"đòn bẩy lớn nhất đo được: DỮ LIỆU"*, và §13.12 vừa kết luận đường nâng
novelty nằm ở **mệnh đề ngắn hơn** hoặc **dữ liệu**. Nhưng ở tầng mệnh đề thì **chưa ai đo**
— trụ F tới giờ mới là một khẳng định mượn từ đường sinh cũ. Probe này biến nó thành số.

**Thiết kế có kiểm soát:** một checkpoint DUY NHẤT, lấy mẫu cha **ngẫu nhiên LỒNG NHAU**
(N nhỏ ⊂ N lớn) ⇒ khác biệt duy nhất là LƯỢNG dữ liệu, không phải miền/vocab/epoch.
(So nhiều checkpoint khác nhau sẽ lẫn cả bốn thứ — không kết luận được gì.)

Ở mỗi N, **cả bộ sinh lẫn thước** đều chỉ thấy đúng tập cha đó — nếu không thì attestation
và novelty đang chấm theo một corpus mà bộ sinh chưa từng thấy.

**⚠️ Nhiễu đã biết, ghi TRƯỚC:** `CopySpanIndex` lớn hơn ⇒ dễ tìm được đoạn trùng dài hơn ⇒
`novelty` bị kéo **XUỐNG** một cách cơ học khi N tăng. Nhiễu này chạy **NGƯỢC** giả thuyết
Trụ F, nên novelty vẫn tăng thì là bằng chứng mạnh; novelty giảm thì **không kết luận được**
ngay — phải tách nhiễu.

**CỔNG G-data (nêu trước):**
  (a) tăng **4×** số cha ⇒ `novelty` tăng ≥ **+0.083** — đúng mức Trụ B đạt được (§13.12)
      nhưng vẫn trượt cổng của nó. Ý: dữ liệu phải **thắng** chỉnh cơ chế.
  (b) `cross_parent_ratio` **≥ 0.95** ở mọi N — tái tổ hợp không được sụp khi scale.
  Phẳng theo dữ liệu ⇒ *"dữ liệu là đòn bẩy lớn nhất"* **KHÔNG đúng ở tầng mệnh đề**.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/probe_data_scaling.py \\
      --checkpoint checkpoints/grade_v3_big_a2_t3_isg.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.eval.ruler import CopySpanIndex, NgramIndex, score_generation  # noqa: E402
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
    join_candidates,
)
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_NOV_PER_4X = 0.083
GATE_CROSS_MIN = 0.95


def _lac(state, prompt_ids, seg_lists) -> float:
    pf = content_folds(state, prompt_ids)
    later = [s for s in seg_lists[1:] if len(s) >= 4]
    if not pf or not later:
        return float("nan")
    return sum(1 for s in later if not (content_folds(state, s) & pf)) / len(later)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/grade_v3_big_a2_t3_isg.pkl")
    ap.add_argument("--scales", default="1000,2500,5000,11000,22000,0",
                    help="số cha mỗi nấc; 0 = hết")
    ap.add_argument("--n-prompt", type=int, default=60)
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--pmi-cut", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default="out/probe_data_scaling.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_data_scaling")
    t0 = time.time()
    logger.log(f"P52 Trụ F — co giãn dữ liệu · {args.checkpoint}")
    logger.log(f"CỔNG G-data (nêu TRƯỚC): 4× cha ⇒ novelty ≥ +{GATE_NOV_PER_4X} · "
               f"cross_parent ≥ {GATE_CROSS_MIN} ở mọi nấc")
    logger.log("⚠️ nhiễu đã biết: index lớn hơn ⇒ novelty bị kéo XUỐNG cơ học "
               "(chạy NGƯỢC giả thuyết ⇒ novelty tăng = bằng chứng mạnh)")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    ps = state.parents
    n_all = len(ps.parents)
    logger.log(f"load {time.time()-t0:.0f}s · {n_all:,} cha")

    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    logger.log(f"W_skip sẵn ({time.time()-t0:.0f}s)")

    # mẫu cha LỒNG NHAU: N nhỏ là tập con của N lớn ⇒ đường cong sạch
    rng0 = random.Random(args.seed)
    perm = list(range(n_all))
    rng0.shuffle(perm)
    scales = [int(x) for x in args.scales.split(",") if x.strip()]
    scales = sorted({(n_all if s <= 0 else min(s, n_all)) for s in scales})

    rows = []
    prompts_fixed: list = []
    for N in scales:
        sub = perm[:N]
        layer = build_clause_layer(
            state, pmi_theta=args.pmi_cut, parent_ids=sub, logger=None,
        )
        if not len(layer):
            continue
        sub_set = set(sub)
        emits = [
            (ps.parents[pi].answer_ids
             if (ps.parents[pi].is_qa and ps.parents[pi].answer_ids)
             else ps.parents[pi].emit_ids)
            for pi in sub
        ]
        ngram = NgramIndex(emits, orders=(2,))
        copyspan = CopySpanIndex(emits)

        # prompt CỐ ĐỊNH qua mọi nấc — chọn từ nấc NHỎ NHẤT để mọi nấc đều có
        if not prompts_fixed:
            cands = [c for c in layer.clauses
                     if c.opens_parent and len(c.tokens) >= 4]
            random.Random(args.seed + 7).shuffle(cands)
            prompts_fixed = cands[: args.n_prompt]
        prompts = [p for p in prompts_fixed if p.parent_idx in sub_set]

        # tái tổ hợp khác-cha (M2 của P47) trên mẫu
        rngm = random.Random(args.seed + 1)
        idxs = rngm.sample(range(len(layer)), min(400, len(layer)))
        cands_n, cross_r = [], []
        for ci in idxs:
            cl = layer.clauses[ci]
            tw = join_candidates(layer, ws, cl.last_content, d_join=4)
            if not tw:
                continue
            tot = sum(len(layer.by_opener[j]) for j in tw)
            own = sum(1 for c2 in layer.by_parent.get(cl.parent_idx, ())
                      if layer.clauses[c2].first_content in tw)
            cands_n.append(tot)
            cross_r.append((tot - own) / max(1, tot))

        novs, atts, lacs, ntok, nrun = [], [], [], [], []
        for i, p in enumerate(prompts):
            gate = CoverageGate(
                theme_folds=frozenset(content_folds(state, p.tokens)),
                anchor_min_folds=1,
            )
            chain, _st = compose_chain(
                state, layer, ws, p.tokens, max_clauses=args.max_clauses,
                gate=gate, rng=random.Random(args.seed + i),
            )
            cont = [int(t) for c in chain for t in c.tokens]
            if len(cont) < 4:
                continue
            sc = score_generation(
                ngram, copyspan, cont, order=2,
                lac=_lac(state, p.tokens, [list(c.tokens) for c in chain]),
            )
            novs.append(sc.novelty)
            atts.append(sc.attestation)
            ntok.append(len(cont))
            nrun.append(copyspan.longest_run(cont)[0])
            if not np.isnan(sc.lac):
                lacs.append(sc.lac)

        row = {
            "n_parents": N,
            "n_clauses": len(layer),
            "n_prompt_used": len(prompts),
            "mean_candidates": float(np.mean(cands_n)) if cands_n else 0.0,
            "cross_parent_ratio": float(np.mean(cross_r)) if cross_r else 0.0,
            "novelty": float(np.mean(novs)) if novs else float("nan"),
            "attestation": float(np.mean(atts)) if atts else float("nan"),
            "lac": float(np.mean(lacs)) if lacs else float("nan"),
            "n_scored": len(novs),
            # ⚠️ nhiễu CHƯA đăng ký trước, phát hiện khi đọc kết quả: novelty = 1 − run/len.
            # Nếu `len` tăng theo N thì novelty tăng CƠ HỌC. Ghi cả hai để tách.
            "mean_tokens": float(np.mean(ntok)) if ntok else float("nan"),
            "mean_run": float(np.mean(nrun)) if nrun else float("nan"),
        }
        rows.append(row)
        logger.log(
            f"[{time.time()-t0:.0f}s] N={N:>6,} cha · {row['n_clauses']:>7,} mệnh đề · "
            f"ứng viên {row['mean_candidates']:>8.0f} · khác-cha "
            f"{row['cross_parent_ratio']:.3f} · nov {row['novelty']:.3f} · "
            f"att {row['attestation']:.3f} · lạc {row['lac']:.3f} (n={row['n_scored']})"
        )

    logger.log(f"\n=== P52 — đường cong co giãn (pmi_cut={args.pmi_cut}) ===")
    logger.log(f"  {'N cha':>8s} {'mệnh đề':>9s} {'ứng viên':>10s} {'khác-cha':>9s} "
               f"{'novelty':>8s} {'attest':>8s} {'tok':>6s} {'run':>6s} {'n':>4s}")
    for r in rows:
        logger.log(f"  {r['n_parents']:>8,} {r['n_clauses']:>9,} "
                   f"{r['mean_candidates']:>10.0f} {r['cross_parent_ratio']:>9.3f} "
                   f"{r['novelty']:>8.3f} {r['attestation']:>8.3f} "
                   f"{r['mean_tokens']:>6.1f} {r['mean_run']:>6.1f} {r['n_scored']:>4d}")
    logger.log("  ⚠️ `tok`/`run` để TÁCH nhiễu: novelty = 1 − run/len. len tăng theo N "
               "⇒ novelty tăng CƠ HỌC, không phải do tái tổ hợp giàu hơn.")

    # cổng: so mỗi cặp cách nhau ~4×
    logger.log(f"\n  CỔNG G-data — mỗi bước ~4× cha:")
    gate_hits = []
    for i, r in enumerate(rows):
        for r2 in rows[i + 1:]:
            ratio = r2["n_parents"] / max(1, r["n_parents"])
            if 3.0 <= ratio <= 5.5:
                d = r2["novelty"] - r["novelty"]
                ok = d >= GATE_NOV_PER_4X
                gate_hits.append(ok)
                logger.log(f"    {r['n_parents']:,} → {r2['n_parents']:,} "
                           f"({ratio:.1f}×): Δnovelty {d:+.3f}  "
                           f"{'ĐẠT' if ok else 'TRƯỢT'}")
    cross_ok = all(r["cross_parent_ratio"] >= GATE_CROSS_MIN for r in rows)
    logger.log(f"    cross_parent ≥ {GATE_CROSS_MIN} ở mọi nấc: "
               f"{'ĐẠT' if cross_ok else 'TRƯỢT'}")
    ok = bool(gate_hits) and any(gate_hits) and cross_ok
    logger.log(f"\n  G-data: {'ĐẠT' if ok else 'TRƯỢT'}")
    if not ok:
        logger.log("  ⇒ ở TẦNG MỆNH ĐỀ, dữ liệu KHÔNG phải đòn bẩy như §0.W nói cho "
                   "đường sinh cũ. Ghi âm tính, đừng lặp lại khẳng định.")

    out = _ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "pmi_cut": args.pmi_cut,
        "gates": {"nov_per_4x": GATE_NOV_PER_4X, "cross_min": GATE_CROSS_MIN},
        "rows": rows, "gate_pass": ok, "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"  → {out}")


if __name__ == "__main__":
    main()
