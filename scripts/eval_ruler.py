#!/usr/bin/env python
"""P50 (TK_045 §13 Trụ E) — chấm hai đường sinh trên CÙNG cây thước.

Trả lời câu §13.10 để ngỏ: *"chuỗi ngắn-đúng-đề có tốt hơn chuỗi dài-trôi-đề không?"* —
`lạc`/`dead` không phán được, phải có trục **attestation × novelty**.

Ba đường được chấm trên cùng bộ prompt đóng băng:

| nhãn | đường |
|---|---|
| `prod` | `generate()` production (W_skip primary + agenda MZ) |
| `clause_nogate` | chuỗi mệnh đề §13 Trụ A, chọn thuần theo mối nối |
| `clause_gated` | + cổng phủ §13 Trụ C (`anchor_min_folds=1`) |

⚠️ **Bảng 2×2 là bắt buộc, không đọc một trục.** Bộ chép đạt `attestation=1.00` tầm thường;
salad đạt `novelty≈1.0` tầm thường. Ô cần đến là **ĐÍCH** (att cao ∧ nov cao). Đọc một trục
là tái phạm đúng lỗi §12.112.

⚠️ **Script này KHÔNG phán "bên nào hay hơn".** Nó cho số khách quan + xuất bộ **cặp A/B
giấu nhãn** để chấm mù riêng (giữ nguyên nguyên tắc §12.111: không tự chấm kẻo thiên lệch
theo kỳ vọng). Cỡ mẫu cần lấy từ `ruler.min_pairs_for` — n=30 chỉ phát hiện nổi ~+26pp.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/eval_ruler.py \\
      --checkpoint checkpoints/mini_sent_ep_qa.pkl
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
from nn_aivie.eval.ruler import (  # noqa: E402
    CopySpanIndex,
    NgramIndex,
    min_pairs_for,
    score_generation,
)
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
)
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.metrics_pmi import PmiCrossIndex  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.slot_swap import ensure_slot_index, swap_slots  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def _lac(state, prompt_ids, seg_lists) -> float:
    pf = content_folds(state, prompt_ids)
    later = [s for s in seg_lists[1:] if len(s) >= 4]
    if not pf or not later:
        return float("nan")
    return sum(1 for s in later if not (content_folds(state, s) & pf)) / len(later)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-prompt", type=int, default=120)
    ap.add_argument("--max-len", type=int, default=48)
    # §13.52 — theo mặc định production (đổi 5 → 1). Để lệch thì `eval_ruler` đo một cấu
    # hình KHÔNG còn là mặc định, và mọi số nó in ra sẽ nói về một hệ khác hệ đang chạy.
    ap.add_argument("--max-clauses", type=int, default=1)
    ap.add_argument("--order", type=int, default=2, help="bậc n-gram cho attestation")
    ap.add_argument("--swap-order", type=int, default=2, help="Trụ B: 2=bigram, 3=trigram")
    ap.add_argument("--swap-max-gidf", type=float, default=4.0,
                    help="Trụ B van fact: chỉ thay token gidf DƯỚI mức này")
    ap.add_argument("--pair", default="clause_nogate,clause_gated",
                    help="hai đường đem CHẤM MÙ (mặc định = câu hỏi Trụ C §13.10)")
    ap.add_argument("--dump-swaps", type=int, default=12,
                    help="số ví dụ thay khe in ra để ĐỌC TAY (metric mù với hỏng fact)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default="out/eval_ruler.json")
    ap.add_argument("--out-pairs", default="out/ruler_blind_pairs.txt")
    ap.add_argument("--out-key", default="out/ruler_blind_key.json")
    args = ap.parse_args()

    logger = get_run_logger("eval_ruler")
    t0 = time.time()
    logger.log(f"P50 Trụ E — cây thước · {args.checkpoint}")
    logger.log(f"cỡ mẫu cần (power 80%): +15pp→{min_pairs_for(15)} cặp · "
               f"+10pp→{min_pairs_for(10)} · +5pp→{min_pairs_for(5)}")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    ps = state.parents
    logger.log(f"load {time.time()-t0:.0f}s · {len(ps.parents):,} cha")

    emits = [
        (r.answer_ids if (r.is_qa and r.answer_ids) else r.emit_ids)
        for r in ps.parents
    ]
    ngram = NgramIndex(emits, orders=(2, 3))
    copyspan = CopySpanIndex(emits)
    pmi = PmiCrossIndex(state)
    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    logger.log(f"thước sẵn · tầng mệnh đề {len(layer):,} ({time.time()-t0:.0f}s)")

    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n_prompt]
    logger.log(f"{len(prompts)} prompt ĐÓNG BĂNG (seed={args.seed})")

    cfg_prod = HCAConfig()
    slot = ensure_slot_index(state)
    logger.log(f"slot index: {len(slot.succ):,} succ · {len(slot.trigrams):,} trigram")
    results: dict[str, list] = {
        "prod": [], "clause_nogate": [], "clause_gated": [], "clause_gated_swap": [],
        # §13.19 — cấu hình TRƯỚC §13.18 (chưa bão hoà hub, chưa luật biên bề mặt).
        # Giữ để chấm mù vòng hai hỏi đúng câu: hai vá đó có đổi CHẤT LƯỢNG CẢM NHẬN không.
        "clause_gated_pre1318": [],
    }
    swap_examples: list[str] = []
    n_swaps_total = 0
    texts: dict[str, dict[int, str]] = {k: {} for k in results}

    for i, p in enumerate(prompts):
        if i % 25 == 0:
            logger.log(f"  {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
        p_ids = list(p.tokens)
        p_txt = decode_surface(state, p_ids).strip()

        # --- prod ---
        try:
            out = generate(state, p_txt, max_len=args.max_len, config=cfg_prod, seed=args.seed)
            enc = list(v.encode_prompt(p_txt))
            cont = [int(t) for t in out[len(enc):]]
        except Exception as e:  # đường production có thể ngã trên prompt lạ — ghi, không giấu
            logger.log(f"  ⚠️ prod lỗi ở prompt {i}: {type(e).__name__}")
            cont = []
        segs = []
        cur = []
        from nn_aivie.hca.sent_intra import _is_boundary
        for t in cont:
            cur.append(int(t))
            if _is_boundary(v, int(t)):
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        results["prod"].append(score_generation(
            ngram, copyspan, cont, order=args.order,
            lac=_lac(state, p_ids, segs),
            pmi_cross=pmi.score(p_ids, cont) if cont else float("nan"),
        ))
        texts["prod"][i] = decode_surface(state, cont).strip()

        # --- clause paths ---
        _g = CoverageGate(
            theme_folds=frozenset(content_folds(state, p_ids)), anchor_min_folds=1,
        )
        for label, gate, kw in (
            ("clause_nogate", None, {}),
            ("clause_gated", _g, {}),
            ("clause_gated_pre1318", _g, {"hub_beta": 0.0, "no_glue": False}),
        ):
            chain, _st = compose_chain(
                state, layer, ws, p_ids,
                max_clauses=args.max_clauses, gate=gate,
                rng=random.Random(args.seed + i), **kw,
            )
            cont_c = [int(t) for c in chain for t in c.tokens]
            results[label].append(score_generation(
                ngram, copyspan, cont_c, order=args.order,
                lac=_lac(state, p_ids, [list(c.tokens) for c in chain]),
                pmi_cross=pmi.score(p_ids, cont_c) if cont_c else float("nan"),
            ))
            texts[label][i] = decode_surface(state, cont_c).strip()

            # --- Trụ B: thay khe TRÊN đường đã có cổng phủ ---
            if label == "clause_gated":
                swapped, segs_sw = [], []
                for c in chain:
                    sw, ns = swap_slots(
                        slot, list(c.tokens), order=args.swap_order,
                        max_gidf=args.swap_max_gidf, rng=random.Random(args.seed + i),
                    )
                    n_swaps_total += ns
                    if ns and len(swap_examples) < args.dump_swaps:
                        before = decode_surface(state, list(c.tokens)).strip()
                        after = decode_surface(state, sw).strip()
                        if before != after:
                            swap_examples.append(f"{before}\n      →  {after}")
                    swapped.extend(sw)
                    segs_sw.append(sw)
                results["clause_gated_swap"].append(score_generation(
                    ngram, copyspan, swapped, order=args.order,
                    lac=_lac(state, p_ids, segs_sw),
                    pmi_cross=pmi.score(p_ids, swapped) if swapped else float("nan"),
                ))
                texts["clause_gated_swap"][i] = decode_surface(state, swapped).strip()

    def _agg(rs, f):
        vals = [f(r) for r in rs]
        vals = [x for x in vals if isinstance(x, float) and not np.isnan(x)]
        return float(np.mean(vals)) if vals else float("nan")

    logger.log(f"\n=== P50 n={len(prompts)} bậc n-gram={args.order} ===")
    logger.log(f"  {'đường':16s} {'attest':>8s} {'novelty':>8s} {'run':>6s} {'lạc':>7s} "
               f"{'pmi':>7s} {'tok':>6s}   ô (ĐÍCH/CHÉP/SALAD)")
    table = {}
    for label, rs in results.items():
        cells: dict[str, int] = {}
        for r in rs:
            c = r.cell()
            cells[c] = cells.get(c, 0) + 1
        n = max(1, len(rs))
        row = {
            "attestation": _agg(rs, lambda r: r.attestation),
            "novelty": _agg(rs, lambda r: r.novelty),
            "lac": _agg(rs, lambda r: r.lac),
            "pmi_cross": _agg(rs, lambda r: r.pmi_cross),
            "n_tokens": _agg(rs, lambda r: float(r.n_tokens)),
            # §13.13 — trục TUYỆT ĐỐI, BẮT BUỘC đi kèm novelty (tỉ lệ đẹp lên chỉ vì
            # đầu ra dài ra). Thêm vào bảng sau khi §13.18 làm chuỗi ngắn đi 2.4× và
            # novelty tụt theo — không có cột này thì đọc thành "tệ đi".
            "copy_run": _agg(rs, lambda r: float(r.copy_run)),
            "cells": {k: v / n for k, v in cells.items()},
        }
        table[label] = row
        cs = " ".join(f"{k}={v/n:.0%}" for k, v in sorted(cells.items(), key=lambda x: -x[1]))
        logger.log(f"  {label:16s} {row['attestation']:8.3f} {row['novelty']:8.3f} "
                   f"{row['copy_run']:6.1f} {row['lac']:7.3f} {row['pmi_cross']:7.3f} "
                   f"{row['n_tokens']:6.1f}   {cs}")

    logger.log(f"\n  Trụ B: {n_swaps_total} lần thay khe "
               f"(order={args.swap_order}, max_gidf={args.swap_max_gidf})")
    logger.log("  ĐỌC TAY ví dụ thay khe — metric MÙ với 'hỏng fact' (dời đô về "
               "Thăng Long → Hoa Lư: hai bigram đều thật, fact SAI):")
    for ex in swap_examples:
        logger.log(f"    · {ex}")

    logger.log("\n  ⚠️ ĐỌC CẶP: attest cao + novelty THẤP = CHÉP (không phải tốt);"
               " novelty cao + attest THẤP = SALAD. Ô cần đến là ĐÍCH.")

    # --- ĐỘ NHẠY NGƯỠNG: `nov_hi=0.50` là do TÔI đặt, chưa calibrate. Nếu tỉ lệ ĐÍCH
    #     sụp khi siết ngưỡng thì con số ĐÍCH KHÔNG dùng để kết luận được. Bày ra,
    #     đừng giấu — đây đúng lớp lỗi §13.10 (θ đặt trước khi biết phân bố).
    logger.log("\n  Độ nhạy ngưỡng novelty (tỉ lệ ô ĐÍCH):")
    hdr = "  " + f"{'đường':16s}" + "".join(f"{f'nov≥{h:.2f}':>10s}" for h in
                                            (0.3, 0.5, 0.7, 0.8, 0.9))
    logger.log(hdr)
    sens = {}
    for label, rs in results.items():
        vals = []
        for h in (0.3, 0.5, 0.7, 0.8, 0.9):
            n = max(1, len(rs))
            vals.append(sum(1 for r in rs if r.cell(nov_hi=h) == "ĐÍCH") / n)
        sens[label] = vals
        logger.log("  " + f"{label:16s}" + "".join(f"{x:9.0%} " for x in vals))
    logger.log("  ⇒ nếu ĐÍCH sụp nhanh theo ngưỡng thì đầu ra chứa khối chép DÀI —"
               " 'ĐÍCH' ở ngưỡng lỏng KHÔNG phải bằng chứng sinh sáng tạo.")

    # novelty phân vị — để lần sau đặt ngưỡng bằng SỐ, không bằng cảm giác
    for label, rs in results.items():
        nv = np.array([r.novelty for r in rs if not np.isnan(r.novelty)])
        if nv.size:
            logger.log(f"  novelty {label:16s} p25={np.percentile(nv,25):.2f} "
                       f"p50={np.percentile(nv,50):.2f} p75={np.percentile(nv,75):.2f} "
                       f"p90={np.percentile(nv,90):.2f}")

    # --- xuất cặp A/B giấu nhãn để chấm mù (KHÔNG tự chấm) ---
    pair_rng = random.Random(20260802)
    pairs, key = [], {}
    for i in range(len(prompts)):
        pa, pb = [x.strip() for x in args.pair.split(",")]
        a_txt, b_txt = texts[pa].get(i, ""), texts[pb].get(i, "")
        if not a_txt or not b_txt or a_txt == b_txt:
            continue
        pid = f"P{len(pairs):03d}"
        left_is_a = pair_rng.random() < 0.5
        left, right = (a_txt, b_txt) if left_is_a else (b_txt, a_txt)
        key[pid] = {"left": pa if left_is_a else pb,
                    "right": pb if left_is_a else pa,
                    "prompt": decode_surface(state, list(prompts[i].tokens)).strip()}
        pairs.append((pid, key[pid]["prompt"], left, right))

    op = _ROOT / args.out_pairs
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        f.write("# CHẤM MÙ CÓ CẶP — chọn X / Y / hoà. Nhãn cấu hình KHÔNG có ở đây.\n")
        f.write(f"# n={len(pairs)} cặp. Cần {min_pairs_for(15)} cặp để phát hiện +15pp"
                f" (power 80%).\n")
        f.write("# Tiêu chí: đúng ngữ cảnh · không salad · văn phong tự nhiên trôi chảy.\n\n")
        for pid, pr, left, right in pairs:
            f.write(f"[{pid}] PROMPT: {pr}\n   X: {left}\n   Y: {right}\n\n")
    (_ROOT / args.out_key).write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")

    out = _ROOT / args.out_json
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "n_prompt": len(prompts), "order": args.order,
        "table": table, "sensitivity_nov": sens, "n_pairs": len(pairs),
        "min_pairs": {"15pp": min_pairs_for(15), "10pp": min_pairs_for(10)},
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"\n  → {out}")
    logger.log(f"  → {op} ({len(pairs)} cặp, nhãn giấu ở {args.out_key})")
    if len(pairs) < min_pairs_for(15):
        logger.log(f"  ⚠️ {len(pairs)} cặp < {min_pairs_for(15)} — CHƯA đủ power cho +15pp."
                   f" Tăng --n-prompt trước khi chấm.")


if __name__ == "__main__":
    main()
