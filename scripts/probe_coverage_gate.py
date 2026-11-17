#!/usr/bin/env python
"""P49 (TK_045 §13 Trụ C) — CỔNG PHỦ ở tầng mệnh đề có cắt được TRÔI ĐỀ không?

**Vì sao:** v4 §1.5 đo được cơ chế DUY NHẤT miễn nhiễm trôi đề là chép-đáp QA (`lạc = 0.00`
ở mọi cap, §12.74), và lý do là nó là cơ chế duy nhất **có cổng phủ nội dung**. Mọi đường
khác trôi 0.12 → 0.27. Trụ C tổng quát hoá đúng cái cổng đó lên tầng mệnh đề.

**⚠️ Định kiến ÂM đã đo, ghi trước:** §12.104 — attestation làm **acceptor ở tầng TOKEN**
TRƯỢT (ba phép đo độc lập). Đây là thí nghiệm KHÁC: acceptor ở tầng **mệnh đề**, dùng
PMI-cross — thước tương quan tốt nhất với chấm mù (Spearman +0.643, §12.117). Nếu vẫn
trượt thì **cả họ acceptor đóng**, và đó là kết quả sạch.

**Thước:** `lạc` = tỉ lệ mệnh đề (từ mệnh đề 2 trở đi) KHÔNG chia fold nội dung nào với
prompt — **dùng lại đúng định nghĩa** của `eval_gen_broad` (`_path_fold` + `gidf ≥ 4.0`),
cố ý, để cổng và thước nói cùng một ngôn ngữ.

**CỔNG G-cov (nêu TRƯỚC):**
  (a) CHÍNH  — `lạc(gated) ≤ 0.5 × lạc(baseline)`: cổng phải cắt ≥ nửa trôi đề.
  (b) GIÁ    — `dead_rate` tăng không quá **+10pp tuyệt đối**: cắt trôi đề bằng cách
               tắc mạch thì không tính là thắng.
  (c) THAM CHIẾU — `lạc(gated) < 0.12` (§13.4). **Báo cáo, KHÔNG phán quyết một mình**:
               0.12 đo trên đường sinh production v12, đây là chuỗi mệnh đề trên mini —
               khác đường, khác quy mô. Tín hiệu chính là cặp (a)+(b) cùng một setup.

Luật 0: chỉ đọc đồ thị. Không train.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/probe_coverage_gate.py \\
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
from nn_aivie.hca.clause_layer import (  # noqa: E402
    CoverageGate,
    build_clause_layer,
    compose_chain,
    content_folds,
)
from nn_aivie.hca.metrics_pmi import PmiCrossIndex  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_LAC_REL = 0.5      # (a) lạc(gated) ≤ 0.5 × lạc(baseline)
GATE_DEAD_PP = 0.10     # (b) dead_rate tăng ≤ +10pp
GATE_LAC_ABS = 0.12     # (c) tham chiếu §13.4 — báo cáo, không phán quyết một mình


def _lac(state, prompt_ids, chain) -> float:
    """Tỉ lệ mệnh đề TỪ CÁI THỨ 2 không chia fold nội dung nào với prompt."""
    pf = content_folds(state, prompt_ids)
    later = [c for c in chain[1:] if len(c.tokens) >= 4]
    if not pf or not later:
        return float("nan")
    return sum(1 for c in later if not (content_folds(state, c.tokens) & pf)) / len(later)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--n-prompt", type=int, default=200)
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--pmi-min", type=float, default=0.5)
    ap.add_argument("--pmi-thetas", default="0.25,0.5,1.0")
    ap.add_argument("--pmi-cut", type=float, default=0.0, help="θ cắt mệnh đề (§13.6)")
    ap.add_argument("--d-join", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump", type=int, default=4)
    ap.add_argument("--out-json", default="out/probe_coverage_gate.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_coverage_gate")
    t0 = time.time()
    logger.log(f"P49 Trụ C — cổng phủ · {args.checkpoint}")
    logger.log(f"CỔNG G-cov (nêu TRƯỚC): (a) lạc ≤ {GATE_LAC_REL}× baseline · "
               f"(b) dead +≤{GATE_DEAD_PP:.0%}pp · (c) tham chiếu lạc < {GATE_LAC_ABS}")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    ps = state.parents
    logger.log(f"load {time.time()-t0:.0f}s · {len(ps.parents):,} cha")

    layer = build_clause_layer(state, pmi_theta=float(args.pmi_cut), logger=logger)
    logger.log(f"tầng mệnh đề: {len(layer):,} · dài TB {layer.mean_len:.2f}")
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    pmi = PmiCrossIndex(state)
    logger.log(f"W_skip + PMI-cross sẵn ({time.time()-t0:.0f}s)")

    # prompt = mệnh đề MỞ của cha ngẫu nhiên (chuỗi phải nối tiếp từ đó)
    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n_prompt]
    logger.log(f"{len(prompts)} prompt (mệnh đề mở, ≥4 token)")

    def _run(gate_fn, label):
        lacs, deads, nclauses, rejs = [], 0, [], []
        samples = []
        for i, p in enumerate(prompts):
            g = gate_fn(p)
            chain, st = compose_chain(
                state, layer, ws, p.tokens,
                max_clauses=int(args.max_clauses), d_join=int(args.d_join),
                gate=g, rng=random.Random(args.seed + i),
            )
            if st["dead"]:
                deads += 1
            nclauses.append(st["n_clauses"])
            rejs.append(st["n_rejected"])
            L = _lac(state, p.tokens, chain)
            if not np.isnan(L):
                lacs.append(L)
            if len(samples) < args.dump and len(chain) >= 2:
                txt = " ‖ ".join(
                    "".join(v.token_of(t) for t in c.tokens).replace("Ġ", " ").strip()
                    for c in chain
                )
                pt = "".join(v.token_of(t) for t in p.tokens).replace("Ġ", " ").strip()
                samples.append(f"[{pt[:60]}] → {txt[:170]}")
        r = {
            "label": label,
            "lac": float(np.mean(lacs)) if lacs else float("nan"),
            "n_lac": len(lacs),
            "dead_rate": deads / max(1, len(prompts)),
            "mean_clauses": float(np.mean(nclauses)) if nclauses else 0.0,
            "mean_rejected": float(np.mean(rejs)) if rejs else 0.0,
        }
        logger.log(f"[{time.time()-t0:.0f}s] {label}: lạc={r['lac']:.3f} (n={r['n_lac']}) "
                   f"· dead={r['dead_rate']:.1%} · mệnh đề TB={r['mean_clauses']:.2f} "
                   f"· loại TB={r['mean_rejected']:.1f}")
        for s in samples:
            logger.log(f"    | {s}")
        return r

    base = _run(lambda _p: None, "A_no_gate")

    def _mk(th=0.0, anchor=0, req=False):
        def _f(p):
            return CoverageGate(
                pmi_index=pmi, pmi_min=th, theme_ids=tuple(p.tokens),
                theme_folds=frozenset(content_folds(state, p.tokens)),
                anchor_min_folds=anchor, require_new_cover=req,
            )
        return _f

    variants: list[tuple[str, dict]] = []
    for th in [float(x) for x in args.pmi_thetas.split(",") if x.strip()]:
        variants.append((f"B_pmi{th:g}", {"th": th}))
        variants.append((f"B_pmi{th:g}_new", {"th": th, "req": True}))
    # ⚠️ TỰ TẠI với thước `lạc` — đo GIÁ (dead/độ dài), không đo "có ăn không". Xem §13.10.
    variants.append(("C_anchor1_TAUT", {"anchor": 1}))
    variants.append(("C_anchor1_new_TAUT", {"anchor": 1, "req": True}))

    rows = []
    for name, kw in variants:
            r = _run(_mk(**kw), name)
            d_lac = r["lac"] - base["lac"]
            d_dead = r["dead_rate"] - base["dead_rate"]
            ga = (not np.isnan(r["lac"])) and r["lac"] <= GATE_LAC_REL * base["lac"]
            gb = d_dead <= GATE_DEAD_PP
            rows.append({**r, "d_lac": d_lac, "d_dead": d_dead,
                         "gate_lac": bool(ga), "gate_dead": bool(gb),
                         "gate_pass": bool(ga and gb),
                         "ref_abs_ok": bool(r["lac"] < GATE_LAC_ABS)})

    logger.log(f"\n=== P49 n={len(prompts)} max_clauses={args.max_clauses} ===")
    logger.log(f"  {'config':16s} {'lạc':>7s} {'Δlạc':>8s} {'dead':>7s} {'Δdead':>8s} "
               f"{'mđ/chuỗi':>9s}  cổng")
    logger.log(f"  {'A_no_gate':16s} {base['lac']:7.3f} {'—':>8s} "
               f"{base['dead_rate']:6.1%} {'—':>8s} {base['mean_clauses']:9.2f}")
    for r in rows:
        logger.log(f"  {r['label']:16s} {r['lac']:7.3f} {r['d_lac']:+8.3f} "
                   f"{r['dead_rate']:6.1%} {r['d_dead']:+7.1%} {r['mean_clauses']:9.2f}  "
                   f"lạc={'✓' if r['gate_lac'] else '✗'} dead={'✓' if r['gate_dead'] else '✗'}"
                   f" ⇒ {'ĐẠT' if r['gate_pass'] else 'TRƯỢT'}")

    ok = any(r["gate_pass"] for r in rows)
    logger.log(f"\n  G-cov: {'ĐẠT' if ok else 'TRƯỢT'}")
    if not ok:
        logger.log("  ⇒ cổng phủ ở tầng mệnh đề KHÔNG cắt được trôi đề mà không tắc mạch."
                   " Cùng kết cục §12.104 (acceptor tầng token) ⇒ ĐÓNG cả họ acceptor.")

    out = _ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "n_prompt": len(prompts),
        "max_clauses": args.max_clauses, "pmi_cut": args.pmi_cut,
        "gates": {"lac_rel": GATE_LAC_REL, "dead_pp": GATE_DEAD_PP,
                  "lac_abs_ref": GATE_LAC_ABS},
        "baseline": base, "rows": rows, "gate_pass": ok,
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"→ {out}")


if __name__ == "__main__":
    main()
