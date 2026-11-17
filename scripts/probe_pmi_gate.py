#!/usr/bin/env python
"""P60 (TK_045 §13.23) — cổng PMI-cross thay cổng fold, hiệu chỉnh theo MỐC VÀNG.

**Vì sao.** Hai cổng dựa trên **fold** đều hỏng: âm tiết lẻ quá lỏng (86.4% bằng chứng
"đúng đề" là mảnh dưới-từ), cặp âm tiết quá thưa (P57: chết 94% ở v12). Đối chứng vàng
(§13.23) chỉ ra vì sao: fold-cặp chỉ khớp **5.2%** ngay cả giữa hai mệnh đề LIỀN KỀ TRONG
CÙNG MỘT CHA — tức đúng đề *theo định nghĩa*. Cổng đó đòi một sự kiện gần như không xảy ra.

`pmi_cross` là ứng viên **liên tục**, và là thứ duy nhất được kiểm **hai lần độc lập**:
Spearman **+0.643** với chấm mù (§12.117), và **mốc vàng** (§13.23).

## MỐC VÀNG — probe tự tính trên CHÍNH checkpoint đang chạy

`gold` = `pmi_cross` trung bình giữa hai mệnh đề **liền kề trong cùng một cha**. Đây là mức
"đúng đề" mà chính corpus đó thể hiện ⇒ **so được giữa các checkpoint**, không phải hằng số
tôi bịa. Trên mini: **2.128**.

## CỔNG G-pmi (nêu TRƯỚC)

Phải tồn tại **một θ** đạt CẢ HAI so với cổng fold đang chạy (`anchor_min_folds=1`):
  (a) `pmi_achieved` ≥ **gold − 0.05** — chạm mức đúng-đề của chính corpus;
  (b) `dead_rate` ≤ `dead_rate(fold)` — **không** trả giá bằng tắc mạch.
Trượt ⇒ cổng PMI không chi phối được cổng fold ở quy mô thật; giữ nguyên hiện trạng và báo.

Trên mini: `pmi≥1.75` → **2.130 vs gold 2.128** @ chết **54%** (fold: 1.844 @ **69%**);
`pmi≥1.5` → 1.883 @ **37%** — áp đảo fold trên **cả hai** trục.

⚠️ **Không dùng `lạc` (bản nào) làm phán quyết.** Cả hai bản fold đã NGHỈ HƯU cho việc phán
chủ đề (§13.23) — chúng đo trùng lặp bề mặt. Arbiter cuối cùng là **chấm mù**; `pmi_cross`
là proxy đã kiểm.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/probe_pmi_gate.py \\
      --checkpoint checkpoints/network_v12_sent.pkl
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
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GOLD_TOL = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--n-prompt", type=int, default=120)
    ap.add_argument("--n-gold", type=int, default=400)
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--pmi-cut", type=float, default=0.0)
    ap.add_argument("--thetas", default="1.25,1.5,1.75,2.0,2.25")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump", type=int, default=4)
    ap.add_argument("--out-json", default="out/probe_pmi_gate.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_pmi_gate")
    t0 = time.time()
    logger.log(f"P60 — cổng PMI-cross hiệu chỉnh theo mốc vàng · {args.checkpoint}")
    logger.log(f"CỔNG G-pmi (nêu TRƯỚC): ∃θ đạt CẢ HAI — pmi ≥ gold−{GOLD_TOL} "
               f"VÀ chết ≤ chết(cổng fold)")
    logger.log("mini để so: pmi≥1.75 → 2.130 vs gold 2.128 @ chết 54% (fold 1.844 @ 69%)")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    logger.log(f"load {time.time()-t0:.0f}s · {len(state.parents.parents):,} cha")

    layer = build_clause_layer(state, pmi_theta=args.pmi_cut, logger=logger)
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    pmi = PmiCrossIndex(state)
    logger.log(f"tầng mệnh đề {len(layer):,} · W_skip + PMI sẵn ({time.time()-t0:.0f}s)")

    # --- MỐC VÀNG: mệnh đề LIỀN KỀ trong CÙNG một cha = đúng đề theo định nghĩa ---
    pairs = []
    for _pi, cids in layer.by_parent.items():
        for a, b in zip(cids, cids[1:]):
            A, B = layer.clauses[a], layer.clauses[b]
            if len(A.tokens) >= 4 and len(B.tokens) >= 4:
                pairs.append((A, B))
    random.Random(0).shuffle(pairs)
    pairs = pairs[: args.n_gold]
    gold = float(np.mean([pmi.score(list(a.tokens), list(b.tokens)) for a, b in pairs])) \
        if pairs else float("nan")
    logger.log(f"\n★ MỐC VÀNG (n={len(pairs)} cặp mệnh đề liền kề cùng cha): "
               f"pmi = **{gold:.3f}**   (mini: 2.128)")

    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n_prompt]
    logger.log(f"{len(prompts)} prompt đóng băng (seed={args.seed})")

    def run(mk, label):
        P, T = [], []
        dead = 0
        samples = []
        for i, p in enumerate(prompts):
            if i and i % 40 == 0:
                logger.log(f"  [{label}] {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
            ch, st_ = compose_chain(
                state, layer, ws, p.tokens, max_clauses=args.max_clauses,
                gate=mk(p), rng=random.Random(args.seed + i),
            )
            if st_["dead"]:
                dead += 1
            if not ch:
                continue
            cont = [int(t) for c in ch for t in c.tokens]
            P.append(float(pmi.score(list(p.tokens), cont)))
            T.append(len(cont))
            if len(samples) < args.dump:
                samples.append(
                    f"[{decode_surface(state, list(p.tokens)).strip()[:40]}] → "
                    f"{decode_surface(state, cont).strip()[:110]}"
                )
        m = lambda a: float(np.mean(a)) if a else float("nan")  # noqa: E731
        row = {"config": label, "pmi": m(P), "dead_rate": dead / max(1, len(prompts)),
               "n_tokens": m(T), "n_alive": len(P)}
        logger.log(f"[{time.time()-t0:.0f}s] {label}: pmi={row['pmi']:.3f} "
                   f"(so mốc {row['pmi']-gold:+.3f}) · chết={row['dead_rate']:.0%} "
                   f"· tok={row['n_tokens']:.1f}")
        for s in samples:
            logger.log(f"    | {s}")
        return row

    rows = [
        run(lambda p: None, "KHONG_cong"),
        run(lambda p: CoverageGate(
            theme_folds=frozenset(content_folds(state, p.tokens)), anchor_min_folds=1,
        ), "fold_dang_chay"),
    ]
    fold = rows[1]
    for th in [float(x) for x in args.thetas.split(",") if x.strip()]:
        rows.append(run(
            lambda p, th=th: CoverageGate(
                pmi_index=pmi, pmi_min=th, theme_ids=tuple(p.tokens), anchor_min_folds=0,
            ),
            f"pmi_{th:g}",
        ))

    logger.log(f"\n=== P60 n={len(prompts)} · MỐC VÀNG {gold:.3f} ===")
    logger.log(f"  {'cổng':18s} {'pmi':>8s} {'so mốc':>9s} {'chết%':>7s} {'tok':>7s}")
    for r in rows:
        logger.log(f"  {r['config']:18s} {r['pmi']:8.3f} {r['pmi']-gold:+9.3f} "
                   f"{r['dead_rate']:6.0%} {r['n_tokens']:7.1f}")

    winners = [
        r for r in rows if r["config"].startswith("pmi_")
        and r["pmi"] >= gold - GOLD_TOL and r["dead_rate"] <= fold["dead_rate"]
    ]
    ok = bool(winners)
    logger.log(f"\n  G-pmi: {'ĐẠT' if ok else 'TRƯỢT'}"
               + (f" — {', '.join(w['config'] for w in winners)}" if ok else ""))
    if not ok:
        near = [r for r in rows if r["config"].startswith("pmi_")]
        best_p = max(near, key=lambda r: r["pmi"]) if near else None
        if best_p is not None:
            logger.log(f"  ⇒ không θ nào vừa chạm mốc vừa không tệ hơn cổng fold "
                       f"({fold['pmi']:.3f} @ {fold['dead_rate']:.0%}). "
                       f"pmi cao nhất: {best_p['config']} = {best_p['pmi']:.3f} @ "
                       f"{best_p['dead_rate']:.0%}. Giữ hiện trạng, báo tác giả.")

    out = _ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "n_prompt": len(prompts),
        "n_clauses": len(layer), "gold_pmi": gold, "gold_pairs": len(pairs),
        "gate": {"gold_tol": GOLD_TOL}, "rows": rows, "gate_pass": ok,
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"  → {out}")


if __name__ == "__main__":
    main()
