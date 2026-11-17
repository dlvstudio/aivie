#!/usr/bin/env python
"""P57 (TK_045 §13.21) — cổng CẶP-ÂM-TIẾT có sống nổi ở quy mô v12 không?

**Bối cảnh.** §13.21 đo trên mini: cổng phủ đang chạy (khớp **âm tiết lẻ**) có `lạc` thật
= **0.966** — 96.6% mệnh đề lạc đề — trong khi thước lỏng báo 0.000, vì **cổng và thước
dùng CÙNG một định nghĩa fold** (tự tại, rỗng). Tiếng Việt đơn âm tiết ⇒ `lục` khớp cả
*lục lạp* lẫn *lục bát*.

Cổng **cặp âm tiết** (`lục|lạp` ≠ `lục|bát`) **đúng chiều** — `pmi_cross` (thước ĐỘC LẬP,
không dùng fold) lên **1.845 → 2.45–2.54** — nhưng trên mini nó **chết đói**: 99–100% chuỗi
tắc, chỉ **3/120** đầu ra chấm được.

**Giả thuyết:** đây là bài toán **VẬT LIỆU**, không phải ngưỡng. §13.13 đo vật liệu co giãn
**tuyến tính** theo dữ liệu (1 083 → 44 940 ứng viên khi 1k → 44k cha). v12 có **1.76 triệu**
mệnh đề (§12.150) vs mini **10 283** = **171×**. Nếu giả thuyết đúng thì ở v12 cổng cặp sẽ
sống.

**CỔNG G-fold (nêu TRƯỚC):** trên v12, cấu hình `D = cặp + bỏ chữ số` phải đạt CẢ BA:
  (a) `dead_rate` ≤ **50%** (mini: 99%) — có đủ vật liệu để sống;
  (b) `lac_strict` ≤ **0.30** (cổng âm-tiết-lẻ hiện tại: 0.966) — thật sự giữ được đề;
  (c) `pmi_cross` ≥ của cấu hình A — không đánh đổi bám đề lấy sự sống.
Trượt (a) ⇒ vật liệu vẫn thiếu ở **171×** ⇒ **đóng hướng cổng-fold**, cần cơ chế khác.

⚠️ **Thước `lac_strict` CỐ ĐỊNH** (cặp + bỏ chữ số) cho MỌI cấu hình — tách khỏi định nghĩa
mà cổng dùng. Nếu để chúng dùng chung thì `lạc` tự tại và không phán được gì (đã vấp §13.10).
`pmi_cross` **không dùng fold** nên là đối chứng hoàn toàn độc lập.

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/probe_fold_gate.py \\
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
from nn_aivie.hca.w_skip import ensure_w_skip  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_DEAD_MAX = 0.50
GATE_LAC_MAX = 0.30

# Bốn cấu hình cổng. Nhãn giữ nguyên §13.21 để so thẳng với bảng mini.
CONFIGS = (
    ("A_don_am_tiet", {}),
    ("B_bo_chu_so", {"drop_digits": True}),
    ("C_cap_am_tiet", {"pair_folds": True}),
    ("D_cap_bo_so", {"pair_folds": True, "drop_digits": True}),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--n-prompt", type=int, default=120)
    ap.add_argument("--max-clauses", type=int, default=5)
    ap.add_argument("--pmi-cut", type=float, default=0.0, help="θ cắt mệnh đề (§13.6)")
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump", type=int, default=4)
    ap.add_argument("--out-json", default="out/probe_fold_gate.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_fold_gate")
    t0 = time.time()
    logger.log(f"P57 — cổng cặp-âm-tiết ở quy mô v12 · {args.checkpoint}")
    logger.log(f"CỔNG G-fold (nêu TRƯỚC) cho D: dead ≤ {GATE_DEAD_MAX:.0%} · "
               f"lac_strict ≤ {GATE_LAC_MAX} · pmi ≥ pmi(A)")
    logger.log("mini để so: A lạc .966/chết 68% · D lạc .000/chết 99%/n=3")

    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    logger.log(f"load {time.time()-t0:.0f}s · {len(state.parents.parents):,} cha")

    layer = build_clause_layer(state, pmi_theta=args.pmi_cut, logger=logger)
    logger.log(f"tầng mệnh đề: {len(layer):,} · dài TB {layer.mean_len:.2f} "
               f"({time.time()-t0:.0f}s)")
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    pmi = PmiCrossIndex(state)
    logger.log(f"W_skip + PMI-cross sẵn ({time.time()-t0:.0f}s)")

    rng = random.Random(args.seed)
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    prompts = openers[: args.n_prompt]
    logger.log(f"{len(prompts)} prompt đóng băng (seed={args.seed})")

    def lac_strict(p, chain) -> float:
        """THƯỚC CỐ ĐỊNH cho mọi cấu hình — cặp + bỏ chữ số, tách khỏi cổng."""
        pf = content_folds(state, p.tokens, pair=True, drop_digits=True)
        later = [c for c in chain[1:] if len(c.tokens) >= 4]
        if not pf or not later:
            return float("nan")
        return sum(
            1 for c in later
            if not (content_folds(state, c.tokens, pair=True, drop_digits=True) & pf)
        ) / len(later)

    rows = []
    for name, kw in CONFIGS:
        L, P, T = [], [], []
        dead = 0
        samples = []
        for i, p in enumerate(prompts):
            if i and i % 25 == 0:
                logger.log(f"  [{name}] {i}/{len(prompts)} ({time.time()-t0:.0f}s)")
            tf = content_folds(
                state, p.tokens, gidf_min=args.gidf_min,
                pair=kw.get("pair_folds", False),
                drop_digits=kw.get("drop_digits", False),
            )
            gate = CoverageGate(
                theme_folds=frozenset(tf), anchor_min_folds=1,
                gidf_min=args.gidf_min, **kw,
            )
            chain, st_ = compose_chain(
                state, layer, ws, p.tokens, max_clauses=args.max_clauses,
                gate=gate, rng=random.Random(args.seed + i), gidf_min=args.gidf_min,
            )
            if st_["dead"]:
                dead += 1
            if not chain:
                continue
            cont = [int(t) for c in chain for t in c.tokens]
            v = lac_strict(p, chain)
            if not np.isnan(v):
                L.append(v)
            P.append(float(pmi.score(list(p.tokens), cont)))
            T.append(len(cont))
            if len(samples) < args.dump:
                from nn_aivie.hca.readout import decode_surface
                samples.append(
                    f"[{decode_surface(state, list(p.tokens)).strip()[:44]}] → "
                    f"{decode_surface(state, cont).strip()[:120]}"
                )
        m = lambda a: float(np.mean(a)) if a else float("nan")  # noqa: E731
        row = {
            "config": name, "lac_strict": m(L), "pmi_cross": m(P),
            "dead_rate": dead / max(1, len(prompts)), "n_tokens": m(T),
            "n_scored": len(L), "n_alive": len(P),
        }
        rows.append(row)
        logger.log(f"[{time.time()-t0:.0f}s] {name}: lạc(NGHIÊM)={row['lac_strict']:.3f} "
                   f"· pmi={row['pmi_cross']:.3f} · chết={row['dead_rate']:.0%} "
                   f"· tok={row['n_tokens']:.1f} · n={row['n_scored']}")
        for s in samples:
            logger.log(f"    | {s}")

    by = {r["config"]: r for r in rows}
    A, D = by.get("A_don_am_tiet"), by.get("D_cap_bo_so")
    logger.log(f"\n=== P57 n={len(prompts)} (mini để so: A .966/68% · D .000/99%/n=3) ===")
    logger.log(f"  {'cổng':18s} {'lạc(NGHIÊM)':>13s} {'pmi':>8s} {'chết%':>7s} "
               f"{'tok':>7s} {'n':>5s}")
    for r in rows:
        logger.log(f"  {r['config']:18s} {r['lac_strict']:13.3f} {r['pmi_cross']:8.3f} "
                   f"{r['dead_rate']:6.0%} {r['n_tokens']:7.1f} {r['n_scored']:5d}")

    ok = False
    if A and D:
        ga = D["dead_rate"] <= GATE_DEAD_MAX
        gb = (not np.isnan(D["lac_strict"])) and D["lac_strict"] <= GATE_LAC_MAX
        gc_ = (not np.isnan(D["pmi_cross"])) and D["pmi_cross"] >= A["pmi_cross"]
        ok = bool(ga and gb and gc_)
        logger.log(f"\n  G-fold: dead={'✓' if ga else '✗'} ({D['dead_rate']:.0%}) · "
                   f"lạc={'✓' if gb else '✗'} ({D['lac_strict']:.3f}) · "
                   f"pmi={'✓' if gc_ else '✗'} ({D['pmi_cross']:.3f} vs A "
                   f"{A['pmi_cross']:.3f}) ⇒ {'ĐẠT' if ok else 'TRƯỢT'}")
        if np.isnan(D["lac_strict"]):
            logger.log(f"  ⚠️ `lac_strict` = nan vì D chỉ còn {D['n_scored']} mẫu chấm "
                       f"được — KHÔNG phải 'giữ đề hoàn hảo', mà là KHÔNG ĐỦ MẪU.")
        if not ga:
            big = len(layer) >= 500_000     # ~ quy mô v12 (1.76M mệnh đề §12.150)
            if big:
                logger.log("  ⇒ chết vẫn cao Ở QUY MÔ v12 ⇒ giả thuyết VẬT LIỆU **BỊ BÁC**; "
                           "đóng hướng cổng-fold, cần cơ chế khớp khác (không phải fold).")
            else:
                logger.log(f"  ⇒ đây là checkpoint NHỎ ({len(layer):,} mệnh đề) — kết quả "
                           f"này KHÔNG bác được giả thuyết vật liệu. Chạy trên v12 mới phán.")

    out = _ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "n_prompt": len(prompts),
        "n_clauses": len(layer), "pmi_cut": args.pmi_cut, "gidf_min": args.gidf_min,
        "gates": {"dead_max": GATE_DEAD_MAX, "lac_max": GATE_LAC_MAX},
        "rows": rows, "gate_pass": ok, "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"  → {out}")


if __name__ == "__main__":
    main()
