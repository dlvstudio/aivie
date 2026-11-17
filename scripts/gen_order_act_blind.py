#!/usr/bin/env python
"""P32/P33 — baseline matrix-core + A/B order→act (opt-in) trên v12, có PMI-cross.

Cổng G-act (§12.31): ON không làm pmi TB xấu hơn OFF; ≥1 tín hiệu dương (Δpmi≥+0.05 hoặc
đọc tay ít salad hơn). Default knob vẫn False — chỉ đo.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.metrics_pmi import PmiCrossIndex  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.sent_intra import clear_sent_intra_cache  # noqa: E402

from gen_samples_blind import PROMPTS  # noqa: E402

BASE = dict(
    ar_matrix_core=True,
    ar_style_payoff=False,
    ar_matrix_stack_weight=0.0,
    ar_no_agenda=True,
    ar_sent_agenda_handoff=False,
    hybrid_temperature=0.0,
)
CONFIGS = {
    "A_off": dict(ar_matrix_order_act=False),
    "B_order": dict(
        ar_matrix_order_act=True,
        ar_matrix_order_dmax=4,
        ar_matrix_order_accum="sum",
        ar_matrix_order_lam=0.5,
        ar_matrix_order_beta=1.5,
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="out/p33")
    ap.add_argument("--only", default="", help="A_off|B_order — trống = cả hai")
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    print(f"[{time.time()-t0:.0f}s] loaded {len(state.parents):,} cha", flush=True)
    pmi = PmiCrossIndex(state)
    print(f"[{time.time()-t0:.0f}s] PMI-cross {len(pmi._c):,} cặp", flush=True)  # noqa: SLF001

    names = [args.only] if args.only else list(CONFIGS)
    rows = []  # (name, prompt, text, pmi, n_tok)
    for name in names:
        over = CONFIGS[name]
        clear_sent_intra_cache()
        cfg = HCAConfig(**{**BASE, **over})
        for p in PROMPTS:
            out = generate(state, p, max_len=args.max_len, config=cfg, seed=args.seed)
            pids = list(v.encode_prompt(p))
            cont = list(out[len(pids):])
            sc = pmi.score(pids, cont)
            txt = decode_surface(state, cont).strip()
            rows.append((name, p, txt, sc, len(cont)))
        mean = sum(r[3] for r in rows if r[0] == name) / len(PROMPTS)
        print(f"[{time.time()-t0:.0f}s] xong {name} · pmi_TB={mean:.3f}", flush=True)

    # bảng
    print("\n=== PMI-cross theo cấu hình ===")
    for name in names:
        rs = [r for r in rows if r[0] == name]
        mean = sum(r[3] for r in rs) / len(rs)
        ntok = sum(r[4] for r in rs) / len(rs)
        print(f"  {name:10s}  pmi={mean:.3f}  n_tok={ntok:.1f}")
    if "A_off" in names and "B_order" in names:
        a = sum(r[3] for r in rows if r[0] == "A_off") / len(PROMPTS)
        b = sum(r[3] for r in rows if r[0] == "B_order") / len(PROMPTS)
        d = b - a
        n_better = sum(
            1 for i in range(len(PROMPTS))
            if rows[i + len(PROMPTS)][3] > rows[i][3] + 1e-9
        )
        n_worse = sum(
            1 for i in range(len(PROMPTS))
            if rows[i + len(PROMPTS)][3] + 1e-9 < rows[i][3]
        )
        n_diff = sum(
            1 for i in range(len(PROMPTS))
            if rows[i][2] != rows[i + len(PROMPTS)][2]
        )
        print(f"\n  Δpmi (B−A)={d:+.3f}  text≠ {n_diff}/{len(PROMPTS)}"
              f"  pmi↑ {n_better} · pmi↓ {n_worse}")
        gate_ok = d >= -0.02 and (d >= 0.05 or n_better >= n_worse)
        # G-act: không xấu hơn (Δ≥−0.02) VÀ (Δ≥+0.05 HOẶC không thua đếm từng prompt)
        print(f"  G-act: {'ĐẠT' if (d >= -0.02 and (d >= 0.05 or n_better > n_worse)) else 'TRƯỢT'}"
              f"  (Δpmi={d:+.3f}, better/worse={n_better}/{n_worse})")

    pref = _ROOT / args.out_prefix
    pref.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260731)
    ids = [f"O{n:03d}" for n in range(len(rows))]
    rng.shuffle(ids)
    key, blind = {}, []
    for sid, (name, p, txt, sc, _n) in zip(ids, rows):
        key[sid] = {"config": name, "prompt": p, "pmi": round(sc, 4)}
        blind.append((sid, p, txt))
    blind.sort(key=lambda r: r[0])
    with open(f"{pref}_samples.txt", "w", encoding="utf-8") as f:
        f.write("# CHẤM MÙ order→act — không lộ cấu hình.\n\n")
        for sid, p, txt in blind:
            f.write(f"[{sid}]  PROMPT: {p}\n        SINH  : {txt}\n\n")
    with open(f"{pref}_key.json", "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    with open(f"{pref}_run.log", "w", encoding="utf-8") as f:
        for name in names:
            rs = [r for r in rows if r[0] == name]
            mean = sum(r[3] for r in rs) / len(rs)
            f.write(f"{name}\tpmi={mean:.4f}\tn={len(rs)}\n")
        for name, p, txt, sc, nt in rows:
            f.write(f"---\n{name}\tpmi={sc:.4f}\tn_tok={nt}\nP: {p}\nS: {txt}\n")
    print(f"[{time.time()-t0:.0f}s] → {pref}_samples.txt", flush=True)


if __name__ == "__main__":
    main()
