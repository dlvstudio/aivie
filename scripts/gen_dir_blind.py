#!/usr/bin/env python
"""§12.124 (③ vào A) — dmax=1 vs dmax=4 trên đường TRUY HỒI, chấm mù.

`directed_bigram_dmax` đổi cách rank_matches chọn cha để CHÉP (cặp-hướng d≤dmax). Đo trên đường
truy hồi (KHÔNG ar_no_agenda) — nơi rank_matches quyết. Câu hỏi: +31pp source-recovery có thành
đáp TỐT HƠN không, hay cov nuốt (answer-selection ≠ source-recovery).
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
from nn_aivie.hca.readout import decode_surface  # noqa: E402

from gen_samples_blind import PROMPTS  # noqa: E402

# Đường TRUY HỒI (mặc định, KHÔNG ar_no_agenda) — nơi rank_matches chọn cha.
BASE = dict(ar_matrix_core=True, ar_style_payoff=False, ar_matrix_stack_weight=0.0)
CONFIGS = {"base_max": dict(directed_bigram_dmax=1, directed_bigram_accum="max"),
           "d4_sum": dict(directed_bigram_dmax=4, directed_bigram_accum="sum")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    print(f"[{time.time()-t0:.0f}s] loaded {len(state.parents):,} cha", flush=True)

    rows = []
    for name, over in CONFIGS.items():
        cfg = HCAConfig(**{**BASE, **over})
        for p in PROMPTS:
            out = generate(state, p, max_len=args.max_len, config=cfg, seed=args.seed)
            pids = list(state.vocab.encode_prompt(p))
            rows.append((name, p, decode_surface(state, out[len(pids):]).strip()))
        print(f"[{time.time()-t0:.0f}s] xong {name}", flush=True)

    n_diff = sum(1 for i in range(len(PROMPTS))
                 if rows[i][2] != rows[i + len(PROMPTS)][2])
    print(f"[{time.time()-t0:.0f}s] dmax1 ≠ dmax4: {n_diff}/{len(PROMPTS)} prompt", flush=True)

    rng = random.Random(20260730)
    ids = [f"D{n:03d}" for n in range(len(rows))]
    rng.shuffle(ids)
    key, blind = {}, []
    for sid, (name, p, cont) in zip(ids, rows):
        key[sid] = {"config": name, "prompt": p}
        blind.append((sid, p, cont))
    blind.sort(key=lambda r: r[0])
    with open(_ROOT / "out/dir_samples.txt", "w") as f:
        f.write("# CHẤM MÙ ③ dmax — không lộ cấu hình.\n\n")
        for sid, p, cont in blind:
            f.write(f"[{sid}]  PROMPT: {p}\n        SINH  : {cont}\n\n")
    with open(_ROOT / "out/dir_key.json", "w") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    print(f"[{time.time()-t0:.0f}s] → out/dir_samples.txt", flush=True)


if __name__ == "__main__":
    main()
