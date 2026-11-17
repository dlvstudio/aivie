#!/usr/bin/env python3
"""§12.3.3 A/B cổng T1.2 bag-midsent (0.35 vs 1.0) — measure-first sau diag H1/H2."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.sent_intra import clear_sent_intra_cache  # noqa: E402

PROMPTS = [
    ("ly", "Nhà Lý dời đô về"),
    ("bach", "Trận Bạch Đằng năm"),
    ("nq_def", "Ngô Quyền là vị vua sáng lập nhà Ngô, được"),
    ("thd", "Trần Hưng Đạo là"),
    ("hung", "Lê Lợi khởi nghĩa ở"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/probe_t12_ab_bag_midsent.log")
    ap.add_argument("--max-len", type=int, default=36)
    args = ap.parse_args()
    t0 = time.time()
    logf = open(_ROOT / args.log_file, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    state, _ = load_checkpoint(args.checkpoint)
    penalties = [("ON_0.35", 0.35), ("OFF_1.0", 1.0)]
    glue = ("vềNhà", "nămTrận", "làTrần", "ởLê")
    log(f"ckpt={args.checkpoint} max_len={args.max_len}")
    for label, pen in penalties:
        clear_sent_intra_cache()
        cfg = HCAConfig(
            ar_matrix_core=True,
            ar_no_agenda=True,
            ar_sent_agenda_handoff=False,
            ar_style_payoff=False,
            hybrid_temperature=0.0,
            ar_matrix_stack_weight=0.0,
            ar_matrix_bag_midsent_penalty=pen,
        )
        log(f"\n########## {label} ar_matrix_bag_midsent_penalty={pen} ##########")
        for name, prompt in PROMPTS:
            t1 = time.time()
            out = generate(state, prompt, max_len=args.max_len, config=cfg, seed=42)
            text = decode_surface(state, out)
            cont = text[len(prompt) :] if text.startswith(prompt) else text
            hits = [g for g in glue if g in text]
            log(f"\n=== {name}  ({time.time()-t1:.1f}s) glue={hits} ===")
            log(f"prompt: {prompt}")
            log(f"out:    {text[:360]}")
            log(f"cont:   {cont[:200]!r}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
