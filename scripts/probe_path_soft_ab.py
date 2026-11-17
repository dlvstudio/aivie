#!/usr/bin/env python3
"""A/B vá(3) path soft — hung/bach/ly/nq. TK_045 §12.2c

    PYTHONPATH=src .venv/bin/python -u scripts/probe_path_soft_ab.py \\
        --checkpoint checkpoints/network_v12_sent.pkl \\
        --log-file out/probe_path_soft_ab.log
"""

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
from nn_aivie.hca.matrix_core import parent_coactivation_logits  # noqa: E402
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
    ap.add_argument("--log-file", default="out/probe_path_soft_ab.log")
    ap.add_argument("--max-len", type=int, default=36)
    args = ap.parse_args()
    t0 = time.time()
    logf = open(_ROOT / args.log_file, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    state, _ = load_checkpoint(args.checkpoint)
    n = len(state.vocab)
    log(f"ckpt={args.checkpoint} max_len={args.max_len}")
    for soft in (False, True):
        clear_sent_intra_cache()
        tag = "SOFT_ON" if soft else "SOFT_OFF"
        log(f"\n########## {tag} path_soft={soft} ##########")
        cfg = HCAConfig(
            ar_matrix_core=True,
            ar_no_agenda=True,
            ar_sent_agenda_handoff=False,
            ar_style_payoff=False,
            hybrid_temperature=0.0,
            ar_matrix_stack_weight=0.0,
            ar_matrix_path_soft=soft,
        )
        for name, prompt in PROMPTS:
            hist = list(state.vocab.encode_prompt(prompt))
            bag, succ, conf, act = parent_coactivation_logits(
                state,
                hist,
                vocab_size=n,
                top_k_parents=48,
                path_soft=soft,
            )
            top_pi = sorted(
                ((float(act[i]), int(i)) for i in range(act.size) if act[i] > 0),
                reverse=True,
            )[:3]
            log(f"\n--- coact {name} conf={conf:.3f} succ.max={succ.max():.4g} ---")
            for a, pi in top_pi:
                em = decode_surface(
                    state, list(state.parents.parents[pi].emit_ids or [])[:22]
                )
                log(f"  act={a:.4g} pi={pi} {em!r}")
            t1 = time.time()
            out = generate(state, prompt, max_len=args.max_len, config=cfg, seed=42)
            text = decode_surface(state, out)
            cont = text[len(prompt) :] if text.startswith(prompt) else text
            log(f"=== {name} gen ({time.time()-t1:.1f}s) ===")
            log(f"out:  {text[:320]}")
            log(f"cont: {cont[:200]!r}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
