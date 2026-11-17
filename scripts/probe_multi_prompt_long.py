#!/usr/bin/env python3
"""Đa prompt — so sánh no_agenda (1 câu) vs agenda+handoff (đa câu)."""

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
    ("nq_def", "Ngô Quyền là vị vua sáng lập nhà Ngô, được"),
    ("nq_full", "Ngô Quyền là vị vua sáng lập nhà Ngô, được xem là người mở đầu thời kỳ độc lập tự chủ"),
    ("thd", "Trần Hưng Đạo là"),
    ("ly", "Nhà Lý dời đô về"),
    ("bach", "Trận Bạch Đằng năm"),
    ("qa_nq", "Hỏi: Ngô Quyền là ai"),
    ("hung", "Lê Lợi khởi nghĩa ở"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/probe_multi_prompt_long.log")
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()
    t0 = time.time()
    logf = open(_ROOT / args.log_file, "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    state, _ = load_checkpoint(args.checkpoint)
    modes = {
        # 1 câu: matrix, dừng ở EOS.
        "no_agenda": dict(ar_no_agenda=True, ar_sent_agenda_handoff=False),
        # Đa câu: VẪN matrix (ar_no_agenda=True) + handoff episode ở EOS.
        # ar_no_agenda=False sẽ lệch sang retrieval MZ — không phải L2R/R2L path.
        "agenda_handoff": dict(
            ar_no_agenda=True,
            ar_sent_agenda_handoff=True,
            ar_sent_agenda_weight=0.0,
            ar_multizone=False,
            ar_pattern_completion=False,
            ar_conv_pool=False,
        ),
    }
    bad_markers = ("Nghĩa", "Giải", "Ví dụ", "Loại từ", "Sirius", "ChatGPT")
    log(f"ckpt={args.checkpoint} max_len={args.max_len}")
    for mode, kw in modes.items():
        clear_sent_intra_cache()
        cfg = HCAConfig(
            ar_matrix_core=True,
            ar_style_payoff=False,
            hybrid_temperature=0.0,
            ar_matrix_stack_weight=0.0,
            **kw,
        )
        log(f"\n########## MODE {mode} ##########")
        for name, prompt in PROMPTS:
            t1 = time.time()
            out = generate(state, prompt, max_len=args.max_len, config=cfg, seed=42)
            text = decode_surface(state, out)
            cont = text[len(prompt) :].strip() if text.startswith(prompt) else text
            n_dot = text.count(".") + text.count("。")
            hits = [b for b in bad_markers if b in text]
            log(f"\n=== {name}  ({time.time()-t1:.1f}s) dots≈{n_dot} bad={hits} ===")
            log(f"prompt: {prompt}")
            log(f"out:    {text[:420]}")
            log(f"cont:   {cont[:280]!r}")
    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
