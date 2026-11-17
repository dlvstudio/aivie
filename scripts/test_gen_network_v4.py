#!/usr/bin/env python3
"""Smoke test sinh trên network_v4 vs baseline + gate on/off."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.generate_hebbian import generate, generate_autoregressive

_ROOT = Path(__file__).resolve().parents[1]

ESSAY_PROMPTS = [
    "Đọc sách giúp",
    "Mùa hè em",
    "Trường học là",
    "Hỏi: Việt Nam là gì Đáp:",
]

BRACKET_PROMPTS = [
    "nếu trời mưa",
    "tuy vất vả",
    "Nếu em học chăm",
    "Không chỉ giỏi",
]

QA_PROMPTS = [
    "Napoléon là ai",
    "Mèo là gì",
]


def _run_one(state, prompt: str, *, gate: float, max_len: int = 24) -> tuple[str, int]:
    cfg = replace(HCAConfig(), ar_frame_bracket_weight=gate)
    trace: list[dict] = []
    ids = generate(state, prompt, max_len=max_len, config=cfg, trace_out=trace)
    text = decode_surface(state, ids)
    fired = sum(1 for t in trace if t.get("frame_bracket_gate"))
    return text, fired


def _run_ar_pure(state, prompt: str, *, gate: float, max_len: int = 20) -> str:
    """A3 thuần (không parent plan) — thấy W_seq + gate trực tiếp."""
    cfg = replace(HCAConfig(), ar_frame_bracket_weight=gate)
    ids = generate_autoregressive(state, prompt, max_len=max_len, config=cfg)
    return decode_surface(state, ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v4.pkl")
    ap.add_argument("--baseline", default="checkpoints/grade_v3_big_a2_t3_isg.pkl")
    args = ap.parse_args()

    print(f"=== Load {args.checkpoint} ===")
    v4, _ = load_checkpoint(args.checkpoint)
    print(f"vocab={len(v4.vocab)} parents={len(v4.parents.parents)}")

    print("\n=== A) Sinh hybrid (generate) — gate ON vs OFF ===\n")
    for p in ESSAY_PROMPTS + BRACKET_PROMPTS + QA_PROMPTS:
        on, f_on = _run_one(v4, p, gate=0.5)
        off, f_off = _run_one(v4, p, gate=0.0)
        same = " (=)" if on == off else ""
        print(f"  [{p!r}]")
        print(f"    gate=0.5 (fb fire {f_on}): {on!r}{same}")
        if on != off:
            print(f"    gate=0.0 (fb fire {f_off}): {off!r}")

    if Path(args.baseline).exists():
        print(f"\n=== B) So sánh v4 vs baseline (_t3_isg) — gate=0.5 ===\n")
        base, _ = load_checkpoint(args.baseline)
        for p in ESSAY_PROMPTS[:2] + BRACKET_PROMPTS[:2] + QA_PROMPTS[:1]:
            t4, _ = _run_one(v4, p, gate=0.5)
            tb, _ = _run_one(base, p, gate=0.5)
            mark = " ✓" if t4 == tb else " ≠"
            print(f"  {p!r}{mark}")
            print(f"    v4:       {t4!r}")
            print(f"    baseline: {tb!r}")

    print("\n=== C) A3 thuần (W_seq + gate) — bracket prompts ===\n")
    for p in BRACKET_PROMPTS:
        on = _run_ar_pure(v4, p, gate=0.5)
        off = _run_ar_pure(v4, p, gate=0.0)
        print(f"  {p!r}")
        print(f"    gate=0.5: {on!r}")
        if on != off:
            print(f"    gate=0.0: {off!r}")


if __name__ == "__main__":
    main()
