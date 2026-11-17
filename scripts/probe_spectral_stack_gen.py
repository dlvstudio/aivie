#!/usr/bin/env python3
"""
Gen FULL đoạn — A3 vs spectral stack (tranche-1 forward).

    PYTHONPATH=src python -u scripts/probe_spectral_stack_gen.py \
        --log-file out/probe_spectral_stack_gen.log
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

# Prefix sát agenda/fact (đo continuation — không phải lệnh “viết đoạn”).
CASES = [
    (
        "Ngô Quyền là vị vua sáng lập nhà Ngô, được",
        ["Ngô", "vua", "Bạch Đằng"],
    ),
    (
        "Trần Hưng Đạo tên thật là Trần Quốc Tuấn, là",
        ["Trần", "tướng", "Nguyên"],
    ),
]


def _strip(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()
    return text


def _fact_cover(text: str, keys: list[str]) -> float:
    if not keys:
        return 1.0
    return sum(1 for k in keys if k.lower() in text.lower()) / len(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", default="out/probe_spectral_stack_gen.log")
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--d", type=int, default=24)
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    ckpt = _ROOT / args.checkpoint
    sidecar = ckpt.with_suffix(".spectral_stack.npz")
    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(ckpt))
    gc.enable()
    state._spectral_stack_sidecar = str(sidecar)  # type: ignore[attr-defined]
    state._checkpoint_path = str(ckpt)  # type: ignore[attr-defined]
    from nn_aivie.hca.context_positional import ensure_varlm

    vlm = ensure_varlm(
        state,
        fallbacks=[
            ckpt.with_suffix(".varlm.pkl"),
            _ROOT / "checkpoints/network_v11.varlm.pkl",
        ],
    )
    log(
        f"[{time.time()-t0:.0f}s] N={len(state.vocab)}  "
        f"varlm={'yes max_ctx='+str(vlm.max_ctx) if vlm else 'NO'}  "
        f"ngram+distant+positional"
    )

    modes = [
        (
            "no_agenda / A3",
            dict(
                ar_no_agenda=True,
                ar_spectral_stack=False,
                ar_multizone=False,
                ar_pattern_completion=False,
                ar_conv_pool=False,
            ),
        ),
        (
            "no_agenda / stack",
            dict(
                ar_no_agenda=True,
                ar_spectral_stack=True,
                ar_spectral_stack_weight=0.55,
                ar_spectral_stack_d=args.d,
                ar_multizone=False,
                ar_pattern_completion=False,
                ar_conv_pool=False,
            ),
        ),
        (
            "MZ / A3",
            dict(
                ar_no_agenda=False,
                ar_spectral_stack=False,
                ar_multizone=True,
                ar_verbatim_fallback=False,
                essay_ar_min_cov=0.0,
            ),
        ),
        (
            "MZ / stack",
            dict(
                ar_no_agenda=False,
                ar_spectral_stack=True,
                ar_spectral_stack_weight=0.55,
                ar_spectral_stack_d=args.d,
                ar_multizone=True,
                ar_verbatim_fallback=False,
                essay_ar_min_cov=0.0,
            ),
        ),
    ]

    for prompt, facts in CASES:
        log("\n" + "=" * 60)
        log(repr(prompt))
        log("=" * 60)
        for name, kw in modes:
            cfg = HCAConfig(ar_style_payoff=False, **kw)
            ids = generate(state, prompt, max_len=args.max_len, config=cfg)
            full = decode_surface(state, ids)
            cont = _strip(full, prompt)
            cover = _fact_cover(full, facts)
            log(f"\n--- {name}  fact={cover:.2f} ---")
            log((prompt + cont)[:900] if cont else full[:900] or "(rỗng)")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
