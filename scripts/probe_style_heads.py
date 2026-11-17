#!/usr/bin/env python3
"""
PROBE FACT/STYLE heads — multizone + a_style opt-in (cho M4).

    PYTHONPATH=src python -u scripts/probe_style_heads.py \
        --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_heads.log

So 4 cấu hình mỗi prompt (verbatim fallback TẮT để thấy A3):
  OFF  — mặc định (1 fact)
  MZ   — ar_multizone=True
  STY  — ar_style_weight=0.5
  BOTH — MZ + STY

Đọc tay: MZ có ghép ≥2 entity? STY có token sắc-thái (khiếu/biếm/cười…)? Không hại fact?
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
from nn_aivie.hca.generate_hebbian import generate, _plan_from_multizone  # noqa: E402
from nn_aivie.hca.style_readout import style_boost_from_coact, style_seeds_from_prompt  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

PROMPTS = [
    "So sánh Nhà Lý và Nhà Trần",
    "Trần Hưng Đạo và trận Bạch Đằng",
    "Viết một đoạn văn hài hước về Ngô Quyền",
    "Kể buồn về mùa thu Hà Nội",
]


def _cfg(**kw) -> HCAConfig:
    c = HCAConfig(ar_verbatim_fallback=False, essay_ar_min_cov=0.0)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--style-weight", type=float, default=0.5)
    ap.add_argument("--log-file", default="out/probe_style_heads.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    log(f"[{time.time()-t0:.0f}s] ✓ parents {len(state.parents):,}")

    modes = [
        ("OFF ", _cfg()),
        ("MZ  ", _cfg(ar_multizone=True)),
        ("STY ", _cfg(ar_style_weight=args.style_weight)),
        ("BOTH", _cfg(ar_multizone=True, ar_style_weight=args.style_weight)),
    ]

    for prompt in PROMPTS:
        log(f"\n═══ {prompt!r} ═══")
        q = list(state.vocab.encode_prompt(prompt))
        seeds = style_seeds_from_prompt(q, state.vocab, state.parents)
        log(f"  style_seeds: {[state.vocab.token_of(t) for t in seeds]}")
        boost, _ = style_boost_from_coact(state, q, len(state.vocab))
        top = sorted(
            ((float(boost[j]), j) for j in range(len(boost)) if boost[j] > 0),
            reverse=True,
        )[:12]
        log(f"  style_top:   {[(state.vocab.token_of(j), round(s, 1)) for s, j in top]}")

        mz = _plan_from_multizone(
            state, q, _cfg(ar_multizone=True), emitted_ids=q,
        )
        if mz is None:
            log("  mz_plan:     ∅")
        else:
            log(f"  mz_plan:     {len(mz.segments)} seg · "
                f"ents={[state.vocab.token_of(t) for t in mz.entity_ids[:8]]}")
            for i, seg in enumerate(mz.segments):
                log(f"    seg[{i}]: {decode_surface(state, seg)[:120]!r}")

        for name, cfg in modes:
            ids = generate(state, prompt, max_len=args.max_len, config=cfg)
            text = decode_surface(state, ids)
            if text.startswith(prompt):
                text = text[len(prompt):].lstrip()
            log(f"  {name}: {text[:200]!r}")

    log(f"\n[{time.time()-t0:.0f}s] → đọc tay: MZ ghép entity? STY có sắc-thái? BOTH không salad?")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
