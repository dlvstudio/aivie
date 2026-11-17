#!/usr/bin/env python3
"""
PROBE style payoff — Setup→Pivot→Payoff soft vs hard.

    PYTHONPATH=src python -u scripts/probe_style_payoff.py \
        --log-file out/probe_style_payoff.log

So MZ / MZ+soft / MZ+hard. Đo fact, mean-bigram, seam cos(pivot_last→next), punch.
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
from nn_aivie.hca.style_fluency import mean_bigram_cosine, token_cosine  # noqa: E402
from nn_aivie.hca.style_payoff import plan_style_payoff  # noqa: E402

CASES = [
    (
        "Viết một đoạn văn hài hước về Ngô Quyền",
        ["Ngô Quyền"],
    ),
    (
        "Viết hài hước về Trần Hưng Đạo",
        ["Trần"],
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


def _core_hit(text: str, tok: str) -> bool:
    c = tok.lstrip("ĠĊ").lower()
    return len(c) >= 2 and c in text.lower()


def _seam_cosine(emb, vocab, text: str, pivot_ids: tuple[int, ...]) -> float:
    """cos(last pivot tok in text, token ngay sau) — mượt seam."""
    if emb is None or not pivot_ids or not text:
        return 0.0
    ids = list(vocab.encode_prompt(text))
    if len(ids) < 2:
        return 0.0
    pivot_set = set(pivot_ids)
    # tìm vị trí cuối cùng khớp pivot token
    last_piv = -1
    for i, t in enumerate(ids):
        if t in pivot_set:
            last_piv = i
    if last_piv < 0 or last_piv + 1 >= len(ids):
        # fallback: tìm core "nhưng"/"cuộc"
        for i, t in enumerate(ids):
            core = vocab.token_of(t).lstrip("ĠĊ").lower()
            if core in {"nhưng", "cuộc"}:
                last_piv = i
        if last_piv < 0 or last_piv + 1 >= len(ids):
            return 0.0
    return token_cosine(emb, ids[last_piv], ids[last_piv + 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_1k.pkl")
    ap.add_argument("--log-file", default="out/probe_style_payoff.log")
    ap.add_argument("--max-len", type=int, default=90)
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
    vocab, emb = state.vocab, state.embedding
    log(f"[{time.time()-t0:.0f}s] ✓")

    payoff_kw = dict(
        ar_style_payoff=True,
        ar_style_payoff_max_tokens=28,
        ar_style_payoff_setup_segs=1,
        ar_style_payoff_setup_max_tokens=48,
        ar_style_payoff_keep_tail=False,
        ar_style_boundary_inject=False,
    )

    for prompt, facts in CASES:
        q = list(vocab.encode_prompt(prompt))
        ents: list[int] = []
        for f in facts:
            ents.extend(vocab.encode_prompt(f))
        po = plan_style_payoff(state, q, ents)
        log(f"\n═══ {prompt!r} ═══")
        if po:
            log(f"  pivot: {po.pivot_text!r}  score={po.score:.3f}")
            log(f"  payoff: {po.payoff_text!r}")
        else:
            log("  payoff: ∅")

        for name, kw in (
            ("MZ only", dict()),
            ("MZ+soft", {**payoff_kw, "ar_style_payoff_hard_copy": False}),
            ("MZ+hard", {**payoff_kw, "ar_style_payoff_hard_copy": True}),
        ):
            cfg = HCAConfig(
                ar_verbatim_fallback=False,
                essay_ar_min_cov=0.0,
                ar_multizone=True,
                **kw,
            )
            ids = generate(state, prompt, max_len=args.max_len, config=cfg)
            text = _strip(decode_surface(state, ids), prompt)
            gen_ids = list(vocab.encode_prompt(text)) if text else []
            bigram = mean_bigram_cosine(emb, gen_ids)
            cover = _fact_cover(text, facts)
            has_pivot = po is not None and any(
                _core_hit(text, vocab.token_of(t)) for t in po.pivot
            )
            pay_hit = False
            if po is not None:
                cores = [
                    vocab.token_of(t).lstrip("ĠĊ").lower()
                    for t in po.payoff
                    if len(vocab.token_of(t).lstrip("ĠĊ")) >= 3
                ]
                pay_hit = sum(1 for c in cores if c and c in text.lower()) >= 3
            seam = 0.0
            if name != "MZ only" and po is not None:
                seam = _seam_cosine(emb, vocab, text, po.pivot)
            log(
                f"  {name:10} fact={cover:.2f} bigram={bigram:.4f} "
                f"seam={seam:.4f} pivot={has_pivot} punch={pay_hit}"
            )
            log(f"    {text[:420]}")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    log("Đạt: soft seam≥hard, fact≈1, punch có thể paraphrase.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
