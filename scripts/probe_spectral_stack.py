#!/usr/bin/env python3
"""
PROBE spectral stack — residual SVD 4 tầng (không gán nghề tầng).

    PYTHONPATH=src python -u scripts/probe_spectral_stack.py \
        --log-file out/probe_spectral_stack.log

Đo: trực giao Wa/Uffn; held-out successor rank per tầng (khám phá);
đọc tay khuôn U/V; gen fact+humor stack vs A3.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.spectral_stack import (  # noqa: E402
    ensure_spectral_stack,
    layer_orthogonality,
    top_patterns,
    wa_successor_scores,
)

CASES = [
    ("Viết một đoạn văn hài hước về Ngô Quyền", ["Ngô Quyền"]),
    ("Viết hài hước về Trần Hưng Đạo", ["Trần"]),
]


def _strip(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()
    return text


def _fact_cover(text: str, keys: list[str]) -> float:
    if not keys:
        return 1.0
    return sum(1 for k in keys if k.lower() in text.lower()) / len(keys)


def _sample_wseq_pairs(state, n_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Lấy cặp (src, dst) từ W_seq có trọng số > 0."""
    layer = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    pairs: list[tuple[int, int]] = []
    n = len(state.vocab)
    # lấy mẫu hàng ngẫu nhiên có nnz
    trials = 0
    while len(pairs) < n_pairs and trials < n_pairs * 40:
        trials += 1
        i = int(rng.integers(0, n))
        row = layer.row(i)
        if not row:
            continue
        items = [(j, w) for j, w in row.items() if j < n and w > 0]
        if not items:
            continue
        weights = np.array([w for _, w in items], dtype=np.float64)
        weights = weights / weights.sum()
        j = int(items[int(rng.choice(len(items), p=weights))][0])
        pairs.append((i, j))
    return pairs


def _mean_rank(scores: np.ndarray, true_j: int) -> float:
    """Rank 1 = tốt nhất (score cao nhất)."""
    if true_j < 0 or true_j >= scores.shape[0]:
        return float(scores.shape[0])
    sj = float(scores[true_j])
    # số token có score > sj (+0.5 ties)
    better = int(np.sum(scores > sj))
    ties = int(np.sum(scores == sj))
    return better + 0.5 * (ties + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", default="out/probe_spectral_stack.log")
    ap.add_argument("--max-len", type=int, default=90)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
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
    log(f"[{time.time()-t0:.0f}s] ✓ N={len(state.vocab)}")

    log(f"\n── build / load spectral stack d={args.d} ──")
    stack = ensure_spectral_stack(
        state,
        d=args.d,
        sidecar=sidecar,
        force_rebuild=args.rebuild,
        logger=log,
    )

    ortho = layer_orthogonality(stack)
    log("\n── orthogonality (‖W_iᵀ W_j‖F / norms; ~0 = tốt) ──")
    for k, v in sorted(ortho.items()):
        log(f"  {k}: {v:.4f}")

    rng = np.random.default_rng(args.seed)
    pairs = _sample_wseq_pairs(state, args.heldout, rng)
    log(f"\n── held-out successor mean-rank (n={len(pairs)}; thấp hơn = tốt hơn) ──")
    log("  (khám phá — không presuppose tầng nào = cú pháp)")
    # baseline: W_seq thô
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    ranks_raw: list[float] = []
    for i, j in pairs:
        row = seq.row(i)
        sc = np.zeros(len(state.vocab), dtype=np.float64)
        for v, w in row.items():
            if v < sc.shape[0]:
                sc[v] = w
        ranks_raw.append(_mean_rank(sc, j))
    log(f"  W_seq raw: mean_rank={np.mean(ranks_raw):.1f}")
    for ℓ in range(len(stack.layers)):
        ranks: list[float] = []
        for i, j in pairs:
            sc = wa_successor_scores(stack, ℓ, i)
            ranks.append(_mean_rank(sc, j))
        log(f"  Wa_L{ℓ}:   mean_rank={np.mean(ranks):.1f}")

    log("\n── top khuôn U/V mỗi tầng (đọc tay — mô tả SAU đo) ──")
    for ℓ in range(len(stack.layers)):
        log(f"  L{ℓ}:")
        for a, b, s in top_patterns(stack, state.vocab, ℓ, k=5):
            log(f"    {a!r} → {b!r}  ({s:.3f})")

    log("\n── gen: KHÔNG agenda (ar_no_agenda) — A3 micro vs +spectral_stack ──")
    for prompt, facts in CASES:
        log(f"\n═══ {prompt!r} ═══")
        for name, kw in (
            ("A3", dict(ar_spectral_stack=False)),
            (
                "stack",
                dict(
                    ar_spectral_stack=True,
                    ar_spectral_stack_weight=0.55,
                    ar_spectral_stack_d=args.d,
                ),
            ),
        ):
            cfg = HCAConfig(
                ar_no_agenda=True,
                ar_verbatim_fallback=False,
                essay_ar_min_cov=0.0,
                ar_multizone=False,
                ar_style_payoff=False,
                ar_pattern_completion=False,
                ar_conv_pool=False,
                **kw,
            )
            ids = generate(state, prompt, max_len=args.max_len, config=cfg)
            text = _strip(decode_surface(state, ids), prompt)
            cover = _fact_cover(text, facts)
            log(f"  {name:6} fact={cover:.2f}")
            log(f"    {text[:420]}")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    log("Đạt: không agenda → stack phải ĐỔI chữ vs A3 (có thể salad); fact không còn bắt buộc.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
