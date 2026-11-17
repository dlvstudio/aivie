#!/usr/bin/env python3
"""
TK_045 — bake PPMI(SENT_L2R/R2L) → Q/K sidecar (offline, không nnz-cap).

    PYTHONPATH=src python -u scripts/bake_sent_qk.py \
        --checkpoint checkpoints/network_v11_sent.pkl

Ghi: checkpoints/network_v11_sent.sent_qk.npz
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
from nn_aivie.hca.sent_intra import (  # noqa: E402
    clear_sent_intra_cache,
    ensure_sent_intra,
    save_sent_qk_sidecar,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_sent.pkl")
    ap.add_argument("--out", default="", help="default: <ckpt>.sent_qk.npz")
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()
    t0 = time.time()
    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else ckpt.with_suffix(".sent_qk.npz")

    def log(m=""):
        print(m, flush=True)

    state, _ = load_checkpoint(str(ckpt))
    state._checkpoint_path = str(ckpt)  # type: ignore[attr-defined]
    state._sent_qk_sidecar = str(out)  # type: ignore[attr-defined]
    cfg = HCAConfig()
    clear_sent_intra_cache()
    log("▶ build SENT CSR …")
    views = ensure_sent_intra(state, config=cfg, force=True)
    log(
        f"  spans={views.n_sent_spans} l2r_pairs={views.n_l2r_pairs} "
        f"r2l_pairs={views.n_r2l_pairs} agenda={views.n_agenda_edges}"
    )
    log(f"  W_l2r nnz={views.W_l2r.nnz} W_r2l nnz={views.W_r2l.nnz}")
    log(f"▶ SVD dim={args.dim} → {out}")
    save_sent_qk_sidecar(state, out, dim=int(args.dim), config=cfg, logger=log)
    log(f"[{time.time() - t0:.0f}s] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
