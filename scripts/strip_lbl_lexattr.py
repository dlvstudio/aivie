#!/usr/bin/env python
"""Zero mọi LEXATTR đụng token Lbl:* (orphan vocab giữ nguyên id)."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402


def strip_lbl_lexattr(state) -> dict[str, int]:
    vocab = state.vocab
    layer = state.adjacency.layer(EdgeKind.LEXATTR)
    hubs = [
        i for i in range(len(vocab))
        if str(vocab.token_of(i)).startswith("Lbl:")
    ]
    n_del = 0
    for lid in hubs:
        for j in list(layer.row(lid).keys()):
            if float(layer.get(lid, j) or 0) > 0:
                layer.set(lid, j, 0.0)
                n_del += 1
            if float(layer.get(j, lid) or 0) > 0:
                layer.set(j, lid, 0.0)
                n_del += 1
    # residual: surface → Lbl that might only appear as col
    residual = 0
    hub_set = set(hubs)
    for i in range(len(vocab)):
        if i in hub_set:
            continue
        for j in list(layer.row(i).keys()):
            if int(j) not in hub_set:
                continue
            if float(layer.get(i, j) or 0) > 0:
                layer.set(i, j, 0.0)
                residual += 1
    return {"n_lbl_tokens": len(hubs), "deleted": n_del, "residual": residual}


def count_lbl_lexattr(state) -> int:
    vocab = state.vocab
    layer = state.adjacency.layer(EdgeKind.LEXATTR)
    n = 0
    hubs = {
        i for i in range(len(vocab))
        if str(vocab.token_of(i)).startswith("Lbl:")
    }
    for lid in hubs:
        for j, w in layer.row(lid).items():
            if float(w) > 0:
                n += 1
        # inbound counted via row of others; also check cols from non-hub
    for i in range(len(vocab)):
        if i in hubs:
            continue
        for j, w in layer.row(i).items():
            if int(j) in hubs and float(w) > 0:
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for ckpt in args.checkpoint:
        print(f"=== {ckpt} ===", flush=True)
        gc.disable()
        state, cache = load_checkpoint(ckpt)
        gc.enable()
        before = count_lbl_lexattr(state)
        stats = strip_lbl_lexattr(state)
        after = count_lbl_lexattr(state)
        print(f"before={before} after={after} stats={stats}", flush=True)
        if args.dry_run:
            print("dry-run: not saved", flush=True)
            continue
        save_checkpoint(state, ckpt, cache=cache, bake_derived=False)
        print(f"saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
