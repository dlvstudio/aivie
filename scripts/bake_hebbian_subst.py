#!/usr/bin/env python
"""P108 (TK_049) — Hebbian: A~B (SYNONYM) ∧ A→P (CO) ⇒ viết B→P.

Ý tác giả: lớp thay thế mang theo tính chất (khỉ~chó ⇒ cả hai gần 'động vật').
Đường decode đa-khe dùng `_co` để xếp ứng viên theo chủ đề — kế thừa CO làm bể
chọn **sạch hơn**, không đổi công thức giao.

Không đè checkpoint gốc.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_subst.pkl")
    ap.add_argument("--inherit-scale", type=float, default=0.45)
    ap.add_argument("--max-inherit", type=int, default=8,
                    help="số đích CO kế thừa tối đa mỗi hướng của một cặp SYNONYM")
    ap.add_argument("--min-w", type=float, default=0.05)
    ap.add_argument("--out-checkpoint", default="checkpoints/network_v12_subst_h.pkl")
    ap.add_argument("--out-meta", default="out/bake_hebbian_v12_meta.json")
    args = ap.parse_args()

    logger = get_run_logger("bake_hebbian_subst")
    t0 = time.time()
    gc.disable()
    state, cache = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, adj = state.vocab, state.adjacency
    n = len(v)

    keep = [False] * n
    for t in range(n):
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        keep[t] = True

    syn = adj.layer(EdgeKind.SYNONYM)
    co = adj.layer(EdgeKind.CO_ACTIVATION)

    # cặp không hướng (a < b)
    pairs: list[tuple[int, int, float]] = []
    if getattr(syn, "_frozen", False):
        for i, j, w in syn.items():
            if i < j and w > 0:
                pairs.append((int(i), int(j), float(w)))
    else:
        seen = set()
        for i, j, w in syn.items():
            if w <= 0:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((int(a), int(b), float(w)))
    logger.log(f"cặp SYNONYM: {len(pairs):,} ({time.time()-t0:.0f}s)")

    n_inherited = 0
    n_pair_touch = 0
    for k, (a, b, w_syn) in enumerate(pairs):
        if k and k % 20000 == 0:
            logger.log(f"  hebbian {k:,}/{len(pairs):,} · inherited={n_inherited:,}")
        scale = args.inherit_scale * min(max(w_syn, 0.0), 1.0)
        touched = False
        for src, dst in ((a, b), (b, a)):
            row = co.row(src)
            if not row:
                continue
            cands = sorted(
                ((float(wp), int(p)) for p, wp in row.items()
                 if float(wp) >= args.min_w
                 and 0 <= int(p) < n and keep[int(p)]
                 and int(p) != src and int(p) != dst),
                reverse=True,
            )[: args.max_inherit]
            for wp, p in cands:
                new_w = wp * scale
                old = float(co.get(dst, p) or 0.0)
                if new_w > old:
                    adj.set(dst, p, new_w, EdgeKind.CO_ACTIVATION)
                    n_inherited += 1
                    touched = True
        if touched:
            n_pair_touch += 1

    out_ck = _ROOT / args.out_checkpoint
    out_ck.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(state, out_ck, cache=cache)
    meta = {
        "checkpoint_in": args.checkpoint,
        "checkpoint_out": args.out_checkpoint,
        "n_synonym_pairs": len(pairs),
        "n_inherited": n_inherited,
        "n_pair_touch": n_pair_touch,
        "inherit_scale": args.inherit_scale,
        "max_inherit": args.max_inherit,
        "min_w": args.min_w,
        "gate_n_inherited_ge_10000": n_inherited >= 10_000,
        "seconds": round(time.time() - t0, 1),
    }
    op = _ROOT / args.out_meta
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(
        f"✓ {out_ck} · inherited={n_inherited:,} · pairs_touch={n_pair_touch:,} "
        f"{'✅' if n_inherited >= 10_000 else '❌'} → {args.out_meta} ({time.time()-t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
