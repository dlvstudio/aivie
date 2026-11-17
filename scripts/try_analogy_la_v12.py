#!/usr/bin/env python
"""Batch analogy_la prompts trên checkpoint product (~430k cha)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.analogy_bridge_gen import (  # noqa: E402
    AnalogyBridgeConfig,
    build_analogy_bridge_index,
    generate_analogy_bridge,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402

PROMPTS = [
    "anh là", "mẹ là", "Thái là", "ông là", "bà là", "Trung là",
    "em là", "cha là", "chị là", "trai là", "cháu là",
    "Hồng là", "Sông là", "con khỉ là", "khỉ là",
    "Quan là", "Vũ là", "Đức là", "Hoàng là", "Công là",
    "Hà Nội là", "Việt Nam là", "người là", "chó là", "mèo là",
    "học sinh là",
]


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v12_subst.pkl"
    stem = Path(ckpt).stem
    out_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else _ROOT / "out" / f"analogy_la_try_{stem}.json"
    )
    t0 = time.time()
    print(f"load {ckpt} …", flush=True)
    import gc
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / ckpt))
    gc.enable()
    n_par = len(state.parents) if state.parents else 0
    print(f"loaded vocab={len(state.vocab):,} parents={n_par:,} ({time.time()-t0:.0f}s)", flush=True)

    cfg = AnalogyBridgeConfig()
    t1 = time.time()
    print("build analogy index …", flush=True)
    idx = build_analogy_bridge_index(state, cfg)
    print(
        f"index L1={idx.meta['n_l1']:,} L2={idx.meta['n_l2']:,} "
        f"triples={idx.meta['n_triples']:,} ({time.time()-t1:.0f}s)",
        flush=True,
    )

    rows = []
    print(f"\n{'prompt':<18} {'output':<36} {'cầu':<20} donors")
    print("-" * 100)
    for p in PROMPTS:
        cont, meta = generate_analogy_bridge(state, p, index=idx, cfg=cfg)
        full = list(state.vocab.encode_prompt(p)) + list(cont)
        out = decode_surface(state, full).strip()
        stop = meta.get("stop")
        row = {"prompt": p, "output": out, "meta": meta}
        rows.append(row)
        if stop != "analogy_la":
            print(f"{p:<18} {out:<36} — {stop}")
        else:
            bridges = ",".join(meta.get("bridges") or [])
            dons = ",".join(meta.get("donors") or [])
            print(
                f"{p:<18} {out:<36} {bridges:<20} {dons} "
                f"score={meta['score']:.2f} n={meta['n_bridges']}"
            )

    payload = {
        "checkpoint": ckpt,
        "vocab": len(state.vocab),
        "parents": n_par,
        "index": idx.meta,
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out_path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
