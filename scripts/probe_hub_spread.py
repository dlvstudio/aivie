#!/usr/bin/env python
"""Probe lan truyền hub W_skip: Khỉ → động/vật ≫ là → peers → nghịch…"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.hub_spread import (  # noqa: E402
    build_hub_spread_index,
    spread_analogy,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="checkpoints/network_v12_subst_isg_pw.pkl",
    )
    ap.add_argument("--seed", default="Khỉ")
    ap.add_argument("--prop-prefix", default="nghịch")
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=int, default=3)
    ap.add_argument("--out", default="out/probe_hub_spread.json")
    args = ap.parse_args()

    gc.disable()
    state, _ = load_checkpoint(str(args.checkpoint))
    gc.enable()
    idx = build_hub_spread_index(
        state, d_max=int(args.d_max), min_count=2,
    )
    result = spread_analogy(
        state,
        idx,
        args.seed,
        prop_prefix=str(args.prop_prefix),
        gidf_min=float(args.gidf_min),
    )
    result["checkpoint"] = str(args.checkpoint)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)

    # Tóm tắt so sánh là vs ý niệm
    raw_ys = [h["Y"] for h in result["hubs_raw_by_weight"][:8]]
    sc_ys = [h["Y"] for h in result["hubs_scored"][:8]]
    print("\n--- tóm tắt ---", flush=True)
    print(f"seed={args.seed!r}  raw_top={raw_ys}", flush=True)
    print(f"scored_top={sc_ys}", flush=True)
    print(
        f"'là' in raw={('là' in raw_ys)}  in scored={('là' in sc_ys)}",
        flush=True,
    )
    print(f"peers={ [p['X'] for p in result['peers'][:10]] }", flush=True)
    print(f"props={ [p['Z'] for p in result['props'][:10]] }", flush=True)


if __name__ == "__main__":
    main()
