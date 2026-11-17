#!/usr/bin/env python
"""Phân tích case: học sinh là ?"""
from __future__ import annotations

import gc
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.analogy_bridge_gen import (  # noqa: E402
    AnalogyBridgeConfig,
    aggregate_scores,
    build_analogy_bridge_index,
    generate_analogy_bridge,
    prop_label,
    surface,
)


def main() -> None:
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / "checkpoints/network_v12_subst_isg_pw.pkl"))
    gc.enable()
    v, ps = state.vocab, state.parents
    cfg = AnalogyBridgeConfig()
    idx = build_analogy_bridge_index(state, cfg)
    cs = idx.compound_surf

    prompt = "học sinh là"
    _cont, meta = generate_analogy_bridge(state, prompt, index=idx, cfg=cfg)
    print("=== META ===")
    for k in (
        "stop", "head", "head_surf", "head_resolved", "pred", "pred_surf",
        "score", "n_bridges", "bridges", "donors", "lop",
    ):
        print(f"{k}: {meta.get(k)}")

    a = int(meta["head"])
    pred = int(meta["pred"])
    print(f"\nhead id={a} (ISG/cluster adj?) compound={a in cs}")
    print("n_props", len(idx.props.get(a, ())))

    _sc, acc_dons, acc_bridges = aggregate_scores(state, idx, a, cfg)
    print("\n=== ALL BRIDGES →", surface(v, pred, cs), "score", _sc.get(pred))
    for key, dons, sc in sorted(acc_bridges.get(pred, ()), key=lambda t: -t[2]):
        lab = prop_label(v, key, cs)
        print(f"\n  cầu={lab!r} strength={sc:.3f}")
        print(f"  donors={[surface(v, d, cs) for d in dons]}")
        shared = idx.edge_left_cls.get((key, a), set()) & set.union(
            *(idx.edge_left_cls.get((key, d), set()) for d in dons)
        )
        print(f"  lop_shared={sorted(shared)}")
        for hid in [a, *dons]:
            print(f"  edge[{surface(v, hid, cs)}]:")
            for line in idx.edge_sents.get((key, hid), [])[:2]:
                print("   ", line)

    print("\n=== TRIPLES donor là", surface(v, pred), "===")
    for d in sorted(acc_dons.get(pred, ())):
        pis = [pi for pi, b, c in idx.triples if b == d and c == pred]
        print(f"donor={surface(v, d, cs)!r} n_parents={len(pis)}")
        for pi in pis[:3]:
            e = [int(t) for t in ps.parents[pi].emit_ids]
            text = " ".join(surface(v, t) for t in e)
            print(f"  [{pi}] {text}")

    # top rival preds
    print("\n=== TOP preds (đối thủ) ===")
    for c, s in sorted(_sc.items(), key=lambda kv: -kv[1])[:6]:
        brs = [prop_label(v, k, cs) for k, _, _ in acc_bridges.get(c, ())[:3]]
        print(f"  {surface(v, c):<12} score={s:.2f} n={len(acc_bridges.get(c,()))} {brs}")


if __name__ == "__main__":
    main()
