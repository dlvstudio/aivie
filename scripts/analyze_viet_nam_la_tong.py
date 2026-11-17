#!/usr/bin/env python
"""Phân tích case: Việt Nam là Tổng."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

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

    prompt = "Việt Nam là"
    _cont, meta = generate_analogy_bridge(state, prompt, index=idx, cfg=cfg)
    print("=== META ===")
    for k in (
        "stop", "head", "head_surf", "head_resolved", "pred", "pred_surf",
        "score", "n_bridges", "bridges", "donors", "lop",
    ):
        print(f"{k}: {meta.get(k)}")

    a = int(meta["head"])
    pred = int(meta["pred"])
    cs = idx.compound_surf

    print("\n=== HEAD n_props", len(idx.props.get(a, ())))
    props_a = sorted(
        idx.props.get(a, ()),
        key=lambda k: (-len(k), prop_label(v, k, cs)),
    )
    print([prop_label(v, k, cs) for k in props_a[:25]])

    _sc, acc_dons, acc_bridges = aggregate_scores(state, idx, a, cfg)
    print("\n=== BRIDGES →", surface(v, pred, cs), "score", _sc.get(pred))
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

    print("\n=== TRIPLES donor là Tổng ===")
    for d in sorted(acc_dons.get(pred, ())):
        pis = [pi for pi, b, c in idx.triples if b == d and c == pred]
        print(
            f"donor={surface(v, d, cs)!r} id={d} "
            f"n_parents={len(pis)} sample={pis[:5]}"
        )
        for pi in pis[:4]:
            e = [int(t) for t in ps.parents[pi].emit_ids]
            text = " ".join(surface(v, t) for t in e)
            print(f"  [{pi}] {text}")

    print("\n=== la_ac[Việt Nam] ===", [surface(v, c) for c in idx.la_ac.get(a, ())])
    print("n parents_a_la", len(idx.parents_a_la.get(a, ())))
    for pi in sorted(idx.parents_a_la.get(a, ()))[:8]:
        e = [int(t) for t in ps.parents[pi].emit_ids]
        text = " ".join(surface(v, t) for t in e)
        print(f"  [{pi}] {text[:180]}")


if __name__ == "__main__":
    main()
