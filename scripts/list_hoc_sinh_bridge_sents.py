#!/usr/bin/env python
"""Quét parents: câu train gắn cầu Z cho học sinh và từng donor → Vương."""
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


def find_sents_for_prop(
    ps,
    v,
    head_toks: tuple[int, ...],
    prop: tuple[int, ...],
    *,
    limit: int = 5,
) -> list[tuple[int, str]]:
    """Tìm cha có … head_toks … ngay sau đó prop (L=1 hoặc L=2)."""
    out: list[tuple[int, str]] = []
    Lh, Lp = len(head_toks), len(prop)
    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        for i in range(0, len(e) - Lh - Lp + 1):
            if tuple(e[i : i + Lh]) != head_toks:
                continue
            if tuple(e[i + Lh : i + Lh + Lp]) != prop:
                continue
            text = " ".join(surface(v, t) for t in e)
            out.append((pi, text))
            break
        if len(out) >= limit:
            break
    return out


def main() -> None:
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / "checkpoints/network_v12_subst_isg_pw.pkl"))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)
    cfg = AnalogyBridgeConfig()
    idx = build_analogy_bridge_index(state, cfg)
    cs = idx.compound_surf

    _cont, meta = generate_analogy_bridge(state, "học sinh là", index=idx, cfg=cfg)
    a = int(meta["head"])
    pred = int(meta["pred"])
    _sc, acc_dons, acc_bridges = aggregate_scores(state, idx, a, cfg)

    # token ids học, sinh
    hoc = sinh = None
    for i in range(n):
        s = surface(v, i)
        if s == "học" and hoc is None:
            hoc = i
        if s == "sinh" and sinh is None:
            sinh = i
    head_span = (hoc, sinh)
    print(f"=== học sinh là → {meta['pred_surf']} ===")
    print(f"ISG head id={a}  token span={head_span}")
    print(f"donors: {[surface(v, d, cs) for d in sorted(acc_dons.get(pred, ()))]}")
    print()

    for key, dons, sc in sorted(acc_bridges.get(pred, ()), key=lambda t: -t[2]):
        lab = prop_label(v, key, cs)
        print("=" * 92)
        print(f"Z = {lab!r}  strength={sc:.3f}  donors={[surface(v, d, cs) for d in dons]}")

        # học sinh (span 2)
        print(f"\n[học sinh] câu có span học+sinh rồi {lab}:")
        for pi, text in find_sents_for_prop(ps, v, head_span, key, limit=4):
            print(f"  [{pi}] {text}")

        for d in dons:
            ds = surface(v, d, cs)
            # donor token: surface match first vocab id with that surface used as head
            # Prefer exact id d if token; else search span from compound_surf
            if d < n:
                d_span = (d,)
            else:
                # compound donor — skip token scan; dùng edge_sents
                print(f"\n[{ds}] (compound id={d}) edge_sents:")
                for line in idx.edge_sents.get((key, d), [])[:3]:
                    print(f"  {line}")
                continue
            print(f"\n[{ds}] câu có token rồi {lab}:")
            hits = find_sents_for_prop(ps, v, d_span, key, limit=4)
            if not hits:
                # fallback edge_sents
                for line in idx.edge_sents.get((key, d), [])[:3]:
                    print(f"  (edge) {line}")
            else:
                for pi, text in hits:
                    same = "  << cùng câu với học sinh" if any(
                        pi == p0 for p0, _ in find_sents_for_prop(ps, v, head_span, key, limit=20)
                    ) else ""
                    # cheaper same-pi check against học sinh hits above
                    print(f"  [{pi}] {text}{same}")

        print(f"\n[là {meta['pred_surf']}] của donors:")
        for d in dons:
            pis = [pi for pi, b, c in idx.triples if b == d and c == pred]
            for pi in pis[:2]:
                e = [int(t) for t in ps.parents[pi].emit_ids]
                text = " ".join(surface(v, t) for t in e)
                print(f"  [{pi}] {surface(v, d, cs)} | {text}")
        print()


if __name__ == "__main__":
    main()
