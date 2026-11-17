#!/usr/bin/env python
"""So sánh analogy_la try: isg_pw vs bản cũ."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
new = json.loads((ROOT / "out/analogy_la_try_network_v12_subst_isg_pw.json").read_text())
old_path = ROOT / "out/analogy_la_try_v12.json"
old = json.loads(old_path.read_text()) if old_path.exists() else None

n_fill = sum(1 for r in new["rows"] if r["meta"].get("stop") == "analogy_la")
print(f"NEW {new['checkpoint']} fill {n_fill}/{len(new['rows'])}")
print("index", {k: new["index"].get(k) for k in (
    "n_l1", "n_l2", "n_triples", "n_head_cluster", "n_head_entity",
    "n_prop_keys_dropped_df",
)})
if old:
    o_fill = sum(1 for r in old["rows"] if r["meta"].get("stop") == "analogy_la")
    print(f"OLD {old.get('checkpoint')} fill {o_fill}/{len(old['rows'])}")

print()
print(f"{'prompt':<16} {'NEW':<28} {'OLD':<28} note")
print("-" * 90)
old_by = {r["prompt"]: r for r in (old["rows"] if old else [])}
for r in new["rows"]:
    p = r["prompt"]
    no = r["output"]
    ns = r["meta"].get("stop")
    oo = old_by[p]["output"] if p in old_by else "—"
    os_ = old_by[p]["meta"].get("stop") if p in old_by else None
    note = ""
    if p in old_by:
        if ns == "analogy_la" and os_ != "analogy_la":
            note = "+fill"
        elif ns != "analogy_la" and os_ == "analogy_la":
            note = "-fill"
        elif no != oo and ns == "analogy_la":
            note = "~pred"
    elif ns == "analogy_la":
        note = "new"
    hs = r["meta"].get("head_surf") or ""
    nbr = r["meta"].get("n_bridges")
    print(f"{p:<16} {no:<28} {oo:<28} {note} head={hs!r} n={nbr}")

hub = [r for r in new["rows"] if "cùng" in r["output"]]
print(f"\nhub 'cùng' in NEW outputs: {len(hub)}")
