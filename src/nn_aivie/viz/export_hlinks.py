"""TK_020 — export CẠNH NGANG XA cho Graph UI: (PMI, power, gap_hist, provenance)."""

from __future__ import annotations

from typing import Any

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.core.frame_bracket_ref import decode_endpoint_label
from nn_aivie.core.vocab import is_surface_token


def hlinks_info(state: GraphState) -> dict[str, Any]:
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    meta = getattr(state.parents, "hlink_meta", None) or {}
    return {"edges": len(fb), "with_meta": len(meta)}


def hlinks_of(state: GraphState, token: str, *, topk: int = 12) -> dict[str, Any]:
    """Cạnh ngang từ một opener: closer + PMI + power + gap-hist + ví dụ CHA NGUỒN."""
    vocab = state.vocab
    ps = state.parents
    tid = vocab._token_to_id.get(token.strip())
    if tid is None:
        # thử fold-variant ("Nếu" ≡ "nếu")
        alt = vocab._token_to_id.get(token.strip().lower())
        tid = alt
    if tid is None:
        return {"found": False, "token": token}
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    meta = getattr(ps, "hlink_meta", None) or {}
    rows: list[dict[str, Any]] = []
    for b, pmi in sorted(fb.row(tid).items(), key=lambda kv: -kv[1])[: topk * 2]:
        m = meta.get((tid, b))
        row: dict[str, Any] = {
            "closer": decode_endpoint_label(
                b, getattr(ps, "hlink_endpoints", None) or {}, vocab
            ),
            "pmi": round(float(pmi), 2),
        }
        if m is not None:
            gaps = m["gaps"]
            tot = m["power"] or 1
            mode_gap, mode_n = max(gaps.items(), key=lambda kv: kv[1]) if gaps else (None, 0)
            row.update({
                "power": m["power"],
                "mode_gap": mode_gap,
                "p_mode": round(mode_n / tot, 2),
                "gaps": {str(g): c for g, c in sorted(gaps.items())},
                "examples": [
                    vocab.detokenize(ps.parents[pi].source_ids[:16])
                    for pi, _pos, _g in m["prov"][:3]
                    if pi < len(ps.parents)
                ],
            })
        rows.append(row)
        if len(rows) >= topk:
            break
    return {"found": True, "token": vocab.token_of(tid), "id": tid, "links": rows}
