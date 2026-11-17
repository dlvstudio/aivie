"""Export checkpoint subgraph to JSON for Graph UI."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import is_rule_token, is_surface_token, is_var_token
from nn_aivie.viz.kinds import node_kind


def list_rule_names(state: GraphState) -> list[str]:
    names = [
        state.vocab.token_of(i)[5:]
        for i in range(len(state.vocab))
        if is_rule_token(state.vocab.token_of(i))
    ]
    return sorted(names)


def _rule_node_id(state: GraphState, rule_name: str) -> int:
    key = rule_name if rule_name.startswith("Rule:") else f"Rule:{rule_name}"
    return state.vocab._token_to_id[key]


_EDGE_KIND_PRIORITY: dict[str, int] = {
    EdgeKind.COLLOCATION.value: 0,
    EdgeKind.ANCHOR.value: 1,
    EdgeKind.GRAMMAR.value: 2,
    EdgeKind.REINFORCED.value: 3,
    EdgeKind.POSITION.value: 4,
    EdgeKind.SECONDARY.value: 5,
    EdgeKind.CO_ACTIVATION.value: 6,
    EdgeKind.EMERGENT.value: 7,
    EdgeKind.ANALOGY.value: 8,  # §180 — loại suy khái niệm (TK_054); ưu tiên thấp nhất
}


def _set_typed_edge(
    edge_map: dict[tuple[int, int], tuple[float, str]],
    i: int,
    j: int,
    w: float,
    kind: EdgeKind,
) -> None:
    key = (i, j)
    pri = _EDGE_KIND_PRIORITY.get(kind.value, 99)
    if key in edge_map:
        old_pri = _EDGE_KIND_PRIORITY.get(edge_map[key][1], 99)
        if old_pri <= pri:
            return
    edge_map[key] = (float(w), kind.value)


def _dominant_edge(
    state: GraphState, i: int, j: int, min_weight: float
) -> tuple[float, str] | None:
    best_w = 0.0
    best_kind = EdgeKind.GRAMMAR.value
    for k in EdgeKind:
        kw = state.adjacency.get(i, j, k)
        if abs(kw) >= min_weight and abs(kw) > abs(best_w):
            best_w = kw
            best_kind = k.value
    if abs(best_w) < min_weight:
        return None
    return best_w, best_kind


def _expand_collocation_neighbors(
    state: GraphState,
    include: set[int],
    *,
    min_weight: float,
    max_extra: int = 12,
    rule_ids: list[int] | None = None,
) -> None:
    """Add surface partners linked by rule-scoped COLLOCATION."""
    added = 0
    scope = set(rule_ids or state.adjacency.rule_collocation_rule_ids())
    for rule_id, i, j, w in state.adjacency.rule_collocation_items():
        if rule_id not in scope or abs(w) < min_weight:
            continue
        if i in include and j not in include and is_surface_token(
            state.vocab.token_of(j)
        ):
            include.add(j)
            added += 1
        elif j in include and i not in include and is_surface_token(
            state.vocab.token_of(i)
        ):
            include.add(i)
            added += 1
        if added >= max_extra:
            return


def structural_subgraph_for_rules(
    state: GraphState,
    rule_names: list[str],
    *,
    min_weight: float = 0.2,
    max_surface_per_var: int = 6,
) -> dict[str, Any]:
    """
    Subgraph cấu trúc teach: Rule ở giữa, cạnh trực tiếp tới Var/Lbl/const.
    Surface chỉ lấy top imprint mỗi Var (tránh nổ node).
    """
    valid = set(list_rule_names(state))
    rule_ids: list[int] = []
    for name in rule_names:
        if name in valid:
            rule_ids.append(_rule_node_id(state, name))

    if not rule_ids:
        return {
            "meta": {
                "rules": rule_names,
                "mode": "structural",
                "n_nodes": 0,
                "n_edges": 0,
                "min_weight": min_weight,
            },
            "nodes": [],
            "edges": [],
        }

    include: set[int] = set(rule_ids)
    var_ids: list[int] = []

    for rid in rule_ids:
        for nid, w in state.adjacency.row(rid).items():
            if abs(w) < min_weight:
                continue
            tok = state.vocab.token_of(nid)
            k = node_kind(tok)
            if k in ("rule", "var", "label", "position", "surface"):
                include.add(nid)
                if is_var_token(tok):
                    var_ids.append(nid)
        for i, j, w in state.adjacency.items():
            if j != rid or abs(w) < min_weight:
                continue
            tok = state.vocab.token_of(i)
            k = node_kind(tok)
            if k in ("rule", "var", "label", "position", "surface"):
                include.add(i)
                if is_var_token(tok):
                    var_ids.append(i)

    for vid in set(var_ids):
        surfaces: list[tuple[int, float]] = []
        for nid, w in state.adjacency.row(vid).items():
            if not is_surface_token(state.vocab.token_of(nid)):
                continue
            if abs(w) >= min_weight:
                surfaces.append((nid, abs(w)))
        surfaces.sort(key=lambda x: -x[1])
        for nid, _ in surfaces[:max_surface_per_var]:
            include.add(nid)

    _expand_collocation_neighbors(
        state, include, min_weight=min_weight, rule_ids=rule_ids
    )

    edge_map: dict[tuple[int, int], tuple[float, str]] = {}
    for i in include:
        for j in state.adjacency.row(i):
            if j not in include:
                continue
            dom = _dominant_edge(state, i, j, min_weight)
            if dom:
                edge_map[(i, j)] = dom

    for i, j, w, kind in state.adjacency.typed_items():
        if kind != EdgeKind.CO_ACTIVATION:
            continue
        if abs(w) < min_weight:
            continue
        if i in include and j in include:
            ti, tj = state.vocab.token_of(i), state.vocab.token_of(j)
            if not (is_surface_token(ti) and is_surface_token(tj)):
                continue
            _set_typed_edge(edge_map, i, j, w, kind)

    for rule_id, i, j, w in state.adjacency.rule_collocation_items():
        if rule_id not in rule_ids or abs(w) < min_weight:
            continue
        if i in include and j in include:
            _set_typed_edge(edge_map, i, j, w, EdgeKind.COLLOCATION)

    nodes = [
        {
            "id": nid,
            "label": state.vocab.token_of(nid),
            "kind": node_kind(state.vocab.token_of(nid)),
        }
        for nid in sorted(include)
    ]
    edges = [
        {"source": i, "target": j, "weight": float(w), "kind": kind}
        for (i, j), (w, kind) in sorted(edge_map.items(), key=lambda x: -abs(x[1][0]))
    ]

    return {
        "meta": {
            "rules": rule_names,
            "mode": "structural",
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "min_weight": min_weight,
        },
        "nodes": nodes,
        "edges": edges,
    }


def subgraph_for_rules(
    state: GraphState,
    rule_names: list[str],
    *,
    min_weight: float = 0.15,
    max_edges: int = 800,
    hops: int = 2,
) -> dict[str, Any]:
    """BFS subgraph from rule node(s), bounded by hops and max_edges."""
    seeds: set[int] = set()
    for name in rule_names:
        if name in list_rule_names(state):
            seeds.add(_rule_node_id(state, name))

    if not seeds:
        return {
            "meta": {
                "rules": rule_names,
                "n_nodes": 0,
                "n_edges": 0,
                "min_weight": min_weight,
                "hops": hops,
            },
            "nodes": [],
            "edges": [],
        }

    seen_nodes: set[int] = set(seeds)
    edge_map: dict[tuple[int, int], float] = {}
    frontier: deque[tuple[int, int]] = deque((s, 0) for s in seeds)

    while frontier and len(edge_map) < max_edges:
        nid, depth = frontier.popleft()
        if depth >= hops:
            continue
        for j, w in state.adjacency.row(nid).items():
            if abs(w) < min_weight:
                continue
            key = (nid, j)
            if key not in edge_map:
                edge_map[key] = w
            if j not in seen_nodes:
                seen_nodes.add(j)
                frontier.append((j, depth + 1))
        for i, j, w in state.adjacency.items():
            if j != nid or abs(w) < min_weight:
                continue
            key = (i, j)
            if key not in edge_map:
                edge_map[key] = w
            if i not in seen_nodes:
                seen_nodes.add(i)
                frontier.append((i, depth + 1))

    nodes = []
    for nid in sorted(seen_nodes):
        tok = state.vocab.token_of(nid)
        nodes.append(
            {
                "id": nid,
                "label": tok,
                "kind": node_kind(tok),
            }
        )

    edges = [
        {"source": i, "target": j, "weight": float(w)}
        for (i, j), w in sorted(edge_map.items(), key=lambda x: -abs(x[1]))
    ]

    return {
        "meta": {
            "rules": rule_names,
            "mode": "bfs",
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "min_weight": min_weight,
            "hops": hops,
        },
        "nodes": nodes,
        "edges": edges,
    }


def subgraph_for_tokens(
    state: GraphState,
    tokens: list[str],
    *,
    kinds: list[EdgeKind] | None = None,
    min_weight: float = 0.0,
    max_edges: int = 400,
    hops: int = 1,
) -> dict[str, Any]:
    """§180 — subgraph quanh TOKEN (không phải rule), duyệt cạnh TYPED.

    Dành cho xem cạnh LOẠI SUY (ANALOGY/REINFORCED) vốn là token→token, thứ view
    rule-centric (`export_graph_json`) không chạm tới. Giải bề mặt token → nút, BFS
    `hops` qua các `kinds` (mặc định analogy+reinforced), chặn bởi `max_edges`.
    """
    from nn_aivie.hca.hub_spread import resolve_surface_ids

    if kinds is None:
        kinds = [EdgeKind.ANALOGY, EdgeKind.REINFORCED]
    seeds: list[int] = []
    for surf in tokens:
        seeds.extend(int(x) for x in resolve_surface_ids(state.vocab, surf))
    seeds = list(dict.fromkeys(seeds))  # unique, giữ thứ tự

    edge_map: dict[tuple[int, int], tuple[float, str]] = {}
    include: set[int] = set(seeds)
    frontier = list(seeds)
    for _ in range(max(1, hops)):
        nxt: list[int] = []
        for u in frontier:
            for kind in kinds:
                for v, w in state.adjacency.row(u, kind).items():
                    if abs(w) < min_weight:
                        continue
                    _set_typed_edge(edge_map, u, int(v), float(w), kind)
                    if int(v) not in include:
                        include.add(int(v))
                        nxt.append(int(v))
                    if len(edge_map) >= max_edges:
                        break
            if len(edge_map) >= max_edges:
                break
        if len(edge_map) >= max_edges:
            break
        frontier = nxt

    nodes = [
        {
            "id": nid,
            "label": state.vocab.token_of(nid),
            "kind": node_kind(state.vocab.token_of(nid)),
            "seed": nid in seeds,
        }
        for nid in sorted(include)
    ]
    edges = [
        {"source": i, "target": j, "weight": float(w), "kind": kind}
        for (i, j), (w, kind) in sorted(edge_map.items(), key=lambda x: -abs(x[1][0]))
    ]
    return {
        "meta": {
            "tokens": tokens,
            "seeds": seeds,
            "kinds": [k.value for k in kinds],
            "mode": "token",
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "hops": hops,
        },
        "nodes": nodes,
        "edges": edges,
    }


def export_graph_json(
    state: GraphState,
    rule_names: list[str],
    *,
    min_weight: float = 0.15,
    max_edges: int = 800,
    hops: int = 2,
    mode: str = "structural",
    checkpoint: str | None = None,
) -> dict[str, Any]:
    if mode == "structural":
        payload = structural_subgraph_for_rules(
            state, rule_names, min_weight=min_weight
        )
    else:
        payload = subgraph_for_rules(
            state,
            rule_names,
            min_weight=min_weight,
            max_edges=max_edges,
            hops=hops,
        )
    payload["meta"]["checkpoint"] = checkpoint
    return payload


def write_graph_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
