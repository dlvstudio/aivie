"""
NN-AIVIE v2 — Unsupervised Neurogenesis & Hierarchical Association.

Discovers co-firing cliques of surface tokens and recruits Concept nodes
to represent latent abstract categories. Implements Neuron Capital for sparsity
and prevents "light neurons" (stop words) from polluting abstractions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from nn_aivie.core.state import GraphState
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.vocab import is_surface_token

logger = logging.getLogger(__name__)


def is_light_neuron(
    node_id: int,
    adjacency,
    vocab,
    max_connections: int = 15,
    avg_weight_threshold: float = 0.08,
) -> bool:
    """
    Identifies a "light neuron" (nơ-ron nhẹ) based on Note 04/01/2011.
    A light neuron has a large number of connections, but most of them are weak.
    Also covers stop words by checking high connectivity with low average weight.
    """
    # Check CO_ACTIVATION edges
    co_edges = adjacency.layer(EdgeKind.CO_ACTIVATION).row(node_id)
    if not co_edges:
        return False
    
    num_conn = len(co_edges)
    if num_conn < max_connections:
        return False
        
    avg_weight = sum(co_edges.values()) / num_conn
    if avg_weight < avg_weight_threshold:
        return True
        
    # Also explicitly check common stop words as fallback
    token = vocab.token_of(node_id).lower()
    if token in {"và", "thì", "là", "mà", "nhưng", "nếu", "của", "cho", "có", "ở", "gì", "nào", "mấy", "?"}:
        return True
        
    return False


def discover_co_firing_cliques(
    state: GraphState,
    sim_threshold: float = 0.03,
    min_clique_size: int = 2,
    max_clique_size: int = 6,
    max_cliques: int = 100,
) -> list[list[int]]:
    """
    Finds cliques of co-firing surface tokens using the Bron-Kerbosch algorithm
    applied to the CO_ACTIVATION edge layer.
    """
    adj = state.adjacency
    vocab = state.vocab
    co_layer = adj.layer(EdgeKind.CO_ACTIVATION)
    
    # 1. Gather eligible surface nodes (excluding structural and light neurons)
    eligible_nodes = []
    for node_id in range(len(vocab)):
        token = vocab.token_of(node_id)
        if not is_surface_token(token):
            continue
        if is_light_neuron(node_id, adj, vocab):
            continue
        eligible_nodes.append(node_id)
        
    if not eligible_nodes:
        return []
        
    # 2. Build adjacency list of connections exceeding threshold
    adj_list: dict[int, set[int]] = defaultdict(set)
    for i in eligible_nodes:
        row_i = co_layer.row(i)
        for j in eligible_nodes:
            if i == j:
                continue
            w_ij = row_i.get(j, 0.0)
            if w_ij >= sim_threshold:
                adj_list[i].add(j)
                
    # 3. Bron-Kerbosch clique finding
    cliques: list[list[int]] = []
    
    def bron_kerbosch(r: set[int], p: set[int], x: set[int]):
        if len(cliques) >= max_cliques:
            return
        if not p and not x:
            if len(r) >= min_clique_size and len(r) <= max_clique_size:
                cliques.append(list(r))
            return
        
        # Simple pivot optimization to reduce recursion tree
        pivot = next(iter(p | x)) if (p | x) else None
        candidates = p - adj_list[pivot] if pivot is not None else p
        
        for v in list(candidates):
            bron_kerbosch(
                r | {v},
                p & adj_list[v],
                x & adj_list[v]
            )
            p.remove(v)
            x.add(v)

    bron_kerbosch(set(), set(eligible_nodes), set())
    
    # Sort cliques by size descending and remove duplicates/sub-cliques
    cliques.sort(key=len, reverse=True)
    unique_cliques: list[list[int]] = []
    seen_nodes: set[int] = set()
    
    for c in cliques:
        # Avoid heavy overlaps to ensure distinct concepts
        num_overlap = len(set(c) & seen_nodes)
        if num_overlap > len(c) // 2:
            continue
        unique_cliques.append(c)
        seen_nodes.update(c)
        if len(unique_cliques) >= max_cliques:
            break
            
    return unique_cliques


def run_neurogenesis_step(
    state: GraphState,
    *,
    sim_threshold: float = 0.03,
    min_clique_size: int = 2,
    max_clique_size: int = 6,
    max_new_concepts: int = 30,
    w_init: float = 1.0,
    inherit_scale: float = 0.45,
) -> dict[str, int]:
    """
    Executes a neurogenesis pass: discovers cliques, spawns Concept nodes,
    creates bidirectional Hebbian connections, and performs hierarchical
    association (inheritance of common targets).
    """
    cliques = discover_co_firing_cliques(
        state,
        sim_threshold=sim_threshold,
        min_clique_size=min_clique_size,
        max_clique_size=max_clique_size,
        max_cliques=max_new_concepts,
    )
    
    adj = state.adjacency
    vocab = state.vocab
    
    stats = {
        "cliques_found": len(cliques),
        "concepts_spawned": 0,
        "bottom_up_edges": 0,
        "top_down_edges": 0,
        "inherited_edges": 0,
    }
    
    for c in cliques:
        # Determine existing concept nodes or spawn a new one
        concept_idx = stats["concepts_spawned"]
        concept_name = f"Concept:{len(vocab):04d}"
        concept_id = vocab.id_of(concept_name)
        stats["concepts_spawned"] += 1
        
        # Step A: Create Bidirectional Connections (Note 16/01/2011 & Note 07/02/2011)
        # 1. Bottom-Up Co-activation: members -> Concept
        # 2. Top-Down Reinforced: Concept -> members
        for node_id in c:
            # Bottom-up (Dương - Yang flow)
            adj.set(node_id, concept_id, w_init, EdgeKind.CO_ACTIVATION)
            stats["bottom_up_edges"] += 1
            
            # Top-down (Âm - Yin flow)
            adj.set(concept_id, node_id, w_init * 0.8, EdgeKind.REINFORCED)
            stats["top_down_edges"] += 1
            
        # Step B: Hierarchical Association (Inheritance of common targets)
        # Find outputs/targets strongly connected (REINFORCED) from all or most clique members
        reinf_layer = adj.layer(EdgeKind.REINFORCED)
        target_weights: dict[int, list[float]] = defaultdict(list)
        
        for node_id in c:
            row = reinf_layer.row(node_id)
            for target_id, w in row.items():
                if w > 0.05:
                    target_weights[target_id].append(w)
                    
        for target_id, weights in target_weights.items():
            # If at least 50% of the clique members share this target
            if len(weights) >= max(2, len(c) // 2):
                avg_w = sum(weights) / len(weights)
                inherited_w = avg_w * inherit_scale
                adj.set(concept_id, target_id, inherited_w, EdgeKind.REINFORCED)
                stats["inherited_edges"] += 1
                
    if stats["concepts_spawned"] > 0:
        logger.info(
            f"Neurogenesis complete: spawned {stats['concepts_spawned']} concept nodes, "
            f"added {stats['bottom_up_edges'] + stats['top_down_edges'] + stats['inherited_edges']} hierarchical edges."
        )
        
    return stats
