"""Passive Hebbian update: ΔA += η from co-activation pairs."""

from __future__ import annotations

import math

from nn_aivie.core.adjacency import TypedAdjacency
from nn_aivie.core.edge_kinds import EdgeKind


def update_from_coactivation(
    adjacency: TypedAdjacency,
    active_ids: list[int],
    eta: float,
) -> None:
    """Outer-product style update for co-active nodes."""
    n = len(active_ids)
    for a in range(n):
        i = active_ids[a]
        for b in range(n):
            if a == b:
                continue
            j = active_ids[b]
            adjacency.add(i, j, eta, EdgeKind.CO_ACTIVATION)


def update_dopamine_feedback(
    adjacency: TypedAdjacency,
    w0: int,
    history_ids: list[int],
    r_slot_ids: list[int],
    gold_id: int,
    pred_id: int,
    eta: float,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    lambda_frame: float = 0.3,
    lambda_bind: float = 0.7,
    wrong_extra_alpha: float = 1.0,
) -> tuple[bool, bool]:
    """
  G3 dopamine POC: củng cố đường tới gold, LTD đường tới pred khi sai.

  Mirrors dual-context keys: (history[k], gold), (R_k, gold), (w0, gold).
  Returns (rewarded, punished).
  """
    history = history_ids[-len(r_slot_ids) :]
    k_len = len(history)
    scale = eta * alpha
    kind = EdgeKind.REINFORCED

    adjacency.add(w0, gold_id, scale, kind)
    if pred_id != gold_id:
        adjacency.add(w0, gold_id, scale * wrong_extra_alpha, kind)

    for k, tok_id in enumerate(history):
        d_frame = math.exp(-lambda_frame * (k_len - 1 - k))
        d_bind = math.exp(-lambda_bind * (k_len - 1 - k))
        adjacency.add(tok_id, gold_id, scale * d_frame, kind)
        adjacency.add(tok_id, gold_id, scale * d_bind * 0.5, kind)
        r_id = r_slot_ids[k] if k < len(r_slot_ids) else r_slot_ids[-1]
        adjacency.add(r_id, gold_id, scale * d_frame * 0.35, kind)

    punished = False
    if pred_id != gold_id and pred_id != w0:
        adjacency.add(w0, pred_id, -eta * beta, kind)
        punished = True

    return True, punished
