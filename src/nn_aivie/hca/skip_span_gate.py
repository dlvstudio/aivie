"""
TK_036 — skip-span gate cho A3: boost cụm đích xa khi w0 = left và agenda có right phrase.

Song song FRAME_BRACKET (từ khung) nhưng cho content skip (phở→(Việt, Nam)).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nn_aivie.core.skip_span_constants import skip_right_tuple
from nn_aivie.core.state import GraphState
from nn_aivie.hca.readout import multiply_scores_by_boost
from nn_aivie.training.skip_span import mine_skip_spans
from nn_aivie.hca.objcache import cache_get, cache_put

if TYPE_CHECKING:
    from nn_aivie.hca.config import HCAConfig

# §12.30: `set[id]` cùng lỗi — id tái dùng ⇒ tưởng ĐÃ mine ⇒ bỏ qua ÂM THẦM.
_mined_skip: dict = {}


def ensure_skip_spans(state: GraphState, *, logger=None) -> int:
    """Lazy-mine skip spans nếu checkpoint chưa có."""
    if cache_get(_mined_skip, state) is not None:
        return len(getattr(state.parents, "skip_spans", None) or [])
    ps = state.parents
    if ps is None:
        return 0
    if getattr(ps, "skip_spans", None):
        cache_put(_mined_skip, state, True)
        return len(ps.skip_spans)
    ledger = getattr(ps, "skip_ledger", None)
    if ledger is not None and (ledger.dirty or ledger.pair):
        ledger.materialize(ps, state.vocab, logger=logger)
        cache_put(_mined_skip, state, True)
        return len(ps.skip_spans or [])
    ckpt = getattr(state, "_checkpoint_path", None)
    if ckpt:
        from nn_aivie.core.skip_span_cache import (
            load_skip_spans_sidecar,
            save_skip_spans_sidecar,
        )

        if not load_skip_spans_sidecar(ps, ckpt, logger=logger):
            if logger is not None:
                logger.log(
                    "skip_spans: checkpoint thiếu — mine lazy "
                    "(chạy scripts/bake_checkpoint_indices.py để bake)"
                )
            mine_skip_spans(state, logger=logger)
            if ps.skip_spans:
                save_skip_spans_sidecar(ps, ckpt)
    else:
        if logger is not None:
            logger.log("skip_spans: mine lazy (không có checkpoint path)")
        mine_skip_spans(state, logger=logger)
    cache_put(_mined_skip, state, True)
    return len(ps.skip_spans or [])


def apply_skip_span_gate(
    scores: np.ndarray,
    state: GraphState,
    out: list[int],
    pending_agenda: list[int],
    agenda_tokens: list[int],
    config: HCAConfig,
    vocab_size: int,
    active_parents: frozenset[int] | None = None,
    gate_trace: dict | None = None,
) -> bool:
    """Boost token đầu cụm right khi w0=left và agenda khớp cả cụm right."""
    w = getattr(config, "ar_skip_span_weight", 0.0)
    if w <= 0 or not out or not pending_agenda:
        return False
    ps = state.parents
    if ps is None:
        return False
    meta = getattr(ps, "skip_span_meta", None) or {}
    left_idx = getattr(ps, "_skip_left_index", None) or {}
    if not meta and not left_idx:
        return False
    if not left_idx and getattr(ps, "_skip_lookup", None):
        ps._rebuild_skip_left_index()
        left_idx = ps._skip_left_index

    w0 = out[-1]
    boost = np.zeros(vocab_size, dtype=np.float64)
    fired = False
    best_left = -1
    best_right = -1
    best_gap = 0

    for right_ids, _si in left_idx.get(w0, ()):
        L = len(right_ids)
        for i in pending_agenda[:8]:
            if i + L > len(agenda_tokens):
                continue
            if tuple(agenda_tokens[i : i + L]) != right_ids:
                continue
            m = meta.get((w0, right_ids))
            if m is None:
                continue
            d = 1
            for j in range(len(out) - 1, -1, -1):
                if out[j] == w0:
                    d = len(out) - 1 - j
                    break
            tot = m["power"] or 1
            p_gap = m["gaps"].get(d, 0) / tot if getattr(config, "ar_skip_gap_prior", True) else 1.0
            if p_gap <= 0 and getattr(config, "ar_skip_gap_prior", True):
                p_gap = max(m["gaps"].values()) / tot if m["gaps"] else 0.25
            val = m["pmi"] * p_gap
            if len(right_ids) > 1:
                val *= 1.0 + 0.15 * (len(right_ids) - 1)
            if (
                active_parents
                and getattr(config, "ar_skip_ctx_weight", 0.0) > 0
                and any(pi in active_parents for pi, _pos, _g in m["prov"])
            ):
                val *= 1.0 + config.ar_skip_ctx_weight
            t0 = right_ids[0]
            if t0 < vocab_size:
                boost[t0] = max(boost[t0], val)
                fired = True
                best_left, best_right, best_gap = w0, t0, d

    if not fired:
        return False
    bmax = boost.max()
    if bmax <= 0:
        return False
    floor = getattr(config, "ar_skip_span_floor", 0.15)
    multiply_scores_by_boost(
        scores, boost, floor=floor, weight=w,
    )
    if gate_trace is not None:
        vocab = state.vocab
        gate_trace.update({
            "skip_left": vocab.token_of(best_left) if best_left >= 0 else None,
            "skip_right": vocab.token_of(best_right) if best_right >= 0 else None,
            "skip_gap": int(best_gap),
        })
    return True
