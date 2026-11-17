"""B1 — batch G1 coactivation + W_seq: tích lũy delta, flush 1 lần cuối batch."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from nn_aivie.core.adjacency import TypedAdjacency
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.hebb import update_from_coactivation
from nn_aivie.core.state import GraphState


@dataclass
class G1BatchBuffer:
    co: Counter = field(default_factory=Counter)
    seq: Counter = field(default_factory=Counter)

    def accumulate(self, ids: list[int], eta: float, *, window: int = 12) -> None:
        co_window_starts(len(ids), window, out=self._co_starts)
        for start in self._co_starts:
            win = ids[start : start + window]
            if len(win) < 2:
                continue
            n = len(win)
            for a in range(n):
                i = win[a]
                for b in range(n):
                    if a == b:
                        continue
                    self.co[(i, win[b])] += eta
        for a, b in zip(ids, ids[1:]):
            self.seq[(a, b)] += eta

    # reuse list buffer — tránh allocate mỗi dòng dài
    _co_starts: list[int] = field(default_factory=list, init=False, repr=False)

    def flush(self, adjacency: TypedAdjacency) -> None:
        if self.co:
            adjacency.add_many(self.co, EdgeKind.CO_ACTIVATION)
        if self.seq:
            adjacency.add_many(self.seq, EdgeKind.SEQUENTIAL)
        self.co.clear()
        self.seq.clear()


# Trên text dài, co-window O(len×window²) — cap số vị trí quét (W_seq vẫn full).
MAX_CO_WINDOW_STARTS = 512


def co_window_starts(n_ids: int, window: int, *, out: list[int] | None = None) -> list[int]:
    """Vị trí start cho co-activation sliding window; subsample nếu text dài."""
    if n_ids < 2:
        return []
    max_start = n_ids - 1
    if max_start <= MAX_CO_WINDOW_STARTS:
        starts = range(max_start + 1)
    else:
        step = max(1, max_start // MAX_CO_WINDOW_STARTS)
        starts = range(0, max_start + 1, step)
    if out is not None:
        out.clear()
        out.extend(starts)
        return out
    return list(starts)


def begin_g1_batch(state: GraphState) -> None:
    """Bắt đầu tích lũy G1 — gọi trước vòng ingest batch."""
    state._g1_batch = G1BatchBuffer()


def flush_g1_batch(state: GraphState, *, logger=None) -> None:
    """Ghi delta G1 xuống adjacency; invalidate readout mỗi hàng 1 lần."""
    buf: G1BatchBuffer | None = getattr(state, "_g1_batch", None)
    if buf is None:
        return
    if logger is not None:
        logger.log(
            f"    flush_g1 co={len(buf.co)} seq={len(buf.seq)} …"
        )
    t0 = time.monotonic()
    buf.flush(state.adjacency)
    state._g1_batch = None
    if logger is not None:
        logger.log(f"✓ flush_g1  ({time.monotonic() - t0:.1f}s)")


def g1_and_wseq(
    state: GraphState, ids: list[int], eta: float, *, window: int = 12
) -> None:
    """Co-occurrence window + W_seq. Batch nếu `begin_g1_batch` đã gọi."""
    buf: G1BatchBuffer | None = getattr(state, "_g1_batch", None)
    if buf is not None:
        buf.accumulate(ids, eta, window=window)
        return
    for start in co_window_starts(len(ids), window):
        win = ids[start : start + window]
        if len(win) >= 2:
            update_from_coactivation(state.adjacency, win, eta)
    for a, b in zip(ids, ids[1:]):
        state.adjacency.add(a, b, eta, EdgeKind.SEQUENTIAL)


# Alias cho test / script cũ
_g1_and_wseq = g1_and_wseq
