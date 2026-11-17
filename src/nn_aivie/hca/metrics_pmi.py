"""PMI-cross prompt↔sinh — thước TRÔI ĐỀ (§12.117), dùng trong eval/A/B.

Công thức khớp `scripts/probe_wskip_accept.py`: với mỗi token-nội-dung sinh, lấy max PMI skip
tới một token-nội-dung prompt; trung bình trên token sinh. Cao = bám đề; thấp = trôi.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from nn_aivie.core.state import GraphState
from nn_aivie.hca.context_positional import is_structural_token
from nn_aivie.hca.embedding_skip import wskip_cooccur


class PmiCrossIndex:
    """Cache đồng-hiện skip một lần / checkpoint; chấm nhiều cặp prompt–cont."""

    def __init__(
        self,
        state: GraphState,
        *,
        d_max: int = 8,
        lam: float = 0.85,
        content_only: bool = True,
    ) -> None:
        v = state.vocab
        self._v = v
        self._struct = [is_structural_token(v, i) for i in range(len(v))]
        c, u, ut = wskip_cooccur(state, d_max=d_max, lam=lam, content_only=content_only)
        self._c = c
        self._u = u
        self._ut = float(ut) or 1.0
        self._tot = float(sum(c.values())) or 1.0

    def content_ids(self, ids: Iterable[int]) -> list[int]:
        v = self._v
        out: list[int] = []
        for t in ids:
            t = int(t)
            if 0 <= t < len(v) and not self._struct[t] and v.is_surface(t):
                out.append(t)
        return out

    def pmi(self, a: int, b: int) -> float | None:
        w = self._c.get((a, b), 0.0)
        if w <= 0:
            return None
        ua, ub = self._u.get(a, 0.0), self._u.get(b, 0.0)
        q = (ua / self._ut) * (ub / self._ut)
        if q <= 0:
            return None
        x = math.log((w / self._tot) / q)
        return x if x > 0 else 0.0

    def score(self, prompt_ids: list[int], cont_ids: list[int]) -> float:
        """Trung bình max-PMI; 0 nếu thiếu nội dung. Không NaN — dễ agg."""
        P = self.content_ids(prompt_ids)
        Q = self.content_ids(cont_ids)
        if not P or not Q:
            return 0.0
        pm: list[float] = []
        for q in Q:
            best = max((self.pmi(p, q) or 0.0 for p in P), default=0.0)
            pm.append(best)
        return sum(pm) / len(pm)
