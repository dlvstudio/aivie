"""
TK_036 S3 — skip span ledger incremental (train-time), nén khi ISG rewire cụm.

Mỗi cha: quét slot_chain (gap theo SLOT, không token thô) → cộng dồn PMI.
Khi promote_pair rewire: refresh cha bị đổi → skip endpoint tự gom cụm (Phở→Việt+Nam → Phở→VN).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from nn_aivie.core.seg_graph import CLUSTER_OFFSET, SegmentGraph, SlotRef
from nn_aivie.core.skip_span_constants import (
    SKIP_MAX_GAP,
    SKIP_MAX_RIGHT_LEN,
    SKIP_MIN_GAP,
    skip_right_tuple,
)
from nn_aivie.training.skip_span import (
    SkipKey,
    _is_skip_content_token,
    _right_spans_at,
    build_phrase_by_start,
)


@dataclass
class SkipLedger:
    """Tích lũy skip span online; materialize → ParentStore.skip_spans."""

    min_gap: int = SKIP_MIN_GAP
    max_gap: int = SKIP_MAX_GAP
    min_count: int = 4
    min_pmi: float = 1.0
    dir_ratio: float = 1.5
    promote_power: int = 10
    max_spans: int = 30_000
    max_prov_per_pair: int = 40

    pair: Counter = field(default_factory=Counter)
    gaps: dict[SkipKey, Counter] = field(default_factory=lambda: defaultdict(Counter))
    prov: dict[SkipKey, list[tuple[int, int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    a_tot: Counter = field(default_factory=Counter)
    b_tot: Counter = field(default_factory=Counter)
    total: int = 0
    _parent_hits: dict[int, list[tuple[SkipKey, int, int]]] = field(default_factory=dict)
    dirty: bool = False
    _synced: bool = False

    def clear(self) -> None:
        self.pair.clear()
        self.gaps.clear()
        self.prov.clear()
        self.a_tot.clear()
        self.b_tot.clear()
        self.total = 0
        self._parent_hits.clear()
        self.dirty = False
        self._synced = False

    def _unrecord_parent(self, pi: int) -> None:
        for key, gap, _pos in self._parent_hits.pop(pi, ()):
            self.pair[key] -= 1
            if self.pair[key] <= 0:
                self.pair.pop(key, None)
            self.gaps[key][gap] -= 1
            if self.gaps[key][gap] <= 0:
                del self.gaps[key][gap]
            if not self.gaps[key]:
                del self.gaps[key]
            a, right = key
            self.a_tot[a] -= 1
            self.b_tot[right] -= 1
            self.total -= 1
            lst = self.prov.get(key)
            if lst:
                self.prov[key] = [(p, pos, g) for p, pos, g in lst if p != pi]

    def scan_parent(
        self,
        pi: int,
        chain: tuple[SlotRef, ...],
        seg: SegmentGraph,
        vocab,
        gdf: dict[int, int],
        n_par: int,
        phrase_by_start: dict[int, list[tuple[int, ...]]],
    ) -> list[tuple[SkipKey, int, int]]:
        """Quét slot_chain — gap = số slot giữa trái và phải (sau rewire cụm)."""
        del pi
        if len(chain) < self.min_gap + 2:
            return []
        units = [tuple(seg.expand_slot(s)) for s in chain]
        # Perf: flat token stream + prefix-offset tính MỘT lần (thay vì dựng lại
        # flat + sum() trong vòng lặp đôi si×sj — O(len²·gap) → O(len·gap)).
        flat: list[int] = []
        prefix: list[int] = [0] * (len(units) + 1)
        for k, u in enumerate(units):
            flat.extend(u)
            prefix[k + 1] = prefix[k] + len(u)
        hits: list[tuple[SkipKey, int, int]] = []

        def _slot_ok(si: int) -> bool:
            toks = units[si]
            return bool(toks) and _is_skip_content_token(toks[0], vocab, gdf, n_par)

        def _right_at_slot(sj: int) -> tuple[int, ...] | None:
            ref = chain[sj]
            if ref.kind == "cluster":
                c = seg.clusters[ref.id - CLUSTER_OFFSET]
                ph = (int(c.left), int(c.right))
                if 2 <= len(ph) <= SKIP_MAX_RIGHT_LEN:
                    return ph
                return None
            rights = _right_spans_at(flat, prefix[sj], vocab, gdf, n_par, phrase_by_start)
            if rights:
                return rights[-1]
            return None

        for si in range(len(chain)):
            if not _slot_ok(si):
                continue
            left = units[si][0]
            off_i = prefix[si]
            j_lo = si + 1
            j_hi = min(len(chain), si + self.max_gap + 1)
            for sj in range(j_lo, j_hi):
                right = _right_at_slot(sj)
                if right is None or len(right) < 2:
                    continue
                off_j = prefix[sj]
                if chain[sj].kind == "cluster":
                    # Khoảng cách theo SLOT sau rewire (Phở→VN = 1)
                    gap = sj - si
                    if gap < 1 or gap > self.max_gap:
                        continue
                else:
                    gap = off_j - off_i - 1
                    if gap < self.min_gap or gap > self.max_gap:
                        continue
                hits.append(((left, right), gap, si))
        return hits

    def record_parent(
        self,
        pi: int,
        ps,
        vocab,
        *,
        phrase_by_start: dict | None = None,
    ) -> None:
        """Ghi lại skip hits của một cha (sau add / rewire)."""
        self._unrecord_parent(pi)
        rec = ps.parents[pi]
        chain = rec.slot_chain
        if not chain:
            self.dirty = True
            return
        n_par = len(ps.parents)
        if phrase_by_start is None:
            phrase_by_start = build_phrase_by_start(ps, min_count=self.min_count)
        scanned = self.scan_parent(
            pi, chain, ps.seg, vocab, ps._gdf, n_par, phrase_by_start,
        )
        # Mỗi cha đóng góp tối đa 1 lần cho cùng cặp (left, right); gap = min trong cha.
        best: dict[SkipKey, tuple[int, int]] = {}
        for key, gap, pos in scanned:
            prev = best.get(key)
            if prev is None or gap < prev[0]:
                best[key] = (gap, pos)
        stored: list[tuple[SkipKey, int, int]] = []
        for key, (gap, pos) in best.items():
            self.pair[key] += 1
            self.gaps[key][gap] += 1
            a, right = key
            self.a_tot[a] += 1
            self.b_tot[right] += 1
            self.total += 1
            if len(self.prov[key]) < self.max_prov_per_pair:
                self.prov[key].append((pi, pos, gap))
            stored.append((key, gap, pos))
        self._parent_hits[pi] = stored
        self.dirty = True
        self._synced = True

    def rebuild_all(self, ps, vocab, *, logger=None) -> None:
        """Bootstrap ledger từ mọi cha (tương đương mine batch, giữ ledger)."""
        self.clear()
        n = len(ps.parents)
        for pi in range(n):
            self.record_parent(pi, ps, vocab)
            if logger is not None and n >= 5000 and (pi + 1) % 5000 == 0:
                logger.progress("skip_ledger", pi + 1, n)

    def ensure_synced(self, ps, vocab) -> None:
        """Checkpoint cũ có skip_spans nhưng chưa ledger — rebuild một lần."""
        if self._synced:
            return
        self.rebuild_all(ps, vocab)

    def materialize(
        self, ps, vocab, *, logger=None, phrase_by_start=None, reuse_hits=None,
    ) -> dict[str, int]:
        """PMI rank → ghi ParentStore.skip_spans + skip_span_ids trên từng cha.

        Fix #4: `reuse_hits` = tập cha VỪA record_parent trong CÙNG flush (cùng
        phrase_by_start + chain hiện tại) → tái dùng `_parent_hits` thay vì scan_parent
        lại (byte-identical: cùng key set). Cha ngoài tập vẫn scan (phrase có thể đã đổi)."""
        if not self.dirty and ps.skip_spans:
            return {"spans": len(ps.skip_spans), "skipped": True}

        def _log(msg: str) -> None:
            if logger is not None:
                logger.log(msg)

        total = max(self.total, 1)
        ranked: list[tuple[float, SkipKey, int]] = []
        for key, c in self.pair.items():
            if c < self.min_count:
                continue
            a, right = key
            if len(right) == 1:
                bwd = self.pair.get((right[0], (a,)), 0)
                if bwd > 0 and c < self.dir_ratio * bwd:
                    continue
            pmi = math.log((c * total) / (max(self.a_tot[a], 1) * max(self.b_tot[right], 1)))
            if pmi < self.min_pmi:
                continue
            ranked.append((pmi * math.log(1 + c), key, c))

        ranked.sort(reverse=True)
        ranked = ranked[: self.max_spans]
        _log(f"skip ledger cặp thô={len(self.pair)} → spans={len(ranked)}")

        spans: list[dict] = []
        lookup: dict[SkipKey, int] = {}
        index: dict[int, list[int]] = defaultdict(list)
        meta: dict[SkipKey, dict] = {}

        for _sc, (a, right), c in ranked:
            pmi = math.log((c * total) / (max(self.a_tot[a], 1) * max(self.b_tot[right], 1)))
            si = len(spans)
            gh = dict(self.gaps[(a, right)])
            rec = {
                "left": a,
                "right": right,
                "power": c,
                "pmi": pmi,
                "gaps": gh,
                "prov": list(self.prov[(a, right)]),
                "promoted": c >= self.promote_power,
            }
            spans.append(rec)
            lookup[(a, right)] = si
            meta[(a, right)] = rec
            for pi, _pos, _g in self.prov[(a, right)]:
                if pi not in index[si]:
                    index[si].append(pi)

        for si in index:
            index[si].sort()

        n_par = len(ps.parents)
        if phrase_by_start is None:
            phrase_by_start = build_phrase_by_start(ps, min_count=self.min_count)
        if logger is not None:
            _log(f"    skip_ledger gán skip_span_ids 0/{n_par}")
        step = 25 if n_par >= 200 else 10
        reused = 0
        for pi, prec in enumerate(ps.parents):
            chain = prec.slot_chain
            if not chain:
                prec.skip_span_ids = ()
                continue
            hits: list[int] = []
            seen: set[int] = set()
            if reuse_hits is not None and pi in reuse_hits:
                scanned = self._parent_hits.get(pi, ())   # đã có từ record_parent (cùng flush)
                reused += 1
            else:
                scanned = self.scan_parent(
                    pi, chain, ps.seg, vocab, ps._gdf, n_par, phrase_by_start,
                )
            for key, _gap, _pos in scanned:
                si = lookup.get(key)
                if si is not None and si not in seen:
                    seen.add(si)
                    hits.append(si)
            prec.skip_span_ids = tuple(sorted(hits))
            if logger is not None and (pi + 1) % step == 0:
                logger.progress("skip_ledger gán skip_span_ids", pi + 1, n_par)
        if logger is not None and reuse_hits is not None:
            _log(f"    skip_ledger gán: reuse {reused}/{n_par} (khỏi scan lại)")

        ps.skip_spans = spans
        ps._skip_lookup = lookup
        ps._rebuild_skip_left_index()
        ps._skip_index = dict(index)
        ps.skip_span_meta = meta
        ps._skip_df = [len(index.get(i, ())) for i in range(len(spans))]
        self.dirty = False

        n_prom = sum(1 for s in spans if s.get("promoted"))
        n_phrase = sum(1 for s in spans if len(skip_right_tuple(s["right"])) > 1)
        _log(f"✓ skip_ledger materialize spans={len(spans)} promoted={n_prom}")
        return {
            "pairs": len(self.pair),
            "spans": len(spans),
            "promoted": n_prom,
            "phrase_right": n_phrase,
        }
