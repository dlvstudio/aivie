"""
TK_013 — Incremental Segment Graph (ISG).

Mỗi nơ-ron cha giữ chuỗi slot + cạnh ngang (parent_idx, slot_index, weight). Bigram token-token
tích lũy qua ingest; đủ ngưỡng → nơ-ron cụm + rewire toàn cục. Index nằm trên link, không
trên neuron.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from nn_aivie.core.parent_store import ParentRecord

CLUSTER_OFFSET = 2 * 10**7
ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class SlotRef:
    kind: Literal["token", "cluster"]
    id: int

    def to_json(self) -> list:
        return [self.kind, self.id]

    @classmethod
    def from_json(cls, data: list) -> "SlotRef":
        return cls(kind=data[0], id=int(data[1]))


@dataclass
class ClusterNeuron:
    left: int
    right: int
    power: int
    # TK_038: span đầy đủ khi promote compositional (cluster+token, …); rỗng = bigram (left, right)
    span: tuple[int, ...] = ()

    def token_span(self) -> tuple[int, ...]:
        return self.span if self.span else (self.left, self.right)


def _slot_pair_key(a: SlotRef, b: SlotRef) -> tuple[str, int, str, int]:
    return (a.kind, a.id, b.kind, b.id)


@dataclass
class SegLink:
    """Cạnh ngang trong một cha; slot_index 1-based."""

    src: SlotRef
    dst: SlotRef
    parent_idx: int
    slot_index: int
    weight: float


@dataclass
class SegmentGraph:
    clusters: list[ClusterNeuron] = field(default_factory=list)
    bigram_count: dict[tuple[int, int], int] = field(default_factory=dict)
    cluster_of_pair: dict[tuple[int, int], int] = field(default_factory=dict)
    # TK_038 — compositional: đếm cặp slot kề (có cluster) → cụm dài hơn
    composite_count: dict[tuple[str, int, str, int], int] = field(default_factory=dict)
    composite_of_pair: dict[tuple[str, int, str, int], int] = field(default_factory=dict)
    # Fix #3 (TK perf): membership-only occurrence index (parents chứa pair/key) —
    # KHÔNG lưu vị trí. Vị trí tính-khi-cần lúc promote (quét chain hiện tại) → bỏ
    # được refresh O(len)/merge. Superset-safe: chỉ thêm lúc register + biên merge,
    # gỡ khi pair được promote; member "chết" chỉ khiến promote quét hụt (no-op).
    _composite_members: dict[tuple[str, int, str, int], set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    _composite_source: dict[int, tuple[str, int, str, int]] = field(default_factory=dict)
    # TK_033 S1: dictionary-seed pairs (a,b) promote không cần đạt min_frequency.
    # Build-time only; runtime chỉ đọc slot_chain/links đã build (Luật 0).
    seed_pairs: set[tuple[int, int]] = field(default_factory=set)
    min_frequency: int = 3
    _links_by_parent: dict[int, list[SegLink]] = field(default_factory=dict)
    _pair_members: dict[tuple[int, int], set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    _cluster_index: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    rewired_parents: set[int] = field(default_factory=set)
    _defer_promotions: bool = False
    # Perf: trong finalize_promotions, links (seg_h) KHÔNG được đọc; hoãn rebuild
    # về một-lần-mỗi-cha ở cuối (thay vì mỗi rewire trên bài dài). link_weight chỉ
    # phụ thuộc bigram_count/power (ổn định suốt finalize) → kết quả link y hệt.
    _defer_link_rebuild: bool = False

    @property
    def links(self) -> list[SegLink]:
        return [ln for lst in self._links_by_parent.values() for ln in lst]

    def cluster_id(self, cluster_idx: int) -> int:
        return CLUSTER_OFFSET + cluster_idx

    def encode_tokens_with_clusters(self, token_ids: list[int]) -> list[int]:
        """Greedy merge token-token pairs into cluster ids when available.

        Input/Output are node ids (surface token ids + cluster ids). This is build-time / scoring-time
        only; runtime `generate()` still emits surface tokens as usual.
        """
        if not token_ids or not self.cluster_of_pair:
            return list(token_ids)
        out: list[int] = []
        i = 0
        n = len(token_ids)
        while i < n:
            a = token_ids[i]
            if i + 1 < n:
                b = token_ids[i + 1]
                ci = self.cluster_of_pair.get((a, b))
                if ci is not None:
                    out.append(self.cluster_id(ci))
                    i += 2
                    continue
            out.append(a)
            i += 1
        return out

    def cluster_slot(self, cluster_idx: int) -> SlotRef:
        return SlotRef("cluster", self.cluster_id(cluster_idx))

    @staticmethod
    def token_slot(token_id: int) -> SlotRef:
        return SlotRef("token", token_id)

    def expand_slot(self, ref: SlotRef) -> tuple[int, ...]:
        if ref.kind == "token":
            return (ref.id,)
        ci = ref.id - CLUSTER_OFFSET
        if ci < 0 or ci >= len(self.clusters):
            return ()
        return self.clusters[ci].token_span()

    def cluster_children(self, cluster_idx: int) -> tuple[tuple[int, int], ...]:
        """(token_id, link_index) — index trên link dọc cluster→token."""
        c = self.clusters[cluster_idx]
        span = c.token_span()
        return tuple((tid, i + 1) for i, tid in enumerate(span))

    def link_weight(self, src: SlotRef, dst: SlotRef) -> float:
        if src.kind == "token" and dst.kind == "token":
            return float(self.bigram_count.get((src.id, dst.id), 1))
        if dst.kind == "cluster":
            ci = dst.id - CLUSTER_OFFSET
            if 0 <= ci < len(self.clusters):
                return float(self.clusters[ci].power)
        if src.kind == "cluster":
            ci = src.id - CLUSTER_OFFSET
            if 0 <= ci < len(self.clusters):
                return float(self.clusters[ci].power)
        return 1.0

    def begin_batch(self) -> None:
        """Rebuild hàng loạt: hoãn promote đến finalize_promotions()."""
        self._defer_promotions = True

    def end_batch(
        self,
        parents: list[ParentRecord],
        *,
        progress: ProgressFn | None = None,
        log_every: int = 500,
        max_promotions: int = 0,
    ) -> int:
        """Chạy promote sau batch ingest; trả số cụm mới."""
        self._defer_promotions = False
        return self.finalize_promotions(
            parents,
            progress=progress,
            log_every=log_every,
            max_promotions=max_promotions,
        )

    def increment_parent_counts(
        self, parent_idx: int, parents: list[ParentRecord]
    ) -> None:
        """TK_038: parent trùng emit_ids — tăng bigram/composite count thêm 1."""
        chain = parents[parent_idx].slot_chain
        if not chain:
            return
        for pos in range(len(chain) - 1):
            a, b = chain[pos], chain[pos + 1]
            if a.kind == "token" and b.kind == "token":
                pair = (a.id, b.id)
                if pair in self.cluster_of_pair:
                    continue
                self.bigram_count[pair] = self.bigram_count.get(pair, 0) + 1
                self._pair_members[pair].add(parent_idx)
            else:
                key = _slot_pair_key(a, b)
                if key in self.composite_of_pair:
                    continue
                self.composite_count[key] = self.composite_count.get(key, 0) + 1
                self._composite_members[key].add(parent_idx)
        if not self._defer_promotions:
            self._check_promotions(parents)

    def ingest_parent(
        self,
        parent_idx: int,
        chain: tuple[SlotRef, ...],
        in_match: tuple[bool, ...],
        parents: list[ParentRecord],
    ) -> None:
        """Ghi chuỗi slot, đếm bigram token-token, có thể promote."""
        self.rewired_parents.clear()
        rec = parents[parent_idx]
        rec.slot_chain = chain
        rec._in_match_flags = in_match
        rec.match_span = _match_span_from_flags(in_match)
        self._rebuild_parent_links(parent_idx, chain)
        for pos in range(len(chain) - 1):
            a, b = chain[pos], chain[pos + 1]
            if a.kind == "token" and b.kind == "token":
                self._register_bigram(a.id, b.id, parent_idx, pos)
            else:
                self._register_composite(a, b, parent_idx, pos)
        if not self._defer_promotions:
            self._check_promotions(parents)

    def should_promote(self, pair: tuple[int, int]) -> bool:
        """Pair đủ điều kiện promote: (a) seed hoặc (b) count>=min_frequency; và chưa có cluster.

        TK_038 overlap: mặc định chặn cặp chồng token giữa; nới nếu seed hoặc
        ``bigram_count(pair) > max(power cụm chồng)`` (power-win, cùng tồn tại).
        """
        if pair in self.cluster_of_pair:
            return False
        if self._bigram_overlap_blocked(pair):
            if pair not in self.seed_pairs and not self._overlap_power_wins(pair):
                return False
        if pair in self.seed_pairs:
            return True
        return self.bigram_count.get(pair, 0) >= self.min_frequency

    def _ensure_overlap_index(self) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        """Chỉ mục nghịch đảo token → cụm, cho kiểm chồng biên.

        Trước đây `_bigram_overlap_blocked` / `_overlap_rival_power` quét TOÀN BỘ
        `cluster_of_pair` mỗi lần gọi; `finalize_promotions` gọi chúng một lần mỗi
        ứng viên pop ⇒ O(ứng_viên × cụm), mà số cụm còn tăng dần trong chính vòng
        lặp (v12 ~2,6·10⁵ cụm).

        KHÔNG phải field dataclass và KHÔNG vào `to_payload`: đây là view dẫn xuất
        của `cluster_of_pair` — dựng lại khi thiếu (sau `from_payload`), cập nhật
        tăng dần ở `promote_pair`.
        """
        by_right = getattr(self, "_ov_by_right", None)
        by_left = getattr(self, "_ov_by_left", None)
        if by_right is None or by_left is None:
            by_right = defaultdict(set)
            by_left = defaultdict(set)
            for (x, y), ci in self.cluster_of_pair.items():
                by_left[int(x)].add(int(ci))
                by_right[int(y)].add(int(ci))
            self._ov_by_right = by_right
            self._ov_by_left = by_left
        return by_right, by_left

    def _overlap_cluster_ids(self, pair: tuple[int, int]) -> set[int]:
        """Cụm chồng biên với `pair`: kết thúc ở `a`, hoặc bắt đầu ở `b`."""
        a, b = pair
        by_right, by_left = self._ensure_overlap_index()
        out: set[int] = set()
        left_of_a = by_right.get(a)
        if left_of_a:
            out |= left_of_a
        right_of_b = by_left.get(b)
        if right_of_b:
            out |= right_of_b
        return out

    def _bigram_overlap_blocked(self, pair: tuple[int, int]) -> bool:
        """TK_038: không 2 cụm bigram chồng token giữa (Ví,dụ) vs (dụ,mẫu)."""
        a, b = pair
        by_right, by_left = self._ensure_overlap_index()
        return bool(by_right.get(a) or by_left.get(b))

    def _overlap_rival_power(self, pair: tuple[int, int]) -> int:
        """Max ``ClusterNeuron.power`` của cụm chồng biên với ``pair`` (0 nếu không chồng)."""
        return max(
            (int(self.clusters[ci].power) for ci in self._overlap_cluster_ids(pair)),
            default=0,
        )

    def _overlap_power_wins(self, pair: tuple[int, int]) -> bool:
        """True khi count pair thắng mọi cụm overlap (strict ``>``)."""
        return self.bigram_count.get(pair, 0) > self._overlap_rival_power(pair)

    def _register_bigram(self, a: int, b: int, parent_idx: int, pos: int = 0) -> None:
        if (a, b) in self.cluster_of_pair:
            return
        pair = (a, b)
        self.bigram_count[pair] = self.bigram_count.get(pair, 0) + 1
        self._pair_members[pair].add(parent_idx)

    def _register_composite(
        self, a: SlotRef, b: SlotRef, parent_idx: int, pos: int = 0
    ) -> None:
        key = _slot_pair_key(a, b)
        if key in self.composite_of_pair:
            return
        self.composite_count[key] = self.composite_count.get(key, 0) + 1
        self._composite_members[key].add(parent_idx)

    def _deregister_composite(self, a: SlotRef, b: SlotRef) -> None:
        """Một occurrence composite (a,b) bị merge phá ở biên → giảm count.
        Giữ nguyên semantics decrement của refresh cũ (chỉ khi key chưa promote).
        Membership giữ nguyên (superset-safe)."""
        key = _slot_pair_key(a, b)
        if key in self.composite_of_pair:
            return
        c = self.composite_count.get(key, 0)
        if c > 0:
            self.composite_count[key] = c - 1
            if self.composite_count[key] <= 0:
                self.composite_count.pop(key, None)

    def should_promote_composite(self, key: tuple[str, int, str, int]) -> bool:
        if key in self.composite_of_pair:
            return False
        return self.composite_count.get(key, 0) >= self.min_frequency

    def finalize_promotions(
        self,
        parents: list[ParentRecord],
        *,
        progress: ProgressFn | None = None,
        log_every: int = 500,
        max_promotions: int = 0,
    ) -> int:
        import heapq
        import time

        # Perf: hoãn rebuild links (seg_h) trong toàn bộ finalize; rebuild
        # một-lần-mỗi-cha-rewire ở cuối (không đọc link giữa chừng).
        self._defer_link_rebuild = True
        n_bigrams = len(self.bigram_count)
        if progress:
            progress(f"scan bigrams 0/{n_bigrams} …")
        heap: list[tuple[int, tuple[int, int]]] = []
        scan_hb = time.monotonic()
        # 1) Heap theo count: gồm cả pair đang overlap-blocked (power-win quyết
        # lúc pop). Chỉ cần đủ freq hoặc seed — không gọi should_promote lúc scan.
        for i, (pair, count) in enumerate(self.bigram_count.items(), 1):
            if pair in self.cluster_of_pair:
                continue
            if pair in self.seed_pairs or count >= self.min_frequency:
                heapq.heappush(heap, (-count, pair))
            if progress and i % 500_000 == 0:
                progress(
                    f"scan bigrams {i}/{n_bigrams} heap={len(heap)}"
                )
                scan_hb = time.monotonic()
            elif progress and time.monotonic() - scan_hb >= 30.0:
                progress(
                    f"scan bigrams {i}/{n_bigrams} heap={len(heap)} …"
                )
                scan_hb = time.monotonic()
        # 2) seed pair không có count (hoặc count=0) → đưa vào cuối heap
        for pair in self.seed_pairs:
            if pair in self.bigram_count:
                continue
            if pair in self.cluster_of_pair:
                continue
            heapq.heappush(heap, (0, pair))
        if progress:
            progress(
                f"bigram heap={len(heap)} candidates — promoting …"
            )
        n = 0
        promote_hb = time.monotonic()
        tick = max(1, min(log_every, 10))
        while heap:
            if max_promotions and n >= max_promotions:
                break
            negc, pair = heapq.heappop(heap)
            count = -negc
            if pair in self.cluster_of_pair:
                continue
            # Seed pair có thể không có count trong bigram_count → bỏ check mismatch.
            if pair not in self.seed_pairs and self.bigram_count.get(pair, 0) != count:
                continue
            if not self.should_promote(pair):
                continue
            self.promote_pair(pair[0], pair[1], parents)
            n += 1
            now = time.monotonic()
            if progress and (n == 1 or n % log_every == 0 or n % tick == 0):
                progress(
                    f"promote {n} clusters={len(self.clusters)} "
                    f"pair_count={count} heap_left={len(heap)}"
                )
                promote_hb = now
            elif progress and now - promote_hb >= 30.0:
                progress(
                    f"promote {n} … clusters={len(self.clusters)} "
                    f"heap_left={len(heap)}"
                )
                promote_hb = now
        # TK_038: composite sau bigram
        n_composite = len(self.composite_count)
        if progress and n_composite:
            progress(f"scan composite 0/{n_composite} …")
        cheap: list[tuple[int, tuple[str, int, str, int]]] = []
        for i, (key, count) in enumerate(self.composite_count.items(), 1):
            if self.should_promote_composite(key):
                heapq.heappush(cheap, (-count, key))
            if progress and i % 500_000 == 0:
                progress(f"scan composite {i}/{n_composite} heap={len(cheap)}")
        if progress and cheap:
            progress(f"composite heap={len(cheap)} — promoting …")
        composite_hb = time.monotonic()
        while cheap:
            if max_promotions and n >= max_promotions:
                break
            negc, key = heapq.heappop(cheap)
            count = -negc
            if key in self.composite_of_pair:
                continue
            if self.composite_count.get(key, 0) != count:
                continue
            if not self.should_promote_composite(key):
                continue
            self.promote_composite_key(key, parents)
            n += 1
            now = time.monotonic()
            if progress and (n % tick == 0 or now - composite_hb >= 30.0):
                progress(
                    f"composite promote total={n} clusters={len(self.clusters)} "
                    f"heap_left={len(cheap)}"
                )
                composite_hb = now
        # Rebuild links một lần cho mọi cha đã rewire trong finalize này.
        self._defer_link_rebuild = False
        rebuilt = self.rewired_parents
        if progress and rebuilt:
            progress(f"rebuild_links parents={len(rebuilt)} …")
        for pi in rebuilt:
            self._rebuild_parent_links(pi, parents[pi].slot_chain)
        if progress:
            progress(f"done promotes={n} clusters={len(self.clusters)}")
        return n

    def _check_promotions(self, parents: list[ParentRecord]) -> None:
        while True:
            promoted = False
            for pair, _count in list(self.bigram_count.items()):
                if not self.should_promote(pair):
                    continue
                self.promote_pair(pair[0], pair[1], parents)
                promoted = True
                break
            if promoted:
                continue
            for pair in list(self.seed_pairs):
                if not self.should_promote(pair):
                    continue
                self.promote_pair(pair[0], pair[1], parents)
                promoted = True
                break
            if promoted:
                continue
            for key, _count in list(self.composite_count.items()):
                if not self.should_promote_composite(key):
                    continue
                self.promote_composite_key(key, parents)
                promoted = True
                break
            if not promoted:
                break

    def promote_pair(self, a: int, b: int, parents: list[ParentRecord]) -> int:
        """Tạo cụm cho bigram (a,b); rewire mọi cha chứa cặp. Trả cluster_idx.

        Fix #3: dùng membership `_pair_members` + quét vị trí on-demand trên chain
        HIỆN TẠI (không lưu/refresh vị trí). Member "chết" → quét hụt → no-op."""
        if (a, b) in self.cluster_of_pair:
            return self.cluster_of_pair[(a, b)]
        count = self.bigram_count.get((a, b), 0)
        ci = len(self.clusters)
        self.clusters.append(ClusterNeuron(left=a, right=b, power=count))
        self.cluster_of_pair[(a, b)] = ci
        # Giữ chỉ mục chồng biên đồng bộ — đây là điểm chèn DUY NHẤT của
        # `cluster_of_pair` ngoài `from_payload` (dựng lại lười ở đó).
        by_right, by_left = self._ensure_overlap_index()
        by_left[int(a)].add(int(ci))
        by_right[int(b)].add(int(ci))
        members = self._pair_members.pop((a, b), None) or ()
        for parent_idx in sorted(members):
            chain = parents[parent_idx].slot_chain
            positions = [
                pos
                for pos in range(len(chain) - 1)
                if chain[pos].kind == "token"
                and chain[pos + 1].kind == "token"
                and chain[pos].id == a
                and chain[pos + 1].id == b
            ]
            if not positions:
                continue
            for pos in sorted(positions, reverse=True):
                self._rewire_at(parent_idx, pos, ci, parents)
            self._cluster_index[ci].append(parent_idx)
        return ci

    def promote_composite_key(
        self, key: tuple[str, int, str, int], parents: list[ParentRecord]
    ) -> int:
        """TK_038: ghép 2 slot kề (có cluster) thành một cụm span dài hơn."""
        if key in self.composite_of_pair:
            return self.composite_of_pair[key]
        a = SlotRef(key[0], key[1])
        b = SlotRef(key[2], key[3])
        span = self.expand_slot(a) + self.expand_slot(b)
        if len(span) < 3:
            return self.composite_of_pair.get(key, -1)
        count = self.composite_count.get(key, 0)
        ci = len(self.clusters)
        self.clusters.append(
            ClusterNeuron(left=span[0], right=span[-1], power=count, span=span)
        )
        self.composite_of_pair[key] = ci
        self._composite_source[ci] = key
        members = self._composite_members.pop(key, None) or ()
        for parent_idx in sorted(members):
            chain = parents[parent_idx].slot_chain
            positions = [
                pos
                for pos in range(len(chain) - 1)
                if _slot_pair_key(chain[pos], chain[pos + 1]) == key
            ]
            if not positions:
                continue
            for pos in sorted(positions, reverse=True):
                self._rewire_composite_at(parent_idx, pos, ci, key, parents)
            self._cluster_index[ci].append(parent_idx)
        return ci

    def _rewire_composite_at(
        self,
        parent_idx: int,
        pos: int,
        cluster_idx: int,
        src_key: tuple[str, int, str, int],
        parents: list[ParentRecord],
    ) -> None:
        rec = parents[parent_idx]
        chain = list(rec.slot_chain)
        flags = list(rec._in_match_flags)
        if pos + 1 >= len(chain):
            return
        if _slot_pair_key(chain[pos], chain[pos + 1]) != src_key:
            return
        cslot = self.cluster_slot(cluster_idx)
        self._apply_boundary_deltas_for(parent_idx, chain, pos, cslot)
        new_flag = flags[pos] or flags[pos + 1]
        chain = chain[:pos] + [cslot] + chain[pos + 2 :]
        flags = flags[:pos] + [new_flag] + flags[pos + 2 :]
        rec.slot_chain = tuple(chain)
        rec._in_match_flags = tuple(flags)
        rec.match_span = _match_span_from_flags(rec._in_match_flags)
        if not self._defer_link_rebuild:
            self._rebuild_parent_links(parent_idx, rec.slot_chain)
        self.rewired_parents.add(parent_idx)

    def _rewire_at(
        self,
        parent_idx: int,
        pos: int,
        cluster_idx: int,
        parents: list[ParentRecord],
    ) -> None:
        rec = parents[parent_idx]
        chain = list(rec.slot_chain)
        flags = list(rec._in_match_flags)
        if pos + 1 >= len(chain):
            return
        if not (
            chain[pos].kind == "token"
            and chain[pos + 1].kind == "token"
            and chain[pos].id == self.clusters[cluster_idx].left
            and chain[pos + 1].id == self.clusters[cluster_idx].right
        ):
            return
        cslot = self.cluster_slot(cluster_idx)
        self._apply_boundary_deltas_for(parent_idx, chain, pos, cslot)
        new_flag = flags[pos] or flags[pos + 1]
        chain = chain[:pos] + [cslot] + chain[pos + 2 :]
        flags = flags[:pos] + [new_flag] + flags[pos + 2 :]
        rec.slot_chain = tuple(chain)
        rec._in_match_flags = tuple(flags)
        rec.match_span = _match_span_from_flags(rec._in_match_flags)
        if not self._defer_link_rebuild:
            self._rebuild_parent_links(parent_idx, rec.slot_chain)
        self.rewired_parents.add(parent_idx)

    def _apply_boundary_deltas_for(
        self, parent_idx: int, chain: list[SlotRef], pos: int, cslot: SlotRef
    ) -> None:
        """Như _apply_boundary_deltas nhưng gắn member vào đúng parent_idx."""
        left = chain[pos - 1] if pos > 0 else None
        right = chain[pos + 2] if pos + 2 < len(chain) else None
        a_slot = chain[pos]
        b_slot = chain[pos + 1]
        if left is not None:
            if not (left.kind == "token" and a_slot.kind == "token"):
                self._deregister_composite(left, a_slot)
            self._register_composite(left, cslot, parent_idx)
        if right is not None:
            if not (b_slot.kind == "token" and right.kind == "token"):
                self._deregister_composite(b_slot, right)
            self._register_composite(cslot, right, parent_idx)

    def _rebuild_parent_links(self, parent_idx: int, chain: tuple[SlotRef, ...]) -> None:
        new_links: list[SegLink] = []
        for i in range(len(chain) - 1):
            src, dst = chain[i], chain[i + 1]
            new_links.append(
                SegLink(
                    src=src,
                    dst=dst,
                    parent_idx=parent_idx,
                    slot_index=i + 1,
                    weight=self.link_weight(src, dst),
                )
            )
        self._links_by_parent[parent_idx] = new_links

    def parent_links(self, parent_idx: int) -> list[SegLink]:
        return self._links_by_parent.get(parent_idx, [])

    def clusters_in_match_span(self, rec: ParentRecord) -> tuple[int, ...]:
        """cluster_idx trong vùng match của cha."""
        if not rec.slot_chain or rec.match_span == (0, 0):
            return ()
        ms, me = rec.match_span
        out: list[int] = []
        for s in rec.slot_chain[ms:me]:
            if s.kind == "cluster":
                out.append(s.id - CLUSTER_OFFSET)
        return tuple(out)

    def to_payload(self) -> dict:
        # Links dạng cột numpy (kind 0=token/1=cluster) — pickle nhị phân, load nhanh
        # (thay 1.65M dict lồng list). from_payload nhận cả dạng cũ (list-of-dict). TK_011.
        import numpy as np

        links = self.links
        n = len(links)
        sk = np.empty(n, dtype=np.int8)
        si = np.empty(n, dtype=np.int32)
        dk = np.empty(n, dtype=np.int8)
        di = np.empty(n, dtype=np.int32)
        pp = np.empty(n, dtype=np.int32)
        slot = np.empty(n, dtype=np.int32)
        w = np.empty(n, dtype=np.float64)
        for k, ln in enumerate(links):
            sk[k] = 0 if ln.src.kind == "token" else 1
            si[k] = ln.src.id
            dk[k] = 0 if ln.dst.kind == "token" else 1
            di[k] = ln.dst.id
            pp[k] = ln.parent_idx
            slot[k] = ln.slot_index
            w[k] = ln.weight
        return {
            "min_frequency": self.min_frequency,
            "seed_pairs": [[a, b] for (a, b) in sorted(self.seed_pairs)],
            "clusters": [
                {
                    "left": c.left,
                    "right": c.right,
                    "power": c.power,
                    **({"span": list(c.span)} if c.span else {}),
                }
                for c in self.clusters
            ],
            "bigram_count": [[a, b, n] for (a, b), n in self.bigram_count.items()],
            "cluster_of_pair": [
                [a, b, ci] for (a, b), ci in self.cluster_of_pair.items()
            ],
            "composite_count": [
                list(key) + [n] for key, n in self.composite_count.items()
            ],
            "composite_of_pair": [
                list(key) + [ci] for key, ci in self.composite_of_pair.items()
            ],
            "links_arr": {
                "sk": sk, "si": si, "dk": dk, "di": di,
                "parent": pp, "slot": slot, "w": w,
            },
        }

    @classmethod
    def from_payload(cls, payload: dict | None) -> "SegmentGraph":
        if not payload:
            return cls()
        sg = cls(min_frequency=int(payload.get("min_frequency", 3)))
        sg.seed_pairs = {
            (int(a), int(b)) for a, b in payload.get("seed_pairs", [])
        }
        sg.clusters = [
            ClusterNeuron(
                left=int(c["left"]),
                right=int(c["right"]),
                power=int(c["power"]),
                span=tuple(int(t) for t in c.get("span", ())),
            )
            for c in payload.get("clusters", [])
        ]
        sg.bigram_count = {
            (int(a), int(b)): int(n) for a, b, n in payload.get("bigram_count", [])
        }
        sg.cluster_of_pair = {
            (int(a), int(b)): int(ci) for a, b, ci in payload.get("cluster_of_pair", [])
        }
        sg.composite_count = {
            (str(k0), int(k1), str(k2), int(k3)): int(n)
            for k0, k1, k2, k3, n in payload.get("composite_count", [])
        }
        sg.composite_of_pair = {
            (str(k0), int(k1), str(k2), int(k3)): int(ci)
            for k0, k1, k2, k3, ci in payload.get("composite_of_pair", [])
        }
        lbp = sg._links_by_parent
        la = payload.get("links_arr")
        if la is not None:  # dạng cột numpy MỚI
            _KIND = ("token", "cluster")
            sk = la["sk"].tolist(); si = la["si"].tolist()
            dk = la["dk"].tolist(); di = la["di"].tolist()
            pp = la["parent"].tolist(); slot = la["slot"].tolist(); ww = la["w"].tolist()
            for a_sk, a_si, a_dk, a_di, a_p, a_slot, a_w in zip(sk, si, dk, di, pp, slot, ww):
                link = SegLink(
                    src=SlotRef(_KIND[a_sk], a_si),
                    dst=SlotRef(_KIND[a_dk], a_di),
                    parent_idx=a_p, slot_index=a_slot, weight=a_w,
                )
                lbp.setdefault(a_p, []).append(link)
        else:  # dạng cũ: list-of-dict
            for ln in payload.get("links", []):
                link = SegLink(
                    src=SlotRef.from_json(ln["src"]),
                    dst=SlotRef.from_json(ln["dst"]),
                    parent_idx=int(ln["parent"]),
                    slot_index=int(ln["slot_index"]),
                    weight=float(ln.get("weight", 1.0)),
                )
                lbp.setdefault(link.parent_idx, []).append(link)
        sg._rebuild_pair_index_from_parents = True  # marker for lazy rebuild
        return sg

    def rebuild_inverted_index(self, parents: list[ParentRecord]) -> None:
        """Sau load checkpoint: tái tạo pair membership + cluster_index từ slot_chain.
        (Fix #3: membership thay occurrence-có-vị-trí. Composite members giữ rỗng như
        hành vi cũ — chỉ bigram được rebuild sau load.)"""
        self._pair_members = defaultdict(set)
        self._composite_members = defaultdict(set)
        self._cluster_index = defaultdict(list)
        for pi, rec in enumerate(parents):
            chain = rec.slot_chain
            if not chain:
                continue
            for pos in range(len(chain) - 1):
                a, b = chain[pos], chain[pos + 1]
                if a.kind == "token" and b.kind == "token":
                    pair = (a.id, b.id)
                    if pair not in self.cluster_of_pair:
                        self._pair_members[pair].add(pi)
            for s in chain:
                if s.kind == "cluster":
                    ci = s.id - CLUSTER_OFFSET
                    self._cluster_index[ci].append(pi)


def _match_span_from_flags(flags: tuple[bool, ...]) -> tuple[int, int]:
    idxs = [i for i, m in enumerate(flags) if m]
    if not idxs:
        return (0, 0)
    return (idxs[0], idxs[-1] + 1)


def build_chain_from_ids(
    ids: list[int],
    *,
    dap_idx: int | None,
    vocab,
    is_match_token,
) -> tuple[tuple[SlotRef, ...], tuple[bool, ...]]:
    """Chuỗi slot token ban đầu + cờ vùng match (câu hỏi / nội dung)."""
    slots = tuple(SegmentGraph.token_slot(i) for i in ids)
    flags: list[bool] = []
    for idx, tid in enumerate(ids):
        if dap_idx is not None and idx >= dap_idx:
            flags.append(False)
        else:
            flags.append(is_match_token(tid, vocab))
    return slots, tuple(flags)
