"""
Parent-neuron store (NN-AIVIE v3, docx §1) — "1 ví dụ = 1 nơ-ron cha, rồi phân rã".

Mỗi ví dụ train (câu/cụm/đoạn) → MỘT nơ-ron cha giữ **con có thứ tự (index)** + **power**
(tần suất). Đây là **A phân cấp** đúng nghĩa v3, dùng cho:
  - lan xuôi (bottom-up): prompt → nơ-ron cha khớp nhiều con nhất tích năng lượng cao nhất;
  - lan ngược (pattern completion): đọc con của cha theo index, dội ra phần CÒN THIẾU (đáp).

Phân rã BPE/Re-Pair (docx §2.1): cụm con lặp ≥ min_frequency → nơ-ron trung gian (chia sẻ giữa
các cha → tổng quát hóa liên-mẫu). Khớp mờ paraphrase dùng embedding E (docx §2.2).

KHÔNG dạy ngữ pháp, KHÔNG hardcode từ: cha/con/power đều từ **dữ liệu thật**. Frame `Đáp:` chỉ
là mốc tách hỏi/đáp (README Mục 0 cho phép frame protocol).
"""

from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace

# TK_025: ReadoutMode (EXPRESSION/KNOWLEDGE) đã gỡ cùng WH-detector (Luật 0).

from nn_aivie.core.seg_graph import (
    CLUSTER_OFFSET,
    ProgressFn,
    SegmentGraph,
    SlotRef,
    _match_span_from_flags,
    build_chain_from_ids,
)
from nn_aivie.core.skip_span_constants import SKIP_MAX_GAP, SKIP_MIN_GAP, skip_right_tuple
from nn_aivie.core.vocab import Vocabulary, is_surface_token


def _repair_grammar(
    sequences: list[list[int]],
    *,
    min_frequency: int,
    max_merges: int,
    first_new_id: int,
) -> tuple[dict[int, tuple[int, int]], list[list[int]]]:
    """
    Re-Pair đệ quy (Larsson & Moffat) — gộp cặp symbol lặp nhiều nhất → symbol mới, lặp lại
    → **hệ phân cấp cụm-của-cụm**. Cài bằng linked-list + heap lazy → O(tổng số lần xuất hiện).

    Trả (rules: symbol → (trái, phải), sequences nén). Symbol < first_new_id là token gốc.
    """
    SENT = -1
    sym: list[int] = []
    for s in sequences:
        sym.extend(s)
        sym.append(SENT)
    n = len(sym)
    if n == 0:
        return {}, [list(s) for s in sequences]
    nxt = list(range(1, n + 1))
    nxt[-1] = -1
    prv = list(range(-1, n - 1))
    alive = [True] * n

    pair_pos: dict[tuple[int, int], set[int]] = defaultdict(set)
    pair_cnt: dict[tuple[int, int], int] = defaultdict(int)
    for i in range(n - 1):
        if sym[i] != SENT and sym[i + 1] != SENT:
            p = (sym[i], sym[i + 1])
            pair_pos[p].add(i)
            pair_cnt[p] += 1
    heap = [(-c, p) for p, c in pair_cnt.items()]
    heapq.heapify(heap)

    rules: dict[int, tuple[int, int]] = {}
    rid = first_new_id
    merges = 0
    while heap and merges < max_merges:
        negc, (a, b) = heapq.heappop(heap)
        if pair_cnt.get((a, b), 0) != -negc:
            continue  # entry cũ (lazy)
        if -negc < min_frequency:
            break
        R = rid
        rid += 1
        rules[R] = (a, b)
        merges += 1
        for i in list(pair_pos.get((a, b), ())):
            j = nxt[i]
            if not (alive[i] and j != -1 and alive[j] and sym[i] == a and sym[j] == b):
                continue
            k = nxt[j]
            h = prv[i]
            pair_pos[(a, b)].discard(i)
            pair_cnt[(a, b)] -= 1
            if h != -1 and alive[h] and sym[h] != SENT:
                pair_cnt[(sym[h], a)] -= 1
                pair_pos[(sym[h], a)].discard(h)
                pair_cnt[(sym[h], R)] += 1
                pair_pos[(sym[h], R)].add(h)
                heapq.heappush(heap, (-pair_cnt[(sym[h], R)], (sym[h], R)))
            if k != -1 and alive[k] and sym[k] != SENT:
                pair_cnt[(b, sym[k])] -= 1
                pair_pos[(b, sym[k])].discard(j)
                pair_cnt[(R, sym[k])] += 1
                pair_pos[(R, sym[k])].add(i)
                heapq.heappush(heap, (-pair_cnt[(R, sym[k])], (R, sym[k])))
            sym[i] = R
            alive[j] = False
            nxt[i] = k
            if k != -1:
                prv[k] = i

    out: list[list[int]] = []
    cur: list[int] = []
    for i in range(n):
        if not alive[i]:
            continue
        if sym[i] == SENT:
            out.append(cur)
            cur = []
        else:
            cur.append(sym[i])
    return rules, out


# Token frame protocol (README Mục 0 cho phép) — KHÔNG phải nội dung: có ở MỌI câu QA nên
# khớp lan xuôi trên chúng chỉ tạo tín hiệu giả ("Cao Bằng là gì" khớp "Huế là gì" qua Hỏi/là/gì).
_FRAME_TOKENS = {"Hỏi:", "Đáp:"}


def _content_ids(ids: list[int], vocab: Vocabulary, *, skip: set[int]) -> tuple[int, ...]:
    out = []
    for i in ids:
        if i in skip:
            continue
        tok = vocab.token_of(i)
        if tok in _FRAME_TOKENS:
            continue
        if is_surface_token(tok):
            out.append(i)
    return tuple(out)


@dataclass(frozen=True)
class QueryMatchUnits:
    """Phân tầng query: cụm trung gian (không chồng lấn) + skip xa + token rời."""

    phrase_units: tuple[int, ...]
    skip_units: tuple[int, ...]
    bare_tokens: tuple[int, ...]
    phrase_covered_tokens: frozenset[int]
    idf_total: float


# TK_045 §13.15 — β mặc định cho phá-thế-hoà Hopfield trong `rank_matches`.
# Chọn 32: thắng trên mini (+2.25pp); trên v12 ba giá trị 8/32/128 đều ĐẠT
# (+8.50/+8.63/+8.88pp) nên chênh lệch nằm trong nhiễu — lấy giá trị GIỮA thay vì
# giá trị tốt nhất của một checkpoint (tránh over-fit vào v12). 0 = tắt.
HOPFIELD_TIEBREAK_BETA = 32.0


@dataclass
class ParentRecord:
    """Một nơ-ron cha: chuỗi con có thứ tự + power + (nếu QA) mốc Đáp:."""

    emit_ids: tuple[int, ...]      # toàn chuỗi con theo index (cái sẽ dội ra)
    match_ids: tuple[int, ...]     # token nội dung dùng để khớp lan xuôi
    answer_ids: tuple[int, ...]    # phần đáp (sau Đáp:) — QA; rỗng nếu là câu phát biểu
    power: int = 1
    is_qa: bool = False
    phrase_ids: tuple[int, ...] = ()  # nơ-ron trung gian (cụm con lặp) chứa trong cha
    skip_span_ids: tuple[int, ...] = ()  # TK_036: cặp content xa (left→right) trong cha
    priority: float = 1.0  # độ tin nguồn: curated=1.0, ngoài miền (wiki)=thấp hơn
    # TK_010 — tầng LINK PHỤ của t1 (episode = cha thật): các fact (cha khác) tìm bằng cộng
    # hưởng từ list token, sắp theo VỊ TRÍ xuất hiện trong chuỗi → (fact_idx, pos). Quan hệ
    # ngang fact↔fact suy từ thứ tự pos. Cha thường: rỗng.
    fact_links: tuple[tuple[int, int], ...] = ()
    # TK_013 — ids gốc khi ingest (toàn câu) — dùng replay seg graph.
    source_ids: tuple[int, ...] = ()
    # TK_013 — chuỗi slot sau rewire (token hoặc cluster); match_span = slice vào slot_chain.
    slot_chain: tuple[SlotRef, ...] = ()
    match_span: tuple[int, int] = (0, 0)
    _in_match_flags: tuple[bool, ...] = ()
    # TK_012: nhãn nguồn ("alpaca:0") ĐÃ TÁCH sang content-cache sidecar (core/content_cache.py).
    # KHÔNG lưu trong nơ-ron nữa — mạng thuần id/cạnh/trọng số. Cache chỉ cho UI/eval/debug.


@dataclass
class ThemeNeuron:
    """
    Nơ-ron chủ đề `Concept:entity` (v4 P1) — vật chất hóa "mọi fact VỀ một thực thể".

    v3 coi mỗi câu là cha độc lập; "các fact về Napoléon" chỉ tồn tại ngầm qua quét
    `facts_about` mỗi lần hỏi. Theme neuron gom chúng thành MỘT nơ-ron: fact list kèm
    **trọng số theme** = thực thể đứng đầu câu thì câu đó NÓI VỀ nó (topic-first tiếng Việt
    — thuần vị trí, cùng nguyên tắc Luật 0 với xếp hạng focus `IDF÷(1+vị_trí)` W3).
    Phân biệt "câu về X" (X là chủ đề: "Napoléon trốn khỏi Elba") với "câu nhắc X"
    (X trong ngữ phụ: "cuộc chiến tranh chống Napoléon của vua Gustav…").

    Là xương sống composition (A1 đọc fact từ đây thay vì quét phẳng) và điểm neo cho
    thuộc tính diễn ngôn của thực thể sau này (đại từ/animacy — P2 lần 2).
    """

    entity: int                              # token id định danh thực thể
    facts: tuple[tuple[int, float], ...]     # (parent_idx, theme_weight ∈ (0,1])
    power: int = 0                           # tổng power fact — độ "quen" của concept


@dataclass
class AbstractNeuron:
    """
    Nơ-ron trừu tượng t1 — EPISODE (v4 Trụ 7, ý tưởng gốc tác giả).

    Một khối diễn ngôn hoàn chỉnh (đoạn văn wiki / bài gold SFT) = MỘT nơ-ron; thành viên là
    các nơ-ron cha (câu) thuộc nó, giữ **thứ tự trình bày** + trọng số. Hai đường nối:
    - **xuất xứ** (train-data có doc_id/para): câu nằm trong đoạn → member, w=1.0;
    - **cộng hưởng** (cơ chế tác giả mô tả): kích hoạt toàn bộ token của đoạn mới → cha nào
      "rung" qua rank_matches trên đồ thị hiện có → nối vào t1 với w=điểm khớp.
    Vai trò: agenda essay (A4), coherence episode khi ghép câu, ngữ cảnh đồng-episode cho
    enrichment nghĩa (T3). Khác Theme (chỉ mục thực-thể→fact): t1 là ký ức tình tiết.
    """

    # TK_012: key/title/source ĐÃ TÁCH sang content-cache sidecar (episode idx → EpisodeContent).
    # Nơ-ron episode thuần: chỉ members (id+trọng số) + match_ids (token khớp query→t1).
    members: tuple[tuple[int, float], ...] = ()  # (parent_idx, w) theo THỨ TỰ trong đoạn
    # T2 (TK_004): chìa khớp query→t1 cho episode SFT — token instruction ("Hãy viết một bài
    # về lợi ích đọc sách"). Episode wiki để rỗng (chưa dùng làm đích khớp trực tiếp).
    match_ids: tuple[int, ...] = ()


@dataclass
class SlotHub:
    """
    Nơ-ron mẫu chung (docx §"Cross-sample Phase" / W_struct) — ví dụ `n_quốc_gia_thủ_đô`.

    Nhiều cha có **cùng skeleton, chỉ lệch đúng 1 vị trí** (vd "Thủ đô của ___ là gì" với
    ___ ∈ {Mỹ, Pháp, Anh}) → tự nổi một hub: khung cố định (`left` + `right`) + **tập filler**
    (thực thể thay vào slot). Filler mới (vd Ý) được **nối vào hub NGAY TRONG TRAIN** qua pha
    quét chéo: bất kỳ câu train nào chứa `left X (right)` → X là filler. KHÔNG hardcode từ vựng:
    skeleton = cụm lặp giữa nhiều mẫu, filler = token phân biệt (IDF cao) tại vị trí biến thiên.
    """

    left: tuple[int, ...]           # token khung TRƯỚC slot (theo index)
    right: tuple[int, ...]          # token khung SAU slot (có thể rỗng)
    fillers: dict[int, int] = field(default_factory=dict)  # filler_id → power (tần suất)
    source_parents: tuple[int, ...] = ()  # cha QA đã tạo hub (để route đáp sau này)

    @property
    def signature(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (self.left, self.right)


@dataclass
class ParentStore:
    parents: list[ParentRecord] = field(default_factory=list)
    # đảo chỉ mục: content token → danh sách parent idx (lan xuôi nhanh)
    _index: dict[int, list[int]] = field(default_factory=dict)
    # document-frequency cho IDF khớp (token phổ biến → trọng số thấp) — CHỈ curated (không để
    # wiki pha loãng độ phân biệt token trong miền; README 0.5.8). Dùng cho retrieval `rank_matches`.
    _df: dict[int, int] = field(default_factory=dict)
    # df TOÀN CỤC (mọi nguồn, gồm wiki) — CHỈ dùng chọn focus/entity (W3): phân biệt "thực thể
    # wiki" (gdf>0, df=0 → không bị lọc oan) với "nhiễu chỉ-trong-câu-hỏi" (gdf=0). KHÔNG dùng cho
    # retrieval để không pha loãng IDF trong miền.
    _gdf: dict[int, int] = field(default_factory=dict)
    _key_to_idx: dict[tuple, int] = field(default_factory=dict)
    # --- nơ-ron trung gian (phân rã BPE/Re-Pair) ---
    phrases: list[tuple[int, ...]] = field(default_factory=list)   # cụm con (token ids)
    phrase_df: list[int] = field(default_factory=list)             # #cha chứa cụm
    _phrase_lookup: dict[tuple, int] = field(default_factory=dict)
    _phrase_max_len: int = 4
    # cụm → cha (lan xuôi phân tầng; token thuộc cụm không còn trong _index)
    _phrase_index: dict[int, list[int]] = field(default_factory=dict)
    # --- nơ-ron mẫu chung (slot hub — docx Cross-sample Phase / W_struct) ---
    hubs: list[SlotHub] = field(default_factory=list)
    # --- nơ-ron trừu tượng t1 (episode — v4 Trụ 7) ---
    episodes: list[AbstractNeuron] = field(default_factory=list)
    _parent_episodes: dict[int, list[int]] = field(default_factory=dict)
    # TK_013 — phân rã tăng dần qua link ngang
    seg: SegmentGraph = field(default_factory=SegmentGraph)
    seg_min_frequency: int = 3
    # Hướng 1 (TK_014) — cờ runtime (KHÔNG serialize): admit cụm phổ biến (WH/frame) vào đơn vị
    # match thay vì hard-gate df. `generate()` set từ HCAConfig.match_common_clusters; eval toggle
    # trực tiếp để A/B. True = cụm "ở đâu"/"là gì" được kích thích (đo trung tính trên QA).
    match_common_clusters: bool = True
    # TK_036 — skip span content (gap 2–8); lazy-mine qua ensure_skip_spans
    skip_spans: list[dict] = field(default_factory=list)
    _skip_lookup: dict[tuple[int, int], int] = field(default_factory=dict)
    _skip_left_index: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    _skip_index: dict[int, list[int]] = field(default_factory=dict)
    skip_span_meta: dict[tuple[int, int], dict] = field(default_factory=dict)
    _skip_df: list[int] = field(default_factory=list)
    # TK_036 S3 — ledger incremental (transient; skip_spans trong payload là bản materialize)
    skip_ledger: object | None = None
    # TK_036 B2 — cha cần record_parent lại (mới / rewire ISG); trống + synced → full flush
    _skip_dirty_parents: set[int] = field(default_factory=set)
    _skip_batch_start: int = 0

    def mark_skip_dirty(self, *parent_indices: int) -> None:
        self._skip_dirty_parents.update(parent_indices)

    def ensure_skip_ledger(self, **kwargs):
        from nn_aivie.core.skip_ledger import SkipLedger

        if self.skip_ledger is None:
            self.skip_ledger = SkipLedger(**kwargs)
        else:
            for k, v in kwargs.items():
                setattr(self.skip_ledger, k, v)
        return self.skip_ledger

    def _touch_skip_ledger(self, pi: int, vocab: Vocabulary, rewired: list[int]) -> None:
        ledger = self.ensure_skip_ledger()
        if self.skip_spans and not ledger._synced:
            ledger.ensure_synced(self, vocab)
        for rpi in rewired:
            if rpi != pi:
                ledger.record_parent(rpi, self, vocab)
        ledger.record_parent(pi, self, vocab)

    def materialize_skip_ledger(
        self, vocab: Vocabulary, *, logger=None, phrase_by_start=None, reuse_hits=None,
    ) -> None:
        ledger = self.skip_ledger
        if ledger is None or not ledger.dirty:
            return
        ledger.materialize(
            self, vocab, logger=logger, phrase_by_start=phrase_by_start,
            reuse_hits=reuse_hits,
        )

    def begin_ingest_batch(self) -> None:
        """Bắt đầu batch ingest: hoãn ISG promote + đánh dấu mốc cha (skip ledger B2)."""
        self._skip_batch_start = len(self.parents)
        self.seg.begin_batch()

    def flush_skip_ledger(self, vocab: Vocabulary, *, logger=None) -> None:
        """Ghi skip ledger; cache phrase_by_start; incremental khi chỉ rewire (không thêm cha)."""
        from nn_aivie.training.skip_span import build_phrase_by_start

        ledger = self.ensure_skip_ledger()
        n = len(self.parents)
        batch_start = getattr(self, "_skip_batch_start", 0)
        dirty = self._skip_dirty_parents
        corpus_grew = batch_start < n and any(pi >= batch_start for pi in dirty)
        if not ledger._synced or corpus_grew:
            if corpus_grew and batch_start > 0:
                targets = sorted(
                    set(range(batch_start, n))
                    | set(dirty)
                    | set(getattr(self.seg, "rewired_parents", ()))
                )
                if logger is not None:
                    logger.log(
                        f"    skip_ledger record_parent incremental "
                        f"{len(targets)}/{n} (batch_from={batch_start})"
                    )
            else:
                targets = list(range(n))
                if logger is not None:
                    logger.log(f"    skip_ledger record_parent 0/{n}")
        elif dirty and len(dirty) < n:
            targets = sorted(dirty)
            if logger is not None:
                logger.log(
                    f"    skip_ledger record_parent incremental {len(targets)}/{n}"
                )
        else:
            targets = list(range(n))
            if logger is not None:
                logger.log(f"    skip_ledger record_parent 0/{n}")
        if logger is not None:
            logger.log("    skip_ledger build_phrase_by_start …")
        phrase_by_start = build_phrase_by_start(self, min_count=ledger.min_count)
        if logger is not None:
            logger.log(
                f"✓ skip_ledger build_phrase_by_start "
                f"starts={len(phrase_by_start)}"
            )
        step = 25 if len(targets) >= 200 else 10
        rec_hb = time.monotonic()
        for i, pi in enumerate(targets):
            ledger.record_parent(
                pi, self, vocab, phrase_by_start=phrase_by_start,
            )
            if logger is not None and len(targets) >= n and (pi + 1) % step == 0:
                logger.progress("skip_ledger record_parent", pi + 1, n)
            elif logger is not None and time.monotonic() - rec_hb >= 30.0:
                logger.log(
                    f"skip_ledger record_parent … {pi + 1}/{n} "
                    f"({100.0 * (pi + 1) / max(1, n):.1f}%)"
                )
                rec_hb = time.monotonic()
        self._skip_dirty_parents.clear()
        if logger is not None:
            logger.log("    skip_ledger materialize …")
        # Fix #4: cha trong `targets` vừa record_parent lại (cùng phrase_by_start) →
        # materialize tái dùng `_parent_hits`, khỏi scan_parent lại (byte-identical).
        self.materialize_skip_ledger(
            vocab, logger=logger, phrase_by_start=phrase_by_start,
            reuse_hits=set(targets),
        )

    def finalize_batch_ingest(
        self,
        vocab: Vocabulary,
        *,
        logger=None,
        log_every: int = 500,
        defer_skip: bool = False,
    ) -> int:
        """Sau vòng `begin_batch` + `add_example(..., defer_index=True)`.

        Promote ISG một lần, đồng bộ phrase/skip, rebuild index, materialize skip ledger.
        Cùng thứ tự cuối như `rebuild_seg_graph` (TK_027).

        Fix #5: `defer_skip=True` → BỎ QUA skip_ledger (record + materialize) ở finalize
        này; caller phải gọi `finalize_skip_ledger_deferred()` MỘT lần ở cuối train.
        skip_span_ids KHÔNG được đọc giữa các dataset nên hoãn được (đổi output vì record
        toàn cha với phrase CUỐI thay vì tích lũy per-dataset — xem doc)."""
        n_par = len(self.parents)
        if logger is not None:
            logger.log(
                f"▶ finalize_batch_ingest parents={n_par} "
                f"clusters={len(self.seg.clusters)} "
                f"bigrams={len(self.seg.bigram_count)}"
            )
        progress = None
        if logger is not None:
            def progress(msg: str) -> None:
                logger.log(f"    ISG: {msg}")
        n_promoted = self.seg.end_batch(
            self.parents,
            progress=progress,
            log_every=log_every,
        )
        if logger is not None:
            logger.log(f"✓ ISG end_batch promoted={n_promoted}")
        batch_start = getattr(self, "_skip_batch_start", 0)
        rewired_after_isg = set(self.seg.rewired_parents)
        sync_targets = sorted(set(range(batch_start, n_par)) | rewired_after_isg)
        n_sync = len(sync_targets)
        sync_step = log_every if log_every > 0 else (500 if n_sync >= 5000 else 100)
        if logger is not None:
            logger.log(
                f"    sync_derived targets={n_sync}/{n_par} "
                f"(batch_from={batch_start} rewired={len(rewired_after_isg)})"
            )
        sync_hb = time.monotonic()
        for si, pi in enumerate(sync_targets):
            self._sync_derived_from_slots(self.parents[pi], vocab)
            if logger is not None and sync_step > 0 and (si + 1) % sync_step == 0:
                logger.progress("sync_derived", si + 1, n_sync)
                sync_hb = time.monotonic()
            elif logger is not None and time.monotonic() - sync_hb >= 30.0:
                logger.log(
                    f"sync_derived … {si + 1}/{n_sync} "
                    f"({100.0 * (si + 1) / max(1, n_sync):.1f}%)"
                )
                sync_hb = time.monotonic()
        if logger is not None:
            logger.log("    rebuild_hierarchical_index …")
        self.rebuild_hierarchical_index()
        if logger is not None:
            logger.log(f"✓ rebuild_hierarchical_index  (parents={n_par})")
        if defer_skip:
            if logger is not None:
                logger.log("    skip_ledger HOÃN (fix#5 — materialize 1 lần ở cuối)")
            return n_promoted
        self.mark_skip_dirty(*self.seg.rewired_parents)
        self.flush_skip_ledger(vocab, logger=logger)
        return n_promoted

    def finalize_skip_ledger_deferred(self, vocab: Vocabulary, *, logger=None) -> None:
        """Fix #5: chạy skip_ledger MỘT lần ở cuối train (thay vì mỗi dataset).

        Clear + build phrase_by_start MỘT lần + record TOÀN BỘ cha (với phrase cuối) +
        materialize (reuse toàn bộ _parent_hits → không scan lại). O(cha) một lần thay
        vì O(dataset × cha). Đổi output vs per-dataset: mọi cha quét với phrase cuối
        cùng đầy đủ (nhất quán hơn), không phải phrase-state tại thời điểm ingest."""
        from nn_aivie.training.skip_span import build_phrase_by_start

        ledger = self.ensure_skip_ledger()
        n = len(self.parents)
        if n == 0:
            return
        ledger.clear()
        if logger is not None:
            logger.log(f"▶ finalize_skip_ledger_deferred (fix#5) parents={n}")
        phrase_by_start = build_phrase_by_start(self, min_count=ledger.min_count)
        if logger is not None:
            logger.log(
                f"✓ build_phrase_by_start starts={len(phrase_by_start)}; record_parent …"
            )
        rec_hb = time.monotonic()
        for pi in range(n):
            ledger.record_parent(pi, self, vocab, phrase_by_start=phrase_by_start)
            if logger is not None and (pi + 1) % 2000 == 0:
                logger.progress("skip_ledger record_parent", pi + 1, n)
                rec_hb = time.monotonic()
            elif logger is not None and time.monotonic() - rec_hb >= 30.0:
                logger.log(f"skip_ledger record_parent … {pi + 1}/{n}")
                rec_hb = time.monotonic()
        self._skip_dirty_parents.clear()
        if logger is not None:
            logger.log("    skip_ledger materialize …")
        self.materialize_skip_ledger(
            vocab, logger=logger, phrase_by_start=phrase_by_start,
            reuse_hits=set(range(n)),
        )

    @staticmethod
    def _is_match_token(tid: int, vocab: Vocabulary) -> bool:
        tok = vocab.token_of(tid)
        if tok in _FRAME_TOKENS:
            return False
        return is_surface_token(tok)

    def _slots_to_content_tokens(
        self, slots: tuple[SlotRef, ...], vocab: Vocabulary
    ) -> tuple[int, ...]:
        out: list[int] = []
        for s in slots:
            for tid in self.seg.expand_slot(s):
                if self._is_match_token(tid, vocab):
                    out.append(tid)
        return tuple(out)

    def resync_match_ids(self, vocab: Vocabulary, *, logger=None) -> int:
        """Dựng lại `match_ids` + `_gdf`/`_df` từ `slot_chain` — gọi MỘT lần lúc load.

        `match_ids` được serialize, nên nó ĐÓNG BĂNG luật `_is_match_token` lúc train. Gỡ
        `is_punct` khỏi luật đó (2026-07-15) mà không dựng lại thì checkpoint cũ giữ thống kê
        cũ: dấu câu vắng khỏi `_gdf` → `_gidf('.')`=7.168 = CAO NHẤT corpus — mà giờ KHÔNG
        còn filter nào chặn nó làm neo. Đo trên grade_lessons_test (code mới, chưa resync):
        query có dấu chấm tụt rank-1 63.33% → 60.00%.

        `slot_chain` + `match_span` CÓ trong checkpoint → tái tạo được, KHÔNG cần train lại.
        Cha không có `slot_chain` (tiền-TK_013, vd `grade_v3.pkl`) giữ nguyên `match_ids`:
        chúng cũng tiền-R0 (`tokenize` không tách) nên vocab KHÔNG có token dấu câu nào —
        luật mới không đổi gì với chúng (đo: grade_v3.pkl 0/2025 token dấu câu).

        Cũng là thiết kế đúng bất kể dấu câu: match_ids luôn khớp luật HIỆN TẠI, thay vì có
        thể lệch âm thầm với code sau mỗi lần đổi `_is_match_token`.
        """
        n = 0
        for rec in self.parents:
            if rec.slot_chain and rec.match_span != (0, 0):
                ms, me = rec.match_span
                rec.match_ids = self._slots_to_content_tokens(rec.slot_chain[ms:me], vocab)
                n += 1
        self._gdf.clear()
        self._df.clear()
        for rec in self.parents:
            for t in set(rec.match_ids):
                self._gdf[t] = self._gdf.get(t, 0) + 1
                if rec.priority >= 0.9:
                    self._df[t] = self._df.get(t, 0) + 1
        self._fold_gdf_ready = -1  # invalidate cache df-theo-fold-group
        self.rebuild_hierarchical_index()
        if logger is not None:
            logger.log(f"    resync match_ids: {n}/{len(self.parents)} cha (df dựng lại)")
        return n

    def _sync_derived_from_slots(self, rec: ParentRecord, vocab: Vocabulary) -> None:
        """Materialize match_ids / phrase_ids từ slot_chain (tương thích rank_matches cũ)."""
        if rec.slot_chain and rec.match_span != (0, 0):
            ms, me = rec.match_span
            rec.match_ids = self._slots_to_content_tokens(rec.slot_chain[ms:me], vocab)
        cis = self.seg.clusters_in_match_span(rec)
        phrase_ids: list[int] = []
        for ci in cis:
            c = self.seg.clusters[ci]
            ph = c.token_span()
            if ph not in self._phrase_lookup:
                idx = len(self.phrases)
                self.phrases.append(ph)
                self.phrase_df.append(c.power)
                self._phrase_lookup[ph] = idx
                self._phrase_max_len = max(self._phrase_max_len, len(ph))
            else:
                idx = self._phrase_lookup[ph]
                self.phrase_df[idx] = max(self.phrase_df[idx], c.power)
            phrase_ids.append(idx)
        rec.phrase_ids = tuple(sorted(set(phrase_ids)))

    def _ensure_seg_fold_index(self, vocab: Vocabulary) -> None:
        """TK_027 bước 6 — index cặp cụm theo FOLD → tra cụm bất kể HOA/thường. "Triều Lý"(hoa,
        đầu câu) khớp cụm học được "triều Lý"(thường). Giữ cụm count cao nhất mỗi fold-pair."""
        if getattr(self, "_seg_fold_ready", False):
            return
        best: dict[tuple[int, int], tuple[int, int]] = {}
        for (a, b), ci in self.seg.cluster_of_pair.items():
            key = (vocab.fold_group_id(a), vocab.fold_group_id(b))
            cnt = getattr(self.seg.clusters[ci], "bigram_count", 0)
            if key not in best or cnt > best[key][1]:
                best[key] = (ci, cnt)
        self._seg_fold_pair = {k: v[0] for k, v in best.items()}
        self._seg_fold_ready = True

    def _seg_spans_in(self, seq: tuple[int, ...]) -> list[tuple[int, int, int]]:
        """(start, end, cluster_idx) cho cụm ISG bigram trong chuỗi query — tra theo VỊ TRÍ
        (O(len) thay O(#cụm)). Miss exact → thử FOLD (case-insensitive) để "Triều Lý" khớp cụm
        "triều Lý" đã học (bước 6) — chống query viết-hoa rã thành token-trần đồng-tự."""
        cop = self.seg.cluster_of_pair
        vocab = getattr(self, "_vocab_ref", None)
        if vocab is not None:
            self._ensure_seg_fold_index(vocab)
        spans: list[tuple[int, int, int]] = []
        for s in range(len(seq) - 1):
            a, b = seq[s], seq[s + 1]
            ci = cop.get((a, b))
            if ci is None and vocab is not None:
                ci = self._seg_fold_pair.get(
                    (vocab.fold_group_id(a), vocab.fold_group_id(b))
                )
            if ci is not None:
                spans.append((s, s + 2, ci))
        return spans

    def _rebuild_skip_left_index(self) -> None:
        """left token → [(right_phrase, skip_idx)] — tra skip span chỉ theo token trong query."""
        idx: dict[int, list[tuple[tuple[int, ...], int]]] = defaultdict(list)
        for key, si in self._skip_lookup.items():
            left, right = key
            idx[left].append((skip_right_tuple(right), si))
        self._skip_left_index = dict(idx)

    def _skip_spans_in(self, seq: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
        """(start, end, skip_idx, gap) — token trái + cụm phải trong query."""
        if len(seq) < SKIP_MIN_GAP + 2:
            return []
        left_idx = self._skip_left_index
        if not left_idx and self._skip_lookup:
            self._rebuild_skip_left_index()
            left_idx = self._skip_left_index
        if not left_idx:
            return []
        spans: list[tuple[int, int, int, int]] = []
        j_lo = SKIP_MIN_GAP + 1
        j_end = SKIP_MAX_GAP + 2
        for i, a in enumerate(seq):
            candidates = left_idx.get(a)
            if not candidates:
                continue
            j_hi = min(len(seq), i + j_end)
            for right_ids, si in candidates:
                L = len(right_ids)
                for j in range(i + j_lo, j_hi):
                    if j + L > len(seq):
                        continue
                    if tuple(seq[j : j + L]) != right_ids:
                        continue
                    gap = j - i - 1
                    if SKIP_MIN_GAP <= gap <= SKIP_MAX_GAP:
                        spans.append((i, j + L, si, gap))
        return spans

    def _skip_idf(self, skip_idx: int) -> float:
        if skip_idx >= len(getattr(self, "_skip_df", ())):
            return 1.0
        n = max(1, len(self.parents))
        return math.log((n + 1) / (1 + self._skip_df[skip_idx])) + 1.0

    def _indexable_for_retrieval(self, rec: ParentRecord) -> bool:
        """Cha pool pattern (priority<0.5) không vào _index — tránh dilution Para/Syn (TK_017 R4b)."""
        return rec.priority >= 0.5

    def _index_parent(self, pi: int) -> None:
        """Ghi một cha vào _index / _phrase_index (incremental, không rebuild toàn graph)."""
        rec = self.parents[pi]
        if not self._indexable_for_retrieval(rec):
            return
        for j in rec.phrase_ids:
            if j < len(self.phrases):
                lst = self._phrase_index.setdefault(j, [])
                if pi not in lst:
                    lst.append(pi)
        for t in set(rec.match_ids):
            lst = self._index.setdefault(t, [])
            if pi not in lst:
                lst.append(pi)

    def finalize_example(
        self,
        idx: int,
        vocab: Vocabulary,
        rewired: list[int] | None = None,
        *,
        logger=None,
    ) -> None:
        """Sau mỗi đơn vị corpus: đồng bộ cụm ISG + skip span + index cho cha mới."""
        rewired = list(rewired or [])
        self._sync_derived_from_slots(self.parents[idx], vocab)
        for rpi in rewired:
            if rpi != idx:
                self._sync_derived_from_slots(self.parents[rpi], vocab)
        self._touch_skip_ledger(idx, vocab, rewired)
        self.materialize_skip_ledger(vocab, logger=logger)
        if rewired:
            self.rebuild_hierarchical_index()
        else:
            self._index_parent(idx)

    def rebuild_hierarchical_index(self) -> None:
        """Sau phân rã: cụm→cha cho khớp phân tầng; `_index` giữ MỌI token→cha (KHÔNG prune).

        (Vá 2026-07-02 — bug "Người X là ai") Prune cũ ("token thuộc cụm chỉ truy hồi qua
        cụm→cha") làm cha BIẾN MẤT khỏi candidate set khi query lệch phân đoạn cụm: "Người
        trị bệnh là ai" không với tới cha "Ai chữa bệnh" vì "bệnh" nằm trong cụm "chữa bệnh"
        của cha mà query không mang nguyên cụm → rank trả cha lạc ("Trung thu là tết trẻ em")
        dù cha đúng tồn tại. Ưu tiên đường cụm là việc của SCORING phân tầng (đã có — cụm
        điểm cao hơn token rời), không phải của candidate set.
        """
        self._index = {}
        self._phrase_index = {}
        for pi, rec in enumerate(self.parents):
            if not self._indexable_for_retrieval(rec):
                continue
            for j in rec.phrase_ids:
                if j >= len(self.phrases):
                    continue
                self._phrase_index.setdefault(j, []).append(pi)
            for t in set(rec.match_ids):
                self._index.setdefault(t, []).append(pi)

    def parent_phrases_for_query(
        self,
        rec: ParentRecord,
        q_content: set[int],
        *,
        phrase_max_df: int | None = 8,
    ) -> tuple[int, ...]:
        """Cụm thuộc cha mà mọi token cụm đều có trong query — greedy longest, không chồng."""
        cands = [
            j
            for j in rec.phrase_ids
            if j < len(self.phrases)
            and (phrase_max_df is None or self.phrase_df[j] <= phrase_max_df)
            and all(t in q_content for t in self.phrases[j])
        ]
        cands.sort(key=lambda j: -len(self.phrases[j]))
        used: set[int] = set()
        picked: list[int] = []
        for j in cands:
            ph = self.phrases[j]
            if any(t in used for t in ph):
                continue
            picked.append(j)
            used.update(ph)
        return tuple(picked)

    @staticmethod
    def _qa_completion_factor(rec: ParentRecord) -> float:
        """§1: khung Hỏi/Đáp — chỉ cha QA (match|answer) có phần đáp pattern-complete."""
        if rec.is_qa and rec.answer_ids:
            n = len(rec.match_ids) + len(rec.answer_ids)
            return len(rec.answer_ids) / max(1, n)
        return 0.0

    def add_example(
        self,
        ids: list[int],
        vocab: Vocabulary,
        *,
        dap_id: int | None,
        priority: float = 1.0,
        defer_index: bool = False,
    ) -> None:
        """Thêm một ví dụ làm nơ-ron cha (gộp power nếu trùng nguyên bản).

        `defer_index=True` (nạp hàng loạt trên store LỚN): BỎ `rebuild_hierarchical_index()`
        mỗi câu (O(#cha) → chết khi loop 10⁴ câu trên 44k cha). Caller BẮT BUỘC gọi
        `seg.begin_batch()` trước + `seg.end_batch(parents)` + `rebuild_hierarchical_index()`
        một lần sau vòng lặp (mẫu `training/ingest_dictionary.py`)."""
        ids = list(ids)
        if len(ids) < 1:
            return
        key = tuple(ids)
        if key in self._key_to_idx:
            idx = self._key_to_idx[key]
            self.parents[idx].power += 1
            self.seg.increment_parent_counts(idx, self.parents)
            return

        is_qa = dap_id is not None and dap_id in ids
        if is_qa:
            di = len(ids) - 1 - ids[::-1].index(dap_id)
            q_part, a_part = ids[:di], ids[di + 1 :]
            match = _content_ids(q_part, vocab, skip=set())
            answer = tuple(a_part)
            emit = answer
            dap_idx = di
        else:
            match = _content_ids(ids, vocab, skip=set())
            answer = ()
            emit = tuple(ids)
            dap_idx = None

        chain, in_match = build_chain_from_ids(
            ids, dap_idx=dap_idx, vocab=vocab, is_match_token=self._is_match_token
        )
        self.seg.min_frequency = self.seg_min_frequency

        rec = ParentRecord(
            emit_ids=emit,
            match_ids=match,
            answer_ids=answer,
            power=1,
            is_qa=is_qa,
            priority=priority,
            source_ids=tuple(ids),
            slot_chain=chain,
            match_span=_match_span_from_flags(in_match),
            _in_match_flags=in_match,
        )
        idx = len(self.parents)
        self.parents.append(rec)
        self._key_to_idx[key] = idx
        self.seg.ingest_parent(idx, chain, in_match, self.parents)
        rewired = list(self.seg.rewired_parents)
        for pi in rewired:
            self._sync_derived_from_slots(self.parents[pi], vocab)
        self.seg.rewired_parents.clear()
        for t in set(rec.match_ids):
            self._gdf[t] = self._gdf.get(t, 0) + 1
            if priority >= 0.9:
                self._df[t] = self._df.get(t, 0) + 1
        if not defer_index:
            self.finalize_example(idx, vocab, rewired)
        else:
            for t in set(rec.match_ids):
                self._index.setdefault(t, []).append(idx)
            self.mark_skip_dirty(idx)

    def _df_gop(self, token_id: int) -> int:
        """§96 — df GỘP hai dạng ranh giới của cùng một từ.

        Token đầu dòng (`con`) và token giữa câu (` con`) là HAI id khác nhau
        với hai `df` riêng. Đa số lần xuất hiện là giữa câu, nên dạng ĐẦU DÒNG
        hiếm ⇒ idf cao vống. Mà từ đầu tiên của MỌI prompt luôn ở dạng đầu dòng:

            từ     đầu dòng  giữa câu          từ     đầu dòng  giữa câu
            là       14,0      8,3             chó      14,0     10,7
            một      14,0     10,7             mèo      13,3     10,4
            con      13,3     10,7

        Hệ quả đo được: «con tê tê là» — `con` được chấm gần như đặc trưng nhất
        câu (13,3 so với `tê` 14,0) nên kéo truy hồi về mọi cha chứa `con`; cha
        đúng tụt hạng 1 → 3 và ĐẦU RA vỡ hẳn (`con chó là gì? dụ`), trong khi
        «tê tê là» ra đúng. Loại từ (`con`, `cái`, `chiếc`, `quả`) đứng đầu là
        cách nói tiếng Việt hoàn toàn tự nhiên, nên lỗi này chạm rất nhiều prompt.

        Vá ở tầng ĐỌC — cộng df hai dạng, không cần train lại.
        """
        d = self._df.get(token_id, 0)
        m = getattr(self, "_ranh_gioi_map", None)
        if m is None:
            m = {}
            v = getattr(self, "_vocab_ref", None)
            if v is not None:
                from nn_aivie.core.tokenizer import MARK_SPACE as _MS
                for i in range(len(v)):
                    t = v.token_of(i)
                    if t and not t.startswith(_MS):
                        j = getattr(v, "_token_to_id", {}).get(_MS + t)
                        if j is not None:
                            m[i] = j
                            m[j] = i
            self._ranh_gioi_map = m
        j = m.get(int(token_id))
        return d + self._df.get(j, 0) if j is not None else d

    def _idf(self, token_id: int) -> float:
        n = max(1, len(self.parents))
        df = (self._df_gop(token_id) if getattr(self, "idf_gop_ranh_gioi", False)
              else self._df.get(token_id, 0))
        return math.log((n + 1) / (1 + df)) + 1.0

    def _gidf(self, token_id: int) -> float:
        """IDF toàn cục (mọi nguồn) — chọn focus/entity; thực thể wiki (df=0) vẫn có IDF thật.

        §97 — GỘP hai dạng ranh giới, y như `_idf`. Đây mới là chỗ lỗi CẮN:
        `pick_primary` chọn thực thể theo `_gidf`, và dạng ĐẦU DÒNG hiếm nên
        token ở vị trí 0 luôn được thổi lên. Bất đối xứng đo được:

            «chó là»      chó ĐẦU DÒNG  gdf=24    gidf=10,8 → primary = chó ✓
            «con chó là»  chó GIỮA CÂU  gdf=1316  gidf= 6,8
                          con ĐẦU DÒNG  gdf=255   gidf= 8,5 → primary = con ✗

        Thêm MỘT từ làm chủ ngữ thật tụt 10,8 → 6,8 — không phải vì nó kém đặc
        trưng đi mà vì nó ĐỔI DẠNG TOKEN. Hệ quả dây chuyền: primary sai ⇒
        entity register sai ⇒ `_build_segment_plan` dựng agenda « gì?» (2 token)
        ⇒ «con chó là gì? dụ». Cùng lỗi với `cái bàn là` (chọn `cái` 9,4 thay
        `bàn` 5,9).
        """
        n = max(1, len(self.parents))
        df = (self._gdf_gop(token_id) if getattr(self, "idf_gop_ranh_gioi", False)
              else self._gdf.get(token_id, 0))
        return math.log((n + 1) / (1 + df)) + 1.0

    def _gdf_gop(self, token_id: int) -> int:
        """`_gdf` gộp hai dạng ranh giới — xem `_gidf`."""
        self._df_gop(0)                      # dựng bản đồ nếu chưa có
        j = (getattr(self, "_ranh_gioi_map", None) or {}).get(int(token_id))
        d = self._gdf.get(token_id, 0)
        return d + self._gdf.get(j, 0) if j is not None else d

    def pick_primary(self, cands, vocab: Vocabulary | None = None, *, tie: float = 0.3) -> int:
        """Token ĐỊNH DANH nhất trong `cands` — chọn neo cho theme/facts_about.

        = max gidf (token PHÂN BIỆT), KHÔNG phải token dẫn/phải nhất: một số đường (span) lấy
        `entity[-1]` (phải nhất) → "Thăng Long" ra 'Long' (va chạm, kéo Gia Long/Hạ Long vào pool)
        → drift. gidf tách đúng ở chênh RÕ ("Thăng" 6.5 > "Long" 4.7). Nhưng THẾ HÒA (token phân
        loại generic "Nhà" nhỉnh gidf sát entity "Trần" ở corpus lệch tần suất) → phá hòa (chênh
        gidf ≤ `tie`) bằng THEME-CONCENTRATION: token có theme_neuron NHỎ hơn = đặc thù hơn.
        DÙNG CHUNG (facts_about + knowledge_extra_fact_plan) để primary NHẤT QUÁN mọi đường.
        """
        cands = list(dict.fromkeys(cands))
        if not cands:
            raise ValueError("pick_primary: cands rỗng")
        best = max(cands, key=self._gidf)
        top = self._gidf(best)
        near = [t for t in cands if top - self._gidf(t) <= tie]
        if len(near) > 1 and vocab is not None:
            def _theme_size(t: int) -> int:
                tn = self.theme_neuron(t, vocab)
                return len(tn.facts) if tn is not None else 10**9
            return min(near, key=_theme_size)
        return best

    def _ensure_fold_gdf(self, vocab: Vocabulary) -> None:
        """TK_020-fix: df gộp theo FOLD-GROUP ("Cách"≡"cách"). Token viết-hoa-đầu-câu là token
        HIẾM riêng → gidf cao ARTIFACT → cướp focus khỏi thực thể thật ("Cách" 6.72 > "phở").
        Đo 07-06: 3/4 ca drift của tác giả cùng gốc này."""
        if getattr(self, "_fold_gdf_ready", 0) == len(self._gdf):
            return
        fold_gdf: dict[int, int] = {}
        for t, d in self._gdf.items():
            try:
                gid = vocab.fold_group_id(t)
            except (IndexError, TypeError):
                continue
            fold_gdf[gid] = fold_gdf.get(gid, 0) + d
        self._fold_gdf = fold_gdf
        self._fold_gdf_ready = len(self._gdf)

    def _fold_gidf(self, token_id: int, vocab: Vocabulary | None = None) -> float:
        """IDF trên fold-group nếu sẵn sàng; fallback _gidf (backward-safe)."""
        fg = getattr(self, "_fold_gdf", None)
        if fg is None or vocab is None:
            return self._gidf(token_id)
        try:
            gid = vocab.fold_group_id(token_id)
        except (IndexError, TypeError):
            return self._gidf(token_id)
        n = max(1, len(self.parents))
        return math.log((n + 1) / (1 + fg.get(gid, 0))) + 1.0

    # ---- phân rã Re-Pair ĐỆ QUY: nơ-ron trung gian nhiều tầng (docx §2.1) ----
    def decompose_repair(
        self,
        vocab: Vocabulary,
        *,
        min_frequency: int = 3,
        max_merges: int = 10000,
        max_span: int = 8,
    ) -> dict[str, int]:
        """
        Re-Pair đệ quy: gộp cặp lặp nhiều nhất → symbol trung gian → lặp → **cụm-của-cụm nhiều
        tầng**. Mỗi symbol phi-kết = một nơ-ron trung gian (span 2..max_span token). Thay
        `decompose` (n-gram phẳng) — đúng tinh thần v3 hơn, cụm dài & phân cấp.
        """
        OFFSET = 10 ** 7  # id symbol phi-kết bắt đầu từ đây (token id < OFFSET)
        seqs = [list(rec.match_ids) for rec in self.parents]
        rules, comp = _repair_grammar(
            seqs, min_frequency=min_frequency, max_merges=max_merges, first_new_id=OFFSET
        )

        span_cache: dict[int, tuple[int, ...]] = {}

        def span(s: int) -> tuple[int, ...]:
            if s < OFFSET:
                return (s,)
            if s in span_cache:
                return span_cache[s]
            a, b = rules[s]
            r = span(a) + span(b)
            span_cache[s] = r
            return r

        parent_nts = [{s for s in cseq if s >= OFFSET} for cseq in comp]
        sym_df: dict[int, int] = defaultdict(int)
        for nts in parent_nts:
            for s in nts:
                sym_df[s] += 1

        self.phrases = []
        self.phrase_df = []
        self._phrase_lookup = {}
        sym_to_phrase: dict[int, int] = {}
        for s, df in sym_df.items():
            sp = span(s)
            if not (2 <= len(sp) <= max_span) or df < min_frequency:
                continue
            if sp in self._phrase_lookup:
                sym_to_phrase[s] = self._phrase_lookup[sp]
                continue
            idx = len(self.phrases)
            self.phrases.append(sp)
            self.phrase_df.append(df)
            self._phrase_lookup[sp] = idx
            sym_to_phrase[s] = idx

        self._phrase_max_len = max((len(p) for p in self.phrases), default=4)
        for rec, nts in zip(self.parents, parent_nts):
            pids = {sym_to_phrase[s] for s in nts if s in sym_to_phrase}
            rec.phrase_ids = tuple(sorted(pids))
        self.rebuild_hierarchical_index()
        return {"phrases": len(self.phrases), "rules": len(rules)}

    # ---- phân rã BPE/Re-Pair (n-gram phẳng — giữ để so sánh) ----
    def decompose(
        self,
        vocab: Vocabulary,
        *,
        min_frequency: int = 2,
        max_len: int = 4,
        max_phrases: int = 4000,
    ) -> dict[str, int]:
        """
        Bóc cụm con **liền kề lặp lại** (≥ min_frequency cha) thành nơ-ron trung gian (docx §2.1).

        Cụm chung giữa nhiều cha → khớp lan xuôi cộng điểm theo CỤM (đặc thù hơn token rời),
        tổng quát hóa liên-mẫu (vd `chữa bệnh`, `Đoàn kết`). Document-frequency = số cha chứa cụm.
        """
        occ: dict[tuple, set[int]] = {}
        for pi, rec in enumerate(self.parents):
            seq = rec.match_ids
            seen: set[tuple] = set()
            for L in range(2, max_len + 1):
                for s in range(len(seq) - L + 1):
                    ng = seq[s : s + L]
                    if ng in seen:
                        continue
                    seen.add(ng)
                    occ.setdefault(ng, set()).add(pi)

        cand = [(ng, len(ps)) for ng, ps in occ.items() if len(ps) >= min_frequency]
        # ưu tiên cụm vừa dài vừa lặp nhiều (xấp xỉ Re-Pair)
        cand.sort(key=lambda x: -(x[1] * len(x[0])))
        cand = cand[:max_phrases]

        self.phrases = [ng for ng, _ in cand]
        self.phrase_df = [d for _, d in cand]
        self._phrase_lookup = {ng: i for i, (ng, _) in enumerate(cand)}
        self._phrase_max_len = max_len

        for rec in self.parents:
            seq = rec.match_ids
            pids: set[int] = set()
            for L in range(2, max_len + 1):
                for s in range(len(seq) - L + 1):
                    j = self._phrase_lookup.get(seq[s : s + L])
                    if j is not None:
                        pids.add(j)
            rec.phrase_ids = tuple(sorted(pids))
        self.rebuild_hierarchical_index()
        return {"phrases": len(self.phrases)}

    def _phrases_in(self, seq: tuple[int, ...]) -> set[int]:
        """Cụm trung gian xuất hiện (liền kề) trong một chuỗi token."""
        if not self._phrase_lookup:
            return set()
        found: set[int] = set()
        for L in range(2, self._phrase_max_len + 1):
            for s in range(len(seq) - L + 1):
                j = self._phrase_lookup.get(seq[s : s + L])
                if j is not None:
                    found.add(j)
        return found

    def _phrase_idf(self, phrase_idx: int) -> float:
        """Độ phân biệt của cụm theo document-frequency của CỤM (cụm phổ biến → thấp)."""
        n = max(1, len(self.parents))
        return math.log((n + 1) / (1 + self.phrase_df[phrase_idx])) + 1.0

    def _phrase_spans_in(self, seq: tuple[int, ...]) -> list[tuple[int, int, int]]:
        """Mọi (start, end, phrase_idx) cụm liền kề trong chuỗi."""
        if not self._phrase_lookup:
            return []
        spans: list[tuple[int, int, int]] = []
        for L in range(2, self._phrase_max_len + 1):
            for s in range(len(seq) - L + 1):
                j = self._phrase_lookup.get(seq[s : s + L])
                if j is not None:
                    spans.append((s, s + L, j))
        return spans

    def query_match_units(
        self,
        q_seq: tuple[int, ...],
        *,
        phrase_max_df: int = 8,
    ) -> QueryMatchUnits:
        """
        Chia query thành đơn vị match phân tầng (docx §2.1): cụm trung gian trước, token rời sau.

        Cụm phân biệt (df ≤ phrase_max_df) được chọn greedy longest-first, không chồng lấn.
        Token thuộc cụm đã chọn **không** còn là bare token (tránh quét kép token→cha).

        Hướng 1 (TK_014): `match_common_clusters` → bỏ cap df, admit CẢ cụm phổ biến (WH/frame
        như "ở đâu"/"là gì") làm unit. Chúng được cân bằng bằng IDF thấp trong `rank_matches`
        (+ phạt distinct-ratio²) nên không nhấn chìm thực thể — nhưng nơ-ron WH được kích thích.
        """
        cap = 10**18 if self.match_common_clusters else phrase_max_df
        spans = [
            (s, e, j)
            for s, e, j in self._phrase_spans_in(q_seq)
            if j < len(self.phrase_df) and self.phrase_df[j] <= cap
        ]
        for s, e, ci in self._seg_spans_in(q_seq):
            c = self.seg.clusters[ci]
            j = self._phrase_lookup.get(c.token_span())
            if j is None or j >= len(self.phrase_df):
                continue
            if self.phrase_df[j] > cap:
                continue
            if (s, e, j) not in spans:
                spans.append((s, e, j))
        spans.sort(key=lambda x: (-self._phrase_idf(x[2]) * (x[1] - x[0]), x[0]))

        used_pos: set[int] = set()
        phrase_units: list[int] = []
        for s, e, j in spans:
            if any(p in used_pos for p in range(s, e)):
                continue
            phrase_units.append(j)
            used_pos.update(range(s, e))

        phrase_units_t = tuple(phrase_units)

        skip_spans_raw = self._skip_spans_in(q_seq)
        skip_spans_raw.sort(key=lambda x: (-self._skip_idf(x[2]), -x[3]))
        skip_units: list[int] = []
        for s, e, si, _gap in skip_spans_raw:
            if any(p in used_pos for p in range(s, e)):
                continue
            skip_units.append(si)
            used_pos.update(range(s, e))

        covered = frozenset(q_seq[p] for p in used_pos)
        q_content = list(dict.fromkeys(q_seq))
        bare = tuple(t for t in q_content if t not in covered)
        skip_units_t = tuple(skip_units)
        idf_total = (
            sum(self._phrase_idf(j) for j in phrase_units_t)
            + sum(self._skip_idf(j) for j in skip_units_t)
            + sum(self._idf(t) for t in bare)
        ) or 1.0
        return QueryMatchUnits(
            phrase_units=phrase_units_t,
            skip_units=skip_units_t,
            bare_tokens=bare,
            phrase_covered_tokens=covered,
            idf_total=idf_total,
        )

    # ---- Hướng 2 (TK_014): anchor gom — X→cụm, Y→cụm ⇒ X,Y chung nơ-ron cụm ----
    def anchor_members(self, cluster_idx: int) -> tuple[int, ...]:
        """Mọi cha chứa cụm (anchor) trong slot_chain — nhóm gom qua nơ-ron chung.

        VD cụm "ở đâu": trả về mọi cha `X ở đâu` (X = token/cụm/cha-câu/bài) → chúng được nối
        vào nhau qua chính cụm, đúng ý niệm "nơi chốn". Dựng sẵn từ `seg._cluster_index`.
        """
        seen: dict[int, None] = {}
        for pi in self.seg._cluster_index.get(cluster_idx, ()):  # noqa: SLF001
            seen.setdefault(pi, None)
        return tuple(seen)

    def anchor_left_fillers(self, cluster_idx: int) -> list[tuple[SlotRef, int]]:
        """Slot đứng NGAY TRƯỚC anchor trong mỗi member = thực thể X được hỏi. (X_slot, parent).

        Đây là "filler" của họ mẫu WH: `Hồ Tây` / `Thủ đô Việt Nam` / `Kinh đô nhà Nguyễn` …
        cho anchor "ở đâu". X có thể là token hoặc cụm (SlotRef.kind).
        """
        cid = self.seg.cluster_id(cluster_idx)
        out: list[tuple[SlotRef, int]] = []
        for pi in self.anchor_members(cluster_idx):
            chain = self.parents[pi].slot_chain
            for pos in range(1, len(chain)):
                s = chain[pos]
                if s.kind == "cluster" and s.id == cid:
                    out.append((chain[pos - 1], pi))
                    break
        return out

    def query_anchor_clusters(self, q_seq: tuple[int, ...]) -> tuple[int, ...]:
        """cluster_idx của MỌI cụm ISG xuất hiện trong query (kể cả phổ biến) — cụm được kích
        thích bởi câu hỏi này, dùng cho hiển thị + tra anchor group. Độc lập với gate scoring."""
        seen: dict[int, None] = {}
        for _s, _e, ci in self._seg_spans_in(q_seq):
            seen.setdefault(ci, None)
        return tuple(seen)

    # TK_025: gỡ TOÀN BỘ WH-detector (wh_anchor_clusters/wh_slot_consistency_mode/
    # _frame_token_ids_in_query) — vi phạm Luật 0 (`==\"gì\"` hardcode + rule phân-loại câu-hỏi).
    # Route giờ 1 đường: retrieval (best_match) quyết theo KÍCH HOẠT, không phân-loại WH.

    # ---- lan xuôi: xếp hạng nơ-ron cha ----
    def rank_matches(
        self,
        query_ids: list[int],
        vocab: Vocabulary,
        embedding=None,
        *,
        fuzzy_topk: int = 6,
        fuzzy_min_sim: float = 0.35,
        fuzzy_scale: float = 0.85,
        phrase_weight: float = 0.35,  # deprecated — cụm là đơn vị match chính, không còn bonus
        phrase_max_df: int = 8,
        min_priority: float = 0.0,
        top_n: int = 8,
        qa_completion_mode: bool = False,
        restrict_parents: frozenset[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Top-N nơ-ron cha theo phủ phân tầng: cụm trung gian → cha, token rời → cha.

        Token thuộc cụm đã chọn chỉ kích hoạt qua cụm (không quét kép token→cha trực tiếp).
        `min_priority`>0 → chỉ xét cha có priority ≥ ngưỡng (dùng cho two-tier: curated trước).
        """
        del phrase_weight  # giữ signature tương thích; không còn bonus additive
        q_seq = _content_ids(query_ids, vocab, skip=set())
        q_content = list(dict.fromkeys(q_seq))
        if not q_content:
            return []

        # §184 — CACHE `ranked` (đầy đủ, TRƯỚC cắt top_n). completion_graph gọi
        # rank_matches ~5× CÙNG query (trực tiếp + agenda + segment_plan) ⇒ 17/19s
        # là quét lặp. Key BỎ top_n (ranked độc lập top_n, chỉ cắt cuối) nên mọi
        # top_n chia sẻ. Phiên bản = len(parents): parents đổi (Hebbian) ⇒ cache rơi.
        _rank_ck = (
            tuple(query_ids), fuzzy_topk, round(fuzzy_min_sim, 4),
            round(fuzzy_scale, 4), phrase_max_df, round(min_priority, 4),
            qa_completion_mode, restrict_parents, embedding is not None,
        )
        _rank_ver = len(self.parents)
        _rank_cache = self.__dict__.get("_rank_cache")
        if _rank_cache is None or _rank_cache.get("__ver__") != _rank_ver:
            _rank_cache = {"__ver__": _rank_ver}
            self._rank_cache = _rank_cache
        _rank_hit = _rank_cache.get(_rank_ck)
        if _rank_hit is not None:
            return _rank_hit[:top_n]

        # TK_029: cặp token CÓ HƯỚNG kề của query (fold) → thưởng fact khớp ĐÚNG hướng-kề. "Triều
        # Lý" (Triều→Lý) chỉ khớp fact "triều Lý …", KHÔNG khớp "Lý … Triều Tiên"(ngược/rời) hay
        # "Triều Kim". Dùng hướng đã học (thứ tự), sửa lỗ bag-of-words vô-hướng của stimulus.
        dir_bonus = getattr(self, "directed_bigram_bonus", 0.0)
        dir_min_idf = getattr(self, "directed_bigram_min_idf", 2.0)
        # TK_045 §12.124 (③): mở rộng cặp-hướng KỀ (d=1) → cặp-hướng theo KHOẢNG CÁCH d≤dmax (W_skip),
        # giảm λ^(d-1). Đo được (§12.119/§12.122): match theo thứ tự thu hồi cha nguồn +31pp v12; và
        # order-làm-multiplier bắt trọn lãi (§12.124 covdir=skip). dmax=1 ⇒ hành vi TK_029 cũ y nguyên.
        dir_dmax = int(getattr(self, "directed_bigram_dmax", 1))
        dir_lam = float(getattr(self, "directed_bigram_lam", 0.5))
        dir_accum = str(getattr(self, "directed_bigram_accum", "max"))  # §12.124 ③: max|sum
        q_dir: dict[tuple[int, int], float] = {}
        if dir_bonus > 0:
            for i in range(len(q_seq) - 1):
                for d in range(1, min(dir_dmax, len(q_seq) - 1 - i) + 1):
                    raw = min(self._idf(q_seq[i]), self._idf(q_seq[i + d]))
                    if raw < dir_min_idf:  # CHỈ cặp-hướng ĐẶC-TRƯNG (Triều→Lý), bỏ cặp hư-từ
                        continue           # (Lý→đã generic) — không thưởng lan fact sai.
                    w = raw * (dir_lam ** (d - 1))
                    key = (vocab.fold_group_id(q_seq[i]), vocab.fold_group_id(q_seq[i + d]))
                    if w > q_dir.get(key, 0.0):
                        q_dir[key] = w

        units = self.query_match_units(q_seq, phrase_max_df=phrase_max_df)
        phrase_units = units.phrase_units
        skip_units = units.skip_units
        bare_tokens = units.bare_tokens
        idf_total = units.idf_total

        fuzzy_map: dict[int, dict[int, float]] = {}
        fuzzy_tokens = set(bare_tokens)
        for j in phrase_units:
            fuzzy_tokens.update(self.phrases[j])
        skip_spans = getattr(self, "skip_spans", None) or []
        for si in skip_units:
            if si < len(skip_spans):
                sp = skip_spans[si]
                fuzzy_tokens.add(sp["left"])
                for t in skip_right_tuple(sp["right"]):
                    fuzzy_tokens.add(t)
        if embedding is not None and fuzzy_topk > 0:
            cand = list(self._index.keys())
            for q in fuzzy_tokens:
                fuzzy_map[q] = {
                    nid: sim
                    for nid, sim in embedding.nearest(
                        q, topk=fuzzy_topk, candidate_ids=cand, min_sim=fuzzy_min_sim
                    )
                }

        unit_idfs = (
            [self._phrase_idf(j) for j in phrase_units]
            + [self._skip_idf(j) for j in skip_units]
            + [self._idf(q) for q in bare_tokens]
        )
        max_q_idf = max(unit_idfs) if unit_idfs else 1.0
        q_set = set(q_content)
        cand_idx: set[int] = set()
        if restrict_parents is not None:
            cand_idx = set(restrict_parents)
        else:
            for j in phrase_units:
                cand_idx.update(self._phrase_index.get(j, ()))
                q_ph = set(self.phrases[j])
                for j2, ph2 in enumerate(self.phrases):
                    if set(ph2) <= q_ph:
                        cand_idx.update(self._phrase_index.get(j2, ()))
            for si in skip_units:
                cand_idx.update(getattr(self, "_skip_index", {}).get(si, ()))
            # §196 — CAP CANDIDATE: chỉ MỞ RỘNG từ token PHÂN BIỆT (IDF ≥ 0.5×đỉnh).
            # Token phổ biến ("và/như/có") phình cand_idx 100k+ cha mà không phân
            # biệt → cold cost lớn; bỏ mở rộng (vẫn TÍNH ĐIỂM nếu candidate chứa
            # chúng). Đáp đúng luôn chứa token phân-biệt (thực thể) nên không mất.
            if bare_tokens:
                _bidf = {q: self._idf(q) for q in bare_tokens}
                _top = max(_bidf.values()) if _bidf else 0.0
                _strong_bare = (
                    {q for q in bare_tokens if _bidf[q] >= 0.5 * _top}
                    if _top > 0 else set(bare_tokens)
                )
            else:
                _strong_bare = set()
            for q in bare_tokens:
                if q not in _strong_bare:
                    continue  # token phổ biến — không mở rộng candidate (§196)
                cand_idx.update(self._index.get(q, ()))
                for j, ph in enumerate(self.phrases):
                    if ph and ph[0] == q:
                        cand_idx.update(self._phrase_index.get(j, ()))
            for q in fuzzy_tokens:
                for nid in fuzzy_map.get(q, {}):
                    cand_idx.update(self._index.get(nid, ()))

        scored: list[tuple[int, float]] = []
        for pi in cand_idx:
            rec = self.parents[pi]
            if rec.priority < min_priority:
                continue
            pset = set(rec.match_ids)
            rec_phrase_set = set(rec.phrase_ids)
            rec_skip_set = set(getattr(rec, "skip_span_ids", ()))
            eff_idf_total = idf_total
            covered = 0.0
            matched_units = 0
            max_cov_idf = 0.0
            tokens_via_phrase: set[int] = set()
            scored_phrases: set[int] = set()

            for j in phrase_units:
                if j in rec_phrase_set:
                    w = self._phrase_idf(j)
                    covered += w
                    matched_units += 1
                    max_cov_idf = max(max_cov_idf, w)
                    scored_phrases.add(j)
                    tokens_via_phrase.update(self.phrases[j])
                else:
                    eff_idf_total -= self._phrase_idf(j)
                    for t in self.phrases[j]:
                        if t in tokens_via_phrase:
                            continue
                        eff_idf_total += self._idf(t)
                        if t in pset:
                            w = self._idf(t)
                            covered += w
                            matched_units += 1
                            max_cov_idf = max(max_cov_idf, w)
                        else:
                            sim = max(
                                (s for nid, s in fuzzy_map.get(t, {}).items() if nid in pset),
                                default=0.0,
                            )
                            if sim > 0:
                                w = self._idf(t) * sim
                                covered += w * fuzzy_scale
                                matched_units += 1
                                max_cov_idf = max(max_cov_idf, w)

            for si in skip_units:
                if si in rec_skip_set:
                    w = self._skip_idf(si)
                    covered += w
                    matched_units += 1
                    max_cov_idf = max(max_cov_idf, w)
                    if si < len(skip_spans):
                        sp = skip_spans[si]
                        tokens_via_phrase.add(sp["left"])
                        for t in skip_right_tuple(sp["right"]):
                            tokens_via_phrase.add(t)
                else:
                    eff_idf_total -= self._skip_idf(si)

            for j in self.parent_phrases_for_query(rec, q_set, phrase_max_df=None):
                if j in scored_phrases:
                    continue
                w = self._phrase_idf(j)
                covered += w
                matched_units += 1
                max_cov_idf = max(max_cov_idf, w)
                scored_phrases.add(j)
                tokens_via_phrase.update(self.phrases[j])
                if j not in phrase_units:
                    eff_idf_total += w
                    for t in self.phrases[j]:
                        if t in bare_tokens:
                            eff_idf_total -= self._idf(t)

            for q in bare_tokens:
                if q in tokens_via_phrase:
                    continue
                if q in pset:
                    w = self._idf(q)
                    covered += w
                    matched_units += 1
                    max_cov_idf = max(max_cov_idf, w)
                else:
                    sim = max(
                        (s for nid, s in fuzzy_map.get(q, {}).items() if nid in pset),
                        default=0.0,
                    )
                    if sim > 0:
                        w = self._idf(q) * sim
                        covered += w * fuzzy_scale
                        matched_units += 1
                        max_cov_idf = max(max_cov_idf, w)
            if matched_units == 0:
                continue
            coverage = covered / (eff_idf_total or 1.0)
            # TK_030 (tác giả): BỎ phạt specificity (matched/len(fact)) — độ DÀI câu không nói lên
            # tính đúng của fact. Trước: score = coverage×(0.5+0.5×specificity) → fact-đúng-dài (khớp
            # ít token/nhiều token, vd "Triều Lý trải qua chín đời vua") bị chia ~nửa điểm, thua
            # fact-sai-ngắn. Giờ score = coverage (distinct_ratio phía dưới vẫn chống khớp-generic).
            score = coverage
            distinct_ratio = max_cov_idf / max_q_idf if max_q_idf > 0 else 1.0
            if distinct_ratio < 0.85:
                score *= distinct_ratio * distinct_ratio
            score *= 1.0 + 0.02 * math.log(1 + rec.power)  # TK_025: WH knowledge-mode đã gỡ
            if q_dir and max_q_idf > 0:  # TK_029/§12.124: thưởng khớp CẶP-CÓ-HƯỚNG (fold) theo d≤dmax
                mids = rec.match_ids
                if dir_accum == "sum":
                    # §12.124 (③): CỘNG DỒN các cặp-hướng KHÁC NHAU của query mà cha khớp (mỗi cặp
                    # tính một lần), chuẩn hoá theo tổng q_dir. MAX (cũ) bão hoà ở cặp d=1 nên mở
                    # rộng d vô ích (§12.124 đo 0/30); SUM mới phân biệt được "cha khớp NHIỀU cặp".
                    seen: set[tuple[int, int]] = set()
                    hit = 0.0
                    for i in range(len(mids) - 1):
                        for d in range(1, min(dir_dmax, len(mids) - 1 - i) + 1):
                            key = (vocab.fold_group_id(mids[i]), vocab.fold_group_id(mids[i + d]))
                            h = q_dir.get(key)
                            if h is not None and key not in seen:
                                seen.add(key)
                                hit += h
                    norm = sum(q_dir.values()) or 1.0
                    if hit > 0:
                        score *= 1.0 + dir_bonus * (hit / norm)
                else:
                    hit = 0.0
                    for i in range(len(mids) - 1):
                        for d in range(1, min(dir_dmax, len(mids) - 1 - i) + 1):
                            h = q_dir.get(
                                (vocab.fold_group_id(mids[i]), vocab.fold_group_id(mids[i + d]))
                            )
                            if h is not None and h > hit:
                                hit = h
                    if hit > 0:
                        score *= 1.0 + dir_bonus * (hit / max_q_idf)
            # độ tin nguồn: cha curated (1.0) áp đảo cha ngoài miền (wiki, <1) cho câu
            # trong miền; wiki chỉ thắng khi là ứng viên tốt nhất/duy nhất (case con kiến).
            score *= rec.priority
            scored.append((pi, score, rec.is_qa and bool(rec.answer_ids)))
        if qa_completion_mode and scored:
            # Chỉ ưu tiên QA khi nó THẬT SỰ cạnh tranh ngang ngửa với ứng viên tốt nhất
            # (README 0.5.12: "khi tranh với QA") — near-tie theo raw score, không phải
            # "có tồn tại một cha QA nào đó, dù yếu, trong top_n". Một cha QA yếu (chỉ
            # trùng vài token generic) không được phép dìm một statement khớp tốt hơn hẳn
            # (case con kiến) — nếu không case con kiến luôn thua vì hầu như câu nào cũng
            # có ít nhất một QA khớp lỏng lẻo.
            top_raw = max(s for _, s, _ in scored)
            best_qa = max((s for _, s, is_qa in scored if is_qa), default=0.0)
            if best_qa >= 0.85 * top_raw:
                scored = [
                    (pi, s if is_qa else s * 0.01, is_qa) for pi, s, is_qa in scored
                ]
        ranked = [(pi, s) for pi, s, _ in scored]
        ranked.sort(key=lambda x: -x[1])
        # TK_045 §13.15 (Trụ D) — PHÁ THẾ HOÀ bằng truy hồi Hopfield hiện đại trên
        # `ParentBipartite.A`. Hoà đỉnh là nút thắt §12.126 đã chỉ: **38.2%** query trên
        # mini, **50.1%** trên v12. Xếp lại CHỈ trong nhóm hoà (bậc điểm chính bất biến):
        #
        #   mini n=400  rank-1 66.50 → 68.75 (+2.25pp) · MRR .697→.750 · hoà 38.2→0.0%
        #   v12  n=800  rank-1 22.25 → 31.13 (+8.88pp) · MRR .288→.409 · hoà 50.1→0.9%
        #
        # ĐẠT cổng G-hop ở CẢ HAI quy mô ⇒ kỷ luật mini→v12 thoả ⇒ tác giả duyệt BẬT
        # mặc định (2026-08-02). `beta=0` để tắt.
        hop_beta = float(getattr(self, "hopfield_tiebreak_beta", HOPFIELD_TIEBREAK_BETA) or 0.0)
        if hop_beta > 0 and len(ranked) >= 2:
            ref = getattr(self, "_owner_ref", None)
            st = ref() if ref is not None else None
            if st is not None:
                from nn_aivie.hca.hopfield import tiebreak_ranked

                ranked = tiebreak_ranked(
                    st, ranked, q_content, [self._gidf(t) for t in q_content],
                    beta=hop_beta,
                    steps=int(getattr(self, "hopfield_tiebreak_steps", 2)),
                )
        _rank_cache[_rank_ck] = ranked  # §184 — lưu ranked đầy đủ để lần sau cắt lại
        return ranked[:top_n]

    def define_entity(
        self, vocab: Vocabulary, entity: str
    ) -> tuple[str, int] | None:
        """§185 (A+B) — TRA ĐỊNH NGHĨA theo CẤU TRÚC cho "X là gì?".

        `rank_matches` (bag-of-content-word) không phân biệt được câu định nghĩa
        với câu khác chứa cùng thực thể (§184: "Sắt là gì?"→"Mỏ… Sắt"); còn
        surface cả câu HỎI ("Chó có chạy không?"). Định nghĩa lại có CẤU TRÚC rõ:
        thực thể X ở VỊ TRÍ CHỦ NGỮ (content token đầu) + copula "là" ngay sau.

        Quét cha STATEMENT (is_qa=False) khớp "ENT là …". Xếp:
          1. ĐƠN-VỊ-NGỮ: chỉ 1 "là" trong ~12 token đầu (loại câu ghép/ví dụ
             "mèo là thú VÀ chó LÀ bạn");
          2. priority (curated > wiki);
          3. power.
        Trả (text, parent_idx) hoặc None. Đây là "A (dùng câu đã truy)" + "B (xếp
        theo cấu trúc thực thể)" — QA định nghĩa trả ĐÚNG mà không cần generate.
        """
        ent_fold = vocab.fold_of(entity)
        la_fold = vocab.fold_of("là")
        if not ent_fold:
            return None

        def _first_content(e: tuple[int, ...]) -> tuple[int, int] | None:
            for k, t in enumerate(e):
                f = vocab.fold(t)
                if f and any(c.isalnum() for c in f):
                    return k, t
            return None

        best_key: tuple = ()
        best: tuple[str, int] | None = None
        for pi, rec in enumerate(self.parents):
            if rec.is_qa:
                continue
            e = rec.emit_ids
            fc = _first_content(e)
            if fc is None:
                continue
            k, t0 = fc
            if vocab.fold(t0) != ent_fold:
                continue          # thực thể phải là CHỦ NGỮ (content đầu tiên)
            if k + 1 >= len(e) or vocab.fold(e[k + 1]) != la_fold:
                continue          # theo sau bởi copula "là"
            n_la = sum(1 for t in e[:12] if vocab.fold(t) == la_fold)
            key = (1 if n_la <= 1 else 0, round(rec.priority, 4), rec.power)
            if best is None or key > best_key:
                best_key = key
                best = (vocab.detokenize(list(e)), pi)
        return best

    def _ctx_vec(self, embedding, vocab: Vocabulary, ids, ent_ids):
        """§188 — VECTOR NGỮ CẢNH: trung bình embedding các content token (trừ
        thực thể), chuẩn hoá. Đại diện NGHĨA của lần xuất hiện trong không gian
        liên tục — "nước sôi" gần "nước chất lỏng" dù không chung TỪ."""
        import numpy as np
        M = embedding.matrix
        cs = [t for t in _content_ids(list(ids), vocab, skip=set())
              if t not in ent_ids and 0 <= t < M.shape[0]]
        if not cs:
            return None
        m = M[np.asarray(cs)].mean(0)
        n = float(np.linalg.norm(m))
        return m / n if n > 0 else None

    def answer_fact(
        self, vocab: Vocabulary, query_ids: list[int], *, top_k: int = 12,
        embedding=None, encoder=None,
    ) -> tuple[str, int] | None:
        """§186 — TRẢ LỜI câu hỏi KHÁC DẠNG ("X ở đâu?", "X ăn gì?", "X sôi ở
        nhiệt độ nào?") bằng truy-hồi-theo-thực-thể `facts_about`.

        `define_entity` (§185) chỉ giải "X là gì?". Câu hỏi khác dạng cần fact VỀ
        thực thể khớp KHÍA CẠNH câu hỏi — đúng việc của `facts_about` (Theme index,
        theme_weight ưu tiên câu thực-thể-đứng-đầu). Không cần danh sách WH viết
        tay: từ hỏi (gì/đâu/nào) tần suất cao ⇒ IDF thấp ⇒ tự nhẹ trong xếp hạng.
        Lọc SAU: bỏ cha `is_qa` / câu kết thúc '?' (fact index đôi khi lọt câu hỏi
        wiki). Trả (text, parent_idx) statement đầu tiên hợp lệ, hoặc None.
        """
        q_content = list(dict.fromkeys(_content_ids(query_ids, vocab, skip=set())))
        if not q_content:
            return None

        # §188 — TÁCH NGHĨA bằng COSINE EMBEDDING neo-định-nghĩa. Thực thể đa nghĩa
        # ("Hổ": cọp vs "hổ thẹn"). Vector ngữ cảnh của ĐỊNH NGHĨA = "nghĩa gốc".
        # Fact đúng nghĩa có ngữ cảnh GẦN (cosine cao) trong không gian embedding;
        # "Hổ thẹn… sống mũi" xa. Liên tục nên KHÔNG loại oan fact khía-cạnh khác
        # từ vựng ("nước sôi" gần "nước chất lỏng"). An toàn: nếu KHÔNG ứng viên
        # nào qua sàn → fallback top khía-cạnh (= hành vi §186, không hồi quy).
        # GATE cụm ghép ("Thủ đô"→"Thủ" không phải entity, def "đầu lợn" sai) →
        # bỏ tách nghĩa.
        _pair_phrase = (
            len(q_content) >= 2
            and (q_content[0], q_content[1]) in self._phrase_lookup
        )
        _ent_surf = vocab.token_of(q_content[0]).lstrip("ĠĊ")
        _def = None
        _ent_ids: set[int] = set()
        if _ent_surf and not _pair_phrase:
            _def = self.define_entity(vocab, _ent_surf)
            if _def is not None:
                _ent_ids = set(self.case_equiv_ids(q_content[0], vocab))

        idxs = self.facts_about(
            tuple(q_content), top_k=top_k, vocab=vocab,
            theme_weight=1.0, min_overlap=0.6,
        )
        cands: list[tuple[str, int]] = []  # (text, pi) — GIỮ thứ tự khía-cạnh
        for pi in idxs:
            rec = self.parents[pi]
            if rec.is_qa:
                continue
            text = vocab.detokenize(list(rec.emit_ids))
            if text.rstrip().endswith("?"):
                continue
            cands.append((text, pi))
        if not cands:
            return None
        import numpy as np

        # §192 — TÁCH NGHĨA bằng ENCODER NGỮ CẢNH (nếu có). Vector-theo-vị-trí tách
        # được nghĩa GẦN (cọp 0.89 vs rắn-hổ-mang 0.27) mà embedding tĩnh không.
        # QUY TẮC KHOẢNG-CÁCH: giữ aspect-top TRỪ KHI có candidate (không phải định
        # nghĩa) cos-ngữ-cảnh CAO HƠN HẲN (> _GAP) → đổi sang nó. Aspect-top đúng
        # nghĩa (Chó ăn thịt) không bị đổi vì không candidate nào vượt xa; sai nghĩa
        # (Hổ thẹn 0.51) bị thay bởi cọp thật (0.89). Loại chính def khỏi so sánh
        # (def cos≈1.0 luôn thắng). Không đủ tín hiệu → fallback tĩnh §188b.
        if encoder is not None and _def is not None:
            _dv = encoder.vec(vocab.encode_prompt(_def[0]), _ent_ids)
            if _dv is not None:
                _GAP = 0.30
                _def_pi = _def[1]
                _top_cos = None
                _best = None  # (cos, text, pi) trong ứng viên KHÔNG-phải-def
                for text, pi in cands:
                    fv = encoder.vec(self.parents[pi].emit_ids, _ent_ids)
                    c = float(np.dot(fv, _dv)) if fv is not None else None
                    if _top_cos is None:
                        _top_cos = c if c is not None else -1.0
                    if pi != _def_pi and c is not None and (_best is None or c > _best[0]):
                        _best = (c, text, pi)
                if _best is not None and _top_cos is not None and _best[0] - _top_cos > _GAP:
                    return (_best[1], _best[2])
                return cands[0]

        # §188b — fallback: cosine embedding TĨNH, lọc sàn giữ thứ tự khía-cạnh.
        if embedding is not None and _def is not None:
            _dvec = self._ctx_vec(embedding, vocab, vocab.encode_prompt(_def[0]), _ent_ids)
            if _dvec is not None:
                _FLOOR = 0.45
                for text, pi in cands:
                    fv = self._ctx_vec(embedding, vocab, self.parents[pi].emit_ids, _ent_ids)
                    if fv is not None and float(np.dot(fv, _dvec)) >= _FLOOR:
                        return (text, pi)
        return cands[0]

    # ---- lan xuôi: chọn nơ-ron cha ----
    def _entity_parent_indices(self, primary: int, vocab: Vocabulary) -> frozenset[int]:
        out: set[int] = set()
        for tid in self.case_equiv_ids(primary, vocab):
            out.update(self._index.get(tid, ()))
        return frozenset(out)

    def _match_has_focus(
        self,
        parent_idx: int,
        focus: tuple[int, ...],
        vocab: Vocabulary,
    ) -> bool:
        """Cha chứa ÍT NHẤT MỘT focus token ĐỦ PHÂN BIỆT (gidf ≥ 75% primary, case-equiv).

        TK_002: bản đầu đòi đúng PRIMARY → giết synonym-paraphrase — "Ai cấy lúa" có
        primary="cấy" (không bao giờ xuất hiện trong cha gold "Ai trồng lúa") dù "lúa",
        cùng độ phân biệt, có mặt → cha đúng bị loại, wiki chứa "cấy" thắng (Syn 33→17).
        Token focus phụ chỉ được tính khi gidf gần primary — chặn token khung yếu ("ai",
        "là" trong "Napoléon là ai") mở toang gate.
        """
        if not focus:
            return True
        top = max(self._gidf(t) for t in focus)
        if top <= 0:
            return True
        strong = [t for t in focus if self._gidf(t) >= 0.75 * top]
        mset = set(self.parents[parent_idx].match_ids)
        return any(self._token_in_seq(t, mset, vocab) for t in strong)

    def _primary_earliest_pos(
        self, parent_idx: int, primary: int, vocab: Vocabulary
    ) -> int:
        """Vị trí đầu tiên primary trong match_ids của cha (nhỏ = topic-first)."""
        mseq = self.parents[parent_idx].match_ids
        variants = self.case_equiv_ids(primary, vocab)
        for i, t in enumerate(mseq):
            if t in variants:
                return i
        return 10**9

    def _distinct_focus_primary(
        self,
        focus: tuple[int, ...],
        vocab: Vocabulary | None = None,
        q_content: tuple[int, ...] | None = None,
    ) -> int:
        """Primary phân biệt — token focus ĐỨNG SAU / PHẢI nhất trong query (Nhà Lý → Lý).

        FB-opener filter không đủ (Lý cũng có cạnh mined). Vị trí query thuần trật tự, Luật 0."""
        if not focus:
            raise ValueError("focus empty")
        if q_content:
            pos = {t: i for i, t in enumerate(q_content)}
            return max(focus, key=lambda t: (pos.get(t, 0), self._gidf(t)))
        fb = getattr(self, "_fb_layer_ref", None)
        cands = list(focus)
        if fb is not None and vocab is not None:
            kept = []
            for t in cands:
                if fb.row(t):
                    continue
                alt = vocab._token_to_id.get(vocab.fold(t))
                if alt is not None and fb.row(alt):
                    continue
                kept.append(t)
            if kept:
                cands = kept
        return max(cands, key=lambda t: self._gidf(t))

    def _parent_has_distinct_primary(
        self,
        parent_idx: int,
        focus: tuple[int, ...],
        vocab: Vocabulary,
        q_content: tuple[int, ...],
    ) -> bool:
        """T3b — cha phải chứa thực thể phân biệt trong phần KHỚP (match_ids), không chỉ khung."""
        try:
            primary = self._distinct_focus_primary(focus, vocab, q_content)
        except ValueError:
            return True
        mset = set(self.parents[parent_idx].match_ids)
        return self._token_in_seq(primary, mset, vocab)

    def _pick_entity_candidate(
        self,
        entity_cands: list[tuple[int, float]],
        focus: tuple[int, ...],
        vocab: Vocabulary,
        q_content: tuple[int, ...] | None = None,
    ) -> tuple[int, float] | None:
        """Chọn ứng viên entity_cands theo score (TK_025: bỏ nhánh topic-first WH-mode)."""
        del focus, vocab, q_content
        if not entity_cands:
            return None
        return max(entity_cands, key=lambda x: x[1])

    def _entity_cands_filtered(
        self,
        allp: list[tuple[int, float]],
        focus: tuple[int, ...],
        vocab: Vocabulary,
        q_content: tuple[int, ...],
    ) -> list[tuple[int, float]]:
        del q_content
        return [(pi, sc) for pi, sc in allp if self._match_has_focus(pi, focus, vocab)]

    def best_match(
        self,
        query_ids: list[int],
        vocab: Vocabulary,
        embedding=None,
        *,
        fuzzy_topk: int = 6,
        fuzzy_min_sim: float = 0.35,
        fuzzy_scale: float = 0.85,
        min_score: float = 0.30,
        curated_priority: float = 0.9,
        pool_min_priority: float = 0.0,
        qa_completion_mode: bool = False,
    ) -> tuple[int, float] | None:
        """
        Two-tier: **thử pool curated (tin cậy) trước**; chỉ rơi xuống pool đầy đủ (gồm wiki) khi
        curated không có cha nào đạt ngưỡng. Nhờ đó câu trong miền LUÔN ổn định dù data wiki lớn
        cỡ nào; wiki chỉ trả lời câu ngoài miền (case con kiến).

        Vá 2026-07-02 (bug "Napoléon là ai"): curated QA khớp khung `là … ai` (token chức năng)
        nhưng KHÔNG chứa thực thể trọng tâm (Napoléon) → không được chặn pool wiki tốt hơn
        (Trung thu là tết của ai). Kiểm tra bằng focus token (gidf), không danh sách từ.
        """
        kw = dict(
            fuzzy_topk=fuzzy_topk, fuzzy_min_sim=fuzzy_min_sim,
            fuzzy_scale=fuzzy_scale,
            qa_completion_mode=qa_completion_mode,
        )
        q_content = _content_ids(query_ids, vocab, skip=set())
        # TK_021: gate crisp GIỮ focus LEGACY (ranking T1 mới lọt vào đây làm Syn 67→33 —
        # đo 07-06; gate đã tinh chỉnh TK_002 quanh focus cũ). Ranking T1 chỉ cho FALLBACK.
        _alpha_save = getattr(self, "focus_pos_alpha", -1.0)
        self.focus_pos_alpha = -1.0
        try:
            focus = self.top_focus_tokens(tuple(q_content), k=2, vocab=vocab) if q_content else ()
        finally:
            self.focus_pos_alpha = _alpha_save

        curated = self.rank_matches(
            query_ids, vocab, embedding, min_priority=curated_priority,
            top_n=1, **kw,
        )
        allp = self.rank_matches(
            query_ids, vocab, embedding, top_n=12, min_priority=pool_min_priority, **kw,
        )

        # TK_025: gỡ WH-mode (readout KNOWLEDGE + theme-first + slot-consistency). Route 1 đường:
        # curated → entity_cands (theo focus) → allp. Không phân-loại câu-hỏi, quyết theo kích hoạt.
        entity_cands = self._entity_cands_filtered(allp, focus, vocab, q_content)

        if curated and curated[0][1] >= min_score:
            if self._match_has_focus(curated[0][0], focus, vocab):
                return curated[0]
            picked = self._pick_entity_candidate(
                entity_cands, focus, vocab, q_content
            )
            if picked is not None:
                return picked
            try:
                primary = self._distinct_focus_primary(focus, vocab, q_content)
            except ValueError:
                primary = max(focus, key=lambda t: self._gidf(t))
            indexed = self._entity_parent_indices(primary, vocab)
            if indexed:
                subset = self.rank_matches(
                    query_ids, vocab, embedding, top_n=1,
                    restrict_parents=indexed, **kw,
                )
                if subset:
                    return subset[0]

        if allp and allp[0][1] >= min_score:
            top_pi, top_sc = allp[0]
            if focus and not self._match_has_focus(top_pi, focus, vocab):
                picked = self._pick_entity_candidate(
                    entity_cands, focus, vocab, q_content
                )
                if picked is not None:
                    return picked
                try:
                    primary = self._distinct_focus_primary(focus, vocab, q_content)
                except ValueError:
                    primary = max(focus, key=lambda t: self._gidf(t))
                indexed = self._entity_parent_indices(primary, vocab)
                if indexed:
                    subset = self.rank_matches(
                        query_ids, vocab, embedding, top_n=1,
                        restrict_parents=indexed, **kw,
                    )
                    if subset:
                        return subset[0]
                return None
            return top_pi, top_sc
        return None

    # ---- composition A1: gom nhiều fact về một thực thể (câu ghép nhiều ý) ----
    def top_focus_tokens(
        self, query_content: tuple[int, ...], *, k: int = 2, vocab: Vocabulary | None = None
    ) -> tuple[int, ...]:
        """Token thực thể phân biệt nhất của câu hỏi (IDF cao) — nơi cần 'chú ý' để gom fact.

        Chỉ xét token **mạng đã biết** (df>0): từ chỉ có trong câu hỏi (vd "thế","nào") có df=0
        → IDF cực đại giả tạo, nhấn chìm thực thể thật ("Hổ") → phải loại khỏi focus.
        """
        reg = self.entity_register_ids(query_content)
        if reg:
            return reg[:k]
        # W3: dùng df TOÀN CỤC — thực thể wiki (df curated=0 nhưng gdf>0) không bị lọc oan;
        # nhiễu chỉ-trong-câu-hỏi ("thế/nào", gdf=0) vẫn bị loại.
        # (TK_020-fix thử fold-sort ở đây: ĐO ÂM TÍNH 07-06 — Syn 67→50 + 4 ca không đổi
        # vì focus thường đi đường entity_register_ids phía trên. REVERT; param vocab giữ
        # tương thích. Gốc thật ở ranking entity_register_ids — việc riêng, xem TK_021.)
        del vocab
        cand = [t for t in dict.fromkeys(query_content) if self._gdf.get(t, 0) > 0]
        cand.sort(key=lambda t: -self._gidf(t))
        return tuple(cand[:k])

    def _ensure_case_equiv(self, vocab: Vocabulary) -> None:
        """Map bề mặt lowercase → mọi token id (Trái Đất ≡ Trái đất — trần tokenize)."""
        if getattr(self, "_case_equiv_ready", False):
            return
        by_key: dict[str, list[int]] = defaultdict(list)
        tid_to_key: dict[int, str] = {}
        for tid in range(len(vocab)):
            if not vocab.is_surface(tid):
                continue
            key = vocab.fold(tid)  # TK_012: thuộc-tính-token precompute
            if not key:
                continue
            by_key[key].append(tid)
            tid_to_key[tid] = key
        self._case_equiv = {k: frozenset(v) for k, v in by_key.items()}
        self._tid_case_key = tid_to_key
        self._case_equiv_ready = True

    def case_equiv_ids(self, token_id: int, vocab: Vocabulary) -> frozenset[int]:
        """Mọi id cùng bề mặt sau chuẩn hóa chữ thường (không đổi vocab — chỉ mở rộng khớp)."""
        self._ensure_case_equiv(vocab)
        key = self._tid_case_key.get(token_id)
        return self._case_equiv.get(key, frozenset({token_id})) if key else frozenset({token_id})

    def _token_in_seq(self, token_id: int, seq: set[int], vocab: Vocabulary | None) -> bool:
        if token_id in seq:
            return True
        if vocab is None:
            return False
        return bool(seq & self.case_equiv_ids(token_id, vocab))

    def entity_register_ids(self, query_content: tuple[int, ...]) -> tuple[int, ...]:
        """
        Thực thể đăng ký cho anaphora (A2) — token từ cụm Re-Pair + bare IDF cao.

        Tương đương "keys" trong attention LLM: các antecedent mà đại từ có thể attend về.
        """
        if not query_content:
            return ()
        q_seq = tuple(dict.fromkeys(query_content))
        units = self.query_match_units(q_seq)
        # W3: xếp hạng MỌI đơn vị (cụm + token rời) CHUNG theo IDF × ưu tiên VỊ TRÍ DẪN ĐẦU.
        # IDF một mình không phân biệt "thế nào" (cụm câu-hỏi hiếm) với "Nhà Lý" (thực thể hiếm)
        # — cả hai IDF cao. Nhưng câu hỏi tiếng Việt đặt THỰC THỂ ở đầu (topic), cụm câu-hỏi ở
        # cuối → chia IDF cho (1 + vị trí bắt đầu) để đơn vị dẫn đầu thắng. Thuần trật tự, Luật 0.
        pos = {t: i for i, t in enumerate(q_seq)}
        ranked: list[tuple[float, int, tuple[int, ...]]] = []
        for j in units.phrase_units:
            if j < len(self.phrases):
                ph = tuple(self.phrases[j])
                start = min((pos.get(t, len(q_seq)) for t in ph), default=len(q_seq))
                ranked.append((self._phrase_idf(j) / (1.0 + start), start, ph))
        # TK_021 T1 (đo 07-06, out/t1_sim.log): ranking BARE cũ `gidf/(1+pos)` giết thực thể
        # đứng sau ("phở" gidf 10.47 CAO NHẤT câu vẫn thua "Cách" nhờ pos=0). Sửa 3 thành phần:
        #   (a) fold-gidf — "Cách/Nhà/Người" hoa-đầu-câu là token hiếm riêng → IDF artifact;
        #   (b) phạt vị trí mềm (1+pos)^alpha, alpha=0.3 — topic-first giữ hướng, bớt độc tài;
        #   (c) demote token là FB-OPENER ("Nếu" có cạnh FRAME_BRACKET → là KHUNG, không phải
        #       thực thể) — đọc từ cạnh mined, Luật 0 sạch.
        # alpha<0 (cờ focus_pos_alpha) → hành vi cũ (ablate).
        alpha = getattr(self, "focus_pos_alpha", 0.3)
        vocab_ref = getattr(self, "_vocab_ref", None)
        use_new = alpha >= 0 and vocab_ref is not None
        if use_new:
            self._ensure_fold_gdf(vocab_ref)
            fb = self._fb_layer_ref  # gán kèm _vocab_ref
        for t in units.bare_tokens:
            if self._gdf.get(t, 0) > 0:  # df toàn cục: giữ thực thể wiki, loại nhiễu câu-hỏi
                start = pos.get(t, len(q_seq))
                if use_new:
                    sc = self._fold_gidf(t, vocab_ref) / ((1.0 + start) ** alpha)
                    if fb is not None and fb.row(t):
                        sc *= 0.1  # opener khung
                    else:
                        alt = vocab_ref._token_to_id.get(vocab_ref.fold(t))
                        if fb is not None and alt is not None and fb.row(alt):
                            sc *= 0.1
                    ranked.append((sc, start, (t,)))
                else:
                    ranked.append((self._gidf(t) / (1.0 + start), start, (t,)))
        ranked.sort(key=lambda x: -x[0])
        out: list[int] = []
        for _score, _start, toks in ranked:
            out.extend(toks)
        return tuple(dict.fromkeys(out))

    def facts_about(
        self,
        focus_ids: tuple[int, ...],
        *,
        top_k: int = 3,
        min_overlap: float = 0.6,
        vocab: Vocabulary | None = None,
        min_priority: float = 0.0,
        cluster_slack: int = 1,
        theme_weight: float = 0.0,
        episode_weight: float = 0.0,
        episode_anchor: int | None = None,
    ) -> list[int]:
        """
        Các nơ-ron cha STATEMENT nói về thực thể `focus_ids`, xếp theo độ liên quan, đã khử trùng.

        Composition A1: thay vì dội một đáp, gom **nhiều fact phân biệt** về cùng thực thể để ghép
        thành câu trả lời nhiều ý. Khử trùng lặp (Jaccard nội dung > min_overlap → giữ power cao).
        `focus_ids[0]` bắt buộc có mặt; các focus phụ chỉ cộng điểm liên quan (IDF).

        Chống wiki rác (đo trực tiếp trên big_a2 — README W-diagnosis):
          - **Cổng cụm liền kề:** khi ≥2 focus token cùng có mặt, vị trí của chúng phải **sát nhau**
            (span ≤ (#present−1)+`cluster_slack`). Loại fact chứa focus token **rải rác** như
            "Phan Trần Chúc (2000) … *Nhà* xuất bản" (Nhà & Trần cách xa) — thực thể phải xuất
            hiện như một CỤM, không phải trùng token tình cờ. Thuần vị trí, không luật từ vựng.
          - **Cân theo priority:** `cov ×= priority` → cha curated (1.0) áp đảo wiki (<1) cùng độ
            phủ, để slot còn lại không bị wiki rác lấp khi đã có fact tin cậy.
          - **`min_priority`>0:** two-tier (caller thử curated trước) như `best_match`.
          - **Dedup theo nội dung bỏ dấu câu** (cần `vocab`): "Bác sĩ chữa bệnh." ≡ "Bác sĩ chữa
            bệnh" (chỉ khác token `.`).
          - **`theme_weight`>0 (v4 P1):** nhân điểm với `(0.5 + theme_weight×w_theme)` từ nơ-ron
            chủ đề — fact mà thực thể đứng ĐẦU câu (câu VỀ nó) thắng fact chỉ NHẮC nó giữa câu
            ("Napoléon trốn khỏi Elba" > "cuộc chiến tranh chống Napoléon của vua Gustav…").
            Sàn 0.5 để fact thực-thể-giữa-câu hợp lệ (Bảo tàng Hồ Chí Minh) không bị triệt.
        """
        if not focus_ids:
            return []
        focus = list(dict.fromkeys(focus_ids))
        primary = self.pick_primary(focus, vocab)

        def fact_ids(rec: ParentRecord) -> tuple[int, ...]:
            # "fact" = phần nội dung dội ra: đáp (QA) hoặc cả câu (statement) — cả hai đều là ý
            return rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids

        _PUNCT_STRIP = ".,;:!?…\"'()[]"

        def content_key(rec: ParentRecord) -> frozenset:
            ids = fact_ids(rec)
            if vocab is None:
                return frozenset(ids)
            # Chuẩn hóa BỀ MẶT: bỏ dấu dính token ("bệnh."→"bệnh") để dedup gộp bản QA (token
            # dính dấu, R0 chưa tách) với bản statement — Jaccard exact-id không gộp được.
            surfaces = set()
            for t in _content_ids(list(ids), vocab, skip=set()):
                surfaces.add(vocab.fold(t))  # TK_012: thuộc-tính-token precompute
            surfaces.discard("")
            return frozenset(surfaces)

        def clustered(fseq: tuple[int, ...]) -> bool:
            """Các focus token có mặt phải nằm sát nhau (thực thể như một cụm, không rải rác)."""
            pos = {}
            for i, t in enumerate(fseq):
                if t in focus and t not in pos:
                    pos[t] = i
            if len(pos) < 2:
                return True  # 0/1 token → không có gì để 'rải rác'
            span = max(pos.values()) - min(pos.values())
            return span <= (len(pos) - 1) + cluster_slack

        # v4 P1: nơ-ron chủ đề = xương sống — ứng viên đọc từ Concept:entity (fact chứa primary,
        # kèm trọng số theme) thay vì quét phẳng toàn bộ cha. theme_weight=0 → quét phẳng như cũ.
        theme = self.theme_neuron(primary, vocab) if theme_weight > 0 else None
        if theme is not None:
            cand: list[tuple[int, float]] = list(theme.facts)
        else:
            cand = [
                (pi, 0.0) for pi, rec in enumerate(self.parents)
                if self._token_in_seq(primary, set(fact_ids(rec)), vocab)
            ]

        scored: list[tuple[int, float, int]] = []
        for pi, w_theme in cand:
            rec = self.parents[pi]
            if rec.priority < min_priority:
                continue
            fseq = fact_ids(rec)
            fset = set(fseq)
            if not clustered(fseq):
                continue
            cov = sum(
                self._idf(t) for t in focus if self._token_in_seq(t, fset, vocab)
            ) * rec.priority
            if theme is not None:
                cov *= 0.5 + theme_weight * w_theme
            if episode_weight > 0 and episode_anchor is not None and self.episodes:
                shared = len(
                    set(self.episodes_of(episode_anchor)) & set(self.episodes_of(pi))
                )
                if shared:
                    cov *= 1.0 + episode_weight * shared
            scored.append((pi, cov, rec.power))
        scored.sort(key=lambda x: (-x[1], -x[2]))
        picked: list[int] = []
        used: list[frozenset[int]] = []
        for pi, _cov, _pw in scored:
            s = content_key(self.parents[pi])
            if any(len(s & u) / max(1, len(s | u)) > min_overlap for u in used):
                continue
            picked.append(pi)
            used.append(s)
            if len(picked) >= top_k:
                break
        return picked

    def _parent_stimulus_raw(
        self,
        rec: ParentRecord,
        pidx: int,
        q_content: tuple[int, ...] | list[int],
        q_set: set[int],
        *,
        seed_tokens: frozenset[int] | None = None,
    ) -> tuple[float, dict[int, float]]:
        """Tổng trọng số kích thích lan xuôi query→cha (TK_023 — cùng quy ước viz).

        Token qua cụm ISG: idf(token)×cluster.power; token rời slot_chain: seg_link.weight;
        phrase khớp: phrase_idf; token còn lại trên match_ids: idf(token). Không tính kép.
        `seed_tokens` — chỉ tính kích thích từ subset query (vd. span entity).
        """
        if seed_tokens is not None:
            q_set = q_set & seed_tokens
            q_content = tuple(t for t in q_content if t in q_set)
            if not q_set:
                return 0.0, {}
        total = 0.0
        by_token: dict[int, float] = {}
        parent_clusters = self.seg.clusters_in_match_span(rec) if self.seg.clusters else ()
        parent_phrases = self.parent_phrases_for_query(rec, q_set, phrase_max_df=None)
        cluster_bigrams: set[tuple[int, int]] = set()
        for ci in parent_clusters:
            c = self.seg.clusters[ci]
            cluster_bigrams.add((c.left, c.right))
        covered: set[int] = set()

        for ci in parent_clusters:
            c = self.seg.clusters[ci]
            cp = float(c.power)
            for t in (c.left, c.right):
                if t not in q_set:
                    continue
                w = self._idf(t) * cp
                total += w
                by_token[t] = by_token.get(t, 0.0) + w
                covered.add(t)

        if rec.slot_chain and rec.match_span != (0, 0):
            ms, me = rec.match_span
            for s in rec.slot_chain[ms:me]:
                if s.kind != "token" or s.id not in q_set:
                    continue
                covered.add(s.id)
                slot_w = 1.0
                for ln in self.seg.parent_links(pidx):
                    if ln.src.kind == "token" and ln.src.id == s.id:
                        slot_w = max(slot_w, ln.weight)
                    if ln.dst.kind == "token" and ln.dst.id == s.id:
                        slot_w = max(slot_w, ln.weight)
                total += slot_w
                by_token[s.id] = by_token.get(s.id, 0.0) + slot_w

        for j in parent_phrases:
            ph_tokens = self.phrases[j]
            if len(ph_tokens) == 2 and (ph_tokens[0], ph_tokens[1]) in cluster_bigrams:
                continue
            w = self._phrase_idf(j)
            total += w
            ph_hits = [t for t in ph_tokens if t in q_set]
            if ph_hits:
                share = w / len(ph_hits)
                for t in ph_hits:
                    by_token[t] = by_token.get(t, 0.0) + share
            covered.update(ph_tokens)

        pset = set(rec.match_ids)
        for qid in q_content:
            if qid not in pset or qid in covered:
                continue
            w = self._idf(qid)
            total += w
            by_token[qid] = by_token.get(qid, 0.0) + w

        return total, by_token

    def parent_stimulus_score(
        self,
        pi: int,
        query_ids: list[int],
        vocab: Vocabulary,
        *,
        theme_anchor: int | None = None,
        theme_weight: float = 0.0,
        seed_tokens: frozenset[int] | None = None,
    ) -> float:
        """Điểm kích thích tổng của cha `pi` từ query (× priority, × theme nếu bật)."""
        q_seq = _content_ids(query_ids, vocab, skip=set())
        q_content = tuple(dict.fromkeys(q_seq))
        if not q_content:
            return 0.0
        rec = self.parents[pi]
        raw, _ = self._parent_stimulus_raw(
            rec, pi, q_content, set(q_content), seed_tokens=seed_tokens,
        )
        score = raw * rec.priority
        if theme_weight > 0 and theme_anchor is not None:
            theme = self.theme_neuron(theme_anchor, vocab)
            if theme is not None:
                w_map = dict(theme.facts)
                if pi in w_map:
                    score *= 0.5 + theme_weight * w_map[pi]
        return score

    def query_coactivation_field(
        self,
        q_content: tuple[int, ...],
        co_layer,
        vocab: Vocabulary,
        *,
        cap_per_seed: int = 40,
        min_field_idf: float = 3.5,
    ) -> dict[int, float]:
        """TK_025 P2 — TRƯỜNG cộng-hưởng của query: lân cận CO_ACTIVATION của MỌI token query,
        gộp lại (trọng số idf seed, chuẩn hoá per-seed chống token phổ biến tràn). Ngữ-cảnh
        'Nhà'+'Lý' cùng nhấn token triều-đại (Thái Tổ, Thường Kiệt); token vật-lý (thuyết, điện)
        yếu vì KHÔNG đồng-kích-hoạt với 'Nhà'. Thuần kích-hoạt đồ-thị (Luật 0), không rule/chuỗi/sense.

        LỌC (đo TK_025 §P2): (1) bỏ chính token query — tránh vòng fact-đồng-tự chứa 'Lý' cộng
        hưởng giả; (2) bỏ dấu câu + phi-bề-mặt; (3) bỏ hư-từ (gidf < min_field_idf) — hư-từ (`,`
        `là` `của`) đồng-kích-hoạt với MỌI thứ → nhấn chìm ngữ-cảnh, làm resonance mất phân biệt."""
        seeds = set(q_content)
        field: dict[int, float] = defaultdict(float)
        for t in dict.fromkeys(q_content):
            row = co_layer.row(t)
            if not row:
                continue
            items = sorted(row.items(), key=lambda kv: -kv[1])[:cap_per_seed]
            norm = sum(w for _, w in items) or 1.0
            wt = self._gidf(t)
            for nb, w in items:
                if nb in seeds:
                    continue
                if not vocab.is_surface(nb):
                    continue
                if self._gidf(nb) < min_field_idf:
                    continue
                field[nb] += wt * (w / norm)
        return dict(field)

    def _ensure_sense_index(self, vocab: Vocabulary) -> None:
        """TK_026 — chỉ mục fold(lemma-đầu) → [Sense:*]. Head 'lý thuyết' → lemma 'lý' (fold 'lý')."""
        if getattr(self, "_sense_by_fold_ready", False):
            return
        idx: dict[str, list[int]] = defaultdict(list)
        for tok, sid in vocab._token_to_id.items():
            if not tok.startswith("Sense:"):
                continue
            head = tok[6:].rsplit("#", 1)[0].split()[0]
            hid = vocab._token_to_id.get(head)
            if hid is None:
                continue
            idx[vocab.fold(hid)].append(sid)
        self._sense_by_fold = dict(idx)
        self._sense_by_fold_ready = True

    def query_sense_profile(
        self,
        q_content: tuple[int, ...],
        co_layer,
        vocab: Vocabulary,
        *,
        min_field_idf: float = 3.0,
    ) -> dict[int, float]:
        """TK_026 — TRƯỜNG NGHĨA của query = gộp hồ-sơ (co-act) các `Sense:*` mà lemma-đầu fold
        trùng token query. Khác `query_coactivation_field`: nguồn là ĐỊNH-NGHĨA từ-điển (curated,
        sạch) chứ không phải đồng-hiện corpus (nhiễu). 'Mây' → hồ sơ mây#1(nước/khí)+mây#2(song/ghế)
        → fact 'Mây bao gồm giọt nước' cộng hưởng, 'Đỉnh Mây Bạc' (tên riêng) = 0. Luật 0 (kích
        hoạt nơ-ron nghĩa sẵn), không rule/chuỗi."""
        self._ensure_sense_index(vocab)
        seeds = set(q_content)
        field: dict[int, float] = defaultdict(float)
        seen: set[int] = set()
        for t in dict.fromkeys(q_content):
            for sid in self._sense_by_fold.get(vocab.fold(t), ()):
                if sid in seen:
                    continue
                seen.add(sid)
                for nb, w in co_layer.row(sid).items():
                    if nb in seeds or w <= 0:
                        continue
                    if not vocab.is_surface(nb):
                        continue
                    if self._gidf(nb) < min_field_idf:
                        continue
                    field[nb] += w
        return dict(field)

    def _ensure_entity_index(self, vocab: Vocabulary) -> None:
        if getattr(self, "_entity_ids_ready", False):
            return
        self._entity_ids = [
            i for t, i in vocab._token_to_id.items() if t.startswith("Entity:")
        ]
        self._entity_id_set = frozenset(self._entity_ids)
        self._entity_ids_ready = True

    def _ensure_entity_title_index(self, vocab: Vocabulary) -> None:
        if getattr(self, "_entity_title_ready", False):
            return
        self._ensure_entity_index(vocab)
        # title fold-token-set mỗi entity → tra thực-thể query theo CỤM (không resonance, tránh
        # bẫy down-weight token tên ở step 2).
        self._entity_title_folds: dict[int, frozenset[int]] = {}
        for eid in self._entity_ids:
            title = vocab.token_of(eid)[7:]  # bỏ "Entity:"
            folds = set()
            for w in title.split():
                tid = vocab._token_to_id.get(w)
                if tid is not None and is_surface_token(w):
                    folds.add(vocab.fold_group_id(tid))
            if folds:
                self._entity_title_folds[eid] = frozenset(folds)
        self._entity_title_ready = True

    def query_entity_field(
        self,
        entity_tokens: tuple[int, ...],
        co_layer,
        vocab: Vocabulary,
        *,
        min_overlap: int = 2,
    ) -> dict[int, float]:
        """TK_027 bước 3 — TRƯỜNG neo từ THỰC-THỂ CỦA QUERY: tìm Entity:title trùng cụm query
        ("Nhà Lý" → Entity:Nhà Lý) bằng OVERLAP fold title (không resonance → tránh bẫy down-weight
        token tên). Hồ-sơ entity đó = trường dynasty SẠCH, độc lập câu-đáp-primary (có thể nhiễm,
        vd 'con cháu Lý sang Cao Ly'). Fact citation/Hàn-Quốc dot thấp với trường này → rớt.

        `min_overlap=2`: chỉ khi khớp ≥2 token title (cụm ĐẦY ĐỦ, vd "Nhà Lý"). Overlap 1 (chỉ
        "lý" của "Triều Lý" → dính Địa-lý/Vật-lý) KHÔNG dùng → trả rỗng, fallback neo primary."""
        self._ensure_entity_title_index(vocab)
        qf = {vocab.fold_group_id(t) for t in entity_tokens}
        if not qf:
            return {}
        # YÊU CẦU: title chứa ĐỦ token entity query (ov == |qf|). "Mây"(1)→title "Mây" khớp;
        # "Nhà Lý"(2)→"Nhà Lý" khớp; "Triều Lý"(2)→chỉ "lý" (không title nào có cả 2) → bỏ (tránh
        # dính "Định lý"/physics). Trong số ĐỦ, ưu tiên title NGẮN NHẤT (frac cao = "Mây" hơn
        # "Mây (thực vật)") → nghĩa chính, không pha loãng.
        need = len(qf)
        scored = []
        for eid, tfolds in self._entity_title_folds.items():
            if len(qf & tfolds) == need:  # title phủ hết token entity query
                scored.append((len(qf) / len(tfolds), eid))
        if not scored:
            return {}
        scored.sort(reverse=True)
        best_frac = scored[0][0]
        field: dict[int, float] = defaultdict(float)
        for frac, eid in scored:
            if frac < best_frac:  # chỉ gộp title khớp ĐẦY ĐỦ NHẤT (ngắn nhất)
                break
            for t, w in co_layer.row(eid).items():
                if vocab.is_surface(t):
                    field[t] += w
        return dict(field)

    def query_entity_resonance_field(
        self,
        entity_tokens: tuple[int, ...],
        co_layer,
        vocab: Vocabulary,
        *,
        top_k: int = 3,
        min_idf: float = 3.0,
        cloud_cap: int = 60,
    ) -> dict[int, float]:
        """TK_027 bước 5 — FALLBACK khi title-match trượt ("Triều Lý" không title nào chứa đủ
        Triều+Lý). Để MẠNG TỰ tìm thực-thể qua CỘNG HƯỞNG ngữ cảnh, không so-chữ/merge-tay:
        (1) mở token entity query → ĐÁM MÂY ngữ cảnh (lân cận co-act, idf, bỏ hư-từ/entity);
        (2) chấm các Entity LÂN CẬN token query bằng đám mây → top-K; (3) gộp hồ-sơ top-K.
        'Triều Lý' → mây {nhà,quân,Tông} → top Lý Nam Đế/Lý Thường Kiệt/Nhà Trần (triều VN),
        Korea chìm. triều≈nhà TỰ NỔI từ đồng-hiện, không phải bảng tay."""
        self._ensure_entity_index(vocab)
        seeds = set(entity_tokens)
        cloud: dict[int, float] = defaultdict(float)
        for t in entity_tokens:
            row = co_layer.row(t)
            if not row:
                continue
            items = [
                (n, w) for n, w in row.items()
                if n not in self._entity_id_set and n not in seeds
                and vocab.is_surface(n)
                and self._gidf(n) >= min_idf
            ]
            items.sort(key=lambda x: -x[1])
            items = items[:cloud_cap]
            nrm = sum(w for _, w in items) or 1.0
            wt = self._gidf(t)
            for n, w in items:
                cloud[n] += wt * (w / nrm)
        if not cloud:
            return {}
        cand: set[int] = set()
        for t in entity_tokens:
            for nb in co_layer.row(t):
                if nb in self._entity_id_set:
                    cand.add(nb)
        scored = []
        for eid in cand:
            p = co_layer.row(eid)
            s = sum(p.get(n, 0.0) * self._gidf(n) for n in cloud)
            if s > 0:
                scored.append((s, eid))
        if not scored:
            return {}
        scored.sort(reverse=True)
        field: dict[int, float] = defaultdict(float)
        for _s, eid in scored[:top_k]:
            for t, w in co_layer.row(eid).items():
                if vocab.is_surface(t):
                    field[t] += w
        return dict(field)

    def entity_anchor_field(
        self,
        anchor_content: tuple[int, ...],
        co_layer,
        vocab: Vocabulary,
        *,
        top_entities: int = 6,
        min_res: float = 0.1,
    ) -> dict[int, float]:
        """TK_027 — TRƯỜNG neo thực-thể: từ token NEO (câu đáp chính), tìm top thực-thể cộng
        hưởng (dot-product hồ-sơ), gộp hồ-sơ chúng (trọng số theo độ liên quan). 'Nhà Trần thay
        nhà Lý' → neo thực-thể triều-đại → trường = từ vựng triều-đại → fact lý-thuyết (0 với
        trường này) rớt. Lan 2-hop qua nơ-ron thực-thể (Luật 0), key cụm không fold."""
        self._ensure_entity_index(vocab)
        toks = list(dict.fromkeys(anchor_content))
        if not toks:
            return {}
        # PERF: entity ỨNG VIÊN = lân cận CO_ACTIVATION của token neo (cạnh token↔entity tạo lúc
        # ingest) — thay quét TẤT CẢ 3711 entity (chậm ×row/lần sinh). Token neo chỉ nối vài entity.
        cand: set[int] = set()
        for t in toks:
            row_t = co_layer.row(t)
            if not row_t:
                continue
            for nb in row_t:
                if nb in self._entity_id_set:
                    cand.add(nb)
        scored: list[tuple[float, int]] = []
        for eid in cand:
            prof = co_layer.row(eid)
            if not prof:
                continue
            s = sum(prof.get(t, 0.0) * self._gidf(t) for t in toks)
            if s > min_res:
                scored.append((s, eid))
        scored.sort(reverse=True)
        field: dict[int, float] = defaultdict(float)
        for s, eid in scored[:top_entities]:
            for t, w in co_layer.row(eid).items():
                if vocab.is_surface(t):
                    field[t] += w * s
        return dict(field)

    def parent_field_dot(self, pi: int, field: dict[int, float]) -> float:
        """Dot-product hồ-sơ fact với TRƯỜNG (không chia độ dài) — fact ngoài domain → ~0."""
        if not field:
            return 0.0
        return sum(
            field.get(t, 0.0) * self._gidf(t)
            for t in dict.fromkeys(self.parents[pi].match_ids)
        )

    def parent_coactivation_resonance(self, pi: int, field: dict[int, float]) -> float:
        """Cộng-hưởng của fact `pi` với TRƯỜNG query = trung bình (idf-weighted) field trên token
        nội dung fact. Fact ngữ-cảnh-khác (lý-thuyết vật-lý) → gần 0 dù CHỨA token 'Lý'."""
        if not field:
            return 0.0
        toks = list(dict.fromkeys(self.parents[pi].match_ids))
        if not toks:
            return 0.0
        num = sum(field.get(t, 0.0) * self._gidf(t) for t in toks)
        den = sum(self._gidf(t) for t in toks) or 1.0
        return num / den

    def parent_salient_stimulus(
        self,
        pi: int,
        query_ids: list[int],
        vocab: Vocabulary,
        salient: int,
        *,
        theme_anchor: int | None = None,
        theme_weight: float = 0.0,
    ) -> float:
        """Phần kích thích từ một token salient (cổng chống generic bleed)."""
        q_seq = _content_ids(query_ids, vocab, skip=set())
        q_content = tuple(dict.fromkeys(q_seq))
        if not q_content or salient not in q_content:
            return 0.0
        rec = self.parents[pi]
        _, by_token = self._parent_stimulus_raw(rec, pi, q_content, set(q_content))
        score = by_token.get(salient, 0.0) * rec.priority
        if theme_weight > 0 and theme_anchor is not None:
            theme = self.theme_neuron(theme_anchor, vocab)
            if theme is not None:
                w_map = dict(theme.facts)
                if pi in w_map:
                    score *= 0.5 + theme_weight * w_map[pi]
        return score

    def rank_facts_by_stimulus(
        self,
        query_ids: list[int],
        vocab: Vocabulary,
        *,
        top_k: int = 8,
        min_priority: float = 0.0,
        min_overlap: float = 0.6,
        theme_anchor: int | None = None,
        theme_weight: float = 0.0,
        salient: int | None = None,
        min_salient_ratio: float = 0.15,
        restrict_parents: frozenset[int] | None = None,
        require_token: int | None = None,
        focus_ids: tuple[int, ...] | None = None,
    ) -> list[tuple[int, float]]:
        """Xếp fact theo tổng trọng số kích thích lan xuôi (TK_023).

        Ứng viên: index từ `focus_ids` hoặc query_match_units; `focus_ids` cũng giới hạn
        token seed khi chấm điểm (bỏ frame WH). Dedup nội dung giống facts_about.
        """
        q_seq = _content_ids(query_ids, vocab, skip=set())
        q_content = tuple(dict.fromkeys(q_seq))
        if not q_content:
            return []

        def fact_ids(rec: ParentRecord) -> tuple[int, ...]:
            return rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids

        _PUNCT_STRIP = ".,;:!?…\"'()[]"

        def content_key(rec: ParentRecord) -> frozenset:
            ids = fact_ids(rec)
            surfaces = set()
            for t in _content_ids(list(ids), vocab, skip=set()):
                surfaces.add(vocab.fold(t))
            surfaces.discard("")
            return frozenset(surfaces)

        cand_idx: set[int] = set()
        if restrict_parents is not None:
            cand_idx = set(restrict_parents)
        elif focus_ids:
            for t in focus_ids:
                cand_idx.update(self._index.get(t, ()))
                for j, ph in enumerate(self.phrases):
                    if ph and t in ph:
                        cand_idx.update(self._phrase_index.get(j, ()))
        elif theme_weight > 0 and theme_anchor is not None:
            theme = self.theme_neuron(theme_anchor, vocab)
            if theme is not None:
                cand_idx = {pi for pi, _ in theme.facts}
        if not cand_idx:
            units = self.query_match_units(q_seq)
            for j in units.phrase_units:
                cand_idx.update(self._phrase_index.get(j, ()))
            for q in units.bare_tokens:
                cand_idx.update(self._index.get(q, ()))

        salient_tok = salient
        if salient_tok is None and q_content:
            focus = self.top_focus_tokens(q_content, k=1, vocab=vocab)
            salient_tok = focus[0] if focus else None
        salient_floor = 0.0
        if salient_tok is not None and min_salient_ratio > 0:
            salient_floor = min_salient_ratio * self._idf(salient_tok)

        seed = frozenset(focus_ids) if focus_ids else None
        scored: list[tuple[int, float, int]] = []
        q_set = set(q_content)
        for pi in cand_idx:
            rec = self.parents[pi]
            if rec.priority < min_priority:
                continue
            if not fact_ids(rec):
                continue
            if require_token is not None and require_token not in set(fact_ids(rec)):
                continue
            raw, _ = self._parent_stimulus_raw(
                rec, pi, q_content, q_set, seed_tokens=seed,
            )
            if raw <= 0:
                continue
            score = raw * rec.priority
            if theme_weight > 0 and theme_anchor is not None:
                theme = self.theme_neuron(theme_anchor, vocab)
                if theme is not None:
                    w_map = dict(theme.facts)
                    if pi in w_map:
                        score *= 0.5 + theme_weight * w_map[pi]
            if salient_tok is not None and salient_floor > 0:
                _, by_tok = self._parent_stimulus_raw(
                    rec, pi, q_content, q_set, seed_tokens=seed,
                )
                sal = by_tok.get(salient_tok, 0.0) * rec.priority
                if theme_weight > 0 and theme_anchor is not None:
                    theme = self.theme_neuron(theme_anchor, vocab)
                    if theme is not None:
                        w_map = dict(theme.facts)
                        if pi in w_map:
                            sal *= 0.5 + theme_weight * w_map[pi]
                if sal < salient_floor:
                    continue
            scored.append((pi, score, rec.power))

        scored.sort(key=lambda x: (-x[1], -x[2]))
        picked: list[tuple[int, float]] = []
        used: list[frozenset] = []
        for pi, sc, _pw in scored:
            s = content_key(self.parents[pi])
            if any(len(s & u) / max(1, len(s | u)) > min_overlap for u in used):
                continue
            picked.append((pi, sc))
            used.append(s)
            if len(picked) >= top_k:
                break
        return picked

    def stimulus_activation_edges(
        self,
        pi: int,
        q_content: list[int],
        vocab: Vocabulary,
        *,
        is_matched: bool,
        nodes: list,
        edges: list,
        tok,
    ) -> float:
        """Vẽ cạnh lan xuôi viz + trả parent_stimulus_score (TK_023 — một nguồn sự thật)."""
        rec = self.parents[pi]
        q_set = set(q_content)
        act = "activate" if is_matched else "activate_alt"
        parent_clusters = self.seg.clusters_in_match_span(rec) if self.seg.clusters else ()
        parent_phrases = self.parent_phrases_for_query(rec, q_set, phrase_max_df=None)
        cluster_bigrams: set[tuple[int, int]] = set()
        for ci in parent_clusters:
            c = self.seg.clusters[ci]
            cluster_bigrams.add((c.left, c.right))
        covered: set[int] = set()

        def append_edge(edge: dict) -> None:
            if edge not in edges:
                edges.append(edge)

        for ci in parent_clusters:
            c = self.seg.clusters[ci]
            cnid = f"cl:{ci}"
            if not any(n["id"] == cnid for n in nodes):
                nodes.append(
                    {
                        "id": cnid,
                        "label": vocab.detokenize([c.left, c.right]),
                        "kind": "cluster",
                        "power": c.power,
                    }
                )
            slot_pos = self._cluster_slot_index_in_parent(rec, ci)
            append_edge(
                {
                    "source": cnid,
                    "target": f"p:{pi}",
                    "weight": float(c.power),
                    "kind": "seg_cluster",
                    "slot_index": slot_pos,
                    "parent": pi,
                }
            )
            for t in (c.left, c.right):
                if t in q_set:
                    append_edge(
                        {
                            "source": f"q:{t}",
                            "target": cnid,
                            "weight": float(c.power),
                            "kind": "seg_compose",
                            "slot_index": self._cluster_token_link_index(ci, t),
                        }
                    )
                    covered.add(t)

        if rec.slot_chain and rec.match_span != (0, 0):
            ms, me = rec.match_span
            for i, s in enumerate(rec.slot_chain[ms:me]):
                if s.kind != "token" or s.id not in q_set:
                    continue
                covered.add(s.id)
                slot_ix = i + 1
                slot_w = 1.0
                for ln in self.seg.parent_links(pi):
                    if ln.src.kind == "token" and ln.src.id == s.id:
                        slot_w = max(slot_w, ln.weight)
                    if ln.dst.kind == "token" and ln.dst.id == s.id:
                        slot_w = max(slot_w, ln.weight)
                append_edge(
                    {
                        "source": f"q:{s.id}",
                        "target": f"p:{pi}",
                        "weight": slot_w,
                        "kind": "seg_slot",
                        "slot_index": slot_ix,
                        "parent": pi,
                    }
                )

        for j in parent_phrases:
            ph_tokens = self.phrases[j]
            if len(ph_tokens) == 2 and (ph_tokens[0], ph_tokens[1]) in cluster_bigrams:
                continue
            covered.update(ph_tokens)
            pnid = f"ph:{j}"
            if not any(n["id"] == pnid for n in nodes):
                nodes.append(
                    {
                        "id": pnid,
                        "label": vocab.detokenize(ph_tokens),
                        "kind": "phrase",
                        "df": self.phrase_df[j],
                    }
                )
            for t in ph_tokens:
                if t in q_set:
                    append_edge(
                        {
                            "source": f"q:{t}",
                            "target": pnid,
                            "weight": 1.0,
                            "kind": "compose",
                        }
                    )
            append_edge(
                {"source": pnid, "target": f"p:{pi}", "weight": 1.0, "kind": act}
            )

        pset = set(rec.match_ids)
        for qid in q_content:
            if qid not in pset or qid in covered:
                continue
            slot_ix = self._standalone_token_slot_index(rec, qid)
            act_edge: dict = {
                "source": f"q:{qid}",
                "target": f"p:{pi}",
                "weight": round(self._idf(qid), 2),
                "kind": act,
            }
            if slot_ix is not None:
                act_edge["slot_index"] = slot_ix
                act_edge["parent"] = pi
            append_edge(act_edge)

        if rec.slot_chain:
            ms, me = rec.match_span
            if me > ms:
                visible: set[str] = set()
                for s in rec.slot_chain[ms:me]:
                    nid = self._slot_viz_id(s, q_content)
                    if nid is not None:
                        visible.add(nid)
                for ln in self.seg.parent_links(pi):
                    src = self._slot_viz_id(ln.src, q_content)
                    dst = self._slot_viz_id(ln.dst, q_content)
                    if src is None or dst is None or src not in visible or dst not in visible:
                        continue
                    append_edge(
                        {
                            "source": src,
                            "target": dst,
                            "weight": ln.weight,
                            "kind": "seg_h",
                            "parent": ln.parent_idx,
                            "slot_index": ln.slot_index,
                        }
                    )

        return self.parent_stimulus_score(pi, list(q_content), vocab)

    def _cluster_slot_index_in_parent(self, rec: ParentRecord, cluster_idx: int) -> int | None:
        """Vị trí 1-based của cụm trong vùng match của cha."""
        if not rec.slot_chain or rec.match_span == (0, 0):
            return None
        ms, me = rec.match_span
        cid = CLUSTER_OFFSET + cluster_idx
        for i, s in enumerate(rec.slot_chain[ms:me]):
            if s.kind == "cluster" and s.id == cid:
                return i + 1
        return None

    def _cluster_token_link_index(self, cluster_idx: int, token_id: int) -> int | None:
        for tid, li in self.seg.cluster_children(cluster_idx):
            if tid == token_id:
                return li
        return None

    def _standalone_token_slot_index(self, rec: ParentRecord, token_id: int) -> int | None:
        if not rec.slot_chain or rec.match_span == (0, 0):
            return None
        ms, me = rec.match_span
        for i, s in enumerate(rec.slot_chain[ms:me]):
            if s.kind == "token" and s.id == token_id:
                return i + 1
        return None

    @staticmethod
    def _slot_viz_id(ref: SlotRef, q_content: list[int]) -> str | None:
        if ref.kind == "cluster":
            return f"cl:{ref.id - CLUSTER_OFFSET}"
        if ref.id in q_content:
            return f"q:{ref.id}"
        return None

    def theme_neuron(self, entity: int, vocab: Vocabulary | None = None) -> ThemeNeuron | None:
        """
        Nơ-ron chủ đề của `entity` — build lazy lần đầu hỏi, cache trong store (tất định,
        read-only; hoạt động trên checkpoint đã build, không cần train lại — như `seq_next`).

        fact = statement/đáp-QA chứa entity; theme_weight = 1/(1+vị_trí_xuất_hiện_đầu) —
        đầu câu ≈1.0 (câu VỀ entity), giữa/cuối câu → nhỏ dần (câu chỉ NHẮC entity).
        """
        themes = getattr(self, "_themes", None)
        if themes is None:
            themes = {}
            self._themes = themes
        if entity in themes:
            return themes[entity]
        facts: list[tuple[int, float]] = []
        power = 0
        variants = self.case_equiv_ids(entity, vocab) if vocab is not None else frozenset({entity})
        for pi, rec in enumerate(self.parents):
            fseq = rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids
            pos = next((i for i, t in enumerate(fseq) if t in variants), None)
            if pos is None:
                continue
            facts.append((pi, 1.0 / (1.0 + pos)))
            power += rec.power
        neuron = ThemeNeuron(entity, tuple(sorted(facts, key=lambda f: -f[1])), power) if facts else None
        themes[entity] = neuron
        return neuron

    # ---- nơ-ron trừu tượng t1 (episode — v4 Trụ 7) ----
    _EP_PUNCT = ".,;:!?…\"'()[]"

    def _fold(self, t: int, vocab: Vocabulary) -> str:
        """Bề mặt case-fold (TK_007: 'Phở'≡'phở' — cùng lớp case-equiv TK_002).
        TK_012: đọc thuộc-tính-token precompute (không strip/lower string lúc match)."""
        return vocab.fold(t)

    def _ensure_phrase_assoc_ids(self, vocab: Vocabulary) -> None:
        """TK_012: đổi phrase_assoc legacy (khóa fold-string) → khóa fold-group-id, 1 lần."""
        legacy = getattr(self, "_phrase_assoc_legacy_str", None)
        if not legacy:
            return
        fg = vocab._fold_to_gid

        def _gid_bg(bg):
            g0, g1 = fg.get(bg[0]), fg.get(bg[1])
            return (g0, g1) if g0 is not None and g1 is not None else None

        converted: dict[tuple[int, int], list] = {}
        for a, row in legacy.items():
            ga = _gid_bg(a)
            if ga is None:
                continue
            new_row = [(gb, w) for b, w in row if (gb := _gid_bg(b)) is not None]
            if new_row:
                converted[ga] = new_row
        self._phrase_assoc = converted
        self._phrase_assoc_legacy_str = None

    def _ensure_ep_bigram_df(self, vocab: Vocabulary) -> None:
        """DF của BIGRAM trên tập instruction episode SFT — lazy. TK_012: khóa fold-group-id."""
        self._ensure_phrase_assoc_ids(vocab)
        if getattr(self, "_ep_bg_ready", False):
            return
        df: dict[tuple[int, int], int] = {}
        n = 0
        for ep in getattr(self, "episodes", []):
            if not ep.match_ids:
                continue
            n += 1
            gs = [vocab.fold_group_id(t) for t in ep.match_ids]
            for bg in set(zip(gs, gs[1:])):
                df[bg] = df.get(bg, 0) + 1
        # TK_010: cha t1 thống nhất (instruction essay vi-alpaca) — cùng màng df-bigram
        for rec in self.parents:
            if not rec.fact_links or not rec.match_ids:
                continue
            n += 1
            gs = [vocab.fold_group_id(t) for t in rec.match_ids]
            for bg in set(zip(gs, gs[1:])):
                df[bg] = df.get(bg, 0) + 1
        self._ep_bg_df = df
        self._ep_bg_n = n
        self._ep_bg_ready = True

    def _bigram_idf(self, bg: tuple[int, int], vocab: Vocabulary) -> float:
        """IDF của CHÍNH bigram (không phải tổng unigram): khung chung nhiều instruction
        ("hướng dẫn nấu") → nhẹ; bigram riêng ("phở bò") → nặng. TK_007 sửa regression bí đỏ."""
        self._ensure_ep_bigram_df(vocab)
        n = max(1, getattr(self, "_ep_bg_n", 1))
        return math.log((n + 1) / (1 + self._ep_bg_df.get(bg, 0))) + 1.0

    def _register_episode(self, neuron: AbstractNeuron) -> int:
        self._ep_bg_ready = False  # TK_007: invalidate df-bigram khi thêm episode
        ei = len(self.episodes)
        self.episodes.append(neuron)
        pe = getattr(self, "_parent_episodes", None)
        if pe is None:
            pe = self._parent_episodes = {}
        for pi, _w in neuron.members:
            pe.setdefault(pi, []).append(ei)
        return ei

    def build_episodes_from_provenance(
        self,
        groups: "list[tuple[str, str, list[tuple[int, ...]]]]",
        *,
        min_members: int = 2,
        cache=None,
    ) -> tuple[int, int]:
        """
        Nối t1 theo XUẤT XỨ: mỗi group = (key, title, [chuỗi token từng câu THEO THỨ TỰ]).

        Câu tra qua `_key_to_idx` (emit đúng chuỗi) — cha nào có mặt thành member w=1.0.
        Nhóm < `min_members` member bỏ (episode 1 câu không mang thông tin diễn ngôn).
        Trả (số episode tạo, số câu match được cha). TK_012: key/title ghi vào `cache` (nhãn
        debug) nếu truyền — nơ-ron KHÔNG giữ nhãn.
        """
        made = matched = 0
        for key, title, sent_seqs in groups:
            members: list[tuple[int, float]] = []
            seen: set[int] = set()
            for seq in sent_seqs:
                pi = self._key_to_idx.get(tuple(seq))
                if pi is None or pi in seen:
                    continue
                seen.add(pi)
                members.append((pi, 1.0))
                matched += 1
            if len(members) >= min_members:
                ei = self._register_episode(
                    AbstractNeuron(members=tuple(members))
                )
                if cache is not None:
                    cache.set_episode(ei, key=key, title=title, source="doc")
                made += 1
        return made, matched

    def bind_episode_by_resonance(
        self,
        passage_ids: list[int],
        vocab: Vocabulary,
        *,
        key: str,
        title: str = "",
        top_n: int = 24,
        min_score: float = 0.05,
        min_members: int = 2,
        cache=None,
    ) -> int | None:
        """
        Nối t1 theo CỘNG HƯỞNG (cơ chế gốc tác giả): kích hoạt toàn bộ token của đoạn văn MỚI
        → cha nào "rung" (rank_matches trên bag đoạn, vượt ngưỡng) → thành member với w=điểm.

        Dùng cho đoạn KHÔNG có xuất xứ trong train data (gold SFT mới, đoạn distill) — cha có
        sẵn từ nguồn khác (np1 "Mèo rất nhanh nhẹn…") vẫn gia nhập t1 của bài văn mèo.
        Trả episode idx hoặc None (không đủ member cộng hưởng). TK_012: nhãn → `cache`.
        """
        ranked = self.rank_matches(
            passage_ids, vocab, None, top_n=top_n, fuzzy_topk=0
        )
        members = [(pi, float(s)) for pi, s in ranked if s >= min_score]
        if len(members) < min_members:
            return None
        ei = self._register_episode(
            AbstractNeuron(members=tuple(members))
        )
        if cache is not None:
            cache.set_episode(ei, key=key, title=title, source="resonance")
        return ei

    def bind_sense_resonance(
        self,
        adjacency,
        vocab: Vocabulary,
        embedding=None,
        *,
        top_n: int = 10,
        min_score: float = 0.06,
        min_priority: float = 0.5,
        max_tokens_per_sense: int = 20,
        logger=None,
    ) -> dict[str, int]:
        """
        TK_016 D1′ — hút occurrence corpus vào sense-neuron (không flat-dump parent từ điển).

        Với mỗi `Sense:word#k`: rank_matches trên bag seed (lemma + co-act sẵn có) → token nội
        dung từ cha corpus khớp → cạnh CO_ACTIVATION có trọng số (cộng hưởng thật).
        """
        from nn_aivie.core.edge_kinds import EdgeKind

        co_layer = adjacency.layer(EdgeKind.CO_ACTIVATION)
        stats = {"senses": 0, "links": 0, "parents_hit": 0}
        sense_ids = [
            sid for tok, sid in vocab._token_to_id.items() if tok.startswith("Sense:")
        ]
        rank_cache: dict[tuple[int, ...], list[tuple[int, float]]] = {}
        total = len(sense_ids)
        for i, sid in enumerate(sense_ids):
            tok = vocab.token_of(sid)
            head = tok[6:].rsplit("#", 1)[0].split()[0]
            lemma_id = vocab._token_to_id.get(head)
            seed: list[int] = []
            if lemma_id is not None:
                seed.append(lemma_id)
            row_co = co_layer.row(sid)
            pairs = [
                (nid, w) for nid, w in row_co.items()
                if w > 0 and vocab.is_surface(nid)
            ]
            pairs.sort(key=lambda x: -x[1])
            seed.extend(n for n, _ in pairs[:12])
            seed = list(dict.fromkeys(seed))
            if not seed:
                continue
            key = tuple(seed)
            ranked = rank_cache.get(key)
            if ranked is None:
                ranked = self.rank_matches(
                    list(seed), vocab, embedding,
                    top_n=top_n, fuzzy_topk=0, min_priority=min_priority,
                )
                rank_cache[key] = ranked
            added = 0
            for pi, score in ranked:
                if score < min_score:
                    break
                stats["parents_hit"] += 1
                rec = self.parents[pi]
                emit = rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids
                for tid in _content_ids(list(emit), vocab, skip=set()):
                    if tid == sid or added >= max_tokens_per_sense:
                        continue
                    adjacency.add(sid, tid, float(score), EdgeKind.CO_ACTIVATION)
                    adjacency.add(tid, sid, float(score), EdgeKind.CO_ACTIVATION)
                    added += 1
                    stats["links"] += 1
            if added:
                stats["senses"] += 1
            if logger is not None and (i + 1) % 1000 == 0:
                logger.progress("bind_sense", i + 1, total, extra=f"links={stats['links']}")
        return stats

    def bind_fact_links(
        self,
        pi: int,
        vocab: Vocabulary,
        *,
        top_n: int = 12,
        min_score: float = 0.05,
        max_links: int = 8,
        exclude: frozenset[int] | None = None,
    ) -> int:
        """
        TK_010 bước (3): từ LIST TOKEN của cha t1 → tìm các FACT đã có (cộng hưởng, tái dùng
        rank_matches) → sắp theo VỊ TRÍ XUẤT HIỆN trong chuỗi t1 → ghi `fact_links` có thứ tự.
        Quan hệ ngang fact↔fact suy từ thứ tự này (fact trước → fact sau).

        pos của một fact = chỉ số sớm nhất trong chuỗi t1 mà token PHÂN BIỆT nhất (gidf cao,
        case-fold) của fact xuất hiện — thuần vị trí, Luật 0.
        """
        rec = self.parents[pi]
        seq = list(rec.emit_ids)
        excl = exclude or frozenset()
        ranked = self.rank_matches(seq, vocab, None, top_n=top_n + 4, fuzzy_topk=0)
        # vị trí xuất hiện đầu của mỗi bề mặt fold trong chuỗi t1
        pos_of: dict[str, int] = {}
        for i, t in enumerate(seq):
            pos_of.setdefault(self._fold(t, vocab), i)
        links: list[tuple[int, int]] = []
        for fi, score in ranked:
            if fi == pi or fi in excl or score < min_score:
                continue
            frec = self.parents[fi]
            fseq = frec.answer_ids if (frec.is_qa and frec.answer_ids) else frec.emit_ids
            cands = sorted(
                set(fseq), key=lambda t: -self._gidf(t)
            )[:3]  # 3 token phân biệt nhất của fact
            ps_hits = [pos_of[k] for t in cands if (k := self._fold(t, vocab)) in pos_of]
            if not ps_hits:
                continue
            links.append((fi, min(ps_hits)))
            if len(links) >= max_links:
                break
        links.sort(key=lambda x: x[1])
        self.parents[pi] = replace(rec, fact_links=tuple(links))
        return len(links)

    def t1_agenda(self, pi: int, vocab: Vocabulary) -> list[list[int]]:
        """Agenda của cha t1 = CHÍNH chuỗi đáp của nó, chia segment theo dấu chấm (TK_010:
        không còn member list — bài là một nơ-ron, tự phân đoạn)."""
        rec = self.parents[pi]
        seq = rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids
        segs: list[list[int]] = []
        cur: list[int] = []
        for t in seq:
            tok = vocab.token_of(t)
            if tok == ".":
                if cur:
                    segs.append(cur)
                    cur = []
            elif is_surface_token(tok):
                cur.append(t)
        if cur:
            segs.append(cur)
        return segs

    def episodes_of(self, parent_idx: int) -> tuple[int, ...]:
        """Các episode t1 chứa cha này (rỗng nếu chưa build tầng t1 / cha ngoài episode)."""
        pe = getattr(self, "_parent_episodes", None) or {}
        return tuple(pe.get(parent_idx, ()))

    def add_sft_episode(
        self,
        *,
        key: str,
        title: str,
        match_ids: tuple[int, ...],
        member_idxs: list[int],
        resonance: list[tuple[int, float]] | None = None,
        cache=None,
    ) -> int:
        """t1 SFT (T2): member xuất xứ THEO THỨ TỰ (w=1.0) + member cộng hưởng tri thức cũ.
        TK_012: key/title là nhãn debug → `cache` (nếu truyền); nơ-ron chỉ giữ id/match_ids."""
        members: list[tuple[int, float]] = [(pi, 1.0) for pi in member_idxs]
        seen = set(member_idxs)
        for pi, w in resonance or []:
            if pi not in seen:
                members.append((pi, float(w)))
                seen.add(pi)
        ei = self._register_episode(
            AbstractNeuron(members=tuple(members), match_ids=tuple(match_ids))
        )
        if cache is not None:
            cache.set_episode(ei, key=key, title=title, source="sft")
        return ei

    def match_episode(
        self,
        query_ids: list[int],
        vocab: Vocabulary,
        *,
        min_score: float = 0.30,
    ) -> tuple[int, float] | None:
        """
        Khớp query → episode t1 SFT qua `match_ids` (instruction) — "đi lên tầng trừu tượng".

        Chấm theo nguyên tắc rank_matches: IDF-coverage (gidf — instruction ngoài miền curated)
        × distinct-guard (token phân biệt nhất ĐƯỢC PHỦ phải ≥75% token phân biệt nhất của
        query — bài học TK_002: không đòi ĐÚNG primary để paraphrase còn đường sống).

        TK_005: đơn vị chấm = token + **BIGRAM LIỀN KỀ** (cấu trúc thuần, cùng tinh thần
        phrase-bonus 0.5.4). Token instruction phổ biến ("lợi/ích") rời rạc không lấn được
        bài khớp CỤM dài ("đọc sách", "biểu hiện" + "căng thẳng") — sửa 2 ca khớp nhầm TK_004.
        Quét phẳng O(#episode-sft) — đủ cho vài nghìn episode, index hoá khi cần.
        """
        q_content = _content_ids(query_ids, vocab, skip=set())
        if not q_content:
            return None
        best: int | None = None
        best_s = 0.0
        for ei, ep in enumerate(getattr(self, "episodes", [])):
            if not ep.match_ids:
                continue
            score = self._instruction_match_score(query_ids, ep.match_ids, vocab)
            if score > best_s:
                best_s, best = score, ei
        return (best, best_s) if best is not None and best_s >= min_score else None

    def _instruction_match_score(
        self,
        query_ids: list[int],
        match_ids: tuple[int, ...],
        vocab: Vocabulary,
    ) -> float:
        """Chấm query vs instruction (match_ids): gidf + bigram-df + distinct-guard (TK_005–007)."""
        q_content = _content_ids(query_ids, vocab, skip=set())
        if not q_content or not match_ids:
            return 0.0
        # TK_012 Phase 3: khớp trên FOLD-GROUP-ID (int), bigram = cặp gid — không string.
        q_keys = [vocab.fold_group_id(t) for t in q_content]
        key_w = {k: self._gidf(t) for t, k in zip(q_content, q_keys)}
        q_key_set = set(q_keys)
        q_bigrams = list(dict.fromkeys(zip(q_keys, q_keys[1:])))
        bg_w = {bg: self._bigram_idf(bg, vocab) for bg in q_bigrams}
        tot = (sum(key_w.values()) + sum(bg_w.values())) or 1.0
        max_q = max(list(key_w.values()) + list(bg_w.values()))
        ep_keys = [vocab.fold_group_id(t) for t in match_ids]
        inter = q_key_set & set(ep_keys)
        if not inter:
            return 0.0
        ep_bigrams = set(zip(ep_keys, ep_keys[1:]))
        cov = (
            sum(key_w[k] for k in inter)
            + sum(w for bg, w in bg_w.items() if bg in ep_bigrams)
        ) / tot
        cov_units = [key_w[k] for k in inter]
        for bg, w in bg_w.items():
            if bg in ep_bigrams:
                cov_units.append(w)
            elif getattr(self, "_phrase_assoc", None):
                if any(b in ep_bigrams for b, _ in self.related_phrases(bg, topk=6)):
                    cov_units.append(0.5 * w)
        ratio = (max(cov_units) / max_q) if max_q > 0 else 1.0
        return cov if ratio >= 0.75 else cov * ratio * ratio

    def match_t1_parent(
        self,
        query_ids: list[int],
        vocab: Vocabulary,
        *,
        min_score: float = 0.15,
    ) -> tuple[int, float] | None:
        """
        TK_010 — khớp query → cha t1 THỐNG NHẤT (ParentRecord có fact_links) qua match_ids.

        Dùng CÙNG máy chấm `match_episode` (bigram-df, case-fold, distinct-guard) — không
        dùng `best_match` BM25-style (regression: token chung "lợi/ích" lấn bigram "đọc sách").
        """
        best: int | None = None
        best_s = 0.0
        for pi, rec in enumerate(self.parents):
            if not rec.fact_links or not rec.match_ids:
                continue
            score = self._instruction_match_score(query_ids, rec.match_ids, vocab)
            if score > best_s:
                best_s, best = score, pi
        return (best, best_s) if best is not None and best_s >= min_score else None

    # ---- TK_008 (2): cạnh liên tưởng CỤM↔CỤM học từ đồng-episode (PMI) — T3 enrichment ----
    def build_phrase_assoc(
        self,
        vocab: Vocabulary,
        *,
        top_bg_per_ep: int = 30,
        min_cooc: int = 2,
        max_df: int = 300,
        min_df: int = 3,
    ) -> dict[str, int]:
        """
        Học "mắt sáng"↔"tinh anh" (ý gốc t1): hai BIGRAM đồng-thành-viên nhiều episode → cạnh
        PMI. Đơn vị = bigram bề mặt case-fold (đa nghĩa token đơn — bài học T0). Mỗi episode
        lấy `top_bg_per_ep` bigram HIẾM nhất (df thấp — bigram khung "của việc" tự rớt) để
        chặn bùng nổ cặp. Offline train-time; đọc qua `related_phrases`.
        """
        # TK_012 Phase 3: đơn vị bigram = cặp FOLD-GROUP-ID (int), không string. Hai token cùng
        # nhóm case-fold ('Phở'≡'phở') → cùng gid → khớp bằng int, đồ thị/serialize thuần id.
        # df bigram trên màng episode (instruction + text member xuất xứ)
        ep_bags: list[set[tuple[int, int]]] = []
        df: dict[tuple[int, int], int] = {}
        for ep in getattr(self, "episodes", []):
            ids: list[int] = list(ep.match_ids)
            for pi, w in ep.members:
                if w >= 1.0:
                    ids.extend(self.parents[pi].emit_ids)
            gids = [vocab.fold_group_id(t) for t in ids]
            folds = [vocab.fold(t) for t in ids]  # để loại bigram có mảnh rỗng (dấu câu)
            bag = {
                (gids[i], gids[i + 1])
                for i in range(len(gids) - 1)
                if folds[i] and folds[i + 1]
            }
            ep_bags.append(bag)
            for bg in bag:
                df[bg] = df.get(bg, 0) + 1
        n = len(ep_bags) or 1
        cooc: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
        def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
            """Cặp NỐI CHUỖI = mảnh của cùng cụm dài ("chú trọng"↔"trọng đến") → nhiễu.
            Cặp chia token CÙNG VỊ TRÍ ("mì kẹp"↔"mì sandwich") = biến thể paradigm → GIỮ."""
            return a[1] == b[0] or b[1] == a[0]

        for bag in ep_bags:
            # band df [min_df, max_df]: bỏ cặp siêu-hiếm df=2 (đồng-hiện tầm thường cùng đoạn
            # — đo vòng 1: 8k cặp toàn mảnh tên riêng) lẫn khung phổ biến; ưu tiên HIẾM trong band
            rare = sorted(
                (bg for bg in bag if min_df <= df[bg] <= max_df), key=lambda b: df[b]
            )[:top_bg_per_ep]
            for i, a in enumerate(rare):
                for b in rare[i + 1:]:
                    if _overlap(a, b):  # nhiễu chính đo vòng 2
                        continue
                    cooc.setdefault(a, {})[b] = cooc.setdefault(a, {}).get(b, 0) + 1
        assoc: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}
        pairs = 0
        for a, row in cooc.items():
            for b, c in row.items():
                if c < min_cooc:
                    continue
                pmi = math.log((c * n) / (df[a] * df[b]))
                if pmi <= 0:
                    continue
                assoc.setdefault(a, []).append((b, pmi))
                assoc.setdefault(b, []).append((a, pmi))
                pairs += 1
        for k in assoc:
            assoc[k].sort(key=lambda x: -x[1])
        self._phrase_assoc = assoc
        return {"episodes": n, "bigrams": len(df), "pairs": pairs}

    def related_phrases(
        self, bg, *, topk: int = 8, vocab: Vocabulary | None = None
    ) -> list:
        """Cụm liên tưởng của một bigram (đã build_phrase_assoc; rỗng nếu chưa).

        TK_012 Phase 3: `_phrase_assoc` khóa cặp FOLD-GROUP-ID (int). `bg` có thể là:
        - cặp int (gid) → tra thẳng (nội bộ, hot path);
        - cặp str (bề mặt) + `vocab` → dịch sang gid trước khi tra (debug/test)."""
        assoc = getattr(self, "_phrase_assoc", {})
        if bg and isinstance(bg[0], str):
            if vocab is None:
                # dịch qua bảng fold-group-id đã build (nếu token có trong vocab)
                fg = getattr(self, "_fold_gid_lookup", None)
                if fg is None:
                    return []
                bg = (fg.get(bg[0]), fg.get(bg[1]))
            else:
                bg = (vocab._fold_to_gid.get(bg[0]), vocab._fold_to_gid.get(bg[1]))
            if bg[0] is None or bg[1] is None:
                return []
        return list(assoc.get(tuple(bg), ())[:topk])

    # ---- nơ-ron mẫu chung: induce hub + quét chéo bind filler (docx Cross-sample) ----
    def induce_slot_hubs(
        self,
        *,
        min_distinct_fillers: int = 2,
        min_len: int = 2,
        max_len: int = 8,
        max_hubs: int = 4000,
    ) -> dict[str, int]:
        """
        Tự nổi hub từ **họ mẫu**: ≥ `min_distinct_fillers` cha cùng skeleton, lệch đúng 1 vị trí.

        Với mỗi cha, thay từng vị trí bằng wildcard → key `(left, right)`; gom cha theo key.
        Key nào có ≥N filler PHÂN BIỆT (token khác nhau) → một hub. Gate Luật 0: filler phải
        **phân biệt hơn khung** (min IDF filler ≥ IDF trung bình token khung) — slot là thực
        thể (Mỹ/Pháp/Anh), khung là phần chung (Thủ đô của / là gì). Không dùng danh sách từ.
        """
        tmpl: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[int, list[int]]] = {}
        for pi, rec in enumerate(self.parents):
            seq = rec.match_ids
            if not (min_len <= len(seq) <= max_len):
                continue
            for i in range(len(seq)):
                key = (seq[:i], seq[i + 1 :])
                tmpl.setdefault(key, {}).setdefault(seq[i], []).append(pi)

        cand: list[tuple[float, SlotHub]] = []
        for (left, right), fillers in tmpl.items():
            if len(fillers) < min_distinct_fillers:
                continue
            skeleton = left + right
            if not skeleton:
                continue
            skel_idf = sum(self._idf(t) for t in skeleton) / len(skeleton)
            min_fill_idf = min(self._idf(f) for f in fillers)
            if min_fill_idf < skel_idf:  # slot không phân biệt hơn khung → bỏ (không phải hub)
                continue
            hub = SlotHub(
                left=left,
                right=right,
                fillers={f: len(ps) for f, ps in fillers.items()},
                source_parents=tuple(sorted({p for ps in fillers.values() for p in ps})),
            )
            # ưu tiên hub nhiều filler + khung dài (đặc thù) khi phải cắt max_hubs
            cand.append((len(fillers) + 0.1 * len(skeleton), hub))

        cand.sort(key=lambda x: -x[0])
        self.hubs = [h for _, h in cand[:max_hubs]]
        return {"hubs": len(self.hubs)}

    def bind_slot_fillers(
        self,
        *,
        extra_sequences: list[tuple[int, ...]] | None = None,
        min_idf_ratio: float = 0.5,
        right_window: int = 4,
    ) -> dict[str, int]:
        """
        Pha quét chéo: nối filler MỚI vào hub **ngay trong train** (điều bạn hỏi — Ý → hub).

        Quét mọi câu train (match_ids cha + `extra_sequences`): gặp `left X …right[0]…` với X là
        token phân biệt (IDF ≥ ratio·IDF filler yếu nhất của hub) → thêm cạnh `hub → X`. Nhờ đó
        một câu phát biểu "Thủ đô của Ý là Rome" đủ để Ý gia nhập `n_quốc_gia_thủ_đô` khi train.
        """
        if not self.hubs:
            return {"bound": 0, "new_fillers": 0}
        seqs = [rec.match_ids for rec in self.parents]
        if extra_sequences:
            seqs += [tuple(s) for s in extra_sequences]

        left_to_hubs: dict[tuple[int, ...], list[SlotHub]] = defaultdict(list)
        for h in self.hubs:
            left_to_hubs[h.left].append(h)
        left_lens = sorted({len(h.left) for h in self.hubs if h.left})

        floors = {
            id(h): (min(self._idf(f) for f in h.fillers) if h.fillers else 1.0)
            for h in self.hubs
        }
        bound = 0
        new_fillers = 0
        for seq in seqs:
            n = len(seq)
            for llen in left_lens:
                if llen == 0 or llen >= n:
                    continue
                for s in range(n - llen):
                    window = seq[s : s + llen]
                    hubs_here = left_to_hubs.get(window)
                    if not hubs_here:
                        continue
                    xi = s + llen
                    if xi >= n:
                        continue
                    x = seq[xi]
                    for hub in hubs_here:
                        if x in hub.fillers:
                            continue
                        if self._idf(x) < floors[id(hub)] * min_idf_ratio:
                            continue
                        if hub.right:
                            tail = seq[xi + 1 : xi + 1 + right_window]
                            if hub.right[0] not in tail:
                                continue
                        if x not in hub.fillers:
                            new_fillers += 1
                        hub.fillers[x] = hub.fillers.get(x, 0) + 1
                        bound += 1
        return {"bound": bound, "new_fillers": new_fillers}

    def hub_for_query(
        self, query_content: tuple[int, ...]
    ) -> tuple[SlotHub, int] | None:
        """Khớp câu hỏi với một hub → trả (hub, filler_token) nếu `left X …right…` khớp.

        Dùng lúc sinh: biết câu thuộc họ mẫu nào + thực thể X ở slot (kể cả X là filler lạ),
        để chú ý vào X và route sang case con kiến. Không sửa hành vi mặc định hiện tại.
        """
        n = len(query_content)
        for hub in self.hubs:
            llen = len(hub.left)
            if llen == 0 or llen >= n:
                continue
            for s in range(n - llen):
                if query_content[s : s + llen] != hub.left:
                    continue
                xi = s + llen
                if xi >= n:
                    continue
                if hub.right:
                    tail = query_content[xi + 1 : xi + 1 + 4]
                    if hub.right[0] not in tail:
                        continue
                return hub, query_content[xi]
        return None

    # ---- Trụ 2: successor bậc cao suy TỪ CHUỖI CHA (docx §1 "index cứng") ----
    def _ensure_seq_index(self) -> None:
        """
        Xây (lazy) thống kê chuyển dịch **bậc-2** từ chuỗi con có thứ tự của cha (đáp QA / cả câu
        statement). Global W_seq chỉ là bigram (bậc-1) → word-salad; cha lưu trật tự đầy đủ nên
        successor bậc-2 cho fluency mạch lạc hơn hẳn mà KHÔNG cần train lại (thuần readout).
        """
        if getattr(self, "_seq_ready", False):
            return
        bi: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        tri: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for rec in self.parents:
            seq = rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids
            w = float(max(1, rec.power))
            for i in range(len(seq) - 1):
                bi[seq[i]][seq[i + 1]] += w
            for i in range(len(seq) - 2):
                tri[(seq[i], seq[i + 1])][seq[i + 2]] += w
        self._seq_bi = {k: dict(v) for k, v in bi.items()}
        self._seq_tri = {k: dict(v) for k, v in tri.items()}
        self._seq_ready = True

    def seq_next(
        self, prev: int | None, cur: int, *, min_hi: float = 2.0
    ) -> dict[int, float]:
        """Phân bố token kế tiếp có điều kiện (prev, cur); backoff về bigram-cha khi bậc-2 thưa."""
        self._ensure_seq_index()
        if prev is not None:
            d = self._seq_tri.get((prev, cur))
            if d and sum(d.values()) >= min_hi:
                return d
        return self._seq_bi.get(cur, {})

    # ---- Trụ 6 (ý 1a): học đại từ theo thực thể từ đồng-hiện trong-cha (PMI) ----
    def completion(self, parent_idx: int) -> tuple[int, ...]:
        """Phần con CÒN THIẾU dội ra: đáp (QA) hoặc cả câu phát biểu (case con kiến)."""
        rec = self.parents[parent_idx]
        return rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids

    def completion_slots(self, parent_idx: int) -> tuple[SlotRef, ...]:
        """TK_013 — chuỗi slot để pattern completion theo index."""
        return self.parents[parent_idx].slot_chain

    def completion_tail_slots(self, parent_idx: int) -> tuple[SlotRef, ...]:
        """Slot sau vùng match — phần còn thiếu theo index (QA: đáp; statement: toàn chuỗi)."""
        rec = self.parents[parent_idx]
        if not rec.slot_chain:
            return ()
        ms, me = rec.match_span
        if rec.is_qa and rec.answer_ids:
            return rec.slot_chain[me:]
        return rec.slot_chain

    def rebuild_seg_graph(
        self,
        vocab: Vocabulary,
        *,
        dap_id: int | None = None,
        progress: ProgressFn | None = None,
        log_every: int = 500,
        max_promotions: int = 0,
    ) -> dict[str, int]:
        """Replay ISG từ source_ids (giữ nguyên danh sách cha, episodes, hubs)."""
        def _log(msg: str) -> None:
            if progress:
                progress(msg)

        total = len(self.parents)
        self.seg = SegmentGraph(min_frequency=self.seg_min_frequency)
        self.seg.begin_batch()
        self.phrases = []
        self.phrase_df = []
        self._phrase_lookup = {}
        self._phrase_max_len = 4
        n = 0
        _log(f"ISG batch ingest start parents={total} min_freq={self.seg_min_frequency}")
        for pi, rec in enumerate(self.parents):
            ids = list(rec.source_ids)
            if not ids and rec.slot_chain:
                for s in rec.slot_chain:
                    ids.extend(self.seg.expand_slot(s))
            if not ids:
                if rec.is_qa and rec.answer_ids and dap_id is not None:
                    ids = list(rec.match_ids) + [dap_id] + list(rec.answer_ids)
                else:
                    ids = list(rec.emit_ids)
            if not ids:
                continue
            dap_idx: int | None = None
            if dap_id is not None and dap_id in ids:
                dap_idx = ids.index(dap_id)
            chain, in_match = build_chain_from_ids(
                ids, dap_idx=dap_idx, vocab=vocab, is_match_token=self._is_match_token
            )
            rec.slot_chain = chain
            rec._in_match_flags = in_match
            rec.match_span = _match_span_from_flags(in_match)
            if not rec.source_ids:
                rec.source_ids = tuple(ids)
            self.seg.ingest_parent(pi, chain, in_match, self.parents)
            n += 1
            if progress and (pi + 1) % log_every == 0:
                progress(
                    f"ingest {pi + 1}/{total} clusters={len(self.seg.clusters)} "
                    f"bigrams={len(self.seg.bigram_count)}"
                )
        _log(f"ingest done n={n} bigrams={len(self.seg.bigram_count)} — finalize promotions")
        n_promoted = self.seg.end_batch(
            self.parents,
            progress=progress,
            log_every=log_every,
            max_promotions=max_promotions,
        )
        _log(f"promotions done clusters={len(self.seg.clusters)} promoted={n_promoted}")
        for pi in range(len(self.parents)):
            self._sync_derived_from_slots(self.parents[pi], vocab)
            if progress and (pi + 1) % (log_every * 4) == 0:
                progress(f"sync derived {pi + 1}/{total}")
        self.rebuild_hierarchical_index()
        _log(
            f"done parents={n} clusters={len(self.seg.clusters)} "
            f"links={len(self.seg.links)} promoted={n_promoted}"
        )
        return {
            "parents": n,
            "clusters": len(self.seg.clusters),
            "links": len(self.seg.links),
            "promoted": n_promoted,
        }

    # ---- serialization ----
    def to_payload(self) -> dict:
        seq_bi: list[list[int | float]] = []
        seq_tri: list[list[int | float]] = []
        if self.parents:
            self._ensure_seq_index()
            for cur, row in getattr(self, "_seq_bi", {}).items():
                for nxt, w in row.items():
                    seq_bi.append([int(cur), int(nxt), float(w)])
            for (a, b), row in getattr(self, "_seq_tri", {}).items():
                for c, w in row.items():
                    seq_tri.append([int(a), int(b), int(c), float(w)])
        return {
            "parents": [
                {
                    "emit": list(p.emit_ids),
                    "match": list(p.match_ids),
                    "answer": list(p.answer_ids),
                    "power": p.power,
                    "is_qa": p.is_qa,
                    "phrase_ids": list(p.phrase_ids),
                    "skip_span_ids": list(getattr(p, "skip_span_ids", ())),
                    "priority": p.priority,
                    "fact_links": [list(fl) for fl in p.fact_links],
                    "source_ids": list(p.source_ids),
                    "slot_chain": [s.to_json() for s in p.slot_chain],
                    "match_span": list(p.match_span),
                    "in_match": list(p._in_match_flags),
                }
                for p in self.parents
            ],
            "seg": self.seg.to_payload(),
            "seg_min_frequency": self.seg_min_frequency,
            # TK_020: meta cạnh ngang XA — power + gap_hist + provenance (cha nguồn)
            "hlink_meta": {
                f"{a},{b}": {
                    "power": m["power"],
                    "gaps": {str(g): c for g, c in m["gaps"].items()},
                    "prov": [list(t) for t in m["prov"]],
                    **({"left_kind": m["left_kind"]} if m.get("left_kind") else {}),
                    **({"right_kind": m["right_kind"]} if m.get("right_kind") else {}),
                }
                for (a, b), m in getattr(self, "hlink_meta", {}).items()
            },
            "hlink_endpoints": {
                str(eid): [ep.kind, list(ep.tokens)]
                for eid, ep in getattr(self, "hlink_endpoints", {}).items()
            },
            "skip_spans": [
                {
                    "left": s["left"],
                    "right": list(skip_right_tuple(s["right"])),
                    "power": s["power"],
                    "pmi": s["pmi"],
                    "gaps": {str(g): c for g, c in s["gaps"].items()},
                    "prov": [list(t) for t in s["prov"]],
                    "promoted": bool(s.get("promoted")),
                }
                for s in getattr(self, "skip_spans", ())
            ],
            "phrases": [list(ph) for ph in self.phrases],
            "phrase_df": list(self.phrase_df),
            "phrase_max_len": self._phrase_max_len,
            "hubs": [
                {
                    "left": list(h.left),
                    "right": list(h.right),
                    "fillers": [[f, p] for f, p in h.fillers.items()],
                    "source_parents": list(h.source_parents),
                }
                for h in self.hubs
            ],
            "phrase_assoc": [
                [list(a), [[list(b), w] for b, w in row]]
                for a, row in getattr(self, "_phrase_assoc", {}).items()
            ],
            "episodes": [
                {
                    "members": [[pi, w] for pi, w in e.members],
                    "match": list(e.match_ids),
                }
                for e in getattr(self, "episodes", [])
            ],
            "seq_bi": seq_bi,
            "seq_tri": seq_tri,
        }

    @classmethod
    def from_payload(cls, payload: dict, *, logger=None) -> "ParentStore":
        store = cls()
        # TK_012: checkpoint cũ có thể còn nhãn nhúng (tag/key/title/source) — bóc ra
        # `_migrated_content` để checkpoint_io gom thành ContentCache (tương thích ngược).
        migrated_parents: dict[int, str] = {}
        migrated_eps: dict[int, dict] = {}
        parents_list = payload.get("parents", [])
        n_total = len(parents_list)
        if logger is not None and n_total:
            logger.log(f"parents: deserialize {n_total} records …")
        for i, p in enumerate(parents_list):
            rec = ParentRecord(
                emit_ids=tuple(p["emit"]),
                match_ids=tuple(p["match"]),
                answer_ids=tuple(p["answer"]),
                power=int(p.get("power", 1)),
                is_qa=bool(p.get("is_qa", False)),
                phrase_ids=tuple(p.get("phrase_ids", ())),
                skip_span_ids=tuple(p.get("skip_span_ids", ())),
                priority=float(p.get("priority", 1.0)),
                fact_links=tuple(
                    (int(a), int(b)) for a, b in p.get("fact_links", ())
                ),
                source_ids=tuple(p.get("source_ids", ())),
                slot_chain=tuple(
                    SlotRef.from_json(s) for s in p.get("slot_chain", ())
                ),
                match_span=tuple(p.get("match_span", (0, 0))),
                _in_match_flags=tuple(p.get("in_match", ())),
            )
            if p.get("tag"):
                migrated_parents[len(store.parents)] = str(p["tag"])
            idx = len(store.parents)
            store.parents.append(rec)
            store._key_to_idx[tuple(rec.emit_ids)] = idx
            for t in set(rec.match_ids):
                store._index.setdefault(t, []).append(idx)
                store._gdf[t] = store._gdf.get(t, 0) + 1  # df toàn cục (W3)
                if rec.priority >= 0.9:
                    store._df[t] = store._df.get(t, 0) + 1
            if logger is not None and n_total >= 5000 and (i + 1) % 5000 == 0:
                logger.progress("parents", i + 1, n_total)
        store.phrases = [tuple(ph) for ph in payload.get("phrases", [])]
        store.phrase_df = list(payload.get("phrase_df", []))
        store._phrase_lookup = {ph: i for i, ph in enumerate(store.phrases)}
        store._phrase_max_len = int(payload.get("phrase_max_len", 4))
        store.hubs = [
            SlotHub(
                left=tuple(h["left"]),
                right=tuple(h["right"]),
                fillers={int(f): int(p) for f, p in h.get("fillers", [])},
                source_parents=tuple(h.get("source_parents", ())),
            )
            for h in payload.get("hubs", [])
        ]
        for e in payload.get("episodes", []):
            ei = store._register_episode(
                AbstractNeuron(
                    members=tuple((int(pi), float(w)) for pi, w in e.get("members", [])),
                    match_ids=tuple(int(t) for t in e.get("match", [])),
                )
            )
            if e.get("key") or e.get("title"):
                migrated_eps[ei] = {
                    "key": str(e.get("key", "")),
                    "title": str(e.get("title", "")),
                    "source": str(e.get("source", "doc")),
                }
        pa = {}
        legacy_str = False
        for a, row in payload.get("phrase_assoc", []):
            key = tuple(a)
            if key and isinstance(key[0], str):
                legacy_str = True  # TK_012: checkpoint cũ khóa string → convert lazy khi có vocab
            pa[key] = [(tuple(b), float(w)) for b, w in row]
        if pa:
            if legacy_str:
                store._phrase_assoc_legacy_str = pa  # chờ vocab để đổi sang fold-group-id
            else:
                store._phrase_assoc = pa
        store.seg = SegmentGraph.from_payload(payload.get("seg"))
        store.seg_min_frequency = int(payload.get("seg_min_frequency", 3))
        store.hlink_meta = {}
        for key, m in (payload.get("hlink_meta") or {}).items():
            a, b = key.split(",")
            entry = {
                "power": int(m["power"]),
                "gaps": {int(g): int(c) for g, c in m["gaps"].items()},
                "prov": [tuple(t) for t in m["prov"]],
            }
            if m.get("left_kind"):
                entry["left_kind"] = str(m["left_kind"])
            if m.get("right_kind"):
                entry["right_kind"] = str(m["right_kind"])
            store.hlink_meta[(int(a), int(b))] = entry
        store.hlink_endpoints = {}
        from nn_aivie.core.frame_bracket_ref import FrameEndpoint

        for eid_s, raw in (payload.get("hlink_endpoints") or {}).items():
            kind = raw[0]
            toks = tuple(int(t) for t in raw[1])
            store.hlink_endpoints[int(eid_s)] = FrameEndpoint(kind, toks)
        store.skip_spans = []
        store._skip_lookup = {}
        store._skip_left_index = {}
        store._skip_index = {}
        store.skip_span_meta = {}
        for s in payload.get("skip_spans") or []:
            right = skip_right_tuple(s["right"])
            rec = {
                "left": int(s["left"]),
                "right": right,
                "power": int(s["power"]),
                "pmi": float(s.get("pmi", 0.0)),
                "gaps": {int(g): int(c) for g, c in s["gaps"].items()},
                "prov": [tuple(t) for t in s["prov"]],
                "promoted": bool(s.get("promoted")),
            }
            si = len(store.skip_spans)
            store.skip_spans.append(rec)
            key = (rec["left"], right)
            store._skip_lookup[key] = si
            store.skip_span_meta[key] = rec
        if store.skip_spans:
            index: dict[int, list[int]] = defaultdict(list)
            for pi, prec in enumerate(store.parents):
                for si in getattr(prec, "skip_span_ids", ()):
                    if si not in index[si]:
                        index[si].append(pi)
            store._skip_index = {k: sorted(v) for k, v in index.items()}
            store._skip_df = [len(store._skip_index.get(i, ())) for i in range(len(store.skip_spans))]
            store._rebuild_skip_left_index()
        if getattr(store.seg, "_rebuild_pair_index_from_parents", False):
            store.seg.rebuild_inverted_index(store.parents)
        if migrated_parents or migrated_eps:
            store._migrated_content = {
                "parents": migrated_parents,
                "episodes": migrated_eps,
            }
        seq_bi = payload.get("seq_bi")
        if seq_bi:
            bi: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
            for cur, nxt, w in seq_bi:
                bi[int(cur)][int(nxt)] += float(w)
            store._seq_bi = {k: dict(v) for k, v in bi.items()}
            tri: dict[tuple[int, int], dict[int, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            for a, b, c, w in payload.get("seq_tri", []):
                tri[(int(a), int(b))][int(c)] += float(w)
            store._seq_tri = {k: dict(v) for k, v in tri.items()}
            store._seq_ready = True
        # Chỉ rebuild MỘT lần: `_migrated_content` (content-cache) không đụng index cấu trúc
        # → bỏ lần rebuild thứ 2 dư thừa (~2s/load trên checkpoint 44k cha — TK_011).
        if logger is not None:
            logger.log("▶ rebuild_hierarchical_index …")
        t0 = time.time() if logger is not None else 0.0
        store.rebuild_hierarchical_index()
        if logger is not None:
            logger.log(f"✓ rebuild_hierarchical_index ({time.time() - t0:.1f}s)")
        return store

    def __len__(self) -> int:
        return len(self.parents)
