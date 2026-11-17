"""
TK_044 P3.1 — SKELETON `run_coref_pass`: học cạnh coreference + after_role từ CORPUS THÔ.

⚠️ SKELETON cho M4 — chưa chạy trên graph (viết ở máy dev không torch/graph). M4 kiểm/tinh các chỗ
`# TODO(M4)` theo API thật rồi chạy. Port thẳng POC (`scripts/poc_binding_continuation.py`) sang
token-id graph. Đọc RUNBOOK_TK_044_P3.1.md.

Cơ chế (mọi cạnh đếm-1-lần, KHÔNG backprop):
  1. Mỗi record (đoạn) → ids. Tìm OCCURRENCE thực-thể = span token HOA liền (`vocab.cap_class≥1` =
     danh-từ-riêng) gate `_gidf` cao. (v2: đổi từ `ps.phrases` — cụm Re-Pair bắt collocation bảng/game
     = lặp cấu-trúc, không phải đồng-chỉ; span-Hoa khớp tín hiệu scan "lặp-tên".)
  2. SIG = ngữ-cảnh trái/phải quanh occurrence (token-id; occurrence KHÁC trong cửa sổ → sentinel -1
     = "<ENT>"), đủ RỘNG bắt frame (bài học P2).
  3. NAME-REPEAT single: trong 1 đoạn, cùng filler → cặp (SIG_a, SIG_b) đếm Hebbian "cùng filler".
  4. after_role[SIG] = token NGAY SAU occurrence (role-backoff).
  5. coref[SIG] = argmax antecedent (ratio>0.55), tie-break TẤT ĐỊNH (ratio, tot, key) — KHÔNG lặp
     set/dict thứ-tự-hash (PYTHONHASHSEED chập chờn — đã cắn ở POC).

Kết quả ghi thẳng vào `state.coref` / `state.after_role` (M4 thêm field vào GraphState + serialize —
RUNBOOK §1/§4). Đại-từ (nó/họ/ông/bà) = P3.3 (mask outlier df-cao + mượn cạnh coref của tên).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from nn_aivie.core.state import GraphState
from nn_aivie.core.tokenizer import MARK_SPACE

SIG_W = 4           # bề rộng ngữ-cảnh mỗi phía (bài học P2: đủ rộng bắt verb-frame)
MAX_OCC = 12        # cap occurrence/đoạn (chặn O(n²) trên bài wiki dài)
ENT = -1            # sentinel "<ENT>" trong signature


def _find_occurrences(ids, vocab, gidf, gidf_min, gidf_solo) -> list[tuple[int, int, tuple[int, ...]]]:
    """(start, end, filler) mỗi occurrence THỰC-THỂ = span HOA bắt đầu bằng token ĐẦU-TỪ, gate gidf.

    v3 (2026-07-22): v2 (span cap_class≥1 thuần) ~73% nhiễu BPE-fragment/từ-Hoa-đầu-câu (out/
    coref_edges_smoke: '[ENT] g', '[ENT] ước'…). Vá: (a) span PHẢI bắt đầu bằng token ĐẦU-TỪ (marker
    `Ġ`, hoặc vị trí 0) — loại subword continuation; (b) nhận nếu ĐA-TOKEN (danh-từ-riêng ≥2 chữ Hoa,
    vd 'Hà Nội','Nguyễn Du') HOẶC single-token đủ đặc thù (`gidf ≥ gidf_solo`) — loại chữ-cái/từ-Hoa-
    đầu-câu tần cao ('Ước','G'). v2 cấu-trúc, không dùng `ps.phrases`."""
    def is_ws(pos):   # đầu-từ: marker Ġ hoặc đầu chuỗi
        return pos == 0 or vocab.token_of(ids[pos]).startswith(MARK_SPACE)
    occ, i, n = [], 0, len(ids)
    while i < n:
        if vocab.cap_class(ids[i]) >= 1 and is_ws(i):
            j = i
            while j < n and vocab.cap_class(ids[j]) >= 1:
                j += 1
            span = tuple(ids[i:j])
            g = max((gidf(t) for t in span), default=0.0)
            if g >= gidf_min and (len(span) >= 2 or g >= gidf_solo):   # đa-token HOẶC single đặc thù
                occ.append((i, j, span))
            i = j
        else:
            i += 1
        if len(occ) >= MAX_OCC:
            break
    return occ


def _sig(ids: list[int], occ: list[tuple[int, int, tuple[int, ...]]], k: int) -> tuple:
    """signature CẤU TRÚC quanh occurrence k: (left, right); token thuộc occurrence KHÁC → ENT."""
    s, e, _ = occ[k]
    ent_pos = set()
    for a, b, _ in occ:
        ent_pos.update(range(a, b))

    def ctx(rng):
        return tuple(ENT if p in ent_pos else ids[p] for p in rng if 0 <= p < len(ids))
    left = ctx(range(s - SIG_W, s))
    right = ctx(range(e, e + SIG_W))
    return (left, right)


def run_coref_pass(
    state: GraphState,
    records: Iterable[str],
    *,
    gidf_min: float = 3.0,       # TODO(M4): chỉnh theo phân bố _gidf graph
    gidf_solo: float = 6.0,      # single-token entity cần đặc thù CAO hơn (loại chữ-cái/từ-Hoa-đầu-câu)
    log=lambda m: None,
    log_every: int = 20000,
) -> dict[str, int]:
    """Quét corpus → điền state.coref / state.after_role (name-repeat single). Trả stats."""
    ps = state.parents
    vocab = state.vocab
    gidf = ps._gidf  # noqa: SLF001 — đặc thù token (Luật 0: idf, không phải bảng từ)

    pair_same: Counter = Counter()
    pair_tot: Counter = Counter()
    after: dict[tuple, Counter] = defaultdict(Counter)
    stats = Counter()

    for r, text in enumerate(records):
        if r and r % log_every == 0:
            log(f"  coref-pass {r:,} record · pairs {len(pair_tot):,}")
        # encode_text (KHÔNG encode_tokens(tokenize())) — bẫy BPE: pre-split words mất Ġ
        # prefix → mid-sentence tokens tạo new ids ngoài surface range, gidf=0, occurrence=0.
        # encode_text dùng tokenizer.encode_pieces → Ġ đúng trên mọi vị trí.
        try:
            ids = vocab.encode_text(text)
        except RuntimeError:
            # legacy vocab (không tokenizer) — fallback encode_tokens
            from nn_aivie.teaching.align import tokenize
            ids = vocab.encode_tokens(tokenize(text))
        if len(ids) < 3:
            continue
        occ = _find_occurrences(ids, vocab, gidf, gidf_min, gidf_solo)
        if len(occ) < 2:
            continue
        stats["records_with_occ"] += 1
        sigs = [_sig(ids, occ, k) for k in range(len(occ))]
        fillers = [o[2] for o in occ]
        # after_role
        for k, (s, e, _) in enumerate(occ):
            if e < len(ids):
                after[sigs[k]][ids[e]] += 1
        # NAME-REPEAT single: cặp occurrence cùng filler trong đoạn
        for a in range(len(occ)):
            for b in range(len(occ)):
                if a == b:
                    continue
                pair_tot[(sigs[a], sigs[b])] += 1
                if fillers[a] == fillers[b]:
                    pair_same[(sigs[a], sigs[b])] += 1

    # finalize coref — tie-break TẤT ĐỊNH
    coref: dict[tuple, tuple] = {}
    by_a: dict[tuple, list[tuple]] = defaultdict(list)
    for (a, b) in pair_tot:
        by_a[a].append(b)
    for a in sorted(by_a):
        best, bkey = None, None
        for b in sorted(by_a[a]):
            ratio = pair_same[(a, b)] / pair_tot[(a, b)]
            key = (ratio, pair_tot[(a, b)])
            if ratio > 0.55 and (bkey is None or key > bkey):
                best, bkey = b, key
        if best is not None:
            coref[a] = best
    after_role = {sig: c.most_common(1)[0][0] for sig, c in after.items() if c}

    state.coref = coref          # field + serialize ĐÃ có (state.py, TK_044 P3.1) — save là persist
    state.after_role = after_role
    stats["coref_edges"] = len(coref)
    stats["after_role"] = len(after_role)
    log(f"✓ coref-pass xong · edges {len(coref):,} · after_role {len(after_role):,} · "
        f"records_with_occ {stats['records_with_occ']:,}")
    return dict(stats)
