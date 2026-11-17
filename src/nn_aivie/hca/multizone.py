"""
MULTI-ZONE retrieval = MULTI-HEAD ATTENTION trên đồ thị (TK_042 nhánh compose, 2026-07-22).

Ý tác giả: prompt phức có nhiều "vùng chú ý" (zone) — vd "hài hước | Trần Quốc Toản bóp cam | gen Z".
Mỗi zone = một HEAD attention: query = embedding của zone, retrieve fact riêng. = multi-head attn
{Q = zone-embedding, K/V = fact (parent) embeddings}, trọng số Hebbian (E đã học), softmax = Hopfield.

## Pipeline (Luật-0 sạch — E + facts_about đã học, không nhãn/không string-rule)

1. **Zone hoá (v5 — CAP-SPAN):** RUN viết-HOA liền kề + bỏ single-token đầu prompt (Viết/Kể/So).
   Giữ thứ tự prompt khi trả zone (không sort cos).
2. **Retrieve per-zone:** facts_about → lọc cover token zone (đa-token) → Hopfield cos re-rank.
3. **Zone-type:** CONTENT nếu top-cos ≥ ngưỡng; STYLE thường bị cap-span bỏ.

Trả cấu trúc để caller (generate) ghép. Wire opt-in qua `HCAConfig.ar_multizone`
(`_plan_from_multizone` trong generate_hebbian) — mặc định tắt, đo A/B trước khi bật.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.hca.generate_hebbian import _completion_from_parent


@dataclass
class Zone:
    tokens: tuple[int, ...]                 # token của vùng chú ý
    max_gidf: float                          # đặc thù cao nhất (entity-like?)
    facts: list[tuple[int, float, float]] = field(default_factory=list)  # (parent_idx, cos, softmax_w)

    @property
    def top_cos(self) -> float:
        return self.facts[0][1] if self.facts else 0.0

    def is_content(self, cos_min: float) -> bool:
        """CONTENT nếu retrieve được fact khớp zone (cosine đủ cao). STYLE/chức-năng → thấp/rỗng."""
        return bool(self.facts) and self.top_cos >= cos_min


def _entity_span_zones(query_ids, vocab, ps, gidf_min: float, gidf_solo: float,
                       *, state=None, left_pmi_min: float = 0.0) -> list[list[int]]:
    """Zone = RUN token VIẾT-HOA liền kề (`cap_class≥1` = danh-từ-riêng), giữ thứ tự prompt.

    v4 (2026-07-22): v3 dùng `ps.phrases` + gidf-gate → nhận NHẦM collocation thường ("hài hước" gidf
    6.9 lọt) vì Re-Pair trộn cụm-entity lẫn cụm-thường; gidf không tách được. cap_class TÁCH đúng:
    "hài hước/về/và" viết-THƯỜNG (cap 0) → bỏ; "Trần Hưng Đạo/Bạch Đằng" viết-HOA liền → 1 span. Đây
    là detector cap-span đã validate ở coref-pass (P3.1): word-start bỏ mảnh-BPE; run đa-token HOẶC
    single gidf≥gidf_solo (bỏ từ-Hoa-đầu-câu 'Kể/Viết' gidf vừa).

    v5 (2026-07-23): single-token == token ĐẦU prompt → BỎ (Viết/Kể/So viết-hoa đầu câu lọt
    dù gidf cao → zone rác ChatGPT/Hội thoại). Vị trí = tín hiệu cấu trúc, Luật 0; không hardcode
    danh sách động-từ. Multi-token đầu câu ("Trần Hưng Đạo…") vẫn giữ.
    """
    zones, cur = [], []
    q0 = query_ids[0] if query_ids else None
    # §12.29 — vị trí "mở đầu câu hỏi" KHÔNG chỉ là pos 0: sau nhãn frame `Hỏi:` cũng vậy.
    # Luật v5 chỉ bỏ pos 0 nên `Kể`(gidf 8.26)/`Viết`(7.70) đứng ngay sau `Hỏi:` vẫn lọt và
    # thành zone rác (đo: zone `Kể` kéo về fact "Hiện vật kể chuyện quá khứ"). So INT trên
    # special nguyên-tử ⇒ Luật 0 sạch, không hardcode danh sách động từ.
    from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL

    _frame = {vocab._token_to_id.get(x) for x in (HOI_LABEL, DAP_LABEL)}  # noqa: SLF001
    opener = set()
    for k, t in enumerate(query_ids):
        if k == 0 or (k > 0 and int(query_ids[k - 1]) in _frame):
            opener.add(k)

    def flush():
        if not cur:
            return
        if len(cur) == 1 and q0 is not None and cur[0] == q0:
            cur.clear()
            return
        if len(cur) >= 2 or max(ps._gidf(x) for x in cur) >= gidf_solo:
            zones.append(list(cur))
        cur.clear()

    for k, t in enumerate(query_ids):
        # §12.29: `Ġ` (khoảng trắng) có cap_class=2 ⇒ độn vào run cho đủ len≥2 và cho lọt
        # entity đơn lẻ một cách TÌNH CỜ (đo: zone `['ĠTrần','Ġ']`). Token cấu-trúc không phải
        # danh-từ-riêng — loại thẳng.
        from nn_aivie.hca.context_positional import is_structural_token

        if is_structural_token(vocab, int(t)):
            flush()
            continue
        if k in opener and vocab.cap_class(t) >= 1:
            # mở đầu câu hỏi: viết hoa là do VỊ TRÍ, không phải vì là tên riêng
            flush()
            continue
        is_cap = vocab.cap_class(t) >= 1 and ps._gdf.get(t, 0) > 0 and ps._gidf(t) >= gidf_min
        ws = (k == 0) or vocab.token_of(t).startswith(MARK_SPACE)   # đầu-từ (bỏ mảnh continuation)
        if is_cap and (cur or ws):
            cur.append(t)
        else:
            flush()
    flush()
    if state is not None and left_pmi_min > 0.0 and zones:
        zones = _extend_left_by_pmi(state, query_ids, zones, left_pmi_min)
    return zones


def _extend_left_by_pmi(state, query_ids, zones, thr: float) -> list[list[int]]:
    """TK_045 §12.46 — nới vùng sang TRÁI khi token liền trước gắn chặt với thực thể.

    Ca đo được (§12.29d): `so sánh nhà Lý và nhà Trần` cho vùng `['Lý']`, MẤT `nhà` ⇒
    `facts_about(('Lý',))` trả gia phả (`Lý Cạnh sinh ra Lý Thiên Bảo…`) thay vì fact triều đại.
    `nhà` viết thường nên cap-run không nuốt nó.

    Tín hiệu: PMI(prev, head) trên emit, KHÔNG gộp hoa/thường (§12.45③). Chỉ nới khi token đó
    THẬT SỰ đứng ngay trước vùng TRONG PROMPT — không bịa thêm token.
    """
    from nn_aivie.hca.matrix_core import left_pmi

    q = [int(t) for t in query_ids]
    pos: dict[int, list[int]] = {}
    for i, t in enumerate(q):
        pos.setdefault(t, []).append(i)
    out: list[list[int]] = []
    for z in zones:
        head = int(z[0])
        added = None
        for i in pos.get(head, ()):
            if i == 0:
                continue
            prev = q[i - 1]
            if prev in z:
                continue
            sc = left_pmi(state, head).get(prev)
            if sc is not None and sc >= float(thr):
                added = prev
                break
        out.append([added] + list(z) if added is not None else list(z))
    return out


def _emb(E: np.ndarray, ids) -> np.ndarray | None:
    ids = [t for t in ids if 0 <= t < E.shape[0]]
    if not ids:
        return None
    v = E[ids].mean(0)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-9 else None


def _fact_cover(ps, pi: int, ztoks) -> float:
    """Tỉ lệ token zone có trong source ∪ match ∪ completion — ưu tiên parent nói ĐỦ cụm entity."""
    rec = ps.parents[pi]
    pool: set[int] = set(_completion_from_parent(ps, pi))
    if rec.source_ids:
        pool.update(rec.source_ids)
    if rec.match_ids:
        pool.update(rec.match_ids)
    if not ztoks:
        return 0.0
    return sum(1 for t in ztoks if t in pool) / float(len(ztoks))


def zone_retrieve(
    state, query_ids: list[int], *,
    gidf_min: float = 3.0, gidf_solo: float = 6.0, cand_per_zone: int = 8,
    beta: float = 5.0, facts_top: int = 2, theme_weight: float = 1.0,
    cover_min: float = 0.5, left_pmi_min: float = 0.0,
) -> list[Zone]:
    """Zone hoá query (cap-span entity) + Hopfield-attention retrieve fact mỗi zone.

    Giữ thứ tự prompt (v5) — "So sánh A và B" phải A rồi B, không sort theo cos.
    `cover_min`: fact đa-token zone phải phủ ≥ tỉ lệ token zone trên source∪match∪completion.
    Điểm xếp = cos × (0.25 + 0.75×cover) — entity dài (Trần Hưng Đạo) thắng đồng-họ lệch.
    """
    ps, vocab = state.parents, state.vocab
    E = np.asarray(state.embedding.matrix)
    zone_toks = _entity_span_zones(list(query_ids), vocab, ps, gidf_min, gidf_solo,
                                   state=state, left_pmi_min=left_pmi_min)
    if not zone_toks:
        return []
    zones: list[Zone] = []
    for ztoks in zone_toks:
        zq = _emb(E, ztoks)
        z = Zone(tokens=tuple(ztoks), max_gidf=max(ps._gidf(t) for t in ztoks))
        if zq is not None:
            # entity dài: lấy pool rộng hơn rồi lọc cover (8 quá hẹp → miss fact đúng)
            pool_k = cand_per_zone * (3 if len(ztoks) >= 3 else 1)
            cands = ps.facts_about(tuple(ztoks), top_k=pool_k, vocab=vocab,
                                   theme_weight=theme_weight)
            scored: list[tuple[float, float, int]] = []  # (rank_score, cos, pi)
            multi = len(ztoks) >= 2
            for pi in cands:
                cov = _fact_cover(ps, pi, ztoks)
                if multi and cover_min > 0 and cov < cover_min:
                    continue
                fe = _emb(E, list(_completion_from_parent(ps, pi)))
                if fe is None:
                    continue
                cos = float(zq @ fe)
                rank = cos * (0.25 + 0.75 * cov)
                scored.append((rank, cos, pi))
            # nếu cover lọc hết → nới (fallback cos thuần) để không mất zone
            if not scored and multi and cover_min > 0:
                for pi in cands:
                    fe = _emb(E, list(_completion_from_parent(ps, pi)))
                    if fe is not None:
                        cos = float(zq @ fe)
                        scored.append((cos, cos, pi))
            scored.sort(reverse=True)
            if scored:
                sims = np.array([c for _, c, _ in scored], dtype=np.float64)
                w = np.exp(beta * sims)
                w /= w.sum()
                z.facts = [
                    (pi, cos, float(wi))
                    for (_, cos, pi), wi in zip(scored[:facts_top], w[:facts_top])
                ]
        zones.append(z)
    return zones  # thứ tự prompt


def zone_handoff_span(state, history_ids: list[int], *, config, emitted_pis: set[int],
                      max_tokens: int = 32) -> tuple[list[int], int] | None:
    """TK_045 §12.29 — span fact của ZONE kế tiếp, cho matrix core dùng khi handoff episode cạn.

    Vì sao cần: `zone_retrieve` (ý tưởng đa-vùng-chú-ý) ĐÃ chạy — đo mini nó lấy đúng một fact
    cho mỗi thực thể (`so sánh nhà Lý và nhà Trần` → 1 fact Trần + 1 fact Lý). Nhưng nó chỉ nằm
    trên **đường plan** (`ar_no_agenda=False`), mà `matrix_core_logits` chỉ được gọi trong
    `_wseq_fallback` (`ar_no_agenda=True`) ⇒ **hai đường loại trừ nhau**: bật MZ là mất sạch
    §12.12–§12.28 (đo: số vỡ `10.09`, mất số `Trận Bạch Đằng năm` + bịa cả khung `Đáp:`).

    Hàm này bắc cầu: giữ nguyên matrix core, chỉ cấp thêm NGUỒN span khi episode-handoff cạn.
    Trả `(span, parent_idx)`; `emitted_pis` để không lặp lại cha đã phát.
    """
    zones = zone_retrieve(
        state, list(history_ids),
        gidf_min=float(getattr(config, "ar_multizone_gidf_min", 3.0)),
        gidf_solo=float(getattr(config, "ar_multizone_gidf_solo", 6.0)),
        beta=float(getattr(config, "ar_multizone_beta", 5.0)),
        facts_top=int(getattr(config, "ar_multizone_facts_top", 1)),
        theme_weight=float(getattr(config, "compose_theme_weight", 1.0)),
        cover_min=float(getattr(config, "ar_multizone_cover_min", 0.5)),
        left_pmi_min=float(getattr(config, "ar_multizone_left_pmi_min", 0.0)),
    )
    if len(zones) < 2:
        return None  # <2 vùng thì đây không phải ca GHÉP — để matrix core tự lo
    cos_min = float(getattr(config, "ar_multizone_cos_min", 0.4))
    ps, vocab = state.parents, state.vocab
    for z in zones:  # GIỮ thứ tự prompt: "so sánh A và B" phải A trước
        for pi, cos, _w in z.facts:
            if pi in emitted_pis or cos < cos_min:
                continue
            rec = ps.parents[pi]
            seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
            span = [int(t) for t in seq
                    if 0 <= int(t) < len(vocab) and vocab.is_surface(int(t))][:max_tokens]
            if span:
                return span, int(pi)
    return None
