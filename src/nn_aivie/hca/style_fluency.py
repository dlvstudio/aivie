"""
Style fluency — đo / gate bằng cosine trên E đã học (Luật 0).

Không hardcode marker. Dùng GraphEmbedding.similarity (PPMI-SVD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def token_cosine(emb, a: int, b: int) -> float:
    if emb is None:
        return 0.0
    return float(emb.similarity(a, b))


def mean_bigram_cosine(emb, token_ids: list[int]) -> float:
    """Độ mượt cục bộ: trung bình cosine(E[t_i], E[t_{i+1}]) trên chuỗi surface."""
    if emb is None or len(token_ids) < 2:
        return 0.0
    s = 0.0
    n = 0
    for a, b in zip(token_ids, token_ids[1:]):
        c = token_cosine(emb, a, b)
        if math.isfinite(c):
            s += c
            n += 1
    return s / n if n else 0.0


def slot_fit_cosine(emb, left: int, mid: int, right: int | None = None) -> float:
    """Độ khớp chỗ chèn: cos(left,mid) [, × cos(mid,right) nếu có]."""
    left_c = token_cosine(emb, left, mid)
    if right is None:
        return left_c
    right_c = token_cosine(emb, mid, right)
    # trung bình (ổn định hơn nhân khi một phía ~0)
    return 0.5 * (left_c + right_c)


@dataclass(frozen=True)
class StylePick:
    unit: tuple[int, ...]
    head: int
    cosine: float


def pick_style_by_cosine(
    emb,
    left_id: int,
    plan: list[tuple[int, ...]],
    *,
    start_i: int = 0,
    cos_thresh: float = 0.0,
    right_id: int | None = None,
) -> StylePick | None:
    """Chọn unit còn lại khớp nhất với left [, và right nếu có].

    Có right (agenda kế): score = 0.5*(cos(left,head)+cos(head,right)) — chỗ chèn
    giữa hai token. Không right: chỉ cos(left,head). Dưới thresh → None.
    """
    if emb is None or start_i >= len(plan):
        return None
    best: StylePick | None = None
    for unit in plan[start_i:]:
        if not unit:
            continue
        head = int(unit[0])
        left_c = token_cosine(emb, left_id, head)
        if right_id is not None:
            c = 0.5 * (left_c + token_cosine(emb, head, right_id))
        else:
            c = left_c
        if c < cos_thresh:
            continue
        if best is None or c > best.cosine:
            best = StylePick(unit=unit, head=head, cosine=c)
    return best


def annotate_inject_fits(
    emb,
    token_ids: list[int],
    style_token_set: set[int],
) -> list[float]:
    """Với mỗi vị trí là style token: slot_fit với left (và right nếu có)."""
    fits: list[float] = []
    for i, t in enumerate(token_ids):
        if t not in style_token_set:
            continue
        if i == 0:
            continue
        right = token_ids[i + 1] if i + 1 < len(token_ids) else None
        fits.append(slot_fit_cosine(emb, token_ids[i - 1], t, right))
    return fits


@dataclass(frozen=True)
class WSeqPeakiness:
    """Peaky = đỉnh hẹp (Việt→Nam); hub = mass phẳng / hub function."""

    peak_frac: float
    top2_gap: float
    n_eff: float
    peaky: bool
    top_ids: tuple[int, ...]


def wseq_peakiness(
    seq_row: dict[int, float] | None,
    *,
    peak_frac_min: float = 0.35,
    tie_eps: float = 0.08,
    gidf_w0: float | None = None,
    row_width_hub: int = 48,
) -> WSeqPeakiness:
    """Đo peaky vs hub trên hàng W_seq (Luật 0 — không list từ)."""
    if not seq_row:
        return WSeqPeakiness(0.0, 0.0, 0.0, False, ())
    items = [(int(k), float(v)) for k, v in seq_row.items() if v > 0]
    if not items:
        return WSeqPeakiness(0.0, 0.0, 0.0, False, ())
    total = sum(w for _, w in items)
    if total <= 0:
        return WSeqPeakiness(0.0, 0.0, 0.0, False, ())
    items.sort(key=lambda x: -x[1])
    probs = [w / total for _, w in items]
    peak = probs[0]
    gap = (probs[0] - probs[1]) if len(probs) > 1 else 1.0
    n_eff = 1.0 / max(1e-12, sum(p * p for p in probs))
    top_ids = tuple(i for i, _ in items[:8])
    peaky = peak >= peak_frac_min and gap >= tie_eps
    if gidf_w0 is not None and gidf_w0 < 3.5 and len(items) >= row_width_hub:
        peaky = False
    if n_eff >= 12.0:
        peaky = False
    return WSeqPeakiness(peak, gap, float(n_eff), peaky, top_ids)


def _norm_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def ctx_mean_emb(emb, token_ids: list[int], *, k: int = 6) -> np.ndarray | None:
    """mean(E[out[-k:]]) đã L2-norm."""
    if emb is None or not token_ids:
        return None
    E = emb.matrix
    ids = [t for t in token_ids[-k:] if 0 <= t < len(E)]
    if not ids:
        return None
    return _norm_vec(np.mean(E[ids], axis=0).astype(np.float64))


def emb_cos_vec_token(emb, vec: np.ndarray | None, tid: int) -> float:
    if emb is None or vec is None:
        return 0.0
    E = emb.matrix
    if tid < 0 or tid >= len(E):
        return 0.0
    return float(np.dot(vec, _norm_vec(E[tid].astype(np.float64))))


def ctx_disambiguate_boosts(
    emb,
    out_ids: list[int],
    cand_ids: list[int],
    *,
    agenda_next: int | None = None,
    ctx_k: int = 6,
    ctx_weight: float = 0.6,
    agenda_weight: float = 0.4,
) -> dict[int, float]:
    """Boost phá hòa: β·cos(ctx, v) + γ·cos(v, agenda_next)."""
    ctx = ctx_mean_emb(emb, out_ids, k=ctx_k)
    out: dict[int, float] = {}
    for v in cand_ids:
        sc = ctx_weight * emb_cos_vec_token(emb, ctx, v)
        if agenda_next is not None:
            sc += agenda_weight * token_cosine(emb, v, agenda_next)
        if sc != 0.0:
            out[int(v)] = float(sc)
    return out


def apply_soft_style_decode(
    scores: np.ndarray,
    *,
    emb,
    w0: int,
    out_ids: list[int],
    seq_row: dict[int, float] | None,
    pending: list[int],
    agenda_seq: list[int],
    vocab_size: int,
    peak_frac_min: float = 0.35,
    tie_eps: float = 0.08,
    hub_wseq_scale: float = 0.35,
    ctx_k: int = 6,
    ctx_weight: float = 0.6,
    agenda_weight: float = 0.75,
    gidf_w0: float | None = None,
    allow_slip: bool = False,
    fluency_weight: float = 0.5,
) -> tuple[WSeqPeakiness, int | None]:
    """Điều chỉnh scores trên payoff soft. Trả (peakiness, soft_agenda_target).

    Peaky + tie → ctx disambiguate. Hub → hạ W_seq thuần + ctx-E.
    Mặc định bám pending[0] (không trượt); fluency_weight·cos(w0,v) giảm salad.
    """
    pk = wseq_peakiness(
        seq_row, peak_frac_min=peak_frac_min, tie_eps=tie_eps, gidf_w0=gidf_w0,
    )
    agenda_next = (
        int(agenda_seq[pending[0]])
        if pending and pending[0] < len(agenda_seq)
        else None
    )
    soft_target: int | None = agenda_next

    fin_idx = np.where(np.isfinite(scores))[0]
    if fin_idx.size:
        top_n = min(32, int(fin_idx.size))
        part = fin_idx[np.argpartition(-scores[fin_idx], top_n - 1)[:top_n]]
        cands = {int(i) for i in part}
    else:
        cands = set()
    cands.update(pk.top_ids)
    for i in pending[:3]:
        if i < len(agenda_seq):
            cands.add(int(agenda_seq[i]))

    fin = scores[np.isfinite(scores)]
    base = float(fin.max()) if fin.size else 0.0

    # Fluency từ w0: mọi cand được nhân nhẹ theo cos(w0, v) — chống nhảy lệch register.
    if fluency_weight > 0 and emb is not None and cands:
        for v in list(cands):
            if v >= vocab_size or not np.isfinite(scores[v]):
                continue
            fc = token_cosine(emb, w0, v)
            scores[v] = float(scores[v]) * (
                (1.0 - fluency_weight) + fluency_weight * max(0.0, 0.15 + 0.85 * fc)
            )

    if pk.peaky and len(pk.top_ids) >= 2 and pk.top2_gap < tie_eps * 1.5:
        boosts = ctx_disambiguate_boosts(
            emb, out_ids, list(pk.top_ids[:4]),
            agenda_next=agenda_next, ctx_k=ctx_k,
            ctx_weight=ctx_weight, agenda_weight=agenda_weight,
        )
        fin = scores[np.isfinite(scores)]
        base = float(fin.max()) if fin.size else 0.0
        for v, b in boosts.items():
            if v < vocab_size and np.isfinite(scores[v]):
                scores[v] = float(scores[v]) + abs(base) * 0.15 * max(0.0, b)
    elif not pk.peaky:
        ctx = ctx_mean_emb(emb, out_ids, k=ctx_k)
        fin = scores[np.isfinite(scores)]
        base = float(fin.max()) if fin.size else 0.0
        for tid in pk.top_ids[:6]:
            if tid >= vocab_size or not np.isfinite(scores[tid]):
                continue
            scores[tid] = (
                float(scores[tid]) * hub_wseq_scale
                + base * (1.0 - hub_wseq_scale) * 0.05
            )
        boosts = ctx_disambiguate_boosts(
            emb, out_ids, list(cands),
            agenda_next=agenda_next if not allow_slip else None,
            ctx_k=ctx_k,
            ctx_weight=ctx_weight,
            agenda_weight=agenda_weight if not allow_slip else 0.0,
        )
        for v, b in boosts.items():
            if v < vocab_size and np.isfinite(scores[v]):
                scores[v] = float(scores[v]) + abs(base) * 0.25 * max(0.0, b)
        if allow_slip:
            best_i = None
            best_c = -1e9
            for off, i in enumerate(pending[:3]):
                t = int(agenda_seq[i])
                c = emb_cos_vec_token(emb, ctx, t)
                c -= 0.03 * off
                if c > best_c:
                    best_c = c
                    best_i = i
            if best_i is not None:
                soft_target = int(agenda_seq[best_i])

    # Luôn kéo mạnh pending[0] (ý payoff kế) — trừ khi allow_slip đã chọn khác.
    if soft_target is not None and soft_target < vocab_size:
        fin = scores[np.isfinite(scores)]
        base = float(fin.max()) if fin.size else 0.0
        # Ưu tiên nếu reachable từ w0 trên E hoặc W_seq
        wseq_hit = bool(seq_row and soft_target in seq_row and seq_row[soft_target] > 0)
        e_hit = token_cosine(emb, w0, soft_target) if emb is not None else 0.0
        boost = 0.45 * agenda_weight + (0.25 if wseq_hit else 0.0) + 0.15 * max(0.0, e_hit)
        if np.isfinite(scores[soft_target]):
            scores[soft_target] = max(float(scores[soft_target]), base * (1.0 + boost))
        else:
            scores[soft_target] = base * (0.55 + 0.45 * agenda_weight)

    return pk, soft_target
