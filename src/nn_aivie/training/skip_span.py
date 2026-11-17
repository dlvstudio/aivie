"""
TK_036 — SKIP_SPAN substrate: token → cụm content xa (gap 2–8, cụm phải 1–4 token).

Đầu trái: token đơn. Đầu phải: tuple token (1–4) — cụm ảo như (Việt, Nam), không chỉ token đầu.
GIỮ NGUYÊN link token→cha qua `_index`. Khi power đủ cao → promote SkipCluster (metadata).

Luật 0: lọc PMI + idf band + hướng — không hardcode từ.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from nn_aivie.core.skip_span_constants import (
    SKIP_MAX_GAP,
    SKIP_MAX_RIGHT_LEN,
    SKIP_MIN_GAP,
    skip_right_tuple,
)
from nn_aivie.core.vocab import is_surface_token

_FRAME_TOKENS = {"Hỏi:", "Đáp:"}

SkipKey = tuple[int, tuple[int, ...]]


def build_phrase_by_start(ps, *, min_count: int = 4) -> dict[int, list[tuple[int, ...]]]:
    """Cụm đã học (phrases + ISG cluster) làm endpoint phải của skip span."""
    n_par = len(ps.parents)
    phrase_lo = min_count
    phrase_hi = 8 if n_par > 20 else 10**18
    phrase_keys: set[tuple[int, ...]] = set()
    for ph, df in zip(getattr(ps, "phrases", ()), getattr(ps, "phrase_df", ())):
        ph_t = tuple(ph)
        if 2 <= len(ph_t) <= SKIP_MAX_RIGHT_LEN and phrase_lo <= int(df) <= phrase_hi:
            phrase_keys.add(ph_t)
    seg = getattr(ps, "seg", None)
    if seg is not None:
        for (_a, _b), ci in getattr(seg, "cluster_of_pair", {}).items():
            c = seg.clusters[ci]
            pw = int(getattr(c, "power", 0) or 0)
            if phrase_lo <= pw <= phrase_hi:
                ph_t = (int(c.left), int(c.right))
                if 2 <= len(ph_t) <= SKIP_MAX_RIGHT_LEN:
                    phrase_keys.add(ph_t)
    phrase_by_start: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for ph in phrase_keys:
        phrase_by_start[ph[0]].append(ph)
    return phrase_by_start


def _is_skip_content_token(
    tid: int, vocab, gdf: dict[int, int], n_par: int, *, min_df: int = 1
) -> bool:
    tok = vocab.token_of(tid)
    if tok in _FRAME_TOKENS:
        return False
    if not is_surface_token(tok):
        return False
    d = gdf.get(tid, 0)
    if d < min_df:
        return False
    if n_par > 20:
        hi = max(min_df + 1, int(0.35 * n_par))
        if d > hi:
            return False
    return True


def _right_spans_at(
    src: list[int],
    j: int,
    vocab,
    gdf: dict[int, int],
    n_par: int,
    phrase_by_start: dict[int, list[tuple[int, ...]]],
) -> list[tuple[int, ...]]:
    """Cụm content đã học bắt đầu tại src[j]. Fallback toy: cụm content dài nhất."""
    if j >= len(src) or not _is_skip_content_token(src[j], vocab, gdf, n_par):
        return []
    if phrase_by_start:
        out: list[tuple[int, ...]] = []
        for ph in phrase_by_start.get(src[j], ()):
            if any(not _is_skip_content_token(t, vocab, gdf, n_par) for t in ph):
                continue
            if tuple(src[j : j + len(ph)]) == ph:
                out.append(ph)
        return out
    best: tuple[int, ...] | None = None
    for L in range(2, min(SKIP_MAX_RIGHT_LEN, len(src) - j) + 1):
        chunk = tuple(src[j : j + L])
        if not all(_is_skip_content_token(t, vocab, gdf, n_par) for t in chunk):
            break
        best = chunk
    return [best] if best is not None else []


def _scan_skip_pairs(
    src: list[int],
    vocab,
    gdf: dict[int, int],
    n_par: int,
    phrase_by_start: dict[int, list[tuple[int, ...]]],
) -> list[tuple[int, int, tuple[int, ...], int]]:
    """(parent_idx placeholder 0, left_pos, right_tokens, gap) — quét một chuỗi."""
    hits: list[tuple[int, int, tuple[int, ...], int]] = []
    content_pos = [
        i for i, t in enumerate(src)
        if _is_skip_content_token(t, vocab, gdf, n_par)
    ]
    for i in content_pos:
        a = src[i]
        j_lo = i + SKIP_MIN_GAP + 1
        j_hi = min(len(src), i + SKIP_MAX_GAP + 2)
        for j in range(j_lo, j_hi):
            rights = _right_spans_at(src, j, vocab, gdf, n_par, phrase_by_start)
            if not rights:
                continue
            right = rights[-1]  # cụm dài nhất tại j
            gap = j - i - 1
            hits.append((i, j, right, gap))
    return hits


def mine_skip_spans(
    state,
    *,
    min_gap: int = SKIP_MIN_GAP,
    max_gap: int = SKIP_MAX_GAP,
    min_count: int = 4,
    min_pmi: float = 1.0,
    dir_ratio: float = 1.5,
    promote_power: int = 10,
    max_spans: int = 30_000,
    max_prov_per_pair: int = 40,
    logger=None,
) -> dict[str, int]:
    """Khai phá SkipSpan — bootstrap ledger từ mọi cha rồi materialize."""
    ps = state.parents
    vocab = state.vocab
    if ps is None:
        return {"spans": 0}

    from nn_aivie.core.skip_ledger import SkipLedger

    ledger = ps.ensure_skip_ledger(
        min_gap=min_gap,
        max_gap=max_gap,
        min_count=min_count,
        min_pmi=min_pmi,
        dir_ratio=dir_ratio,
        promote_power=promote_power,
        max_spans=max_spans,
        max_prov_per_pair=max_prov_per_pair,
    )
    ledger.rebuild_all(ps, vocab, logger=logger)
    return ledger.materialize(ps, vocab, logger=logger)
