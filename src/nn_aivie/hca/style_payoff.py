"""
Style PAYOFF — Setup(fact) → Pivot(frame) → Payoff(snippet styled).

Không chỉ inject n-gram: retrieve một MẨU thân hài (đời thường / lệch kỳ vọng),
ghép sau fact. Luật 0: E-cosine + corpus label-conditioned, không hardcode punchline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nn_aivie.hca.style_corpus import (
    _iter_styled_texts,
    _split_label_body,
    ranked_discourse_ngrams,
)
from nn_aivie.hca.style_readout import _core, _is_artifact, style_seeds_from_prompt
from nn_aivie.hca.objcache import cache_get, cache_put

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_PIVOT_SPLIT = re.compile(r"(?<=[.!?…])?\s*(?:Thế nhưng|Rốt cuộc|thế nhưng|rốt cuộc)\s+")
# Meta / echo label — không phải punchline đời thường.
_META_PAYOFF = re.compile(
    r"(?i)vi[eế]t\s+v[aă]n|đo[aạ]n\s+v[aă]n\s+h[aà]i|h[aà]i\s+h[uứ][oớ]c\s+kh[oô]ng|"
    r"phong\s+c[aá]ch|style\s+prompt|nh[aãn]\s+style"
)
_PAYOFF_CACHE: dict[tuple, "PayoffStore"] = {}


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def _mean_emb(E: np.ndarray, ids: list[int]) -> np.ndarray | None:
    rows = [E[i] for i in ids if 0 <= i < len(E)]
    if not rows:
        return None
    return _norm(np.mean(rows, axis=0))


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def _prefer_word_start(vocab, tid: int) -> int:
    tok = vocab.token_of(tid)
    if tok.startswith("Ġ") or tok.startswith("Ċ"):
        return tid
    for cand in (f"Ġ{tok}", f"Ġ{_core(tok)}"):
        alt = vocab._token_to_id.get(cand)  # noqa: SLF001
        if alt is not None:
            return int(alt)
    return tid


@dataclass
class PayoffSnippet:
    token_ids: tuple[int, ...]
    text: str
    vec: np.ndarray
    after_pivot: bool = False
    parent_low: str = ""  # cả thân bài — neo entity dù punch không lặp tên


@dataclass
class PayoffStore:
    snippets: list[PayoffSnippet] = field(default_factory=list)


def build_payoff_store(
    state,
    paths: list[Path] | None = None,
    *,
    gidf_min: float = 4.0,
    max_snips: int = 800,
) -> PayoffStore:
    """Câu/mệnh đề từ thân styled có seed HOW (hài…)."""
    vocab, ps = state.vocab, state.parents
    E = state.embedding.matrix if state.embedding is not None else None
    if ps is None or E is None:
        return PayoffStore()
    if paths is None:
        root = Path(__file__).resolve().parents[3]
        paths = [
            root / "data/text_corpus/styled_humor_history.jsonl",
            root / "data/text_corpus/styled_sft_1k.jsonl",
            root / "data/text_corpus/styled_sft.jsonl",
        ]
    paths = [Path(p) for p in paths if Path(p).is_file()]
    store = PayoffStore()
    for text in _iter_styled_texts(paths):
        lab, bod = _split_label_body(text)
        if not bod.strip() or bod.strip() == lab.strip():
            continue
        label_ids = list(vocab.encode_prompt(lab))
        seeds = style_seeds_from_prompt(
            label_ids, vocab, ps, gidf_min=gidf_min, max_seeds=3,
        )
        if not seeds:
            continue
        # Câu thường + mẩu SAU pivot (Thế nhưng / Rốt cuộc) — ưu tiên after_pivot khi rank.
        parts: list[tuple[str, bool]] = []
        seen: set[str] = set()
        for p in _SENT_SPLIT.split(bod.strip()):
            p = p.strip()
            if p and p not in seen:
                parts.append((p, False))
                seen.add(p)
        piv_parts = [c.strip() for c in _PIVOT_SPLIT.split(bod.strip()) if c.strip()]
        for i, chunk in enumerate(piv_parts):
            after = i > 0  # phần trước pivot đầu = setup
            if chunk in seen:
                if after:
                    parts = [(p, True if p == chunk else ap) for p, ap in parts]
                continue
            parts.append((chunk, after))
            seen.add(chunk)
        for part, after_piv in parts:
            if len(part) < 20 or len(part) > 320:
                continue
            if _META_PAYOFF.search(part):
                continue
            if lab and lab.strip().lower()[:12] and lab.strip().lower()[:12] in part.lower():
                continue
            ids = [_prefer_word_start(vocab, t) for t in vocab.encode_prompt(part)
                   if vocab.is_surface(t)]
            ids = [t for t in ids if not _is_artifact(vocab.token_of(t))]
            cores = [_core(vocab.token_of(t)) for t in ids]
            if any(len(c) == 1 and c.isalpha() for c in cores):
                continue
            if len(ids) < 6 or len(ids) > 56:
                continue
            vec = _mean_emb(E, ids)
            if vec is None:
                continue
            store.snippets.append(
                PayoffSnippet(
                    tuple(ids), part, vec,
                    after_pivot=after_piv,
                    parent_low=bod.lower(),
                )
            )
            if len(store.snippets) >= max_snips:
                return store
    return store


def get_payoff_store(state, paths: list[Path] | None = None) -> PayoffStore:
    if paths is None:
        root = Path(__file__).resolve().parents[3]
        paths = [
            root / "data/text_corpus/styled_humor_history.jsonl",
            root / "data/text_corpus/styled_sft_1k.jsonl",
            root / "data/text_corpus/styled_sft.jsonl",
        ]
    paths = [Path(p) for p in paths if Path(p).is_file()]
    _sub = ("payoff_v10_hist", "|".join(str(p.resolve()) for p in paths))
    hit = cache_get(_PAYOFF_CACHE, state.vocab, _sub)
    if hit is not None:
        return hit
    store = build_payoff_store(state, paths)
    cache_put(_PAYOFF_CACHE, state.vocab, store, _sub)
    return store


def _first_sentence_ids(vocab, ids: list[int], *, max_tokens: int = 48) -> list[int]:
    """Cắt setup tới dấu câu đầu (hoặc max_tokens) — giữ fact liền, chừa chỗ payoff."""
    if not ids:
        return []
    out: list[int] = []
    for tid in ids:
        out.append(tid)
        tok = vocab.token_of(tid).lstrip("ĠĊ")
        if tok in ".!?…" and len(out) >= 8:
            break
        if len(out) >= max_tokens:
            break
    return out


def _surface_humor_bonus(text: str) -> float:
    low = text.lower()
    sc = 0.0
    if "thế nhưng" in low or " nhưng " in f" {low} ":
        sc += 0.06
    if "hệt" in low or "kiểu" in low or "y như" in low:
        sc += 0.05
    if "phết" in low or "hơi lố" in low:
        sc += 0.04
    # redup thô: lặp âm tiết 2–3 ký tự (đời thường)
    if re.search(r"(\b\w{2,4})\s+\1\b", low):
        sc += 0.04
    return sc


@dataclass(frozen=True)
class StylePayoffPlan:
    pivot: tuple[int, ...]
    payoff: tuple[int, ...]
    pivot_text: str
    payoff_text: str
    score: float


def plan_style_payoff(
    state,
    query_ids: list[int],
    entity_ids: list[int] | tuple[int, ...] = (),
    *,
    max_payoff_tokens: int = 28,
) -> StylePayoffPlan | None:
    """Chọn pivot (glue discourse) + payoff snippet lệch entity (incongruity trên E)."""
    vocab, ps = state.vocab, state.parents
    if ps is None or state.embedding is None:
        return None
    seeds = style_seeds_from_prompt(query_ids, vocab, ps, max_seeds=2)
    if not seeds:
        return None
    E = state.embedding.matrix
    # Chỉ entity viết-hoa (cap≥1) — bỏ HOW/style lọt entity_register («Viết/hài»).
    ents = [int(e) for e in entity_ids if vocab.cap_class(e) >= 1]
    if not ents:
        ents = list(entity_ids)[:4]
    # Không kéo style_v = «hài hước» (→ meta essay). Incongruity = XA entity + bonus bề mặt.
    ent_v = _mean_emb(E, ents) if ents else None
    # centroid nhẹ của vài snippet đầu (đời thường) làm neo tone — không dùng seed label.
    store = get_payoff_store(state)
    if not store.snippets:
        return None
    tone_rows = [s.vec for s in store.snippets[:64]]
    tone_v = _norm(np.mean(tone_rows, axis=0)) if tone_rows else None

    # Pivot: glue-frame đầu trong discourse rank
    pivot: tuple[int, ...] = ()
    pivot_text = ""
    for unit, _sc in ranked_discourse_ngrams(state, query_ids, top_k=8):
        cores = [_core(vocab.token_of(t)).lower() for t in unit]
        if len(unit) >= 2 and (
            "nhưng" in cores or "cuộc" in cores or cores[0][:1] == cores[1][:1]
        ):
            pivot = tuple(_prefer_word_start(vocab, t) for t in unit)
            pivot_text = " ".join(vocab.token_of(t) for t in pivot)
            break
    if not pivot:
        disc = ranked_discourse_ngrams(state, query_ids, top_k=1)
        if not disc:
            return None
        pivot = tuple(_prefer_word_start(vocab, t) for t in disc[0][0])
        pivot_text = " ".join(vocab.token_of(t) for t in pivot)

    # query cores (bỏ HOW ngắn) để neo snippet gắn đề
    q_cores = [
        _core(vocab.token_of(t)).lower()
        for t in query_ids
        if vocab.is_surface(t) and len(_core(vocab.token_of(t))) >= 3
    ]
    q_cores = [
        c for c in q_cores
        if c not in {"viết", "hài", "hước", "về", "một", "đoạn", "văn", "kể", "hãy"}
    ]
    # cụm entity viết-hoa liền (Ngô Quyền) — tránh "quyền" khớp "chủ quyền"
    ent_phrases: list[str] = []
    i = 0
    while i < len(query_ids):
        tid = query_ids[i]
        if vocab.is_surface(tid) and vocab.cap_class(tid) >= 1:
            run = [_core(vocab.token_of(tid)).lower()]
            j = i + 1
            while j < len(query_ids):
                tj = query_ids[j]
                if not (vocab.is_surface(tj) and vocab.cap_class(tj) >= 1):
                    break
                run.append(_core(vocab.token_of(tj)).lower())
                j += 1
            ph = " ".join(c for c in run if c)
            # chỉ cụm ≥2 token (Ngô Quyền) — tránh "Viết" đầu câu thành phrase giả
            if len(run) >= 2 and len(ph) >= 4:
                ent_phrases.append(ph)
            i = j
        else:
            i += 1

    def _core_hit_low(low: str, c: str) -> bool:
        pad = f" {low} "
        return f" {c} " in pad

    best: PayoffSnippet | None = None
    best_sc = -1e9
    for sn in store.snippets:
        if _META_PAYOFF.search(sn.text):
            continue
        low = sn.text.lower()
        # gần tone đời thường; không đẩy xa entity quá mạnh (cần neo lịch sử)
        sc = 0.40 * _cos(sn.vec, tone_v) - 0.25 * _cos(sn.vec, ent_v)
        sc += _surface_humor_bonus(sn.text)
        ent_hit = 0
        for e in ents[:6]:
            et = _core(vocab.token_of(e)).lower()
            if len(et) >= 3 and _core_hit_low(low, et):
                ent_hit += 1
        # neo nhẹ 1–3 entity token = thưởng; nhồi quá = phạt
        if 1 <= ent_hit <= 3:
            sc += 0.16 * ent_hit
        elif ent_hit > 3:
            sc -= 0.10 * (ent_hit - 3)
        phrase_hit = sum(1 for ph in ent_phrases if ph in low)
        parent_hit = sum(1 for ph in ent_phrases if ph and ph in (sn.parent_low or ""))
        sc += 0.35 * phrase_hit
        sc += 0.40 * parent_hit  # bài đúng entity dù punch không nhắc tên
        q_hit = sum(1 for c in q_cores[:8] if _core_hit_low(low, c))
        sc += 0.08 * min(q_hit, 4)
        # ưu tiên mẩu SAU pivot trong bài đúng entity
        if sn.after_pivot and parent_hit >= 1:
            sc += 0.50
        elif sn.after_pivot and phrase_hit >= 1:
            sc += 0.40
        elif sn.after_pivot and parent_hit == 0:
            sc -= 0.35
        hum = _surface_humor_bonus(sn.text)
        if not sn.after_pivot and ent_hit >= 2 and hum < 0.04:
            sc -= 0.40
        if sc > best_sc:
            best_sc = sc
            best = sn
    if best is None or best_sc < -0.15:
        return None
    ids = list(best.token_ids)[:max_payoff_tokens]
    while ids and pivot and ids[0] == pivot[0]:
        ids = ids[1:]
    if len(ids) < 5:
        return None
    return StylePayoffPlan(
        pivot=pivot,
        payoff=tuple(ids),
        pivot_text=pivot_text,
        payoff_text=best.text[:160],
        score=float(best_sc),
    )


def append_payoff_segments(
    segments: list[list[int]],
    entity_ids: tuple[int, ...],
    payoff: StylePayoffPlan,
    *,
    vocab=None,
    setup_segs: int = 1,
    setup_max_tokens: int = 48,
    keep_tail_facts: bool = False,
) -> tuple[list[list[int]], list[int], frozenset[int], frozenset[int]]:
    """Setup → pivot → payoff. Trả (segs, flat, style_segs, pivot_lock_segs)."""
    del entity_ids
    if not segments:
        segs = [list(payoff.pivot), list(payoff.payoff)]
        style = frozenset({0, 1})
        pivot_lock = frozenset({0})
    else:
        n = max(1, min(setup_segs, len(segments)))
        head = [list(s) for s in segments[:n]]
        if vocab is not None and head:
            head[0] = _first_sentence_ids(vocab, head[0], max_tokens=setup_max_tokens)
        tail = [list(s) for s in segments[n:]] if keep_tail_facts else []
        segs = head + [list(payoff.pivot), list(payoff.payoff)] + tail
        style = frozenset({n, n + 1})
        pivot_lock = frozenset({n})
    flat: list[int] = []
    for s in segs:
        flat.extend(s)
    return segs, flat, style, pivot_lock
