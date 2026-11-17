"""
Style inventory từ thân styled corpus (label-conditioned) — 2026-07-23.

Graph CO_ACT «hài hước» → neighbor giải trí (`cốt/kịch/phim`), không phải marker
giọng trong body. Cách Luật-0: đếm n-gram thân ĐÁP có điều kiện nhãn HOW trùng
seed prompt, xếp theo distinctive vs mọi thân (tf × IDF thân).

Không hardcode danh sách marker.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from nn_aivie.hca.style_readout import _core, _is_artifact, style_seeds_from_prompt
from nn_aivie.hca.objcache import cache_get, cache_put

# 1k: "Hỏi: …? Đáp: …" cùng một dòng — không neo ^/\n.
_DAP = re.compile(r"Đáp\s*:\s*", re.IGNORECASE)
_COLON_LABEL = re.compile(
    r"^(?:Viết|Kể|Hãy viết|Hãy kể)[^:]{0,80}:\s*",
    re.IGNORECASE,
)

# cache theo (vocab_id, paths_key)
_INDEX_CACHE: dict[tuple[int, str], "StyleBodyIndex"] = {}


def _split_label_body(text: str) -> tuple[str, str]:
    m = _DAP.search(text)
    if m:
        return text[: m.start()], text[m.end() :]
    m2 = _COLON_LABEL.match(text)
    if m2:
        return text[: m2.end()], text[m2.end() :]
    if text.lower().startswith("hỏi:"):
        return text, text
    return text, text


@dataclass
class StyleBodyIndex:
    """seed_core → đếm unigram/bigram/trigram trong thân; df toàn cục theo số thân."""

    n_bodies: int = 0
    seed_uni: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    seed_bi: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    seed_tri: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    uni_df: Counter = field(default_factory=Counter)
    bi_df: Counter = field(default_factory=Counter)
    tri_df: Counter = field(default_factory=Counter)


def _iter_styled_texts(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("text") if isinstance(obj, dict) else None
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
    return out


def build_style_body_index(
    state,
    paths: list[Path],
    *,
    gidf_min: float = 4.0,
    max_seeds: int = 3,
) -> StyleBodyIndex:
    """Xây index từ JSONL styled; cần vocab + parents (gidf) của checkpoint."""
    vocab, ps = state.vocab, state.parents
    idx = StyleBodyIndex()
    texts = _iter_styled_texts(paths)
    for text in texts:
        label_txt, body_txt = _split_label_body(text)
        if not body_txt.strip() or body_txt.strip() == label_txt.strip():
            continue
        label_ids = list(vocab.encode_prompt(label_txt))
        body_ids = list(vocab.encode_prompt(body_txt))
        seeds = style_seeds_from_prompt(
            label_ids, vocab, ps, gidf_min=gidf_min, max_seeds=max_seeds,
        )
        if not seeds:
            seeds = style_seeds_from_prompt(
                label_ids, vocab, ps, gidf_min=max(3.0, gidf_min - 1.0), max_seeds=4,
            )
        if not seeds or len(body_ids) < 2:
            continue

        # body tokens: surface thường. Uni chặt (gidf); n-gram lỏng hơn để giữ frame
        # «hệt như / thế nhưng» (như/nhưng gidf thấp — eval 2026-07-23).
        keep_uni: list[int] = []
        keep_ng: list[int] = []
        seen_uni: set[int] = set()
        for t in body_ids:
            if not vocab.is_surface(t):
                continue
            if vocab.cap_class(t) >= 1:
                continue
            tok = vocab.token_of(t)
            if _is_artifact(tok) or len(_core(tok)) <= 1:
                continue
            # Giữ cả mảnh BPE không Ġ (thế/hệt đầu cụm) — cần cho frame «thế nhưng».
            g = float(ps._gidf(t))  # noqa: SLF001
            if g >= max(2.5, gidf_min - 1.5):
                keep_ng.append(t)
            if g >= gidf_min and (tok.startswith("Ġ") or tok.startswith("Ċ")):
                keep_uni.append(t)
                seen_uni.add(t)

        if len(keep_ng) < 2:
            continue

        idx.n_bodies += 1
        for t in seen_uni:
            idx.uni_df[t] += 1
        seen_bi: set[tuple[int, int]] = set()
        for a, b in zip(keep_ng, keep_ng[1:]):
            if a == b:
                continue
            seen_bi.add((a, b))
        for pair in seen_bi:
            idx.bi_df[pair] += 1
        seen_tri: set[tuple[int, int, int]] = set()
        for a, b, c in zip(keep_ng, keep_ng[1:], keep_ng[2:]):
            if len({a, b, c}) < 2:
                continue
            seen_tri.add((a, b, c))
        for trip in seen_tri:
            idx.tri_df[trip] += 1

        for s in seeds:
            core = _core(vocab.token_of(s)).lower()
            if not core:
                continue
            cu = idx.seed_uni[core]
            cb = idx.seed_bi[core]
            ct = idx.seed_tri[core]
            for t in keep_uni:
                cu[t] += 1
            for a, b in zip(keep_ng, keep_ng[1:]):
                if a != b:
                    cb[(a, b)] += 1
            for a, b, c in zip(keep_ng, keep_ng[1:], keep_ng[2:]):
                if len({a, b, c}) >= 2:
                    ct[(a, b, c)] += 1
    return idx


def get_style_body_index(
    state,
    paths: list[Path] | None = None,
    *,
    gidf_min: float = 4.0,
) -> StyleBodyIndex | None:
    """Lazy cache theo VOCAB (weakref, `objcache`) + paths."""
    if paths is None:
        root = Path(__file__).resolve().parents[3]
        paths = [
            root / "data/text_corpus/styled_humor_history.jsonl",
            root / "data/text_corpus/styled_sft_1k.jsonl",
            root / "data/text_corpus/styled_sft.jsonl",
        ]
    paths = [Path(p) for p in paths if Path(p).is_file()]
    if not paths:
        return None
    _sub = ("v6glue", "|".join(str(p.resolve()) for p in paths))
    hit = cache_get(_INDEX_CACHE, state.vocab, _sub)
    if hit is not None:
        return hit
    idx = build_style_body_index(state, paths, gidf_min=gidf_min)
    if idx.n_bodies <= 0:
        return None
    cache_put(_INDEX_CACHE, state.vocab, idx, _sub)
    return idx


def _idf(n_bodies: int, df: int) -> float:
    return math.log(1.0 + n_bodies / (1.0 + df))


def ranked_corpus_inventory(
    state,
    query_ids: list[int],
    *,
    top_k: int = 24,
    gidf_min: float = 4.0,
    token_gidf_min: float = 5.5,
    max_seeds: int = 2,
    paths: list[Path] | None = None,
    index: StyleBodyIndex | None = None,
) -> list[tuple[int, float]]:
    """[(token_id, score)] distinctive unigram trong thân khớp seed HOW.

    token_gidf_min: sàn gidf ứng viên (5.5) — `mua/phút` ~4.2–4.6; `hệt/xịn` ~7–8.
    """
    vocab, ps = state.vocab, state.parents
    if ps is None:
        return []
    seeds = style_seeds_from_prompt(
        query_ids, vocab, ps, gidf_min=gidf_min, max_seeds=max_seeds,
    )
    if not seeds:
        return []
    idx = index or get_style_body_index(state, paths, gidf_min=gidf_min)
    if idx is None or idx.n_bodies <= 0:
        return []

    seed_cores = {_core(vocab.token_of(s)).lower() for s in seeds}
    agg: Counter = Counter()
    for core in seed_cores:
        for tid, c in idx.seed_uni.get(core, {}).items():
            agg[tid] += c
    if not agg:
        return []

    scored: list[tuple[float, int]] = []
    for tid, tf in agg.items():
        if tid in set(seeds):
            continue
        tok = vocab.token_of(tid)
        if _is_artifact(tok) or vocab.cap_class(tid) >= 1:
            continue
        # bỏ mảnh BPE giữa từ (ll/lay từ Unfriend…) — chỉ word-start
        if not (tok.startswith("Ġ") or tok.startswith("Ċ")):
            continue
        if _core(tok).lower() in seed_cores:
            continue
        if len(_core(tok)) <= 2 and _core(tok).isascii():
            continue
        g = float(ps._gidf(tid))  # noqa: SLF001
        if g < token_gidf_min:
            continue
        df = int(idx.uni_df.get(tid, 0))
        sc = float(tf) * _idf(idx.n_bodies, df) * math.log(1.0 + df) * g
        if sc > 0:
            scored.append((sc, tid))
    scored.sort(reverse=True)
    return [(j, sc) for sc, j in scored[:top_k]]


def ranked_corpus_phrases(
    state,
    query_ids: list[int],
    *,
    top_k: int = 12,
    gidf_min: float = 4.0,
    max_seeds: int = 2,
    paths: list[Path] | None = None,
    index: StyleBodyIndex | None = None,
    min_seed_df: int = 2,
    token_gidf_min: float = 5.5,
) -> list[tuple[tuple[int, int], float]]:
    """[((u,v), score)] bigram thân distinctive + tái xuất hiện trong ≥min_seed_df thân style.

    min_seed_df≥2: bỏ one-shot content (`cắm cọc` trong 1 bài lịch sử hài).
    """
    vocab, ps = state.vocab, state.parents
    if ps is None:
        return []
    seeds = style_seeds_from_prompt(
        query_ids, vocab, ps, gidf_min=gidf_min, max_seeds=max_seeds,
    )
    if not seeds:
        return []
    idx = index or get_style_body_index(state, paths, gidf_min=gidf_min)
    if idx is None or idx.n_bodies <= 0:
        return []

    seed_cores = {_core(vocab.token_of(s)).lower() for s in seeds}
    # df trong các thân khớp seed (không chỉ tf cộng dồn)
    agg: Counter = Counter()
    for core in seed_cores:
        for pair, c in idx.seed_bi.get(core, {}).items():
            agg[pair] += c
    if not agg:
        return []

    scored: list[tuple[float, tuple[int, int]]] = []
    for (u, v), tf in agg.items():
        if u == v:
            continue
        if vocab.cap_class(u) >= 1 or vocab.cap_class(v) >= 1:
            continue
        tu, tv = vocab.token_of(u), vocab.token_of(v)
        if not (tu.startswith("Ġ") or tu.startswith("Ċ")):
            continue
        if not (tv.startswith("Ġ") or tv.startswith("Ċ")):
            continue
        cu = _core(tu).lower()
        cv = _core(tv).lower()
        if cu in seed_cores or cv in seed_cores:
            continue
        if _is_artifact(tu) or _is_artifact(tv):
            continue
        sdf = int(idx.bi_df.get((u, v), 0))
        if sdf < min_seed_df:
            continue
        gu = float(ps._gidf(u))  # noqa: SLF001
        gv = float(ps._gidf(v))  # noqa: SLF001
        if gu < token_gidf_min or gv < token_gidf_min:
            continue
        sc = float(tf) * _idf(idx.n_bodies, sdf) * math.log(1.0 + sdf) * (0.5 * (gu + gv))
        if sc > 0:
            scored.append((sc, (u, v)))
    scored.sort(reverse=True)
    return [(pair, sc) for sc, pair in scored[:top_k]]


def _alliterative(cores: list[str]) -> bool:
    """Cùng chữ cái đầu (ngơ/ngác, phũ/phàng) — tín hiệu cụm tu từ, không hardcode từ."""
    letters = [c[0].lower() for c in cores if c]
    if len(letters) < 2:
        return False
    return letters[0] == letters[1]


def _content_leak_light(state, token_id: int, controls: list[int]) -> float:
    from nn_aivie.core.edge_kinds import EdgeKind
    if not controls:
        return 0.0
    co = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
    s = 0.0
    for c in controls:
        s += float(co.get(c, token_id)) + float(co.get(token_id, c))
    return s


def ranked_discourse_ngrams(
    state,
    query_ids: list[int],
    *,
    top_k: int = 12,
    gidf_min: float = 4.0,
    max_seeds: int = 2,
    paths: list[Path] | None = None,
    index: StyleBodyIndex | None = None,
    min_df: int = 2,
    token_gidf_min: float = 5.0,
) -> list[tuple[tuple[int, ...], float]]:
    """Cụm discourse theo eval_humor_lexicon (Luật 0).

    Ưu tiên: (A) điệp âm  (B) frame df cao («hệt như/thế nhưng»)  (C) W_seq khóa.
    Frame: cho phép 1 token gidf thấp (như/nhưng) nếu đối tác gidf cao + df≥8.
    """
    from nn_aivie.core.edge_kinds import EdgeKind

    vocab, ps = state.vocab, state.parents
    if ps is None:
        return []
    seeds = style_seeds_from_prompt(
        query_ids, vocab, ps, gidf_min=gidf_min, max_seeds=max_seeds,
    )
    if not seeds:
        return []
    idx = index or get_style_body_index(state, paths, gidf_min=gidf_min)
    if idx is None or idx.n_bodies <= 0:
        return []

    seed_cores = {_core(vocab.token_of(s)).lower() for s in seeds}
    controls: list[int] = []
    for w in ("Trần Hưng Đạo", "Ngô Quyền", "Hà Nội", "Nhà Lý"):
        controls.extend(vocab.encode_prompt(w))
    controls = list(dict.fromkeys(c for c in controls if c not in set(query_ids)))
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)

    def _surface_ok(tid: int) -> bool:
        if vocab.cap_class(tid) >= 1:
            return False
        tok = vocab.token_of(tid)
        if _is_artifact(tok):
            return False
        c = _core(tok)
        if len(c) <= 1 or _core(tok).lower() in seed_cores:
            return False
        # Cho phép thiếu Ġ (BPE «thế» trong thế nhưng).
        if not (tok.startswith("Ġ") or tok.startswith("Ċ")):
            if not any(ch.isalpha() for ch in c):
                return False
        return True

    def _pair_ok(u: int, v: int, df: int) -> bool:
        if not (_surface_ok(u) and _surface_ok(v)):
            return False
        gu = float(ps._gidf(u))  # noqa: SLF001
        gv = float(ps._gidf(v))  # noqa: SLF001
        # Frame dày (thế nhưng df~54): cả hai có thể gidf thấp — eval humor lexicon.
        if df >= 12 and min(gu, gv) >= 2.4:
            return True
        if df >= 8 and max(gu, gv) >= token_gidf_min and min(gu, gv) >= 2.5:
            return True
        return gu >= token_gidf_min and gv >= token_gidf_min

    scored: list[tuple[float, tuple[int, ...], int]] = []  # sc, unit, tier

    agg_bi: Counter = Counter()
    for core in seed_cores:
        for pair, c in idx.seed_bi.get(core, {}).items():
            agg_bi[pair] += c
    for (u, v), tf in agg_bi.items():
        if u == v:
            continue
        df = int(idx.bi_df.get((u, v), 0))
        if df < min_df or not _pair_ok(u, v, df):
            continue
        gu = float(ps._gidf(u))  # noqa: SLF001
        gv = float(ps._gidf(v))  # noqa: SLF001
        w_seq = float(seq.get(u, v))
        cores = [_core(vocab.token_of(u)), _core(vocab.token_of(v))]
        leak = _content_leak_light(state, u, controls) + _content_leak_light(state, v, controls)
        sc = (
            float(tf) * math.log(1.0 + df) * max(0.5 * (gu + gv), 2.5)
            * math.log(1.0 + w_seq)
            / (1.0 + 0.05 * leak)
        )
        allit = _alliterative(cores)
        strong_allit = allit and max(gu, gv) >= 5.5
        glue_frame = (
            df >= 15
            and min(gu, gv) <= 3.5
            and max(gu, gv) < 5.2
        )  # thế nhưng / rốt cuộc — loại tiết kiệm (min≈4) & điện thoại
        if strong_allit:
            sc *= 2.8
        elif allit:
            sc *= 1.15
        if glue_frame:
            sc *= 2.8
        elif strong_allit:
            pass  # already boosted
        elif df >= 10 and min(gu, gv) >= 4.0:
            sc *= 0.35  # compound topic (tiết kiệm/điện thoại/cà phê)
        elif df >= 8:
            sc *= 1.2
        # tier: 0=glue frame, 1=redup mạnh, 2=khác
        if glue_frame:
            tier = 0
        elif strong_allit:
            tier = 1
        else:
            tier = 2
        if sc > 0:
            scored.append((sc, (u, v), tier))

    agg_tri: Counter = Counter()
    for core in seed_cores:
        for trip, c in idx.seed_tri.get(core, {}).items():
            agg_tri[trip] += c
    for (u, v, w), tf in agg_tri.items():
        if not (_surface_ok(u) and _surface_ok(v) and _surface_ok(w)):
            continue
        df = int(idx.tri_df.get((u, v, w), 0))
        if df < min_df:
            continue
        gs = [float(ps._gidf(t)) for t in (u, v, w)]  # noqa: SLF001
        if max(gs) < token_gidf_min or min(gs) < 2.5:
            continue
        w_seq = float(seq.get(u, v)) * float(seq.get(v, w))
        cores = [_core(vocab.token_of(t)) for t in (u, v, w)]
        leak = sum(_content_leak_light(state, t, controls) for t in (u, v, w))
        sc = (
            float(tf) * math.log(1.0 + df) * (sum(gs) / 3.0)
            * math.log(1.0 + w_seq)
            / (1.0 + 0.05 * leak)
        ) * 1.15
        allit = _alliterative(cores[:2]) or _alliterative(cores[1:])
        if allit:
            sc *= 1.4
        if df >= 8:
            sc *= 1.8
        tier = 0 if allit else (1 if df >= 8 else 2)
        if sc > 0:
            scored.append((sc, (u, v, w), tier))

    scored.sort(key=lambda x: (x[2], -x[0]))
    seen: set[tuple[str, ...]] = set()
    out: list[tuple[tuple[int, ...], float]] = []
    # Ưu tiên frame/redup; chỉ lấy compound topic nếu thiếu.
    for allow_tier_max in (1, 2):
        for sc, unit, tier in scored:
            if tier > allow_tier_max:
                continue
            key = tuple(_core(vocab.token_of(t)).lower() for t in unit)
            if key in seen:
                continue
            seen.add(key)
            out.append((unit, sc))
            if len(out) >= top_k:
                return out
    return out
