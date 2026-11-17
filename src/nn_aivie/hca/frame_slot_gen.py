"""TK_050 P113/P114 — khung–đa-khe: rank cha → emit → điền khe giao.

Cấu hình sống (P112): licence=giao · slot_rank=bam · min_gain=1 · max_thay=2 ·
không CO-graph · SYNONYM bake · cấm khe `số`.

P114: ``decode_unit=thuc_the`` — điền theo từ/entity hub (thay cả cụm tên riêng).

Tách từ `scripts/gen_three_arm.py` để CLI / `generate(mode=frame_da_khe)` dùng chung luật.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.hca.context_positional import is_structural_token
from nn_aivie.hca.readout import decode_surface


@dataclass
class UnitSpan:
    """Một đơn vị thuc_the trên emit (token đơn hoặc cụm tên riêng)."""

    tokens: tuple[int, ...]
    start: int
    end: int
    rep: int


@dataclass
class FrameSlotConfig:
    """Defaults = P112 cổng (4) ĐẠT."""

    licence: str = "giao"  # bo_ba | lop | giao
    slot_rank: str = "bam"  # bam | ncand
    min_gain: int = 1
    max_thay: int = 2
    no_co_graph: bool = True
    decode_unit: str = "token"  # token | thuc_the (P114)
    # P115: siết chất lượng đa-khe
    era_filter: bool = False
    soft_giao_single_only: bool = False  # soft lop chỉ cho đơn vị 1-token
    hybrid_if_zero: bool = False  # token fallback chỉ khi 0 thay đơn vị
    second_min_gain: int = 0  # khe thứ 2+ cần gain ≥ ngưỡng (0 = dùng min_gain)
    topk_lop: int = 8
    min_share: int = 3
    gidf_min: float = 4.0
    top_n: int = 8
    seed: int = 42
    label_map: str = "out/token_labels_v12.json"
    ws_model: str = "out/models/ws_crf_uts_full"
    require_synonym: bool = True  # P112: cần bake SYNONYM


# Triều / niên đại hay bị nhảy khi thay khe (TK_047 §17 · TK_050 P115).
_ERA_MARKERS: tuple[str, ...] = (
    "Lý", "Trần", "Lê", "Nguyễn", "Đinh", "Ngô", "Hồ", "Mạc", "Tây Sơn",
    "Đường", "Tống", "Minh", "Nguyên", "Thanh", "Tần", "Hán", "Đại Việt",
    "Thăng Long", "Huế", "Phú Xuân",
)


def _syllables(text: str) -> list[str]:
    """Tách âm tiết theo khoảng trắng, bỏ dấu câu ở hai biên mỗi âm tiết."""
    out: list[str] = []
    for raw in str(text).replace("Ġ", " ").split():
        syl = raw.strip(".,;:!?…\"'()[]{}«»–—-")
        if syl:
            out.append(syl)
    return out


def eras_in_text(text: str) -> set[str]:
    """Niên đại xuất hiện trong `text`, khớp theo ÂM TIẾT (không phải chuỗi con).

    Khớp chuỗi con làm `Lê` trúng trong `Lên`, `Hồ` trong `Hồng`, `Ngô` trong
    `Ngôn` — dương tính giả có hệ thống ở bộ lọc niên đại (`compose_2cha`).
    Vẫn phân biệt hoa/thường: `hồ` (danh từ chung) không phải triều `Hồ`.

    ⚠️ `_ERA_MARKERS` là danh sách viết tay (nợ Luật 0). Khớp-âm-tiết chỉ sửa
    lỗi chuỗi con; nó KHÔNG sửa được đa nghĩa cùng mặt chữ (`Lý` triều đại vs
    `Lý` đầu câu của `lý thuyết`) — đó là trần của chính cách tiếp cận danh sách.
    """
    syls = _syllables(text)
    if not syls:
        return set()
    found: set[str] = set()
    for e in _ERA_MARKERS:
        parts = e.split()
        if not parts:
            continue
        n = len(parts)
        for i in range(len(syls) - n + 1):
            if syls[i : i + n] == parts:
                found.add(e)
                break
    return found


def eras_of_rep(
    state: GraphState,
    rep: int,
    rep_tokens: dict[int, tuple[int, ...]],
) -> set[str]:
    toks = rep_tokens.get(rep)
    if toks:
        return eras_in_text(_unit_surface(state.vocab, toks))
    tok = state.vocab.token_of(rep) if hasattr(state.vocab, "token_of") else str(rep)
    if str(tok).startswith("Entity:"):
        return eras_in_text(str(tok)[7:])
    return eras_in_text(str(tok).replace("Ġ", " "))


def era_ok(
    theme_eras: set[str],
    old_eras: set[str],
    cand_eras: set[str],
) -> bool:
    """True nếu ứng viên không nhảy niên đại/triều so với đề + khe gốc."""
    if not cand_eras:
        return True
    allowed = theme_eras | old_eras
    if not allowed:
        return True  # đề/gốc không neo triều → không cấm
    return bool(cand_eras & allowed)


@dataclass
class FrameSlotIndex:
    keep: np.ndarray
    so: set[int]
    tri: dict[tuple[int, int], set[int]]
    co: dict[int, set[int]]
    lop_nb: dict[int, list[int]]
    ws_model: Any = None
    n_syn_edges: int = 0
    n_co_graph: int = 0
    pn: set[int] = field(default_factory=set)
    unit_tri: dict[tuple[int, int], set[int]] | None = None
    unit_lop: dict[int, list[int]] | None = None
    rep_tokens: dict[int, tuple[int, ...]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def load_so_ban(label_map: str | Path) -> set[int]:
    p = Path(label_map)
    so: set[int] = set()
    if not p.exists():
        return so
    m = json.loads(p.read_text(encoding="utf-8")).get("tu_loai") or {}
    for ts, d in m.items():
        if d and max(d, key=d.get) == "số":
            so.add(int(ts))
    return so


def load_proper_noun_tokens(label_map: str | Path) -> set[int]:
    p = Path(label_map)
    pn: set[int] = set()
    if not p.exists():
        return pn
    m = json.loads(p.read_text(encoding="utf-8")).get("tu_loai") or {}
    for ts, d in m.items():
        if d and max(d, key=d.get) == "tên riêng":
            pn.add(int(ts))
    return pn


def _unit_surface(v, unit: tuple[int, ...]) -> str:
    parts = [(v.token_of(t) if hasattr(v, "token_of") else str(t)) for t in unit]
    return "".join(parts).replace("Ġ", " ").strip()


def rep_for_unit(
    state: GraphState,
    unit: tuple[int, ...],
    *,
    rep_tokens: dict[int, tuple[int, ...]],
) -> int:
    """Đại diện đơn vị: token đơn hoặc hub Entity: (P106 bake)."""
    if len(unit) == 1:
        return int(unit[0])
    v = state.vocab
    title = _unit_surface(v, unit)
    if not title:
        return int(unit[0])
    eid = v.entity_id(title)
    rep_tokens.setdefault(eid, unit)
    return int(eid)


def tokens_for_rep(
    state: GraphState,
    rep: int,
    rep_tokens: dict[int, tuple[int, ...]],
    keep: np.ndarray,
) -> tuple[int, ...]:
    if rep in rep_tokens:
        return rep_tokens[rep]
    v = state.vocab
    tok = v.token_of(rep) if hasattr(v, "token_of") else ""
    if not str(tok).startswith("Entity:"):
        return (int(rep),)
    co_layer = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
    row = co_layer.row(rep) or {}
    members = sorted(
        (int(j) for j in row if 0 <= int(j) < len(keep) and keep[int(j)]),
        key=lambda t: (int(t),),
    )
    if members:
        rep_tokens[rep] = tuple(members)
        return rep_tokens[rep]
    return (int(rep),)


def segment_emit_thuc_the(
    state: GraphState,
    ids: list[int],
    wsm,
    pn: set[int],
) -> list[UnitSpan]:
    """Tách emit → đơn vị thuc_the (WS-CRF + ghép tên riêng liền)."""
    rep_cache: dict[int, tuple[int, ...]] = {}
    if not ids:
        return []
    if len(ids) == 1:
        u = (int(ids[0]),)
        return [UnitSpan(tokens=u, start=0, end=1, rep=rep_for_unit(state, u, rep_tokens=rep_cache))]

    whole = decode_surface(state, ids)
    syl = whole.split()
    if not syl:
        return []
    tags = wsm.tag(syl)
    if len(tags) != len(syl):
        return []

    si_of: list[int | None] = []
    for k in range(len(ids)):
        pre = decode_surface(state, ids[: k + 1])
        si_of.append(
            len(pre.split()) - 1
            if whole.startswith(pre) and not whole[len(pre):][:1].isalnum()
            else None
        )
    w_of_syl: list[int] = []
    cur = -1
    for tg in tags:
        if tg != "I-W" or cur < 0:
            cur += 1
        w_of_syl.append(cur)
    word_units: list[list[int]] = [[] for _ in range(cur + 1)]
    word_start_tok: list[int | None] = [None] * (cur + 1)
    for k, t in enumerate(ids):
        si = si_of[k]
        if si is not None and 0 <= si < len(w_of_syl):
            wi = w_of_syl[si]
            if word_start_tok[wi] is None:
                word_start_tok[wi] = k
            word_units[wi].append(int(t))

    raw: list[tuple[int, ...]] = []
    spans: list[tuple[int, int]] = []
    for wi, wu in enumerate(word_units):
        if not wu:
            continue
        start = word_start_tok[wi] if word_start_tok[wi] is not None else 0
        raw.append(tuple(wu))
        spans.append((start, start + len(wu)))

    if not pn:
        return [
            UnitSpan(tokens=u, start=s, end=e, rep=rep_for_unit(state, u, rep_tokens=rep_cache))
            for u, (s, e) in zip(raw, spans)
        ]

    is_pn = lambda u: bool(u) and all(t in pn for t in u)
    merged_u: list[tuple[int, ...]] = []
    merged_s: list[tuple[int, int]] = []
    i = 0
    while i < len(raw):
        if is_pn(raw[i]):
            j = i + 1
            while j < len(raw) and is_pn(raw[j]):
                j += 1
            toks = tuple(t for u in raw[i:j] for t in u)
            merged_u.append(toks)
            merged_s.append((spans[i][0], spans[j - 1][1]))
            i = j
        else:
            merged_u.append(raw[i])
            merged_s.append(spans[i])
            i += 1

    return [
        UnitSpan(tokens=u, start=s, end=e, rep=rep_for_unit(state, u, rep_tokens=rep_cache))
        for u, (s, e) in zip(merged_u, merged_s)
    ]


def ws_tags_ok(wsm, syl: list[str]) -> bool:
    if len(syl) < 2:
        return True
    tags = wsm.tag(syl)
    if len(tags) != len(syl):
        return False
    for i, tg in enumerate(tags):
        if tg == "I-W" and (i == 0 or tags[i - 1] not in ("B-W", "I-W")):
            return False
    return True


def build_unit_slot_maps(
    state: GraphState,
    keep: np.ndarray,
    cfg: FrameSlotConfig,
    wsm,
    pn: set[int],
    *,
    logger=None,
) -> tuple[dict[tuple[int, int], set[int]], dict[int, list[int]], dict[int, tuple[int, ...]]]:
    """Dựng tri/lop ở mức rep (token | Entity hub) từ toàn bộ cha."""
    ps, v = state.parents, state.vocab
    syn = state.adjacency.layer(EdgeKind.SYNONYM)
    nv = len(v)
    rep_tokens: dict[int, tuple[int, ...]] = {}
    unit_tri: dict[tuple[int, int], set[int]] = {}
    n_parents = len(ps.parents)

    for pi, rec in enumerate(ps.parents):
        if pi and pi % 2000 == 0 and logger:
            logger.log(f"  unit index {pi:,}/{n_parents:,}")
        ids = [int(t) for t in rec.emit_ids]
        units = segment_emit_thuc_the(state, ids, wsm, pn)
        if len(units) < 3:
            continue
        reps = [u.rep for u in units]
        for u in units:
            rep_tokens.setdefault(u.rep, u.tokens)
        for i in range(1, len(reps) - 1):
            r = reps[i]
            if not _rep_keep(r, keep, v):
                continue
            unit_tri.setdefault((reps[i - 1], reps[i + 1]), set()).add(r)

    unit_lop: dict[int, list[int]] = {}
    reps_seen = set(rep_tokens) | set(r for rs in unit_tri.values() for r in rs)
    for r in reps_seen:
        if not _rep_keep(r, keep, v):
            continue
        row = syn.row(r)
        if not row:
            continue
        ranked = sorted(
            ((float(w), int(j)) for j, w in row.items()
             if 0 <= int(j) < nv and int(j) != r and _rep_keep(int(j), keep, v)),
            reverse=True,
        )
        if ranked:
            unit_lop[r] = [j for _w, j in ranked[: cfg.topk_lop]]

    return unit_tri, unit_lop, rep_tokens


def _rep_keep(rep: int, keep: np.ndarray, v) -> bool:
    if 0 <= rep < len(keep) and keep[rep]:
        return True
    tok = v.token_of(rep) if hasattr(v, "token_of") else ""
    return str(tok).startswith("Entity:")


def bam_rep(
    rep: int,
    theme: set[int],
    co: dict[int, set[int]],
    rep_tokens: dict[int, tuple[int, ...]],
) -> int:
    toks = rep_tokens.get(rep, (rep,))
    s: set[int] = set()
    for t in toks:
        s |= co.get(int(t), set())
    return len(s & theme)


def build_keep_mask(state: GraphState, *, gidf_min: float = 4.0) -> np.ndarray:
    v, ps = state.vocab, state.parents
    n = len(v)
    keep = np.zeros(n, dtype=bool)
    for t in range(n):
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            if float(ps._gidf(t)) >= gidf_min:  # noqa: SLF001
                keep[t] = True
        except Exception:
            continue
    return keep


def build_frame_slot_index(
    state: GraphState,
    cfg: FrameSlotConfig | None = None,
    *,
    root: Path | None = None,
    logger=None,
) -> FrameSlotIndex:
    """Dựng tri / co / lop_nb một lần — tái dùng cho nhiều đề."""
    cfg = cfg or FrameSlotConfig()
    root = root or Path.cwd()
    ps, v = state.parents, state.vocab
    if ps is None or len(ps) == 0:
        raise RuntimeError("frame_slot_gen cần ParentStore")

    keep = build_keep_mask(state, gidf_min=cfg.gidf_min)
    so = load_so_ban(root / cfg.label_map)
    pn = load_proper_noun_tokens(root / cfg.label_map)
    nv = len(v)
    tri: dict[tuple[int, int], set[int]] = {}
    co: dict[int, set[int]] = {}
    for rec in ps.parents:
        e = [int(t) for t in rec.emit_ids]
        for i in range(1, len(e) - 1):
            if 0 <= e[i] < nv and keep[e[i]]:
                tri.setdefault((e[i - 1], e[i + 1]), set()).add(e[i])
        u = {t for t in e if 0 <= t < nv and keep[t]}
        for t in u:
            co.setdefault(t, set()).update(u)

    n_co_g = 0
    if not cfg.no_co_graph:
        co_layer = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
        for t in list(co.keys()):
            row = co_layer.row(t)
            if not row:
                continue
            top = sorted(
                ((float(w), int(j)) for j, w in row.items()
                 if float(w) >= 0.05 and 0 <= int(j) < nv and keep[int(j)] and int(j) != t),
                reverse=True,
            )[:16]
            for _w, j in top:
                if j not in co[t]:
                    co[t].add(j)
                    n_co_g += 1

    syn = state.adjacency.layer(EdgeKind.SYNONYM)
    if getattr(syn, "_frozen", False):
        n_syn = int(len(getattr(syn, "_indices", ())))
    else:
        n_syn = len(getattr(syn, "_data", {}) or {})

    lop_nb: dict[int, list[int]] = {}
    if n_syn > 0:
        for t in range(nv):
            if not keep[t]:
                continue
            row = syn.row(t)
            if not row:
                continue
            ranked = sorted(
                ((float(w), int(j)) for j, w in row.items()
                 if 0 <= int(j) < nv and keep[int(j)] and int(j) != t),
                reverse=True,
            )
            if ranked:
                lop_nb[t] = [j for _w, j in ranked[: cfg.topk_lop]]
        if logger:
            logger.log(f"lớp thay thế từ SYNONYM bake: {len(lop_nb):,} token ({n_syn:,} cạnh)")
    elif cfg.require_synonym:
        raise RuntimeError(
            "frame_slot_gen cần cạnh SYNONYM (bake P106) — "
            "checkpoints/network_v12_subst.pkl"
        )
    else:
        ctx_all: dict[int, set] = defaultdict(set)
        for rec in ps.parents:
            e = [int(t) for t in rec.emit_ids]
            for i in range(1, len(e) - 1):
                if 0 <= e[i] < nv and keep[e[i]]:
                    ctx_all[e[i]].add((e[i - 1], e[i + 1]))
        ctx_all = {t: c for t, c in ctx_all.items() if len(c) >= 3}
        inv: dict[tuple, list[int]] = defaultdict(list)
        for t, cs in ctx_all.items():
            for c in cs:
                inv[c].append(t)
        for x, cs in ctx_all.items():
            sh: dict[int, int] = defaultdict(int)
            for c in cs:
                for y in inv[c]:
                    if y != x:
                        sh[y] += 1
            nx = len(cs) ** 0.5
            sc = sorted(
                ((m / (nx * len(ctx_all[y]) ** 0.5), y)
                 for y, m in sh.items() if m >= cfg.min_share),
                reverse=True,
            )
            if sc:
                lop_nb[x] = [y for _s, y in sc[: cfg.topk_lop]]
        if logger:
            logger.log(f"lớp thay thế (tính lại): {len(lop_nb):,}")

    from nn_aivie.ws_crf.ws_model import WSCRFModel  # noqa: PLC0415
    wsm = WSCRFModel.load(root / cfg.ws_model)

    unit_tri = unit_lop = rep_tokens = None
    if cfg.decode_unit == "thuc_the":
        if not pn:
            raise RuntimeError(
                "decode_unit=thuc_the cần nhãn `tên riêng` trong label_map"
            )
        if logger:
            logger.log(f"token tên riêng: {len(pn):,} · dựng chỉ mục đơn vị thuc_the…")
        unit_tri, unit_lop, rep_tokens = build_unit_slot_maps(
            state, keep, cfg, wsm, pn, logger=logger)
        if logger:
            logger.log(
                f"đơn vị: bối cảnh {len(unit_tri):,} · lớp {len(unit_lop):,} · "
                f"rep {len(rep_tokens):,}"
            )

    if logger:
        logger.log(
            f"khuôn: bối cảnh {len(tri):,} · đồng hiện {len(co):,} · "
            f"+CO graph {n_co_g:,}" + (" (tắt)" if cfg.no_co_graph else "")
            + f" · cấm số {len(so):,}"
        )

    return FrameSlotIndex(
        keep=keep, so=so, tri=tri, co=co, lop_nb=lop_nb, ws_model=wsm,
        n_syn_edges=n_syn, n_co_graph=n_co_g, pn=pn,
        unit_tri=unit_tri, unit_lop=unit_lop, rep_tokens=rep_tokens,
        meta={
            "licence_default": cfg.licence,
            "n_lop": len(lop_nb),
            "decode_unit": cfg.decode_unit,
            "n_unit_tri": len(unit_tri or {}),
            "n_unit_lop": len(unit_lop or {}),
        },
    )


def breaks_word(state: GraphState, wsm, ids: list[int], k: int) -> bool:
    """True nếu thay ở k làm vỡ từ (WS-CRF)."""
    txt = decode_surface(state, ids)
    syl = txt.split()
    if len(syl) < 2:
        return False
    tags = wsm.tag(syl)
    if len(tags) != len(syl):
        return False
    pre = decode_surface(state, ids[: k + 1])
    if not txt.startswith(pre) or txt[len(pre):][:1].isalnum():
        return True
    si = len(pre.split()) - 1
    if not (0 <= si < len(tags)):
        return True
    return tags[si] == "I-W" or (si + 1 < len(tags) and tags[si + 1] == "I-W")


def rank_other_parent(
    state: GraphState,
    prompt_ids: list[int],
    *,
    skip_parent: int | None = None,
    top_n: int = 8,
) -> int | None:
    ps = state.parents
    got = ps.rank_matches(
        list(prompt_ids), state.vocab, getattr(state, "embedding", None), top_n=top_n)
    for pi, _sc in got:
        if skip_parent is None or int(pi) != int(skip_parent):
            return int(pi)
    return None


def fill_da_khe(
    state: GraphState,
    emit_ids: list[int],
    theme: set[int],
    index: FrameSlotIndex,
    cfg: FrameSlotConfig | None = None,
    *,
    parent_idx: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Điền nhiều khe trên emit khuôn. Trả (ids mới, meta)."""
    cfg = cfg or FrameSlotConfig()
    if cfg.decode_unit == "thuc_the" and index.unit_tri is not None:
        return _fill_da_khe_thuc_the(
            state, emit_ids, theme, index, cfg, parent_idx=parent_idx)
    return _fill_da_khe_token(
        state, emit_ids, theme, index, cfg, parent_idx=parent_idx)


def _fill_da_khe_token(
    state: GraphState,
    emit_ids: list[int],
    theme: set[int],
    index: FrameSlotIndex,
    cfg: FrameSlotConfig,
    *,
    parent_idx: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    goc = list(emit_ids)
    ids = list(goc)
    keep, so, tri, co, lop = index.keep, index.so, index.tri, index.co, index.lop_nb
    nv = len(keep)
    r = random.Random(cfg.seed * 7919 + (parent_idx or 0) + 17)
    rep_tokens: dict[int, tuple[int, ...]] = {}
    theme_txt = " ".join(
        (state.vocab.token_of(t) if hasattr(state.vocab, "token_of") else str(t))
        for t in sorted(theme)
    ).replace("Ġ", " ")
    theme_eras = eras_in_text(theme_txt) if cfg.era_filter else set()

    def bam(c: int) -> int:
        return len(co.get(c, ()) & theme)

    prop: list[tuple[int, int, int, list[int]]] = []
    for i in range(1, len(ids) - 1):
        x = ids[i]
        if not (0 <= x < nv) or not keep[x] or x in so:
            continue
        tri_c = tri.get((ids[i - 1], ids[i + 1]), set()) - {x} - so
        lop_c = set(lop.get(x, ())) - {x} - so
        if cfg.licence == "lop":
            cs = lop_c
        elif cfg.licence == "giao":
            cs = tri_c & lop_c
        else:
            cs = tri_c
        if cfg.era_filter and cs:
            old_eras = eras_of_rep(state, x, rep_tokens)
            cs = {
                c for c in cs
                if era_ok(theme_eras, old_eras, eras_of_rep(state, c, rep_tokens))
            }
        if not cs:
            continue
        cslist = sorted(cs)
        best = max(cslist, key=lambda z: (bam(z), -z)) if theme else cslist[0]
        gain = bam(best) - bam(x) if theme else 0
        if cfg.slot_rank == "bam" and theme:
            need = max(cfg.min_gain, 1)
            if gain < need:
                continue
        rank = gain if cfg.slot_rank == "bam" else len(cs)
        prop.append((rank, len(cs), i, cslist))
    prop.sort(reverse=True)
    n_de_xuat = len(prop)
    if cfg.max_thay > 0:
        prop = prop[: cfg.max_thay]
    n_thay = 0
    for _rk, _nc, i, cs in prop:
        if theme:
            c = max(cs, key=lambda z: (bam(z), -z))
        else:
            c = r.choice(cs)
        old = ids[i]
        ids[i] = c
        if breaks_word(state, index.ws_model, ids, i):
            ids[i] = old
            continue
        n_thay += 1
    meta = {
        "n_thay": n_thay,
        "n_khe_de_xuat": n_de_xuat,
        "trung_nguyen_van": ids == goc,
        "licence": cfg.licence,
        "max_thay": cfg.max_thay or None,
        "slot_rank": cfg.slot_rank,
        "min_gain": cfg.min_gain or None,
        "decode_unit": "token",
        "era_filter": cfg.era_filter,
        "parent": parent_idx,
        "stop": f"ep_da_khe_{cfg.licence}",
    }
    return ids, meta


def _unit_banned(unit: UnitSpan, so: set[int]) -> bool:
    return any(t in so for t in unit.tokens)


def _fill_da_khe_thuc_the(
    state: GraphState,
    emit_ids: list[int],
    theme: set[int],
    index: FrameSlotIndex,
    cfg: FrameSlotConfig,
    *,
    parent_idx: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    goc = list(emit_ids)
    ids = list(goc)
    keep, so = index.keep, index.so
    co = index.co
    unit_tri = index.unit_tri or {}
    unit_lop = index.unit_lop or {}
    rep_tokens = index.rep_tokens or {}
    wsm = index.ws_model
    pn = index.pn
    r = random.Random(cfg.seed * 7919 + (parent_idx or 0) + 17)

    theme_txt = " ".join(
        (state.vocab.token_of(t) if hasattr(state.vocab, "token_of") else str(t))
        for t in sorted(theme)
    ).replace("Ġ", " ")
    theme_eras = eras_in_text(theme_txt) if cfg.era_filter else set()

    def bam(rid: int) -> int:
        return bam_rep(rid, theme, co, rep_tokens)

    prop: list[tuple[int, int, int, int, int, list[int]]] = []
    units = segment_emit_thuc_the(state, ids, wsm, pn)
    if len(units) < 3:
        return ids, {
            "n_thay": 0,
            "n_khe_de_xuat": 0,
            "trung_nguyen_van": True,
            "licence": cfg.licence,
            "decode_unit": "thuc_the",
            "parent": parent_idx,
            "stop": f"ep_da_khe_{cfg.licence}",
        }

    for ui in range(1, len(units) - 1):
        u = units[ui]
        if _unit_banned(u, so) or not _rep_keep(u.rep, keep, state.vocab):
            continue
        tri_c = unit_tri.get((units[ui - 1].rep, units[ui + 1].rep), set()) - {u.rep}
        lop_c = set(unit_lop.get(u.rep, ())) - {u.rep}
        tri_c = {c for c in tri_c if not any(t in so for t in tokens_for_rep(
            state, c, rep_tokens, keep))}
        lop_c = {c for c in lop_c if not any(t in so for t in tokens_for_rep(
            state, c, rep_tokens, keep))}
        if cfg.licence == "lop":
            cs = lop_c
        elif cfg.licence == "giao":
            cs = tri_c & lop_c
            if not cs:
                allow_soft = (not cfg.soft_giao_single_only) or len(u.tokens) == 1
                if allow_soft:
                    cs = lop_c
        else:
            cs = tri_c
        if cfg.era_filter and cs:
            old_eras = eras_of_rep(state, u.rep, rep_tokens)
            cs = {
                c for c in cs
                if era_ok(theme_eras, old_eras, eras_of_rep(state, c, rep_tokens))
            }
        if not cs:
            continue
        cslist = sorted(cs)
        best = max(cslist, key=lambda z: (bam(z), -z)) if theme else cslist[0]
        gain = bam(best) - bam(u.rep) if theme else 0
        if cfg.slot_rank == "bam" and theme:
            need = max(cfg.min_gain, 1)
            if gain < need:
                continue
        rank = gain if cfg.slot_rank == "bam" else len(cs)
        prop.append((rank, len(cs), u.start, u.end, u.rep, cslist))

    prop.sort(reverse=True)
    n_de_xuat = len(prop)
    if cfg.max_thay > 0:
        prop = prop[: cfg.max_thay]
    n_thay = 0
    for _rk, _nc, start, end, orep, cs in sorted(prop, key=lambda x: x[2], reverse=True):
        need = cfg.min_gain
        if n_thay >= 1 and cfg.second_min_gain > 0:
            need = max(need, cfg.second_min_gain)
        if theme:
            ranked = sorted(cs, key=lambda z: (bam(z), -z), reverse=True)
            cand_rep = None
            for z in ranked:
                if bam(z) - bam(orep) >= max(need, 1):
                    cand_rep = z
                    break
            if cand_rep is None:
                continue
        else:
            cand_rep = r.choice(cs)
        new_toks = list(tokens_for_rep(state, cand_rep, rep_tokens, keep))
        if not new_toks or tuple(new_toks) == tuple(ids[start:end]):
            continue
        trial = ids[:start] + new_toks + ids[end:]
        syl = decode_surface(state, trial).split()
        if not ws_tags_ok(wsm, syl):
            continue
        ids = trial
        n_thay += 1

    do_hybrid = False
    if cfg.max_thay > 0 and n_thay < cfg.max_thay:
        if cfg.hybrid_if_zero:
            do_hybrid = n_thay == 0
        else:
            do_hybrid = True  # P114: luôn lấp phần còn
    if do_hybrid:
        tok_cfg = FrameSlotConfig(
            licence=cfg.licence,
            slot_rank=cfg.slot_rank,
            min_gain=cfg.min_gain,
            max_thay=cfg.max_thay - n_thay,
            no_co_graph=cfg.no_co_graph,
            decode_unit="token",
            seed=cfg.seed,
            era_filter=False,
        )
        tok_idx = FrameSlotIndex(
            keep=index.keep, so=index.so, tri=index.tri, co=index.co,
            lop_nb=index.lop_nb, ws_model=index.ws_model,
            n_syn_edges=index.n_syn_edges, n_co_graph=index.n_co_graph,
        )
        ids, tok_meta = _fill_da_khe_token(
            state, ids, theme, tok_idx, tok_cfg, parent_idx=parent_idx)
        n_thay += int(tok_meta.get("n_thay", 0))

    meta = {
        "n_thay": n_thay,
        "n_khe_de_xuat": n_de_xuat,
        "trung_nguyen_van": ids == goc,
        "licence": cfg.licence,
        "max_thay": cfg.max_thay or None,
        "slot_rank": cfg.slot_rank,
        "min_gain": cfg.min_gain or None,
        "decode_unit": "thuc_the",
        "era_filter": cfg.era_filter,
        "soft_giao_single_only": cfg.soft_giao_single_only,
        "hybrid_if_zero": cfg.hybrid_if_zero,
        "second_min_gain": cfg.second_min_gain or None,
        "parent": parent_idx,
        "stop": f"ep_da_khe_{cfg.licence}",
    }
    return ids, meta


def generate_frame_slot(
    state: GraphState,
    prompt: str,
    *,
    index: FrameSlotIndex | None = None,
    cfg: FrameSlotConfig | None = None,
    root: Path | None = None,
    skip_parent: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Đường sinh đầy đủ: encode đề → rank cha → fill_da_khe → ids (không gồm đề)."""
    cfg = cfg or FrameSlotConfig()
    root = root or Path.cwd()
    if index is None:
        index = build_frame_slot_index(state, cfg, root=root)
    prompt_ids = list(state.vocab.encode_prompt(prompt))
    if not prompt_ids:
        return [], {"stop": "prompt_rong", "n_thay": 0}
    pi = rank_other_parent(
        state, prompt_ids, skip_parent=skip_parent, top_n=cfg.top_n)
    if pi is None:
        return [], {"stop": "khong_truy_hoi", "n_thay": 0}
    goc = [int(t) for t in state.parents.parents[pi].emit_ids]
    nv = len(index.keep)
    theme = {int(t) for t in prompt_ids if 0 <= int(t) < nv and index.keep[int(t)]}
    ids, meta = fill_da_khe(
        state, goc, theme, index, cfg, parent_idx=pi)
    meta["stop"] = meta.get("stop") or "frame_da_khe"
    return ids, meta
