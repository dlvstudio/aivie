"""TK_051 ship — ghép ≥2 cha có phép (P203).

Công thức: dual retrieve (theme overlap + era) → bản lề W_skip d≤8 ·
min_count≥1 → chọn B theo (ent_keep, join_score) → nối emit A+B;
không có bản lề → fallback P112 trên cha #1.

Tách từ `scripts/gen_three_arm.py` để CLI / `generate(mode=compose_2cha)` dùng chung.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nn_aivie.core.state import GraphState
from nn_aivie.hca.context_positional import is_structural_token
from nn_aivie.hca.frame_slot_gen import (
    FrameSlotConfig,
    FrameSlotIndex,
    build_frame_slot_index,
    eras_in_text,
    fill_da_khe,
)
from nn_aivie.hca.readout import decode_surface
from nn_aivie.hca.w_skip import ensure_w_skip


@dataclass
class ComposeConfig:
    """Defaults = P203 cổng (3) ĐẠT."""

    join_d_max: int = 8
    join_min_count: float = 1.0
    top_n: int = 24
    max_cands: int = 12
    edge_width: int = 3
    skip_parent: int | None = None
    seed: int = 42
    label_map: str = "out/token_labels_v12.json"
    ws_model: str = "out/models/ws_crf_uts_full"
    # fallback P112
    max_thay: int = 2
    min_gain: int = 1
    slot_rank: str = "bam"
    no_co_graph: bool = True


def ent_keep(
    prompt_ids: list[int],
    out_ids: list[int],
    keep,
) -> float:
    """Tỷ lệ token keep của đề còn xuất hiện trong emit."""
    nv = len(keep)
    th = {int(t) for t in prompt_ids if 0 <= int(t) < nv and keep[int(t)]}
    if not th:
        return 1.0
    ou = {int(t) for t in out_ids if 0 <= int(t) < nv}
    return round(len(th & ou) / len(th), 4)


def _edge_contents(
    state: GraphState,
    ids: list[int],
    keep,
    *,
    side: str,
    width: int = 3,
) -> list[int]:
    nv = len(keep)
    ks = [t for t in ids if 0 <= t < nv and keep[t]]
    if not ks:
        ks = [
            t for t in ids
            if 0 <= t < nv and state.vocab.is_surface(t)
            and not is_structural_token(state.vocab, t)
        ]
    if not ks:
        return []
    if side == "last":
        return ks[-width:]
    return ks[:width]


def join_score(
    state: GraphState,
    ws,
    emit_a: list[int],
    emit_b: list[int],
    keep,
    *,
    d_max: int,
    width: int = 3,
) -> float:
    lasts = _edge_contents(state, emit_a, keep, side="last", width=width)
    firsts = _edge_contents(state, emit_b, keep, side="first", width=width)
    if not lasts or not firsts:
        return 0.0
    best = 0.0
    d_cap = min(int(d_max), int(ws.d_max))
    first_set = set(firsts)
    for last in lasts:
        for d in range(1, d_cap + 1):
            row = ws.mats[d].getrow(int(last))
            if row.nnz == 0:
                continue
            for j, w in zip(row.indices, row.data):
                if int(j) in first_set:
                    best = max(best, float(w))
    return best


def rank_compose_parents(
    state: GraphState,
    prompt_ids: list[int],
    keep,
    *,
    skip_parent: int | None = None,
    top_n: int = 24,
    max_cands: int = 12,
) -> list[int]:
    """Dual retrieve: theme keep-overlap + era filter; bỏ cha nguồn đề."""
    ps = state.parents
    if ps is None or len(ps) == 0:
        return []
    nv = len(keep)
    got = ps.rank_matches(
        list(prompt_ids), state.vocab, getattr(state, "embedding", None),
        top_n=max(int(top_n), 16),
    )
    theme = {int(t) for t in prompt_ids if 0 <= int(t) < nv and keep[int(t)]}
    prompt_txt = decode_surface(state, list(prompt_ids))
    theme_eras = eras_in_text(prompt_txt)
    cands: list[int] = []
    for pi, _sc in got:
        pi = int(pi)
        if skip_parent is not None and pi == int(skip_parent):
            continue
        emit = [int(t) for t in ps.parents[pi].emit_ids]
        emit_keep = {t for t in emit if 0 <= t < nv and keep[t]}
        if theme and not (emit_keep & theme):
            continue
        if theme_eras:
            emit_eras = eras_in_text(decode_surface(state, emit))
            if emit_eras and not (emit_eras & theme_eras):
                continue
        cands.append(pi)
        if len(cands) >= int(max_cands):
            break
    return cands


def compose_two_parents(
    state: GraphState,
    prompt_ids: list[int],
    *,
    index: FrameSlotIndex,
    cfg: ComposeConfig | None = None,
    ws=None,
    skip_parent: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Trả (ids, meta). ids = emit nối hoặc P112 fallback; rỗng nếu không truy hồi."""
    cfg = cfg or ComposeConfig()
    ps = state.parents
    if ps is None or len(ps) == 0:
        return [], {
            "stop": "khong_parents", "n_src": 0, "parents": [],
            "ent_keep": 0.0, "joined": False, "fallback_p112": False,
            "join_ok": False, "join_score": 0.0,
        }
    if ws is None:
        ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)

    skip = cfg.skip_parent if skip_parent is None else skip_parent
    cands = rank_compose_parents(
        state, prompt_ids, index.keep,
        skip_parent=skip, top_n=cfg.top_n, max_cands=cfg.max_cands,
    )
    theme = {
        int(t) for t in prompt_ids
        if 0 <= int(t) < len(index.keep) and index.keep[int(t)]
    }

    def _fallback(pi: int, *, join_sc: float = 0.0, tried: list[int] | None = None):
        goc = [int(t) for t in ps.parents[pi].emit_ids]
        fcfg = FrameSlotConfig(
            licence="giao",
            slot_rank=cfg.slot_rank,
            min_gain=cfg.min_gain,
            max_thay=cfg.max_thay,
            no_co_graph=cfg.no_co_graph,
            seed=cfg.seed,
            decode_unit="token",
            label_map=cfg.label_map,
            ws_model=cfg.ws_model,
        )
        ids, m = fill_da_khe(
            state, goc, theme, index, fcfg, parent_idx=pi)
        return ids, {
            "stop": "compose_fallback_p112",
            "n_src": 1,
            "parents": tried or [pi],
            "ent_keep": ent_keep(prompt_ids, ids, index.keep),
            "joined": False,
            "fallback": True,
            "fallback_p112": True,
            "join_ok": False,
            "join_score": round(join_sc, 4),
            **{k: m[k] for k in (
                "n_thay", "n_khe_de_xuat", "trung_nguyen_van",
                "licence", "max_thay", "slot_rank", "min_gain",
            ) if k in m},
        }

    if not cands:
        return [], {
            "stop": "khong_truy_hoi", "n_src": 0, "parents": [],
            "ent_keep": 0.0, "joined": False, "fallback": True,
            "fallback_p112": False, "join_ok": False, "join_score": 0.0,
        }

    if len(cands) == 1:
        return _fallback(cands[0], tried=cands)

    emit_a = [int(t) for t in ps.parents[cands[0]].emit_ids]
    best_j = -1
    best_key = (-1.0, -1.0)
    thr = float(cfg.join_min_count)
    for j in range(1, len(cands)):
        emit_b = [int(t) for t in ps.parents[cands[j]].emit_ids]
        sc = join_score(
            state, ws, emit_a, emit_b, index.keep,
            d_max=cfg.join_d_max, width=cfg.edge_width,
        )
        if sc < thr:
            continue
        ek = ent_keep(prompt_ids, emit_a + emit_b, index.keep)
        key = (ek, sc)
        if key > best_key:
            best_key = key
            best_j = j

    if best_j < 0:
        return _fallback(cands[0], join_sc=0.0, tried=cands[:2])

    emit_b = [int(t) for t in ps.parents[cands[best_j]].emit_ids]
    joined_ids = emit_a + emit_b
    return joined_ids, {
        "stop": "compose_2cha",
        "n_src": 2,
        "parents": [cands[0], cands[best_j]],
        "ent_keep": ent_keep(prompt_ids, joined_ids, index.keep),
        "joined": True,
        "fallback": False,
        "fallback_p112": False,
        "join_ok": True,
        "join_score": round(best_key[1], 4),
    }


def generate_compose_2cha(
    state: GraphState,
    prompt: str,
    *,
    index: FrameSlotIndex | None = None,
    cfg: ComposeConfig | None = None,
    root: Path | None = None,
    skip_parent: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Đường sinh đầy đủ: encode đề → dual retrieve → nối / fallback P112."""
    cfg = cfg or ComposeConfig()
    root = root or Path.cwd()
    fcfg = FrameSlotConfig(
        seed=cfg.seed,
        label_map=cfg.label_map,
        ws_model=cfg.ws_model,
        no_co_graph=cfg.no_co_graph,
        max_thay=cfg.max_thay,
        min_gain=cfg.min_gain,
        slot_rank=cfg.slot_rank,
        require_synonym=True,
    )
    if index is None:
        index = build_frame_slot_index(state, fcfg, root=root)
    prompt_ids = list(state.vocab.encode_prompt(prompt))
    if not prompt_ids:
        return [], {"stop": "prompt_rong", "n_src": 0, "joined": False}
    ws = ensure_w_skip(state, d_max=8, min_count=1, content_only=False)
    return compose_two_parents(
        state, prompt_ids, index=index, cfg=cfg, ws=ws,
        skip_parent=skip_parent,
    )
