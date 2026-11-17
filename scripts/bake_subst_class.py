#!/usr/bin/env python
"""P106 (TK_049) — đông cứng lớp thay thế + hub Entity: vào checkpoint.

Probe TK_048 tính lại lớp mỗi lần rồi vứt. Ở đây ghi cạnh `SYNONYM` (kênh paradigmatic
đã có, priority 0) + spawn `Entity:…` cho đơn vị đa-token (`thuc_the`) với `CO_ACTIVATION`
tới thành viên.

Không đè checkpoint gốc — ghi `--out-checkpoint`.

Chạy: xem RUNBOOK_TK_049_P106.md
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def _segment_seqs(state, unit: str, label_map: Path, ws_model: Path, logger, t0: float):
    """Tách cha → chuỗi đơn vị (token | thuc_the). Logic khớp probe_subst_class.py."""
    v, ps = state.vocab, state.parents
    n = len(v)
    keep = np.zeros(n, dtype=bool)
    for t in range(n):
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            if float(ps._gidf(t)) >= 4.0:  # noqa: SLF001
                keep[t] = True
        except Exception:
            continue

    seqs: list[list[tuple[int, ...]]] = []
    if unit == "token":
        for rec in ps.parents:
            seqs.append([(int(t),) for t in rec.emit_ids])
        return seqs, keep

    from nn_aivie.ws_crf.ws_model import WSCRFModel  # noqa: PLC0415
    wsm = WSCRFModel.load(ws_model)
    for pi, rec in enumerate(ps.parents):
        if pi and pi % 1500 == 0:
            logger.log(f"  tách từ {pi:,}/{len(ps.parents):,} ({time.time()-t0:.0f}s)")
        ids = [int(t) for t in rec.emit_ids]
        whole = decode_surface(state, ids)
        syl = whole.split()
        if not syl:
            continue
        tags = wsm.tag(syl)
        if len(tags) != len(syl):
            continue
        si_of: list[int | None] = []
        for k in range(len(ids)):
            pre = decode_surface(state, ids[: k + 1])
            si_of.append(len(pre.split()) - 1
                         if whole.startswith(pre) and not whole[len(pre):][:1].isalnum()
                         else None)
        w_of_syl: list[int] = []
        cur = -1
        for tg in tags:
            if tg != "I-W" or cur < 0:
                cur += 1
            w_of_syl.append(cur)
        units: list[list[int]] = [[] for _ in range(cur + 1)]
        for k, t in enumerate(ids):
            si = si_of[k]
            if si is not None and 0 <= si < len(w_of_syl):
                units[w_of_syl[si]].append(t)
        seqs.append([tuple(u) for u in units if u])

    if unit != "thuc_the":
        return seqs, keep

    pn: set[int] = set()
    if label_map.exists():
        for ts, d in (json.loads(label_map.read_text(encoding="utf-8")).get("tu_loai") or {}).items():
            if d and max(d, key=d.get) == "tên riêng":
                pn.add(int(ts))
    if not pn:
        raise SystemExit(f"⛔ {label_map} không có nhãn `tên riêng` ⇒ không ghép thuc_the")
    logger.log(f"token mang nhãn `tên riêng`: {len(pn):,}")
    is_pn = lambda u: bool(u) and all(t in pn for t in u)
    out2, n_merge = [], 0
    for sq in seqs:
        r, i = [], 0
        while i < len(sq):
            if is_pn(sq[i]):
                j = i + 1
                while j < len(sq) and is_pn(sq[j]):
                    j += 1
                if j > i + 1:
                    n_merge += 1
                r.append(tuple(t for u in sq[i:j] for t in u))
                i = j
            else:
                r.append(sq[i])
                i += 1
        out2.append(r)
    logger.log(f"chuỗi tên riêng được ghép: {n_merge:,}")
    return out2, keep


def _unit_name(v, u: tuple[int, ...]) -> str:
    _tn = lambda t: (v.token_of(t) if hasattr(v, "token_of") else str(t))
    return "".join(_tn(t) for t in u).replace("Ġ", " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--label-map", default="out/token_labels_v12.json")
    ap.add_argument("--ws-model", default="out/models/ws_crf_uts_full")
    ap.add_argument("--unit", choices=("token", "thuc_the"), default="thuc_the")
    ap.add_argument("--min-ctx", type=int, default=3)
    ap.add_argument("--min-share", type=int, default=3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out-checkpoint", default="checkpoints/network_v12_subst.pkl")
    ap.add_argument("--out-meta", default="out/bake_subst_v12_meta.json")
    args = ap.parse_args()

    logger = get_run_logger("bake_subst_class")
    t0 = time.time()
    logger.log(f"P106 bake · {args.checkpoint} · unit={args.unit}")

    gc.disable()
    state, cache = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps, adj = state.vocab, state.parents, state.adjacency
    n0 = len(v)

    seqs, keep = _segment_seqs(
        state, args.unit, _ROOT / args.label_map, _ROOT / args.ws_model, logger, t0)

    def ok(u: tuple[int, ...]) -> bool:
        return all(0 <= t < n0 and keep[t] for t in u)

    # --- spawn Entity: cho đơn vị đa-token + CO_ACTIVATION tới thành viên ---------------
    n_entity = 0
    n_co = 0
    hub_of: dict[tuple[int, ...], int] = {}
    for sq in seqs:
        for u in sq:
            if len(u) < 2 or u in hub_of:
                continue
            title = _unit_name(v, u)
            if not title:
                continue
            eid = v.entity_id(title)
            hub_of[u] = eid
            w = 1.0 / max(len(u), 1)
            for t in u:
                adj.set(eid, int(t), w, EdgeKind.CO_ACTIVATION)
                adj.set(int(t), eid, w, EdgeKind.CO_ACTIVATION)
                n_co += 2
            n_entity += 1
    logger.log(f"Entity: spawned {n_entity:,} · CO links {n_co:,} ({time.time()-t0:.0f}s)")

    # --- lớp thay thế trên đơn vị --------------------------------------------------------
    ctx: dict[tuple[int, ...], set] = defaultdict(set)
    for sq in seqs:
        for i in range(1, len(sq) - 1):
            if ok(sq[i]):
                ctx[sq[i]].add((sq[i - 1], sq[i + 1]))
    ctx = {u: c for u, c in ctx.items() if len(c) >= args.min_ctx}
    logger.log(f"đơn vị ≥{args.min_ctx} bối cảnh: {len(ctx):,}")

    inv: dict[tuple, list] = defaultdict(list)
    for u, cs in ctx.items():
        for c in cs:
            inv[c].append(u)

    # --- lớp TOKEN (để decode giao đọc SYNONYM trên emit_ids) ---------------------------
    # Đơn vị 1-token → cạnh trực tiếp. Đơn vị đa-token → cạnh giữa MỌI thành viên của hai
    # đơn vị (cùng lớp) — decode token vẫn thấy hàng xóm paradigmatic.
    ctx_tok: dict[int, set] = defaultdict(set)
    for sq in seqs:
        flat = [t for u in sq for t in u]
        for i in range(1, len(flat) - 1):
            if 0 <= flat[i] < n0 and keep[flat[i]]:
                ctx_tok[flat[i]].add((flat[i - 1], flat[i + 1]))
    ctx_tok = {t: c for t, c in ctx_tok.items() if len(c) >= args.min_ctx}
    inv_tok: dict[tuple, list[int]] = defaultdict(list)
    for t, cs in ctx_tok.items():
        for c in cs:
            inv_tok[c].append(t)

    n_syn = 0
    n_unit_with_nb = 0
    seen_pair: set[tuple[int, int]] = set()

    def _syn(a: int, b: int, w: float) -> None:
        nonlocal n_syn
        if a == b or a < 0 or b < 0:
            return
        key = (a, b) if a < b else (b, a)
        if key in seen_pair:
            return
        seen_pair.add(key)
        adj.set(a, b, w, EdgeKind.SYNONYM)
        adj.set(b, a, w, EdgeKind.SYNONYM)
        n_syn += 2

    # (A) token↔token từ lớp token
    tok_cov = 0
    for x, cs in ctx_tok.items():
        sh: dict[int, int] = defaultdict(int)
        for c in cs:
            for y in inv_tok[c]:
                if y != x:
                    sh[y] += 1
        nx = len(cs) ** 0.5
        sc = sorted(((m / (nx * len(ctx_tok[y]) ** 0.5), y)
                     for y, m in sh.items() if m >= args.min_share), reverse=True)
        top = sc[: args.topk]
        if top:
            tok_cov += 1
        for s, y in top:
            _syn(int(x), int(y), float(s))

    # (B) hub↔hub + thành viên↔thành viên từ lớp đơn vị (thuc_the)
    for x, cs in ctx.items():
        sh: dict[tuple[int, ...], int] = defaultdict(int)
        for c in cs:
            for y in inv[c]:
                if y != x:
                    sh[y] += 1
        nx = len(cs) ** 0.5
        sc = sorted(((m / (nx * len(ctx[y]) ** 0.5), y)
                     for y, m in sh.items() if m >= args.min_share), reverse=True)
        top = sc[: args.topk]
        if top:
            n_unit_with_nb += 1
        for s, y in top:
            # Chỉ hub↔hub / token-đơn↔token-đơn ở tầng đơn vị. Decode token đọc lớp (A).
            hx = hub_of.get(x) if len(x) > 1 else (x[0] if len(x) == 1 else None)
            hy = hub_of.get(y) if len(y) > 1 else (y[0] if len(y) == 1 else None)
            if hx is not None and hy is not None:
                _syn(int(hx), int(hy), float(s))

    # pad E nếu vocab mọc Entity:
    if state.embedding is not None and state.embedding.matrix.shape[0] < len(v):
        pad = len(v) - state.embedding.matrix.shape[0]
        state.embedding.matrix = np.pad(
            state.embedding.matrix, ((0, pad), (0, 0)), mode="constant")
        logger.log(f"pad E +{pad} hàng (Entity mới)")

    # invalidate parent entity cache nếu có
    if hasattr(ps, "_entity_ids_ready"):
        ps._entity_ids_ready = False

    out_ck = _ROOT / args.out_checkpoint
    out_ck.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(state, out_ck, cache=cache)
    logger.log(f"✓ ghi {out_ck} ({time.time()-t0:.0f}s)")

    ty_le = tok_cov / max(len(ctx_tok), 1)
    meta = {
        "checkpoint_in": args.checkpoint,
        "checkpoint_out": args.out_checkpoint,
        "unit": args.unit,
        "min_share": args.min_share,
        "topk": args.topk,
        "n_vocab_in": n0,
        "n_vocab_out": len(v),
        "n_entity": n_entity,
        "n_co_activation_writes": n_co,
        "n_synonym_directed": n_syn,
        "n_synonym_undirected_pairs": len(seen_pair),
        "n_token_ctx_ge_min": len(ctx_tok),
        "n_token_co_synonym": tok_cov,
        "ty_le_token_co_synonym": round(ty_le, 4),
        "n_unit_ctx_ge_min": len(ctx),
        "n_unit_co_lang_gieng": n_unit_with_nb,
        "gate1_ty_le_ge_0_20": ty_le >= 0.20,
        "gate2_n_entity_ge_100": n_entity >= 100,
        "seconds": round(time.time() - t0, 1),
    }
    op = _ROOT / args.out_meta
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(
        f"cổng bake: ty_le={ty_le:.1%} {'✅' if ty_le >= 0.20 else '❌'} · "
        f"n_entity={n_entity:,} {'✅' if n_entity >= 100 else '❌'} → {args.out_meta}")


if __name__ == "__main__":
    main()
