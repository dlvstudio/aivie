#!/usr/bin/env python
"""P113/ship — CLI khung–đa-khe (defaults = P112).

Xem RUNBOOK_FRAME_P112_SHIP.md. Nhãn: truy hồi + khung có phép (không quảng cáo sáng tạo).

  PYTHONPATH=src .venv/bin/python scripts/gen_frame_slot.py \\
    --checkpoint checkpoints/network_v12_subst.pkl \\
    --prompt "Nhà Lý đóng đô ở đâu"

  PYTHONPATH=src .venv/bin/python scripts/gen_frame_slot.py --verify-p112
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.frame_slot_gen import (  # noqa: E402
    FrameSlotConfig,
    build_frame_slot_index,
    fill_da_khe,
    generate_frame_slot,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Sinh khung–đa-khe (P112 defaults)")
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_subst.pkl")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--label-map", default="out/token_labels_v12.json")
    ap.add_argument("--ws-model", default="out/models/ws_crf_uts_full")
    ap.add_argument("--max-thay", type=int, default=2)
    ap.add_argument("--min-gain", type=int, default=1)
    ap.add_argument("--slot-rank", choices=("bam", "ncand"), default="bam")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode-unit", choices=("token", "thuc_the"), default="token",
                    help="P114: điền khe theo token hoặc thuc_the/entity")
    ap.add_argument("--verify-p112", action="store_true",
                    help="tai lap fill tren parent P112, so n_thay (cong 1: >=95%% khop)")
    ap.add_argument("--out-meta", default="out/frame_p113_verify_meta.json")
    args = ap.parse_args()

    logger = get_run_logger("gen_frame_slot")
    cfg = FrameSlotConfig(
        max_thay=args.max_thay,
        min_gain=args.min_gain,
        slot_rank=args.slot_rank,
        seed=args.seed,
        label_map=args.label_map,
        ws_model=args.ws_model,
        no_co_graph=True,
        require_synonym=True,
        decode_unit=args.decode_unit,
    )

    t0 = time.time()
    import gc
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    logger.log(f"loaded ({time.time()-t0:.0f}s)")

    index = build_frame_slot_index(state, cfg, root=_ROOT, logger=logger)

    if args.verify_p112:
        _verify_p112(state, index, cfg, logger, _ROOT / args.out_meta)
        return

    if not args.prompt:
        raise SystemExit("cần --prompt hoặc --verify-p112")
    ids, meta = generate_frame_slot(
        state, args.prompt, index=index, cfg=cfg, root=_ROOT)
    print(decode_surface(state, ids).strip())
    print("# meta:", json.dumps(meta, ensure_ascii=False), file=sys.stderr)


def _verify_p112(state, index, cfg, logger, out_path: Path) -> None:
    """Cổng (1): tái lập đúng slice opener + theme p.tokens như gen_three_arm P112."""
    import random as _rnd
    from nn_aivie.hca.clause_layer import build_clause_layer
    from nn_aivie.hca.frame_slot_gen import rank_other_parent

    meta_p = json.loads((_ROOT / "out/frame_p112_meta.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in meta_p["rows"] if r.get("id")}

    logger.log("dựng tầng mệnh đề để tái lập opener P112…")
    layer = build_clause_layer(state, pmi_theta=0.0, logger=logger)
    rng = _rnd.Random(42)  # gen_three_arm --seed 42
    openers = [c for c in layer.clauses if c.opens_parent and len(c.tokens) >= 4]
    rng.shuffle(openers)
    # P112 arms: chep_cau, ep_mot_khe_giao, ep_da_khe_giao · n=120
    n = 120
    ep_da_prompts = openers[2 * n : 3 * n]
    if len(ep_da_prompts) < n:
        raise SystemExit(f"⛔ không đủ opener: {len(openers)}")

    # hàng trước shuffle (cùng thứ tự gen_three_arm)
    pre_meta = []
    for p in ep_da_prompts:
        pi = rank_other_parent(
            state, list(p.tokens), skip_parent=int(p.parent_idx), top_n=cfg.top_n)
        if pi is None:
            pre_meta.append({"n_thay": 0, "parent": None, "stop": "khong_truy_hoi"})
            continue
        theme = {
            int(t) for t in p.tokens
            if 0 <= int(t) < len(index.keep) and index.keep[int(t)]
        }
        goc = [int(t) for t in state.parents.parents[pi].emit_ids]
        _ids, m = fill_da_khe(state, goc, theme, index, cfg, parent_idx=pi)
        pre_meta.append(m)

    # 3 nhánh × 120; chỉ nhánh 3 có ep_da — pad giả 240 rồi nối ep_da rồi shuffle
    dummy = [{"n_thay": -1, "cfg": "pad"} for _ in range(2 * n)]
    for m in pre_meta:
        m["cfg"] = "ep_da_khe_giao"
    full = dummy + pre_meta
    order = list(range(len(full)))
    _rnd.Random(20260804).shuffle(order)
    match = 0
    hist_old: Counter = Counter()
    hist_new: Counter = Counter()
    n_cmp = 0
    for j, oi in enumerate(order):
        row = full[oi]
        if row.get("cfg") != "ep_da_khe_giao":
            continue
        sid = f"A{j:03d}"
        old = by_id.get(sid)
        if not old:
            continue
        n_cmp += 1
        hist_old[int(old.get("n_thay", 0))] += 1
        hist_new[int(row.get("n_thay", 0))] += 1
        if int(row.get("n_thay", -1)) == int(old.get("n_thay", -2)):
            match += 1

    rate = match / max(n_cmp, 1)
    tot_o = max(sum(hist_old.values()), 1)
    tot_n = max(sum(hist_new.values()), 1)
    ge1_old = sum(c for k, c in hist_old.items() if k >= 1) / tot_o
    ge1_new = sum(c for k, c in hist_new.items() if k >= 1) / tot_n
    out = {
        "n": n_cmp,
        "n_thay_match": match,
        "ty_le_khop": round(rate, 4),
        "gate1_ge_0_95": rate >= 0.95,
        "ge1_old": round(ge1_old, 4),
        "ge1_new": round(ge1_new, 4),
        "hist_old": {str(k): v for k, v in sorted(hist_old.items())},
        "hist_new": {str(k): v for k, v in sorted(hist_new.items())},
        "config": {
            "max_thay": cfg.max_thay,
            "min_gain": cfg.min_gain,
            "slot_rank": cfg.slot_rank,
            "seed": cfg.seed,
        },
        "method": "opener_slice_replay",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(
        f"verify P112: n={n_cmp} khớp n_thay {match}/{n_cmp} = {rate:.1%} "
        f"{'✅' if rate >= 0.95 else '❌'} · ge1 {ge1_old:.1%}→{ge1_new:.1%} → {out_path}"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
