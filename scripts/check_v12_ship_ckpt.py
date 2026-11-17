#!/usr/bin/env python3
"""Kiểm checkpoint ship v12: đủ cấu trúc chưa, và ngưỡng có cắt sạch ứng viên không.

Trả lời ba câu trước khi tin bất kỳ số đo nào:

1. Ckpt có thứ đường ship cần không? (`subst` cần cạnh SYNONYM;
   `isg_pw` cần cụm ISG power-win)
2. `analogy_la` có ứng viên không, hay ngưỡng hub cắt sạch? — `n_triples=0`
   nghĩa là cơ chế CÂM vì cấu hình, không phải vì âm tính.
3. Vá `la_emit_id` (chọn biến thể `là` theo df, không theo id nhỏ nhất) có
   đổi hành vi trên v12 không?

Ghi tiến trình ra `--log` (gồm cả khâu nạp — nạp ckpt v12 có thể rất lâu).

    PYTHONPATH=src python scripts/check_v12_ship_ckpt.py \
      --checkpoint checkpoints/network_v12_subst_isg_pw.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

T0 = time.monotonic()
_LOG = None


def log(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824
    line = f"[{time.monotonic() - T0:7.1f}s RSS {rss:5.2f}GB] {msg}"
    print(line, flush=True)
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def main() -> int:
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--prompts",
        default="anh là,mẹ là,học sinh là,Việt Nam là,con mèo là,nhà Trần là",
    )
    args = ap.parse_args()
    _LOG = args.log or f"out/check_ckpt_{Path(args.checkpoint).stem}.log"
    Path(_LOG).parent.mkdir(parents=True, exist_ok=True)

    # gc.disable(): nạp checkpoint lớn nhanh hơn ~1300× (ghi nhớ dự án).
    gc.disable()
    log(f"gc.disable() · nạp {args.checkpoint} …")

    from nn_aivie.core.state import GraphState

    st = GraphState.load(args.checkpoint)
    log(f"nạp xong · parents={len(st.parents)} vocab={len(st.vocab)}")

    rep: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "n_parents": len(st.parents),
        "n_vocab": len(st.vocab),
    }

    # (1) cấu trúc: SYNONYM (đường frame/compose) + ISG (đường analogy)
    from nn_aivie.core.edge_kinds import EdgeKind

    try:
        syn = st.adjacency.layer(EdgeKind.SYNONYM)
        n_syn = sum(len(syn.row(i) or {}) for i in range(len(st.vocab)))
    except Exception as exc:  # lớp chưa bake
        n_syn = -1
        log(f"SYNONYM: không đọc được ({exc})")
    rep["n_synonym_edges"] = n_syn
    log(f"cạnh SYNONYM = {n_syn}  (frame_da_khe/compose_2cha cần >0)")

    seg = getattr(st.parents, "seg", None)
    n_cl = len(seg.clusters) if seg is not None else 0
    n_span = sum(1 for c in (seg.clusters if seg else []) if len(c.token_span()) >= 3)
    rep["n_isg_clusters"] = n_cl
    rep["n_isg_span_ge3"] = n_span
    log(f"cụm ISG = {n_cl} · span≥3 token = {n_span}")

    # (2) analogy: ngưỡng có để lại ứng viên nào không
    from nn_aivie.hca.analogy_bridge_gen import (
        AnalogyBridgeConfig,
        build_analogy_bridge_index,
        generate_analogy_bridge,
    )

    cfg = AnalogyBridgeConfig()
    t = time.monotonic()
    idx = build_analogy_bridge_index(st, cfg)
    log(f"dựng index analogy {time.monotonic() - t:.1f}s")
    for k in (
        "n_l1", "n_l2", "n_triples", "n_prop_keys_dropped_df",
        "n_heads", "n_head_cluster", "n_head_phrase", "n_head_entity",
    ):
        log(f"  {k} = {idx.meta.get(k)}")
    rep["analogy_meta"] = dict(idx.meta)
    if not idx.meta.get("n_triples"):
        log("🔴 n_triples=0 — cơ chế CÂM vì ngưỡng, mọi kết quả âm tính ở đây VÔ NGHĨA")

    # (3) vá la_emit_id có đổi hành vi không
    la_ids = sorted(idx.la_ids)
    gdf = {t_: int(st.parents._gdf.get(t_, 0)) for t_ in la_ids}  # noqa: SLF001
    rep["la_ids"] = la_ids
    rep["la_gdf"] = gdf
    rep["la_emit_id"] = int(idx.la_emit_id)
    rep["la_emit_doi_so_voi_min_id"] = bool(idx.la_emit_id != min(la_ids))
    log(
        f"la_ids={la_ids} gdf={gdf} → emit={idx.la_emit_id} "
        f"({'ĐỔI' if idx.la_emit_id != min(la_ids) else 'trùng'} so với min-id cũ)"
    )

    outs = []
    for de in [p.strip() for p in args.prompts.split(",") if p.strip()]:
        ids, meta = generate_analogy_bridge(st, de, index=idx, cfg=cfg)
        txt = "".join(st.vocab.token_of(i) for i in ids).replace("Ġ", " ").strip()
        outs.append({"de": de, "ra": txt, "stop": meta.get("stop"),
                     "donors": meta.get("donors"), "bridges": meta.get("bridges")})
        log(f"  {de!r} → {txt!r}  stop={meta.get('stop')} donors={meta.get('donors')}")
    rep["prompts"] = outs

    out = args.out or f"out/check_ckpt_{Path(args.checkpoint).stem}.json"
    Path(out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
