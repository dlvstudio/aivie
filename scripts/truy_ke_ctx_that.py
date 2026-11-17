"""Vì sao KỀ NGỮ CẢNH (§60) KHÔNG chuyển gì trên `classrich`?

Thước sinh A/B (`--ke-ctx 0` vs `2.0`) ra HAI FILE Y HỆT. Ba khả năng, script
này tách chúng ra:

  a) bảng KỀ rỗng      — prompt chỉ có MỘT token nội dung và nó không có cạnh kề
  b) bảng KỀ có nhưng không giao với bể ③
  c) có giao nhưng trọng số ③ trên dữ liệu thật KHÔNG hoà nên tie-break câm

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/truy_ke_ctx_that.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classrich.pkl"
    ca_json = sys.argv[2] if len(sys.argv) > 2 else "out/gen_ke0.json"
    cfg = HCAConfig()
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    state, _ = load_checkpoint(ck)
    v = state.vocab
    wsp = H.load_ws_spans("checkpoints/classrich.ws_spans.json")
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, label_indeg_q=cfg.hub_chain_indeg_q,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm, ws_spans=wsp)
    ca = json.load(open(ca_json))["ca"]
    print(f"\n### {ck} · {len(state.parents.parents)} cha · {len(ca)} ca\n")

    dem = Counter()
    for c in ca[:12]:
        pr = c["P"]
        ctx = list(v.encode_prompt(pr))
        bm = [s for s in (H._surf(v, int(t)) for t in ctx)
              if s and len(s) > 1 and s not in H._FUNC]
        ke: dict[str, float] = {}
        for s in reversed(bm):
            for x in (H.resolve_surface_ids(v, s) or [])[:6]:
                for dst, (w_, d_) in idx.fwd.get(int(x), {}).items():
                    if int(d_) <= 1:
                        sd = H._surf(v, int(dst))
                        if sd:
                            ke[sd] = max(ke.get(sd, 0.0), float(w_))
            if ke:
                break
        # bể ③ của nhãn mà ② thực sự chọn
        moi = [t for t in ctx if H._surf(v, int(t)) in bm]
        s1 = H.score_hubs_from(state, idx, moi[:4], topk=10, gidf_min=0.0,
                               channel="typed_first", kenh_thuoc_tinh=True)
        _c = [h for h in s1 if h.tid not in set(ctx)]
        _t = [h for h in _c if h.typed]
        hat = (_t if _t else _c)[:4]
        if not hat:
            dem["①_rỗng"] += 1
            continue
        s2 = H.score_hubs_from(state, idx, [h.tid for h in hat], topk=10,
                               gidf_min=0.0, channel="typed_first",
                               rank=cfg.hub_chain_rank, loc_pos=True,
                               ctx_ids=ctx)
        cam = {h.surface for h in hat} | set(bm)
        ok = [h for h in s2 if h.surface not in cam]
        if not ok:
            dem["②_rỗng"] += 1
            continue
        nb = H.resolve_surface_ids(v, ok[0].surface) or [ok[0].tid]
        ms = H.reverse_peers(state, idx, [ok[0].tid] + nb[:3], topk=30,
                             gidf_min=0.0, channel="typed_first")
        w = Counter(round(m.weight, 3) for m in ms)
        giao = [m.surface for m in ms if m.surface in ke]
        hoa = sum(n for _, n in w.items() if n > 1)
        dem["ca"] += 1
        dem["ke_rỗng"] += 1 if not ke else 0
        dem["giao_rỗng"] += 1 if ke and not giao else 0
        print(f"  P={pr!r:<14} nhãn={ok[0].surface!r:<12} bể③={len(ms):>3}"
              f"  |KỀ|={len(ke):>5}  giao={len(giao):>2}"
              f"  hạng hoà={hoa}/{len(ms)}  {giao[:4]}")
    print(f"\n  {dict(dem)}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
