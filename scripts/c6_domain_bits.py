#!/usr/bin/env python3
"""TK_108 C6: đo bits/token theo TỪNG LĨNH VỰC trên phần val giữ riêng (200K token cuối shard cuối mỗi nguồn — trainer
không bao giờ lấy mẫu train ở đó) → biết C6 có thật sự giúp CODE không, và tiếng Việt có bị quên không.

Khác val_bits của trainer (20 lô ngẫu nhiên trộn theo tỉ lệ): ở đây đi TUẦN TỰ hết các cửa sổ ctx không chồng nhau
trong vùng val của từng nguồn → số chính xác, so được giữa các checkpoint.

    PYTHONPATH=scripts:src python scripts/c6_domain_bits.py --ckpt /workspace/c6/cpt_b_stable.pt ckpt/c6.pt \
        --out out/c6_domain_bits.json
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F

import train_hybrid_scale as T

SRC = {   # nguồn → nhóm
    "data/cx3/gh": "code", "data/cx3/opc_algo": "code", "data/cx3/opc_snip": "code", "data/cx3/opc_files": "code",
    "data/cpt/code": "code", "data/cx3/web": "tài liệu kỹ thuật", "data/cpt/en_edu": "Anh giáo dục",
    "data/cpt/vi_web": "tiếng Việt", "data/cpt/long": "tiếng Việt", "data/cpt/vi_wiki": "tiếng Việt", "data/cx3/vi": "tiếng Việt",
    "data/cpt/math": "toán",
}


@torch.no_grad()
def bits_of(m, src, ctx, bs, V):
    d = T.ShardData(src, ctx)
    mm, lo, hi = d.mm[-1], d.val_from, d.lens[-1]
    offs = list(range(lo, hi - ctx - 1, ctx))
    tot, n = 0.0, 0
    for i in range(0, len(offs), bs):
        x = torch.from_numpy(np.stack([np.asarray(mm[o:o + ctx + 1], dtype=np.int64) for o in offs[i:i + bs]])).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo_ = m(x[:, :-1])
        tot += float(F.cross_entropy(lo_.float().reshape(-1, V), x[:, 1:].reshape(-1), reduction="sum")); n += x[:, 1:].numel()
    return tot / n / math.log(2), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--out", default="out/c6_domain_bits.json")
    ap.add_argument("--bf16", action="store_true", help="trọng số bf16 (~5GB) → đo được cạnh lúc đang train; lệch fp32 cỡ 1e-3 bit")
    a = ap.parse_args()
    V = 32002
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for ck in a.ckpt:
        t0 = time.time()
        o = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
        mc = {k: o["cfg"][k] for k in T.MODEL_KEYS if k in o["cfg"]}
        mc["attn_layers"] = T.attn_layers_of(mc)
        m = T.build_from_cfg(mc, V)
        m.load_state_dict({k: v.float() for k, v in o["state"].items()})
        m = (m.bfloat16() if a.bf16 else m).cuda().eval()
        tag = f"{os.path.basename(ck)}@{o.get('step', '?')}" + ("·bf16" if a.bf16 else "")
        del o
        per = {s: bits_of(m, s, a.ctx, a.bs, V) for s in SRC}
        grp = {}
        for s, (b, n) in per.items():
            g = grp.setdefault(SRC[s], [0.0, 0]); g[0] += b * n; g[1] += n
        res[tag] = {"nhóm": {g: round(v[0] / v[1], 4) for g, v in grp.items()},
                    "nguồn": {s: round(b, 4) for s, (b, _) in per.items()}}
        print(f"### {tag} · {time.time()-t0:.0f}s · " + " · ".join(f"{g} {v}" for g, v in res[tag]["nhóm"].items()), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
