"""Đo bits/token trên tập kiểm tra CỐ ĐỊNH theo từng nhóm dữ liệu (cùng cửa sổ cho mọi checkpoint) — để so checkpoint
công bằng khi `bits` trong log train dao động (log lấy mẫu val ngẫu nhiên, thành phần nhóm đổi mỗi lần).

    python scripts/eval_fixed_val.py --ckpts ckpt/hybrid_2p6b_x.pt,ckpt/cpt_a1.pt --dirs data/cpt/vi_web,data/cpt/code --ctx 2048
Nạp trọng số bf16 (≈5GB) để chạy được cạnh tiến trình train đang chiếm GPU.
"""
import argparse, glob, json, math, os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")]
import numpy as np, torch, torch.nn.functional as F


def windows(d, ctx, n, val_tokens=200_000):
    m = np.load(sorted(glob.glob(os.path.join(d, "shard_*.npy")))[-1], mmap_mode="r")
    lo = max(0, len(m) - val_tokens); hi = len(m) - ctx - 1
    offs = np.linspace(lo, hi, n).astype(int)                     # cùng cửa sổ trong đuôi val như trình train
    return torch.from_numpy(np.stack([np.asarray(m[o:o + ctx + 1], dtype=np.int64) for o in offs]))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True); ap.add_argument("--dirs", required=True)
    ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--rope-base", type=float, default=0, help="ép RoPE base (0 = theo ckpt)")
    a = ap.parse_args()
    import train_hybrid_scale as ths
    data = {os.path.basename(d.rstrip("/")): windows(d, a.ctx, a.n) for d in a.dirs.split(",")}
    res = {}
    for c in a.ckpts.split(","):
        obj = torch.load(c, map_location="cpu", weights_only=False, mmap=True)
        cfg = dict(obj["cfg"]); step = obj.get("step")
        if a.rope_base: cfg["rope_base"] = a.rope_base
        m = ths.build_from_cfg(cfg); m.load_state_dict(obj["state"]); del obj
        m = m.bfloat16().cuda().eval()
        r = {}
        for k, x in data.items():
            tot = 0.0
            for i in range(0, len(x), 2):
                b = x[i:i + 2].cuda()
                lo = m(b[:, :-1]).float()
                tot += float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]), b[:, 1:].reshape(-1))) * len(b)
            r[k] = round(tot / len(x) / math.log(2), 4)
        r["TB"] = round(sum(r.values()) / len(r), 4)
        res[f"{os.path.basename(c)}@{step}"] = r
        print(f"{os.path.basename(c)} @step {step}: " + " · ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
        del m; torch.cuda.empty_cache()
    print("### " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
