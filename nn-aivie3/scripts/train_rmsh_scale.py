"""nn-aivie3 · SCALE — train MultimodalRMSH (ảnh+caption) QUY-MÔ trên 1×GPU CUDA.

Nối kernel `mamba-ssm` (selective-scan CUDA, nhanh ~10-50× vòng-lặp thuần-PyTorch của MPS)
qua HybridBackbone(use_kernel=True). Tái-dùng khung train đã kiểm ở train_hybrid_scale.py:
bf16 autocast · grad-checkpointing · cosine-LR + warmup · fused-AdamW · resume · log tok/s.

CÀI kernel (trên box CUDA):  bash scripts/setup_container.sh   (đã ghim causal-conv1d 1.4.0)
SMOKE tại-chỗ (MPS/CPU, data TỔNG-HỢP, cấu-hình tí hon):
    PYTHONPATH=src python nn-aivie3/scripts/train_rmsh_scale.py --smoke \
        --d 128 --layers 4 --attn-every 2 --img 64 --patch 16 --nq 8 --steps 20
CHẠY THẬT (H100, data ảnh-caption đã build):
    PYTHONPATH=src python nn-aivie3/scripts/train_rmsh_scale.py --data /workspace/data/rmsh_vi \
        --d 1024 --layers 24 --attn-every 6 --img 224 --patch 16 --nq 32 \
        --bs 32 --grad-accum 8 --lr 3e-4 --tokens 8e9 --grad-ckpt --out /workspace/checkpoints/rmsh_mini.pt

DATA THẬT: thư-mục chứa shard_*.npz, mỗi shard:  img=uint8[N,3,H,W] · cap=int16[N,L] (token
caption tiếng Việt, đã tokenize bằng BPE của ta). Build trên M4 rồi upload (TK_101: FPT
serverless CHẶN ssh/rsync → dùng HF Hub hoặc bundle như gpu_bundle.tgz).
"""
import os, sys, glob, time, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
from nn_aivie3.rmsh import MultimodalRMSH
from nn_aivie3.model import HAVE_MAMBA_KERNEL
T0 = time.time()


def lr_at(step, warmup, total, lr, min_lr):
    if step < warmup:
        return lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * r))


class MMData:
    """shard_*.npz (img uint8[N,3,H,W], cap int16[N,L]); nếu không có → TỔNG-HỢP (smoke)."""
    def __init__(self, path, img, vocab, cap_len, smoke):
        self.img, self.vocab, self.L = img, vocab, cap_len
        self.paths = sorted(glob.glob(os.path.join(path, "shard_*.npz"))) if path else []
        self.smoke = smoke or not self.paths
        if self.smoke:
            print(f"[{time.time()-T0:.0f}s] DATA tổng-hợp (smoke) — CHƯA có shard thật ở '{path}'", flush=True)
        else:
            self.mm = [np.load(p, mmap_mode="r") for p in self.paths]
            self.lens = [len(m["cap"]) for m in self.mm]
            print(f"[{time.time()-T0:.0f}s] DATA: {len(self.paths)} shard · "
                  f"{sum(self.lens)/1e3:.0f}K cặp ảnh-caption", flush=True)

    def batch(self, bs, dev, gen):
        if self.smoke:
            img = torch.rand(bs, 3, self.img, self.img)
            cap = torch.from_numpy(gen.integers(0, self.vocab, (bs, self.L)).astype(np.int64))
            return img.to(dev), cap.to(dev)
        imgs, caps = [], []
        for _ in range(bs):
            si = int(gen.integers(0, len(self.mm))); m = self.mm[si]
            j = int(gen.integers(0, self.lens[si]))
            imgs.append(np.asarray(m["img"][j], dtype=np.float32) / 255.0)
            caps.append(np.asarray(m["cap"][j], dtype=np.int64))
        img = torch.from_numpy(np.stack(imgs)); cap = torch.from_numpy(np.stack(caps))
        return img.to(dev), cap.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=""); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--vocab", type=int, default=32002); ap.add_argument("--cap-len", type=int, default=32)
    ap.add_argument("--d", type=int, default=1024); ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--attn-every", type=int, default=6); ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--heads", type=int, default=16); ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--d-v", type=int, default=768); ap.add_argument("--nq", type=int, default=32)
    ap.add_argument("--img", type=int, default=224); ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--tokens", type=float, default=8e9); ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--init-from", default=None); ap.add_argument("--out", default="checkpoints/rmsh_mini.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    use_bf16 = dev == "cuda"
    use_kernel = dev == "cuda"                                        # kernel chỉ trên CUDA
    attn_layers = list(range(a.attn_every - 1, a.layers, a.attn_every)) if a.attn_every > 0 else []
    data = MMData(a.data, a.img, a.vocab, a.cap_len, a.smoke)

    m = MultimodalRMSH(vocab=a.vocab, d=a.d, n_layer=a.layers, attn_layers=attn_layers,
                       d_state=a.d_state, heads=a.heads, window=a.window, d_v=a.d_v,
                       n_query=a.nq, img=a.img, patch=a.patch, use_kernel=use_kernel).to(dev)
    m.backbone.grad_ckpt = a.grad_ckpt
    step0, resume_opt = 0, None
    if a.init_from and os.path.exists(a.init_from):
        obj = torch.load(a.init_from, map_location="cpu")
        m.load_state_dict(obj["state"]); resume_opt = obj.get("opt"); step0 = obj.get("step", 0)
        print(f"[{time.time()-T0:.0f}s] RESUME {a.init_from} @step {step0}", flush=True)

    tok_per_step = a.bs * a.grad_accum * a.cap_len                    # token CHỮ học/step
    steps = a.steps or int(a.tokens // tok_per_step)
    npar = sum(p.numel() for p in m.parameters())
    try:
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1,
                                fused=(dev == "cuda"))
    except TypeError:
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    if resume_opt:
        opt.load_state_dict(resume_opt)
    if a.compile and dev == "cuda":
        m = torch.compile(m)

    kern = "CUDA-kernel(mamba-ssm)" if (use_kernel and HAVE_MAMBA_KERNEL) else \
        ("thuần-PyTorch (kernel THIẾU!)" if use_kernel else "thuần-PyTorch(fallback)")
    print(f"[{time.time()-T0:.0f}s] dev={dev} · RMSH {npar/1e6:.1f}M · d={a.d} L={a.layers} "
          f"attn@{attn_layers} · Mamba={kern} · vision d_v={a.d_v} nq={a.nq} img={a.img}", flush=True)
    print(f"[{time.time()-T0:.0f}s] bs={a.bs}×ga{a.grad_accum} cap_len={a.cap_len} "
          f"→ {tok_per_step} chữ/step · steps={steps} · bf16={use_bf16} grad_ckpt={a.grad_ckpt}", flush=True)

    gen = np.random.default_rng(0)
    actx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)
    m.train(); t1 = time.time(); seen = 0
    for step in range(step0 + 1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.warmup, steps, a.lr, a.min_lr)
        opt.zero_grad(set_to_none=True); lsum = 0.0
        for _ in range(a.grad_accum):
            img, cap = data.batch(a.bs, dev, gen)
            with actx:
                loss, _ = m(img, cap); loss = loss / a.grad_accum
            loss.backward(); lsum += loss.detach().item() * a.grad_accum
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); seen += tok_per_step
        if step % a.log_every == 0:
            dt = time.time() - t1; tps = seen / max(dt, 1e-6)
            print(f"[{time.time()-T0:.0f}s] step {step}/{steps} · loss {lsum:.3f} · "
                  f"{tps:.0f} chữ/s · lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if a.save_every and step % a.save_every == 0:
            sd = (m._orig_mod if hasattr(m, "_orig_mod") else m).state_dict()
            torch.save({"state": sd, "opt": opt.state_dict(), "step": step,
                        "cfg": vars(a)}, a.out)
            print(f"[{time.time()-T0:.0f}s] SAVE {a.out} @step {step}", flush=True)
    print(f"[{time.time()-T0:.0f}s] XONG {steps} step.", flush=True)


if __name__ == "__main__":
    main()
