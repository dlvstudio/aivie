"""Cắt checkpoint train (có optimizer, fp32) → bản chạy thử gọn: chỉ cfg + state bf16 + step (2,6B ≈ 5GB, 1.3B ≈ 2,6GB).
    python scripts/slim_ckpt.py ckpt/cpt_a1.pt ckpt/cpt_a1_bf16.pt
Chạy trên CPU (mmap) — không đụng GPU đang train. hybrid_infer.load() nạp bf16 vào model fp32 bình thường."""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
obj = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
out = {"cfg": obj["cfg"], "step": obj.get("step"),
       "state": {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in obj["state"].items()}}
torch.save(out, dst + ".tmp")
import os; os.replace(dst + ".tmp", dst)
n = sum(v.numel() for v in out["state"].values())
print(f"### {dst} · {n/1e9:.3f}B tham số · step {out['step']} · {os.path.getsize(dst)/1e9:.2f}GB")
