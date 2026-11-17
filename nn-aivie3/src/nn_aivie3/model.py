"""nn-aivie3 · M1 — Backbone Hybrid đa-phương-thức (RMSH) với HAI đường khớp nhau:

  • forward()      : theo-chuỗi, CÓ gradient  → dùng để TRAIN (GPU).
  • stream_prefill/stream_step : hồi-quy giữ-state → dùng để SERVE realtime (M4).

Bất-biến (đã kiểm ở smoke): hai đường cho ra CÙNG kết-quả (cùng weight) → train một
lần, serve realtime không tái-huấn. Mamba dùng MambaBlock(train)/MambaStep(serve)
[[nn-aivie3-rmsh-streaming]]; Attention dùng StreamAttn (một module, hai lối gọi).
"""
from __future__ import annotations
import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "scripts"))   # train_mamba.MambaBlock
from train_mamba import MambaBlock                                      # noqa: E402
from nn_aivie3.streaming import MambaStep, RMSNorm                      # noqa: E402

# ── Đường CUDA nhanh: kernel selective-scan chính-thức (mamba-ssm) nếu cài được ──
# Giống train_hybrid_scale.py: kernel trên GPU (nhanh ~10-50×), fallback thuần-PyTorch trên
# MPS/CPU để smoke tại-chỗ. LƯU: kernel chỉ dùng cho đường FORWARD (train scan); streaming
# serve (MambaStep) vẫn đi fallback pure-PyTorch (đúng, đủ nhanh khi phục-vụ) — dựng
# use_kernel=True KHI TRAIN trên CUDA, mặc-định False để mọi PoC streaming chạy được.
try:
    from mamba_ssm import Mamba as _MambaKernel                        # noqa: N814
    HAVE_MAMBA_KERNEL = True
except Exception:
    HAVE_MAMBA_KERNEL = False


def make_mamba(d, d_state, use_kernel):
    """Kernel CUDA khi bật+có (train nhanh) → nếu không, khối thuần-PyTorch của ta (chạy MPS)."""
    if use_kernel and HAVE_MAMBA_KERNEL:
        return _MambaKernel(d_model=d, d_state=d_state, d_conv=4, expand=2)
    return MambaBlock(d, d_state)


# ── RoPE (vị-trí tuyệt-đối → streamable) ──────────────────────────────────────────
def _rope(x, pos, base=10000.0):                      # x:(b,h,l,hd) · pos:(l,)
    b, h, l, hd = x.shape
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    ang = torch.outer(pos.float(), inv)               # (l, hd/2)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


class StreamAttn(nn.Module):
    """Self-attention nhân-quả RoPE, hai lối: forward (batched) · step (KV-cache cửa-sổ)."""

    def __init__(self, d, heads=4, window=4096):
        super().__init__()
        self.h, self.hd, self.window = heads, d // heads, window
        assert self.hd % 2 == 0, "head_dim phải chẵn cho RoPE"
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, x, pos0=0):                     # x:(b,l,d) — đường TRAIN
        b, l, d = x.shape
        pos = torch.arange(pos0, pos0 + l, device=x.device)
        q = self.wq(x).view(b, l, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(b, l, self.h, self.hd).transpose(1, 2)
        v = self.wv(x).view(b, l, self.h, self.hd).transpose(1, 2)
        q, k = _rope(q, pos), _rope(k, pos)
        if self.window >= l:                          # đủ rộng → full-causal (khớp step cửa-sổ rộng)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:                                         # cửa-sổ trượt: i thấy j khi 0≤i-j<W (KHỚP .step)
            i = torch.arange(l, device=x.device)[:, None]
            j = torch.arange(l, device=x.device)[None, :]
            allow = (i - j >= 0) & (i - j < self.window)          # (l,l) bool: True=được dự
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[None, None])
        return self.wo(o.transpose(1, 2).reshape(b, l, d))

    @torch.no_grad()
    def step(self, x_t, state):                       # x_t:(b,d) — đường SERVE
        b, d = x_t.shape
        t = state["pos"]
        pos = torch.tensor([t], device=x_t.device)
        q = self.wq(x_t).view(b, 1, self.h, self.hd).transpose(1, 2)
        k = self.wk(x_t).view(b, 1, self.h, self.hd).transpose(1, 2)
        v = self.wv(x_t).view(b, 1, self.h, self.hd).transpose(1, 2)
        q, k = _rope(q, pos), _rope(k, pos)
        K, V = state.get("K"), state.get("V")
        K = k if K is None else torch.cat([K, k], 2)[:, :, -self.window:]
        V = v if V is None else torch.cat([V, v], 2)[:, :, -self.window:]
        state["K"], state["V"] = K, V
        o = F.scaled_dot_product_attention(q, K, V)   # q chỉ 1 token → không cần mask
        return self.wo(o.transpose(1, 2).reshape(b, d))


class HybridBackbone(nn.Module):
    """Xen Mamba (mặc-định) và StreamAttn ở các tầng trong `attn_layers`."""

    def __init__(self, d, n_layer, attn_layers=(), d_state=16, heads=4, window=4096,
                 use_kernel=False):
        super().__init__()
        self.aset = set(attn_layers)
        self.window = window
        self.use_kernel = use_kernel and HAVE_MAMBA_KERNEL              # bật kernel CUDA khi train
        self.blocks = nn.ModuleList([
            StreamAttn(d, heads, window) if i in self.aset else make_mamba(d, d_state, self.use_kernel)
            for i in range(n_layer)])
        self.norms = nn.ModuleList([RMSNorm(d) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d)
        self.grad_ckpt = False                        # bật khi train quy-mô (tiết-kiệm VRAM)

    def forward(self, x):                             # (b,l,d) → (b,l,d) — TRAIN
        for blk, nm in zip(self.blocks, self.norms):
            if self.grad_ckpt and self.training:
                x = x + torch.utils.checkpoint.checkpoint(lambda t, b=blk, n=nm: b(n(t)), x,
                                                          use_reentrant=False)
            else:
                x = x + blk(nm(x))
        return self.norm_f(x)

    # ── đường SERVE: prefill (nuốt cả cụm token) rồi step (từng token) ──
    def init_stream(self, batch, device, dtype=torch.float32):
        if self.use_kernel:
            raise RuntimeError(
                "Streaming (MambaStep) chưa nối mamba-ssm kernel step. Dựng backbone với "
                "use_kernel=False để serve realtime (fallback pure-PyTorch), hoặc dùng "
                "mamba_ssm .step() cho serve trên CUDA (TODO). Kernel chỉ dành đường TRAIN.")
        steps = []
        for i, blk in enumerate(self.blocks):
            if i in self.aset:
                steps.append(blk)                     # StreamAttn dùng lại chính nó
            else:
                steps.append(MambaStep.load_from(blk).to(device))
        state = {"steps": steps, "layer": []}
        for i, blk in enumerate(self.blocks):
            if i in self.aset:
                state["layer"].append({"pos": 0})
            else:
                state["layer"].append(steps[i].init_state(batch, device, dtype))
        return state

    @torch.no_grad()
    def stream_step(self, x_t, state):                # (b,d) → (b,d) một token
        h = x_t
        for i, (step_mod, nm) in enumerate(zip(state["steps"], self.norms)):
            ls = state["layer"][i]
            h = h + step_mod.step(nm(h), ls)
            if i in self.aset:
                ls["pos"] += 1
        return self.norm_f(h)

    @torch.no_grad()
    def stream_prefill(self, xs, state):              # (b,l,d) → (b,l,d)
        return torch.stack([self.stream_step(xs[:, t], state) for t in range(xs.shape[1])], 1)
