"""nn-aivie3 · M1 — Front-end thị-giác (VFE) TỪ-SCRATCH.

Chuỗi: ảnh → patchify(conv) → ViT-S encoder (full-attn trên lưới patch) →
PerceiverResampler (m query nén → k token/frame CỐ-ĐỊNH) → Projector(d_model).

k token/frame cố-định = chốt-trên chi-phí mỗi frame ở backbone (điều-kiện cần cho
24fps, xem TK3_000 §2.1). VFE from-scratch train được ở cỡ nhỏ; giữ ablation
warm-start (nạp encoder mở, đông-băng) làm baseline — §7 rủi-ro.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderBlock(nn.Module):
    """Transformer encoder chuẩn (non-causal): MHSA + MLP, pre-norm."""
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class VisionEncoder(nn.Module):
    """ViT-S tí-hon từ-scratch: (b,3,H,W) → (b, N, d_v) token lưới patch."""
    def __init__(self, d_v=256, patch=16, img=224, heads=4, layers=6):
        super().__init__()
        self.patch, self.grid = patch, img // patch
        n_pos = self.grid * self.grid
        self.proj = nn.Conv2d(3, d_v, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_pos, d_v)); nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_EncoderBlock(d_v, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_v)

    def forward(self, img):                            # (b,3,H,W)
        x = self.proj(img).flatten(2).transpose(1, 2)  # (b, N, d_v)
        x = x + self.pos[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class PerceiverResampler(nn.Module):
    """m learned-query cross-attend lưới patch → k=m token/frame cố-định."""
    def __init__(self, d_v=256, n_query=16, heads=4, layers=2):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, n_query, d_v)); nn.init.normal_(self.q, std=0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "nq": nn.LayerNorm(d_v), "nk": nn.LayerNorm(d_v),
                "attn": nn.MultiheadAttention(d_v, heads, batch_first=True),
                "nm": nn.LayerNorm(d_v),
                "mlp": nn.Sequential(nn.Linear(d_v, 4 * d_v), nn.GELU(), nn.Linear(4 * d_v, d_v)),
            }) for _ in range(layers)])
        self.n_query = n_query

    def forward(self, grid):                           # (b, N, d_v) → (b, m, d_v)
        b = grid.shape[0]
        q = self.q.expand(b, -1, -1)
        for L in self.layers:
            a, _ = L["attn"](L["nq"](q), L["nk"](grid), L["nk"](grid), need_weights=False)
            q = q + a
            q = q + L["mlp"](L["nm"](q))
        return q


class VisionFrontEnd(nn.Module):
    """VFE trọn gói: ảnh → k token đã chiếu về d_model (+ modality embedding)."""
    def __init__(self, d_model, d_v=256, n_query=16, patch=16, img=224,
                 enc_layers=6, res_layers=2, heads=4):
        super().__init__()
        self.encoder = VisionEncoder(d_v, patch, img, heads, enc_layers)
        self.resampler = PerceiverResampler(d_v, n_query, heads, res_layers)
        self.proj = nn.Linear(d_v, d_model)
        self.k = n_query

    def forward(self, img):                            # (b,3,H,W) → (b, k, d_model)
        return self.proj(self.resampler(self.encoder(img)))
