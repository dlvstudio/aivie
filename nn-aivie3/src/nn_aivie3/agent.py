"""nn-aivie3 · M3 — RUNTIME REALTIME: perceive O(1) / decide anytime.

Ghép các mảnh đã kiểm-chứng: VFE (M1) → HybridBackbone StreamState (M0) → ActionHead.
Hai đường KHỚP nhau (bất-biến train==serve):
  • forward()  : nuốt cả stream theo-chuỗi, action-logits MỖI frame → TRAIN (GPU).
  • perceive()/decide() : nuốt frame TỪNG CÁI qua StreamState (O(1)); decide() đọc
    state đang-ấm ra hành-động BẤT-CỨ-LÚC-NÀO trong 1 bước head → SERVE realtime (M4).

Đầu hành-động là O(1)/tick, độc-lập độ-dài luồng ⇒ "quyết-định gần tức-thì" là
thuộc-tính hệ-thống (TK3_000 §9.3). Bài đoán-HƯỚNG-chuyển-động buộc dùng bộ-nhớ
streaming (so frame hiện-tại với trước) — chứng-minh vòng quyết-định đọc được ngữ-cảnh sống.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from nn_aivie3.model import HybridBackbone
from nn_aivie3.vision import VisionFrontEnd


class StreamingActionModel(nn.Module):
    def __init__(self, n_actions=4, d=128, n_layer=6, attn_layers=(2, 4), d_state=16,
                 heads=4, window=256, d_v=128, n_query=4, img=48, patch=8, tokens="grid"):
        super().__init__()
        self.d = d
        self.tokens = tokens                              # "grid"=giữ-vị-trí (bài học M2.5) | "resampler"=nén
        self.vfe = VisionFrontEnd(d, d_v, n_query, patch, img, heads=heads)
        self.k = (img // patch) ** 2 if tokens == "grid" else n_query
        self.vis_ln = nn.LayerNorm(d)
        self.backbone = HybridBackbone(d, n_layer, attn_layers, d_state, heads, window)
        self.action = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_actions))

    def _frame_tokens(self, imgs):                        # imgs:(N,3,H,W) → (N,k,d)
        if self.tokens == "grid":                         # patchify+encoder → proj (GIỮ vị-trí, bỏ resampler)
            return self.vis_ln(self.vfe.proj(self.vfe.encoder(imgs)))
        return self.vis_ln(self.vfe(imgs))                # resampler (nén)

    def _tok(self, frames):                               # frames:(b,T,3,H,W) → (b,T,k,d)
        b, T = frames.shape[:2]
        return self._frame_tokens(frames.flatten(0, 1)).view(b, T, self.k, self.d)

    def forward(self, frames):                            # TRAIN → action-logits mỗi frame (b,T,A)
        b, T = frames.shape[:2]
        x = self._tok(frames).reshape(b, T * self.k, self.d)
        h = self.backbone(x)                              # (b, T*k, d)
        last = h.view(b, T, self.k, self.d)[:, :, -1]     # hidden sau token cuối MỖI frame
        return self.action(last)                          # (b,T,A)

    # ── SERVE realtime ──
    @torch.no_grad()
    def new_stream(self, device, batch=1):
        return self.backbone.init_stream(batch, device)

    @torch.no_grad()
    def perceive(self, frame, state):                     # frame:(b,3,H,W) → hidden (b,d), O(1)
        toks = self._frame_tokens(frame)                  # (b,k,d)
        h = None
        for j in range(self.k):
            h = self.backbone.stream_step(toks[:, j], state)
        return h                                          # hidden sau frame này (đọc để decide)

    @torch.no_grad()
    def decide(self, hidden):                             # anytime: hidden→hành-động, O(1)
        return self.action(hidden).argmax(-1)


class SalientActionModel(nn.Module):
    """M3.5 — GIẢI đánh-đổi token/vị-trí: gate salience per-patch chọn VÀI patch quan-trọng
    (GIỮ vị-trí, ÍT token). Train dense-gated (học được, kiểu M2.5); serve giữ top-k patch."""

    def __init__(self, n_actions=4, d=128, n_layer=6, attn_layers=(2, 4), d_state=16,
                 heads=4, window=128, d_v=128, img=32, patch=8, l1=0.02, gate_mode="softmax"):
        super().__init__()
        self.d, self.l1, self.gate_mode = d, l1, gate_mode
        self.np = (img // patch) ** 2                     # số patch/frame
        self.vfe = VisionFrontEnd(d, d_v, 4, patch, img, heads=heads)   # dùng encoder+proj
        self.vis_ln = nn.LayerNorm(d)
        self.gate = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.backbone = HybridBackbone(d, n_layer, attn_layers, d_state, heads, window)
        self.action = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_actions))

    def _grid(self, imgs):                                # (N,3,H,W) → (N,np,d) GIỮ vị-trí
        return self.vis_ln(self.vfe.proj(self.vfe.encoder(imgs)))

    def forward(self, frames, topk=None):
        """TRAIN (topk=None): dense-gated. EVAL (topk=k): chỉ giữ k patch salient nhất/frame."""
        b, T = frames.shape[:2]
        grid = self._grid(frames.flatten(0, 1)).view(b, T, self.np, self.d)
        score = self.gate(grid)                            # (b,T,np,1)
        if self.gate_mode == "softmax":                   # PHÂN-BỔ ngân-sách cố-định trên patch
            g = torch.softmax(score, dim=2) * self.np      # dồn vào patch có tin (không co-đều)
        else:
            g = torch.sigmoid(score)
        if topk is not None:                              # giữ top-k patch salient nhất (mask phần còn lại)
            thr = g.topk(topk, dim=2).values[:, :, -1:, :]
            g = g * (g >= thr).float()
        x = (grid * g).reshape(b, T * self.np, self.d)
        h = self.backbone(x).view(b, T, self.np, self.d)[:, :, -1]      # đọc sau frame
        return self.action(h), g                          # (b,T,A), gate


class GumbelSelectActionModel(nn.Module):
    """M3.5 (đúng) — CHỌN-TOKEN thích-nghi KHẢ-VI bằng Gumbel-softmax. m slot, mỗi slot
    chọn ~1 patch (near-one-hot) GIỮ vị-trí → chỉ m token/frame (rẻ, realtime) mà bám đúng
    patch quan-trọng. Train tau cao→thấp (mềm→cứng); serve hard=argmax. Tránh sụp bootstrapping
    của L1/hard-topk (đường khả-vi xuyên suốt)."""

    def __init__(self, n_actions=4, m=4, d=128, n_layer=6, attn_layers=(2, 4), d_state=16,
                 heads=4, window=128, d_v=128, img=32, patch=8):
        super().__init__()
        self.d, self.m = d, m
        self.np = (img // patch) ** 2
        self.vfe = VisionFrontEnd(d, d_v, 4, patch, img, heads=heads)
        self.vis_ln = nn.LayerNorm(d)
        self.sel = nn.Linear(d, m)                        # patch → điểm cho m slot
        self.backbone = HybridBackbone(d, n_layer, attn_layers, d_state, heads, window)
        self.action = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_actions))

    def _grid(self, imgs):
        return self.vis_ln(self.vfe.proj(self.vfe.encoder(imgs)))       # (N,np,d) GIỮ vị-trí

    def forward(self, frames, tau=1.0, hard=False):
        b, T = frames.shape[:2]
        grid = self._grid(frames.flatten(0, 1)).view(b, T, self.np, self.d)
        logits = self.sel(grid)                           # (b,T,np,m)
        w = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, dim=2)   # chọn patch/slot
        slots = torch.einsum('btpm,btpd->btmd', w, grid)  # (b,T,m,d) — m token đã chọn (giữ vị-trí)
        x = slots.reshape(b, T * self.m, self.d)
        h = self.backbone(x).view(b, T, self.m, self.d)[:, :, -1]
        return self.action(h), w                          # (b,T,A), trọng-số chọn
