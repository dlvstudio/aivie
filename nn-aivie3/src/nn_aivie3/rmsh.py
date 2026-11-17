"""nn-aivie3 · M1 — MultimodalRMSH: lắp VFE + backbone Hybrid + LM head.

Một luồng token DUY-NHẤT: [<img> · k token thị-giác][chữ...]. Modality-embedding
báo backbone loại token. Train = forward theo-chuỗi (loss chữ). Serve = prefill
(hình+prompt) rồi step sinh chữ từ state đã-ấm (trễ token-đầu ~ vài token).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_aivie3.model import HybridBackbone
from nn_aivie3.vision import VisionFrontEnd

MOD_TEXT, MOD_VISION = 0, 1


class MultimodalRMSH(nn.Module):
    def __init__(self, vocab, d=256, n_layer=8, attn_layers=(2, 5), d_state=16,
                 heads=4, window=4096, d_v=256, n_query=16, img=224, patch=16,
                 max_frames=64, use_kernel=False):
        super().__init__()
        self.d, self.V = d, vocab
        self.tok_emb = nn.Embedding(vocab, d); nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.mod_emb = nn.Embedding(2, d); nn.init.zeros_(self.mod_emb.weight)
        self.temp_emb = nn.Embedding(max_frames, d)                     # vị-trí THỜI-GIAN/frame
        nn.init.normal_(self.temp_emb.weight, std=0.20)                 # scale ĐỦ LỚN để attn bắt chỉ-số
        self.vfe = VisionFrontEnd(d, d_v, n_query, patch, img,
                                  heads=heads)
        self.vis_ln = nn.LayerNorm(d)                                   # chuẩn scale token thị-giác (cân với temp/mod)
        self.backbone = HybridBackbone(d, n_layer, attn_layers, d_state, heads, window,
                                       use_kernel=use_kernel)
        self.k = self.vfe.k

    # ── embed một chuỗi hỗn-hợp (hình rồi chữ) → (b, L, d) + nhãn loss ──
    def _embed(self, img, text_ids):
        vis = self.vis_ln(self.vfe(img)) + self.mod_emb.weight[MOD_VISION]   # (b,k,d)
        txt = self.tok_emb(text_ids) + self.mod_emb.weight[MOD_TEXT]    # (b,T,d)
        return torch.cat([vis, txt], 1)                                 # (b,k+T,d)

    def forward(self, img, text_ids):
        """TRAIN: trả loss next-token TRÊN PHẦN CHỮ (bỏ vị-trí hình)."""
        x = self._embed(img, text_ids)
        h = self.backbone(x)                                            # (b,k+T,d)
        logits = h @ self.tok_emb.weight.T                              # weight-tied
        k = self.k
        # dự-đoán token chữ kế: dùng hidden tại (k-1 .. k+T-2) → nhãn text_ids[:,0:]
        pred = logits[:, k - 1: -1]                                     # (b,T,V)
        loss = F.cross_entropy(pred.reshape(-1, self.V), text_ids.reshape(-1))
        return loss, logits

    # ── ĐA-FRAME (M2: bộ-nhớ thời-gian) ──────────────────────────────────────────
    def embed_multi(self, frames, text_ids, query_frame=None):
        """frames:(b,T,3,H,W) · text_ids:(b,S) → (b, T*k+S, d) với temporal-emb/frame.

        query_frame:(b,) — nếu có, BƠM temp_emb[query_frame] vào token chữ CUỐI (token
        hỏi). "Hỏi frame nào" là VỊ-TRÍ THỜI-GIAN → mã bằng đúng positional-code mà frame
        đeo (chia-sẻ), KHÔNG dùng token chỉ-số rời (bài học M2 debug2).
        """
        b, T = frames.shape[:2]
        vis = self.vis_ln(self.vfe(frames.flatten(0, 1))).view(b, T, self.k, self.d)  # (b,T,k,d)
        vis = vis + self.mod_emb.weight[MOD_VISION]
        vis = vis + self.temp_emb.weight[:T][None, :, None, :]           # bind frame→index
        vis = vis.reshape(b, T * self.k, self.d)
        txt = self.tok_emb(text_ids) + self.mod_emb.weight[MOD_TEXT]
        if query_frame is not None:
            txt = txt.clone()
            txt[:, -1] = txt[:, -1] + self.temp_emb(query_frame)        # câu hỏi neo vị-trí thời-gian
        return torch.cat([vis, txt], 1)

    def logits_multi(self, frames, text_ids, query_frame=None):         # TRAIN → (b,L,V)
        x = self.embed_multi(frames, text_ids, query_frame)
        return self.backbone(x) @ self.tok_emb.weight.T

    @torch.no_grad()
    def stream_answer(self, frames, text_ids, device, query_frame=None):
        """SERVE: nuốt frame-rồi-chữ TỪNG TOKEN qua state; trả logits vị-trí cuối + state."""
        x = self.embed_multi(frames, text_ids, query_frame)
        state = self.backbone.init_stream(x.shape[0], device, x.dtype)
        h_last = self.backbone.stream_prefill(x, state)[:, -1]
        return h_last @ self.tok_emb.weight.T, state

    # ── SERVE (1 frame, M1) ──
    @torch.no_grad()
    def prefill(self, img, prompt_ids, device):
        x = self._embed(img, prompt_ids)                                # (b, k+P, d)
        state = self.backbone.init_stream(x.shape[0], device, x.dtype)
        h_last = self.backbone.stream_prefill(x, state)[:, -1]          # (b,d)
        return state, h_last

    @torch.no_grad()
    def generate(self, img, prompt_ids, max_new=16, device="cpu", greedy=True):
        state, h = self.prefill(img, prompt_ids, device)
        out = []
        for _ in range(max_new):
            logit = h @ self.tok_emb.weight.T                           # (b,V)
            nxt = logit.argmax(-1) if greedy else torch.multinomial(logit.softmax(-1), 1)[:, 0]
            out.append(nxt)
            e = self.tok_emb(nxt) + self.mod_emb.weight[MOD_TEXT]
            h = self.backbone.stream_step(e, state)
        return torch.stack(out, 1)                                      # (b, max_new)
