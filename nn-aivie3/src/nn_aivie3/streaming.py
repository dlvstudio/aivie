"""nn-aivie3 · M0 — StreamState + MambaStep: step hồi-quy giữ-state cho SSM.

Ý-tưởng lõi (RMSH, xem TK3_000_KIEN_TRUC.md §2.4/§4): `MambaBlock` của ta
(scripts/train_mamba.py) tính selective-scan reset h=0 mỗi forward — đúng để TRAIN
theo chuỗi, nhưng KHÔNG dùng lại được cho luồng realtime. `MambaStep` ở đây làm
đúng cùng phép-tính đó nhưng ở dạng HỒI-QUY MỘT-TOKEN, GIỮ state (h + ring-buffer
conv) giữa các lần gọi → nuốt luồng 24fps ở O(1)/token, RAM phẳng.

QUAN TRỌNG: MambaStep dùng ĐÚNG các tham-số của MambaBlock (in_proj, conv1d, x_proj,
dt_proj, A_log, D, out_proj) → `MambaStep.load_from(block)` nạp thẳng weight đã train.
Nghĩa là bản train (theo chuỗi, nhanh trên GPU) và bản serve (step, realtime trên M4)
là CÙNG MỘT model, không tái-huấn.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaStep(nn.Module):
    """Bản step hồi-quy của scripts/train_mamba.py:MambaBlock (khớp weight 1-1)."""

    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank=None):
        super().__init__()
        self.d = d
        self.d_inner = expand * d
        self.dt_rank = dt_rank or math.ceil(d / 16)
        self.d_state = d_state
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)

    @classmethod
    def load_from(cls, block: nn.Module) -> "MambaStep":
        """Dựng MambaStep từ một MambaBlock đã train (chia-sẻ trọng-số)."""
        m = cls(d=block.out_proj.out_features, d_state=block.d_state, d_conv=block.d_conv,
                expand=block.d_inner // block.out_proj.out_features)
        m.load_state_dict(block.state_dict(), strict=True)
        return m

    def init_state(self, batch: int, device, dtype=torch.float32):
        """Trạng-thái RỖNG cho một luồng mới. Kích-thước CỐ-ĐỊNH (không theo T)."""
        h = torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype)
        conv_buf = torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device, dtype=dtype)
        return {"h": h, "conv_buf": conv_buf}

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: dict) -> torch.Tensor:
        """Một token. x_t: (b, d) → out: (b, d). Cập-nhật state TẠI-CHỖ, O(1)."""
        b = x_t.shape[0]
        xz = self.in_proj(x_t)                                  # (b, 2*d_inner)
        xx, z = xz.chunk(2, dim=-1)                             # (b, d_inner) mỗi

        # ── causal depthwise conv qua ring-buffer (thay conv1d theo-chuỗi) ──
        buf = state["conv_buf"]                                 # (b, d_inner, d_conv-1)
        window = torch.cat([buf, xx.unsqueeze(-1)], dim=-1)     # (b, d_inner, d_conv)
        w = self.conv1d.weight.squeeze(1)                       # (d_inner, d_conv)
        xx = (window * w.unsqueeze(0)).sum(-1) + self.conv1d.bias
        state["conv_buf"] = window[..., 1:]                     # trượt cửa-sổ
        xx = F.silu(xx)

        # ── selective SSM một-bước: h = dA*h + dBx; y = (h*C).sum + xx*D ──
        A = -torch.exp(self.A_log)                              # (d_inner, n)
        db = self.x_proj(xx)                                    # (b, dt_rank+2n)
        dt, B, C = db.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                       # (b, d_inner)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))       # (b, d_inner, n)
        dBx = dt.unsqueeze(-1) * B.unsqueeze(1) * xx.unsqueeze(-1)   # (b, d_inner, n)
        h = dA * state["h"] + dBx
        state["h"] = h
        y = (h * C.unsqueeze(1)).sum(-1) + xx * self.D          # (b, d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class StreamState:
    """Bộ-nhớ-làm-việc tri-giác của cả backbone: gom state mọi tầng Mamba +
    KV cửa-sổ-trượt cho các tầng Attn. Kích-thước CHẶN-TRÊN (không theo độ-dài luồng)."""

    def __init__(self, mamba_steps, attn_window: int, batch: int, device, dtype=torch.float32):
        self.states = [ms.init_state(batch, device, dtype) for ms in mamba_steps]
        self.attn_window = attn_window
        self.kv = {}                                           # layer_idx -> (K,V) cửa-sổ
        self.n_tokens = 0

    def push_kv(self, layer_idx, k_t, v_t):
        K, V = self.kv.get(layer_idx, (None, None))
        k_t = k_t.unsqueeze(1); v_t = v_t.unsqueeze(1)         # (b,1,·)
        K = k_t if K is None else torch.cat([K, k_t], 1)[:, -self.attn_window:]
        V = v_t if V is None else torch.cat([V, v_t], 1)[:, -self.attn_window:]
        self.kv[layer_idx] = (K, V)
        return K, V
