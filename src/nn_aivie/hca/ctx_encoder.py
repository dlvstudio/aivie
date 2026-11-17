"""§192 — ENCODER NGỮ CẢNH (TK_055 Mức 2). Self-attention nông trên embedding
tĩnh; cho VECTOR THEO VỊ TRÍ (mỗi lần xuất hiện một vector) → tách nghĩa.

Huấn luyện: `scripts/train_ctx_encoder.py` (masked-LM, MPS). Runtime: nạp
`ctx_encoder.pt` một lần, gọi `vec(emit_ids, ent_ids)` lấy vector ngữ cảnh của
thực thể trong câu. Torch là phụ thuộc TÙY CHỌN — thiếu torch/checkpoint thì
`load_ctx_encoder` trả None và retrieval rơi về vector tĩnh (§188b).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MAXLEN = 48
_DIM = 96
_VOCAB = 32002


class ContextEncoder:
    """Bọc encoder torch + tiện ích lấy vector ngữ cảnh của một token trong câu."""

    def __init__(self, model, torch_mod) -> None:
        self._m = model
        self._t = torch_mod
        # §TK_064 — maxlen THẬT của model (bảng pos), KHÔNG hằng 48; model
        # ngữ-cảnh-dài (128) mới dùng đủ cửa sổ lúc suy luận.
        try:
            self._ml = int(model.pos.num_embeddings)
        except Exception:
            self._ml = MAXLEN

    def vec(self, emit_ids, ent_ids) -> np.ndarray | None:
        """Vector ngữ cảnh (chuẩn hoá) của token thuộc `ent_ids` xuất hiện ĐẦU
        tiên trong `emit_ids`; None nếu không có token nào thuộc `ent_ids`."""
        e = [int(t) for t in emit_ids if 0 <= int(t) < _VOCAB][:MAXLEN]
        pos = next((k for k, t in enumerate(e) if t in ent_ids), None)
        if pos is None or not e:
            return None
        t = self._t
        x = t.zeros(1, MAXLEN, dtype=t.long)
        x[0, : len(e)] = t.tensor(e)
        pm = t.arange(MAXLEN)[None] >= len(e)
        with t.no_grad():
            h = self._m.hidden(x, pm)[0, pos].numpy()
        n = float(np.linalg.norm(h))
        return h / n if n > 0 else None

    def causal_last(self, out_ids) -> np.ndarray | None:
        """§TK_056 — vector LEFT-context (mask nhân quả) tại VỊ TRÍ CUỐI, để dự
        đoán token kế (dùng cho kNN-LM sinh). Trả vector chuẩn hoá hoặc None."""
        ml = self._ml
        e = [int(t) for t in out_ids if 0 <= int(t) < _VOCAB][-ml:]
        if not e:
            return None
        t = self._t
        x = t.zeros(1, ml, dtype=t.long)
        x[0, : len(e)] = t.tensor(e)
        pm = t.arange(ml)[None] >= len(e)
        with t.no_grad():
            h = self._m.hidden(x, pm, causal=True)[0, len(e) - 1].numpy()
        n = float(np.linalg.norm(h))
        return h / n if n > 0 else None

    def causal_logits_last(self, out_ids) -> np.ndarray | None:
        """§TK_061 — LOGITS ngôn ngữ (weight-tied) của token KẾ, từ LEFT-context
        (mask nhân quả) tại vị trí cuối. Dùng h chuẩn-LayerNorm (khớp lúc train),
        KHÔNG L2-chuẩn hoá. Trả vector logits[V] (np) hoặc None."""
        ml = self._ml
        e = [int(t) for t in out_ids if 0 <= int(t) < _VOCAB][-ml:]
        if not e:
            return None
        t = self._t
        x = t.zeros(1, ml, dtype=t.long)
        x[0, : len(e)] = t.tensor(e)
        pm = t.arange(ml)[None] >= len(e)
        with t.no_grad():
            h = self._m.hidden(x, pm, causal=True)[0, len(e) - 1]
            lo = (h @ self._m.emb.weight[:_VOCAB].T).numpy()
        return lo


def _build_module(torch_mod, nn, d=_DIM, ml=MAXLEN, nl=2, nh=4):
    class _Enc(nn.Module):
        def __init__(self, V, d, ml, nl, nh):
            super().__init__()
            self.emb = nn.Embedding(V + 1, d)
            self.pos = nn.Embedding(ml, d)
            layer = nn.TransformerEncoderLayer(
                d, nh, 4 * d, batch_first=True, dropout=0.0
            )
            self.enc = nn.TransformerEncoder(layer, nl, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.V = V

        def hidden(self, x, pm, causal=False):
            p = torch_mod.arange(x.shape[1])
            h = self.emb(x) + self.pos(p)[None]
            cm = None
            if causal:
                cm = torch_mod.triu(
                    torch_mod.ones(x.shape[1], x.shape[1], dtype=torch_mod.bool), 1
                )
            return self.norm(self.enc(h, mask=cm, src_key_padding_mask=pm))

    return _Enc(_VOCAB, d, ml, nl, nh)


def load_ctx_encoder(path: str | Path = "checkpoints/ctx_encoder.pt") -> ContextEncoder | None:
    """Nạp encoder đã train. Đọc CONFIG (d/layers/heads) kèm checkpoint để dựng
    đúng kiến trúc; tương thích format cũ (state_dict trần, d=96/2/4). Trả None
    (âm thầm) nếu thiếu torch/file — retrieval tự rơi về vector tĩnh, không vỡ."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    try:
        obj = torch.load(str(p), map_location="cpu")
        if isinstance(obj, dict) and "cfg" in obj and "state" in obj:
            c = obj["cfg"]
            m = _build_module(torch, nn, c["d"], c.get("maxlen", MAXLEN),
                              c["layers"], c["heads"])
            m.load_state_dict(obj["state"])
        else:  # format cũ: state_dict trần
            m = _build_module(torch, nn)
            m.load_state_dict(obj)
    except Exception:
        return None
    m.eval()
    return ContextEncoder(m, torch)
