"""§TK_103 · TEST Zamba2-2.7B-Instruct (hybrid Mamba2+Attn, base đã train ~3T token).

So với con 1.3B của ta: kiểm tiếng Anh, tiếng Việt, + các use-case gốc (email/văn-phòng,
lập-trình, dịch-thuật). Chạy qua transformers (Mamba2 có fallback pure-torch trên M4).

    PYTHONPATH=scripts:src .venv/bin/python scripts/test_zamba2.py --smoke     # tải + 1 prompt, đo tok/s
    PYTHONPATH=scripts:src .venv/bin/python scripts/test_zamba2.py             # full suite
"""
from __future__ import annotations
import argparse, time, sys
import torch

MODEL = "Zyphra/Zamba2-2.7B-instruct"

SUITE = [
    ("EN · kiến-thức", "How many moons does Mars have, and what are their names?"),
    ("VI · kiến-thức", "Sao Hỏa có mấy vệ tinh? Tên là gì?"),
    ("VI · email công-sở", "Viết email xin nghỉ phép 2 ngày gửi sếp, lý do việc gia đình, giọng lịch sự."),
    ("VI · lập-trình", "Viết hàm Python tính số Fibonacci thứ n bằng đệ quy có memoization."),
    ("EN→VI · dịch", "Dịch sang tiếng Việt: 'Artificial intelligence is transforming how businesses operate.'"),
    ("VI→EN · dịch", "Translate to English: 'Hôm nay trời đẹp, tôi muốn đi dạo trong công viên.'"),
]


def patch_zamba2_block_id():
    """Vá bug transformers 5.16: get_layers gán block_id = layer_id % num_mem_blocks, nhưng
    adapter được chọn theo THỨ-TỰ trong hybrid_layer_ids. Với layout 2.7B [6,12,…,42,47,51]
    tầng 12/47/51 bị lệch chẵn-lẻ → validation tie_weights fail (checkpoint: tầng 12 = block 1,
    adapter 1,3,5,7). Sửa: block_id = hybrid_index % num_mem_blocks."""
    from itertools import cycle
    import torch.nn as nn
    from transformers.models.zamba2 import modeling_zamba2 as Z

    def get_layers(self):
        layers, unique = [], []
        self._tied_weights_keys = {}
        self.first_transformer_layer_id = 0
        hyb_idx = 0
        for layer_id, layer_type in enumerate(self.layers_block_type):
            mamba_layer = Z.Zamba2MambaDecoderLayer(self.config, layer_idx=layer_id)
            if layer_type == "hybrid":
                prefix = f"layers.{layer_id}.shared_transformer"
                if not isinstance(unique, list) or len(unique) >= self.config.num_mem_blocks:
                    if isinstance(unique, list):
                        unique = cycle(unique)
                    self._tied_weights_keys.update({prefix: next(unique)})
                else:
                    unique.append(prefix)
                block_id = hyb_idx % self.config.num_mem_blocks      # ← sửa (gốc: layer_id % …)
                hyb_idx += 1
                attn = Z.Zamba2AttentionDecoderLayer(self.config, block_id=block_id)
                lin = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
                layers.append(Z.Zamba2HybridLayer(attn, lin, mamba_layer))
            else:
                layers.append(mamba_layer)
        return nn.ModuleList(layers)

    Z.Zamba2Model.get_layers = get_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prompt", action="append", default=None, help="thay suite bằng prompt tùy ý (lặp được)")
    a = ap.parse_args()
    T0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    patch_zamba2_block_id()
    print(f"[{time.time()-T0:.0f}s] tải/nạp {a.model} (dev={a.device})…", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    dt = torch.float32 if a.device == "cpu" else torch.float16
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(a.device).eval()
    npar = sum(p.numel() for p in m.parameters())
    print(f"[{time.time()-T0:.0f}s] nạp xong · {npar/1e9:.2f}B params", flush=True)

    suite = ([("custom", p) for p in a.prompt] if a.prompt
             else SUITE[:1] if a.smoke else SUITE)
    for label, q in suite:
        msgs = [{"role": "user", "content": q}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)      # transformers 5.x trả dict
        ids = enc["input_ids"].to(a.device)
        t = time.time()
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        nt = out.shape[1] - ids.shape[1]
        dt_ = time.time() - t
        print(f"\n{'='*72}\n[{label}]  ({dt_:.1f}s · {nt/dt_:.1f} tok/s)\n❓ {q}\n{'-'*72}\n{gen}", flush=True)
    print(f"\n[{time.time()-T0:.0f}s] ### xong", flush=True)


if __name__ == "__main__":
    main()
