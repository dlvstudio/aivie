"""Sinh text từ HybridLM 1.3B trên M4 (MPS) — base LM, nối tiếp prompt.

Chỉ nạp `state` (bỏ Adam) để tiết kiệm RAM. Không cần mamba-ssm CUDA.

    PYTHONPATH=scripts:src .venv/bin/python scripts/gen_hybrid_scale.py \\
        --ckpt checkpoints/hf_hybrid/hybrid_1p3b_step62000.pt \\
        --prompt "Hà Nội là thành phố" --n 64
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "scripts")
from train_hybrid_scale import HybridLM, HAVE_KERNEL
from nn_aivie.core.tokenizer import BPETokenizer
from nn_aivie.core.vocab import Vocabulary


def chat_ids(tok, question):
    """Khớp build_example của train_hybrid_sft: mỗi đoạn khuôn encode RIÊNG rồi nối."""
    return tok.encode("Hỏi: " + question.strip() + "\n") + tok.encode("Đáp: ")


@torch.no_grad()
def gen(m, vocab, prompt, V, dev, n=64, temp=0.8, top_p=0.9, rep_pen=1.5, recent=48, ctx=512,
        chat=False, eos=None):
    ids = chat_ids(vocab.tokenizer, prompt) if chat else list(vocab.encode_prompt(prompt))
    n0 = len(ids)
    rng = np.random.default_rng(0)
    for i in range(n):
        x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=dev)
        lo = m(x)[0, -1].float().cpu().numpy().astype(np.float64)
        for t in set(ids[-recent:]):
            if 0 <= t < V:
                lo[t] -= rep_pen
        lo = lo / max(temp, 1e-6)
        lo -= lo.max()
        p = np.exp(lo)
        p /= p.sum()
        order = np.argsort(p)[::-1]
        cdf = np.cumsum(p[order])
        keep = order[: max(1, int(np.searchsorted(cdf, top_p)) + 1)]
        pk = p[keep] / p[keep].sum()
        ids.append(int(rng.choice(keep, p=pk)))
        if eos is not None and ids[-1] == eos:
            ids.pop()
            break
        if (i + 1) % 8 == 0:
            print(f"  …{i+1}/{n} tok", flush=True)
    return vocab.detokenize(ids[n0:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="checkpoints/bpe_v1.json")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--ctx", type=int, default=256, help="cửa sổ context lúc gen (nhỏ hơn = nhanh hơn trên M4)")
    ap.add_argument("--rep-pen", type=float, default=1.5)
    ap.add_argument("--chat", action="store_true", help="ckpt SFT: khuôn 'Hỏi: …\\nĐáp: ', dừng ở <EOS>")
    a = ap.parse_args()
    prompts = a.prompt or [
        "Hà Nội là thành phố",
        "Việt Nam là một quốc gia",
        "Sông Mê Kông là",
        "Trong cơ thể người, tế bào",
    ]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"dev={dev} · HAVE_KERNEL={HAVE_KERNEL} · load {a.ckpt}", flush=True)
    t0 = time.time()
    obj = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    step = obj.get("step")
    c = obj["cfg"]
    state = obj["state"]
    del obj  # bỏ opt ngay
    ae = c.get("attn_every", 0)
    attn = list(range(ae - 1, c["layers"], ae)) if ae and ae > 0 else []
    V = 32002
    m = HybridLM(V, c["d"], c["layers"], attn, c.get("d_state", 16), c.get("heads", 16), c.get("kv_heads"))
    m.load_state_dict(state)
    del state
    m.eval()
    m.to(dev)
    npar = sum(p.numel() for p in m.parameters())
    print(f"step={step} · {npar/1e9:.3f}B · d={c['d']} L={c['layers']} attn@{attn} · load {time.time()-t0:.0f}s", flush=True)
    tok = BPETokenizer.load(a.tokenizer)
    vocab = Vocabulary(tok)
    eos = tok.token_to_id["<EOS>"] if a.chat else None
    for s in prompts:
        print(f"\n▶ «{s}»", flush=True)
        t1 = time.time()
        out = gen(m, vocab, s, V, dev, n=a.n, temp=a.temp, ctx=a.ctx, rep_pen=a.rep_pen,
                  chat=a.chat, eos=eos)
        head = "" if a.chat else s
        print(f"  → {head}{out}\n  ({time.time()-t1:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
