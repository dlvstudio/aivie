"""§TK_093 — SINH CHỮ THẬT từ Mamba/hybrid của ta (base LM, NỐI-TIẾP văn bản).

Mamba là BASE LM (không instruction-tuned) → NỐI tiếp prompt, không trả lời hỏi.
9.9M/bits~7 = đồ-chơi kiến-trúc, không phải trợ lý. Đây là để THẤY nó sinh gì.

    PYTHONPATH=src .venv/bin/python scripts/gen_mamba_text.py
"""
import sys, argparse
import numpy as np, torch
sys.path.insert(0, "scripts")
from train_mamba import Mamba
from nn_aivie.core.tokenizer import BPETokenizer
from nn_aivie.core.vocab import Vocabulary

SEEDS = [
    "Chó là loài động vật",
    "Hà Nội là thành phố",
    "Sông Mê Kông là con sông",
    "Mặt Trời là",
    "Việt Nam là một quốc gia",
    "Trong cơ thể người, tế bào",
]


def load_model(ckpt, V, dev):
    import re
    obj = torch.load(ckpt, map_location="cpu"); c = obj["cfg"]; state = obj["state"]
    # SUY attn-layers từ state-dict (bền — không phụ thuộc cfg có thể lưu sai khi resume)
    al = sorted({int(m.group(1)) for k in state
                 for m in [re.match(r"blocks\.(\d+)\.attn\.", k)] if m})
    if not al:
        al = [int(x) for x in str(c.get("attn_layers", "")).split(",") if x.strip() != ""]
    m = Mamba(V, c["d"], c["layers"], None, c["d_state"], al, c.get("heads", 4)).to(dev)
    m.load_state_dict(state); m.eval()
    tag = f"HYBRID(attn@{al})" if al else "Mamba thuần"
    return m, tag, c


@torch.no_grad()
def gen(m, vocab, prompt, V, dev, n=45, temp=0.8, top_p=0.9, rep_pen=3.0, recent=24):
    ids = list(vocab.encode_prompt(prompt)); n0 = len(ids)
    rng = np.random.default_rng(0)
    for _ in range(n):
        x = torch.tensor([ids[-256:]], dtype=torch.long, device=dev)
        lo = m(x)[0, -1].float().cpu().numpy().astype(np.float64)
        for t in set(ids[-recent:]):
            if 0 <= t < V: lo[t] -= rep_pen
        lo = lo / max(temp, 1e-6); lo -= lo.max()
        p = np.exp(lo); p /= p.sum()
        order = np.argsort(p)[::-1]; cdf = np.cumsum(p[order])
        keep = order[:max(1, int(np.searchsorted(cdf, top_p)) + 1)]
        pk = p[keep] / p[keep].sum()
        ids.append(int(rng.choice(keep, p=pk)))
    return vocab.detokenize(ids[n0:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+",
                    default=["checkpoints/hybrid_d256_2k.pt", "checkpoints/mamba_d256_2k.pt"])
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    vocab = Vocabulary(BPETokenizer.load("checkpoints/bpe_v1.json"))
    V = 32002
    print(f"vocab V={V} · dev={dev}\n", flush=True)
    for ck in a.ckpts:
        m, tag, c = load_model(ck, V, dev)
        npar = sum(pp.numel() for pp in m.parameters())
        print(f"{'#'*70}\n# {tag} ({npar/1e6:.1f}M, {ck})\n{'#'*70}", flush=True)
        for s in SEEDS:
            out = gen(m, vocab, s, V, dev)
            print(f"\n▶ «{s}»\n  → {s} {out}", flush=True)
        print(flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
