"""§TK_095 — TEST SFT: hỏi câu MỚI, xem model SFT trả-lời-đúng-KHUÔN nhưng có bịa fact không.

Kiểm chứng luận điểm: SFT dạy ĐỊNH-DẠNG (Hỏi:→Đáp:), KHÔNG dạy KIẾN-THỨC.
Base yếu (bits 6.79) → sẽ trả lời đúng khuôn "Đáp: …" nhưng fact word-salad.

    PYTHONPATH=src .venv/bin/python scripts/gen_sft_qa.py --ckpt checkpoints/mamba1024_sft.pt
"""
import sys, argparse
import numpy as np, torch
sys.path.insert(0, "scripts")
from train_mamba import Mamba
from nn_aivie.core.tokenizer import BPETokenizer
from nn_aivie.core.vocab import Vocabulary

# câu hỏi FACT rõ ràng, KHÔNG có trong data train (data train là "viết đoạn văn/dịch/toán")
QUESTIONS = [
    "Thủ đô của nước Pháp là thành phố nào?",
    "Ai là tác giả của Truyện Kiều?",
    "Nước sôi ở nhiệt độ bao nhiêu độ C?",
    "Một năm có bao nhiêu ngày?",
    "Sông dài nhất thế giới là sông nào?",
    "Mặt Trời là gì?",
]


def load_model(ckpt, V, dev):
    import re
    obj = torch.load(ckpt, map_location="cpu"); c = obj["cfg"]; state = obj["state"]
    al = sorted({int(m.group(1)) for k in state
                 for m in [re.match(r"blocks\.(\d+)\.attn\.", k)] if m})
    m = Mamba(V, c["d"], c["layers"], None, c["d_state"], al, c.get("heads", 4)).to(dev)
    m.load_state_dict(state); m.eval()
    return m


@torch.no_grad()
def answer(m, vocab, tok, q, V, dev, n=60, temp=0.7, top_p=0.9, rep_pen=3.0, recent=24):
    EOS = tok.token_to_id["<EOS>"]
    prompt = f"Hỏi: {q}\nĐáp:"
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
        nxt = int(rng.choice(keep, p=pk))
        if nxt == EOS:
            break
        ids.append(nxt)
    return vocab.detokenize(ids[n0:]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mamba1024_sft.pt")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = BPETokenizer.load("checkpoints/bpe_v1.json")
    vocab = Vocabulary(tok); V = 32002
    m = load_model(a.ckpt, V, dev)
    npar = sum(pp.numel() for pp in m.parameters())
    print(f"# SFT Q&A test · {npar/1e6:.1f}M · {a.ckpt}\n{'='*70}", flush=True)
    for q in QUESTIONS:
        ans = answer(m, vocab, tok, q, V, dev)
        print(f"\n▶ Hỏi: {q}\n  Đáp: {ans}", flush=True)
    print("\n### hết", flush=True)


if __name__ == "__main__":
    main()
