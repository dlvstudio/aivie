"""§TK_097 — DEMO PROMPT THỰC TẾ: cloze-fact, so combiner THUẦN-nơ-ron vs CÓ-RAG.

Base-LM nối-tiếp (không hỏi-đáp) → cho câu dẫn tới token-SỰ-THẬT, xem top-5 dự-đoán
token-kế-tiếp. RAG (kho viwiki) có kéo đúng fact lên không? + nối-tiếp ngắn tham-khảo.

    PYTHONPATH=src .venv/bin/python scripts/rag_prompt_demo.py
"""
import sys, argparse
sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F
from fact_eval import load_combiner, logp_of
from nn_aivie.core.tokenizer import BPETokenizer
from nn_aivie.core.vocab import Vocabulary

PROMPTS = [
    "Thủ đô của Việt Nam là thành phố",
    "Hà Nội là thủ đô của nước",
    "Sông dài nhất Việt Nam là sông",
    "Chủ tịch Hồ Chí Minh sinh năm",
    "Nước có công thức hóa học là",
    "Mặt Trời là một ngôi",
]


@torch.no_grad()
def next_top(bundle, ids_list, k=5):
    experts, knn, comb = bundle
    ids = torch.tensor([ids_list], dtype=torch.long, device=DEV)
    logp = logp_of(experts, knn, comb, ids)[0, -1]         # (V,) log-prob token kế
    p = logp.exp(); tp, ti = p.topk(k)
    return [(int(t), float(w)) for t, w in zip(ti, tp)]


@torch.no_grad()
def generate(bundle, ids_list, n=12, temp=0.3):
    ids = list(ids_list)
    for _ in range(n):
        top = next_top(bundle, ids, k=8)
        toks = [t for t, _ in top]; ws = np.array([w for _, w in top]) ** (1 / temp)
        ws /= ws.sum()
        ids.append(int(np.random.default_rng(len(ids)).choice(toks, p=ws)))
    return ids[len(ids_list):]


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", default="checkpoints/combiner_wiki_qa.pt")
    ap.add_argument("--rag", default="checkpoints/combiner_rag_big.pt")
    ap.add_argument("--rag-datastore", default="checkpoints/knn_datastore_2m.npz")
    a = ap.parse_args()
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = BPETokenizer.load("checkpoints/bpe_v1.json"); voc = Vocabulary(tok); V = 32002
    neural = load_combiner(a.neural, V, DEV)
    rag = load_combiner(a.rag, V, DEV, a.rag_datastore)
    dec = lambda t: tok.decode([t]).strip() or "·"

    for pr in PROMPTS:
        ids = list(voc.encode_prompt(pr))
        print(f"\n{'='*72}\n▶ «{pr} ___»")
        rag_lab = "CÓ-RAG (" + a.rag_datastore.split("/")[-1].split(".")[0].replace("knn_", "") + ")"
        for name, b in [("THUẦN-nơ-ron", neural), (rag_lab, rag)]:
            top = next_top(b, ids)
            cells = "  ".join(f"{dec(t)}·{w*100:.0f}%" for t, w in top)
            gen = tok.decode(generate(b, ids)).strip()
            print(f"  [{name:>13}] top5: {cells}")
            print(f"  {'':>15} nối: …{gen}")
    print("\n### hết")


if __name__ == "__main__":
    main()
