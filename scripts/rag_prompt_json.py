"""§TK_099 — test prompt thực tế nhiều cấu-hình → JSON (top-5 token-kế + nối ngắn).

    PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python scripts/rag_prompt_json.py \
        --combiners A.pt B.pt --labels d256-gốc d512+RAG-8M \
        --knn-overrides none checkpoints/knn_faiss_8m_d512.index --out out/rag_prompt_results.json
"""
import sys, argparse, json
sys.path.insert(0, "scripts")
import numpy as np, torch
from fact_eval import load_combiner, logp_of
from nn_aivie.core.tokenizer import BPETokenizer
from nn_aivie.core.vocab import Vocabulary

PROMPTS = [
    "Thủ đô của Việt Nam là thành phố",
    "Hà Nội là thủ đô của nước",
    "Sông dài nhất Việt Nam là sông",
    "Chủ tịch Hồ Chí Minh sinh năm",
    "Mặt Trời là một ngôi",
    "Nước có công thức hóa học là",
    "Truyện Kiều là tác phẩm của",
    "Việt Nam nằm ở khu vực",
]


@torch.no_grad()
def top_next(bundle, ids, dev, k=5):
    logp = logp_of(*bundle, torch.tensor([ids], dtype=torch.long, device=dev))[0, -1]
    p = logp.exp(); tp, ti = p.topk(k)
    return [(int(t), round(float(w), 4)) for t, w in zip(ti, tp)]


@torch.no_grad()
def gen(bundle, ids0, dev, n=12, temp=0.3):
    ids = list(ids0)
    for _ in range(n):
        top = top_next(bundle, ids, dev, k=8)
        toks = [t for t, _ in top]; w = np.array([x for _, x in top]) ** (1 / temp)
        ids.append(int(np.random.default_rng(len(ids)).choice(toks, p=w / w.sum())))
    return ids[len(ids0):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combiners", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--knn-overrides", nargs="+", default=None)
    ap.add_argument("--out", default="out/rag_prompt_results.json")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = BPETokenizer.load("checkpoints/bpe_v1.json"); voc = Vocabulary(tok); V = 32002
    dec = lambda t: tok.decode([t]).strip() or "·"
    ovr = a.knn_overrides or ["none"] * len(a.combiners)
    bundles = [(load_combiner(p, V, dev, None if o == "none" else o), lab)
               for p, lab, o in zip(a.combiners, a.labels, ovr)]

    out = {"prompts": []}
    for pr in PROMPTS:
        ids = list(voc.encode_prompt(pr))
        entry = {"prompt": pr, "configs": []}
        for bundle, lab in bundles:
            top = top_next(bundle, ids, dev)
            entry["configs"].append({
                "label": lab,
                "top5": [{"token": dec(t), "prob": w} for t, w in top],
                "noi_tiep": tok.decode(gen(bundle, ids, dev)).strip(),
            })
        out["prompts"].append(entry)
        print(f"✓ {pr}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"LƯU {a.out} ({len(PROMPTS)} prompt × {len(bundles)} cấu-hình)", flush=True)


if __name__ == "__main__":
    main()
