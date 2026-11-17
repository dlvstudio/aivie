"""§TK_097 — EVAL FACT-ĐÚNG: đo grounding THẬT của RAG (không chỉ bits).

Trên viwiki HELD-OUT (rời datastore+train), so combiner thuần-nơ-ron vs combiner-có-RAG
ở độ-chính-xác dự-đoán token-kế-tiếp, TÁCH riêng token NỘI-DUNG (hiếm — mang fact: tên,
số, thực-thể) khỏi token CHỨC-NĂNG (hư từ, thường). Grounding ăn tiền ở token nội-dung.

    PYTHONPATH=src .venv/bin/python scripts/fact_eval.py \
        --combiners checkpoints/combiner_wiki_qa.pt checkpoints/combiner_rag.pt \
        --labels thuần-nơ-ron có-RAG
"""
import sys, argparse
sys.path.insert(0, "scripts")
import numpy as np, torch
from train_combiner import Combiner, load_expert, make_knn


def load_combiner(path, V, dev, knn_override=None):
    obj = torch.load(path, map_location="cpu")
    experts = [load_expert(c, V, dev)[0] for c in obj["experts"]]
    ds_path = knn_override or (obj["knn"]["datastore"] if obj.get("knn") else None)  # §TK_097 hoán kho giữ combiner
    knn = make_knn(ds_path, dev, V, obj["knn"]["k"], obj["knn"]["temp"]) \
        if obj.get("knn") else None
    comb = Combiner(obj["dims"], obj["dh"], mode=obj.get("mode", "prob"), V=V).to(dev)
    comb.load_state_dict(obj["state"]); comb.eval()
    return experts, knn, comb


@torch.no_grad()
def logp_of(experts, knn, comb, ids):
    lh = [m(ids, return_hidden=True) for m in experts]
    los = [l for l, _ in lh]; hs = [h for _, h in lh]
    if knn is not None:
        los.append(knn.logits(hs[0])); hs.append(hs[0])
    logp, _ = comb(hs, los)
    return logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combiners", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--knn-overrides", nargs="+", default=None,
                    help="§TK_097 hoán kho cho từng combiner (giữ NGUYÊN combiner, cô-lập hiệu-ứng cỡ-kho); 'none'=dùng kho gốc")
    ap.add_argument("--data", default="checkpoints/enc_data_viwiki.npz")
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--rare-top", type=int, default=200, help="token nằm trong top-N tần-suất = CHỨC-NĂNG; còn lại = NỘI-DUNG")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = np.load(a.data); X = d["X"][:, :128]; L = d["L"]; V = d["E"].shape[0]
    keep = np.where(L >= 64)[0]

    # tần-suất token (định-nghĩa nội-dung/chức-năng) từ mẫu lớn
    freq = np.bincount(X[keep[:8000]].reshape(-1), minlength=V)
    func_tokens = set(np.argsort(freq)[::-1][:a.rare_top].tolist())   # top-N thường = chức-năng
    is_func = torch.zeros(V, dtype=torch.bool); is_func[list(func_tokens)] = True
    is_func = is_func.to(dev)

    # eval set RỜI datastore(seed123) + train(seed0): seed 999
    ev = keep.copy(); np.random.default_rng(999).shuffle(ev); ev = ev[:a.n_eval]
    Xt = torch.tensor(X[ev], dtype=torch.long)

    print(f"eval {len(ev)} seq · token chức-năng={a.rare_top} thường nhất · còn lại=nội-dung(fact)\n")
    print(f"{'model':>14} | {'top1 chung':>10} {'top5 chung':>10} | {'top1 NỘI-DUNG':>13} {'top5 NỘI-DUNG':>13}")
    print("-" * 74)
    ovr = a.knn_overrides or ["none"] * len(a.combiners)
    for path, lab, ov in zip(a.combiners, a.labels, ovr):
        experts, knn, comb = load_combiner(path, V, dev, None if ov == "none" else ov)
        c1 = c5 = n = 0; fc1 = fc5 = fn = 0
        for i in range(0, len(Xt), 8):
            ids = Xt[i:i + 8].to(dev)
            logp = logp_of(experts, knn, comb, ids)[:, :-1]      # (b,l-1,V)
            gold = ids[:, 1:]                                     # (b,l-1)
            top5 = logp.topk(5, dim=-1).indices                  # (b,l-1,5)
            ok1 = top5[..., 0] == gold
            ok5 = (top5 == gold.unsqueeze(-1)).any(-1)
            content = ~is_func[gold]                              # token nội-dung
            c1 += int(ok1.sum()); c5 += int(ok5.sum()); n += gold.numel()
            fc1 += int((ok1 & content).sum()); fc5 += int((ok5 & content).sum()); fn += int(content.sum())
        print(f"{lab:>14} | {100*c1/n:9.1f}% {100*c5/n:9.1f}% | {100*fc1/fn:12.1f}% {100*fc5/fn:12.1f}%")
    print("\n(NỘI-DUNG = token hiếm mang fact: tên/số/thực-thể. Grounding ăn tiền ở cột này.)")


if __name__ == "__main__":
    main()
