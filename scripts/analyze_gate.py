"""§TK_096 — phân-tích gate combiner theo NGUỒN: chứng minh định-tuyến ngữ-nghĩa.

Nhóm token theo nhãn nguồn (src trong npz trộn) → gate trung bình. Đường chéo trội
= combiner tự học "token miền X → tin expert miền X" (thuần backprop, không cho nhãn).

    PYTHONPATH=src .venv/bin/python scripts/analyze_gate.py \
        --combiner checkpoints/combiner_3expert.pt --data checkpoints/mix3_wqn.npz --names wiki QA news
"""
import sys, argparse
sys.path.insert(0, "scripts")
import numpy as np, torch
from train_combiner import Combiner, load_expert, KnnExpert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combiner", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--names", nargs="+", required=True, help="tên NGUỒN (hàng)")
    ap.add_argument("--expert-names", nargs="+", default=None, help="tên EXPERT/cột (gồm RAG-kNN); mặc-định=names")
    ap.add_argument("--cap", type=int, default=200)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = np.load(a.data); X = d["X"]; src = d["src"]; V = d["E"].shape[0]
    obj = torch.load(a.combiner, map_location="cpu")
    experts = [load_expert(c, V, dev)[0] for c in obj["experts"]]
    comb = Combiner(obj["dims"], obj["dh"], mode=obj.get("mode", "prob"), V=V).to(dev)
    comb.load_state_dict(obj["state"]); comb.eval()
    knn = KnnExpert(obj["knn"]["datastore"], dev, V, obj["knn"]["k"], obj["knn"]["temp"]) \
        if obj.get("knn") else None
    Xt = torch.tensor(X, dtype=torch.long)
    row_names = a.names                                    # tên NGUỒN (hàng)
    col_names = a.expert_names or a.names                  # tên EXPERT/cột (gồm RAG-kNN)
    hdr = "  ".join(f"→{n:>7}" for n in col_names)
    print(f"gate trung bình per-token, nhóm theo NGUỒN câu:\n{'nguồn':>8} | {hdr}")

    def run(ids):
        lh = [m(ids, return_hidden=True) for m in experts]
        los = [l for l, _ in lh]; hs = [h for _, h in lh]
        if knn is not None:
            los.append(knn.logits(hs[0])); hs.append(hs[0])
        return comb(hs, los)

    with torch.no_grad():
        for s in range(len(row_names)):
            idx = np.where(src == s)[0][:a.cap]; gs = []
            for i in range(0, len(idx), 16):
                _, g = run(Xt[idx[i:i + 16]].to(dev))
                if g is not None:
                    gs.append(g.mean(dim=(0, 1)).cpu())
            if not gs:
                print(f"{row_names[s]:>8} | (mode không có gate)"); continue
            g = torch.stack(gs).mean(0).tolist()
            star = col_names[int(np.argmax(g))]
            row = "  ".join(f"{x:7.2f}" for x in g)
            print(f"{row_names[s]:>8} | {row}   (cao nhất → {star})")


if __name__ == "__main__":
    main()
