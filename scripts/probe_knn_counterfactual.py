#!/usr/bin/env python3
"""
TK_042 G13 — PROBE PHẢN-SỰ-THẬT: datastore có LÁI ĐƯỢC LM không?

    PYTHONPATH=src python -u scripts/probe_knn_counterfactual.py

## Câu hỏi (§7.15f #4) — phép đo THẬT SỰ trả lời câu hỏi của dự án

LM một mình **bịa**: `'Thủ đô Việt Nam là'` → *"tỉnh Nam Cộng hòa"* (§7.14a).
Nhét fact vào **prompt** → **KHÔNG đọc** (§7.14c: fact ĐÚNG ≡ fact SAI ⇒ 11.4M không có
in-context learning).

**kNN-LM trộn ở tầng PHÂN BỐ, không ở tầng ngữ cảnh** ⇒ LM không cần "hiểu" gì:

    P = λ·P_LM + (1−λ)·P_kNN(h)

⇒ Câu hỏi: **datastore có lái được không?**

## Vì sao PHẢN-SỰ-THẬT, không phải chỉ "fact đúng"

Nếu chỉ thử fact ĐÚNG (Hà Nội) và LM nói *"Hà Nội"* — **KHÔNG chứng minh gì**: nó có thể
nói đúng vì **đã nhớ sẵn** (corpus train có Hà Nội). Phải thử fact **SAI** (Huế):

* datastore "Huế" → sinh **"Huế"**  ⇒ nó **ĐỌC datastore** ✅ (đồ thị lái được LM)
* datastore "Huế" → vẫn "Hà Nội"/bịa ⇒ nó đọc **TRÍ NHỚ**, datastore vô dụng ❌

Đây đúng phương pháp §7.14c đã dùng để giết RAG-prompt (B≡C ⇒ không đọc).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from nn_aivie.core.neural_lm import NeuralLM  # noqa: E402
from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402


class KnnStore:
    """datastore (h, next) — LM tự encode CẢ HAI đầu ⇒ cùng không gian (§7.15a)."""

    def __init__(self, lm, tok):
        self.lm, self.tok, self.m = lm, tok, lm.model
        self.H: list[np.ndarray] = []
        self.Y: list[int] = []
        self.T: list[str] = []

    def _hidden(self, ids):
        """h TRƯỚC head (chuẩn kNN-LM: input của projection cuối)."""
        with torch.no_grad():
            x = torch.tensor(ids[-self.lm.ctx :])[None]
            T = x.shape[1]
            h = self.m.drop(self.m.tok(x) + self.m.pos(torch.arange(T)))
            msk = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            for b in self.m.blocks:
                h = b(h, msk)
            return self.m.ln(h)[0].numpy()

    def add(self, text: str):
        ids = self.tok.encode(text)
        H = self._hidden(ids)
        for i in range(len(ids) - 1):
            v = H[i]
            self.H.append(v / (np.linalg.norm(v) + 1e-8))
            self.Y.append(ids[i + 1])
            self.T.append(self.tok.decode(ids[max(0, i - 4) : i + 1]))

    def finalize(self):
        self.Hm = np.array(self.H, dtype=np.float32)
        self.Ya = np.array(self.Y)

    def probs(self, ids, V, k, temp):
        h = self._hidden(ids)[-1]
        h = h / (np.linalg.norm(h) + 1e-8)
        sims = self.Hm @ h
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        w = np.exp((sims[top] - sims[top].max()) / temp)
        w /= w.sum()
        p = np.zeros(V, dtype=np.float64)
        for j, i in enumerate(top):
            p[self.Ya[i]] += w[j]
        return p, [(float(sims[i]), self.T[i], self.tok.token_of if False else i) for i in top[:3]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="out/g7_best_model.pt")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--lams", default="1.0,0.5,0.2,0.0")
    ap.add_argument("--n", type=int, default=14)
    args = ap.parse_args()

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok)
    lm = NeuralLM.load(_ROOT / args.lm, V, device="cpu")
    print(f"LM {sum(p.numel() for p in lm.model.parameters())/1e6:.1f}M · vocab {V:,}\n")

    Q = "Hỏi: Thủ đô Việt Nam là gì ? Đáp: Thủ đô Việt Nam là"
    FACTS = {
        "THẬT  (Hà Nội)": "Thủ đô Việt Nam là Hà Nội. Hà Nội nằm ở đồng bằng Bắc Bộ, ven sông Hồng.",
        "PHẢN  (Huế)": "Thủ đô Việt Nam là Huế. Huế nằm bên bờ sông Hương, miền Trung.",
        "PHẢN  (Cà Mau)": "Thủ đô Việt Nam là Cà Mau. Cà Mau nằm ở cực nam đất nước.",
    }

    def gen(store, lam, n):
        ids = tok.encode(Q)
        for _ in range(n):
            with torch.no_grad():
                lg = lm.model(torch.tensor(ids[-lm.ctx :])[None])[0, -1]
                p_lm = torch.softmax(lg, -1).numpy().astype(np.float64)
            if store is not None and lam < 1.0:
                p_kn, _ = store.probs(ids, V, args.k, args.temp)
                p = lam * p_lm + (1 - lam) * p_kn
            else:
                p = p_lm
            ids.append(int(np.argmax(p)))
        return tok.decode(ids)[len(Q) :]

    lams = [float(x) for x in args.lams.split(",")]
    print(f"prompt: {Q!r}\n")
    print(f"{'datastore':>16} " + " ".join(f"{'λ='+str(l):>26}" for l in lams))
    print(f"{'':>16} " + " ".join(f"{'(1.0 = LM thuần)' if l==1.0 else '':>26}" for l in lams))
    for name, fact in FACTS.items():
        st = KnnStore(lm, tok)
        st.add(fact)
        st.finalize()
        row = [gen(st, l, args.n)[:24].replace("\n", "⏎") for l in lams]
        print(f"{name:>16} " + " ".join(f"{r!r:>26}" for r in row))

    print("\n⇒ ĐỌC: cột λ=1.0 phải GIỐNG NHAU (LM thuần, datastore không dùng).")
    print("  Cột λ thấp: 'Huế'/'Cà Mau' xuất hiện ⇒ datastore LÁI ĐƯỢC LM ✅")
    print("               vẫn 'Hà Nội'/bịa      ⇒ LM đọc TRÍ NHỚ, datastore vô dụng ❌")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
