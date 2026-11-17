#!/usr/bin/env python3
"""
TK_042 G14 — G13 có phải ẢO GIÁC của n=1 không? Cho kNN CƠ HỘI SAI.

    PYTHONPATH=src python -u scripts/probe_knn_compete.py

## Vì sao G13 CHƯA chứng minh gì (tự phản biện)

G13 (§7.15) khoe: datastore ghi *"Thủ đô Việt Nam là **Huế**"* → mạng sinh **"Huế"** ⇒
*"datastore lái được LM"*. Hai lỗ hổng, cả hai do tác giả probe (tôi) tự tạo:

**1. Datastore MỘT CÂU ⇒ kNN KHÔNG THỂ SAI.** Sổ có 20 dòng, tất cả từ cùng một câu.
Lấy top-8 hàng xóm ⇒ 8 trong 20 dòng đó, **không có gì khác để chọn**. Không đối thủ,
không nhiễu, không cơ hội chọn nhầm. Phép đo **không hề kiểm tra việc truy hồi** —
nó chỉ kiểm tra *"p_kNN có chạy tới đầu ra không"*.

**2. Output CHÍNH LÀ datastore, chép nguyên văn:**

    datastore   : "Thủ đô Việt Nam là Huế. Huế nằm bên bờ sông Hương, miền Trung."
    output λ=0.5:              " Huế. Huế nằm bên bờ sôn…"

⇒ G13 trình bày một cơ chế **verbatim copy** như thành tựu — trong đúng ticket sinh ra để
**giết** verbatim copy (§7.13).

## Hai câu hỏi probe này trả lời

**Q1 — LM có phải hành khách?** So λ=0.0 (kNN THUẦN) với λ=0.5.
* giống nhau ⇒ LM **không đóng góp gì**, toàn bộ là băng ghi âm phát lại ❌
* khác nhau ⇒ có trộn thật, LM còn tiếng nói ✅

**Q2 — kNN có chọn đúng khi CÓ đối thủ?** Nhồi sổ thêm câu cạnh tranh, trong đó có
`HARD` = **cùng khuôn "Thủ đô X là Y"** (ngữ cảnh gần trùng ⇒ h gần trùng ⇒ ép kNN
phải phân biệt bằng *"Việt Nam"* vs *"Pháp"*/*"Nhật Bản"*):
* vẫn ra "Huế" ⇒ truy hồi THẬT, G13 sống ✅
* ra "Paris"/"Tokyo" ⇒ kNN **không phân biệt nổi** ⇒ G13 là ảo giác của n=1 ❌

Đây là điều kiện CẦN cho kNN-LM ở quy mô 151k cha. Nếu ngã ở 12 câu thì khỏi bàn 151k.

## ⚠️ Vẫn chưa đo bits/byte

Sinh đúng token ≠ mô hình tốt hơn. Xác nhận cơ chế, không xác nhận HỮU ÍCH (§7.15 caveat).
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

QUERY = "Hỏi: Thủ đô Việt Nam là gì ? Đáp: Thủ đô Việt Nam là"

# Câu PHẢN-SỰ-THẬT: 'Huế' KHÔNG THỂ đến từ trí nhớ LM ⇒ nếu sinh ra thì chỉ có thể từ sổ.
TARGET = "Thủ đô Việt Nam là Huế. Huế nằm bên bờ sông Hương, miền Trung."

# HARD: CÙNG khuôn "Thủ đô X là Y" ⇒ h gần trùng ⇒ kNN phải phân biệt bằng TÊN NƯỚC.
HARD = [
    "Thủ đô Pháp là Paris. Paris nằm bên bờ sông Seine, miền bắc nước Pháp.",
    "Thủ đô Nhật Bản là Tokyo. Tokyo nằm bên vịnh Tokyo, phía đông đảo Honshu.",
    "Thủ đô Thái Lan là Bangkok. Bangkok nằm bên bờ sông Chao Phraya.",
    "Thủ đô Hàn Quốc là Seoul. Seoul nằm bên bờ sông Hán, phía tây bắc bán đảo.",
    "Thủ đô Trung Quốc là Bắc Kinh. Bắc Kinh nằm ở phía bắc đồng bằng Hoa Bắc.",
    "Thủ đô Lào là Viêng Chăn. Viêng Chăn nằm bên bờ sông Mê Kông.",
]

# SOFT: khác chủ đề hoàn toàn — chỉ để bơm nhiễu vào sổ.
SOFT = [
    "Sông Hồng chảy qua đồng bằng Bắc Bộ rồi đổ ra biển Đông.",
    "Nhà Trần ba lần đánh bại quân Nguyên Mông trong thế kỷ mười ba.",
    "Tiếng Việt thuộc ngữ hệ Nam Á, có sáu thanh điệu.",
    "Cà phê Việt Nam chủ yếu trồng ở Tây Nguyên, sản lượng lớn thứ hai thế giới.",
    "Vịnh Hạ Long có hàng nghìn hòn đảo đá vôi, là di sản thiên nhiên thế giới.",
]


class KnnStore:
    """datastore (h, next). LM tự mã hoá CẢ khoá lẫn truy vấn ⇒ chung không gian."""

    def __init__(self, lm, tok):
        self.lm, self.tok, self.m = lm, tok, lm.model
        self.H: list[np.ndarray] = []
        self.Y: list[int] = []
        self.SRC: list[str] = []

    def _hidden(self, ids):
        """h TRƯỚC head — chuẩn kNN-LM (Khandelwal 2020): input của projection cuối."""
        with torch.no_grad():
            x = torch.tensor(ids[-self.lm.ctx :])[None]
            T = x.shape[1]
            h = self.m.drop(self.m.tok(x) + self.m.pos(torch.arange(T)))
            msk = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            for b in self.m.blocks:
                h = b(h, msk)
            return self.m.ln(h)[0].numpy()

    def add(self, text: str, src: str):
        ids = self.tok.encode(text)
        H = self._hidden(ids)
        for i in range(len(ids) - 1):
            v = H[i]
            self.H.append(v / (np.linalg.norm(v) + 1e-8))
            self.Y.append(ids[i + 1])
            self.SRC.append(src)

    def finalize(self):
        self.Hm = np.array(self.H, dtype=np.float32)
        self.Ya = np.array(self.Y)

    def probs(self, ids, V, k, temp):
        h = self._hidden(ids)[-1]
        h = h / (np.linalg.norm(h) + 1e-8)
        sims = self.Hm @ h
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]  # argpartition KHÔNG sắp — phải sort tường minh
        w = np.exp((sims[top] - sims[top].max()) / temp)
        w /= w.sum()
        p = np.zeros(V, dtype=np.float64)
        for j, i in enumerate(top):
            p[self.Ya[i]] += w[j]
        nb = [(float(sims[i]), self.SRC[i], self.tok.decode([int(self.Ya[i])])) for i in top[:4]]
        return p, nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="out/g7_best_model.pt")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=14)
    args = ap.parse_args()

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok)
    lm = NeuralLM.load(_ROOT / args.lm, V, device="cpu")
    print(f"LM {sum(p.numel() for p in lm.model.parameters())/1e6:.1f}M · vocab {V:,}")
    print(f"prompt: {QUERY!r}\n")

    def gen(store, lam, n):
        ids = tok.encode(QUERY)
        for _ in range(n):
            with torch.no_grad():
                lg = lm.model(torch.tensor(ids[-lm.ctx :])[None])[0, -1]
                p_lm = torch.softmax(lg, -1).numpy().astype(np.float64)
            p = p_lm if (store is None or lam >= 1.0) else lam * p_lm + (1 - lam) * store.probs(ids, V, args.k, args.temp)[0]
            ids.append(int(np.argmax(p)))
        return tok.decode(ids)[len(QUERY) :]

    # sổ dựng dần: n=1 (G13, kNN KHÔNG THỂ SAI) → +HARD (phải phân biệt) → +SOFT (nhiễu)
    setups = [
        ("n=1  (G13 gốc)", [(TARGET, "TARGET")]),
        ("+6 HARD cùng khuôn", [(TARGET, "TARGET")] + [(t, "HARD") for t in HARD]),
        ("+6 HARD +5 SOFT", [(TARGET, "TARGET")] + [(t, "HARD") for t in HARD] + [(t, "SOFT") for t in SOFT]),
    ]
    lams = [1.0, 0.5, 0.2, 0.0]

    print(f"{'sổ':>20} {'#dòng':>6} " + " ".join(f"{'λ='+str(l):>24}" for l in lams))
    print(f"{'':>20} {'':>6} " + " ".join(f"{'(LM thuần)' if l==1.0 else ('(kNN thuần)' if l==0.0 else ''):>24}" for l in lams))
    last = None
    for name, docs in setups:
        st = KnnStore(lm, tok)
        for text, src in docs:
            st.add(text, src)
        st.finalize()
        row = [gen(st, l, args.n)[:22].replace("\n", "⏎") for l in lams]
        print(f"{name:>20} {len(st.Y):>6} " + " ".join(f"{r!r:>24}" for r in row))
        last = st

    # hàng xóm tại ĐÚNG bước quyết định (ngay sau '… Việt Nam là')
    print(f"\n─── 4 hàng xóm gần nhất ở bước quyết định (sổ đầy đủ, {len(last.Y)} dòng) ───")
    _, nb = last.probs(tok.encode(QUERY), V, args.k, args.temp)
    for sim, src, nxt in nb:
        print(f"   cos={sim:.4f}  từ={src:<7} bỏ phiếu cho → {nxt!r}")

    print("\n⇒ ĐỌC:")
    print("  Q1 λ=0.0 vs λ=0.5 GIỐNG NHAU ⇒ LM là HÀNH KHÁCH, chỉ là băng phát lại ❌")
    print("  Q2 thêm HARD mà VẪN ra 'Huế' ⇒ truy hồi THẬT ✅ | ra 'Paris'/'Tokyo' ⇒ ảo giác n=1 ❌")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
