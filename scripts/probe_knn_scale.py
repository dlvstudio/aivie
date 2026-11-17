#!/usr/bin/env python3
"""
TK_042 G15 — kNN-LM ở QUY MÔ THẬT: datastore từ CORPUS THẬT (không phải 1-2 câu tay viết).

    PYTHONPATH=src python -u scripts/probe_knn_scale.py

## Vì sao cần (nối tiếp G13 §7.15, G14 §7.16e-outlier)

G13/G14 chứng minh CƠ CHẾ kNN-LM chạy đúng — nhưng datastore chỉ có 1 câu (G13) hoặc 1 câu +
6 câu HARD viết tay (G14). Không câu nào trong đó THẬT SỰ đến từ corpus train. Đây là lỗ hổng
quy mô cuối cùng trước khi kết luận kNN-LM có ích ở kích thước thật.

Script này dựng datastore từ **`history_articles.jsonl` THẬT** (158 bài, cùng corpus đã train
`network_v11.pkl`) và thử lại **ĐÚNG 8 prompt lịch sử** đã dùng xuyên suốt ticket
(`probe_granularity.py`), để so trực tiếp: LM-thuần vs LM+kNN ở corpus THẬT.

## Vì sao phải CẮT CỬA SỔ (chunk), không encode nguyên bài

`lm.ctx = 256` token. Bài history trung vị dài **~2000 token ước tính** (median 6077 ký tự) —
dài hơn ctx **~8 lần**. `_hidden()` gốc trong `probe_knn_compete.py` chỉ lấy **ctx token CUỐI**
của `ids` — nếu dùng y nguyên trên bài dài, vòng lặp gán `(h, next)` sẽ LỆCH VỊ TRÍ (hoặc lỗi
out-of-bounds) vì `H` chỉ có `ctx` hàng còn `ids` dài hơn. Ở đây CẮT mỗi bài thành các cửa sổ
**không chồng lấn**, kích thước `ctx`, mã hoá riêng từng cửa sổ — mất ngữ cảnh xuyên cửa sổ,
nhưng ĐÚNG vị trí, không có lỗi ngầm.

## Đọc kết quả

* So với G14 (datastore đồ chơi): nếu ở quy mô thật retrieval **vẫn khớp đúng** (hàng xóm gần
  nhất là đoạn liên quan tới prompt) ⇒ G13/G14 đứng vững ở corpus thật.
* Nếu hàng xóm gần nhất **không liên quan** ⇒ ở quy mô ~15-20k dòng datastore, tín hiệu bị
  nhiễu lấn át — kNN-LM cần datastore LỚN HƠN NỮA (Khandelwal dùng ~103M token) hoặc cách biểu
  diễn tốt hơn `h` thô.
* So sánh với `probe_granularity.py` trên `v10.pkl`/`v11.pkl` (MỚI 5-gram ≈ 0%, khảm thuần):
  nếu kNN-LM+LM-nhỏ sinh câu **KHÔNG trùng y hệt bất kỳ đoạn nào** trong datastore mà vẫn *đúng
  ngữ nghĩa* ⇒ đây là bằng chứng đầu tiên cho "sinh thật" mà TK_042 tìm kiếm từ đầu.

⚠️ Đây vẫn không phải câu trả lời cuối — chưa đo bits/byte, datastore chỉ 158 bài (một phần nhỏ
của 65,098 cha trong `v11.pkl`). Là bước tiếp theo hợp lý, không phải kết luận.

## v3 (2026-07-18) — LẤY MẪU thay ARGMAX

G15 v1/v2 sinh bằng `argmax` — LUÔN chọn token xác suất cao nhất, TẤT ĐỊNH. Nhưng câu hỏi mở
ra TOÀN BỘ TK_042 mô tả **lấy mẫu ngẫu nhiên**: *"lúc thì sinh 'chó', lúc thì sinh 'cún', chạy
theo phân bố xác suất hợp ngữ cảnh"*. Ta chưa từng thử đúng thứ được hỏi.

Đây cũng có thể giải thích vòng lặp thoái hoá ở §7.17b: argmax tất định dễ mắc kẹt (chọn token
cao nhất → token đó đẩy xác suất token tương tự lên cao nhất lượt sau → lặp mãi). Lấy mẫu có
nhiễu ngẫu nhiên thường tự phá kiểu vòng lặp này.

`--mode sample`: rút mẫu trực tiếp từ phân bố ĐÃ TRỘN `p = λ·P_LM + (1-λ)·P_kNN` (không argmax).
`--seeds N`: chạy N lần/prompt/λ với seed khác nhau — quan sát TRỰC TIẾP biến thiên "lúc chó lúc
cún" có xảy ra không. `--sample-temp`: định hình lại `p` trước khi rút mẫu
(`p^(1/T)/Σp^(1/T)`, T=1 = rút thẳng từ `p`, T>1 làm phẳng/ngẫu nhiên hơn, T<1 làm nhọn/gần
argmax hơn).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from nn_aivie.core.neural_lm import NeuralLM  # noqa: E402
from nn_aivie.core.tokenizer import BPETokenizer  # noqa: E402

# Đúng 8 prompt đã dùng trong probe_granularity.py — so được xuyên suốt ticket.
PROMPTS = [
    "Nhà Trần là",
    "Khúc Thừa Dụ",
    "Ngô Quyền",
    "Lý Thái Tổ",
    "Hai Bà Trưng",
    "Nhà Lý",
    "Trần Hưng Đạo",
    "Lê Lợi",
]


class KnnStoreScale:
    def __init__(self, lm, tok):
        self.lm, self.tok, self.m = lm, tok, lm.model
        self.H: list[np.ndarray] = []
        self.Y: list[int] = []
        self.T: list[str] = []

    def _hidden_window(self, ids: list[int]) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(ids)[None]
            T = x.shape[1]
            h = self.m.drop(self.m.tok(x) + self.m.pos(torch.arange(T)))
            msk = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            for b in self.m.blocks:
                h = b(h, msk)
            return self.m.ln(h)[0].numpy()

    def add_document(self, text: str) -> int:
        """Cắt thành cửa sổ KHÔNG CHỒNG LẤN kích thước ctx, mã hoá riêng từng cửa sổ."""
        ids = self.tok.encode(text)
        ctx = self.lm.ctx
        n_added = 0
        for start in range(0, len(ids), ctx):
            window = ids[start : start + ctx]
            if len(window) < 4:
                continue
            H = self._hidden_window(window)
            for i in range(len(window) - 1):
                v = H[i]
                self.H.append(v / (np.linalg.norm(v) + 1e-8))
                self.Y.append(window[i + 1])
                self.T.append(self.tok.decode(window[max(0, i - 6) : i + 1]))
            n_added += len(window) - 1
        return n_added

    def finalize(self):
        self.Hm = np.array(self.H, dtype=np.float32)
        self.Ya = np.array(self.Y)

    def probs(self, ids: list[int], V: int, k: int, temp: float):
        with torch.no_grad():
            x = torch.tensor(ids[-self.lm.ctx :])[None]
            T = x.shape[1]
            h = self.m.drop(self.m.tok(x) + self.m.pos(torch.arange(T)))
            msk = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            for b in self.m.blocks:
                h = b(h, msk)
            h = self.m.ln(h)[0, -1].numpy()
        h = h / (np.linalg.norm(h) + 1e-8)
        sims = self.Hm @ h
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        w = np.exp((sims[top] - sims[top].max()) / temp)
        w /= w.sum()
        p = np.zeros(V, dtype=np.float64)
        for j, i in enumerate(top):
            p[self.Ya[i]] += w[j]
        nb = [(float(sims[i]), self.T[i], self.tok.decode([int(self.Ya[i])])) for i in top[:5]]
        return p, nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="out/g7_best_model.pt")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--corpus", default="data/text_corpus/history_articles.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lams", default="1.0,0.5,0.2,0.0")
    ap.add_argument("--mode", choices=["argmax", "sample"], default="argmax")
    ap.add_argument("--sample-lams", default="0.4")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--sample-temps", default="0.3,0.5,0.7", help="quét — list ngăn bởi dấu phẩy")
    ap.add_argument("--top-k", type=int, default=40, help="0 = tắt")
    ap.add_argument("--top-p", type=float, default=0.0, help="0 = tắt (nucleus)")
    args = ap.parse_args()

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok)
    lm = NeuralLM.load(_ROOT / args.lm, V, device="cpu")
    print(f"LM {sum(p.numel() for p in lm.model.parameters())/1e6:.1f}M · vocab {V:,} · ctx={lm.ctx}")

    print(f"\ndựng datastore từ {args.corpus} …")
    store = KnnStoreScale(lm, tok)
    n_docs = n_windows_tok = 0
    docs: list[str] = []
    for line in open(_ROOT / args.corpus, encoding="utf-8"):
        text = json.loads(line)
        docs.append(text)
        n_windows_tok += store.add_document(text)
        n_docs += 1
    store.finalize()
    print(f"✓ {n_docs} bài · {len(store.Y):,} cặp (h,next) trong datastore")

    # 5-gram MỚI — CÙNG thước dùng xuyên suốt ticket (probe_granularity.py §0.5/§7.16).
    # Không đo cái này thì KHÔNG được gọi là "sinh thật" — chỉ là "đọc thấy có vẻ mới".
    N = 5
    print(f"dựng index {N}-gram từ CHÍNH {args.corpus} (nguồn datastore) …")
    ngram_idx: set[tuple[int, ...]] = set()
    for text in docs:
        ids = tok.encode(text)
        for i in range(len(ids) - N + 1):
            ngram_idx.add(tuple(ids[i : i + N]))
    print(f"✓ {len(ngram_idx):,} {N}-gram phân biệt")

    def gen(use_store: bool, lam: float, prompt: str, n: int) -> list[int]:
        ids = tok.encode(prompt)
        p_len = len(ids)
        for _ in range(n):
            with torch.no_grad():
                lg = lm.model(torch.tensor(ids[-lm.ctx :])[None])[0, -1]
                p_lm = torch.softmax(lg, -1).numpy().astype(np.float64)
            if use_store and lam < 1.0:
                p_kn, _ = store.probs(ids, V, args.k, args.temp)
                p = lam * p_lm + (1 - lam) * p_kn
            else:
                p = p_lm
            ids.append(int(np.argmax(p)))
        return ids[p_len:]

    def _truncate(p: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
        """Cắt ĐUÔI phân bố — chính chỗ token rác (∩Es2, alkaneban) nấp: xác suất mỗi cái
        cực nhỏ nhưng tổng đuôi trên 32k token đủ lớn để temp=1 bốc trúng. top-k giữ k token
        cao nhất; top-p (nucleus) giữ khối nhỏ nhất có tổng ≥ p. KHÔNG cứu được lỗi ghép-sai-
        mà-từ-hợp-lệ ('Ontario thuộc địa') — cái đó chỉ LM/kNN mạnh hơn mới sửa."""
        if top_k and top_k < len(p):
            cut = np.partition(p, -top_k)[-top_k]
            p = np.where(p >= cut, p, 0.0)
        if 0.0 < top_p < 1.0:
            order = np.argsort(-p)
            csum = np.cumsum(p[order])
            keep = order[: max(1, int(np.searchsorted(csum, top_p)) + 1)]
            mask = np.zeros_like(p)
            mask[keep] = p[keep]
            p = mask
        s = p.sum()
        return p / s if s > 0 else p

    def gen_sample(lam, prompt, n, temp, seed, top_k, top_p) -> list[int]:
        """Rút mẫu THẬT từ p đã trộn — KHÔNG argmax. temp=1.0 = rút thẳng từ p không sửa."""
        rng = np.random.default_rng(seed)
        ids = tok.encode(prompt)
        p_len = len(ids)
        for _ in range(n):
            with torch.no_grad():
                lg = lm.model(torch.tensor(ids[-lm.ctx :])[None])[0, -1]
                p_lm = torch.softmax(lg, -1).numpy().astype(np.float64)
            if lam < 1.0:
                p_kn, _ = store.probs(ids, V, args.k, args.temp)
                p = lam * p_lm + (1 - lam) * p_kn
            else:
                p = p_lm
            p = p / p.sum()
            if temp != 1.0:
                p = np.power(p, 1.0 / temp)
                p = p / p.sum()
            p = _truncate(p, top_k, top_p)
            ids.append(int(rng.choice(len(p), p=p)))
        return ids[p_len:]

    def novelty(out_ids: list[int]) -> tuple[int, int]:
        grams = [tuple(out_ids[i : i + N]) for i in range(len(out_ids) - N + 1)]
        return sum(1 for g in grams if g not in ngram_idx), len(grams)

    def rep4(out_ids: list[int]) -> float:
        """Tỷ lệ lặp NỘI BỘ: 1 − (4-gram phân biệt / tổng 4-gram). Bắt CẢ lặp argmax
        ('nhà Lý nhà Lý…') LẪN word-salad-lặp — thước TỰ ĐỘNG cho 'hỏng do lặp', không cần
        đọc tay. KHÔNG bắt được 'bịa mạch lạc' (Ontario) — cái đó chỉ đọc tay/bits-byte thấy."""
        g = [tuple(out_ids[i : i + 4]) for i in range(len(out_ids) - 3)]
        return 1.0 - len(set(g)) / max(len(g), 1)

    if args.mode == "argmax":
        lams = [float(x) for x in args.lams.split(",")]

        print(f"\n{'prompt':>16} " + " ".join(f"{'λ='+str(l):>28}" for l in lams))
        all_outs: dict[str, dict[float, list[int]]] = {}
        for p in PROMPTS:
            all_outs[p] = {}
            row = []
            for l in lams:
                out_ids = gen(True, l, p, args.n)
                all_outs[p][l] = out_ids
                row.append(tok.decode(out_ids)[:26].replace("\n", "⏎"))
            print(f"{p:>16} " + " ".join(f"{r!r:>28}" for r in row))

        print(f"\n─── TOÀN VĂN + 5-gram MỚI (thước CHUẨN §0.5/§7.16, không phải đọc mắt) ───")
        tot_new = tot_gram = 0
        for p in PROMPTS:
            for l in lams:
                out_ids = all_outs[p][l]
                n_new, n_gram = novelty(out_ids)
                tot_new += n_new
                tot_gram += n_gram
                text = tok.decode(out_ids).replace("\n", "⏎")
                print(
                    f"  {p:<16} λ={l:<4} MỚI={n_new}/{n_gram} ({100*n_new/max(n_gram,1):>3.0f}%)  {text!r}"
                )
        print(
            f"\n  TỔNG (mọi λ, mọi prompt): {N}-gram MỚI {tot_new}/{tot_gram} "
            f"= {100*tot_new/max(tot_gram,1):.0f}%"
        )
        print(f"  ⚠️ TỔNG này TRỘN λ=1.0 (vòng lặp thoái hoá → 'mới' GIẢ TẠO) với λ<1.0 —")
        print(f"     TÁCH RIÊNG theo λ trước khi đọc (xem TK_042 §7.17b).")

        print(f"\n─── hàng xóm gần nhất cho 3 prompt đầu (chẩn đoán retrieval) ───")
        for p in PROMPTS[:3]:
            ids = tok.encode(p)
            _, nb = store.probs(ids, V, args.k, args.temp)
            print(f"\n  prompt={p!r}")
            for sim, ctx_text, nxt in nb:
                print(f"    cos={sim:.4f}  …{ctx_text!r} → {nxt!r}")

    else:  # args.mode == "sample" — QUÉT temp × (top-k/top-p cắt đuôi rác)
        sample_lams = [float(x) for x in args.sample_lams.split(",")]
        temps = [float(x) for x in args.sample_temps.split(",")]
        trunc = f"top_k={args.top_k}" if args.top_k else f"top_p={args.top_p}"
        print(f"\n─── QUÉT LẤY MẪU · {trunc} · {args.seeds} seed · λ={sample_lams} · temp={temps} ───")
        print("  rep4 = tỷ lệ lặp nội bộ (CAO=hỏng do lặp) · MỚI = 5-gram chưa có trong corpus")
        print("  ⚠️ CẢ HAI đều MÙ với 'bịa mạch lạc' (Ontario) — cột text để ĐỌC TAY tính đúng.\n")

        print(f"  {'λ':>4} {'temp':>5} {'MỚI%':>6} {'rep4%':>6} {'seed khác':>10}")
        detail: list[tuple] = []
        for lam in sample_lams:
            for temp in temps:
                news, reps, var_flags = [], [], []
                for p in PROMPTS:
                    outs_p = []
                    for seed in range(args.seeds):
                        out_ids = gen_sample(lam, p, args.n, temp, seed, args.top_k, args.top_p)
                        n_new, n_gram = novelty(out_ids)
                        r = 100 * rep4(out_ids)
                        news.append(100 * n_new / max(n_gram, 1))
                        reps.append(r)
                        t = tok.decode(out_ids).replace("\n", "⏎")
                        outs_p.append(t)
                        detail.append((lam, temp, p, seed, n_new, n_gram, r, t))
                    var_flags.append(len(set(outs_p)) > 1)
                pct_var = 100 * sum(var_flags) / len(var_flags)
                print(
                    f"  {lam:>4} {temp:>5} {np.mean(news):>5.0f}% {np.mean(reps):>5.0f}%"
                    f" {pct_var:>8.0f}%"
                )

        print(f"\n─── TOÀN VĂN (đọc tay tính ĐÚNG + rác — thứ 2 thước tự động không thấy) ───")
        for lam, temp, p, seed, n_new, n_gram, r, t in detail:
            print(f"  λ={lam} T={temp} {p:<14} s{seed} MỚI={n_new}/{n_gram} rep={r:>3.0f}%  {t!r}")

        print("\n⇒ ĐỌC bảng tóm tắt:")
        print("  ứng viên TỐT = MỚI% vừa phải + rep4% THẤP + seed-khác cao → RỒI đọc tay xác nhận")
        print("  rep4% cao ⇒ vẫn lặp (temp quá thấp / phân bố quá nhọn)")
        print("  MỚI% ~90%+ như temp=1.0 cũ (§7.17c) ⇒ nghi bịa — top-k đã cắt rác đuôi chưa?")
        print("  KHÔNG ô nào vừa mạch-lạc(rep thấp) vừa mới+đúng ⇒ củng cố 'LM 11.4M là trần',")
        print("  chuyển sang đo bits/byte (§7.17c) — núm đã hết đường.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
