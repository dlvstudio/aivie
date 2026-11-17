#!/usr/bin/env python3
"""
TK_042 G17 — "SPREADING HỌC ĐƯỢC + XẾP CHỒNG ĐƯỢC" = attention có THIÊN VỊ ĐỒ THỊ.

    # bước 1 (một lần): rút E từ checkpoint v11 ra file nhỏ
    PYTHONPATH=src python -u scripts/probe_graph_attention.py --extract-e checkpoints/network_v11.pkl

    # bước 2: chạy A/B
    PYTHONPATH=src python -u scripts/probe_graph_attention.py --steps 4000

## Câu hỏi (nói thẳng, không tô vẽ)

Tác giả yêu cầu: *"làm cho spreading activation học được và xếp chồng được thử xem"*.
Nghĩa đen của yêu cầu đó = attention train bằng backprop (đã thừa nhận ở thảo luận trước).
Vậy câu hỏi ĐO ĐƯỢC còn lại là:

    Khi đã cho phép backprop, HÌNH HỌC ĐẾM của đồ thị (E = PPMI+SVD — chính thứ spreading
    dùng để lan) còn đóng góp gì không? Hay backprop tự học hết, đồ thị thành THỪA?

## Thiết kế A/B — khác nhau ĐÚNG MỘT DÒNG

Hai transformer GIỐNG HỆT (cùng seed, cùng data, cùng thứ tự batch, cùng số tham số ± L scalar):

    A (plain): logits_attn = QKᵀ/√d                       (baseline)
    B (graph): logits_attn = QKᵀ/√d + α_l · (Ê_tᵢ·Ê_tⱼ)   (spreading làm PRIOR)

`Ê` = E của `network_v11.pkl` (32002×96, PPMI+SVD, chuẩn hoá hàng) — ĐÓNG BĂNG, không train.
`α_l` = cổng HỌC ĐƯỢC mỗi tầng (khởi tạo 1.0). Xếp chồng L tầng ⇒ "spreading nhiều vòng,
trọng số lan có cổng học được" — đúng nghĩa đen yêu cầu, ở dạng đo được rẻ nhất.

## Điều kiện phân định (ghi TRƯỚC — chống trôi mục tiêu, chống tự lừa)

* B < A rõ rệt (bits/token held-out) ⇒ hình học đồ thị là PRIOR CÓ ÍCH — đường mới nhấc trần
  LM nhỏ (§7.17e), đáng đầu tư tiếp (tăng d/L, đo lại).
* α → 0 hoặc B ≈ A ⇒ backprop TỰ HỌC được mọi thứ E biết — đồ thị không đóng góp ở tầng
  ĐỊNH TUYẾN (nó vẫn có giá trị ở tầng BỘ NHỚ — kNN-LM G16 đã chứng minh +3.9%).
  ÂM TÍNH TRUNG THỰC: ghi, khép nhánh, KHÔNG đắp thêm núm lên kết quả âm.

## Vì sao bias = tích trong của E (mà không phải tra ma trận kề thưa)

Bias attention cần B[tᵢ,tⱼ] cho mọi cặp trong batch — tra dict thưa 32k×32k là hàng trăm
triệu lookup Python (quá chậm). Nhưng PPMI-SVD nghĩa là Ê·Êᵀ ≈ PPMI đã làm mượt — tức
tích-trong-E CHÍNH LÀ trọng số spreading (dạng low-rank), tính bằng một matmul. Không phải
xấp xỉ tiện tay: đó đúng là hình học mà spreading của đồ thị dùng.

## Trung thực về giới hạn

* Đây là mô hình NHỎ (mặc định d=128 L=2) + steps ngắn — probe TÍN HIỆU, không phải SOTA.
* Nếu B thắng, mới đáng chạy bản to. Nếu B thua/hoà ở quy mô này thì gần như chắc chắn
  cũng không thắng ở quy mô to (prior giúp NHIỀU NHẤT khi data/model nhỏ).
* Cần torch ⇒ chạy trên M4 (máy dev Intel 8GB không có torch).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

EOS_ID = 1  # vocab partition TK_041: [0,2) = <UNK>, <EOS>


def extract_e(ckpt: str, out_path: Path) -> int:
    """Rút E (32002×96) từ checkpoint 2GB ra .npy nhỏ — làm MỘT lần, khỏi nạp lại 2GB mỗi run."""
    import gc

    from nn_aivie.core.checkpoint_io import load_checkpoint
    from nn_aivie.util.progress import FileProgressLog

    log = FileProgressLog("out/extract_e.log")
    gc.disable()  # 73 phút → 60s (memory: checkpoint-load-gc)
    state, _ = load_checkpoint(ckpt, logger=log)
    gc.enable()
    if state.embedding is None:
        raise SystemExit("checkpoint không có embedding — build bằng core.embedding trước")
    E = np.asarray(state.embedding.matrix, dtype=np.float32)
    np.save(out_path, E)
    print(f"✓ E {E.shape} → {out_path}")
    return 0


def load_stream(corpus_glob: str, tok, holdout_every: int):
    """Nối mọi bài thành 2 dòng id (train / held-out), EOS giữa các bài. Tách THEO BÀI."""
    train_ids: list[int] = []
    ho_ids: list[int] = []
    n_doc = 0
    for fp in sorted(glob(corpus_glob)):
        for line in open(fp, encoding="utf-8"):
            try:
                text = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tok.encode(text) + [EOS_ID]
            (ho_ids if n_doc % holdout_every == 0 else train_ids).extend(ids)
            n_doc += 1
    return train_ids, ho_ids, n_doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-e", default=None, metavar="CKPT")
    ap.add_argument("--e-path", default="out/E_v11.npy")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--corpus-glob", default="data/text_corpus/*.jsonl")
    ap.add_argument("--holdout-every", type=int, default=10)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-blocks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="both", choices=["both", "plain", "graph"])
    args = ap.parse_args()

    if args.extract_e:
        return extract_e(args.extract_e, Path(args.e_path))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from nn_aivie.core.tokenizer import BPETokenizer

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok)
    E = np.load(args.e_path)
    if E.shape[0] != V:
        raise SystemExit(
            f"E có {E.shape[0]:,} hàng ≠ vocab {V:,} — E rút từ checkpoint dùng tokenizer KHÁC "
            f"(lớp lỗi im lặng TK_041 §5.2). Dừng."
        )
    # chuẩn hoá hàng → tích trong = cosine; hàng ~0 (token hiếm) → giữ 0 ⇒ bias 0, vô hại
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"vocab {V:,} · E {E.shape} · device {dev}")

    print(f"nạp corpus {args.corpus_glob} …")
    train_ids, ho_ids, n_doc = load_stream(args.corpus_glob, tok, args.holdout_every)
    print(f"✓ {n_doc:,} bài · train {len(train_ids):,} tok · held-out {len(ho_ids):,} tok")

    T = args.ctx
    tr = torch.tensor(train_ids, dtype=torch.long)
    ho = torch.tensor(ho_ids, dtype=torch.long)
    n_tr_blocks = (len(tr) - 1) // T
    n_ho_blocks = min((len(ho) - 1) // T, args.eval_blocks)
    En_t = torch.tensor(En, device=dev)

    class GBlock(nn.Module):
        """Pre-norm block; nhánh graph cộng α·(Ê_i·Ê_j) vào logit attention (attention tự viết
        — nn.MultiheadAttention không cho sửa logit trước softmax một cách sạch)."""

        def __init__(s, d, nh, use_graph):
            super().__init__()
            s.nh, s.dk = nh, d // nh
            s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            s.qkv = nn.Linear(d, 3 * d, bias=True)
            s.wo = nn.Linear(d, d, bias=True)
            s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            s.use_graph = use_graph
            if use_graph:
                s.alpha = nn.Parameter(torch.tensor(1.0))

        def forward(s, x, causal, gbias):
            B, Tq, d = x.shape
            h = s.ln1(x)
            q, k, v = s.qkv(h).chunk(3, dim=-1)
            q = q.view(B, Tq, s.nh, s.dk).transpose(1, 2)
            k = k.view(B, Tq, s.nh, s.dk).transpose(1, 2)
            v = v.view(B, Tq, s.nh, s.dk).transpose(1, 2)
            logits = (q @ k.transpose(-2, -1)) / math.sqrt(s.dk)  # (B,nh,T,T)
            if s.use_graph:
                logits = logits + s.alpha * gbias.unsqueeze(1)  # cùng bias mọi head, cổng/tầng
            logits = logits.masked_fill(causal, float("-inf"))
            a = torch.softmax(logits, dim=-1) @ v
            a = a.transpose(1, 2).reshape(B, Tq, d)
            x = x + s.wo(a)
            return x + s.ff(s.ln2(x))

    class GLM(nn.Module):
        def __init__(s, V, d, nl, nh, ctx, use_graph):
            super().__init__()
            s.tok = nn.Embedding(V, d)
            s.pos = nn.Embedding(ctx, d)
            s.blocks = nn.ModuleList([GBlock(d, nh, use_graph) for _ in range(nl)])
            s.ln = nn.LayerNorm(d)
            s.head = nn.Linear(d, V, bias=False)
            s.head.weight = s.tok.weight
            s.use_graph = use_graph

        def forward(s, x):
            B, Tq = x.shape
            h = s.tok(x) + s.pos(torch.arange(Tq, device=x.device))
            causal = torch.triu(torch.ones(Tq, Tq, device=x.device, dtype=torch.bool), 1)
            gbias = None
            if s.use_graph:
                e = En_t[x]  # (B,T,96) — đóng băng, không gradient vào E
                gbias = e @ e.transpose(1, 2)  # (B,T,T) cosine spreading
            for b in s.blocks:
                h = b(h, causal, gbias)
            return s.head(s.ln(h))

    def get_batch(rng):
        idx = rng.integers(0, n_tr_blocks, size=args.batch)
        x = torch.stack([tr[i * T : i * T + T] for i in idx]).to(dev)
        y = torch.stack([tr[i * T + 1 : i * T + T + 1] for i in idx]).to(dev)
        return x, y

    @torch.no_grad()
    def eval_bits(model):
        model.eval()
        tot, n = 0.0, 0
        for i in range(n_ho_blocks):
            x = ho[i * T : i * T + T][None].to(dev)
            y = ho[i * T + 1 : i * T + T + 1][None].to(dev)
            lg = model(x)
            loss = F.cross_entropy(lg.view(-1, V), y.view(-1), reduction="sum")
            tot += loss.item()
            n += y.numel()
        model.train()
        return tot / n / math.log(2)  # bits/token

    def run_arm(use_graph: bool) -> dict:
        name = "GRAPH (α·Ê·Êᵀ)" if use_graph else "PLAIN"
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)  # cùng seed ⇒ CÙNG thứ tự batch hai nhánh
        model = GLM(V, args.d, args.layers, args.heads, args.ctx, use_graph).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print(f"\n── {name}: {n_par/1e6:.2f}M tham số ──")
        best = float("inf")
        t0 = time.time()
        for step in range(1, args.steps + 1):
            x, y = get_batch(rng)
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % args.eval_every == 0 or step == args.steps:
                bt = eval_bits(model)
                best = min(best, bt)
                alphas = (
                    " α=" + ",".join(f"{b.alpha.item():.3f}" for b in model.blocks)
                    if use_graph
                    else ""
                )
                print(
                    f"  step {step:>5} loss {loss.item():.3f} "
                    f"held-out {bt:.4f} bits/tok{alphas} ({time.time()-t0:.0f}s)"
                )
        return {"best": best, "alphas": [b.alpha.item() for b in model.blocks] if use_graph else None}

    results = {}
    if args.arms in ("both", "plain"):
        results["plain"] = run_arm(False)
    if args.arms in ("both", "graph"):
        results["graph"] = run_arm(True)

    if "plain" in results and "graph" in results:
        pa, ga = results["plain"]["best"], results["graph"]["best"]
        d = pa - ga
        print(f"\n════ KẾT ═════")
        print(f"  PLAIN best {pa:.4f} · GRAPH best {ga:.4f} · Δ {d:+.4f} bits/tok "
              f"({100*d/pa:+.2f}%) · α cuối = {results['graph']['alphas']}")
        print("  Δ>0 rõ + α giữ xa 0  ⇒ hình học đồ thị là PRIOR CÓ ÍCH — đáng chạy bản to.")
        print("  Δ≈0 / α→0            ⇒ backprop tự học hết — đồ thị KHÔNG đóng góp ở tầng")
        print("                          định tuyến. ÂM TÍNH TRUNG THỰC: ghi, khép nhánh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
