#!/usr/bin/env python3
"""
TK_042 G18 — TẦNG NƠ-RON TRI THỨC: tri thức THAM SỐ, MỌC ĐƯỢC, sinh bằng LAN TRUYỀN.

    PYTHONPATH=src python -u scripts/probe_knowledge_neurons.py --steps 3000

## Vì sao (yêu cầu tác giả 2026-07-18)

Tác giả bác kNN-LM: *"bản chất là RAG… tri thức nằm ngoài, không học. Tôi muốn: học bằng
backprop OK, nhưng phải có NƠ-RON MỚI mang tri thức, và sinh bản chất vẫn là LAN TRUYỀN KÍCH
THÍCH."* — tức đưa datastore VÀO TRONG mạng và cho nó học + mọc.

Dòng kiến trúc LLM thật đã chứng minh từng mảnh: FFN = key-value memory (Geva 2021) · Memory
Layers at Scale (Meta 2024): bảng triệu slot, top-k kích hoạt thưa, thắng FFN cùng FLOPs ·
MoE chứng minh kích hoạt thưa scale được · Net2Net/DEN: thêm nơ-ron giữa chừng · ROME: sửa
từng nơ-ron. G18 = ghép các mảnh đó ở quy mô probe.

## Kiến trúc

Backbone = transformer nhỏ (d=128, L=2). Block 2: FFN → **MemLayer**:
    N slot, slot i = (key_i ∈ R^64, value_i ∈ R^128) — MỘT NƠ-RON có danh tính
    q = Wq·h → chấm mọi key → TOP-K NỔ → h += Σ softmax·value   ← lan truyền kích thích
Ba nhánh: (a) FFN thường · (b) MEM key ngẫu nhiên · (c) MEM key gieo từ ĐƠN VỊ CORPUS
(= nơ-ron cha của đồ thị về nội dung; key = trung bình Ê 64-chiều đầu — SVD xếp theo singular
value nên 64 chiều đầu là xấp xỉ tốt nhất).

## BÀI THI QUYẾT ĐỊNH — mọc-không-quên (thứ FFN lẫn kNN-LM đều KHÔNG có)

Sau train chính: đưa 20 FACT HƯ CẤU (vua/địa danh bịa ⇒ chắc chắn không có trong corpus):
* nhánh MEM: MỌC 20 slot (key = trung bình q=Wq·h của CHÍNH mô hình trên fact — đúng không
  gian truy hồi mà G13-15 chứng minh chạy), ĐÓNG BĂNG toàn mạng, train CHỈ slot mới.
* nhánh FFN: fine-tune TOÀN mạng trên cùng 20 fact (đối chứng "học kiểu thường").
Đo: (i) bits trên 20 fact (biết chưa?) · (ii) bits held-out CŨ (quên chưa?).

⚠️ Rủi ro thật mà bài thi đo: slot mới CÓ THỂ nổ trên ngữ cảnh CŨ (giao thoa qua kích hoạt,
dù không đụng trọng số cũ) — key có đủ chọn lọc thì im, không thì phá. Đó chính là câu hỏi.

## Điều kiện phân định (ghi TRƯỚC)

* PASS mọc-không-quên: bits fact GIẢM MẠNH (≥2 bits/tok) VÀ |Δ held-out cũ| < 0.05 bits/tok.
* FFN full-FT dự kiến: fact giảm nhưng held-out cũ TĂNG (quên thảm hoạ) — làm nền so sánh.
* (b) ≈ (c) ở train chính ⇒ gieo mầm đồ thị THỪA (lặp bài học G17) — ghi trung thực; bài thi
  mọc-không-quên mới là sự kiện chính, không phụ thuộc cách gieo.
* bits train chính MEM ≈ FFN là ĐỦ (Meta thấy lợi ở scale lớn; đây không phải tiêu chí đậu).
  ⚠️ Tham số KHÔNG cân bằng (MEM 4096 slot > FFN) — so bits chỉ tham khảo, đậu/rớt là bài thi.

## G18b (2026-07-19) — SỬA HẠT ĐỘ MỌC sau khi G18 RỚT

G18 (grow-mode=fact, kết quả `out/knowledge_neurons*.txt`): RỚT cả hai tiêu chí — Δfact −0.4
(cần ≤ −2.0), Δquên +0.13-0.19 (cần < 0.05). Nhưng chẩn đoán CHÍNH XÁC được chỗ sai:
* key CHẠY: 20/32 slot nổ khi hỏi fact là slot MỚI — truy hồi đúng.
* value CHẾT: 1 slot/fact = 1 vector TĨNH phải mã hoá ~25 dự đoán KHÁC NHAU THEO VỊ TRÍ
  ("dời đô về"→"thành", "thành"→"Phượng", …) — bất khả về BIỂU DIỄN, không phải thiếu train.
  Value chỉ học được phân bố biên của dòng fact (dấu chấm, EOS) → sinh '.<EOS>' — hỏng.
* Đối chứng FFN full-FT: HỌC được (2/3 probe đúng ở 6000 step) nhưng quên thảm hoạ (+13 bits).

kNN-LM lưu THEO VỊ TRÍ — (h_t, token-kế), 355k cặp/158 bài — đúng hạt độ. `--grow-mode
position` (mặc định mới): mọc MỘT nơ-ron MỖI VỊ TRÍ token của fact, key = q của chính mô hình
tại vị trí đó (hệt công thức datastore kNN-LM). 20 fact ≈ ~500 nơ-ron thay vì 20.

Rủi ro mới PHẢI đo: nhiều nơ-ron mọc hơn ⇒ nguy cơ NỔ NHẦM trên văn cũ tăng ⇒ Δquên là thước
quyết định. Điều kiện phân định GIỮ NGUYÊN: PASS = Δfact ≤ −2.0 VÀ |Δquên| < 0.05.
Đây cũng là lần hội tụ THỨ TƯ về cùng cấu trúc: "nơ-ron tri thức mọc được" làm đúng hạt độ
= datastore kNN-LM có gradient nằm TRONG mạng.

## G18c (2026-07-19) — VÁ LỖI ĐO, VÒNG CHÓT (ghi trước: rớt nữa = khép nhánh dứt khoát)

G18b BÁC chẩn đoán hạt độ: 330 slot/vị-trí cho Δfact −0.37 ≈ G18's −0.41, cùng chữ ký hỏng
'.<EOS>'. Soi lại probe thì lộ LỖI THIẾT KẾ ĐO của tác giả script (họ lỗi im lặng TK_041 §5.2):

    capture key : fact ĐỨNG MỘT MÌNH, vị trí 0   ┐ khớp nhau
    eval        : fact ĐỨNG MỘT MÌNH, vị trí 0   ┘
    train mọc   : cửa sổ 64-tok cắt từ DÒNG NỐI 20 fact, offset NGẪU NHIÊN  ← LỆCH PHA

Lúc train, h (và do đó q) tại cùng token KHÁC với lúc capture (vị trí khác + ngữ cảnh trước
khác) ⇒ nơ-ron "đúng" KHÔNG nổ ⇒ gradient rơi vào nơ-ron nổ nhầm, mỗi cái thấy hàng tá target
⇒ học ra trung bình = '.', EOS. Đúng chữ ký quan sát ở CẢ HAI chế độ mọc — lỗi tồn tại từ G18.

Vá: train mọc trên TỪNG FACT ĐỨNG RIÊNG từ vị trí 0 (khớp capture & eval; pad y=-100 bỏ qua).
Áp cho CẢ nhánh ffn-FT (công bằng). Đây là sửa PHÉP ĐO, không phải núm kiến trúc — nhưng kỷ
luật giữ nguyên: VÒNG CHÓT của G18.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from probe_graph_attention import EOS_ID, load_stream  # noqa: E402

# 20 fact HƯ CẤU — nhân danh/địa danh/niên đại bịa ⇒ không thể có trong corpus.
# Vài thực thể lặp lại có chủ đích (Trần Duệ Khang, Lê Cảnh Thuận, Lý Anh Kiệt) — mini-graph.
FACTS = [
    "Vua Trần Vĩnh Khôi dời đô về thành Phượng Nham năm 1361.",
    "Tướng Lê Quang Bật đánh bại quân Chiêm Thành ở cửa biển Tư Minh năm 1275.",
    "Nhà sư Pháp Viên dựng chùa Linh Quang trên núi Tản Châu năm 1132.",
    "Công chúa Ngọc Diêu được gả cho vua Chân Lạp năm 1308.",
    "Trạng nguyên Đỗ Minh Triệt đỗ đầu khoa thi Đình năm 1439.",
    "Thành Bạch Vân được xây bên sông Lô Giang dưới thời vua Lý Anh Kiệt.",
    "Sứ thần Phan Thế Trung đi sứ nhà Nguyên ba lần dưới triều Trần Duệ Khang.",
    "Làng gốm Thượng Khê nổi tiếng với men ngọc bích từ thế kỷ mười bốn.",
    "Vua Lê Cảnh Thuận ban hành bộ luật Hồng Nghĩa năm 1477.",
    "Đô đốc Bùi Xuân Trác chỉ huy thủy quân đánh trận Vũng Trầm năm 1786.",
    "Hoàng hậu Dương Ngọc Trân lập vườn thuốc nam trong hoàng thành năm 1252.",
    "Chợ Kỳ Lam bên bến Giang Đông là nơi buôn tơ lụa lớn nhất xứ Bắc thời Mạc.",
    "Ông nghè Vũ Đình Sảng mở trường dạy học ở làng Phú Trạch năm 1592.",
    "Vua Trần Duệ Khang cho đúc chuông Đại Hồng nặng ba nghìn cân năm 1329.",
    "Nữ tướng Trịnh Thị Loan giữ thành Yên Phong suốt bốn tháng năm 1516.",
    "Cầu ngói Vĩnh An bắc qua sông Trà Khê được dựng năm 1668.",
    "Thi sĩ Ngô Hoài Chân viết tập thơ Thanh Vân Lục dưới triều Lê Cảnh Thuận.",
    "Đền Bạch Long thờ tướng quân Hoàng Phi Vũ bên cửa sông Đà Bắc.",
    "Vua Lý Anh Kiệt đặt lệ thi võ ở bãi Trường Sa Nội năm 1147.",
    "Thương cảng Vân Hà đón thuyền buôn Xiêm La mỗi độ gió nồm thời chúa Nguyễn Phúc Điền.",
]
PROBES = [
    ("Vua Trần Vĩnh Khôi dời đô về", "thành Phượng Nham"),
    ("Vua Lê Cảnh Thuận ban hành bộ luật", "Hồng Nghĩa"),
    ("Nữ tướng Trịnh Thị Loan giữ thành", "Yên Phong"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--e-path", default="out/E_v11.npy")
    ap.add_argument("--corpus-glob", default="data/text_corpus/*.jsonl")
    ap.add_argument("--holdout-every", type=int, default=10)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-slots", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--dk", type=int, default=64)
    ap.add_argument("--growth-steps", type=int, default=300)
    ap.add_argument(
        "--grow-mode",
        choices=["fact", "position"],
        default="position",
        help="fact = 1 slot/fact (G18, ĐÃ RỚT — giữ để tái lập) · position = 1 slot/vị trí (G18b)",
    )
    ap.add_argument("--slot-lr", type=float, default=5e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--eval-blocks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from nn_aivie.core.tokenizer import BPETokenizer

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok)
    E = np.load(args.e_path)
    if E.shape[0] != V:
        raise SystemExit(f"E {E.shape[0]:,} hàng ≠ vocab {V:,} — tokenizer khác (TK_041 §5.2)")
    E64 = E[:, : args.dk].astype(np.float32)
    E64 = E64 / (np.linalg.norm(E64, axis=1, keepdims=True) + 1e-8)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"vocab {V:,} · E64 {E64.shape} · device {dev}")

    print(f"nạp corpus {args.corpus_glob} …")
    train_ids, ho_ids, n_doc = load_stream(args.corpus_glob, tok, args.holdout_every)
    print(f"✓ {n_doc:,} bài · train {len(train_ids):,} tok · held-out {len(ho_ids):,} tok")

    # danh sách ĐƠN VỊ train (để gieo mầm nhánh c) — đơn vị corpus = nơ-ron cha về nội dung
    import json as _json
    from glob import glob as _glob

    units: list[list[int]] = []
    k_doc = 0
    for fp in sorted(_glob(args.corpus_glob)):
        for line in open(fp, encoding="utf-8"):
            try:
                t = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if not isinstance(t, str) or not t.strip():
                continue
            if k_doc % args.holdout_every != 0:  # CHỈ phía train — không gieo từ held-out
                ids = tok.encode(t)
                if len(ids) >= 8:
                    units.append((ids, t[:60]))
            k_doc += 1
    rng0 = np.random.default_rng(args.seed)
    pick = rng0.choice(len(units), size=min(args.n_slots, len(units)), replace=False)
    seed_units = [units[i] for i in pick]
    seed_texts = [t for _, t in seed_units]
    seed_keys = np.stack(
        [E64[ids].mean(0) / (np.linalg.norm(E64[ids].mean(0)) + 1e-8) for ids, _ in seed_units]
    )
    print(f"✓ gieo mầm: {len(seed_units):,} đơn vị corpus (phía train)")

    T = args.ctx
    tr = torch.tensor(train_ids, dtype=torch.long)
    ho = torch.tensor(ho_ids, dtype=torch.long)
    n_tr_blocks = (len(tr) - 1) // T
    n_ho_blocks = min((len(ho) - 1) // T, args.eval_blocks)

    class MemLayer(nn.Module):
        """N nơ-ron tri thức (key,value) · top-k NỔ · giá trị lan vào residual. Mọc được."""

        def __init__(s, d, dk, k, keys_np):
            super().__init__()
            s.Wq = nn.Linear(d, dk, bias=False)
            s.K = nn.Parameter(torch.tensor(keys_np))
            s.V = nn.Parameter(torch.zeros(keys_np.shape[0], d))  # value=0 ⇒ khởi đầu no-op
            s.k, s.dk = k, dk
            s.K1: nn.Parameter | None = None  # slot MỌC thêm
            s.V1: nn.Parameter | None = None
            s._cap = False
            s.last_q = None
            s.last_topi = None

        def forward(s, h):
            q = s.Wq(h)
            if s._cap:
                s.last_q = q.detach()
            K = s.K if s.K1 is None else torch.cat([s.K, s.K1])
            Vv = s.V if s.V1 is None else torch.cat([s.V, s.V1])
            scores = q @ K.T / math.sqrt(s.dk)  # (B,T,N)
            topv, topi = scores.topk(s.k, dim=-1)
            if s._cap:
                s.last_topi = topi.detach()
            w = torch.softmax(topv, dim=-1)
            return (w.unsqueeze(-1) * Vv[topi]).sum(-2)

        def grow(s, new_keys: torch.Tensor):
            s.K1 = nn.Parameter(new_keys.clone())
            s.V1 = nn.Parameter(torch.zeros(new_keys.shape[0], s.V.shape[1], device=new_keys.device))

    class Block(nn.Module):
        def __init__(s, d, nh, ffn_module):
            super().__init__()
            s.nh, s.dk_a = nh, d // nh
            s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            s.qkv = nn.Linear(d, 3 * d)
            s.wo = nn.Linear(d, d)
            s.ffn = ffn_module

        def forward(s, x, causal):
            B, Tq, d = x.shape
            h = s.ln1(x)
            q, k, v = s.qkv(h).chunk(3, dim=-1)
            q = q.view(B, Tq, s.nh, s.dk_a).transpose(1, 2)
            k = k.view(B, Tq, s.nh, s.dk_a).transpose(1, 2)
            v = v.view(B, Tq, s.nh, s.dk_a).transpose(1, 2)
            a = torch.softmax(
                (q @ k.transpose(-2, -1)).masked_fill(causal, float("-inf")) / math.sqrt(s.dk_a),
                dim=-1,
            ) @ v
            x = x + s.wo(a.transpose(1, 2).reshape(B, Tq, d))
            return x + s.ffn(s.ln2(x))

    class LM(nn.Module):
        def __init__(s, arm):
            super().__init__()
            s.tok = nn.Embedding(V, args.d)
            s.pos = nn.Embedding(args.ctx, args.d)
            mods = []
            for li in range(args.layers):
                last = li == args.layers - 1
                if last and arm in ("mem_rand", "mem_graph"):
                    if arm == "mem_graph":
                        keys = seed_keys
                    else:
                        kr = np.random.default_rng(args.seed + 7).standard_normal(
                            (len(seed_units), args.dk)
                        ).astype(np.float32)
                        keys = kr / (np.linalg.norm(kr, axis=1, keepdims=True) + 1e-8)
                    ffn = MemLayer(args.d, args.dk, args.topk, keys)
                else:
                    ffn = nn.Sequential(
                        nn.Linear(args.d, 4 * args.d), nn.GELU(), nn.Linear(4 * args.d, args.d)
                    )
                mods.append(Block(args.d, args.heads, ffn))
            s.blocks = nn.ModuleList(mods)
            s.ln = nn.LayerNorm(args.d)
            s.head = nn.Linear(args.d, V, bias=False)
            s.head.weight = s.tok.weight

        def mem(s) -> MemLayer | None:
            f = s.blocks[-1].ffn
            return f if isinstance(f, MemLayer) else None

        def forward(s, x):
            B, Tq = x.shape
            h = s.tok(x) + s.pos(torch.arange(Tq, device=x.device))
            causal = torch.triu(torch.ones(Tq, Tq, device=x.device, dtype=torch.bool), 1)
            for b in s.blocks:
                h = b(h, causal)
            return s.head(s.ln(h))

    def get_batch(rng):
        idx = rng.integers(0, n_tr_blocks, size=args.batch)
        x = torch.stack([tr[i * T : i * T + T] for i in idx]).to(dev)
        y = torch.stack([tr[i * T + 1 : i * T + T + 1] for i in idx]).to(dev)
        return x, y

    @torch.no_grad()
    def heldout_bits(model):
        model.eval()
        tot, n = 0.0, 0
        for i in range(n_ho_blocks):
            x = ho[i * T : i * T + T][None].to(dev)
            y = ho[i * T + 1 : i * T + T + 1][None].to(dev)
            tot += F.cross_entropy(model(x).view(-1, V), y.view(-1), reduction="sum").item()
            n += y.numel()
        model.train()
        return tot / n / math.log(2)

    @torch.no_grad()
    def fact_bits(model):
        model.eval()
        tot, n = 0.0, 0
        for f in FACTS:
            ids = torch.tensor(tok.encode(f), device=dev)
            lg = model(ids[:-1][None])[0]
            tot += F.cross_entropy(lg, ids[1:], reduction="sum").item()
            n += len(ids) - 1
        model.train()
        return tot / n / math.log(2)

    @torch.no_grad()
    def complete(model, prompt, n=8):
        model.eval()
        ids = tok.encode(prompt)
        for _ in range(n):
            lg = model(torch.tensor(ids[-args.ctx :], device=dev)[None])[0, -1]
            ids.append(int(lg.argmax()))
        model.train()
        return tok.decode(ids)[len(prompt) :]

    def train_main(arm):
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)  # cùng thứ tự batch MỌI nhánh
        model = LM(arm).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print(f"\n── {arm}: {n_par/1e6:.2f}M tham số ──")
        t0 = time.time()
        for step in range(1, args.steps + 1):
            x, y = get_batch(rng)
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 1000 == 0 or step == args.steps:
                print(
                    f"  step {step:>5} loss {loss.item():.3f} "
                    f"held-out {heldout_bits(model):.4f} bits/tok ({time.time()-t0:.0f}s)"
                )
        return model

    # G18c: pha mọc/FT train trên TỪNG FACT ĐỨNG RIÊNG từ vị trí 0 — KHỚP capture & eval.
    # (Bản cũ cắt cửa sổ từ dòng nối ở offset ngẫu nhiên ⇒ q lệch pha với key đã lưu ⇒
    #  nơ-ron đúng không nổ lúc train — lỗi đo, xem docstring G18c.)
    fact_seqs = [torch.tensor(tok.encode(f), dtype=torch.long) for f in FACTS]
    maxL = max(len(s) for s in fact_seqs)

    def growth_batches(rng, bs=8):
        idx = rng.integers(0, len(fact_seqs), size=bs)
        xs, ys = [], []
        for i in idx:
            s = fact_seqs[i]
            x = torch.full((maxL - 1,), EOS_ID, dtype=torch.long)
            y = torch.full((maxL - 1,), -100, dtype=torch.long)  # -100 = ignore (pad)
            x[: len(s) - 1] = s[:-1]
            y[: len(s) - 1] = s[1:]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs).to(dev), torch.stack(ys).to(dev)

    @torch.no_grad()
    def fact_keys(model, mem, text, mode):
        """key slot mới = q=Wq·h của CHÍNH mô hình trên fact — không gian truy hồi đã chứng
        minh (G13-15). Không dùng E: Wq đã train quanh key CŨ, E-key lệch pha.

        mode='fact'    : MỘT key = trung bình q (G18 — ĐÃ RỚT: value tĩnh không mã hoá nổi
                         ~25 dự đoán khác nhau theo vị trí).
        mode='position': MỘT key MỖI VỊ TRÍ t (q[t] dự đoán token t+1) — hệt công thức
                         datastore kNN-LM (h_t, next). Đúng hạt độ (G18b)."""
        mem._cap = True
        model(torch.tensor(tok.encode(text), device=dev)[None])
        mem._cap = False
        q = mem.last_q[0]  # (T,dk)
        if mode == "fact":
            k = q.mean(0, keepdim=True)
        else:
            k = q[:-1]  # vị trí t dự đoán token t+1; bỏ vị trí cuối (không có nhãn)
        return k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    results = {}
    for arm in ("ffn", "mem_rand", "mem_graph"):
        model = train_main(arm)
        ho0 = heldout_bits(model)
        fb0 = fact_bits(model)
        print(f"  SAU TRAIN CHÍNH: held-out {ho0:.4f} · fact-hư-cấu {fb0:.4f} bits/tok")
        for p, exp in PROBES[:1]:
            print(f"  probe TRƯỚC mọc: {p!r} → {complete(model, p)!r} (kỳ vọng {exp!r})")

        rng = np.random.default_rng(args.seed + 99)
        mem = model.mem()
        if mem is None:
            opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr)  # FFN: FT TOÀN mạng
            label = f"full-FT lr={args.ft_lr}"
        else:
            new_keys = torch.cat([fact_keys(model, mem, f, args.grow_mode) for f in FACTS])
            for p in model.parameters():
                p.requires_grad_(False)  # ĐÓNG BĂNG toàn mạng
            mem.grow(new_keys)  # MỌC nơ-ron mới
            opt = torch.optim.AdamW([mem.K1, mem.V1], lr=args.slot_lr)
            label = f"mọc {new_keys.shape[0]} slot ({args.grow_mode}), train CHỈ slot mới lr={args.slot_lr}"
        for step in range(args.growth_steps):
            x, y = growth_batches(rng)
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        ho1 = heldout_bits(model)
        fb1 = fact_bits(model)
        print(f"  MỌC ({label}):")
        print(f"    fact-hư-cấu {fb0:.4f} → {fb1:.4f}  (Δ {fb1-fb0:+.4f})")
        print(f"    held-out CŨ {ho0:.4f} → {ho1:.4f}  (Δ {ho1-ho0:+.4f} ← thước QUÊN)")
        for p, exp in PROBES:
            print(f"    probe SAU mọc: {p!r} → {complete(model, p)!r} (kỳ vọng {exp!r})")

        if mem is not None:  # nơ-ron CHỈ ĐƯỢC: slot nào nổ với câu hỏi fact?
            mem._cap = True
            model(torch.tensor(tok.encode(PROBES[0][0]), device=dev)[None])
            mem._cap = False
            top = mem.last_topi[0, -1].tolist()
            n_new = sum(1 for i in top if i >= mem.K.shape[0])
            print(f"    slot nổ (vị trí cuối): {n_new}/{args.topk} là slot MỚI; top-5={top[:5]}")
            if arm == "mem_graph":
                old = [i for i in top if i < mem.K.shape[0]][:2]
                for i in old:
                    print(f"      slot cũ #{i} = đơn vị: {seed_texts[i]!r}")
        results[arm] = dict(ho0=ho0, ho1=ho1, fb0=fb0, fb1=fb1)

    print("\n════ BẢNG KẾT (điều kiện phân định ghi TRƯỚC trong docstring) ════")
    print(f"{'nhánh':>10} {'held-out':>9} {'fact trước':>11} {'fact sau':>9} {'Δfact':>8} {'Δquên':>8}")
    for arm, r in results.items():
        print(
            f"{arm:>10} {r['ho0']:>9.4f} {r['fb0']:>11.4f} {r['fb1']:>9.4f} "
            f"{r['fb1']-r['fb0']:>+8.4f} {r['ho1']-r['ho0']:>+8.4f}"
        )
    print("\n⇒ PASS mọc-không-quên = Δfact ≤ −2.0 VÀ |Δquên| < 0.05 (nhánh mem).")
    print("  FFN full-FT dự kiến Δfact âm nhưng Δquên DƯƠNG (quên) — nền so sánh.")
    print("  mem_rand ≈ mem_graph ở held-out ⇒ gieo mầm đồ thị thừa (bài học G17) — ghi thật.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
