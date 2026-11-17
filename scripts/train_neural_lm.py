#!/usr/bin/env python3
"""
TK_042 G7 ("Bước 1") — ĐO CÁI GAP CHO RA SỐ THẬT: LM nơ-ron vs ĐẾM, cùng mọi thứ.

    pip install torch
    PYTHONPATH=src python -u scripts/train_neural_lm.py --log-file out/g7_neural_lm.log

⚠️ **CẦN MÁY ≥ 16GB** (máy dev hiện tại 8GB — xem TK_042 §7.7). Apple Silicon → MPS.

## Câu hỏi G7 trả lời (nhị phân)

> **Corpus 19.6M token của TA còn bao nhiêu "room" mà mô hình ĐẾM không lấy được?**

Đã đo (TK_042 §7.5b): mọi mô hình nhìn **1 token ngữ cảnh** chặn ở **~6.4 bits/token**, kể cả
điền số 0 HOÀN HẢO. Suy ra R1 (sinh thật) cần **ngữ cảnh dài** ⇒ cần **nén ngữ cảnh** ⇒ cần
**học một mô hình thật**. G7 đo xem học thật lấy được bao nhiêu **trên chính data này**.

* LM nơ-ron **thắng đậm** ⇒ gap là THẬT, do thiếu điều-kiện-hoá-ngữ-cảnh. Câu hỏi kiến trúc
  (B: luật cục bộ vs C: backprop) trở thành có nghĩa.
* LM nơ-ron **KHÔNG hơn đáng kể** ⇒ 19.6M token quá nhỏ, **A là câu trả lời đúng** — đóng
  TK_042, sửa định vị README, không tiếc.

## Vì sao MỘT script làm CẢ HAI mô hình

Bẫy lớn nhất của G7 là **so sánh khập khiễng**. Số đếm ta có (`1.288 bits/byte`) đến từ
`network_v10_histwiki` — **corpus train KHÁC**, **tokenizer KHÁC** (legacy 113k). So nó với một
LM train trên 99MB + BPE 32k là so hai thứ khác nhau ở BA biến cùng lúc.

→ Script này dựng **CẢ HAI trên cùng một dòng dữ liệu**: cùng manifest, cùng `bpe_v1.json`,
cùng held-out (`load_heldout` của `eval_bits.py`, cùng seed), cùng công thức
`bits = -log2 P(gold)` trên **MỌI vị trí**. Khác biệt duy nhất còn lại = **mô hình**.
Đó là điều kiện để con số nói được điều gì.

Baseline đếm ở đây **không cần** load checkpoint 20GB — `W_seq` chỉ là bigram count, dựng
thẳng từ corpus đã tokenize.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

from nn_aivie.core.tokenizer import BPETokenizer, normalize

_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(_ROOT / "scripts"))
from eval_bits import load_heldout  # noqa: E402  — CÙNG held-out với thước G0/G1


# ---------------------------------------------------------------- data
def load_corpus(manifest: Path, tok: BPETokenizer, max_texts: int | None, log) -> np.ndarray:
    spec = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    ids: list[int] = []
    for d in spec["datasets"]:
        p = _ROOT / d["path"]
        if not p.exists():
            log(f"  ⚠ bỏ qua (không có): {p.name}")
            continue
        # `repeat` = SAMPLING WEIGHT (vắng = 1, giữ mọi manifest cũ byte-identical). Vì vòng
        # train lấy cửa sổ NGẪU NHIÊN ĐỀU trên mảng nối, lặp file N lần ⇒ nó được lấy mẫu N× —
        # đúng "sampling weights" §7.20b. Cần cho v12: CulturaX ~500M sẽ NHẤN CHÌM QA 34k token
        # (0.007%) nếu không nâng trọng số nguồn QA/grade.
        rep = int(d.get("repeat", 1))
        file_ids: list[int] = []
        with p.open(encoding="utf-8") as f:
            for k, line in enumerate(f):
                if max_texts and k >= max_texts:
                    break
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = o if isinstance(o, str) else (o.get("text") or "")
                if not t:
                    continue
                file_ids.extend(tok.encode(t))
                file_ids.append(tok.token_to_id["<EOS>"])
        n0 = len(ids)
        for _ in range(rep):
            ids.extend(file_ids)
        tag = f" ×{rep}" if rep != 1 else ""
        log(f"  {p.name}{tag}: +{len(ids)-n0:,} token (tổng {len(ids):,})")
    return np.array(ids, dtype=np.int32)


# ---------------------------------------------------------------- baseline: ĐẾM
def bits_counts(train_ids: np.ndarray, heldout: list[list[int]], nbytes: int, V: int, log) -> float:
    """Bigram + unigram nội suy — CÙNG mô hình mà `W_seq` của đồ thị là (chỉ là count)."""
    from collections import defaultdict

    uni = np.bincount(train_ids, minlength=V).astype(np.float64)
    bi: dict[int, dict[int, int]] = defaultdict(dict)
    for a, b in zip(train_ids[:-1], train_ids[1:]):
        d = bi[int(a)]
        d[int(b)] = d.get(int(b), 0) + 1
    rowsum = {i: sum(d.values()) for i, d in bi.items()}
    p_uni = (uni + 1.0) / (uni.sum() + V)

    best = None
    for lam in (0.5, 0.7, 0.85, 0.95):
        s = 0.0
        n = 0
        for ids in heldout:
            for i in range(1, len(ids)):
                w0, g = int(ids[i - 1]), int(ids[i])
                d = bi.get(w0)
                pb = (d.get(g, 0) / rowsum[w0]) if d and rowsum.get(w0) else 0.0
                p = lam * pb + (1 - lam) * float(p_uni[g])
                s += -math.log2(max(p, 1e-12))
                n += 1
        bpb = s / nbytes
        log(f"    ĐẾM lam={lam:.2f}: {s/n:7.3f} bits/token · {bpb:.3f} bits/BYTE")
        if best is None or bpb < best:
            best = bpb
    return best


# ---------------------------------------------------------------- Muon (TK_042 §7.20)
def _make_muon(torch):
    """Muon (Keller Jordan 2024): trực giao hoá MA TRẬN CẬP NHẬT bằng Newton-Schulz — mọi giá
    trị kỳ dị của update về ~bằng nhau ⇒ update đẳng hướng ⇒ trọng số không sụp hạng, các head
    không thoái hoá về giống nhau (câu hỏi tác giả 2026-07-19 — đây là cơ chế GIỮ trực giao
    trong train, bổ sung cho orthogonal init tại t=0). Chứng minh ở speedrun nanoGPT (đúng cỡ
    mô hình này) và Kimi K2 (1T). CHỈ cho ma trận ẩn 2D — embedding/norm/bảng memory đi AdamW."""

    def ns5(G, steps=5, eps=1e-7):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G / (G.norm() + eps)
        tp = X.size(0) > X.size(1)
        if tp:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            X = a * X + (b * A + c * (A @ A)) @ X
        return X.T if tp else X

    class Muon(torch.optim.Optimizer):
        def __init__(s, params, lr=0.02, momentum=0.95):
            super().__init__(params, dict(lr=lr, momentum=momentum))

        @torch.no_grad()
        def step(s):
            for g in s.param_groups:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = s.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p.grad)
                    buf = st["buf"]
                    buf.mul_(g["momentum"]).add_(p.grad)
                    u = ns5(p.grad.add(buf, alpha=g["momentum"]))  # nesterov
                    p.add_(u, alpha=-g["lr"] * max(1.0, p.size(-2) / p.size(-1)) ** 0.5)

    return Muon


def _log_head_similarity(model, nh, log, torch):
    """ĐO-không-ép: |cos| trung bình giữa các lát Wq của các head (proxy phẳng hoá — không
    phải góc chính không gian con, đủ để phát hiện sụp). Gần 0 = head khác nhau; gần 1 = sụp."""
    for li, blk in enumerate(model.blocks):
        W = blk.attn.qkv.weight  # (3d, d)
        d = W.size(1)
        dk = d // nh
        heads = W[:d].view(nh, dk, d)  # phần Wq
        sims = []
        for i in range(nh):
            for j in range(i + 1, nh):
                a, b = heads[i].flatten(), heads[j].flatten()
                sims.append((a @ b / (a.norm() * b.norm())).abs().item())
        log(f"    head-sim(Wq) L{li}: mean|cos|={sum(sims)/len(sims):.4f}")


# ---------------------------------------------------------------- neural
def bits_neural(train_ids, heldout, nbytes, V, args, log) -> float:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        log("✗ CẦN torch: `pip install torch`  (bỏ qua phần nơ-ron)")
        return float("nan"), float("nan")

    dev = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # MPS (Apple Silicon) không hỗ trợ dropout trong scaled_dot_product_attention →
    # dùng attn_dropout=0.0 trên MPS; dropout FF vẫn giữ.
    attn_dropout = 0.0 if dev == "mps" else args.dropout
    log(f"  device={dev}  attn_dropout={attn_dropout} (MPS: attention dropout buộc=0)")
    torch.manual_seed(args.seed)

    class Block(nn.Module):
        def __init__(s, d, h, p, ad):
            super().__init__()
            s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            s.attn = nn.MultiheadAttention(d, h, dropout=ad, batch_first=True)
            s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))

        def forward(s, x, mask):
            h = s.ln1(x)
            a, _ = s.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            return x + s.ff(s.ln2(x))

    class LM(nn.Module):
        def __init__(s, V, d, nl, nh, ctx, p, ad):
            super().__init__()
            s.tok = nn.Embedding(V, d)
            s.pos = nn.Embedding(ctx, d)
            s.drop = nn.Dropout(p)
            s.blocks = nn.ModuleList([Block(d, nh, p, ad) for _ in range(nl)])
            s.ln = nn.LayerNorm(d)
            s.head = nn.Linear(d, V, bias=False)
            s.head.weight = s.tok.weight  # tied — vocab 32k chiếm phần lớn tham số
            s.ctx = ctx

        def forward(s, x):
            T = x.shape[1]
            h = s.drop(s.tok(x) + s.pos(torch.arange(T, device=x.device)))
            m = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            for b in s.blocks:
                h = b(h, m)
            return s.head(s.ln(h))

    v2_cfg = None
    if args.arch == "v2":
        # §7.20 — kiến trúc từ neural_lm.py (MỘT nguồn duy nhất), init (orthogonal + emb 0.02)
        # nằm TRONG build_model_v2 — KHÔNG apply(_init) đè lên (sẽ xoá mất orthogonal init!)
        from nn_aivie.core.neural_lm import build_model_v2

        mem_layer = args.layers - 3 if args.mem_slots > 0 else -1  # gần cuối, không phải cuối
        v2_cfg = dict(
            V=V, d=args.d_model, nl=args.layers, nh=args.heads, ctx=args.ctx,
            mem_slots=args.mem_slots, mem_dk=64, mem_topk=32, mem_layer=mem_layer,
            mos_k=args.mos_k,
        )
        if args.mos_k:
            log(f"  v2 + MoS head: K={args.mos_k} softmax trộn (§7.21 phá softmax-bottleneck)")
        model = build_model_v2(**v2_cfg, p=args.dropout)
        if args.mem_slots:
            log(f"  v2 + MemLayer from-scratch: {args.mem_slots} slot @ block {mem_layer} "
                f"(G18 §7.19d: train CÙNG backbone, không ghép sau)")
    else:
        model = LM(V, args.d_model, args.layers, args.heads, args.ctx, args.dropout, attn_dropout)

        def _init(m):
            """Init kiểu GPT-2: N(0, 0.02).

            BẮT BUỘC khi TIED WEIGHTS — bug đã bắt bằng smoke test 2026-07-16: init mặc định
            của `nn.Embedding` là N(0,1); head dùng CHUNG weight đó ⇒ logits ~ N(0, d_model)
            ⇒ std ~8 ⇒ logits cực đoan ⇒ **loss khởi đầu 42 nats = 60 bits/token**, trong khi
            trần lý thuyết lúc init chỉ là ln(32002) = 10.37 nats = 15 bits (phân bố đều).
            """
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        model.apply(_init)
    model = model.to(dev)

    dev_type = "mps" if dev == "mps" else ("cuda" if dev == "cuda" else "cpu")
    import contextlib as _ctxlib

    _bf16 = args.bf16 and dev_type != "cpu"  # cpu: bf16 không nhanh
    _fp16 = args.fp16 and dev_type != "cpu"

    def autocast():
        if _bf16:
            return torch.autocast(device_type=dev_type, dtype=torch.bfloat16)
        if _fp16:
            return torch.autocast(device_type=dev_type, dtype=torch.float16)
        return _ctxlib.nullcontext()

    autocast_eval = autocast  # eval dùng chung factory

    def save_ckpt(path):
        if v2_cfg is not None:
            torch.save({"cfg": v2_cfg, "state_dict": model.state_dict()}, path)
        else:
            torch.save(model.state_dict(), path)
    with torch.no_grad():
        x0 = torch.zeros(1, 8, dtype=torch.long, device=dev)
        l0 = F.cross_entropy(model(x0).view(-1, V), x0.reshape(-1)).item()
    ceil0 = math.log(V)
    log(f"  sanity init: loss={l0:.2f} nats (trần đều = ln({V:,}) = {ceil0:.2f})")
    if l0 > ceil0 * 1.25:
        log(f"  ✗ INIT HỎNG: loss init > trần đều ⇒ logits cực đoan (nghi tied-weight init).")
        return float("nan"), float("nan")
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  model: {n_par/1e6:.1f}M tham số (d={args.d_model} L={args.layers} ctx={args.ctx})")
    log(f"  ⚠ Chinchilla-optimal cho {len(train_ids)/1e6:.1f}M token ≈ ~1M tham số → mô hình này")
    log(f"    THỪA tham số (embedding 32k×{args.d_model} chiếm phần lớn) → dropout + early stop")

    data = torch.from_numpy(train_ids.astype(np.int64))
    n_steps = args.steps
    opts = []
    if args.opt == "muon" and args.arch == "v2":
        # lai chuẩn: Muon cho ma trận ẩn 2D (attn qkv/wo + SwiGLU w1/w2/w3 + mem Wq);
        # AdamW cho embedding(tied head) + norm + bảng memory K/V (embedding-like).
        muon_p, adamw_p = [], []
        for name, prm in model.named_parameters():
            is_hidden_2d = prm.ndim == 2 and (
                ".attn." in name or ".ffn.w" in name or ".ffn.Wq." in name
            )  # KHÔNG gồm: tok.weight (tied head), RMSNorm .w (1D), mem K/V (bảng ~embedding)
            (muon_p if is_hidden_2d else adamw_p).append(prm)
        Muon = _make_muon(torch)
        opt_m = Muon(muon_p, lr=args.muon_lr, momentum=0.95)
        opt_a = torch.optim.AdamW(adamw_p, lr=args.lr, weight_decay=0.1)
        opts = [opt_m, opt_a]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt_a, args.lr, total_steps=n_steps, pct_start=0.05
        )
        # ⚠️ FIX G19 (§7.20a): bản đầu để Muon LR PHẲNG — comment cũ "chuẩn speedrun" SAI,
        # speedrun nanoGPT ANNEAL Muon LR về 0. LR phẳng ⇒ ma trận ẩn nhận update đẳng hướng
        # lớn tới cuối ⇒ không lắng vào đáy ⇒ eval nảy + xấu đi sau step 5000 + sinh lặp thoái
        # hoá. Nay anneal tay: warmup 5% linear rồi cosine → ~0. (Muon là custom optimizer nên
        # scale lr trong vòng lặp, không qua torch scheduler.)
        muon_peak = args.muon_lr
        warm = max(1, int(0.05 * n_steps))

        def muon_lr_at(step):
            if step < warm:
                return muon_peak * (step + 1) / warm
            prog = (step - warm) / max(1, n_steps - warm)
            return muon_peak * 0.5 * (1 + math.cos(math.pi * prog))

        log(
            f"  opt=MUON lai (LR ANNEAL cosine, fix §7.20a): "
            f"{sum(p.numel() for p in muon_p)/1e6:.1f}M Muon (peak lr={args.muon_lr}) + "
            f"{sum(p.numel() for p in adamw_p)/1e6:.1f}M AdamW (lr={args.lr})"
        )
    else:
        if args.opt == "muon":
            log("  ⚠ --opt muon chỉ hỗ trợ --arch v2 — dùng AdamW")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
        opts = [opt]
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=n_steps, pct_start=0.05)

    # ---- RESUME: nối tiếp run đang dở (§TK_043 câu 3a) — KHÔNG train lại từ đầu ----
    # Lưu ĐỦ để tiếp: model + MỌI optimizer (momentum) + scheduler + step + best. Thiếu
    # optimizer/step ⇒ "resume" giả (momentum về 0, LR schedule về đầu) = train kém, im lặng.
    start_step = 0
    best_resume = None

    def save_resume(path, step, best):
        torch.save(
            {
                "cfg": v2_cfg,
                "model": model.state_dict(),
                "opts": [o.state_dict() for o in opts],
                "sched": sched.state_dict(),
                "step": step,
                "best": best,
            },
            path,
        )

    if args.resume:
        rp = Path(args.resume)
        if not rp.exists():
            raise SystemExit(f"--resume {rp} không tồn tại")
        st = torch.load(rp, map_location=dev)
        model.load_state_dict(st["model"])
        for o, osd in zip(opts, st["opts"]):
            o.load_state_dict(osd)
        sched.load_state_dict(st["sched"])
        start_step = st["step"] + 1
        best_resume = tuple(st["best"])
        log(f"  ▶ RESUME từ {rp.name}: tiếp step {start_step}/{n_steps} · best cũ {best_resume[0]:.3f} bits/tok")

    is_logprob = getattr(model, "returns_logprob", False)  # MoS trả LOG-P sẵn (§7.21)

    def _eval_heldout():
        """bits trên held-out — CÙNG vị trí, CÙNG công thức với `bits_counts`/`eval_bits.py`."""
        model.eval()
        s = 0.0
        n = 0
        with torch.no_grad():
            for ids in heldout:
                if len(ids) < 2:
                    continue
                x = torch.tensor(ids[: args.ctx], dtype=torch.long, device=dev)[None]
                with autocast_eval():
                    out = model(x)[0].float()  # về fp32 cho log_softmax ổn định
                lp = out if is_logprob else torch.log_softmax(out, -1)  # KHÔNG double-softmax
                m = min(len(ids), args.ctx)
                if m > 1:  # gom log-prob 1 phép gather thay .item() MỖI token (sync device→host)
                    rows = torch.arange(m - 1, device=dev)      # 0..m-2
                    cols = torch.tensor(ids[1:m], dtype=torch.long, device=dev)  # ids[1..m-1]
                    s += (-lp[rows, cols].sum().item()) / math.log(2)
                    n += m - 1
        model.train()
        return s / max(1, n), s / nbytes, n

    # EVAL GIỮA CHỪNG + giữ BẢN TỐT NHẤT.
    #
    # ⚠️ Bản đầu chỉ eval Ở CUỐI. Với 12 epoch trên 19.7M token / 11.4M tham số, mô hình có
    # thể ĐI QUA điểm tối ưu rồi overfit — khi đó "eval ở step cuối" là số SAI, và cả lần
    # chạy 2.3 giờ thành vô ích vì không giữ lại bản tốt nhất. Số cần đọc luôn là **eval TỐT
    # NHẤT**, không phải eval cuối.
    best = best_resume or (float("inf"), float("inf"), -1)  # (bits/token, bits/byte, step)
    model.train()
    t0 = time.time()
    use_muon = args.opt == "muon" and args.arch == "v2"
    if args.bf16:
        log(f"  bf16 autocast: {'BẬT' if _bf16 else 'BỎ (cpu)'} · batch={args.batch}")
    if args.fp16:
        log(f"  fp16 autocast: {'BẬT' if _fp16 else 'BỎ (cpu)'} · batch={args.batch}")
    for step in range(start_step, n_steps):
        if use_muon:  # anneal Muon LR (fix §7.20a)
            opt_m.param_groups[0]["lr"] = muon_lr_at(step)
        # gather vectorized: bỏ vòng Python + stack mỗi step, copy host→device 1 lần.
        # ix (B,1) + off (ctx,) broadcast → index (B,ctx); cùng chỉ số như bản stack cũ.
        ix = torch.randint(0, len(data) - args.ctx - 1, (args.batch, 1))
        off = torch.arange(args.ctx)
        x = data[ix + off].to(dev)
        y = data[ix + off + 1].to(dev)
        with autocast():
            out = model(x).view(-1, V)
            loss = F.nll_loss(out, y.reshape(-1)) if is_logprob else F.cross_entropy(out, y.reshape(-1))
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        sched.step()
        if step % args.log_every == 0 or step == n_steps - 1:
            el = time.time() - t0
            log(
                f"    step {step:>6}/{n_steps}  loss {loss.item():.3f} "
                f"({loss.item()/math.log(2):.3f} bits/token)  {el:.0f}s "
                f"· ETA {el/(step+1)*(n_steps-step-1)/60:.0f}m"
            )
        if args.eval_every and (step + 1) % args.eval_every == 0:
            ebt, ebb, _ = _eval_heldout()
            mark = ""
            if ebb < best[1]:
                best = (ebt, ebb, step)
                mark = " ★ tốt nhất"
                if args.save:
                    save_ckpt(args.save)
            if args.save:  # resume-state (model+opt+step) MỖI lần eval → ngắt lúc nào tiếp lúc đó
                save_resume(args.save + ".resume", step, best)
            log(f"      ↳ EVAL @{step+1}: {ebt:.3f} bits/token · {ebb:.3f} bits/BYTE{mark}")

    eval_bt, eval_bb, n = _eval_heldout()
    log(f"    nơ-ron (CUỐI): {eval_bt:.3f} bits/token · {eval_bb:.3f} bits/BYTE ({n:,} vị trí)")
    if best[2] >= 0 and best[1] < eval_bb - 1e-9:
        log(
            f"    ⚠ eval TỐT NHẤT ở step {best[2]+1}: {best[0]:.3f} bits/token · "
            f"{best[1]:.3f} bits/BYTE — CUỐI đã XẤU ĐI ⇒ ĐÃ OVERFIT. Dùng số TỐT NHẤT."
        )
        eval_bt, eval_bb = best[0], best[1]
    s = eval_bb * nbytes  # giữ tương thích phần dưới

    # ---- CHẨN ĐOÁN FIT — thay cho guard `steps < 500` (VÔ DỤNG, xem dưới) ----
    #
    # ⚠️ BÀI HỌC 2026-07-17: lần chạy G7 đầu (3000 step) cho Δ=−0.027 và kết luận "A đúng,
    # đóng ticket". SAI. Guard cũ `undertrained = steps < 500` báo XANH vì 3000 > 500 — một
    # ngưỡng bịa ra cho smoke test, không phải cho hội tụ. Đúng lớp lỗi im lặng của cả TK_041:
    # một phép kiểm tra TRÔNG NHƯ đang canh gác, thực ra không nổ khi cần.
    #
    # Chẩn đoán ĐÚNG có sẵn trong tay mà không dùng: **so loss TRAIN với loss EVAL**.
    #   train ≈ eval  → UNDERFIT: mô hình chưa fit nổi data của CHÍNH NÓ → train thêm.
    #                   KHÔNG được kết luận gì về "data đủ hay không".
    #   train ≪ eval  → đã vắt hết data → LÚC ĐÓ mới kết luận được.
    # (Log lần đầu: train 6.875 vs eval 6.938 bits/token — KHÔNG có khoảng cách nào.)
    tr_s = 0.0
    tr_n = 0
    with torch.no_grad():
        for _ in range(16):
            ix = torch.randint(0, len(data) - args.ctx - 1, (args.batch, 1))
            off = torch.arange(args.ctx)
            x = data[ix + off].to(dev)
            y = data[ix + off + 1].to(dev)
            out = model(x)
            lp = out if is_logprob else torch.log_softmax(out, -1)
            tr_s += -lp.gather(2, y[..., None]).sum().item() / math.log(2)
            tr_n += y.numel()
    train_bt = tr_s / tr_n
    gap = eval_bt - train_bt
    log(f"    TRAIN {train_bt:.3f} vs EVAL {eval_bt:.3f} bits/token · khoảng cách {gap:+.3f}")
    if gap < 0.5:
        log("    ✗ UNDERFIT: mô hình CHƯA FIT NỔI data train của chính nó (train≈eval).")
        log("      ⇒ KHÔNG kết luận 'data quá nhỏ'. Train thêm (--steps lớn hơn) rồi đo lại.")
    else:
        log(f"    ✓ đã có khoảng cách train/eval ⇒ mô hình đã vắt được data; kết luận có nghĩa.")

    if args.arch == "v2":
        _log_head_similarity(model, args.heads, log, torch)  # đo-không-ép (§7.20)
    if args.save:
        if best[2] < 0:  # chưa từng eval giữa chừng — lưu bản cuối
            save_ckpt(args.save)
        log(f"    → {args.save} (bản TỐT NHẤT theo eval giữa chừng)")
    return eval_bb, gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/train_v10.yaml")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--holdout", default="data/real_vi/essays.jsonl")
    ap.add_argument("--n-holdout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-texts", type=int, default=None, help="cắt mỗi file (dev)")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--steps",
        type=int,
        default=30000,
        help="⚠️ 3000 (mặc định CŨ) = UNDERFIT — đã chứng minh 2026-07-17: train 6.875 ≈ "
             "eval 6.938 bits/token. Đọc `khoảng cách train/eval` cuối log, KHÔNG đọc Δ "
             "cho tới khi gap ≥ 0.5.",
    )
    ap.add_argument("--eval-every", type=int, default=2000, help="0 = tắt; giữ bản TỐT NHẤT")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--arch", choices=["v1", "v2"], default="v1",
                    help="v2 = RoPE+RMSNorm+SwiGLU+orthogonal-init (§7.20); v1 = giữ tái lập G7")
    ap.add_argument("--opt", choices=["adamw", "muon"], default="adamw",
                    help="muon (chỉ v2): lai — Muon cho ma trận ẩn 2D, AdamW cho emb/norm/mem")
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--mem-slots", type=int, default=0,
                    help="v2: >0 = thay FFN block (layers-3) bằng MemLayer train CÙNG backbone"
                         " từ đầu (arm G18-from-scratch, §7.19d/§7.20)")
    ap.add_argument("--mos-k", type=int, default=0,
                    help="v2: >0 = MoS head K softmax (§7.21 G20 — phá softmax-bottleneck; CHỈ"
                         " chạy khi data đã gỡ + chẩn đoán chỉ vào output, không depth)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--log-file", default="out/g7_neural_lm.log")
    ap.add_argument("--save", default=None)
    ap.add_argument("--bf16", action="store_true",
                    help="autocast bfloat16 (~1.5-2× nhanh, ½ RAM; MPS/GPU). Canh loss không nổ.")
    ap.add_argument("--fp16", action="store_true",
                    help="autocast float16 (nhanh hơn fp32, ít ổn định hơn bf16 trên MPS/GPU).")
    ap.add_argument("--resume", default=None,
                    help="nối tiếp từ <save>.resume (model+opt+step) — KHÔNG train lại từ đầu")
    ap.add_argument("--skip-neural", action="store_true")
    args = ap.parse_args()
    if args.bf16 and args.fp16:
        raise SystemExit("Chỉ chọn 1 trong 2: --bf16 hoặc --fp16")

    lf = Path(args.log_file)
    lf.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(m: str) -> None:
        line = f"[{time.time()-t0:7.1f}s] {m}"
        print(line, flush=True)
        with lf.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"=== TK_042 G7 — LM nơ-ron vs ĐẾM (cùng corpus/tokenizer/held-out/công thức) ===")
    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok)
    log(f"tokenizer {args.bpe}: {V:,} token · fingerprint {tok.fingerprint()}")

    log("nạp corpus …")
    train_ids = load_corpus(_ROOT / args.manifest, tok, args.max_texts, log)
    log(f"train: {len(train_ids):,} token")

    texts = load_heldout(_ROOT / args.holdout, args.n_holdout, args.seed)
    heldout = [tok.encode(t) for t in texts]
    nbytes = sum(len(normalize(t).encode("utf-8")) for t in texts)
    log(f"held-out: {len(texts)} đơn vị · {nbytes:,} byte · {sum(len(i) for i in heldout):,} token")

    log("\n--- BASELINE: ĐẾM (bigram+unigram) ---")
    b_cnt = bits_counts(train_ids, heldout, nbytes, V, log)
    log(f"  → ĐẾM tốt nhất: {b_cnt:.3f} bits/BYTE")

    b_nn, gap = float("nan"), float("nan")
    if not args.skip_neural:
        log("\n--- LM NƠ-RON ---")
        b_nn, gap = bits_neural(train_ids, heldout, nbytes, V, args, log)

    log("\n" + "=" * 64)
    log(f"  ĐẾM     : {b_cnt:.3f} bits/byte")
    if not math.isnan(b_nn):
        log(f"  NƠ-RON  : {b_nn:.3f} bits/byte   (Δ {b_nn-b_cnt:+.3f})")
        log("=" * 64)
        # ⚠️ KHÔNG kết luận từ lần chạy chưa train đủ.
        # Ngưỡng CŨ `steps < 500` là số BỊA cho smoke test → báo XANH ở 3000 step (lần chạy
        # 2026-07-17) → kết luận SAI "A đúng, đóng ticket", trong khi train 6.875 ≈ eval 6.938
        # bits/token = UNDERFIT rành rành. Nay dùng CHẨN ĐOÁN FIT (khoảng cách train/eval).
        undertrained = (
            args.steps < 500
            or len(train_ids) < 1_000_000
            or (not math.isnan(gap) and gap < 0.5)  # ← chốt THẬT
        )
        if undertrained:
            log(
                f"\n⚠ CHƯA TRAIN ĐỦ (steps={args.steps}, {len(train_ids)/1e6:.2f}M token) — "
                "KHÔNG kết luận gì từ số này. Đây là smoke test, không phải phép đo."
            )
        elif b_nn < b_cnt - 0.1:
            log("\n→ GAP THẬT. Đếm KHÔNG lấy được ngữ cảnh dài. Câu hỏi B vs C có nghĩa.")
        else:
            log("\n→ KHÔNG hơn đáng kể ⇒ 19.6M token quá nhỏ ⇒ **A là câu trả lời đúng**.")
            log("   Đóng TK_042, sửa định vị README. Ghi âm tính (§7.4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
