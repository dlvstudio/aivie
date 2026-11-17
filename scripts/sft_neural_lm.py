#!/usr/bin/env python3
"""
SFT LM v2 — dạy khuôn `Hỏi:… Đáp:…<EOS>` trên data chất lượng (TK_042 nhánh SFT, 2026-07-20).

    # SAU KHI G19c xong (out/lm_v2_v10_fp32.pt) + đã chạy scripts/ingest_sft_vi.py:
    PYTHONPATH=src python -u scripts/sft_neural_lm.py --ckpt out/lm_v2_v10_fp32.pt \\
        --save out/lm_v2_sft.pt --log-file out/sft_neural_lm.log

## Cơ chế

* **Loss CHỈ trên phần đáp**: mask mọi vị trí ≤ token `Đáp:` (special nguyên tử — tìm id,
  so INT, không đụng string). Mô hình học TRẢ LỜI + đóng `<EOS>`, không học lặp lại câu hỏi.
* **Replay chống quên** (mặc định mỗi 4 step 1 batch pretrain): G18 đo full-FT không bảo vệ
  quên +13-14 bits — SFT không replay là đặt cược vào đúng chỗ đã đo là thua.
* **AdamW lr nhỏ, KHÔNG Muon**: SFT là tinh chỉnh quanh nghiệm pretrain; update đẳng hướng
  mạnh của Muon là dao mổ trâu ở đây.

## Điều kiện phân định (GHI TRƯỚC — đọc verdict cuối log)

1. **bits-đáp sft-val** (thước chính): SAU phải THẤP hơn TRƯỚC ≥ 0.5 bits ⇒ SFT ăn.
2. **Guard quên** — bits/byte held-out essays (CÙNG load_heldout/seed với thước 5.594):
   Δ ≤ +0.05 sạch · +0.05–0.2 cảnh báo · > +0.2 QUÊN THẢM HOẠ (chữ ký G18) ⇒ tăng replay
   / hạ lr — ghi lại, KHÔNG đắp núm quá một vòng.
3. **Đọc tay bắt buộc**: bits MÙ với "bịa mạch lạc" (bài học G15 §7.17d) — cuối log in
   greedy trên QA val nhà + sft-val; PHẢI đọc, không tin số suông.

⚠️ Chạy trên M4 (máy dev không torch). Log dán nguyên văn — không tóm tắt.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from nn_aivie.core.tokenizer import BPETokenizer, DAP_LABEL, SPECIAL_EOS, normalize

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from eval_bits import load_heldout  # noqa: E402 — CÙNG held-out/công thức với mốc 5.594
from train_neural_lm import load_corpus  # noqa: E402 — replay đọc CÙNG corpus pretrain


# ---------------------------------------------------------------- data
def _iter_qa(args, log):
    """Yield (prompt, answer) từ sft_train + (tuỳ chọn) QA nhà. Prompt CHƯA có 'Đáp:'."""
    p = _ROOT / args.data
    n = 0
    for line in p.open(encoding="utf-8"):
        o = json.loads(line)
        yield o["prompt"], o["answer"]
        n += 1
    log(f"  {p.name}: {n:,} cặp")
    if not args.no_repo_qa:
        n = 0
        for f in sorted(_ROOT.glob("data/grade_0*/qa/train.jsonl")):
            for line in f.open(encoding="utf-8"):
                o = json.loads(line)
                yield f"Hỏi: {o['question']}", o["answer"]
                n += 1
        log(f"  QA nhà (grade_0*/qa/train): {n:,} cặp")


def build_examples(tok, pairs, max_len, dap_id, eos_id, log):
    """encode CHUNG một lần (seam-safe: special là ranh cứng của pre-tokenizer — TK_041),
    tìm vị trí `Đáp:` bằng id. Trả list (ids, dap_pos)."""
    exs, n_long, n_nodap = [], 0, 0
    for prompt, answer in pairs:
        ids = tok.encode(f"{prompt} {DAP_LABEL} {answer}")
        if dap_id not in ids:
            n_nodap += 1
            continue
        ids.append(eos_id)
        if len(ids) > max_len:
            n_long += 1
            continue
        exs.append((ids, ids.index(dap_id)))
    log(f"  giữ {len(exs):,} · bỏ {n_long:,} dài >{max_len} tok · {n_nodap} thiếu Đáp: "
        f"({'⚠ KIỂM DATA' if n_nodap else 'ok'})")
    return exs


def make_batches(exs, max_tokens, rng):
    """Xếp theo độ dài rồi cắt theo NGÂN SÁCH TOKEN (B×Lmax ≤ max_tokens) — giữ đỉnh RAM
    logits ngang pretrain b16 (bài học swap batch-48); xáo THỨ TỰ batch mỗi epoch."""
    order = sorted(range(len(exs)), key=lambda i: len(exs[i][0]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        lm = max(cur_max, len(exs[i][0]))
        if cur and (len(cur) + 1) * lm > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            lm = len(exs[i][0])
        cur.append(i)
        cur_max = lm
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="SFT LM v2 — loss mask sau Đáp:, replay chống quên")
    ap.add_argument("--ckpt", required=True, help="checkpoint v2 pretrain (dict cfg+state_dict)")
    ap.add_argument("--save", default="out/lm_v2_sft.pt")
    ap.add_argument("--data", default="data/sft/sft_train.jsonl")
    ap.add_argument("--val", default="data/sft/sft_val.jsonl")
    ap.add_argument("--bpe", default="checkpoints/bpe_v1.json")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-batch-tokens", type=int, default=8192,
                    help="B×Lmax mỗi batch — 8192 giữ đỉnh RAM ngang pretrain batch16 (M4)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--replay-every", type=int, default=4,
                    help="mỗi N step SFT chèn 1 batch pretrain (0=tắt — sẽ quên, G18 đã đo)")
    ap.add_argument("--manifest", default="config/train_v10.yaml", help="corpus cho replay")
    ap.add_argument("--holdout", default="data/real_vi/essays.jsonl")
    ap.add_argument("--n-holdout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-samples", type=int, default=12)
    # decode polish (§7.17d/G15 — argmax thuần lặp thoái hoá). Mặc định: greedy + anti-loop
    # (no-repeat-ngram 3 + rep-penalty 1.2) — cắt vòng lặp mà vẫn tất định, dễ đọc tay.
    ap.add_argument("--gen-temp", type=float, default=0.0, help="0 = greedy; >0 = lấy mẫu")
    ap.add_argument("--gen-top-p", type=float, default=1.0)
    ap.add_argument("--gen-rep-penalty", type=float, default=1.2)
    ap.add_argument("--gen-no-repeat", type=int, default=3)
    ap.add_argument("--no-repo-qa", action="store_true")
    ap.add_argument("--log-file", default="out/sft_neural_lm.log")
    args = ap.parse_args()

    lf = Path(args.log_file)
    lf.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(m: str) -> None:
        line = f"[{time.time()-t0:7.1f}s] {m}"
        print(line, flush=True)
        with lf.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    import torch
    import torch.nn.functional as F

    from nn_aivie.core.neural_lm import build_model_v2

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    tok = BPETokenizer.load(_ROOT / args.bpe)
    V = len(tok.tokens)
    dap_id, eos_id = tok.token_to_id[DAP_LABEL], tok.token_to_id[SPECIAL_EOS]
    log(f"tokenizer {args.bpe}: {V:,} token · fingerprint {tok.fingerprint()}")

    # ---- model: nạp cfg tường minh (nh KHÔNG suy được từ shape — TK_041 §5.2) ----
    obj = torch.load(_ROOT / args.ckpt, map_location="cpu")
    if not (isinstance(obj, dict) and "cfg" in obj):
        raise SystemExit(f"✗ {args.ckpt} không phải checkpoint v2 (cần dict cfg+state_dict)")
    cfg = obj["cfg"]
    if cfg["V"] != V:
        raise SystemExit(f"✗ vocab ckpt {cfg['V']:,} ≠ tokenizer {V:,} — tokenizer KHÁC")
    model = build_model_v2(**cfg, p=args.dropout)
    model.load_state_dict(obj["state_dict"])
    model.to(dev).train()
    ctx = cfg["ctx"]
    log(f"model {sum(p.numel() for p in model.parameters())/1e6:.1f}M · ctx {ctx} · "
        f"dropout {args.dropout} · device={dev}")
    is_logprob = getattr(model, "returns_logprob", False)

    # ---- data ----
    log("nạp SFT data …")
    exs = build_examples(tok, _iter_qa(args, log), ctx, dap_id, eos_id, log)
    if not exs:
        raise SystemExit("✗ 0 example — chạy scripts/ingest_sft_vi.py trước")
    val_pairs = [(o["prompt"], o["answer"])
                 for o in map(json.loads, (_ROOT / args.val).open(encoding="utf-8"))]
    val_exs = build_examples(tok, val_pairs, ctx, dap_id, eos_id, log)

    # QA val nhà (35 cặp) — thước trong-miền + đọc tay; BÁO nếu lộ đề vào train
    train_qs = {normalize(p) for p, _ in _iter_qa(args, lambda _m: None)}
    grade_val, n_leak = [], 0
    for f in sorted(_ROOT.glob("data/grade_0*/qa/val.jsonl")):
        for line in f.open(encoding="utf-8"):
            o = json.loads(line)
            if normalize(f"Hỏi: {o['question']}") in train_qs:
                n_leak += 1
                continue
            grade_val.append((o["question"], o["answer"]))
    log(f"  QA val nhà: {len(grade_val)} cặp (loại {n_leak} lộ đề vào train)")
    grade_exs = build_examples(
        tok, [(f"Hỏi: {q}", a) for q, a in grade_val], ctx, dap_id, eos_id, log)

    # ---- guard quên: CÙNG held-out/công thức với mốc 5.594 ----
    texts = load_heldout(_ROOT / args.holdout, args.n_holdout, args.seed)
    heldout = [tok.encode(t) for t in texts]
    nbytes = sum(len(normalize(t).encode("utf-8")) for t in texts)
    log(f"guard held-out: {len(texts)} đơn vị · {nbytes:,} byte")

    replay = None
    if args.replay_every > 0:
        log("nạp corpus replay …")
        replay = torch.from_numpy(
            load_corpus(_ROOT / args.manifest, tok, None, log).astype(np.int64))

    # ---- evals (dùng chung; mọi log-prob gom 1 gather — không .item() từng token) ----
    @torch.no_grad()
    def bits_answers(examples):
        """bits/token CHỈ trên phần đáp (sau Đáp:, gồm EOS)."""
        model.eval()
        s, n = 0.0, 0
        for ids, dp in examples:
            x = torch.tensor(ids[:-1], dtype=torch.long, device=dev)[None]
            out = model(x)[0].float()
            lp = out if is_logprob else torch.log_softmax(out, -1)
            rows = torch.arange(dp, len(ids) - 1, device=dev)
            cols = torch.tensor(ids[dp + 1:], dtype=torch.long, device=dev)
            s += (-lp[rows, cols].sum().item()) / math.log(2)
            n += len(ids) - 1 - dp
        model.train()
        return s / max(1, n)

    @torch.no_grad()
    def guard_bpb():
        model.eval()
        s = 0.0
        for ids in heldout:
            if len(ids) < 2:
                continue
            x = torch.tensor(ids[:ctx], dtype=torch.long, device=dev)[None]
            out = model(x)[0].float()
            lp = out if is_logprob else torch.log_softmax(out, -1)
            m = min(len(ids), ctx)
            rows = torch.arange(m - 1, device=dev)
            cols = torch.tensor(ids[1:m], dtype=torch.long, device=dev)
            s += (-lp[rows, cols].sum().item()) / math.log(2)
        model.train()
        return s / nbytes

    from nn_aivie.core.neural_lm import NeuralLM

    gen_lm = NeuralLM(model, ctx, dev)  # bọc để dùng bộ decode chung (repetition penalty…)

    @torch.no_grad()
    def greedy(prompt: str, max_new: int = 60) -> str:
        ids = gen_lm.generate(
            tok.encode(f"{prompt} {DAP_LABEL}"), max_new, eos_id=eos_id,
            temperature=args.gen_temp, top_p=args.gen_top_p,
            repetition_penalty=args.gen_rep_penalty, no_repeat_ngram=args.gen_no_repeat,
            seed=args.seed,
        )
        model.train()  # generate gọi eval() — trả lại train cho vòng SFT
        return tok.decode(ids)

    b_val0, b_grade0, g0 = bits_answers(val_exs), bits_answers(grade_exs), guard_bpb()
    log(f"TRƯỚC SFT: sft-val {b_val0:.3f} bits-đáp · QA-nhà {b_grade0:.3f} · guard {g0:.4f} bpb")

    # ---- train ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    batches = make_batches(exs, args.max_batch_tokens, rng)
    total = args.epochs * len(batches)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    log(f"train: {len(exs):,} ví dụ · {len(batches)} batch/epoch · {args.epochs} epoch · "
        f"lr {args.lr} · replay mỗi {args.replay_every or '∅'} step")

    best = (b_val0, -1)
    gstep = 0
    for ep in range(args.epochs):
        for bi, batch in enumerate(batches):
            lm = max(len(exs[i][0]) for i in batch)
            x = torch.full((len(batch), lm - 1), eos_id, dtype=torch.long)
            y = torch.full((len(batch), lm - 1), eos_id, dtype=torch.long)
            mask = torch.zeros((len(batch), lm - 1), dtype=torch.bool)
            for r, i in enumerate(batch):
                ids, dp = exs[i]
                x[r, :len(ids) - 1] = torch.tensor(ids[:-1])
                y[r, :len(ids) - 1] = torch.tensor(ids[1:])
                mask[r, dp:len(ids) - 1] = True  # CHỈ học đáp (gồm EOS)
            x, y, mask = x.to(dev), y.to(dev), mask.to(dev)
            out = model(x)
            lp = out if is_logprob else torch.log_softmax(out, -1)
            loss = -(lp.gather(2, y[..., None])[..., 0][mask]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            gstep += 1
            if args.replay_every and gstep % args.replay_every == 0:
                ix = torch.randint(0, len(replay) - 257, (8, 1))
                off = torch.arange(256)
                rx, ry = replay[ix + off].to(dev), replay[ix + off + 1].to(dev)
                r_out = model(rx)
                r_lp = r_out if is_logprob else torch.log_softmax(r_out, -1)
                r_loss = -r_lp.gather(2, ry[..., None]).mean()
                opt.zero_grad(set_to_none=True)
                r_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if bi % 100 == 0:
                log(f"  ep{ep} {bi:>5}/{len(batches)} loss {loss.item()/math.log(2):.3f} bits "
                    f"· ETA {(time.time()-t0)/max(1,gstep)*(total-gstep)/60:.0f}m")
        bv, bg, gb = bits_answers(val_exs), bits_answers(grade_exs), guard_bpb()
        mark = ""
        if bv < best[0]:
            best = (bv, ep)
            torch.save({"cfg": cfg, "state_dict": model.state_dict()}, _ROOT / args.save)
            mark = " ★ lưu"
        log(f"EPOCH {ep}: sft-val {bv:.3f} (Δ{bv-b_val0:+.3f}) · QA-nhà {bg:.3f} "
            f"(Δ{bg-b_grade0:+.3f}) · guard {gb:.4f} (Δ{gb-g0:+.4f}){mark}")

    # ---- verdict theo điều kiện GHI TRƯỚC ----
    gb_end = guard_bpb()
    dv, dg = best[0] - b_val0, gb_end - g0
    log("=" * 64)
    log(f"1) sft-val bits-đáp {b_val0:.3f} → {best[0]:.3f} (Δ{dv:+.3f}) — "
        f"{'✅ SFT ăn (≤ −0.5)' if dv <= -0.5 else '❌ chưa đạt ngưỡng −0.5'}")
    verdict = ("✅ sạch (≤ +0.05)" if dg <= 0.05 else
               "⚠ cảnh báo +0.05–0.2" if dg <= 0.2 else
               "🔴 QUÊN THẢM HOẠ (>+0.2, chữ ký G18) — tăng replay / hạ lr")
    log(f"2) guard essays {g0:.4f} → {gb_end:.4f} bpb (Δ{dg:+.4f}) — {verdict}")
    log("3) ĐỌC TAY (bits mù với bịa — G15):")
    for q, gold in grade_val[: args.gen_samples]:
        log(f"   Hỏi: {q}\n     GOLD: {gold}\n     SINH: {greedy(f'Hỏi: {q}')}")
    for p, gold in val_pairs[: max(0, args.gen_samples - len(grade_val))]:
        log(f"   [sft-val] {p[-120:]}\n     GOLD: {gold[:120]}\n     SINH: {greedy(p)[-200:]}")
    log(f"→ checkpoint tốt nhất (epoch {best[1]}): {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
