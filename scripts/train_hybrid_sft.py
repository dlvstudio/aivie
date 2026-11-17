"""§TK_100 · M1 — SFT trên HybridLM (nạp checkpoint pretrain → dạy KHUÔN chỉ-dẫn).

Nối bài-học [[mamba-ssm-tren-m4]]: pretrain dạy KIẾN-THỨC, SFT dạy ĐỊNH-DẠNG/hành-vi.
Nạp checkpoint từ `train_hybrid_scale.py` → finetune trên `sft_mix.jsonl` (từ
`build_sft_mix.py`). **Mask loss:** chỉ tính lỗi trên token của ASSISTANT (bỏ qua
system/user + phần đệm) → model học TRẢ LỜI, không học nhại câu hỏi.

Khuôn hội-thoại (dùng special-token có sẵn của bpe_v1): user→"Hỏi:", assistant→"Đáp:",
kết bằng <EOS>. Prompt lúc suy-luận: "…Hỏi: {q}\nĐáp:" rồi để model sinh.

    PYTHONPATH=src python scripts/train_hybrid_sft.py --init-from checkpoints/hybrid_1p3b.pt \
        --data data/sft_vi/sft_mix.jsonl --bs 8 --grad-accum 4 --lr 1e-5 --epochs 2 \
        --grad-ckpt --out checkpoints/hybrid_1p3b_sft.pt
"""
import sys, os, json, time, math, argparse, random
sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from train_hybrid_scale import build_from_cfg
from hf_backup_ckpt import maybe_hf_backup, start_hf_backup, wait_hf_backups, warn_if_no_hf_repo
from nn_aivie.core.tokenizer import BPETokenizer
T0 = time.time()


def build_example(msgs, tok, max_len):
    """Trả (ids, labels) — labels=-100 ở token KHÔNG train (system/user/khuôn), =id ở assistant."""
    EOS = tok.token_to_id["<EOS>"]
    ids, lab = [], []

    def add(text, train):
        e = tok.encode(text)
        ids.extend(e); lab.extend(e if train else [-100] * len(e))

    for m in msgs:
        r, c = m["role"], m["content"].strip()
        if not c:
            continue
        if r == "system":
            add(c + "\n", False)
        elif r == "user":
            add("Hỏi: " + c + "\n", False)
        elif r == "assistant":
            learn = not m.get("nolearn")                   # lượt chèn sẵn (vd bản sửa SAI để dạy phục hồi) → không học
            add("Đáp: ", False)                            # khuôn "Đáp:" = prompt, KHÔNG train
            add(c, learn)                                  # nội-dung trả-lời = TRAIN
            ids.append(EOS); lab.append(EOS if learn else -100)   # học dừng đúng chỗ
    if len(ids) > max_len:                                 # cắt phải (giữ đầu hội-thoại)
        ids, lab = ids[:max_len], lab[:max_len]
    return ids, lab


class SFTData:
    def __init__(self, path, tok, max_len, eval_frac=0.02):
        raw = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.ex = []
        skip = 0
        for r in raw:
            ids, lab = build_example(r["messages"], tok, max_len)
            if any(x != -100 for x in lab):                # phải có ít nhất 1 token assistant
                self.ex.append((ids, lab))
            else:
                skip += 1
        random.Random(0).shuffle(self.ex)
        n_val = max(1, int(len(self.ex) * eval_frac))
        self.val, self.train = self.ex[:n_val], self.ex[n_val:]
        print(f"[{time.time()-T0:.0f}s] SFT: {len(self.train)} train · {len(self.val)} val "
              f"(bỏ {skip} rỗng-assistant)", flush=True)

    def batch(self, bs, split, dev, gen, pad=0, tok_budget=0):
        pool = self.train if split == "train" else self.val
        idx = gen.integers(0, len(pool), bs)
        rows = [pool[i] for i in idx]
        if tok_budget:                                     # mẫu dài 8K lẫn mẫu ngắn: giữ lô ≤ ngân sách token (sau đệm)
            rows.sort(key=lambda r: -len(r[0]))            # → giữ mẫu dài nhất, bớt mẫu ngắn
            rows = rows[:max(1, tok_budget // len(rows[0][0]))]
            bs = len(rows)
        ml = max(len(r[0]) for r in rows)
        X = np.full((bs, ml), pad, np.int64); Y = np.full((bs, ml), -100, np.int64)
        for j, (ids, lab) in enumerate(rows):
            X[j, :len(ids)] = ids; Y[j, :len(lab)] = lab
        return torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True, help="checkpoint pretrain (train_hybrid_scale)")
    ap.add_argument("--data", default="data/sft_vi/sft_mix.jsonl")
    ap.add_argument("--tokenizer", default="checkpoints/bpe_v1.json")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--device", default="", help="cuda | mps | cpu (mặc định: tự chọn)")
    ap.add_argument("--resume", action="store_true",
                    help="--out đã có → chạy TIẾP từ bước đã lưu (box chết giữa chừng): optimizer mới, ấm lại --warmup bước")
    ap.add_argument("--tok-budget", type=int, default=0,
                    help="tối đa bs×độ dài (sau đệm) mỗi lô vi mô; 0 = không giới hạn. Cần khi trộn mẫu dài 8K")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"],
                    help="adamw8bit (bitsandbytes) bớt ~3/4 bộ nhớ trạng thái optimizer — thân 2,5B trên 1×H100")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out", default="checkpoints/hybrid_sft.pt")
    ap.add_argument("--hf-repo", default="",
                    help="HF model repo để backup ckpt (cần token WRITE). Rỗng = không đẩy.")
    ap.add_argument("--hf-every", type=int, default=500,
                    help="đẩy HF mỗi N step (0=tắt). Nên là bội của --save-every.")
    ap.add_argument("--hf-name", default="hybrid_sft.pt")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    use_bf16 = dev == "cuda"
    tok = BPETokenizer.load(a.tokenizer); V = len(tok.tokens)

    obj = torch.load(a.init_from, map_location="cpu", weights_only=False, mmap=True); c = obj["cfg"]
    m = build_from_cfg(c, V)                               # đúng attn_layers + rope_base (thân 2,5B: RoPE 1e6 cho 8K)
    m.load_state_dict(obj["state"]); m = m.to(dev); m.grad_ckpt = a.grad_ckpt
    del obj
    npar = sum(p.numel() for p in m.parameters())
    print(f"[{time.time()-T0:.0f}s] nạp pretrain {a.init_from} · {npar/1e9:.3f}B · d={c['d']} L={c['layers']} "
          f"attn@{m.attn_layers} · rope {c.get('rope_base', 10000.0):g}", flush=True)

    data = SFTData(a.data, tok, a.max_len)
    steps_per_ep = max(1, len(data.train) // (a.bs * a.grad_accum))
    total = steps_per_ep * a.epochs
    if a.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    else:
        try:
            opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=(dev == "cuda"))
        except TypeError:
            opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if a.compile and dev == "cuda":
        m = torch.compile(m)
    print(f"[{time.time()-T0:.0f}s] dev={dev} · {steps_per_ep} step/epoch × {a.epochs} = {total} · bf16={use_bf16}", flush=True)
    warn_if_no_hf_repo(a.hf_repo, a.hf_every)

    step0 = 0
    if a.resume and os.path.exists(a.out):
        ro = torch.load(a.out, map_location="cpu", weights_only=False, mmap=True)
        m.load_state_dict(ro["state"]); step0 = int(ro.get("step") or 0); del ro
        print(f"[{time.time()-T0:.0f}s] CHẠY TIẾP từ {a.out} @step {step0}/{total} (optimizer mới, ấm lại {a.warmup} bước)", flush=True)
    gen = np.random.default_rng(step0); vgen = np.random.default_rng(999)       # lô khác với đoạn đã học
    actx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / max(1, a.warmup)
        r = (s - a.warmup) / max(1, total - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * min(1.0, r)))

    @torch.no_grad()
    def val_loss(nb=10):
        m.eval(); tot = 0.0
        for _ in range(nb):
            x, y = data.batch(a.bs, "val", dev, vgen, tok_budget=a.tok_budget)
            with actx:
                lo = m(x)
            tot += float(F.cross_entropy(lo[:, :-1].reshape(-1, V), y[:, 1:].reshape(-1), ignore_index=-100))
        m.train(); return tot / nb

    print(f"[{time.time()-T0:.0f}s] val-loss TRƯỚC SFT: {val_loss(5):.3f}", flush=True)
    t1 = time.time()
    for step in range(step0 + 1, total + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * (min(1.0, (step - step0) / max(1, a.warmup)) if step0 else 1.0)
        opt.zero_grad(set_to_none=True); lsum = 0.0
        for _ in range(a.grad_accum):
            x, y = data.batch(a.bs, "train", dev, gen, tok_budget=a.tok_budget)
            with actx:
                lo = m(x)
                loss = F.cross_entropy(lo[:, :-1].reshape(-1, V), y[:, 1:].reshape(-1),
                                       ignore_index=-100) / a.grad_accum
            loss.backward(); lsum += loss.detach().item() * a.grad_accum
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0:
            print(f"[{time.time()-T0:.0f}s] step {step}/{total} · loss {lsum/a.grad_accum:.3f} · "
                  f"val {val_loss():.3f} · lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if step % a.save_every == 0:
            sd = (m._orig_mod if hasattr(m, "_orig_mod") else m).state_dict()
            torch.save({"cfg": c, "state": sd, "step": step}, a.out)
            maybe_hf_backup(a.out, a.hf_repo, step, a.hf_every, a.hf_name)
    sd = (m._orig_mod if hasattr(m, "_orig_mod") else m).state_dict()
    torch.save({"cfg": c, "state": sd, "step": total}, a.out)
    print(f"[{time.time()-T0:.0f}s] XONG SFT · LƯU {a.out} · val cuối {val_loss():.3f}", flush=True)
    if a.hf_repo:
        wait_hf_backups()
        start_hf_backup(a.out, a.hf_repo, total, a.hf_name, wait=True)


if __name__ == "__main__":
    main()
