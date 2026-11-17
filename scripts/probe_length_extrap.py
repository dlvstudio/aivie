"""§TK_103 · PROBE NGOẠI-SUY ĐỘ-DÀI — model train ở ctx=2048 có dùng được ngữ-cảnh dài hơn?

Đo bits/token THEO VỊ-TRÍ trong một chuỗi dài (nối bài viwiki + EOS, khớp cách pack lúc train).
- bits phẳng khi vượt 2048 → kiến-trúc NGOẠI-SUY được (tín-hiệu tốt cho hướng long-context).
- bits vọt lên sau 2048 → SẬP → cần fine-tune giãn độ-dài (RoPE scaling / continue-train).

    PYTHONPATH=scripts:src .venv/bin/python scripts/probe_length_extrap.py --len 8192
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, torch

sys.path.insert(0, "scripts")
import train_hybrid_scale as THS
from train_hybrid_scale import HybridLM, HAVE_KERNEL
from nn_aivie.core.tokenizer import BPETokenizer

V = 32002
VIWIKI = "data/text_corpus/viwiki.jsonl"


def make_rope(mode, trained_ctx):
    """Chế-độ giãn context 0-đồng (monkey-patch rope toàn-cục của model).
    none = gốc · ntk = tăng base (NTK-aware) · pi = nén vị-trí (Position Interpolation)."""
    def rope_scaled(x, base=10000.0):
        b, h, l, hd = x.shape
        pos = torch.arange(l, device=x.device, dtype=torch.float32)
        if l > trained_ctx and mode == "ntk":
            scale = l / trained_ctx
            base = base * (scale ** (hd / (hd - 2)))          # NTK-aware: nới tần-số cao
        elif l > trained_ctx and mode == "pi":
            pos = pos * (trained_ctx / l)                     # PI: nén mọi vị-trí về vùng đã học
        inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
        ang = torch.outer(pos, inv)
        cos, sin = ang.cos()[None, None], ang.sin()[None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)
    return rope_scaled


def load_model(ckpt, dev):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    c, state, step = obj["cfg"], obj["state"], obj.get("step")
    del obj
    ae = c.get("attn_every", 0)
    attn = list(range(ae - 1, c["layers"], ae)) if ae and ae > 0 else []
    m = HybridLM(V, c["d"], c["layers"], attn, c.get("d_state", 16),
                 c.get("heads", 16), c.get("kv_heads"))
    m.load_state_dict(state); del state
    m.eval().to(dev)
    return m, c, step


def build_stream(tok, need, eos=1):
    """Nối text viwiki + EOS tới khi đủ `need` token (khớp packing lúc train)."""
    ids = []
    with open(VIWIKI, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj if isinstance(obj, str) else obj.get("text", "")   # viwiki: mỗi dòng là 1 chuỗi JSON
            if not t or len(t) < 40:
                continue
            ids.extend(tok.encode(t)); ids.append(eos)
            if len(ids) >= need + 1:
                break
    return ids[:need + 1]


@torch.no_grad()
def per_pos_bits(m, ids, dev, chunk_v=True):
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
    y = torch.tensor(ids[1:], dtype=torch.long, device=dev)
    logits = m(x)[0].float()                              # [L, V]
    loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")  # [L]
    return (loss / np.log(2)).cpu().numpy()              # bits/token theo vị-trí


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf_hybrid/backup_hybrid_1p3b_pretrain_step143433.pt")
    ap.add_argument("--tokenizer", default="checkpoints/bpe_v1.json")
    ap.add_argument("--len", type=int, default=8192, help="độ dài chuỗi probe (thử; tự lùi nếu OOM)")
    ap.add_argument("--bucket", type=int, default=512)
    ap.add_argument("--trained-ctx", type=int, default=2048)
    ap.add_argument("--rope-modes", nargs="+", default=["none", "ntk", "pi"],
                    help="thử giãn context 0-đồng: none/ntk/pi")
    a = ap.parse_args()
    T0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"dev={dev} · kernel={HAVE_KERNEL} · load {a.ckpt}", flush=True)
    m, c, step = load_model(a.ckpt, dev)
    print(f"[{time.time()-T0:.0f}s] step={step} · d={c['d']} L={c['layers']} "
          f"ctx-train={c.get('ctx')} · dựng chuỗi probe…", flush=True)
    tok = BPETokenizer.load(a.tokenizer)
    ids = build_stream(tok, a.len)
    print(f"[{time.time()-T0:.0f}s] chuỗi {len(ids)-1} token · forward…", flush=True)

    # chọn độ dài forward được (thử lùi nếu OOM) — dùng chung cho mọi chế-độ
    Luse = None
    for L in [a.len, 6144, 4096, 3072, 2560]:
        if L <= len(ids) - 1:
            Luse = L; break
    if Luse is None:
        Luse = len(ids) - 1

    results = {}
    for mode in a.rope_modes:
        THS.rope = make_rope(mode, a.trained_ctx)             # monkey-patch rope toàn-cục
        bits = None
        for L in [Luse, 6144, 4096, 3072, 2560]:
            if L > len(ids) - 1:
                continue
            try:
                bits = per_pos_bits(m, ids[:L + 1], dev)
                break
            except RuntimeError as e:
                print(f"⚠️ [{mode}] L={L} lỗi ({str(e)[:50]}) → lùi", flush=True)
                if dev == "mps":
                    torch.mps.empty_cache()
        if bits is None:
            print(f"⚠️ [{mode}] không forward được", flush=True); continue
        results[mode] = bits
        print(f"[{time.time()-T0:.0f}s] [{mode}] forward OK ở L={len(bits)}", flush=True)

    if not results:
        sys.exit("không forward được ở bất-kỳ chế-độ nào")

    # bảng chi-tiết cho chế-độ gốc + tổng-hợp so-sánh
    ref = results["none"] if "none" in results else next(iter(results.values()))
    L = len(ref)
    print(f"\n===== BITS/TOKEN THEO VỊ-TRÍ · chế-độ 'none' (ctx-train={a.trained_ctx}) =====", flush=True)
    for s in range(0, L, a.bucket):
        e = min(s + a.bucket, L)
        flag = "  ← vượt ngưỡng train" if (s < a.trained_ctx <= e) else ("  (ngoại-suy)" if s >= a.trained_ctx else "")
        print(f"{s:6d}-{e:<6d} | {float(np.mean(ref[s:e])):6.3f}{flag}", flush=True)

    print(f"\n===== SO-SÁNH GIÃN CONTEXT 0-ĐỒNG (in-dist <{a.trained_ctx} vs ngoại-suy >{a.trained_ctx}) =====", flush=True)
    print(f"{'chế-độ':>8} | {'in-dist':>8} | {'ngoại-suy':>10} | {'Δ':>7}", flush=True)
    for mode, bits in results.items():
        if len(bits) <= a.trained_ctx:
            continue
        ind = float(np.mean(bits[:a.trained_ctx]))
        ext = float(np.mean(bits[a.trained_ctx:]))
        print(f"{mode:>8} | {ind:8.3f} | {ext:10.3f} | {100*(ext-ind)/ind:+6.0f}%", flush=True)
    print("\n(Δ nhỏ = ngoại-suy tốt. Nếu ntk/pi kéo Δ về gần 0 → có long-context MIỄN PHÍ, "
          "không cần train. Nếu vẫn cao → cần fine-tune giãn độ-dài.)", flush=True)


if __name__ == "__main__":
    main()
