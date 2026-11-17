"""§TK_103 · NEEDLE-IN-HAYSTACK tiếng Việt — model CÓ DÙNG được ngữ-cảnh dài không?

Perplexity phẳng (probe §11) mới là điều-kiện CẦN. Test này là điều-kiện ĐỦ: giấu 1
"mã bí mật = <từ>" ở độ sâu d trong haystack viwiki dài, rồi hỏi lại ở cuối — đo model
có LÔI ĐÚNG từ ra không (argmax trên 5 từ-mã single-token; chance = 20%).

Dùng RoPE NTK (giãn context 0-đồng, đã validate §11). Control = không cài needle → phải ~chance.

    PYTHONPATH=scripts:src .venv/bin/python scripts/needle_haystack.py --len 8192 --rope ntk
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np, torch

sys.path.insert(0, "scripts")
import train_hybrid_scale as THS
from train_hybrid_scale import HybridLM, HAVE_KERNEL
from probe_length_extrap import make_rope, load_model, build_stream
from nn_aivie.core.tokenizer import BPETokenizer

# 5 từ-mã single-token (đã kiểm) — needle chọn 1, hỏi argmax trên cả 5
CODES = ["táo", "chuối", "xoài", "nho", "dứa"]


@torch.no_grad()
def ask(m, ids, cand_ids, dev):
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    lo = m(x)[0, -1].float()
    c = torch.tensor(cand_ids, device=dev)
    return int(lo[c].argmax())                            # index trong CODES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hf_hybrid/backup_hybrid_1p3b_pretrain_step143433.pt")
    ap.add_argument("--tokenizer", default="checkpoints/bpe_v1.json")
    ap.add_argument("--len", type=int, default=8192)
    ap.add_argument("--rope", default="ntk", choices=["none", "ntk", "pi"])
    ap.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--trained-ctx", type=int, default=2048)
    a = ap.parse_args()
    T0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"dev={dev} · kernel={HAVE_KERNEL} · rope={a.rope} · load {a.ckpt}", flush=True)
    m, c, step = load_model(a.ckpt, dev)
    THS.rope = make_rope(a.rope, a.trained_ctx)           # giãn context
    tok = BPETokenizer.load(a.tokenizer)
    cand_ids = [tok.encode(" " + w)[0] for w in CODES]
    query_ids = tok.encode("\n\nHỏi: Mã bí mật là gì?\nĐáp: Mã bí mật là")

    # haystack nền (đủ dài để sau khi chèn needle + query vẫn ~--len)
    hay = build_stream(tok, a.len)[:a.len - 40]
    print(f"[{time.time()-T0:.0f}s] haystack {len(hay)} token · chạy {len(a.depths)} độ sâu × {len(CODES)} từ…", flush=True)

    # CONTROL: không needle → chỉ kiểm model có thiên-lệch 1 từ không.
    # (needle-test đã tự có chance 20% vì xoay đủ 5 từ làm đáp-án đúng.)
    dflt = ask(m, hay + query_ids, cand_ids, dev)
    print(f"[{time.time()-T0:.0f}s] CONTROL (không needle): model mặc-định chọn "
          f"'{CODES[dflt]}' — chỉ để kiểm bias.", flush=True)

    # NEEDLE ở từng độ sâu, thử cả 5 từ
    print(f"\n{'độ sâu':>8} | {'hit/5':>6} | {'%':>5}", flush=True)
    rows = []
    for d in a.depths:
        pos = int(d * len(hay))
        hit = 0
        for gi, w in enumerate(CODES):
            needle = tok.encode(f" Ghi nhớ: mã bí mật là {w}. ")
            ids = hay[:pos] + needle + hay[pos:] + query_ids
            pred = ask(m, ids, cand_ids, dev)
            hit += int(pred == gi)
        rows.append((d, hit))
        print(f"{d:8.2f} | {hit}/5   | {100*hit/5:4.0f}%", flush=True)

    avg = 100 * sum(h for _, h in rows) / (5 * len(rows))
    print(f"\n[{time.time()-T0:.0f}s] TRUNG-BÌNH retrieval = {avg:.0f}% (chance 20%)", flush=True)
    if avg >= 70:
        print("KẾT-LUẬN: ✅ DÙNG ĐƯỢC ngữ-cảnh dài — long-context THỰC-CHẤT (không chỉ perplexity).", flush=True)
    elif avg >= 40:
        print("KẾT-LUẬN: ⚠️ retrieval MỘT PHẦN — hơn chance rõ nhưng chưa chắc; cần fine-tune giãn.", flush=True)
    else:
        print("KẾT-LUẬN: ❌ ~chance — perplexity phẳng nhưng KHÔNG lôi được fact xa; cần fine-tune.", flush=True)


if __name__ == "__main__":
    main()
