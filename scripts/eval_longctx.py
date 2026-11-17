"""P0 · BỘ ĐO NGỮ CẢNH DÀI cho thân HybridLM (1.3B gốc 2K → bản nuôi lớn 8K).

  pos   loss theo VỊ TRÍ token trên văn bản dài ghép (0–1K, 1–2K, 2–4K, 4–8K): model dùng được ngữ cảnh dài thì loss
        ở đoạn sau ≤ đoạn đầu; vượt độ dài đã học thì loss vọt lên.
  kim   "kim trong đống rơm" tiếng Việt: cài "Mã số bí mật của kho hàng là <5 số>." vào văn bản dài (1K/2K/4K/8K token,
        độ sâu 10/50/90%), cuối văn bản hỏi lại → model viết tiếp có đúng 5 số không.

    python scripts/eval_longctx.py --ckpt <ckpt> --texts data/vlm/news/docs.jsonl --lens 1024,2048,4096,8192 --n 8
    (CUDA dùng kernel Mamba; CPU/MPS dùng khối thuần-PyTorch — chậm, chỉ để thử nhỏ)
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys, time
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")]
import torch
import torch.nn.functional as F


def load_texts(path, n=4000):
    if path.endswith(".jsonl"):
        out = []
        for l in open(path, encoding="utf-8"):
            x = json.loads(l)
            out.append("\n".join(x) if isinstance(x, list) else x.get("text", ""))
            if len(out) >= n:
                break
        return out
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=["text"]).column("text").to_pylist()[:n]


def load_model(ckpt, dev):
    import train_hybrid_scale as ths
    if not dev.startswith("cuda"):
        ths.HAVE_KERNEL = False
    obj = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    m = ths.build_from_cfg(obj["cfg"])
    m.load_state_dict(obj["state"])
    return m.to(dev).eval(), obj["cfg"]


class Shards:
    """thư mục shard uint16 (build_cpt_data.py / pretrain) → lấy đoạn token liền mạch ngẫu nhiên."""
    def __init__(self, path):
        import glob
        import numpy as np
        self.mm = [np.load(p, mmap_mode="r") for p in sorted(glob.glob(os.path.join(path, "shard_*.npy")))]
        assert self.mm, f"không thấy shard trong {path}"

    def take(self, n, r):
        m = r.choice([x for x in self.mm if len(x) > n + 1] or self.mm)
        o = r.randrange(0, max(1, len(m) - n - 1))
        return [int(x) for x in m[o:o + n]]


def filler_ids(tok, texts, n_tok, r, eos):
    if isinstance(texts, Shards):
        return texts.take(n_tok, r)
    ids = []
    while len(ids) < n_tok:
        ids += tok.encode(r.choice(texts)) + [eos]
    return ids[:n_tok]


@torch.no_grad()
def pos_loss(m, tok, texts, L, n, r, dev, eos, amp):
    buckets = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384)]
    sums = {b: [0.0, 0] for b in buckets if b[0] < L}
    for _ in range(n):
        ids = torch.tensor([filler_ids(tok, texts, L + 1, r, eos)], device=dev)
        with amp():
            lo = m(ids[:, :-1])
        l = F.cross_entropy(lo.float()[0], ids[0, 1:], reduction="none")
        for (a, b) in sums:
            seg = l[a:min(b, L)]
            sums[(a, b)][0] += float(seg.sum()); sums[(a, b)][1] += seg.numel()
    return {f"{a // 1024}-{b // 1024}K": round(s / max(1, k), 3) for (a, b), (s, k) in sums.items()}


@torch.no_grad()
def greedy(m, ids, n, dev, amp):
    out = []
    x = torch.tensor([ids], device=dev)
    for _ in range(n):
        with amp():
            lo = m(x)[0, -1]
        t = int(lo.argmax())
        out.append(t)
        x = torch.cat([x, torch.tensor([[t]], device=dev)], 1)
    return out


def needle(m, tok, voc, texts, L, depth, r, dev, eos, amp):
    code = str(r.randint(10000, 99999))
    nd = tok.encode(f" Mã số bí mật của kho hàng là {code}. Hãy nhớ mã số này. ")
    q = tok.encode(" Câu hỏi: Mã số bí mật của kho hàng là gì? Trả lời: Mã số bí mật của kho hàng là")
    hay = filler_ids(tok, texts, L - len(nd) - len(q), r, eos)
    k = int(len(hay) * depth)
    ids = hay[:k] + nd + hay[k:] + q
    got = voc.detokenize(greedy(m, ids, 6, dev, amp))
    return code in re.sub(r"\D", "", got)[:6], code, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--texts", default="data/fineweb2hq/vie_Latn/000_00000.parquet")
    ap.add_argument("--lens", default="1024,2048,4096,8192")
    ap.add_argument("--n", type=int, default=8, help="số mẫu mỗi độ dài (kim: × 3 độ sâu)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from nn_aivie.core.tokenizer import BPETokenizer
    from nn_aivie.core.vocab import Vocabulary
    tok = BPETokenizer.load("checkpoints/bpe_v1.json"); voc = Vocabulary(tok); eos = tok.token_to_id["<EOS>"]
    texts = Shards(a.texts) if os.path.isdir(a.texts) else [t for t in load_texts(a.texts) if len(t) > 300]
    m, c = load_model(a.ckpt, a.device)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if a.device.startswith("cuda") else (lambda: torch.autocast("cpu", enabled=False))
    r = random.Random(0)
    res = {"ckpt": a.ckpt, "tag": a.tag, "layers": c["layers"], "rope_base": c.get("rope_base", 10000.0), "pos": {}, "kim": {}}
    T0 = time.time()
    for L in [int(x) for x in a.lens.split(",")]:
        res["pos"][L] = pos_loss(m, tok, texts, L, max(2, a.n // 2), r, a.device, eos, amp)
        hits = []
        for d in (0.1, 0.5, 0.9):
            for _ in range(a.n):
                ok, code, got = needle(m, tok, voc, texts, L, d, r, a.device, eos, amp)
                hits.append(ok)
        res["kim"][L] = f"{sum(hits)}/{len(hits)}"
        print(f"[{time.time()-T0:.0f}s] L={L} · loss theo vị trí {res['pos'][L]} · kim {res['kim'][L]} (vd: {code} → {got!r})", flush=True)
    print("### " + json.dumps(res, ensure_ascii=False), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
