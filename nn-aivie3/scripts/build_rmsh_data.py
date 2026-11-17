"""nn-aivie3 · BUILD DATA ảnh-caption → shard cho train_rmsh_scale.py.

Ra `shard_*.npz`: img=uint8[N,3,H,W] · cap=int16[N,L] (caption token, BPE CỦA TA
`checkpoints/bpe_v1.json` — CÙNG vocab với LM để weight-tie). Build TRÊN M4 rồi upload
lên box (TK_101: FPT chặn ssh/rsync → HF Hub / bundle).

NGUỒN (--source):
  synthetic         — ảnh ngẫu-nhiên + caption mẫu (KIỂM pipeline, không tải mạng)
  local:<dir>       — <dir>/captions.jsonl {"file","caption"} + ảnh trong <dir>
  hf:<dataset>      — HF image-caption (cần `datasets`+`Pillow`); map cột --img-field/--cap-field

DỊCH SANG TIẾNG VIỆT: caption English (COCO/Flickr…) nên dịch trước bằng công-cụ đã có
([[rag-seallm-huong-di]] Ollama SeaLLM, hoặc PhoMT/MTet) → đưa vào dạng local:<dir> đã-dịch.
Script này KHÔNG tự dịch (giữ đơn-nhiệm); caption vào sao tokenize vậy.

CHẠY (kiểm pipeline tại-chỗ, không mạng):
    PYTHONPATH=src python nn-aivie3/scripts/build_rmsh_data.py --source synthetic \
        --limit 40 --img 64 --cap-len 16 --shard-size 20 --out /tmp/rmsh_smoke
THẬT (ví-dụ Flickr30k đã có caption vi ở local):
    PYTHONPATH=src python nn-aivie3/scripts/build_rmsh_data.py --source local:data/flickr_vi \
        --img 224 --cap-len 32 --out data/rmsh_vi
"""
import os, sys, json, glob, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import numpy as np
from nn_aivie.core.tokenizer import BPETokenizer
T0 = time.time()


def load_tok(path):
    tok = BPETokenizer.load(path)
    eos = tok.token_to_id.get("<EOS>", 1)
    print(f"[{time.time()-T0:.0f}s] BPE {path}: |vocab|={len(tok)} · <EOS>={eos}", flush=True)
    return tok, eos


def encode_cap(tok, eos, text, L):
    ids = tok.encode(text)[: L - 1] + [eos]                          # cắt + EOS
    ids = ids + [eos] * (L - len(ids))                               # pad EOS tới L
    return np.asarray(ids[:L], dtype=np.int16)


def to_chw(pil_img, S):
    img = pil_img.convert("RGB").resize((S, S))
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)        # (3,S,S)


def iter_source(src, limit, S):
    """yield (chw_uint8, caption_str)."""
    if src == "synthetic":
        rng = np.random.default_rng(0)
        caps = ["một bức ảnh mẫu", "con mèo trên bàn", "người đi trên đường",
                "bầu trời xanh và mây trắng", "chiếc xe màu đỏ"]
        for i in range(limit or 40):
            yield rng.integers(0, 256, (3, S, S), dtype=np.uint8), caps[i % len(caps)]
        return
    if src.startswith("local:"):
        d = src[6:]
        rows = [json.loads(l) for l in open(os.path.join(d, "captions.jsonl"), encoding="utf-8")]
        from PIL import Image
        for i, r in enumerate(rows):
            if limit and i >= limit:
                break
            p = os.path.join(d, r["file"])
            if not os.path.exists(p):
                continue
            yield to_chw(Image.open(p), S), r["caption"]
        return
    if src.startswith("hf:"):
        from datasets import load_dataset                            # noqa
        name = src[3:]
        ds = load_dataset(name, split="train", streaming=True)
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            img = r[IMG_FIELD]; cap = r[CAP_FIELD]
            cap = cap[0] if isinstance(cap, list) else cap
            yield to_chw(img, S), cap
        return
    raise SystemExit(f"--source lạ: {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--tokenizer", default="checkpoints/bpe_v1.json")
    ap.add_argument("--img", type=int, default=224); ap.add_argument("--cap-len", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=2000); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--img-field", default="image"); ap.add_argument("--cap-field", default="caption")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global IMG_FIELD, CAP_FIELD; IMG_FIELD, CAP_FIELD = a.img_field, a.cap_field

    os.makedirs(a.out, exist_ok=True)
    tok, eos = load_tok(a.tokenizer)
    imgs, caps, shard, total = [], [], 0, 0

    def flush():
        nonlocal imgs, caps, shard, total
        if not imgs:
            return
        p = os.path.join(a.out, f"shard_{shard:05d}.npz")
        np.savez_compressed(p, img=np.stack(imgs), cap=np.stack(caps))
        total += len(imgs)
        print(f"[{time.time()-T0:.0f}s] {p}: {len(imgs)} cặp (tổng {total})", flush=True)
        imgs, caps, shard = [], [], shard + 1

    for chw, cap in iter_source(a.source, a.limit, a.img):
        imgs.append(chw); caps.append(encode_cap(tok, eos, cap, a.cap_len))
        if len(imgs) >= a.shard_size:
            flush()
    flush()
    print(f"[{time.time()-T0:.0f}s] XONG: {total} cặp ảnh-caption → {a.out}/shard_*.npz "
          f"(img {a.img}² uint8 · cap {a.cap_len} int16). Kiểm: train_rmsh_scale.py --data {a.out}", flush=True)


if __name__ == "__main__":
    main()
