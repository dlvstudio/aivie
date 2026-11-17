"""Model HybridLM (checkpoint bất-kỳ) làm bài eval_v1 TRỰC-TIẾP — không cần server UI.
Chạy được trên box CUDA ngay sau khi train → chỉ kéo file kết-quả nhỏ về Mac để chấm (eval_clean.py judge).

    PYTHONPATH=scripts:src python scripts/gen_evalclean_local.py --ckpt ckpt/hybrid_1p3b_sft_v2.pt --tag sftv2
    → out/evalclean_sftv2.jsonl  (cùng định-dạng eval_clean.py gen-http: nhiệt-độ 0, rep-pen 1.1, ≤500 token)
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path[:0] = ["scripts", "src"]
import torch
from hybrid_infer import load, Runner
from nn_aivie.core.tokenizer import BPETokenizer
from nn_aivie.core.vocab import Vocabulary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src", default="data/eval_clean/eval_v1.jsonl")
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--rep-pen", type=float, default=1.1)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    m, _, step = load(a.ckpt, dev)
    r = Runner(m, dev)
    tok = BPETokenizer.load("checkpoints/bpe_v1.json")
    vocab, eos = Vocabulary(tok), tok.token_to_id["<EOS>"]
    qs = [json.loads(l) for l in open(a.src, encoding="utf-8")]
    os.makedirs("out", exist_ok=True)
    path = f"out/evalclean_{a.tag}.jsonl"
    T0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        for i, q in enumerate(qs, 1):
            t = time.time()
            ids = tok.encode("Hỏi: " + q["q"].strip() + "\n") + tok.encode("Đáp: ")
            out = list(r.generate(ids, n=a.max_new, temp=0, rep_pen=a.rep_pen, eos=eos))
            f.write(json.dumps({**q, "tag": a.tag, "answer": vocab.detokenize(out).strip(),
                                "sec": round(time.time() - t, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"[{time.time()-T0:.0f}s] {i}/{len(qs)}", flush=True)
    print(f"### LƯU {path} · checkpoint bước {step} · {dev}", flush=True)


if __name__ == "__main__":
    main()
