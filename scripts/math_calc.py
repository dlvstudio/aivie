"""1.3B DÙNG MÁY TÍNH khi giải toán — không cần train lại.

Model (SFT v2) đã viết lời-giải dạng "50 + 80 = 130", nhưng hay TÍNH SAI. Lúc sinh chữ, hễ model vừa viết xong một
biểu-thức số học kết thúc bằng "=" thì hệ-thống TÍNH HỘ và chèn kết-quả đúng vào (model đọc tiếp kết-quả đó).
Bài-học TK_104: model học được CÁCH trình-bày nhưng không tự tính đúng → giao phần tính cho công-cụ.

  build    Gemma dịch 250 bài GSM8K (phần TEST — model chưa từng thấy) sang tiếng Việt → data/math_test/gsm8k_vi.jsonl
  eval     --ckpt … --mode plain|calc  → % đúng đáp-số (GSM8K-vi 250 bài + 18 bài toán eval_v1)
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path[:0] = ["scripts", "src"]

EXPR = re.compile(r"(\(?\s*-?\d[\d.,]*\s*(?:[\+\-\*x×/:]\s*\(?\s*-?\d[\d.,]*\s*\)?\s*)+)=\s*$")


def calc(expr):
    """Tính biểu-thức số học an-toàn (chỉ số, + - * / ngoặc). Số kiểu '1.200' (nghìn) / '1,5' (thập-phân)."""
    e = expr.replace("×", "*").replace("x", "*").replace(":", "/")
    e = re.sub(r"(\d)\.(\d{3})(?!\d)", r"\1\2", e)             # 1.200 → 1200
    e = e.replace(",", ".")
    if not re.fullmatch(r"[\d.\s\+\-\*/()]+", e):
        return None
    try:
        v = eval(e, {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(v, float):
        v = round(v, 4)
        if v == int(v):
            v = int(v)
    return str(v).replace(".", ",") if isinstance(v, float) else str(v)


def build(a):
    from datasets import load_dataset
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    ds = list(load_dataset("openai/gsm8k", "main", split="test"))[: a.n]

    def tr(r):
        body = {"model": "gemma4:26b-a4b-it-qat", "stream": False, "think": False,
                "options": {"temperature": 0, "num_ctx": 4096},
                "messages": [{"role": "user", "content": "Dịch bài toán sau sang tiếng Việt tự nhiên, GIỮ NGUYÊN mọi con số. "
                              "Chỉ trả về bài toán đã dịch, không giải.\n\n" + r["question"]}]}
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        q = json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"].strip()
        return {"q": q, "gold": r["answer"].split("####")[-1].strip().replace(",", ""), "q_en": r["question"]}

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(tr, ds))
    os.makedirs("data/math_test", exist_ok=True)
    with open("data/math_test/gsm8k_vi.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} bài → data/math_test/gsm8k_vi.jsonl", flush=True)


DIGIT_IDS = None


def digit_ids(vocab, V):
    """token bắt đầu bằng chữ-số → cấm ngay sau khi chèn kết-quả (tránh '= 130 130')."""
    global DIGIT_IDS
    if DIGIT_IDS is None:
        DIGIT_IDS = [i for i in range(V) if re.match(r"\s*\d", vocab.detokenize([i]) or "")]
    return DIGIT_IDS


def solve(r, tok, vocab, eos, q, use_calc, n=400, rep_pen=1.1):
    """Giải tham-lam; use_calc: gặp '<biểu-thức> =' → chèn kết-quả tính hộ (ép token)."""
    import numpy as np
    ids = tok.encode("Hỏi: " + q.strip() + "\n") + tok.encode("Đáp: ")
    lo, st = r.prefill(ids)
    lo = lo[-1]
    hist, out, n_calc, just = list(ids), [], 0, False
    while len(out) < n:
        l = lo.float().cpu().numpy().astype(np.float64)
        for t in set(hist[-64:]):
            l[t] -= rep_pen
        if just:
            l[digit_ids(vocab, len(l))] = -1e9
            just = False
        nxt = int(l.argmax())
        if nxt == eos:
            break
        out.append(nxt); hist.append(nxt)
        lo, st = r.step(nxt, st)
        if use_calc:
            text = vocab.detokenize(out)
            m = EXPR.search(text[-80:])
            if m and text.endswith("="):
                v = calc(m.group(1))
                if v is not None:
                    for t in tok.encode(" " + v):                # ÉP kết-quả đúng vào chuỗi
                        out.append(t); hist.append(t)
                        lo, st = r.step(t, st)
                    n_calc += 1
                    just = True
    return vocab.detokenize(out).strip(), n_calc


def num(s):
    s = s.replace(".", "").replace(",", ".") if re.search(r"\d\.\d{3}", s) else s.replace(",", ".")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def evaluate(a):
    import torch
    from hybrid_infer import load, Runner
    from nn_aivie.core.tokenizer import BPETokenizer
    from nn_aivie.core.vocab import Vocabulary
    dev = "cuda" if torch.cuda.is_available() else "mps"
    m, _, _ = load(a.ckpt, dev)
    r = Runner(m, dev)
    tok = BPETokenizer.load("checkpoints/bpe_v1.json")
    vocab, eos = Vocabulary(tok), tok.token_to_id["<EOS>"]
    sets = {"gsm8k_vi": [json.loads(l) for l in open("data/math_test/gsm8k_vi.jsonl", encoding="utf-8")][: a.n],
            "eval_v1": [json.loads(l) for l in open("data/eval_clean/eval_v1.jsonl", encoding="utf-8") if '"math"' in l]}
    res, rows = {"ckpt": a.ckpt, "mode": a.mode}, []
    for name, qs in sets.items():
        ok, nc = 0, 0
        for i, x in enumerate(qs):
            ans, c = solve(r, tok, vocab, eos, x["q"], a.mode == "calc")
            tail = ans.split("Đáp án:")[-1] if "Đáp án:" in ans else ans[-60:]
            g, p = num(str(x["gold"])), num(tail)
            hit = g is not None and p is not None and abs(g - p) < 1e-6
            ok += hit; nc += c
            rows.append({"set": name, "q": x["q"][:200], "gold": x["gold"], "ans": ans[-400:], "ok": hit, "calc": c})
            if i % 50 == 0:
                print(f"[{a.tag}] {name} {i}/{len(qs)} · đúng {ok}", flush=True)
        res[name] = round(ok / len(qs), 3)
        res[f"{name}_calc_per_bai"] = round(nc / len(qs), 2)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    os.makedirs("out/math_calc", exist_ok=True)
    json.dump({**res, "rows": rows}, open(f"out/math_calc/{a.tag}.json", "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--ckpt", default="ckpt/hybrid_1p3b_sft_v2.pt")
    ap.add_argument("--mode", default="calc", choices=["plain", "calc"])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.tag = a.tag or f"{os.path.basename(a.ckpt)[:-3]}_{a.mode}"
    build(a) if a.cmd == "build" else evaluate(a)


if __name__ == "__main__":
    main()
