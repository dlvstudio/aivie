"""So-sánh CÔNG-BẰNG mọi model giao-tiếp trên CÙNG bộ câu: model nhỏ tự train (GPT 33M/100M/220M, khuôn "A:/B:")
và model 1.3B fine-tune hội-thoại (khuôn "Hỏi:/Đáp:").

  • Hội-thoại DailyDialog SẠCH (không lượt nào trùng train): cho k lượt đầu (k lẻ → lượt kế là của model), model nói
    lượt tiếp; Gemma chấm tự-nhiên + hợp ngữ-cảnh 1–5 (chấm cả câu gốc làm mốc).
  • Nhớ chi-tiết (probe.jsonl, đã gỡ tên người nói): câu cuối hỏi lại chi-tiết đã nêu → đáp-án có đúng chi-tiết không.

    python curriculum/giaotiep/eval_chat.py --gt gt2_big            (out/curriculum/gt/gt2_big.pt, tok v2)
    PYTHONPATH=scripts:src python curriculum/giaotiep/eval_chat.py --hybrid ckpt/hybrid_1p3b_chat.pt --name chat13
    → out/curriculum/gt/chat_<tên>.json
"""
from __future__ import annotations
import argparse, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_gt import JUDGE, gemma, complete, D  # noqa: E402


def hit(ans, resp):
    n_ = lambda t: re.sub(r"[^\w ]", " ", t.lower()).split()
    digits = re.findall(r"\d+", ans)
    if digits:
        return all(d in re.findall(r"\d+", resp) for d in digits)
    aw = [w for w in n_(ans) if len(w) > 1]
    return bool(aw) and all(w in n_(resp) for w in aw)


def items(n_dialog, seed=0):
    rng = random.Random(seed)
    norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()
    seen = {norm(t) for l in open(f"{D}/dd_envi.jsonl", encoding="utf-8") for r in [json.loads(l)]
            if r["split"] == "train" for t in r["en"] if len(norm(t).split()) >= 4}
    test = [json.loads(l) for l in open(f"{D}/test_dialog.jsonl", encoding="utf-8")]
    test = [d for d in test if len(d["en"]) >= 2 and
            not any(norm(t) in seen for t in d["en"] if len(norm(t).split()) >= 4)]
    rng.shuffle(test)
    out = []
    for d in test[:n_dialog]:
        k = rng.choice([x for x in range(1, len(d["en"])) if x % 2 == 1])   # k lẻ → lượt kế là "B"/model
        for lang in ("vi", "en"):
            out.append({"kind": "dialog", "lang": lang, "turns": d[lang][:k], "ref": d[lang][k]})
    for p in map(json.loads, open(f"{D}/probe.jsonl", encoding="utf-8")):
        for lang in ("vi", "en"):
            out.append({"kind": "probe", "lang": lang, "turns": p[lang], "ref": p[f"dap_an_{lang}"]})
    return out


class GT:
    def __init__(self, tag, tok_path, dev):
        from tokenizers import Tokenizer
        from train_gt import GPT
        ck = torch.load(f"out/curriculum/gt/{tag}.pt", map_location="cpu")
        st = ck["state"]
        nl = 1 + max(int(k.split(".")[1]) for k in st if k.startswith("blocks."))
        self.m = GPT(st["emb.weight"].shape[0], ck["args"]["d"], nl, ck["args"]["heads"])
        self.m.load_state_dict(st)
        self.m.to(dev).eval()
        self.tok, self.dev = Tokenizer.from_file(tok_path), dev

    def reply(self, turns):
        ctx = "\n".join(f"{'AB'[i % 2]}: {t}" for i, t in enumerate(turns))
        return complete(self.m, self.tok, f"{ctx}\n{'AB'[len(turns) % 2]}:", self.dev)


class Hybrid:
    def __init__(self, ckpt, dev):
        from hybrid_infer import load, Runner
        from nn_aivie.core.tokenizer import BPETokenizer
        from nn_aivie.core.vocab import Vocabulary
        m, _, _ = load(ckpt, dev)
        self.r = Runner(m, dev)
        self.tok = BPETokenizer.load("checkpoints/bpe_v1.json")
        self.vocab, self.eos = Vocabulary(self.tok), self.tok.token_to_id["<EOS>"]

    def reply(self, turns):
        n = len(turns)                                        # lượt CUỐI luôn là người dùng
        ids = []
        for i, t in enumerate(turns):
            if (n - 1 - i) % 2 == 0:
                ids += self.tok.encode("Hỏi: " + t.strip() + "\n")
            else:
                ids += self.tok.encode("Đáp: ") + self.tok.encode(t.strip()) + [self.eos]
        ids += self.tok.encode("Đáp: ")
        out = list(self.r.generate(ids, n=80, temp=0, rep_pen=1.1, eos=self.eos))
        return self.vocab.detokenize(out).strip().split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="", help="tag model nhỏ (out/curriculum/gt/<tag>.pt)")
    ap.add_argument("--tok", default=f"{D}/tok.json")
    ap.add_argument("--hybrid", default="", help="checkpoint 1.3B (khuôn Hỏi/Đáp)")
    ap.add_argument("--name", default="")
    ap.add_argument("--n-dialog", type=int, default=100)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = GT(a.gt, a.tok, dev) if a.gt else Hybrid(a.hybrid, dev)
    name = a.name or a.gt
    its = items(a.n_dialog)
    for i, it in enumerate(its):
        it["resp"] = model.reply(it["turns"])
        if i % 100 == 0:
            print(f"[{name}] {i}/{len(its)}", flush=True)
    res = {"name": name}
    for lang in ("vi", "en"):
        pr = [x for x in its if x["kind"] == "probe" and x["lang"] == lang]
        res[f"nho_chi_tiet_{lang}"] = round(sum(hit(x["ref"], x["resp"]) for x in pr) / len(pr), 3)

    def judge(it):
        lang = "tiếng Việt" if it["lang"] == "vi" else "tiếng Anh"
        ctx = "\n".join(f"{'AB'[i % 2]}: {t}" for i, t in enumerate(it["turns"]))
        spk = "AB"[len(it["turns"]) % 2]
        o = {}
        for key in ("resp", "ref"):
            try:
                j = gemma(JUDGE.format(lang=lang, ctx=ctx, spk=spk, resp=it[key] or "(im lặng)"))
                o[key] = {"tu_nhien": int(j["tu_nhien"]), "hop_ngu_canh": int(j["hop_ngu_canh"])}
            except Exception:
                o[key] = None
        return o

    dl = [x for x in its if x["kind"] == "dialog"]
    with ThreadPoolExecutor(16) as ex:
        for it, j in zip(dl, ex.map(judge, dl)):
            it["judge"] = j
    for lang in ("vi", "en"):
        for key, tag in (("resp", "model"), ("ref", "goc")):
            js = [x["judge"][key] for x in dl if x["lang"] == lang and x["judge"].get(key)]
            for c in ("tu_nhien", "hop_ngu_canh"):
                res[f"{c}_{lang}_{tag}"] = round(sum(x[c] for x in js) / max(1, len(js)), 2)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    json.dump({**res, "items": its}, open(f"out/curriculum/gt/chat_{name}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
