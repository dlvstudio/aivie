"""TK_104 · Đo NGOẠI-SUY CHIỀU SÂU: model train tới 4 bước (lớp 5) giải được bài 5-6 / 7-8 / 10-12 bước không?

    python curriculum/eval_deep.py quote_s0 rand_wsd_s0 ...   (đọc out/curriculum/<tag>.pt)
    → out/curriculum/deep_eval.json (gộp, ghi đè theo tag)
"""
import json, os, sys
import torch
import loop_lm
from loop_lm import LoopLM, Vocab, load_level, evaluate, index_row

OUT = "out/curriculum/deep_eval.json"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v = Vocab()
    tests = {lv: load_level(lv, "test")[:300] for lv in (5, 6, 7, 8)}
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag in sys.argv[1:]:
        ck = torch.load(f"out/curriculum/{tag}.pt", map_location="cpu")
        a = ck["args"]
        st = ck["state"]
        n = lambda pre: 1 + max(int(k.split(".")[1]) for k in st if k.startswith(pre + "."))
        m = LoopLM(st["emb.weight"].shape[0], a["d"], 8, n("pre"), n("core"), n("post"), a.get("arch", ""))
        m.load_state_dict(st)
        m.to(dev)
        loop_lm.RANDPOS["max"] = a.get("rand_pos", 0)
        quote = a.get("quote", False) or a.get("index", "") == "quote"
        out = {}
        for lv, rows in tests.items():
            if a.get("index"):
                rows = [index_row(r, a["index"] == "quote") for r in rows]
            mx = {5: 170, 6: 250, 7: 330, 8: 480}[lv] * (2 if quote else 1) + (60 if a.get("index") else 0)
            out[f"lop{lv}"] = round(evaluate(m, v, rows, min(4, a.get("r_train", 4)), dev, max_new=mx), 4)
        res[tag] = out
        print(tag, out, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
