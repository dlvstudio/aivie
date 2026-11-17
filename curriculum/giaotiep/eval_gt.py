"""TK_104 · Môn GIAO-TIẾP: chấm model song-ngữ (train_gt.py) trên phần GIỮ RIÊNG của DailyDialog.

1. DỊCH từng lượt nói (en→vi, vi→en): chrF (n-gram ký-tự, β=2) so với bản tham-chiếu.
LỌC SẠCH: DailyDialog có ~43% hội-thoại test trùng lượt nói với train (model chép nguyên câu gốc) → chỉ chấm
hội-thoại KHÔNG có lượt dài (≥4 từ) nào xuất-hiện trong train.
2. HỘI-THOẠI: cho k lượt đầu → model nói lượt tiếp; đo
   • đúng ngôn-ngữ (khách-quan: lượt tiếng Việt phải có dấu, lượt tiếng Anh không có dấu Việt)
   • Gemma chấm 1–5: tự-nhiên + hợp ngữ-cảnh (chấm CẢ câu trả-lời gốc của bộ dữ-liệu làm mốc trần).

    python curriculum/giaotiep/eval_gt.py --tag gt_base [--n-trans 200 --n-dialog 100 --no-judge]
    → out/curriculum/gt/<tag>_eval.json
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, urllib.request
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gt import GPT, D

VI = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


def chrf(hyp, ref, n=6, beta=2.0):
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", "")
    ps, rs = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        m = sum((h & r).values())
        if sum(h.values()):
            ps.append(m / sum(h.values()))
        if sum(r.values()):
            rs.append(m / sum(r.values()))
    p = sum(ps) / len(ps) if ps else 0.0
    r = sum(rs) / len(rs) if rs else 0.0
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


@torch.no_grad()
def complete(m, tok, prompt, dev, n=60, stop="\n"):
    """giải tham-lam đến hết dòng / <eos>."""
    ids = tok.encode(prompt).ids
    start, eos = len(ids), tok.token_to_id("<eos>")
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            nxt = int(m(torch.tensor([ids[-1024:]], device=dev))[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        if stop in tok.decode(ids[start:]):
            break
    return tok.decode(ids[start:]).split(stop)[0].strip()


def gemma(prompt):
    body = {"model": "gemma4:26b-a4b-it-qat", "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0, "num_ctx": 4096}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=300).read())["message"]["content"])


JUDGE = """Chấm câu nói tiếp theo trong một đoạn hội thoại đời thường ({lang}).

Ngữ cảnh:
{ctx}

Câu nói tiếp theo của {spk}: "{resp}"

Chấm 1-5 cho mỗi tiêu chí:
- tu_nhien: có tự nhiên, đúng ngữ pháp, giống người thật nói không (1 = vô nghĩa/sai ngữ pháp, 5 = hoàn toàn tự nhiên)
- hop_ngu_canh: có đáp đúng ý câu trước, hợp ngữ cảnh không (1 = lạc đề, 5 = rất hợp)
Trả về JSON: {{"tu_nhien": n, "hop_ngu_canh": n}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-trans", type=int, default=200)
    ap.add_argument("--n-dialog", type=int, default=100)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--tok", default=f"{D}/tok.json", help="tokenizer đúng với checkpoint (v1: data/giaotiep/v1/tok.json)")
    ap.add_argument("--out-suffix", default="_eval")
    ap.add_argument("--probe", default=f"{D}/probe.jsonl", help="bài NHỚ CHI-TIẾT (gen_dialog_grounded.py --probe)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    ck = torch.load(f"out/curriculum/gt/{a.tag}.pt", map_location="cpu")
    st = ck["state"]
    nl = 1 + max(int(k.split(".")[1]) for k in st if k.startswith("blocks."))
    m = GPT(tok.get_vocab_size(), ck["args"]["d"], nl, ck["args"]["heads"])
    m.load_state_dict(st)
    m.to(dev).eval()
    rng = random.Random(0)
    test = [json.loads(l) for l in open(f"{D}/test_dialog.jsonl", encoding="utf-8")]
    norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()
    seen = {norm(t) for l in open(f"{D}/dd_envi.jsonl", encoding="utf-8") for r in [json.loads(l)]
            if r["split"] == "train" for t in r["en"] if len(norm(t).split()) >= 4}
    n0 = len(test)
    test = [d for d in test if not any(norm(t) in seen for t in d["en"] if len(norm(t).split()) >= 4)]
    out = {"tag": a.tag, "n_dialog_clean": len(test), "n_dialog_all": n0}
    print(f"[{a.tag}] hội-thoại chấm SẠCH: {len(test)}/{n0}", flush=True)

    turns = [(e, v) for d in test for e, v in zip(d["en"], d["vi"]) if 3 <= len(e.split()) <= 30]
    rng.shuffle(turns)
    turns = turns[: a.n_trans]
    ev = [chrf(complete(m, tok, f"Dịch sang tiếng Việt: {e}\n=>", dev), v) for e, v in turns]
    ve = [chrf(complete(m, tok, f"Translate to English: {v}\n=>", dev), e) for e, v in turns]
    out["chrf_en_vi"], out["chrf_vi_en"] = round(sum(ev) / len(ev), 1), round(sum(ve) / len(ve), 1)
    print(f"[{a.tag}] chrF en→vi {out['chrf_en_vi']} · vi→en {out['chrf_vi_en']}", flush=True)

    dl = [d for d in test if len(d["en"]) >= 3]
    rng.shuffle(dl)
    rows = []
    for d in dl[: a.n_dialog]:
        k = rng.randint(1, len(d["en"]) - 1)
        for lang in ("vi", "en"):
            turns_ = d[lang]
            ctx = "\n".join(f"{'AB'[i % 2]}: {t}" for i, t in enumerate(turns_[:k]))
            spk = "AB"[k % 2]
            resp = complete(m, tok, f"{ctx}\n{spk}:", dev)
            ok_lang = bool(VI.search(resp)) if lang == "vi" else not VI.search(resp)
            rows.append({"lang": lang, "ctx": ctx, "spk": spk, "resp": resp, "ref": turns_[k], "lang_ok": ok_lang})
    for lang in ("vi", "en"):
        rs = [r for r in rows if r["lang"] == lang]
        out[f"lang_ok_{lang}"] = round(sum(r["lang_ok"] for r in rs) / len(rs), 3)
        out[f"empty_{lang}"] = round(sum(not r["resp"] for r in rs) / len(rs), 3)
    if not a.no_judge:
        from concurrent.futures import ThreadPoolExecutor

        def judge(r):
            lang = "tiếng Việt" if r["lang"] == "vi" else "tiếng Anh"
            res = {}
            for key in ("resp", "ref"):
                try:
                    j = gemma(JUDGE.format(lang=lang, ctx=r["ctx"], spk=r["spk"], resp=r[key] or "(im lặng)"))
                    res[key] = {"tu_nhien": int(j["tu_nhien"]), "hop_ngu_canh": int(j["hop_ngu_canh"])}
                except Exception:
                    res[key] = None
            return res

        with ThreadPoolExecutor(16) as ex:
            for r, j in zip(rows, ex.map(judge, rows)):
                r["judge"] = j
        for lang in ("vi", "en"):
            for key in ("resp", "ref"):
                js = [r["judge"][key] for r in rows if r["lang"] == lang and r["judge"].get(key)]
                for c in ("tu_nhien", "hop_ngu_canh"):
                    out[f"{c}_{lang}_{'model' if key == 'resp' else 'goc'}"] = round(sum(x[c] for x in js) / max(1, len(js)), 2)
    if os.path.exists(a.probe):                              # NHỚ CHI-TIẾT: khách-quan, không cần Gemma
        def hit(ans, resp):
            n_ = lambda t: re.sub(r"[^\w ]", " ", t.lower()).split()
            digits = re.findall(r"\d+", ans)
            if digits:
                return all(d_ in re.findall(r"\d+", resp) for d_ in digits)
            aw = [w for w in n_(ans) if len(w) > 1]
            return bool(aw) and all(w in n_(resp) for w in aw)
        probes = [json.loads(l) for l in open(a.probe, encoding="utf-8")]
        for lang in ("vi", "en"):
            ok_, ex_ = 0, []
            for p_ in probes:
                t_ = p_[lang]
                ctx = "\n".join(f"{'AB'[i % 2]}: {x}" for i, x in enumerate(t_))
                resp = complete(m, tok, f"{ctx}\n{'AB'[len(t_) % 2]}:", dev)
                h = hit(p_[f"dap_an_{lang}"], resp)
                ok_ += h
                if len(ex_) < 8:
                    ex_.append({"hoi": t_[-1], "dap_an": p_[f"dap_an_{lang}"], "model": resp, "dung": h})
            out[f"nho_chi_tiet_{lang}"] = round(ok_ / len(probes), 3)
            out[f"probe_examples_{lang}"] = ex_
        print(f"[{a.tag}] nhớ chi-tiết vi {out['nho_chi_tiet_vi']} · en {out['nho_chi_tiet_en']} ({len(probes)} bài)", flush=True)
    out["examples"] = rows[:30]
    print(json.dumps({k: v for k, v in out.items() if "examples" not in k}, ensure_ascii=False), flush=True)
    json.dump({**out, "rows": rows}, open(f"out/curriculum/gt/{a.tag}{a.out_suffix}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
