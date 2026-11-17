"""Thay TẦNG GLOBAL (5,11,…,47) của Gemma 4 12B bằng Mamba P1 — đo KL (2K, 8K) và thử 'kim đáy bể' 8K.

  python scripts/exp2_global.py --run gl [--needles 5] [--ctx 7500]
Biến thể: gốc · Mamba từng tầng global · Mamba cả 8 · bỏ cả 8 (· bỏ từng tầng cho KL).
Kết quả: out/exp2_mamba_layer/<run>/global_eval.json
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import exp2_mamba_layer as E

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="gl")
ap.add_argument("--needles", type=int, default=5)
ap.add_argument("--ctx", type=int, default=7500)
ap.add_argument("--skip-kl", action="store_true")
ap.add_argument("--kl-vars", default="")
ap.add_argument("--kl-sets", default="2k,8k")
ap.add_argument("--kim-vars", default="")
ap.add_argument("--keep", default="", help="vd '11,47;11,29,47' — giữ attention ở các tầng này, Mamba ở các tầng global còn lại")
a = ap.parse_args()

model, tok = E.load_model(); lm = E.lm_inner(model)
d = os.path.join(E.OUT_BASE, a.run)
lay = json.load(open(os.path.join(d, "train_log.json")))["layers"]
bag = E.Bag([E.Mamba2Mixer(lm.layers[i].self_attn) for i in lay]); bag.load_weights(os.path.join(d, "mixers.safetensors"))
mix = dict(zip(lay, bag.mixers))
allL = list(range(len(lm.layers)))
orig = {i: lm.layers[i].self_attn for i in allL}
E.log(f"Mamba cho tầng global {lay}")


def setv(S_m=(), S_z=()):
    for i in allL:
        lm.layers[i].self_attn = orig[i]
    for i in S_m:
        lm.layers[i].self_attn = E.Swap(mix[i])
    for i in S_z:
        lm.layers[i].self_attn = E.Swap(None, zero=True)


variants = {"goc": ((), ())}
for i in lay:
    variants[f"mamba_{i}"] = ((i,), ())
variants["mamba_all"] = (tuple(lay), ())
variants["bo_all"] = ((), tuple(lay))
no11 = tuple(i for i in lay if i != 11)
variants["mamba_no11"] = (no11, ())
variants["mamba_no11_bo11"] = (no11, (11,))
for ks in [x for x in a.keep.split(";") if x]:
    kp = {int(v) for v in ks.split(",")}
    variants["giu_" + "_".join(str(v) for v in sorted(kp))] = (tuple(i for i in lay if i not in kp), ())
out_p = os.path.join(d, "global_eval.json")
res = json.load(open(out_p)) if os.path.exists(out_p) else {}

# ── KL ──
if not a.skip_kl:
    mt = os.path.join(E.ROOT, "data/multiturn/train.jsonl")
    sets = {"2k": E.pack(tok, E.jsonl_chats(tok, mt, 400, from_end=True), 2048, 3),
            "8k": E.pack(tok, E.jsonl_chats(tok, mt, 400), 8192, 1)}
    kv = dict(variants); kv.update({f"bo_{i}": ((), (i,)) for i in lay})
    for sname, seqs in sets.items():
        toks = [mx.array(s)[None] for s in seqs]
        base = []
        for t in toks:
            setv(); lo = model(t); mx.eval(lo); base.append(lo)
        if sname not in a.kl_sets.split(","):
            continue
        for v, (Sm, Sz) in kv.items():
            if v == "goc" or (a.kl_vars and v not in a.kl_vars.split(",")):
                continue
            kl = ag = 0.0
            for t, lo in zip(toks, base):
                setv(Sm, Sz); ls = model(t); mx.eval(ls)
                k, g, _, _ = E.kl_stats(lo, ls, t); kl += k; ag += g
            res.setdefault("KL", {}).setdefault(sname, {})[v] = {"KL": kl / len(toks), "dongy": ag / len(toks)}
            E.log(f"KL {sname} {v:10s} {kl/len(toks):.3f} ({ag/len(toks)*100:.1f}%)")
            json.dump(res, open(out_p, "w"), ensure_ascii=False, indent=1)
        del base
    setv()

# ── kim đáy bể ──
from mlx_lm import generate
rng = random.Random(7)
mt_lines = open(os.path.join(E.ROOT, "data/multiturn/train.jsonl")).read().splitlines()
filler_parts = []
for l in mt_lines[:3000]:
    for m in json.loads(l)["messages"]:
        if m["role"] == "assistant":
            filler_parts.append(m["content"].strip())
rng.shuffle(filler_parts)
items = ["kho hàng số 3", "phòng họp tầng 7", "két sắt của công ty", "tủ hồ sơ phòng kế toán", "cổng phụ nhà máy",
         "máy chủ dự phòng", "xe tải biển 29C"]
cases = []
for n in range(a.needles):
    code = str(rng.randint(1000, 9999)); thing = items[n % len(items)]
    needle = f"Ghi chú quan trọng: mã mở khoá của {thing} là {code}. Hãy nhớ mã này."
    paras, ntok = [], 0
    while ntok < a.ctx:
        p = filler_parts[rng.randrange(len(filler_parts))]
        paras.append(p); ntok += len(tok.encode(p, add_special_tokens=False))
    pos = max(1, int(len(paras) * [0.05, 0.25, 0.5, 0.1, 0.4][n % 5]))
    paras.insert(pos, needle)
    q = "\n\n".join(paras) + f"\n\n---\nCâu hỏi: Theo tài liệu trên, mã mở khoá của {thing} là bao nhiêu? Chỉ trả lời con số."
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    cases.append((code, thing, prompt, pos / len(paras)))
E.log(f"kim: {len(cases)} ca · ~{len(tok.encode(cases[0][2]))} token")

nres = res.get("kim", {})
for v, (Sm, Sz) in variants.items():
    if a.kim_vars and v not in a.kim_vars.split(","):
        continue
    mx.clear_cache()
    ok, outs = 0, []
    for code, thing, prompt, frac in cases:
        setv(Sm, Sz)
        cache = model.make_cache()
        for i in list(Sm) + list(Sz):
            cache[i] = E.SSMCache()
        t0 = time.time()
        ans = generate(model, tok, prompt, max_tokens=12, verbose=False, prompt_cache=cache)
        hit = code in ans; ok += hit
        outs.append({"ma": code, "vi_tri": round(frac, 2), "tra_loi": ans.strip(), "dung": hit, "s": round(time.time() - t0, 1)})
    nres[v] = {"dung": ok, "tong": len(cases), "ca": outs}
    E.log(f"kim {v:10s} {ok}/{len(cases)} · " + " | ".join(o["tra_loi"][:14] for o in outs))
    res["kim"] = nres
    json.dump(res, open(out_p, "w"), ensure_ascii=False, indent=1)
setv()
E.log("XONG global")
