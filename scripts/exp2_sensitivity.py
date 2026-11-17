"""Quét độ nhạy cả 40 tầng cửa sổ trượt của Gemma 4 12B: thay TỪNG tầng (bỏ ↔ Mamba P1) → KL đầu ra so với gốc.

  python scripts/exp2_sensitivity.py --runs chat8,sg1,sg2,sg3,sg4 [--greedy 12]
Mamba của mỗi tầng lấy từ run tương ứng (train_log.json ghi danh sách tầng).
--greedy K: sau khi quét, thêm dần tầng dễ nhất (theo KL đơn lẻ) và đo KL tổ hợp, tới K tầng.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import exp2_mamba_layer as E

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="chat8,sg1,sg2,sg3,sg4")
ap.add_argument("--greedy", type=int, default=16)
ap.add_argument("--out", default="sensitivity")
a = ap.parse_args()

model, tok = E.load_model(); lm = E.lm_inner(model)
sliding = [i for i, l in enumerate(lm.layers) if l.self_attn.is_sliding]
mix = {}
for r in a.runs.split(","):
    d = os.path.join(E.OUT_BASE, r)
    lay = json.load(open(os.path.join(d, "train_log.json")))["layers"]
    bag = E.Bag([E.Mamba2Mixer(lm.layers[i].self_attn) for i in lay]); bag.load_weights(os.path.join(d, "mixers.safetensors"))
    for j, i in enumerate(lay):
        mix[i] = bag.mixers[j]
E.log(f"{len(sliding)} tầng sliding · có Mamba cho {len(mix)} tầng")
orig = {i: lm.layers[i].self_attn for i in sliding}
mt = os.path.join(E.ROOT, "data/multiturn/train.jsonl")
seqs = E.pack(tok, E.jsonl_chats(tok, mt, 400, from_end=True), 2048, 2) + \
       E.pack(tok, E.jsonl_chats(tok, os.path.join(E.ROOT, "data/code_sft/val.jsonl"), 100), 2048, 1)
toks = [mx.array(s)[None] for s in seqs]

def reset():
    for i in sliding:
        lm.layers[i].self_attn = orig[i]

base = []
for t in toks:
    reset(); lo = model(t); mx.eval(lo); base.append(lo)

def measure(setter):
    kl = ag = 0.0
    for t, lo in zip(toks, base):
        reset(); setter()
        ls = model(t); mx.eval(ls)
        k, g, _, _ = E.kl_stats(lo, ls, t); kl += k; ag += g
    reset()
    return kl / len(toks), ag / len(toks)

od = os.path.join(E.OUT_BASE, a.out); os.makedirs(od, exist_ok=True)
res = {}
for i in sliding:
    r = {}
    r["bo_KL"], r["bo_dongy"] = measure(lambda: setattr(lm.layers[i], "self_attn", E.Swap(None, zero=True)))
    if i in mix:
        r["mamba_KL"], r["mamba_dongy"] = measure(lambda: setattr(lm.layers[i], "self_attn", E.Swap(mix[i])))
    res[i] = r
    E.log(f"tầng {i:2d} · bỏ KL {r['bo_KL']:.3f} ({r['bo_dongy']*100:.1f}%) · Mamba KL {r.get('mamba_KL', float('nan')):.3f} ({r.get('mamba_dongy', float('nan'))*100:.1f}%)")
    json.dump(res, open(os.path.join(od, "per_layer.json"), "w"), indent=1)

# tổ hợp tham lam: thêm dần tầng có KL Mamba đơn lẻ thấp nhất
order = sorted([i for i in res if "mamba_KL" in res[i]], key=lambda i: res[i]["mamba_KL"])
greedy = []
for k in range(1, min(a.greedy, len(order)) + 1):
    S = order[:k]
    def setS():
        for i in S:
            lm.layers[i].self_attn = E.Swap(mix[i])
    kl, ag = measure(setS)
    greedy.append({"k": k, "tang": S, "KL": kl, "dongy": ag})
    E.log(f"tham lam {k:2d} tầng {S} · KL {kl:.3f} ({ag*100:.1f}%)")
    json.dump({"per_layer": res, "greedy": greedy}, open(os.path.join(od, "sensitivity.json"), "w"), indent=1)
E.log("XONG quét")
