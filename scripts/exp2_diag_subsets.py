"""Chẩn đoán thí nghiệm 8 tầng: KL khi thay từng TẬP CON tầng (Mamba P1 của run chat8)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import exp2_mamba_layer as E
run = sys.argv[1] if len(sys.argv) > 1 else "chat8"
layers = [1, 3, 7, 9, 14, 20, 26, 30]
model, tok = E.load_model(); lm = E.lm_inner(model)
orig = {i: lm.layers[i].self_attn for i in layers}
bag = E.Bag([E.Mamba2Mixer(orig[i]) for i in layers]); bag.load_weights(os.path.join(E.OUT_BASE, run, "mixers.safetensors"))
mt = os.path.join(E.ROOT, "data/multiturn/train.jsonl")
seqs = E.pack(tok, E.jsonl_chats(tok, mt, 400, from_end=True), 2048, 2)
subsets = [[i] for i in layers] + [[1, 3], [1, 3, 7], [1, 3, 7, 9], [1, 3, 7, 9, 14], [14, 20, 26, 30], [1, 14, 30], [20, 26, 30], layers]
res = []
for sub in subsets:
    kl = ag = 0.0
    for sq in seqs:
        t = mx.array(sq)[None]
        for i in layers: lm.layers[i].self_attn = orig[i]
        lo = model(t); mx.eval(lo)
        for j, i in enumerate(layers):
            if i in sub: lm.layers[i].self_attn = E.Swap(bag.mixers[j])
        ls = model(t); mx.eval(ls)
        k, a_, _, _ = E.kl_stats(lo, ls, t); kl += k; ag += a_
    r = {"tang": sub, "KL": kl / len(seqs), "dong_y": ag / len(seqs)}
    res.append(r); E.log(f"{str(sub):28s} KL {r['KL']:.3f} · đồng-ý {r['dong_y']*100:.1f}%")
for i in layers: lm.layers[i].self_attn = orig[i]
json.dump(res, open(os.path.join(E.OUT_BASE, run, "diag_subsets.json"), "w"))
