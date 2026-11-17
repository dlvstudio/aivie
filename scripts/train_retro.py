"""§TK_074 — TRAIN mini-RETRO (retrieval-augmented) vs BASELINE (no-retrieval).

So sánh cốt lõi: retrieval-TRONG-TRAINING (chunked cross-attention) có hơn model
thường cùng kiến trúc không? Cùng data, cùng size, chỉ khác --retrieval.

    .venv/bin/python scripts/train_retro.py --retrieval --steps 6000 --out checkpoints/retro_ret.pt
    .venv/bin/python scripts/train_retro.py            --steps 6000 --out checkpoints/retro_base.pt
"""
import time, argparse
import numpy as np, torch, torch.nn.functional as F
from retro_pytorch import RETRO

T0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="checkpoints/retro_data.npz")
    ap.add_argument("--retrieval", action="store_true", help="bật chunked-cross-attention (RETRO); tắt = baseline")
    ap.add_argument("--dec-dim", type=int, default=384)
    ap.add_argument("--dec-depth", type=int, default=6)
    ap.add_argument("--enc-dim", type=int, default=256)
    ap.add_argument("--enc-depth", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", default="checkpoints/retro_model.pt")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    d = np.load(a.data)
    S = torch.tensor(d["seqs"], dtype=torch.long)          # (N,128)
    R = torch.tensor(d["retrieved"], dtype=torch.long)     # (N,2,r,128)
    V = int(S.max().item()) + 1; V = max(V, 32002)
    N = len(S); n_ho = max(400, N // 20)
    idx = np.random.default_rng(0).permutation(N)
    ho, tr = idx[:n_ho], idx[n_ho:]
    print(f"[{time.time()-T0:.0f}s] dev={dev} N={N} train={len(tr)} ho={n_ho} retrieval={a.retrieval}", flush=True)

    retro = RETRO(num_tokens=V, max_seq_len=256, chunk_size=64,
                  enc_dim=a.enc_dim, enc_depth=a.enc_depth,
                  dec_dim=a.dec_dim, dec_depth=a.dec_depth,
                  dec_cross_attn_layers=(2, 5) if a.dec_depth >= 6 else (2,),
                  heads=8, dim_head=48, pad_id=0).to(dev)
    npar = sum(p.numel() for p in retro.parameters())
    opt = torch.optim.AdamW(retro.parameters(), lr=a.lr)
    print(f"[{time.time()-T0:.0f}s] RETRO {npar/1e6:.1f}M tham số", flush=True)

    def loss_on(ids):
        seq = S[ids].to(dev)                                # (b,128)
        ret = R[ids].to(dev) if a.retrieval else None       # (b,2,r,128) | None
        logits = retro(seq, ret) if a.retrieval else retro(seq)   # (b,128,V)
        return F.cross_entropy(logits[:, :-1].reshape(-1, V), seq[:, 1:].reshape(-1),
                               ignore_index=0)

    @torch.no_grad()
    def ho_bits(cap=400):
        retro.eval(); tot = 0.0; c = 0
        for i in range(0, min(len(ho), cap), a.bs):
            l = loss_on(ho[i:i + a.bs]); tot += float(l); c += 1
        retro.train()
        return tot / max(c, 1) / np.log(2)

    print(f"[{time.time()-T0:.0f}s] bits TRƯỚC: {ho_bits():.3f}", flush=True)
    rng = np.random.default_rng(1)
    for step in range(1, a.steps + 1):
        ids = tr[rng.integers(0, len(tr), a.bs)]
        loss = loss_on(ids)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(retro.parameters(), 1.0)
        if a.warmup > 0 and step <= a.warmup:
            for g in opt.param_groups:
                g["lr"] = a.lr * step / a.warmup
        opt.step()
        if step % a.eval_every == 0:
            print(f"[{time.time()-T0:.0f}s] step {step} loss {loss.item():.3f} | bits {ho_bits():.3f}", flush=True)
            torch.save({"cfg": vars(a), "state": retro.state_dict()}, a.out)
    torch.save({"cfg": vars(a), "state": retro.state_dict()}, a.out)
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out}", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
