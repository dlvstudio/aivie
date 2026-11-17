"""nn-aivie3 · M3.5(2) — CHỌN-TOKEN thích-nghi bằng GUMBEL-SOFTMAX (giải mảnh khó).

m=4 slot, mỗi slot chọn ~1 patch (near-one-hot, GIỮ vị-trí) → chỉ 4 token/frame mà bám
đúng patch quan-trọng. Train tau 2.0→0.5 (mềm→cứng); EVAL hard=argmax (chọn cứng thật).
Nếu eval-hard đạt acc cao ⇒ chọn-token thích-nghi HỌC ĐƯỢC từ-scratch (khác L1/hard-topk).

CHẠY:  python nn-aivie3/scripts/rmsh_m35_gumbel.py
"""
import os, sys, time
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nn_aivie3.agent import GumbelSelectActionModel                     # noqa: E402

G, CELL, IMG, T = 4, 8, 32, 4
DIRS = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def make(b, gen):
    d = torch.randint(0, 4, (b,), generator=gen); vel = DIRS[d]
    r = torch.full((b,), G // 2).float() - (vel[:, 0] > 0).float()
    c = torch.full((b,), G // 2).float() - (vel[:, 1] > 0).float()
    fr = torch.zeros(b, T, 3, IMG, IMG)
    for t in range(T):
        rr = (r + vel[:, 0] * t).clamp(0, G - 1).long(); cc = (c + vel[:, 1] * t).clamp(0, G - 1).long()
        for i in range(b):
            fr[i, t, :, rr[i] * CELL:rr[i] * CELL + CELL, cc[i] * CELL:cc[i] * CELL + CELL] = 1.0
    fr = fr + 0.03 * torch.randn(b, T, 3, IMG, IMG, generator=gen)
    return fr.to(dev), d.to(dev)


def main():
    print(f"=== M3.5 Gumbel · dev={dev} · lưới {G}x{G} ({(IMG//CELL)**2} patch) · m=4 slot · ngẫu-nhiên 25% ===\n")
    torch.manual_seed(0)
    m = GumbelSelectActionModel(n_actions=4, m=4, d=128, img=IMG, patch=CELL, window=64).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3); gen = torch.Generator().manual_seed(1)
    m.train(); t0 = time.time(); STEPS = 350
    for it in range(STEPS):
        tau = max(0.5, 2.0 * (1 - it / STEPS))                          # anneal 2.0→0.5
        fr, d = make(48, gen)
        logit, w = m(fr, tau=tau, hard=False)
        loss = F.cross_entropy(logit[:, 1:].reshape(-1, 4), d[:, None].expand(-1, T - 1).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == STEPS - 1:
            print(f"    it {it:>4} tau {tau:.2f} loss {loss.item():.3f} · {time.time()-t0:.0f}s", flush=True)

    m.eval(); g2 = torch.Generator().manual_seed(999); fr, d = make(256, g2)
    with torch.no_grad():
        soft = m(fr, tau=0.5, hard=False)[0][:, -1].argmax(-1).eq(d).float().mean().item()
        hard = m(fr, tau=0.5, hard=True)[0][:, -1].argmax(-1).eq(d).float().mean().item()
    print(f"\n  acc hướng — SOFT(train-mode): {soft:.1%} · HARD(argmax-serve): {hard:.1%} "
          f"(chỉ m=4 token/frame, giữ vị-trí)")
    print("  ⇒ " + ("CHỌN-TOKEN thích-nghi HỌC ĐƯỢC từ-scratch (Gumbel giải mảnh khó M3.5)."
                     if hard > 0.6 else "chưa đạt — chỉnh tau/m/steps."))


if __name__ == "__main__":
    main()
