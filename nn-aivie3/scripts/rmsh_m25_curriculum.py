"""nn-aivie3 · M2.5 curriculum — DẠY thị-giác from-scratch định-vị-giữa-nhiễu bằng
curriculum (nhiễu 0→10 tăng dần) + cosine-LR + nhiều bước. GIỮ cam-kết from-scratch.

Chiến-lược 2 nấc (tránh dead-end):
  (G) GATE: ca tốt-nhất (flatten toàn grid, KHÔNG bottleneck) có học nổi nhiễu=10
      nhờ curriculum không? Nếu KHÔNG → bài/ngân-sách còn quá khó, dừng báo.
  (AB) nếu G đạt → train resampler UNIFORM vs TOPDOWN (m=4) cùng curriculum → đo A/B
       salience (top-down có định-vị tốt hơn ở bottleneck hẹp không).

CHẠY:  python nn-aivie3/scripts/rmsh_m25_curriculum.py
"""
import os, sys, math, time
import torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m25_salience as M                                           # noqa: E402
from nn_aivie3.vision import VisionEncoder                             # noqa: E402
dev = M.dev_()
MAXD = 10


class Best(nn.Module):
    def __init__(s, d=128, el=2):
        super().__init__()
        s.enc = VisionEncoder(d_v=d, patch=M.CELL, img=M.IMG, heads=4, layers=el)
        s.tgt = nn.Embedding(M.N_COLORS, d); nn.init.normal_(s.tgt.weight, std=0.2)
        s.head = nn.Sequential(nn.Linear(d * M.G * M.G + d, d), nn.GELU(), nn.Linear(d, 4))
    def forward(s, img, c):
        return s.head(torch.cat([s.enc(img).flatten(1), s.tgt(c)], -1))


def n_dist_at(it, steps, ramp=0.6):
    return min(MAXD, int(MAXD * it / (steps * ramp)) + 0) if it < steps * ramp else MAXD


def lr_at(it, steps, lr, warm=0.05):
    if it < steps * warm:
        return lr * it / (steps * warm)
    p = (it - steps * warm) / (steps * (1 - warm))
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train(net, steps, dev, bs=128, lr=1.5e-3, tag=""):
    opt = torch.optim.AdamW(net.parameters(), lr=lr); gen = torch.Generator().manual_seed(1)
    net.train(); t0 = time.time()
    for it in range(steps):
        M.N_DIST = n_dist_at(it, steps)
        for g in opt.param_groups:
            g["lr"] = lr_at(it, steps, lr)
        img, c, quad = M.render(bs, gen, dev)
        loss = F.cross_entropy(net(img, c), quad)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 400 == 0:
            print(f"    [{tag}] it {it:>4} nd={M.N_DIST} lr={opt.param_groups[0]['lr']:.1e} "
                  f"loss {loss.item():.3f} · {time.time()-t0:.0f}s", flush=True)
    return net


@torch.no_grad()
def ev(net, dev, nd=MAXD, n=512):
    M.N_DIST = nd; net.eval(); g2 = torch.Generator().manual_seed(999)
    img, c, quad = M.render(n, g2, dev)
    return net(img, c).argmax(-1).eq(quad).float().mean().item()


def main():
    STEPS = 2500
    print(f"=== M2.5 curriculum · dev={dev} · nhiễu 0→{MAXD} · {STEPS} bước · ngẫu-nhiên 25% ===\n")

    print("G) GATE — ca tốt-nhất (flatten) + curriculum:")
    torch.manual_seed(0)
    best = train(Best().to(dev), STEPS, dev, tag="best")
    ag = ev(best, dev, MAXD)
    print(f"  → best-case acc @nhiễu=10: {ag:.1%}  {'ĐẠT (task học được from-scratch)' if ag > 0.6 else 'CHƯA ĐẠT'}\n")
    if ag <= 0.6:
        print("Dừng: bài-nền chưa vượt ngưỡng — cần thêm bước/curriculum khác hoặc warm-start.")
        return

    print("AB) A/B salience — resampler UNIFORM vs TOPDOWN (m=4) + curriculum:")
    res = {}
    for cond in (False, True):
        torch.manual_seed(0)
        net = train(M.SalienceNet(m=4, cond=cond, enc_layers=2).to(dev), STEPS, dev,
                    tag=("topdown" if cond else "uniform"))
        res[cond] = ev(net, dev, MAXD)
    print(f"\n  uniform @10: {res[False]:.1%} · topdown @10: {res[True]:.1%} · "
          f"lợi top-down: {res[True]-res[False]:+.1%}")
    print("  ⇒ " + ("TOP-DOWN salience trội ở bottleneck hẹp (biết nhìn cái quan-trọng)."
                     if res[True] - res[False] > 0.08 else
                     "chưa tách rõ ở m=4 — thử siết m nhỏ hơn / nhiễu cao hơn."))


if __name__ == "__main__":
    main()
