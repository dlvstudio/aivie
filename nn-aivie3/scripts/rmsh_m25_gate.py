"""nn-aivie3 · M2.5 gate — CHÚ-Ý HỌC-ĐƯỢC bằng CỔNG salience per-patch (from-scratch).

Thay resampler-nén (kẹt bootstrapping) bằng **gate g_p∈[0,1] mỗi patch** (GIỮ vị-trí →
kế-thừa tính học-được của ca flatten). gated = grid·g → flatten → head. Phạt thưa L1·mean(g)
= "ngân-sách token": model bị ép chỉ BẬT vài patch quan-trọng.
  • uniform : g phụ-thuộc grid THÔI (không biết mục-tiêu c lúc chạy).
  • topdown : g phụ-thuộc grid + target-emb(c) → BẬT đúng patch màu-c, TẮT nhiễu.
Kỳ-vọng: cùng phạt thưa, topdown đạt acc CAO với mean-gate THẤP (ít token=nhanh) và
gate TẬP-TRUNG vào patch mục-tiêu; uniform phải bật nhiều mới bằng, hoặc thua.

Curriculum nhiễu 0→10 + cosine-LR (đã chứng-minh phá rào from-scratch).
CHẠY:  python nn-aivie3/scripts/rmsh_m25_gate.py
"""
import os, sys, math, time
import torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m25_salience as M                                           # noqa: E402
from nn_aivie3.vision import VisionEncoder                             # noqa: E402
import rmsh_m25_curriculum as C                                        # tái-dùng n_dist_at / lr_at
dev = M.dev_()
NP = M.G * M.G                                                          # 36 patch


class GateNet(nn.Module):
    def __init__(s, cond, d=128, el=2):
        super().__init__()
        s.cond = cond
        s.enc = VisionEncoder(d_v=d, patch=M.CELL, img=M.IMG, heads=4, layers=el)
        s.tgt = nn.Embedding(M.N_COLORS, d); nn.init.normal_(s.tgt.weight, std=0.2)
        gin = d * 2 if cond else d
        s.gate = nn.Sequential(nn.Linear(gin, d), nn.GELU(), nn.Linear(d, 1))
        s.head = nn.Sequential(nn.Linear(d * NP + d, d), nn.GELU(), nn.Linear(d, 4))

    def forward(s, img, c):
        grid = s.enc(img)                                  # (b,NP,d)
        cvec = s.tgt(c)                                     # (b,d)
        gin = torch.cat([grid, cvec[:, None].expand(-1, NP, -1)], -1) if s.cond else grid
        g = torch.sigmoid(s.gate(gin))                     # (b,NP,1) — bản-đồ chú-ý
        if getattr(s, "serve_topk", None):                 # SERVE: chỉ giữ top-k patch salient nhất
            k = s.serve_topk
            thr = g.topk(k, dim=1).values[:, -1:, :]
            g = g * (g >= thr).float()
        feat = torch.cat([(grid * g).flatten(1), cvec], -1)
        return s.head(feat), g.squeeze(-1)


def train(net, steps, lam, dev, bs=128, lr=1.5e-3, tag=""):
    opt = torch.optim.AdamW(net.parameters(), lr=lr); gen = torch.Generator().manual_seed(1)
    net.train(); t0 = time.time()
    for it in range(steps):
        M.N_DIST = C.n_dist_at(it, steps)
        for gp in opt.param_groups:
            gp["lr"] = C.lr_at(it, steps, lr)
        img, c, quad = M.render(bs, gen, dev)
        logit, g = net(img, c)
        loss = F.cross_entropy(logit, quad) + lam * g.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"    [{tag}] it {it:>4} nd={M.N_DIST} loss {loss.item():.3f} "
                  f"gate {g.mean().item():.3f} · {time.time()-t0:.0f}s", flush=True)
    return net


@torch.no_grad()
def ev(net, dev, nd=10, n=512):
    M.N_DIST = nd; net.eval(); g2 = torch.Generator().manual_seed(999)
    # dựng batch kèm biết ô mục-tiêu để đo gate-tập-trung
    import rmsh_m25_salience as MM
    perm = torch.argsort(torch.rand(n, NP, generator=g2), dim=1)
    occ = perm[:, :1 + nd]; tgt_cell = occ[:, 0]; dist = occ[:, 1:]
    c = torch.randint(0, M.N_COLORS, (n,), generator=g2)
    dc = torch.randint(0, M.N_COLORS - 1, (n, nd), generator=g2); dc = dc + (dc >= c[:, None]).long()
    grid = torch.full((n, NP), -1, dtype=torch.long)
    grid.scatter_(1, tgt_cell[:, None], c[:, None]); grid.scatter_(1, dist, dc)
    pal = torch.cat([torch.zeros(1, 3), MM._PAL], 0)
    rgb = pal[grid + 1].view(n, M.G, M.G, 3).permute(0, 3, 1, 2)
    img = (rgb.repeat_interleave(M.CELL, 2).repeat_interleave(M.CELL, 3)
           + 0.03 * torch.randn(n, 3, M.IMG, M.IMG, generator=g2)).to(dev)
    quad = ((tgt_cell // M.G >= M.G // 2).long() * 2 + (tgt_cell % M.G >= M.G // 2).long()).to(dev)
    logit, g = net(img, c.to(dev))
    acc = logit.argmax(-1).eq(quad).float().mean().item()
    gate_tgt = g[torch.arange(n), tgt_cell.to(dev)].mean().item()      # gate TRÊN patch mục-tiêu
    return acc, g.mean().item(), gate_tgt


def main():
    STEPS, LAM = 2500, 0.03
    print(f"=== M2.5 gate · dev={dev} · curriculum nhiễu 0→10 · L1={LAM} · ngẫu-nhiên 25% ===\n")
    out = {}
    for cond in (False, True):
        tag = "topdown" if cond else "uniform"
        print(f"{tag}:")
        torch.manual_seed(0)
        net = train(GateNet(cond).to(dev), STEPS, LAM, dev, tag=tag)
        out[cond] = ev(net, dev, 10)
        print(f"  → acc@10 {out[cond][0]:.1%} · mean-gate {out[cond][1]:.3f} "
              f"(ngân-sách) · gate-trên-mục-tiêu {out[cond][2]:.3f}\n")
    au, at = out[False], out[True]
    print(f"KẾT · uniform: acc {au[0]:.1%} gate {au[1]:.3f} | topdown: acc {at[0]:.1%} gate {at[1]:.3f}")
    if at[0] - au[0] > 0.08 or (at[0] > 0.6 and at[1] < au[1] - 0.05):
        print("⇒ TOP-DOWN salience THẮNG: đúng hơn và/hoặc ít token hơn — 'biết nhìn cái quan-trọng'.")
        print(f"  (gate top-down tập-trung mục-tiêu {at[2]:.2f} vs nền {at[1]:.2f} ⇒ bản-đồ chú-ý có nghĩa)")
    else:
        print("⇒ chưa tách rõ — chỉnh L1/steps.")


if __name__ == "__main__":
    main()
