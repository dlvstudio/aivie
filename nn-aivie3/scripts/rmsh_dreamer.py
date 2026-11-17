"""nn-aivie3 · WORLD-MODEL LATENT GROUNDED TỪ THỊ-GIÁC (Dreamer-lite cho mê-cung).

Nối phần khó: đọc mê-cung từ PIXEL → latent CÓ CẤU-TRÚC → planning-bằng-tưởng-tượng.
  • Encoder: frame (điểm-ảnh) → latent mỗi-ô (ViT nhỏ giữ-vị-trí).
  • Giải-mã latent (grounding, tự-học): TƯỜNG (mỗi ô) · AGENT (ô nào) · ĐÍCH (ô nào).
  • IMAGINATION-PLANNING: BFS trong mê-cung ĐÃ CẢM-NHẬN (từ pixel) → bước đầu; act ở env thật;
    re-cảm-nhận mỗi bước (MPC). = mô-phỏng trong đầu trên thế-giới ĐỌC-ĐƯỢC-TỪ-MẮT.

So reflex nguyên-khối (§9b, 8%). Đo cả độ-chính-xác CẢM-NHẬN (mắt đọc mê-cung tốt không).

CHẠY:  python nn-aivie3/scripts/rmsh_dreamer.py
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import rmsh_dual as D                                                  # gen_maze, draw, free, DIRS, bfs_first_dir, greedy_dir, G, CELL, IMG
from nn_aivie3.vision import VisionEncoder
G, CELL, IMG, GG = D.G, D.CELL, D.IMG, D.G * D.G
free, DIRS = D.free, D.DIRS
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def labels(walls, a, t):
    wl = torch.zeros(GG)
    for (r, c) in walls: wl[r*G+c] = 1.0
    return wl, a[0]*G+a[1], t[0]*G+t[1]


class Perceiver(nn.Module):
    """frame → latent mỗi-ô → giải-mã tường/agent/đích (world-model grounded từ pixel)."""
    def __init__(s, d=128):
        super().__init__()
        s.enc = VisionEncoder(d_v=d, patch=CELL, img=IMG, heads=4, layers=5)
        s.wall = nn.Linear(d, 1); s.agent = nn.Linear(d, 1); s.target = nn.Linear(d, 1)
    def forward(s, frame):
        z = s.enc(frame)                                             # (b, GG, d) latent mỗi-ô
        return s.wall(z).squeeze(-1), s.agent(z).squeeze(-1), s.target(z).squeeze(-1)


def main():
    print(f"=== nn-aivie3 · world-model latent grounded từ thị-giác · mê-cung {G}x{G} · dev={dev} ===\n")
    torch.manual_seed(0)
    m = Perceiver().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3)
    g = torch.Generator().manual_seed(1); WP = 0.15

    # LUYỆN cảm-nhận: đọc tường/agent/đích từ pixel (tự-giám-sát bằng nhãn dựng-hình)
    m.train(); t0 = time.time(); BS = 64
    for it in range(700):
        frames, wls, ais, tis = [], [], [], []
        for _ in range(BS):
            walls, a, t = D.gen_maze(g, WP)
            frames.append(D.draw(walls, a, t)); wl, ai, ti = labels(walls, a, t)
            wls.append(wl); ais.append(ai); tis.append(ti)
        fr = torch.stack(frames).to(dev)
        wl = torch.stack(wls).to(dev); ai = torch.tensor(ais, device=dev); ti = torch.tensor(tis, device=dev)
        pw, pa, pt = m(fr)
        loss = F.binary_cross_entropy_with_logits(pw, wl) + F.cross_entropy(pa, ai) + F.cross_entropy(pt, ti)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0 or it == 699:
            print(f"    luyện it {it:>3} loss {loss.item():.3f} · {time.time()-t0:.0f}s", flush=True)

    # độ-chính-xác CẢM-NHẬN (held-out)
    m.eval(); g2 = torch.Generator().manual_seed(999); N = 200
    mazes = [D.gen_maze(g2, WP) for _ in range(N)]
    wa = aa = ta = 0
    with torch.no_grad():
        for (walls, a, t) in mazes:
            pw, pa, pt = m(D.draw(walls, a, t)[None].to(dev))
            wl, ai, ti = labels(walls, a, t)
            wa += ((pw[0].sigmoid() > 0.5).cpu().float() == wl).float().mean().item()
            aa += (int(pa[0].argmax()) == ai); ta += (int(pt[0].argmax()) == ti)
    print(f"\n  CẢM-NHẬN từ pixel: tường {wa/N:.0%}/ô · agent {aa/N:.0%} · đích {ta/N:.0%}")

    @torch.no_grad()
    def perceive(frame):                                            # pixel → mê-cung latent (đọc-được)
        pw, pa, pt = m(frame[None].to(dev))
        wl = (pw[0].sigmoid() > 0.5).cpu()
        pset = {(i//G, i%G) for i in range(GG) if wl[i]}
        ai = int(pa[0].argmax()); ti = int(pt[0].argmax())
        return pset, (ai//G, ai%G), (ti//G, ti%G)

    # IMAGINATION-PLANNING trên thế-giới CẢM-NHẬN-TỪ-MẮT (re-cảm-nhận mỗi bước)
    def run(policy, K=24):
        s = 0
        for (walls, a, t) in mazes:
            cur = a
            for _ in range(K):
                if cur == t: break
                action = policy(walls, cur, t)
                nx = (cur[0]+DIRS[action][0], cur[1]+DIRS[action][1])
                if free(nx, walls): cur = nx
            s += (cur == t)
        return s / N

    def dreamer_policy(walls, cur, t):
        pw, pa, pt = perceive(D.draw(walls, cur, t))                 # ĐỌC mê-cung từ pixel
        act = D.bfs_first_dir(pw, pa, pt)                           # BFS trong đầu (tưởng-tượng)
        return act if act is not None else D.greedy_dir(cur, t)

    dr = run(dreamer_policy)
    gr = run(lambda w, c, t: D.greedy_dir(c, t))
    bf = run(lambda w, c, t: D.bfs_first_dir(w, c, t))

    print(f"\n  tỉ-lệ TỚI ĐÍCH ({N} mê-cung, tường {WP}):")
    print(f"    reflex nguyên-khối (§9b)                    : ~8%     ← FAIL")
    print(f"    tham-lam                                     : {gr:.0%}")
    print(f"    DREAMER (mắt đọc mê-cung → tưởng-tượng BFS)   : {dr:.0%}   ← world-model latent + Hệ 2")
    print(f"    BFS thật (biết mê-cung sẵn)                   : {bf:.0%}")
    ok = dr > 0.8
    print(f"\n  ⇒ " + ("THÀNH ✓ — world-model GROUNDED TỪ THỊ-GIÁC: mắt đọc mê-cung từ pixel → mô-phỏng "
                       "trong đầu → giải được (gần BFS), vượt xa reflex. Đây là Dreamer trên RMSH."
                       if ok else f"dreamer {dr:.0%} (cảm-nhận tường {wa/N:.0%}) — perception còn sai → planning lệch; cần luyện mắt kỹ hơn."))


if __name__ == "__main__":
    main()
