"""nn-aivie3 · WORLD-MODEL tường-minh + IMAGINATION-PLANNING cho MÊ-CUNG.

Giải lại chính bài §9b mà reflex-nguyên-khối FAIL (~8%). Ý: TÁCH planning =
  • World-model học ĐỘNG-LỰC 1-BƯỚC: (ô, tường, hành-động) → ô kế (đi được / đụng tường).
    Đây là bài DỄ (cục-bộ), khác hẳn "đoán cả tuyến trong 1 forward" (khó, reflex fail).
  • IMAGINATION-PLANNING (Hệ 2): BFS TRONG world-model đã học ("tua tưởng-tượng") → tuyến →
    bước đầu. Re-plan mỗi bước (MPC) để bền với sai-số model.

Chứng-minh: lập-kế-hoạch-bằng-tưởng-tượng đạt gần BFS-thật, VƯỢT XA reflex (8%) & tham-lam (60%).

CHẠY:  python nn-aivie3/scripts/rmsh_imagine.py
"""
import os, sys, time
from collections import deque
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmsh_dual as D                                                  # gen_maze, bfs_first_dir, free, DIRS, G, greedy_dir
G, DIRS, free = D.G, D.DIRS, D.free
GG = G * G
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def idx(cell): return cell[0] * G + cell[1]
def walls_vec(walls):
    v = torch.zeros(GG)
    for (r, c) in walls: v[r*G+c] = 1.0
    return v
def true_next(cell, action, walls):
    dr, dc = DIRS[action]; nx = (cell[0]+dr, cell[1]+dc)
    return nx if free(nx, walls) else cell


class WorldModel(nn.Module):
    """(ô hiện-tại, bản-đồ tường, hành-động) → ô KẾ. Học động-lực CỤC-BỘ."""
    def __init__(s, h=256):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2*GG+4, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, GG))
    def forward(s, ag, wv, ac):                                       # ag:(b,) ac:(b,) wv:(b,GG)
        x = torch.cat([F.one_hot(ag, GG).float(), wv, F.one_hot(ac, 4).float()], -1)
        return s.net(x)


@torch.no_grad()
def plan_first_action(wm, wv, a, t):
    """IMAGINATION: dựng chuyển-tiếp bằng world-model → BFS trong đầu → bước đầu."""
    ag = torch.arange(GG, device=dev).repeat_interleave(4)
    ac = torch.arange(4, device=dev).repeat(GG)
    nxt = wm(ag, wv[None].expand(GG*4, -1).to(dev), ac).argmax(-1).view(GG, 4).cpu()   # (GG,4)
    ai, ti = idx(a), idx(t)
    prev = {ai: None}; q = deque([ai])
    while q:
        cur = q.popleft()
        if cur == ti: break
        for action in range(4):
            n = int(nxt[cur, action])
            if n != cur and n not in prev:                           # "đi được" trong tưởng-tượng
                prev[n] = (cur, action); q.append(n)
    if ti not in prev: return 0                                       # world-model bí → bừa
    cur, first = ti, 0
    while prev[cur] is not None:
        p, action = prev[cur]; first = action; cur = p
    return first


def main():
    print(f"=== nn-aivie3 · world-model + imagination-planning · mê-cung {G}x{G} · dev={dev} ===\n")
    torch.manual_seed(0)
    wm = WorldModel().to(dev); opt = torch.optim.AdamW(wm.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(1); WP = 0.15

    # LUYỆN world-model: đoán ô-kế từ (ô, tường, hành-động) — bài cục-bộ DỄ
    wm.train(); t0 = time.time(); BS = 256
    for it in range(1500):
        ags, wvs, acs, nxs = [], [], [], []
        for _ in range(BS):
            walls, _, _ = D.gen_maze(g, WP)
            frees = [(r, c) for r in range(G) for c in range(G) if (r, c) not in walls]
            cell = frees[int(torch.randint(len(frees), (1,), generator=g))]
            action = int(torch.randint(4, (1,), generator=g))
            ags.append(idx(cell)); wvs.append(walls_vec(walls)); acs.append(action)
            nxs.append(idx(true_next(cell, action, walls)))
        ag = torch.tensor(ags, device=dev); wv = torch.stack(wvs).to(dev)
        ac = torch.tensor(acs, device=dev); nx = torch.tensor(nxs, device=dev)
        loss = F.cross_entropy(wm(ag, wv, ac), nx)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 400 == 0 or it == 1499:
            print(f"    luyện WM it {it:>4} loss {loss.item():.4f} · {time.time()-t0:.0f}s", flush=True)

    # độ-chính-xác động-lực 1-bước (held-out)
    wm.eval(); g2 = torch.Generator().manual_seed(999); acc = 0; M = 3000
    for _ in range(M):
        walls, _, _ = D.gen_maze(g2, WP)
        frees = [(r, c) for r in range(G) for c in range(G) if (r, c) not in walls]
        cell = frees[int(torch.randint(len(frees), (1,), generator=g2))]
        action = int(torch.randint(4, (1,), generator=g2))
        p = int(wm(torch.tensor([idx(cell)], device=dev), walls_vec(walls)[None].to(dev),
                   torch.tensor([action], device=dev)).argmax())
        acc += (p == idx(true_next(cell, action, walls)))
    wm_acc = acc / M

    # ĐÁNH-GIÁ vòng-kín: planning-bằng-tưởng-tượng (re-plan mỗi bước) vs baselines
    g3 = torch.Generator().manual_seed(7); N = 200
    mazes = [D.gen_maze(g3, WP) for _ in range(N)]

    def run(policy, K=24):
        s = 0
        for (walls, a, t) in mazes:
            wv = walls_vec(walls); cur = a
            for _ in range(K):
                if cur == t: break
                action = policy(walls, wv, cur, t)
                cur = true_next(cur, action, walls)
            s += (cur == t)
        return s / N

    imag = run(lambda w, wv, c, t: plan_first_action(wm, wv, c, t))
    greedy = run(lambda w, wv, c, t: D.greedy_dir(c, t))
    bfs = run(lambda w, wv, c, t: D.bfs_first_dir(w, c, t))

    print(f"\n  world-model đoán ĐÚNG ô-kế (1 bước): {wm_acc:.0%}")
    print(f"\n  tỉ-lệ TỚI ĐÍCH ({N} mê-cung, tường {WP}):")
    print(f"    reflex nguyên-khối (đã đo §9b)         : ~8%     ← FAIL (đoán cả tuyến 1 forward)")
    print(f"    phản-xạ THAM-LAM                        : {greedy:.0%}")
    print(f"    IMAGINATION-PLANNING (WM + BFS-trong-đầu): {imag:.0%}   ← Hệ 2")
    print(f"    BFS thật (thầy)                         : {bfs:.0%}")
    ok = imag > 0.8
    print(f"\n  ⇒ " + ("THÀNH ✓ — TÁCH 'động-lực cục-bộ (học) + tìm-trong-tưởng-tượng' GIẢI được mê-cung "
                       "mà reflex nguyên-khối FAIL. Đây là Hệ-2 = mô-phỏng trong đầu (Dreamer/MuZero)."
                       if ok else f"imagination {imag:.0%} — WM {wm_acc:.0%}; cần WM chuẩn hơn."))
    print("    Bài học: planning = MÔ-HÌNH cục-bộ dễ-học + TÌM-KIẾM; không phải phản-xạ nguyên-khối.")


if __name__ == "__main__":
    main()
