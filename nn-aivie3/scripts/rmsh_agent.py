"""nn-aivie3 · AGENT TỰ-HỌC TÍCH-HỢP (capstone) — "Bug Find Food" đầy-đủ, từ số 0.

Thả agent vào thế-giới LẠ: KHÔNG biết đích ở đâu, KHÔNG biết bẫy ở đâu. Nó tự học tất-cả:
  • World-model (vật-lý): (ô,tường,hành-động)→ô-kế — LUYỆN TRƯỚC theo CURRICULUM (dễ→khó)
    = "vật-lý bẩm-sinh"; rồi biết động-lực mê-cung mới ngay.
  • Curiosity: thưởng nội-sinh cho ô LẠ (ít thăm) → tự đi khám-phá để TÌM đích.
  • Học thưởng/nguy từ TRẢI-NGHIỆM (KTTT/KTNH): chạm đích→nhớ +thưởng; dẫm bẫy→ĐAU→nhớ ô nguy.
  • Imagination-planning (value-iteration TRONG world-model): tối-đa [thưởng-đã-học + curiosity]
    → sớm khám-phá, muộn đi thẳng đích & né bẫy đã biết. Re-plan mỗi bước.

Không ai dạy đích/bẫy ở đâu. Đo đường-cong TỰ-HỌC qua các tập: thưởng↑ · dẫm-bẫy↓ · tới-đích↑,
vượt agent đi bừa.

CHẠY:  python nn-aivie3/scripts/rmsh_agent.py
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmsh_dual as D
G, DIRS, free = D.G, D.DIRS, D.free
GG = G * G
dev = "mps" if torch.backends.mps.is_available() else "cpu"
R_GOAL, R_TRAP, R_STEP, GAMMA, BETA = 10.0, -8.0, -1.0, 0.95, 2.0


def idx(c): return c[0]*G + c[1]
def wvec(walls):
    v = torch.zeros(GG)
    for (r, c) in walls: v[r*G+c] = 1.0
    return v
def truenext(c, a, walls):
    d = DIRS[a]; n = (c[0]+d[0], c[1]+d[1]); return n if free(n, walls) else c


class WorldModel(nn.Module):
    def __init__(s, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2*GG+4, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, GG))
    def forward(s, c, wv, a):
        return s.net(torch.cat([F.one_hot(c, GG).float(), wv, F.one_hot(a, 4).float()], -1))


def pretrain_wm(wm, gen):
    """CURRICULUM: luyện vật-lý (ô,tường,hành-động)→ô-kế trên mê-cung tường 0→0.20."""
    opt = torch.optim.AdamW(wm.parameters(), lr=2e-3); wm.train()
    for it in range(1200):
        wp = 0.20 * min(1.0, it/700)                                  # curriculum dễ→khó
        cs, wv, as_, nx = [], [], [], []
        for _ in range(128):
            walls, frees = _maze(gen, wp)
            cell = frees[int(torch.randint(len(frees), (1,), generator=gen))]
            a = int(torch.randint(4, (1,), generator=gen))
            cs.append(idx(cell)); wv.append(wvec(walls)); as_.append(a); nx.append(idx(truenext(cell, a, walls)))
        loss = F.cross_entropy(wm(torch.tensor(cs, device=dev), torch.stack(wv).to(dev), torch.tensor(as_, device=dev)),
                               torch.tensor(nx, device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
    wm.eval()


def _maze(gen, wp):
    while True:
        walls = {(r, c) for r in range(G) for c in range(G) if torch.rand(1, generator=gen).item() < wp}
        frees = [(r, c) for r in range(G) for c in range(G) if (r, c) not in walls]
        if len(frees) > 0.6*GG: return walls, frees


@torch.no_grad()
def transitions(wm, wv):
    ag = torch.arange(GG, device=dev).repeat_interleave(4); ac = torch.arange(4, device=dev).repeat(GG)
    return wm(ag, wv[None].expand(GG*4, -1).to(dev), ac).argmax(-1).view(GG, 4).cpu()


def plan(nxt, cur, known_goal, known_traps, N):
    """imagination value-iteration: tối-đa [thưởng-đã-học + curiosity] trong world-model."""
    def rew(ci):
        cell = (ci//G, ci % G)
        if known_goal is not None and cell == known_goal: return R_GOAL
        if cell in known_traps: return R_TRAP
        return R_STEP + BETA/((N.get(cell, 0)+1)**0.5)                # curiosity: ô lạ hấp-dẫn hơn
    goal_i = idx(known_goal) if known_goal is not None else -1
    V = [0.0]*GG
    for _ in range(50):
        nV = [0.0]*GG
        for ci in range(GG):
            if ci == goal_i: nV[ci] = R_GOAL; continue
            nV[ci] = max(rew(int(nxt[ci, a])) + GAMMA*V[int(nxt[ci, a])] for a in range(4))
        V = nV
    ci = idx(cur); best, ba = -1e9, 0
    for a in range(4):
        ni = int(nxt[ci, a]); q = rew(ni) + GAMMA*V[ni]
        if q > best: best, ba = q, a
    return ba


def run_agent(wm, walls, start, goal, traps, episodes, K=40):
    wv = wvec(walls); known_goal = None; known_traps = set(); N = {}
    log = []
    for ep in range(episodes):
        cur = start; rew = 0; hits = 0; reached = False
        nxt = transitions(wm, wv)                                     # world-model của mê-cung này
        for step in range(K):
            N[cur] = N.get(cur, 0) + 1
            a = plan(nxt, cur, known_goal, known_traps, N)
            cur = truenext(cur, a, walls)                            # hành-động trong env thật
            if cur == goal:
                known_goal = goal; rew += R_GOAL; reached = True; break   # KTTT: nhớ đích
            elif cur in traps:
                known_traps.add(cur); rew += R_TRAP; hits += 1; rew += R_STEP   # KTNH: nhớ ô nguy + đau
            else:
                rew += R_STEP
        log.append((rew, hits, reached))
    return log


def run_random(walls, start, goal, traps, episodes, gen, K=40):
    log = []
    for ep in range(episodes):
        cur = start; rew = 0; hits = 0; reached = False
        for _ in range(K):
            a = int(torch.randint(4, (1,), generator=gen)); cur = truenext(cur, a, walls)
            if cur == goal: rew += R_GOAL; reached = True; break
            elif cur in traps: rew += R_TRAP + R_STEP; hits += 1
            else: rew += R_STEP
        log.append((rew, hits, reached))
    return log


def avg(log, i): return sum(x[i] for x in log)/len(log)


def main():
    print(f"=== nn-aivie3 · AGENT TỰ-HỌC TÍCH-HỢP (capstone) · lưới {G}x{G} · dev={dev} ===\n")
    t0 = time.time()
    torch.manual_seed(0); gen = torch.Generator().manual_seed(1)
    wm = WorldModel().to(dev)
    print("① CURRICULUM: luyện 'vật-lý bẩm-sinh' (world-model, tường 0→0.20)…", flush=True)
    pretrain_wm(wm, gen); print(f"   xong ({time.time()-t0:.0f}s)\n", flush=True)

    # thế-giới LẠ: đích + bẫy ẨN (agent KHÔNG biết ở đâu)
    g2 = torch.Generator().manual_seed(7); walls, frees = _maze(g2, 0.12)
    perm = [frees[i] for i in torch.randperm(len(frees), generator=g2).tolist()]
    start, goal = perm[0], perm[1]; traps = set(perm[2:6])            # 4 bẫy ẩn
    print(f"② THẢ vào thế-giới lạ: {len(frees)} ô · đích + 4 bẫy ẨN (agent phải TỰ tìm)\n")

    E = 40
    log = run_agent(wm, walls, start, goal, traps, E)
    rnd = run_random(walls, start, goal, traps, E, torch.Generator().manual_seed(9))
    print(f"   (chạy {time.time()-t0:.0f}s)\n")
    print(f"  {'tập':>7} | {'thưởng (agent / bừa)':^22} | {'dẫm bẫy':^16} | {'tới đích':^14}")
    print("  " + "-"*64)
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40)]:
        a = log[lo:hi]; r = rnd[lo:hi]
        print(f"  {lo+1:>2}-{hi:<3} | {avg(a,0):>8.1f} / {avg(r,0):>8.1f}       | "
              f"{avg(a,1):>5.2f} / {avg(r,1):>5.2f}     | {avg(a,2):>4.0%} / {avg(r,2):>4.0%}")
    e = log[-10:]; e0 = log[:10]
    print(f"\n  ⇒ TỰ-HỌC: thưởng {avg(e0,0):+.1f} (tập đầu) → {avg(e,0):+.1f} (tập cuối); "
          f"dẫm bẫy {avg(e0,1):.2f}→{avg(e,1):.2f}; tới đích {avg(e0,2):.0%}→{avg(e,2):.0%}.")
    ok = avg(e, 0) > avg(e0, 0) + 2 and avg(e, 1) < avg(e0, 1) and avg(e, 0) > avg(rnd[-10:], 0) + 5
    print("  " + ("THÀNH ✓ — agent TỰ khám-phá đích, TỰ nhớ bẫy, TỰ lập-kế-hoạch né → cải-thiện qua trải-nghiệm, "
                  "KHÔNG được dạy gì. Đây là 'Bug Find Food' sống: nhìn+mô-hình+tò-mò+thưởng+tưởng-tượng."
                  if ok else "chưa rõ đường-cong — chỉnh curriculum/beta/episodes."))


if __name__ == "__main__":
    main()
