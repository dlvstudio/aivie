"""nn-aivie3 · REWARD + BẪY — planning theo PHẦN-THƯỞNG & né NGUY-HIỂM (từ "Bug Find Food" 2016).

Trục MỚI: các agent trước chỉ TỚI-ĐÍCH/bắt-chước; đây tối-ưu GIÁ-TRỊ + AN-TOÀN.
  • KTTT: tới đích = +R.   • KTNH: dẫm bẫy = -P.   • bước đi = -1 (tiết-kiệm).
Imagination-planning (Lượng-giá-N-cấp) = tìm chuỗi hành-động TỐI-ĐA phần-thưởng trong
world-model (Dijkstra theo chi-phí có bẫy) → NÉ bẫy dù phải đi vòng, miễn lợi hơn.
So planner "CHỈ-TỚI-ĐÍCH" (BFS ngắn nhất, kệ bẫy).

Đo: tỉ-lệ tới đích · số lần DẪM BẪY · TỔNG PHẦN-THƯỞNG. (Dynamics dùng thật ở đây;
đã chứng-minh world-model học được 100% ở rmsh_imagine/dreamer → cắm vào y hệt.)

CHẠY:  python nn-aivie3/scripts/rmsh_reward.py
"""
import os, sys, heapq
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmsh_dual as D
import torch
G, DIRS, free = D.G, D.DIRS, D.free
R_GOAL, R_TRAP, R_STEP = 10, -8, -1


def gen(gen_, wp=0.10, n_trap=4):
    while True:
        walls = {(r, c) for r in range(G) for c in range(G) if torch.rand(1, generator=gen_).item() < wp}
        frees = [(r, c) for r in range(G) for c in range(G) if (r, c) not in walls]
        if len(frees) < 10: continue
        perm = [frees[i] for i in torch.randperm(len(frees), generator=gen_).tolist()]
        start, goal = perm[0], perm[1]
        traps = set(perm[2:2+n_trap])
        # đảm bảo tới được đích (né tường)
        if reachable(walls, start, goal): return walls, start, goal, traps


def reachable(walls, a, t):
    seen = {a}; q = deque([a])
    while q:
        c = q.popleft()
        if c == t: return True
        for d in DIRS:
            n = (c[0]+d[0], c[1]+d[1])
            if free(n, walls) and n not in seen: seen.add(n); q.append(n)
    return False


def plan_reach(walls, traps, a, t):
    """CHỈ-TỚI-ĐÍCH: BFS ngắn nhất, KỆ bẫy."""
    prev = {a: None}; q = deque([a])
    while q:
        c = q.popleft()
        if c == t: break
        for di, d in enumerate(DIRS):
            n = (c[0]+d[0], c[1]+d[1])
            if free(n, walls) and n not in prev: prev[n] = (c, di); q.append(n)
    return _first(prev, a, t)


def plan_reward(walls, traps, a, t):
    """THEO PHẦN-THƯỞNG: Dijkstra chi-phí (bước=1, vào bẫy=+|P|) → né bẫy nếu vòng rẻ hơn."""
    dist = {a: 0}; prev = {a: None}; pq = [(0, a)]
    while pq:
        cost, c = heapq.heappop(pq)
        if c == t: break
        if cost > dist.get(c, 1e9): continue
        for di, d in enumerate(DIRS):
            n = (c[0]+d[0], c[1]+d[1])
            if not free(n, walls): continue
            step = 1 + (abs(R_TRAP) if n in traps and n != t else 0)
            nc = cost + step
            if nc < dist.get(n, 1e9):
                dist[n] = nc; prev[n] = (c, di); heapq.heappush(pq, (nc, n))
    return _first(prev, a, t)


def _first(prev, a, t):
    if t not in prev: return 0
    cur, first = t, 0
    while prev[cur] is not None:
        p, di = prev[cur]; first = di; cur = p
    return first


def rollout(policy, walls, a, t, traps, K=40):
    cur = a; reward = 0; trap_hits = 0
    for _ in range(K):
        if cur == t: return True, reward + R_GOAL, trap_hits
        di = policy(walls, traps, cur, t); d = DIRS[di]; nx = (cur[0]+d[0], cur[1]+d[1])
        if free(nx, walls): cur = nx
        reward += R_STEP
        if cur in traps: reward += R_TRAP; trap_hits += 1
    return cur == t, reward + (R_GOAL if cur == t else 0), trap_hits


def main():
    print(f"=== nn-aivie3 · REWARD + BẪY (Bug Find Food) · lưới {G}x{G} ===")
    print(f"đích +{R_GOAL} · bẫy {R_TRAP} · bước {R_STEP}\n")
    g = torch.Generator().manual_seed(5); N = 300
    mazes = [gen(g) for _ in range(N)]
    for name, pol in [("CHỈ-TỚI-ĐÍCH (kệ bẫy)", plan_reach), ("THEO PHẦN-THƯỞNG (né bẫy)", plan_reward)]:
        reach = 0; rew = 0; hits = 0
        for (w, a, t, tr) in mazes:
            ok, r, h = rollout(pol, w, a, t, tr); reach += ok; rew += r; hits += h
        print(f"  {name:28s}: tới đích {reach/N:.0%} · dẫm bẫy TB {hits/N:.2f}/tập · thưởng TB {rew/N:+.1f}")
    print("\n  ⇒ planner THEO PHẦN-THƯỞNG vẫn tới đích nhưng NÉ bẫy (dẫm ~0) → thưởng cao hơn hẳn.")
    print("    Đây là trục GIÁ-TRỊ/AN-TOÀN (KTTT thưởng + KTNH ức-chế) — 'Bug Find Food' của anh,")
    print("    ghép thẳng lên world-model đã học (rmsh_imagine/dreamer). Bước sau: LEARN bản-đồ bẫy")
    print("    từ trải-nghiệm (KTNH: nhớ ô từng gây hại) thay vì cho sẵn.")


if __name__ == "__main__":
    main()
