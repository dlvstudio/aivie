"""nn-aivie3 · MẠNG LẬP-KẾ-HOẠCH (VIN) — giờ HỌC CẢ NÉ BỌ ĐỘNG (2026-09-08).

So với bản trước: thêm CON BỌ ĐỘNG vào thế-giới + KÊNH "bọ" thứ-6 vào input mạng
(trong tầm cảm-nhận BSENSE). Thầy BFS né ô có bọ (+lề an-toàn). ⇒ VIN chưng-cất được
CẢ tìm-food LẪN né-bọ-động trong MỘT mạng → bỏ được vector-đẩy viết-tay ở demo.

Kiến-trúc VIN (Tamar 2016): conv học reward-map R (giờ R âm ở ô tường/nguy/BỌ), lặp
value-iteration K vòng, đọc-ra tại ô robot. Chưng-cất BFS bằng BC + DAgger trên M4.

Input mạng = lưới 20×20 × 6 kênh: [chưa-biết, trống, tường, food, nguy-tĩnh, BỌ].
5 kênh đầu từ bản-đồ đã-lộ B (one-hot theo B); kênh BỌ = ô có bọ trong tầm BSENSE (độc-lập).

CHẠY:  .venv/bin/python nn-aivie3/scripts/rmsh_navpolicy.py
"""
import os, math, time, random, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ---- hằng KHỚP demo ----
G = 20
VIS, FOV, NR = 7.2, 1.15, 72
STEP_T = 0.05
SPD, TURN = 0.075, 0.19
NBUG, BUGSPD, BUGR, HITR = 6, 0.026, 0.42, 0.5
BSENSE = 6.0                              # tầm robot cảm-nhận bọ (đưa vào input mạng)
NCH = 6                                   # 0 chưa-biết,1 trống,2 tường,3 food,4 nguy,5 BỌ
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_CH = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4}
SP = "/private/tmp/claude-501/-Users-user-Documents-learn-nn-aivie/bbbd11fd-4078-4299-874a-72c1068fd8b1/scratchpad/navpolicy.json"

def inb(r, c): return 0 <= r < G and 0 <= c < G

def conn_free(T, sr, sc):
    seen = {sr * G + sc}; q = [sr * G + sc]; n = 0
    while q:
        x = q.pop(); r, c = divmod(x, G); n += 1
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if inb(nr, nc) and T[nr][nc] != 1 and nr * G + nc not in seen:
                seen.add(nr * G + nc); q.append(nr * G + nc)
    return seen, n

def gen():
    while True:
        T = np.zeros((G, G), dtype=np.int8)
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 1
        for r in range(2, G - 2):
            for c in range(2, G - 2):
                if random.random() < 0.14: T[r][c] = 1
        while True:
            sr, sc = random.randrange(G), random.randrange(G)
            if T[sr][sc] == 0: break
        seen, n = conn_free(T, sr, sc)
        if n < 0.5 * G * G: continue
        free = [i for i in seen if math.hypot(i % G - sc, i // G - sr) > 4]
        if len(free) < 20: continue
        random.shuffle(free)
        for _ in range(2):
            i = free.pop(); T[i // G][i % G] = 2
        for _ in range(14):
            if not free: break
            i = free.pop(); T[i // G][i % G] = 3
        ax, ay, ath = sc + 0.5, sr + 0.5, random.random() * 6.28
        B = -np.ones((G, G), dtype=np.int8); B[sr][sc] = T[sr][sc]
        # con bọ động: ô trống, xa robot
        spots = [(r, c) for r in range(1, G - 1) for c in range(1, G - 1)
                 if T[r][c] == 0 and math.hypot(c - ax, r - ay) > 4]
        random.shuffle(spots)
        bugs = [{'x': c + 0.5, 'y': r + 0.5, 'th': random.random() * 6.28, 'cd': 0}
                for r, c in spots[:NBUG]]
        return T, B, ax, ay, ath, bugs

def move_bugs(T, bugs):
    for b in bugs:
        if b['cd'] > 0: b['cd'] -= 1
        b['th'] += (random.random() - 0.5) * 0.5
        nx = b['x'] + math.cos(b['th']) * BUGSPD; ny = b['y'] + math.sin(b['th']) * BUGSPD
        c, r = int(nx), int(ny)
        if inb(r, c) and T[r][c] != 1: b['x'], b['y'] = nx, ny
        else: b['th'] += 2.1

def collide(bugs, ax, ay):               # va bọ → knockback (khớp demo), trả (ax,ay,hit?)
    for b in bugs:
        if math.hypot(b['x'] - ax, b['y'] - ay) < HITR and b['cd'] <= 0:
            b['cd'] = 30
            aw = math.atan2(ay - b['y'], ax - b['x'])
            kx = ax + math.cos(aw) * 0.4; ky = ay + math.sin(aw) * 0.4
            if inb(int(ky), int(kx)): ax, ay = kx, ky
            return ax, ay, True
    return ax, ay, False

def reveal_cone(T, B, ax, ay, ath):
    for i in range(NR):
        ang = ath - FOV / 2 + FOV * i / (NR - 1)
        dx, dy = math.cos(ang), math.sin(ang); t = 0.0
        while t < VIS:
            t += STEP_T
            c, r = int(ax + dx * t), int(ay + dy * t)
            if not inb(r, c): break
            if B[r][c] == -1: B[r][c] = T[r][c]
            if T[r][c] in (1, 2, 3): break

def bug_blocked(bugs, ax, ay, margin):   # ô bị bọ chiếm (+lề) trong tầm cảm-nhận → thầy né
    s = set()
    for b in bugs:
        if math.hypot(b['x'] - ax, b['y'] - ay) > BSENSE: continue
        br, bc = int(b['y']), int(b['x']); s.add(br * G + bc)
        if margin:
            for dr, dc in DIRS:
                nr, nc = br + dr, bc + dc
                if inb(nr, nc): s.add(nr * G + nc)
    return s

def bfs_core(B, ax, ay, blocked):
    sr, sc = int(ay), int(ax)
    def run(goal):
        s = sr * G + sc; prev = {s: -1}; q = [s]; head = 0
        while head < len(q):
            x = q[head]; head += 1; r, c = divmod(x, G)
            if goal(r, c):
                path = []; k = x
                while k != -1: path.append(k); k = prev[k]
                path.reverse(); return path
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc; ni = nr * G + nc
                if inb(nr, nc) and ni not in prev and B[nr][nc] in (0, 2) and ni not in blocked:
                    prev[ni] = x; q.append(ni)
        return None
    known_food = bool((B == 2).any())
    def frontier(r, c):
        if B[r][c] != 0: return False
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if inb(nr, nc) and B[nr][nc] == -1: return True
        return False
    path = run(lambda r, c: B[r][c] == 2) if known_food else run(frontier)
    if not path or len(path) < 2: path = run(frontier)
    if not path or len(path) < 2: return None
    nr, nc = divmod(path[1], G); dr, dc = nr - sr, nc - sc
    for k, (a, b) in enumerate(DIRS):
        if (a, b) == (dr, dc): return k
    return None

def bfs_dir(B, bugs, ax, ay):            # thầy: né bọ (lề→chỉ-ô-bọ→bỏ-qua) để không kẹt
    for mode in (2, 1, 0):
        blocked = set() if mode == 0 else bug_blocked(bugs, ax, ay, margin=(mode == 2))
        a = bfs_core(B, ax, ay, blocked)
        if a is not None: return a
    return None

def observe(B, bugs, ax, ay):            # [6,G,G] — 5 kênh từ B + kênh BỌ (trong tầm BSENSE)
    o = np.zeros((NCH, G, G), dtype=np.float32)
    for r in range(G):
        for c in range(G):
            o[_CH[int(B[r][c])], r, c] = 1.0
    for b in bugs:
        if math.hypot(b['x'] - ax, b['y'] - ay) <= BSENSE:
            br, bc = int(b['y']), int(b['x'])
            if inb(br, bc): o[5, br, bc] = 1.0
    return o

def step_move(T, B, ax, ay, ath, a):
    sr, sc = int(ay), int(ax)
    wx, wy = (sc + DIRS[a][1]) + 0.5, (sr + DIRS[a][0]) + 0.5
    desired = math.atan2(wy - ay, wx - ax)
    d = ((desired - ath + math.pi * 3) % (math.pi * 2)) - math.pi
    ath += max(-TURN, min(TURN, d)); eaten = False
    if abs(d) < 0.6:
        nx, ny = ax + math.cos(ath) * SPD, ay + math.sin(ath) * SPD
        r, c = int(ny), int(nx)
        if inb(r, c):
            v = T[r][c]
            if v == 1: B[r][c] = 1
            elif v == 3: B[r][c] = 3
            else:
                if v == 2: T[r][c] = 0; eaten = True
                ax, ay = nx, ny
                if B[r][c] == -1: B[r][c] = T[r][c]
    return ax, ay, ath, eaten

def respawn_food(T, ax, ay):
    cand = [(r, c) for r in range(1, G - 1) for c in range(1, G - 1)
            if T[r][c] == 0 and math.hypot(c - ax, r - ay) > 3]
    if cand:
        r, c = random.choice(cand); T[r][c] = 2

def legal_moves(B, ax, ay):
    sr, sc = int(ay), int(ax)
    return [k for k in range(4)
            if inb(sr + DIRS[k][0], sc + DIRS[k][1])
            and B[sr + DIRS[k][0]][sc + DIRS[k][1]] in (0, 2)]

# ---------------- VIN (6 kênh input) ----------------
class VIN(nn.Module):
    def __init__(self, ch_h=64, ch_q=10, K=40):
        super().__init__()
        self.K = K; self.ch_q = ch_q
        self.hR = nn.Conv2d(NCH, ch_h, 3, padding=1)
        self.R = nn.Conv2d(ch_h, 1, 1, bias=False)
        self.Q = nn.Conv2d(2, ch_q, 3, padding=1, bias=False)
        self.pi = nn.Linear(ch_q, 4)

    def forward(self, obs, pr, pc):
        B = obs.shape[0]
        r = self.R(F.relu(self.hR(obs)))
        v = torch.zeros_like(r)
        for _ in range(self.K):
            q = self.Q(torch.cat([r, v], 1))
            v = q.max(1, keepdim=True)[0]
        qf = q.reshape(B, self.ch_q, G * G)
        idx = (pr * G + pc).view(B, 1, 1).expand(B, self.ch_q, 1)
        return self.pi(qf.gather(2, idx).squeeze(2))

# ---------------- thu dữ-liệu (có bọ) ----------------
def collect(n_ep, steps, log=True, tag=""):
    X, P, Y = [], [], []
    t0 = time.time()
    for ep in range(n_ep):
        T, B, ax, ay, ath, bugs = gen(); reveal_cone(T, B, ax, ay, ath); spins = 0
        for _ in range(steps):
            a = bfs_dir(B, bugs, ax, ay)
            if a is None:
                ath += 0.4; reveal_cone(T, B, ax, ay, ath); move_bugs(T, bugs)
                ax, ay, _ = collide(bugs, ax, ay); spins += 1
                if spins > 24: break
                continue
            spins = 0
            X.append(observe(B, bugs, ax, ay)); P.append((int(ay), int(ax))); Y.append(a)
            ax, ay, ath, eaten = step_move(T, B, ax, ay, ath, a)
            reveal_cone(T, B, ax, ay, ath); move_bugs(T, bugs); ax, ay, _ = collide(bugs, ax, ay)
            if eaten: respawn_food(T, ax, ay)
        if log and (ep + 1) % max(1, n_ep // 8) == 0:
            print(f"   {tag}{ep+1}/{n_ep} ep · {len(X)} mẫu · {time.time()-t0:.0f}s")
    return (np.stack(X), np.array(P, dtype=np.int64), np.array(Y, dtype=np.int64))

def net_action(net, dev, B, bugs, ax, ay, legal):
    o = torch.from_numpy(observe(B, bugs, ax, ay)).to(dev).unsqueeze(0)
    pr = torch.tensor([int(ay)], device=dev); pc = torch.tensor([int(ax)], device=dev)
    with torch.no_grad():
        logits = net(o, pr, pc)[0].cpu().numpy()
    return max(legal, key=lambda k: logits[k])

def dagger_collect(net, dev, n_ep, steps):
    X, P, Y = [], [], []
    net.eval()
    for _ in range(n_ep):
        T, B, ax, ay, ath, bugs = gen(); reveal_cone(T, B, ax, ay, ath); spins = 0
        for _ in range(steps):
            legal = legal_moves(B, ax, ay)
            if not legal:
                ath += 0.4; reveal_cone(T, B, ax, ay, ath); move_bugs(T, bugs)
                ax, ay, _ = collide(bugs, ax, ay); spins += 1
                if spins > 24: break
                continue
            spins = 0
            exp = bfs_dir(B, bugs, ax, ay)
            if exp is not None and exp in legal:
                X.append(observe(B, bugs, ax, ay)); P.append((int(ay), int(ax))); Y.append(exp)
            a = net_action(net, dev, B, bugs, ax, ay, legal)
            ax, ay, ath, eaten = step_move(T, B, ax, ay, ath, a)
            reveal_cone(T, B, ax, ay, ath); move_bugs(T, bugs); ax, ay, _ = collide(bugs, ax, ay)
            if eaten: respawn_food(T, ax, ay)
    if not X: return None
    return (np.stack(X), np.array(P, dtype=np.int64), np.array(Y, dtype=np.int64))

def train_epochs(net, opt, Xtr, Ptr, Ytr, Xva, Pva, Yva, epochs, bs=1024, cap=40000):
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr), device=Xtr.device)[:cap]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(net(Xtr[b], Ptr[b, 0], Ptr[b, 1]), Ytr[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        acc = 0
        for i in range(0, len(Xva), 1024):
            pred = net(Xva[i:i+1024], Pva[i:i+1024, 0], Pva[i:i+1024, 1]).argmax(1)
            acc += (pred == Yva[i:i+1024]).sum().item()
    return acc / len(Xva)

def rollout_eval(net, dev, n_ep=30, steps=800, use_net=True):
    net.eval(); tot = 0; hits = 0; agree = 0; an = 0
    for _ in range(n_ep):
        T, B, ax, ay, ath, bugs = gen(); reveal_cone(T, B, ax, ay, ath); spins = 0
        for _ in range(steps):
            legal = legal_moves(B, ax, ay)
            exp = bfs_dir(B, bugs, ax, ay)
            if use_net:
                if not legal: a = None
                else:
                    a = net_action(net, dev, B, bugs, ax, ay, legal)
                    if exp is not None: an += 1; agree += (a == exp)
            else:
                a = exp
            if a is None:
                ath += 0.4; reveal_cone(T, B, ax, ay, ath); move_bugs(T, bugs)
                ax, ay, h = collide(bugs, ax, ay); hits += h; spins += 1
                if spins > 24: break
                continue
            spins = 0
            ax, ay, ath, eaten = step_move(T, B, ax, ay, ath, a)
            reveal_cone(T, B, ax, ay, ath); move_bugs(T, bugs)
            ax, ay, h = collide(bugs, ax, ay); hits += h
            if eaten: tot += 1; respawn_food(T, ax, ay)
    return tot / n_ep, hits / n_ep, (agree / an if an else 0.0)

def export_json(net, path):
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    obj = {"meta": {"G": G, "NCH": NCH, "ch_q": net.ch_q, "K": net.K, "BSENSE": BSENSE,
                    "chan_order": "unknown,free,wall,food,hazard,bug", "dirs": "N,E,S,W"}}
    for k, v in sd.items():
        obj[k] = np.round(v, 5).tolist()
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    return os.path.getsize(path)

def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== VIN + NÉ BỌ ĐỘNG (input {NCH} kênh) chưng-cất BFS · dev={dev} ===\n")

    print("① THU DỮ-LIỆU (BFS-né-bọ lái)…")
    Xtr, Ptr, Ytr = collect(160, 450, tag="train ")
    print(f"   train {len(Xtr)} mẫu · phân-bố {np.bincount(Ytr, minlength=4).tolist()} (N,E,S,W)")
    Xva, Pva, Yva = collect(30, 450, log=False)
    print(f"   val (bản-đồ RIÊNG) {len(Xva)} mẫu\n")

    Xall, Pall, Yall = [Xtr], [Ptr], [Ytr]
    def to_dev():
        return (torch.from_numpy(np.concatenate(Xall)).to(dev),
                torch.from_numpy(np.concatenate(Pall)).to(dev),
                torch.from_numpy(np.concatenate(Yall)).to(dev))
    XvaT = torch.from_numpy(Xva).to(dev); PvaT = torch.from_numpy(Pva).to(dev); YvaT = torch.from_numpy(Yva).to(dev)

    net = VIN(K=40).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    print("② BEHAVIOR-CLONING…")
    Xc, Pc, Yc = to_dev()
    acc = train_epochs(net, opt, Xc, Pc, Yc, XvaT, PvaT, YvaT, epochs=25)
    fn, hn, ag = rollout_eval(net, dev)
    fb, hb, _ = rollout_eval(net, dev, use_net=False)
    print(f"   BC · val-acc {acc*100:.1f}% · MẠNG {fn:.2f} food, {hn:.2f} va-bọ/sân · trùng {ag*100:.0f}% (BFS {fb:.2f} food, {hb:.2f} va)\n")

    print("③ DAgger…")
    for i in range(3):
        d = dagger_collect(net, dev, 90, 450)
        if d: Xall.append(d[0]); Pall.append(d[1]); Yall.append(d[2])
        Xc, Pc, Yc = to_dev()
        acc = train_epochs(net, opt, Xc, Pc, Yc, XvaT, PvaT, YvaT, epochs=12)
        fn, hn, ag = rollout_eval(net, dev)
        print(f"   vòng {i+1} · Σ{sum(len(a) for a in Xall)} · val {acc*100:.1f}% · "
              f"MẠNG {fn:.2f} food, {hn:.2f} va-bọ · trùng {ag*100:.0f}%")
    print()

    print("④ CHẤM CUỐI (40 sân × 800 bước)…")
    fn, hn, ag = rollout_eval(net, dev, n_ep=40)
    fb, hb, _ = rollout_eval(net, dev, n_ep=40, use_net=False)
    print(f"   • MẠNG(VIN): {fn:.2f} food · {hn:.2f} va-bọ/sân · trùng {ag*100:.1f}%")
    print(f"   • BFS-né-bọ: {fb:.2f} food · {hb:.2f} va-bọ/sân  (trần)")
    print(f"   ⇒ VIN {fn/max(fb,1e-9)*100:.0f}% food-BFS; va-bọ {hn:.2f} vs {hb:.2f}. Thuần forward-pass K={net.K}.\n")

    sz = export_json(net, SP)
    print(f"⑤ XUẤT → {SP}  ({sz/1024:.0f} KB)")
    return fn, hn, fb, hb

if __name__ == "__main__":
    main()
