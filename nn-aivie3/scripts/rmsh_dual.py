"""nn-aivie3 · TƯ-DUY 2 HỆ (PoC) — "luyện nhiều → phản-xạ": distill Hệ-2 vào Hệ-1.

Ánh-xạ nhận-thức người:
  • Hệ 2 (suy-nghĩ SÂU, nhiều bước, CHẬM) = BFS lập-kế-hoạch đường trong mê-cung có tường.
  • Hệ 1 (PHẢN-XẠ tức-thì, O(1)) = reflex-head nhìn khung → ra hướng ngay (StreamingActionModel).
Đi trong mê-cung cần KẾ-HOẠCH: phản-xạ THAM-LAM (đi thẳng về đích) kẹt tường. Nhưng khi
LUYỆN reflex bắt-chước BFS (distill) → reflex đi được mê-cung ở O(1)/bước = "muscle memory".

Chứng-minh: (1) tham-lam kẹt tường (thấp), (2) BFS ~100% nhưng phải TÌM mỗi bước (chậm),
(3) reflex-đã-luyện ≈ BFS ở tốc-độ phản-xạ → di-chuyển/robot quy về phản-xạ (sau luyện).

CHẠY:  python nn-aivie3/scripts/rmsh_dual.py
"""
import os, sys, time
from collections import deque
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from nn_aivie3.agent import StreamingActionModel

G, CELL, IMG = 6, 8, 48
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]                              # lên/xuống/trái/phải
RED, GRN, BLU = torch.tensor([1., .2, .2]), torch.tensor([.2, 1., .3]), torch.tensor([.3, .5, 1.])
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def free(cell, walls): return 0 <= cell[0] < G and 0 <= cell[1] < G and cell not in walls


def bfs_first_dir(walls, a, t):
    """Hệ 2: BFS → hướng bước ĐẦU từ a tới t (né tường). None nếu không tới."""
    prev = {a: None}; q = deque([a])
    while q:
        cur = q.popleft()
        if cur == t: break
        for di, (dr, dc) in enumerate(DIRS):
            nx = (cur[0] + dr, cur[1] + dc)
            if free(nx, walls) and nx not in prev:
                prev[nx] = (cur, di); q.append(nx)
    if t not in prev: return None
    cur, first = t, None
    while prev[cur] is not None:
        p, di = prev[cur]; first = di; cur = p
    return first


def gen_maze(gen, wall_p=0.24):
    """mê-cung giải-được: tường ngẫu-nhiên (mật-độ wall_p), agent/đích tự-do & thông nhau."""
    while True:
        walls = {(r, c) for r in range(G) for c in range(G)
                 if torch.rand(1, generator=gen).item() < wall_p}
        frees = [(r, c) for r in range(G) for c in range(G) if (r, c) not in walls]
        if len(frees) < 6: continue
        a = frees[int(torch.randint(len(frees), (1,), generator=gen))]
        t = frees[int(torch.randint(len(frees), (1,), generator=gen))]
        if a != t and bfs_first_dir(walls, a, t) is not None:
            return walls, a, t


def draw(walls, a, t):
    fr = torch.zeros(3, IMG, IMG)
    for (r, c) in walls: fr[:, r*CELL:r*CELL+CELL, c*CELL:c*CELL+CELL] = BLU[:, None, None]
    fr[:, t[0]*CELL:t[0]*CELL+CELL, t[1]*CELL:t[1]*CELL+CELL] = GRN[:, None, None]
    fr[:, a[0]*CELL:a[0]*CELL+CELL, a[1]*CELL:a[1]*CELL+CELL] = RED[:, None, None]
    return fr + 0.02 * torch.randn(3, IMG, IMG)


def greedy_dir(a, t):
    dr, dc = t[0]-a[0], t[1]-a[1]
    if abs(dr) >= abs(dc): return 0 if dr < 0 else 1
    return 2 if dc < 0 else 3


def rollout(policy, walls, a, t, K=24):
    """policy(a)->dir. Trả (chạm?, số bước). Đâm tường/biên = đứng yên."""
    cur = a
    for k in range(K):
        if cur == t: return True, k
        di = policy(cur); dr, dc = DIRS[di]; nx = (cur[0]+dr, cur[1]+dc)
        if free(nx, walls): cur = nx
    return cur == t, K


def main():
    print(f"=== nn-aivie3 · tư-duy 2 hệ (mê-cung {G}x{G}) · dev={dev} ===\n")
    torch.manual_seed(0)
    reflex = StreamingActionModel(n_actions=4, d=128, img=IMG, patch=CELL, window=64, tokens="grid").to(dev)
    opt = torch.optim.AdamW(reflex.parameters(), lr=1.5e-3); gen = torch.Generator().manual_seed(1)

    # LUYỆN: reflex bắt-chước BFS (distill Hệ-2 → Hệ-1)
    t0 = time.time(); BS, WP = 48, 0.15
    g2 = torch.Generator().manual_seed(999); N = 120
    eval_mazes = [gen_maze(g2, WP) for _ in range(N)]

    @torch.no_grad()
    def rdir(w, a, t):
        return int(reflex(draw(w, a, t)[None, None].to(dev))[:, 0].argmax())

    def evalset():                                                    # (bước-acc, vòng-kín) — 1 lần
        sa = sum(1 for (w, a, t) in eval_mazes if rdir(w, a, t) == bfs_first_dir(w, a, t)) / N
        cr = sum(rollout(lambda c, w=w, t=t: rdir(w, c, t), w, a, t)[0] for (w, a, t) in eval_mazes) / N
        return sa, cr

    def train_on(buf, steps, gd):
        reflex.train()
        for _ in range(steps):
            idx = torch.randint(len(buf), (BS,), generator=gd).tolist()
            fr = torch.stack([draw(*buf[i][:3]) for i in idx])[:, None].to(dev)
            lb = torch.tensor([buf[i][3] for i in idx], device=dev)
            loss = F.cross_entropy(reflex(fr)[:, 0], lb)
            opt.zero_grad(); loss.backward(); opt.step()

    # buffer AGGREGATE: seed bằng trạng-thái THẦY (curriculum tường 0→WP), giữ luôn qua các vòng
    def expert_states(n, wall_p):
        out = []
        for _ in range(n):
            w, a, t = gen_maze(gen, wall_p); out.append((w, a, t, bfs_first_dir(w, a, t)))
        return out
    buf = expert_states(2000, 0.06) + expert_states(4000, WP)         # dễ→khó

    gd = torch.Generator().manual_seed(7)
    gr = sum(rollout(lambda c, t=t: greedy_dir(c, t), w, a, t)[0] for (w, a, t) in eval_mazes) / N

    # ===== 1) BC — chỉ XEM thầy (luyện trên buffer thầy) =====
    train_on(buf, 800, gd); reflex.eval()
    sa, cr = evalset()
    print(f"  [BC · chỉ xem thầy]  bước {sa:.0%} · vòng-kín {cr:.0%}   (tham-lam {gr:.0%} · BFS 100%) · {time.time()-t0:.0f}s", flush=True)

    # ===== 2) DAgger — LUYỆN TRÊN CHÍNH LỖI CỦA MÌNH (aggregate) =====
    for rnd in range(5):
        reflex.eval()                                                # thu trạng-thái reflex ĐI QUA
        for _ in range(120):
            w, a, t = gen_maze(gd, WP); cur = a
            for _ in range(20):
                if cur == t: break
                lab = bfs_first_dir(w, cur, t)
                if lab is None: break
                buf.append((w, cur, t, lab))                          # thầy gán nhãn tại state của reflex
                dr, dc = DIRS[rdir(w, cur, t)]; nx = (cur[0] + dr, cur[1] + dc)
                if free(nx, w): cur = nx                              # đi theo hành-động CỦA REFLEX
        if len(buf) > 30000: buf = buf[-30000:]
        train_on(buf, 300, gd); reflex.eval()                        # luyện trên TOÀN buffer (thầy + lỗi mình)
        sa, cr = evalset()
        print(f"  [DAgger vòng {rnd+1}]  buffer {len(buf):>5} · bước {sa:.0%} · vòng-kín {cr:.0%}"
              f"  {'✓ VƯỢT tham-lam' if cr > gr else ''} · {time.time()-t0:.0f}s", flush=True)

    print(f"\n  ⇒ KẾT: reflex vòng-kín {cr:.0%} vs tham-lam {gr:.0%} vs BFS 100%.")
    print("    " + ("THÀNH ✓ — luyện trên CHÍNH lỗi của mình (DAgger) đưa phản-xạ O(1) VƯỢT tham-lam."
                    if cr > gr else f"chưa vượt tham-lam ({gr:.0%}) — dung-lượng model đồ-chơi là trần; scale sẽ mở."))
    print("    Đúng cách người luyện muscle-memory: không chỉ XEM thầy, mà lặp-SỬA trên sai-lầm của mình.")


if __name__ == "__main__":
    main()
