"""nn-aivie3 · DEMO tích-hợp PRE-GPU — chạy trên M4, một lệnh, xuất JSON để dựng hình.

Gói 3 cơ-chế đã kiểm vào một mạch, dùng model TỰ-HỌC tại-chỗ (không GPU, không data ngoài):
  [A] STREAMING O(1): step≡scan + latency/RAM phẳng khi luồng dài.
  [B] CHÚ-Ý (salience): gate dồn vào mục-tiêu giữa nhiễu → bản-đồ chú-ý THẬT.
  [C] ROBOT VLA vòng-kín: agent nhìn→đi tới đích → quỹ-đạo THẬT.
Xuất `--out` JSON (mặc-định scratchpad) để render artifact.

CHẠY:  python nn-aivie3/scripts/rmsh_demo.py --out /tmp/rmsh_demo.json
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn.functional as F
from nn_aivie3.streaming import MambaStep
import rmsh_m25_gate as GK, rmsh_m25_salience as MS
import rmsh_m5_vla as VLA
from nn_aivie3.agent import StreamingActionModel
dev = "mps" if torch.backends.mps.is_available() else "cpu"
T0 = time.time()


def demo_streaming():
    print(f"[{time.time()-T0:.0f}s] [A] STREAMING O(1)…", flush=True)
    torch.manual_seed(0)
    from train_mamba import MambaBlock
    blk = MambaBlock(64, 16).to(dev).eval(); st = MambaStep.load_from(blk).to(dev).eval()
    x = torch.randn(1, 30, 64, device=dev)
    with torch.no_grad():
        y_seq = blk(x); s = st.init_state(1, dev)
        y_str = torch.stack([st.step(x[:, t], s) for t in range(30)], 1)
    err = (y_seq - y_str).abs().max().item()
    steps = [MambaStep(256, 16).to(dev).eval() for _ in range(8)]
    curve = []
    for n in (100, 400, 1600):
        state = [m.init_state(1, dev) for m in steps]; xx = torch.randn(1, 256, device=dev)
        if dev == "mps": torch.mps.synchronize()
        t = time.time()
        for _ in range(n):
            h = xx
            for m, s2 in zip(steps, state): h = h + m.step(h, s2)
        if dev == "mps": torch.mps.synchronize()
        curve.append({"tokens": n, "ms_per_token": (time.time() - t) / n * 1e3})
    elems = sum(s["h"].numel() + s["conv_buf"].numel() for s in state)
    return {"step_scan_err": err, "latency_curve": curve, "state_elems": elems}


def demo_salience(steps=1500):
    print(f"[{time.time()-T0:.0f}s] [B] CHÚ-Ý salience (train {steps} bước)…", flush=True)
    import rmsh_m25_curriculum as C
    torch.manual_seed(0)
    net = GK.GateNet(cond=True).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3); gen = torch.Generator().manual_seed(1)
    net.train()
    for it in range(steps):
        MS.N_DIST = C.n_dist_at(it, steps)
        for g in opt.param_groups: g["lr"] = C.lr_at(it, steps, 1.5e-3)
        img, c, quad = MS.render(96, gen, dev)
        lg, ga = net(img, c)
        loss = F.cross_entropy(lg, quad) + 0.03 * ga.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # dump vài ví-dụ: lưới màu + gate map + đúng/sai
    net.eval(); MS.N_DIST = 10; g2 = torch.Generator().manual_seed(7)
    NP = GK.NP; G = MS.G
    perm = torch.argsort(torch.rand(4, NP, generator=g2), 1); occ = perm[:, :11]
    tgt = occ[:, 0]; dist = occ[:, 1:]
    c = torch.randint(0, MS.N_COLORS, (4,), generator=g2)
    dc = torch.randint(0, MS.N_COLORS - 1, (4, 10), generator=g2); dc = dc + (dc >= c[:, None]).long()
    grid = torch.full((4, NP), -1, dtype=torch.long)
    grid.scatter_(1, tgt[:, None], c[:, None]); grid.scatter_(1, dist, dc)
    pal = torch.cat([torch.zeros(1, 3), MS._PAL], 0)
    img = pal[grid + 1].view(4, G, G, 3).permute(0, 3, 1, 2).repeat_interleave(MS.CELL, 2).repeat_interleave(MS.CELL, 3)
    img = (img + 0.03 * torch.randn(4, 3, MS.IMG, MS.IMG, generator=g2)).to(dev)
    with torch.no_grad():
        logit, gate = net(img, c.to(dev))
    quad = ((tgt // G >= G // 2).long() * 2 + (tgt % G >= G // 2).long())
    ex = []
    for i in range(4):
        ex.append({"grid": grid[i].tolist(), "target_cell": int(tgt[i]), "target_color": int(c[i]),
                   "gate": gate[i].tolist(), "pred_ok": bool(logit[i].argmax().item() == quad[i].item())})
    # gate-trên-mục-tiêu + top-k retention
    _, _, gtgt = GK.ev(net, dev, 10)
    topk = {}
    for k in (GK.NP, 8, 4, 2):
        net.serve_topk = k if k != GK.NP else None
        topk[str(k)] = GK.ev(net, dev, 10)[0]
    net.serve_topk = None
    return {"grid_size": G, "n_colors": MS.N_COLORS, "examples": ex, "gate_on_target": gtgt, "topk_acc": topk}


def demo_reach(steps=300):
    print(f"[{time.time()-T0:.0f}s] [C] ROBOT VLA vòng-kín (train {steps} bước)…", flush=True)
    torch.manual_seed(0)
    m = StreamingActionModel(n_actions=4, d=128, img=VLA.IMG, patch=VLA.CELL, window=64, tokens="grid").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3); gen = torch.Generator().manual_seed(1)
    m.train()
    for it in range(steps):
        ar, ac, tr, tc = VLA.rand_pos(64, gen)
        same = (ar == tr) & (ac == tc); ar = torch.where(same, (ar + 1) % VLA.G, ar)
        fr = VLA.draw(ar, ac, tr, tc, 64)[:, None].to(dev)
        loss = F.cross_entropy(m(fr)[:, 0], VLA.expert_dir(ar, ac, tr, tc).to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    # rollout vài tập, ghi quỹ-đạo
    m.eval(); g3 = torch.Generator().manual_seed(11); E, K = 6, 12
    ar, ac, tr, tc = VLA.rand_pos(E, g3)
    same = (ar == tr) & (ac == tc); ar = torch.where(same, (ar + 1) % VLA.G, ar)
    trajs = [[[int(ar[i]), int(ac[i])]] for i in range(E)]
    reached = torch.zeros(E, dtype=torch.bool)
    with torch.no_grad():
        st = m.new_stream(dev, batch=E)
        for k in range(K):
            fr = VLA.draw(ar, ac, tr, tc, E).to(dev); a = m.decide(m.perceive(fr, st)).cpu()
            mv = VLA.DIRS[a]; hit = (ar == tr) & (ac == tc); reached = reached | hit
            ar = (ar + torch.where(hit, torch.zeros_like(mv[:, 0]), mv[:, 0])).clamp(0, VLA.G - 1)
            ac = (ac + torch.where(hit, torch.zeros_like(mv[:, 1]), mv[:, 1])).clamp(0, VLA.G - 1)
            for i in range(E): trajs[i].append([int(ar[i]), int(ac[i])])
    reached = reached | ((ar == tr) & (ac == tc))
    return {"grid": int(VLA.G), "targets": [[int(tr[i]), int(tc[i])] for i in range(E)],
            "trajectories": trajs, "reached": reached.tolist()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/tmp/rmsh_demo.json")
    a = ap.parse_args()
    print(f"=== nn-aivie3 DEMO tích-hợp PRE-GPU · dev={dev} ===", flush=True)
    data = {"device": dev, "streaming": demo_streaming(),
            "salience": demo_salience(2200), "reach": demo_reach(600)}
    json.dump(data, open(a.out, "w"), ensure_ascii=False)
    s = data["streaming"]; sa = data["salience"]; rc = data["reach"]
    print(f"\n=== KẾT DEMO ({time.time()-T0:.0f}s) ===")
    print(f"[A] streaming: step≡scan Δ={s['step_scan_err']:.1e} · "
          f"latency {s['latency_curve'][0]['ms_per_token']:.2f}→{s['latency_curve'][-1]['ms_per_token']:.2f} ms/token "
          f"(luồng ×16) · state {s['state_elems']/1e3:.0f}K PHẲNG")
    print(f"[B] chú-ý: gate-trên-mục-tiêu {sa['gate_on_target']:.2f} · "
          f"top-8 giữ {sa['topk_acc']['8']:.0%} (full {sa['topk_acc'][str(GK.NP)]:.0%})")
    print(f"[C] robot: chạm đích {sum(rc['reached'])}/{len(rc['reached'])} tập (quỹ-đạo đã ghi)")
    print(f"→ JSON: {a.out}  (dùng dựng artifact trực-quan)")


if __name__ == "__main__":
    main()
