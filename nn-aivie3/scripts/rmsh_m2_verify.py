"""Verify nhanh giả-thuyết cân-scale (vis_ln + temp scale) trên joint recall, T=6."""
import os, sys, time, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "nn-aivie3", "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import rmsh_m2_smoke as S                                               # noqa: E402

dev = S.pick_device()
torch.manual_seed(0)
m = S.build(dev); m.train()
opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
gen = torch.Generator().manual_seed(1); t0 = time.time(); T = 6
for it in range(250):
    fr, tx, ans, _ = S.make_batch(128, T, dev, gen)
    loss = F.cross_entropy(m.logits_multi(fr, tx)[:, -1], ans)
    opt.zero_grad(); loss.backward(); opt.step()
acc = S.eval_recall(m, dev, T=T, n=512)
print(f"JOINT recall T={T} sau 250 bước: loss {loss.item():.3f} · acc {acc:.1%} "
      f"(ngẫu-nhiên {1/S.N_COLORS:.1%}) · {time.time()-t0:.0f}s → {'HỌC ĐƯỢC' if acc>0.6 else 'VẪN KẸT'}")
