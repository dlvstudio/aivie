"""TK_107 P3 — thử nhanh code expert/SFT trên thân MỚI (96 khối, 12 attention) trước khi thuê box. Chạy được trên Mac.

  1. nạp thân qua build_from_cfg → điểm cắm = 12 khối attention, mỗi FFN nửa cỡ (tổng tham số expert giữ nguyên)
  2. cắm 3 bộ expert mới (code/văn phòng/học thuật) → logit KHỚP thân gốc (lớp ra = 0)
  3. vài bước học CHỈ expert trên 1 lô → loss giảm, thân không đổi
  4. Runner từng bước (prefill + step) có expert → khớp forward cả chuỗi
  5. SFT toàn bộ tham số trên thân TÍ HON cùng kiểu cấu hình (attn_layers list + RoPE 1e6) + mẫu dài + ngân sách token

    PYTHONPATH=scripts:src python scripts/smoke_p3.py --base checkpoints/hf_hybrid/hybrid_2p6b_b9536_bf16.pt
"""
import argparse, json, os, subprocess, sys, tempfile, time
sys.path[:0] = ["scripts", "src"]
import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/hf_hybrid/hybrid_2p6b_b9536_bf16.pt")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--sft-data", default="data/sft_p3/sft_p3.jsonl")
    a = ap.parse_args()
    import train_hybrid_scale as ths
    import expert_ffn_1p3b as E
    import expert_office_1p3b as X
    import hybrid_infer as hi
    dev, T0 = a.device, time.time()
    ok = {}

    # 0 ── cfg 1.3B cũ (attn_every=8) → 6 điểm cũ, hid giữ nguyên (checkpoint v5 / vlm_v4 vẫn nạp được)
    E.set_pos({"layers": 48, "attn_every": 8})
    ok["compat_1p3b"] = E.POS == [7, 15, 23, 31, 39, 47] and E.hid_eff(8192) == 8192
    print(f"[0] cfg 1.3B: POS {E.POS} · hid 8192 → {E.hid_eff(8192)}", flush=True)

    # 1 ── thân + điểm cắm
    E.BASE = a.base
    trunk, c = E.load_trunk("cpu", kernel=False)
    trunk = trunk.to(dev, torch.bfloat16).eval()
    attn = ths.attn_layers_of(c)
    ok["pos"] = E.POS == attn and E.POS[0] == attn[0]
    print(f"[1] thân {c['layers']} khối · rope {c.get('rope_base', 1e4):g} · POS {E.POS} · "
          f"hid 8192 → {E.hid_eff(8192)} / 4096 → {E.hid_eff(4096)} ({time.time()-T0:.0f}s)", flush=True)

    # 2 ── 3 bộ expert mới: khớp thân gốc
    sets = {"code": X.make_set(c["d"], 8192, -2.0), "vp": X.make_set(c["d"], 4096, 0.0),
            "hoc": X.make_set(c["d"], 8192, 0.0)}
    m = X.Multi(trunk, sets).to(dev, torch.bfloat16)
    n_exp = sum(p.numel() for p in m.sets.parameters())
    ids = torch.randint(5, 32000, (1, 64), device=dev)
    with torch.no_grad():
        ref = trunk(ids).float(); got = m(ids).float()
    d = (ref - got).abs().max().item()
    ok["identity"] = d == 0.0
    print(f"[2] 3 bộ expert × {len(E.POS)} điểm = {n_exp/1e6:.0f}M tham số · sai lệch logit so thân {d:.2e}", flush=True)

    # 3 ── học chỉ expert vài bước trên 1 lô (cổng + FFN), thân phải giữ nguyên
    for p in trunk.parameters():
        p.requires_grad_(False)
    params = [p for p in m.sets.parameters()]
    for p in params:
        p.requires_grad_(True)
    w0 = trunk.blocks[E.POS[-1]].wq.weight.detach().clone()
    opt = torch.optim.AdamW(params, lr=1e-4)
    X_ = torch.randint(5, 32000, (2, 48), device=dev)
    m.train(); ls = []
    for s in range(6):
        lo = m(X_[:, :-1])
        loss = nn.functional.cross_entropy(lo.float().reshape(-1, lo.size(-1)), X_[:, 1:].reshape(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); ls.append(round(float(loss), 3))
    m.eval()
    ok["learn"] = ls[-1] < ls[0] and torch.equal(w0, trunk.blocks[E.POS[-1]].wq.weight)
    print(f"[3] loss chỉ-expert {ls} · thân không đổi: {torch.equal(w0, trunk.blocks[E.POS[-1]].wq.weight)}", flush=True)

    del trunk, m
    if dev == "mps":
        torch.mps.empty_cache()

    # 4 ── Runner từng bước có expert = forward cả chuỗi. fp32 trên thân TÍ HON cùng kiểu cfg (attn_layers list,
    #      RoPE 1e6, expert lớp ra KHÁC 0) — bf16 qua 96 khối trôi số nên không dùng để kiểm logic
    cfg_t = {"d": 128, "layers": 16, "attn_every": 0, "attn_layers": [3, 6, 11, 14], "d_state": 16, "heads": 4,
             "kv_heads": None, "rope_base": 1e6}
    torch.manual_seed(1)
    tt = ths.build_from_cfg(cfg_t, 32002).eval()
    E.set_pos(cfg_t)
    ts = {"a": X.make_set(128, 512, 0.0), "b": X.make_set(128, 256, -2.0)}
    for st_ in ts.values():
        for b in st_.values():
            nn.init.normal_(b.w3.weight, std=0.02)
    mt = X.Multi(tt, ts).eval()
    per = {str(i): nn.Sequential(*[s[str(i)] for s in mt.sets.values()]) for i in E.POS}
    r = hi.Runner(tt, "cpu", nn.ModuleDict(per).eval())
    seq = torch.randint(5, 32000, (40,)).tolist()
    with torch.no_grad():
        full = mt(torch.tensor([seq]))[0].float()
        lo, st = r.prefill(seq[:12], all_logits=True)
        lo2, st = r.prefill(seq[12:20], all_logits=True, state=st)
        outs = [lo.float(), lo2.float()]
        for t in seq[20:]:
            l, st = r.step(t, st); outs.append(l.float()[None])
    got = torch.cat(outs)[:40]
    d4 = (got[:39] - full[:39]).abs().max().item()
    ok["runner"] = d4 < 1e-3
    print(f"[4] Runner (prefill 12+8, step 20) có expert, fp32: sai lệch {d4:.2e} · argmax khớp "
          f"{int((got[:39].argmax(-1) == full[:39].argmax(-1)).sum())}/39", flush=True)

    # 5 ── SFT toàn tham số trên thân tí hon cùng kiểu cfg (Mac không đủ RAM cho AdamW 3,4B)
    tiny = ths.HybridLM(32002, 128, 16, [3, 7, 11, 15], 16, 4, None, 1e6)
    cfg = {"d": 128, "layers": 16, "attn_every": 0, "attn_layers": [3, 7, 11, 15], "d_state": 16, "heads": 4,
           "kv_heads": None, "rope_base": 1e6}
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "tiny.pt")
        torch.save({"cfg": cfg, "state": tiny.state_dict(), "step": 0}, ck)
        sub = os.path.join(td, "sft.jsonl")
        with open(a.sft_data, encoding="utf-8") as f, open(sub, "w", encoding="utf-8") as g:
            n = nl = 0
            for line in f:
                isl = '"source": "long_' in line
                if (isl and nl < 40) or (not isl and n < 200):
                    g.write(line); nl += isl; n += not isl
                if nl >= 40 and n >= 200:
                    break
        out = os.path.join(td, "sft_out.pt")
        L = "8192" if dev == "cuda" else "3000"            # Mac: Mamba thuần PyTorch quét tuần tự → 8K mất cả phút/bước
        cmd = [sys.executable, "-u", "scripts/train_hybrid_sft.py", "--init-from", ck, "--data", sub,
               "--max-len", L, "--bs", "8", "--grad-accum", "1", "--tok-budget", L, "--epochs", "1",
               "--lr", "1e-3", "--warmup", "2", "--eval-every", "10", "--save-every", "100000", "--out", out,
               "--device", "cuda" if dev == "cuda" else "cpu"]  # MPS báo hết bộ nhớ thì KHÔNG ném lỗi → loss 0 giả
        p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "scripts:src"})
        tail = "\n".join(p.stdout.strip().splitlines()[-4:])
        saved = torch.load(out, weights_only=False) if os.path.exists(out) else {}
        import re
        vals = [float(x) for x in re.findall(r"· val ([0-9.]+)", p.stdout)]
        ok["sft"] = (p.returncode == 0 and saved.get("cfg", {}).get("rope_base") == 1e6 and "rope 1e+06" in p.stdout
                     and len(vals) >= 2 and 0 < vals[-1] < vals[0])
        print(f"[5] SFT tí hon: mã thoát {p.returncode}\n{tail}\n{p.stderr[-800:] if p.returncode else ''}", flush=True)

    print(json.dumps({"ok": ok, "all": all(ok.values()), "sec": round(time.time() - T0)}, ensure_ascii=False))
    sys.exit(0 if all(ok.values()) else 1)


if __name__ == "__main__":
    main()
