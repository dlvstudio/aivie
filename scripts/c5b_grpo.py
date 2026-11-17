"""TK_108 C5b — RL THEO TEST (GRPO) trên expert code (thân + expert văn phòng/học thuật đóng băng).

Mỗi bước: lấy K đề có test → mỗi đề model TỰ sinh G lời giải (song song: nạp đề một lần rồi nhân trạng thái lên G)
→ CHẠY THẬT test trong sandbox → thưởng = 1 nếu qua (+0,1 nếu đúng khuôn một khối code) → lợi thế = (thưởng − TB nhóm) / độ lệch
→ đẩy log-xác suất các lời giải tốt hơn trung bình nhóm lên, kém hơn xuống (on-policy, một lần cập nhật / lượt sinh)
+ phạt KL nhỏ so với expert gốc (đóng băng) để không trôi xa.

Đề:  python   data/c2/oci_verified.jsonl (OpenCodeInstruct đã chạy qua; đã bỏ đề giống HumanEval/MBPP)
     c2b      data/c2b/verified_*.jsonl (thầy viết test + chạy thật: JS/TS/Java/C#/PHP/Go/Bash; dùng test của thầy)
Hai model cùng trọng số: bản HUẤN LUYỆN (kernel Mamba nhanh, có gradient) và bản SINH (Mamba từng bước) — chung bộ expert.

    python scripts/c5b_grpo.py --code ckpt/expert_code/c5a_code.pt --steps 300 --tag c5b
    python scripts/c5b_grpo.py --smoke     (thử nhanh: 2 bước, G=4, 64 token)
"""
from __future__ import annotations
import argparse, copy, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert_ffn_1p3b as E
import expert_office_1p3b as O
import code_agent as CA

OUT = "ckpt/expert_code"


# ── đề ──────────────────────────────────────────────────────────────────────────────────────
def load_tasks(a, rng):
    if a.tasks_file:                                    # đề tự chọn {id, q, tests} (thử nhỏ)
        return [dict(json.loads(l), kind="python") for l in open(a.tasks_file, encoding="utf-8")]
    tasks = []
    if a.py_frac > 0:
        rows = [json.loads(l) for l in open("data/c2/oci_verified.jsonl", encoding="utf-8")]
        rng.shuffle(rows)
        for r in rows[: a.n_py]:
            tasks.append({"id": r["id"], "kind": "python", "q": r["messages"][0]["content"],
                          "tests": "\n".join(t.strip() for t in r["tests"])})
    if a.py_frac < 1:
        import c2b_teacher as T
        for p in sorted(__import__("glob").glob("data/c2b/verified_*.jsonl")):
            lang = os.path.basename(p)[9:-6]
            if lang not in T.LANG:                     # sql / tsql / html_css: chấm khác — chưa dùng ở C5b
                continue
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                if r["id"].endswith("_vi") or "files" not in r:
                    continue
                tasks.append({"id": r["id"], "kind": lang, "q": r["messages"][0]["content"], "files": r["files"], "cmd": r["cmd"]})
    return tasks


def code_of(ans, kind):
    bl = re.findall(r"```[\w#+.-]*[^\n]*\n(.*?)```", ans, re.S)
    return max(bl, key=len) if bl else None


def reward(t, ans):
    code = code_of(ans, t["kind"])
    if code is None:
        return 0.0, "không có khối code"
    fmt = 0.1 if len(re.findall(r"```", ans)) == 2 else 0.0
    if t["kind"] == "python":
        r = CA.run_python(code + "\n\n" + t["tests"] + "\n", timeout=10)
        ok = r["rc"] == 0 and not r["timeout"]
        return (1.0 if ok else 0.0) + fmt, r["out"][-300:]
    import c2b_teacher as T
    S = T.LANG[t["kind"]]
    want = list(S["files"])
    files = dict(t["files"]); files[want[0]] = code
    r = T.run_project({**files, **S.get("extra", {})}, S["cmd"], 60)
    ok = r["rc"] == 0 and not r["timeout"]
    return (1.0 if ok else 0.0) + fmt, r["out"][-300:]


# ── sinh song song G lời giải cho một đề ──────────────────────────────────────────────────────
def make_sampler(gtrunk, sets, dev):
    import hybrid_infer as hi
    blocks, norms = gtrunk.blocks, gtrunk.norms

    def ap(i, x):
        for s in sets:
            if str(i) in s:
                x = s[str(i)](x)
        return x

    def logits(x):
        return gtrunk.norm_f(x) @ gtrunk.emb.weight.T

    @torch.no_grad()
    def sample(ids, G, max_new, temp, top_p, rep_pen, eos):
        x = gtrunk.emb(torch.tensor([ids], dtype=torch.long, device=dev))
        sts = []
        for i, (blk, nm) in enumerate(zip(blocks, norms)):
            o, st = (hi.attn_prefill(blk, nm(x), None, 0) if isinstance(blk, hi.ScaledAttn) else hi.mamba_prefill(blk, nm(x), None))
            x = ap(i, x + o); sts.append(st)
        lo = logits(x[:, -1:])[:, -1].float().repeat(G, 1)                         # (G, V)
        sts = [tuple(s.repeat(G, *([1] * (s.dim() - 1))) for s in st) for st in sts]
        pos = len(ids)
        V = lo.shape[-1]
        out = torch.full((G, max_new), eos, dtype=torch.long, device=dev)
        done = torch.zeros(G, dtype=torch.bool, device=dev)
        recent = torch.tensor(ids[-64:], device=dev).repeat(G, 1)
        for t_ in range(max_new):
            l = lo.clone()
            if rep_pen:
                pen = torch.zeros(G, V, device=dev, dtype=torch.bool)
                pen.scatter_(1, recent, True)
                l[pen] -= rep_pen
            l = l / max(temp, 1e-4)
            p = torch.softmax(l, -1)
            sp, si = p.sort(-1, descending=True)
            keep = sp.cumsum(-1) - sp < top_p
            sp = sp * keep
            nxt = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            out[:, t_] = nxt
            done |= nxt == eos
            if bool(done.all()):
                break
            recent = torch.cat([recent[:, 1:], nxt[:, None]], 1)
            x = gtrunk.emb(nxt[:, None])
            new = []
            for i, (blk, nm, st) in enumerate(zip(blocks, norms, sts)):
                o, st2 = (hi.attn_step(blk, nm(x), st, pos) if isinstance(blk, hi.ScaledAttn) else hi.mamba_step(blk, nm(x), st))
                x = ap(i, x + o); new.append(st2)
            sts, pos = new, pos + 1
            lo = logits(x)[:, -1].float()
        res = []
        for g in range(G):
            seq = out[g].tolist()
            res.append(seq[: seq.index(eos)] if eos in seq else seq)
        return res
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ckpt/p3_sft.pt")
    ap.add_argument("--code", default="ckpt/expert_code/c5a_code.pt")
    ap.add_argument("--office", default="ckpt/expert_code/p3r1xl_vp.pt")
    ap.add_argument("--hoc", default="ckpt/expert_code/p3r1_hoc.pt")
    ap.add_argument("--hid-code", type=int, default=8192); ap.add_argument("--hid-off", type=int, default=4096)
    ap.add_argument("--hid-hoc", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--K", type=int, default=4, help="số đề mỗi bước")
    ap.add_argument("--G", type=int, default=8, help="số lời giải mỗi đề")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-pen", type=float, default=1.1)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--kl", type=float, default=0.02)
    ap.add_argument("--py-frac", type=float, default=0.6, help="tỉ lệ đề Python (còn lại: C2b đa ngôn ngữ)")
    ap.add_argument("--n-py", type=int, default=20000)
    ap.add_argument("--hours", type=float, default=0, help="dừng sau N giờ (0 = theo --steps)")
    ap.add_argument("--tag", default="c5b")
    ap.add_argument("--eval-n", type=int, default=24, help="số đề Python giữ riêng để kiểm xu hướng")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks-file", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.K, a.G, a.n_py = 2, 2, 4, 200
        a.max_new = min(a.max_new, 64)
    dev = a.device
    E.BASE = a.base
    rng = random.Random(608)
    tok = E.tok_(); eos = tok.token_to_id["<EOS>"]
    from nn_aivie.core.vocab import Vocabulary
    voc = Vocabulary(tok)

    # model huấn luyện (kernel nếu có) + model sinh (Mamba từng bước) — CHUNG bộ expert
    trunk, c = E.load_trunk(dev, kernel=dev == "cuda")
    if dev != "cuda":                                   # Mac (thử nhỏ): bản gói bf16 → fp32 cho khớp kiểu với expert
        trunk = trunk.float()
    gtrunk, _ = E.load_trunk(dev, kernel=False) if dev == "cuda" else (trunk, c)
    gtrunk.eval()
    sets = {}
    for name, path, hid, gb in (("code", a.code, a.hid_code, -2.0), ("vp", a.office, a.hid_off, 0.0), ("hoc", a.hoc, a.hid_hoc, 0.0)):
        if path:
            s = O.load_set(c["d"], path, gb); sets[name] = s.to(dev)
    ref = copy.deepcopy(sets["code"]).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    for n_, s in sets.items():
        for p in s.parameters():
            p.requires_grad_(n_ == "code")
    m = O.Multi(trunk, sets).to(dev)
    m.grad_from = min(int(k) for k in sets["code"])           # expert lớn cắm sớm hơn khối attention đầu
    mref = O.Multi(trunk, {**sets, "code": ref})
    sample = make_sampler(gtrunk, list(sets.values()), dev)
    params = list(sets["code"].parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    tasks = load_tasks(a, rng)
    py = [t for t in tasks if t["kind"] == "python"]; ml = [t for t in tasks if t["kind"] != "python"]
    held = py[-a.eval_n:] if a.eval_n else []                 # bộ kiểm CỐ ĐỊNH, không dùng để học
    py = py[:-a.eval_n] if a.eval_n else py

    def heldout():
        ok = 0
        for t in held:
            ids = (tok.encode("Hỏi: " + t["q"].strip() + "\n") + tok.encode("Đáp: "))[-1536:]
            o = sample(ids, 1, a.max_new, 1e-4, 1.0, a.rep_pen, eos)[0]          # gần như tham lam
            ok += reward(t, voc.detokenize(o).strip())[0] >= 1
        return ok
    print(f"[grpo] đề: python {len(py)} · đa ngôn ngữ {len(ml)} · K={a.K} G={a.G} max_new={a.max_new}", flush=True)
    os.makedirs("out/c5b", exist_ok=True)
    logf = open(f"out/c5b/{a.tag}_rollouts.jsonl", "a", encoding="utf-8")
    pool = ThreadPoolExecutor(16)
    T0, deadline = time.time(), time.time() + a.hours * 3600 if a.hours else float("inf")
    hist_r = []
    if held:
        print(f"[grpo] kiểm cố định trước khi học: qua {heldout()}/{len(held)}", flush=True)
    for step in range(a.steps):
        if time.time() > deadline:
            break
        batch = [rng.choice(py) if (rng.random() < a.py_frac or not ml) else rng.choice(ml) for _ in range(a.K)]
        seqs, advs, R_all = [], [], []
        t_gen = time.time()
        for t in batch:
            ids = tok.encode("Hỏi: " + t["q"].strip() + "\n") + tok.encode("Đáp: ")
            ids = ids[-1536:]
            m.eval()
            outs = sample(ids, a.G, a.max_new, a.temp, a.top_p, a.rep_pen, eos)
            texts = [voc.detokenize(o).strip() for o in outs]
            rw = list(pool.map(lambda x: reward(t, x), texts))
            R = torch.tensor([r for r, _ in rw])
            R_all += R.tolist()
            for g in range(a.G):
                logf.write(json.dumps({"step": step, "id": t["id"], "kind": t["kind"], "r": rw[g][0], "out": rw[g][1][-200:],
                                       "ans": texts[g][:1500]}, ensure_ascii=False) + "\n")
            if R.std() < 1e-6:                          # cả nhóm như nhau → không có tín hiệu
                continue
            A_ = (R - R.mean()) / (R.std() + 1e-4)
            for g in range(a.G):
                if outs[g]:
                    seqs.append((ids, outs[g] + [eos])); advs.append(float(A_[g]))
        logf.flush()
        t_gen = time.time() - t_gen
        hist_r += R_all
        if not seqs:
            print(f"[grpo] bước {step} · thưởng TB {sum(R_all)/len(R_all):.3f} · không nhóm nào có tín hiệu · sinh {t_gen:.0f}s", flush=True)
            continue
        m.train()
        opt.zero_grad(set_to_none=True)
        tot_tok = sum(len(o) for _, o in seqs)
        loss_v = kl_v = 0.0
        for (p_, o_), adv in zip(seqs, advs):           # từng chuỗi một (dài) → cộng dồn gradient
            ids_ = torch.tensor([p_ + o_], device=dev)
            lab = torch.tensor([[-100] * len(p_) + o_], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lo = m(ids_[:, :-1])
            lp = torch.log_softmax(lo.float(), -1)
            tgt = lab[:, 1:]; mask = tgt != -100
            g_ = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lr_ = torch.log_softmax(mref(ids_[:, :-1]).float(), -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            kl = (torch.exp(lr_ - g_) - (lr_ - g_) - 1)                # ước lượng KL k3 (luôn ≥ 0)
            loss = (-(adv * g_) + a.kl * kl)[mask].sum() / tot_tok
            loss.backward()
            loss_v += loss.item(); kl_v += kl[mask].sum().item() / tot_tok
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last = hist_r[-a.K * a.G * 10:]
        print(f"[grpo] bước {step} · thưởng TB {sum(R_all)/len(R_all):.3f} (10 bước: {sum(last)/len(last):.3f}) · "
              f"qua {sum(1 for r in R_all if r >= 1)}/{len(R_all)} · chuỗi học {len(seqs)} · loss {loss_v:.4f} · KL {kl_v:.4f} · "
              f"sinh {t_gen:.0f}s · {time.time()-T0:.0f}s", flush=True)
        if held and (step + 1) % a.eval_every == 0:
            print(f"[grpo] kiểm cố định bước {step + 1}: qua {heldout()}/{len(held)}", flush=True)
        if (step + 1) % 50 == 0 and not a.smoke:
            torch.save(sets["code"].state_dict(), f"{OUT}/{a.tag}_code.pt")
    if not a.smoke:
        torch.save(sets["code"].state_dict(), f"{OUT}/{a.tag}_code.pt")
    print(f"### c5b {a.tag}: {step + 1} bước · thưởng TB toàn bộ {sum(hist_r)/max(1,len(hist_r)):.3f} · {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
