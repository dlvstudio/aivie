"""TK_104 §15 · EXPERT VĂN PHÒNG cắm thêm cạnh EXPERT CODE — trợ lý code + văn phòng trên cùng thân 1.3B.

Thân 1.3B (chat_v2) ĐÓNG BĂNG. Sau mỗi khối attention (7,15,…,47) có HAI tầng GatedFFN nối tiếp:
    x ← x + g_code(x)·FFN_code(x)        expert code (đã học 1,1 tỷ token + cầu nối tiếng Việt) — ĐÓNG BĂNG
    x ← x + g_vp(x)·FFN_vp(x)            expert văn phòng — HỌC (FFN + cổng), lớp ra khởi-tạo 0
Cổng theo từng token → mỗi expert tự mở đúng chỗ của mình (bài học BTX: cắm expert đóng băng = học không quên).
Dữ liệu trộn: văn phòng 50% · hội thoại 20% · code tiếng Việt 15% · code tiếng Anh 15% (ôn — TK_104: không ôn là quên).

    PYTHONPATH=scripts:src python scripts/expert_office_1p3b.py trainOFF --base ckpt/hybrid_1p3b_chat_v2.pt \
        --code ckpt/expert_code/code_VI.pt --steps 2000 --tag office_v1
    … evaloff  --office ckpt/expert_code/office_v1.pt      (đề văn phòng giữ riêng: Gemma chấm + công thức Excel kiểm thật)
    … mbpp / evalv1 / ask  --office …                      (kiểm KHÔNG QUÊN code + tiếng Việt chung)
"""
from __future__ import annotations
import argparse, collections, glob, json, math, os, random, re, sys, time
sys.path[:0] = ["scripts", "src"]
import torch
import torch.utils.checkpoint                  # NGC torch không tự nạp submodule này (Mac thì có)
import torch.nn as nn
import torch.nn.functional as F
import expert_ffn_1p3b as E

OUT = "ckpt/expert_code"


def make_set(d, hid, gate_bias, pos=None, exact=False):
    """pos=None → cắm tại các khối attention (E.POS, cỡ quy về 6 điểm); pos + exact → expert LỚN (vị trí + cỡ ẩn thật)."""
    s = nn.ModuleDict({str(i): E.GatedFFN(d, hid, exact=exact) for i in (pos or E.POS)})
    for b in s.values():
        b.use_gate = True
        nn.init.constant_(b.gate.bias, gate_bias)
    return s


def set_from_sd(d, sd, gate_bias=0.0):
    """dựng bộ expert ĐÚNG theo file: vị trí cắm = khoá '<khối>.…', cỡ ẩn = w3.weight.shape[1] (expert cũ lẫn expert lớn)."""
    pos = sorted({int(k.split(".")[0]) for k in sd})
    hid = sd[f"{pos[0]}.w3.weight"].shape[1]
    s = make_set(d, hid, gate_bias, pos=pos, exact=True)
    s.load_state_dict(sd)
    return s


def load_set(d, path, gate_bias=0.0):
    return set_from_sd(d, torch.load(path, map_location="cpu"), gate_bias)


class Multi(nn.Module):
    """thân + các bộ expert áp nối tiếp tại mỗi điểm cắm."""
    def __init__(self, trunk, sets):
        super().__init__()
        self.t = trunk
        self.sets = nn.ModuleDict(sets)
        self.active = None                        # None = mọi bộ; tập tên = chỉ bật các bộ đó (học 2 giai đoạn)
        self.grad_from = None                     # khối đầu tiên cần gradient (None = E.POS[0]); expert lớn cắm sớm hơn

    def apply_sets(self, i, x):
        for name, s in self.sets.items():
            if str(i) in s and (self.active is None or name in self.active):
                x = s[str(i)](x)
        return x

    def forward(self, ids):
        t = self.t
        x = t.emb(ids)
        for i, (blk, nm) in enumerate(zip(t.blocks, t.norms)):
            if i < (self.grad_from if self.grad_from is not None else E.POS[0]):
                with torch.no_grad():
                    x = x + blk(nm(x))
            else:
                x = x + torch.utils.checkpoint.checkpoint(lambda z, b=blk, n=nm: b(n(z)), x, use_reentrant=False)
            x = self.apply_sets(i, x)
        return t.norm_f(x) @ t.emb.weight.T


def build(dev, a, kernel=True):
    trunk, c = E.load_trunk(dev, kernel)
    sets = {}
    if a.code:
        cs = make_set(c["d"], a.hid_code, -2.0)
        cs.load_state_dict(torch.load(a.code, map_location="cpu"))
        for b in cs.values():
            b.use_gate = not a.code_open
        for p in cs.parameters():
            p.requires_grad_(False)
        sets["code"] = cs
    if getattr(a, "no_office", False):
        return Multi(trunk, sets).to(dev), c
    vp = make_set(c["d"], a.hid_off, 0.0)
    if a.office:
        vp.load_state_dict(torch.load(a.office, map_location="cpu"))
    sets["vp"] = vp
    return Multi(trunk, sets).to(dev), c


def gate_means(m, exs, dev):
    out = {}
    X, _ = E.batch(exs, random.Random(3), 4, dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(X[:, :-1])
    for name, s in m.sets.items():
        out[name] = round(float(torch.stack([b.g.float().mean() for b in s.values()]).mean()), 3)
    return out


def train_off(a):
    dev = "cuda"
    tok = E.tok_()
    m, c = build(dev, a)
    L = lambda p: E.load_ex(p, tok)
    off_tr, off_va = L(a.office_data + "/train.jsonl"), L(a.office_data + "/val.jsonl")
    chat_tr, chat_va = L("data/code_sft/chat_mix.jsonl"), L("data/code_sft/chat_val.jsonl")
    vi_all = L("data/code_sft/vi_code.jsonl")
    vi_tr, vi_va = vi_all[100:], vi_all[:100]
    code_tr, code_va = L("data/code_sft/train.jsonl"), L("data/code_sft/val.jsonl")
    vals = {"vp": off_va, "chat": chat_va, "vi_code": vi_va, "code": code_va}
    mix = [(off_tr, 0.50), (chat_tr, 0.20), (vi_tr, 0.15), (code_tr, 0.15)]
    if a.calib and os.path.exists(a.calib):        # hiệu chỉnh trung thực: câu phổ thông phải đáp + thực thể bịa phải từ chối
        cal = L(a.calib)
        cal_tr, cal_va = cal[150:], cal[:150]
        vals["calib"] = cal_va
        mix = [(off_tr, 0.45), (cal_tr, 0.15), (chat_tr, 0.15), (vi_tr, 0.125), (code_tr, 0.125)]
    ffn = [p for k, p in m.sets["vp"].named_parameters() if ".gate." not in k]
    gate = [p for k, p in m.sets["vp"].named_parameters() if ".gate." in k]
    opt = torch.optim.AdamW([{"params": ffn, "lr": a.lr, "base": a.lr},
                             {"params": gate, "lr": 1e-3, "base": 1e-3}], weight_decay=0.1)
    log = {"before": {k: E.val_loss(m, v, dev) for k, v in vals.items()}}
    print(f"[off] trước {log['before']} · học {sum(p.numel() for p in ffn + gate)/1e6:.0f}M · "
          f"văn phòng {len(off_tr)} mẫu", flush=True)
    rng = random.Random(1); T0 = time.time(); steps = a.steps
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1, (s + 1) / 50) * min(1.0, (steps - s) / (0.2 * steps))
        loss = 0.0
        for _ in range(a.accum):
            exs = rng.choices([d for d, _ in mix], weights=[w for _, w in mix])[0]
            X, Y = E.batch(exs, rng, a.bs, dev)
            l = E.lm_loss(m, X, Y) / a.accum
            l.backward(); loss += float(l)
        torch.nn.utils.clip_grad_norm_(ffn + gate, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s % 100 == 0 or s == steps - 1:
            print(f"[off] bước {s}/{steps} · loss {loss:.3f} · {time.time()-T0:.0f}s", flush=True)
        if a.val_every and s and s % a.val_every == 0:
            print(f"[off] val {{'vp': {E.val_loss(m, off_va, dev)}, 'chat': {E.val_loss(m, chat_va, dev)}}}", flush=True)
    log["after"] = {k: E.val_loss(m, v, dev) for k, v in vals.items()}
    log["gates"] = {k: gate_means(m, v, dev) for k, v in vals.items()}
    print(f"[off] sau {log['after']}\n[off] cổng {log['gates']}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    torch.save(m.sets["vp"].state_dict(), f"{OUT}/{a.tag}.pt")
    json.dump(log, open(f"{OUT}/trainOFF_{a.tag}.json", "w"), indent=1)


def train_code2(a):
    """CODE v2: expert code học TIẾP (từ code_VI) trên "sách giáo khoa" tiếng Việt kiểu Phi-1 (bài tập có test chạy qua +
    bài giảng có ví dụ chạy được), ôn code Việt/Anh + hội thoại. FFN LR nhỏ (giữ tri thức 1,1 tỷ token), cổng LR lớn."""
    dev = "cuda"
    tok = E.tok_()
    a.no_office = True
    m, c = build(dev, a)
    cs = m.sets["code"]
    for p in cs.parameters():
        p.requires_grad_(True)
    L = lambda p: E.load_ex(p, tok)
    tb = L("data/code_tb/train.jsonl")
    tb_tr, tb_va = tb[150:], tb[:150]
    chat_tr, chat_va = L("data/code_sft/chat_mix.jsonl"), L("data/code_sft/chat_val.jsonl")
    vi_all = L("data/code_sft/vi_code.jsonl")
    vi_tr, vi_va = vi_all[100:], vi_all[:100]
    code_tr, code_va = L("data/code_sft/train.jsonl"), L("data/code_sft/val.jsonl")
    vals = {"sgk": tb_va, "chat": chat_va, "vi_code": vi_va, "code": code_va}
    mix = [(tb_tr, 0.40), (vi_tr, 0.20), (code_tr, 0.20), (chat_tr, 0.20)]
    ffn = [p for k, p in cs.named_parameters() if ".gate." not in k]
    gate = [p for k, p in cs.named_parameters() if ".gate." in k]
    opt = torch.optim.AdamW([{"params": ffn, "lr": a.lr, "base": a.lr},
                             {"params": gate, "lr": 1e-3, "base": 1e-3}], weight_decay=0.0)
    log = {"before": {k: E.val_loss(m, v, dev) for k, v in vals.items()}}
    print(f"[code2] trước {log['before']} · sách giáo khoa {len(tb_tr)} bài", flush=True)
    rng = random.Random(2); T0 = time.time(); steps = a.steps
    for s_ in range(steps):
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1, (s_ + 1) / 50) * min(1.0, (steps - s_) / (0.2 * steps))
        loss = 0.0
        for _ in range(a.accum):
            exs = rng.choices([d for d, _ in mix], weights=[w for _, w in mix])[0]
            X, Y = E.batch(exs, rng, a.bs, dev)
            l = E.lm_loss(m, X, Y) / a.accum
            l.backward(); loss += float(l)
        torch.nn.utils.clip_grad_norm_(ffn + gate, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s_ % 100 == 0 or s_ == steps - 1:
            print(f"[code2] bước {s_}/{steps} · loss {loss:.3f} · {time.time()-T0:.0f}s", flush=True)
    log["after"] = {k: E.val_loss(m, v, dev) for k, v in vals.items()}
    log["gates"] = {k: gate_means(m, v, dev) for k, v in vals.items()}
    print(f"[code2] sau {log['after']}\n[code2] cổng {log['gates']}", flush=True)
    torch.save(cs.state_dict(), f"{OUT}/{a.tag}.pt")
    json.dump(log, open(f"{OUT}/trainCODE2_{a.tag}.json", "w"), indent=1)


# ── suy-luận: runner có NHIỀU bộ expert (thay E.adapter_runner để dùng lại mbpp/evalv1/ask) ──
def multi_runner(a, dev):
    import hybrid_infer as hi
    trunk, c = E.load_trunk(dev, kernel=False)
    trunk.eval()
    sets = []
    if a.code:
        cs = load_set(c["d"], a.code, -2.0)
        for b in cs.values():
            b.use_gate = not a.code_open
        sets.append(cs)
    if a.office:
        vp = load_set(c["d"], a.office); sets.append(vp)
    if getattr(a, "hoc", ""):
        hc = load_set(c["d"], a.hoc); sets.append(hc)
    sets = [s.to(dev).eval() for s in sets]

    def ap(i, x):
        for s in sets:
            if str(i) in s:
                x = s[str(i)](x)
        return x

    class R(hi.Runner):
        @torch.no_grad()
        def prefill(self, ids, all_logits=False, state=None, vis=None):   # vis: hàm sinh chung (có ảnh) truyền vào — runner chữ bỏ qua
            x = self.m.emb(torch.tensor([ids], dtype=torch.long, device=self.dev))
            pos0 = state["pos"] if state else 0
            olds = state["layers"] if state else [None] * len(self.m.blocks)
            sts = []
            for i, (blk, nm, st) in enumerate(zip(self.m.blocks, self.m.norms, olds)):
                o, st2 = (hi.attn_prefill(blk, nm(x), st, pos0) if isinstance(blk, hi.ScaledAttn)
                          else hi.mamba_prefill(blk, nm(x), st))
                x = ap(i, x + o); sts.append(st2)
            lo = self._logits(x if all_logits else x[:, -1:])
            return lo[0], {"layers": sts, "pos": pos0 + len(ids)}

        @torch.no_grad()
        def step(self, tok, state):
            x = self.m.emb(torch.tensor([[tok]], dtype=torch.long, device=self.dev))
            new, pos = [], state["pos"]
            for i, (blk, nm, st) in enumerate(zip(self.m.blocks, self.m.norms, state["layers"])):
                o, st2 = (hi.attn_step(blk, nm(x), st, pos) if isinstance(blk, hi.ScaledAttn)
                          else hi.mamba_step(blk, nm(x), st))
                x = ap(i, x + o); new.append(st2)
            return self._logits(x)[0, -1], {"layers": new, "pos": pos + 1}
    return R(trunk, dev)


def extract_formula(ans):
    f_ = None
    for blk in re.findall(r"```[a-zA-Z]*\n(.*?)```", ans, re.S):
        f_ = next((ln.strip() for ln in blk.splitlines() if ln.strip().startswith("=")), None)
        if f_:
            break
    if not f_:
        m_ = re.search(r"(=[A-Z]+\(.*\))", ans)
        f_ = m_.group(1).strip() if m_ else None
    if f_ and ";" in f_ and "," not in f_:          # Excel vùng Việt Nam dùng ; — đổi về , để máy tính kiểm
        f_ = f_.replace(";", ",")
    return f_


def eval_off(a):
    """Đề văn phòng GIỮ RIÊNG (data/office/eval_office.jsonl): sinh → Excel kiểm thật bằng `formulas`; còn lại Gemma chấm."""
    from gen_office import check_excel, judge_work
    tok = E.tok_()
    r = multi_runner(a, getattr(a, "device", "cuda"))
    rows = [json.loads(l) for l in open("data/office/eval_office.jsonl", encoding="utf-8")]
    out = f"out/office_eval_{a.tag}.jsonl"
    res = []
    for k, q in enumerate(rows):
        ans = E.answer(r, tok, q["q"], n=600)
        rec = {**q, "answer": ans}
        if "formula" in q:
            f_ = extract_formula(ans)
            try:
                ok, got = check_excel(q["table"], f_, q["expected"]) if f_ else (False, None)
            except Exception:
                ok, got = False, None
            rec["excel_ok"], rec["formula_model"] = bool(ok), f_
        res.append(rec)
        if k % 20 == 0:
            print(f"[evaloff] {k}/{len(rows)}", flush=True)
    judged = [judge_work(({**rec, "request": rec["q"]}, a.ollama)) for rec in res]
    with open(out, "w", encoding="utf-8") as f:
        for rec in judged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    from collections import defaultdict
    by = defaultdict(list)
    for rec in judged:
        if "j" in rec:
            by[rec["cat"]].append(rec["j"])
    summ = {c: {"n": len(v), "ok": round(sum(x["ok"] for x in v) / len(v), 2), "vi": round(sum(x["vi"] for x in v) / len(v), 2)}
            for c, v in by.items()}
    xl = [rec["excel_ok"] for rec in judged if "excel_ok" in rec]
    summ["excel_dung"] = f"{sum(xl)}/{len(xl)}"
    print("[evaloff]", json.dumps(summ, ensure_ascii=False), flush=True)
    json.dump(summ, open(f"{OUT}/evaloff_{a.tag}.json", "w"), ensure_ascii=False, indent=1)


def humaneval(a):
    """HumanEval (164 bài, thước đo của Phi-1): hoàn thành hàm → chạy check() thật → pass@1 (tham lam)."""
    import subprocess, tempfile
    from datasets import load_dataset
    tok = E.tok_()
    r = multi_runner(a, getattr(a, "device", "cuda"))
    ds = list(load_dataset("openai/openai_humaneval", split="test"))
    ok, rows = 0, []
    for k, ex in enumerate(ds):
        q = f"Complete the following Python function:\n```python\n{ex['prompt']}```"
        resp = E.answer(r, tok, q, n=500)
        m_ = re.search(r"```(?:python)?\n(.*?)```", resp, re.S)
        code = m_.group(1) if m_ else resp
        if f"def {ex['entry_point']}" not in code:
            code = ex["prompt"] + code                      # model chỉ viết thân hàm
        prog = code + "\n\n" + ex["test"] + f"\ncheck({ex['entry_point']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog)
        try:
            passed = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0
        except subprocess.TimeoutExpired:
            passed = False
        os.unlink(f.name)
        ok += passed
        rows.append({"task": ex["task_id"], "pass": passed, "resp": resp[:800]})
        if k % 20 == 0:
            print(f"[humaneval] {k}/{len(ds)} · qua {ok}", flush=True)
    res = {"tag": a.tag, "code": a.code, "code_open": a.code_open, "office": a.office, "n": len(ds), "pass": ok,
           "pass_rate": round(ok / len(ds), 3)}
    print(json.dumps(res), flush=True)
    json.dump({**res, "rows": rows}, open(f"{OUT}/humaneval_{a.tag}.json", "w"), ensure_ascii=False, indent=1)


def easy20(a):
    """20 câu kiến thức phổ thông dễ (data/eval_clean/easy20.jsonl) — đo lỗi "trung thực quá tay" (tham lam, nhiệt độ 0)."""
    tok = E.tok_()
    r = multi_runner(a, getattr(a, "device", "cuda"))
    rows = [json.loads(l) for l in open("data/eval_clean/easy20.jsonl", encoding="utf-8")]
    ABS = ["không chắc", "không biết", "không có thông tin", "kiểm tra lại"]
    ok = ab = 0
    out = []
    for q in rows:
        ans = E.answer(r, tok, q["q"], n=80)
        hit = q["gold"].lower() in ans.lower()
        abst = (not hit) and any(x in ans.lower() for x in ABS)
        ok += hit; ab += abst
        out.append({**q, "answer": ans, "hit": hit, "abstain": abst})
    res = {"tag": a.tag, "dung": ok, "tu_choi": ab, "n": len(rows)}
    print("[easy20]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump({**res, "rows": out}, open(f"{OUT}/easy20_{a.tag}.json", "w"), ensure_ascii=False, indent=1)


ASK_OFF = ["Soạn giúp tôi email xin nghỉ phép 2 ngày gửi sếp vì việc gia đình.",
           "Viết công thức Excel tính tổng doanh thu cột D khi cột B là 'Hà Nội'. Dữ liệu từ dòng 2 đến 100.",
           "Làm sao tạo mục lục tự động trong Word?",
           "Viết lại câu này cho lịch sự hơn: 'Sao giờ này chưa gửi báo cáo, làm ăn kiểu gì vậy?'",
           "Cho tôi đoạn code Python đệ quy tính giai thừa.",
           "Hôm nay trời mưa, bạn có lời khuyên gì không?"]


# ── v4: bộ train CHUNG (expert học thuật + bước cuối "chat dài") ─────────────────────────────────────────────
DATA = {"sgk": "data/sgk/train.jsonl", "math": "data/math/train.jsonl", "mt": "data/multiturn/train.jsonl",
        "chat": "data/code_sft/chat_mix.jsonl", "off": "data/office_v2/train.jsonl", "vi_code": "data/code_sft/vi_code.jsonl",
        "code": "data/code_sft/train.jsonl", "tb": "data/code_tb/train.jsonl", "calib": "data/calib/train.jsonl", "fix": "data/v5/sft.jsonl",
        "offfix": "data/office_fix/off.jsonl", "fixfix": "data/office_fix/fix.jsonl",
        # TK_108 C2 — dữ liệu code đã kiểm bằng chạy thật (c2_build_verified.py, c2_teacher.py)
        "c2oci": "data/c2/oci_verified.jsonl", "c2vi": "data/c2/vi.jsonl", "c2dbg": "data/c2/debug.jsonl",
        "c2pred": "data/c2/predict.jsonl", "c2sql": "data/c2/sql.jsonl",
        "c3": "data/c3/train.jsonl",                    # quỹ đạo dùng công cụ thành công (c3_teacher.py) — nhiều lượt, dài
        # TK_108 C2b — code đa ngôn ngữ (c2b_mix.py): A kiểm cú pháp · plus hiện đại · verified thầy viết test + chạy thật
        "c2bA": "data/c2b/mix_A.jsonl", "c2bP": "data/c2b/mix_plus.jsonl", "c2bV": "data/c2b/mix_verified.jsonl",
        "c2dbg2": "data/c2/debug_v2.jsonl",
        "c3r": "data/c3/train_r.jsonl", "c3r2": "data/c3/train_r2.jsonl",   # C3-R2 (18/10): đợt 2, dồn kiểu "sửa sai rồi lặp y hệt"
        "c2bF": "data/c2b/mix_fix.jsonl",
        "c7": "data/c7/train.jsonl"}                     # TK_108 C7 (22/10): đề THỰC TẾ tiếng Việt nhiều ràng buộc, 2 lớp kiểm chạy thật (c7_real.py)
# c2bF: sai → lỗi thật → sửa (nhật ký thầy C2b B + C2 debug v2; lượt sai nolearn)                 # C3-R: quỹ đạo PHỤC HỒI (lượt sai chèn sẵn có nolearn → chỉ học phần sửa)
LONG = {"c3": 4096, "c3r": 4096, "c3r2": 4096,                                     # khoá cần ngữ cảnh dài hơn 1536 (quỹ đạo: trung vị ~1,8K, p90 ~2,8K token)
        "c2bA": 2560, "c2bP": 2560, "c2bV": 2560, "c2bF": 3072, "c7": 3072}       # code đa ngôn ngữ: bỏ mẫu dài hơn (không cắt giữa code)


def load_key(k, tok):
    if k in LONG:                                       # load_ex2 cắt TRÁI → bỏ quỹ đạo dài hơn (mất phần đầu nhiệm vụ)
        return [e for e in load_ex2(DATA[k], tok, max_len=LONG[k]) if len(e[0]) < LONG[k]]
    return load_ex2(DATA[k], tok) if k in ("fix", "mt", "fixfix") else E.load_ex(DATA[k], tok)       # công thức luôn trong khối ```excel (build_office_fix.py)


def train_gen(a):
    """Cắm đủ 3 bộ expert (code, văn phòng, học thuật) — học bộ trong --train, trộn dữ liệu theo --mix.
    Bộ MỚI (chưa có file) khởi tạo lớp ra = 0 → lúc đầu y hệt model cũ. FFN bộ cũ học LR nhỏ (--lr-old), cổng LR 1e-3."""
    dev = "cuda"
    tok = E.tok_()
    trunk, c = E.load_trunk(dev)
    spec = {"code": (a.code, a.hid_code, -2.0), "vp": (a.office, a.hid_off, 0.0), "hoc": (a.hoc, a.hid_hoc, 0.0)}
    sets = {}
    for name, (path, hid, gb) in spec.items():
        if path or name in a.train:
            st_ = load_set(c["d"], path, gb) if path else make_set(c["d"], hid, gb)
            sets[name] = st_
    m = Multi(trunk, sets).to(dev)
    m.grad_from = min(int(k) for n_ in a.train.split(",") if n_ in sets for k in sets[n_])   # expert lớn cắm từ khối 3
    lora_ps = []
    if getattr(a, "lora_r", 0):                           # C8: LoRA khối trộn từ khối grad_from trở lên (dưới đó chạy no_grad)
        lora_ps = E.add_code_lora(trunk, a.lora_r, start=m.grad_from, attn_only=a.lora_attn_only)
        print(f"[gen] CodeLoRA r={a.lora_r} · {len(lora_ps) // 2} lớp chiếu · {sum(p.numel() for p in lora_ps)/1e6:.1f}M tham số · lr {a.lora_lr}", flush=True)
    groups, params = [], []
    for name, st_ in m.sets.items():
        on = name in a.train.split(",")
        for p in st_.parameters():
            p.requires_grad_(on)
        if not on:
            continue
        new = not spec[name][0]
        ffn = [p for k, p in st_.named_parameters() if ".gate." not in k]
        gate = [p for k, p in st_.named_parameters() if ".gate." in k]
        lr_f = a.lr if new else a.lr_old
        groups += [{"params": ffn, "lr": lr_f, "base": lr_f}, {"params": gate, "lr": 1e-3, "base": 1e-3}]
        params += ffn + gate
    if lora_ps:
        groups.append({"params": lora_ps, "lr": a.lora_lr, "base": a.lora_lr})
        params += lora_ps
    mix, vals = [], {}
    for item in a.mix.split(","):
        k, w = item.split(":")
        ex = load_key(k, tok)
        nv = 100 if k == "vi_code" else 150
        vals[k], tr = ex[:nv], ex[nv:]
        mix.append((tr, float(w)))
        print(f"[gen] {k}: train {len(tr)} · val {len(vals[k])} · trọng số {w}", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    log = {"before": {k: E.val_loss(m, v, dev) for k, v in vals.items()}}
    print(f"[gen] học {a.train} · {sum(p.numel() for p in params)/1e6:.0f}M tham số · trước {log['before']}", flush=True)
    rng = random.Random(9); T0 = time.time(); steps = a.steps
    for s_ in range(steps):
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1, (s_ + 1) / 50) * min(1.0, (steps - s_) / (0.2 * steps))
        loss = 0.0
        for _ in range(a.accum):
            exs = rng.choices([d for d, _ in mix], weights=[w for _, w in mix])[0]
            X, Y = E.batch(exs, rng, a.bs, dev)
            l = E.lm_loss(m, X, Y) / a.accum
            l.backward(); loss += float(l)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s_ % 100 == 0 or s_ == steps - 1:
            print(f"[gen] bước {s_}/{steps} · loss {loss:.3f} · {time.time()-T0:.0f}s", flush=True)
    log["after"] = {k: E.val_loss(m, v, dev) for k, v in vals.items()}
    log["gates"] = {k: gate_means(m, v, dev) for k, v in vals.items()}
    print(f"[gen] sau {log['after']}\n[gen] cổng {log['gates']}", flush=True)
    for name in a.train.split(","):
        torch.save(m.sets[name].state_dict(), f"{OUT}/{a.tag}_{name}.pt")
    if lora_ps:
        torch.save(E.code_lora_state(trunk), f"{OUT}/{a.tag}_lora.pt")
    json.dump(log, open(f"{OUT}/trainGEN_{a.tag}.json", "w"), indent=1)


def train_cptx(a):
    """TK_108 pilot EXPERT CODE LỚN: PRETRAIN RIÊNG expert (thân + expert văn phòng/học thuật ĐÓNG BĂNG, vẫn bật) trên code
    thô (data/cx, build_code_expert_data.py) — expert mang kiến thức, thân làm cầu nối. Cắm --pos (mặc định cứ 4 khối),
    cỡ ẩn THẬT --raw-hid, CỔNG MỞ HẲN khi học; lưu cổng khởi ~sigmoid(3) cho bước SFT sau. Loss trên mọi token.
    Nhiều GPU: torchrun --nproc_per_node N … trainCPTX … — mỗi GPU một bản đủ model, dữ liệu ngẫu nhiên riêng theo rank,
    gộp gradient expert (all_reduce) mỗi bước; GPU 0 đo + lưu. --bs/--accum là MỖI GPU (token/bước = N × bs × accum × seq).
    --resume: nạp lại expert + optimizer + bước từ {tag}_state.pt (lưu mỗi --eval-every bước) → chạy tiếp đúng lịch LR."""
    import numpy as np
    import torch.distributed as dist
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    main_ = rank == 0
    pr = print if main_ else (lambda *x, **k: None)
    dev = getattr(a, "device", "cuda")
    trunk, c = E.load_trunk(dev, kernel=dev == "cuda")
    if dev != "cuda":                                   # Mac (thử nhỏ): bản gói bf16 → fp32
        trunk = trunk.float()
    nb = len(trunk.blocks)
    pos = list(range(3, nb, 4)) if a.pos == "every4" else [int(x) for x in a.pos.split(",")]
    st_path = f"{OUT}/{a.tag}_state.pt"
    resume = a.resume and os.path.exists(st_path)
    src = f"{OUT}/{a.tag}_code.pt" if resume else a.code
    big = make_set(c["d"], a.raw_hid, 3.0, pos=pos, exact=True) if not src else load_set(c["d"], src, 3.0)
    for b in big.values():
        b.use_gate = False                            # cổng MỞ HẲN khi pretrain
    sets = {"code": big}
    for name, path in (("vp", a.office), ("hoc", a.hoc)):
        if path:
            sets[name] = load_set(c["d"], path)
    m = Multi(trunk, sets).to(dev)
    m.grad_from = min(int(k) for k in big)
    for n_, st_ in m.sets.items():
        for p_ in st_.parameters():
            p_.requires_grad_(n_ == "code")
    params = list(m.sets["code"].parameters())
    pos = sorted(int(k) for k in big)
    pr(f"[cptx] {world} GPU · expert {len(big)} điểm {pos[:4]}… · cỡ ẩn {a.raw_hid} · {sum(p.numel() for p in params)/1e9:.2f} tỷ tham số học", flush=True)
    mix = []
    for item in a.cx_mix.split(","):
        g, w = item.split(":")
        arrs = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(f"{a.cx_dir}/{g}/*.npy"))]
        assert arrs, f"thiếu dữ liệu {a.cx_dir}/{g}"
        mix.append((g, arrs, float(w)))
        pr(f"[cptx] {g}: {sum(len(x) for x in arrs)/1e6:.0f} triệu token · trọng số {w}", flush=True)
    vals = {g: np.load(f"{a.cx_dir}/val/{g}.npy", mmap_mode="r") for g, _, _ in mix if os.path.exists(f"{a.cx_dir}/val/{g}.npy")}
    rng = np.random.default_rng(11 + 1000 * rank + int(time.time()) % 1000 * bool(resume))   # mỗi GPU / lần chạy tiếp một dòng dữ liệu khác
    L = a.seq

    def window(arrs):
        x = arrs[rng.integers(len(arrs))]
        i = int(rng.integers(0, len(x) - L - 1))
        return torch.from_numpy(np.asarray(x[i:i + L + 1], dtype=np.int64))

    @torch.no_grad()
    def evals(n=24):
        m.eval(); out = {}
        r0 = np.random.default_rng(5)
        for g, v in vals.items():
            ls = []
            for k in range(0, n, 8):
                W = torch.stack([torch.from_numpy(np.asarray(v[j:j + L + 1], dtype=np.int64))
                                 for j in r0.integers(0, len(v) - L - 1, 8)]).to(dev)
                ls.append(float(E.lm_loss(m, W, W)))
            out[g] = round(sum(ls) / len(ls), 4)
        m.train(); return out
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    s_, used = 0, 0.0                                   # used = giờ đã chạy ở các lần trước (lịch LR theo giờ)
    if resume:
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); s_, used, log = st["step"], st["hours"], st["log"]
        pr(f"[cptx] chạy tiếp từ bước {s_} ({used:.2f} giờ đã chạy)", flush=True)
    else:
        log = {"pos": pos, "raw_hid": a.raw_hid, "world": world, "eval": [{"step": 0, **(evals() if main_ else {})}]}
        pr(f"[cptx] trước (thân + expert = đồng nhất): {log['eval'][0]}", flush=True)
    s0 = s_
    T0 = time.time(); deadline = T0 + (a.hours - used) * 3600 if a.hours else float("inf")
    w_ = np.array([w for _, _, w in mix]); w_ = w_ / w_.sum()

    def save():
        torch.save(m.sets["code"].state_dict(), f"{OUT}/{a.tag}_code.pt")
        json.dump(log, open(f"{OUT}/trainCPTX_{a.tag}.json", "w"), indent=1)
        torch.save({"opt": opt.state_dict(), "step": s_, "hours": used + (time.time() - T0) / 3600, "log": log}, st_path + ".tmp")
        os.replace(st_path + ".tmp", st_path)
    while True:
        go = torch.tensor([float(s_ < a.steps and time.time() < deadline)], device=dev)
        if ddp:
            dist.all_reduce(go, op=dist.ReduceOp.MIN)   # mọi GPU dừng CÙNG bước (lệch là treo all_reduce)
        if not go.item():
            break
        warm, frac = min(1, (s_ + 1) / 300), s_ / max(1, a.steps)
        if a.hours:                                     # chạy theo giờ → lịch cosine theo thời gian đã dùng
            frac = max(frac, (used + (time.time() - T0) / 3600) / a.hours)
        for g_ in opt.param_groups:
            g_["lr"] = a.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))
        loss = 0.0
        for _ in range(a.accum):
            W = torch.stack([window(mix[int(rng.choice(len(mix), p=w_))][1]) for _ in range(a.bs)]).to(dev)
            l = E.lm_loss(m, W, W) / a.accum
            l.backward(); loss += float(l)
        if ddp:
            hs = [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True) for p in params if p.grad is not None]
            for h in hs:
                h.wait()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        s_ += 1
        if s_ % 50 == 0:
            tps = (s_ - s0) * world * a.bs * a.accum * L / (time.time() - T0)
            pr(f"[cptx] bước {s_} · loss {loss:.3f} · {tps/1e3:.1f}K token/s · {used + (time.time()-T0)/3600:.2f} giờ", flush=True)
        if s_ % a.eval_every == 0:
            if main_:
                e = evals(); log["eval"].append({"step": s_, **e}); pr(f"[cptx] đo bước {s_}: {e}", flush=True)
                save()
            if ddp:
                dist.barrier()
    log["tokens"] = s_ * world * a.bs * a.accum * L
    if main_:
        e = evals(); log["eval"].append({"step": s_, **e})
        pr(f"[cptx] sau {s_} bước ({log['tokens']/1e6:.0f} triệu token): {e}", flush=True)
        save()
    else:
        e = {}
    if ddp:
        dist.barrier(); dist.destroy_process_group()
    pr(f"### cptx {a.tag}: {s_} bước · {log['tokens']/1e6:.0f}M token · trước {log['eval'][0]} → sau {e}", flush=True)


def train_gen2(a):
    """Học expert 2 GIAI ĐOẠN (thay cho học FFN + cổng cùng lúc — kiểm chứng 14/10: cổng sập ~0,003, FFN phóng to bù,
    mở cổng thì ra rác). Thân đóng băng; chỉ dùng cho bộ MỚI (chưa có file).
      A) cổng MỞ HẲN, mỗi bước chỉ bật MỘT expert và cho học trên dữ liệu miền của nó (--domain) → FFN học đúng biên độ, chuyên miền
      B) đóng băng FFN, bật cổng (khởi tạo mở ~sigmoid(--gate-init)), bật đủ các expert, học cổng trên --mix chung
         (gồm cả dữ liệu ngoài miền) → cổng học đóng ở chỗ expert làm hại, mở ở chỗ có ích."""
    dev = a.device
    tok = E.tok_()
    trunk, c = E.load_trunk(dev, kernel=dev == "cuda")
    spec = {"code": (a.code, a.hid_code), "vp": (a.office, a.hid_off), "hoc": (a.hoc, a.hid_hoc)}
    names = a.train.split(",")
    sets = {}
    for name, (path, hid) in spec.items():
        if name in names:
            assert not path, f"train_gen2 chỉ học bộ MỚI — bỏ --{name} cũ hoặc bỏ {name} khỏi --train"
            sets[name] = make_set(c["d"], hid, a.gate_init)
        elif path:                                    # bộ đã có, cắm vào nhưng đóng băng (chỉ bật ở giai đoạn B)
            st_ = load_set(c["d"], path)
            for p in st_.parameters():
                p.requires_grad_(False)
            sets[name] = st_
    m = Multi(trunk, sets).to(dev)
    frozen = {n_ for n_ in sets if n_ not in names}
    cache = {}
    def data(k):
        if k not in cache:
            ex = load_key(k, tok)
            if a.max_len:
                ex = [e for e in ex if len(e[0]) <= a.max_len]
            if a.max_ex:
                ex = ex[:a.max_ex]
            nv = min(100 if k == "vi_code" else 150, len(ex) // 10)
            cache[k] = (ex[nv:], ex[:nv])
            print(f"[gen2] {k}: train {len(ex) - nv} · val {nv}", flush=True)
        return cache[k]
    dom = {}
    for item in a.domain.split(","):
        n_, ks = item.split(":")
        if n_ in names:
            dom[n_] = [data(k) for k in ks.split("+")]
    mix = [(data(k)[0], float(w)) for k, w in (x.split(":") for x in a.mix.split(","))]
    keys = sorted(cache)
    def vals():
        return {k: E.val_loss(m, cache[k][1], dev, n=a.val_n) for k in keys}
    def sched(opt, s_, steps):
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1, (s_ + 1) / 50) * min(1.0, (steps - s_) / (0.2 * steps))
    log = {"before": vals()}
    print(f"[gen2] trước {log['before']}", flush=True)
    rng = random.Random(9); T0 = time.time()

    # ── A: FFN, cổng mở, một expert / bước trên miền của nó ──
    for n_ in names:
        for b in m.sets[n_].values():
            b.use_gate = False
    ffn = {n_: [p for k, p in m.sets[n_].named_parameters() if ".gate." not in k] for n_ in names}
    for n_ in names:
        for k, p in m.sets[n_].named_parameters():
            p.requires_grad_(".gate." not in k)
    opt = torch.optim.AdamW([{"params": ffn[n_], "lr": a.lr, "base": a.lr} for n_ in names], weight_decay=0.0)
    for s_ in range(a.steps):
        sched(opt, s_, a.steps)
        n_ = names[s_ % len(names)]
        m.active = {n_} | frozen                       # bộ cũ đóng băng vẫn bật (như lúc dùng thật), bộ mới khác thì tắt
        loss = 0.0
        for _ in range(a.accum):
            tr, _v = rng.choice(dom[n_])
            X, Y = E.batch(tr, rng, a.bs, dev)
            l = E.lm_loss(m, X, Y) / a.accum
            l.backward(); loss += float(l)
        torch.nn.utils.clip_grad_norm_(ffn[n_], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s_ % 100 == 0 or s_ == a.steps - 1:
            print(f"[gen2] A bước {s_}/{a.steps} · {n_} · loss {loss:.3f} · {time.time()-T0:.0f}s", flush=True)
    m.active = None
    log["after_A_open"] = vals()                      # đủ expert, cổng mở hẳn
    print(f"[gen2] sau A (mở hẳn, đủ expert) {log['after_A_open']}", flush=True)

    # ── B: cổng, FFN đóng băng, đủ expert, dữ liệu chung ──
    gates = []
    for n_ in names:
        for k, p in m.sets[n_].named_parameters():
            p.requires_grad_(".gate." in k)
            if ".gate." in k:
                gates.append(p)
        for b in m.sets[n_].values():
            b.use_gate = True
    opt = torch.optim.AdamW([{"params": gates, "lr": a.lr_gate, "base": a.lr_gate}], weight_decay=0.0)
    for s_ in range(a.gate_steps):
        sched(opt, s_, a.gate_steps)
        loss = 0.0
        for _ in range(a.accum):
            exs = rng.choices([d for d, _ in mix], weights=[w for _, w in mix])[0]
            X, Y = E.batch(exs, rng, a.bs, dev)
            l = E.lm_loss(m, X, Y) / a.accum
            l.backward(); loss += float(l)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s_ % 100 == 0 or s_ == a.gate_steps - 1:
            print(f"[gen2] B bước {s_}/{a.gate_steps} · loss {loss:.3f} · {time.time()-T0:.0f}s", flush=True)
    log["after"] = vals()
    log["gates"] = {k: gate_means(m, cache[k][1], dev) for k in keys}
    print(f"[gen2] sau B {log['after']}\n[gen2] cổng {log['gates']}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    for n_ in names:
        torch.save(m.sets[n_].state_dict(), f"{OUT}/{a.tag}_{n_}.pt")
    json.dump(log, open(f"{OUT}/trainGEN2_{a.tag}.json", "w"), indent=1)


# ── v4: bộ chấm mới ──────────────────────────────────────────────────────────────────────────
def gen_ids(r, tok, ids, n, calc=False, rep_pen=1.1):
    """sinh tham lam từ ids (khuôn Hỏi/Đáp đã ghép sẵn); calc=True → gặp 'biểu thức =' thì ép kết quả đúng (TK_104 §14.4)."""
    import numpy as np
    from nn_aivie.core.vocab import Vocabulary
    from math_calc import EXPR, calc as calc_f, digit_ids
    voc = Vocabulary(tok); eos = tok.token_to_id["<EOS>"]
    lo, state = r.prefill(ids); lo = lo[-1]
    hist, gen, just = list(ids), [], False
    while len(gen) < n:
        l = lo.float().cpu().numpy().astype(np.float64)
        for t in set(hist[-64:]):
            l[t] -= rep_pen
        if just:
            l[digit_ids(voc, len(l))] = -1e9; just = False
        nxt = int(l.argmax())
        if nxt == eos:
            break
        forced = [nxt]; gen.append(nxt)
        if calc:
            text = voc.detokenize(gen)
            if text.endswith("="):
                tail = text[-80:]
                mm = EXPR.search(tail)
                cut = bool(mm) and re.search(r"[\+\-\*x×/:]\s*$", tail[:mm.start()]) is not None
                v = calc_f(mm.group(1)) if mm and not cut else None
                if v is not None:
                    extra = tok.encode(" " + v); forced += extra; gen += extra; just = True
        for t in forced:
            hist.append(t); lo, state = r.step(t, state)
    return voc.detokenize(gen).strip()


def hist_ids(tok, msgs):
    ids = []
    eos = tok.token_to_id["<EOS>"]
    for m_ in msgs:
        if m_["role"] == "user":
            ids += tok.encode("Hỏi: " + m_["content"].strip() + "\n")
        else:
            ids += tok.encode("Đáp: ") + tok.encode(m_["content"].strip()) + [eos]
    return ids + tok.encode("Đáp: ")


def llm_json(prompt, url):
    from gen_office import ollama, loads_loose
    return loads_loose(ollama(prompt, url, temp=0, fmt="json", num_predict=200))


P_SGK = """Câu hỏi kiến thức phổ thông (tiếng Việt): {q}
Đáp án chuẩn: {ref}
Câu trả lời cần chấm: {ans}
Chấm: "dung" nếu câu trả lời nêu đúng ý chính của đáp án chuẩn (diễn đạt khác vẫn được); "mot_phan" nếu đúng một phần;
"sai" nếu sai, bịa, lạc đề hoặc từ chối. Trả về DUY NHẤT JSON {{"v": "dung|mot_phan|sai"}}"""

P_MT = """Đây là hội thoại giữa người dùng và trợ lý. Hãy chấm CÂU TRẢ LỜI CUỐI của trợ lý.
{hist}
NGƯỜI DÙNG (câu mới nhất): {q}
TRỢ LÝ (cần chấm): {ans}
Cho điểm 1–5:
- "bam": trả lời đúng vào câu mới nhất VÀ nhất quán với các lượt trước (nhớ thông tin đã nêu), không lạc đề/lan man.
- "ok": đúng và hữu ích.
Trả về DUY NHẤT JSON {{"bam": 4, "ok": 4}}"""


def eval_sgk(a):
    tok = E.tok_(); r = multi_runner(a, getattr(a, "device", "cuda"))
    rows = [json.loads(l) for l in open("data/sgk/test.jsonl", encoding="utf-8")][: a.n]
    cnt = collections.Counter(); out = []
    for q in rows:
        ans = gen_ids(r, tok, hist_ids(tok, [{"role": "user", "content": q["q"]}]), 200)
        try:
            v = llm_json(P_SGK.format(q=q["q"], ref=q["ref"], ans=ans), a.llm).get("v", "?")
        except Exception:
            v = "?"
        cnt[v] += 1; out.append({**q, "answer": ans, "v": v})
    res = {"tag": a.tag, "n": len(rows), **{k: round(100 * c / len(rows), 1) for k, c in cnt.items()}}
    print("[evalsgk]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump({**res, "rows": out}, open(f"{OUT}/evalsgk_{a.tag}.json", "w"), ensure_ascii=False, indent=1)


def eval_math(a):
    from gen_math import num_last, close
    tok = E.tok_(); r = multi_runner(a, getattr(a, "device", "cuda"))
    rows = [json.loads(l) for l in open("data/math/test.jsonl", encoding="utf-8")]
    ev = [json.loads(l) for l in open("data/eval_clean/eval_v1.jsonl", encoding="utf-8")]
    rows += [{"q": x["q"], "gold": num_last(x["gold"]), "src": "eval_v1"} for x in ev if x["type"] == "math"]
    res = collections.defaultdict(lambda: [0, 0, 0]); out = []
    for q in rows:
        ids = hist_ids(tok, [{"role": "user", "content": q["q"]}])
        a0 = gen_ids(r, tok, ids, 400); a1 = gen_ids(r, tok, ids, 400, calc=True)
        k = res[q["src"]]; k[0] += 1; k[1] += close(num_last(a0), q["gold"]); k[2] += close(num_last(a1), q["gold"])
        out.append({**q, "plain": a0, "calc": a1})
    summ = {src: {"n": v[0], "dung": v[1], "dung_may_tinh": v[2]} for src, v in res.items()}
    print("[evalmath]", json.dumps(summ, ensure_ascii=False), flush=True)
    json.dump({"tag": a.tag, "tom_tat": summ, "rows": out[:80]}, open(f"{OUT}/evalmath_{a.tag}.json", "w"), ensure_ascii=False, indent=1)


def eval_mt(a):
    tok = E.tok_(); r = multi_runner(a, getattr(a, "device", "cuda"))
    rows = [json.loads(l) for l in open("data/multiturn/probe.jsonl", encoding="utf-8")]
    by_turn = collections.defaultdict(list); out = []
    for c in rows:
        msgs, conv = [], []
        for t, u in enumerate(c["users"]):
            msgs.append({"role": "user", "content": u})
            ids = hist_ids(tok, msgs)[-1800:]
            ans = gen_ids(r, tok, ids, 350)
            hist = "\n".join(f"{'NGƯỜI DÙNG' if m_['role'] == 'user' else 'TRỢ LÝ'}: {m_['content'][:600]}" for m_ in msgs[:-1])
            try:
                j = llm_json(P_MT.format(hist=hist or "(chưa có)", q=u, ans=ans[:2000]), a.llm)
                by_turn[t + 1].append((float(j.get("bam", 0)), float(j.get("ok", 0))))
            except Exception:
                j = {}
            msgs.append({"role": "assistant", "content": ans}); conv.append({"u": u, "a": ans, "j": j})
        out.append({"id": c["id"], "dom": c["dom"], "conv": conv})
    summ = {f"luot_{t}": {"n": len(v), "bam": round(sum(x for x, _ in v) / len(v), 2), "ok": round(sum(y for _, y in v) / len(v), 2)}
            for t, v in sorted(by_turn.items())}
    print("[evalmt]", json.dumps(summ, ensure_ascii=False), flush=True)
    json.dump({"tag": a.tag, "tom_tat": summ, "rows": out}, open(f"{OUT}/evalmt_{a.tag}.json", "w"), ensure_ascii=False, indent=1)


# ── v5: dữ liệu từ VÒNG THẦY CHAT–CHẤM–SỬA → SFT (chỉ lượt cuối) + DPO ─────────────────────────────────────
def build_v5(a):
    sft, dpo, n = [], [], 0
    for tag in a.v5_tags.split(","):
        for l in open(f"data/v5/{tag}.jsonl", encoding="utf-8"):
            x = json.loads(l); n += 1
            ctx = x["history"] + [{"role": "user", "content": x["user"]}]
            if 0 <= x["diem"] < 8 and len(x["sua"]) > 10:
                sft.append({"messages": ctx + [{"role": "assistant", "content": x["sua"]}], "last_only": True, "src": "v5_sua"})
                if x["diem"] <= 5:
                    dpo.append({"ctx": ctx, "chosen": x["sua"], "rejected": x["answer"]})
            elif x["diem"] >= 9:
                sft.append({"messages": ctx + [{"role": "assistant", "content": x["answer"]}], "last_only": True, "src": "v5_tot"})
    for name, rows in (("sft", sft), ("dpo", dpo)):
        with open(f"data/v5/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[v5] {n} lượt → SFT {len(sft)} (sửa {sum(r['src'] == 'v5_sua' for r in sft)}, tốt {sum(r['src'] == 'v5_tot' for r in sft)}) · DPO {len(dpo)} cặp")


def load_ex2(path, tok, max_len=1536):
    """như E.load_ex nhưng: 'last_only' → chỉ học lượt trợ lý CUỐI; quá dài → cắt TRÁI (giữ lượt cuối)."""
    from train_hybrid_sft import build_example
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        msgs = r["messages"]
        ids, lab = build_example(msgs, tok, 10 ** 6)
        if r.get("last_only"):
            pre, _ = build_example(msgs[:-1], tok, 10 ** 6)
            lab = [-100] * len(pre) + lab[len(pre):]
        if len(ids) > max_len:
            ids, lab = ids[-max_len:], lab[-max_len:]
        if any(x != -100 for x in lab):
            out.append((ids, lab))
    return out


def seq_logp(m, X, Y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(X[:, :-1])
    lp = torch.log_softmax(lo.float(), -1)
    tgt = Y[:, 1:]
    mask = tgt != -100
    g = lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (g * mask).sum(-1), mask.sum(-1)


def train_dpo(a):
    """DPO trên các bộ expert (thân đóng băng): đẩy bản sửa của thầy lên, câu kém của chính model xuống, so với model THAM CHIẾU
    (chính nó trước khi học). Thêm 0,2 × NLL trên bản sửa cho ổn định."""
    dev = "cuda"
    tok = E.tok_()
    eos = tok.token_to_id["<EOS>"]
    trunk, c = E.load_trunk(dev)
    spec = {"code": (a.code, a.hid_code, -2.0), "vp": (a.office, a.hid_off, 0.0), "hoc": (a.hoc, a.hid_hoc, 0.0)}
    sets = {}
    for name, (path, hid, gb) in spec.items():
        if path:
            st_ = load_set(c["d"], path, gb); sets[name] = st_
    m = Multi(trunk, sets).to(dev)
    params = []
    for name, st_ in m.sets.items():
        on = name in a.train.split(",")
        for p in st_.parameters():
            p.requires_grad_(on)
        if on:
            params += list(st_.parameters())
    pairs = [json.loads(l) for l in open(a.dpo_data, encoding="utf-8")]
    RM, ML = a.resp_max, (a.max_len or 1536)                # code: câu trả lời dài → --resp-max 1024 --max-len 2048

    def enc(ctx, resp):
        p_ = hist_ids(tok, ctx)
        r_ = tok.encode(resp.strip())[:RM] + [eos]
        ids = (p_ + r_)[-ML:]
        lab = ([-100] * len(p_) + r_)[-ML:]
        return ids, lab
    data = [(enc(x["ctx"], x["chosen"]), enc(x["ctx"], x["rejected"])) for x in pairs]
    data = [d for d in data if any(t != -100 for t in d[0][1]) and any(t != -100 for t in d[1][1])]
    print(f"[dpo] {len(data)} cặp từ {a.dpo_data}", flush=True)

    def mk(items):
        L = max(len(i) for i, _ in items)
        X = torch.zeros(len(items), L, dtype=torch.long); Y = torch.full((len(items), L), -100, dtype=torch.long)
        for j, (i, l) in enumerate(items):
            X[j, :len(i)] = torch.tensor(i); Y[j, :len(l)] = torch.tensor(l)
        return X.to(dev), Y.to(dev)
    ref = []
    m.eval()
    with torch.no_grad():
        for k in range(0, len(data), 8):
            ch = data[k:k + 8]
            Xc, Yc = mk([c_ for c_, _ in ch]); Xr, Yr = mk([r_ for _, r_ in ch])
            ref += list(zip(seq_logp(m, Xc, Yc)[0].tolist(), seq_logp(m, Xr, Yr)[0].tolist()))
    m.train()
    opt = torch.optim.AdamW(params, lr=a.lr_old, weight_decay=0.0)
    rng = random.Random(11); T0 = time.time(); beta = a.beta
    for s_ in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr_old * min(1, (s_ + 1) / 30) * min(1.0, (a.steps - s_) / (0.2 * a.steps))
        idx = rng.sample(range(len(data)), min(a.bs, len(data)))
        Xc, Yc = mk([data[i][0] for i in idx]); Xr, Yr = mk([data[i][1] for i in idx])
        lc, nc = seq_logp(m, Xc, Yc); lr_, _ = seq_logp(m, Xr, Yr)
        rc = torch.tensor([ref[i][0] for i in idx], device=dev); rr = torch.tensor([ref[i][1] for i in idx], device=dev)
        margin = beta * ((lc - rc) - (lr_ - rr))
        loss = -F.logsigmoid(margin).mean() + 0.2 * (-(lc / nc.clamp(min=1))).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if s_ % 25 == 0 or s_ == a.steps - 1:
            print(f"[dpo] bước {s_}/{a.steps} · loss {loss.item():.3f} · biên {margin.mean().item():.2f} · "
                  f"thắng {(margin > 0).float().mean().item():.2f} · {time.time()-T0:.0f}s", flush=True)
    for name in a.train.split(","):
        torch.save(m.sets[name].state_dict(), f"{OUT}/{a.tag}_{name}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trainOFF", "trainCODE2", "trainGEN", "trainGEN2", "trainDPO", "trainCPTX", "buildV5", "evaloff", "evalsgk", "evalmath", "evalmt", "mbpp", "humaneval", "evalv1", "ask", "easy20"])
    ap.add_argument("--base", default="ckpt/hybrid_1p3b_chat_v2.pt")
    ap.add_argument("--code", default="ckpt/expert_code/code_VI.pt", help="'' = không cắm expert code")
    ap.add_argument("--office", default="", help="trọng-số expert văn phòng đã học; '' = không cắm")
    ap.add_argument("--code-open", action="store_true", help="expert code MỞ HẲN (bỏ cổng)")
    ap.add_argument("--office-data", default="data/office", help="thư mục có train.jsonl/val.jsonl văn phòng")
    ap.add_argument("--calib", default="", help="dữ liệu hiệu chỉnh trung thực (data/calib/train.jsonl)")
    ap.add_argument("--hid-code", type=int, default=8192)
    ap.add_argument("--hid-off", type=int, default=4096)
    ap.add_argument("--hoc", default="", help="expert học thuật (SGK + toán) đã học")
    ap.add_argument("--hid-hoc", type=int, default=8192)
    ap.add_argument("--train", default="hoc", help="bộ expert được học (trainGEN), vd hoc hoặc code,vp,hoc")
    ap.add_argument("--mix", default="sgk:0.4,math:0.3,mt:0.1,chat:0.1,off:0.05,vi_code:0.05")
    ap.add_argument("--domain", default="code:code+vi_code+tb,vp:off,hoc:sgk+math",
                    help="trainGEN2 giai đoạn A: dữ liệu miền của từng expert (nối bằng +)")
    ap.add_argument("--gate-steps", type=int, default=400, help="trainGEN2 giai đoạn B: số bước học cổng")
    ap.add_argument("--gate-init", type=float, default=3.0, help="trainGEN2: bias cổng lúc bật (sigmoid(3)≈0,95 — gần mở)")
    ap.add_argument("--lr-gate", type=float, default=1e-3)
    ap.add_argument("--val-n", type=int, default=200, help="số mẫu đo val mỗi nguồn (trainGEN2)")
    ap.add_argument("--max-len", type=int, default=0, help="bỏ mẫu dài hơn N token (thử nhỏ trên Mac)")
    ap.add_argument("--max-ex", type=int, default=0, help="giới hạn số mẫu mỗi nguồn (thử nhỏ trên Mac)")
    ap.add_argument("--lr-old", type=float, default=2e-5, help="LR cho FFN của bộ expert đã có")
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1", help="giám khảo (vLLM/Ollama)")
    ap.add_argument("--v5-tags", default="r1", help="các vòng dữ liệu v5 (data/v5/<tag>.jsonl)")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    ap.add_argument("--dpo-data", default="data/v5/dpo.jsonl", help="cặp {ctx, chosen, rejected}")
    ap.add_argument("--pos", default="every4", help="trainCPTX: điểm cắm expert lớn ('every4' hoặc '3,7,11,…')")
    ap.add_argument("--raw-hid", type=int, default=8192, help="trainCPTX: cỡ ẩn THẬT mỗi FFN")
    ap.add_argument("--cx-mix", default="opc_algo:0.25,opc_snip:0.2,opc_files:0.25,web:0.15,vi:0.15")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--cx-dir", default="data/cx", help="trainCPTX: thư mục dữ liệu (data/cx = pilot, data/cx3 = chạy thật)")
    ap.add_argument("--resume", action="store_true", help="trainCPTX: chạy tiếp từ {tag}_state.pt nếu có")
    ap.add_argument("--hours", type=float, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--resp-max", type=int, default=400, help="DPO: số token tối đa của câu trả lời")
    ap.add_argument("--lora-r", type=int, default=0, help="trainGEN: >0 = học thêm CodeLoRA trên khối trộn của thân (TK_108 C8)")
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--lora-attn-only", action="store_true", help="CodeLoRA chỉ trên attention (bỏ in_proj/out_proj Mamba)")
    ap.add_argument("--device", default="cuda", help="cuda | mps (chấm trên Mac M4)")
    ap.add_argument("--tag", default="office_v1")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    E.BASE = a.base
    torch.manual_seed(0)
    if a.cmd == "trainOFF":
        return train_off(a)
    if a.cmd == "trainCODE2":
        return train_code2(a)
    if a.cmd == "trainGEN":
        return train_gen(a)
    if a.cmd == "trainGEN2":
        return train_gen2(a)
    if a.cmd == "trainDPO":
        return train_dpo(a)
    if a.cmd == "trainCPTX":
        return train_cptx(a)
    if a.cmd == "buildV5":
        return build_v5(a)
    if a.cmd in ("evalsgk", "evalmath", "evalmt"):
        return {"evalsgk": eval_sgk, "evalmath": eval_math, "evalmt": eval_mt}[a.cmd](a)
    if a.cmd == "easy20":
        return easy20(a)
    if a.cmd == "evaloff":
        return eval_off(a)
    if a.cmd == "humaneval":
        return humaneval(a)
    # dùng lại mbpp / evalv1 / ask của expert code, chỉ thay runner bằng bản nhiều-expert
    E.adapter_runner = lambda which, dev, path=None: multi_runner(a, dev)
    a.which, a.adapters = "multi", a.office
    if a.cmd == "ask":
        E.ASK = ASK_OFF
        return E.ask(a)
    return E.mbpp(a) if a.cmd == "mbpp" else E.evalv1(a)


if __name__ == "__main__":
    main()
