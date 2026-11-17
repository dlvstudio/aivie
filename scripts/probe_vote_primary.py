#!/usr/bin/env python
"""Vote W_skip CHỦ ĐẠO + bonus CHUNG-CÂU + vòng quay — đúng đặc tả tác giả (§12.106 note).

Đặc tả gốc: đếm phiếu W_skip[d] → giảm theo d → **cộng thêm khi chung ngữ cảnh/chung câu** →
softmax → vòng quay (không argmax). §12.106 dựng vote nhưng (1) làm KÊNH PHỤ, không chủ đạo; (2)
THIẾU bonus chung-câu. Đây dựng ĐỦ cả hai để đo bản trung thành nhất.

W_skip = bộ lái DUY NHẤT (không succ/path/matrix). Bonus chung-câu: candidate xuất hiện trong CÙNG
cha với ngữ cảnh gần → thưởng — kéo sinh bám một câu nhớ, giả thuyết: giảm salad/trôi.

Ba cấu hình chấm mù: vote-only · vote+chung-câu · (đối chứng: copy-corpus qua generate mặc định
lấy từ file trước). Luật 0: đếm + softmax, không backprop.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import SPECIAL_EOS  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.readout import decode_surface, mask_non_surface  # noqa: E402
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans  # noqa: E402
from nn_aivie.hca.w_skip import vote_logits  # noqa: E402

from gen_samples_blind import PROMPTS  # noqa: E402


def build_content_index(state, struct):
    """token → list cha chứa nó (theo content-position); và content-set mỗi cha."""
    v = state.vocab
    tok2par = defaultdict(list)
    par_cont: list[set] = []
    for pi, rec in enumerate(state.parents.parents):
        sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        toks = set()
        for sp in iter_sentence_spans(sq, v):
            for t in _content_positions(sp, v, content_only=True, struct_mask=struct):
                toks.add(int(t))
        par_cont.append(toks)
        for t in toks:
            tok2par[t].append(pi)
    return tok2par, par_cont


def same_sentence_bonus(hist_content, tok2par, par_cont, n_vocab, *, window=4, top_m=8):
    """Thưởng candidate CÙNG CHA với ngữ cảnh gần. Chọn top-M cha khớp `window` token gần nhất,
    rồi cộng cho mọi content-token của các cha ấy (chuẩn hoá theo overlap)."""
    recent = hist_content[-window:]
    if not recent:
        return np.zeros(n_vocab)
    cand: dict[int, int] = defaultdict(int)
    for t in recent:
        for pi in tok2par.get(t, ()):
            cand[pi] += 1
    if not cand:
        return np.zeros(n_vocab)
    top = sorted(cand.items(), key=lambda kv: -kv[1])[:top_m]
    out = np.zeros(n_vocab)
    for pi, ov in top:
        w = ov / len(recent)          # cha khớp nhiều token gần → trọng số cao
        for b in par_cont[pi]:
            if 0 <= b < n_vocab:
                out[b] += w
    return out


def gen_one(state, prompt, *, struct, tok2par, par_cont, same_sent, beta, T, topk,
            d_max, lam, max_len, rng):
    v = state.vocab
    n = len(v)
    eos = v._token_to_id.get(SPECIAL_EOS)
    hist = list(v.encode_prompt(prompt))
    out_ids = []
    for _ in range(max_len):
        vote = vote_logits(state, hist, n_vocab=n, d_max=d_max, lam=lam, topk=topk)
        score = vote.astype(np.float64)
        if same_sent and beta > 0:
            hc = [int(t) for t in hist if 0 <= int(t) < n and not struct[int(t)] and v.is_surface(int(t))]
            bonus = same_sentence_bonus(hc, tok2par, par_cont, n, window=4, top_m=8)
            rv = float(score.max()) or 1.0
            rb = float(bonus.max()) or 1.0
            score = score + beta * rv * (bonus / rb)     # chung-câu tăng thêm chút
        score = mask_non_surface(score, v)
        # cho phép EOS cạnh tranh: EOS lấy điểm = trung vị điểm dương (không dìm, không ép)
        pos = score[np.isfinite(score) & (score > 0)]
        if eos is not None and eos < n:
            score[eos] = float(np.median(pos)) if pos.size else 0.0
        finite = np.where(np.isfinite(score), score, -np.inf)
        if not np.isfinite(finite).any() or finite.max() <= 0:
            break
        k = min(topk, int(np.isfinite(finite).sum()))
        idx = np.argpartition(-finite, k - 1)[:k]
        z = finite[idx] / max(T, 1e-6)
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        nxt = int(rng.choice(idx, p=p))              # VÒNG QUAY đúng xác suất softmax
        if nxt == eos:
            break
        out_ids.append(nxt)
        hist.append(nxt)
    return out_ids, decode_surface(state, out_ids).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--d-max", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="out/vote_primary")
    ap.add_argument("--T-grid", default="", help="vd '0.4,0.6' — chạy nhiều T; trống = một --T")
    ap.add_argument("--beta-grid", default="", help="vd '0,1' — kết hợp với T-grid")
    args = ap.parse_args()

    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    struct = [is_structural_token(v, i) for i in range(len(v))]
    tok2par, par_cont = build_content_index(state, struct)
    print(f"[{time.time()-t0:.0f}s] index xong · {len(state.parents):,} cha", flush=True)

    from nn_aivie.hca.metrics_pmi import PmiCrossIndex
    pmi = PmiCrossIndex(state)
    print(f"[{time.time()-t0:.0f}s] PMI-cross sẵn", flush=True)

    Ts = [float(x) for x in args.T_grid.split(",") if x.strip()] or [args.T]
    betas = [float(x) for x in args.beta_grid.split(",") if x.strip()] or [args.beta]

    # (label, same_sent, T, beta)
    jobs = []
    for T in Ts:
        for beta in betas:
            jobs.append((f"vote_only_T{T}", False, T, 0.0))
            if beta > 0:
                jobs.append((f"vote_chung_T{T}_b{beta}", True, T, beta))
    # dedupe labels
    seen = set()
    uniq = []
    for j in jobs:
        if j[0] in seen:
            continue
        seen.add(j[0])
        uniq.append(j)
    jobs = uniq

    rows = []
    for name, ss, T, beta in jobs:
        rng = np.random.default_rng(args.seed)
        for p in PROMPTS:
            cids, txt = gen_one(state, p, struct=struct, tok2par=tok2par, par_cont=par_cont,
                          same_sent=ss, beta=beta, T=T, topk=args.topk,
                          d_max=args.d_max, lam=args.lam, max_len=args.max_len, rng=rng)
            pids = list(v.encode_prompt(p))
            sc = pmi.score(pids, cids)
            rows.append((name, p, txt, sc))
        mean = sum(r[3] for r in rows if r[0] == name) / len(PROMPTS)
        print(f"[{time.time()-t0:.0f}s] xong {name} · pmi_TB={mean:.3f}", flush=True)

    print("\n=== PMI vote-primary ===")
    for name, *_ in jobs:
        rs = [r for r in rows if r[0] == name]
        print(f"  {name:28s} pmi={sum(r[3] for r in rs)/len(rs):.3f}")

    pref = str(_ROOT / args.out_prefix)
    rng2 = random.Random(20260730)
    ids = [f"V{n:03d}" for n in range(len(rows))]
    rng2.shuffle(ids)
    key, blind = {}, []
    for sid, (name, p, txt, sc) in zip(ids, rows):
        key[sid] = {"config": name, "prompt": p, "pmi": round(sc, 4)}
        blind.append((sid, p, txt))
    blind.sort(key=lambda r: r[0])
    with open(f"{pref}_samples.txt", "w", encoding="utf-8") as f:
        f.write("# CHẤM MÙ vote chủ đạo — không lộ cấu hình.\n\n")
        for sid, p, txt in blind:
            f.write(f"[{sid}]  PROMPT: {p}\n        SINH  : {txt}\n\n")
    with open(f"{pref}_key.json", "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    with open(f"{pref}_run.log", "w", encoding="utf-8") as f:
        for name, *_ in jobs:
            rs = [r for r in rows if r[0] == name]
            f.write(f"{name}\tpmi={sum(r[3] for r in rs)/len(rs):.4f}\n")
    print(f"[{time.time()-t0:.0f}s] → {pref}_samples.txt", flush=True)


if __name__ == "__main__":
    main()
