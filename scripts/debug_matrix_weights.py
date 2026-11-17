#!/usr/bin/env python3
"""
Debug trọng số ma trận continuation — từng kênh + đề xuất sửa weight.

    PYTHONPATH=src python -u scripts/debug_matrix_weights.py \
        --log-file out/debug_matrix_weights.log
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402
from nn_aivie.hca.attention_qk import build_directed_ppmi, ensure_directed_qk  # noqa: E402
from nn_aivie.hca.context_positional import (  # noqa: E402
    content_token_ids,
    distant_content_wseq_micro,
    ensure_varlm,
    ngram_varlm_micro,
    order2_micro,
    positional_wseq_micro,
)


def _topk(vocab, sc: np.ndarray, k: int = 8):
    idx = np.argpartition(-sc, min(k, len(sc) - 1) - 0)[:k]
    idx = idx[np.argsort(-sc[idx])]
    return [(vocab.token_of(int(i)), float(sc[i])) for i in idx if sc[i] > 0]


def _rank(sc: np.ndarray, gold: int) -> int:
    if gold < 0 or gold >= len(sc):
        return -1
    return int(np.sum(sc > sc[gold])) + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", default="out/debug_matrix_weights.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    ckpt = _ROOT / args.checkpoint
    gc.disable()
    state, _ = load_checkpoint(str(ckpt))
    gc.enable()
    state._checkpoint_path = str(ckpt)  # type: ignore[attr-defined]
    vlm = ensure_varlm(state, fallbacks=[_ROOT / "checkpoints/network_v11.varlm.pkl"])
    vocab = state.vocab
    n = len(vocab)
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    log(f"varlm={vlm is not None} max_ctx={getattr(vlm,'max_ctx',None)}")

    # --- A. Hub mass trên W_seq raw vs PPMI ---
    log("\n══ A. W_seq raw vs PPMI — hub sau Ċ / sau Quyền ══")
    ppmi = build_directed_ppmi(state)
    for label, tid in (("Ċ", vocab.encode_prompt("\n")[0] if False else None),):
        pass
    # find newline token
    nl = None
    for i in range(n):
        if vocab.token_of(i) in ("Ċ", "\n", "ĠĊ"):
            nl = i
            break
    # often Ċ is id 437 from earlier debug
    if nl is None:
        nl = 437
    quyen = list(vocab.encode_prompt("Quyền"))[-1]
    dao = list(vocab.encode_prompt("Đạo"))[-1]

    def row_raw(i):
        sc = np.zeros(n)
        for j, w in seq.row(i).items():
            if j < n:
                sc[j] = w
        return sc

    def row_ppmi(i):
        sc = np.zeros(n)
        row = ppmi.getrow(i)
        for j, w in zip(row.indices, row.data):
            if j < n:
                sc[j] = w
        return sc

    for name, tid in (("Ċ", nl), ("Quyền", quyen), ("Đạo", dao)):
        raw, ppm = row_raw(tid), row_ppmi(tid)
        log(f"\n  {name!r} id={tid}")
        log(f"    RAW top:  {_topk(vocab, raw, 6)}")
        log(f"    PPMI top: {_topk(vocab, ppm, 6)}")
        # entropy / concentration
        def conc(sc):
            s = sc[sc > 0]
            if s.size == 0:
                return 0, 0
            p = s / s.sum()
            return float(-(p * np.log(p + 1e-12)).sum()), float(p.max())

        er, cr = conc(raw)
        ep, cp = conc(ppm)
        log(f"    RAW  H={er:.2f} peak={cr:.3f}  |  PPMI H={ep:.2f} peak={cp:.3f}")

    # --- B. Kênh tại đầu essay (no agenda) ---
    prompt = "Viết một đoạn văn hài hước về Ngô Quyền"
    hist = list(vocab.encode_prompt(prompt))
    content = content_token_ids(hist, vocab)
    log(f"\n══ B. Kênh tại hết prompt {prompt!r} ══")
    log(f"  content={ [vocab.token_of(t) for t in content] }")
    ch = {
        "pos_wseq": positional_wseq_micro(state, hist, vocab_size=n, lam=0.85, ctx_len=12),
        "varlm": ngram_varlm_micro(state, hist, vocab_size=n),
        "order2": order2_micro(state, hist, vocab_size=n),
        "distant": distant_content_wseq_micro(state, content, vocab_size=n),
    }
    # PPMI positional
    ppm_pos = np.zeros(n)
    ctx = hist[-12:]
    ages = np.arange(len(ctx) - 1, -1, -1)
    for tid, age in zip(ctx, ages):
        w = 0.85 ** float(age)
        row = ppmi.getrow(int(tid))
        for j, c in zip(row.indices, row.data):
            if j < n and c > 0:
                ppm_pos[j] += w * float(c)
    if ppm_pos.max() > 0:
        ppm_pos /= ppm_pos.max()
    ch["pos_ppmi"] = ppm_pos

    # E-similarity to mean content
    emb = state.embedding.matrix
    C = emb[np.array(content)]
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    cmean = C.mean(axis=0)
    cmean /= np.linalg.norm(cmean) + 1e-9
    En = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    ch["cos_content"] = np.maximum(En @ cmean, 0.0)

    for name, sc in ch.items():
        log(f"  {name:12} top={_topk(vocab, sc, 6)}")

    # blend hiện tại vs blend PPMI
    blend = (
        ch["pos_wseq"]
        + 1.0 * ch["varlm"]
        + 0.5 * ch["order2"]
        + 0.45 * ch["distant"]
    )
    blend_ppmi = (
        ch["pos_ppmi"]
        + 1.0 * ch["varlm"]
        + 0.5 * ch["order2"]
        + 0.35 * ch["cos_content"]
    )
    log(f"  blend_raw   top={_topk(vocab, blend, 6)}")
    log(f"  blend_ppmiE top={_topk(vocab, blend_ppmi, 6)}")

    # --- C. Sau khi đã sinh ',' hoặc '.' — kênh nào kéo vào salad ---
    for extra in ([","], [".", "Ċ"], list(vocab.encode_prompt(", một"))):
        h2 = hist + [vocab.encode_prompt(t)[-1] if isinstance(t, str) and len(vocab.encode_prompt(t)) else t for t in extra]
        # simplify: encode joined
        pass
    for suffix in [",", ".\n", ", một nhận"]:
        h2 = hist + list(vocab.encode_prompt(suffix))
        log(f"\n══ C. sau prompt+{suffix!r}  w0={vocab.token_of(h2[-1])!r} ══")
        pos = positional_wseq_micro(state, h2, vocab_size=n)
        vl = ngram_varlm_micro(state, h2, vocab_size=n)
        dist = distant_content_wseq_micro(state, content, vocab_size=n)
        # ppmi pos
        ppm_pos = np.zeros(n)
        ctx = h2[-12:]
        for i, tid in enumerate(ctx):
            w = 0.85 ** (len(ctx) - 1 - i)
            row = ppmi.getrow(int(tid))
            for j, c in zip(row.indices, row.data):
                if j < n and c > 0:
                    ppm_pos[j] += w * float(c)
        if ppm_pos.max() > 0:
            ppm_pos /= ppm_pos.max()
        log(f"  pos_wseq  {_topk(vocab, pos, 5)}")
        log(f"  varlm     {_topk(vocab, vl, 5)}")
        log(f"  distant   {_topk(vocab, dist, 5)}")
        log(f"  pos_ppmi  {_topk(vocab, ppm_pos, 5)}")
        br = pos + vl + 0.45 * dist
        bp = ppm_pos + vl + 0.35 * ch["cos_content"]
        log(f"  → blend_raw argmax={vocab.token_of(int(np.argmax(br)))!r}  "
            f"blend_ppmiE argmax={vocab.token_of(int(np.argmax(bp)))!r}")

    # --- D. Agenda gold: ma trận có nâng được không ---
    from nn_aivie.hca.config import HCAConfig
    from nn_aivie.hca.generate_hebbian import _build_segment_plan

    plan = _build_segment_plan(
        state, hist, HCAConfig(ar_multizone=True), max_facts=2, emitted_ids=hist
    )
    agenda = plan.flat_agenda
    if len(agenda) > 20:
        # prefix agenda as if writing fact — gold = next
        for k in (4, 8, 12):
            h = agenda[:k]
            gold = agenda[k]
            pos = positional_wseq_micro(state, h, vocab_size=n)
            ppm_pos = np.zeros(n)
            for i, tid in enumerate(h[-12:]):
                w = 0.85 ** (min(11, len(h[-12:])) - 1 - i)
                row = ppmi.getrow(int(tid))
                for j, c in zip(row.indices, row.data):
                    if j < n:
                        ppm_pos[j] += w * float(c)
            if ppm_pos.max() > 0:
                ppm_pos /= ppm_pos.max()
            vl = ngram_varlm_micro(state, h, vocab_size=n)
            log(f"\n══ D. agenda-prefix k={k} gold={vocab.token_of(gold)!r} ══")
            log(f"  rank raw_pos={_rank(pos, gold)}  ppmi_pos={_rank(ppm_pos, gold)}  "
                f"varlm={_rank(vl, gold)}")

    # qk directed
    qk = ensure_directed_qk(state, dim=64)
    qs = np.maximum(qk.scores(quyen), 0.0)
    log(f"\n══ E. q·k(Quyền) top={_topk(vocab, qs, 6)} ══")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
