#!/usr/bin/env python3
"""
DEBUG spectral stack với prompt/history SÁT agenda (vẫn đo no-agenda blend).

Lấy flat_agenda MZ thật → dùng prefix agenda làm history; so W_seq / Wa_ℓ / α / rank(gold).

    PYTHONPATH=src python -u scripts/debug_spectral_agenda_prompt.py \
        --log-file out/debug_spectral_agenda_prompt.log
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
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import _build_segment_plan  # noqa: E402
from nn_aivie.hca.readout import (  # noqa: E402
    mask_non_surface,
    multiply_scores_by_boost,
    spreading_activation_read,
)
from nn_aivie.hca.spectral_stack import (  # noqa: E402
    ensure_spectral_stack,
    forward_spectral_stack,
    spectral_forward_kwargs,
    wa_successor_scores,
)


def _topk(sc: np.ndarray, k: int = 8) -> list[tuple[int, float]]:
    s = np.where(np.isfinite(sc), sc, -np.inf)
    if not np.isfinite(s).any():
        return []
    k = min(k, int(np.isfinite(s).sum()))
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]
    return [(int(i), float(s[i])) for i in idx]


def _fmt(vocab, pairs: list[tuple[int, float]], k: int = 8) -> str:
    return "  ".join(f"{vocab.token_of(i)!r}:{v:.3f}" for i, v in pairs[:k])


def _rank(sc: np.ndarray, gold: int) -> int:
    if gold < 0 or gold >= sc.shape[0] or not np.isfinite(sc[gold]):
        return -1
    return int(np.sum(sc > sc[gold])) + 1


def _jac(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def _micro_scores(state, history: list[int], config: HCAConfig) -> np.ndarray:
    vocab_size = len(state.vocab)
    w0 = history[-1]
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    base = np.zeros(vocab_size, dtype=np.float64)
    for v, w in state.adjacency.row_readout(w0).items():
        if v < vocab_size and w > 0:
            base[v] = w
    micro = np.zeros(vocab_size, dtype=np.float64)
    for v, w in seq.row(w0).items():
        if v < vocab_size and w > 0:
            micro[v] = w
    mmax = micro.max() or 1.0
    scores = base * (config.hybrid_micro_floor + config.hybrid_micro_weight * (micro / mmax))
    seeds = [(t, 1.0) for t in dict.fromkeys(history) if state.vocab.is_surface(t)]
    if config.ar_continuation_topic_weight > 0 and seeds:
        topic = spreading_activation_read(
            w0, seeds, state.adjacency, config, vocab_size
        )
        tmax = topic.max() or 1.0
        scores = scores * (
            config.ar_continuation_topic_floor
            + config.ar_continuation_topic_weight * topic / tmax
        )
    return scores


def dump_at(
    log,
    state,
    stack,
    history: list[int],
    gold: int | None,
    label: str,
) -> None:
    vocab = state.vocab
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    w0 = history[-1]
    log(f"\n══ {label} ══")
    log(f"  H_tail={vocab.detokenize(history[-10:])!r}")
    log(f"  w0={vocab.token_of(w0)!r}  |H|={len(history)}")
    if gold is not None:
        log(f"  gold next={vocab.token_of(gold)!r}")

    raw = np.zeros(len(vocab), dtype=np.float64)
    for j, w in seq.row(w0).items():
        if j < len(raw):
            raw[j] = w
    raw_top = _topk(raw, 10)
    extra = f"  rank(gold)={_rank(raw, gold)}" if gold is not None else ""
    log(f"  W_seq: {_fmt(vocab, raw_top)}{extra}")

    for ell in range(len(stack.layers)):
        sc = np.maximum(wa_successor_scores(stack, ell, w0), 0.0)
        tp = _topk(sc, 8)
        jac = _jac([i for i, _ in raw_top[:8]], [i for i, _ in tp[:8]])
        g = f"  rank(gold)={_rank(sc, gold)}" if gold is not None else ""
        log(f"  Wa_L{ell}: {_fmt(vocab, tp)}  jac@8={jac:.2f}{g}")

    emb = state.embedding.matrix if state.embedding is not None else None
    cfg = HCAConfig()
    fwd_kw = spectral_forward_kwargs(cfg, state, history)
    # giả lập pending: nhét gold (+ vài token) vào candidates như AR có agenda
    cands = [gold] if gold is not None else None
    a_out, dbg = forward_spectral_stack(
        stack,
        history_ids=history,
        candidate_ids=cands,
        emb_matrix=emb,
        debug=True,
        **fwd_kw,
    )
    log(
        f"  |S|={dbg['|S|']}  combine={dbg.get('combine')}  "
        f"query={dbg.get('query_mode')}  a_out: {_fmt(vocab, dbg.get('mix_top', []), 8)}"
    )
    if dbg.get("succ_l0_top"):
        log(f"  succ_L0: {_fmt(vocab, dbg.get('succ_l0_top', []), 6)}")
    if gold is not None:
        log(f"  gold_in_S={gold in dbg['S']}  a_out[gold]={float(a_out[gold]):.4f}")

    for L in dbg["layers"]:
        log(
            f"  L{L['ℓ']}: Hα={L['alpha_entropy']:.2f} z={L['z_l2']:.3f} "
            f"drop={L['dropped']} ov0={L['overlap_h0']:.3f}"
        )
        log(f"      α: {_fmt(vocab, L['alpha_top'], 5)}")
        log(f"      h: {_fmt(vocab, L['h_top'], 5)}")
        if gold is not None:
            h_ids = [i for i, _ in L["h_top"]]
            a_ids = [i for i, _ in L["alpha_top"]]
            log(
                f"      gold_in_α_top5={gold in a_ids}  gold_in_h_top5={gold in h_ids}"
            )

    a3 = _micro_scores(state, history, cfg)
    st = a3.copy()
    multiply_scores_by_boost(st, a_out, floor=0.15, weight=0.55)
    a3 = mask_non_surface(a3, vocab)
    st = mask_non_surface(st, vocab)
    a3_arg = int(np.argmax(np.where(np.isfinite(a3), a3, -np.inf)))
    st_arg = int(np.argmax(np.where(np.isfinite(st), st, -np.inf)))
    log(
        f"  argmax A3={vocab.token_of(a3_arg)!r}  stack={vocab.token_of(st_arg)!r}  "
        f"same={a3_arg == st_arg}"
    )
    log(f"  A3 top: {_fmt(vocab, _topk(a3, 6))}")
    log(f"  stack top: {_fmt(vocab, _topk(st, 6))}")
    if gold is not None:
        log(f"  rank(gold) A3={_rank(a3, gold)}  stack={_rank(st, gold)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", default="out/debug_spectral_agenda_prompt.log")
    ap.add_argument("--prompt", default="Viết hài hước về Trần Hưng Đạo")
    ap.add_argument("--d", type=int, default=24)
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    ckpt = _ROOT / args.checkpoint
    sidecar = ckpt.with_suffix(".spectral_stack.npz")
    log(f"[{time.time()-t0:.0f}s] load …")
    gc.disable()
    state, _ = load_checkpoint(str(ckpt))
    gc.enable()
    state._spectral_stack_sidecar = str(sidecar)  # type: ignore[attr-defined]
    stack = ensure_spectral_stack(state, d=args.d, sidecar=sidecar, logger=log)
    vocab = state.vocab

    q = list(vocab.encode_prompt(args.prompt))
    plan = _build_segment_plan(
        state, q, HCAConfig(ar_multizone=True), max_facts=3, emitted_ids=q
    )
    agenda = list(plan.flat_agenda)
    log(f"prompt={args.prompt!r}")
    log(f"agenda n={len(agenda)} text={vocab.detokenize(agenda)[:360]!r}")
    log(f"entities={[vocab.token_of(t) for t in plan.entity_ids[:8]]}")
    if len(agenda) < 8:
        log("⚠ agenda quá ngắn — dừng")
        logf.close()
        return 1

    # A) prefix thuần agenda (giả sử đang viết đúng fact)
    for k in (4, 8, 12, 20):
        if k >= len(agenda):
            break
        hist = agenda[:k]
        gold = agenda[k]
        dump_at(
            log,
            state,
            stack,
            hist,
            gold,
            f"A agenda-prefix k={k}  H={vocab.detokenize(hist)!r}",
        )

    # B) essay prompt + prefix agenda (như đã viết setup rồi vào thân fact)
    for k in (6, 12, 20):
        if k >= len(agenda):
            break
        hist = q + agenda[:k]
        gold = agenda[k]
        dump_at(
            log,
            state,
            stack,
            hist,
            gold,
            f"B essay+agenda k={k}",
        )

    # C) so với essay-only (prompt mở — baseline xấu đã biết)
    dump_at(
        log,
        state,
        stack,
        q,
        agenda[0] if agenda else None,
        "C essay-only (gold=agenda[0] — khó)",
    )

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
