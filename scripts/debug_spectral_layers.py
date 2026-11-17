#!/usr/bin/env python3
"""
DEBUG từng tầng spectral stack — luôn ar_no_agenda.

    PYTHONPATH=src python -u scripts/debug_spectral_layers.py \
        --log-file out/debug_spectral_layers.log

Mỗi bước decode (continuation): W_seq top vs Wa_Lℓ top; α/canal từng tầng;
argmax trước/sau blend; overlap với successor thô.
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
from nn_aivie.core.vocab import SPECIAL_EOS, is_surface_token  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import _wseq_fallback  # noqa: E402
from nn_aivie.hca.readout import (  # noqa: E402
    apply_repetition_penalty,
    mask_non_surface,
    multiply_scores_by_boost,
    spreading_activation_read,
)
from nn_aivie.hca.spectral_stack import (  # noqa: E402
    ensure_spectral_stack,
    forward_spectral_stack,
    wa_successor_scores,
)


def _tok(vocab, i: int) -> str:
    return vocab.token_of(int(i))


def _fmt_pairs(vocab, pairs: list[tuple[int, float]], k: int = 8) -> str:
    parts = []
    for i, s in pairs[:k]:
        parts.append(f"{_tok(vocab, i)!r}:{s:.3f}")
    return "  ".join(parts)


def _topk_ids(scores: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
    finite = np.isfinite(scores)
    if not finite.any():
        return []
    s = np.where(finite, scores, -np.inf)
    k = min(k, int(finite.sum()))
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]
    return [(int(i), float(s[i])) for i in idx]


def _jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _one_step_scores(state, out: list[int], config: HCAConfig, stack, *, use_stack: bool):
    """Tái lập 1 bước _wseq_fallback (+ optional stack debug)."""
    vocab_size = len(state.vocab)
    w0 = out[-1]
    base = np.zeros(vocab_size, dtype=np.float64)
    for v, w in state.adjacency.row_readout(w0).items():
        if v < vocab_size and w > 0:
            base[v] = w
    micro = np.zeros(vocab_size, dtype=np.float64)
    for v, w in state.adjacency.layer(EdgeKind.SEQUENTIAL).row(w0).items():
        if v < vocab_size and w > 0:
            micro[v] = w
    mmax = micro.max() or 1.0
    micro_n = micro / mmax
    scores = base * (config.hybrid_micro_floor + config.hybrid_micro_weight * micro_n)
    topic_seeds = [
        (t, 1.0) for t in dict.fromkeys(out) if state.vocab.is_surface(t)
    ]
    if config.ar_continuation_topic_weight > 0 and topic_seeds:
        topic = spreading_activation_read(
            w0, topic_seeds, state.adjacency, config, vocab_size
        )
        tmax = topic.max() or 1.0
        scores = scores * (
            config.ar_continuation_topic_floor
            + config.ar_continuation_topic_weight * topic / tmax
        )
    before = scores.copy()
    dbg = None
    if use_stack and stack is not None:
        emb = getattr(state, "embedding", None)
        emb_mat = getattr(emb, "matrix", None) if emb is not None else None
        a_out, dbg = forward_spectral_stack(
            stack,
            history_ids=out,
            candidate_ids=None,
            entity_ids=(),
            emb_matrix=emb_mat,
            floor=float(config.ar_spectral_stack_floor),
            debug=True,
        )
        if a_out.shape[0] != vocab_size:
            a_out = np.pad(a_out, (0, max(0, vocab_size - a_out.shape[0])))[:vocab_size]
        multiply_scores_by_boost(
            scores,
            a_out,
            floor=float(config.ar_spectral_stack_floor),
            weight=float(config.ar_spectral_stack_weight),
        )
    scores = mask_non_surface(scores, state.vocab)
    scores = apply_repetition_penalty(scores, out, state.vocab, prompt_len=len(out))
    eos = state.vocab.id_of(SPECIAL_EOS)
    if eos < len(scores):
        fm = scores[np.isfinite(scores)].max() if np.any(np.isfinite(scores)) else 0.0
        if state.adjacency.get_readout(w0, eos) >= config.hybrid_eos_threshold:
            scores[eos] = float(fm) * 1.5 + 1.0
        else:
            scores[eos] = -np.inf
    return before, scores, dbg, micro


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11_styled_hist.pkl")
    ap.add_argument("--log-file", default="out/debug_spectral_layers.log")
    ap.add_argument("--prompt", default="Viết hài hước về Trần Hưng Đạo")
    ap.add_argument("--steps", type=int, default=6)
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
    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(ckpt))
    gc.enable()
    state._spectral_stack_sidecar = str(sidecar)  # type: ignore[attr-defined]
    stack = ensure_spectral_stack(state, d=args.d, sidecar=sidecar, logger=log)
    vocab = state.vocab
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)

    # ── A. Năng lượng phổ / tầng (closed-form) ──
    log("\n══ A. Năng lượng phổ mỗi tầng (Σ Wffn + ‖Wa‖F) ══")
    for i, ly in enumerate(stack.layers):
        log(
            f"  L{i}: ‖wq‖F={np.linalg.norm(ly.wq):.2f}  "
            f"‖wk‖F={np.linalg.norm(ly.wk):.2f}  "
            f"Σs_ffn={ly.s_ffn.sum():.2f}  s_max={ly.s_ffn.max():.2f}  "
            f"n_keep={ly.n_keep}"
        )

    # ── B. Successor: W_seq vs từng Wa trên vài seed ──
    seeds = list(vocab.encode_prompt(args.prompt))
    w0 = seeds[-1]
    log(f"\n══ B. Successor từ w0={_tok(vocab, w0)!r} (cuối prompt) ══")
    row = seq.row(w0)
    raw = np.zeros(len(vocab), dtype=np.float64)
    for j, w in row.items():
        if j < len(raw):
            raw[j] = w
    raw_top = _topk_ids(raw, 12)
    log(f"  W_seq raw top: {_fmt_pairs(vocab, raw_top)}")
    for ℓ in range(len(stack.layers)):
        sc = wa_successor_scores(stack, ℓ, w0)
        sc = np.maximum(sc, 0.0)
        top = _topk_ids(sc, 12)
        jac = _jaccard([i for i, _ in raw_top[:10]], [i for i, _ in top[:10]])
        # rank of best raw successor in Wa
        if raw_top:
            best_j = raw_top[0][0]
            better = int(np.sum(sc > sc[best_j]))
            rank = better + 1
        else:
            rank = -1
        log(
            f"  Wa_L{ℓ} top: {_fmt_pairs(vocab, top)}  "
            f"jaccard@10={jac:.2f}  rank(best_raw)={rank}"
        )

    # ── C. Cột phổ: token hub (degree cao trong |U| / |V|) ──
    log("\n══ C. Hub phổ — token |wq| / |wk| lớn nhất mỗi tầng (5) ══")
    for ℓ, ly in enumerate(stack.layers):
        left = np.argsort(-np.linalg.norm(ly.wq, axis=1))[:5]
        right = np.argsort(-np.linalg.norm(ly.wk, axis=1))[:5]
        log(
            f"  L{ℓ} query-hub: {[repr(_tok(vocab, i)) for i in left]}"
        )
        log(
            f"  L{ℓ} key-hub:   {[repr(_tok(vocab, i)) for i in right]}"
        )

    # ── D. Bước decode no-agenda: dump từng tầng ──
    cfg = HCAConfig(
        ar_no_agenda=True,
        ar_spectral_stack=True,
        ar_spectral_stack_weight=0.55,
        ar_spectral_stack_floor=0.15,
        ar_spectral_stack_d=args.d,
        ar_pattern_completion=False,
        ar_conv_pool=False,
    )
    cfg0 = HCAConfig(
        ar_no_agenda=True,
        ar_spectral_stack=False,
        ar_pattern_completion=False,
        ar_conv_pool=False,
    )
    out = list(seeds)
    log(f"\n══ D. Decode no-agenda — {args.steps} bước đầu ══")
    log(f"  prompt: {args.prompt!r}")

    for step in range(args.steps):
        before, after, dbg, micro = _one_step_scores(
            state, out, cfg, stack, use_stack=True
        )
        _, after0, _, _ = _one_step_scores(state, out, cfg0, None, use_stack=False)
        w0 = out[-1]
        log(f"\n── step {step}  w0={_tok(vocab, w0)!r}  |out|={len(out)} ──")
        log(f"  W_seq/micro top: {_fmt_pairs(vocab, _topk_ids(micro, 8))}")
        log(f"  scores A3 top:   {_fmt_pairs(vocab, _topk_ids(after0, 8))}")
        log(f"  scores+stack top:{_fmt_pairs(vocab, _topk_ids(after, 8))}")
        a3_arg = int(np.argmax(np.where(np.isfinite(after0), after0, -np.inf)))
        st_arg = int(np.argmax(np.where(np.isfinite(after), after, -np.inf)))
        log(
            f"  argmax A3={_tok(vocab, a3_arg)!r}  stack={_tok(vocab, st_arg)!r}  "
            f"same={a3_arg == st_arg}"
        )
        if dbg is not None:
            log(f"  |S|={dbg.get('|S|')}  out_nnz={dbg.get('out_nnz')}")
            # S có bao nhiêu token surface / punct
            n_surf = sum(1 for i in dbg["S"] if is_surface_token(_tok(vocab, i)))
            log(f"  S surface={n_surf}/{len(dbg['S'])}")
            log(f"  a_out top: {_fmt_pairs(vocab, dbg.get('mix_top', []))}")
            log(f"  succ_L0 top: {_fmt_pairs(vocab, dbg.get('succ_l0_top', []))}")
            for L in dbg["layers"]:
                log(
                    f"  L{L['ℓ']}: H={L['alpha_entropy']:.2f}  "
                    f"z={L['z_l2']:.3f}  drop={L['dropped']}  "
                    f"ov_h0={L['overlap_h0']:.3f}  "
                    f"Σs={L['s_ffn_sum']:.2f}"
                )
                log(f"      α: {_fmt_pairs(vocab, L['alpha_top'], 6)}")
                log(f"      h: {_fmt_pairs(vocab, L['h_top'], 6)}")

        # chọn theo stack path (giống gen khi flag on)
        if not np.any(np.isfinite(after)) or float(np.nanmax(after)) <= 0:
            log("  BREAK: no finite scores")
            break
        nxt = st_arg
        if nxt == vocab.id_of(SPECIAL_EOS):
            log("  EOS")
            break
        out.append(nxt)
        log(f"  → emit {_tok(vocab, nxt)!r}")

    # full short gen A3 vs stack
    log("\n══ E. Gen ngắn no-agenda (A3 vs stack) ══")
    eos = vocab.id_of(SPECIAL_EOS)
    rng = np.random.default_rng(0)
    for name, c in (("A3", cfg0), ("stack", cfg)):
        ids = _wseq_fallback(state, list(seeds), eos, 40, c, rng)
        tail = ids[len(seeds) :]
        text = state.vocab.detokenize(tail) if tail else ""
        log(f"  {name}: {text[:240]!r}")

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
