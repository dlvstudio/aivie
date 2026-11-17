#!/usr/bin/env python3
"""Chẩn QA answer-copy trên v12 — TK_045 §12.40 tiếp.

Vì sao T4 chỉ ~3/10 đọc được khi bật `ar_matrix_qa_answer_copy`?
Với mỗi prompt T4: in top-act cha, is_qa, phủ, span copy (nếu có), và vì sao cổng từ chối.

    PYTHONPATH=src python -u scripts/diag_qa_answer_copy.py \\
        --checkpoint checkpoints/network_v12_sent.pkl \\
        --log-file out/diag_qa_answer_copy_v12.log
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
from nn_aivie.core.tokenizer import DAP_LABEL  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.matrix_core import (  # noqa: E402
    _path_fold,
    answer_span_from_active,
    emit_match_ratio,
    ensure_parent_bipartite,
    matrix_core_logits,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.sent_intra import clear_sent_intra_cache  # noqa: E402

# Cùng bộ T4 của eval_gen_broad (§12.38).
PROMPTS = [
    "Hỏi: Nhà Lý dời đô về đâu? Đáp:",
    "Hỏi: Ai đánh bại quân Nam Hán trên sông Bạch Đằng? Đáp:",
    "Hỏi: Hoa Lư nằm ở tỉnh nào? Đáp:",
    "Hỏi: Lê Lợi khởi nghĩa ở đâu? Đáp:",
    "Hỏi: Ngô Quyền đánh bại quân Nam Hán năm nào? Đáp:",
    "Hỏi: Kinh đô đầu tiên của nước Đại Cồ Việt là gì? Đáp:",
    "Hỏi: Ai chỉ huy trận Bạch Đằng năm 1288? Đáp:",
    "Hỏi: Nhà Lý dời đô năm nào? Đáp:",
    "Hỏi: Đinh Bộ Lĩnh đặt tên nước là gì? Đáp:",
    "Hỏi: Lê Hoàn đánh bại quân nào trên sông Bạch Đằng? Đáp:",
]


def _tok(v, t: int) -> str:
    try:
        return v.token_of(int(t))
    except Exception:
        return f"<{t}>"


def _explain_gate(state, act, history_ids, *, form_min: float, gidf_hi: float = 4.0,
                  min_cover: float = 0.8, min_match: int = 3) -> dict:
    """Giải thích vì sao answer_span_from_active trả [] (hoặc sẽ chép gì)."""
    ps = state.parents
    bi = ensure_parent_bipartite(state)
    vocab = state.vocab
    pi = int(np.argmax(act))
    rec = ps.parents[pi]
    out: dict = {
        "pi": pi,
        "act": float(act[pi]),
        "is_qa": bool(getattr(rec, "is_qa", False)),
        "match": decode_surface(state, list(rec.match_ids))[:120],
        "emit": decode_surface(
            state,
            list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids),
        )[:120],
    }
    if rec.is_qa and rec.answer_ids:
        mfolds = {_path_fold(vocab, int(t)) for t in rec.match_ids}
        mfolds.discard("")
        hfolds = {_path_fold(vocab, int(t)) for t in history_ids}
        cover = (len(mfolds & hfolds) / len(mfolds)) if mfolds else 0.0
        out["gate"] = "qa_cover"
        out["n_match"] = len(mfolds)
        out["cover"] = round(cover, 3)
        out["ok"] = len(mfolds) >= min_match and cover >= min_cover
        if len(mfolds) < min_match:
            out["reject"] = f"match_ids quá ngắn ({len(mfolds)}<{min_match})"
        elif cover < min_cover:
            out["reject"] = f"history phủ match_ids {cover:.2f}<{min_cover}"
        return out

    # statement path
    er = emit_match_ratio(bi)
    q: set[str] = set()
    dropped: list[str] = []
    for t in history_ids:
        t = int(t)
        if not (0 <= t < bi.A.shape[0]):
            continue
        s = _tok(vocab, t)
        if bi.A.getrow(t).nnz == 0:
            dropped.append(f"{s}:A=0")
            continue
        if er[t] < form_min:
            dropped.append(f"{s}:form={er[t]:.2f}")
            continue
        if float(ps._gidf(t)) < gidf_hi:  # noqa: SLF001
            dropped.append(f"{s}:gidf={ps._gidf(t):.1f}")  # noqa: SLF001
            continue
        f = _path_fold(vocab, t)
        if f:
            q.add(f)
    pfolds = {_path_fold(vocab, int(t)) for t in rec.emit_ids}
    cover = (len(q & pfolds) / len(q)) if q else 0.0
    out["gate"] = "statement_cover"
    out["q_content"] = sorted(q)
    out["n_q"] = len(q)
    out["cover"] = round(cover, 3)
    out["dropped"] = dropped[:12]
    out["ok"] = len(q) >= min_match and cover >= min_cover
    if len(q) < min_match:
        out["reject"] = f"nội dung câu hỏi còn {len(q)} token (<{min_match}) sau lọc form/gidf"
    elif cover < min_cover:
        missing = sorted(q - pfolds)
        out["reject"] = f"cha phủ câu hỏi {cover:.2f}<{min_cover}; thiếu={missing[:8]}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/diag_qa_answer_copy_v12.log")
    ap.add_argument("--form-min", type=float, default=0.9)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    logf = (_ROOT / args.log_file).open("w", encoding="utf-8")
    t0 = time.time()

    def log(m: object = "") -> None:
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    ensure_parent_bipartite(state)
    n_qa = sum(1 for r in state.parents.parents if getattr(r, "is_qa", False))
    log(f"[{time.time()-t0:.0f}s] ✓ {len(state.parents):,} cha · is_qa={n_qa:,}")

    clear_sent_intra_cache()
    dap = state.vocab._token_to_id.get(DAP_LABEL)  # noqa: SLF001
    cfg_on = HCAConfig(
        ar_matrix_core=True, ar_style_payoff=False, hybrid_temperature=0.0,
        ar_matrix_stack_weight=0.0, ar_no_agenda=True, ar_sent_agenda_handoff=False,
        ar_matrix_qa_answer_copy=True, ar_matrix_and_form_min=args.form_min,
    )
    cfg_off = HCAConfig(
        ar_matrix_core=True, ar_style_payoff=False, hybrid_temperature=0.0,
        ar_matrix_stack_weight=0.0, ar_no_agenda=True, ar_sent_agenda_handoff=False,
        ar_matrix_qa_answer_copy=False,
    )

    n_copy = n_readable = 0
    for i, prompt in enumerate(PROMPTS):
        pids = list(state.vocab.encode_prompt(prompt))
        if dap is None or not pids or int(pids[-1]) != int(dap):
            log(f"\n[{i}] SKIP (không kết Đáp:): {prompt!r}")
            continue

        lock: dict = {}
        matrix_core_logits(state, pids, config=cfg_on, lock_out=lock)
        act = lock.get("act")
        info = _explain_gate(state, act, pids, form_min=args.form_min)
        span = answer_span_from_active(
            state, act, pids, form_min=args.form_min, statement=True,
        )
        out_on = generate(state, prompt, max_len=48, config=cfg_on, seed=42)
        out_off = generate(state, prompt, max_len=48, config=cfg_off, seed=42)
        cont_on = decode_surface(state, list(out_on[len(pids):]))
        cont_off = decode_surface(state, list(out_off[len(pids):]))
        readable = len(out_on) - len(pids) >= 4
        n_copy += int(bool(span))
        n_readable += int(readable)

        # top-k act
        tops = []
        if act is not None and len(act):
            order = np.argsort(-act)[: args.top]
            for pi in order:
                if float(act[pi]) <= 0:
                    break
                r = state.parents.parents[int(pi)]
                tops.append(
                    f"pi={int(pi)} act={float(act[pi]):.3f} qa={int(r.is_qa)} "
                    f"match={decode_surface(state, list(r.match_ids))[:60]!r}"
                )

        log(f"\n══ [{i}] {prompt}")
        log(f"  top-act: {tops[0] if tops else '(none)'}")
        for line in tops[1:]:
            log(f"           {line}")
        log(f"  gate={info.get('gate')} ok={info.get('ok')} cover={info.get('cover')} "
            f"n={info.get('n_q', info.get('n_match'))} reject={info.get('reject')}")
        if info.get("q_content") is not None:
            log(f"  q_content={info['q_content']}")
            log(f"  dropped={info.get('dropped')}")
        log(f"  span_len={len(span)} span={decode_surface(state, span)[:100]!r}")
        log(f"  OFF → {cont_off!r}")
        log(f"  ON  → {cont_on!r}  readable={readable}")

    log(f"\n── tổng: copy_fire={n_copy}/10 · readable(n_tok≥4)={n_readable}/10 "
        f"· form_min={args.form_min}")
    log(f"[{time.time()-t0:.0f}s] → {args.log_file}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
