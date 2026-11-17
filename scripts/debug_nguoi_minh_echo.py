#!/usr/bin/env python3
"""Debug case Người Minh echo trên v10 paragraph checkpoint."""

from __future__ import annotations

from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import (
    _agenda_ids_from_parent_index,
    _completion_from_parent,
    _longest_suffix_cut,
    _query_content_folds,
    _split_frame,
    generate,
    generate_autoregressive,
)
from nn_aivie.hca.readout import decode_surface
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Người Minh dùng hỏa tiễn bắn voi"
CKPT = _ROOT / "checkpoints/network_v10_histwiki_paragraph.pkl"


def main() -> None:
    log = get_run_logger("debug_nguoi_minh")
    with log.phase("load"):
        state, _ = load_checkpoint(CKPT, logger=log)
    ps = state.parents
    gen_ids = list(state.vocab.encode_tokens(PROMPT.split()))
    q_ids, ans_so_far = _split_frame(gen_ids, state.vocab)
    query = q_ids if q_ids else gen_ids

    log.log(f"prompt={PROMPT!r}")
    log.log(f"gen_ids n={len(gen_ids)} q_ids n={len(q_ids)} ans_so_far n={len(ans_so_far)}")

    ranked = ps.rank_matches(query, state.vocab, state.embedding, fuzzy_topk=0, top_n=5)
    log.log("top-5 rank_matches:")
    for pi, sc in ranked:
        rec = ps.parents[pi]
        src = list(rec.source_ids)
        emit = list(ps.completion(pi))
        agenda = _agenda_ids_from_parent_index(ps, pi, gen_ids, state.vocab)
        qf = _query_content_folds(gen_ids, state.vocab)
        cut, mk = _longest_suffix_cut(src, qf, state.vocab.fold_group_id, min_k=1)
        tail_surf = decode_surface(state, src[cut + 1 :]) if cut >= 0 else ""
        log.log(
            f"  pi={pi} sc={sc:.3f} pri={rec.priority:.2f} "
            f"src_len={len(src)} emit_len={len(emit)} cut={cut} mk={mk}"
        )
        log.log(f"    source: {decode_surface(state, src)[:120]!r}")
        log.log(f"    emit:   {decode_surface(state, emit)[:120]!r}")
        log.log(f"    tail:   {tail_surf!r}")
        log.log(f"    agenda: {decode_surface(state, agenda)!r}")

    m = ps.best_match(
        query, state.vocab, state.embedding,
        fuzzy_topk=0, qa_completion_mode=True,
    )
    log.log(f"best_match={m}")
    if not m:
        return

    pi = m[0]
    rec = ps.parents[pi]
    completion = _completion_from_parent(ps, pi)
    agenda = _agenda_ids_from_parent_index(ps, pi, gen_ids, state.vocab)
    log.log(f"chosen pi={pi} is_qa={rec.is_qa} match_span={rec.match_span}")
    log.log(f"completion={decode_surface(state, list(completion))!r}")
    log.log(f"agenda_for_AR={decode_surface(state, agenda)!r}")

    with log.phase("generate"):
        out_ids = generate(state, PROMPT, max_len=48, multi=True)
    out = decode_surface(state, out_ids)
    log.log(f"OUT={out!r}")
    log.log(f"echo_count={out.count(PROMPT)}")

    # AR only với agenda đã cắt — tách retrieval khỏi decode
    from nn_aivie.hca.generate_hebbian import _SegmentPlan

    q_content = [t for t in gen_ids if t]
    plan = _SegmentPlan([agenda], ps.entity_register_ids(tuple(q_content)), list(agenda))
    with log.phase("generate_autoregressive_only"):
        ar_ids = generate_autoregressive(
            state, PROMPT, max_len=48, plan=plan, config=HCAConfig(),
        )
    ar_out = decode_surface(state, ar_ids)
    log.log(f"AR_only={ar_out!r}")
    log.log("DONE")


if __name__ == "__main__":
    main()
