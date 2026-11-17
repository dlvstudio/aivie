"""
Train pipeline — một luồng: mỗi dòng jsonl = một text → graph (cạnh + cha + ISG + skip ledger).

TK_045: tuỳ chọn `sentence_parents` — 1 câu = 1 parent + episode hub theo bài.
"""

from __future__ import annotations

import time
from pathlib import Path

from nn_aivie.core.embedding import build_embedding
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import SPECIAL_EOS
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.data.quality import normalize_ws, normalize_ws_keep_newlines
from nn_aivie.training.dataset_spec import DatasetSpec
from nn_aivie.training.g1_batch import begin_g1_batch, flush_g1_batch, g1_and_wseq
from nn_aivie.training.ingest import encode_unit
from nn_aivie.training.text_corpus import iter_texts
from nn_aivie.teaching.align import tokenize_punct


def _strip_trailing_eos(ids: list[int], eos_id: int | None) -> list[int]:
    if eos_id is not None and ids and int(ids[-1]) == int(eos_id):
        return list(ids[:-1])
    return list(ids)


def _with_eos(ids: list[int] | tuple[int, ...], eos_id: int | None) -> list[int]:
    out = [int(t) for t in ids]
    if eos_id is None:
        return out
    if out and int(out[-1]) == int(eos_id):
        return out
    return [*out, int(eos_id)]


def _is_qa_unit(ids: list[int], text: str, dap_id: int | None) -> bool:
    if dap_id is not None and dap_id in ids:
        return True
    t = text.strip()
    return t.startswith("Hỏi:") or "Đáp:" in t


def ingest_text_file(
    state: GraphState,
    corpus_path: Path | str,
    spec: DatasetSpec,
    *,
    logger=None,
    log_every: int = 2000,
    verbose: bool = False,
    progress_jsonl: Path | str | None = None,
    defer_skip: bool = False,
    sentence_parents: bool = False,
    sentence_min_content: int = 3,
    sentence_max_tokens: int = 64,
    sentence_max_spans: int = 32,
    episode_groups: list | None = None,
) -> dict[str, int]:
    """Nạp file text_corpus — mỗi dòng một chuỗi JSON.

    `sentence_parents=True` (TK_045): tách câu → mỗi câu một parent; bài ≥2 câu → episode.
    G1/W_seq vẫn học trên **cả bài** (một lần) để giữ bigram xuyên câu.
    """
    from nn_aivie.core.parent_store import ParentStore
    from nn_aivie.hca.sentence_parents import split_emit_to_sentence_seqs
    from nn_aivie.training.train_progress import (
        append_progress_jsonl,
        count_corpus_lines,
        format_item_log,
        global_graph_stats,
        parent_item_stats,
        text_label,
    )

    path = Path(corpus_path)
    if not path.is_absolute():
        from nn_aivie.training.dataset_spec import REPO_ROOT

        path = REPO_ROOT / path
    ps = state.parents
    if ps is None:
        ps = ParentStore()
        ps.ensure_skip_ledger(min_count=4, min_pmi=1.0)
        state.parents = ps
    else:
        ps.ensure_skip_ledger(min_count=4, min_pmi=1.0)

    total_lines = count_corpus_lines(path) if verbose else 0
    if verbose and logger is not None:
        logger.log(f"  {path.name}: {total_lines} đơn vị (bài/câu)")

    dap_id = state.vocab.id_of(DAP_LABEL)
    eos_id = state.vocab.id_of(SPECIAL_EOS)
    is_bpe = state.vocab.tokenizer is not None
    tok = tokenize_punct if spec.split_punct else lambda s: s.split()
    n = 0
    n_split = 0
    n_sent_parents = 0
    begin_g1_batch(state)
    ps.begin_ingest_batch()
    pending_verbose: list[tuple[str, tuple[int, ...], bool, int]] = []
    ingest_hb = time.monotonic()
    for k, text in enumerate(iter_texts(path)):
        if spec.max_texts and k >= spec.max_texts:
            break
        text = normalize_ws_keep_newlines(text) if is_bpe else normalize_ws(text)
        ids = encode_unit(state, text, legacy_tok=tok)
        if len(ids) < 3:  # ≥2 token nội dung + EOS
            continue

        use_split = sentence_parents and not _is_qa_unit(ids, text, dap_id)
        if use_split:
            body = _strip_trailing_eos(ids, eos_id)
            sents = split_emit_to_sentence_seqs(
                body,
                state.vocab,
                min_content=sentence_min_content,
                max_tokens=sentence_max_tokens,
                # §13.54 — MỒI NGUỘI: ở đây `W_seq` đang được dựng nên tín hiệu EOS-lift
                # chưa tồn tại. Đây là chỗ DUY NHẤT còn dùng danh sách dấu câu, và nó được
                # truyền TƯỜNG MINH để không ai tưởng hệ đang học biên ở khâu này.
                bootstrap=True,
            )
            if len(sents) > sentence_max_spans:
                sents = sents[:sentence_max_spans]
            if len(sents) >= 2:
                g1_and_wseq(state, ids, spec.eta)
                member_seqs: list[tuple[int, ...]] = []
                for s in sents:
                    sent_ids = _with_eos(s, eos_id)
                    if len(sent_ids) < 3:
                        continue
                    key = tuple(sent_ids)
                    dup = key in ps._key_to_idx  # noqa: SLF001
                    ps.add_example(
                        sent_ids,
                        state.vocab,
                        dap_id=dap_id,
                        priority=spec.priority,
                        defer_index=True,
                    )
                    n_sent_parents += 1
                    if key in ps._key_to_idx:  # noqa: SLF001
                        member_seqs.append(key)
                    if verbose:
                        pending_verbose.append((text, key, dup, n + 1))
                if episode_groups is not None and len(member_seqs) >= 2:
                    episode_groups.append(
                        (f"{path.name}:{k}", text_label(text)[:80], list(member_seqs))
                    )
                n += 1
                n_split += 1
                if not verbose and logger is not None and (k + 1) % log_every == 0:
                    logger.progress(
                        "ingest_text",
                        k + 1,
                        spec.max_texts or k + 1,
                        extra=f"n={n} sent_p={n_sent_parents} parents={len(ps.parents)}",
                    )
                    ingest_hb = time.monotonic()
                elif logger is not None and time.monotonic() - ingest_hb >= 30.0:
                    logger.log(
                        f"ingest_text … line={k + 1} n={n} split={n_split} "
                        f"sent_parents={n_sent_parents} parents={len(ps.parents)}"
                    )
                    ingest_hb = time.monotonic()
                continue

        # --- 1 parent / đơn vị (QA, 1 câu, hoặc tắt sentence_parents) ---
        key = tuple(ids)
        dup = key in ps._key_to_idx  # noqa: SLF001
        g1_and_wseq(state, ids, spec.eta)
        ps.add_example(
            ids,
            state.vocab,
            dap_id=dap_id,
            priority=spec.priority,
            defer_index=True,
        )
        n += 1
        if verbose:
            pending_verbose.append((text, key, dup, n))
        elif logger is not None and (k + 1) % log_every == 0:
            logger.progress(
                "ingest_text", k + 1, spec.max_texts or k + 1, extra=f"n={n}"
            )
            ingest_hb = time.monotonic()
        elif logger is not None and time.monotonic() - ingest_hb >= 30.0:
            logger.log(
                f"ingest_text … line={k + 1} n={n} parents={len(ps.parents)}"
            )
            ingest_hb = time.monotonic()
    if logger is not None and n > 0:
        extra = ""
        if sentence_parents:
            extra = f" split_docs={n_split} sent_parents={n_sent_parents}"
        logger.log(f"ingest_text done — n={n} parents={len(ps.parents)}{extra}")
    if n > 0:
        flush_g1_batch(state, logger=logger)
        fin_every = log_every if log_every > 0 else 500
        ps.finalize_batch_ingest(
            state.vocab,
            logger=logger,
            log_every=fin_every,
            defer_skip=defer_skip,
        )
    if verbose and logger is not None and pending_verbose:
        ds_name = path.name
        gstats = global_graph_stats(ps, state.vocab)
        for text, key, dup, idx_n in pending_verbose:
            show = (
                log_every <= 1
                or idx_n % log_every == 0
                or idx_n == (total_lines or idx_n)
            )
            if not show:
                continue
            pi = ps._key_to_idx[key]
            label = text_label(text)
            item = parent_item_stats(ps, pi, state.vocab)
            logger.log(
                format_item_log(
                    ds_name, idx_n, total_lines or idx_n, label, item, gstats, duplicate=dup
                )
            )
            if progress_jsonl:
                append_progress_jsonl(
                    progress_jsonl,
                    {
                        "dataset": ds_name,
                        "index": idx_n,
                        "total": total_lines or idx_n,
                        "label": label,
                        "duplicate": dup,
                        **item,
                        "graph": gstats,
                    },
                )
    if verbose and logger is not None:
        g = global_graph_stats(ps, state.vocab)
        logger.log(
            f"  ✓ {path.name} xong: {n} đơn vị | "
            f"cha={g['parents']} cụm={g['clusters']} skip={g['skip_spans']}"
        )
    return {
        "texts": n,
        "split_docs": n_split,
        "sent_parents": n_sent_parents,
    }


def finalize_graph(state: GraphState, *, dim: int = 96, logger=None) -> None:
    """Sau khi nạp xong mọi dataset — embedding + slot hubs."""
    ps = state.parents
    if ps is None:
        return
    if logger is not None:
        logger.log("▶ slot_hubs")
    ih = ps.induce_slot_hubs(min_distinct_fillers=2)
    bh = ps.bind_slot_fillers()
    if logger is not None:
        logger.log(f"  hubs={ih['hubs']} bindings={bh['bound']}")
    if logger is not None:
        logger.log("▶ embedding")
    state.embedding = build_embedding(state.adjacency, state.vocab, dim=dim)
    if logger is not None:
        logger.log(f"  E shape={state.embedding.matrix.shape}")


def run_manifest(
    state: GraphState,
    specs: list[DatasetSpec],
    *,
    repo_root: Path,
    logger=None,
    dim: int = 96,
    verbose: bool = False,
    log_every: int = 0,
    progress_jsonl: Path | str | None = None,
    defer_skip: bool = True,
    sentence_parents: bool = False,
    sentence_min_content: int = 3,
    sentence_max_tokens: int = 64,
    sentence_max_spans: int = 32,
) -> dict[str, int]:
    totals: dict[str, int] = {}
    episode_groups: list = [] if sentence_parents else []
    for i, spec in enumerate(specs):
        path = Path(spec.path)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            if logger is not None:
                logger.log(f"⚠ bỏ qua (không tìm thấy): {path}")
            continue
        label = path.name
        ds_sent = sentence_parents
        if "sentence_parents" in spec.extra:
            ds_sent = bool(spec.extra["sentence_parents"])
        if logger is not None:
            logger.log(
                f"▶ [{i + 1}/{len(specs)}] {label} priority={spec.priority}"
                + (" sentence_parents=ON" if ds_sent else "")
            )
        every = log_every
        if every <= 0:
            every = 1 if verbose else 2000
        kw = dict(
            logger=logger,
            log_every=every,
            verbose=verbose,
            progress_jsonl=progress_jsonl,
            defer_skip=defer_skip,
            sentence_parents=ds_sent,
            sentence_min_content=sentence_min_content,
            sentence_max_tokens=sentence_max_tokens,
            sentence_max_spans=sentence_max_spans,
            episode_groups=episode_groups if ds_sent else None,
        )
        if logger is not None:
            with logger.phase(f"ingest {label}"):
                stats = ingest_text_file(state, path, spec, **kw)
        else:
            stats = ingest_text_file(state, path, spec, **kw)
        totals[label] = stats.get("texts", 0)
        if ds_sent and logger is not None:
            logger.log(
                f"  split_docs={stats.get('split_docs', 0)} "
                f"sent_parents_added≈{stats.get('sent_parents', 0)}"
            )
    if sentence_parents and state.parents is not None and episode_groups:
        if logger is not None:
            logger.log(f"▶ build episodes from {len(episode_groups)} groups")
        made, matched = state.parents.build_episodes_from_provenance(
            episode_groups, min_members=2
        )
        if logger is not None:
            logger.log(f"  episodes={made} matched_sents={matched}")
        totals["_episodes"] = made
    if defer_skip and state.parents is not None:
        state.parents.finalize_skip_ledger_deferred(state.vocab, logger=logger)
    finalize_graph(state, dim=dim, logger=logger)
    return totals
