"""
TK_045 P0.1 — tách mega-parent thành 1 câu = 1 parent + episode members theo thứ tự.

Không thay pipeline ingest gốc; migrate offline từ checkpoint hiện có.
"""

from __future__ import annotations

from dataclasses import dataclass

from nn_aivie.core.parent_store import AbstractNeuron, ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.sent_intra import (
    _bound_of,
    _is_boundary,
    clear_sent_intra_cache,
    iter_sentence_spans,
)


@dataclass
class SplitStats:
    old_parents: int = 0
    new_parents: int = 0
    kept_as_is: int = 0
    split_sources: int = 0
    episodes: int = 0
    skipped_short: int = 0
    truncated_spans: int = 0


def _content_len(span: list[int], vocab: Vocabulary, bound=None, *, bootstrap=False) -> int:
    n = 0
    for t in span:
        if _is_boundary(vocab, int(t), bound, bootstrap=bootstrap):
            continue
        if vocab.is_surface(int(t)):
            n += 1
    return n


def split_emit_to_sentence_seqs(
    seq: list[int] | tuple[int, ...],
    vocab: Vocabulary,
    *,
    min_content: int = 3,
    max_tokens: int = 64,
    bound=None,
    bootstrap: bool = False,
) -> list[tuple[int, ...]]:
    """Tách emit → list câu (token ids), lọc ngắn / cắt quá dài.

    ⚠️ `bound` (§13.54) là tín hiệu biên **học được**. Không truyền ⇒ chỉ còn ngắt theo
    khoảng trắng, tức **gần như không tách câu** — cố ý ồn ào thay vì im lặng quay lại danh
    sách dấu câu cứng. Mọi đường gọi trong hệ đều truyền `bound`.
    """
    out: list[tuple[int, ...]] = []
    for span in iter_sentence_spans(list(seq), vocab, bound, bootstrap=bootstrap):
        if _content_len(span, vocab, bound, bootstrap=bootstrap) < min_content:
            continue
        if len(span) > max_tokens:
            # cắt theo max_tokens nhưng giữ biên nếu có
            span = span[:max_tokens]
        out.append(tuple(int(t) for t in span))
    return out


def migrate_parents_to_sentences(
    state: GraphState,
    *,
    bootstrap: bool = False,
    only_mega_gt: int = 50,
    min_content: int = 3,
    max_tokens: int = 64,
    max_spans_per_source: int = 24,
    max_new_parents: int = 200_000,
    min_episode_members: int = 2,
    dap_id: int | None = None,
    logger=None,
) -> SplitStats:
    """Thay `state.parents` bằng store đã segment; dựng episode từ nguồn multi-câu.

    Cha ngắn (emit ≤ only_mega_gt) giữ nguyên 1 parent.
    Cha dài / nhiều span → mỗi câu 1 parent + episode nối các câu cùng nguồn.
    """
    def log(msg: str) -> None:
        if logger is not None:
            logger.log(msg)
        else:
            print(msg, flush=True)

    # §13.54 — biên đơn vị HỌC ĐƯỢC, thay danh sách dấu câu cứng. Migrate chạy trên
    # checkpoint ĐÃ TRAIN nên tín hiệu có sẵn; `bootstrap=True` chỉ dành cho state chưa có
    # `W_seq` (ví dụ fixture test dựng tay) — và khi đó phải nói ra, không đoán ngầm.
    _bd = None if bootstrap else _bound_of(state)
    if _bd is not None and _bd.eos_base <= 0:
        raise ValueError(
            "§13.54: state chưa có tín hiệu EOS (eos_base = 0) ⇒ không tách câu được. "
            "Train trước, hoặc truyền bootstrap=True nếu cố ý dùng mồi nguội."
        )
    old = state.parents
    if old is None:
        return SplitStats()
    vocab = state.vocab
    if dap_id is None:
        from nn_aivie.data.qa import DAP_LABEL

        dap_id = vocab.id_of(DAP_LABEL)

    stats = SplitStats(old_parents=len(old.parents))
    new_ps = ParentStore()
    new_ps.begin_ingest_batch()
    episode_groups: list[tuple[str, str, list[tuple[int, ...]]]] = []

    for pi, rec in enumerate(old.parents):
        if len(new_ps.parents) >= max_new_parents:
            stats.truncated_spans += max(0, stats.old_parents - pi)
            break
        seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        priority = float(getattr(rec, "priority", 1.0) or 1.0)
        # QA: giữ nguyên 1 cha (không xé answer thành nhiều QA)
        if rec.is_qa:
            src = list(rec.source_ids) if rec.source_ids else (
                list(rec.match_ids) + [dap_id] + list(rec.answer_ids)
                if dap_id is not None
                else list(seq)
            )
            new_ps.add_example(
                src,
                vocab,
                dap_id=dap_id,
                priority=priority,
                defer_index=True,
            )
            stats.kept_as_is += 1
            continue

        need_split = len(seq) > only_mega_gt
        sents = split_emit_to_sentence_seqs(
            seq, vocab, min_content=min_content, max_tokens=max_tokens,
            bound=_bd, bootstrap=bootstrap,
        )
        if not need_split or len(sents) <= 1:
            if not sents:
                if _content_len(seq, vocab, _bd, bootstrap=bootstrap) >= min_content:
                    sents = [tuple(seq)]
                else:
                    stats.skipped_short += 1
                    continue
            new_ps.add_example(
                list(sents[0]),
                vocab,
                dap_id=dap_id,
                priority=priority,
                defer_index=True,
            )
            stats.kept_as_is += 1
            continue

        # multi-sentence source
        stats.split_sources += 1
        if len(sents) > max_spans_per_source:
            sents = sents[:max_spans_per_source]
            stats.truncated_spans += 1
        member_seqs: list[tuple[int, ...]] = []
        for s in sents:
            if len(new_ps.parents) >= max_new_parents:
                break
            new_ps.add_example(
                list(s), vocab, dap_id=dap_id, priority=priority, defer_index=True
            )
            if tuple(s) in new_ps._key_to_idx:  # noqa: SLF001
                member_seqs.append(s)
        if len(member_seqs) >= min_episode_members:
            episode_groups.append((f"src:{pi}", f"parent-{pi}", member_seqs))

        if (pi + 1) % 5000 == 0:
            log(f"  migrate {pi + 1}/{stats.old_parents} → new_parents={len(new_ps.parents)}")

    log("  finalize_batch_ingest …")
    new_ps.finalize_batch_ingest(vocab, logger=logger, defer_skip=True)
    log(f"  build episodes from {len(episode_groups)} groups …")
    made, matched = new_ps.build_episodes_from_provenance(
        episode_groups, min_members=min_episode_members
    )
    stats.episodes = made
    stats.new_parents = len(new_ps.parents)
    state.parents = new_ps

    # invalidate matrix / sent caches
    try:
        from nn_aivie.hca import matrix_core as mc

        mc._PARENT_BI.clear()
        mc._KNOW_QK.clear()
        mc._STRUCT_MASK.clear()
    except Exception:
        pass
    clear_sent_intra_cache()
    log(
        f"✓ migrate: {stats.old_parents}→{stats.new_parents} parents, "
        f"episodes={stats.episodes} (matched_sents={matched}), "
        f"split_sources={stats.split_sources}, kept={stats.kept_as_is}"
    )
    return stats
