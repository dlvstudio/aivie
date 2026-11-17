"""G2 Imitation — teacher-forced strengthening on gold answer paths (SFT-style)."""

from __future__ import annotations

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.hebb import update_from_coactivation
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import SPECIAL_EOS
from nn_aivie.data.corpus import TextCorpus
from nn_aivie.data.qa import DAP_LABEL, HOI_LABEL
from nn_aivie.data.types import CorpusRecord
from nn_aivie.hca.config import HCAConfig, hca_config_for_grade
from nn_aivie.teaching.align import tokenize


def _should_train_step(mask: list[int] | None, token_index: int) -> bool:
    if mask is None:
        return True
    if token_index < 0 or token_index >= len(mask):
        return False
    return mask[token_index] == 1


def _imitate_sentence(
    state: GraphState,
    record: CorpusRecord,
    eta: float,
    stats: dict[str, int],
    *,
    config: HCAConfig | None = None,
    context_bind: float = 0.45,
) -> None:
    config = config or hca_config_for_grade(record.grade)
    tokens = tokenize(record.text)
    if len(tokens) < 2:
        return
    ids = state.vocab.encode_tokens(tokens)
    mask = record.train_mask
    stats["tokens"] += len(ids)

    answer_ids = [ids[i] for i in range(len(ids)) if _should_train_step(mask, i)]
    if len(answer_ids) >= 2:
        for i in range(len(answer_ids) - 1):
            state.adjacency.add(
                answer_ids[i],
                answer_ids[i + 1],
                eta * 1.2,
                EdgeKind.REINFORCED,
            )
        update_from_coactivation(
            state.adjacency, answer_ids, eta * 0.6
        )
        stats["answer_bigrams"] = stats.get("answer_bigrams", 0) + len(answer_ids) - 1

    for i in range(1, len(ids)):
        if not _should_train_step(mask, i):
            stats["skipped"] = stats.get("skipped", 0) + 1
            continue
        w0, gold = ids[i - 1], ids[i]
        state.adjacency.add(w0, gold, eta, EdgeKind.REINFORCED)
        prefix = ids[:i]
        for hid in prefix[-config.k_max :]:
            state.adjacency.add(hid, gold, eta * context_bind, EdgeKind.REINFORCED)
        stats["steps"] += 1

    if DAP_LABEL in tokens:
        dap_i = tokens.index(DAP_LABEL)
        hoi_i = tokens.index(HOI_LABEL) if HOI_LABEL in tokens else 0
        q_ids = ids[hoi_i + 1 : dap_i]
        a_ids = answer_ids or ids[dap_i + 1 :]
        # TK_019: bỏ QFocus/fingerprint/rule_collocation (v2). Giữ bind anchor→đáp thuần.
        salient = list(q_ids)
        anchor_ids = salient[:1]
        if a_ids:
            for q in anchor_ids:
                state.adjacency.add(q, a_ids[0], eta * 0.95, EdgeKind.REINFORCED)
            a_set = set(a_ids)
            for q in q_ids:
                if q in a_set:
                    state.adjacency.add(q, q, eta * 0.5, EdgeKind.REINFORCED)
                tok_q = state.vocab.token_of(q)
                for a in a_ids:
                    if state.vocab.token_of(a) == tok_q:
                        state.adjacency.add(q, a, eta * 0.65, EdgeKind.REINFORCED)
        stats["qa_binds"] = stats.get("qa_binds", 0) + len(anchor_ids) * max(
            len(a_ids), 1
        )

    if answer_ids:
        eos_id = state.vocab.id_of(SPECIAL_EOS)
        state.adjacency.add(answer_ids[-1], eos_id, eta * 0.5, EdgeKind.REINFORCED)


def run_g2_imitation(
    state: GraphState,
    corpus: TextCorpus,
    eta: float = 0.04,
    max_records: int | None = None,
    *,
    config: HCAConfig | None = None,
) -> dict[str, int]:
    """Teacher-forced imitation — củng cố đường gold, không LTD."""
    stats: dict[str, int] = {
        "records": 0,
        "tokens": 0,
        "steps": 0,
        "skipped": 0,
    }
    for i, record in enumerate(corpus):
        if max_records is not None and i >= max_records:
            break
        stats["records"] += 1
        _imitate_sentence(
            state,
            record,
            eta,
            stats,
            config=config or hca_config_for_grade(record.grade),
        )
    return stats
