"""G1 Passive Listening — Hebbian co-occurrence từ corpus."""

from __future__ import annotations

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.hebb import update_from_coactivation
from nn_aivie.core.state import GraphState
from nn_aivie.data.corpus import TextCorpus
from nn_aivie.data.types import CorpusRecord
from nn_aivie.core.vocab import SPECIAL_EOS
from nn_aivie.teaching.align import tokenize
from nn_aivie.training.neurogenesis import run_neurogenesis_step


def run_g1_passive(
    state: GraphState,
    corpus: TextCorpus,
    eta: float = 0.01,
    max_records: int | None = None,
    *,
    window: int = 8,
    sentence_bind: bool = True,
    sentence_eta_scale: float = 0.5,
    enable_neurogenesis: bool = True,
) -> dict[str, int]:
    """Nạp corpus: cửa sổ cục bộ + (tuỳ chọn) co-activate toàn câu."""
    stats: dict[str, int] = {
        "records": 0,
        "tokens": 0,
        "updates": 0,
        "sentence_updates": 0,
    }
    for i, record in enumerate(corpus):
        if max_records is not None and i >= max_records:
            break
        stats["records"] += 1
        _passive_sentence(
            state,
            record,
            eta,
            stats,
            window=window,
            sentence_bind=sentence_bind,
            sentence_eta_scale=sentence_eta_scale,
        )
    
    if enable_neurogenesis:
        # Run unsupervised neurogenesis to extract abstract concept nodes
        neuro_stats = run_neurogenesis_step(
            state,
            sim_threshold=0.03,
            min_clique_size=2,
            max_clique_size=6,
            w_init=1.0,
        )
        for k, v in neuro_stats.items():
            stats[f"neurogenesis_{k}"] = v
            
    return stats


def _passive_sentence(
    state: GraphState,
    record: CorpusRecord,
    eta: float,
    stats: dict[str, int],
    *,
    window: int = 8,
    sentence_bind: bool = True,
    sentence_eta_scale: float = 0.5,
) -> None:
    tokens = tokenize(record.text)
    ids = state.vocab.encode_tokens(tokens)
    stats["tokens"] += len(ids)

    for start in range(len(ids)):
        end = min(len(ids), start + window)
        window_ids = ids[start:end]
        if len(window_ids) < 2:
            continue
        update_from_coactivation(state.adjacency, window_ids, eta)
        stats["updates"] += 1

    # W_seq — chuyển dịch vị trí có hướng (bigram a→b) cho kênh micro hybrid v3.
    for a, b in zip(ids, ids[1:]):
        state.adjacency.add(a, b, eta, EdgeKind.SEQUENTIAL)
    if ids:
        stats["wseq_updates"] = stats.get("wseq_updates", 0) + max(0, len(ids) - 1)

    if sentence_bind and len(ids) >= 2:
        update_from_coactivation(
            state.adjacency, ids, eta * sentence_eta_scale
        )
        stats["sentence_updates"] = stats.get("sentence_updates", 0) + 1
        eos_id = state.vocab.id_of(SPECIAL_EOS)
        update_from_coactivation(
            state.adjacency, [ids[-1], eos_id], eta * sentence_eta_scale
        )
