#!/usr/bin/env python3
"""
Train v3 — KHÔNG dạy ngữ pháp (no NVL/teach pack, no Rule/Var imprint).

Ngữ pháp & cấu trúc đáp **tự nổi** từ dữ liệu thật lớp 3–7 qua:
  G1 (co-occurrence + W_seq) → neurogenesis (Concept) → G2 (đường đáp REINFORCED + q→a bind,
  khối rule_collocation tự bỏ khi không có Rule node) → emergent consolidation → E (PPMI+SVD).

Chỉ dùng **dữ liệu kiến thức thật**; loại corpus HF nhiễu (general, vi_sentences_hf).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nn_aivie.cli.train import run_train
from nn_aivie.core.embedding import build_embedding
from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.data.corpus import TextCorpus, load_manifest
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.data.registry import grade_manifest_path
from nn_aivie.teaching.align import tokenize
from nn_aivie.training.consolidate import consolidate_emergent
from nn_aivie.util.progress import DEFAULT_LOG_DIR


def build_parents(state: GraphState, grades: list[int], dataset_ids: list[str]) -> ParentStore:
    """Mỗi record thật → MỘT nơ-ron cha (v3 §1). Vocab đã ổn định sau train.

    qa_corpus: nạp cả shard val — đó là kiến thức Hỏi/Đáp (bộ nhớ cha), không phải benchmark
    ẩn; paraphrase_heldout / synonym_probe vẫn giữ riêng cho đo tổng quát hóa.
    """
    store = ParentStore()
    store.ensure_skip_ledger(min_count=4, min_pmi=1.0)
    dap_id = state.vocab.id_of(DAP_LABEL)
    allowed = set(dataset_ids)
    store.seg.begin_batch()
    for grade in grades:
        manifest = load_manifest(grade_manifest_path(grade))
        specs = [s for s in manifest.for_phase("G1") if s.id in allowed]
        if not specs:
            continue
        for spec in specs:
            splits = ("train", "val") if spec.id == "qa_corpus" else ("train",)
            for split in splits:
                for rec in TextCorpus([spec], grade=grade, split=split):
                    ids = state.vocab.encode_tokens(tokenize(rec.text))
                    store.add_example(ids, state.vocab, dap_id=dap_id, defer_index=True)
    store.seg.end_batch(store.parents)
    store.rebuild_hierarchical_index()
    store.flush_skip_ledger(state.vocab)
    return store

# Dữ liệu thật, kiến thức thật lớp 3–7 (không HF general/sentences nhiễu).
REAL_KNOWLEDGE = [
    "curated_simple",
    "compound_corpus",
    "copula_corpus",
    "qa_corpus",
    "dialogue_corpus",
    "toan_corpus",
    "dia_ly_corpus",
    "lich_su_corpus",
    "khoa_hoc_corpus",
    "xa_hoi_corpus",
]
G2_DATASETS = ["qa_corpus", "curated_simple", "compound_corpus", "dialogue_corpus"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Train v3 (teaching-free) grade 3→7.")
    ap.add_argument("--out", default="checkpoints/grade_v3.pkl")
    ap.add_argument("--eta-g1", type=float, default=0.05)
    ap.add_argument("--eta-g2", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument(
        "--extra-dir",
        default=None,
        help="Thư mục chứa statements.jsonl / qa.jsonl (ingest HF) để nạp thêm dữ liệu thật",
    )
    ap.add_argument(
        "--legacy-repair",
        action="store_true",
        help="Chạy decompose_repair batch (cũ) thay vì chỉ ISG tăng dần (TK_013)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    ckpt = str(out)

    from nn_aivie.util.progress import PerfSummary, get_run_logger

    log = get_run_logger("train_v3")
    perf = PerfSummary()
    log.log(f"out={ckpt} extra_dir={args.extra_dir} legacy_repair={args.legacy_repair}")

    GraphState().save(ckpt)  # fresh — KHÔNG teach pack

    for grade in [3, 4, 5, 6, 7]:
        log.log("=" * 60 + f"  GRADE {grade:02d} (no teaching)")

        with log.phase(f"G1_grade{grade:02d}", summary=perf):
            run_train(grade=grade, phase="G1", checkpoint=ckpt, output=ckpt,
                      eta=args.eta_g1, dataset=REAL_KNOWLEDGE)

        with log.phase(f"G2_grade{grade:02d}", summary=perf):
            run_train(grade=grade, phase="G2", checkpoint=ckpt, output=ckpt,
                      eta=args.eta_g2, dataset=G2_DATASETS)

        with log.phase(f"consolidate_grade{grade:02d}", summary=perf):
            state = GraphState.load(ckpt)
            consol = consolidate_emergent(state)
            state.save(ckpt)
            log.log(f"  sources={consol['sources']} emergent_edges={consol['emergent_edges']}")

    with log.phase("build_parents", summary=perf):
        state = GraphState.load(ckpt)
        state.parents = build_parents(state, [3, 4, 5, 6, 7], REAL_KNOWLEDGE)
        qa_n = sum(p.is_qa for p in state.parents.parents)
        log.log(f"  parents={len(state.parents)} (QA={qa_n}, statement={len(state.parents)-qa_n})")

    if args.extra_dir:
        import json
        from nn_aivie.training.ingest import ingest_qa, ingest_statements

        extra = Path(args.extra_dir)
        sp = extra / "statements.jsonl"
        qp = extra / "qa.jsonl"
        ap_file = extra / "anaphora.jsonl"  # A2: câu `, nó/họ/chúng` (R0 tách dấu câu)
        with log.phase("ingest_extra", summary=perf):
            if sp.exists():
                sents = [json.loads(l)["text"] for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
                r = ingest_statements(state, sents, eta=args.eta_g1, split_punct=True, logger=log)
                log.log(f"  ingest statements: +{r['statements']} cha (từ {sp})")
            if ap_file.exists():
                asents = [json.loads(l)["text"] for l in ap_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                r = ingest_statements(state, asents, eta=args.eta_g1, split_punct=True, logger=log)
                log.log(f"  ingest anaphora (A2): +{r['statements']} cha (từ {ap_file})")
            if qp.exists():
                pairs = [(json.loads(l)["question"], json.loads(l)["answer"]) for l in qp.read_text(encoding="utf-8").splitlines() if l.strip()]
                r = ingest_qa(state, pairs, eta=args.eta_g2, split_punct=True, logger=log)
                log.log(f"  ingest QA: +{r['qa']} cha (từ {qp})")
            qa_n = sum(p.is_qa for p in state.parents.parents)
            log.log(f"  parents now={len(state.parents)} (QA={qa_n}, statement={len(state.parents)-qa_n})")

    with log.phase("decompose", summary=perf):
        if args.legacy_repair:
            max_merges = min(30000, max(4000, len(state.parents)))
            deco = state.parents.decompose_repair(
                state.vocab, min_frequency=3, max_merges=max_merges, max_span=8
            )
            log.log(f"  Re-Pair intermediate phrase neurons={deco['phrases']} (rules={deco['rules']})")
        else:
            log.log(f"  ISG (TK_013) online clusters={len(state.parents.seg.clusters)}"
                    f" links={len(state.parents.seg.links)}")
            state.parents.rebuild_hierarchical_index()
            state.parents.flush_skip_ledger(state.vocab, logger=log)

    with log.phase("slot_hubs", summary=perf):
        ih = state.parents.induce_slot_hubs(min_distinct_fillers=2)
        bh = state.parents.bind_slot_fillers()
        log.log(f"  slot hubs={ih['hubs']}  filler bindings (train-time)={bh['bound']}"
                f" (mới={bh['new_fillers']})")

    with log.phase("embedding", summary=perf):
        state.embedding = build_embedding(state.adjacency, state.vocab, dim=args.dim)
        state.save(ckpt)
        log.log(f"  E shape={state.embedding.matrix.shape}")

    log.log("=" * 60)
    log.log(f"V3 TRAINING COMPLETE  vocab={len(state.vocab)}  edges={len(state.adjacency)}  checkpoint={ckpt}")
    perf.print_summary()
    perf.write(Path(DEFAULT_LOG_DIR) / "train_v3_perf.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
