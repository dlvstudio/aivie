"""Curriculum runner — chọn corpus theo grade + phase."""

from __future__ import annotations

from pathlib import Path

from nn_aivie.core.state import GraphState
from nn_aivie.data.corpus import Phase, TextCorpus, load_manifest
from nn_aivie.data.registry import grade_manifest_path
from nn_aivie.training.g1_passive import run_g1_passive
from nn_aivie.training.g2_imitation import run_g2_imitation


def run_curriculum_step(
    state: GraphState,
    grade: int,
    phase: Phase,
    *,
    split: str = "train",
    eta: float | None = None,
    max_records: int | None = None,
    manifest_path: Path | None = None,
    dataset_ids: list[str] | None = None,
    sentence_bind: bool = True,
    sentence_eta_scale: float = 0.5,
    alpha: float = 1.5,
    beta: float = 1.0,
    teach_eos: bool = True,
) -> dict:
    manifest = load_manifest(manifest_path or grade_manifest_path(grade))
    if manifest.grade != grade:
        raise ValueError(f"manifest grade {manifest.grade} != requested {grade}")

    specs = manifest.for_phase(phase)
    if dataset_ids:
        allowed = set(dataset_ids)
        specs = [s for s in specs if s.id in allowed]
    if not specs:
        raise ValueError(f"no datasets for grade {grade} phase {phase}")

    corpus = TextCorpus(specs, grade=grade, split=split)  # type: ignore[arg-type]

    if phase == "G1":
        stats = run_g1_passive(
            state,
            corpus,
            eta=eta if eta is not None else 0.01,
            max_records=max_records,
            sentence_bind=sentence_bind,
            sentence_eta_scale=sentence_eta_scale,
            window=12,
        )
        return {"phase": phase, "grade": grade, "datasets": [s.id for s in specs], **stats}

    if phase == "G2":
        stats = run_g2_imitation(
            state,
            corpus,
            eta=eta if eta is not None else 0.04,
            max_records=max_records,
        )
        return {"phase": phase, "grade": grade, "datasets": [s.id for s in specs], **stats}

    if phase == "G3":
        stats = {"skipped": "g3 removed (TK_019)"}  # legacy v2 matcher-based
        return {"phase": phase, "grade": grade, "datasets": [s.id for s in specs], **stats}

    raise NotImplementedError(
        f"phase {phase} chưa triển khai — scaffold corpus đã sẵn; G1 passive đã có"
    )
