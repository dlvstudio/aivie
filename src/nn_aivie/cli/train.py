"""aivie train — corpus curriculum theo lớp và G1–G4."""

from __future__ import annotations

from pathlib import Path

from nn_aivie.core.state import GraphState
from nn_aivie.training.runner import run_curriculum_step


def run_train(
    grade: int,
    phase: str,
    checkpoint: str | None,
    output: str,
    *,
    eta: float = 0.01,
    max_records: int | None = None,
    dataset: list[str] | None = None,
    manifest: str | None = None,
    sentence_bind: bool = True,
    sentence_eta_scale: float = 0.5,
    alpha: float = 1.5,
    beta: float = 1.0,
    teach_eos: bool = True,
) -> GraphState:
    state = GraphState.load(checkpoint) if checkpoint else GraphState()
    stats = run_curriculum_step(
        state,
        grade=grade,
        phase=phase,  # type: ignore[arg-type]
        eta=eta,
        max_records=max_records,
        manifest_path=Path(manifest) if manifest else None,
        dataset_ids=dataset,
        sentence_bind=sentence_bind,
        sentence_eta_scale=sentence_eta_scale,
        alpha=alpha,
        beta=beta,
        teach_eos=teach_eos,
    )
    state.save(output)
    sent = stats.get("sentence_updates", 0)
    if phase == "G3":
        print(
            f"train grade={grade} {phase}: "
            f"{stats['records']} records, {stats.get('tokens', 0)} tokens, "
            f"{stats.get('steps', 0)} steps, "
            f"{stats.get('correct', 0)} correct / {stats.get('wrong', 0)} wrong, "
            f"{stats.get('skipped', 0)} skipped, "
            f"{stats.get('punishments', 0)} LTD, "
            f"{len(state.adjacency)} edges -> {output}"
        )
    elif phase == "G2":
        print(
            f"train grade={grade} {phase}: "
            f"{stats['records']} records, {stats.get('tokens', 0)} tokens, "
            f"{stats.get('steps', 0)} imitation steps, "
            f"{stats.get('qa_binds', 0)} qa binds, "
            f"{len(state.adjacency)} edges -> {output}"
        )
    else:
        print(
            f"train grade={grade} {phase}: "
            f"{stats['records']} records, {stats.get('tokens', 0)} tokens, "
            f"{stats['updates']} window updates, {sent} sentence binds, "
            f"{len(state.adjacency)} edges -> {output}"
        )
    return state
