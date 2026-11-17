"""
TK_012 — Checkpoint IO: graph thuần (.pkl) + content-cache sidecar (.content.pkl).

`load_checkpoint`/`save_checkpoint` là cửa DUY NHẤT nên dùng khi cần cả nhãn (viz/eval).
Inference & training vẫn dùng `GraphState.load` thuần (không cache).

Auto-migrate: checkpoint cũ (nhãn còn nhúng trong graph pickle) → `GraphState.load` bóc nhãn
ra `ParentStore._migrated_content` khi from_payload; hàm này gom lại thành ContentCache.
"""

from __future__ import annotations

from pathlib import Path

from nn_aivie.core.content_cache import ContentCache
from nn_aivie.core.state import GraphState


def content_sidecar_path(checkpoint: str | Path) -> Path:
    """`checkpoints/foo.pkl` → `checkpoints/foo.content.pkl`."""
    p = Path(checkpoint)
    return p.with_suffix(".content.pkl")


def load_checkpoint(
    checkpoint: str | Path, *, log: bool = True, logger=None, freeze_adjacency: bool = False
) -> tuple[GraphState, ContentCache]:
    """Load graph thuần + content-cache. Nếu sidecar thiếu nhưng graph cũ còn nhãn nhúng →
    dùng nhãn migrate (tương thích ngược, không cần rebuild).

    `log=True` (mặc định): ghi tiến độ TỪNG BƯỚC ra `out/logs/load_<stem>_<ts>.log` + stdout —
    quan sát load đến đâu/đúng chưa (quy tắc TK_011: mọi process dài ghi file). `log=False` tắt."""
    if logger is None and log:
        from nn_aivie.util.progress import get_run_logger

        logger = get_run_logger(f"load_{Path(checkpoint).stem}")
    state = GraphState.load(str(checkpoint), logger=logger, freeze_adjacency=freeze_adjacency)
    sidecar = content_sidecar_path(checkpoint)
    if logger is not None:
        logger.log(f"sidecar {'có' if sidecar.exists() else 'thiếu → migrate'}: {sidecar.name}")
    if sidecar.exists():
        cache = ContentCache.load(sidecar)
    else:
        cache = _cache_from_migrated(state)
    # TK_031: sidecar VarOrderLM (n-gram biến-bậc — văn mượt) nếu đã build
    from nn_aivie.core.varlm import VarOrderLM, varlm_sidecar_path

    vpath = varlm_sidecar_path(checkpoint)
    if vpath.exists():
        if logger is not None:
            logger.log(f"▶ varlm sidecar: {vpath.name}")
        state.varlm = VarOrderLM.load(vpath)
        if logger is not None:
            logger.log(f"✓ varlm: {state.varlm.stats()}")
    else:
        state.varlm = None
    # TK_036: skip spans sidecar (khi checkpoint chưa embed spans)
    if state.parents is not None and not getattr(state.parents, "skip_spans", None):
        from nn_aivie.core.skip_span_cache import load_skip_spans_sidecar

        load_skip_spans_sidecar(state.parents, checkpoint, logger=logger)
    state._checkpoint_path = str(checkpoint)
    # TK_045: PPMI_l2r/r2l QK sidecar (nếu đã bake)
    sq = Path(checkpoint).with_suffix(".sent_qk.npz")
    state._sent_qk_sidecar = str(sq)  # type: ignore[attr-defined]
    if logger is not None:
        logger.log("✓ load_checkpoint XONG")
    return state, cache


def save_checkpoint(
    state: GraphState,
    checkpoint: str | Path,
    cache: ContentCache | None = None,
    *,
    bake_derived: bool = False,
    logger=None,
) -> None:
    """Ghi graph thuần + sidecar cache (nếu có nhãn).

    `bake_derived=True`: mine skip spans / seq index / FRAME_BRACKET (nếu thiếu) trước save —
    chỉ dùng lúc build/migrate, không bật mỗi lần save nhẹ."""
    if bake_derived:
        from nn_aivie.training.checkpoint_bake import bake_derived_indices

        bake_derived_indices(state, logger=logger)
    state.save(str(checkpoint))
    if cache:
        cache.save(content_sidecar_path(checkpoint))


def _cache_from_migrated(state: GraphState) -> ContentCache:
    """Gom nhãn bóc từ checkpoint cũ (ParentStore giữ tạm trong `_migrated_content`)."""
    cache = ContentCache()
    ps = state.parents
    if ps is None:
        return cache
    migrated = getattr(ps, "_migrated_content", None)
    if not migrated:
        return cache
    for pi, tag in migrated.get("parents", {}).items():
        cache.set_parent_tag(pi, tag)
    for ei, ec in migrated.get("episodes", {}).items():
        cache.set_episode(ei, key=ec.get("key", ""), title=ec.get("title", ""),
                          source=ec.get("source", "doc"))
    return cache
