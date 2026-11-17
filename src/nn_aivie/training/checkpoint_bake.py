"""
Bake chỉ số suy dẫn vào checkpoint — mine một lần lúc build, load không làm lại.

Gồm: skip spans (TK_036), seq index bậc-2 (seq_bi/seq_tri), FRAME_BRACKET (nếu thiếu).
"""

from __future__ import annotations

from nn_aivie.core.adjacency import SparseAdjacency
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.training.frame_bracket import mine_frame_brackets
from nn_aivie.training.skip_span import mine_skip_spans


def bake_derived_indices(
    state: GraphState,
    *,
    logger=None,
    skip_spans: bool = True,
    seq_index: bool = True,
    frame_brackets: bool = True,
    force_skip_spans: bool = False,
    force_frame_brackets: bool = False,
) -> dict[str, object]:
    """Mine/ghi chỉ số suy dẫn lên state TRƯỚC save — `to_payload` sẽ embed vào .pkl."""
    stats: dict[str, object] = {}
    ps = state.parents
    if ps is None or len(ps.parents) == 0:
        return stats

    def _log(msg: str) -> None:
        if logger is not None:
            logger.log(msg)

    if frame_brackets:
        layer = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
        has_meta = bool(getattr(ps, "hlink_meta", None))
        # `has_eps` phải khớp `hca/frame_bracket_gate.ensure_frame_brackets`: runtime coi
        # checkpoint thiếu `hlink_endpoints` (TK_037) là CHƯA bake và đòi mine lại. Nếu bake
        # chỉ hỏi `has_meta`, checkpoint bake trước TK_037 (có layer + meta, endpoints rỗng)
        # sẽ kẹt: bake báo "skipped", runtime bị guard checkpoint-lớn chặn → phrase endpoint
        # chết âm thầm. Đo 2026-07-14: network_v10_histwiki{,_paragraph}.pkl đúng trạng thái này.
        has_eps = bool(getattr(ps, "hlink_endpoints", None))
        if force_frame_brackets or len(layer) == 0 or not has_meta or not has_eps:
            if len(layer) > 0:  # re-mine: bỏ layer cũ, tránh trộn cạnh 2 lần mine
                state.adjacency._layers[EdgeKind.FRAME_BRACKET] = SparseAdjacency()
            _log("▶ mine FRAME_BRACKET …")
            stats["frame_brackets"] = mine_frame_brackets(state, logger=logger)
        else:
            stats["frame_brackets"] = {"skipped": True, "edges": len(layer)}

    if skip_spans:
        have = bool(getattr(ps, "skip_spans", None))
        if force_skip_spans or not have:
            _log("▶ mine skip_spans …")
            stats["skip_spans"] = mine_skip_spans(state, logger=logger)
        else:
            stats["skip_spans"] = {"skipped": True, "spans": len(ps.skip_spans)}

    if seq_index:
        _log("▶ build seq index (seq_bi / seq_tri) …")
        ps._ensure_seq_index()
        stats["seq_index"] = {
            "bi_rows": len(getattr(ps, "_seq_bi", {})),
            "tri_rows": len(getattr(ps, "_seq_tri", {})),
        }

    return stats
