"""Sidecar cache cho skip spans đã mine — tránh re-mine ~15 phút mỗi process."""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.skip_span_constants import skip_right_tuple

SIDECAR_VERSION = 1


def skip_spans_sidecar_path(checkpoint: str | Path) -> Path:
    """`checkpoints/foo.pkl` → `checkpoints/foo.skip_spans.pkl`."""
    p = Path(checkpoint)
    return p.with_name(p.stem + ".skip_spans.pkl")


def apply_skip_spans_to_store(ps: ParentStore, spans_data: list[dict]) -> None:
    """Nạp skip spans + rebuild lookup/index (giống from_payload)."""
    ps.skip_spans = []
    ps._skip_lookup = {}
    ps._skip_left_index = {}
    ps._skip_index = {}
    ps.skip_span_meta = {}
    for s in spans_data:
        right = skip_right_tuple(s["right"])
        rec = {
            "left": int(s["left"]),
            "right": right,
            "power": int(s["power"]),
            "pmi": float(s.get("pmi", 0.0)),
            "gaps": {int(g): int(c) for g, c in s["gaps"].items()},
            "prov": [tuple(t) for t in s["prov"]],
            "promoted": bool(s.get("promoted")),
        }
        si = len(ps.skip_spans)
        ps.skip_spans.append(rec)
        key = (rec["left"], right)
        ps._skip_lookup[key] = si
        ps.skip_span_meta[key] = rec
    if ps.skip_spans:
        index: dict[int, list[int]] = defaultdict(list)
        for pi, prec in enumerate(ps.parents):
            for si in getattr(prec, "skip_span_ids", ()):
                if si not in index[si]:
                    index[si].append(pi)
        ps._skip_index = {k: sorted(v) for k, v in index.items()}
        ps._skip_df = [len(ps._skip_index.get(i, ())) for i in range(len(ps.skip_spans))]
        ps._rebuild_skip_left_index()


def save_skip_spans_sidecar(ps: ParentStore, checkpoint: str | Path) -> Path:
    """Ghi sidecar sau khi mine (hoặc khi checkpoint chưa embed spans)."""
    path = skip_spans_sidecar_path(checkpoint)
    payload = {
        "version": SIDECAR_VERSION,
        "n_parents": len(ps.parents),
        "skip_spans": [
            {
                "left": s["left"],
                "right": list(skip_right_tuple(s["right"])),
                "power": s["power"],
                "pmi": s["pmi"],
                "gaps": {str(g): c for g, c in s["gaps"].items()},
                "prov": [list(t) for t in s["prov"]],
                "promoted": bool(s.get("promoted")),
            }
            for s in (ps.skip_spans or [])
        ],
        "parent_skip_span_ids": [
            list(getattr(p, "skip_span_ids", ())) for p in ps.parents
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_skip_spans_sidecar(
    ps: ParentStore,
    checkpoint: str | Path,
    *,
    logger=None,
) -> bool:
    """Load sidecar nếu khớp n_parents; trả True nếu thành công."""
    path = skip_spans_sidecar_path(checkpoint)
    if not path.exists():
        return False
    with path.open("rb") as f:
        data = pickle.load(f)
    if data.get("version") != SIDECAR_VERSION:
        return False
    if data.get("n_parents") != len(ps.parents):
        if logger is not None:
            logger.log(
                f"skip_spans sidecar stale (parents {data.get('n_parents')} "
                f"!= {len(ps.parents)})"
            )
        return False
    spans = data.get("skip_spans") or []
    if not spans:
        return False
    parent_sids = data.get("parent_skip_span_ids")
    if parent_sids and len(parent_sids) == len(ps.parents):
        for i, sids in enumerate(parent_sids):
            ps.parents[i].skip_span_ids = tuple(int(x) for x in sids)
    apply_skip_spans_to_store(ps, spans)
    if logger is not None:
        logger.log(f"✓ skip_spans sidecar: {path.name} ({len(spans)} spans)")
    return True
