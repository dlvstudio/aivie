"""Unit: hub spread ưu tiên gidf·diversity hơn weight thuần."""
from __future__ import annotations

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.hub_spread import (
    build_hub_spread_index,
    resolve_surface_ids,
    score_hubs_from,
    score_hubs_raw_compare,
    spread_analogy,
)


def _encode(v: Vocabulary, text: str) -> list[int]:
    return [v.id_of(t) for t in text.split()]


def test_hub_spread_prefers_concept_over_la() -> None:
    """Khỉ/chó/mèo … động vật; ranking scored hạ «là»."""
    v = Vocabulary()
    st = GraphState(vocab=v)
    st.parents = ParentStore()
    # Lặp để W_skip min_count=2; thêm nhiều câu nền để «là» df cao → gidf thấp
    filler = [
        "anh là người <EOS>",
        "em là người <EOS>",
        "nó là việc <EOS>",
        "đây là cách <EOS>",
        "kia là nơi <EOS>",
        "hắn là kẻ <EOS>",
        "ta là ai <EOS>",
        "họ là bạn <EOS>",
    ]
    animal = [
        "con khỉ là động vật <EOS>",
        "con chó là động vật <EOS>",
        "con mèo là động vật <EOS>",
        "con trâu là động vật <EOS>",
        "con khỉ nghịch ngợm <EOS>",
        "con chó nghịch ngợm <EOS>",
        "con mèo nghịch ngợm <EOS>",
    ]
    for s in filler + animal:
        for _ in range(2):
            st.parents.add_example(_encode(v, s), v, dap_id=None, defer_index=True)
    st.parents.rebuild_hierarchical_index()

    idx = build_hub_spread_index(st, d_max=3, min_count=2)
    seeds = resolve_surface_ids(v, "khỉ")
    assert seeds
    raw = score_hubs_raw_compare(st, idx, seeds, topk=10)
    scored = score_hubs_from(st, idx, seeds, topk=10, gidf_min=0.0, ban_func=True)
    raw_surfs = [h.surface for h in raw]
    scored_surfs = [h.surface for h in scored]
    # scored cấm func → không còn «là»
    assert "là" not in scored_surfs
    # có tín hiệu lớp (động hoặc vật)
    assert any(s in scored_surfs for s in ("động", "vật", "động vật"))

    out = spread_analogy(st, idx, "khỉ", prop_prefix="nghịch", gidf_min=0.0)
    peer_x = {p["X"] for p in out["peers"]}
    assert peer_x & {"chó", "mèo", "trâu"}
    prop_z = {p["Z"] for p in out["props"]}
    assert "nghịch" in prop_z or "ngợm" in prop_z or any(
        z.startswith("nghịch") for z in prop_z
    )
    # raw có thể vẫn thấy «là» (weight) — đó là điểm so sánh
    _ = raw_surfs
