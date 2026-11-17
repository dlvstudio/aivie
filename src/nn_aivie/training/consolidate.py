"""
Composition #3 — Emergent consolidation (Hebb thật, tạo cạnh mới).

"Neurons that fire together, wire together" ở mức **cấu trúc**: sau khi train,
nếu hai node nguồn cùng dẫn mạnh tới một tập đích chung (đồng vai trong nhiều
frame), mạng tự tạo cạnh `EdgeKind.EMERGENT` để mỗi node **kế thừa** các đích
mạnh của node anh em. Hệ quả: khái quát hóa nổi lên từ các mẫu rời rạc — không
cần teacher liệt kê mọi tổ hợp, không có rule ngôn ngữ nào.

Guard: chỉ xét token bề mặt (bỏ special/var/rule/QFocus/dấu câu), yêu cầu chia
sẻ đủ nhiều đích (min_shared) + độ tương đồng Jaccard tối thiểu, cap số cặp và
số cạnh để tránh phình đồ thị. Cạnh EMERGENT vào readout ở scale nhỏ
(`READOUT_EM_SCALE`) nên chỉ *nudge*, không lấn đường gold đã học.
"""

from __future__ import annotations

import math
from collections import defaultdict

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import is_surface_token


def _eligible_source(vocab, nid: int) -> bool:
    tok = vocab.token_of(nid)
    if not is_surface_token(tok):
        return False
    return True


def consolidate_emergent(
    state: GraphState,
    *,
    min_weight: float = 0.05,
    min_shared: int = 2,
    sim_min: float = 0.22,
    top_targets: int = 10,
    inherit_scale: float = 0.45,
    max_inherit_per_pair: int = 4,
    max_sources: int = 600,
    max_edges: int = 4000,
    hub_quantile: float = 0.90,
) -> dict[str, int]:
    """
    Tạo cạnh EMERGENT kế thừa giữa các node nguồn đồng vai.

    Tương đồng dùng **Jaccard có trọng số IDF** theo in-degree của đích: chia sẻ
    một đích hiếm là bằng chứng đồng vai mạnh, chia sẻ một hub (đích phổ biến) thì
    yếu. Không kế thừa các đích hub (>= `hub_quantile`) để tránh nhiễu lan rộng.
    Đây là thống kê đồ thị thuần — không có luật ngôn ngữ.

    Trả về thống kê {sources, sibling_pairs, emergent_edges}.
    """
    adj = state.adjacency
    vocab = state.vocab

    out: dict[int, dict[int, float]] = defaultdict(dict)
    in_degree: dict[int, int] = defaultdict(int)
    for i, j, w in adj.items(EdgeKind.REINFORCED):
        if w <= min_weight:
            continue
        if not _eligible_source(vocab, i):
            continue
        out[i][j] = max(out[i].get(j, 0.0), w)
        in_degree[j] += 1

    # Ngưỡng hub theo phân vị in-degree (đích bị quá nhiều nguồn trỏ tới).
    degs = sorted(in_degree.values())
    hub_cut = degs[min(len(degs) - 1, int(len(degs) * hub_quantile))] if degs else 0

    def idf(t: int) -> float:
        return 1.0 / (1.0 + math.log(1.0 + in_degree.get(t, 0)))

    sources = [s for s, tgts in out.items() if len(tgts) >= min_shared]
    sources.sort(key=lambda s: -len(out[s]))
    sources = sources[:max_sources]

    top_sets: dict[int, dict[int, float]] = {}
    for s in sources:
        items = sorted(out[s].items(), key=lambda kv: -kv[1])[:top_targets]
        top_sets[s] = dict(items)

    stats = {"sources": len(sources), "sibling_pairs": 0, "emergent_edges": 0}

    for a_idx in range(len(sources)):
        a = sources[a_idx]
        keys_a = set(top_sets[a])
        for b_idx in range(a_idx + 1, len(sources)):
            b = sources[b_idx]
            keys_b = set(top_sets[b])
            inter = keys_a & keys_b
            if len(inter) < min_shared:
                continue
            union = keys_a | keys_b
            inter_w = sum(idf(t) for t in inter)
            union_w = sum(idf(t) for t in union)
            jac = inter_w / union_w if union_w > 0 else 0.0
            if jac < sim_min:
                continue
            stats["sibling_pairs"] += 1

            # a kế thừa các đích mạnh, *không phải hub* của b mà a chưa có.
            for src, donor_set, donor in ((a, keys_b - keys_a, b), (b, keys_a - keys_b, a)):
                donated = sorted(
                    (t for t in donor_set if in_degree.get(t, 0) < hub_cut),
                    key=lambda t: -top_sets[donor][t],
                )[:max_inherit_per_pair]
                for t in donated:
                    w = jac * top_sets[donor][t] * inherit_scale
                    if w <= 0:
                        continue
                    adj.add(src, t, w, EdgeKind.EMERGENT)
                    stats["emergent_edges"] += 1
                    if stats["emergent_edges"] >= max_edges:
                        return stats
    return stats
