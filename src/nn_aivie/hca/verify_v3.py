"""G4 Verify cho lõi sinh v3 — không cần QASession / Rule v2."""

from __future__ import annotations

import numpy as np

from nn_aivie.core.adjacency import TypedAdjacency
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.config import HCAConfig


def apply_g4_verify_v3(
    scores: np.ndarray,
    w0: int,
    anchors: list[int],
    adjacency: TypedAdjacency,
    vocab: Vocabulary,
    eos_id: int,
    *,
    config: HCAConfig | None = None,
) -> np.ndarray:
    """
    Ức chế token không có đường dẫn học được từ anchor (entity, agenda, w0, đáp gần).
    Giống G4 v2 nhưng anchor = entity register + agenda thay vì QFocus.
    """
    config = config or HCAConfig()
    if not config.verify_enabled:
        return scores

    seen: set[int] = set()
    uniq_anchors: list[int] = []
    for a in [w0, *anchors]:
        if a not in seen:
            seen.add(a)
            uniq_anchors.append(a)

    out = scores.copy()
    n = len(out)

    # TK_011 B1 — grounding vectorized: dựng g[v] = max_anchor learned(anchor→v)
    # MỘT lần bằng cách gom hàng của vài anchor, thay vì quét V ứng viên × anchor ×
    # lớp. Kết quả token-identical với vòng grounding_score cũ (chỉ đổi độ phức tạp).
    g: dict[int, float] = {}
    for a in uniq_anchors:
        for v, w in adjacency.learned_row(a).items():
            if v < n and w > g.get(v, 0.0):
                g[v] = w

    vmin = config.verify_min
    surface = vocab.surface_mask()  # precompute — bỏ is_surface_token/ứng viên (TK_011)
    ns = len(surface)
    # Chỉ xét ứng viên còn sống (điểm hữu hạn dương) — phần lớn V bị loại ngay,
    # khỏi phải token_of/grounding cho từng node chết.
    for v in np.nonzero(np.isfinite(out) & (out > 0.0))[0]:
        v = int(v)
        if v == eos_id:
            continue
        if v < ns and not surface[v]:
            continue
        # Dấu câu KHÔNG được miễn G4 — nó là token như mọi token, phải có đường dẫn đã học
        # như mọi token (Luật 0: không đặc-cách từ vựng lúc suy luận).
        # Bản cũ miễn trừ qua `token_of(v) in {",",".",";","!","?"}` (bản sao thứ 4 của
        # PUNCT_TOKENS, lại thiếu `…`/`:`). Miễn trừ đó THỪA: G1 đếm dấu câu vào
        # CO_ACTIVATION bình thường nên grounding của chúng rất cao — đo 2026-07-15 trên
        # grade_lessons_test (anchor Việt/là/học, vmin=0.04): '.'=227.45, ':'=285.30,
        # ','=10.50, '?'=8.90 — đều vượt ngưỡng hàng nghìn lần, còn cao hơn 'Nam'=44.70.
        if g.get(v, 0.0) < vmin:
            out[v] = -np.inf
    return out
