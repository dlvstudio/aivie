"""
TK_019 (dọn legacy) — các hàm readout SỐNG của đường sinh token-by-token, gom về một chỗ.

Trước đây kẹt trong file v2 đã xóa: `matcher.py` (mask/repetition), `forward.py`
(spreading), `generate.py` (decode). Đường sinh duy nhất còn lại: `generate_hebbian`.
"""

from __future__ import annotations

import numpy as np

from nn_aivie.core.adjacency import TypedAdjacency
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import SPECIAL_EOS, Vocabulary
from nn_aivie.hca.config import HCAConfig


def _spreading_ctx_dict(
    seeds: list[tuple[int, float]],
    adjacency: TypedAdjacency,
    config: HCAConfig,
    vocab_size: int,
) -> dict[int, float]:
    """BFS dict spreading (legacy fallback khi spread_use_csr=False)."""
    gamma = config.spread_gamma
    hops = max(1, config.spread_hops)
    max_frontier = config.spread_max_frontier
    threshold = config.spread_threshold

    ctx: dict[int, float] = {}
    frontier: dict[int, float] = {}
    for sid, w in seeds:
        if 0 <= sid < vocab_size and w > 0:
            frontier[sid] = frontier.get(sid, 0.0) + w

    for _ in range(hops):
        if not frontier:
            break
        items = sorted(frontier.items(), key=lambda kv: -kv[1])[:max_frontier]
        nxt: dict[int, float] = {}
        for u, au in items:
            if au < threshold:
                continue
            row = adjacency.row_readout(u)
            if not row:
                continue
            row_max = max(row.values()) or 1.0
            for v, a_uv in row.items():
                if v >= vocab_size:
                    continue
                contrib = au * (a_uv / row_max) * gamma
                if contrib < threshold:
                    continue
                ctx[v] = ctx.get(v, 0.0) + contrib
                nxt[v] = nxt.get(v, 0.0) + contrib
        frontier = nxt
    return ctx


def _spreading_ctx(
    seeds: list[tuple[int, float]],
    adjacency: TypedAdjacency,
    config: HCAConfig,
    vocab_size: int,
) -> dict[int, float]:
    if getattr(config, "spread_use_csr", True):
        from nn_aivie.hca.spreading_csr import spreading_ctx_csr

        return spreading_ctx_csr(seeds, adjacency, config, vocab_size)
    return _spreading_ctx_dict(seeds, adjacency, config, vocab_size)


def _gate_spreading_by_w0(
    ctx: dict[int, float],
    w0: int,
    adjacency: TypedAdjacency,
    vocab_size: int,
) -> dict[int, float]:
    row_w0 = adjacency.row_readout(w0)
    scores: dict[int, float] = {}
    for v, a_w0v in row_w0.items():
        if v < vocab_size:
            cv = ctx.get(v, 0.0)
            if cv > 0:
                scores[v] = cv * a_w0v
    return scores


def decode_surface(state: GraphState, ids: list[int]) -> str:
    """Chuỗi chỉ gồm token bề mặt (lọc node cấu trúc nếu có trong prompt).

    TK_041: nối qua `vocab.detokenize` — detokenizer DUY NHẤT. Trên BPE, khoảng trắng là
    thứ mạng ĐÃ NẠP, không phải thứ `" ".join` BỊA ra; hết `Hà Nội .` (space thừa trước dấu
    chấm) — lỗi hiển thị mà mọi output của dự án mang suốt.
    """
    return state.vocab.detokenize(ids)


def mask_non_surface(scores: np.ndarray, vocab: Vocabulary) -> np.ndarray:
    out = scores.copy()
    mask_non_surface_inplace(out, vocab)
    return out


def mask_non_surface_inplace(scores: np.ndarray, vocab: Vocabulary) -> None:
    mask = vocab.surface_mask()  # precompute cache (TK_011)
    n = min(len(scores), len(mask))
    eos = vocab._token_to_id.get(SPECIAL_EOS)
    for v in np.where(np.isfinite(scores[:n]))[0]:
        v = int(v)
        if eos is not None and v == eos:
            continue
        if not mask[v]:
            scores[v] = -np.inf


def _fold_equiv(vocab):
    """tid → mọi id CÙNG bề mặt sau fold (cache trên vocab). None nếu vocab không hỗ trợ."""
    if vocab is None or not hasattr(vocab, "fold"):
        return None
    idx = getattr(vocab, "_fold_to_ids_cache", None)
    if idx is None:
        idx = {}
        for t in range(len(vocab)):
            try:
                k = vocab.fold(t)
            except Exception:
                continue
            if k:
                idx.setdefault(k, []).append(t)
        try:
            vocab._fold_to_ids_cache = idx  # noqa: SLF001
        except Exception:
            pass

    def _eq(tid: int):
        try:
            return idx.get(vocab.fold(int(tid)), (int(tid),))
        except Exception:
            return (int(tid),)

    return _eq


def apply_repetition_penalty(
    scores: np.ndarray,
    generated: list[int],
    vocab: Vocabulary,
    *,
    window: int = 6,
    factor: float = 0.25,
    prompt_len: int = 0,
) -> np.ndarray:
    """Penalty lặp THUẦN window trên phần ĐÃ SINH (TK_019: thay exempt QA-session v2).

    KHÔNG phạt token thuộc PROMPT (`prompt_len` đầu) — đáp lặp từ câu hỏi là ngôn ngữ đúng
    ("Đoàn kết giúp em…" lặp "Đoàn kết giúp em"; đo: phạt prompt làm QA gen-hy 100→86)."""
    out = scores.copy()
    apply_repetition_penalty_inplace(
        out, generated, window=window, factor=factor, prompt_len=prompt_len,
        equiv=_fold_equiv(vocab),
    )
    return out


def apply_repetition_penalty_inplace(
    scores: np.ndarray,
    generated: list[int],
    *,
    window: int = 6,
    factor: float = 0.25,
    prompt_len: int = 0,
    equiv=None,  # tid -> iterable[tid] cùng lớp bề mặt (None = hành vi cũ theo-id)
) -> None:
    lo = max(prompt_len, len(generated) - window)
    for tid in generated[lo:]:
        if tid < len(scores) and np.isfinite(scores[tid]):
            scores[tid] *= factor
    # TK_045: chặn lặp trên LỚP TƯƠNG-ĐƯƠNG bề mặt, không chỉ id. BPE có nhiều biến-thể cùng
    # surface (`0`/`Ġ0`/`00`, `Nhà`/`Ġnhà`) → luân phiên id NÉ được `-inf` theo-id ⇒ loop
    # "1 0 0 0 0…×61" (đo v12, out/probe_multi_prompt_long_post_merge.log). Dùng fold
    # (thuộc-tính token precompute — Luật 0, KHÔNG bảng từ), nên vá cả lớp loop chứ không
    # đặc-cách chữ số.
    def _ban(tid: int) -> None:
        for t in (equiv(tid) if equiv is not None else (tid,)):
            if 0 <= t < len(scores):
                scores[t] = -np.inf

    if generated and generated[-1] < len(scores):
        _ban(int(generated[-1]))
    if len(generated) >= 2 and generated[-1] == generated[-2]:
        _ban(int(generated[-1]))


def spreading_activation_read(
    w0: int,
    seeds: list[tuple[int, float]],
    adjacency: TypedAdjacency,
    config: HCAConfig | None = None,
    vocab_size: int | None = None,
) -> np.ndarray:
    """
    Composition #2 — multi-hop spreading activation (graph-only, không rule từ vựng).

    Activation lan từ `seeds` qua readout edges `spread_hops` hop có decay; nội suy nổi
    ở hub chung; field cuối gate theo reachability từ w0.
    """
    config = config or HCAConfig()
    if vocab_size is None:
        max_from_adj = 0
        for i, j, _ in adjacency.items():
            max_from_adj = max(max_from_adj, i, j)
        vocab_size = max_from_adj + 1

    ctx = _spreading_ctx(seeds, adjacency, config, vocab_size)
    scores = np.zeros(vocab_size, dtype=np.float64)
    for v, s in _gate_spreading_by_w0(ctx, w0, adjacency, vocab_size).items():
        scores[v] = s
    return scores


def spreading_activation_read_sparse(
    w0: int,
    seeds: list[tuple[int, float]],
    adjacency: TypedAdjacency,
    config: HCAConfig | None = None,
    vocab_size: int | None = None,
) -> dict[int, float]:
    """Giống spreading_activation_read nhưng trả dict — không cấp phát np.zeros(V)."""
    config = config or HCAConfig()
    if vocab_size is None:
        max_from_adj = 0
        for i, j, _ in adjacency.items():
            max_from_adj = max(max_from_adj, i, j)
        vocab_size = max_from_adj + 1

    ctx = _spreading_ctx(seeds, adjacency, config, vocab_size)
    return _gate_spreading_by_w0(ctx, w0, adjacency, vocab_size)


def multiply_scores_by_boost(
    scores: np.ndarray,
    boost: np.ndarray,
    *,
    floor: float,
    weight: float,
) -> bool:
    bmax = float(boost.max())
    if bmax <= 0:
        return False
    finite = np.where(np.isfinite(scores))[0]
    if finite.size == 0:
        return False
    mult = floor + weight * boost[finite] / bmax
    scores[finite] *= mult
    return True
