"""Bits-per-byte and simple corpus metrics (nanochat-style smoke).

Trụ 1 / bước 0 (docs/study/08, TK_015): thước **KHÔNG bão hoà** cho next-token modeling.
`bits_per_byte` cũ = proxy nhị phân (p=0.85 nếu rank-1 else 0.05) → tự bão hoà + gọi generate()
mỗi vị trí (chậm). `next_token_lm` mới đo **thứ hạng token gold** trong phân bố điểm base-LM
(W_seq ⊙ readout, KHÔNG agenda) trên corpus held-out → log2(rank) là chi phí mã hoá liên tục,
nhạy với cải tiến substrate (D1–D4). Cùng họ đo A0 doc 03 (held-out rank) — tái dùng cho Q/K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import is_surface_token
from nn_aivie.hca.readout import mask_non_surface
from nn_aivie.teaching.align import tokenize


@dataclass
class LMRuler:
    """Kết quả thước next-token không bão hoà (cao/thấp: xem field)."""
    bits: float       # mean log2(rank token gold) — chi phí mã hoá, THẤP = tốt (không bão hoà)
    mrr: float        # mean 1/rank — CAO = tốt
    top1: float       # % gold là argmax
    top5: float
    top10: float
    n_pos: int        # số vị trí chấm (gold in-vocab + surface + có ứng viên finite)
    n_oov: int        # số vị trí bỏ qua vì gold OOV/non-surface


def _base_lm_scores(
    state: GraphState,
    w0: int,
    vocab_size: int,
    micro_floor: float,
    ctx_ids: list[int] | None = None,
    varlm_weight: float = 0.0,
) -> np.ndarray:
    """Phân bố điểm base-LM tại w0: readout ⊙ (floor + W_seq bigram), mask non-surface.

    ĐÂY là 'mô hình ngôn ngữ' của substrate KHÔNG có agenda/retrieval — thước đo thuần
    khả năng dự đoán token kế của đồ thị. D2 (q·k) sẽ cắm thêm kênh vào đây để đo delta.
    TK_031: `varlm_weight`>0 + state.varlm → trộn kênh biến-bậc (ctx_ids = prefix) vào micro.
    """
    base = np.zeros(vocab_size, dtype=np.float64)
    for v, w in state.adjacency.row_readout(w0).items():
        if v < vocab_size and w > 0:
            base[v] = w
    micro = np.zeros(vocab_size, dtype=np.float64)
    for v, w in state.adjacency.layer(EdgeKind.SEQUENTIAL).row(w0).items():
        if v < vocab_size and w > 0:
            micro[v] = w
    varlm = getattr(state, "varlm", None)
    if varlm is not None and varlm_weight > 0 and ctx_ids:
        ctx = ctx_ids
        if getattr(varlm, "use_clusters", False) and state.parents is not None:
            ctx = state.parents.seg.encode_tokens_with_clusters(ctx_ids)
        pp = varlm.scores(ctx[-varlm.max_ctx:])
        if pp:
            bmax = micro.max() or 1.0
            pmax = max(pp.values()) or 1.0
            for v, p in pp.items():
                if v < vocab_size:
                    micro[v] += varlm_weight * (p / pmax) * bmax
    mmax = micro.max() or 1.0
    scores = base * (micro_floor + micro / mmax)
    return mask_non_surface(scores, state.vocab)


def next_token_lm(
    state: GraphState,
    texts: list[str],
    *,
    max_len: int = 40,
    micro_floor: float = 0.15,
    varlm_weight: float = 0.0,
) -> LMRuler:
    """Thước next-token KHÔNG bão hoà: log2(rank) + MRR + top-k trên corpus held-out.

    Với mỗi vị trí i (prefix → gold=ids[i]): tính phân bố base-LM tại w0=ids[i-1], xếp hạng
    gold trong các ứng viên finite. Bỏ qua gold OOV/non-surface (không chấm được rank).
    """
    vocab_size = len(state.vocab)
    sum_bits = 0.0
    sum_rr = 0.0
    t1 = t5 = t10 = 0
    n_pos = 0
    n_oov = 0
    for text in texts:
        tokens = tokenize(text)
        if len(tokens) < 2:
            continue
        ids = list(state.vocab.encode_tokens(tokens))
        for i in range(1, min(len(ids), max_len)):
            gold = ids[i]
            gtok = state.vocab.token_of(gold)
            if gold < 0 or gold >= vocab_size or not is_surface_token(gtok):
                n_oov += 1
                continue
            scores = _base_lm_scores(
                state, ids[i - 1], vocab_size, micro_floor,
                ctx_ids=ids[:i], varlm_weight=varlm_weight,
            )
            gscore = scores[gold]
            if not np.isfinite(gscore):
                n_oov += 1
                continue
            finite = np.isfinite(scores)
            # rank = 1 + số ứng viên finite có điểm > gold (ties → cùng rank tối thiểu)
            rank = 1 + int(np.sum(scores[finite] > gscore))
            sum_bits += math.log2(rank)
            sum_rr += 1.0 / rank
            t1 += rank <= 1
            t5 += rank <= 5
            t10 += rank <= 10
            n_pos += 1
    if n_pos == 0:
        return LMRuler(0.0, 0.0, 0.0, 0.0, 0.0, 0, n_oov)
    return LMRuler(
        bits=sum_bits / n_pos,
        mrr=sum_rr / n_pos,
        top1=t1 / n_pos,
        top5=t5 / n_pos,
        top10=t10 / n_pos,
        n_pos=n_pos,
        n_oov=n_oov,
    )


# TK_019: bits_per_byte (proxy giả 0.85/0.05, đường v2) ĐÃ XÓA — dùng next_token_lm.
