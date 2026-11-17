"""W_skip[d] — cạnh có hướng theo khoảng cách (TK_045, ý tác giả).

Mỗi d=1..D: CSR (N×N) đếm a→b khi a đứng trước b đúng d token (trên cả emit;
không cắt câu — §12.143). Decode: mỗi token lịch sử đề xuất next ở đúng d,
rồi GIAO / tích mềm các đề xuất.

Khác `pos_rel` (1 scalar/token) và khác SENT_L2R gộp λ^d: giữ TỪNG bin d để giao.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from nn_aivie.core.state import GraphState
from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.hca.context_positional import is_structural_token
from nn_aivie.hca.objcache import cache_get, cache_put
from nn_aivie.hca.sent_intra import _content_positions, iter_sentence_spans

_WSKIP: dict[int, "WSkip"] = {}


@dataclass
class WSkip:
    """mats[d] = CSR a→b ở khoảng cách d; mats[0] = ma trận rỗng placeholder."""

    mats: list[csr_matrix]
    d_max: int
    min_count: int
    n_spans: int = 0
    n_edges: int = 0  # tổng lượt cộng (trước gộp)

    def row_topk(
        self,
        src: int,
        d: int,
        topk: int = 50,
        *,
        gidf_fn=None,
        gidf_min: float = 0.0,
    ) -> list[tuple[int, float]]:
        if d < 1 or d > self.d_max or topk <= 0:
            return []
        row = self.mats[d].getrow(int(src))
        if row.nnz == 0:
            return []
        idx = row.indices
        dat = row.data
        keep = []
        for j, w in zip(idx, dat):
            if w < self.min_count:
                continue
            j = int(j)
            if gidf_fn is not None and float(gidf_fn(j)) < float(gidf_min):
                continue
            keep.append((j, float(w)))
        keep.sort(key=lambda x: -x[1])
        return keep[:topk]

    def row_score_map(
        self,
        src: int,
        d: int,
        *,
        gidf_fn=None,
        gidf_min: float = 0.0,
    ) -> dict[int, float]:
        """P̃(b|src,d) ∝ count nếu count≥min_count; chuẩn hoá theo tổng hàng đã lọc."""
        if d < 1 or d > self.d_max:
            return {}
        row = self.mats[d].getrow(int(src))
        if row.nnz == 0:
            return {}
        items = []
        for j, w in zip(row.indices, row.data):
            if w < self.min_count:
                continue
            j = int(j)
            if gidf_fn is not None and float(gidf_fn(j)) < float(gidf_min):
                continue
            items.append((j, float(w)))
        if not items:
            return {}
        s = sum(w for _, w in items)
        if s <= 0:
            return {}
        return {j: w / s for j, w in items}


# §170 — cạnh LOẠI SUY (REINFORCED, TK_054) được cắm vào W_skip mats[1] để
# THAM GIA phiếu như mọi cạnh khác (đáp câu hỏi tác giả §169: cạnh bị "bỏ qua"
# CHỈ vì để nhầm hộp TypedAdjacency; vote đọc W_skip). Scale × trọng số lưu để
# tương thích thang phiếu câu. 0 = không cắm.
ANALOGY_TO_WSKIP_SCALE = 60.0


def build_w_skip(
    state: GraphState,
    *,
    d_max: int = 12,
    min_count: int = 2,
    content_only: bool = True,
) -> WSkip:
    """Skip-edge trên CẢ emit (không cắt câu).

    Trước: `iter_sentence_spans` cắt ở `.!?` → `Long`↛`Đây` dù cùng parent.
    Tác giả §12.143: chuỗi liền — `đẹp. Việt` vẫn có `đẹp→Việt`
    (d=1 nếu content_only bỏ `.`; d=2 nếu giữ `.`).
    """
    vocab = state.vocab
    ps = state.parents
    n = len(vocab)
    empty = csr_matrix((n, n), dtype=np.float64)
    if ps is None or len(ps.parents) == 0:
        return WSkip([empty] * (d_max + 1), d_max=d_max, min_count=min_count)

    struct = [is_structural_token(vocab, i) for i in range(n)]
    # lists per distance
    rows: list[list[int]] = [[] for _ in range(d_max + 1)]
    cols: list[list[int]] = [[] for _ in range(d_max + 1)]
    vals: list[list[float]] = [[] for _ in range(d_max + 1)]
    n_spans = 0
    n_edges = 0

    for rec in ps.parents:
        seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        if len(seq) < 2:
            continue
        pw = float(max(1, rec.power)) * float(getattr(rec, "priority", 1.0) or 1.0)
        # §12.143 — không cắt span; skip xuyên `.` trong cùng emit.
        toks = _content_positions(
            seq, vocab, content_only=content_only, struct_mask=struct
        )
        if len(toks) < 2:
            continue
        n_spans += 1
        L = len(toks)
        for i in range(L):
            for j in range(i + 1, min(L, i + 1 + d_max)):
                d = j - i
                a, b = int(toks[i]), int(toks[j])
                if 0 <= a < n and 0 <= b < n:
                    rows[d].append(a)
                    cols[d].append(b)
                    vals[d].append(pw)
                    n_edges += 1

    mats: list[csr_matrix] = [empty]
    for d in range(1, d_max + 1):
        if not rows[d]:
            mats.append(empty.copy())
            continue
        m = coo_matrix(
            (np.asarray(vals[d], dtype=np.float64),
             (np.asarray(rows[d], dtype=np.int32),
              np.asarray(cols[d], dtype=np.int32))),
            shape=(n, n),
        ).tocsr()
        # gộp trùng
        m.sum_duplicates()
        mats.append(m)

    # §170 — CẮM cạnh REINFORCED (loại suy đã xác nhận) vào mats[1] như cạnh
    # W_skip thật. state.adjacency là hộp KHÁC (vote không đọc), nên phải chuyển
    # vào mats. Trọng số × scale để có tiếng nói tương xứng phiếu câu.
    if ANALOGY_TO_WSKIP_SCALE > 0 and getattr(state, "adjacency", None) is not None:
        try:
            from nn_aivie.core.edge_kinds import EdgeKind as _EK
            _lay = state.adjacency._layers.get(_EK.REINFORCED)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            _lay = None
        if _lay is not None:
            _ri = []; _rj = []; _rv = []
            for _i, _j, _w in _lay.items():
                if _w > 0 and 0 <= int(_i) < n and 0 <= int(_j) < n:
                    _ri.append(int(_i)); _rj.append(int(_j))
                    _rv.append(float(_w) * ANALOGY_TO_WSKIP_SCALE)
            if _ri:
                _add = coo_matrix(
                    (np.asarray(_rv, dtype=np.float64),
                     (np.asarray(_ri, dtype=np.int32),
                      np.asarray(_rj, dtype=np.int32))),
                    shape=(n, n),
                ).tocsr()
                mats[1] = (mats[1] + _add).tocsr()
                mats[1].sum_duplicates()
                n_edges += len(_ri)

    return WSkip(
        mats=mats, d_max=d_max, min_count=min_count, n_spans=n_spans, n_edges=n_edges
    )


def ensure_w_skip(
    state: GraphState,
    *,
    d_max: int = 12,
    min_count: int = 2,
    content_only: bool = True,
) -> WSkip:
    sub = (int(d_max), int(min_count), bool(content_only))
    cached = cache_get(_WSKIP, state, sub)
    if cached is not None:
        return cached
    built = build_w_skip(
        state, d_max=d_max, min_count=min_count, content_only=content_only
    )
    return cache_put(_WSKIP, state, built, sub)


def clear_w_skip_cache() -> None:
    _WSKIP.clear()


def constrain_next(
    wskip: WSkip,
    history_ids: list[int],
    vocab,
    *,
    topk: int = 64,
    ctx_max: int = 8,
    content_only: bool = True,
    lam: float = 0.85,
    gidf_fn=None,
    gidf_min: float = 0.0,
) -> dict:
    """Đề xuất next từ mỗi token lịch sử ở đúng d, rồi giao cứng + điểm mềm.

    soft[tid] = n_hits + ε·Σ λ^{d-1} log P̃
      n_hits = số proposer có tid trong top-k  ← đúng ý "giao / bỏ phiếu"
      hạng phụ có giảm theo d để d=1 không bị dìm bởi d xa nhiễu.
    gidf_min>0: bỏ hub (là/có/không…) khỏi tập đề xuất — bắt buộc trên v12.
    """
    from nn_aivie.hca.context_positional import is_structural_token

    n = len(vocab)
    struct = [is_structural_token(vocab, i) for i in range(n)]
    hist = []
    for t in history_ids:
        t = int(t)
        if content_only:
            if 0 <= t < n and struct[t]:
                continue
            if not vocab.is_surface(t):
                continue
        hist.append(t)
    if not hist:
        return {"proposers": [], "hard": set(), "soft": {}, "empty_hard": True,
                "n_active": 0}

    window = hist[-ctx_max:]
    L = len(window)
    proposers = []
    sets: list[set[int]] = []
    hits: dict[int, int] = {}
    soft_aux: dict[int, float] = {}

    for i, src in enumerate(window):
        d = L - i
        if d > wskip.d_max:
            continue
        top = wskip.row_topk(
            src, d, topk=topk, gidf_fn=gidf_fn, gidf_min=gidf_min
        )
        pmap = wskip.row_score_map(
            src, d, gidf_fn=gidf_fn, gidf_min=gidf_min
        )
        proposers.append({
            "src": int(src),
            "d": int(d),
            "n_prop": len(top),
            "top": top[:12],
        })
        if not top:
            continue
        sset = {j for j, _ in top}
        sets.append(sset)
        w_d = float(lam) ** (d - 1)
        for j in sset:
            hits[j] = hits.get(j, 0) + 1
        for j, p in pmap.items():
            soft_aux[j] = soft_aux.get(j, 0.0) + w_d * float(np.log(max(p, 1e-12)))

    if not sets:
        return {"proposers": proposers, "hard": set(), "soft": {}, "empty_hard": True,
                "n_active": 0}

    hard = sets[0].copy()
    for s in sets[1:]:
        hard &= s

    soft: dict[int, float] = {}
    for j, h in hits.items():
        soft[j] = float(h) + 1e-3 * soft_aux.get(j, 0.0)

    return {
        "proposers": proposers,
        "hard": hard,
        "soft": soft,
        "empty_hard": len(hard) == 0,
        "n_active": len(sets),
    }


def soft_constraint_logits(
    state: GraphState,
    history_ids: list[int],
    *,
    d_max: int = 12,
    min_count: int = 2,
    topk: int = 64,
    ctx_max: int = 8,
    hard_bonus: float = 2.0,
    lam: float = 0.85,
    hard_gate: bool = False,
    gidf_min: float = 4.0,
) -> np.ndarray:
    """Vector (N,) điểm ràng buộc — dùng làm kênh matrix_core (mặc định OFF).

    hard_gate=True: nếu hard∩ ≠ ∅ chỉ ghi điểm trên ∩ (ý tác giả); soft chỉ khi ∩ rỗng.
    gidf_min mặc định 4.0 — lọc hub khỏi đề xuất (cùng ngưỡng cột `lạc`).
    """
    n = len(state.vocab)
    out = np.zeros(n, dtype=np.float64)
    wskip = ensure_w_skip(state, d_max=d_max, min_count=min_count)
    gidf_fn = None
    if state.parents is not None and float(gidf_min) > 0:
        ps = state.parents
        gidf_fn = lambda tid, _ps=ps: float(_ps._gidf(int(tid)))  # noqa: SLF001
    res = constrain_next(
        wskip,
        history_ids,
        state.vocab,
        topk=topk,
        ctx_max=ctx_max,
        lam=lam,
        gidf_fn=gidf_fn,
        gidf_min=float(gidf_min),
    )
    soft = res["soft"]
    hard = res["hard"]
    if hard_gate and hard:
        for tid in hard:
            if 0 <= tid < n:
                out[tid] = float(soft.get(tid, 1.0)) + float(hard_bonus)
        return out
    if not soft:
        return out
    for tid, s in soft.items():
        if 0 <= tid < n:
            out[tid] = float(s)
    for tid in hard:
        if 0 <= tid < n:
            out[tid] += float(hard_bonus)
    return out


def _row_score_map(
    ws: WSkip,
    src: int,
    d: int,
    *,
    n_vocab: int,
    topk: int,
    thr: float,
    blocked_edges: set[tuple[int, int, int]] | None = None,
    blocked_edge_scale: float = 0.0,
) -> dict[int, float]:
    """Hàng W_skip[d][src] → {dst: weight}, lọc thr; topk<=0 = full CSR.

    `blocked_edges`: (src,d,dst) đã consume (§12.144). `blocked_edge_scale`=0 → bỏ;
    >0 → nhân trọng số (giảm điểm mềm).
    """
    out: dict[int, float] = {}
    use_full = int(topk) <= 0
    blk = blocked_edges
    scale = float(blocked_edge_scale)
    src_i = int(src)
    d_i = int(d)

    def _accept(bi: int, c: float) -> float | None:
        w = float(c)
        if w < thr:
            return None
        if blk is not None and (src_i, d_i, bi) in blk:
            if scale <= 0.0:
                return None
            w *= scale
            if w < thr:
                return None
        return w

    if use_full:
        if d > ws.d_max:
            return out
        row = ws.mats[d].getrow(src_i)
        if row is None or row.nnz == 0:
            return out
        for b, c in zip(row.indices, row.data):
            bi = int(b)
            w = _accept(bi, c)
            if w is None or not (0 <= bi < n_vocab):
                continue
            out[bi] = float(w)
        return out
    want = max(int(topk) * 20, int(topk))
    got = 0
    for b, c in ws.row_topk(src_i, d_i, max(want, int(topk))):
        bi = int(b)
        w = _accept(bi, c)
        if w is None or not (0 <= bi < n_vocab):
            continue
        out[bi] = float(w)
        got += 1
        if got >= int(topk):
            break
    return out


def mark_w_skip_consume(
    state: GraphState,
    hist_before: list[int],
    nxt: int,
    *,
    d_max: int,
    blocked_edges: set[tuple[int, int, int]],
    consume_edges: bool = True,
    reverse_window: int = 4,
    edge_age: dict[tuple[int, int, int], int] | None = None,
    step: int = 0,
) -> None:
    """§12.144/145 — sau sample `nxt`: consume cạnh bag→nxt (+ ngược chọn lọc).

    1) Mọi `(src∈bag, d, nxt)` trên W_skip.
    2) Chiều ngược `(nxt,d,t)`:
       - `t` trong `reverse_window` gần: luôn (chu kỳ liền).
       - `t` xa: ngược trừ khi `(nxt,1,t)` argmax d=1 (giữ `kinh→đô`).
    3) `edge_age` + hồi (§12.145): cho phép tái dùng cạnh khi đích đã ra khỏi túi.
    """
    nxt = int(nxt)
    if not hist_before or not consume_edges:
        return
    ws = ensure_w_skip(state, d_max=int(d_max), min_count=2, content_only=False)
    vocab = state.vocab
    n = len(vocab)
    bag: list[int] = []
    seen_src: set[int] = set()
    for src in hist_before:
        src = int(src)
        if src in seen_src or not (0 <= src < n):
            continue
        seen_src.add(src)
        if is_structural_token(vocab, src) or not vocab.is_surface(src):
            continue
        bag.append(src)

    def _block_row(src: int, dst: int) -> None:
        for d in range(1, int(d_max) + 1):
            if d > ws.d_max:
                break
            row = ws.mats[d].getrow(src)
            if row is None or row.nnz == 0:
                continue
            if dst in row.indices:
                key = (src, int(d), dst)
                blocked_edges.add(key)
                if edge_age is not None:
                    edge_age[key] = int(step)

    def _is_strong_d1(src: int, dst: int) -> bool:
        if 1 > ws.d_max:
            return False
        row = ws.mats[1].getrow(src)
        if row is None or row.nnz == 0:
            return False
        return int(row.indices[int(np.argmax(row.data))]) == int(dst)

    for src in bag:
        _block_row(src, nxt)
    rw = max(0, int(reverse_window))
    if rw <= 0 or not (0 <= nxt < n):
        return
    if is_structural_token(vocab, nxt) or not vocab.is_surface(nxt):
        return
    recent = set(bag[-rw:]) if bag else set()
    for t in bag:
        if t == nxt:
            continue
        if t not in recent and _is_strong_d1(nxt, t):
            continue
        _block_row(nxt, t)


def expire_w_skip_consume(
    blocked_edges: set[tuple[int, int, int]],
    edge_age: dict[tuple[int, int, int], int],
    *,
    step: int,
    recover_steps: int,
    hold_dst: set[int] | None = None,
) -> int:
    """§12.145 — mở lại cạnh đã consume sau `recover_steps` (0 = không hồi).

    `hold_dst`: không hồi cạnh có đích vẫn nằm cửa sổ gần (tránh xoay lại hub
    đang trong túi; `đô` đã ra khỏi cửa sổ → cạnh cũ vào `đô` mới được hồi).
    """
    rec = int(recover_steps)
    if rec <= 0 or not edge_age:
        return 0
    hold = hold_dst or set()
    dead = [
        e for e, t0 in edge_age.items()
        if int(step) - int(t0) >= rec and int(e[2]) not in hold
    ]
    for e in dead:
        blocked_edges.discard(e)
        del edge_age[e]
    return len(dead)


def apply_local_repeat_penalty(
    scores: np.ndarray,
    out: list[int],
    *,
    prompt_len: int,
    n_vocab: int,
    vocab,
    window: int = 8,
    scale: float = 0.15,
    min_count: int = 1,
    near_k: int = 3,
    near_scale: float = 0.08,
    outsider_prefer: float = 0.05,
) -> np.ndarray:
    """§12.145 — giảm điểm mềm token đã xuất hiện trong cửa sổ gần (túi xáo).

    Không cấm token vĩnh viễn: hết cửa sổ → không phạt.
    Nếu còn ứng viên **ngoài** cửa sổ (score>0), nhân thêm `outsider_prefer` lên
    token trong túi — phá chu kỳ hub khi chỉ xoay quan↔là↔trọng.
    """
    win = int(window)
    sc = float(scale)
    if win <= 0 or scores.size == 0:
        return scores
    lo = max(0, int(prompt_len))
    tail = out[lo:]
    content: list[int] = []
    for t in tail[-win:]:
        tid = int(t)
        if not (0 <= tid < int(n_vocab)):
            continue
        if not vocab.is_surface(tid) or is_structural_token(vocab, tid):
            continue
        content.append(tid)
    if not content:
        return scores
    counts: dict[int, int] = {}
    for tid in content:
        counts[tid] = counts.get(tid, 0) + 1
    thr = max(1, int(min_count))
    out_s = np.array(scores, dtype=np.float64, copy=True)
    if sc < 1.0 - 1e-15:
        for tid, c in counts.items():
            if c < thr:
                continue
            out_s[tid] *= sc ** float(c)
    nk = max(0, int(near_k))
    ns = float(near_scale)
    if nk > 0 and ns < 1.0 - 1e-15:
        for tid in set(content[-nk:]):
            out_s[tid] *= ns
    # Còn outsider sống → hạ mạnh túi gần (tránh xếp hạng tương đối giữa hub).
    op = float(outsider_prefer)
    recent_set = set(content)
    if op < 1.0 - 1e-15 and recent_set:
        has_out = False
        for tid in range(int(n_vocab)):
            if tid in recent_set:
                continue
            v = float(out_s[tid])
            if np.isfinite(v) and v > 0.0:
                has_out = True
                break
        if has_out:
            for tid in recent_set:
                out_s[tid] *= op
    return out_s


def blocked_dst_from_hist(
    state: GraphState,
    out: list[int],
    *,
    prompt_len: int = 0,
    gidf_min: float = 4.0,
) -> set[int]:
    """§12.144 — đích cấm = token nội dung ĐÃ CÓ trong hist (trước next).

    Không phân biệt câu / biên `.`. Hub gidf < min (`là`) vẫn được lặp.
    `prompt_len`: không block token chỉ thuộc prompt.
    """
    vocab = state.vocab
    n = len(vocab)
    lo = max(0, int(prompt_len))
    gmin = float(gidf_min)
    out_set: set[int] = set()
    for t in out[lo:]:
        tid = int(t)
        if not (0 <= tid < n) or not vocab.is_surface(tid):
            continue
        if is_structural_token(vocab, tid):
            continue
        if gmin > 0.0 and state.parents is not None:
            try:
                if float(state.parents._gidf(tid)) < gmin:  # noqa: SLF001
                    continue
            except Exception:
                pass
        out_set.add(tid)
        tok = vocab.token_of(tid)
        bare = tok.lstrip(MARK_SPACE).lstrip("Ċ")
        for alt in (bare, (MARK_SPACE + bare) if bare else ""):
            if not alt:
                continue
            aid = vocab._token_to_id.get(alt)  # noqa: SLF001
            if aid is None:
                continue
            aid = int(aid)
            if 0 <= aid < n and not is_structural_token(vocab, aid):
                if gmin <= 0.0 or state.parents is None:
                    out_set.add(aid)
                else:
                    try:
                        if float(state.parents._gidf(aid)) >= gmin:  # noqa: SLF001
                            out_set.add(aid)
                    except Exception:
                        out_set.add(aid)
    return out_set


# alias cũ
blocked_dst_from_closed_sents = blocked_dst_from_hist


def _oja_scale_row(
    weights: dict[int, float],
    *,
    oja_row: bool = True,
    degree_beta: float = 1.0,
) -> dict[int, float]:
    """§12.140 — Oja-L2 hàng + bão hoà theo nnz (mọi token, không if dấu câu).

    tilde_w = w / ||w||_2; alpha = 1/(1+beta_d*log(1+k)); trả alpha * tilde_w.
    """
    if not weights:
        return weights
    scaled = dict(weights)
    if oja_row:
        nrm = float(sum(float(c) * float(c) for c in scaled.values())) ** 0.5
        if nrm > 0.0:
            scaled = {b: float(c) / nrm for b, c in scaled.items()}
    beta = float(degree_beta)
    if beta > 0.0:
        k = len(scaled)
        alpha = 1.0 / (1.0 + beta * float(np.log1p(k)))
        scaled = {b: float(c) * alpha for b, c in scaled.items()}
    return scaled


def vote_logits(
    state: GraphState,
    history_ids: list[int],
    *,
    n_vocab: int,
    d_max: int = 16,
    lam: float = 0.5,
    topk: int = 12,
    min_count: int = 2,
    edge_min: float | None = None,
    agg: str = "sum",
    prod_window: int = 0,
    oja_row: bool = True,
    oja_degree_beta: float = 1.0,
    oja_compete_beta: float = 0.0,
    oja_indeg_beta: float = 0.75,
    blocked_edges: set[tuple[int, int, int]] | None = None,
    blocked_dst: set[int] | None = None,
    blocked_edge_scale: float = 0.0,
) -> np.ndarray:
    """TK_045 §12.106 / §12.139 / §12.140 / §12.144 — kênh PHIẾU.

    `agg="sum"`: mỗi `h[-d]` cộng `λ^(d-1)` (cũ).
    `agg="prod"`: **hợp** mọi đề xuất W_skip[d][h[-d]] (không giao); điểm =
    geo-mean trọng số của các proposer đã đề cử × số phiếu, rồi caller softmax
    (§12.139). Token chỉ một phía đề cử vẫn vào list.

    §12.140: mỗi hàng đề cử qua Oja-L2 + α(nnz); ứng viên × α(in_deg);
    tuỳ chọn oja_compete trên vector (mặc định tắt — tránh bóp thang xếp hạng).
    Không nhánh theo loại dấu câu.

    §12.144: `blocked_edges` — cạnh đã consume (nhân `blocked_edge_scale`, 0=tắt);
    `blocked_dst` tuỳ chọn (mặc định tắt ở generate — không chặn token).

    `prod_window`: số proposer nội dung gần nhất; ≤0 = hết `d_max`.
    `topk<=0`: cả hàng CSR. `edge_min=None` → 0 khi full-row.
    """
    out = np.zeros(int(n_vocab), dtype=np.float64)
    if not history_ids or d_max < 1:
        return out
    ws = ensure_w_skip(
        state, d_max=int(d_max), min_count=int(min_count), content_only=False
    )
    hist = [int(t) for t in history_ids]
    use_full = int(topk) <= 0
    thr = (
        float(edge_min)
        if edge_min is not None
        else (0.0 if use_full else float(ws.min_count))
    )
    mode = str(agg or "sum").strip().lower()
    blk_e = blocked_edges
    blk_d = blocked_dst
    blk_scale = float(blocked_edge_scale)

    def _finish(vec: np.ndarray) -> np.ndarray:
        # Bão hoà bậc vào (mọi ứng viên): hub đích bị hạ theo in-deg W_skip[d=1].
        ib = float(oja_indeg_beta)
        if ib > 0.0 and ws.d_max >= 1:
            # in-degree ≈ số nguồn có cạnh tới b trên bin d=1 (proxy bậc vào)
            indices = ws.mats[1].indices
            # đếm cột: bincount trên indices
            indeg = np.bincount(indices, minlength=int(n_vocab)).astype(np.float64)
            pos = np.flatnonzero(vec > 0)
            if pos.size:
                vec = vec.copy()
                for bi in pos:
                    k = float(indeg[int(bi)]) if int(bi) < indeg.size else 0.0
                    vec[int(bi)] *= 1.0 / (1.0 + ib * float(np.log1p(k)))
        if blk_d:
            vec = vec.copy()
            for bi in blk_d:
                if 0 <= int(bi) < int(n_vocab):
                    vec[int(bi)] = 0.0
        beta = float(oja_compete_beta)
        if beta <= 0.0:
            return vec
        from nn_aivie.hca.knowledge_seq import oja_compete

        return oja_compete(vec, beta=beta)

    if mode in {"prod", "product", "hadamard", "mul"}:
        # §12.139 — HỢP đề xuất (không giao!).
        from collections import defaultdict

        vocab = state.vocab
        maps: list[dict[int, float]] = []
        for d in range(1, min(int(d_max), len(hist)) + 1):
            src = hist[-d]
            if not (0 <= src < n_vocab):
                continue
            if is_structural_token(vocab, src):
                continue
            if not vocab.is_surface(src):
                continue
            bare = vocab.token_of(src).lstrip(MARK_SPACE).lstrip("Ċ")
            if bare.isdigit():
                continue
            m = _row_score_map(
                ws, src, d, n_vocab=n_vocab, topk=int(topk), thr=thr,
                blocked_edges=blk_e, blocked_edge_scale=blk_scale,
            )
            if m:
                maps.append(
                    _oja_scale_row(
                        m, oja_row=bool(oja_row), degree_beta=float(oja_degree_beta),
                    )
                )
        if not maps:
            return out

        def _cand_ok(b: int) -> bool:
            # Chỉ lọc non-surface / structural graph — KHÔNG if punct (§12.140).
            if not (0 <= int(b) < n_vocab):
                return False
            if blk_d is not None and int(b) in blk_d:
                return False
            if is_structural_token(vocab, int(b)):
                return False
            if not vocab.is_surface(int(b)):
                return False
            return True

        win = int(prod_window)
        if win > 0:
            maps = maps[:win]

        hits: dict[int, list[float]] = defaultdict(list)
        for i, m in enumerate(maps):
            decay = float(lam) ** i
            for b, c in m.items():
                if _cand_ok(b) and float(c) > 0.0:
                    hits[int(b)].append(float(c) * decay)
        for b, vals in hits.items():
            k = len(vals)
            prod = 1.0
            for x in vals:
                prod *= float(x)
            geo = prod ** (1.0 / float(k))
            out[int(b)] = geo * float(k)
        return _finish(out)

    # --- agg=sum (cũ) + §12.140 Oja trên từng hàng ---
    for d in range(1, min(int(d_max), len(hist)) + 1):
        src = hist[-d]
        if not (0 <= src < n_vocab):
            continue
        w = float(lam) ** (d - 1)
        m = _row_score_map(
            ws, src, d, n_vocab=n_vocab, topk=int(topk), thr=thr,
            blocked_edges=blk_e, blocked_edge_scale=blk_scale,
        )
        m = _oja_scale_row(
            m, oja_row=bool(oja_row), degree_beta=float(oja_degree_beta),
        )
        for bi, c in m.items():
            if blk_d is not None and int(bi) in blk_d:
                continue
            out[bi] += w * float(c)
    return _finish(out)


def ensure_same_sent_index(state: GraphState) -> tuple[dict, list[set], np.ndarray]:
    """Cache token→cha / content-set / structural mask (chống salad §12.132)."""
    cached = getattr(state, "_same_sent_idx", None)
    if cached is not None:
        return cached
    from collections import defaultdict

    v = state.vocab
    n = len(v)
    struct = np.array(
        [is_structural_token(v, i) for i in range(n)], dtype=bool
    )
    tok2par: dict[int, list[int]] = defaultdict(list)
    par_cont: list[set[int]] = []
    parents = state.parents.parents if state.parents is not None else ()
    for pi, rec in enumerate(parents):
        sq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        toks: set[int] = set()
        for sp in iter_sentence_spans(sq, v):
            for t in _content_positions(sp, v, content_only=True, struct_mask=struct):
                toks.add(int(t))
        par_cont.append(toks)
        for t in toks:
            tok2par[t].append(pi)
    cached = (dict(tok2par), par_cont, struct)
    setattr(state, "_same_sent_idx", cached)
    return cached


def same_sentence_bonus(
    hist_content: list[int],
    tok2par: dict,
    par_cont: list[set],
    n_vocab: int,
    *,
    window: int = 4,
    top_m: int = 8,
) -> np.ndarray:
    """Thưởng candidate CÙNG CHA với ngữ cảnh gần (probe_vote_primary / §12.106 note)."""
    from collections import defaultdict

    recent = hist_content[-window:]
    if not recent:
        return np.zeros(n_vocab, dtype=np.float64)
    cand: dict[int, int] = defaultdict(int)
    for t in recent:
        for pi in tok2par.get(int(t), ()):
            cand[int(pi)] += 1
    if not cand:
        return np.zeros(n_vocab, dtype=np.float64)
    top = sorted(cand.items(), key=lambda kv: -kv[1])[:top_m]
    out = np.zeros(n_vocab, dtype=np.float64)
    for pi, ov in top:
        w = ov / len(recent)
        for b in par_cont[pi]:
            if 0 <= b < n_vocab:
                out[b] += w
    return out


def d1_fluency_row(
    state: GraphState,
    w0: int,
    n_vocab: int,
    *,
    topk: int = 64,
    d_max: int = 8,
    min_count: int = 2,
    blocked_edges: set[tuple[int, int, int]] | None = None,
    blocked_edge_scale: float = 0.0,
) -> np.ndarray:
    """W_skip[d=1] từ token cuối (= view W_seq) — cổng fluency cục bộ chống salad.

    Tôn trọng §12.144 consume cạnh (w0,1,dst).
    """
    out = np.zeros(int(n_vocab), dtype=np.float64)
    if not (0 <= int(w0) < n_vocab):
        return out
    ws = ensure_w_skip(
        state, d_max=int(d_max), min_count=int(min_count), content_only=False
    )
    w0i = int(w0)
    scale = float(blocked_edge_scale)
    for b, c in ws.row_topk(w0i, 1, int(topk)):
        if not (0 <= b < n_vocab):
            continue
        w = float(c)
        if blocked_edges is not None and (w0i, 1, int(b)) in blocked_edges:
            if scale <= 0.0:
                continue
            w *= scale
        if w > 0.0:
            out[int(b)] = w
    return out


# §12.141/P42–P43 — EOS end_after, SENT_SUCC mở câu, content gate, hist sau biên
from nn_aivie.hca._w_skip_p42_helpers import (  # noqa: E402
    content_channel_gate,
    count_boundaries,
    eos_from_end_after,
    hist_after_last_boundary,
    sent_open_from_w_agenda,
    sent_open_from_w_skip,
)
