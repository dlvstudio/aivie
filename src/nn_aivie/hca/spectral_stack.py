"""
Spectral stack 4 tầng — Attn(Wa) → Amp(Wffn) → top-n, đồng cấu.

Wa/Wffn chiết từ PPMI có hướng (W_seq) bằng residual SVD deflation.
Không gán sẵn “tầng = cú pháp/style/fact” — chuyên hóa nổi sau khi đo.

Luật 0: view offline từ A, không backprop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, svds

from nn_aivie.core.state import GraphState
from nn_aivie.hca.attention_qk import build_directed_ppmi
from nn_aivie.hca.objcache import cache_get, cache_put

N_LAYERS = 4
DEFAULT_D = 32  # nửa phổ mỗi tầng; 2d = Wa+Wffn; 4×2d = 256 svds max
DEFAULT_NS = (64, 48, 32, 24)
_STACK_CACHE: dict[int, "SpectralStack"] = {}


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Softmax theo hàng; ổn định số."""
    if logits.size == 0:
        return logits
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(z, -40, 40))
    s = e.sum(axis=-1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return e / s


def _l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


@dataclass
class SpectralLayer:
    """Một tầng: Wa=(Wq,Wk), Wffn=(U_ffn, s_ffn). Không nhãn chức năng."""

    wq: np.ndarray  # (N, d)
    wk: np.ndarray  # (N, d)
    u_ffn: np.ndarray  # (N, d)
    s_ffn: np.ndarray  # (d,)
    n_keep: int = 48

    @property
    def d(self) -> int:
        return int(self.wq.shape[1])


@dataclass
class SpectralStack:
    layers: list[SpectralLayer] = field(default_factory=list)
    vocab_size: int = 0
    source: str = "ppmi_directed_wseq"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload: dict = {
            "vocab_size": self.vocab_size,
            "n_layers": len(self.layers),
            "source": self.source,
        }
        for i, ly in enumerate(self.layers):
            payload[f"L{i}_wq"] = ly.wq.astype(np.float32)
            payload[f"L{i}_wk"] = ly.wk.astype(np.float32)
            payload[f"L{i}_uffn"] = ly.u_ffn.astype(np.float32)
            payload[f"L{i}_sffn"] = ly.s_ffn.astype(np.float32)
            payload[f"L{i}_nkeep"] = np.array([ly.n_keep], dtype=np.int32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "SpectralStack":
        z = np.load(path, allow_pickle=False)
        n = int(z["n_layers"])
        layers = []
        for i in range(n):
            layers.append(
                SpectralLayer(
                    wq=z[f"L{i}_wq"].astype(np.float64),
                    wk=z[f"L{i}_wk"].astype(np.float64),
                    u_ffn=z[f"L{i}_uffn"].astype(np.float64),
                    s_ffn=z[f"L{i}_sffn"].astype(np.float64),
                    n_keep=int(z[f"L{i}_nkeep"][0]),
                )
            )
        return cls(
            layers=layers,
            vocab_size=int(z["vocab_size"]),
            source=str(z["source"]) if "source" in z.files else "ppmi_directed_wseq",
        )


def _svd_lm(op_or_mat, k: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """svds largest-magnitude; trả U (n,k), s (k,), Vt (k,n) đã sort giảm dần."""
    k = max(1, min(k, n - 2))
    u, s, vt = svds(op_or_mat, k=k, which="LM")
    order = np.argsort(-s)
    return u[:, order], s[order], vt[order]


def _residual_operator(
    M: csr_matrix, factors: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> LinearOperator:
    """matvec = M x − Σ_f U (s ⊙ (Vᵀ x))."""

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        y = M @ x
        for U, s, Vt in factors:
            y = y - U @ (s * (Vt @ x))
        return y

    def rmatvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        y = M.T @ x
        for U, s, Vt in factors:
            y = y - Vt.T @ (s * (U.T @ x))
        return y

    return LinearOperator(M.shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64)


def build_spectral_stack(
    state: GraphState,
    *,
    d: int = DEFAULT_D,
    n_layers: int = N_LAYERS,
    n_keeps: tuple[int, ...] = DEFAULT_NS,
    logger=None,
) -> SpectralStack:
    """Residual SVD deflation trên PPMI(W_seq) → 4 tầng Wa/Wffn trực giao phổ."""
    n = len(state.vocab)
    if logger:
        logger(f"spectral_stack: PPMI directed W_seq (N={n}) …")
    M = build_directed_ppmi(state)
    if M.nnz == 0:
        if logger:
            logger("spectral_stack: PPMI rỗng → zero stack")
        z = np.zeros((n, d))
        layers = [
            SpectralLayer(
                z.copy(),
                z.copy(),
                z.copy(),
                np.zeros(d),
                n_keeps[min(i, len(n_keeps) - 1)],
            )
            for i in range(n_layers)
        ]
        return SpectralStack(layers=layers, vocab_size=n)

    factors: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    layers: list[SpectralLayer] = []
    k_full = 2 * d

    for ℓ in range(n_layers):
        if logger:
            logger(f"  layer {ℓ+1}/{n_layers}: svds k={k_full} …")
        op = _residual_operator(M, factors) if factors else M.asfptype()
        try:
            U, s, Vt = _svd_lm(op, k_full, n)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger(f"  svds failed ({exc}); pad zero layer")
            z = np.zeros((n, d))
            layers.append(
                SpectralLayer(
                    z,
                    z.copy(),
                    z.copy(),
                    np.zeros(d),
                    n_keeps[min(ℓ, len(n_keeps) - 1)],
                )
            )
            continue
        mid = d
        if U.shape[1] < k_full:
            pad = k_full - U.shape[1]
            U = np.pad(U, ((0, 0), (0, pad)))
            s = np.pad(s, (0, pad))
            Vt = np.pad(Vt, ((0, pad), (0, 0)))
        root = np.sqrt(np.maximum(s[:mid], 0.0))
        wq = U[:, :mid] * root
        wk = Vt[:mid, :].T * root
        u_ffn = U[:, mid : mid + d]
        s_ffn = np.maximum(s[mid : mid + d], 0.0)
        if u_ffn.shape[1] < d:
            u_ffn = np.pad(u_ffn, ((0, 0), (0, d - u_ffn.shape[1])))
            s_ffn = np.pad(s_ffn, (0, d - s_ffn.shape[0]))
        nk = n_keeps[min(ℓ, len(n_keeps) - 1)]
        layers.append(SpectralLayer(wq=wq, wk=wk, u_ffn=u_ffn, s_ffn=s_ffn, n_keep=nk))
        factors.append((U[:, :k_full], s[:k_full], Vt[:k_full, :]))

    if logger:
        logger(f"✓ spectral_stack {n_layers} layers d={d}")
    return SpectralStack(layers=layers, vocab_size=n, source="ppmi_wseq_nostruct_v2")


def ensure_spectral_stack(
    state: GraphState,
    *,
    d: int = DEFAULT_D,
    sidecar: str | Path | None = None,
    force_rebuild: bool = False,
    logger=None,
) -> SpectralStack:
    """Lazy cache; optional load/save sidecar npz cạnh checkpoint."""
    _hit = None if force_rebuild else cache_get(_STACK_CACHE, state)
    if _hit is not None:
        return _hit
    path = Path(sidecar) if sidecar else None
    if path is not None and path.is_file() and not force_rebuild:
        try:
            stack = SpectralStack.load(path)
            if (
                stack.vocab_size == len(state.vocab)
                and len(stack.layers) == N_LAYERS
                and getattr(stack, "source", "") == "ppmi_wseq_nostruct_v2"
            ):
                cache_put(_STACK_CACHE, state, stack)
                if logger:
                    logger(f"spectral_stack loaded {path}")
                return stack
            if logger:
                logger("spectral_stack sidecar stale source → rebuild")
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger(f"spectral_stack load fail ({exc}); rebuild")
    stack = build_spectral_stack(state, d=d, logger=logger)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stack.save(path)
        if logger:
            logger(f"spectral_stack saved {path}")
    cache_put(_STACK_CACHE, state, stack)
    return stack


def layer_orthogonality(stack: SpectralStack) -> dict[str, float]:
    """‖Wa_iᵀ Wa_j‖F / norms — gần 0 nếu residual ổn."""
    out: dict[str, float] = {}
    L = stack.layers
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            qi, qj = L[i].wq, L[j].wq
            ni = np.linalg.norm(qi) * np.linalg.norm(qj) + 1e-12
            out[f"wq_{i}_{j}"] = float(np.linalg.norm(qi.T @ qj) / ni)
            ui, uj = L[i].u_ffn, L[j].u_ffn
            nu = np.linalg.norm(ui) * np.linalg.norm(uj) + 1e-12
            out[f"uffn_{i}_{j}"] = float(np.linalg.norm(ui.T @ uj) / nu)
    return out


def _amp_ffn(layer: SpectralLayer, h_S: np.ndarray, S: np.ndarray) -> np.ndarray:
    """z = U_S · σ(Σ · (U_Sᵀ h)) — spectral Hopfield trên cửa sổ S."""
    U = layer.u_ffn[S]  # (|S|, d)
    s = layer.s_ffn
    code = U.T @ h_S  # (d,)
    code = _relu(s * code)
    return U @ code  # (|S|,)


def wseq_topk_ids(state, w0: int, k: int = 64) -> list[int]:
    """Top-k successor theo W_seq thô — dùng mở cửa sổ S (không dùng Wa hub)."""
    from nn_aivie.core.edge_kinds import EdgeKind

    if k <= 0:
        return []
    row = state.adjacency.layer(EdgeKind.SEQUENTIAL).row(int(w0))
    if not row:
        return []
    items = [(int(j), float(w)) for j, w in row.items() if w > 0]
    if not items:
        return []
    items.sort(key=lambda t: -t[1])
    return [j for j, _ in items[:k]]


def spectral_forward_kwargs(config, state, history_ids: list[int]) -> dict:
    """Tham số forward: positional pool seq trên cả history (+ decay)."""
    from nn_aivie.hca.context_positional import positional_wseq_pool

    k = int(getattr(config, "ar_spectral_s_wseq_k", 64))
    lam = float(getattr(config, "ar_ctx_lambda", 0.85))
    ctx_len = int(getattr(config, "ar_ctx_len", 12))
    use_pos = bool(getattr(config, "ar_ctx_positional", True))
    use_ppmi = bool(getattr(config, "ar_wseq_ppmi", True))
    skip_struct = bool(getattr(config, "ar_wseq_skip_structural", True))
    w0 = int(history_ids[-1]) if history_ids else 0
    if use_pos and history_ids:
        neighbors, wseq_scores = positional_wseq_pool(
            state,
            history_ids,
            lam=lam,
            ctx_len=ctx_len,
            k_per=max(8, k // 3),
            k_total=k,
            use_ppmi=use_ppmi,
            skip_structural=skip_struct,
        )
    else:
        from nn_aivie.core.edge_kinds import EdgeKind
        from nn_aivie.hca.context_positional import is_structural_token

        src = w0
        if skip_struct and history_ids and is_structural_token(state.vocab, w0):
            for tid in reversed(history_ids):
                if not is_structural_token(state.vocab, tid):
                    src = int(tid)
                    break
        if use_ppmi:
            from nn_aivie.hca.attention_qk import ensure_directed_ppmi

            row = ensure_directed_ppmi(state).getrow(src)
            wseq_scores = {
                int(j): float(c) for j, c in zip(row.indices, row.data) if c > 0
            }
        else:
            row = state.adjacency.layer(EdgeKind.SEQUENTIAL).row(src)
            wseq_scores = (
                {int(j): float(w) for j, w in row.items() if w > 0} if row else {}
            )
        neighbors = sorted(wseq_scores, key=wseq_scores.get, reverse=True)[:k]
    qmode = str(getattr(config, "ar_spectral_query_mode", "positional"))
    if use_pos and qmode == "w0":
        qmode = "positional"
    return {
        "wseq_neighbor_ids": neighbors,
        "wseq_scores": wseq_scores,
        "query_mode": qmode,
        "combine": str(getattr(config, "ar_spectral_combine", "residual")),
        "succ_inject": bool(getattr(config, "ar_spectral_succ_inject", False)),
        "use_amp": bool(getattr(config, "ar_spectral_use_amp", True)),
        "attn_temp": float(getattr(config, "ar_spectral_attn_temp", 0.45)),
        "use_wa_neighbors": False,
        "wseq_k": k,
        "wseq_attn_beta": float(getattr(config, "ar_spectral_wseq_attn_beta", 2.0)),
        "pos_lambda": lam,
        "pos_ctx_len": ctx_len,
        "floor": float(getattr(config, "ar_spectral_stack_floor", 0.15)),
    }


def _build_window_S(
    stack: SpectralStack,
    history_ids: list[int],
    candidate_ids: list[int] | None,
    entity_ids: list[int] | tuple[int, ...],
    *,
    wseq_neighbor_ids: list[int] | None = None,
    use_wa_neighbors: bool = False,
    wseq_k: int = 64,
) -> np.ndarray:
    """S = history ∪ entity ∪ candidates ∪ W_seq(w0) top-k.

    Mặc định KHÔNG thêm neighbor Wa (đo: hub Thiên/Thần, gold lệch S).
    """
    n = stack.vocab_size
    S_set: set[int] = set()
    for t in history_ids[-96:]:
        if 0 <= t < n:
            S_set.add(int(t))
    for t in entity_ids:
        if 0 <= t < n:
            S_set.add(int(t))
    if candidate_ids:
        for t in candidate_ids:
            if 0 <= t < n:
                S_set.add(int(t))
    if wseq_neighbor_ids:
        for t in wseq_neighbor_ids[: max(wseq_k, 0)]:
            if 0 <= t < n:
                S_set.add(int(t))
    if use_wa_neighbors and history_ids and stack.layers:
        w0 = int(history_ids[-1])
        if 0 <= w0 < n:
            s0 = stack.layers[0].wk @ stack.layers[0].wq[w0]
            top = np.argpartition(-s0, min(32, n - 1))[:32]
            S_set.update(int(j) for j in top if j >= 0)
    if len(S_set) < 4:
        S_set.update(range(min(32, n)))
    return np.array(sorted(S_set), dtype=np.int64)


def forward_spectral_stack(
    stack: SpectralStack,
    *,
    history_ids: list[int],
    candidate_ids: list[int] | None = None,
    entity_ids: list[int] | tuple[int, ...] = (),
    emb_matrix: np.ndarray | None = None,
    floor: float = 0.15,
    debug: bool = False,
    wseq_neighbor_ids: list[int] | None = None,
    wseq_scores: dict[int, float] | None = None,
    query_mode: str = "w0",
    combine: str = "residual",
    succ_inject: bool = False,
    use_amp: bool = True,
    attn_temp: float = 0.45,
    use_wa_neighbors: bool = False,
    wseq_k: int = 64,
    wseq_attn_beta: float = 2.0,
    pos_lambda: float = 0.85,
    pos_ctx_len: int = 48,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Chạy 4 tầng đồng cấu; trả scores dày (vocab_size,).

    query_mode:
      - positional: q = Σ λ^{T-i} Wq[w_i]  (ngữ cảnh + vị trí)
      - w0: chỉ token cuối
      - history: bag Wqᵀ h trên S
    """
    from nn_aivie.hca.context_positional import positional_query

    n = stack.vocab_size
    info: dict = {
        "layers": [],
        "S": [],
        "mix_top": [],
        "succ_l0_top": [],
        "query_mode": query_mode,
        "combine": combine,
    }
    if not stack.layers:
        return (np.ones(n), info) if debug else np.ones(n)

    S = _build_window_S(
        stack,
        history_ids,
        candidate_ids,
        entity_ids,
        wseq_neighbor_ids=wseq_neighbor_ids,
        use_wa_neighbors=use_wa_neighbors,
        wseq_k=wseq_k,
    )
    m = len(S)
    inv = {int(S[i]): i for i in range(m)}
    if debug:
        info["S"] = [int(x) for x in S]
        info["|S|"] = m

    # H₀: mass trên S — recency + entity
    h = np.zeros(m, dtype=np.float64)
    w0 = int(history_ids[-1]) if history_ids else -1
    ctx_h = history_ids[-pos_ctx_len:] if history_ids else []
    if ctx_h:
        from nn_aivie.hca.context_positional import positional_weights

        ws = positional_weights(len(ctx_h), pos_lambda)
        for tid, wt in zip(ctx_h, ws):
            if tid in inv:
                h[inv[tid]] += float(wt)
    if w0 in inv and h.sum() <= 0:
        h[inv[w0]] = 1.0
    for tid in entity_ids:
        if tid in inv:
            h[inv[int(tid)]] += 0.2
    if h.sum() <= 0:
        h[:] = 1.0 / m
    h = h / (h.sum() + 1e-12)
    h0 = h.copy()

    layer_canals: list[np.ndarray] = []
    temp = max(float(attn_temp), 1e-3)

    for li, ly in enumerate(stack.layers):
        Wk = ly.wk[S]
        if query_mode == "history":
            Wq = ly.wq[S]
            q = Wq.T @ h
        elif query_mode == "w0":
            q = ly.wq[w0] if 0 <= w0 < n else ly.wq[S].T @ h
        else:
            # positional (default)
            q = positional_query(
                ly.wq, history_ids, lam=pos_lambda, ctx_len=pos_ctx_len
            )
        logits = (Wk @ q) / (max(np.sqrt(ly.d), 1e-6) * temp)
        if wseq_scores and wseq_attn_beta > 0:
            bias = np.array(
                [np.log1p(wseq_scores.get(int(S[i]), 0.0)) for i in range(m)],
                dtype=np.float64,
            )
            logits = logits + float(wseq_attn_beta) * bias
        alpha = _softmax_rows(logits.reshape(1, -1)).ravel()
        if emb_matrix is not None and emb_matrix.shape[0] >= n:
            V = emb_matrix[S].astype(np.float64)
            V = _l2_normalize(V, axis=1)
            ctx_e = alpha @ V
            ctx = V @ ctx_e
        else:
            ctx = alpha.copy()
        z = _amp_ffn(ly, ctx, S) if use_amp else np.zeros_like(ctx)
        h_pre = alpha + 0.35 * ctx + (0.25 * z if use_amp else 0.0)
        h = _l2_normalize(np.abs(h_pre).reshape(1, -1), axis=1).ravel()
        nk = min(ly.n_keep, m)
        dropped = 0
        if nk < m:
            keep_idx = np.argpartition(-h, nk - 1)[:nk]
            mask = np.zeros(m, dtype=bool)
            mask[keep_idx] = True
            dropped = int((~mask).sum())
            h = np.where(mask, h, 0.0)
            h = h / (h.sum() + 1e-12)
        layer_canals.append(h.copy())
        if debug:

            def _topk(vec: np.ndarray, k: int = 8) -> list[tuple[int, float]]:
                k = min(k, len(vec))
                idx = np.argpartition(-vec, k - 1)[:k]
                idx = idx[np.argsort(-vec[idx])]
                return [(int(S[i]), float(vec[i])) for i in idx]

            info["layers"].append(
                {
                    "ℓ": li,
                    "s_ffn_sum": float(ly.s_ffn.sum()),
                    "s_ffn_max": float(ly.s_ffn.max()) if ly.s_ffn.size else 0.0,
                    "alpha_entropy": float(-np.sum(alpha * np.log(alpha + 1e-12))),
                    "alpha_top": _topk(alpha),
                    "ctx_top": _topk(np.abs(ctx)),
                    "z_l2": float(np.linalg.norm(z)),
                    "h_top": _topk(h),
                    "n_keep": nk,
                    "dropped": dropped,
                    "overlap_h0": float(
                        np.dot(h, h0)
                        / (np.linalg.norm(h) * np.linalg.norm(h0) + 1e-12)
                    ),
                }
            )

    if combine == "hadamard":
        mix = np.ones(m, dtype=np.float64)
        for canal in layer_canals:
            mix = mix * (floor + canal)
    elif combine == "last":
        mix = layer_canals[-1].copy() if layer_canals else np.ones(m) / m
    else:
        mix = np.zeros(m, dtype=np.float64)
        for canal in layer_canals:
            mix = mix + canal
    mix = mix / (mix.max() + 1e-12)

    out = np.zeros(n, dtype=np.float64)
    out[S] = mix
    if succ_inject and history_ids and stack.layers and 0 <= w0 < n:
        succ = stack.layers[0].wk @ stack.layers[0].wq[w0]
        succ = np.maximum(succ, 0.0)
        if succ.max() > 0:
            succ = succ / succ.max()
            if debug:
                top = np.argpartition(-succ, min(8, n - 1))[:8]
                top = top[np.argsort(-succ[top])]
                info["succ_l0_top"] = [(int(j), float(succ[j])) for j in top]
            out = out * (floor + 0.35 * succ) + 0.15 * succ
    if debug:
        nz = np.flatnonzero(out > 0)
        if nz.size:
            top = nz[np.argsort(-out[nz])[:12]]
            info["mix_top"] = [(int(j), float(out[j])) for j in top]
            info["out_nnz"] = int(nz.size)
        return out, info
    return out


def wa_successor_scores(stack: SpectralStack, layer_idx: int, token_id: int) -> np.ndarray:
    """q·k của Wa_ℓ — đo held-out / khám phá tầng."""
    ly = stack.layers[layer_idx]
    if token_id < 0 or token_id >= ly.wq.shape[0]:
        return np.zeros(ly.wk.shape[0])
    return ly.wk @ ly.wq[token_id]


def top_patterns(
    stack: SpectralStack, vocab, layer_idx: int, k: int = 5
) -> list[tuple[str, str, float]]:
    """Đọc tay: cặp (trái mạnh U, phải mạnh V) theo cột phổ Wa — khám phá."""
    ly = stack.layers[layer_idx]
    d = ly.d
    pairs = []
    for c in range(min(k, d)):
        left = int(np.argmax(np.abs(ly.wq[:, c])))
        right = int(np.argmax(np.abs(ly.wk[:, c])))
        pairs.append(
            (
                vocab.token_of(left),
                vocab.token_of(right),
                float(ly.wq[left, c] * ly.wk[right, c]),
            )
        )
    return pairs
