"""
Graph-derived soft embeddings (E) — Layer 1 of the NN-AIVIE v3 hybrid core.

KHÔNG pretrain, KHÔNG backprop. Embedding **dẫn xuất từ chính thống kê đồ thị** của mạng:
PPMI trên lớp CO_ACTIVATION → SVD rút gọn. Tương đồng ngữ nghĩa ("chó≈cún") nổi lên từ
phân phối hàng xóm mà mạng tự học (tôn trọng Luật 0 — README Mục 0).

Tham chiếu cổ điển: Levy & Goldberg 2014 (PPMI+SVD ≈ word2vec). Concept node init bằng
mean-pool embedding thành viên (docx §Bước1.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

from nn_aivie.core.adjacency import TypedAdjacency
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.vocab import Vocabulary, is_concept_token


def build_ppmi(
    adjacency: TypedAdjacency,
    vocab_size: int,
    *,
    smoothing: float = 0.75,
) -> csr_matrix:
    """
    Ma trận PPMI **thưa** `[N×N]` từ counts lớp CO_ACTIVATION (scale tới vocab 10k–50k).

    `smoothing` (context distribution smoothing, Levy et al. 2015): nâng P(j) lên luỹ thừa
    <1 để giảm thiên lệch về node tần suất cao. PPMI[i,j] = max(0, log P(i,j)/(P(i)·P_smooth(j))).
    Chỉ tính trên các ô khác 0 → bộ nhớ & thời gian ∝ số cạnh, không phải N².
    """
    co = adjacency.layer(EdgeKind.CO_ACTIVATION)
    rows, cols, vals = [], [], []
    for i, j, w in co.items():
        if i < vocab_size and j < vocab_size and w > 0:
            rows.append(i)
            cols.append(j)
            vals.append(w)
    if not vals:
        return csr_matrix((vocab_size, vocab_size), dtype=np.float64)

    c = coo_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(vocab_size, vocab_size),
    ).tocsr()
    total = float(c.sum())
    if total <= 0:
        return csr_matrix((vocab_size, vocab_size), dtype=np.float64)

    row_sum = np.asarray(c.sum(axis=1)).ravel()  # P(i)
    col_sum = np.asarray(c.sum(axis=0)).ravel()  # context marginal
    col_smooth = np.power(col_sum, smoothing)
    col_smooth_total = col_smooth.sum() or 1.0
    p_i = row_sum / total
    p_j = col_smooth / col_smooth_total

    coo = c.tocoo()
    denom = p_i[coo.row] * p_j[coo.col]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((coo.data / total) / denom)
    mask = np.isfinite(pmi) & (denom > 0) & (pmi > 0)
    return csr_matrix(
        (pmi[mask], (coo.row[mask], coo.col[mask])),
        shape=(vocab_size, vocab_size),
    )


def truncated_svd(ppmi, d: int) -> np.ndarray:
    """
    E = U_d · sqrt(Σ_d) từ SVD rút gọn của PPMI (embedding chuẩn PPMI+SVD).

    Ma trận lớn → `scipy.sparse.linalg.svds` (chỉ d trị kỳ dị lớn nhất); ma trận nhỏ → SVD dày
    numpy cho ổn định.
    """
    if not (hasattr(ppmi, "shape")):
        ppmi = np.asarray(ppmi)
    n = ppmi.shape[0]
    if n == 0:
        return np.zeros((0, d), dtype=np.float64)
    k = min(d, n - 1)
    use_dense = n <= 64 or k < 1 or k >= min(ppmi.shape)
    if use_dense:
        dense = ppmi.toarray() if hasattr(ppmi, "toarray") else np.asarray(ppmi)
        u, s, _ = np.linalg.svd(dense, full_matrices=False)
        m = min(d, len(s))
        e = u[:, :m] * np.sqrt(np.maximum(s[:m], 0.0))
    else:
        mat = ppmi.asfptype() if hasattr(ppmi, "asfptype") else ppmi
        u, s, _ = svds(mat, k=k, which="LM")
        order = np.argsort(-s)  # svds trả tăng dần → đảo về giảm dần
        e = u[:, order] * np.sqrt(np.maximum(s[order], 0.0))
    if e.shape[1] < d:  # pad giữ chiều cố định
        e = np.pad(e, ((0, 0), (0, d - e.shape[1])))
    return e


def concept_meanpool(
    e: np.ndarray,
    adjacency: TypedAdjacency,
    vocab: Vocabulary,
) -> np.ndarray:
    """
    Init/refine embedding Concept node = mean-pool embedding thành viên surface (docx §Bước1.2).

    Thành viên = node surface trỏ vào concept qua CO_ACTIVATION (bottom-up). Concept nào không
    có thành viên thì giữ nguyên vector PPMI+SVD.
    """
    co = adjacency.layer(EdgeKind.CO_ACTIVATION)
    out = e.copy()
    n = e.shape[0]
    # MỘT PASS gom member theo concept (trước: O(#concept × #cạnh) quét lại toàn bộ cạnh mỗi
    # concept-node → chết khi vocab lớn; giờ O(#cạnh)).
    from collections import defaultdict

    members_by_concept: dict[int, list[int]] = defaultdict(list)
    for i, j, w in co.items():
        if w > 0 and i < n and j < n and is_concept_token(vocab.token_of(j)) \
                and vocab.is_surface(i):
            members_by_concept[j].append(i)
    for cid, members in members_by_concept.items():
        out[cid] = e[members].mean(axis=0)
    return out


@dataclass
class GraphEmbedding:
    """E `[N×d]` + tra cứu cosine. Là view dẫn xuất của A — tính lại offline mỗi epoch."""

    matrix: np.ndarray  # (N, d)

    @property
    def dim(self) -> int:
        return self.matrix.shape[1] if self.matrix.size else 0

    def vector(self, node_id: int) -> np.ndarray:
        if 0 <= node_id < self.matrix.shape[0]:
            return self.matrix[node_id]
        return np.zeros(self.dim, dtype=np.float64)

    def _norms(self) -> np.ndarray:
        if not hasattr(self, "_norm_cache") or self._norm_cache is None:
            self._norm_cache = np.linalg.norm(self.matrix, axis=1)
        return self._norm_cache

    def similarity(self, i: int, j: int) -> float:
        vi, vj = self.vector(i), self.vector(j)
        ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
        if ni == 0 or nj == 0:
            return 0.0
        return float(np.dot(vi, vj) / (ni * nj))

    def nearest(
        self,
        node_id: int,
        topk: int = 5,
        *,
        candidate_ids: list[int] | None = None,
        min_sim: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Top-k node cosine gần nhất (loại chính nó). Lọc theo `candidate_ids` nếu có."""
        v = self.vector(node_id)
        nv = np.linalg.norm(v)
        if nv == 0:
            return []
        norms = self._norms()
        sims = self.matrix @ v
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = np.where(norms > 0, sims / (norms * nv), 0.0)
        if candidate_ids is not None:
            mask = np.full(self.matrix.shape[0], -np.inf)
            valid = [c for c in candidate_ids if 0 <= c < self.matrix.shape[0]]
            mask[valid] = sims[valid]
            sims = mask
        order = np.argsort(-sims)
        out: list[tuple[int, float]] = []
        for idx in order:
            if idx == node_id:
                continue
            s = float(sims[idx])
            if not np.isfinite(s) or s < min_sim:
                break
            out.append((int(idx), s))
            if len(out) >= topk:
                break
        return out

    # --- serialization ---
    def to_payload(self) -> dict:
        return {"matrix": self.matrix.astype(np.float32).tolist()}

    @classmethod
    def from_payload(cls, payload: dict) -> "GraphEmbedding":
        return cls(matrix=np.asarray(payload["matrix"], dtype=np.float64))


def build_embedding(
    adjacency: TypedAdjacency,
    vocab: Vocabulary,
    *,
    dim: int = 96,
    smoothing: float = 0.75,
    meanpool_concepts: bool = True,
) -> GraphEmbedding:
    """Pipeline đầy đủ: PPMI → SVD → (tuỳ chọn) mean-pool concept → GraphEmbedding."""
    n = len(vocab)
    ppmi = build_ppmi(adjacency, n, smoothing=smoothing)
    e = truncated_svd(ppmi, dim)
    if meanpool_concepts:
        e = concept_meanpool(e, adjacency, vocab)
    return GraphEmbedding(matrix=e)
