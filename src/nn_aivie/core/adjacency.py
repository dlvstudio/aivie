"""Sparse adjacency — flat and typed multi-layer."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import numpy as np

from nn_aivie.core.edge_kinds import (
    ALL_KINDS,
    ACTIVATION_KINDS,
    EdgeKind,
    READOUT_CO_SCALE,
    READOUT_EM_SCALE,
    READOUT_RF_SCALE,
    STRUCTURAL_KINDS,
)


class SparseAdjacency:
    def __init__(self) -> None:
        self._data: dict[tuple[int, int], float] = defaultdict(float)
        self._rows: dict[int, dict[int, float]] = defaultdict(dict)

    # --- TK_040 M1: CSR read-only (freeze sau ingest → bỏ _data+_rows, hết lưu-2-lần) ---
    def freeze(self) -> None:
        """Nén sang CSR (row_ids/indptr/indices/vals) read-only; bỏ _data+_rows.
        Chỉ gọi ở pha ĐỌC (sau train/load). Mutate (set/add) sẽ tự thaw."""
        import numpy as np
        if getattr(self, "_frozen", False):
            return
        items = sorted(self._data.items())  # ((i,j),v) theo (i,j)
        n = len(items)
        # int32 an toàn: node-id (token < ~1e5, cluster < CLUSTER_OFFSET+len ~2e7) < 2^31.
        # vals GIỮ float64 → byte-identical (không mất precision).
        indices = np.empty(n, dtype=np.int32)
        vals = np.empty(n, dtype=np.float64)
        row_ids: list[int] = []
        indptr: list[int] = [0]
        for k, ((i, j), v) in enumerate(items):
            if not row_ids or row_ids[-1] != i:
                if row_ids:
                    indptr.append(k)
                row_ids.append(i)
            indices[k] = j
            vals[k] = v
        indptr.append(n)
        self._row_ids = np.asarray(row_ids, dtype=np.int32)
        self._indptr = np.asarray(indptr, dtype=np.int32)
        self._indices = indices
        self._vals = vals
        self._frozen = True
        self._data = {}
        self._rows = {}

    def _thaw(self) -> None:
        if not getattr(self, "_frozen", False):
            return
        self._data = defaultdict(float)
        self._rows = defaultdict(dict)
        for i, j, v in self._csr_items():
            self._data[(i, j)] = v
            self._rows[i][j] = v
        self._frozen = False
        self._row_ids = self._indptr = self._indices = self._vals = None

    def _row_span(self, i: int):
        import numpy as np
        r = int(np.searchsorted(self._row_ids, i))
        if r >= len(self._row_ids) or int(self._row_ids[r]) != i:
            return None
        return int(self._indptr[r]), int(self._indptr[r + 1])

    def _csr_items(self):
        for r in range(len(self._row_ids)):
            i = int(self._row_ids[r])
            for k in range(int(self._indptr[r]), int(self._indptr[r + 1])):
                yield i, int(self._indices[k]), float(self._vals[k])

    def get(self, i: int, j: int) -> float:
        if getattr(self, "_frozen", False):
            import numpy as np
            span = self._row_span(i)
            if span is None:
                return 0.0
            lo, hi = span
            pos = int(np.searchsorted(self._indices[lo:hi], j))
            if pos < hi - lo and int(self._indices[lo + pos]) == j:
                return float(self._vals[lo + pos])
            return 0.0
        return self._data.get((i, j), 0.0)

    def set(self, i: int, j: int, value: float) -> None:
        if getattr(self, "_frozen", False):
            self._thaw()
        if value == 0.0:
            self._data.pop((i, j), None)
            if i in self._rows:
                self._rows[i].pop(j, None)
                if not self._rows[i]:
                    self._rows.pop(i, None)
        else:
            self._data[(i, j)] = value
            self._rows[i][j] = value

    def add(self, i: int, j: int, delta: float) -> None:
        if delta == 0.0:
            return
        if getattr(self, "_frozen", False):
            self._thaw()
        key = (i, j)
        new_val = self._data.get(key, 0.0) + delta
        if new_val == 0.0:
            self._data.pop(key, None)
            if i in self._rows:
                self._rows[i].pop(j, None)
                if not self._rows[i]:
                    self._rows.pop(i, None)
        else:
            self._data[key] = new_val
            self._rows[i][j] = new_val

    def add_delta(self, i: int, j: int, delta: float) -> None:
        self.add(i, j, delta)

    def row(self, i: int) -> dict[int, float]:
        if getattr(self, "_frozen", False):
            span = self._row_span(i)
            if span is None:
                return {}
            lo, hi = span
            return {
                int(self._indices[k]): float(self._vals[k]) for k in range(lo, hi)
            }
        if not hasattr(self, "_rows"):
            self._rows = defaultdict(dict)
            for (ri, rj), rv in self._data.items():
                self._rows[ri][rj] = rv
        return self._rows.get(i, {})

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_rows" not in self.__dict__:
            self._rows = defaultdict(dict)
            for (ri, rj), rv in self._data.items():
                self._rows[ri][rj] = rv

    def items(self) -> Iterator[tuple[int, int, float]]:
        if getattr(self, "_frozen", False):
            yield from self._csr_items()
            return
        for (i, j), v in self._data.items():
            yield i, j, v

    def to_dict(self) -> dict[str, float]:
        return {f"{i},{j}": v for (i, j), v in self._data.items()}

    def to_arrays(self) -> dict:
        """Serialize dạng 3 mảng numpy (i, j, w) — pickle nhị phân, load nhanh.

        Thay `{"i,j": w}` (key chuỗi): tránh 3.6M `str.split` + pickle key chuỗi lúc load
        (TK_011 — load checkpoint 44k cha ~11.7s → vài giây). `from_dict` tự nhận cả 2 dạng."""
        import numpy as np

        if getattr(self, "_frozen", False):
            # CSR: dựng i (broadcast từ row_ids theo indptr) + j + w từ mảng có sẵn
            n = len(self._indices)
            i = np.empty(n, dtype=np.int32)
            for r in range(len(self._row_ids)):
                i[int(self._indptr[r]):int(self._indptr[r + 1])] = int(self._row_ids[r])
            return {
                "i": i,
                "j": self._indices.astype(np.int32),
                "w": self._vals.astype(np.float64),
            }
        n = len(self._data)
        i = np.empty(n, dtype=np.int32)
        j = np.empty(n, dtype=np.int32)
        w = np.empty(n, dtype=np.float64)
        for k, ((a, b), v) in enumerate(self._data.items()):
            i[k] = a
            j[k] = b
            w[k] = v
        return {"i": i, "j": j, "w": w}

    @staticmethod
    def _fill(layer: "SparseAdjacency", data) -> None:
        """Nạp layer_data (dạng mảng MỚI hoặc key-chuỗi CŨ) vào _data + _rows trực tiếp."""
        _data = layer._data
        _rows = layer._rows
        if isinstance(data, dict) and "i" in data and "j" in data:
            ii = data["i"]
            jj = data["j"]
            ww = data["w"]
            ii = ii.tolist() if hasattr(ii, "tolist") else list(ii)
            jj = jj.tolist() if hasattr(jj, "tolist") else list(jj)
            ww = ww.tolist() if hasattr(ww, "tolist") else list(ww)
            for a, b, v in zip(ii, jj, ww):
                if v == 0.0:
                    continue
                _data[(a, b)] = v
                _rows[a][b] = v
        else:  # dạng cũ: {"i,j": w}
            for key, v in data.items():
                a, b = map(int, key.split(","))
                v = float(v)
                if v == 0.0:
                    continue
                _data[(a, b)] = v
                _rows[a][b] = v

    @staticmethod
    def _fill_csr(layer: "SparseAdjacency", data) -> None:
        """TK_040 M3: nạp THẲNG vào CSR frozen từ mảng {i,j,w} (không dựng _data/_rows dict
        → hạ peak RAM lúc load). Chỉ nhận dạng mảng. Kết quả = như freeze() (cùng thứ tự sort)."""
        import numpy as np

        i = np.asarray(data["i"])
        j = np.asarray(data["j"])
        w = np.asarray(data["w"], dtype=np.float64)
        mask = w != 0.0
        if not mask.all():
            i = i[mask]; j = j[mask]; w = w[mask]
        n = len(i)
        if n == 0:
            layer._row_ids = np.empty(0, dtype=np.int32)
            layer._indptr = np.zeros(1, dtype=np.int32)
            layer._indices = np.empty(0, dtype=np.int32)
            layer._vals = np.empty(0, dtype=np.float64)
        else:
            order = np.lexsort((j, i))  # sort theo i rồi j (= sorted((i,j)))
            i = i[order].astype(np.int64)
            layer._indices = j[order].astype(np.int32)
            layer._vals = w[order]
            row_ids, first = np.unique(i, return_index=True)  # i đã sort → first tăng dần
            layer._row_ids = row_ids.astype(np.int32)
            layer._indptr = np.append(first, n).astype(np.int32)
        layer._frozen = True
        layer._data = {}
        layer._rows = {}

    @classmethod
    def from_dict(cls, data, *, csr: bool = False) -> SparseAdjacency:
        adj = cls()
        if csr and isinstance(data, dict) and "i" in data and "j" in data:
            cls._fill_csr(adj, data)
        else:
            cls._fill(adj, data)
        return adj

    def __len__(self) -> int:
        if getattr(self, "_frozen", False):
            return len(self._indices)
        return len(self._data)


class TypedAdjacency:
    """Multi-layer adjacency: one sparse matrix per EdgeKind."""

    def __init__(self) -> None:
        self._layers: dict[EdgeKind, SparseAdjacency] = {
            kind: SparseAdjacency() for kind in ALL_KINDS
        }
        # Teach collocation S↔A scoped by Rule node (only used when that rule is active).
        self._rule_collocation: dict[int, SparseAdjacency] = {}

    def add_rule_collocation(self, rule_id: int, i: int, j: int, delta: float) -> None:
        if delta == 0.0:
            return
        if rule_id not in self._rule_collocation:
            self._rule_collocation[rule_id] = SparseAdjacency()
        self._rule_collocation[rule_id].add(i, j, delta)

    def get_rule_collocation(self, rule_id: int, i: int, j: int) -> float:
        layer = self._rule_collocation.get(rule_id)
        return layer.get(i, j) if layer else 0.0

    def row_rule_collocation(self, rule_id: int, i: int) -> dict[int, float]:
        layer = self._rule_collocation.get(rule_id)
        return layer.row(i) if layer else {}

    def rule_collocation_items(self) -> Iterator[tuple[int, int, int, float]]:
        for rule_id, layer in self._rule_collocation.items():
            for i, j, w in layer.items():
                yield rule_id, i, j, w

    def rule_collocation_rule_ids(self) -> list[int]:
        return list(self._rule_collocation.keys())

    def layer(self, kind: EdgeKind) -> SparseAdjacency:
        return self._layers[kind]

    def freeze(self) -> None:
        """TK_040 M1: nén mọi layer sang CSR read-only (gọi sau train/load, pha ĐỌC)."""
        for layer in self._layers.values():
            layer.freeze()
        for layer in self._rule_collocation.values():
            layer.freeze()

    def _ro_invalidate(self, i: int) -> None:
        """Xoá cache readout/learned của HÀNG i (đổi khi cạnh i→· đổi)."""
        c = getattr(self, "_ro_cache", None)
        if c is not None:
            c.pop(i, None)
        lc = getattr(self, "_lr_cache", None)
        if lc is not None:
            lc.pop(i, None)
        ric = getattr(self, "_ro_ids_cache", None)
        if ric is not None:
            ric.pop(i, None)
        lic = getattr(self, "_lr_ids_cache", None)
        if lic is not None:
            lic.pop(i, None)
        self._ro_csr_norm = None
        self._ro_csr_vocab = 0

    def readout_csr_normalized(self, vocab_size: int):
        """Ma trận readout hàng-chuẩn-hóa (R[i,j] = readout(i,j)/max_j readout(i,j)).

        Dùng cho spreading matvec: nxt = gamma * R_norm.T @ frontier. Cache lazy; invalidate
        khi ghi cạnh. Không pickle (transient, dựng lại sau load)."""
        from collections import defaultdict

        from scipy.sparse import coo_matrix, csr_matrix

        cached = getattr(self, "_ro_csr_norm", None)
        if cached is not None and getattr(self, "_ro_csr_vocab", 0) == vocab_size:
            return cached

        # TK_011 D1: khi mọi readout-layer đã frozen (CSR) → build THẲNG từ mảng
        # (vectorized scipy) thay vì Python-loop qua items() (O(cạnh) → chậm graph lớn).
        ro_kinds = [(k, 1.0) for k in STRUCTURAL_KINDS] + [
            (EdgeKind.CO_ACTIVATION, READOUT_CO_SCALE),
            (EdgeKind.REINFORCED, READOUT_RF_SCALE),
            (EdgeKind.EMERGENT, READOUT_EM_SCALE),
        ]
        if all(getattr(self._layers[k], "_frozen", False) for k, _ in ro_kinds):
            mat = self._readout_norm_from_csr(ro_kinds, vocab_size)
            self._ro_csr_norm = mat
            self._ro_csr_vocab = vocab_size
            return mat

        row_data: dict[int, dict[int, float]] = defaultdict(dict)
        for k in STRUCTURAL_KINDS:
            for i, j, w in self._layers[k].items():
                if w != 0.0:
                    row_data[i][j] = row_data[i].get(j, 0.0) + w
        for i, j, w in self._layers[EdgeKind.CO_ACTIVATION].items():
            if w != 0.0:
                row_data[i][j] = row_data[i].get(j, 0.0) + w * READOUT_CO_SCALE
        for i, j, w in self._layers[EdgeKind.REINFORCED].items():
            if w != 0.0:
                row_data[i][j] = row_data[i].get(j, 0.0) + w * READOUT_RF_SCALE
        for i, j, w in self._layers[EdgeKind.EMERGENT].items():
            if w != 0.0:
                row_data[i][j] = row_data[i].get(j, 0.0) + w * READOUT_EM_SCALE

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i, jd in row_data.items():
            if i >= vocab_size:
                continue
            rmax = max(jd.values()) or 1.0
            for j, w in jd.items():
                if j < vocab_size and w > 0.0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(w / rmax)

        if not vals:
            mat = csr_matrix((vocab_size, vocab_size), dtype=np.float64)
        else:
            mat = coo_matrix(
                (np.asarray(vals, dtype=np.float64), (rows, cols)),
                shape=(vocab_size, vocab_size),
            ).tocsr()

        self._ro_csr_norm = mat
        self._ro_csr_vocab = vocab_size
        return mat

    def _readout_norm_from_csr(self, ro_kinds, vocab_size):
        """TK_011 D1: readout-norm matrix build THẲNG từ mảng CSR frozen (vectorized).
        Khớp logic loop cũ: sum cạnh×scale trên (i,j); rmax = max CẢ hàng (gồm j≥vocab);
        output j<vocab, w>0, chia rmax. Float-order coo-sum ≈ (như CSR spreading)."""
        from scipy.sparse import coo_matrix, csr_matrix, diags

        ii, jj, ww = [], [], []
        for kind, scale in ro_kinds:
            L = self._layers[kind]
            if len(L._indices) == 0:
                continue
            i = np.repeat(L._row_ids, np.diff(L._indptr).astype(np.int64))
            ii.append(i)
            jj.append(L._indices.astype(np.int64))
            ww.append(L._vals * float(scale))
        if not ii:
            return csr_matrix((vocab_size, vocab_size), dtype=np.float64)
        I = np.concatenate(ii)
        J = np.concatenate(jj)
        W = np.concatenate(ww)
        m = (I < vocab_size) & (W != 0.0)
        I, J, W = I[m], J[m], W[m]
        if len(I) == 0:
            return csr_matrix((vocab_size, vocab_size), dtype=np.float64)
        ncol = max(int(J.max()) + 1, vocab_size)
        full = coo_matrix((W, (I, J)), shape=(vocab_size, ncol)).tocsr()  # sum duplicates
        rmax = np.asarray(full.max(axis=1).todense()).ravel()  # max CẢ hàng
        rmax[rmax <= 0.0] = 1.0
        norm = (diags(1.0 / rmax) @ full)[:, :vocab_size].tocsr()  # chia hàng + lọc cột
        if norm.nnz:
            norm.data[norm.data <= 0.0] = 0.0  # w>0 filter
            norm.eliminate_zeros()
        return norm

    def __getstate__(self):
        # KHÔNG pickle cache readout/learned/CSR (transient, dựng lại lazy sau load).
        state = self.__dict__.copy()
        state.pop("_ro_cache", None)
        state.pop("_lr_cache", None)
        state.pop("_ro_ids_cache", None)
        state.pop("_lr_ids_cache", None)
        state.pop("_ro_csr_norm", None)
        state.pop("_ro_csr_vocab", None)
        return state

    def add(
        self,
        i: int,
        j: int,
        delta: float,
        kind: EdgeKind = EdgeKind.GRAMMAR,
    ) -> None:
        if delta == 0.0:
            return
        self._layers[kind].add(i, j, delta)
        self._ro_invalidate(i)

    def add_many(
        self,
        deltas: dict[tuple[int, int], float] | Counter,
        kind: EdgeKind,
    ) -> None:
        """B1: ghi nhiều delta cùng layer — invalidate mỗi hàng nguồn một lần."""
        if not deltas:
            return
        layer = self._layers[kind]
        touched: set[int] = set()
        for (i, j), delta in deltas.items():
            if delta == 0.0:
                continue
            layer.add(i, j, delta)
            touched.add(i)
        for i in touched:
            self._ro_invalidate(i)

    def set(
        self,
        i: int,
        j: int,
        value: float,
        kind: EdgeKind = EdgeKind.GRAMMAR,
    ) -> None:
        self._layers[kind].set(i, j, value)
        self._ro_invalidate(i)

    def get(self, i: int, j: int, kind: EdgeKind | None = None) -> float:
        if kind is not None:
            return self._layers[kind].get(i, j)
        return sum(self._layers[k].get(i, j) for k in STRUCTURAL_KINDS)

    def get_all(self, i: int, j: int) -> float:
        return sum(layer.get(i, j) for layer in self._layers.values())

    def get_readout(self, i: int, j: int) -> float:
        """HCA readout: structural + scaled corpus co-activation + G3 + emergent."""
        co = self._layers[EdgeKind.CO_ACTIVATION].get(i, j)
        rf = self._layers[EdgeKind.REINFORCED].get(i, j)
        em = self._layers[EdgeKind.EMERGENT].get(i, j)
        return (
            self.get(i, j)
            + co * READOUT_CO_SCALE
            + rf * READOUT_RF_SCALE
            + em * READOUT_EM_SCALE
        )

    def row_readout(self, i: int) -> dict[int, float]:
        """HÀNG readout của i (structural + CO/RF/EM scaled), MEMO hoá.

        Trước đây merge 4 lớp thành dict mới MỖI lần gọi — spreading/dual-context gọi
        lặp cùng node hàng nghìn lần/token. Cache theo hàng, invalidate khi ghi cạnh i→·
        (mọi write qua `add`/`set`). Inference read-only → cache bền; train pop hàng đã sửa.
        TK_011 B2. KHÔNG mutate dict trả về (mọi caller hiện chỉ đọc)."""
        cache = getattr(self, "_ro_cache", None)
        if cache is None:
            cache = {}
            self._ro_cache = cache
        hit = cache.get(i)
        if hit is not None:
            return hit
        merged = dict(self.row(i))
        for j, w in self._layers[EdgeKind.CO_ACTIVATION].row(i).items():
            merged[j] = merged.get(j, 0.0) + w * READOUT_CO_SCALE
        for j, w in self._layers[EdgeKind.REINFORCED].row(i).items():
            merged[j] = merged.get(j, 0.0) + w * READOUT_RF_SCALE
        for j, w in self._layers[EdgeKind.EMERGENT].row(i).items():
            merged[j] = merged.get(j, 0.0) + w * READOUT_EM_SCALE
        cache[i] = merged
        return merged

    def learned_row(self, i: int) -> dict[int, float]:
        """Tổng RAW các lớp học (REINFORCED+EMERGENT+CO_ACTIVATION) từ node i.

        = `sum_k adjacency.get(i, ·, k)` cho k trong _LEARNED_KINDS, nhưng lấy MỘT
        lần cả hàng thay vì dò từng đích. Dùng cho G4 grounding vectorized (TK_011 B1):
        gom hàng của vài anchor thay vì quét toàn vocab × anchor × lớp.
        """
        cache = getattr(self, "_lr_cache", None)
        if cache is None:
            cache = {}
            self._lr_cache = cache
        hit = cache.get(i)
        if hit is not None:
            return hit
        merged: dict[int, float] = {}
        for k in (
            EdgeKind.REINFORCED,
            EdgeKind.EMERGENT,
            EdgeKind.CO_ACTIVATION,
        ):
            for j, w in self._layers[k].row(i).items():
                merged[j] = merged.get(j, 0.0) + w
        cache[i] = merged
        return merged

    def readout_row_ids(self, i: int, vocab_size: int):
        """TK_011 D3: neighbor-id <vocab của readout-row i, LỌC + CACHE dạng mảng.
        ar_candidates gom candidate mỗi token trên agenda-seed tái diễn → cache mảng
        (khỏi lặp genexpr+filter mỗi token). Invalidate cùng _ro_cache."""
        cache = getattr(self, "_ro_ids_cache", None)
        if cache is None:
            cache = {}
            self._ro_ids_cache = cache
        hit = cache.get(i)
        if hit is not None:
            return hit
        ids = np.fromiter(
            (v for v in self.row_readout(i) if v < vocab_size), dtype=np.int64
        )
        cache[i] = ids
        return ids

    def learned_row_ids(self, i: int, vocab_size: int):
        """TK_011 D3: neighbor-id <vocab của learned-row i, cache dạng mảng."""
        cache = getattr(self, "_lr_ids_cache", None)
        if cache is None:
            cache = {}
            self._lr_ids_cache = cache
        hit = cache.get(i)
        if hit is not None:
            return hit
        ids = np.fromiter(
            (v for v in self.learned_row(i) if v < vocab_size), dtype=np.int64
        )
        cache[i] = ids
        return ids

    def row_activation(self, i: int) -> dict[int, float]:
        merged: dict[int, float] = defaultdict(float)
        for k in ACTIVATION_KINDS:
            for j, w in self._layers[k].row(i).items():
                merged[j] += w
        return dict(merged)

    def get_weighted(
        self, i: int, j: int, kind: EdgeKind, *, scale_priority: bool = True
    ) -> float:
        w = self.get(i, j, kind)
        if not scale_priority or w == 0.0:
            return w
        from nn_aivie.core.edge_kinds import priority_weight

        return priority_weight(kind, w)

    def row(self, i: int, kind: EdgeKind | None = None) -> dict[int, float]:
        if kind is not None:
            return self._layers[kind].row(i)
        merged: dict[int, float] = defaultdict(float)
        for k in STRUCTURAL_KINDS:
            for j, w in self._layers[k].row(i).items():
                merged[j] += w
        return dict(merged)

    def items(
        self, kind: EdgeKind | None = None
    ) -> Iterator[tuple[int, int, float]]:
        if kind is not None:
            yield from self._layers[kind].items()
            return
        seen: set[tuple[int, int]] = set()
        for k in STRUCTURAL_KINDS:
            for i, j, w in self._layers[k].items():
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                yield i, j, self.get(i, j)

    def typed_items(self) -> Iterator[tuple[int, int, float, EdgeKind]]:
        for k in ALL_KINDS:
            for i, j, w in self._layers[k].items():
                yield i, j, w, k

    @property
    def _data(self) -> dict[tuple[int, int], float]:
        """Merged all layers (tests/debug)."""
        merged: dict[tuple[int, int], float] = defaultdict(float)
        for layer in self._layers.values():
            for key, v in layer._data.items():
                merged[key] += v
        return dict(merged)

    def to_dict(self) -> dict[str, dict[str, float] | float]:
        layers = {
            kind.value: layer.to_dict()
            for kind, layer in self._layers.items()
            if len(layer) > 0
        }
        rule_colloc = {
            str(rule_id): layer.to_dict()
            for rule_id, layer in self._rule_collocation.items()
            if len(layer) > 0
        }
        if rule_colloc:
            layers["rule_collocation"] = rule_colloc
        if layers:
            return layers
        return {}

    def to_arrays(self) -> dict:
        """Serialize numpy-array cho pickle (load nhanh). Cùng cấu trúc `to_dict` nhưng
        mỗi layer = {i,j,w} mảng thay {"i,j": w}. `from_dict` nhận cả hai (back-compat)."""
        layers = {
            kind.value: layer.to_arrays()
            for kind, layer in self._layers.items()
            if len(layer) > 0
        }
        rule_colloc = {
            str(rule_id): layer.to_arrays()
            for rule_id, layer in self._rule_collocation.items()
            if len(layer) > 0
        }
        if rule_colloc:
            layers["rule_collocation"] = rule_colloc
        return layers

    @classmethod
    def from_dict(cls, data: dict, *, csr: bool = False) -> TypedAdjacency:
        adj = cls()
        if not data:
            return adj
        sample_key = next(iter(data))
        if "," in str(sample_key):  # dạng flat GRAMMAR cũ nhất ({"i,j": w} ở gốc)
            SparseAdjacency._fill(
                adj._layers[EdgeKind.GRAMMAR],
                {str(k): float(v) for k, v in data.items()},
            )
            return adj

        def _load(layer, ldata):
            # TK_040: csr=True + dạng mảng → nạp thẳng CSR (low-peak). Ngược lại dict.
            if csr and isinstance(ldata, dict) and "i" in ldata and "j" in ldata:
                SparseAdjacency._fill_csr(layer, ldata)
            else:
                SparseAdjacency._fill(layer, ldata)

        kind_values = {k.value for k in EdgeKind}
        for kind_name, layer_data in data.items():
            if kind_name == "rule_collocation":
                if not isinstance(layer_data, dict):
                    continue
                for rule_key, edges in layer_data.items():
                    rule_id = int(rule_key)
                    layer = SparseAdjacency()
                    _load(layer, edges)
                    if len(layer) > 0:
                        adj._rule_collocation[rule_id] = layer
                continue
            if kind_name not in kind_values:
                continue
            _load(adj._layers[EdgeKind(kind_name)], layer_data)
        return adj

    def __len__(self) -> int:
        return sum(len(layer) for layer in self._layers.values())
