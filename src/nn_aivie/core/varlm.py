"""
TK_031 — Mô-hình-chuỗi BIẾN-BẬC (variable-order / PPM-style), thuần ĐẾM (không backprop).

Vì sao: W_seq là order-1 (nhìn 1 token trước) → mất trí nhớ ngữ cảnh sau mỗi từ → văn rời.
VarOrderLM nhớ "token kế theo CHUỖI-KHỚP-DÀI-NHẤT còn dữ liệu" (ctx-len 1..4), backoff mềm
kiểu PPM: điểm = trộn P_k(t|ctx_k) các bậc có dữ liệu, trọng số λ^(max_ctx−k) (bậc dài tin hơn).

Probe (out/tk031_ppm_probe*.log, held-out 3097 vị trí):
  bigram 4.369 bits / top1 16.4%  →  PPM 4.155 bits / top1 20.6%  (Δ−0.21 bits, +4pp — lớn nhất
  từng đo trên họ thước). λ vô cảm (0.3–0.75); PRUNE ctx singleton k≥3 MIỄN PHÍ (bits không đổi,
  bỏ ~58% context). Mật độ held-out: 76% vị trí có ctx≥2 trong corpus → trần data không chặn.

Luật 0 sạch: chỉ đếm đồng-hiện tuần-tự (Hebbian tầm xa hơn bigram), không rule/label runtime.
"""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path


class VarOrderLM:
    """N-gram biến-bậc trên token id. tables[k-1]: ctx-tuple(len k) → {next_id: count}."""

    def __init__(self, max_ctx: int = 4, lam: float = 0.45, *, use_clusters: bool = False):
        self.max_ctx = max_ctx
        self.lam = lam
        # If True, ctx_ids/ids may include SegmentGraph cluster node ids (>= CLUSTER_OFFSET).
        # This does NOT change decoding; it only makes VarLM context denser.
        self.use_clusters = use_clusters
        self.tables: list[dict[tuple[int, ...], dict[int, int]]] = [
            {} for _ in range(max_ctx)
        ]
        self.n_tokens = 0

    # ---- build ----
    def add_sequence(self, ids: list[int]) -> None:
        for i, t in enumerate(ids):
            self.n_tokens += 1
            for k in range(1, self.max_ctx + 1):
                if i - k < 0:
                    break
                ctx = tuple(ids[i - k : i])
                d = self.tables[k - 1].setdefault(ctx, {})
                d[t] = d.get(t, 0) + 1

    def prune(self, min_total: int = 2, from_ctx: int = 3) -> dict[str, int]:
        """Bỏ ctx tổng-count < min_total ở bậc ≥ from_ctx (probe: MIỄN PHÍ về bits)."""
        stats = {}
        for k in range(from_ctx, self.max_ctx + 1):
            tab = self.tables[k - 1]
            before = len(tab)
            self.tables[k - 1] = {
                c: v for c, v in tab.items() if sum(v.values()) >= min_total
            }
            stats[f"k{k}"] = before - len(self.tables[k - 1])
        return stats

    # ---- score ----
    def scores(self, ctx_ids) -> dict[int, float]:
        """Điểm token kế: trộn P_k các bậc có ctx, trọng số λ^(max_ctx−k). Rỗng nếu mù ctx."""
        out: dict[int, float] = defaultdict(float)
        wsum = 0.0
        n = len(ctx_ids)
        for k in range(min(self.max_ctx, n), 0, -1):
            c = self.tables[k - 1].get(tuple(ctx_ids[n - k : n]))
            if not c:
                continue
            w = self.lam ** (self.max_ctx - k)
            wsum += w
            tot = sum(c.values())
            for t, cnt in c.items():
                out[t] += w * (cnt / tot)
        if not out:
            return {}
        return {t: v / wsum for t, v in out.items()}

    # ---- Kneser-Ney (interpolated, absolute discount) ----
    # Mục đích: thay nội-suy λ^(max−k) cố định bằng KN chuẩn — bậc cao dùng count thật,
    # bậc thấp dùng CONTINUATION count (số ngữ cảnh-trái phân biệt mà một gram theo sau),
    # nền unigram = xác suất continuation. Thuần ĐẾM (Luật 0 sạch). Additive: không đụng
    # `scores()`; bật qua flag đo A/B (TK_031 bước 2).
    _kn_cont: list[dict] | None = None
    _kn_uni: dict[int, int] | None = None
    _kn_uni_total: int = 0

    def _ensure_kn(self) -> None:
        if self._kn_cont is not None:
            return
        # cont[k] (k=1..max_ctx-1): ctx(len k) → {next: số x phân biệt sao cho (x,)+ctx+(next,) xuất hiện}
        cont: list[dict] = [None] * self.max_ctx  # index k-1; chỉ dùng 1..max_ctx-1
        for k in range(1, self.max_ctx):
            higher = self.tables[k]  # contexts of length k+1: (x, c_1..c_k)
            ctab: dict = {}
            for hctx, nexts in higher.items():
                base_ctx = hctx[1:]  # bỏ token trái x → gram gốc len k
                d = ctab.setdefault(base_ctx, {})
                for w in nexts:
                    d[w] = d.get(w, 0) + 1  # mỗi hctx khác nhau ở x → +1 x phân biệt
            cont[k] = ctab
        # unigram continuation: n1_left[w] = #x phân biệt với bigram (x,w); tổng = #kiểu bigram
        uni: dict[int, int] = {}
        total = 0
        for _ctx, nexts in self.tables[0].items():
            for w in nexts:
                uni[w] = uni.get(w, 0) + 1
                total += 1
        self._kn_cont = cont
        self._kn_uni = uni
        self._kn_uni_total = total

    def scores_kn(self, ctx_ids, d: float = 0.75) -> dict[int, float]:
        """Interpolated Kneser-Ney next-token distribution trên tập ứng viên. Rỗng nếu mù ctx."""
        self._ensure_kn()
        n = len(ctx_ids)
        K = min(self.max_ctx, n)
        if K == 0:
            return {}
        order_tab: dict[int, dict | None] = {}
        cands: set[int] = set()
        for k in range(1, K + 1):
            ctx = tuple(ctx_ids[n - k : n])
            if k == K:
                tab = self.tables[k - 1].get(ctx)
            else:
                tab = self._kn_cont[k].get(ctx) if self._kn_cont[k] else None
            order_tab[k] = tab
            if tab:
                cands.update(tab.keys())
        if not cands:
            return {}
        # nền: xác suất continuation unigram (uniform nếu chưa có bằng chứng)
        tot = self._kn_uni_total or 1
        P: dict[int, float] = {
            w: self._kn_uni.get(w, 0) / tot for w in cands
        }
        # nội-suy từ bậc thấp lên bậc cao K
        for k in range(1, K + 1):
            tab = order_tab[k]
            if not tab:
                continue  # backoff toàn phần: P giữ nguyên (bậc dưới)
            denom = sum(tab.values())
            if denom <= 0:
                continue
            gamma = d * len(tab) / denom
            P = {
                w: max(tab.get(w, 0) - d, 0.0) / denom + gamma * P.get(w, 0.0)
                for w in cands
            }
        return P

    # ---- io ----
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "max_ctx": self.max_ctx,
                    "lam": self.lam,
                    "use_clusters": self.use_clusters,
                    "tables": self.tables,
                    "n_tokens": self.n_tokens,
                },
                f, protocol=4,
            )

    @classmethod
    def load(cls, path: str | Path) -> "VarOrderLM":
        with open(path, "rb") as f:
            d = pickle.load(f)
        lm = cls(max_ctx=d["max_ctx"], lam=d["lam"], use_clusters=bool(d.get("use_clusters", False)))
        lm.tables = d["tables"]
        lm.n_tokens = d.get("n_tokens", 0)
        return lm

    def stats(self) -> str:
        return ", ".join(f"k={k+1}:{len(t)}" for k, t in enumerate(self.tables))


def varlm_sidecar_path(checkpoint: str | Path) -> Path:
    p = Path(checkpoint)
    return p.with_name(p.stem + ".varlm.pkl")
