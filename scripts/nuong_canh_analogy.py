"""§158 — NƯỚNG CẠNH LOẠI SUY c→Y VÀO BẢNG KỀ (thụ động, tồn tại lúc train).

Yêu cầu tác giả: «tôi quan tâm việc đã tạo THỤ ĐỘNG kết nối giữa c và Y chưa.
Hiện khi sinh cạnh c→Y còn YẾU nên có thể không tham gia sinh, nhưng tôi muốn
PHẢI CÓ CẠNH Ở ĐÓ, sau này sẽ có thuật toán khác củng cố».

Ba dữ kiện đã đo trong TK_053:
  · §143: `build_w_skip` (đường sinh) đọc `ps.parents`, KHÔNG đọc `TypedAdjacency`
    ⇒ cạnh thêm vào bảng kề TỒN TẠI mà KHÔNG nhiễu sinh — đúng ý «còn yếu».
  · §149: bước TƯƠNG TỰ `S` hoạt động (`sắt → mangan/nhôm/oxide`); bước thừa
    hưởng `E = S·W` cho c→Y.
  · `EdgeKind.ANALOGY` priority 0.0 ⇒ hiện KHÔNG cộng vào readout sinh.

Cơ chế (đặc tả tác giả `X…a,X…b,X…c + a…Y,b…Y ⟹ c→Y`):
  S = Bn·Bnᵀ   (c ~ a: cùng nối tới nhiều ngữ cảnh X; cosine, cắt cột hub)
  E = S·W      (c thừa hưởng cạnh của anh em ⇒ c→Y)
  E -= E∩W     (chỉ giữ cạnh MỚI — đó là ngoại suy)
  top-k mỗi hàng, trọng số THẤP ⇒ nạp vào adjacency kind=ANALOGY.

    PYTHONPATH=src python scripts/nuong_canh_analogy.py CKPT --ra CKPT_MOI
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint, save_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402
from nn_aivie.hca.w_skip import build_w_skip  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

T0 = time.time()


def _topk_row(M: sp.csr_matrix, k: int) -> sp.csr_matrix:
    M = M.tocsr()
    r: list[int] = []
    c: list[int] = []
    d: list[float] = []
    for i in range(M.shape[0]):
        s, e = M.indptr[i], M.indptr[i + 1]
        if e <= s:
            continue
        dd = M.data[s:e]
        cc = M.indices[s:e]
        idx = np.argsort(-dd)[:k]
        r += [i] * len(idx)
        c += list(cc[idx])
        d += list(dd[idx])
    return sp.coo_matrix((d, (r, c)), shape=M.shape).tocsr()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--sim-topk", type=int, default=10, help="số anh em mỗi c")
    ap.add_argument("--edge-topk", type=int, default=15, help="số đích c→Y mỗi c")
    ap.add_argument("--weight", type=float, default=0.05,
                    help="trọng số cạnh (THẤP: tồn tại, chưa tham gia)")
    a = ap.parse_args()

    state, _ = load_checkpoint(a.ckpt)
    v = state.vocab
    ws = build_w_skip(state, d_max=6, min_count=2)
    W = ws.mats[1]
    for d in range(2, 5):
        W = W + ws.mats[d]
    W = W.tocsr()
    W.eliminate_zeros()
    print(f"[{time.time()-T0:.0f}s] W (d=1..4): nnz={W.nnz:,}", flush=True)

    # ── S = tương tự qua ngữ cảnh chung (cắt CỘT hub, cosine)
    indeg = np.asarray((W > 0).sum(0)).ravel()
    q = np.quantile(indeg[indeg > 0], 0.90)
    B = (W @ sp.diags((indeg <= q).astype(np.float64))).tocsr()
    B.eliminate_zeros()
    B.data[:] = 1.0
    nrm = np.sqrt(np.asarray(B.sum(1)).ravel())
    nrm[nrm == 0] = 1.0
    Bn = (sp.diags(1.0 / nrm) @ B).tocsr()
    print(f"[{time.time()-T0:.0f}s] bat dau S = Bn@BnT", flush=True)
    S = (Bn @ Bn.T).tocsr()
    S.setdiag(0)
    # §158b — CẮT NGƯỠNG trước vòng Python top-k. S thô ~400M phần tử ⇒ vòng
    # `_topk_row` thuần Python treo. Cosine ≥ 0,05 giữ lại phần tương tự thật
    # (ca tốt 0,196–0,321, §149) và bỏ đuôi trùng-hợp, thu S xuống cỡ nạp nổi.
    S.data[S.data < 0.05] = 0.0
    S.eliminate_zeros()
    print(f"[{time.time()-T0:.0f}s] S sau cắt ngưỡng 0,05: nnz={S.nnz:,}", flush=True)
    Sk = _topk_row(S, a.sim_topk)
    print(f"[{time.time()-T0:.0f}s] S top-{a.sim_topk}: nnz={Sk.nnz:,}", flush=True)

    # ── E = S·W (thừa hưởng), bỏ cạnh đã có, chuẩn hoá đích, top-k
    E = (Sk @ W).tocsr()
    E = E - E.multiply(W > 0)
    E.eliminate_zeros()
    E = (E @ sp.diags(1.0 / np.maximum(1.0, indeg.astype(np.float64)))).tocsr()
    Ek = _topk_row(E, a.edge_topk)
    print(f"[{time.time()-T0:.0f}s] E (c→Y mới) top-{a.edge_topk}: nnz={Ek.nnz:,}", flush=True)

    # ── nạp vào adjacency kind=ANALOGY, trọng số THẤP
    Ek = Ek.tocoo()
    deltas = {(int(i), int(j)): a.weight
              for i, j in zip(Ek.row, Ek.col)}
    state.adjacency.add_many(deltas, EdgeKind.ANALOGY)
    print(f"[{time.time()-T0:.0f}s] nạp {len(deltas):,} cạnh ANALOGY (w={a.weight})", flush=True)

    # §158c — KHÔNG bake_derived: chỉ thêm cạnh ANALOGY vào adjacency, còn
    # W_skip/seq index đọc từ ps.parents (§143) nên KHÔNG phụ thuộc cạnh mới.
    # bake_derived mine lại toàn bộ ⇒ 28 phút chưa xong, thừa. Lưu thẳng.
    save_checkpoint(state, a.ra)
    print(f"### lưu → {a.ra}")

    # ── XÁC MINH cạnh TỒN TẠI
    print("\n### xác minh c→Y có trong đồ thị:")
    for w in ("sắt", "kẽm", "đồng", "chó", "lúa"):
        ids = H.resolve_surface_ids(v, w)
        if not ids:
            continue
        i = int(ids[0])
        got = [(H._surf(v, int(j)).strip(),
                round(state.adjacency.get(i, int(j), EdgeKind.ANALOGY), 3))
               for j in Ek.col[Ek.row == i][:6]]
        print(f"   «{w}» → {got}")
    print(f"[{time.time()-T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
