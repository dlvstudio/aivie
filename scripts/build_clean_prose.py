"""§TK_067/069 — MẶT NẠ SẠCH-PROSE: lọc truyện tu tiên/kiếm hiệp (~40% corpus,
nguồn trôi-nghĩa), giữ PROSE FACTUAL + từ-điển.

Phương pháp KẾT HỢP (đo TK_067 — keyword-đơn hoặc token-set-đơn đều dở):
  1) KEYWORD chắc chắn: đại từ truyện (hắn/nàng/ngươi/gã/chàng/thiếp, token đơn)
     + cụm tu-tiên n-gram (khớp chính xác, không dính component word thường).
  2) EMBEDDING score: centroid câu-truyện (mầm keyword) vs centroid từ-điển
     (factual) → mỗi câu chấm `cos(s,novel)-cos(s,fact)`; >THR = nghiêng truyện.
     Bắt được truyện KHÔNG-từ-khoá (tên riêng hiện đại).
  3) LUÔN GIỮ từ-điển (Từ:/Giải nghĩa:/Ví dụ:) — factual chắc chắn.
Loại nếu (1) HOẶC (2 & không phải từ-điển). Giữ ~50% = factual tinh khiết cao.

    PYTHONPATH=src python scripts/build_clean_prose.py [--thr 0.06]
"""
import sys, time, argparse, re
sys.path.insert(0, "src")
import gc; gc.disable()
import numpy as np
from nn_aivie.core.checkpoint_io import load_checkpoint
T0 = time.time()

PRON_WORDS = ["hắn", "nàng", "ngươi", "gã", "chàng", "thiếp"]
COMPOUNDS = [
    "tu luyện", "tu tiên", "tu vi", "cảnh giới", "nguyên anh", "kim đan",
    "trúc cơ", "hóa thần", "độ kiếp", "nguyên thần", "linh thạch", "pháp bảo",
    "pháp khí", "linh khí", "yêu thú", "kiếm khí", "đan điền", "thần thức",
    "luyện khí", "trận pháp", "đạo hữu", "sư huynh", "sư đệ", "chưởng môn",
    "tông môn", "yêu tộc", "ma tộc", "kết đan", "phi thăng", "phi kiếm",
]
DICT = re.compile(r"^\s*(Từ|Giải nghĩa|Ví dụ|Đồng nghĩa|Trái nghĩa)\s*:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.06, help="ngưỡng embedding-score (cao=loại ít hơn)")
    ap.add_argument("--ckpt", default="checkpoints/net_ls_concept_conf.pkl")
    ap.add_argument("--out", default="checkpoints/clean_mask_prose.npy")
    a = ap.parse_args()

    st, _ = load_checkpoint(a.ckpt)
    v = st.vocab; ps = st.parents
    E = st.embedding.matrix.astype(np.float32); V = E.shape[0]; N = len(ps.parents)
    print(f"[{time.time()-T0:.0f}s] {N} câu, V={V}", flush=True)

    PRON = set()
    for w in PRON_WORDS:
        ids = v.encode_prompt(w)
        if len(ids) == 1:
            PRON.add(v.fold_group_id(int(ids[0])))
    ng = [tuple(v.fold_group_id(int(t)) for t in v.encode_prompt(c)) for c in COMPOUNDS]

    def kw_novel(emit):
        fg = [v.fold_group_id(int(t)) for t in emit]
        if set(fg) & PRON:
            return True
        for mi in ng:
            L = len(mi)
            for i in range(len(fg) - L + 1):
                if tuple(fg[i:i + L]) == mi:
                    return True
        return False

    def semb(emit):
        e = [int(t) for t in emit if 0 <= int(t) < V]
        if not e:
            return None
        x = E[e].mean(0); n = np.linalg.norm(x)
        return x / n if n > 0 else None

    # PASS 1: centroid mầm
    S = np.zeros((N, E.shape[1]), np.float32); ok = np.zeros(N, bool)
    kwf = np.zeros(N, bool); isdict = np.zeros(N, bool)
    novel_c = np.zeros(E.shape[1]); fact_c = np.zeros(E.shape[1]); nn = nf = 0
    for i, rec in enumerate(ps.parents):
        s = semb(rec.emit_ids)
        if s is None:
            continue
        S[i] = s; ok[i] = True
        kwf[i] = kw_novel(rec.emit_ids)
        txt = v.detokenize([int(t) for t in rec.emit_ids]).strip()
        isdict[i] = bool(DICT.match(txt))
        if kwf[i]:
            novel_c += s; nn += 1
        elif isdict[i]:
            fact_c += s; nf += 1
        if i % 100000 == 0:
            print(f"[{time.time()-T0:.0f}s]  pass1 {i}/{N}", flush=True)
    novel_c /= max(nn, 1); fact_c /= max(nf, 1)
    nc = novel_c / (np.linalg.norm(novel_c) + 1e-9)
    fc = fact_c / (np.linalg.norm(fact_c) + 1e-9)
    score = S @ nc - S @ fc
    print(f"[{time.time()-T0:.0f}s] mầm novel={nn} fact={nf}", flush=True)

    # PASS 2: quyết định
    clean = np.ones(N, bool); nk = 0
    for i in range(N):
        if not ok[i]:
            continue
        novel = kwf[i] or (score[i] > a.thr and not isdict[i])
        if novel:
            clean[i] = False; nk += 1
    np.save(a.out, clean)
    print(f"[{time.time()-T0:.0f}s] LƯU {a.out} — giữ {int(clean.sum())}/{N} "
          f"({100*nk/N:.1f}% loại truyện), thr={a.thr}", flush=True)
    print("### hết", flush=True)


if __name__ == "__main__":
    main()
