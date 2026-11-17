"""§189 — DỰNG PER-SENSE EMBEDDING (offline). Gom occurrence mỗi token content
đa nghĩa thành các NGHĨA bằng k-means trên vector-ngữ-cảnh, lưu (centroid, tập
cha) mỗi nghĩa. Token ĐƠN-NGHĨA (spread thấp) KHÔNG lưu → retrieval không lọc.

    PYTHONPATH=src python scripts/dung_sense_clusters.py CKPT --ra SIDECAR
"""
from __future__ import annotations
import argparse, gc, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src")); gc.disable()
from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.parent_store import _content_ids  # noqa: E402
T0 = time.time()

def kmeans(X, K, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()
    a = np.zeros(len(X), int)
    for _ in range(iters):
        a = (X @ C.T).argmax(1)
        for k in range(K):
            m = X[a == k]
            if len(m):
                c = m.mean(0); n = np.linalg.norm(c)
                C[k] = c / n if n > 0 else C[k]
    return C, a

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--ra", required=True)
    ap.add_argument("--df-min", type=int, default=40)
    ap.add_argument("--df-max", type=int, default=2500)
    ap.add_argument("--max-occ", type=int, default=300)
    ap.add_argument("--spread", type=float, default=0.55,
                    help="min cos GIỮA centroid < ngưỡng ⇒ đa nghĩa (khác hướng)")
    a = ap.parse_args()
    st, _ = load_checkpoint(a.ckpt); v = st.vocab; ps = st.parents; E = st.embedding.matrix
    print(f"[{time.time()-T0:.0f}s] load; E {E.shape}", flush=True)
    # precompute ctx-vec mỗi cha (mean embedding content-token của cha)
    N = len(ps.parents)
    pv = np.zeros((N, E.shape[1]), np.float32)
    for i, rec in enumerate(ps.parents):
        cs = [t for t in _content_ids(list(rec.emit_ids), v, skip=set()) if 0 <= t < E.shape[0]]
        if cs:
            m = E[np.asarray(cs)].mean(0); n = np.linalg.norm(m)
            if n > 0: pv[i] = m / n
        if i % 100000 == 0: print(f"[{time.time()-T0:.0f}s]  pv {i}/{N}", flush=True)
    print(f"[{time.time()-T0:.0f}s] pv xong", flush=True)
    # df per token
    from collections import Counter
    df = Counter()
    for rec in ps.parents:
        for t in set(rec.match_ids): df[t] += 1
    senses = {}  # token_id -> list[(centroid float32[d], frozenset(parent_ids))]
    n_multi = 0; n_seen = 0
    for t, d in df.items():
        if not (a.df_min <= d <= a.df_max): continue
        s = v.token_of(t).lstrip("ĠĊ")
        if not s or not any(c.isalpha() for c in s) or any(c.isdigit() for c in s): continue
        variants = ps.case_equiv_ids(t, v)
        occ = set()
        for tv in variants: occ.update(ps._index.get(tv, ()))
        occ = [pi for pi in occ if pv[pi].any()]
        if len(occ) < 25: continue
        n_seen += 1
        if len(occ) > a.max_occ:
            occ = list(np.random.default_rng(0).choice(occ, a.max_occ, replace=False))
        X = pv[np.asarray(occ)]
        # k-means K=3, giữ cụm ≥5 thành viên
        K = 3 if len(occ) >= 60 else 2
        C, asg = kmeans(X, K)
        cl = []
        for k in range(K):
            members = [occ[j] for j in range(len(occ)) if asg[j] == k]
            if len(members) >= 5:
                cl.append((C[k].astype(np.float32), frozenset(int(m) for m in members)))
        if len(cl) < 2: continue
        # ĐA-NGHĨA khi CENTROID khác hướng: min cos-giữa-cặp < ngưỡng. Đơn-nghĩa
        # chẻ bừa ⇒ centroid GIỐNG nhau (cos cao) ⇒ bỏ. Hổ: xấu-hổ vs động-vật xa.
        cens = np.stack([c[0] for c in cl])
        S = cens @ cens.T
        mincos = min(S[i, j] for i in range(len(cl)) for j in range(i+1, len(cl)))
        if mincos > a.spread: continue  # centroid quá giống → đơn nghĩa
        senses[int(t)] = cl; n_multi += 1
        if n_seen % 2000 == 0:
            print(f"[{time.time()-T0:.0f}s]  xét {n_seen} token, đa-nghĩa {n_multi}", flush=True)
    pickle.dump(senses, open(a.ra, "wb"))
    print(f"[{time.time()-T0:.0f}s] LƯU {a.ra}: {n_multi} token đa nghĩa / {n_seen} xét", flush=True)
    # xác minh hổ/chó
    for w in ["hổ", "chó", "đồng", "nước"]:
        tid = v.encode_prompt(w)[-1]
        for tv in ps.case_equiv_ids(tid, v):
            if int(tv) in senses:
                print(f"  «{w}»({tv}): {len(senses[int(tv)])} nghĩa, cỡ {[len(c[1]) for c in senses[int(tv)]]}", flush=True); break
        else:
            print(f"  «{w}»: đơn nghĩa (không lưu)", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
