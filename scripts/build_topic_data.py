"""§TK_096 — dựng data expert THEO CHỦ-ĐỀ (khó hơn theo văn-phong: cùng register bách-khoa).

- math: 5CD-AI/Vietnamese-meta-math-MetaMathQA-40K (query_vi + response_vi).
- geo/history/science: cắt Vietnamese Wikipedia thô theo TỪ-KHOÁ (một lượt stream),
  gán bài vào chủ-đề có điểm từ-khoá trội nhất (>=MIN_HITS, và trội gấp đôi á-quân).

    PYTHONPATH=src .venv/bin/python scripts/build_topic_data.py
"""
import sys, itertools
sys.path.insert(0, "src")
import numpy as np
from datasets import load_dataset
from nn_aivie.core.tokenizer import BPETokenizer

CTX, CAP, MIN_HITS = 128, 360_000, 4
t = BPETokenizer.load("checkpoints/bpe_v1.json")
EOS = t.token_to_id["<EOS>"]
E = np.load("checkpoints/enc_data_viwiki.npz")["E"]


def pack(stream, name):
    nr = len(stream) // CTX
    X = np.array(stream[:nr * CTX], dtype=np.int32).reshape(nr, CTX)
    np.savez(f"checkpoints/enc_{name}.npz", X=X, L=np.full(nr, CTX, np.int32), E=E)
    print(f"LƯU enc_{name}.npz: {X.shape}", flush=True)


# ---- MATH ----
sm, n = [], 0
ds = load_dataset("5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated", split="train", streaming=True)
for r in ds:
    q = (r.get("query_vi") or "").strip(); a = (r.get("response_vi") or "").strip()
    if len(q) < 10 or len(a) < 10:
        continue
    sm.extend(list(t.encode(f"Hỏi: {q}\nĐáp: {a}")) + [EOS]); n += 1
    if len(sm) >= CAP:
        break
print(f"math: {n} bài → {len(sm)} tok", flush=True); pack(sm, "math")

# ---- GEO / HISTORY / SCIENCE từ wiki thô ----
KW = {
    "geo": ["tỉnh", "thành phố", "quốc gia", "sông", "núi", "biển", "đảo", "thủ đô", "dân số",
            "diện tích", "khí hậu", "huyện", "vùng", "lãnh thổ", "giáp", "châu"],
    "hist": ["triều đại", "vua", "hoàng đế", "chiến tranh", "khởi nghĩa", "cách mạng", "thế kỷ",
             "vương quốc", "trận", "nhà Nguyễn", "nhà Lê", "nhà Trần", "thời kỳ", "lịch sử", "đế quốc"],
    "sci": ["nguyên tố", "phân tử", "nguyên tử", "tế bào", "hóa học", "vật lý", "sinh học", "định lý",
            "phản ứng", "năng lượng", "protein", "enzyme", "hạt nhân", "electron", "công thức", "loài"],
}
buf = {k: [] for k in KW}; cnt = {k: 0 for k in KW}; done = set()
ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
scanned = 0
for r in ds:
    scanned += 1
    txt = r["text"]
    if len(txt) < 300:
        continue
    low = txt[:1500].lower()
    score = {k: sum(low.count(w.lower()) for w in ws) for k, ws in KW.items()}
    top = max(score, key=score.get); s = sorted(score.values())
    if score[top] < MIN_HITS or (len(s) > 1 and s[-2] * 2 > s[-1]):   # phải trội & gấp đôi á-quân
        continue
    if top in done:
        continue
    buf[top].extend(list(t.encode(txt[:2000])) + [EOS]); cnt[top] += 1
    if len(buf[top]) >= CAP:
        done.add(top)
    if len(done) == len(KW) or scanned > 60000:
        break
print(f"wiki: quét {scanned} bài → " + " ".join(f"{k}:{cnt[k]}bài/{len(buf[k])}tok" for k in KW), flush=True)
for k in KW:
    pack(buf[k], {"geo": "geo", "hist": "hist", "sci": "sci"}[k])
print("### hết", flush=True)
