#!/usr/bin/env python3
"""
PROBE TÁCH-ASPECT — prompt đa-góc-nhìn → gom token theo E → truy hồi TỪNG aspect (TK_042).

    ⚠️ CHẠY M4 — checkpoint đồ thị 2GB → ~20GB RAM (TK_040).
    PYTHONPATH=src python -u scripts/probe_aspect_split.py --checkpoint checkpoints/network_v11.pkl

## Ý tưởng tác giả 2026-07-20

Thay vì query 1 fact, tách prompt thành nhiều "chú ý" (aspect), truy hồi RIÊNG từng aspect rồi
trộn. VD "bằng văn phong hài hước hiện đại, miêu tả trận Bạch Đằng của Ngô Quyền" → aspect
{hài hước, hiện đại} / {miêu tả} / {Bạch Đằng} / {Ngô Quyền}.

## Probe NÀY đo phần YẾU NHẤT trước (chặn build vô ích)

Trước khi xây spreading đa-seed (trộn), đo: **truy hồi TỪNG aspect trả về gì?**
* aspect NỘI DUNG (Bạch Đằng, Ngô Quyền) → ra fact LIÊN QUAN? (⇒ trục nội dung sống, đáng trộn)
* aspect VĂN PHONG (hài hước, hiện đại) → ra gì? Dự đoán: định nghĩa/rác — vì "hài hước" là
  CÁCH NÓI không phải fact-để-truy-hồi (graph lưu WHAT không lưu HOW — probe tầng-khuôn đã đo).
  Nếu đúng ⇒ xác nhận tường style bằng SỐ ⇒ style/instruction về LM+SFT, KHÔNG nhét vào graph.

Tách aspect: gom token nội dung prompt (gdf>0, gidf cao) theo **E-cosine** (union-find,
ngưỡng `--sim`) — Luật 0 sạch (cluster E, không nhãn). Mỗi nhóm → `facts_about` → decode đọc tay.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.generate_hebbian import _completion_from_parent, _content_ids  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402

PROMPTS = [
    "bằng văn phong hài hước hiện đại, hãy miêu tả trận đánh Bạch Đằng của Ngô Quyền",
    "kể một câu chuyện buồn về mùa thu Hà Nội",
    "giải thích ngắn gọn dễ hiểu cho trẻ em vì sao trời mưa",
    "so sánh nhà Lý và nhà Trần một cách khách quan",
]


def group_by_e(tokens, E, sim: float):
    """Union-find: gom token có E-cosine ≥ sim thành aspect (Luật 0 — cluster phân bố)."""
    n = len(tokens)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    V = np.stack([E[t] for t in tokens]).astype(np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    S = V @ V.T
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= sim:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(tokens[i])
    return list(groups.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--sim", type=float, default=0.35, help="ngưỡng E-cosine gom aspect")
    ap.add_argument("--max-anchors", type=int, default=8, help="số token nội dung xét (gidf cao)")
    ap.add_argument("--facts", type=int, default=3, help="fact/aspect")
    args = ap.parse_args()

    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    log(f"load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    ps = state.parents
    if ps is None or state.embedding is None:
        raise SystemExit("✗ checkpoint thiếu parents/embedding")
    E = np.asarray(state.embedding.matrix)
    vocab = state.vocab
    sur = vocab.token_of
    log(f"✓ loaded · parents {len(ps):,} · E {E.shape}")

    for prompt in PROMPTS:
        log(f"\n═══ PROMPT: {prompt} ═══")
        qids = list(vocab.encode_prompt(prompt))
        content = _content_ids(qids, vocab, skip=set())
        # token mạng ĐÃ BIẾT (gdf>0), xếp theo gidf (đặc thù) giảm dần, lấy top
        cand = [t for t in dict.fromkeys(content) if ps._gdf.get(t, 0) > 0 and t < E.shape[0]]
        cand.sort(key=lambda t: -ps._gidf(t))
        cand = cand[: args.max_anchors]
        if not cand:
            log("  (không token nội dung nào mạng biết)")
            continue
        aspects = group_by_e(cand, E, args.sim)
        log(f"  {len(cand)} anchor → {len(aspects)} aspect (E-cosine ≥ {args.sim}):")
        for gi, grp in enumerate(aspects):
            names = " ".join(f"{sur(t)!r}" for t in grp)
            gidfs = ",".join(f"{ps._gidf(t):.1f}" for t in grp)
            log(f"  ── aspect {gi}: {names}  (gidf {gidfs})")
            facts = ps.facts_about(tuple(grp), top_k=args.facts, vocab=vocab)
            if not facts:
                log("       → (facts_about: RỖNG — không fact nào)")
                continue
            for pi in facts:
                ids = list(_completion_from_parent(ps, pi))
                txt = decode_surface(state, ids)[:180]
                log(f"       → {txt!r}")

    log("\n═══ ĐỌC TAY: aspect NỘI DUNG (Bạch Đằng/Ngô Quyền/Hà Nội) ra fact LIÊN QUAN? "
        "aspect VĂN PHONG (hài hước/hiện đại) ra định nghĩa/rác? ═══")
    log("→ nội dung sống + văn phong rác ⇒ build spreading đa-seed cho NỘI DUNG, style về LM+SFT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
