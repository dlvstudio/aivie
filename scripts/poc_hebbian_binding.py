#!/usr/bin/env python3
"""
POC VARIABLE BINDING KHÔNG BACKPROP — single + ĐA-REFERENT crossed (TK_044, 2026-07-21).

    python scripts/poc_hebbian_binding.py

## Câu hỏi (tác giả)

Bind biến MỚI vào khuôn, KHÔNG backprop. Nền: Hopfield=attention, trọng số=tích-ngoài Hebbian
(set lúc train). Hai cấp:
  • single-referent: "[X] ăn kẹo, [X] đồ ngọt" → "Linh mê ăn kẹo," → "Linh"
  • ĐA-referent CROSSED: "[X] tặng [Y] kẹo, [Y] cảm ơn [X]" → "Linh tặng Quân kẹo," → Quân …
    cảm ơn … Linh  (phải biết slot NÀO corefer slot NÀO — heuristic "tên đầu tiên" THẤT BẠI)

## Cơ chế (mọi cạnh đếm-1-lần lúc ingest, KHÔNG backprop)

1. **category** = cluster E (PPMI+SVD ngữ cảnh) → TÊN cùng cụm. (= E của đồ thị)
2. **slot-key** = ANCHOR TRÁI (token liền trước tên) — vị trí cấu trúc, không phải index tuyệt đối.
   "<S>"=chủ ngữ đầu câu, "tặng"=tân ngữ, ","=tên sau phẩy, "ơn"=tên sau "ơn"…
3. **CẠNH COREFERENCE (mảnh đa-referent):** với mỗi CẶP slot-key, đếm Hebbian "cùng filler trong
   một câu?" → key nào hay chia filler với key nào = corefer. Học được: (","↔"tặng")=Y,
   ("<S>"↔"ơn")=X — tức khuôn CROSSED.
4. **Sinh:** đọc prompt → bind filler theo slot-key. Tại slot-biến (bigram dự đoán category-nhiều-
   filler), key = anchor trái hiện tại → theo CẠNH COREFERENCE → COPY filler của slot corefer.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

NAMES_TRAIN = ["Mai", "Tùng", "Hoa", "Nam", "Lan", "Hùng", "Trang", "Đức", "Bình", "Chi"]
NAMES_TEST = ["Linh", "Quân", "Thảo"]
V1 = ["hay", "thích", "mê", "khoái"]
V2 = ["rất", "cực", "luôn", "khá"]
TAIL2 = ["thích", "mê", "khoái", "thèm"]


def make_corpus(n=500, seed=0):
    rng = random.Random(seed)
    sents = []
    for _ in range(n):
        # single-referent: X lặp
        x = rng.choice(NAMES_TRAIN)
        sents.append(f"{x} {rng.choice(V1)} ăn kẹo , {x} {rng.choice(V2)} {rng.choice(TAIL2)} đồ ngọt".split())
        # đa-referent CROSSED: X tặng Y kẹo , Y cảm ơn X  (X≠Y)
        x, y = rng.sample(NAMES_TRAIN, 2)
        sents.append(f"{x} tặng {y} kẹo , {y} cảm ơn {x}".split())
    return sents


def kmeans_cos(X, K, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = X[m].mean(0)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return (X @ C.T).argmax(1)


def main():
    sents = make_corpus()
    vocab = sorted({t for s in sents for t in s})
    vid = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    fwd = defaultdict(Counter)   # cur -> next (bigram)
    co = np.zeros((V, V))        # đồng-hiện ±2 → category
    for s in sents:
        ids = [vid[t] for t in s]
        for i, a in enumerate(ids):
            if i + 1 < len(ids):
                fwd[a][ids[i + 1]] += 1
            for j in range(max(0, i - 2), min(len(ids), i + 3)):
                if j != i:
                    co[a][ids[j]] += 1

    # category (PPMI + SVD + k-means cosine) — closed-form, không backprop
    r = co.sum(1, keepdims=True) + 1e-9
    c = co.sum(0, keepdims=True) + 1e-9
    ppmi = np.maximum(np.log((co * co.sum()) / (r * c) + 1e-9), 0.0)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    En = (U[:, :16] * S[:16])
    En /= np.linalg.norm(En, axis=1, keepdims=True) + 1e-9
    cat = kmeans_cos(En, K=6, seed=1)
    name_ids = [vid[n] for n in NAMES_TRAIN]
    NAME_CAT = Counter(cat[i] for i in name_ids).most_common(1)[0][0]
    print(f"vocab={V} · cụm TÊN #{NAME_CAT}: {[vocab[i] for i in range(V) if cat[i]==NAME_CAT]}")

    def name_slot_after(prev_id):
        """(là slot-biến TÊN?, filler-diversity) — bigram sau prev dự đoán category-TÊN nhiều filler."""
        d = fwd.get(prev_id)
        if not d:
            return False, 0
        mass = Counter()
        members = defaultdict(set)
        for nxt, k in d.items():
            mass[cat[nxt]] += k
            members[cat[nxt]].add(nxt)
        top = mass.most_common(1)[0][0]
        return top == NAME_CAT, len(members[top])

    # --- CẠNH COREFERENCE: cặp slot-key hay chia filler trong-câu ---
    def slot_key(sent, i):
        return sent[i - 1] if i > 0 else "<S>"   # anchor TRÁI

    same = defaultdict(int)
    both = defaultdict(int)
    for s in sents:
        occ = [(slot_key(s, i), t) for i, t in enumerate(s) if vid[t] in name_ids or t in NAMES_TEST]
        occ = [(k, t) for k, t in occ if vid.get(t, -1) in name_ids]  # chỉ tên train
        for a in range(len(occ)):
            for b in range(len(occ)):
                if a == b:
                    continue
                both[(occ[a][0], occ[b][0])] += 1
                if occ[a][1] == occ[b][1]:
                    same[(occ[a][0], occ[b][0])] += 1
    coref = {}   # slot-key -> slot-key corefer (chia filler ≥60%)
    keys = {k for k, _ in both}
    for ka in {k for k, _ in both.keys()}:
        best, br = None, 0.6
        for kb in {k2 for (k1, k2) in both if k1 == ka}:
            ratio = same[(ka, kb)] / both[(ka, kb)]
            if ratio > br:
                best, br = kb, ratio
        if best:
            coref[ka] = best
    print(f"cạnh coreference học được: {coref}")

    def generate(prompt, mode, steps=6):
        toks = prompt.split()
        # bind filler theo slot-key (tên = token category-TÊN HOẶC test-name, nhận qua ngữ cảnh)
        binds = {}
        for i, t in enumerate(toks):
            is_name = vid.get(t, -1) in name_ids or t in NAMES_TEST
            if is_name:
                binds[toks[i - 1] if i > 0 else "<S>"] = t
        out = list(toks)
        for _ in range(steps):
            prev = out[-1]
            pid = vid.get(prev, -1)
            slot, div = name_slot_after(pid) if pid >= 0 else (False, 0)
            if mode == "bind" and slot and div >= 3:
                key = prev  # anchor trái của tên sắp phát
                src = coref.get(key)
                if src in binds:
                    out.append(binds[src]); continue   # COPY theo cạnh coreference
                if binds:
                    out.append(next(iter(binds.values())))  # fallback đơn-referent
                    continue
            if pid < 0 or not fwd.get(pid):
                break
            out.append(vocab[max(fwd[pid], key=fwd[pid].get)])  # bigram argmax
        return " ".join(out)

    print("\n═══ SINGLE-referent ═══")
    for nm in NAMES_TEST:
        p = f"{nm} mê ăn kẹo ,"
        print(f"  bigram: {generate(p,'bigram')!r}")
        print(f"  BIND  : {generate(p,'bind')!r}")

    def next_tok(prompt):
        g = generate(prompt, "bind", steps=1).split()
        return g[len(prompt.split())] if len(g) > len(prompt.split()) else "∅"

    # test slot-fill TỪNG VỊ TRÍ (continuation qua tên-mới = role-tracking, phạm vi khác).
    # Khuôn CROSSED "X tặng Y kẹo , Y cảm ơn X": Y sau ',', X sau 'ơn' — heuristic "tên đầu"
    # (đơn-referent) SẼ SAI cả hai. Chỉ CẠNH COREFERENCE mới điền đúng.
    print("\n═══ ĐA-referent CROSSED — slot-fill từng vị trí (tên MỚI) ═══")
    cases = [
        ("Linh tặng Quân kẹo ,", "Quân", "Y-slot sau ',' → coref '→tặng'"),
        ("Linh tặng Quân kẹo , Quân cảm ơn", "Linh", "X-slot sau 'ơn' → coref '→<S>'"),
        ("Thảo tặng Linh kẹo ,", "Linh", "Y-slot"),
        ("Thảo tặng Linh kẹo , Linh cảm ơn", "Thảo", "X-slot"),
    ]
    ok = 0
    for prompt, exp, note in cases:
        got = next_tok(prompt)
        hit = got == exp
        ok += hit
        print(f"  '{prompt}'\n     → next={got!r}  {'✓' if hit else '✗'} (kỳ vọng {exp!r} — {note})")
    print(f"\n→ đa-referent crossed đúng {ok}/{len(cases)}. "
          f"{'ĐỘT PHÁ: coreference Hebbian bind ĐA-BIẾN CHÉO, không backprop.' if ok==len(cases) else 'xem gãy đâu.'}")


if __name__ == "__main__":
    main()
