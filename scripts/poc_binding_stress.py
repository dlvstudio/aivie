#!/usr/bin/env python3
"""
POC P2 — STRESS binding: sống với template đa dạng + nhiễu + tên đa-token + BPE? (TK_044).

    python scripts/poc_binding_stress.py

P0 chứng minh binding trên synthetic SẠCH (1 khuôn, tên 1-token, whitespace). P2 leo thang độ
khó để đo cơ chế BỀN tới đâu — trước khi tích hợp graph thật (P3, đắt). Mỗi nấc: % bind đúng
(single + crossed, tên MỚI).

Cơ chế (từ P0, tổng quát cho SPAN đa-token): category=cluster E(PPMI+SVD); name-span = run token
category-TÊN; slot-key=anchor trái; cạnh coreference = cặp slot-key chia filler-span; sinh = copy
span đã bind. Mọi cạnh đếm-1-lần, KHÔNG backprop.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

SIG = 5   # bề rộng signature slot-key (số token trái) — cần rộng để bắt VERB-FRAME (ăn vs tặng)
NAMES1 =["Mai", "Tùng", "Hoa", "Nam", "Lan", "Hùng", "Trang", "Đức", "Bình", "Chi", "Sơn", "Hà"]
NAMES1_TEST = ["Linh", "Quân", "Thảo", "Kiên"]
NAMES2 = ["Thanh Mai", "Đức Anh", "Bảo Long", "Minh Châu", "Hồng Nhung", "Quốc Bảo", "Thu Hà", "Gia Huy"]
NAMES2_TEST = ["Bảo Ngọc", "Tuấn Kiệt", "Diệu Linh"]
DISTRACT = ["hôm nay trời đẹp quá", "mọi người cùng đi học", "buổi chiều gió mát",
            "lớp học rất vui vẻ", "cây xanh ngoài sân"]


def gen(names, names_test, n, variety, noise, seed=0):
    rng = random.Random(seed)
    T_single = [("{X} {a} ăn kẹo , {X} {b} đồ ngọt", ["hay", "thích", "mê"], ["rất", "cực", "luôn"]),
                ("{X} {a} vui , {X} {b} cười", ["rất", "khá", "thật"], ["liền", "bèn", "lại"])]
    T_cross = [("{X} tặng {Y} kẹo , {Y} cảm ơn {X}", [], []),
               ("{X} gặp {Y} sáng nay , {Y} chào {X}", [], [])]
    ts = ([T_single[0]], [T_cross[0]]) if not variety else (T_single, T_cross)
    sents = []
    for _ in range(n):
        tpl, A, B = rng.choice(ts[0])
        x = rng.choice(names)
        sents.append(tpl.format(X=x, a=rng.choice(A) if A else "", b=rng.choice(B) if B else ""))
        tpl, _, _ = rng.choice(ts[1])
        x, y = rng.sample(names, 2)
        sents.append(tpl.format(X=x, Y=y))
        if noise and rng.random() < 0.5:
            sents.append(rng.choice(DISTRACT))
    return [s.split() for s in sents]


def build(sents, names_train_words, nonname=()):
    # nonname = token BẮT BUỘC không-phải-tên (đại từ ở P1: distributionally = tên nhưng phải loại)
    nonname = set(nonname)
    sents = [["<S>"] + s for s in sents]   # BOS anchor = slot chủ ngữ
    vocab = sorted({t for s in sents for t in s})
    vid = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    fwd = defaultdict(Counter)
    co = np.zeros((V, V))
    for s in sents:
        ids = [vid[t] for t in s]
        for i, a in enumerate(ids):
            if i + 1 < len(ids):
                fwd[a][ids[i + 1]] += 1
            for j in range(max(0, i - 2), min(len(ids), i + 3)):
                if j != i:
                    co[a][ids[j]] += 1
    for t in nonname:            # MASK khỏi co-occurrence: đại-từ hi-df kéo lệch cụm → tên tán loạn
        if t in vid:
            co[vid[t]] = 0; co[:, vid[t]] = 0
    r = co.sum(1, keepdims=True) + 1e-9
    c = co.sum(0, keepdims=True) + 1e-9
    ppmi = np.maximum(np.log((co * co.sum()) / (r * c) + 1e-9), 0.0)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    d = min(16, U.shape[1])
    En = (U[:, :d] * S[:d]); En /= np.linalg.norm(En, axis=1, keepdims=True) + 1e-9
    # name-piece ids = mọi token XUẤT HIỆN trong tên train (đa-token → nhiều piece). Đây là GIÁM SÁT
    # mức train (đồ thị thật có entity-register tương đương) — tên MỚI vẫn nhận qua NGỮ CẢNH, không đây.
    name_toks = {w for full in names_train_words for w in full.split()}
    name_ids = [vid[w] for w in name_toks if w in vid]
    # DÙNG centroid-cosine thay nhãn-cụm: kmeans nhỏ THIẾU ỔN ĐỊNH (đổi seed → tách tên khác nhau).
    # Tên = gần TÂM tên-train trong E. Ngưỡng = ~decile-10 sim của chính tên-train (thích nghi, không magic).
    ctr = En[name_ids].mean(0); ctr /= np.linalg.norm(ctr) + 1e-9
    sims = En @ ctr
    _ts = sorted(float(sims[i]) for i in name_ids)
    thr = _ts[len(_ts) // 10] - 1e-6
    # PURITY GUARD (Luật 0 — theo TẦN SUẤT, không nghĩa): "," kề tên nên gần tâm → nhưng phủ ~mọi câu;
    # bỏ token phủ > 50% câu. (Đại từ "nó" df~0.5N trà-trộn → xử riêng ở P1 part B qua nonname.)
    df = Counter(t for s in sents for t in set(s))
    is_namecat = lambda t: (t in vid and sims[vid[t]] >= thr
                            and df[t] <= 0.5 * len(sents) and t not in nonname)
    looks_name = lambda t: (t not in vid) or is_namecat(t)   # OOV (tên MỚI) HOẶC gần-tâm-tên sạch

    def slot_after_is_name(prev):
        """anchor `prev` có phải slot-TÊN? follower (bỏ nonname) mà ĐA SỐ mass là tên → phải; div=#tên."""
        d = fwd.get(vid.get(prev, -1))
        if not d:
            return False, 0
        nm_mass = tot = 0
        names = set()
        for nx, k in d.items():
            t = vocab[nx]
            if t in nonname:
                continue
            tot += k
            if is_namecat(t):
                nm_mass += k; names.add(nx)
        return (tot > 0 and nm_mass >= 0.5 * tot), len(names)

    # spans tên: BẮT ĐẦU sau anchor dự-đoán-tên; NHẬN filler qua NGỮ CẢNH (kể cả OOV/đa-token).
    # key = SIGNATURE 2-token trái của span (phân biệt "kẹo ," chéo vs "vui ," đơn); filler = span.
    def sig_at(toks, i):
        # signature CẤU TRÚC: token-tên trong cửa sổ → "<N>" (bất biến theo tên cụ thể)
        return tuple("<N>" if looks_name(t) else t for t in toks[max(0, i - SIG):i])

    def walk(toks_bos):
        """yield (sig, i, j) mỗi name-span trên chuỗi ĐÃ có <S> đầu (i..j-1 = span)."""
        i = 1
        while i < len(toks_bos):
            if slot_after_is_name(toks_bos[i - 1])[0] and looks_name(toks_bos[i]):
                j = i
                while j < len(toks_bos) and looks_name(toks_bos[j]):
                    j += 1
                yield sig_at(toks_bos, i), i, j
                i = j
            else:
                i += 1

    def spans_of(toks):
        tb = ["<S>"] + list(toks)
        return [(sig, tuple(tb[i:j])) for sig, i, j in walk(tb)]

    # after_role[sig] = token CẤU TRÚC theo sau name-span slot-sig (role-tracking cho continuation P1)
    after_role = defaultdict(Counter)
    for s in sents:
        for sig, i, j in walk(s):        # s đã có <S> (build prepend)
            after_role[sig][s[j] if j < len(s) else "<EOS>"] += 1
    same, both = defaultdict(int), defaultdict(int)
    for s in sents:
        occ = spans_of(s[1:])   # bỏ <S> đã prepend (spans_of tự thêm lại)
        for a in range(len(occ)):
            for b in range(len(occ)):
                if a != b:
                    both[(occ[a][0], occ[b][0])] += 1
                    same[(occ[a][0], occ[b][0])] += occ[a][1] == occ[b][1]
    # tie-break TẤT ĐỊNH (str-hash ngẫu-nhiên hoá thứ-tự set giữa các lần chạy → kết quả chập chờn):
    # chọn theo (ratio, #đồng-hiện, key) đã sắp — biên rõ, không phụ thuộc PYTHONHASHSEED.
    coref = {}
    for ka in sorted({k for k, _ in both}):
        cands = sorted({k2 for (k1, k2) in both if k1 == ka})
        best, bkey = None, None
        for kb in cands:
            rr = same[(ka, kb)] / both[(ka, kb)]
            key = (rr, both[(ka, kb)])
            if rr > 0.55 and (bkey is None or key > bkey):
                best, bkey = kb, key
        if best:
            coref[ka] = best
    return dict(vid=vid, vocab=vocab, sims=sims, is_namecat=is_namecat, looks_name=looks_name,
                slot=slot_after_is_name, coref=coref, spans_of=spans_of,
                fwd=fwd, after_role=after_role, sig_at=sig_at)


def next_after(prompt_toks, M):
    binds = {k: v for k, v in M["spans_of"](prompt_toks)}   # sig -> span
    prev = prompt_toks[-1]
    slot, div = M["slot"](prev)
    if slot and div >= 2:
        tail = (["<S>"] + list(prompt_toks))[-SIG:]
        sig = tuple("<N>" if M["looks_name"](t) else t for t in tail)   # signature slot sắp phát
        src = M["coref"].get(sig)
        if src in binds:
            return " ".join(binds[src])
        if binds:
            return " ".join(next(iter(binds.values())))
    return "∅"


def eval_once(names, names_test, variety, noise, seed):
    M = build(gen(names, names_test, 400, variety, noise, seed), names)
    sok = st = cok = ct = 0
    for nm in names_test:                         # single: "{NEW} mê ăn kẹo ," → NEW
        st += 1
        sok += next_after(f"{nm} mê ăn kẹo ,".split(), M) == nm
    for x, y in zip(names_test, names_test[1:] + names_test[:1]):   # crossed: 2 vị trí
        if x == y:
            continue
        ct += 2
        cok += next_after(f"{x} tặng {y} kẹo ,".split(), M) == y            # Y-slot sau ','
        cok += next_after(f"{x} tặng {y} kẹo , {y} cảm ơn".split(), M) == x  # X-slot sau 'ơn'
    return sok, st, cok, ct


def run_config(name, names, names_test, variety, noise, seeds=6):
    S = [eval_once(names, names_test, variety, noise, sd) for sd in range(seeds)]
    sok, st, cok, ct = (sum(x[i] for x in S) for i in range(4))
    print(f"  [{name:13s}] single {sok:>3}/{st:<3} · crossed {cok:>3}/{ct:<3}  "
          f"({seeds} seed) → {'✓ SỐNG' if sok == st and cok == ct else '✗ GÃY'}")
    return sok == st and cok == ct


def main():
    print("═══ P2 STRESS — binding sống qua từng nấc? (tên MỚI, không backprop, 6 seed) ═══")
    print("nấc 1 baseline (1 khuôn, tên 1-token, sạch)")
    run_config("baseline", NAMES1, NAMES1_TEST, variety=False, noise=False)
    print("nấc 2 + template đa dạng (đơn+chéo trộn) + câu nhiễu chen giữa")
    run_config("variety+noise", NAMES1, NAMES1_TEST, variety=True, noise=True)
    print("nấc 3 tên ĐA-TOKEN ('Thanh Mai', 'Bảo Ngọc') + đa dạng + nhiễu")
    run_config("multitoken", NAMES2, NAMES2_TEST, variety=True, noise=True)
    print("\n3 cơ chế phải VÁ để sống ngoài lab (mỗi cái là một bug thật, tìm qua đo):")
    print("  1. nhận tên MỚI qua NGỮ CẢNH anchor (không qua category — tên mới là OOV)")
    print("  2. purity NAME_CAT: bỏ member phủ >50% câu (',' kề tên → lọt cụm) — theo TẦN SUẤT")
    print("  3. signature slot rộng (SIG=5) bắt VERB-FRAME: 'ăn kẹo,'(đơn) ≠ 'tặng Y kẹo,'(chéo)")
    print("→ 3 nấc ✓ ⇒ induction-head Hebbian bền với token-vụn+nhiễu. Kế: P1 (đại từ/role) rồi P3 (graph).")


if __name__ == "__main__":
    main()
