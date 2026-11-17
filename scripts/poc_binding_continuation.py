#!/usr/bin/env python3
"""
POC P1 — CONTINUATION (role-tracking) + COREFERENCE qua ĐẠI TỪ (TK_044, 2026-07-22).

    python scripts/poc_binding_continuation.py

Nối tiếp P0/P2 (`poc_hebbian_binding.py`, `poc_binding_stress.py`). Hai mảnh P1:

  A. ROLE-TRACKING — sinh FULL câu qua tên-MỚI. P0/P2 chỉ điền 1 token rồi dừng: sau khi copy tên
     OOV "Quân", bigram(Quân) RỖNG → tắc. Vá: token sau một tên do VAI (slot-signature) quyết định,
     KHÔNG do danh tính tên → back-off `after_role[sig]` (đếm Hebbian: sau tên ở slot-sig là token gì).
     Mục tiêu: "Linh tặng Quân kẹo ," → "… Quân cảm ơn Linh".

  B. ĐẠI TỪ ("… , nó …") — "nó" là token ĐƠN tần-suất-cao → purity guard loại khỏi NAME_CAT → cơ
     chế gốc KHÔNG cấp cạnh coref cho nó (đây là giới hạn thật, đo ở B0). Vá tối thiểu: "nó" =
     BIẾN-COREF (đứng vị-trí-tên nhưng không phải tên) → GIẢI bằng cách MƯỢN cạnh coref của TÊN
     (sig chỗ nó đứng → antecedent slot → filler). Đo: "nó" trỏ đúng tiền-ngữ (tên MỚI) chưa.

Mọi cạnh đếm-1-lần lúc ingest, KHÔNG backprop.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import poc_binding_stress as B
from poc_binding_stress import SIG, NAMES1, NAMES1_TEST, NAMES2, NAMES2_TEST, build, gen


def _sig(toks_bos, end, M):
    """signature cấu trúc của slot kết ở vị trí `end` (loại trừ end), tên→<N>."""
    return tuple("<N>" if M["looks_name"](t) else t for t in toks_bos[max(0, end - SIG):end])


# ─────────────────────────────── A. ROLE-TRACKING CONTINUATION ───────────────────────────────
def continue_gen(prompt, M, max_steps=10):
    toks = ["<S>"] + prompt.split()
    last_sig = None
    for _ in range(max_steps):
        prev = toks[-1]
        slot, div = M["slot"](prev)
        if slot and div >= 2:                                  # 1. copy-slot: điền tên bind qua coref
            sig = _sig(toks, len(toks), M)
            binds = {k: v for k, v in M["spans_of"](toks[1:])}
            src = M["coref"].get(sig)
            span = binds.get(src) or (next(iter(binds.values())) if binds else None)
            if span:
                toks.extend(span); last_sig = sig; continue
        if M["vid"].get(prev, -1) < 0 and last_sig is not None:  # 2. prev = tên-MỚI OOV → ROLE-BACKOFF
            ar = M["after_role"].get(last_sig); last_sig = None
            nxt = max(ar, key=ar.get) if ar else "<EOS>"
            if nxt == "<EOS>":
                break
            toks.append(nxt); continue
        d = M["fwd"].get(M["vid"].get(prev, -1))                # 3. bigram cấu trúc
        if not d:
            break
        toks.append(M["vocab"][max(d, key=d.get)])
    return toks[1:]


def name_seq(tokens, cands):
    """chuỗi tên xuất hiện theo thứ tự, KHỚP với tên-test đã biết (bền khi tên chia mảnh với train,
    vd 'Bảo' của 'Bảo Ngọc' cũng là mảnh train → không thể dựa OOV)."""
    parts = [(c, c.split()) for c in cands]
    out, i = [], 0
    while i < len(tokens):
        hit = next((c for c, ts in parts if tokens[i:i + len(ts)] == ts), None)
        if hit:
            out.append(hit); i += len(hit.split())
        else:
            i += 1
    return out


def part_A(names, names_test, multi, seeds=6):
    # ĐÚNG = mọi slot-tên trong continuation điền đúng biến + KHÔNG trôi sang tên TRAIN (drift).
    # (Greedy có thể lạc sang khuôn khác → dài hơn nhưng vẫn copy đúng — không phạt độ dài.)
    ok_s = ok_c = n = 0
    sample = None
    for sd in range(seeds):
        M = build(gen(names, names_test, 400, True, True, sd), names)
        for x, y in zip(names_test, names_test[1:] + names_test[:1]):
            if x == y:
                continue
            n += 1
            gs = continue_gen(f"{x} mê ăn kẹo ,", M)
            gc = continue_gen(f"{x} tặng {y} kẹo ,", M)
            no_drift = lambda g: not name_seq(g, names)          # không tên TRAIN nào lọt vào
            ns, nc = name_seq(gs, [x]), name_seq(gc, [x, y])
            ok_s += len(ns) >= 2 and all(t == x for t in ns) and no_drift(gs)
            ok_c += nc[:4] == [x, y, y, x] and no_drift(gc)
            if sd == 0 and sample is None:
                sample = (" ".join(gs), " ".join(gc))
    tag = "multitoken" if multi else "single-word"
    print(f"  [{tag:11s}] single-cont {ok_s}/{n} · crossed-cont {ok_c}/{n}  ({seeds} seed)")
    print(f"      vd single : {sample[0]}")
    print(f"      vd crossed: {sample[1]}")
    return ok_s == n and ok_c == n


# ─────────────────────────────── B. COREFERENCE QUA ĐẠI TỪ ───────────────────────────────
def gen_pron(names, n, seed, pron_frac=0.5):
    """khuôn TRỘN tên-lặp (dạy cạnh coref) + đại-từ 'nó' (single & crossed)."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.choice(names)
        if rng.random() < pron_frac:
            out.append(f"{x} mê ăn kẹo , nó rất thích chè".split())
        else:
            out.append(f"{x} mê ăn kẹo , {x} rất thích chè".split())
        x, y = rng.sample(names, 2)
        if rng.random() < pron_frac:
            out.append(f"{x} tặng {y} kẹo , nó cảm ơn {x}".split())
        else:
            out.append(f"{x} tặng {y} kẹo , {y} cảm ơn {x}".split())
    return out


def detect_pron(sents, names_train):
    """BIẾN-COREF, phát hiện KHÔNG cần category (category bị đại-từ làm hỏng): một SLOT-BIẾN = anchor
    có NHIỀU follower rời-rạc (nhiều tên khác nhau điền); trong đám đó, type TẦN-SUẤT-CAO (df lớn) =
    đại từ (tên thì mỗi cái hiếm). Phân biệt bằng cấu-trúc phân-bố + df, không dùng nghĩa (Luật 0)."""
    sb = [["<S>"] + s for s in sents]
    n = len(sb)
    df = Counter(t for s in sb for t in set(s))
    nxt = defaultdict(Counter)
    for s in sb:
        for i in range(len(s) - 1):
            nxt[s[i]][s[i + 1]] += 1
    votes = Counter()
    for a, d in nxt.items():
        F = list(d)
        if len(F) < 4:                                 # cần ≥4 follower để là slot-biến
            continue
        dfs = sorted(df[f] for f in F)
        med = dfs[len(dfs) // 2]                        # slot-tên: đa số follower HIẾM → med thấp
        for f in F:                                     # đại từ = OUTLIER df cao lạc giữa đám tên hiếm
            if f != "," and df[f] > max(3 * med, 0.25 * n):
                votes[f] += 1
    return {t for t, v in votes.items() if v >= 1}


def resolve_pron(prompt, M):
    """prompt KẾT bằng đại-từ → mượn cạnh coref của TÊN (sig chỗ nó đứng) → tiền-ngữ."""
    toks = ["<S>"] + prompt.split()
    sig = _sig(toks, len(toks) - 1, M)                  # signature slot của đại-từ (loại nó)
    src = M["coref"].get(sig)
    binds = {k: v for k, v in M["spans_of"](toks[1:-1])}
    span = binds.get(src)
    return " ".join(span) if span else "∅"


def part_B(names, names_test, seeds=6):
    # B0: cơ chế GỐC (không xử đại từ) — "nó" phân-bố Y HỆT tên → lọt NAME_CAT, tán loạn tên,
    # KHÔNG cấp được cạnh coref name-repeat sạch. Đây là giới hạn THẬT (đo bằng resolve trên M gốc).
    sents0 = gen_pron(names, 400, 0)
    prons = detect_pron(sents0, names)
    M0 = build(sents0, names)                     # gốc: KHÔNG loại đại từ
    base = sum(resolve_pron(f"{x} mê ăn kẹo , nó", M0) == x for x in names_test)
    print(f"  B0 GỐC (không xử): biến-coref phát hiện={sorted(prons)} · giải nó=chủ-ngữ "
          f"{base}/{len(names_test)} — 'nó' phân-bố=tên → nhiễu category, coref hụt.")
    # B: loại đại từ khỏi TÊN (build nonname=prons) → tên cụm sạch, coref name-repeat học được →
    #    GIẢI 'nó' bằng cách MƯỢN cạnh coref của TÊN (sig chỗ nó đứng → antecedent slot → filler).
    ok_s = ok_c = n = 0
    sample = []
    for sd in range(seeds):
        sents = gen_pron(names, 400, sd)
        pr = detect_pron(sents, names)
        M = build(sents, names, nonname=pr)
        for x, y in zip(names_test, names_test[1:] + names_test[:1]):
            if x == y:
                continue
            n += 1
            ok_s += resolve_pron(f"{x} mê ăn kẹo , nó", M) == x        # nó = x (chủ ngữ)
            ok_c += resolve_pron(f"{x} tặng {y} kẹo , nó", M) == y     # nó = y (tân ngữ)
            if sd == 0 and len(sample) < 2:
                rs = resolve_pron(f"{x} mê ăn kẹo , nó", M)
                rc = resolve_pron(f"{x} tặng {y} kẹo , nó", M)
                sample.append((f"{x} … nó→{rs}(mong {x})", f"{x} tặng {y} … nó→{rc}(mong {y})"))
    print(f"  B  loại đại-từ+mượn coref TÊN: nó=chủ-ngữ {ok_s}/{n} · nó=tân-ngữ {ok_c}/{n} ({seeds} seed)")
    for a, b in sample:
        print(f"      {a} · {b}")
    return ok_s == n and ok_c == n


def main():
    print("═══ P1 — CONTINUATION (role-tracking) + ĐẠI TỪ (tên MỚI, không backprop, 6 seed) ═══")
    print("A. ROLE-TRACKING — sinh FULL câu qua tên-mới (bigram tên-OOV rỗng → back-off theo VAI):")
    a1 = part_A(NAMES1, NAMES1_TEST, multi=False)
    a2 = part_A(NAMES2, NAMES2_TEST, multi=True)
    print("\nB. COREFERENCE qua ĐẠI TỪ ('… , nó …' → nó trỏ ai?):")
    b = part_B(NAMES1, NAMES1_TEST)
    print()
    print(f"→ A role-tracking: {'✓ sinh full câu qua tên-mới' if a1 and a2 else '✗ xem gãy đâu'}. "
          f"B đại từ: {'✓ giải đúng tiền-ngữ (mượn cạnh coref tên)' if b else '✗ chưa'}.")
    print("  Giới hạn thật (B0): đại từ KHÔNG tự có coref — phải MƯỢN cạnh của tên-lặp; corpus toàn"
          " đại-từ (không tên-lặp) sẽ KHÔNG học được slot-coref. Kế: P3 cắm 2 kênh vào graph thật.")


if __name__ == "__main__":
    main()
