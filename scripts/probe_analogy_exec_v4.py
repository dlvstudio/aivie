#!/usr/bin/env python
"""P98F+ (TK_046) — cầu thuộc tính · lop_trai · CỘNG NHIỀU CẦU.

Giữ: bigram d=1, p→x, khớp lop_trai.
Mới: score(c) = Σ_p strength(p,c) trên mọi cầu khớp (không chỉ argmax 1 p).
     càng nhiều cầu → trọng số càng cao.

Cổng: RUNBOOK_TK_046_P98F.md
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_N_POS = 10
GATE_HIT = 0.40
GATE_FP = 0.15

CLASSIFIER = frozenset({"con", "cái", "một", "các", "những"})
GEO_LEFT = frozenset({
    "Sông", "sông", "Núi", "núi", "biển", "Biển", "Đảo", "đảo",
    "vịnh", "Vịnh", "hồ", "Hồ", "đồng", "Đồng", "phố", "Phố",
})
PERSON_LEFT = CLASSIFIER | frozenset({
    "người", "Người", "cha", "Cha", "mẹ", "Mẹ", "ông", "Ông", "bà", "Bà",
    "gái", "trai", "cháu", "anh", "chị", "em", "vợ", "chồng", "con",
})
STOP = frozenset({
    "con", "cái", "là", "màu", "các", "những", "một", "của", "và",
    "cho", "với", "trong", "trên", "dưới", "vào", "từ", "đến", "về", "như",
    "khi", "mà", "để", "bị", "được", "đã", "đang", "sẽ", "cũng", "rất",
    "này", "đó", "kia", "nào", "gì", "ai", "đây", "Đây", "Đó", "đó",
    "người", "nơi", "việc", "theo", "sau", "trước", "giữa", "hay", "hoặc",
    "nhưng", "vì", "nếu", "do", "bởi", "qua", "lại", "ra", "lên", "xuống",
    "đi", "làm", "có", "không", "chỉ", "còn", "vẫn", "thì", "gọi", "tức",
    "tên", "di", "hành", "cách", "coi", "xem", "EOS", "<EOS>",
})
BAD_HEAD = frozenset({
    "gọi", "coi", "xem", "tức", "tên", "hay", "dù", "phải", "còn", "vẫn",
    "cũng", "dưới", "trên", "nhất", "nay", "từng", "nghĩa", "truyền", "chính",
    "sự", "Đây", "đây", "Đó", "đó", "Nó", "nó", "họ", "mình", "người",
    "việc", "chỉ", "vua", "thứ", "hiệu", "xưng", "nguyên", "trái", "Vì",
    "vì", "Do", "Nếu", "Khi", "Sau", "Tuy", "Dù", "Chính", "Bit", "ADN",
    "vừa", "thường", "dẫu", "thực", "hoặc", "như", "lần", "Gen", "Thuế",
    "Nghĩa", "Kết", "kết",
})
BAD_PRED = frozenset({
    "người", "nơi", "vị", "số", "hai", "lần", "cách", "di", "hành", "cũng",
    "rất", "không", "có", "được", "đã", "này", "đó", "một", "các", "những",
    "biểu", "loại", "vùng", "cố", "kinh",
})
KINSHIP = frozenset({
    "cha", "Cha", "mẹ", "Mẹ", "ông", "Ông", "bà", "Bà", "gái", "trai",
    "cháu", "con", "anh", "chị", "em", "vợ", "chồng",
})


def _surface(vocab, tid: int) -> str:
    return vocab.token_of(int(tid)).replace(MARK_SPACE, "").strip()


def _is_punct(vocab, tid: int) -> bool:
    s = _surface(vocab, tid)
    return (not s) or s in {"<EOS>"} or all(not ch.isalnum() for ch in s)


def _content_ok(vocab, tid: int, *, stop: set[str]) -> bool:
    s = _surface(vocab, tid)
    if not s or s in stop or s.lower() in stop:
        return False
    if s.isdigit() or len(s) <= 1:
        return False
    if all(not ch.isalnum() for ch in s):
        return False
    return True


def _lop_trai(left_surf: str | None, entity_surf: str) -> str:
    if left_surf in PERSON_LEFT:
        return "nguoi"
    if left_surf in GEO_LEFT:
        return "dia_ly"
    if left_surf is None or left_surf == "":
        if entity_surf in KINSHIP:
            return "nguoi"
        if entity_surf in GEO_LEFT:
            return "dia_ly"
        return "khac"
    return "khac"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--prop-gidf-min", type=float, default=4.5)
    ap.add_argument("--pred-top-ban", type=int, default=15)
    ap.add_argument("--min-strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pos", type=int, default=80)
    ap.add_argument("--max-neg", type=int, default=80)
    ap.add_argument("--out", default="out/probe_analogy_exec_v4.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_analogy_exec_v4")
    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)
    stop = set(STOP) | {s.lower() for s in STOP}

    la_ids = {i for i in range(n) if _surface(v, i) == "là"}
    if not la_ids:
        raise SystemExit("⛔ không tìm thấy token 'là'")

    # props[x] = {p | cạnh p→x}. Bigram d=1 (x, p) ⇒ p→x.
    props: dict[int, set[int]] = defaultdict(set)
    edge_left_cls: dict[tuple[int, int], set[str]] = defaultdict(set)
    edge_sents: dict[tuple[int, int], list[str]] = defaultdict(list)
    la_ac: dict[int, set[int]] = defaultdict(set)
    parents_a_la: dict[int, set[int]] = defaultdict(set)
    triples: list[tuple[int, int, int]] = []

    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        text = " ".join(_surface(v, t) for t in e)

        # bigram d=1: (entity, prop) ⇒ prop → entity + lop_trai
        for i in range(len(e) - 1):
            x, p = e[i], e[i + 1]
            if _is_punct(v, x) or _is_punct(v, p):
                continue
            if x in la_ids or p in la_ids:
                continue
            if not _content_ok(v, x, stop=stop) or not _content_ok(v, p, stop=stop):
                continue
            left_surf = None
            if i > 0 and not _is_punct(v, e[i - 1]):
                left_surf = _surface(v, e[i - 1])
            lc = _lop_trai(left_surf, _surface(v, x))
            props[x].add(p)
            edge_left_cls[(p, x)].add(lc)
            if len(edge_sents[(p, x)]) < 2:
                edge_sents[(p, x)].append(
                    f"[{pi}] lop={lc} trái={left_surf!r} | {text}"
                )

        for i in range(len(e) - 2):
            if e[i + 1] not in la_ids:
                continue
            b, c = e[i], e[i + 2]
            parents_a_la[b].add(pi)
            bs, cs = _surface(v, b), _surface(v, c)
            if bs in BAD_HEAD or cs in BAD_PRED or bs in BAD_PRED:
                continue
            if not _content_ok(v, b, stop=stop) or not _content_ok(v, c, stop=stop):
                continue
            ok_left = (
                i == 0
                or _is_punct(v, e[i - 1])
                or _surface(v, e[i - 1]) in CLASSIFIER
            )
            if not ok_left:
                continue
            la_ac[b].add(c)
            triples.append((pi, b, c))

    prop_ok = np.zeros(n, dtype=bool)
    for t in range(n):
        if not _content_ok(v, t, stop=stop) or t in la_ids:
            continue
        try:
            if float(ps._gidf(t)) >= args.prop_gidf_min:  # noqa: SLF001
                prop_ok[t] = True
        except Exception:
            continue
    for a in list(props):
        props[a] = {p for p in props[a] if prop_ok[p]}

    n_edges = sum(len(ps_) for ps_ in props.values())
    logger.log(
        f"triples sạch={len(triples):,} · cạnh p→x (d=1)={n_edges:,} · "
        f"prop_ok={int(prop_ok.sum()):,} ({time.time()-t0:.0f}s)"
    )

    pred_df = np.zeros(n, dtype=np.int32)
    for _h, cs in la_ac.items():
        for c in cs:
            pred_df[c] += 1
    top_ban: set[int] = set()
    for i in np.argsort(-pred_df)[: max(args.pred_top_ban * 3, 0)]:
        if pred_df[i] <= 0:
            continue
        if _surface(v, int(i))[:1].isupper():
            continue
        top_ban.add(int(i))
        if len(top_ban) >= args.pred_top_ban:
            break
    pred_hub = set(top_ban)
    for c in range(n):
        if _surface(v, c) in BAD_PRED or not _content_ok(v, c, stop=stop):
            pred_hub.add(c)
    for h in list(la_ac):
        la_ac[h] = {c for c in la_ac[h] if c not in pred_hub}
        if not la_ac[h]:
            del la_ac[h]
    triples = [(pi, b, c) for pi, b, c in triples if c not in pred_hub and b in la_ac]

    # p → heads (thực_thể nhận cạnh từ p)
    p_heads: dict[int, set[int]] = defaultdict(set)
    for head in la_ac:
        for p in props.get(head, ()):
            p_heads[p].add(head)

    def compatible(p: int, a: int, b: int) -> bool:
        return bool(
            edge_left_cls.get((p, a), set()) & edge_left_cls.get((p, b), set())
        )

    def strength(n_donors: int, p: int, c: int) -> float:
        try:
            g = float(ps._gidf(p))  # noqa: SLF001
        except Exception:
            g = 1.0
        return float(n_donors) * g / (1.0 + np.log1p(pred_df[c]))

    rng = random.Random(args.seed)
    pos_raw: list[dict] = []
    neg_cases: list[dict] = []

    for a, sp in props.items():
        if not sp:
            continue
        sa = _surface(v, a)
        if sa in BAD_HEAD or sa in stop:
            continue
        if sa.startswith("Ċ") or not sa or not any(ch.isalnum() for ch in sa):
            continue
        if not (sa[:1].isupper() or sa in KINSHIP):
            continue
        if is_structural_token(v, a):
            continue

        # c → cộng mọi cầu p (không chỉ best p)
        acc_score: dict[int, float] = defaultdict(float)
        acc_dons: dict[int, set[int]] = defaultdict(set)
        acc_bridges: dict[int, list[tuple[int, list[int], float]]] = defaultdict(list)
        acc_left: dict[int, set[str]] = defaultdict(set)
        weak: dict[int, set[int]] = defaultdict(set)
        for p in sp:
            if not prop_ok[p]:
                continue
            cd: dict[int, set[int]] = defaultdict(set)
            for b in p_heads.get(p, ()):
                if b == a or not compatible(p, a, b):
                    continue
                for c in la_ac.get(b, ()):
                    if c == a or c in pred_hub:
                        continue
                    cd[c].add(b)
                    weak[c].add(b)
            for c, dons in cd.items():
                if len(dons) < args.min_support:
                    continue
                sc = strength(len(dons), p, c)
                shared = edge_left_cls[(p, a)] & set.union(
                    *(edge_left_cls[(p, d)] for d in dons)
                )
                acc_score[c] += sc
                acc_dons[c] |= dons
                acc_bridges[c].append((p, sorted(dons), sc))
                acc_left[c] |= shared
        for c, sc in acc_score.items():
            if c in la_ac.get(a, ()):
                continue
            if sc < args.min_strength:
                continue
            dons = sorted(acc_dons[c])
            bridges = sorted(acc_bridges[c], key=lambda t: -t[2])
            n_kin = sum(1 for d in dons if _surface(v, d) in KINSHIP)
            pred_cap = _surface(v, c)[:1].isupper()
            if not (n_kin >= 2 or (pred_cap and len(dons) >= 2)):
                continue
            p0 = bridges[0][0]
            evid = []
            for p_i, d_i, _ in bridges[:3]:
                for d in d_i[:2]:
                    evid.extend(edge_sents.get((p_i, d), [])[:1])
                evid.extend(edge_sents.get((p_i, a), [])[:1])
            pos_raw.append({
                "a": a, "c": c, "donors": dons, "bridge_prop": p0,
                "bridges": [_surface(v, p) for p, _, _ in bridges],
                "n_bridges": len(bridges),
                "n_donors": len(dons), "strength": sc, "n_kinship": n_kin,
                "left_cls": sorted(acc_left[c]),
                "edge_sents": evid[:6],
            })
        supported_cs = set(acc_score)
        for c, dons in weak.items():
            if c in la_ac.get(a, ()) or c in supported_cs or c in pred_hub:
                continue
            if len(dons) == 1:
                neg_cases.append({
                    "a": a, "c": c, "donors": sorted(dons), "bridge_prop": None,
                    "bridges": [], "n_bridges": 0,
                    "n_donors": 1, "strength": 0.0, "left_cls": [],
                    "edge_sents": [],
                })

    pos_raw.sort(key=lambda x: (-x["n_bridges"], -x["strength"], -x["n_kinship"]))
    per_a: dict[int, int] = defaultdict(int)
    per_c: dict[int, int] = defaultdict(int)
    seen_surf: set[str] = set()
    pos_cases: list[dict] = []
    for case in pos_raw:
        sa = _surface(v, case["a"]).lower()
        if sa in seen_surf or per_a[case["a"]] >= 1 or per_c[case["c"]] >= 8:
            continue
        pos_cases.append(case)
        per_a[case["a"]] += 1
        per_c[case["c"]] += 1
        seen_surf.add(sa)
        if len(pos_cases) >= args.max_pos:
            break

    rng.shuffle(neg_cases)
    neg_cases = neg_cases[: args.max_neg]
    logger.log(
        f"ca dương={len(pos_cases)} (raw {len(pos_raw)}) âm={len(neg_cases)} "
        f"({time.time()-t0:.0f}s)"
    )

    b_to_c_all: dict[int, set[int]] = defaultdict(set)
    b_to_c_pi: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pi, b, c in triples:
        b_to_c_all[b].add(c)
        b_to_c_pi[b].append((pi, c))

    E = np.asarray(state.embedding.matrix, dtype=np.float64)
    if E.shape[0] < n:
        E = np.pad(E, ((0, n - E.shape[0]), (0, 0)))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    def b_to_c_excluding(a: int) -> dict[int, set[int]]:
        ban = parents_a_la.get(a, ())
        if not ban:
            return {b: set(cs) for b, cs in b_to_c_all.items()}
        out: dict[int, set[int]] = defaultdict(set)
        for b, lst in b_to_c_pi.items():
            for pi, c in lst:
                if pi not in ban:
                    out[b].add(c)
        return out

    def write_bridge_edges(a: int, b2c: dict[int, set[int]]) -> dict[int, tuple[float, dict]]:
        """score(c) = Σ_p strength — nhiều cầu → trọng số cao hơn."""
        acc_score: dict[int, float] = defaultdict(float)
        acc_dons: dict[int, set[int]] = defaultdict(set)
        acc_bridges: dict[int, list[tuple[int, float]]] = defaultdict(list)
        acc_left: dict[int, set[str]] = defaultdict(set)
        for p in props.get(a, ()):
            if not prop_ok[p]:
                continue
            cd: dict[int, set[int]] = defaultdict(set)
            for b in p_heads.get(p, ()):
                if b == a or not compatible(p, a, b):
                    continue
                for c in b2c.get(b, ()):
                    if c != a and c not in pred_hub:
                        cd[c].add(b)
            for c, dons in cd.items():
                if len(dons) < args.min_support:
                    continue
                sc = strength(len(dons), p, c)
                shared = edge_left_cls[(p, a)] & set.union(
                    *(edge_left_cls[(p, d)] for d in dons)
                )
                acc_score[c] += sc
                acc_dons[c] |= dons
                acc_bridges[c].append((p, sc))
                acc_left[c] |= shared
        edges: dict[int, tuple[float, dict]] = {}
        for c, sc in acc_score.items():
            bridges = sorted(acc_bridges[c], key=lambda t: -t[1])
            edges[c] = (sc, {
                "prop": bridges[0][0],
                "bridges": [p for p, _ in bridges],
                "n_bridges": len(bridges),
                "donors": sorted(acc_dons[c])[:6],
                "left_cls": sorted(acc_left[c]),
            })
        return edges

    def fill_bridge(a: int, b2c: dict[int, set[int]]) -> tuple[int | None, dict]:
        edges = write_bridge_edges(a, b2c)
        if not edges:
            return None, {}
        best = max(edges.items(), key=lambda kv: kv[1][0])[0]
        return best, edges[best][1]

    def fill_embedding(a: int, b2c: dict[int, set[int]], top_b: int = 30) -> int | None:
        fa = props.get(a, set())
        cand = [b for b in la_ac if b != a and props.get(b)]
        base = sorted(
            ((len(fa & props.get(b, set())), b) for b in cand), reverse=True,
        )[: top_b * 5]
        if not base:
            return None
        neigh = [
            b for _s, b in sorted(
                ((float(En[a] @ En[b]), b) for _o, b in base), reverse=True,
            )[:top_b]
        ]
        sc: dict[int, float] = defaultdict(float)
        for b in neigh:
            s = float(En[a] @ En[b])
            if s <= 0:
                continue
            for c in b2c.get(b, ()):
                if c != a and c not in pred_hub:
                    sc[c] += s
        return max(sc, key=lambda c: sc[c]) if sc else None

    def fill_random(a: int, b2c: dict[int, set[int]]) -> int | None:
        pool: set[int] = set()
        for p in props.get(a, ()):
            for b in p_heads.get(p, ()):
                if b != a and compatible(p, a, b):
                    pool |= {c for c in b2c.get(b, set()) if c not in pred_hub}
        pool.discard(a)
        return rng.choice(sorted(pool)) if pool else None

    def run_split(cases: list[dict]) -> dict:
        out = {m: {"n_fill": 0, "hit": 0, "examples": []}
               for m in ("bridge", "embedding", "ngau_nhien")}
        for case in cases:
            a, c_star = case["a"], case["c"]
            b2c = b_to_c_excluding(a)
            pred_b, meta_b = fill_bridge(a, b2c)
            preds = {
                "bridge": pred_b,
                "embedding": fill_embedding(a, b2c),
                "ngau_nhien": fill_random(a, b2c),
            }
            for mode, pred in preds.items():
                if pred is None:
                    continue
                out[mode]["n_fill"] += 1
                ok = pred == c_star
                if ok:
                    out[mode]["hit"] += 1
                if len(out[mode]["examples"]) < 15:
                    bp = case.get("bridge_prop")
                    left_cls = case.get("left_cls") or []
                    bridges = case.get("bridges") or []
                    n_br = case.get("n_bridges", len(bridges))
                    if mode == "bridge" and meta_b.get("prop") is not None:
                        bp = meta_b["prop"]
                        left_cls = meta_b.get("left_cls") or left_cls
                        br_ids = meta_b.get("bridges") or []
                        bridges = [_surface(v, p) for p in br_ids]
                        n_br = meta_b.get("n_bridges", len(bridges))
                    out[mode]["examples"].append({
                        "prompt": f"{_surface(v, a)} là",
                        "expect": f"{_surface(v, a)} là {_surface(v, c_star)}",
                        "actual": (
                            f"{_surface(v, a)} là {_surface(v, pred)}"
                            if pred is not None else None
                        ),
                        "ok": ok,
                        "donors": [_surface(v, d) for d in case["donors"][:5]],
                        "bridge_prop": _surface(v, bp) if bp is not None else None,
                        "bridges": bridges,
                        "n_bridges": n_br,
                        "left_cls": left_cls,
                        "strength": round(case.get("strength", 0), 2),
                        "edge_sents": case.get("edge_sents", [])[:4],
                    })
        return out

    pos_r = run_split(pos_cases)
    neg_r = run_split(neg_cases)

    def show(title, block):
        print(f"\n=== {title} ===")
        print(f"{'bộ':<14}{'n_điền':>8}{'hit':>6}{'rate':>8}")
        rates = {}
        for m in ("bridge", "embedding", "ngau_nhien"):
            s = block[m]
            r = s["hit"] / max(s["n_fill"], 1)
            rates[m] = r
            print(f"{m:<14}{s['n_fill']:>8}{s['hit']:>6}{r:>7.1%}")
        return rates

    pos_rates = show("DƯƠNG (P98F+ · Σ cầu · lop_trai)", pos_r)
    neg_rates = show("ÂM (1 donor · FP)", neg_r)

    he_n = pos_r["bridge"]["n_fill"]
    he_hit = pos_rates["bridge"]
    fp = neg_rates["bridge"] if neg_r["bridge"]["n_fill"] else 0.0
    creative = all(
        c not in la_ac.get(a, ())
        for a, c in ((x["a"], x["c"]) for x in pos_cases)
    )
    g1 = he_n >= GATE_N_POS
    g2 = he_hit >= GATE_HIT
    g3 = fp <= GATE_FP
    g4 = creative
    ok_gate = g1 and g2 and g3 and g4

    prompts_pos = {_surface(v, c["a"]) for c in pos_cases}
    q1_dead = ("Hồng" not in prompts_pos) and ("Sông" not in prompts_pos)
    bad_q1 = [
        ex for ex in pos_r["bridge"]["examples"]
        if ex["prompt"].startswith(("Hồng ", "Sông "))
        and ex.get("bridge_prop") == "nuôi"
    ]
    q1 = q1_dead and len(bad_q1) == 0

    print(f"\nCỔNG G-exec-v4 (P98F):")
    print(f"  (1) n≥{GATE_N_POS}: {he_n} {'✅' if g1 else '❌'}")
    print(f"  (2) hit≥{GATE_HIT:.0%}: {he_hit:.1%} {'✅' if g2 else '❌'}")
    print(f"  (3) FP≤{GATE_FP:.0%}: {fp:.1%} {'✅' if g3 else '❌'}")
    print(f"  (4) creative: {'✅' if g4 else '❌'}")
    print(f"  Q1 Hồng/Sông chết: {'✅' if q1 else '❌'}")
    print(f"⇒ {'ĐẠT' if ok_gate else 'TRƯỢT'}")

    print("\n| prompt | mong đợi | thực tế | donors | cầu(s) | n_cầu |")
    print("|---|---|---|---|---|---:|")
    for ex in pos_r["bridge"]["examples"]:
        br = ",".join(ex.get("bridges") or ([ex["bridge_prop"]] if ex.get("bridge_prop") else []))
        print(
            f"| `{ex['prompt']}` | `{ex['expect']}` | `{ex['actual']}` | "
            f"{', '.join(ex['donors'])} | `{br}` | {ex.get('n_bridges', 1)} |"
        )

    n_multi = sum(1 for c in pos_cases if c.get("n_bridges", 1) >= 2)
    print(f"\nca dương có ≥2 cầu: {n_multi}/{len(pos_cases)}")

    payload = {
        "n_pos": len(pos_cases), "n_neg": len(neg_cases),
        "n_pos_multi_bridge": n_multi,
        "filters": {
            "protocol": "creative_d1_left_cls_sum_bridges",
            "bigram_d": 1,
            "edge_dir": "prop→entity",
            "left_cls_match": True,
            "score": "sum_over_bridges",
            "prop_gidf_min": args.prop_gidf_min,
            "min_strength": args.min_strength,
        },
        "q1": {"hong_song_dead": q1},
        "pos": {m: {"n_fill": pos_r[m]["n_fill"], "hit": pos_r[m]["hit"],
                    "rate": pos_rates[m], "examples": pos_r[m]["examples"]}
                for m in pos_r},
        "neg": {m: {"n_fill": neg_r[m]["n_fill"], "hit": neg_r[m]["hit"],
                    "rate": neg_rates[m], "examples": neg_r[m]["examples"]}
                for m in neg_r},
        "gate": {
            "g1": g1, "g2": g2, "g3": g3, "g4": g4, "q1": q1, "pass": ok_gate,
        },
        "seconds": round(time.time() - t0, 1),
    }
    (_ROOT / args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
