#!/usr/bin/env python
"""P98G (TK_046) — cầu CỤM (L=1..2) + lop_trai + Σ cầu.

Đo cổng / try-prompts. Luật sinh ship nằm ở
`nn_aivie.hca.analogy_bridge_gen` (`aivie generate --mode analogy_la`).

Cổng: RUNBOOK_TK_046_P98G.md · ship: RUNBOOK_ANALOGY_P98G_SHIP.md
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

PropKey = tuple[int, ...]

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


def _prop_label(vocab, key: PropKey) -> str:
    return "+".join(_surface(vocab, t) for t in key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--prop-gidf-min", type=float, default=4.5)
    ap.add_argument("--compound-bonus", type=float, default=2.5,
                    help="nhân thêm cho cầu L=2")
    ap.add_argument("--pred-top-ban", type=int, default=15)
    ap.add_argument("--min-strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pos", type=int, default=80)
    ap.add_argument("--max-neg", type=int, default=80)
    ap.add_argument("--try-prompts", default="",
                    help="vd: 'anh là,mẹ là,Thái là' — điền thử rồi thoát phần bảng ngắn")
    ap.add_argument("--out", default="out/probe_analogy_exec_v5.json")
    # A/B scaffold từ vựng — bao nhiêu phần kết quả đến từ bảng từ viết tay?
    ap.add_argument(
        "--scaffold", choices=("full", "off", "gidf"), default="full",
        help="full=hiện trạng · off=bảng rỗng · gidf=thay bằng ngưỡng gidf từ đồ thị",
    )
    ap.add_argument(
        "--case-scaffold", choices=("keep", "off"), default="keep",
        help="keep=chọn ca như cũ (KINSHIP định tập ca) · off=bỏ KINSHIP khỏi CHỌN CA",
    )
    ap.add_argument("--stop-gidf-max", type=float, default=4.5,
                    help="scaffold=gidf: surface có gidf < ngưỡng ⇒ coi là hư từ")
    args = ap.parse_args()

    logger = get_run_logger("probe_analogy_exec_v5")
    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    n = len(v)

    # --- A/B scaffold: rebind bảng từ TRƯỚC khi dựng index ---------------
    global STOP, BAD_HEAD, BAD_PRED, KINSHIP, GEO_LEFT, PERSON_LEFT, CLASSIFIER
    _scaffold_meta: dict[str, object] = {"scaffold": args.scaffold,
                                         "case_scaffold": args.case_scaffold}
    if args.scaffold == "off":
        STOP = frozenset()
        BAD_HEAD = frozenset()
        BAD_PRED = frozenset()
        KINSHIP = frozenset()
        GEO_LEFT = frozenset()
        PERSON_LEFT = frozenset()
        CLASSIFIER = frozenset()
    elif args.scaffold == "gidf":
        # Hư từ = token df cao ⇒ gidf thấp (§0.4.2: thống kê tự dìm chúng).
        # Cùng call-site, nội dung DẪN XUẤT TỪ ĐỒ THỊ thay vì viết tay.
        hu_tu = {
            _surface(v, t)
            for t in range(n)
            if ps._gidf(t) < args.stop_gidf_max  # noqa: SLF001
        }
        hu_tu.discard("")
        STOP = frozenset(hu_tu)
        BAD_HEAD = frozenset(hu_tu)
        BAD_PRED = frozenset(hu_tu)
        # Lớp trái là TRI THỨC THẾ GIỚI, không dẫn xuất được bằng gidf → bỏ hẳn.
        KINSHIP = frozenset()
        GEO_LEFT = frozenset()
        PERSON_LEFT = frozenset()
        _scaffold_meta["n_hu_tu_gidf"] = len(hu_tu)
    if args.case_scaffold == "off":
        # KINSHIP dùng ở CHỌN CA (dòng ~385/397) — tách riêng để thấy nó
        # định nghĩa tập ca chứ không chỉ lọc cơ chế.
        KINSHIP = frozenset()
    _scaffold_meta["n_STOP"] = len(STOP)
    _scaffold_meta["n_BAD_PRED"] = len(BAD_PRED)
    _scaffold_meta["n_KINSHIP"] = len(KINSHIP)
    logger.log(f"scaffold={args.scaffold} case={args.case_scaffold} · {_scaffold_meta}")

    stop = set(STOP) | {s.lower() for s in STOP}

    la_ids = {i for i in range(n) if _surface(v, i) == "là"}
    if not la_ids:
        raise SystemExit("⛔ không tìm thấy token 'là'")

    props: dict[int, set[PropKey]] = defaultdict(set)
    edge_left_cls: dict[tuple[PropKey, int], set[str]] = defaultdict(set)
    edge_sents: dict[tuple[PropKey, int], list[str]] = defaultdict(list)
    la_ac: dict[int, set[int]] = defaultdict(set)
    parents_a_la: dict[int, set[int]] = defaultdict(set)
    triples: list[tuple[int, int, int]] = []

    def add_edge(x: int, key: PropKey, lc: str, pi: int, text: str) -> None:
        props[x].add(key)
        edge_left_cls[(key, x)].add(lc)
        if len(edge_sents[(key, x)]) < 2:
            edge_sents[(key, x)].append(
                f"[{pi}] L={len(key)} lop={lc} cầu={_prop_label(v, key)!r} | {text}"
            )

    for pi, rec in enumerate(ps.parents):
        e = [int(t) for t in rec.emit_ids]
        text = " ".join(_surface(v, t) for t in e)
        for i in range(len(e) - 1):
            x = e[i]
            if _is_punct(v, x) or x in la_ids or not _content_ok(v, x, stop=stop):
                continue
            left_surf = None
            if i > 0 and not _is_punct(v, e[i - 1]):
                left_surf = _surface(v, e[i - 1])
            lc = _lop_trai(left_surf, _surface(v, x))

            p1 = e[i + 1]
            if (
                not _is_punct(v, p1)
                and p1 not in la_ids
                and _content_ok(v, p1, stop=stop)
            ):
                add_edge(x, (p1,), lc, pi, text)
                # L=2 cụm
                if i + 2 < len(e):
                    p2 = e[i + 2]
                    if (
                        not _is_punct(v, p2)
                        and p2 not in la_ids
                        and _content_ok(v, p2, stop=stop)
                    ):
                        add_edge(x, (p1, p2), lc, pi, text)

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

    tok_ok = np.zeros(n, dtype=bool)
    for t in range(n):
        if not _content_ok(v, t, stop=stop) or t in la_ids:
            continue
        try:
            if float(ps._gidf(t)) >= args.prop_gidf_min:  # noqa: SLF001
                tok_ok[t] = True
        except Exception:
            continue

    def key_ok(key: PropKey) -> bool:
        return all(tok_ok[t] for t in key)

    for a in list(props):
        props[a] = {k for k in props[a] if key_ok(k)}

    n_l1 = sum(1 for s in props.values() for k in s if len(k) == 1)
    n_l2 = sum(1 for s in props.values() for k in s if len(k) == 2)
    logger.log(
        f"triples={len(triples):,} · cạnh L1={n_l1:,} L2={n_l2:,} ({time.time()-t0:.0f}s)"
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

    p_heads: dict[PropKey, set[int]] = defaultdict(set)
    for head in la_ac:
        for key in props.get(head, ()):
            p_heads[key].add(head)

    def compatible(key: PropKey, a: int, b: int) -> bool:
        return bool(
            edge_left_cls.get((key, a), set()) & edge_left_cls.get((key, b), set())
        )

    def strength(n_donors: int, key: PropKey, c: int) -> float:
        g = 1.0
        for t in key:
            try:
                g *= float(ps._gidf(t))  # noqa: SLF001
            except Exception:
                g *= 1.0
        # geometric-ish: root
        g = float(g ** (1.0 / max(len(key), 1)))
        bonus = args.compound_bonus if len(key) == 2 else 1.0
        return float(n_donors) * g * bonus / (1.0 + np.log1p(pred_df[c]))

    rng = random.Random(args.seed)

    def aggregate_for(a: int, b2c: dict[int, set[int]] | None = None):
        """Trả score/donors/bridges cho mọi c từ a (creative filter nếu b2c=None dùng la_ac)."""
        acc_score: dict[int, float] = defaultdict(float)
        acc_dons: dict[int, set[int]] = defaultdict(set)
        acc_bridges: dict[int, list[tuple[PropKey, list[int], float]]] = defaultdict(list)
        acc_left: dict[int, set[str]] = defaultdict(set)
        weak: dict[int, set[int]] = defaultdict(set)
        for key in props.get(a, ()):
            if not key_ok(key):
                continue
            cd: dict[int, set[int]] = defaultdict(set)
            for b in p_heads.get(key, ()):
                if b == a or not compatible(key, a, b):
                    continue
                src = b2c.get(b, ()) if b2c is not None else la_ac.get(b, ())
                for c in src:
                    if c == a or c in pred_hub:
                        continue
                    cd[c].add(b)
                    weak[c].add(b)
            for c, dons in cd.items():
                if len(dons) < args.min_support:
                    continue
                sc = strength(len(dons), key, c)
                shared = edge_left_cls[(key, a)] & set.union(
                    *(edge_left_cls[(key, d)] for d in dons)
                )
                acc_score[c] += sc
                acc_dons[c] |= dons
                acc_bridges[c].append((key, sorted(dons), sc))
                acc_left[c] |= shared
        return acc_score, acc_dons, acc_bridges, acc_left, weak

    # --- try prompts early path after graph built ---
    def resolve_head(word: str) -> int | None:
        # Cùng surface có thể nhiều token-id; chọn id có props mạnh nhất (vd anh=2149 vs 1550).
        cands = [i for i in range(n) if _surface(v, i) == word and props.get(i)]
        if cands:
            return int(max(cands, key=lambda i: len(props[i])))
        cands = [i for i in range(n) if _surface(v, i) == word]
        return int(cands[0]) if cands else None

    if args.try_prompts.strip():
        print("\n=== TRY PROMPTS (cầu cụm L=1..2) ===")
        b_to_c_all: dict[int, set[int]] = defaultdict(set)
        for _pi, b, c in triples:
            b_to_c_all[b].add(c)
        for raw in args.try_prompts.split(","):
            raw = raw.strip()
            if not raw:
                continue
            phrase = raw[: -len("là")].strip() if raw.endswith("là") else raw.strip()
            parts = phrase.split()
            a = None
            head_w = phrase
            # thử cụm đầy đủ rồi từng token cuối (vd "con khỉ" → khỉ)
            for cand in ([phrase] if phrase else []) + list(reversed(parts)):
                if not cand:
                    continue
                a = resolve_head(cand)
                if a is not None and props.get(a):
                    head_w = cand
                    break
                if a is not None and head_w == phrase:
                    head_w = cand
            if a is None and parts:
                head_w = parts[-1]
                a = resolve_head(head_w)
            print(f"\n## prompt: {raw!r} (head={head_w!r} id={a})")
            if a is None:
                print("  ⛔ không resolve được head")
                continue
            acc_score, acc_dons, acc_bridges, acc_left, _ = aggregate_for(a, b_to_c_all)
            if not acc_score:
                print("  (không điền được)")
                continue
            ranked = sorted(acc_score.items(), key=lambda kv: -kv[1])[:5]
            for c, sc in ranked:
                bridges = sorted(acc_bridges[c], key=lambda t: -t[2])
                labels = [_prop_label(v, k) for k, _, _ in bridges]
                dons = [_surface(v, d) for d in sorted(acc_dons[c])[:6]]
                print(
                    f"  → {_surface(v, a)} là {_surface(v, c)}"
                    f"  score={sc:.2f}  n_cầu={len(bridges)}  cầu={labels}"
                    f"  donors={dons}  lop={sorted(acc_left[c])}"
                )
                for key, d_list, sc_i in bridges[:3]:
                    for line in edge_sents.get((key, a), [])[:1]:
                        print(f"     seed: {line}")
                    for d in d_list[:2]:
                        for line in edge_sents.get((key, d), [])[:1]:
                            print(f"     donor: {line}")
        # vẫn ghi json ngắn
        payload = {"try_prompts": args.try_prompts, "seconds": round(time.time() - t0, 1)}
        (_ROOT / args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.log(f"try-prompts xong → {args.out}")
        return

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
        acc_score, acc_dons, acc_bridges, acc_left, weak = aggregate_for(a, None)
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
            n_l2b = sum(1 for k, _, _ in bridges if len(k) == 2)
            evid = []
            for key, d_i, _ in bridges[:3]:
                evid.extend(edge_sents.get((key, a), [])[:1])
                for d in d_i[:1]:
                    evid.extend(edge_sents.get((key, d), [])[:1])
            pos_raw.append({
                "a": a, "c": c, "donors": dons,
                "bridge_prop": bridges[0][0],
                "bridges": [k for k, _, _ in bridges],
                "n_bridges": len(bridges),
                "n_l2": n_l2b,
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
                    "a": a, "c": c, "donors": sorted(dons),
                    "bridge_prop": None, "bridges": [], "n_bridges": 0, "n_l2": 0,
                    "n_donors": 1, "strength": 0.0, "left_cls": [], "edge_sents": [],
                })

    pos_raw.sort(key=lambda x: (-x["n_l2"], -x["n_bridges"], -x["strength"]))
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
    logger.log(f"ca dương={len(pos_cases)} âm={len(neg_cases)} ({time.time()-t0:.0f}s)")

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

    def fill_bridge(a: int, b2c: dict[int, set[int]]):
        acc_score, acc_dons, acc_bridges, acc_left, _ = aggregate_for(a, b2c)
        if not acc_score:
            return None, {}
        best = max(acc_score.items(), key=lambda kv: kv[1])[0]
        bridges = sorted(acc_bridges[best], key=lambda t: -t[2])
        return best, {
            "prop": bridges[0][0],
            "bridges": [k for k, _, _ in bridges],
            "n_bridges": len(bridges),
            "donors": sorted(acc_dons[best])[:6],
            "left_cls": sorted(acc_left[best]),
        }

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
        for key in props.get(a, ()):
            for b in p_heads.get(key, ()):
                if b != a and compatible(key, a, b):
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
                    bridges = case.get("bridges") or []
                    n_br = case.get("n_bridges", len(bridges))
                    labels = [_prop_label(v, k) for k in bridges]
                    if mode == "bridge" and meta_b.get("bridges"):
                        bridges = meta_b["bridges"]
                        labels = [_prop_label(v, k) for k in bridges]
                        n_br = meta_b.get("n_bridges", len(bridges))
                    out[mode]["examples"].append({
                        "prompt": f"{_surface(v, a)} là",
                        "expect": f"{_surface(v, a)} là {_surface(v, c_star)}",
                        "actual": f"{_surface(v, a)} là {_surface(v, pred)}",
                        "ok": ok,
                        "donors": [_surface(v, d) for d in case["donors"][:5]],
                        "bridges": labels,
                        "n_bridges": n_br,
                        "n_l2": sum(1 for k in bridges if len(k) == 2),
                        "left_cls": case.get("left_cls") or [],
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

    pos_rates = show("DƯƠNG (P98G · cầu cụm L=1..2)", pos_r)
    neg_rates = show("ÂM (1 donor · FP)", neg_r)

    he_n = pos_r["bridge"]["n_fill"]
    he_hit = pos_rates["bridge"]
    fp = neg_rates["bridge"] if neg_r["bridge"]["n_fill"] else 0.0
    creative = all(c not in la_ac.get(a, ()) for a, c in ((x["a"], x["c"]) for x in pos_cases))
    g1, g2, g3, g4 = he_n >= GATE_N_POS, he_hit >= GATE_HIT, fp <= GATE_FP, creative
    ok_gate = g1 and g2 and g3 and g4
    prompts_pos = {_surface(v, c["a"]) for c in pos_cases}
    q1 = ("Hồng" not in prompts_pos) and ("Sông" not in prompts_pos)
    n_multi = sum(1 for c in pos_cases if c.get("n_bridges", 1) >= 2)
    n_with_l2 = sum(1 for c in pos_cases if c.get("n_l2", 0) >= 1)

    print(f"\nCỔNG G-exec-v5 (P98G cầu cụm):")
    print(f"  (1) n≥{GATE_N_POS}: {he_n} {'✅' if g1 else '❌'}")
    print(f"  (2) hit≥{GATE_HIT:.0%}: {he_hit:.1%} {'✅' if g2 else '❌'}")
    print(f"  (3) FP≤{GATE_FP:.0%}: {fp:.1%} {'✅' if g3 else '❌'}")
    print(f"  (4) creative: {'✅' if g4 else '❌'}")
    print(f"  Q1 Hồng/Sông chết: {'✅' if q1 else '❌'}")
    print(f"  ca ≥2 cầu: {n_multi}/{len(pos_cases)} · có L2: {n_with_l2}/{len(pos_cases)}")
    print(f"⇒ {'ĐẠT' if ok_gate else 'TRƯỢT'}")

    print("\n| prompt | mong đợi | thực tế | donors | cầu(s) | n_cầu |")
    print("|---|---|---|---|---|---:|")
    for ex in pos_r["bridge"]["examples"]:
        br = ",".join(ex.get("bridges") or [])
        print(
            f"| `{ex['prompt']}` | `{ex['expect']}` | `{ex['actual']}` | "
            f"{', '.join(ex['donors'])} | `{br}` | {ex.get('n_bridges', 1)} |"
        )

    payload = {
        "n_pos": len(pos_cases), "n_neg": len(neg_cases),
        "n_pos_multi_bridge": n_multi, "n_pos_with_l2": n_with_l2,
        "filters": {
            "protocol": "creative_compound_prop_L1L2",
            "compound_bonus": args.compound_bonus,
            "left_cls_match": True,
            "score": "sum_over_bridges",
        },
        "q1": {"hong_song_dead": q1},
        "pos": {m: {"n_fill": pos_r[m]["n_fill"], "hit": pos_r[m]["hit"],
                    "rate": pos_rates[m], "examples": pos_r[m]["examples"]}
                for m in pos_r},
        "neg": {m: {"n_fill": neg_r[m]["n_fill"], "hit": neg_r[m]["hit"],
                    "rate": neg_rates[m], "examples": neg_r[m]["examples"]}
                for m in neg_r},
        "gate": {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "q1": q1, "pass": ok_gate},
        "scaffold": _scaffold_meta,
        "seconds": round(time.time() - t0, 1),
    }
    (_ROOT / args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
