#!/usr/bin/env python
"""P98A (TK_046) — đồ chơi: viết cạnh Hebbian → thi hành khung `X là Y`.

Trả lời §17: cạnh suy ra có thành câu đọc được không — đúng ví dụ tác giả.

Cơ chế viết (kiến trúc tác giả, đồ chơi không có E):
  khỉ chia thuộc tính với chó/mèo (≥2 bạn) → thừa hưởng pred trong khung `là`
  bàn chỉ chia `nâu` với chó (1 bạn) → KHÔNG thừa hưởng

Thi hành: prompt `… là` + điền Y từ cạnh đã viết — không AR, không chép cha.

Cổng: RUNBOOK_TK_046_P98.md §0.A
"""
from __future__ import annotations

from collections import defaultdict

T1 = [
    "con mèo là động vật",
    "con chó là động vật",
    "con mèo nghịch ngợm",
    "con chó nghịch ngợm",
    "con khỉ nghịch ngợm",
]
T2_EXTRA = [
    "cái bàn màu nâu",
    "con chó màu nâu",
]

STOP = frozenset({"con", "cái", "là", "màu"})


def words(s: str) -> list[str]:
    return s.split()


def build(sents: list[str]):
    co: dict[str, set[str]] = defaultdict(set)
    la_pred: dict[str, set[str]] = defaultdict(set)  # head → preds after "là"
    for s in sents:
        ws = words(s)
        for i, a in enumerate(ws):
            for b in ws:
                if a != b:
                    co[a].add(b)
        if " là " in f" {s} ":
            left, right = s.split(" là ", 1)
            head = words(left)[-1]
            la_pred[head].add(right.strip())
    return co, la_pred


def props(co: dict[str, set[str]], tok: str) -> set[str]:
    return {t for t in co.get(tok, ()) if t not in STOP}


def write_la_edges(
    co: dict[str, set[str]],
    la_pred: dict[str, set[str]],
    seeds: list[str],
    *,
    min_support: int = 2,
) -> dict[str, dict[str, float]]:
    """Hebbian lớp: ≥ min_support head khác chia thuộc tính với seed và mang cùng pred."""
    edges: dict[str, dict[str, float]] = defaultdict(dict)
    heads = sorted(set(la_pred) | set(seeds))
    for seed in seeds:
        sp = props(co, seed)
        if not sp:
            continue
        # pred → list donors
        donors: dict[str, list[str]] = defaultdict(list)
        for h in heads:
            if h == seed:
                continue
            hp = props(co, h)
            shared = sp & hp
            if not shared:
                continue
            for pred in la_pred.get(h, ()):
                if pred in la_pred.get(seed, ()):
                    continue
                donors[pred].append(h)
        for pred, hs in donors.items():
            uniq = sorted(set(hs))
            if len(uniq) >= min_support:
                edges[seed][pred] = float(len(uniq))
    return edges


def execute_la(prompt: str, edges: dict[str, dict[str, float]]) -> str:
    p = prompt.strip()
    if not p.endswith("là"):
        raise ValueError("prompt phải kết thúc bằng 'là'")
    left = p[: -len("là")].strip()
    head = words(left)[-1]
    cands = edges.get(head) or {}
    if not cands:
        return p
    pred = max(cands.items(), key=lambda kv: kv[1])[0]
    return f"{left} là {pred}".strip()


def run(sents: list[str], label: str) -> dict:
    co, la_pred = build(sents)
    edges = write_la_edges(co, la_pred, seeds=["khỉ", "bàn", "mèo", "chó"])
    print(f"\n=== {label} ===")
    print("  pred `là` attested:", {k: sorted(v) for k, v in sorted(la_pred.items())})
    print("  cạnh viết (head → pred):")
    any_e = False
    for h, preds in sorted(edges.items()):
        for pred, sc in sorted(preds.items(), key=lambda x: -x[1]):
            print(f"    {h} → {pred!r}  (support={sc:.0f})")
            any_e = True
    if not any_e:
        print("    (không có)")

    out_khi = execute_la("con khỉ là", edges)
    out_ban = execute_la("cái bàn là", edges)
    print(f"  thi hành 'con khỉ là' → {out_khi!r}")
    print(f"  thi hành 'cái bàn là' → {out_ban!r}")

    a1 = ("động vật" in edges.get("khỉ", {})
          and "động vật" not in edges.get("bàn", {}))
    a2 = "động vật" in out_khi and out_khi != "con khỉ là"
    a3 = "động vật" not in out_ban
    print(f"  A1 viết cạnh tách lớp: {'✅' if a1 else '❌'}")
    print(f"  A2 khỉ → câu có động vật: {'✅' if a2 else '❌'}")
    print(f"  A3 bàn không ra động vật: {'✅' if a3 else '❌'}")
    return {
        "edges": {h: dict(p) for h, p in edges.items()},
        "out_khi": out_khi,
        "out_ban": out_ban,
        "A1": a1, "A2": a2, "A3": a3,
        "pass": bool(a1 and a2 and a3),
    }


def main() -> int:
    print("P98A — Hebbian viết cạnh → thi hành khung X là Y (đồ chơi tác giả)")
    for s in T1:
        print(f"  train: {s}")
    r1 = run(T1, "T1 (không bẫy bàn)")
    print("\n  thêm bẫy:")
    for s in T2_EXTRA:
        print(f"  train: {s}")
    r2 = run(T1 + T2_EXTRA, "T2 (+ bàn/nâu)")

    ok = r2["pass"] and r1["A2"]
    print(f"\n⇒ cổng A (T2): {'ĐẠT' if r2['pass'] else 'TRƯỢT'}"
          f" · T1 A2 sanity: {'✅' if r1['A2'] else '❌'}"
          f"  ⇒ {'ĐI TIẾP mini' if ok else 'DỪNG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
