#!/usr/bin/env python3
"""Export audit TSV/JSON từ checkpoint + sanity checks."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.frame_bracket_ref import decode_endpoint_label
from nn_aivie.core.seg_graph import CLUSTER_OFFSET
from nn_aivie.core.vocab import is_surface_token
from nn_aivie.training.checkpoint_bake import bake_derived_indices
from nn_aivie.training.frame_bracket import mine_frame_brackets


_BOILERPLATE_PHRASES = (
    "Ví dụ", ": Tiết", "Tiết này", "mẫu câu", "Em đọc", "Bài 1",
    ": Lớp", "Tiếng Việt", "Toán :", "Địa lý :",
)


def _tok(v, i: int) -> str:
    try:
        return v.token_of(int(i))
    except Exception:
        return f"<{i}>"


def _parent_text(ps, v, pi: int, n: int = 100) -> str:
    s = " ".join(_tok(v, t) for t in ps.parents[pi].source_ids)
    return s[:n].replace("\t", " ")


def _cluster_usage(ps) -> dict[int, set[int]]:
    usage: dict[int, set[int]] = defaultdict(set)
    for pi, rec in enumerate(ps.parents):
        for s in rec.slot_chain:
            if s.kind == "cluster":
                usage[s.id - CLUSTER_OFFSET].add(pi)
    return usage


def _isg_flag(phrase: str, parent_df: int, live: bool) -> str:
    if not live:
        return "orphan"
    if not phrase or phrase[0] in ".,:;«»" or phrase[-1] in ".,:;«»":
        return "punct"
    for bp in _BOILERPLATE_PHRASES:
        if bp in phrase:
            return "boilerplate"
    if parent_df >= 10 and len(phrase.split()) <= 4:
        return "boilerplate"
    return "ok"


def _skip_flag(left: str, right: str, parent_df: int) -> str:
    for bp in ("Ví dụ", "mẫu", "Tiết này", "Em đọc"):
        if bp in right or bp in left:
            return "boilerplate"
    return "ok"


def _right_label(v, right) -> str:
    if isinstance(right, (list, tuple)):
        return " ".join(_tok(v, t) for t in right)
    return _tok(v, int(right))


def export_all(
    state,
    out_dir: Path,
    *,
    prefix: str,
    bake: bool = True,
    train_time_s: float | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ps, v = state.parents, state.vocab
    if bake:
        bake_derived_indices(state, skip_spans=False, frame_brackets=True, seq_index=False)
    elif not getattr(ps, "hlink_meta", None):
        mine_frame_brackets(state)

    usage = _cluster_usage(ps)

    # --- skip spans TSV ---
    skip_path = out_dir / f"{prefix}_skip_spans.tsv"
    with skip_path.open("w", encoding="utf-8") as f:
        f.write("id\tleft\tright\tpower\tpmi\tpromoted\tgaps\tparents\texample\n")
        for i, s in enumerate(ps.skip_spans or []):
            prov = s.get("prov", [])
            pis = sorted({int(x[0]) for x in prov})
            gaps = s.get("gaps", {})
            gap_s = ",".join(f"{k}:{v}" for k, v in sorted(gaps.items()))
            ex = _parent_text(ps, v, pis[0]) if pis else ""
            f.write(
                f"{i}\t{_tok(v, s['left'])}\t{_right_label(v, s['right'])}\t"
                f"{s['power']}\t{s['pmi']:.3f}\t{int(s.get('promoted', False))}\t"
                f"{{{gap_s}}}\t{len(pis)}\t{ex}\n"
            )

    # --- ISG clusters TSV ---
    isg_path = out_dir / f"{prefix}_isg_clusters.tsv"
    with isg_path.open("w", encoding="utf-8") as f:
        f.write(
            "id\tleft\tright\tphrase\tpower\tparent_df\tspan_len\t"
            "compositional\tparents\texample\n"
        )
        for i, c in enumerate(ps.seg.clusters):
            pis = sorted(usage.get(i, set()))
            span = c.token_span()
            phrase = " ".join(_tok(v, t) for t in span)
            comp = len(span) > 2 or (bool(c.span) and len(c.span) > 2)
            pdf = len(pis)
            ex = _parent_text(ps, v, pis[0]) if pis else ""
            f.write(
                f"{i}\t{_tok(v, c.left)}\t{_tok(v, c.right)}\t{phrase}\t"
                f"{c.power}\t{pdf}\t{len(span)}\t{comp}\t"
                f"{','.join(map(str, pis))}\t{ex}\n"
            )

    # --- frame brackets TSV ---
    fb_path = out_dir / f"{prefix}_frame_brackets.tsv"
    meta = getattr(ps, "hlink_meta", {}) or {}
    endpoints = getattr(ps, "hlink_endpoints", {}) or {}
    layer = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    with fb_path.open("w", encoding="utf-8") as f:
        f.write(
            "id\topener\tcloser\tleft_kind\tright_kind\tpower\tpmi\t"
            "parents\tgaps\tflag\n"
        )
        rows: list[tuple] = []
        for a, b, w in layer.items():
            m = meta.get((a, b), {})
            power = m.get("power", 0)
            gaps = m.get("gaps", {})
            prov = m.get("prov", [])
            pis = len({int(x[0]) for x in prov}) if prov else 0
            opener = decode_endpoint_label(a, endpoints, v)
            closer = decode_endpoint_label(b, endpoints, v)
            lk = m.get("left_kind", "token")
            rk = m.get("right_kind", "token")
            gap_s = ",".join(f"{k}:{v}" for k, v in sorted(gaps.items()))
            flag = "boilerplate" if any(x in opener + closer for x in (
                "Tiết", "Ví dụ", "Bài", "Lớp", "mẫu",
            )) else "ok"
            rows.append((power, a, b, opener, closer, lk, rk, w, pis, gap_s, flag))
        rows.sort(key=lambda r: -r[0])
        for idx, (_p, _a, _b, op, cl, lk, rk, w, pis, gap_s, flag) in enumerate(rows):
            f.write(
                f"{idx}\t{op}\t{cl}\t{lk}\t{rk}\t{_p}\t{w:.3f}\t"
                f"{pis}\t{gap_s}\t{flag}\n"
            )

    # --- summaries ---
    skip_flags = Counter()
    for i, s in enumerate(ps.skip_spans or []):
        prov = s.get("prov", [])
        pis = len({int(x[0]) for x in prov})
        skip_flags[_skip_flag(_tok(v, s["left"]), _right_label(v, s["right"]), pis)] += 1

    isg_flags = Counter()
    span_lens = Counter()
    n_bigram = n_comp = 0
    for i, c in enumerate(ps.seg.clusters):
        span = c.token_span()
        span_lens[len(span)] += 1
        if len(span) <= 2 and not c.span:
            n_bigram += 1
        else:
            n_comp += 1
        pis = sorted(usage.get(i, set()))
        phrase = " ".join(_tok(v, t) for t in span)
        isg_flags[_isg_flag(phrase, len(pis), bool(pis))] += 1

    fb_flags = Counter()
    for row in rows:
        fb_flags[row[-1]] += 1

    audit = {
        "source": "",
        "train_time_s": train_time_s,
        "parents": len(ps.parents),
        "clusters": len(ps.seg.clusters),
        "clusters_live": sum(1 for i in range(len(ps.seg.clusters)) if usage.get(i)),
        "skip_spans": len(ps.skip_spans or []),
        "frame_brackets": len(rows),
        "vocab": len(v),
        "isg_bigram": n_bigram,
        "isg_compositional": n_comp,
        "isg_span_lens": dict(sorted(span_lens.items())),
        "note": "B1 checkpoint export",
    }
    (out_dir / f"{prefix}_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source": audit.get("source"),
        "stats": {
            "skip_total": len(ps.skip_spans or []),
            "isg_total": len(ps.seg.clusters),
            "skip_by_flag": dict(skip_flags),
            "isg_by_flag": dict(isg_flags),
            "frame_by_flag": dict(fb_flags),
        },
    }
    (out_dir / f"{prefix}_cluster_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audit


def run_checks(state, out_dir: Path, prefix: str) -> dict:
    ps, v = state.parents, state.vocab
    checks: dict = {"pass": [], "warn": [], "fail": []}

    def ok(msg: str) -> None:
        checks["pass"].append(msg)

    def warn(msg: str) -> None:
        checks["warn"].append(msg)

    def fail(msg: str) -> None:
        checks["fail"].append(msg)

    # basic counts
    if len(ps.parents) == 476:
        ok("parents=476")
    else:
        warn(f"parents={len(ps.parents)} (expected 476)")

    # dụ → Ví dụ: (Ví,dụ) phrase right; dụ bị lọc idf khi làm đầu trái (gdf≈207/476)
    vid = v._token_to_id.get("Ví")
    did = v._token_to_id.get("dụ")
    if vid is not None and did is not None:
        for i, s in enumerate(ps.skip_spans or []):
            if s["left"] == did and tuple(s["right"]) == (vid, did):
                prov = {int(x[0]) for x in s.get("prov", [])}
                if s["power"] == len(prov):
                    ok(f"dụ→Ví dụ power={s['power']} == parents={len(prov)}")
                else:
                    fail(f"dụ→Ví dụ power={s['power']} != parents={len(prov)}")
                break
        else:
            from nn_aivie.training.skip_span import _is_skip_content_token
            if did and not _is_skip_content_token(did, v, ps._gdf, len(ps.parents)):
                ok(
                    f"dụ→Ví dụ absent — dụ gdf={ps._gdf.get(did)} "
                    f"lọc idf đầu trái (đúng luật hiện tại)"
                )
            else:
                warn("dụ→Ví dụ skip span not materialized")

    # khong nhung ma con S — batch ISG có thể dừng ở bước trung gian
    usage = _cluster_usage(ps)
    found_phrase = False
    for i, c in enumerate(ps.seg.clusters):
        phrase = " ".join(_tok(v, t) for t in c.token_span())
        if phrase in ("khong nhung ma con S", "khong nhung ma"):
            pis = usage.get(i, set())
            found_phrase = True
            ok(f"cluster#{i} '{phrase}' power={c.power} live_parents={len(pis)}")
            if phrase == "khong nhung ma con S" and len(pis) == 2:
                ok("khong nhung ma con S in 2 parents (Lớp 6 + 7)")
            elif phrase == "khong nhung ma" and len(pis) == 2:
                ok("khong nhung ma (bước compositional) in parents 346,433")
            break
    if not found_phrase:
        warn("cluster khong nhung* not found")

    # spurious frame bracket should be gone
    bad_fb = False
    meta = getattr(ps, "hlink_meta", {}) or {}
    endpoints = getattr(ps, "hlink_endpoints", {}) or {}
    layer = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    for a, b, w in layer.items():
        op = decode_endpoint_label(a, endpoints, v)
        cl = decode_endpoint_label(b, endpoints, v)
        if "Ví dụ mẫu" in op and "là hành tinh" in cl:
            m = meta.get((a, b), {})
            fail(f"spurious FB still present: {op}→{cl} power={m.get('power')}")
            bad_fb = True
    if not bad_fb:
        ok("no spurious '. Ví dụ mẫu → là hành tinh' FRAME_BRACKET")

    # G1 edges exist
    co_n = len(state.adjacency.layer(EdgeKind.CO_ACTIVATION))
    seq_n = len(state.adjacency.layer(EdgeKind.SEQUENTIAL))
    if co_n > 1000 and seq_n > 500:
        ok(f"G1 edges co={co_n} seq={seq_n}")
    else:
        warn(f"G1 thin: co={co_n} seq={seq_n}")

    report_path = out_dir / f"{prefix}_verify.json"
    report_path.write_text(
        json.dumps(checks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description="Export grade_lessons audit files")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--prefix", default="grade_lessons_b1")
    ap.add_argument("--no-bake", action="store_true")
    ap.add_argument("--train-time", type=float, default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    state, _ = load_checkpoint(ckpt)
    state.__dict__["_audit_source"] = str(ckpt)  # for audit json
    out_dir = Path(args.out_dir)

    audit = export_all(
        state, out_dir,
        prefix=args.prefix,
        bake=not args.no_bake,
        train_time_s=args.train_time,
    )
    audit["source"] = str(ckpt)
    (out_dir / f"{args.prefix}_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_path = out_dir / f"{args.prefix}_cluster_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["source"] = str(ckpt)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    checks = run_checks(state, out_dir, args.prefix)
    print(json.dumps({"audit": audit, "checks": checks}, indent=2, ensure_ascii=False))
    return 1 if checks["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
