#!/usr/bin/env python3
"""
PROBE CO_ACT-STYLE — "hài hước" → từ-SẮC-THÁI? xếp theo ĐẶC THÙ (cho M4).

    ⚠️ CHẠY M4 (checkpoint ~1.8GB — đừng chạy trên máy dev yếu).
    PYTHONPATH=src python -u scripts/probe_style_coact.py \
        --checkpoint checkpoints/network_v11.pkl --log-file out/probe_style_coact_rank.log

## Câu hỏi (2026-07-23)

Nơ-ron style («hài hước») yếu → raw CO_ACT luôn bị át bởi hub (`,` `.` `và`).
Probe xếp (Luật 0 — idf/PPMI/contrast, không hardcode marker):

  1) RAW     — trọng số Hebbian thô (baseline; kỳ vọng hub)
  2) W×IDF   — C(seed,w) × gidf(w)
  3) PPMI*   — ego-PPMI thô (hay bị one-shot BPE / `X:` át)
  4) PPMI·C  — PPMI* × log(1+C)  → hạ one-shot cực hiếm
  5) ΔPPMI   — PPMI(seed) − mean PPMI(control) trên cùng w
               control mặc định: «là» «nghĩa» «Trần Hưng Đạo» (dict/content)
  6) (lọc)   — bỏ surface chỉ-dấu / kết thúc ':' (artifact nhãn `… về X:`)

Đọc tay «hài hước»: cột PPMI·C / ΔPPMI có marker sắc-thái không.
  * Có → liên kết ĐẶC THÙ sống → a_style.
  * Vẫn dict → Style hub / label↔body ingest (không eta mù).
Control «Trần Hưng Đạo» — ΔPPMI phải ra nội dung (Vương/Viễn…), sanity check.
"""

from __future__ import annotations

import argparse
import gc
import math
import string
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402

WORDS = ["hài hước", "vui", "buồn", "tự nhiên", "Trần Hưng Đạo"]
DEFAULT_CONTROLS = ["là", "nghĩa", "Trần Hưng Đạo"]
_PUNCT = set(string.punctuation + "…“”‘’—–·•")


def _fmt(sur, ranked, *, prec: str = ".2f") -> str:
    if not ranked:
        return "∅"
    parts = []
    for j, sc in ranked:
        if prec == ".0f":
            parts.append(f"{sur(j)!r}({sc:.0f})")
        else:
            parts.append(f"{sur(j)!r}({sc:.2f})")
    return "  ".join(parts)


def _row_sum(co, i: int) -> float:
    return float(sum(co.row(i).values()))


def _surface_core(tok: str) -> str:
    s = tok
    if s.startswith("Ġ") or s.startswith("Ċ"):
        s = s[1:]
    return s


def _is_artifact_surface(tok: str) -> bool:
    """Bỏ dấu thuần + đuôi ':' (artifact nhãn `học:` / `Trần:`). Luật-0-ish: hình thái token."""
    core = _surface_core(tok)
    if not core:
        return True
    if core.endswith(":"):
        return True
    if all(ch in _PUNCT or ch.isspace() for ch in core):
        return True
    return False


def _aggregate(co, ids: list[int]) -> tuple[Counter, float]:
    seed_set = set(ids)
    agg: Counter = Counter()
    for t in ids:
        for j, wt in co.row(t).items():
            if j not in seed_set and wt > 0:
                agg[j] += wt
    c_seed = sum(_row_sum(co, t) for t in ids) or 1.0
    return agg, c_seed


def _ego_ppmi(
    agg: Counter,
    c_seed: float,
    pool: list[int],
    strength_j: dict[int, float],
    *,
    gidf,
    gidf_min: float,
) -> dict[int, float]:
    n_ego = c_seed + sum(strength_j[j] for j in pool if j in strength_j)
    out: dict[int, float] = {}
    if n_ego <= 0:
        return out
    for j in pool:
        g = gidf(j)
        if g < gidf_min:
            continue
        c_ij = agg[j]
        c_j = strength_j.get(j, 0.0)
        if c_j <= 0:
            continue
        pmi = math.log((c_ij * n_ego) / (c_seed * c_j) + 1e-12)
        if pmi > 0:
            out[j] = pmi
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--gidf-min", type=float, default=2.0,
                    help="bỏ ứng viên gidf < ngưỡng khỏi các cột strength")
    ap.add_argument("--max-cand", type=int, default=2000,
                    help="giới hạn ứng viên theo RAW trước khi tính strength")
    ap.add_argument("--c-min", type=float, default=2.0,
                    help="sàn C_ij cho PPMI·C / ΔPPMI (bỏ one-shot)")
    ap.add_argument("--controls", default=",".join(DEFAULT_CONTROLS),
                    help="seed control cho ΔPPMI, cách nhau bởi dấu phẩy")
    ap.add_argument("--keep-artifacts", action="store_true",
                    help="giữ surface ':' / punct (mặc định lọc)")
    ap.add_argument("--log-file", default="out/probe_style_coact.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    vocab, ps = state.vocab, state.parents
    sur = vocab.token_of
    gidf = ps._gidf  # noqa: SLF001 — Luật 0: idf học được
    co = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
    log(f"[{time.time()-t0:.0f}s] ✓ loaded · parents {len(ps):,}")

    controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    # Precompute control PPMI maps (cùng máy ego trên pool riêng của từng control)
    ctrl_ppmi: list[dict[int, float]] = []
    for cw in controls:
        cids = list(vocab.encode_prompt(cw))
        cagg, c_seed = _aggregate(co, cids)
        cpool = [j for j, _ in cagg.most_common(args.max_cand)]
        cstrength = {j: _row_sum(co, j) for j in cpool}
        ctrl_ppmi.append(
            _ego_ppmi(cagg, c_seed, cpool, cstrength, gidf=gidf, gidf_min=args.gidf_min)
        )
    log(f"    controls ΔPPMI: {controls}")

    log(f"\n═══ CO_ACT «seed» → neighbors — RAW / đặc thù / floor / contrast (top {args.top}) ═══")
    log("    RAW · W×IDF=C×gidf · PPMI*=ego · PPMI·C=PPMI×log(1+C) · ΔPPMI=PPMI−mean(control)")
    log(f"    lọc artifact surface ':'/punct={'OFF' if args.keep_artifacts else 'ON'} · C_ij≥{args.c_min:g}\n")

    for w in WORDS:
        ids = list(vocab.encode_prompt(w))
        seed_set = set(ids)
        seed_cores = {_surface_core(sur(t)).lower() for t in ids}
        agg, c_seed = _aggregate(co, ids)
        raw = agg.most_common(args.top)
        pool = [j for j, _ in agg.most_common(args.max_cand)]
        strength_j = {j: _row_sum(co, j) for j in pool}

        scored_idf: list[tuple[float, int]] = []
        scored_ppmi: list[tuple[float, int]] = []
        scored_ppmi_c: list[tuple[float, int]] = []
        scored_dppmi: list[tuple[float, int]] = []

        ppmi_map = _ego_ppmi(
            agg, c_seed, pool, strength_j, gidf=gidf, gidf_min=args.gidf_min
        )

        for j in pool:
            c_ij = float(agg[j])
            g = gidf(j)
            if g < args.gidf_min:
                continue
            tok = sur(j)
            if not args.keep_artifacts and _is_artifact_surface(tok):
                continue
            # bỏ mảnh surface trùng seed (Ġhài / hước…)
            if _surface_core(tok).lower() in seed_cores:
                continue

            scored_idf.append((c_ij * g, j))
            pmi = ppmi_map.get(j)
            if pmi is None:
                continue
            scored_ppmi.append((pmi, j))
            if c_ij < args.c_min:
                continue
            scored_ppmi_c.append((pmi * math.log(1.0 + c_ij), j))

            ctrl_vals = [m[j] for m in ctrl_ppmi if j in m]
            # không có trong control → coi baseline 0 (ứng viên chỉ-style)
            ctrl_mean = (sum(ctrl_vals) / len(ctrl_vals)) if ctrl_vals else 0.0
            # nếu seed chính là một control («Trần Hưng Đạo»), Δ vs controls khác
            if w in controls and ctrl_vals:
                others = []
                for cw, m in zip(controls, ctrl_ppmi):
                    if cw == w:
                        continue
                    if j in m:
                        others.append(m[j])
                ctrl_mean = (sum(others) / len(others)) if others else 0.0
            d = pmi - ctrl_mean
            if d > 0:
                scored_dppmi.append((d * math.log(1.0 + c_ij), j))

        def _top(scored: list[tuple[float, int]]):
            scored.sort(reverse=True)
            return [(j, sc) for sc, j in scored[: args.top]]

        log(f"  «{w}» (ids {ids} · C_seed={c_seed:.0f} · neighbors={len(agg):,} · pool={len(pool)})")
        log(f"    RAW   : {_fmt(sur, raw, prec='.0f')}")
        log(f"    W×IDF : {_fmt(sur, _top(scored_idf))}")
        log(f"    PPMI* : {_fmt(sur, _top(scored_ppmi))}")
        log(f"    PPMI·C: {_fmt(sur, _top(scored_ppmi_c))}")
        log(f"    ΔPPMI : {_fmt(sur, _top(scored_dppmi))}")
        log()

    log(f"[{time.time()-t0:.0f}s] → đọc PPMI·C / ΔPPMI: marker sắc-thái = liên kết ĐẶC THÙ sống → a_style. "
        "Vẫn dict/hub = Style hub / label↔body ingest, không chỉ eta mù.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
