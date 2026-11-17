#!/usr/bin/env python
"""P0/P45 — baseline chất lượng W_skip primary + PMI-cross (không dùng mạng LM).

Xuất:
  out/quality_baseline_v12.json
  out/quality_baseline_v12.log

Chạy:
  PYTHONPATH=src .venv/bin/python scripts/quality_baseline_v12.py \\
    --checkpoint checkpoints/network_v12_sent.pkl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import generate  # noqa: E402
from nn_aivie.hca.metrics_pmi import PmiCrossIndex  # noqa: E402
from nn_aivie.hca.readout import decode_surface  # noqa: E402
from nn_aivie.hca.w_skip import clear_w_skip_cache  # noqa: E402

# Bộ cố định — gate P42–P45
PROMPTS = [
    "Năm 1010, vua Lý Thái Tổ",
    "Nhà Lý dời đô về",
    "Ngô Quyền đánh bại quân Nam Hán năm",
    "Hỏi: Nhà Lý dời đô về đâu? Đáp:",
    "Việt Nam là một quốc gia nằm ở",
    "Phở là món ăn",
    "Trong thời kỳ Bắc thuộc, nhân dân ta",
    "Thủ đô Hà Nội có lịch sử hơn nghìn năm, từ khi",
]

HUB = ("quan", "trọng", "mốc", "là")


def _hub_ratio(text: str) -> float:
    toks = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if not toks:
        return 0.0
    hit = sum(1 for t in toks if t in HUB)
    return hit / len(toks)


def _after_period_salad(text: str) -> bool:
    """True nếu sau dấu chấm đầu có hub-ratio cao (salad)."""
    parts = re.split(r"[.!?]", text, maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        return False
    return _hub_ratio(parts[1]) >= 0.35


def _first_clause_ok_doi_do(text: str) -> bool:
    t = text.lower()
    return ("thăng long" in t or "thănglong" in t) and ("đô" in t or "dời" in t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mz", action="store_true", help="bật MZ agenda")
    ap.add_argument("--no-tiebreak", action="store_true",
                    help="TẮT phá-thế-hoà Hopfield trong rank_matches (§13.15) — dùng để "
                         "A/B: baseline §12.149 được đo KHI CƠ CHẾ NÀY CHƯA TỒN TẠI")
    ap.add_argument("--out-json", default="out/quality_baseline_v12.json")
    ap.add_argument("--out-log", default="out/quality_baseline_v12.log")
    args = ap.parse_args()

    t0 = time.time()
    clear_w_skip_cache()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    pmi = PmiCrossIndex(state)
    # §13.15 — phá-thế-hoà Hopfield nay BẬT mặc định trong `rank_matches`, tức đường
    # retrieval (⇒ agenda MZ) đã đổi so với lúc đo baseline §12.149. `--no-tiebreak`
    # cho phép A/B sạch: cùng checkpoint, cùng seed, chỉ khác một cờ.
    if args.no_tiebreak and state.parents is not None:
        state.parents.hopfield_tiebreak_beta = 0.0
    tb = "TẮT" if args.no_tiebreak else "BẬT (mặc định §13.15)"
    print(f"[{time.time()-t0:.0f}s] loaded parents={len(state.parents):,} · "
          f"phá-thế-hoà: {tb}", flush=True)

    cfg = replace(
        HCAConfig(),
        ar_w_skip_use_mz_agenda=bool(args.mz),
        ar_multizone=bool(args.mz),
    )
    rows = []
    for prompt in PROMPTS:
        for seed in range(args.seeds):
            out = generate(state, prompt, max_len=args.max_len, config=cfg, seed=seed)
            pids = list(state.vocab.encode_prompt(prompt))
            cont_ids = [int(t) for t in out[len(pids):]]
            text = decode_surface(state, out)
            cont = decode_surface(state, out[len(pids):]).strip()
            score = pmi.score(pids, cont_ids)
            row = {
                "prompt": prompt,
                "seed": seed,
                "text": text,
                "cont": cont,
                "pmi": round(float(score), 4),
                "hub_ratio": round(_hub_ratio(cont), 4),
                "salad_after_period": _after_period_salad(text),
                "doi_do_clause_ok": (
                    _first_clause_ok_doi_do(text)
                    if "Lý Thái Tổ" in prompt or "dời đô" in prompt.lower()
                    else None
                ),
                "n_tok": len(cont_ids),
            }
            from nn_aivie.core.vocab import SPECIAL_EOS
            row["stopped_eos"] = bool(out) and int(out[-1]) == int(
                state.vocab.id_of(SPECIAL_EOS)
            )
            rows.append(row)
            print(
                f"pmi={row['pmi']:.3f} hub={row['hub_ratio']:.2f} "
                f"salad={row['salad_after_period']} | {prompt[:28]}… → {cont[:60]}",
                flush=True,
            )

    n = len(rows)
    summary = {
        "n": n,
        "pmi_mean": round(sum(r["pmi"] for r in rows) / max(1, n), 4),
        "hub_ratio_mean": round(sum(r["hub_ratio"] for r in rows) / max(1, n), 4),
        "salad_after_period_rate": round(
            sum(1 for r in rows if r["salad_after_period"]) / max(1, n), 4
        ),
        "eos_rate": round(sum(1 for r in rows if r["stopped_eos"]) / max(1, n), 4),
        "doi_do_ok_rate": None,
        "mz": bool(args.mz),
        "elapsed_s": round(time.time() - t0, 1),
    }
    dd = [r for r in rows if r["doi_do_clause_ok"] is not None]
    if dd:
        summary["doi_do_ok_rate"] = round(
            sum(1 for r in dd if r["doi_do_clause_ok"]) / len(dd), 4
        )

    # P45 gates (ngưỡng mềm — baseline + theo dõi cải thiện)
    gates = {
        "pmi_ge_0_8": summary["pmi_mean"] >= 0.8,
        "salad_le_0_4": summary["salad_after_period_rate"] <= 0.4,
        "doi_do_ge_0_6": (
            summary["doi_do_ok_rate"] is None or summary["doi_do_ok_rate"] >= 0.6
        ),
    }
    summary["gates"] = gates
    summary["quality_pass"] = all(gates.values())
    summary["hopfield_tiebreak"] = (not args.no_tiebreak)

    out_json = _ROOT / args.out_json
    out_log = _ROOT / args.out_log
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "=== quality_baseline_v12 ===",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "",
    ]
    for r in rows:
        lines.append(
            f"seed{r['seed']} pmi={r['pmi']} salad={r['salad_after_period']}: {r['text']}"
        )
    out_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out_json} · {out_log}", flush=True)


if __name__ == "__main__":
    main()
