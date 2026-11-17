#!/usr/bin/env python3
"""
P3.0 HARNESS — đo BINDING trên đồ thị THẬT (TK_044, cho M4).

    ⚠️ CHẠY M4 — checkpoint đồ thị 2GB → ~20GB RAM.
    PYTHONPATH=src python -u scripts/probe_binding_graph.py \
        --checkpoint checkpoints/network_v11.pkl --probe data/sft/bind_probe.jsonl \
        --log-file out/probe_bind_P30.log

## Mục đích (đọc RUNBOOK_TK_044_P3.md §2 trước)

Thước A/B cho P3: binding = điền thực-thể MỚI (chưa/hiếm trong graph) vào khuôn đã học rồi COPY lại ở
slot sau. Nấc P3.0 = đo **NỀN** (chưa cắm kênh copy) — kỳ vọng ≈ 0 (graph chép/mosaic phát thực-thể
TRAIN, không bind biến mới), giống baseline-bigram của POC. Mọi thay đổi P3.1/3.2 so lại BẢNG này.

## Đo gì (mỗi item)

* **novelty**: thực-thể probe có THẬT là mới không (entity_register + gidf). Nếu là entity mạnh sẵn có
  → item KHÔNG hợp lệ (đo trí-nhớ, không phải bind) → hạ cờ `novel=False`, LOẠI khỏi tỉ lệ.
* **copy**: continuation có chứa lại thực-thể-biến ở slot sau không (single/pronoun: 1 slot; crossed:
  2 slot đúng THỨ TỰ X…Y…Y…X).
* **drift** (đọc tay): continuation có kéo thực-thể TRAIN lạ vào không (in nguyên văn để soi — bits mù
  với bịa, G15). Không tự chấm drift bằng máy (cần mắt người); in ra.

## Cách chấm (bền với BPE/bề mặt)

Chấm bằng SUBSTRING bề mặt trên phần continuation (bỏ prefix prompt). copy_ok = surface thực-thể-biến
xuất hiện lại. crossed_ok = cả hai surface xuất hiện + đúng thứ tự. Thô nhưng đủ cho tín hiệu nền≈0 vs
có-copy; nấc sau nếu cần tinh hơn thì chấm theo vị-trí-slot token.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import (  # noqa: E402
    _SegmentPlan, _content_ids, generate_autoregressive,
)
from nn_aivie.hca.readout import decode_surface  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", default="checkpoints/network_v11.pkl")
    ap.add_argument("--probe", default="data/sft/bind_probe.jsonl")
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-channel", default="off", choices=["off", "hard", "hopfield"],
                    help="P3.2: kênh copy slot. off=nền P3.0, hard=argmax cứng (Plan B), hopfield=mềm (Plan A).")
    ap.add_argument("--copy-boost", type=float, default=15.0,
                    help="Plan B: boost score entity khi copy slot (ar_copy_boost).")
    ap.add_argument("--plan-mode", default="retrieval", choices=["retrieval", "continuation"],
                    help="continuation = bỏ _build_segment_plan (nguồn drift), agenda=prompt-content, "
                         "entity_ids=entity_register → cô lập copy-channel khỏi plan-drift (§14)")
    ap.add_argument("--log-file", default="out/probe_bind_P30.log")
    args = ap.parse_args()

    logf = open(_ROOT / args.log_file, "w")
    t0 = time.time()

    def log(m=""):
        line = f"[{time.time()-t0:6.0f}s] {m}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    log(f"load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    ps, vocab = state.parents, state.vocab
    if ps is None:
        raise SystemExit("✗ thiếu parents")
    log(f"✓ loaded · parents {len(ps):,}")

    cfg = HCAConfig()
    # P3.2: set copy channel theo --copy-channel
    if hasattr(cfg, "ar_copy_channel"):
        cfg.ar_copy_channel = args.copy_channel
        cfg.ar_copy_boost = args.copy_boost
    elif args.copy_channel != "off":
        log("⚠ config CHƯA có ar_copy_channel — kênh copy chưa hiện thực (P3.2). Đang đo NỀN.")

    items = [json.loads(l) for l in (_ROOT / args.probe).read_text().splitlines() if l.strip()]
    log(f"probe: {len(items)} item")

    # Inverted index theo token đầu để check "surface sequence" nhanh hơn quét mù toàn bộ parents.
    first_tok_to_parents: dict[int, list[int]] = {}
    for i, rec in enumerate(ps.parents):
        if rec.emit_ids:
            first_tok_to_parents.setdefault(rec.emit_ids[0], []).append(i)

    def _contains_subseq(seq: tuple[int, ...], ids: tuple[int, ...]) -> bool:
        if not seq or len(seq) > len(ids):
            return False
        m = len(seq)
        for j in range(0, len(ids) - m + 1):
            if ids[j:j + m] == seq:
                return True
        return False

    def _surface_exists_exact(ids: tuple[int, ...]) -> bool:
        """Surface "đã có trong graph" = có parent chứa ĐÚNG chuỗi token liên tiếp."""
        if not ids:
            return False
        cands = first_tok_to_parents.get(ids[0], [])
        for pi in cands:
            if _contains_subseq(ids, ps.parents[pi].emit_ids):
                return True
        return False

    def novelty(surface: str):
        """thực-thể có MỚI không: ưu tiên exact-surface, reg chỉ để debug."""
        try:
            ids = list(vocab.encode_prompt(surface))
        except Exception:
            return True, "enc?"
        reg = ps.entity_register_ids(tuple(ids))
        strong = any(t in reg for t in ids)
        gidf = min((ps._gidf(t) for t in ids), default=0.0)
        exact = _surface_exists_exact(tuple(ids))
        # Novel nếu surface KHÔNG xuất hiện nguyên chuỗi trong graph.
        return (not exact), f"exact={'Y' if exact else 'n'} reg={'Y' if strong else 'n'} gidf={gidf:.1f}"

    def continuation(prompt: str) -> str:
        plan = None
        if args.plan_mode == "continuation":
            # cô lập khỏi plan-drift: agenda = content prompt (KHÔNG truy hồi fact train),
            # entity_ids = entity_register (thực-thể prompt) → copy-channel vẫn có cái boost.
            qids = list(vocab.encode_prompt(prompt))
            qc = list(dict.fromkeys(_content_ids(qids, vocab, skip=set())))
            reg = ps.entity_register_ids(tuple(qc))
            plan = _SegmentPlan(segments=[qc], entity_ids=reg, flat_agenda=list(qc))
        ids = generate_autoregressive(state, prompt, max_len=args.max_len, config=cfg,
                                      seed=args.seed, plan=plan)
        full = decode_surface(state, ids)
        # bỏ phần prompt ở đầu (so bề mặt thô)
        p = decode_surface(state, list(vocab.encode_prompt(prompt)))
        return full[len(p):].strip() if full.startswith(p[:20]) else full

    tally = Counter()
    log("\n═══ P3.0 baseline (kênh copy TẮT) — kỳ vọng copy ≈ 0 ═══")
    for it in items:
        typ = it["type"]
        ents = it.get("slots") or ([it["antecedent"]] if "antecedent" in it else [])
        # novelty của mọi thực-thể trong item
        novs = [novelty(e) for e in dict.fromkeys(ents)]
        novel = all(n for n, _ in novs)
        cont = continuation(it["prompt"])
        if typ in ("single", "pronoun"):
            gold = ents[0]
            copy_ok = gold in cont
        else:  # crossed: X tặng Y … → mong Y…X (đúng thứ tự trong continuation)
            x, y = it["order"][0], it["order"][1]
            iy, ix = cont.find(y), cont.rfind(x)
            copy_ok = (iy >= 0 and ix >= 0 and iy < ix)
        tag = typ if novel else typ + "·SKIP(not-novel)"
        if novel:
            tally[f"{typ}_tot"] += 1
            tally[f"{typ}_ok"] += int(copy_ok)
        log(f"[{tag}] copy={'✓' if copy_ok else '✗'} novelty={[d for _,d in novs]}")
        log(f"    prompt: {it['prompt']!r}")
        log(f"    cont  : {cont!r}   (mong copy: {ents})")

    log("\n═══ BẢNG (chỉ item novel hợp lệ) ═══")
    for typ in ("single", "crossed", "pronoun"):
        tot, ok = tally[f"{typ}_tot"], tally[f"{typ}_ok"]
        if tot:
            log(f"  {typ:8s}: copy {ok}/{tot}")
    log("\n→ P3.0: xác nhận nền (copy thấp) + harness chạy. Số này là MỐC cho P3.2 A/B.")
    log("  Drift: đọc tay 'cont' ở trên — có thực-thể TRAIN lạ bị kéo vào không (G15).")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
