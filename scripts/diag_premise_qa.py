#!/usr/bin/env python3
"""QA nổ 100% ở §12.17 là do ĐÂU? — TK_045 §12.18 (⚠️ CHẠY M4, dev không load nổi v12).

    PYTHONPATH=src python -u scripts/diag_premise_qa.py \
        --checkpoint checkpoints/network_v12_sent.pkl \
        --log-file out/diag_premise_qa.log

## Vì sao cần

`unattested_premise` hỏi "có cha nào chứa ĐỦ các token đặc-hiệu này không" — tức nó CHÍNH LÀ
phép thử "tổ hợp này có trong đồ thị không". Nên `qa_supervised` nổ **300/300** ở §12.17 có
đúng hai cách đọc, và chúng dẫn tới hai hành động NGƯỢC nhau:

  (a) `qa_supervised.jsonl` KHÔNG nằm trong corpus đã nạp vào v12 ⇒ nhãn "ĐÃ train" tôi gán cho
      nhóm đó là **giả định chưa kiểm** ⇒ 100% là ĐÚNG, cổng không sai, số §12.17 vô nghĩa với
      việc quyết bật/tắt.
  (b) Cha QA CÓ trong v12 nhưng cổng vẫn nổ ⇒ **cổng hỏng với đường QA** ⇒ phải sửa, không phải
      chỉ tắt.

Script này tách (a) khỏi (b). Ba mục, đọc theo thứ tự:

  1. Đếm cha `is_qa` trong checkpoint. =0 ⇒ (a), xong.
  2. Chạy cổng trên CHÍNH `match_ids` của cha QA. Theo cấu tạo `bi.A` dựng từ `match_ids` nên
     phải ra **0%**. Khác 0 ⇒ (b), cổng hỏng — dán ngay.
  3. Với vài prompt QA từ file: in `hi` (token đặc-hiệu) và kích thước GIAO sau MỖI token, để
     thấy token nào làm rỗng. Đây là thứ nói cho ta biết cổng nhầm KIỂU gì.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.context_positional import is_structural_token  # noqa: E402
from nn_aivie.hca.matrix_core import ensure_parent_bipartite, unattested_premise  # noqa: E402

QA_FILES = ["data/text_corpus/qa_supervised.jsonl", "data/sft/qa_heldout.jsonl"]
HOI, DAP = "Hỏi:", "Đáp:"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_sent.pkl")
    ap.add_argument("--log-file", default="out/diag_premise_qa.log")
    ap.add_argument("--last-k", type=int, default=6)
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()

    logf = (_ROOT / args.log_file).open("w", encoding="utf-8")
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
        logf.flush()

    log(f"[{time.time()-t0:.0f}s] load {args.checkpoint} (gc.disable) …")
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v, ps = state.vocab, state.parents
    bi = ensure_parent_bipartite(state)
    sur = v.token_of
    log(f"[{time.time()-t0:.0f}s] ✓ parents {len(ps):,}")

    # ── 1. có cha QA nào không ──────────────────────────────────────────────
    qa_idx = [i for i, r in enumerate(ps.parents) if getattr(r, "is_qa", False)]
    log(f"\n(1) cha is_qa = {len(qa_idx):,} / {len(ps.parents):,}"
        f"  ({len(qa_idx)/max(1,len(ps.parents)):.2%})")
    if not qa_idx:
        log("    ⇒ KẾT LUẬN (a): v12 KHÔNG có cha QA. Nhóm `qa_supervised` ở §12.17 bị tôi gán")
        log("      nhãn 'ĐÃ train' mà chưa kiểm ⇒ 100% là ĐÚNG, không phải lỗi cổng.")

    # ── 2. cổng chạy trên chính match_ids của cha QA (phải 0%) ──────────────
    if qa_idx:
        random.Random(42).shuffle(qa_idx)
        sample = qa_idx[: args.sample]
        fired = sum(
            unattested_premise(state, [int(t) for t in ps.parents[i].match_ids],
                               last_k=args.last_k)
            for i in sample
        )
        log(f"\n(2) cổng trên match_ids của {len(sample)} cha QA: nổ {fired}/{len(sample)}"
            f" = {fired/len(sample):.1%}")
        log("    0% ⇒ cổng ĐÚNG với cha QA ⇒ 100% ở §12.17 là do prompt file khác đồ thị (a).")
        log("    >0% ⇒ KẾT LUẬN (b): CỔNG HỎNG — dán nguyên dòng này về, phải sửa chứ không chỉ tắt.")

    # ── 3. giao rỗng ở token nào ────────────────────────────────────────────
    # Tự dựng frame ids — KHÔNG đọc `_frame_label_ids_cache` (mục (2) có thể bị bỏ qua nên
    # cache chưa được tạo ⇒ `Hỏi:`/`Đáp:` lọt vào vết và làm hiểu sai nguyên nhân).
    from nn_aivie.core.tokenizer import DAP_LABEL as _D, HOI_LABEL as _H

    frame_ids = frozenset(
        int(v._token_to_id[lab]) for lab in (_H, _D) if lab in v._token_to_id  # noqa: SLF001
    )
    log(f"\n(3) vết GIAO từng bước (last_k={args.last_k}) — token nào làm rỗng?"
        f"  [frame ids loại: {sorted(frame_ids)}]")
    for rel in QA_FILES:
        p = _ROOT / rel
        if not p.exists():
            continue
        shown = 0
        for line in p.open(encoding="utf-8"):
            if shown >= 3:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            txt = rec if isinstance(rec, str) else (
                rec.get("text", "") if isinstance(rec, dict) else "")
            if HOI not in txt or DAP not in txt:
                continue
            prompt = txt[: txt.rindex(DAP) + len(DAP)]
            ids = list(v.encode_prompt(prompt))
            frame = frame_ids
            hi = [t for t in dict.fromkeys(int(x) for x in ids)
                  if 0 <= t < bi.A.shape[0]
                  and t not in frame
                  and not is_structural_token(v, t)
                  and float(ps._gidf(t)) >= 4.0]  # noqa: SLF001
            hi = hi[-args.last_k:] if args.last_k > 0 else hi
            inter, steps = None, []
            for t in hi:
                tids = {t}
                try:
                    tids |= {int(x) for x in ps.case_equiv_ids(t, v)}
                except Exception:
                    pass
                r: set[int] = set()
                for u in tids:
                    if 0 <= u < bi.A.shape[0]:
                        r |= set(bi.A.getrow(u).indices.tolist())
                inter = r if inter is None else (inter & r)
                steps.append(f"{sur(t)}(df={len(r)})→∩{len(inter)}")
            log(f"\n  [{rel.rsplit('/', 1)[-1]}] {prompt[-110:]!r}")
            log(f"    hi={[sur(t) for t in hi]}")
            log(f"    {'  '.join(steps)}")
            shown += 1

    log(f"\n[{time.time()-t0:.0f}s] → {args.log_file}")
    log("Dán mục (1)(2) + vài dòng (3) vào ticket §12.18.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
