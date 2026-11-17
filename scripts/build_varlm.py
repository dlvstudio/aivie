#!/usr/bin/env python3
"""
TK_031 — build sidecar VarOrderLM (n-gram biến-bậc) cho checkpoint.

Đếm ctx-len 1..4 trên corpus thật (statements + doc + add + history), encode bằng vocab
checkpoint, prune singleton k≥3 (probe: miễn phí) → `<ckpt>.varlm.pkl`.
Ghi progress ra file (Luật observability).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.varlm import VarOrderLM, varlm_sidecar_path
from nn_aivie.core.vocab import SPECIAL_EOS, is_surface_token
from nn_aivie.data.paragraph_units import iter_units_from_doc_jsonl
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]
CORPUS = [
    "data/real_vi/statements.jsonl",
    "data/real_vi/statements_doc.jsonl",
    "data/real_vi/statements_doc_add.jsonl",
    "data/real_vi/history_vi.jsonl",
    "data/real_vi/history_wiki_para.jsonl",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v9_histwiki_paragraph.pkl")
    ap.add_argument("--max-ctx", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.45)
    ap.add_argument(
        "--use-clusters",
        action="store_true",
        help="Dùng SegmentGraph cluster neuron để nén context trước khi đếm n-gram (cluster-aware VarLM).",
    )
    ap.add_argument(
        "--paragraph-units",
        action="store_true",
        help="TK_035: đếm trên đơn vị 2–4 câu (ctx xuyên dấu phẩy/chấm trong đoạn)",
    )
    ap.add_argument("--min-sent", type=int, default=2)
    ap.add_argument("--max-sent", type=int, default=4)
    ap.add_argument("--extra-corpus", nargs="*", default=[],
                    help="corpus THÊM (vd data/text_corpus/styled_sft.jsonl) — SFT-ingest styled vào VarLM")
    args = ap.parse_args()

    global CORPUS
    CORPUS = CORPUS + list(args.extra_corpus)   # nối corpus styled để VarLM học phrasing (Luật-0: đếm)

    log = get_run_logger("build_varlm")
    state, _cache = load_checkpoint(_ROOT / args.checkpoint, logger=log)
    vocab = state.vocab
    tid = vocab._token_to_id

    lm = VarOrderLM(max_ctx=args.max_ctx, lam=args.lam, use_clusters=bool(args.use_clusters))
    n_s = 0
    seg = state.parents.seg if (args.use_clusters and state.parents is not None) else None

    eos_id = state.vocab.id_of(SPECIAL_EOS)

    def _feed_text(text: str) -> None:
        """Nối EOS cuối mỗi đơn vị — ĐƯỜNG NẠP THỨ BA, phải khớp `ingest._with_eos`.

        Bẫy đã cắn (verify 2026-07-16: `varlm học mốc kết = 0 ctx có EOS làm token kế`):
        `_with_eos` được thêm vào `ingest.py` rồi `pipeline.py`, nhưng `build_varlm` tự
        tokenize lấy → sidecar KHÔNG thấy EOS → VarLM không bao giờ đề xuất mốc kết, dù đồ
        thị đã học. Ba đường nạp phải nối EOS như nhau; xem VERIFY_CHECKPOINT.md §3.

        EOS chỉ nối vào đoạn CUỐI: token lạ (không có trong vocab) cắt chuỗi thành nhiều
        khúc, nhưng chỉ khúc cuối mới thật sự kết thúc đơn vị.

        TK_041: trên vocab BPE, nhánh "token lạ" **chết theo cấu trúc** — vocab bề mặt ĐÓNG
        + byte-fallback ⇒ mọi text encode được, không bao giờ có id lạ, không bao giờ cắt
        khúc. Chuỗi VarLM giờ khớp CHÍNH XÁC chuỗi mà `ingest` nạp vào đồ thị (cùng
        `encode_text`), thay vì hai lần tokenize độc lập tình cờ trùng nhau.
        """
        nonlocal n_s
        if state.vocab.tokenizer is not None:
            seq = state.vocab.encode_text(text)
        else:  # legacy: token lạ cắt chuỗi thành khúc
            seq = []
            for tok in tokenize_punct(text):
                i = tid.get(tok)
                if i is None:
                    if seq:
                        lm.add_sequence(seq)
                        seq = []
                    continue
                seq.append(i)
        if seq:
            ids = seg.encode_tokens_with_clusters(seq) if seg is not None else seq
            lm.add_sequence([*ids, eos_id])
        n_s += 1

    with log.phase("đếm n-gram"):
        if args.paragraph_units:
            for p in CORPUS:
                fp = _ROOT / p
                if not fp.exists():
                    continue
                for unit in iter_units_from_doc_jsonl(
                    fp, min_sent=args.min_sent, max_sent=args.max_sent,
                ):
                    _feed_text(unit)
                    if n_s % 20000 == 0:
                        log.log(f"  {n_s} đoạn, {lm.n_tokens} token")
        else:
            for p in CORPUS:
                fp = _ROOT / p
                if not fp.exists():
                    continue
                with fp.open(encoding="utf-8") as f:
                    for line in f:
                        text = json.loads(line).get("text", "")
                        if not text:
                            continue
                        # OOV cắt chuỗi (không nối n-gram xuyên token lạ)
                        _feed_text(text)
                        if n_s % 40000 == 0:
                            log.log(f"  {n_s} câu, {lm.n_tokens} token")
    log.log(f"đếm xong: {n_s} câu, {lm.n_tokens} token; {lm.stats()}")

    with log.phase("prune singleton k≥3"):
        st = lm.prune(min_total=2, from_ctx=3)
    log.log(f"prune: {st} → {lm.stats()}")

    out = varlm_sidecar_path(_ROOT / args.checkpoint)
    with log.phase("save"):
        lm.save(out)
    log.log(f"✓ SAVED {out.name} ({out.stat().st_size // (1 << 20)}MB)")


if __name__ == "__main__":
    main()
