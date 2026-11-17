#!/usr/bin/env python3
"""
Roadmap — Discourse/paragraph prompt eval (dataset-derived, không cần gold).

Mục tiêu: cover sát corpus thật hơn so với 7 prompt khung, bằng cách:
- lấy prompt trực tiếp từ các nguồn jsonl (history_wiki_para / statements_doc / conv),
- chạy generate() multi=True với ar_varlm_weight sweep (0 / 0.35 / 0.5)
- đo: copy (verbatim thật, LCS với corpus), multi-sent, commas, entity-keep.
  `overlap` (cũ) ĐÃ XOÁ: nó là %token-prompt-trong-output, mà output luôn bắt đầu
  bằng prompt → LUÔN 1.00 (129/129 dòng), không phân biệt nổi sinh với tra cứu.

Log realtime: mọi dòng ghi ra stdout (flush) + `out/logs/eval_discourse_prompts_<ts>.log`
+ append từng dòng vào `--out` (tail -f được ngay).

Chạy:
  PYTHONPATH=src python -u scripts/eval_discourse_prompts.py \
    --checkpoint checkpoints/network_v10_histwiki_paragraph.pkl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.data.quality import normalize_ws, split_sentences
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.generate_hebbian import generate
from nn_aivie.hca.readout import decode_surface
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.util.progress import get_run_logger

_ROOT = Path(__file__).resolve().parents[1]


_WORD = re.compile(r"\w+", re.UNICODE)
_COMMON_NOUN_PROMPTS = [
    # bắt sổ nợ cap-gate: chủ đề danh từ thường vẫn cần discourse mượt
    "phở là",
    "mùa hè em",
    "ẩm thực Việt Nam",
    "Tết Nguyên Đán",
    "bóng đá là",
    "âm nhạc giúp",
    "trường học là",
    "đọc sách giúp",
    "sức khỏe là",
    "môi trường là",
]


def _entity_spans(text: str) -> list[str]:
    """Maximal run of Title-ish tokens (đơn giản, đủ để đo 'giữ entity')."""
    toks = _WORD.findall(text)
    spans: list[str] = []
    run: list[str] = []
    for tok in toks:
        cap = bool(tok) and tok[0].isupper() and not tok.isupper()
        if cap:
            run.append(tok)
        else:
            if len(run) >= 2:
                spans.append(" ".join(run))
            run = []
    if len(run) >= 2:
        spans.append(" ".join(run))
    return spans[:6]


COPY_TOPK = 20  # số đoạn ứng viên đem so LCS (chọn theo số token chung)


class CorpusIndex:
    """Đoạn corpus + inverted index token→đoạn, để đo `copy` (verbatim thật).

    Vì sao cần `copy`: `overlap` KHÔNG đo verbatim — nó là % token PROMPT có trong output,
    mà `generate()` luôn trả prompt + phần sinh → nó **luôn = 1.00** dù chép nguyên xi hay
    sinh thật (đo 2026-07-15: 129/129 dòng = 1.00). Bộ eval không có thước nào phân biệt
    sinh với tra cứu — đúng thứ Trụ 4 lấy làm mục tiêu.

    CẬN DƯỚI: chỉ phủ history_wiki_para + statements_doc. Chép từ nguồn train khác
    (conv/lesson/dictionary) KHÔNG bị bắt → copy thật có thể cao hơn.
    """

    def __init__(self, units: list[list[str]]):
        self.units = units
        self.inv: dict[str, list[int]] = defaultdict(list)
        for ui, u in enumerate(units):
            for tok in set(u):
                self.inv[tok].append(ui)


def build_corpus_index(limit_units: int = 0) -> CorpusIndex:
    from nn_aivie.data.paragraph_units import iter_units_from_doc_jsonl

    texts: list[str] = []
    hist = _ROOT / "data/real_vi/history_wiki_para.jsonl"
    if hist.exists():
        texts.extend(iter_units_from_doc_jsonl(hist, min_sent=1, max_sent=8, dedup=True))
    doc = _ROOT / "data/real_vi/statements_doc.jsonl"
    if doc.exists():
        texts.extend(_load_texts_from_jsonl(doc, limit=limit_units or 10**9))
    return CorpusIndex([tokenize_punct(normalize_ws(t)) for t in texts])


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def _copy_frac(gen_text: str, index: CorpusIndex, topk: int = COPY_TOPK) -> float:
    """Chuỗi con chung dài nhất giữa PHẦN SINH và đoạn corpus khớp nhất, / độ dài phần sinh.

    1.0 = phần sinh là bản sao (có thể sứt) của một đoạn corpus. Dùng CHUỖI CON chứ không
    n-gram liền, và KHÔNG lọc token nào — đó là điểm mấu chốt:

    - Luật 0: không cần khái niệm "dấu câu", không danh sách ký tự. Bản n-gram trước đó
      phải lọc `PUNCT_TOKENS` để khớp, tức thước tự mang tri thức ngôn ngữ đi chấm mạng.
    - Bền với token rơi: A3 recite đúng chuỗi từ corpus nhưng KHÔNG sinh nổi dấu câu
      (đo 2026-07-15). N-gram liền lệch pha ở mọi chỗ có `.` → báo 0.29 trong khi sự thật
      là chép sạch. LCS chỉ mất 1 vị trí/dấu → vẫn 1.00. Đúng với MỌI token rơi, không
      riêng dấu câu.

    SÀN ≈0.2: câu bịa vẫn khớp hư từ theo thứ tự trên corpus lớn. Đọc số theo tương đối.
    """
    g = tokenize_punct(normalize_ws(gen_text))
    if not g or not index.units:
        return 0.0
    cnt: dict[int, int] = defaultdict(int)
    for tok in set(g):
        for ui in index.inv.get(tok, ()):
            cnt[ui] += 1
    if not cnt:
        return 0.0
    cands = sorted(cnt, key=cnt.__getitem__, reverse=True)[:topk]
    best = max((_lcs_len(g, index.units[ui]) for ui in cands), default=0)
    return best / len(g)


def _stats(
    prompt: str, out: str, index: CorpusIndex | None = None
) -> dict[str, float]:
    sents = len(re.findall(r"[.!?]", out))
    ent = _entity_spans(prompt)
    ent_keep = 0.0
    if ent:
        ent_keep = sum(1 for e in ent if e in out) / len(ent)
    # copy đo trên PHẦN SINH THÊM (bỏ prompt) — nếu không bỏ thì prompt tự nó là verbatim
    gen_only = out[len(prompt) :] if out.startswith(prompt) else out
    copy = _copy_frac(gen_only, index) if index is not None else 0.0
    return {
        "chars": float(len(out)),
        "commas": float(out.count(",")),
        "sents": float(sents),
        "multi": float(sents >= 2 or ("," in out and len(out) > 60)),
        "ent_keep": float(ent_keep),
        "copy": float(copy),
    }


def _pick_prefix(text: str, *, min_tok: int, max_tok: int) -> str:
    toks = tokenize_punct(normalize_ws(text))
    toks = [t for t in toks if t]
    if len(toks) < min_tok:
        return ""
    k = min(max_tok, len(toks))
    # cắt ở biên câu nếu gặp dấu kết câu sớm (prompt tự nhiên hơn)
    cut = k
    for i in range(min(k, len(toks))):
        if toks[i] in {".", "!", "?"} and i + 1 >= min_tok:
            cut = i + 1
            break
    return normalize_ws(" ".join(toks[:cut]))


def _load_texts_from_jsonl(path: Path, *, key: str = "text", limit: int = 0) -> list[str]:
    out: list[str] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = (row.get(key) or "").strip()
            if t:
                out.append(t)
                if limit and len(out) >= limit:
                    break
    return out


def _build_prompts(
    *,
    rng: random.Random,
    n_from_history: int,
    n_from_doc: int,
    n_from_conv: int,
    min_tok: int,
    max_tok: int,
) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []

    # 1) history wiki paragraphs: dùng paragraph-units 2–4 câu → lấy prefix
    from nn_aivie.data.paragraph_units import iter_units_from_doc_jsonl

    hist = _ROOT / "data/real_vi/history_wiki_para.jsonl"
    units = list(iter_units_from_doc_jsonl(hist, min_sent=2, max_sent=4, dedup=True))
    rng.shuffle(units)
    for u in units[:n_from_history]:
        p = _pick_prefix(u, min_tok=min_tok, max_tok=max_tok)
        if p:
            prompts.append(("history_wiki_para", p))

    # 2) statements_doc: câu fact wiki/curated có doc_id+para, sát retrieval
    doc = _ROOT / "data/real_vi/statements_doc.jsonl"
    doc_texts = _load_texts_from_jsonl(doc, limit=n_from_doc * 6)
    rng.shuffle(doc_texts)
    for t in doc_texts:
        p = _pick_prefix(t, min_tok=min_tok, max_tok=max_tok)
        if p:
            prompts.append(("statements_doc", p))
        if sum(1 for s, _ in prompts if s == "statements_doc") >= n_from_doc:
            break

    # 3) conv stories/chat: lấy "text"/turns blob → tách câu → prefix
    conv_path = _ROOT / "data/conv/stories_raw.jsonl"
    conv_rows = _load_texts_from_jsonl(conv_path, limit=n_from_conv * 2)
    rng.shuffle(conv_rows)
    for blob in conv_rows:
        # chọn 1 câu đầu trong blob để prompt giống người dùng
        ss = split_sentences(blob)
        if not ss:
            continue
        p = _pick_prefix(ss[0], min_tok=min_tok, max_tok=max_tok)
        if p:
            prompts.append(("conv", p))
        if sum(1 for s, _ in prompts if s == "conv") >= n_from_conv:
            break

    # 4) thêm vài prompt frame QA cho coverage “Hỏi/Đáp”
    prompts.extend(
        [
            ("frame", "Hỏi: Việt Nam là gì Đáp:"),
            ("frame", "Hỏi: Văn Miếu là gì Đáp:"),
            ("frame", "Hỏi: Nhà Lý là gì Đáp:"),
        ]
    )
    return prompts


class _OutWriter:
    """Append kết quả eval ra --out; mỗi dòng flush ngay."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def emit(self, line: str = "") -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def main() -> None:
    # Đảm bảo stdout không buffer khi pipe/tee.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v10_histwiki_paragraph.pkl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-history", type=int, default=10)
    ap.add_argument("--n-doc", type=int, default=10)
    ap.add_argument("--n-conv", type=int, default=10)
    ap.add_argument("--n-common", type=int, default=10, help="prompt danh từ thường (anti cap-only)")
    ap.add_argument("--min-tok", type=int, default=3)
    ap.add_argument("--max-tok", type=int, default=7)
    ap.add_argument("--max-len", type=int, default=72)
    ap.add_argument("--max-modes", type=int, default=0, help="chỉ chạy N mode đầu (0=all)")
    ap.add_argument("--max-prompts", type=int, default=0, help="chỉ chạy N prompt đầu (0=all)")
    ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.35, 0.5])
    ap.add_argument(
        "--no-verbatim-nets",
        action="store_true",
        help="TẮT 2 lưới dội verbatim (ar_tail_as_agenda_opener=True, ar_tail_verbatim_fallback"
        "=False, ar_verbatim_fallback=False) → buộc vòng A3 SINH thật. Mặc định 2 lưới này "
        "return/vứt kết quả A3 trước khi tới kênh micro, nên sweep varlm là no-op "
        "(đo 2026-07-15: varlm 0/0.35/0.5 ra text GIỐNG TỪNG BYTE ở 129/129 dòng).",
    )
    ap.add_argument(
        "--no-copy",
        action="store_true",
        help="bỏ đo copy (thước verbatim) — dựng index corpus tốn ~17s",
    )
    ap.add_argument("--out", default="out/tk035_discourse_prompts.log")
    args = ap.parse_args()

    log = get_run_logger("eval_discourse_prompts")
    outw = _OutWriter(_ROOT / args.out)
    log.log(f"results → {outw.path}")

    rng = random.Random(args.seed)

    with log.phase("load_checkpoint"):
        state, _ = load_checkpoint(_ROOT / args.checkpoint, logger=log)

    with log.phase("build_prompts"):
        prompts = _build_prompts(
            rng=rng,
            n_from_history=args.n_history,
            n_from_doc=args.n_doc,
            n_from_conv=args.n_conv,
            min_tok=args.min_tok,
            max_tok=args.max_tok,
        )
        common = list(_COMMON_NOUN_PROMPTS)
        rng.shuffle(common)
        for p in common[: max(0, args.n_common)]:
            prompts.append(("common_noun", p))

    base_cfg = HCAConfig()
    tag = ""
    if args.no_verbatim_nets:
        base_cfg = replace(
            base_cfg,
            ar_tail_as_agenda_opener=True,   # pool tail → agenda cho A3, không dội thẳng
            ar_tail_verbatim_fallback=False,  # A3 phủ agenda thấp → GIỮ kết quả A3
            ar_verbatim_fallback=False,       # crisp plan cũng không dội verbatim
        )
        tag = " nets=off"
    modes: list[tuple[str, HCAConfig]] = []
    for w in args.weights:
        modes.append((f"gen multi varlm={w:g}{tag}", replace(base_cfg, ar_varlm_weight=w)))
    if args.max_modes and args.max_modes > 0:
        modes = modes[: args.max_modes]

    if args.max_prompts and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    header = [
        f"checkpoint={Path(args.checkpoint).name}",
        f"parents={len(state.parents.parents)} vocab={len(state.vocab)}",
        f"prompts={len(prompts)} (history={args.n_history}, doc={args.n_doc}, conv={args.n_conv}, "
        f"common={args.n_common}, +frame=3)",
        "",
        "=== Discourse prompts (dataset-derived) ===",
    ]
    for line in header:
        outw.emit(line)
    log.log(
        f"eval: {len(prompts)} prompts × {len(modes)} modes = {len(prompts) * len(modes)} generations"
    )

    index: CorpusIndex | None = None
    if not args.no_copy:
        with log.phase("build_corpus_index (copy metric — LCS)"):
            index = build_corpus_index()
            log.log(f"corpus: {len(index.units):,} đoạn, {len(index.inv):,} token")

    sums: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[str, str], int] = defaultdict(int)

    total_gen = len(prompts) * len(modes)
    gen_i = 0

    with log.phase("eval_generate"):
        for pi, (src, prompt) in enumerate(prompts, 1):
            outw.emit("")
            outw.emit(f"[{src}] {prompt!r}")
            log.log(f"prompt {pi}/{len(prompts)} [{src}] {prompt[:70]!r}")

            for label, cfg in modes:
                gen_i += 1
                log.log(f"▶ gen {gen_i}/{total_gen} {label}")
                t0 = time.time()
                ids = generate(state, prompt, max_len=args.max_len, config=cfg, multi=True)
                dt = time.time() - t0
                out_text = decode_surface(state, ids)
                st = _stats(prompt, out_text, index)
                key = (src, label)
                for k, v in st.items():
                    sums[key][k] += float(v)
                counts[key] += 1
                detail = (
                    f"  - {label}: chars={int(st['chars'])} sents={int(st['sents'])} "
                    f"commas={int(st['commas'])} multi={int(st['multi'])} "
                    f"copy={st['copy']:.2f} ent_keep={st['ent_keep']:.2f} ({dt:.1f}s)"
                )
                outw.emit(detail)
                outw.emit(f"      {out_text!r}")
                log.log(
                    f"✓ gen {gen_i}/{total_gen} ({dt:.1f}s) "
                    f"copy={st['copy']:.2f} chars={int(st['chars'])} "
                    f"multi={int(st['multi'])} ent_keep={st['ent_keep']:.2f}"
                )
                log.log(f"    → {out_text!r}")

    summary_header = [
        "",
        "=== Summary (mean metrics by bucket/mode) ===",
        "bucket | mode | n | chars | sents | commas | multi% | copy | ent_keep",
        "-" * 92,
    ]
    for line in summary_header:
        outw.emit(line)

    summary_lines: list[str] = []
    for bucket, mode in sorted(counts.keys()):
        n = counts[(bucket, mode)]
        if n <= 0:
            continue
        m = {
            k: sums[(bucket, mode)][k] / n
            for k in ("chars", "sents", "commas", "multi", "copy", "ent_keep")
        }
        row = (
            f"{bucket:<12} | {mode:<24} | {n:>2d} | "
            f"{m['chars']:>5.0f} | {m['sents']:>5.2f} | {m['commas']:>6.2f} | "
            f"{(100 * m['multi']):>5.1f}% | {m['copy']:>4.2f} | {m['ent_keep']:>7.2f}"
        )
        summary_lines.append(row)
        outw.emit(row)

    log.log("=== Summary ===")
    for row in summary_lines:
        log.log(row)
    log.log(f"✓ DONE → {outw.path}")


if __name__ == "__main__":
    main()
