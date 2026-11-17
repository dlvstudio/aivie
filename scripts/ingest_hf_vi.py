#!/usr/bin/env python3
"""
Ingest dữ liệu thật tiếng Việt từ Hugging Face → jsonl đã lọc chất lượng.

Nguồn (preset):
  wiki      — Wikipedia VI (wikimedia/wikipedia 20231101.vi), tách câu → statements.jsonl
  sentences — thanhdath/vietnamese-sentences (câu Việt) → statements.jsonl
  viquad    — bộ QA đọc hiểu (mặc định taidng/UIT-ViQuAD2.0) → qa.jsonl

Mọi câu qua cổng `data/quality.py`. Kết quả ghi vào data/real_vi/{statements,qa}.jsonl (dedup).
Lưu ý: tải HF có thể chậm — chạy nền và để `--max` hợp lý (POC ~10–20k).

Ví dụ:
  python scripts/ingest_hf_vi.py --source sentences --max 8000
  python scripts/ingest_hf_vi.py --source wiki --max 12000
  python scripts/ingest_hf_vi.py --source viquad --max 4000
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from nn_aivie.data.quality import is_quality_vi_sentence, normalize_ws, split_sentences

# A2/anaphora: câu có đại từ hồi chỉ ngay sau dấu phẩy — nguồn học `, → nó/chúng/họ`.
_ANAPHORA_RE = re.compile(r",\s*(nó|chúng|họ|chúng nó|chúng ta)\b", re.IGNORECASE)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "real_vi"

WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.vi"
# Gold dài cho t1 SFT (Trụ 7 T2 — TK_004): instruction VI → bài luận nhiều đoạn
ALPACA_REPO = "bkai-foundation-models/vi-alpaca"
# ViQuAD: RC dataset cho QA held-out (TK_042)
VIQUAD_REPO = "taidng/UIT-ViQuAD2.0"
# datasets-server REST: trả rows tức thì (KHÔNG cần builder nặng → nhanh hơn streaming nhiều)
_ROWS_API = "https://datasets-server.huggingface.co/rows"

# Logger tiến trình (set trong main) — network fetch chậm, cần biết đang tải đến đâu.
RUN_LOG = None


def _log(msg: str) -> None:
    if RUN_LOG is not None:
        RUN_LOG.log(msg)
    else:
        print(msg, flush=True)


def _hf_rows(repo: str, max_rows: int, page: int = 100, split: str = "train"):
    """Stream row dict qua datasets-server /rows (phân trang, retry) — dùng chung mọi repo.
    `split` mặc định 'train' (giữ mọi caller cũ); build_qa_heldout dùng 'test'."""
    from urllib.parse import quote

    offset = 0
    while offset < max_rows:
        length = min(page, max_rows - offset)
        url = (
            f"{_ROWS_API}?dataset={quote(repo, safe='')}&config=default"
            f"&split={quote(split, safe='')}&offset={offset}&length={length}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "nn-aivie"})
        for attempt in range(3):
            try:
                data = json.load(urllib.request.urlopen(req, timeout=40))
                break
            except Exception:
                if attempt == 2:
                    return
                time.sleep(2)
        rows = data.get("rows", [])
        if not rows:
            return
        for r in rows:
            yield r.get("row_idx", offset), r.get("row", {})
        offset += length
        _log(f"[hf/{repo}] fetched {offset}/{max_rows} rows")


def ingest_vi_alpaca(max_n: int, *, scan_limit: int = 0, min_sentences: int = 3) -> int:
    """
    Gold dài t1 SFT (Trụ 7 T2): instruction + bài luận → essays.jsonl (câu đã lọc, GIỮ THỨ TỰ).

    Lọc THUẦN CẤU TRÚC (không regex từ khóa lệnh — Luật 0 cho cả heuristic data): giữ item có
    ≥ `min_sentences` câu qua `is_quality_vi_sentence` — bài liệt kê từ vựng / code / đáp ngắn
    tự rớt vì câu không đạt gate. Ghi {instruction, input, sentences[], doc_id}.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "essays.jsonl"
    seen = _load_existing(out, "instruction")

    kept = scanned = 0
    with out.open("a", encoding="utf-8") as f:
        for row_idx, row in _hf_rows(ALPACA_REPO, max_rows=scan_limit or 10 * max_n):
            scanned += 1
            instr = normalize_ws(row.get("instruction", "") or "")
            extra = normalize_ws(row.get("input", "") or "")
            output = row.get("output", "") or ""
            if not instr or instr in seen or not output:
                continue
            sentences: list[str] = []
            for para in output.split("\n"):
                for s in split_sentences(para):
                    s = normalize_ws(s)
                    if is_quality_vi_sentence(s):
                        sentences.append(s)
            if len(sentences) < min_sentences:
                continue
            seen.add(instr)
            f.write(json.dumps(
                {"instruction": instr, "input": extra, "sentences": sentences,
                 "doc_id": f"alpaca:{row_idx}"},
                ensure_ascii=False,
            ) + "\n")
            kept += 1
            if kept >= max_n:
                break
    print(f"vi_alpaca: scanned {scanned} rows")
    return kept


def _wiki_articles(max_articles: int, page: int = 100, start_offset: int = 0):
    """Stream (doc_id, title, text) bài Wikipedia VI qua datasets-server /rows (phân trang).

    doc_id/title giữ cho tầng t1 (nơ-ron trừu tượng episode — v4 Trụ 7): câu cùng bài = cùng
    episode; E mức episode (đồng hiện trong-bài) nhắm synonym paradigmatic (T0).
    `start_offset`>0: bỏ qua N bài đầu (đã kéo trước) → lấy VÙNG MỚI, không re-scan (TK_027 bổ data).
    """
    offset = start_offset
    end = start_offset + max_articles
    while offset < end:
        length = min(page, end - offset)
        url = (
            f"{_ROWS_API}?dataset=wikimedia%2Fwikipedia&config={WIKI_CONFIG}"
            f"&split=train&offset={offset}&length={length}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "nn-aivie"})
        for attempt in range(3):
            try:
                data = json.load(urllib.request.urlopen(req, timeout=40))
                break
            except Exception:
                if attempt == 2:
                    return
                time.sleep(2)
        rows = data.get("rows", [])
        if not rows:
            return
        for r in rows:
            row = r.get("row", {})
            did = str(row.get("id", "") or f"off{r.get('row_idx', offset)}")
            yield did, (row.get("title", "") or ""), (row.get("text", "") or "")
        offset += length
        _log(f"[hf/wiki] fetched {offset}/{max_articles} articles")


def _load_existing(path: Path, key: str) -> set[str]:
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)[key])
                except Exception:
                    pass
    return seen


def _wiki_data_files() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(WIKI_REPO, repo_type="dataset")
    shards = sorted(
        f for f in files if f.startswith(f"{WIKI_CONFIG}/") and f.endswith(".parquet")
    )
    base = f"https://huggingface.co/datasets/{WIKI_REPO}/resolve/main/"
    return [base + s for s in shards]


def ingest_statements(
    source: str,
    max_n: int,
    *,
    max_articles: int = 20000,
    start_offset: int = 0,
    anaphora_only: bool = False,
    scan_limit: int = 0,
    out_name: str = "statements.jsonl",
) -> int:
    """Ghi câu chất lượng vào statements.jsonl.

    `anaphora_only=True`: chỉ giữ câu có đại từ hồi chỉ sau dấu phẩy (`, nó/chúng/họ`) — dùng
    để gom dữ liệu A2 (tần suất thấp ~0.1%, cần `scan_limit` lớn). `scan_limit`>0 = trần số câu
    duyệt qua để không stream vô hạn.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    seen = _load_existing(out, "text")

    if source == "wiki":
        articles = _wiki_articles(max_articles, start_offset=start_offset)
    else:  # sentences (general — lưu ý: thiên tin tức, không thuần kiến thức)
        from datasets import load_dataset

        ds = load_dataset("thanhdath/vietnamese-sentences", split="train", streaming=True)
        articles = (("", "", row.get("text", "") or "") for row in ds)

    kept = scanned = 0
    with out.open("a", encoding="utf-8") as f:
        for did, title, raw in articles:
            # wiki: tách ĐOẠN trước (ranh giới \n của dump) rồi mới tách câu — episode t1 =
            # đoạn văn topically-coherent, không phải cả bài trộn trăm chủ đề (đo T0 lần 1:
            # 1118 bài/45k câu → term×doc quá thô, E-episode âm tính; đoạn cho ngữ cảnh mịn).
            if source == "wiki":
                paras = [(pi, p) for pi, p in enumerate(raw.split("\n")) if p.strip()]
            else:
                paras = [(0, raw)]
            for pi, para in paras:
                sentences = split_sentences(para) if source == "wiki" else [para]
                for s in sentences:
                    scanned += 1
                    if scan_limit and scanned > scan_limit:
                        return kept
                    s = normalize_ws(s)
                    if s in seen or not is_quality_vi_sentence(s):
                        continue
                    if anaphora_only and not _ANAPHORA_RE.search(s):
                        continue
                    seen.add(s)
                    rec: dict = {"text": s}
                    if did:  # nhóm episode cho tầng t1 / E-episode (v4 Trụ 7)
                        rec["doc_id"] = did
                        rec["para"] = pi
                        if title:
                            rec["title"] = title
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    if kept >= max_n:
                        return kept
    return kept


def ingest_viquad(repo: str, max_n: int) -> int:
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "qa.jsonl"
    seen = _load_existing(out, "question")
    ds = load_dataset(repo, split="train", streaming=True)

    kept = 0
    with out.open("a", encoding="utf-8") as f:
        for row in ds:
            q = normalize_ws(row.get("question", "") or "")
            ans = row.get("answers") or {}
            a = ""
            if isinstance(ans, dict) and ans.get("text"):
                a = ans["text"][0]
            elif isinstance(row.get("answer"), str):
                a = row["answer"]
            a = normalize_ws(a)
            if not q or not a or q in seen:
                continue
            # câu hỏi cần là tiếng Việt; đáp ngắn gọn
            if len(q.split()) < 3 or len(a.split()) > 24:
                continue
            seen.add(q)
            f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= max_n:
                break
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest real Vietnamese HF data → jsonl.")
    ap.add_argument("--source", choices=["wiki", "sentences", "viquad", "vi_alpaca"], required=True)
    ap.add_argument("--max", type=int, default=10000)
    ap.add_argument("--repo", default="taidng/UIT-ViQuAD2.0", help="repo cho viquad")
    ap.add_argument(
        "--anaphora-only",
        action="store_true",
        help="chỉ giữ câu có `, nó/chúng/họ` (dữ liệu A2 — cần scan-limit lớn)",
    )
    ap.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="trần số câu duyệt (0 = không giới hạn); dùng với --anaphora-only",
    )
    ap.add_argument("--max-articles", type=int, default=20000, help="trần số bài wiki quét")
    ap.add_argument("--start-offset", type=int, default=0, help="bỏ N bài đầu (lấy vùng mới)")
    ap.add_argument(
        "--out-name",
        default="statements.jsonl",
        help="tên file output trong data/real_vi (vd anaphora.jsonl)",
    )
    args = ap.parse_args()

    global RUN_LOG
    from nn_aivie.util.progress import get_run_logger

    RUN_LOG = get_run_logger(f"ingest_hf_{args.source}")
    RUN_LOG.log(f"source={args.source} max={args.max} out={args.out_name}")

    if args.source == "viquad":
        n = ingest_viquad(args.repo, args.max)
        RUN_LOG.log(f"viquad: wrote {n} QA pairs → {OUT_DIR/'qa.jsonl'}")
    elif args.source == "vi_alpaca":
        n = ingest_vi_alpaca(args.max, scan_limit=args.scan_limit)
        RUN_LOG.log(f"vi_alpaca: wrote {n} essays → {OUT_DIR/'essays.jsonl'}")
    else:
        n = ingest_statements(
            args.source,
            args.max,
            max_articles=args.max_articles,
            start_offset=args.start_offset,
            anaphora_only=args.anaphora_only,
            scan_limit=args.scan_limit,
            out_name=args.out_name,
        )
        RUN_LOG.log(f"{args.source}: wrote {n} sentences → {OUT_DIR/args.out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
