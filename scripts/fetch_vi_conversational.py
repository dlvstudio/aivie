#!/usr/bin/env python3
"""
TK_017 R3-data — tải corpus VĂN NÓI/hội thoại tiếng Việt (fix trần continuation, §13).

Nguồn (đọc qua datasets-server /rows như wiki/tsdocode; resumable + throttle):
  - `truongpdd/vietnamese_story` (99k truyện/tản văn — khuôn "nếu…thì/tuy…nhưng" tự nhiên)
  - `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` (12.7k hội thoại đa lượt)

Output: data/conv/stories_raw.jsonl ({text}), data/conv/chat_raw.jsonl ({turns:[...]}).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DIR = _ROOT / "data/conv"


def _fetch(ds: str, offset: int, length: int, retries: int = 8) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": ds, "config": "default", "split": "train",
                                "offset": offset, "length": length})
    url = f"https://datasets-server.huggingface.co/rows?{q}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return [row["row"] for row in json.load(r).get("rows", [])]
        except urllib.error.HTTPError as e:  # noqa: PERF203
            last = e
            time.sleep((10.0 if e.code == 429 else 2.0) * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


def _pull(ds: str, out: Path, max_rows: int, to_row) -> None:
    have = sum(1 for _ in out.open()) if out.exists() else 0
    print(f"{ds}: đã có {have} → tiếp tới {max_rows}", flush=True)
    offset = have
    with out.open("a", encoding="utf-8") as f:
        while offset < max_rows:
            rows = _fetch(ds, offset, min(100, max_rows - offset))
            if not rows:
                break
            for row in rows:
                f.write(json.dumps(to_row(row), ensure_ascii=False) + "\n")
            f.flush()
            offset += len(rows)
            if offset % 1000 == 0:
                print(f"  {ds} offset={offset}", flush=True)
            time.sleep(0.4)
    print(f"✓ {ds}: {offset} dòng → {out.relative_to(_ROOT)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stories", type=int, default=6000)
    ap.add_argument("--max-chat", type=int, default=4000)
    args = ap.parse_args()
    _DIR.mkdir(parents=True, exist_ok=True)
    _pull("truongpdd/vietnamese_story", _DIR / "stories_raw.jsonl", args.max_stories,
          lambda r: {"text": r.get("text", ""), "the_loai": r.get("the_loai", "")})
    _pull("5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca", _DIR / "chat_raw.jsonl", args.max_chat,
          lambda r: {"turns": [t.get("value", "") for t in r.get("conversations", [])]})


if __name__ == "__main__":
    main()
