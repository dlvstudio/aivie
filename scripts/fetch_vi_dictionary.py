#!/usr/bin/env python3
"""
Tải từ điển tiếng Việt (HF `tsdocode/vietnamese-dictionary`, 36.7k mục) qua datasets-server
/rows REST — cùng cách nạp wiki (README 0.5.7). Resumable: append raw rows, đếm dòng để tiếp.

Schema mỗi row: {word, part_of_speech, meaning, example}. Nhiều row cùng `word` = nhiều nghĩa.
Output: data/dict/vi_dict_raw.jsonl (mỗi dòng 1 nghĩa, giữ thứ tự gốc → group by word khi ingest).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "data/dict/vi_dict_raw.jsonl"
_DATASET = "tsdocode/vietnamese-dictionary"
_PAGE = 100


def _fetch(offset: int, length: int, retries: int = 8) -> list[dict]:
    """Trả rows; rỗng CHỈ khi hết data thật. Lỗi (429…) cạn retry → RAISE (main resume lại)."""
    q = urllib.parse.urlencode(
        {"dataset": _DATASET, "config": "default", "split": "train",
         "offset": offset, "length": length}
    )
    url = f"https://datasets-server.huggingface.co/rows?{q}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return [row["row"] for row in json.load(r).get("rows", [])]
        except urllib.error.HTTPError as e:  # noqa: PERF203
            last = e
            time.sleep((10.0 if e.code == 429 else 2.0) * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


def main() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    have = sum(1 for _ in _OUT.open()) if _OUT.exists() else 0
    print(f"đã có {have} dòng → tiếp từ offset {have}")
    offset = have
    with _OUT.open("a", encoding="utf-8") as f:
        while True:
            rows = _fetch(offset, _PAGE)
            if not rows:
                break
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            offset += len(rows)
            if offset % 2000 == 0 or len(rows) < _PAGE:
                print(f"  offset={offset}", flush=True)
            if len(rows) < _PAGE:
                break
            time.sleep(0.5)  # throttle chống 429
    print(f"XONG: {offset} dòng → {_OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
