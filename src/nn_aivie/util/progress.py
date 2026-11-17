"""Progress + timing logging cho eval/train — biết đang ở giai đoạn nào (TK_011 Phase A)."""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LOG_DIR = os.environ.get("AIVIE_LOG_DIR", "out/logs")


@dataclass
class PerfSummary:
    """Gom thời gian theo section (+ per-case) — export JSON so sánh trước/sau tối ưu."""

    sections: dict[str, float] = field(default_factory=dict)
    cases: dict[str, list[dict]] = field(default_factory=dict)

    def add_section(self, name: str, seconds: float) -> None:
        self.sections[name] = self.sections.get(name, 0.0) + seconds

    def add_case(self, section: str, label: str, seconds: float) -> None:
        self.cases.setdefault(section, []).append(
            {"label": label[:80], "seconds": round(seconds, 3)}
        )

    def to_payload(self) -> dict:
        return {
            "sections": {k: round(v, 3) for k, v in self.sections.items()},
            "total_seconds": round(sum(self.sections.values()), 3),
            "cases": self.cases,
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def print_summary(self) -> None:
        print("\n=== Perf summary (section → s) ===", flush=True)
        for k, v in sorted(self.sections.items(), key=lambda x: -x[1]):
            print(f"  {k:22s} {v:7.1f}s", flush=True)
        print(f"  {'TOTAL':22s} {sum(self.sections.values()):7.1f}s", flush=True)


@contextmanager
def stage(name: str, *, quiet: bool = False, summary: PerfSummary | None = None):
    """Context manager in '▶ {name}' … '✓ {name} ({elapsed}s)'; cộng vào summary nếu có."""
    t0 = time.time()
    if not quiet:
        print(f"  ▶ {name} …", flush=True)
    try:
        yield
    finally:
        dt = time.time() - t0
        if summary is not None:
            summary.add_section(name, dt)
        if not quiet:
            print(f"  ✓ {name}  ({dt:.1f}s)", flush=True)


# alias theo tên trong ticket (StageTimer)
StageTimer = stage


def case_progress(section: str, i: int, n: int, label: str, elapsed: float | None = None) -> None:
    """In tiến độ từng ca: '[essay 3/8] 45.2s — Nêu những lợi ích…'."""
    t = f" {elapsed:.1f}s" if elapsed is not None else ""
    print(f"  [{section} {i}/{n}]{t} — {label[:60]}", flush=True)
    sys.stdout.flush()


case_line = case_progress


class FileProgressLog:
    """Ghi log tiến trình ra file (append) + stdout — rebuild/train dài."""

    def __init__(self, path: str | Path, *, also_stdout: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.also_stdout = also_stdout
        self._t0 = time.time()

    def log(self, msg: str) -> None:
        elapsed = time.time() - self._t0
        line = f"[{elapsed:8.1f}s] {msg}\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
        if self.also_stdout:
            print(line.rstrip(), flush=True)

    def progress(
        self,
        stage: str,
        done: int,
        total: int,
        *,
        extra: str = "",
    ) -> None:
        pct = 100.0 * done / max(1, total)
        msg = f"{stage} {done}/{total} ({pct:.1f}%)"
        if extra:
            msg += f" — {extra}"
        self.log(msg)

    @contextmanager
    def phase(self, name: str, *, summary: PerfSummary | None = None):
        """Log '▶ name' … '✓ name (…s)' vào file + stdout; cộng vào summary."""
        self.log(f"▶ {name}")
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            if summary is not None:
                summary.add_section(name, dt)
            self.log(f"✓ {name}  ({dt:.1f}s)")


def get_run_logger(
    name: str, *, also_stdout: bool = True, log_dir: str | Path | None = None
) -> FileProgressLog:
    """Logger dùng CHUNG cho mọi process dài (train/ingest/eval/build/generate).

    Ghi `out/logs/<name>_<YYYYmmdd_HHMMSS>.log` (append) + stdout, prefix elapsed,
    flush mỗi dòng → biết mạng đang xử lý đến đâu và soi tính đúng đắn lúc chạy.
    Đổi thư mục qua `AIVIE_LOG_DIR`. TK_011 Phase A (mở rộng cho MỌI process).
    """
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(log_dir or DEFAULT_LOG_DIR) / f"{name}_{ts}.log"
    lg = FileProgressLog(path, also_stdout=also_stdout)
    lg.log(f"=== {name} START (pid={os.getpid()}) → {path} ===")
    return lg
