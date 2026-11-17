"""
Đặc tả dataset train — chỉ path + tham số ngoài (priority, eta, …).

Mọi file corpus dùng schema: mỗi dòng jsonl = ["text", ...].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PRIORITY = 0.6
DEFAULT_ETA = 0.05

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "train_v10.yaml"


@dataclass
class DatasetSpec:
    path: str
    priority: float = DEFAULT_PRIORITY
    eta: float = DEFAULT_ETA
    split_punct: bool = True
    max_texts: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetSpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        core = {k: v for k, v in raw.items() if k in known and k != "extra"}
        extra = dict(raw.get("extra") or {})
        for k, v in raw.items():
            if k not in known:
                extra[k] = v
        return cls(extra=extra, **core)

    @classmethod
    def load_manifest(cls, path: Path) -> list["DatasetSpec"]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "datasets" in data:
            items = data["datasets"]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError(f"manifest phải là list hoặc {{datasets: [...]}}: {path}")
        return [cls.from_dict(item) for item in items]
