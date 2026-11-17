"""
TK_012 — Content cache: nhãn/text của nơ-ron TÁCH khỏi graph thuần.

Luật 0.0 (README, 2026-07-03): mạng nơ-ron KHÔNG chứa text/label/content. Đồ thị chỉ có
id + cạnh (kiểu/trọng số) + kiểu nơ-ron + chuỗi token-id cấu trúc. Mọi text/label
(tag "alpaca:0", title bài, key episode, source) sống trong file cache RIÊNG
(`<checkpoint>.content.pkl`) — CHỈ phục vụ UI / tracing / debug.

**Bất biến bắt buộc:** inference VÀ training không đọc cache này. Chỉ eval-metric (đo "khớp
đúng bài nguồn") và viz được đọc, qua API tường minh (đánh dấu "debug read").
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParentContent:
    """Nhãn debug của một ParentRecord (id → nội dung). KHÔNG vào graph."""

    tag: str = ""  # nhãn nguồn ("alpaca:0")


@dataclass
class EpisodeContent:
    """Nhãn debug của một AbstractNeuron (episode)."""

    key: str = ""      # định danh ("wiki:4#2", "alpaca:12")
    title: str = ""    # chủ đề (title wiki / instruction SFT)
    source: str = "doc"  # doc | resonance | sft


@dataclass
class ContentCache:
    """Map neuron-id → nội dung debug. Sidecar của checkpoint, không phải graph."""

    parents: dict[int, ParentContent] = field(default_factory=dict)
    episodes: dict[int, EpisodeContent] = field(default_factory=dict)

    # ---- ghi (build-time) ----
    def set_parent_tag(self, parent_idx: int, tag: str) -> None:
        if not tag:
            return
        self.parents.setdefault(parent_idx, ParentContent()).tag = tag

    def set_episode(
        self, episode_idx: int, *, key: str = "", title: str = "", source: str = "doc"
    ) -> None:
        self.episodes[episode_idx] = EpisodeContent(key=key, title=title, source=source)

    # ---- đọc (debug/eval/UI only) ----
    def parent_tag(self, parent_idx: int) -> str:
        pc = self.parents.get(parent_idx)
        return pc.tag if pc else ""

    def episode_key(self, episode_idx: int) -> str:
        ec = self.episodes.get(episode_idx)
        return ec.key if ec else ""

    def episode_title(self, episode_idx: int) -> str:
        ec = self.episodes.get(episode_idx)
        return ec.title if ec else ""

    def episode_source(self, episode_idx: int) -> str:
        ec = self.episodes.get(episode_idx)
        return ec.source if ec else "doc"

    # ---- serialize (sidecar .pkl RIÊNG, không nằm trong GraphState pickle) ----
    def to_payload(self) -> dict:
        return {
            "parents": {pi: {"tag": pc.tag} for pi, pc in self.parents.items()},
            "episodes": {
                ei: {"key": ec.key, "title": ec.title, "source": ec.source}
                for ei, ec in self.episodes.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ContentCache":
        cache = cls()
        for pi, d in payload.get("parents", {}).items():
            cache.parents[int(pi)] = ParentContent(tag=str(d.get("tag", "")))
        for ei, d in payload.get("episodes", {}).items():
            cache.episodes[int(ei)] = EpisodeContent(
                key=str(d.get("key", "")),
                title=str(d.get("title", "")),
                source=str(d.get("source", "doc")),
            )
        return cache

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(self.to_payload(), f)

    @classmethod
    def load(cls, path: str | Path) -> "ContentCache":
        p = Path(path)
        if not p.exists():
            return cls()
        with p.open("rb") as f:
            return cls.from_payload(pickle.load(f))

    def __bool__(self) -> bool:
        return bool(self.parents or self.episodes)
