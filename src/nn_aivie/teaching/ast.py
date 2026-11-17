"""Lesson AST — canonical model for Direct Teaching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class VarSlot:
    var: str


@dataclass(frozen=True)
class ConstSlot:
    const: str


@dataclass(frozen=True)
class GapSlot:
    gap: str = "s+"
    min: int = 1


@dataclass(frozen=True)
class OneOfSlot:
    vars: tuple[str, ...]


Slot = Union[VarSlot, ConstSlot, GapSlot, OneOfSlot]


@dataclass
class LessonAST:
    pattern_id: str
    slots: list[Slot]
    examples: list[str]
    lesson_id: str = ""
    grade: int | None = None
    repeat: int = 1
    eta_teach: float = 0.5
    w_init: float = 2.0
    bind: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.pattern_id:
            raise ValueError("pattern_id is required")
        if not self.slots:
            raise ValueError("slots must not be empty")
        if not self.examples:
            raise ValueError("examples must not be empty")
        for i, slot in enumerate(self.slots):
            if not isinstance(slot, (VarSlot, ConstSlot, GapSlot, OneOfSlot)):
                raise ValueError(f"invalid slot at index {i}: {slot!r}")
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")

    def normalized_key(self) -> tuple:
        """Key for comparing lessons across formats."""
        return (
            self.pattern_id,
            tuple(_slot_key(s) for s in self.slots),
            tuple(self.examples),
            tuple(sorted(self.bind.items())),
        )


def _slot_key(slot: Slot) -> tuple:
    if isinstance(slot, VarSlot):
        return ("var", slot.var)
    if isinstance(slot, ConstSlot):
        return ("const", slot.const)
    if isinstance(slot, GapSlot):
        return ("gap", slot.gap, slot.min)
    return ("one_of", slot.vars)
