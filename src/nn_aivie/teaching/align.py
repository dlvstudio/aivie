"""Align example sentences to slot spans."""

from __future__ import annotations

from dataclasses import dataclass

from nn_aivie.teaching.ast import (
    ConstSlot,
    GapSlot,
    LessonAST,
    OneOfSlot,
    Slot,
    VarSlot,
)


def tokenize(text: str) -> list[str]:
    """Tách token theo khoảng trắng; phân biệt hoa/thường (Em ≠ em — mỗi chuỗi là một neuron)."""
    return text.strip().split()


# R0: dấu câu tách thành token riêng (cho dữ liệu thật/HF, nơi `ra,` `mỏi.` dính token).
# KHÔNG dùng cho lesson alignment (pattern const `,` đã giả định dấu đã tách sẵn).
_PUNCT_CHARS = ",.;:!?"


def tokenize_punct(text: str) -> list[str]:
    """
    Như `tokenize` nhưng tách **dấu câu cuối/đầu** khỏi từ → token dấu câu độc lập.

    Cần cho A2/discourse: W_seq phải học `, → nó` với `,` là **một token**, không phải
    dính `ra,`. Chỉ tách dấu ở biên (đầu/cuối cụm), giữ nguyên `2.5`, `U.S` giữa từ.
    """
    from nn_aivie.data.qa import DAP_LABEL, HOI_LABEL

    out: list[str] = []
    for word in text.strip().split():
        if not word:
            continue
        # nhãn frame (Hỏi:/Đáp:) là token GIAO THỨC nguyên vẹn (Mục 0 — frame protocol được
        # phép tường minh). Không tách ":" — nếu tách, ingest_qa(split_punct=True) mất mốc
        # Đáp: và bỏ qua lặng lẽ mọi cặp (bug nền lộ khi TK_010 nạp Hỏi-Đáp dài).
        if word in (HOI_LABEL, DAP_LABEL):
            out.append(word)
            continue
        i = 0
        while i < len(word) and word[i] in _PUNCT_CHARS:
            out.append(word[i])
            i += 1
        word = word[i:]
        trailing: list[str] = []
        while word and word[-1] in _PUNCT_CHARS:
            trailing.append(word[-1])
            word = word[:-1]
        if word:
            out.append(word)
        out.extend(reversed(trailing))
    return out


@dataclass(frozen=True)
class SlotSpan:
    slot: Slot
    start: int
    end: int  # exclusive
    var_name: str | None = None


@dataclass(frozen=True)
class Alignment:
    tokens: list[str]
    spans: list[SlotSpan]

    def var_token_spans(self) -> dict[str, tuple[int, int]]:
        """Map variable name to token span (start, end) for imprint."""
        result: dict[str, tuple[int, int]] = {}
        for span in self.spans:
            if span.var_name:
                result[span.var_name] = (span.start, span.end)
        return result


def _should_use_const_anchor(slots: list[Slot], const_info: tuple[int, ConstSlot]) -> bool:
    """Chỉ tách câu tại const khi mệnh đề sau const là một slot vị ngữ (copula, không, đang…)."""
    ci, _ = const_info
    post = slots[ci + 1 :]
    if any(isinstance(s, OneOfSlot) for s in post):
        return True
    if any(isinstance(s, ConstSlot) for s in post):
        return False
    pre_vars = sum(1 for s in slots[:ci] if isinstance(s, VarSlot))
    post_vars = sum(1 for s in post if isinstance(s, VarSlot))
    return pre_vars >= 1 and post_vars == 1


def align_example(lesson: LessonAST, example: str) -> Alignment:
    tokens = tokenize(example)
    const_slots = [(i, s) for i, s in enumerate(lesson.slots) if isinstance(s, ConstSlot)]
    if len(const_slots) == 1 and _should_use_const_anchor(lesson.slots, const_slots[0]):
        spans = _align_const_anchor(lesson.slots, tokens, const_slots[0])
        if spans is not None:
            return Alignment(tokens=tokens, spans=spans)
    spans = _align_backtrack(lesson.slots, tokens, 0)
    if spans is None:
        raise ValueError(f"alignment failed for {example!r}")
    return Alignment(tokens=tokens, spans=spans)


def _align_const_anchor(
    slots: list[Slot],
    tokens: list[str],
    const_info: tuple[int, ConstSlot],
) -> list[SlotSpan] | None:
    """Align patterns like S … là … N|A by splitting on the const anchor."""
    ci, const_slot = const_info
    try:
        const_pos = tokens.index(const_slot.const)
    except ValueError:
        return None

    subject = tokens[:const_pos]
    predicate = tokens[const_pos + 1 :]
    if not subject or not predicate:
        return None

    pre = slots[:ci]
    post = slots[ci + 1 :]

    if not _region_matches_slots(subject, pre):
        return None
    var_name = _match_predicate_var(predicate, post)
    if var_name is None:
        return None
    post_spans = _align_backtrack(post, predicate, 0)
    pred_start = const_pos + 1
    pred_end = len(tokens)
    if post_spans is not None:
        var_spans = [s for s in post_spans if s.var_name]
        if var_spans:
            main = max(var_spans, key=lambda s: s.end - s.start)
            var_name = main.var_name
            if main.end - main.start >= len(predicate):
                pred_start = const_pos + 1 + main.start
                pred_end = const_pos + 1 + main.end

    spans: list[SlotSpan] = []
    first_var = next((s for s in pre if isinstance(s, VarSlot)), None)
    if first_var:
        spans.append(
            SlotSpan(slot=first_var, start=0, end=len(subject), var_name=first_var.var)
        )
    spans.append(SlotSpan(slot=const_slot, start=const_pos, end=const_pos + 1))
    spans.append(
        SlotSpan(
            slot=VarSlot(var=var_name),
            start=pred_start,
            end=pred_end,
            var_name=var_name,
        )
    )
    return spans


def _region_matches_slots(region: list[str], slots: list[Slot]) -> bool:
    if not region:
        return False
    min_t = 0
    for slot in slots:
        if isinstance(slot, (VarSlot, ConstSlot)):
            min_t += 1
        elif isinstance(slot, GapSlot):
            min_t += slot.min
        elif isinstance(slot, OneOfSlot):
            min_t += 1
    return len(region) >= min_t


def _match_predicate_var(predicate: list[str], slots: list[Slot]) -> str | None:
    spans = _align_backtrack(slots, predicate, 0)
    if spans is None:
        return None
    for sp in reversed(spans):
        if sp.var_name:
            return sp.var_name
    return None


def _min_tokens_for_slots(slots: list[Slot]) -> int:
    total = 0
    for slot in slots:
        if isinstance(slot, (VarSlot, ConstSlot)):
            total += 1
        elif isinstance(slot, GapSlot):
            total += slot.min
        elif isinstance(slot, OneOfSlot):
            total += 1
    return total


def _align_backtrack(
    slots: list[Slot],
    tokens: list[str],
    pos: int,
    acc: list[SlotSpan] | None = None,
) -> list[SlotSpan] | None:
    if acc is None:
        acc = []
    if not slots:
        return acc if pos == len(tokens) else None

    slot = slots[0]
    rest = slots[1:]

    if isinstance(slot, VarSlot):
        if pos >= len(tokens):
            return None
        if rest and isinstance(rest[0], ConstSlot):
            try:
                end = tokens.index(rest[0].const, pos + 1)
            except ValueError:
                return None
        elif not rest:
            end = len(tokens)
        else:
            end = pos + 1
        span = SlotSpan(slot=slot, start=pos, end=end, var_name=slot.var)
        return _align_backtrack(rest, tokens, end, acc + [span])

    if isinstance(slot, ConstSlot):
        if pos >= len(tokens) or tokens[pos] != slot.const:
            return None
        span = SlotSpan(slot=slot, start=pos, end=pos + 1)
        return _align_backtrack(rest, tokens, pos + 1, acc + [span])

    if isinstance(slot, GapSlot):
        min_need = _min_tokens_for_slots(rest)
        max_gap = len(tokens) - pos - min_need
        if max_gap < slot.min:
            return None
        for count in range(slot.min, max_gap + 1):
            span = SlotSpan(slot=slot, start=pos, end=pos + count)
            out = _align_backtrack(rest, tokens, pos + count, acc + [span])
            if out is not None:
                return out
        return None

    if isinstance(slot, OneOfSlot):
        min_need = _min_tokens_for_slots(rest)
        max_take = len(tokens) - pos - min_need
        if max_take < 1:
            return None
        best: list[SlotSpan] | None = None
        best_var_len = 0
        for var_name in slot.vars:
            for count in range(1, max_take + 1):
                var_span = SlotSpan(
                    slot=VarSlot(var=var_name),
                    start=pos,
                    end=pos + count,
                    var_name=var_name,
                )
                out = _align_backtrack(rest, tokens, pos + count, acc + [var_span])
                if out is not None and count > best_var_len:
                    best_var_len = count
                    best = out
        return best

    raise TypeError(f"unknown slot: {slot}")


def align_all_examples(lesson: LessonAST) -> list[Alignment]:
    return [align_example(lesson, ex) for ex in lesson.examples]
