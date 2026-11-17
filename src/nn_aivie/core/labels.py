"""Structural label names shared by teaching and inference."""

SENTENCE_START_LABEL = "đầu câu"
BEFORE_ANCHOR_LABEL = "trước neo"
AFTER_ANCHOR_LABEL = "sau neo"  # legacy fallback
NEXT_SLOT_LABEL = "next_slot"
MUST_COVER_LABEL = "must_cover"
ANSWER_COMPOUND_LABEL = "answer_compound"


def segment_label_name(seg: int) -> str:
    return f"seg_{seg}"
