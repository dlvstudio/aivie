"""TK_036 — gap window cho skip span mining và query match."""

SKIP_MIN_GAP = 2
SKIP_MAX_GAP = 8
SKIP_MAX_RIGHT_LEN = 4


def skip_right_tuple(right: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Chuẩn hoá đầu phải skip span — luôn tuple token (cụm hoặc đơn)."""
    if isinstance(right, tuple):
        return right
    if isinstance(right, list):
        return tuple(int(x) for x in right)
    return (int(right),)
