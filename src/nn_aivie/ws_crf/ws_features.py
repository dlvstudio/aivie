from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WSFeatureConfig:
    """Feature template gần giống underthesea (±2 context + lower/digit/title + is_in_dict)."""

    window: int = 2


def _istitle_token(tok: str) -> bool:
    """Title-case theo token đầu (đủ dùng cho tiếng Việt kiểu 'Hà', 'Nội')."""
    return bool(tok) and tok[0].isupper()


def build_ws_features(
    tokens: list[str],
    idx: int,
    *,
    dictionary_lower: set[str],
    cfg: WSFeatureConfig = WSFeatureConfig(),
) -> dict[str, str]:
    """
    CRFsuite expects feature dict {name: value}. Value thường là "1" (present).

    Template tương đương nhóm features trong underthesea:
    - T[k], T[k].lower
    - T[k].isdigit, T[k].istitle
    - T[k].is_in_dict
    - bigram T[i,j] và các biến thể `.lower/.is_in_dict` cho một số cặp
    """
    feats: dict[str, str] = {}

    def tok_at(off: int) -> str:
        j = idx + off
        if j < 0:
            return "__BOS__"
        if j >= len(tokens):
            return "__EOS__"
        return tokens[j]

    def add(name: str, value: str = "1") -> None:
        feats[name] = value

    # unigram ±2
    ctx = {off: tok_at(off) for off in range(-cfg.window, cfg.window + 1)}
    for off, tok in ctx.items():
        add(f"T[{off}]", tok)
        low = tok.lower()
        add(f"T[{off}].lower", low)
        add(f"T[{off}].isdigit", str(tok.isdigit()))
        add(f"T[{off}].istitle", str(_istitle_token(tok)))
        add(f"T[{off}].is_in_dict", str(low in dictionary_lower))

    # selected bigrams (mirrors underthesea list)
    pairs = [
        (-2, -1), (-1, 0), (0, 1), (1, 2),
        (-2, 0), (-1, 1), (0, 2),
    ]
    for a, b in pairs:
        ta, tb = ctx[a], ctx[b]
        add(f"T[{a},{b}]", f"{ta}|{tb}")
        add(f"T[{a},{b}].lower", f"{ta.lower()}|{tb.lower()}")
        add(
            f"T[{a},{b}].is_in_dict",
            str(f"{ta.lower()} {tb.lower()}" in dictionary_lower),
        )

    return feats

