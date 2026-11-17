"""Node kind labels for graph export."""

from __future__ import annotations

from nn_aivie.core.vocab import (
    SPECIAL_EOS,
    SPECIAL_UNK,
    is_rule_token,
    is_surface_token,
    is_var_token,
    is_concept_token,
)


def node_kind(token: str) -> str:
    if is_rule_token(token):
        return "rule"
    if is_var_token(token):
        return "var"
    if is_concept_token(token):
        return "concept"
    if token.startswith("Lbl:"):
        return "label"
    if token.startswith("R_-"):
        return "position"
    if token in (SPECIAL_EOS, SPECIAL_UNK):
        return "special"
    if is_surface_token(token):
        return "surface"
    return "other"
