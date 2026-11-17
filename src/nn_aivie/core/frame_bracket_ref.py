"""TK_037 — FRAME_BRACKET endpoint refs (token / ISG cluster / learned phrase)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nn_aivie.core.seg_graph import CLUSTER_OFFSET, SegmentGraph, SlotRef

PHRASE_OFFSET = 3 * 10**7

FrameKind = Literal["token", "cluster", "phrase"]


@dataclass(frozen=True)
class FrameEndpoint:
    kind: FrameKind
    tokens: tuple[int, ...]

    def adj_id(self) -> int:
        if self.kind == "token":
            return self.tokens[0]
        if self.kind == "cluster":
            return self.tokens[0]
        return self.tokens[0]

    def label_tokens(self) -> tuple[int, ...]:
        if self.kind == "token":
            return self.tokens
        if self.kind == "cluster":
            return self.tokens[1:]
        return self.tokens[1:]


def is_cluster_adj_id(eid: int) -> bool:
    return CLUSTER_OFFSET <= eid < PHRASE_OFFSET


def is_phrase_adj_id(eid: int) -> bool:
    return eid >= PHRASE_OFFSET


def phrase_adj_id(phrase_idx: int) -> int:
    return PHRASE_OFFSET + phrase_idx


def cluster_adj_id(cluster_idx: int) -> int:
    return CLUSTER_OFFSET + cluster_idx


def endpoint_from_slot(ref: SlotRef, seg: SegmentGraph) -> FrameEndpoint | None:
    if ref.kind == "token":
        return FrameEndpoint("token", (ref.id,))
    toks = seg.expand_slot(ref)
    if not toks:
        return None
    return FrameEndpoint("cluster", (ref.id, *toks))


def endpoint_from_phrase(phrase: tuple[int, ...], phrase_idx: int) -> FrameEndpoint:
    return FrameEndpoint("phrase", (phrase_adj_id(phrase_idx), *phrase))


def expand_endpoint_tokens(
    eid: int,
    endpoints: dict[int, FrameEndpoint] | None,
    seg: SegmentGraph | None,
    vocab_size: int,
) -> tuple[int, ...]:
    """Token ids to boost at generation — first token primary for cluster/phrase."""
    if endpoints and eid in endpoints:
        ep = endpoints[eid]
        if ep.kind == "token":
            return ep.tokens
        return ep.label_tokens()
    if is_cluster_adj_id(eid) and seg is not None:
        ci = eid - CLUSTER_OFFSET
        if 0 <= ci < len(seg.clusters):
            c = seg.clusters[ci]
            return (c.left, c.right)
    if eid < vocab_size:
        return (eid,)
    return ()


def decode_endpoint_label(
    eid: int,
    endpoints: dict[int, FrameEndpoint] | None,
    vocab,
) -> str:
    """Nhãn HIỂN THỊ của endpoint (trace/viz/audit — Luật 0.0: không đường inference nào đọc).

    TK_041: nhận `vocab` thay `token_of` — nhãn phải qua `vocab.detokenize` (detokenizer DUY
    NHẤT, §2.6). Nối tay bằng `" ".join` trên vocab BPE sẽ hiện `cl:Việt ĠNam` (lộ marker)
    thay vì `cl:Việt Nam`.
    """
    if endpoints and eid in endpoints:
        ep = endpoints[eid]
        if ep.kind == "cluster":
            return "cl:" + vocab.detokenize(ep.label_tokens())
        if ep.kind == "phrase":
            return "ph:" + vocab.detokenize(ep.label_tokens())
        return vocab.token_of(ep.tokens[0])
    if is_cluster_adj_id(eid):
        return f"cl:{eid - CLUSTER_OFFSET}"
    if is_phrase_adj_id(eid):
        return f"ph:{eid - PHRASE_OFFSET}"
    return token_of(eid)
