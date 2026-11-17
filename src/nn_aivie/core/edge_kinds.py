"""Edge kinds and inference priority weights."""

from __future__ import annotations

from enum import Enum


class EdgeKind(str, Enum):
    GRAMMAR = "grammar"
    SECONDARY = "secondary"
    POSITION = "position"
    COLLOCATION = "collocation"
    CO_ACTIVATION = "co_activation"
    REINFORCED = "reinforced"
    ANCHOR = "anchor"
    EMERGENT = "emergent"
    # W_seq — ma trận chuyển dịch vị trí có hướng (bigram a→b) cho kênh micro/fluency
    # của lõi sinh hybrid v3. Populate ở G1 passive; đọc có row-normalize. Tách riêng,
    # KHÔNG nằm trong STRUCTURAL/ACTIVATION/LEARNING để không nhiễm readout HCA cũ.
    SEQUENTIAL = "sequential"
    # TK_015 (D1) — substrate từ điển. Tách riêng như SEQUENTIAL: đọc qua kênh riêng, KHÔNG
    # cộng vào readout structural (priority 0). SENSE: lemma↔sense-neuron; POS_ANCHOR: sense→
    # nơ-ron từ-loại (hub); SYNONYM/ANTONYM: cạnh nghĩa tường minh; LEXATTR: từ→lớp-từ.
    SENSE = "sense"
    POS_ANCHOR = "pos_anchor"
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    LEXATTR = "lexattr"
    # TK_016 (D3) — link ngang XA có hướng: token khung không-kề (nếu→thì, Hỏi:→Đáp:).
    FRAME_BRACKET = "frame_bracket"
    # TK_045 P7 — trong câu: mọi cặp có hướng + khoảng cách (đọc kênh riêng, không readout HCA).
    SENT_L2R = "sent_l2r"  # t_i → t_j (i<j), weight f(d)
    SENT_R2L = "sent_r2l"  # t_j → t_i (i<j), weight f(d)
    # TK_045 P0.2 — hết câu k → đầu câu k+1 (cùng episode).
    SENT_SUCC = "sent_succ"
    # TK_053 §158 — canh LOAI SUY thu dong: c->Y suy ra tu bao dong tuong tu
    # tren W_skip luc bake (c ~ a qua ngu canh chung, a->Y, => c->Y). Ton tai
    # trong do thi de thuat toan sau cung co; INFERENCE_PRIORITY = 0.0 nen HIEN
    # TAI KHONG tham gia sinh (dung yeu cau tac gia: "phai co canh o do, con
    # yeu nen khong tham gia").
    ANALOGY = "analogy"


ALL_KINDS: tuple[EdgeKind, ...] = tuple(EdgeKind)

STRUCTURAL_KINDS: tuple[EdgeKind, ...] = (
    EdgeKind.GRAMMAR,
    EdgeKind.ANCHOR,
    EdgeKind.SECONDARY,
    EdgeKind.POSITION,
)

LEARNING_KINDS: tuple[EdgeKind, ...] = (
    EdgeKind.CO_ACTIVATION,
    EdgeKind.REINFORCED,
    EdgeKind.EMERGENT,
)

ACTIVATION_KINDS: tuple[EdgeKind, ...] = STRUCTURAL_KINDS + (EdgeKind.CO_ACTIVATION,)

READOUT_CO_SCALE = 0.25
READOUT_RF_SCALE = 0.35
# Composition #3 — emergent (consolidated) edges contribute to readout at a small
# scale so generalization nudges, never dominates, taught reinforced paths.
READOUT_EM_SCALE = 0.12

# Inference priority π — scales contribution per kind at read/match time.
INFERENCE_PRIORITY: dict[EdgeKind, float] = {
    EdgeKind.GRAMMAR: 1.0,
    EdgeKind.ANCHOR: 0.95,
    EdgeKind.SECONDARY: 0.35,
    EdgeKind.POSITION: 0.40,
    EdgeKind.COLLOCATION: 0.90,
    EdgeKind.CO_ACTIVATION: 0.20,
    EdgeKind.REINFORCED: 0.50,
    EdgeKind.EMERGENT: 0.15,
    EdgeKind.SEQUENTIAL: 0.0,  # đọc riêng qua kênh micro, không cộng vào readout structural
    # TK_015 D1 — kênh riêng (đọc tường minh), không cộng readout structural
    EdgeKind.SENSE: 0.0,
    EdgeKind.POS_ANCHOR: 0.0,
    EdgeKind.SYNONYM: 0.0,
    EdgeKind.ANTONYM: 0.0,
    EdgeKind.LEXATTR: 0.0,
    EdgeKind.FRAME_BRACKET: 0.0,  # đọc riêng qua kênh D3, không cộng readout structural
    EdgeKind.SENT_L2R: 0.0,  # TK_045 — kênh matrix_core / sent_intra
    EdgeKind.SENT_R2L: 0.0,
    EdgeKind.SENT_SUCC: 0.0,
    EdgeKind.ANALOGY: 0.0,  # ton tai, chua tham gia sinh
}

CO_FIRE_SCALE = 0.1
COLLOCATION_WEIGHT = 0.75


def priority_weight(kind: EdgeKind, w: float) -> float:
    """Apply inference priority; LTD (negative REINFORCED) keeps sign."""
    pi = INFERENCE_PRIORITY.get(kind, 1.0)
    return w * pi
