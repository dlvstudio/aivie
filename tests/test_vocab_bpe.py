"""TK_041 B2 — vocab BPE: phân vùng id, fold không-FOLD_STRIP, detokenize duy nhất."""

from __future__ import annotations

import pytest

from nn_aivie.core.tokenizer import SPECIAL_EOS, SPECIAL_UNK, BPETokenizer
from nn_aivie.core.vocab import Vocabulary

CORPUS = [
    "Hà Nội là thủ đô của Việt Nam .",
    "Nhà Trần đánh thắng quân Nguyên năm 1288 .",
    "Con mèo màu trắng , nó thích ngủ .",
    "Hỏi: Thủ đô Việt Nam là gì ? Đáp: Hà Nội .",
]


@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    return Vocabulary(BPETokenizer.train(CORPUS * 30, vocab_size=900))


# ---- phân vùng id (§2.4) ------------------------------------------------------------


def test_hidden_specials_first(vocab: Vocabulary) -> None:
    assert vocab.token_of(0) == SPECIAL_UNK
    assert vocab.token_of(1) == SPECIAL_EOS
    assert not vocab.is_surface(0) and not vocab.is_surface(1)


def test_is_surface_is_int_compare(vocab: Vocabulary) -> None:
    """Luật 0.0 (README §0.0 mục B): inference KHÔNG re-parse token-string."""
    assert vocab.is_bpe
    ent = vocab.entity_id("Hồ Chí Minh")
    assert ent >= vocab._surface_hi
    assert not vocab.is_surface(ent)
    assert vocab.is_surface(vocab.id_of("Ġthủ") if "Ġthủ" in vocab._token_to_id else 5)


def test_structural_nodes_live_above_bpe_range(vocab: Vocabulary) -> None:
    for tid in [
        vocab.entity_id("Nhà Trần"),
        vocab.concept_id("mèo"),
        vocab.pos_id("danh từ"),
        vocab.sense_id("bảo", 1),
        vocab.position_id(3),
    ]:
        assert tid >= vocab._surface_hi
        assert not vocab.is_surface(tid)


def test_surface_range_closed_under_text(vocab: Vocabulary) -> None:
    """Nạp text lạ KHÔNG mọc thêm id bề mặt — vocab bề mặt ĐÓNG (§2.4)."""
    hi = vocab._surface_hi
    n_before = len(vocab)
    ids = vocab.encode_text("một câu lạ hoắc với 🦖 và 98765")
    assert all(i < hi for i in ids)
    assert len(vocab) == n_before


def test_surface_mask_matches_is_surface(vocab: Vocabulary) -> None:
    vocab.entity_id("X")  # buộc vocab mọc → mask phải rebuild đúng
    mask = vocab.surface_mask()
    assert len(mask) == len(vocab)
    for i in range(len(vocab)):
        assert bool(mask[i]) == vocab.is_surface(i)


# ---- fold (§2.5) — FOLD_STRIP đã chết ------------------------------------------------


def test_fold_groups_case_and_space_marker(vocab: Vocabulary) -> None:
    """`Ġthủ` ≡ `Thủ` ≡ `thủ` — cùng fold-group."""
    gids = {vocab.fold_group_id(vocab.id_of(t)) for t in ["Ġthủ", "Thủ", "thủ"]}
    assert len(gids) == 1


def test_punctuation_folds_are_distinct(vocab: Vocabulary) -> None:
    """Bug cũ: `fold_of('.') == fold_of('?') == ''` → MỌI dấu câu là MỘT token với mạng
    → `_longest_suffix_cut` cắt trúng mọi dấu → agenda RỖNG. FOLD_STRIP xoá ⇒ hết theo
    CẤU TRÚC, không phải nhờ vá `folded or token.lower()`."""
    gids = [vocab.fold_group_id(vocab.id_of(t)) for t in [".", ",", "?", ":", "!"]]
    assert len(set(gids)) == len(gids)


def test_fold_never_empty(vocab: Vocabulary) -> None:
    for t in [".", "Ġ", "Ġ.", "thủ", "Ġthủ"]:
        assert vocab.fold_of(t) != ""


def test_cap_class_ignores_space_marker(vocab: Vocabulary) -> None:
    """cap-gate đại từ (`discourse.py` ngưỡng 0.12) đọc cap_class — marker không được
    làm lệch nó."""
    assert vocab.cap_class(vocab.id_of("ĠHà")) == vocab.cap_class(vocab.id_of("Hà")) == 1
    assert vocab.cap_class(vocab.id_of("Ġthủ")) == 0


# ---- detokenize (§2.6) ---------------------------------------------------------------


@pytest.mark.parametrize("text", CORPUS)
def test_detokenize_roundtrip(vocab: Vocabulary, text: str) -> None:
    """Không còn `Hà Nội .` — khoảng trắng là cái đã NẠP, không phải cái ta BỊA."""
    assert vocab.detokenize(vocab.encode_text(text)) == text


def test_detokenize_drops_structural_not_spaces(vocab: Vocabulary) -> None:
    ids = vocab.encode_text("Hà Nội là thủ đô")
    out = vocab.detokenize([vocab.entity_id("Hà Nội"), *ids, vocab.id_of(SPECIAL_EOS)])
    assert out == "Hà Nội là thủ đô"


def test_no_space_before_period(vocab: Vocabulary) -> None:
    assert vocab.detokenize(vocab.encode_text("Hà Nội .")) == "Hà Nội ."
    assert vocab.detokenize(vocab.encode_text("Hà Nội.")) == "Hà Nội."


# ---- TRAIN ≡ PROMPT: cùng một đường tokenize (câu hỏi tác giả 2026-07-16) --------------


@pytest.mark.parametrize("text", CORPUS)
def test_prompt_encodes_identically_to_train(vocab: Vocabulary, text: str) -> None:
    """Prompt lúc SINH phải ra ĐÚNG ids mà TRAIN đã nạp — nếu không, mạng nhận một chuỗi
    chưa từng tồn tại và mọi cạnh W_seq/cha đều trật.

    Đây là bug #3 (bắt bằng đọc tay, không test nào đỏ): đường sinh còn
    `prompt.strip().split()` → `'Hỏi: Thủ đô'` thành `'Hỏi:Thủđô'`. Test này khoá nó lại.
    """
    from nn_aivie.core.state import GraphState
    from nn_aivie.training.ingest import encode_unit

    state = GraphState(vocab=vocab)
    train_ids = encode_unit(state, text)[:-1]  # bỏ EOS
    assert vocab.encode_prompt(text) == train_ids


def test_prompt_robust_to_stray_whitespace(vocab: Vocabulary) -> None:
    """Khoảng trắng thừa của người dùng KHÔNG được đẩy prompt ra ngoài phân bố train.

    Trước TK_041, `.strip()` chỉ là thừa. Nay khoảng trắng LÀ token → `'  Hỏi:'` là
    `['Ġ','Ġ','Hỏi:']` ≠ `['Hỏi:']`. Train chuẩn hoá qua `normalize_ws_keep_newlines`,
    nên prompt phải chuẩn hoá y hệt. (Tính vô-tổn-hao giữ trung thành cả rác đầu vào —
    nên chuẩn hoá phải làm TRƯỚC khi encode.)
    """
    base = "Hỏi: Thủ đô Việt Nam là gì ? Đáp:"
    for messy in [f"   {base}   ", f"{base}\n", f"Hỏi:  Thủ   đô Việt Nam là gì ? Đáp:"]:
        assert vocab.encode_prompt(messy) == vocab.encode_prompt(base), repr(messy)


# ---- legacy còn sống ------------------------------------------------------------------


def test_legacy_vocab_unchanged() -> None:
    """Checkpoint tiền-TK_041 (`grade_v3*.pkl`) phải còn load/test được."""
    v = Vocabulary()
    assert not v.is_bpe
    assert v.token_of(0) == SPECIAL_UNK and v.token_of(1) == SPECIAL_EOS
    a = v.id_of("Việt")
    assert v.is_surface(a)
    assert not v.is_surface(v.entity_id("Việt Nam"))
    with pytest.raises(RuntimeError):
        v.encode_text("không có tokenizer")
