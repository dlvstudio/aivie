"""TK_041 — 4 tính chất BẤT KHẢ XÂM PHẠM của đáy mới (core/tokenizer.py).

Nếu một test ở đây đỏ thì đáy hỏng, và mọi số đo ở tầng trên đều vô nghĩa.
"""

from __future__ import annotations

import unicodedata

import pytest

from nn_aivie.core.tokenizer import (
    _BYTE_RE,
    DAP_LABEL,
    HOI_LABEL,
    MARK_NL,
    MARK_SPACE,
    SPECIAL_EOS,
    BPETokenizer,
    normalize,
    pre_tokenize,
)

CORPUS = [
    "Hà Nội là thủ đô của Việt Nam .",
    "Nhà Trần đánh thắng quân Nguyên năm 1288 .",
    "Con mèo màu trắng , nó thích ngủ .",
    "1 cộng 1 bằng 2 .",
    "12 cộng 345 bằng 357 .",
    "Hỏi: Thủ đô Việt Nam là gì ? Đáp: Hà Nội .",
    "Lớp 3:\nTiết 1: 1 cộng 1 bằng 2\nBài 1:\n",
    "Trần Hưng Đạo ( chữ Nho : 陳興道 ) là danh tướng .",
    "Phở là món ăn phổ biến ở Việt Nam , nhưng phở Hà Nội có cách riêng .",
]


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS * 20, vocab_size=1200)


# ---- 1. VÔ-TỔN-HAO ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        *CORPUS,
        "",
        " ",
        "\n",
        "\n\n\n",
        "  hai space  và   ba   ",
        "\tTab\tvà\r\nCRLF\n",
        "dòng một\ndòng hai\n\nđoạn mới",
        "emoji 🐈 và 🇻🇳 lẫn chữ Hán 陳興道 và tiếng Nga Привет",
        "1.000.000 đồng ; 3,14 ; 2.5 ; U.S.A",
        "Ġ và Ċ là marker — text thật chứa chúng vẫn phải roundtrip",
        "<EOS> giữa câu <EOS>",
        "Hỏi: gì ? Đáp: đây .",
        "x" * 500,
    ],
)
def test_roundtrip_byte_exact(tok: BPETokenizer, text: str) -> None:
    """`decode(encode(x)) == NFC(x)` TỪNG BYTE — định nghĩa vận hành của 'không bỏ qua
    bất kỳ cái gì, kể cả dấu cách'. Tokenizer cũ KHÔNG THỂ pass test này."""
    assert tok.decode(tok.encode(text)) == normalize(text)


def test_roundtrip_on_nfd_input(tok: BPETokenizer) -> None:
    """Input NFD (`ế` = e + 2 dấu tổ hợp) → chuẩn hoá về NFC rồi roundtrip.

    Đo B0: 100% đơn vị corpus chứa dấu tổ hợp được. NFC/NFD nhìn GIỐNG HỆT nhau nhưng
    BPE tách khác hẳn — bug này ẩn suốt vì `.split()` không quan tâm.
    """
    nfd = unicodedata.normalize("NFD", "Việt Nam thống nhất")
    assert nfd != normalize(nfd)  # bảo đảm mẫu thử THẬT SỰ là NFD
    assert tok.decode(tok.encode(nfd)) == normalize(nfd)


def test_marker_chars_in_raw_text_survive(tok: BPETokenizer) -> None:
    """Marker Ġ/Ċ do tokenizer sinh; text thật chứa chúng → byte-fallback, không nhập nhằng."""
    text = f"chữ {MARK_SPACE} và {MARK_NL} thật"
    assert tok.decode(tok.encode(text)) == text


def test_no_unk_ever(tok: BPETokenizer) -> None:
    """Byte-fallback ⇒ <UNK> BẤT KHẢ. Vocab train không hề có emoji/Kirin."""
    unk = tok.token_to_id["<UNK>"]
    for text in ["🐈🇻🇳", "Привет мир", "こんにちは", "\x00\x01"]:
        assert unk not in tok.encode(text)
        assert tok.decode(tok.encode(text)) == normalize(text)


# ---- 2. VOCAB ĐÓNG ------------------------------------------------------------------


def test_vocab_closed(tok: BPETokenizer) -> None:
    """Bề mặt không mọc thêm dù gặp text lạ — trái ngược vocab MỞ hiện tại (126 518 và tăng)."""
    before = len(tok)
    tok.encode("một câu hoàn toàn lạ với 🦖 và 9876543210")
    assert len(tok) == before


def test_encode_ids_in_range(tok: BPETokenizer) -> None:
    for text in CORPUS + ["🦖 lạ hoắc"]:
        assert all(0 <= i < len(tok) for i in tok.encode(text))


# ---- 3. CHỮ SỐ RỜI ------------------------------------------------------------------


def test_digits_never_merge(tok: BPETokenizer) -> None:
    """Mỗi chữ số một token. Đo B0: 8 590 loại token chứa số = 14.4% vocab → còn 10.

    `1288` và `1287` phải CHIA nơ-ron, nếu không mạng không có cơ quan để học số.
    """
    pieces = tok.encode_pieces("1288 và 1287")
    assert pieces[:4] == ["1", "2", "8", "8"]
    assert pieces[-4:] == ["Ġ1", "2", "8", "7"]


def test_digit_never_merges_with_letter(tok: BPETokenizer) -> None:
    """`năm1288` không thể thành một token — regex tách nhánh \\d vs [^\\W\\d_]+."""
    trained = BPETokenizer.train(["năm1288 " * 200], vocab_size=400)
    for p in trained.tokens:
        if _BYTE_RE.match(p) or p in trained.specials:
            continue  # `<0x41>` mang cả chữ lẫn số trong TÊN, không phải bề mặt
        has_d = any(c.isdecimal() for c in p)
        has_a = any(c.isalpha() for c in p)
        assert not (has_d and has_a), f"token trộn chữ+số: {p!r}"


def test_superscript_is_not_an_arithmetic_digit() -> None:
    """`km²` ĐƯỢC gộp: `²` là ký hiệu đơn vị, không phải chữ số tham gia phép toán.

    `'²'.isdigit()` = True nhưng regex `\\d` (= Nd) không khớp — verify mini bắt đúng lằn
    ranh này (`Ġkm²`/`Ġm³` từng làm mục §7.2 #15 trượt oan).
    """
    trained = BPETokenizer.train(["diện tích 5 km² và 3 m³ " * 200], vocab_size=400)
    assert trained.encode_pieces("Ġkm²" and " km²") == ["Ġkm²"]


def test_all_ten_digits_are_neurons_even_if_absent_from_corpus() -> None:
    """Corpus train KHÔNG có `2`/`7` → chúng vẫn phải là nơ-ron chữ số, không byte-fallback.

    Bug đã bắt: alphabet suy từ tần suất ⇒ chữ số hiếm rớt ⇒ `52` → `['5','<0x32>']`.
    Roundtrip vẫn đúng nên bug im lặng — mà `2` thôi không còn dùng chung nơ-ron với `2`
    ở chỗ khác, tức tái phát đúng bệnh §3.1 mà digit-split sinh ra để chữa.
    """
    trained = BPETokenizer.train(["chỉ có 5 và 3 thôi " * 200], vocab_size=400)
    for d in "0123456789":
        assert trained.encode_pieces(d) == [d], f"chữ số {d} không phải nơ-ron"
    assert trained.encode_pieces("52") == ["5", "2"]


def test_digit_split_regardless_of_corpus(tok: BPETokenizer) -> None:
    """Data lặp `2026` mười nghìn lần cũng KHÔNG mua được một merge — luật ở pre-token,
    không ở merge table."""
    trained = BPETokenizer.train(["2026 " * 10000], vocab_size=500)
    assert trained.encode_pieces("2026") == ["2", "0", "2", "6"]


# ---- 4. SPECIAL NGUYÊN TỬ -----------------------------------------------------------


def test_specials_atomic(tok: BPETokenizer) -> None:
    """`Đáp:` tách thành `Đáp`+`:` = mất mốc → ingest_qa bỏ qua LẶNG LẼ mọi cặp QA
    (bug nền TK_010). Phải là MỘT token."""
    pieces = tok.encode_pieces("Hỏi: x ? Đáp: y .")
    assert HOI_LABEL in pieces and DAP_LABEL in pieces
    assert pieces[0] == HOI_LABEL


def test_eos_atomic(tok: BPETokenizer) -> None:
    """EOS = mốc kết đơn vị — LIFT ≥ 2.0 của `at_boundary` (README §0.4.2) đứng trên nó."""
    pieces = tok.encode_pieces("câu xong .<EOS>")
    assert pieces[-1] == SPECIAL_EOS


def test_special_ids_stable(tok: BPETokenizer) -> None:
    """Id special CỐ ĐỊNH ở đầu — code hạ nguồn so id, không so chuỗi."""
    assert tok.tokens[:4] == ["<UNK>", SPECIAL_EOS, HOI_LABEL, DAP_LABEL]


# ---- khoảng trắng: thứ trước đây bị VỨT ---------------------------------------------


def test_space_is_a_token(tok: BPETokenizer) -> None:
    """`Ġ` = space đứng trước. Đây là 'dấu cách cũng là token' của chỉ đạo tác giả."""
    pieces = tok.encode_pieces("thủ đô")
    assert any(p.startswith(MARK_SPACE) for p in pieces), pieces


def test_newline_is_a_token(tok: BPETokenizer) -> None:
    """`\\n` phải sống tới mạng. `pipeline.py` cũ ghi thẳng '\\n → khoảng trắng lúc
    tokenize' — cấu trúc đoạn văn bị huỷ NGAY LÚC NẠP."""
    assert any(MARK_NL in p for p in tok.encode_pieces("dòng một\ndòng hai"))


def test_pre_tokenize_keeps_all_whitespace() -> None:
    for text in ["a  b", "a\n\nb", "a \n b", " a", "a ", "a\t\tb", "a\r\nb"]:
        syms = [s for piece in pre_tokenize(normalize(text)) for s in piece]
        assert BPETokenizer.decode_pieces(syms) == text, f"{text!r} → {syms!r}"


# ---- artifact -----------------------------------------------------------------------


def test_save_load_identical(tok: BPETokenizer, tmp_path) -> None:
    p = tmp_path / "bpe.json"
    tok.save(p)
    back = BPETokenizer.load(p)
    assert back.fingerprint() == tok.fingerprint()
    assert back.tokens == tok.tokens
    for text in CORPUS:
        assert back.encode(text) == tok.encode(text)


def test_fingerprint_changes_with_merges(tok: BPETokenizer) -> None:
    """Tokenizer lệch checkpoint = mọi id vô nghĩa mà không crash → phải bắt được."""
    other = BPETokenizer(tok.merges[:-5], tok.alphabet)
    assert other.fingerprint() != tok.fingerprint()
