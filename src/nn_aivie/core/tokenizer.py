"""
TK_041 — BPE tokenizer TỰ-TRAIN trên corpus nhà. Đáy MỚI của tháp.

Vì sao (đo B0, TK_041 §0): tokenizer cũ (`text.split()` + `FOLD_STRIP`) là **đáy giả** —
giả định "dấu cách = ranh giới đơn vị", VỨT khoảng trắng/xuống dòng, vocab MỞ vô hạn
(53% là hapax df=1 = nơ-ron ma), và mỗi con số là một atom vô can (`1287` ⊥ `1288`).

Đáy mới, bốn tính chất BẤT KHẢ XÂM PHẠM (mỗi cái có test riêng):

1. **VÔ-TỔN-HAO**: `decode(encode(x)) == NFC(x)` từng byte — kể cả khoảng trắng, xuống
   dòng, emoji, chữ Hán. Đây là định nghĩa vận hành của "không bỏ qua bất kỳ cái gì".
2. **VOCAB ĐÓNG**: bề mặt = đúng `vocab_size` token từ merge table. Không mọc thêm.
3. **CHỮ SỐ RỜI**: mỗi chữ số một token, không bao giờ merge với nhau hay với chữ cái —
   cưỡng chế ở PRE-TOKEN (regex), nên merge table không thể lách dù data nói gì.
4. **SPECIAL NGUYÊN TỬ**: `<EOS>`/`Hỏi:`/`Đáp:` là giao thức frame (README Mục 0), BPE
   không đụng tới. Tách `Đáp:` thành `Đáp`+`:` = mất mốc → `ingest_qa` bỏ qua LẶNG LẼ mọi
   cặp QA (bug nền đã vấp ở TK_010 — xem comment cũ trong `align.tokenize_punct`).

Luật 0 SẠCH: merge table học từ chính corpus mình (data-derived), không mượn tri thức
ngoài. Chốt tác giả 2026-07-16.

Alphabet là KÝ TỰ (không phải byte) + byte-fallback cho ký tự lạ — cố ý: token đọc được
bằng mắt (`Ġthủ`, không phải `Ġth\xe1\xbb\xa7`) là điều kiện để "đọc tay" — kỷ luật nền
của dự án (TK_041 §8 R6).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# ---- marker khoảng trắng: quy ước MƯỢN của GPT-2, KHÔNG phải tất yếu -----------------
#
# Mọi tokenizer LLM đều hiện khoảng trắng bằng marker, không dùng dấu cách thật — hai họ:
#   * GPT-2 / GPT-4 / Llama-3 (tiktoken):      Ġ (space) · Ċ (newline)
#   * Llama-1,2 / T5 / Mistral (SentencePiece): ▁ (space) · byte <0x0A> (newline)
# Ta theo quy ước GPT-2. `Ġ`/`Ċ` KHÔNG phải ai chọn cho đẹp: GPT-2 là byte-level nên phải
# ánh xạ 256 byte sang ký tự in được, và byte 0x20 (space) rơi trúng U+0120 = `Ġ`, 0x0A →
# U+010A = `Ċ`. Chúng là **tai nạn của một bảng byte**.
#
# ⚠️ VÀ ĐÓ LÀ CHỖ THIẾT KẾ NÀY LAI (nói thẳng, đừng để người sau tưởng là định luật):
# ánh xạ byte của GPT-2 là **song ánh trên byte** ⇒ chữ `Ġ` THẬT trong text = bytes
# [196,160] → thành hai ký tự KHÁC → không bao giờ đụng marker `Ġ`(=byte 0x20). Va chạm là
# **bất khả về cấu trúc**, nên GPT-2 không cần một dòng escape nào (đo bằng tiktoken
# 2026-07-16: `'chữ Ġ thật'` roundtrip sạch).
#
# Ta là **char-level** (cố ý — xem note đầu file: token phải đọc được bằng mắt cho R6).
# Char-level thì marker `Ġ` và `Ġ` trong text là **CÙNG một ký tự** → va chạm THẬT →
# `_escape_raw` phải tồn tại. Tức: ta mượn marker của GPT-2 mà KHÔNG mượn được cơ chế làm
# marker đó an toàn, rồi chắp lớp escape để dựng lại tính chất đã đánh rơi.
#
# Ba phương án nhất quán + giá (đo/thử 2026-07-16):
#   (a) byte-level thuần   : hết escape, nhưng token thành `Ġth\xe1\xbb\xa7` → giết R6 (đọc
#                            tay) — chính lớp đã bắt bug #3. BỎ.
#   (b) char + marker + escape  ← ĐANG DÙNG. Giá: ~10 dòng escape + test khoá.
#   (c) char + dấu cách THẬT: hết escape (thử nghiệm: 38/40 test pass, 2 cái đỏ chỉ vì viết
#                            cứng chuỗi 'Ġ1'). Giá: khoảng trắng VÔ HÌNH trong log và
#                            Graph UI (`<div> Nam</div>` render ra "Nam") → hỏng đúng R6. BỎ.
#
# Chốt tác giả 2026-07-16: giữ (b). `▁` (SentencePiece) sẽ "đúng họ" hơn — kiến trúc ta
# (char + byte-fallback) CHÍNH LÀ kiến trúc Llama-2 — nhưng `▁` không có quy ước xuống dòng
# (Llama-2 dùng byte <0x0A>, và thế thì `\n\n` KHÔNG merge được → mất tín hiệu ngắt đoạn mà
# ISG vừa học). Cặp `Ġ`/`Ċ` ít nhất nhất quán với nhau.
MARK_SPACE = "Ġ"  # U+0120 — GPT-2: byte 0x20 (space)
MARK_NL = "Ċ"     # U+010A — GPT-2: byte 0x0A (newline)
MARK_TAB = "ĉ"    # U+0109 — byte 0x09
MARK_CR = "č"     # U+010D — byte 0x0D

_WS_TO_MARK = {" ": MARK_SPACE, "\n": MARK_NL, "\t": MARK_TAB, "\r": MARK_CR}
_MARK_TO_WS = {v: k for k, v in _WS_TO_MARK.items()}
MARKS = frozenset(_MARK_TO_WS)

# Mười nơ-ron chữ số — BẢO ĐẢM có mặt, không phụ thuộc corpus (§3.2).
#
# Bẫy đã cắn (test `test_superscript_is_not_an_arithmetic_digit`, 2026-07-16): alphabet suy
# từ tần suất corpus + `min_char_count` ⇒ chữ số HIẾM rớt khỏi alphabet → byte-fallback:
# `52` → `['5', '<0x32>']`. Roundtrip VẪN ĐÚNG nên bug này trốn thoát im lặng, chỉ có điều
# `2` thôi không còn là nơ-ron chữ số dùng chung — tức đúng cái bệnh §3.1 mà digit-split
# sinh ra để chữa, nay tái phát ở tầng alphabet. Corpus thật chắc chắn có đủ 0–9, nhưng
# "chắc chắn" không phải bảo đảm: ghim cứng 10 id.
DIGITS = frozenset("0123456789")

# ---- special token (nguyên tử — BPE không đụng) -------------------------------------
SPECIAL_EOS = "<EOS>"
SPECIAL_UNK = "<UNK>"  # giữ id cho tương thích; byte-fallback ⇒ count PHẢI = 0 (verify §7.2)
HOI_LABEL = "Hỏi:"
DAP_LABEL = "Đáp:"

# Thứ tự CỐ ĐỊNH — id special phụ thuộc nó. Thêm vào CUỐI, không chèn giữa.
SPECIALS: tuple[str, ...] = (SPECIAL_UNK, SPECIAL_EOS, HOI_LABEL, DAP_LABEL)

# ---- pre-token: nơi cưỡng chế mọi luật cứng ----------------------------------------
#   \d              → MỘT chữ số (không phải \d+): chữ số không merge với nhau, cũng
#                     không với chữ cái (nhánh khác) → `Ġnăm1288` BẤT KHẢ.
#   [^\W\d_]+       → chuỗi chữ cái có dấu (\w trừ số/underscore)
#   \s+             → khoảng trắng ĐƯỢC GIỮ, không vứt
#   [^\s\w]|_       → mỗi dấu/underscore là đơn vị riêng
_PAT = re.compile(r"\d|[^\W\d_]+|\s+|[^\s\w]|_", re.UNICODE)

_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def byte_token(b: int) -> str:
    return f"<0x{b:02X}>"


BYTE_TOKENS: tuple[str, ...] = tuple(byte_token(b) for b in range(256))


def normalize(text: str) -> str:
    """NFC — BẮT BUỘC, bước đầu tiên.

    Đo B0: 100% (28 999/28 999) đơn vị corpus chứa dấu tổ hợp được. `ế` dạng NFC (1 code
    point) và NFD (`e`+`◌̂`+`◌́`) NHÌN GIỐNG HỆT nhau nhưng BPE tách khác hẳn → merge table
    học hai họ song song cho cùng một chữ. Tokenizer cũ `.split()` không quan tâm nên bug
    này ẩn suốt.
    """
    return unicodedata.normalize("NFC", text)


def _escape_raw(ch: str) -> list[str]:
    """Ký tự BỀ MẶT THẬT → symbol. Ký tự trùng marker → byte-fallback.

    **Đây là CÁI GIÁ của việc chọn marker `Ġ`/`Ċ` trên nền char-level — KHÔNG phải một
    định luật của tokenization.** GPT-2 không có hàm này và không cần: nó byte-level nên va
    chạm marker là bất khả (xem note ở khối MARK_* đầu file). Ta char-level → `Ġ` marker và
    `Ġ` trong text là cùng một ký tự → phải tự tay tách hai nghĩa đó ra.

    Cụ thể: marker `Ġ`/`Ċ` nghĩa là "khoảng trắng DO TOKENIZER sinh". Nếu text gốc chứa chữ
    `Ġ` mà để nguyên, decode trả về dấu cách → **mất chữ**. Nên symbol phải mang XUẤT XỨ —
    và đó là lý do `pre_tokenize` trả `list[list[str]]` (symbol) chứ không phải `list[str]`
    (chuỗi): trong một chuỗi thì `Ġ` do ai sinh cũng như nhau.

    Bug thật, `test_marker_chars_in_raw_text_survive` bắt ngay vòng đầu. Bỏ hàm này thì phải
    bỏ luôn marker (phương án (c)) — hai thứ sống chết cùng nhau.
    """
    return [ch] if ch not in MARKS else [byte_token(b) for b in ch.encode("utf-8")]


def _encode_ws_run(run: str) -> list[str]:
    """Khoảng trắng → symbol marker. Ký tự trắng lạ (nbsp…) → byte-fallback."""
    out: list[str] = []
    for ch in run:
        mark = _WS_TO_MARK.get(ch)
        if mark is not None:
            out.append(mark)
        else:
            out.extend(byte_token(b) for b in ch.encode("utf-8"))
    return out


def pre_tokenize(text: str) -> list[list[str]]:
    """NFC-text → danh sách 'mảnh', mỗi mảnh = list SYMBOL. BPE chỉ merge TRONG một mảnh.

    Symbol = 1 ký tự bề mặt, HOẶC marker khoảng trắng, HOẶC byte token `<0xNN>`.

    Khoảng trắng: **đúng một** space trước một mảnh không-trắng → gộp làm tiền tố `Ġ`
    (kiểu GPT-2 — rẻ nhất, vì đó là ~99% trường hợp). Mọi run khác (xuống dòng, space
    lặp, tab) → mảnh riêng gồm marker (`ĊĊ` = ngắt đoạn, BPE có thể merge thành 1 token).

    → Sau bước này, xuống dòng là thứ mạng HỌC được. Trước đây `pipeline.py` ghi thẳng
    "`\\n` → khoảng trắng lúc tokenize", tức cấu trúc đoạn văn bị huỷ NGAY LÚC NẠP, rồi
    TK_035 phải dựng lại 'paragraph unit' bằng cơ chế khác ở tầng trên.
    """
    matches = list(_PAT.finditer(text))
    out: list[list[str]] = []
    pending: list[str] = []
    for i, m in enumerate(matches):
        s = m.group(0)
        if s.isspace():
            nxt = matches[i + 1].group(0) if i + 1 < len(matches) else None
            if s == " " and nxt is not None and not nxt.isspace():
                pending = [MARK_SPACE]
            else:
                out.append(_encode_ws_run(s))
            continue
        piece = list(pending)
        pending = []
        for ch in s:
            piece.extend(_escape_raw(ch))
        out.append(piece)
    if pending:
        out.append(pending)
    return out


def _split_unknown(symbols: list[str], alphabet: frozenset[str]) -> list[list[str]]:
    """Mảnh → các mảnh-con chỉ gồm symbol trong alphabet; symbol lạ → mảnh-con RIÊNG.

    Byte token nằm mảnh-con riêng ⇒ **không bao giờ bị merge** ⇒ decode gom byte liền kề
    là an toàn (`decode_pieces`).
    """
    out: list[list[str]] = []
    cur: list[str] = []
    for sym in symbols:
        if sym in alphabet:
            cur.append(sym)
            continue
        if cur:
            out.append(cur)
            cur = []
        if _BYTE_RE.match(sym):
            out.append([sym])
        else:
            for b in sym.encode("utf-8"):
                out.append([byte_token(b)])
    if cur:
        out.append(cur)
    return out


class BPETokenizer:
    """BPE trên mảnh pre-token. Merge table CỐ ĐỊNH sau train — artifact ghim (`bpe_v1.json`).

    Bề mặt token = `specials + bytes + alphabet + merged`, thứ tự CỐ ĐỊNH → id ổn định
    giữa các lần load. Đổi merge table = đổi vocab = MỌI checkpoint vô giá trị (TK_041 §R5).
    """

    def __init__(
        self,
        merges: list[tuple[str, str]],
        alphabet: Iterable[str],
        *,
        specials: tuple[str, ...] = SPECIALS,
    ) -> None:
        self.specials = tuple(specials)
        # Marker + chữ số LUÔN trong alphabet: corpus train có thể không có `\t`/`\r` hay
        # một chữ số hiếm, nhưng lúc encode gặp thì phải có id — nếu không sẽ rơi
        # byte-fallback: roundtrip vẫn đúng (nên bug im lặng) mà nơ-ron thì vô nghĩa. Rẻ (14 id).
        self.alphabet = frozenset(alphabet) | MARKS | DIGITS
        self.merges = [tuple(m) for m in merges]
        self.ranks: dict[tuple[str, str], int] = {m: i for i, m in enumerate(self.merges)}
        self._cache: dict[tuple[str, ...], list[str]] = {}
        self.tokens: list[str] = self._build_surface()
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        if len(self.token_to_id) != len(self.tokens):
            raise ValueError("bề mặt BPE có token trùng — merge table hỏng")

    # ---- bề mặt ----
    def _build_surface(self) -> list[str]:
        out = list(self.specials)
        out.extend(BYTE_TOKENS)
        seen = set(out)
        for ch in sorted(self.alphabet):
            if ch not in seen:
                out.append(ch)
                seen.add(ch)
        for a, b in self.merges:
            t = a + b
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out

    def __len__(self) -> int:
        return len(self.tokens)

    # ---- train ----
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32000,
        *,
        specials: tuple[str, ...] = SPECIALS,
        min_char_count: int = 1,
        progress=None,
    ) -> "BPETokenizer":
        piece_freq: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for p in pre_tokenize(normalize(text)):
                piece_freq[tuple(p)] += 1

        sym_count: Counter[str] = Counter()
        for p, f in piece_freq.items():
            for sym in p:
                sym_count[sym] += f
        # Ký tự quá hiếm → byte-fallback (không đốt id cho ký tự thấy 1 lần). Byte token
        # KHÔNG vào alphabet (đã có sẵn 256 id, và phải giữ nguyên tử để decode gom được).
        alphabet = frozenset(
            s
            for s, c in sym_count.items()
            if c >= min_char_count and not _BYTE_RE.match(s)
        )

        n_base = len(specials) + 256 + len(alphabet - set(specials))
        n_merge_target = max(0, vocab_size - n_base)
        if progress:
            progress(f"pre-token: {sum(piece_freq.values()):,} lượt, {len(piece_freq):,} loại")
            progress(f"alphabet: {len(alphabet):,} · base {n_base:,} · cần {n_merge_target:,} merge")

        merges = cls._learn_merges(piece_freq, alphabet, n_merge_target, progress=progress)
        return cls(merges, alphabet, specials=specials)

    @staticmethod
    def _learn_merges(
        piece_freq: Counter[tuple[str, ...]],
        alphabet: frozenset[str],
        n_target: int,
        *,
        progress=None,
    ) -> list[tuple[str, str]]:
        """BPE cổ điển trên từ-điển-tần-suất (GPT-2 style) + index ngược `where`.

        Chỉ đụng lại các mảnh THẬT SỰ chứa cặp vừa merge → O(số lần xuất hiện), không
        quét lại toàn bộ mỗi vòng (bản recount ngây thơ mất hàng chục phút; bản này ~3
        phút cho 32k trên 16MB — đo B0).
        """
        # Mỗi mảnh rã sẵn thành mảnh-con; cặp KHÔNG bao giờ bắc cầu qua ranh giới mảnh-con
        # (đó là chỗ byte-fallback ngồi) → `subwords[p]` là list các list symbol.
        subwords: dict[tuple[str, ...], list[list[str]]] = {
            p: _split_unknown(list(p), alphabet) for p in piece_freq
        }

        pair_freq: Counter[tuple[str, str]] = Counter()
        where: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
        for p, subs in subwords.items():
            f = piece_freq[p]
            for syms in subs:
                for pair in zip(syms, syms[1:]):
                    pair_freq[pair] += f
                    where[pair].add(p)

        merges: list[tuple[str, str]] = []
        while len(merges) < n_target and pair_freq:
            best, cnt = pair_freq.most_common(1)[0]
            if cnt < 2:
                break  # hết cặp lặp — vocab tự đóng dưới đích, không bịa merge rác
            merges.append(best)
            new = best[0] + best[1]
            for p in list(where[best]):
                f = piece_freq[p]
                for si, syms in enumerate(subwords[p]):
                    i = 0
                    out: list[str] = []
                    while i < len(syms):
                        if i < len(syms) - 1 and syms[i] == best[0] and syms[i + 1] == best[1]:
                            out.append(new)
                            i += 2
                        else:
                            out.append(syms[i])
                            i += 1
                    if out == syms:
                        continue
                    for pair in zip(syms, syms[1:]):
                        pair_freq[pair] -= f
                        if pair_freq[pair] <= 0:
                            del pair_freq[pair]
                    for pair in zip(out, out[1:]):
                        pair_freq[pair] += f
                        where[pair].add(p)
                    subwords[p][si] = out
            pair_freq.pop(best, None)
            where.pop(best, None)
            if progress and len(merges) % 4000 == 0:
                progress(f"  merge {len(merges):,}/{n_target:,} (count {cnt:,})")
        return merges

    # ---- encode ----
    def _bpe_piece(self, syms: list[str]) -> list[str]:
        while len(syms) > 1:
            best_i: int | None = None
            best_rank: int | None = None
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            syms[best_i : best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        return syms

    def encode_pieces(self, text: str) -> list[str]:
        """NFC(text) → token bề mặt. Special tách TRƯỚC pre-token (nguyên tử)."""
        out: list[str] = []
        for chunk, is_special in self._split_specials(normalize(text)):
            if is_special:
                out.append(chunk)
                continue
            for piece in pre_tokenize(chunk):
                key = tuple(piece)
                cached = self._cache.get(key)
                if cached is None:
                    cached = []
                    for sub in _split_unknown(piece, self.alphabet):
                        if len(sub) == 1 and _BYTE_RE.match(sub[0]):
                            cached.append(sub[0])
                        else:
                            cached.extend(self._bpe_piece(list(sub)))
                    self._cache[key] = cached
                out.extend(cached)
        return out

    def _split_specials(self, text: str) -> list[tuple[str, bool]]:
        # Special CHỈ gồm `<EOS>`/`Hỏi:`/`Đáp:` — không regex-escape động cho gọn/nhanh.
        pat = getattr(self, "_special_pat", None)
        if pat is None:
            if self.specials:
                pat = re.compile("|".join(re.escape(s) for s in self.specials))
            else:
                pat = re.compile(r"(?!x)x")
            self._special_pat = pat
        out: list[tuple[str, bool]] = []
        last = 0
        for m in pat.finditer(text):
            if m.start() > last:
                out.append((text[last : m.start()], False))
            out.append((m.group(0), True))
            last = m.end()
        if last < len(text):
            out.append((text[last:], False))
        return out

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[p] for p in self.encode_pieces(text)]

    # ---- decode ----
    @staticmethod
    def decode_pieces(pieces: Iterable[str]) -> str:
        """Token bề mặt → text gốc. Nghịch đảo ĐÚNG của `encode_pieces` (test roundtrip).

        Byte token liền kề gom thành một run rồi mới utf-8 decode — ký tự lạ luôn tách
        thành mảnh-con riêng lúc encode nên không bao giờ nằm giữa một token merged.
        """
        out: list[str] = []
        buf: bytearray = bytearray()

        def flush() -> None:
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf.clear()

        for p in pieces:
            m = _BYTE_RE.match(p)
            if m is not None:
                buf.append(int(m.group(1), 16))
                continue
            flush()
            out.append("".join(_MARK_TO_WS.get(ch, ch) for ch in p))
        flush()
        return "".join(out)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_pieces(self.tokens[i] for i in ids)

    # ---- artifact ----
    def to_payload(self) -> dict:
        return {
            "version": 1,
            "specials": list(self.specials),
            "alphabet": sorted(self.alphabet),
            "merges": [list(m) for m in self.merges],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "BPETokenizer":
        return cls(
            [tuple(m) for m in payload["merges"]],
            payload["alphabet"],
            specials=tuple(payload["specials"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_payload(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))

    def fingerprint(self) -> str:
        """Hash merge table — checkpoint ghim cái này (verify §7.2 mục 2).

        Tokenizer lệch checkpoint = mọi id vô nghĩa mà KHÔNG crash (chỉ ra rác lặng lẽ) —
        đúng loại lỗi phải chết ồn ào.
        """
        import hashlib

        h = hashlib.sha256()
        h.update(json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True).encode())
        return h.hexdigest()[:16]
