"""Token and structural node vocabulary.

TK_041 — vocab có HAI CHẾ ĐỘ:

* **BPE (mới)**: `Vocabulary(tokenizer=...)` → không gian id PHÂN VÙNG (§2.4):

      [0, 2)                  <UNK>, <EOS>          — non-surface
      [2, len(tokenizer))     Hỏi:/Đáp: + subword   — BỀ MẶT, ĐÓNG (không mọc)
      [len(tokenizer), ∞)     Entity:/Concept:/…    — cấu trúc, mọc được

  → `is_surface(id)` là **phép so int**, không `startswith` 9 prefix.
  → `surface_mask` là **slice**, không quét string 34k phần tử mỗi lần vocab lớn thêm.

* **Legacy**: `Vocabulary()` không tokenizer → hành vi cũ (so chuỗi). Giữ cho checkpoint
  train trước TK_041 (`grade_v3*.pkl`) còn load/test được.
"""

from __future__ import annotations

from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL, SPECIAL_EOS, SPECIAL_UNK, BPETokenizer

STRUCTURAL_PREFIXES = ("Rule:", "Var:", "Lbl:", "R_-", "QFocus:", "Concept:", "Sense:", "POS:", "Entity:")

# ⚠️ LEGACY-ONLY (vocab không tokenizer). Trên đường BPE hằng này KHÔNG được đọc — xem
# `Vocabulary.fold_of`. Còn sống chỉ để checkpoint tiền-TK_041 tái lập đúng baseline §7.1.
_LEGACY_FOLD_STRIP = ".,;:!?…\"'()[]"

# Special KHÔNG hiển thị — phải đứng ĐẦU `tokenizer.SPECIALS` để `[0, N_HIDDEN)` liền mạch.
_HIDDEN_SPECIALS = (SPECIAL_UNK, SPECIAL_EOS)


def _strip_mark(token: str) -> str:
    """Bề mặt token, bỏ marker khoảng trắng: `Ġthủ` → `thủ`, `ĊĊ` → `ĊĊ` (chỉ marker → giữ).

    Token thuần marker (`Ġ`, `ĊĊ`) giữ nguyên: chúng LÀ nội dung (khoảng trắng), không phải
    tiền tố của nội dung khác. Nếu strip trọn thì `Ġ` và `ĊĊ` cùng fold rỗng → đúng lại cái
    bug fold-rỗng vừa giết ở `fold_of`.
    """
    from nn_aivie.core.tokenizer import MARK_SPACE

    stripped = token.lstrip(MARK_SPACE)
    return stripped or token


def is_surface_token(token: str) -> bool:
    """Token hiển thị được khi generate (không phải node cấu trúc).

    ⚠️ **Đường LEGACY (so chuỗi).** Code mới dùng `vocab.is_surface(id)` — so int, và là
    thứ Luật 0.0 đòi (README §0.0 mục B: "inference re-parse token-string"). Hàm này còn
    sống chỉ để đọc checkpoint tiền-TK_041.
    """
    if token in (SPECIAL_EOS, SPECIAL_UNK):
        return False
    return not token.startswith(STRUCTURAL_PREFIXES)


def is_rule_token(token: str) -> bool:
    return token.startswith("Rule:")


def is_qfocus_token(token: str) -> bool:
    return token.startswith("QFocus:")


def is_var_token(token: str) -> bool:
    return token.startswith("Var:")


def is_concept_token(token: str) -> bool:
    return token.startswith("Concept:")


class Vocabulary:
    """Maps string tokens to integer node ids (case-sensitive: Em và em là hai node khác nhau)."""

    def __init__(self, tokenizer: BPETokenizer | None = None) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []
        # TK_012: thuộc-tính-token precompute (không xử lý string lúc match/infer)
        self._cap_class: list[int] = []      # 0=không hoa | 1=hoa đầu | 2=HOA HẾT
        self._fold: list[str] = []           # bề mặt case-fold (bỏ marker Ġ + lower)
        self._fold_to_gid: dict[str, int] = {}
        self._fold_gid: list[int] = []       # id nhóm fold (so khớp bằng int, không string)
        self.tokenizer = tokenizer
        if tokenizer is None:
            self._surface_hi: int | None = None  # legacy → is_surface so chuỗi
            self._ensure(SPECIAL_UNK)
            self._ensure(SPECIAL_EOS)
            return
        if tokenizer.specials[: len(_HIDDEN_SPECIALS)] != _HIDDEN_SPECIALS:
            raise ValueError(
                f"tokenizer.SPECIALS phải mở đầu bằng {_HIDDEN_SPECIALS} — "
                "phân vùng id (§2.4) đứng trên thứ tự này"
            )
        # Nạp TRỌN bề mặt BPE theo đúng thứ tự tokenizer → id ổn định mọi lần load.
        for token in tokenizer.tokens:
            self._ensure(token)
        self._surface_hi = len(tokenizer)  # = STRUCT_LO: từ đây trở lên là nơ-ron cấu trúc

    # ---- phân vùng id (TK_041 §2.4) ----
    @property
    def is_bpe(self) -> bool:
        return self._surface_hi is not None

    def is_surface(self, idx: int) -> bool:
        """Token hiển thị được — **so int** trên vocab BPE (Luật 0.0: không re-parse string)."""
        if self._surface_hi is None:
            return is_surface_token(self._id_to_token[idx])
        return len(_HIDDEN_SPECIALS) <= idx < self._surface_hi

    @staticmethod
    def _cap_class_of(token: str) -> int:
        letters = [c for c in token if c.isalpha()]
        if not letters:
            return 0
        if all(c.isupper() for c in letters):
            return 2
        if letters[0].isupper():
            return 1
        return 0

    def _ensure(self, token: str) -> int:
        if token not in self._token_to_id:
            idx = len(self._id_to_token)
            self._token_to_id[token] = idx
            self._id_to_token.append(token)
            # precompute thuộc tính 1 lần khi token vào vocab
            surf = _strip_mark(token)
            self._cap_class.append(self._cap_class_of(surf))
            fold = self.fold_of(token)
            self._fold.append(fold)
            gid = self._fold_to_gid.get(fold)
            if gid is None:
                gid = len(self._fold_to_gid)
                self._fold_to_gid[fold] = gid
            self._fold_gid.append(gid)
        return self._token_to_id[token]

    # ---- thuộc-tính-token (TK_012): match/infer đọc thuộc tính, không string-op ----
    def cap_class(self, idx: int) -> int:
        return self._cap_class[idx]

    def fold(self, idx: int) -> str:
        return self._fold[idx]

    def fold_of(self, token: str) -> str:
        """Fold một chuỗi bề-mặt theo ĐÚNG luật `_ensure` dùng lúc build vocab.

        Để caller ngoài (viz/API/eval) khỏi chép tay — chép tay thì lệch một ký tự là tra
        ra kết quả khác mạng. KHÔNG thêm token vào vocab (khác `id_of`) → gọi lúc chạy an toàn.

        **BPE: `FOLD_STRIP` CHẾT.** Fold = bỏ marker khoảng trắng + lower, KHÔNG strip dấu
        câu. `Ġthủ` ≡ `Thủ` ≡ `thủ`; `.` giữ fold `.` của riêng nó.

        Vì sao xoá được: `FOLD_STRIP = ".,;:!?…\\"'()[]"` là **tri thức chính tả viết tay** —
        đúng thứ §0.4.2 đã dọn ở tầng inference nhưng KHÔNG dọn nổi ở tokenizer (TK_012
        §9.5: "gỡ = đổi vocab = mọi checkpoint vô giá trị"). TK_041 đổi vocab → trả nợ.
        Nó cũng giết bug fold-rỗng theo CẤU TRÚC chứ không bằng vá: bản cũ cho
        `fold_of('.') == fold_of('?') == ''` → mọi dấu câu là MỘT token với mạng → agenda
        RỖNG; bản cũ vá bằng `folded or token.lower()`. Không strip thì không thể ra rỗng.

        **Legacy giữ nguyên `FOLD_STRIP`** — cố ý, không phải quên: vocab tiền-TK_041 có
        token dính dấu (`phở.`, `Việt.` — pre-R0), và ta CẦN checkpoint cũ (`network_v10`)
        tái lập đúng số baseline §7.1, nếu không thì không có gì để so v11 vào. Đổi fold ở
        legacy = baseline trôi = mất luôn thước.
        """
        if self.tokenizer is None:
            folded = token.strip(_LEGACY_FOLD_STRIP).lower()
            return folded or token.lower()
        return _strip_mark(token).lower()

    def fold_group_id(self, idx: int) -> int:
        """Id nhóm case-fold: hai token cùng nhóm ('Phở'≡'phở') → cùng id."""
        return self._fold_gid[idx]

    def id_of(self, token: str) -> int:
        return self._ensure(token)

    def token_of(self, idx: int) -> str:
        return self._id_to_token[idx]

    def surface_mask(self):
        """Mảng bool [n]: token hiển thị được (không phải node cấu trúc/EOS/UNK).

        TK_041: trên vocab BPE đây là một **SLICE** — bề mặt là dải id liền mạch
        `[2, _surface_hi)`, mọi thứ trên `_surface_hi` là cấu trúc. Không đụng string.

        Legacy (checkpoint tiền-TK_041): quét `is_surface_token` như cũ. Comment TK_011 ghi
        nó tốn ~0.47s/prompt trên vocab 34k — chi phí đó nay chỉ còn ở đường legacy.
        """
        import numpy as np

        n = len(self._id_to_token)
        cache = getattr(self, "_surface_mask_cache", None)
        if cache is not None and len(cache) == n:
            return cache
        if self._surface_hi is None:
            cache = np.fromiter(
                (is_surface_token(t) for t in self._id_to_token), dtype=bool, count=n
            )
        else:
            cache = np.zeros(n, dtype=bool)
            cache[len(_HIDDEN_SPECIALS) : self._surface_hi] = True
        self._surface_mask_cache = cache
        return cache

    def __len__(self) -> int:
        return len(self._id_to_token)

    def var_id(self, name: str) -> int:
        return self.id_of(f"Var:{name}")

    def rule_id(self, name: str) -> int:
        return self.id_of(f"Rule:{name}")

    def qfocus_id(self, focus: str) -> int:
        """Question fingerprint node from q_signature (e.g. QFocus:Chó_màu_gì)."""
        return self.id_of(f"QFocus:{focus}")

    def label_id(self, name: str) -> int:
        return self.id_of(f"Lbl:{name}")

    def concept_id(self, name: str) -> int:
        return self.id_of(f"Concept:{name}")

    def sense_id(self, word: str, k: int) -> int:
        """TK_015 D1 — nơ-ron nghĩa thứ k của từ (non-surface, không sinh ra)."""
        return self.id_of(f"Sense:{word}#{k}")

    def entity_id(self, title: str) -> int:
        """TK_027 — nơ-ron THỰC-THỂ tên-riêng (title bài), non-surface. Key theo CỤM title,
        không fold → tách 'Lý Thái Tổ' khỏi 'lý thuyết'."""
        return self.id_of(f"Entity:{title}")

    def pos_id(self, pos: str) -> int:
        """TK_015 D1 — nơ-ron từ-loại (hub gom mọi từ cùng POS)."""
        return self.id_of(f"POS:{pos.replace(' ', '_')}")

    def position_id(self, k: int) -> int:
        """Preallocated position node R_-k (k >= 1)."""
        return self.id_of(f"R_-{k}")

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        """Encode từng token đúng như đã tokenize — không chuẩn hóa hoa/thường.

        ⚠️ Nhận token ĐÃ TÁCH SẴN → trên vocab BPE nó **mọc thêm id ngoài dải bề mặt**
        (phá tính đóng). Chỉ dùng cho token cấu trúc (`Entity:…`) hoặc đường legacy. Nạp
        text thật → `encode_text`.
        """
        return [self.id_of(t) for t in tokens]

    def encode_text(self, text: str) -> list[int]:
        """Text thô → id. **Đường nạp DUY NHẤT cho text thật** (TK_041 §4.3).

        Trước TK_041 mỗi đường nạp tự `.split()` / `tokenize_punct` → 3 đường tokenize
        độc lập, và lịch sử chứng minh nó sót: đợt EOS sót `pipeline.py` (suýt bake 7.7h ra
        checkpoint không mốc kết) rồi sót `build_varlm.py` (verify FAIL). Nay một cửa.
        """
        if self.tokenizer is None:
            raise RuntimeError("encode_text cần vocab BPE — vocab legacy phải tự tokenize")
        return [self.id_of(p) for p in self.tokenizer.encode_pieces(text)]

    def encode_prompt(self, prompt: str) -> list[int]:
        """Prompt lúc SINH → id. Phải khớp CHÍNH XÁC luật tokenize + chuẩn hoá lúc TRAIN.

        **Hai bẫy, cùng một gốc: prompt và train phải là MỘT đường, không phải hai đường
        tình cờ trùng.**

        (1) Bẫy đã cắn (đọc tay đầu ra mini, 2026-07-16): đường sinh còn
        `prompt.strip().split()` + `encode_tokens` → mỗi TỪ tra thẳng vào vocab. Trên BPE,
        `id_of("Thủ")` trúng biến thể KHÔNG-có-Ġ (token đầu-chuỗi) → prompt
        `'Hỏi: Thủ đô Việt Nam'` thành `'Hỏi:ThủđôViệtNam'`: mạng nhận một chuỗi **chưa từng
        tồn tại lúc train**, mọi cạnh W_seq/cha đều trật. Tệ hơn: từ nào chưa có trong vocab
        (`'gì?'`) thì `id_of` **ĐẺ id mới** → phá luôn tính ĐÓNG của dải bề mặt.

        (2) Bẫy MỚI, lộ ra khi tác giả hỏi "tại sao phải strip": trước TK_041 `.strip()` chỉ
        là thừa (`"  a b  ".split()` vốn đã bỏ trắng thừa). **Nay nó KHÔNG thừa** — khoảng
        trắng LÀ token, nên `'  Hỏi:'` → `['Ġ','Ġ','Hỏi:']` ≠ `['Hỏi:']`. Mà `pipeline` lúc
        train chuẩn hoá text (`normalize_ws_keep_newlines`, kết thúc bằng `.strip()`) → nếu
        prompt không chuẩn hoá y hệt thì một khoảng trắng thừa của người dùng đẩy prompt ra
        **ngoài phân bố train**. Tính vô-tổn-hao khiến rác ở đầu vào cũng được giữ trung
        thành — nên chuẩn hoá phải làm TRƯỚC, và làm GIỐNG train.
        """
        if self.tokenizer is None:
            return self.encode_tokens(prompt.strip().split())
        from nn_aivie.data.quality import normalize_ws_keep_newlines

        return self.encode_text(normalize_ws_keep_newlines(prompt))

    def decode_ids(self, ids: list[int]) -> list[str]:
        return [self.token_of(i) for i in ids]

    def detokenize(self, ids) -> str:
        """Id → text. **Detokenizer DUY NHẤT** — mọi `" ".join(token_of(t) …)` phải chết.

        Vì sao: ~20 chỗ trong repo làm `" ".join(...)`, tức **BỊA ra khoảng trắng**, vì
        tokenizer cũ đã vứt mất khoảng trắng thật. Hệ quả: mọi output của dự án mang lỗi
        `Hà Nội .` (space thừa trước dấu chấm) — ai cũng sống chung với nó. Nay khoảng
        trắng là token, nên detok chỉ việc nối lại đúng cái đã nạp.
        """
        if self.tokenizer is None:  # legacy: khoảng trắng không có trong id → đành bịa
            return " ".join(self.token_of(i) for i in ids if is_surface_token(self.token_of(i)))
        return self.tokenizer.decode_pieces(
            self.token_of(i) for i in ids if self.is_surface(i)
        )

    def position_ids(self, k_max: int) -> list[int]:
        return [self.position_id(k) for k in range(1, k_max + 1)]
