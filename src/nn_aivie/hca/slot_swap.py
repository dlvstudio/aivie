r"""TK_045 §13 Trụ B — THAY KHE: nguồn sáng tạo thật, vẫn không salad.

## Vì sao trụ này tồn tại — thước đã chỉ đúng chỗ

P50 (§13.11) đo: đường mệnh đề có `novelty` p90 chỉ **0.76–0.77**, và tỉ lệ ô ĐÍCH sụp
**82% → 0%** khi siết `nov ≥ 0.80`. novelty 0.70 nghĩa là **khối chép liên tục = 30% đầu ra**.
Đó là **tính chất CẤU TRÚC** của soạn-theo-mệnh-đề: chọn nguyên mệnh đề thì mệnh đề dài nhất
chính là khối chép. Trần đó **chỉ phá được bằng cách sửa BÊN TRONG mệnh đề** — tức trụ này.

## Cơ chế: khe = vị trí THAY THẾ ĐƯỢC ĐÃ CHỨNG THỰC

Với vị trí `i` trong mệnh đề, gọi `a = tok[i-1]`, `b = tok[i+1]`. Token `x` được phép thay
`tok[i]` khi **cả hai mối nối đều có thật** trong `emit_ids` của cha:

    (a → x) đã thấy   ∧   (x → b) đã thấy

⇒ `x ∈ succ(a) ∩ pred(b)`. Đây là **lớp mẫu theo khe** — định nghĩa gốc của paradigm class,
và là thứ §13.2 nói *mạnh hơn tương đồng embedding*: §12.92 + v4 §3.7 đã đo tương đồng
embedding **một mình KHÔNG** cứu `chó≈cún`; còn "thay được trong đúng khe đã chứng thực" là
tín hiệu **đếm được**.

## Tính chất then chốt (vì sao nó cho ĐÍCH chứ không cho SALAD)

Thay `tok[i]` chỉ đụng đúng hai bigram `(a,x)` và `(x,b)` — mà cả hai **đã được chứng thực**
theo điều kiện trên. ⇒ **`attestation` giữ nguyên theo CẤU TẠO**, trong khi đoạn chép liên
tục bị **cắt đôi** tại `i` ⇒ `novelty` tăng. Đó chính là ô **ĐÍCH** của §13.11.

## ⚠️ RỦI RO THẬT: thay khe có thể LÀM HỎNG FACT

`dời đô về Thăng Long` → `dời đô về Hoa Lư`: hai bigram đều có thật, **fact thì SAI**. Điều
này đánh thẳng vào thứ §13.5 nêu là lợi thế **duy nhất** vượt LLM — *trung thực có dẫn nguồn*.
Nên trụ này **không được** thả tự do. Ba van, đều đo được:

* **van SỐ** (`has_digit`) — token chứa chữ số, **và láng giềng của nó**, bất khả xâm phạm.
  ⚠️ Van này sinh ra vì `max_gidf` **THẤT BẠI** với số: TK_041 §0.4.3 cưỡng chế tách chữ số
  ⇒ mỗi chữ số là atom **phổ biến** ⇒ gidf **thấp** ⇒ lọt van gidf tầm thường. Đọc tay bắt
  được `năm 1989`→`1929`, `năm 1290`→`1890`, `(1864)`→`11264` — **không metric nào báo động**
  (đúng bài học TK_041 §5.2: lỗi im lặng chỉ đọc tay mới thấy).
* **`max_gidf`** — chỉ thay token có gidf **dưới** ngưỡng: thực thể/tên riêng bất khả xâm phạm.
* **`order=3`** — đòi cả **trigram** `(a,x,b)` đã thấy, không chỉ hai bigram rời. An toàn hơn
  hẳn, đổi lại ít khe hơn.

Cấu hình lỏng nhất (`order=2`, `max_gidf` cao) phải được coi là **thí nghiệm**, không phải
mặc định — và tỉ lệ hỏng fact phải được đo chứ không phải suy đoán.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nn_aivie.hca.context_positional import is_structural_token
from nn_aivie.hca.objcache import cache_get, cache_put

_SLOT_CACHE: dict = {}

# Ngưỡng gidf mặc định: dưới mức này coi là từ CHỨC NĂNG (thay được). `content_folds`
# của thước dùng gidf ≥ 4.0 làm "nội dung đặc trưng" ⇒ để 4.0 thì thực thể/fact được
# bảo vệ đúng bằng ranh giới mà thước dùng để chấm trôi đề.
DEFAULT_MAX_GIDF = 4.0
DEFAULT_ORDER = 2


@dataclass
class SlotIndex:
    """succ/pred từ emit_ids + gidf — đủ để hỏi 'ai thay được vào khe (a,_,b)'."""

    succ: dict[int, set[int]] = field(default_factory=dict, repr=False)
    pred: dict[int, set[int]] = field(default_factory=dict, repr=False)
    trigrams: set[tuple[int, int, int]] = field(default_factory=set, repr=False)
    gidf: np.ndarray = field(default=None, repr=False)
    struct: np.ndarray = field(default=None, repr=False)
    has_digit: np.ndarray = field(default=None, repr=False)

    def fillers(
        self, a: int, b: int, *, order: int = DEFAULT_ORDER, max_gidf: float = DEFAULT_MAX_GIDF,
    ) -> set[int]:
        """Token thay được vào khe giữa `a` và `b`, đã lọc theo hai van."""
        sa = self.succ.get(int(a))
        pb = self.pred.get(int(b))
        if not sa or not pb:
            return set()
        cand = sa & pb
        if not cand:
            return set()
        out = set()
        for x in cand:
            if self.struct is not None and bool(self.struct[x]):
                continue
            if self.has_digit is not None and bool(self.has_digit[x]):
                continue  # van SỐ (xem `swap_slots`)
            if self.gidf is not None and float(self.gidf[x]) > float(max_gidf):
                continue  # van fact: thực thể/số/tên riêng bất khả xâm phạm
            if int(order) >= 3 and (int(a), int(x), int(b)) not in self.trigrams:
                continue
            out.add(int(x))
        return out


def build_slot_index(state, *, with_trigrams: bool = True) -> SlotIndex:
    """succ/pred dựng từ **`emit_ids` của cha** — CÙNG nguồn với thước `NgramIndex`.

    ⚠️ Bản đầu dựng từ `W_seq` (`EdgeKind.SEQUENTIAL`). Đo được: **60.6%** bigram W_seq
    KHÔNG có trong emit của cha (134 069 vs 52 885 trên mini) — W_seq gom cả các đường nạp
    khác. Hệ quả: cổng sinh kiểm bằng một định nghĩa "đã chứng thực", thước chấm bằng định
    nghĩa khác ⇒ tuyên bố *"attestation giữ theo cấu tạo"* của tôi **SAI** (đo: 0.979→0.882).
    Đúng lớp lỗi `core/neural_lm.py` cảnh báo: hai bên định nghĩa riêng → lệch **âm thầm**.
    Nay một nguồn duy nhất; test đối chứng khoá lại.
    """
    v = state.vocab
    n = len(v)
    succ: dict[int, set[int]] = {}
    pred: dict[int, set[int]] = {}
    tri: set[tuple[int, int, int]] = set()
    if state.parents is not None:
        for rec in state.parents.parents:
            e = [int(t) for t in rec.emit_ids]
            for k in range(len(e) - 1):
                succ.setdefault(e[k], set()).add(e[k + 1])
                pred.setdefault(e[k + 1], set()).add(e[k])
            if with_trigrams:
                for k in range(len(e) - 2):
                    tri.add((e[k], e[k + 1], e[k + 2]))
    ps = state.parents
    gidf = np.zeros(n, dtype=np.float64)
    if ps is not None:
        for t in range(n):
            try:
                gidf[t] = float(ps._gidf(t))  # noqa: SLF001
            except Exception:
                gidf[t] = 0.0
    struct = np.array([is_structural_token(v, i) for i in range(n)], dtype=bool)
    # Van SỐ — token nào CHỨA chữ số thì bất khả xâm phạm. Lớp KÝ TỰ tính sẵn lúc
    # build vocab (§0.0 cho phép; tiền lệ `matrix_core.inner_digit_ids`), KHÔNG bảng-từ.
    has_digit = np.array(
        [any(ch.isdigit() for ch in (v.token_of(i) or "")) for i in range(n)], dtype=bool
    )
    return SlotIndex(succ=succ, pred=pred, trigrams=tri, gidf=gidf, struct=struct,
                     has_digit=has_digit)


def ensure_slot_index(state, *, with_trigrams: bool = True) -> SlotIndex:
    got = cache_get(_SLOT_CACHE, state, bool(with_trigrams))
    if got is not None:
        return got
    return cache_put(
        _SLOT_CACHE, state, build_slot_index(state, with_trigrams=with_trigrams),
        bool(with_trigrams),
    )


def clear_slot_cache() -> None:
    _SLOT_CACHE.clear()


def swap_slots(
    index: SlotIndex,
    tokens,
    *,
    order: int = DEFAULT_ORDER,
    max_gidf: float = DEFAULT_MAX_GIDF,
    rng=None,
    max_swaps: int | None = None,
    min_gap: int = 2,
    pmi_index=None,
    theme_ids=None,
    pmi_min: float = 0.0,
) -> tuple[list[int], int]:
    """Thay token ở các khe hợp lệ. Trả `(chuỗi mới, số lần thay)`.

    `min_gap` — hai lần thay phải cách nhau ≥ ngần này vị trí, để không đụng cùng một
    bigram hai lần (nếu thay liền kề thì mối nối `(x_i, x_{i+1})` **chưa được chứng thực**
    và bảo chứng attestation gãy — đây là chỗ dễ hỏng nhất của cơ chế).
    """
    import random as _rnd

    rng = rng or _rnd.Random(0)
    toks = [int(t) for t in tokens]
    if len(toks) < 3:
        return toks, 0
    positions = list(range(1, len(toks) - 1))
    rng.shuffle(positions)
    out = list(toks)
    used: list[int] = []
    n_swap = 0
    for i in positions:
        if max_swaps is not None and n_swap >= max_swaps:
            break
        if any(abs(i - u) < min_gap for u in used):
            continue
        cur = out[i]
        if index.gidf is not None and float(index.gidf[cur]) > float(max_gidf):
            continue
        # Van SỐ: `gidf` KHÔNG bảo vệ được năm/số. TK_041 §0.4.3 cưỡng chế tách chữ số
        # ⇒ mỗi chữ số là atom PHỔ BIẾN ⇒ gidf THẤP ⇒ lọt van gidf một cách tầm thường.
        # Đọc tay bắt được: `năm 1989`→`1929`, `năm 1290`→`1890`, `(1864)`→`11264`.
        # Không metric nào báo động — đúng bài học TK_041 §5.2.
        if index.has_digit is not None and (
            bool(index.has_digit[cur])
            or bool(index.has_digit[out[i - 1]])
            or bool(index.has_digit[out[i + 1]])
        ):
            continue  # không đụng token số, cũng không đụng LÁNG GIỀNG của số
        fill = index.fillers(out[i - 1], out[i + 1], order=order, max_gidf=max_gidf)
        fill.discard(cur)
        if not fill:
            continue
        cands = sorted(fill)
        if pmi_index is not None and pmi_min > 0 and theme_ids:
            cands = [
                x for x in cands
                if float(pmi_index.score(list(theme_ids), [x])) >= float(pmi_min)
            ]
            if not cands:
                continue
        out[i] = int(rng.choice(cands))
        used.append(i)
        n_swap += 1
    return out, n_swap
