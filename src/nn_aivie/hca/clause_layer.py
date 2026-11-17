r"""TK_045 §13 Trụ A — TẦNG NƠ-RON MỆNH ĐỀ (ClauseNeuron).

## Vì sao tồn tại (một câu)

Đồ thị đếm là **bộ CHỌN xuất sắc** (match-theo-`d` +31pp §12.122, `rank_matches` +13.5pp
§12.126) và **bộ SINH-TOKEN vô vọng** (`H(next|w0)=6.42 bits`, v4 §3.7; W_skip dự đoán
−14.5pp §12.120; salad §12.132). Hệ chỉ có hai độ hạt — **token** (6.42 bits → salad) và
**cả emit của cha** (→ chép 0.93–1.00 §12.72). Tầng này là **khoảng giữa**: sinh = CHỌN một
mệnh đề đã được chứng thực, nên bài toán sinh trở thành bài toán đồ thị làm giỏi.

## Hai tín hiệu biên — CẢ HAI đều HỌC, không danh sách dấu câu (Luật 0)

| biên | tín hiệu | vì sao |
|---|---|---|
| **đơn vị** (câu) | `P(EOS\|w)` lift ≥ θ | §0.4.2 — `ingest._with_eos` dạy; `'.'` lift 18.5 vs `'là'` 0.0 |
| **mệnh đề** (dưới câu) | thung lũng `pmi(a→b)` trên `W_seq` | mối nối lỏng nhất; đúng tiêu chí ISG/BPE dùng để GỘP, đọc ngược thành CẮT |

⚠️ EOS-lift **một mình là KHÔNG ĐỦ** — đo P47: chỉ `!`/`.`/`?` có lift > 0, mọi token khác
= 0.0 ⇒ sweep θ là no-op, mệnh đề dài 16.4 token. `,` không bao giờ đứng trước EOS.
§0.4.2 đã bác hai tín hiệu dẫn xuất khác (entropy hàng W_seq, tỉ lệ đứng-cuối) — cả hai đo
**token đơn**; PMI đo **CẶP**, đại lượng khác, và nó ăn.

## Số đo mở cổng (P47, `mini_sent_ep_qa`, §13.6)

`PMI=0.0` → 10 283 mệnh đề · dài TB 10.72 · ứng viên kế TB 1 821 · **KHÁC-CHA 99.8%** ·
**0.569 bits/token** (bigram = 6.42). Tự kiểm: mệnh đề co về ~5.9 token thì `H_type → 6.59`
≈ 6.42 đo độc lập ⇒ instrument đúng.

⚠️ `bits/token` đo **TRÔI CHẢY**, KHÔNG đo **đúng nội dung** — chọn mệnh đề là cam kết luôn
fact trong đó. Giá nội dung là việc của Trụ C (cổng phủ), đo riêng (`lạc`/PMI-cross).

## MỘT định nghĩa DUY NHẤT

`scripts/probe_clause_layer.py` **import từ đây**, không tự định nghĩa lại — tránh đúng lớp
lỗi mà `core/neural_lm.py` cảnh báo (hai bên định nghĩa riêng → lệch âm thầm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nn_aivie.core.adjacency import EdgeKind
from nn_aivie.core.tokenizer import SPECIAL_EOS
from nn_aivie.hca.context_positional import is_structural_token
from nn_aivie.hca.objcache import cache_get, cache_put

_LAYER_CACHE: dict = {}
_BOUND_CACHE: dict = {}

# Vùng vận hành đo được (§13.6) — PMI ∈ [0.0, 0.5]; 0.0 = dài TB ~10.7 token.
DEFAULT_PMI_THETA = 0.0
DEFAULT_EOS_LIFT = 2.0  # ≡ ar_eos_boundary_min_lift
DEFAULT_MIN_LEN = 2
# §13.18 — bão hoà hub token-mở. 0 = tắt. Chẩn mini: top 10 token-mở chiếm 50% mối nối
# (`Ġlà` 16.8%, `Ġvà` 10.1%) ⇒ "chuỗi hút" §13.17. Cùng họ `oja_indeg` §12.140.
DEFAULT_HUB_BETA = 1.0


# --------------------------------------------------------------- tín hiệu biên
@dataclass
class BoundarySignal:
    """Hai bảng biên đọc từ W_seq. Dựng một lần / state, cache weakref."""

    lift: np.ndarray          # lift[w] = P(EOS|w) / P(EOS) nền
    eos_base: float
    _rowsum: np.ndarray = field(repr=False, default=None)
    _colmass: np.ndarray = field(repr=False, default=None)
    _total: float = field(repr=False, default=0.0)
    _seq: object = field(repr=False, default=None)
    _rowcache: dict = field(repr=False, default_factory=dict)

    def pmi(self, a: int, b: int) -> float:
        """log2(P(b|a)/P(b)); −inf nếu cặp CHƯA từng thấy ⇒ mối nối không chứng thực ⇒ cắt."""
        a, b = int(a), int(b)
        row = self._rowcache.get(a)
        if row is None:
            row = self._seq.row(a) or {}
            self._rowcache[a] = row
        w = row.get(b, 0.0)
        if w <= 0 or self._rowsum[a] <= 0 or self._colmass[b] <= 0 or self._total <= 0:
            return float("-inf")
        return float(math.log2((w / self._rowsum[a]) / (self._colmass[b] / self._total)))

    def _w(self, a: int, b: int) -> float:
        a = int(a)
        row = self._rowcache.get(a)
        if row is None:
            row = self._seq.row(a) or {}
            self._rowcache[a] = row
        return float(row.get(int(b), 0.0))

    def p_after(self, a: int, b: int) -> float:
        """P(b đứng NGAY SAU a) — chuẩn hoá theo tổng ra của a."""
        s = float(self._rowsum[int(a)]) if 0 <= int(a) < self._rowsum.size else 0.0
        return (self._w(a, b) / s) if s > 0 else 0.0

    def p_before(self, a: int, b: int) -> float:
        """P(a đứng NGAY TRƯỚC b) — chuẩn hoá theo tổng vào của b."""
        m = float(self._colmass[int(b)]) if 0 <= int(b) < self._colmass.size else 0.0
        return (self._w(a, b) / m) if m > 0 else 0.0

    def ends_unit(self, tid: int, theta: float = DEFAULT_EOS_LIFT) -> bool:
        t = int(tid)
        return 0 <= t < self.lift.size and float(self.lift[t]) >= float(theta)


def build_boundary_signal(state, *, logger=None) -> BoundarySignal:
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    n = len(state.vocab)
    eos_id = int(state.vocab.id_of(SPECIAL_EOS))
    rowsum = np.zeros(n, dtype=np.float64)
    colmass = np.zeros(n, dtype=np.float64)
    to_eos = tot = 0.0
    for i, j, w in seq.items():
        if 0 <= i < n:
            rowsum[i] += w
        if 0 <= j < n:
            colmass[j] += w
        tot += w
        if j == eos_id:
            to_eos += w
    base = (to_eos / tot) if tot > 0 else 0.0
    lift = np.zeros(n, dtype=np.float64)
    if base > 0:
        for w0 in range(n):
            r = seq.row(w0)
            if not r:
                continue
            tt = sum(r.values())
            if tt > 0:
                lift[w0] = (r.get(eos_id, 0.0) / tt) / base
    elif logger is not None:
        logger.log("⚠️ P(EOS) nền = 0 — checkpoint train TRƯỚC đợt EOS (§0.4.2).")
    return BoundarySignal(
        lift=lift, eos_base=base, _rowsum=rowsum, _colmass=colmass,
        _total=tot, _seq=seq,
    )


def ensure_boundary_signal(state, *, logger=None) -> BoundarySignal:
    got = cache_get(_BOUND_CACHE, state)
    if got is not None:
        return got
    return cache_put(_BOUND_CACHE, state, build_boundary_signal(state, logger=logger))


# ------------------------------------------------------------------ mệnh đề
@dataclass(frozen=True)
class ClauseNeuron:
    parent_idx: int
    tokens: tuple[int, ...]
    first_content: int   # token nội dung ĐẦU — dùng làm chữ ký TRÁI (mối nối vào)
    last_content: int    # token nội dung CUỐI — chữ ký PHẢI (mối nối ra)
    pos_in_parent: int
    n_in_parent: int

    @property
    def opens_parent(self) -> bool:
        return self.pos_in_parent == 0


@dataclass
class ClauseLayer:
    clauses: list[ClauseNeuron]
    by_opener: dict[int, list[int]]   # first_content → clause ids
    by_parent: dict[int, list[int]]
    pmi_theta: float
    eos_theta: float
    mean_len: float = 0.0

    def __len__(self) -> int:
        return len(self.clauses)


def cut_clauses(
    emit,
    bound: BoundarySignal,
    *,
    eos_theta: float = DEFAULT_EOS_LIFT,
    pmi_theta: float | None = DEFAULT_PMI_THETA,
    min_len: int = DEFAULT_MIN_LEN,
) -> list[list[int]]:
    """Cắt sau token có EOS-lift ≥ θ (biên đơn vị) và ở thung lũng PMI (biên mệnh đề).

    `pmi_theta=None` ⇒ chỉ cắt câu.
    """
    out: list[list[int]] = []
    cur: list[int] = []
    n_e = len(emit)
    for k, t in enumerate(emit):
        t = int(t)
        cur.append(t)
        if bound.ends_unit(t, eos_theta):
            out.append(cur)
            cur = []
            continue
        if pmi_theta is not None and k + 1 < n_e and len(cur) >= min_len:
            if bound.pmi(t, int(emit[k + 1])) <= float(pmi_theta):
                out.append(cur)
                cur = []
    if cur:
        out.append(cur)
    return out


def make_content_filter(state, bound: BoundarySignal, *, eos_theta: float = DEFAULT_EOS_LIFT):
    """Token nội dung = surface ∧ không cấu trúc ∧ không phải mốc kết đơn vị."""
    v = state.vocab
    n = len(v)
    struct = np.array([is_structural_token(v, i) for i in range(n)], dtype=bool)
    surf = np.array([v.is_surface(i) for i in range(n)], dtype=bool)
    lift = bound.lift

    def is_content(t: int) -> bool:
        t = int(t)
        return (
            0 <= t < n
            and bool(surf[t])
            and not bool(struct[t])
            and float(lift[t]) < float(eos_theta)
        )

    return is_content


def build_clause_layer(
    state,
    *,
    pmi_theta: float | None = DEFAULT_PMI_THETA,
    eos_theta: float = DEFAULT_EOS_LIFT,
    min_len: int = DEFAULT_MIN_LEN,
    max_parents: int = 0,
    parent_ids=None,
    logger=None,
) -> ClauseLayer:
    """`parent_ids` — chỉ dựng trên tập cha này (dùng cho đường cong co-giãn dữ liệu).

    ⚠️ `max_parents` lấy N cha ĐẦU, mà thứ tự cha = thứ tự nạp ⇒ N đầu là **một dataset**,
    không phải mẫu đại diện. Đo co-giãn thì phải truyền `parent_ids` lấy mẫu NGẪU NHIÊN.
    """
    ps = state.parents
    if ps is None or not len(ps.parents):
        return ClauseLayer([], {}, {}, float(pmi_theta or 0.0), eos_theta)
    bound = ensure_boundary_signal(state, logger=logger)
    is_content = make_content_filter(state, bound, eos_theta=eos_theta)

    clauses: list[ClauseNeuron] = []
    by_opener: dict[int, list[int]] = {}
    by_parent: dict[int, list[int]] = {}
    n_par = len(ps.parents)
    if parent_ids is not None:
        order = [int(x) for x in parent_ids if 0 <= int(x) < n_par]
    else:
        lim = n_par if max_parents <= 0 else min(n_par, max_parents)
        order = range(lim)
    for pi in order:
        emit = ps.parents[pi].emit_ids
        if not emit:
            continue
        spans = cut_clauses(
            emit, bound, eos_theta=eos_theta, pmi_theta=pmi_theta, min_len=min_len,
        )
        kept: list[tuple[tuple[int, ...], int, int]] = []
        for span in spans:
            cont = [t for t in span if is_content(t)]
            if not cont:
                continue
            kept.append((tuple(span), int(cont[0]), int(cont[-1])))
        for pos, (toks, first, last) in enumerate(kept):
            ci = len(clauses)
            clauses.append(ClauseNeuron(
                parent_idx=pi, tokens=toks, first_content=first, last_content=last,
                pos_in_parent=pos, n_in_parent=len(kept),
            ))
            by_opener.setdefault(first, []).append(ci)
            by_parent.setdefault(pi, []).append(ci)
        if logger is not None and pi and pi % 50000 == 0:
            logger.log(f"  … mệnh đề: cha {pi:,} → {len(clauses):,}")

    mean_len = (
        float(np.mean([len(c.tokens) for c in clauses])) if clauses else 0.0
    )
    return ClauseLayer(
        clauses=clauses, by_opener=by_opener, by_parent=by_parent,
        pmi_theta=float(pmi_theta) if pmi_theta is not None else float("nan"),
        eos_theta=float(eos_theta), mean_len=mean_len,
    )


def ensure_clause_layer(
    state,
    *,
    pmi_theta: float | None = DEFAULT_PMI_THETA,
    eos_theta: float = DEFAULT_EOS_LIFT,
    min_len: int = DEFAULT_MIN_LEN,
    logger=None,
) -> ClauseLayer:
    sub = (pmi_theta, float(eos_theta), int(min_len))
    got = cache_get(_LAYER_CACHE, state, sub)
    if got is not None:
        return got
    built = build_clause_layer(
        state, pmi_theta=pmi_theta, eos_theta=eos_theta, min_len=min_len, logger=logger,
    )
    return cache_put(_LAYER_CACHE, state, built, sub)


def clear_clause_layer_cache() -> None:
    _LAYER_CACHE.clear()
    _BOUND_CACHE.clear()


# ------------------------------------------------------------------ mối nối
def opener_is_word_start(vocab, tid: int) -> bool:
    """Token này có MỞ ĐẦU một từ mới không (`Ġ`/`Ċ`)?

    §13.18 — nối hai mệnh đề mà token đầu của mệnh đề sau KHÔNG mở từ thì bề mặt **dính
    liền**: `"năm 1989"` + `"000 hố va chạm"` → `"1989000"`. Chấm mù bắt được (§13.17 lỗi 2);
    không thước nào trong sáu thước báo động — `attestation` còn thấy hợp lệ vì cả hai
    bigram đều có thật, chỉ **ranh giới đơn vị** là hỏng.
    Thuộc-tính KÝ TỰ tính từ vocab (tiền lệ `matrix_core.inner_digit_ids`), không bảng-từ.
    """
    if not getattr(vocab, "is_bpe", False):
        # Vocab KHÔNG phải BPE (checkpoint legacy, fixture test) thì không có marker `Ġ`
        # ⇒ không có hiện tượng dính bề mặt ⇒ luật này phải là NO-OP, chứ không phải
        # lọc sạch mọi ứng viên. Bắt được khi test đồ chơi đỏ ngay sau khi wire.
        return True
    s = vocab.token_of(int(tid)) or ""
    return s[:1] in ("Ġ", "Ċ")


def join_candidates(
    layer: ClauseLayer,
    ws,
    last_content: int,
    *,
    d_join: int = 4,
    hub_beta: float = 0.0,
    vocab=None,
    require_word_start: bool = False,
) -> dict[int, float]:
    """Mệnh đề kế ứng viên → sức mạnh mối nối, ĐÃ ĐƯỢC CHỨNG THỰC trên W_skip.

    Mối nối = (token nội dung CUỐI của mệnh đề đang có) → (token nội dung ĐẦU của ứng
    viên). Chỉ cạnh có thật mới vào ⇒ **chỗ ghép không thể là salad về cấu trúc**.
    Trả `{opener_token: trọng số}` — caller chọn mệnh đề trong `layer.by_opener[opener]`.
    """
    out: dict[int, float] = {}
    if not layer.by_opener:
        return out
    for d in range(1, int(d_join) + 1):
        if d > ws.d_max:
            break
        row = ws.mats[d].getrow(int(last_content))
        if row.nnz == 0:
            continue
        for j, w in zip(row.indices, row.data):
            j = int(j)
            if j in layer.by_opener:
                out[j] = out.get(j, 0.0) + float(w)
    if require_word_start and vocab is not None:
        out = {j: w for j, w in out.items() if opener_is_word_start(vocab, j)}
    if hub_beta > 0.0 and out:
        # §13.18 — BÃO HOÀ HUB. Chẩn (mini, n=800): `Ġlà` đứng đầu **16.8%** mối nối
        # (200 mệnh đề cùng mở), `Ġvà` 10.1%, `0` 7.7%; **top 10 token-mở chiếm 50%**
        # và chỉ 178/2159 token-mở từng được chọn. Đó chính là "chuỗi hút" mà chấm mù
        # tả (§13.17 lỗi 1): mọi prompt đều rơi về cùng vài hư-từ mở.
        # Hạ theo CỠ NHÓM: token-mở có nhiều mệnh đề = mở chung chung, ít thông tin.
        # Cùng họ với `oja_indeg` §12.140, và Luật 0 sạch (đếm, không bảng-từ).
        out = {
            j: w / (1.0 + hub_beta * math.log1p(len(layer.by_opener.get(j, ()))))
            for j, w in out.items()
        }
    return out


# --------------------------------------------------------- Trụ C — CỔNG PHỦ
GIDF_CONTENT_MIN = 4.0  # ≡ ngưỡng fold nội dung của thước `lạc` (eval_gen_broad §12.73)


def content_folds(
    state,
    ids,
    *,
    gidf_min: float = GIDF_CONTENT_MIN,
    drop_digits: bool = False,
    pair: bool = False,
) -> set[str]:
    """Tập fold nội dung ĐẶC TRƯNG của một chuỗi — **cùng định nghĩa với thước `lạc`**.

    🔴 **§13.20 — fold ÂM TIẾT LẺ gần như vô nghĩa trong tiếng Việt.** Đo được: 86.4%
    bằng chứng "đúng đề" của cổng phủ là fold ≤4 ký tự, và đứng đầu là **chữ số**
    (`'4'` 10 lượt, `'3'` 7, `'7'` 7) cùng âm tiết chung (`'ông'`, `'từ'`, `'dân'`).
    Chấm mù bắt đúng hệ quả: `lục lạp`→`lục bát`, `Mã Lai`→`Gia Lai`, `ly tâm`→`thoát ly`.
    Tiếng Việt đơn âm tiết ⇒ một âm tiết là **đồng âm hàng loạt**; khớp trên nó ≈ khớp
    trên không có gì. ⇒ `lạc = 0.000` của cổng phủ **phần lớn RỖNG**.

    Hai van, đều Luật 0 (đếm / lớp ký tự, không bảng-từ):

    * `drop_digits` — bỏ fold chứa chữ số (tiền lệ `matrix_core.inner_digit_ids`).
    * `pair` — đơn vị khớp là **CẶP fold nội dung KỀ NHAU** (`"lục|lạp"` ≠ `"lục|bát"`)
      thay vì âm tiết lẻ. Đây là van chính: nó đưa đơn vị khớp lên trên mức đồng âm.
    """
    from nn_aivie.hca.matrix_core import _path_fold

    ps = state.parents
    v = state.vocab
    seq: list[str] = []
    for t in ids:
        t = int(t)
        # Token CẤU TRÚC (`.`, `<EOS>`, newline…) không bao giờ là bằng chứng "cùng đề".
        # Trước đây chỉ lọc bằng gidf — mà `<EOS>` lọt, làm cặp `'.|<eos>'` thành fold
        # chung của MỌI câu. Test đồ chơi bắt được (`lục lạp` vs `lục bát` giao đúng ở đó).
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            g = float(ps._gidf(t))  # noqa: SLF001
        except Exception:
            continue
        f = _path_fold(v, t)
        if not f or g < gidf_min:
            continue
        if drop_digits and any(ch.isdigit() for ch in f):
            continue
        seq.append(f)
    if not pair:
        return set(seq)
    # CẶP KỀ trong dòng fold nội dung — giữ thứ tự, nên `"lục|lạp"` ≠ `"lạp|lục"`
    return {f"{a}|{b}" for a, b in zip(seq, seq[1:])}


def fold_gidf_map(state, ids, *, gidf_min: float = GIDF_CONTENT_MIN) -> dict[str, float]:
    """fold nội dung → `gidf` (độ HIẾM) của nó. Cùng bộ lọc với `content_folds`.

    Tách riêng vì `content_folds` trả `set` — mất trọng số, mà trọng số mới là thứ đáng
    giá (§13.31). `gidf` do `ParentStore` đếm sẵn ⇒ Luật 0 sạch.
    """
    from nn_aivie.hca.matrix_core import _path_fold

    ps, v = state.parents, state.vocab
    out: dict[str, float] = {}
    for t in ids:
        t = int(t)
        if is_structural_token(v, t) or not v.is_surface(t):
            continue
        try:
            g = float(ps._gidf(t))  # noqa: SLF001
        except Exception:
            continue
        f = _path_fold(v, t)
        if f and g >= gidf_min:
            out[f] = max(out.get(f, 0.0), g)
    return out


def prompt_anchor_rescorer(state, prompt_ids, *, gidf_min: float = GIDF_CONTENT_MIN):
    """§13.31 — xếp mối nối theo **neo vào ĐỀ, có trọng số ĐỘ HIẾM**.

    Trả một `join_rescore(prev, cand) -> float` cho `compose_chain`; điểm là
    `Σ gidf(f)` trên các fold nội dung mà `cand` **dùng chung với prompt**.

    ## Vì sao là cái này, sau sáu vòng chấm mù

    Mọi vòng đều nói cùng một điều: **nới lỏng ràng buộc trùng MẶT CHỮ với prompt là thua.**
    Cổng fold thắng không-cổng 92%, thắng cổng `pmi` 85.6% (mini) và 71.7% (v12); và bộ xếp
    `pmi(prev,·)` thua bộ xếp một-token 59.7% (§13.30). Thứ thắng luôn là thứ neo **thô và
    theo mặt chữ** vào **đề**, không phải thứ đo mạch lạc **cục bộ** giữa hai mệnh đề.

    Hai chỗ sửa so với các lần trước:

    1. **XẾP HẠNG, không phải CỔNG.** Mọi tiêu chí chặt trước đây đều dựng thành cổng và
       chết vì tắc mạch (P57: cặp-fold ⇒ chết 94%). Bể ứng viên có **trung vị 1.550 mệnh
       đề** (đo §13.31) ⇒ làm bộ xếp thì chặt tới đâu cũng **không bao giờ tắc**, chỉ đổi
       thứ tự. Đây là điều chưa từng thử.
    2. **Trọng số `gidf` thay vì ĐẾM.** §13.20: 86.4% bằng chứng "đúng đề" của cổng phủ là
       mảnh dưới-từ, đứng đầu là **chữ số** và âm tiết chung (`ông`, `từ`, `dân`). Đếm fold
       ⇒ những thứ đó lấn át. Nhân với `gidf` ⇒ `Trịnh`, `lạp` nặng hơn `4`, `dân` rất
       nhiều, mà **không** cần cặp-fold — cặp chỉ nổ **4.2%** số mối nối (khớp P57 5.2%),
       tức gần như bất động.

    ⚠️ **Chưa phải kết luận.** Đo được tới đây: sống ở **100%** mối nối (trung vị trọng số
    5.3), đổi đầu ra **38.3%/120 prompt**, và đọc tay 3/3 ca đổi đều bỏ được câu toán chen
    ngang. Arbiter là **chấm mù** — §13.31 đã ghi vì sao mọi proxy ngoại tuyến ở đây đều
    không đáng tin (nhiệm vụ vàng ≠ nhiệm vụ sinh).
    """
    pg = fold_gidf_map(state, prompt_ids, gidf_min=gidf_min)

    def rescore(_prev, cand) -> float:
        if not pg:
            return 0.0
        cf = fold_gidf_map(state, cand.tokens, gidf_min=gidf_min)
        return sum(pg[f] for f in cf if f in pg)

    return rescore


def word_pivot_strength(bound: BoundarySignal, w_left: int, hinge: int, w_right: int) -> float:
    """§13.39 — bản lề có đang **ĐỔI VAI TRONG TỪ** không?

    Bệnh §13.25 đọc lại cho đúng: không phải "đồng tự" chung chung, mà là bản lề vừa là âm
    tiết **CUỐI** của một từ trong A, vừa bị mối nối đọc thành âm tiết **ĐẦU** của một từ
    khác — `quốc gia` ⇒ `gia vị` · `tối đa` ⇒ `đa số` · `tư duy` ⇒ `duy nhất`.

    Chữ ký đếm được: bản lề bị buộc chặt **cả hai phía**. Trả `min` của hai xác suất — cao
    nghĩa là **cả hai** liên kết đều mạnh, tức đúng cái bập bênh gây trượt nghĩa.

    P73 đo trên mini: **17,3%** mối nối SINH có `min ≥ 0.05`, so với **0,6%** mối nối THẬT
    của người viết (§13.39) — chênh ~29 lần. Đây là con số cứng đầu tiên cho bệnh gốc sau
    tám vòng chấm mù chỉ thấy nó bằng ví dụ.

    Chỉ đọc `W_seq` (đếm) ⇒ Luật 0 sạch.
    """
    if w_left is None or hinge is None or w_right is None:
        return 0.0
    return min(bound.p_before(int(w_left), int(hinge)),
               bound.p_after(int(hinge), int(w_right)))


@dataclass
class CoverageGate:
    """TK_045 §13 Trụ C — cổng phủ ở tầng MỆNH ĐỀ.

    **Vì sao tồn tại:** v4 §1.5 đo được cơ chế **DUY NHẤT** miễn nhiễm trôi đề là chép-đáp QA
    (`lạc = 0.00` ở mọi cap, §12.74) — và lý do là nó là cơ chế duy nhất **có cổng phủ nội
    dung**. Mọi đường khác trôi 0.12→0.27. Trụ C tổng quát hoá đúng cái cổng đó.

    ⚠️ §12.104 đã đo **attestation làm acceptor ở tầng TOKEN → TRƯỢT**. Đây là thí nghiệm
    KHÁC: acceptor ở tầng **mệnh đề**, dùng PMI-cross — thước tương quan tốt nhất với điểm
    chấm mù (Spearman **+0.643**, §12.117). Nếu vẫn trượt thì họ acceptor đóng hẳn.

    Ba điều kiện TRỰC GIAO (AND), bật riêng được:

    | # | trường | nghĩa | tự tại? |
    |---|---|---|---|
    | 1 | `anchor_min_folds` | chia ≥k fold nội dung với chủ đề — cổng bám đề CỨNG | ⚠️ **TỰ TẠI với thước `lạc`** (xem dưới) |
    | 2 | `pmi_min` | PMI-cross(mệnh đề, chủ đề) ≥ θ — bám đề MỀM | không — tín hiệu khác thước |
    | 3 | `require_new_cover` | mang ≥1 fold nội dung CHƯA phủ — đẩy mạch, không giậm chân | không |

    ⚠️ **Điều kiện 1 tự tại với `lạc`**: `lạc` được định nghĩa đúng là *"không chia fold nội
    dung nào với prompt"*, nên bật (1) thì `lạc → 0` **theo cấu tạo**, không phải theo phát
    hiện. Nó vẫn đáng đo — nhưng câu hỏi hợp lệ là **GIÁ** (`dead_rate`, độ dài chuỗi), không
    phải "có ăn không". Đây đúng là điều chép-đáp QA đang làm (`lạc=0.00`, §12.74); vấn đề
    luôn là nó có sống nổi ngoài khung QA không.

    Điều kiện 2 là cái **KHÔNG tự tại** — nó phải được calibrate theo phân bố PMI thật
    (đo mini: p10=0.53 · p50=1.25 · p90=2.23; đặt θ=0.5 là **no-op**, dưới cả p10).
    """

    pmi_index: object | None = None
    pmi_min: float = 0.0
    theme_ids: tuple[int, ...] = ()
    theme_folds: frozenset[str] = frozenset()
    anchor_min_folds: int = 0
    require_new_cover: bool = False
    # Thang gidf phụ thuộc corpus — corpus nhỏ có gidf thấp hơn hẳn. Để 4.0 cứng trong
    # cổng thì trên checkpoint khác nó lặng lẽ trả fold RỖNG ⇒ cổng thành no-op (hoặc
    # loại sạch) mà không báo gì. Test đồ chơi đã bắt đúng ca này.
    gidf_min: float = GIDF_CONTENT_MIN
    # §13.20 — định nghĩa fold của CỔNG, tách khỏi định nghĩa của THƯỚC. Nếu hai bên
    # dùng chung một định nghĩa thì `lạc` tự tại và không phán được gì (đã vấp §13.10).
    drop_digits: bool = False
    pair_folds: bool = False
    # §13.48 — VÁ BẪY IM LẶNG. `accept` chỉ xét neo khi `theme_folds` KHÁC RỖNG, nên nâng
    # `gidf_min` lên mức ĐẶC TRƯNG (≥7) sẽ làm nhiều đề không còn fold nào ⇒ `theme_folds`
    # rỗng ⇒ cổng **TẮT HẲN** và nhận mọi ứng viên — tức siết cổng lại hoá ra **mở toang**
    # nó, âm thầm, đúng ở những đề khó nhất. Chính docstring của lớp này đã cảnh báo dạng
    # bẫy đó ("trả fold RỖNG ⇒ cổng thành no-op … mà không báo gì") mà cơ chế thì chưa vá.
    # Bật cờ này ⇒ đề không có fold đặc trưng thì **không sinh gì**, đúng §13.37 (im lặng
    # hơn nói bậy). Mặc định False = giữ nguyên hành vi cũ.
    reject_when_no_theme: bool = False

    def _folds(self, state, ids) -> set[str]:
        return content_folds(
            state, ids, gidf_min=self.gidf_min,
            drop_digits=self.drop_digits, pair=self.pair_folds,
        )

    def accept(self, state, clause: ClauseNeuron, covered: set[str]) -> bool:
        cf = None
        if (self.reject_when_no_theme and self.anchor_min_folds > 0
                and not self.theme_folds):
            return False
        if self.anchor_min_folds > 0 and self.theme_folds:
            cf = self._folds(state, clause.tokens)
            if len(cf & self.theme_folds) < self.anchor_min_folds:
                return False
        if self.require_new_cover:
            if cf is None:
                cf = self._folds(state, clause.tokens)
            # "đẩy mạch" = mang fold nội dung MỚI so với những gì ĐÃ phủ.
            # (Bản đầu so với `agenda_folds` mà `covered` đã chứa sẵn ⇒ luôn FALSE ⇒
            #  loại sạch, dead=100%. Lỗi ngữ nghĩa của tôi, ghi lại ở §13.10.)
            if not (cf - covered):
                return False
        if self.pmi_index is not None and self.pmi_min > 0 and self.theme_ids:
            if float(self.pmi_index.score(list(self.theme_ids), list(clause.tokens))) < self.pmi_min:
                return False
        return True


def order_join_candidates(
    layer: ClauseLayer,
    order,
    prev: "ClauseNeuron | None",
    *,
    join_rescore=None,
    rng=None,
    top_p_pool: int = 8,
) -> list[tuple[int, list[int]]]:
    """Xếp danh sách (token-mở, ứng viên) — tách ra để TEST ĐƯỢC.

    §13.28 — hai lỗi của bản §13.26, cả hai đều **im lặng**:

    1. `members.sort(rescore)` rồi NGAY SAU đó `rng.shuffle(members)` ⇒ shuffle **xoá sạch**
       thứ tự vừa xếp ⇒ `join_rescore` là **NO-OP**. Đầu ra vẫn "đổi 69/120" chỉ vì shuffle
       nhận list đầu vào khác thứ tự nên cho hoán vị khác ⇒ **phép đo end-to-end KHÔNG phát
       hiện được**, và chấm mù vòng năm so **nhiễu với nhiễu** (47.3%).
    2. Kể cả khi sort sống, nó chỉ xếp lại **TRONG** một token-mở — mà **bản lề đồng âm
       CHÍNH LÀ token-mở** (`nhân`, `bình`, `tĩnh`, §13.27). Xếp lại nội bộ nhóm nằm **hạ
       nguồn** của sai lầm nên không đời nào chữa được nó.

    Đúng: bốc ngẫu nhiên phá thế hoà **TRƯỚC**, chấm từng (token-mở, ứng viên tốt nhất), rồi
    xếp **cả danh sách token-mở** theo điểm đó.
    """
    import random as _rnd

    rng = rng or _rnd.Random(0)
    rows: list[tuple[int, list[int]]] = []
    for opener, _w in order:
        mem = list(layer.by_opener.get(opener, ()))
        rng.shuffle(mem)                       # phá thế hoà NGẪU NHIÊN trước
        rows.append((opener, mem[:top_p_pool]))
    if join_rescore is None or prev is None:
        return rows
    scored = [
        (max((float(join_rescore(prev, layer.clauses[c])) for c in mem),
             default=float("-inf")), op, mem)
        for op, mem in rows
    ]
    scored.sort(key=lambda x: -x[0])
    return [(op, mem) for _s, op, mem in scored]


def compose_chain(
    state,
    layer: ClauseLayer,
    ws,
    prompt_ids,
    *,
    # §13.52 — MẶC ĐỊNH 1, đổi từ 6. Ba vòng chấm mù độc lập, mỗi vòng ít mối nối hơn thì
    # thắng: `=2` thắng `=5` **78,7%** (§13.41) · `len2` thắng `dong_cau` **70,7%** (§13.42) ·
    # thí nghiệm CÓ GÁN `len1` vs `len2` **13,0% vs 4,5%**, z=3,01 (§13.52).
    #
    # Lý do cơ chế, đã đo: **mỗi mối nối đều làm hỏng đầu ra** — tỷ lệ hỏng mỗi mối nối
    # ~10% và **phẳng theo vị trí** (§13.41), nên chuỗi dài chỉ có nhiều cơ hội hỏng hơn.
    #
    # ⚠️ Nói thẳng nghĩa của mặc định này: `max_clauses=1` ⇒ hệ **KHÔNG NỐI GÌ CẢ**. Nó chọn
    # một mệnh đề đã được chứng thực rồi dán vào đề — **truy hồi + ghép một bước**. Vế
    # "tái tổ hợp NHIỀU mệnh đề" của Trụ A nay có bằng chứng **phản đối**, không phải chỉ
    # thiếu bằng chứng ủng hộ. Ai muốn nâng lại thì phải bác §13.52, không phải chỉnh số.
    max_clauses: int = 1,
    d_join: int = 4,
    gate: CoverageGate | None = None,
    rng=None,
    top_p_pool: int = 8,
    gidf_min: float = GIDF_CONTENT_MIN,
    hub_beta: float = DEFAULT_HUB_BETA,
    no_glue: bool = True,
    join_rescore=None,
    pivot_max: float | None = None,
    stop_on_close: bool = False,
    min_clauses: int = 2,
    join_seq_max: float | None = None,
) -> tuple[list[ClauseNeuron], dict]:
    """Soạn một chuỗi mệnh đề: mỗi bước CHỌN một mệnh đề đã được chứng thực.

    Không có `gate` ⇒ chọn thuần theo sức mạnh mối nối (đối chứng cho Trụ C).
    Trả `(chuỗi, thống kê)`; `stats["dead"]`=True nếu tắc mạch trước `max_clauses`.
    """
    import random as _rnd

    rng = rng or _rnd.Random(0)
    chosen: list[ClauseNeuron] = []
    # neo = token nội dung CUỐI của prompt (chữ ký PHẢI để tìm mối nối)
    # `gidf_min` là THAM SỐ, không phải hằng 4.0 cứng: thang gidf phụ thuộc corpus —
    # trên corpus nhỏ mọi token đều dưới 4.0 ⇒ `content_folds` rỗng ⇒ `last=None` ⇒
    # chuỗi chết ngay ở bước 0, âm thầm. `CoverageGate` đã vá lỗi này; bộ soạn thì chưa.
    cont_prompt = [
        t for t in (int(x) for x in prompt_ids)
        if content_folds(state, (t,), gidf_min=gidf_min)
    ]
    last = cont_prompt[-1] if cont_prompt else None
    # §13.39 — chữ đứng NGAY TRƯỚC bản lề, thứ quyết định bản lề là âm tiết cuối của từ nào.
    # Bước 0 lấy từ ĐỀ; các bước sau lấy từ mệnh đề vừa chọn.
    _pi = list(prompt_ids)
    _k = len(_pi) - 1 - _pi[::-1].index(last) if (last is not None and last in _pi) else 0
    prev_left = int(_pi[_k - 1]) if _k > 0 else None
    # §13.42 — DỪNG CÓ CĂN CỨ. §13.41 đo được: mỗi mối nối hỏng ~10% **đều nhau theo vị
    # trí**, nên chuỗi dài thua vì TÍCH LUỸ chứ không vì đuôi tệ hơn. `len2` thắng 78,7%
    # nhưng là hằng số mù: 37% điểm dừng rơi GIỮA CÂU (giám khảo phạt 8 ca "cụt").
    # `stop_on_close` dừng ở mệnh đề ĐÓNG ĐƯỢC CÂU đầu tiên (EOS-lift, tín hiệu đã có sẵn),
    # sau khi đã đủ `min_clauses` ⇒ giữ cái thắng của len2, bỏ cái giá của nó.
    # §13.44 — ĐIỀU KIỆN HỢP LỆ của mối nối, không phải một bộ xếp nữa.
    #
    # Đo được: 46,6% mối nối sinh đặt cạnh nhau hai token **chưa bao giờ đứng kề nhau**
    # trong corpus (văn thật 5,2%) — vì `W_skip` chứng thực ĐỒNG HIỆN TRONG CỬA SỔ 8, không
    # chứng thực KỀ NHAU. Và pmi trung vị tại chỗ ghép là **+0,57**, trong khi `cut_clauses`
    # định nghĩa biên là `pmi ≤ pmi_theta` (=0.0) ⇒ bộ CẮT và bộ NỐI đang dùng hai tiêu chí
    # MÂU THUẪN nhau: nối ở đúng loại điểm mà chính hệ không coi là biên.
    #
    # `join_seq_max` bắt hai bộ dùng CÙNG một tiêu chí: chuyển tiếp bề mặt phải (a) được
    # chứng thực KỀ NHAU trên `W_seq`, và (b) nằm trong dải biên. Thuần đếm ⇒ Luật 0 sạch.
    _need_bound = pivot_max is not None or stop_on_close or join_seq_max is not None
    bound_pv = ensure_boundary_signal(state) if _need_bound else None
    covered: set[str] = content_folds(state, prompt_ids, gidf_min=gidf_min)
    n_rejected = 0
    dead = False
    # §13.36 — VÌ SAO DỪNG. `dead` chỉ nói CÓ tắc, không nói tắc ở đâu; suốt §13 mọi vòng đo
    # *chọn cái gì*, chưa vòng nào đo *vì sao dừng* trong khi chuỗi chết sớm ở đa số đề
    # (2,08/5 mệnh đề mỗi đề — §13.35). Đếm ở đây, trong mã THẬT, để probe khỏi phải chép lại
    # vòng lặp và trôi khỏi nó.
    stop = "du_max"          # chạy hết max_clauses — không tắc
    n_dup = n_glue = n_gate = n_pivot = n_seq = 0
    # đuôi BỀ MẶT của vế trước — token cuối thật, không phải `last_content`: đó mới là chỗ
    # hai mệnh đề chạm nhau khi đọc, và là chỗ `quốc gia` + `gia vị` dính lại thành từ.
    prev_tail = int(prompt_ids[-1]) if len(prompt_ids) else None
    pool_at_stop = 0
    for _ in range(int(max_clauses)):
        if last is None:
            dead = True
            stop = "khong_neo"      # không còn token nội dung làm neo
            break
        cand_open = join_candidates(
            layer, ws, last, d_join=d_join,
            hub_beta=hub_beta, vocab=state.vocab, require_word_start=no_glue,
        )
        if not cand_open:
            dead = True
            stop = "khong_ung_vien"  # neo có, nhưng W_skip không cho token-mở nào
            break
        # xếp token mở theo sức mạnh mối nối, duyệt xuống cho tới khi qua cổng
        order = sorted(cand_open.items(), key=lambda x: -x[1])[:top_p_pool]
        pick = None
        # §13.28 — XẾP LẠI Ở TẦNG TOKEN-MỞ, không phải trong nhóm.
        #
        # ⚠️ Bản §13.26 hỏng HAI chỗ, và chấm mù không phát hiện được vì nó đo nhầm thứ:
        #   (1) `members.sort(rescore)` rồi NGAY SAU đó `rng.shuffle(members)` — shuffle
        #       **xoá sạch** thứ tự vừa xếp ⇒ `join_rescore` là NO-OP. Đầu ra vẫn "đổi
        #       69/120" chỉ vì shuffle nhận list đầu vào khác thứ tự ⇒ hoán vị khác.
        #       Chấm mù vòng năm vì thế so **nhiễu với nhiễu** (47.3%, không tách) —
        #       kết quả đó KHÔNG nói gì về S3.
        #   (2) Kể cả khi sort sống, nó chỉ xếp lại TRONG một token-mở — mà **bản lề đồng
        #       âm CHÍNH LÀ token-mở** (`nhân`, `bình`, `tĩnh`). Xếp lại nội bộ nhóm nằm
        #       DƯỚI HẠ NGUỒN của sai lầm; không đời nào chữa được nó.
        #
        # Đúng: chấm từng (token-mở, ứng viên tốt nhất) rồi xếp **cả danh sách token-mở**.
        order_iter = order_join_candidates(
            layer, order, chosen[-1] if chosen else None,
            join_rescore=join_rescore, rng=rng, top_p_pool=top_p_pool,
        )

        for opener, members in order_iter:
            for ci in members:
                cl = layer.clauses[ci]
                if any(cl is c for c in chosen):
                    n_dup += 1
                    continue
                # §13.18 — biên bề mặt tại CHỖ GHÉP. `require_word_start` lọc theo
                # `first_content`, nhưng mệnh đề có thể bắt đầu bằng token KHÁC (dấu,
                # chữ số) đứng trước token nội dung ⇒ vẫn dính. Kiểm token ĐẦU THẬT.
                if no_glue and chosen and cl.tokens:
                    if not opener_is_word_start(state.vocab, cl.tokens[0]):
                        n_glue += 1
                        continue
                if join_seq_max is not None and prev_tail is not None and cl.tokens:
                    _p = bound_pv.pmi(prev_tail, int(cl.tokens[0]))
                    if not (math.isfinite(_p) and _p <= float(join_seq_max)):
                        n_seq += 1
                        continue
                if (pivot_max is not None and prev_left is not None
                        and cl.tokens and word_pivot_strength(
                            bound_pv, prev_left, last, int(cl.first_content)) >= pivot_max):
                    n_pivot += 1
                    continue
                if gate is not None and not gate.accept(state, cl, covered):
                    n_rejected += 1
                    n_gate += 1
                    continue
                pick = cl
                break
            if pick is not None:
                break
        if pick is None:
            dead = True
            stop = "loai_het"        # có ứng viên, nhưng bị loại sạch (cổng / trùng / dính)
            pool_at_stop = len(cand_open)
            break
        chosen.append(pick)
        covered |= content_folds(state, pick.tokens, gidf_min=gidf_min)
        pt = list(pick.tokens)
        _j = (len(pt) - 1 - pt[::-1].index(int(pick.last_content))
              if int(pick.last_content) in pt else 0)
        prev_left = int(pt[_j - 1]) if _j > 0 else None
        prev_tail = int(pt[-1]) if pt else None
        last = pick.last_content
        if (stop_on_close and len(chosen) >= int(min_clauses)
                and pt and bound_pv.ends_unit(int(pt[-1]))):
            stop = "dong_cau"
            break
    return chosen, {
        "dead": dead, "n_rejected": n_rejected, "n_clauses": len(chosen),
        # §13.36 — vì sao dừng: du_max | khong_neo | khong_ung_vien | loai_het
        "stop": stop, "n_dup": n_dup, "n_glue": n_glue, "n_gate": n_gate,
        "n_pivot": n_pivot, "n_seq": n_seq,
        "pool_at_stop": pool_at_stop,
    }
