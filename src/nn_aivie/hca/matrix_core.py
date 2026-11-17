"""
Matrix generative core — forward kiểu LLM (chỉ phép tính ma trận).

Không chọn cha / khớp suffix thủ công. Tri thức = W_know (từ chuỗi cha) → SVD Q/K
+ residual matvec; fluency = directed PPMI Q/K + spectral stack (nếu bật).

    logits = β_know · (ctxᵀ W_know) + β_qk · (q·K) + β_stack · stack(S)
    next   = argmax(logits)   [hoặc softmax sample]

Luật 0: mọi ma trận chiết từ A / ParentStore đã học.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

from nn_aivie.core.state import GraphState
from nn_aivie.hca.knowledge_seq import ensure_w_know
from nn_aivie.hca.attention_qk import ensure_directed_qk
from nn_aivie.hca.objcache import cache_get, cache_put


@dataclass
class KnowFactors:
    """W_know ≈ U√Σ · (V√Σ)ᵀ — cùng tinh thần directed Q/K."""

    wq: np.ndarray  # (N, d)
    wk: np.ndarray  # (N, d)

    def scores(self, i: int) -> np.ndarray:
        if i < 0 or i >= self.wq.shape[0]:
            return np.zeros(self.wk.shape[0], dtype=np.float64)
        return self.wk @ self.wq[i]


@dataclass
class ParentBipartite:
    """Lớp cha bipartite — khuếch đại token cùng cha khi ctx kích hoạt cha.

    A: (N, P)  token → cha   (match × √power × gidf)
    E: (P, N)  cha → token   (emit mass)
    succ: prev → [(pi, next, w), ...]
    emits: list emit_id arrays — khớp tiếp nối full-history (anti-drift)
    end_after: (N,) mass token đứng ngay trước biên câu trong emit
    """

    A: csr_matrix
    E: csr_matrix
    n_parents: int
    succ: dict  # int -> list[tuple[int, int, float]]
    emits: list  # list[np.ndarray[int32]]
    end_after: np.ndarray  # (N,) mass đứng trước biên câu
    cont_after: np.ndarray  # (N,) mass đứng trước token tiếp (không biên)


_KNOW_QK: dict[int, KnowFactors] = {}
_PARENT_BI: dict[int, ParentBipartite] = {}
_STRUCT_MASK: dict[int, np.ndarray] = {}
_WH_SLOT: dict = {}
_QA_COUNT: dict = {}


def build_parent_bipartite(state: GraphState) -> ParentBipartite:
    """Chiết A, E, succ, end_after từ ParentRecord — offline (kể cả cạnh → EOS)."""
    from nn_aivie.core.vocab import SPECIAL_EOS
    from nn_aivie.hca.context_positional import is_structural_token

    ps = state.parents
    n = len(state.vocab)
    vocab = state.vocab
    eos_id = vocab._token_to_id.get(SPECIAL_EOS)  # noqa: SLF001
    # §13.55 — biên câu HỌC ĐƯỢC (EOS-lift trên W_seq), thay danh sách dấu câu cứng.
    # Import cục bộ: `clause_layer` gọi ngược file này ở `_path_fold`.
    from nn_aivie.hca.clause_layer import ensure_boundary_signal

    _bd_lbl = ensure_boundary_signal(state)
    if ps is None or len(ps.parents) == 0:
        z = csr_matrix((n, 0), dtype=np.float64)
        return ParentBipartite(
            z,
            csr_matrix((0, n), dtype=np.float64),
            0,
            {},
            [],
            np.zeros(n),
            np.zeros(n),
        )
    struct = np.array(
        [is_structural_token(vocab, i) for i in range(n)], dtype=bool
    )
    P = len(ps.parents)
    a_r, a_c, a_v = [], [], []
    e_r, e_c, e_v = [], [], []
    succ: dict[int, list[tuple[int, int, float]]] = {}
    emits: list[np.ndarray] = []
    end_after = np.zeros(n, dtype=np.float64)
    cont_after = np.zeros(n, dtype=np.float64)
    for pi, rec in enumerate(ps.parents):
        pw = float(max(1, rec.power)) * float(getattr(rec, "priority", 1.0) or 1.0)
        root = float(np.sqrt(pw))
        seen: set[int] = set()
        for t in rec.match_ids:
            t = int(t)
            if 0 <= t < n and t not in seen and not struct[t]:
                seen.add(t)
                g = float(ps._gidf(t))  # noqa: SLF001
                a_r.append(t)
                a_c.append(pi)
                a_v.append(root * g)
        seq = list(
            rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids
        )
        emits.append(np.asarray(seq, dtype=np.int32))
        for i, t in enumerate(seq):
            t = int(t)
            if not (0 <= t < n):
                continue
            # Chiết cạnh w→EOS khi corpus kết thúc câu/emit (nhãn train, không phải rule runtime).
            at_end = i + 1 >= len(seq) or _is_emit_boundary_label(
                vocab, int(seq[i + 1]), _bd_lbl
            )
            if at_end:
                end_after[t] += pw
                if eos_id is not None:
                    succ.setdefault(t, []).append((pi, int(eos_id), pw))
            else:
                cont_after[t] += pw
        for pos, t in enumerate(seq):
            t = int(t)
            if not (0 <= t < n) or struct[t]:
                continue
            e_r.append(pi)
            e_c.append(t)
            e_v.append((pw / (1.0 + 0.15 * pos)) * float(ps._gidf(t)))  # noqa: SLF001
        for i in range(len(seq) - 1):
            a_i, b_i = int(seq[i]), int(seq[i + 1])
            if not (0 <= a_i < n and 0 <= b_i < n):
                continue
            if struct[a_i] or struct[b_i]:
                continue
            w_s = pw * (1.0 + 0.5 * (i / max(len(seq) - 1, 1))) * float(
                ps._gidf(b_i)  # noqa: SLF001
            )
            succ.setdefault(a_i, []).append((pi, b_i, w_s))
            e_r.append(pi)
            e_c.append(b_i)
            e_v.append(0.25 * pw * float(ps._gidf(b_i)))  # noqa: SLF001
    A = coo_matrix((a_v, (a_r, a_c)), shape=(n, P), dtype=np.float64).tocsr()
    E = coo_matrix((e_v, (e_r, e_c)), shape=(P, n), dtype=np.float64).tocsr()
    return ParentBipartite(
        A=A,
        E=E,
        n_parents=P,
        succ=succ,
        emits=emits,
        end_after=end_after,
        cont_after=cont_after,
    )


def ensure_parent_bipartite(state: GraphState) -> ParentBipartite:
    """Cache bipartite theo state, xác thực bằng weakref (`objcache`, §12.29/§12.30).

    `id(state)` chỉ duy nhất khi đối tượng còn SỐNG: state cũ bị thu hồi, state mới trúng cùng
    địa chỉ ⇒ nhận lại bipartite dựng cho vocab khác. Bắt được vì suite chạy 3 lần ra
    284 / 283+1 / 284 trên CÙNG cây mã (`vocab.fold(7) IndexError`) — mà crash là trường hợp
    MAY: thường id vẫn hợp lệ ⇒ truy hồi sai ÂM THẦM."""
    cached = cache_get(_PARENT_BI, state)
    if cached is None or getattr(cached, "_v", 0) < 10:
        cached = build_parent_bipartite(state)
        cached._v = 10  # type: ignore[attr-defined]
        cache_put(_PARENT_BI, state, cached)
    return cached


def succ_arrays(bi: ParentBipartite) -> dict:
    """TK_045 §12.33 — `bi.succ` (dict prev → list[(pi, nxt, w)]) đổi sang 3 mảng numpy/khoá.

    Vòng cũ duyệt PYTHON toàn bộ danh sách MỖI token, và bên trong còn `bi.A[prev, pi]` —
    tra phần tử ma trận thưa trong vòng lặp. Độ dài danh sách tỉ lệ với CỠ CORPUS: trên mini
    (5.664 cha) `succ['Ġlà']` đã 1.556 mục; v12 gấp ~76× ⇒ **~118.000 mục/token**.

    Quan hệ tĩnh theo checkpoint ⇒ dựng một lần, cache trên `bi`. Sau đó lọc bằng mặt nạ numpy
    và cộng bằng `np.add.at` — cùng kết quả, không còn vòng Python.
    """
    arr = getattr(bi, "_succ_arr", None)
    if arr is not None:
        return arr
    arr = {}
    for w0, lst in bi.succ.items():
        if not lst:
            continue
        pis = np.fromiter((int(p) for p, _n, _w in lst), dtype=np.int64, count=len(lst))
        nxs = np.fromiter((int(n) for _p, n, _w in lst), dtype=np.int64, count=len(lst))
        wts = np.fromiter((float(w) for _p, _n, w in lst), dtype=np.float64, count=len(lst))
        arr[int(w0)] = (pis, nxs, wts)
    bi._succ_arr = arr  # type: ignore[attr-defined]
    return arr


_UNI_EMIT: dict = {}
_LEFT_CNT: dict = {}


def emit_unigrams(bi: ParentBipartite, n_vocab: int):
    """(đếm token trên MỌI emit, tổng) — nền cho PMI. Một lần, cache trên `bi`."""
    hit = cache_get(_UNI_EMIT, bi)
    if hit is not None:
        return hit
    u = np.zeros(int(n_vocab), dtype=np.float64)
    tot = 0
    for e in bi.emits:
        for t in e:
            t = int(t)
            if 0 <= t < n_vocab:
                u[t] += 1.0
                tot += 1
    return cache_put(_UNI_EMIT, bi, (u, max(1, tot)))


def left_pmi(state, head: int, *, min_count: int = 3) -> dict[int, float]:
    """TK_045 §12.46 — PMI(prev, `head`) trên emit, **KHÔNG gộp hoa/thường**, cache theo head.

    §12.45③: gộp fold thì `nhà lý` (PMI 3.4) THUA `quản lý` 5.4 / `xử lý` 4.9 / `địa lý` 4.3 —
    vì fold trộn danh-từ-riêng `Lý` với từ thường `lý`. Tách theo id (cap_class khác nhau):

        'Lý' (cap=1) → Nhà 4.2 · nhà 3.8 · vua 3.6      ← toàn LOẠI TỪ
        'lý' (cap=0) → Địa 7.3 · quản 6.8 · xử 6.2      ← toàn TỪ GHÉP thường

    ⇒ "loại từ đứng trước danh từ riêng" HỌC ĐƯỢC, miễn là đừng fold. Nguyên tắc §12.45:
    fold khi so CHỨC NĂNG, đừng fold khi so DANH TÍNH.

    Tính LƯỜI theo từng head: chỉ quét emit của các cha có phát `head` (qua `emit_membership`),
    nên không phải dựng bảng bigram toàn corpus (v12 ~10M bigram = quá tốn).
    """
    hit = cache_get(_LEFT_CNT, state, int(head))
    if hit is not None:
        return hit
    bi = ensure_parent_bipartite(state)
    n = bi.A.shape[0]
    if not (0 <= int(head) < n):
        return cache_put(_LEFT_CNT, state, {}, int(head))
    from nn_aivie.hca.sent_intra import emit_membership

    memb = emit_membership(bi)
    cnt: dict[int, int] = {}
    if int(head) < memb.shape[0]:
        for pi in memb.getrow(int(head)).indices:
            seq = bi.emits[int(pi)]
            for i in range(1, len(seq)):
                if int(seq[i]) == int(head):
                    prev = int(seq[i - 1])
                    if 0 <= prev < n:
                        cnt[prev] = cnt.get(prev, 0) + 1
    u, tot = emit_unigrams(bi, n)
    uh = float(u[int(head)])
    out: dict[int, float] = {}
    if uh > 0:
        for prev, c in cnt.items():
            if c < int(min_count) or u[prev] <= 0:
                continue
            out[prev] = float(np.log((c / tot) / ((u[prev] / tot) * (uh / tot))))
    return cache_put(_LEFT_CNT, state, out, int(head))


def qa_ask_answer_counts(state) -> tuple[dict[str, int], dict[str, int]]:
    """(#cha QA có fold ở phần HỎI, ở phần ĐÁP) — nền của `ask_form`, cache trên ParentStore.

    Đếm theo FOLD chứ không theo id: `Ai` (đầu câu) và `ai` là hai token khác nhau; tách ra thì
    `Ai` không đủ `min_ask` và lọt lưới (§12.44 đo 2/8 câu hỏi bỏ sót trước khi gộp).
    """
    ps = getattr(state, "parents", None)
    if ps is None:
        return {}, {}
    hit = cache_get(_QA_COUNT, ps)
    if hit is not None:
        return hit
    vocab = state.vocab
    ask: dict[str, int] = {}
    ans: dict[str, int] = {}
    for rec in ps.parents:
        if not (getattr(rec, "is_qa", False) and getattr(rec, "answer_ids", ())):
            continue
        for f in {_path_fold(vocab, int(x)) for x in rec.match_ids}:
            ask[f] = ask.get(f, 0) + 1
        for f in {_path_fold(vocab, int(x)) for x in rec.answer_ids}:
            ans[f] = ans.get(f, 0) + 1
    return cache_put(_QA_COUNT, ps, (ask, ans))


def ask_form_of(state, tid: int) -> float | None:
    """`n_Q/(n_Q+n_A)` của fold chứa `tid`; None nếu chưa từng bị hỏi (không kết luận)."""
    ask, ans = qa_ask_answer_counts(state)
    f = _path_fold(state.vocab, int(tid))
    q = ask.get(f, 0)
    return None if q <= 0 else q / float(q + ans.get(f, 0))


def _content_veto(state, tid: int, veto_max: float) -> bool:
    """True = BÁC đề cử của luật vị trí vì token này rõ ràng là NỘI DUNG.

    `ask_form` thấp = được ĐÁP nhiều hơn được HỎI ⇒ nội dung. Đo (mini ‖ v12): `ai` 0.46 ‖ 0.53
    nhưng `Ngô` 0.40 · `Lê` 0.25 · `Lý` 0.23 ⇒ ngưỡng ~0.45 bác đúng thực thể mà tha `ai`.
    Biên mỏng và chỉ dựa hai lần đo — nhưng đây là quyền PHỦ QUYẾT: không nổ thì hành vi trở về
    đúng luật vị trí, tức trường hợp xấu nhất bằng bản M4 đã đo (§12.47).
    Chưa từng bị hỏi (`None`) ⇒ KHÔNG bác (thiếu bằng chứng thì để luật vị trí quyết).
    """
    if veto_max <= 0.0:
        return False
    af = ask_form_of(state, int(tid))
    return af is not None and af < float(veto_max)


def wh_slot_ids(state, *, min_ask: int = 5, ask_form_min: float = 0.8) -> frozenset[int]:
    """TK_045 §12.43 — token HỎI mà không bao giờ ĐƯỢC ĐÁP, học từ chính cha QA.

    §12.42 (M4) nhận đúng bệnh — WH-slot đầu độc truy hồi — nhưng bắt bằng **vị trí + độ dài bề
    mặt**, và luật đó giòn. Đo trên mini-có-QA, 4/8 câu hỏi hợp lý bị bắn nhầm:

        'Lê Lợi khởi nghĩa ở đâu?'      cấm `đâu` ✓ và `Lê`  ✗ (thực thể)
        'Ai lập ra nhà Ngô?'            cấm `Ai`  ✓ và `Ngô` ✗ (chính thực thể được hỏi)
        'Trận Bạch Đằng do ai chỉ huy?' cấm `huy` ✗, BỎ SÓT `ai` giữa câu
        'Nước Đại Cồ Việt do ai lập?'   cấm `lập` ✗, BỎ SÓT `ai`

    Tín hiệu ĐÚNG có sẵn trong đồ thị: với cha QA, `match_ids` là CÂU HỎI còn `answer_ids` là
    CÂU ĐÁP. Token chỉ nằm bên hỏi = WH-slot. Đo mini (660 cha QA), tách sạch ở ngưỡng 0.9:

        mấy .00 · gì .06 · nào .07 · đâu .22 · sao .39 · ai .73
        ‖ Ngô 1.50 · Lý 1.75 · Lê 3.00 · năm 3.35 · lập 4.14

    Bắt được `ai` GIỮA câu (thứ luật vị trí bỏ sót) và không đụng thực thể ngắn. Thuần thống kê
    đồ thị ⇒ Luật 0 sạch, không bảng từ, không đoán theo mặt chữ.

    `min_ask`: dưới ngưỡng đếm thì KHÔNG kết luận (thà bỏ sót còn hơn cấm nhầm thực thể).
    Trả `frozenset()` khi checkpoint không có cha QA — caller tự lo phương án khác.
    """
    ps = getattr(state, "parents", None)
    if ps is None:
        return frozenset()
    hit = cache_get(_WH_SLOT, ps, (round(float(ask_form_min), 3), int(min_ask)))
    if hit is not None:
        return hit
    vocab = state.vocab
    ask, ans = qa_ask_answer_counts(state)
    # `ask_form = n_Q / (n_Q + n_A)` — thang M4 đề xuất (§12.43 M4c). Nó là biến đổi đơn điệu
    # của `n_A/n_Q` nhưng chặn ở [0,1] nên ngưỡng dễ đọc, và θ=0.8 CHẶT hơn `n_A/n_Q < 0.9`
    # (tương đương θ≈0.53) ⇒ ít bắn nhầm hơn. Dùng thang của M4 để hai bên nói cùng ngôn ngữ.
    wh_folds = {
        f for f, q in ask.items()
        if f and q >= int(min_ask)
        and (q / float(q + ans.get(f, 0))) >= float(ask_form_min)
    }
    out = frozenset(
        t for t in range(len(vocab)) if _path_fold(vocab, t) in wh_folds
    ) if wh_folds else frozenset()
    return cache_put(_WH_SLOT, ps, out, (round(float(ask_form_min), 3), int(min_ask)))


def emit_match_ratio(bi: ParentBipartite) -> np.ndarray:
    """TK_045 §12.28 — tỉ lệ (số cha PHÁT token) / (số cha KHỚP bằng token), cache trên `bi`.

    Token chỉ được dùng để KHỚP mà hầu như không bao giờ được PHÁT RA = từ chỉ mang HÌNH THỨC
    câu hỏi. Đo mini-có-QA, tách rất sạch và KHÔNG cần bảng từ tiếng Việt (Luật 0):

        gì .12 · mấy .15 · nào .22 · đâu .43 · sao .59 · ai .87
        ‖ Lý 1.02 · Nhà 1.02 · dời 1.10 · năm 1.07 · Hồ 1.09 · Thăng 1.17

    Dùng để loại chúng khỏi ô neo hard-AND: hỏi `Nhà Lý dời đô về ĐÂU?` thì neo hoá ra là
    `đâu ∧ Nhà` ⇒ chỉ cha chứa CẢ HAI mới sống ⇒ "Nhà mình sắp đón Tết, dọn dẹp từ đâu?" và
    "Thành nhà Hồ ở đâu?" thắng, còn "Lý Thái Tổ dời kinh đô về Thăng Long năm 1010" bị nhân 0.
    """
    r = getattr(bi, "_emit_match_ratio", None)
    if r is not None:
        return r
    n = bi.A.shape[0]
    a_cnt = np.diff(bi.A.tocsr().indptr).astype(np.float64)
    ec = bi.E.tocsc()
    e_cnt = np.diff(ec.indptr).astype(np.float64)
    if e_cnt.shape[0] < n:
        e_cnt = np.pad(e_cnt, (0, n - e_cnt.shape[0]))
    ratio = np.where(a_cnt[:n] > 0, e_cnt[:n] / np.maximum(a_cnt[:n], 1e-9), np.inf)
    bi._emit_match_ratio = ratio  # type: ignore[attr-defined]
    return ratio


def _is_emit_boundary_label(vocab, tid: int, bound=None) -> bool:
    """Biên câu trong corpus → cạnh ·→EOS.

    §13.55 — trả nốt nợ Luật 0 (§13.54 đã trả ở `sent_intra`). Bản cũ hardcode
    `(".", "!", "?", "…", ";", "。")`. Nay dùng `BoundarySignal.ends_unit` — EOS-lift đếm
    được từ `W_seq`.

    **KHÔNG vòng vo**: `lift` dựng từ cạnh `SEQUENTIAL → EOS` mà `ingest._with_eos` đã ghi
    lúc train; hàm này chạy khi **chiết ma trận từ checkpoint ĐÃ CÓ**, nên tín hiệu có sẵn
    và không phải do chính nó tạo ra. (Khác hẳn `ingest`, nơi vòng luẩn quẩn là thật — xem
    `sent_intra._is_boundary`.)

    ⚠️ `bound=None` ⇒ **chỉ** còn ngắt khoảng trắng. Không có fallback im lặng về danh sách
    dấu câu; mọi đường gọi trong hệ đều truyền `bound`.
    """
    tok = vocab.token_of(int(tid)).lstrip("Ġ")
    if not tok:
        return True
    if tok.startswith("Ċ") or tok in ("\n", "\r"):
        return True
    return bool(bound is not None and bound.ends_unit(int(tid)))


def _path_fold(vocab, tid: int) -> str:
    try:
        return str(vocab.fold(int(tid)))
    except Exception:
        tok = vocab.token_of(int(tid)).lstrip("ĠĊ")
        return tok.casefold()


def _path_tokens_equal(a: int, b: int, vocab=None) -> bool:
    if int(a) == int(b):
        return True
    if vocab is None:
        return False
    return _path_fold(vocab, a) == _path_fold(vocab, b)


def _is_path_light(vocab, tid: int, *, gidf=None, gidf_max: float = 3.0) -> bool:
    """Hư-từ / punct = token ÍT PHÂN BIỆT — bỏ qua khi soft-align; content giữ.

    LUẬT 0 (docs/study/09 §2): đây là đường RUNTIME (sinh) ⇒ CẤM bảng-từ tiếng Việt.
    Bản trước dùng `_PATH_LIGHT_FOLDS` (40 hư-từ hardcode) = vi phạm → thay bằng **gidf**
    (IDF học được từ corpus): gidf thấp = phổ biến = hư-từ. Cùng tín hiệu hard-AND đang dùng."""
    if vocab is None:
        return False
    tid = int(tid)
    try:
        if hasattr(vocab, "is_surface") and not vocab.is_surface(tid):
            return True
    except Exception:
        pass
    tok = vocab.token_of(tid).lstrip("ĠĊ")
    if tok.isdigit():
        # TK_045 §12.13: chữ-số là CONTENT, không phải hư-từ. BPE tách số thành token 1 ký tự
        # nên luật `len<2` bên dưới nuốt sạch → chuỗi content của "…năm 1288" ≡ "…năm" ⇒ path
        # khớp lại chính chỗ vừa sinh và trỏ về `Ġ1` mãi (đo mini: lặp câu + salad chữ-số).
        return False
    if len(tok) < 2:
        return True
    if all(not ch.isalnum() for ch in tok):
        return True
    if gidf is None:
        return False  # không có thống-kê → KHÔNG đoán (thà giữ token)
    try:
        return float(gidf(tid)) < gidf_max
    except Exception:
        return False


def _path_content_index(
    ids: list[int], vocab, *, soft: bool, gidf=None, gidf_max: float = 3.0
) -> list[tuple[int, int]]:
    """(vị trí trong seq đầy đủ, token_id) — soft thì bỏ light (gidf thấp)."""
    out: list[tuple[int, int]] = []
    for i, tid in enumerate(ids):
        tid = int(tid)
        if soft and _is_path_light(vocab, tid, gidf=gidf, gidf_max=gidf_max):
            continue
        out.append((i, tid))
    return out


def _path_continuation_logits(
    bi: ParentBipartite,
    act: np.ndarray,
    history_ids: list[int],
    *,
    vocab_size: int,
    eos_id: int | None = None,
    struct_mask: np.ndarray | None = None,
    vocab=None,
    soft: bool = False,
    gidf=None,
    floor_retry: bool = True,   # TK_045 §12.13; False = ablate
    bound=None,                 # §13.55
) -> tuple[np.ndarray, int]:
    """Khớp suffix history↔emit → next; quét lại **sàn 0** nếu lượt đầu trắng tay (§12.13).

    `act_floor = 5%·act_peak` là sàn TƯƠNG ĐỐI, mà đỉnh có thể thuộc cha KHÔNG path-match nổi:
    `Trận Bạch Đằng năm` → đỉnh là "Trận Bạch Đằng của Ngô Quyền…" (emit không có `năm` ⇒ k=0),
    act 0.484 dựng sàn 0.024, giết cha đúng "…Bạch Đằng năm 1288" (act 0.0071, k=3) ⇒ path=0
    ⇒ bag soup (đo mini §12.13). Một cha VÔ DỤNG với path không được quyền dựng sàn.
    Lượt 2 chỉ chạy khi lượt 1 **không ra cha nào** ⇒ mọi ca path đang chạy giữ NGUYÊN.
    """
    out, lock_pi = _path_scan_once(
        bi, act, history_ids, vocab_size=vocab_size, eos_id=eos_id,
        struct_mask=struct_mask, vocab=vocab, soft=soft, gidf=gidf, act_floor_frac=0.05,
        bound=bound,
    )
    if floor_retry and not out.any():
        out, lock_pi = _path_scan_once(
            bi, act, history_ids, vocab_size=vocab_size, eos_id=eos_id,
            struct_mask=struct_mask, vocab=vocab, soft=soft, gidf=gidf, act_floor_frac=0.0,
            bound=bound,
            )
    return out, lock_pi


def _path_scan_once(
    bi: ParentBipartite,
    act: np.ndarray,
    history_ids: list[int],
    *,
    vocab_size: int,
    eos_id: int | None = None,
    struct_mask: np.ndarray | None = None,
    vocab=None,
    soft: bool = False,  # TK_045 §12.2c: khớp content / bỏ hư-từ; True = bật (A/B: regress số)
    gidf=None,           # Luật 0: nhận hư-từ bằng IDF học được (KHÔNG bảng từ tiếng Việt)
    act_floor_frac: float = 0.05,
    bound=None,          # §13.55 — biên câu HỌC ĐƯỢC; None ⇒ chỉ ngắt khoảng trắng
) -> tuple[np.ndarray, int]:
    """Một lượt quét cha với sàn act = `act_floor_frac`·đỉnh — chỉ cộng điểm, không if-dừng.

    soft=True (vá 3): chiếu sang chuỗi content (bỏ hư-từ `ở`/`của`/…), so fold hoa/thường.
    Không bỏ content (`năm`/`về`) — tránh khớp nhầm cha sai.
    """
    out = np.zeros(vocab_size, dtype=np.float64)
    lock_pi = -1
    if not history_ids or bi.n_parents <= 0:
        return out, lock_pi
    hist = [int(t) for t in history_ids]
    hlen = len(hist)
    hist_idx = _path_content_index(hist, vocab, soft=bool(soft), gidf=gidf)
    hc_len = len(hist_idx)
    if hc_len < 1:
        return out, lock_pi
    active = np.flatnonzero(act > 0)
    best_global_k = 0
    best_global_pi = -1
    best_global_next = -1
    best_global_score = -1.0
    act_peak = float(np.max(act)) if act.size else 0.0
    act_floor = act_floor_frac * act_peak
    for pi in active:
        a = float(act[pi])
        if a < act_floor or pi >= len(bi.emits):
            continue
        seq = [int(x) for x in bi.emits[pi]]
        slen = len(seq)
        if slen < 1:
            continue
        emit_idx = _path_content_index(seq, vocab, soft=bool(soft), gidf=gidf)
        ec_len = len(emit_idx)
        if ec_len < 1:
            continue
        best_k = 0
        best_e_full = -1  # vị trí cuối khớp trên emit ĐẦY ĐỦ
        for ec in range(ec_len):
            kmax = min(ec + 1, hc_len)
            k = 0
            while k < kmax and _path_tokens_equal(
                emit_idx[ec - k][1], hist_idx[hc_len - 1 - k][1], vocab
            ):
                k += 1
            if k > best_k:
                best_k = k
                best_e_full = int(emit_idx[ec][0])
        if best_k <= 0 or best_e_full < 0:
            continue
        # k=1 trên function word dễ lệch kỷ — cần ≥2 content (trừ hist ngắn)
        if best_k < 2 and hc_len >= 2:
            continue
        lead = 1.0 + 2.0 / (1.0 + float(max(0, best_e_full - best_k + 1)))
        score = a * float((1 + min(best_k, 24)) ** 2) * lead
        if soft:
            # phạt nhẹ vs exact — ưu tiên khớp khít khi cùng k
            score *= 0.92
        j = best_e_full + 1
        # Bỏ punct nhẹ; gặp biên câu / hết emit → next = EOS (token trong logits).
        while j < slen:
            nxt = int(seq[j])
            is_struct = struct_mask is not None and (
                0 <= nxt < len(struct_mask) and bool(struct_mask[nxt])
            )
            if is_struct:
                if vocab is not None and _is_emit_boundary_label(vocab, nxt, bound):
                    if eos_id is not None and 0 <= eos_id < vocab_size:
                        out[eos_id] += score
                        if best_k > best_global_k or (
                            best_k == best_global_k and score > best_global_score
                        ):
                            best_global_k = best_k
                            best_global_pi = int(pi)
                            best_global_next = int(eos_id)
                            best_global_score = score
                    break
                j += 1
                continue
            if 0 <= nxt < vocab_size:
                out[nxt] += score
                if best_k > best_global_k or (
                    best_k == best_global_k and score > best_global_score
                ):
                    best_global_k = best_k
                    best_global_pi = int(pi)
                    best_global_next = nxt
                    best_global_score = score
            break
        else:
            # Hết emit → EOS
            if eos_id is not None and 0 <= eos_id < vocab_size:
                out[eos_id] += score
                if best_k > best_global_k or (
                    best_k == best_global_k and score > best_global_score
                ):
                    best_global_k = best_k
                    best_global_pi = int(pi)
                    best_global_next = int(eos_id)
                    best_global_score = score
    if best_global_k >= 4 and best_global_pi >= 0 and best_global_next >= 0:
        lock_pi = best_global_pi
        peaked = np.zeros_like(out)
        peaked[best_global_next] = (
            out[best_global_next] + float(act[best_global_pi]) * 50.0
        )
        return peaked, lock_pi
    return out, lock_pi


def _order_act_multiplier(
    state: GraphState,
    hist: list[int],
    act: np.ndarray,
    ps,
    *,
    dmax: int,
    lam: float,
    beta: float,
    min_idf: float,
    accum: str,
) -> np.ndarray:
    """Nhân act[pi] *= 1 + β·(order_hit/norm) — cùng cơ chế P31 trên rank_matches.

    Chỉ đụng cha đã có act>0 (sau bag). Không rebuild A.
    """
    vocab = state.vocab
    # content sequence từ hist (bỏ frame/structural thô: dùng surface + gidf≥min nhẹ)
    q_seq: list[int] = []
    for tid in hist:
        tid = int(tid)
        if not (0 <= tid < len(vocab)):
            continue
        try:
            if not vocab.is_surface(tid):
                continue
            s = vocab.token_of(tid).lstrip("ĠĊ")
        except Exception:
            continue
        if not s or s in ".!?;:,":
            continue
        q_seq.append(tid)
    if len(q_seq) < 2:
        return act
    q_dir: dict[tuple[int, int], float] = {}
    for i in range(len(q_seq) - 1):
        for d in range(1, min(dmax, len(q_seq) - 1 - i) + 1):
            raw = min(float(ps._idf(q_seq[i])), float(ps._idf(q_seq[i + d])))  # noqa: SLF001
            if raw < min_idf:
                continue
            w = raw * (lam ** (d - 1))
            key = (vocab.fold_group_id(q_seq[i]), vocab.fold_group_id(q_seq[i + d]))
            if w > q_dir.get(key, 0.0):
                q_dir[key] = w
    if not q_dir:
        return act
    norm = sum(q_dir.values()) or 1.0
    use_sum = str(accum).lower() == "sum"
    out = act.copy()
    # chỉ cha đã kích hoạt — O(số cha dương), không quét 430k
    cand = np.nonzero(act > 0)[0]
    for pi in cand:
        pi = int(pi)
        try:
            mids = ps.parents[pi].match_ids
        except Exception:
            continue
        if not mids or len(mids) < 2:
            continue
        if use_sum:
            seen: set[tuple[int, int]] = set()
            hit = 0.0
            for i in range(len(mids) - 1):
                for d in range(1, min(dmax, len(mids) - 1 - i) + 1):
                    key = (vocab.fold_group_id(mids[i]), vocab.fold_group_id(mids[i + d]))
                    h = q_dir.get(key)
                    if h is not None and key not in seen:
                        seen.add(key)
                        hit += h
            if hit > 0:
                out[pi] *= 1.0 + beta * (hit / norm)
        else:
            hit = 0.0
            for i in range(len(mids) - 1):
                for d in range(1, min(dmax, len(mids) - 1 - i) + 1):
                    h = q_dir.get(
                        (vocab.fold_group_id(mids[i]), vocab.fold_group_id(mids[i + d]))
                    )
                    if h is not None and h > hit:
                        hit = h
            if hit > 0:
                out[pi] *= 1.0 + beta * (hit / max(norm, 1e-12))
    return out


def parent_coactivation_logits(
    state: GraphState,
    history_ids: list[int],
    *,
    vocab_size: int,
    lam: float = 0.85,  # giữ chữ ký; parent path không dùng decay/window
    ctx_len: int = 0,  # 0 / âm = full history (không cửa sổ)
    gamma: float = 1.5,
    succ_boost: float = 3.0,
    top_k_parents: int = 48,
    gidf_hi: float = 4.0,
    casefold: bool = True,  # TK_045 §12.2: hard-AND gộp hoa/thường (`Nhà`≡`Ġnhà`); False = ablate
    path_soft: bool = False,  # TK_045 §12.2c vá(3); default off — A/B regress số Lý/Bạch
    path_floor_retry: bool = True,  # TK_045 §12.13: quét lại sàn 0 khi path trắng tay
    and_form_min: float = 0.0,      # TK_045 §12.28: ngưỡng E/A cho neo AND; 0 = tắt
    wh_learned: bool = False,       # TK_045 §12.43: WH-slot học từ cha QA; False = luật vị trí
    ask_form_min: float = 0.8,      # θ của M4: n_Q/(n_Q+n_A) ≥ θ ⇒ WH-slot
    wh_veto_max: float = 0.0,       # §12.48: ask_form < ngưỡng ⇒ BÁC đề cử luật vị trí; 0 = tắt
    gidf=None,                # Luật 0: nhận hư-từ bằng IDF học được (không bảng từ)
    lock_out: dict | None = None,  # TK_045 §12.5: out-param → cha đang KHOÁ path (chép số)
    # TK_045 P33: order multiplier trên act (opt-in; default tắt qua config)
    order_act: bool = False,
    order_dmax: int = 4,
    order_lam: float = 0.5,
    order_beta: float = 1.5,
    order_min_idf: float = 2.0,
    order_accum: str = "sum",
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """(bag, succ, confidence, parent_act) — neo trên toàn bộ history, không cắt window."""
    bag = np.zeros(vocab_size, dtype=np.float64)
    succ_out = np.zeros(vocab_size, dtype=np.float64)
    conf = 0.0
    empty_act = np.zeros(0, dtype=np.float64)
    if not history_ids or state.parents is None:
        return bag, succ_out, conf, empty_act

    bi = ensure_parent_bipartite(state)
    if bi.n_parents <= 0:
        return bag, succ_out, conf, empty_act
    ps = state.parents
    # Full history — không [-ctx_len:].
    hist = list(history_ids) if int(ctx_len) <= 0 else list(history_ids[-int(ctx_len) :])
    del lam  # parent neo: presence × gidf, không positional decay (tránh soft-window)

    from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL

    frame_ids = getattr(state.vocab, "_frame_label_ids_cache", None)
    if frame_ids is None:
        raw = []
        for t in (HOI_LABEL, DAP_LABEL):
            tid = state.vocab._token_to_id.get(t)  # noqa: SLF001
            if tid is not None:
                raw.append(int(tid))
        frame_ids = frozenset(raw)
        state.vocab._frame_label_ids_cache = frame_ids  # type: ignore[attr-defined]

    # §12.42: trên v12, emit_match_ratio(WH)≈1.0 (tín hiệu mini chết). WH-slot theo
    # VỊ TRÍ + độ dài bề mặt (Luật 0, không bảng từ):
    #   · cuối câu hỏi, len≤3, không phải chữ-số → đâu/nào/gì (không Đằng/1288)
    #   · đầu sau Hỏi:, len≤2 → Ai/gì (không Nhà)
    # Cấm khỏi ctx + hard-AND ⇒ hết `đâu∧Nhà` giết cha "Lý dời đô…".
    wh_ban: set[int] = set()
    # §12.43: ưu tiên tín hiệu HỌC ĐƯỢC (hỏi-mà-không-đáp trên cha QA). Luật vị trí+độ dài của
    # §12.42 chỉ dùng khi checkpoint KHÔNG có cha QA — nó bắn nhầm thực thể ngắn (đo: 4/8).
    _wh_learned: frozenset[int] = frozenset()
    if bool(wh_learned):
        _wh_learned = wh_slot_ids(state, ask_form_min=ask_form_min)
    _dap = state.vocab._token_to_id.get(DAP_LABEL)  # noqa: SLF001
    # §12.48 — HỢP, không loại trừ. Bản §12.43 viết `if learned … elif vị-trí` nên khi tập học
    # được KHÁC RỖNG thì luật vị trí bị bỏ hẳn ⇒ `Ai` (ask_form 0.53 < θ) không ai cấm ⇒ M4 đo
    # T4 lùi 7/10 → 5/10 (§12.47b). Nay: học-được ∪ vị-trí, và thống kê có QUYỀN PHỦ QUYẾT đề cử
    # của luật vị trí khi token rõ ràng là NỘI DUNG (ask_form thấp) — đó là chỗ luật vị trí bắn
    # nhầm `Ngô` 0.40 / `Lê` 0.25 / `Lý` 0.23, còn `ai` 0.53 thì không bị bác.
    # Tính chất quan trọng: khi phủ quyết không nổ, hành vi ≡ luật vị trí (bản M4 đo tốt nhất)
    # ⇒ hợp này chỉ có thể BẰNG hoặc HƠN, không thể kém.
    if _wh_learned and _dap is not None and hist and int(hist[-1]) == int(_dap):
        wh_ban = {int(t) for t in hist if int(t) in _wh_learned}
    if _dap is not None and hist and int(hist[-1]) == int(_dap):
        content_tids: list[int] = []
        for tid in hist:
            tid = int(tid)
            if tid in frame_ids:
                continue
            try:
                surface = state.vocab.token_of(tid).lstrip("ĠĊ")
            except Exception:
                surface = ""
            if not surface or surface in ".!?;:,":
                continue
            content_tids.append(tid)
        if content_tids:
            last = content_tids[-1]
            try:
                last_s = state.vocab.token_of(last).lstrip("ĠĊ")
            except Exception:
                last_s = ""
            if (last_s and not last_s.isdigit() and len(last_s) <= 3
                    and not _content_veto(state, last, wh_veto_max)):
                wh_ban.add(last)
            first = content_tids[0]
            try:
                first_s = state.vocab.token_of(first).lstrip("ĠĊ")
            except Exception:
                first_s = ""
            if (first_s and not first_s.isdigit() and len(first_s) <= 2
                    and not _content_veto(state, first, wh_veto_max)):
                wh_ban.add(first)

    # §12.28: khi and_form_min>0, nhân thêm E/A (còn ăn trên mini; v12 ~no-op).
    _form = emit_match_ratio(bi) if and_form_min > 0.0 else None

    ctx = np.zeros(bi.A.shape[0], dtype=np.float64)
    first_pos: dict[int, int] = {}
    for i, tid in enumerate(hist):
        tid = int(tid)
        if not (0 <= tid < ctx.shape[0]):
            continue
        if tid in wh_ban:
            continue  # §12.42: WH-slot không được kéo act
        if tid not in first_pos:
            first_pos[tid] = i
        g = float(ps._gidf(tid))  # noqa: SLF001
        w = max(1.0, g)
        if _form is not None:
            w *= float(max(_form[tid], 1e-3))
        ctx[tid] += w

    act = np.asarray(bi.A.T @ ctx).ravel()
    if float(act.max()) <= 0:
        return bag, succ_out, conf, act
    act = np.maximum(act, 0.0)

    # TK_045 P33: order→act — cùng họ directed_bigram / P31 trên rank_matches, áp lên bag act.
    if order_act and order_beta > 0 and order_dmax >= 1:
        act = _order_act_multiplier(
            state, hist, act, ps,
            dmax=int(order_dmax), lam=float(order_lam), beta=float(order_beta),
            min_idf=float(order_min_idf), accum=str(order_accum),
        )

    # Độ đặc hiệu: cha match gọn / phần lớn nằm trong hist.
    hist_mask = (ctx > 0).astype(np.float64)
    match_mass = np.asarray(bi.A.sum(axis=0)).ravel() + 1e-12
    explained = np.asarray(bi.A.T @ hist_mask).ravel()
    spec = explained / match_mass
    act = act * (spec * spec)

    # Soft-AND: entity sticky gidf/(1+pos)^α; bỏ token không index trong A (frame `Hỏi:`…).
    # Fix QA salad (Hỏi: wipe act) + Nhà Lý→Tây Đô (dời∧Nhà thắng Lý).
    and_pos_alpha = 1.5
    _emit_ratio = _form  # đã tính ở trên nếu and_form_min>0
    scored: list[tuple[float, int, int]] = []
    for tid, pos0 in first_pos.items():
        if not (0 <= tid < bi.A.shape[0]):
            continue
        if tid in frame_ids or tid in wh_ban:
            continue
        row = bi.A.getrow(tid)
        if row.nnz == 0:
            continue
        g = float(ps._gidf(tid))  # noqa: SLF001
        if g < gidf_hi:
            continue
        # §12.28: token chỉ để HỎI (được khớp, không bao giờ được phát) không được làm neo AND.
        if and_form_min > 0.0 and _emit_ratio is not None and _emit_ratio[tid] < and_form_min:
            continue
        sticky = g / ((1.0 + float(pos0)) ** and_pos_alpha)
        scored.append((sticky, -pos0, tid))
    scored.sort(reverse=True)
    # Hard-AND: (token ĐUÔI hist) ∧ (entity ĐẦU — không nằm late).
    # Chỉ Nhà∧Lý → xóa "Lý Thái Tổ dời đô về Thăng Long" (không có Nhà).
    # Chỉ dời∧đô → giữ cả "Nhà Hồ … dời đô về Tây Đô".
    # dời∧Lý → đúng fact Lý + Thăng Long (đo v12: probe_h1_late_and3).
    hlen = len(hist)
    late_cut = max(0, hlen - 3)
    late_ids = {
        tid for tid, pos0 in first_pos.items()
        if int(pos0) >= late_cut and tid not in wh_ban
    }
    early_ids = {
        tid for tid, pos0 in first_pos.items()
        if int(pos0) < late_cut and tid not in wh_ban
    }
    hi: list[int] = []
    for _sticky, _negpos, tid in scored:
        if tid in late_ids:
            hi.append(int(tid))
            break
    early_by_gidf: list[tuple[int, float, int]] = []  # (pos, gidf, tid)
    for tid in early_ids:
        if not (0 <= tid < bi.A.shape[0]):
            continue
        row = bi.A.getrow(tid)
        if row.nnz == 0:
            continue
        g = float(ps._gidf(int(tid)))  # noqa: SLF001
        if g < gidf_hi:
            continue
        # §12.28: PHẢI lọc form ở ĐÂY nữa. `early_by_gidf` dựng độc lập với `scored`, và nó sắp
        # theo (-pos0) nên token MUỘN NHẤT lên đầu — đúng chỗ từ để hỏi đứng. Chỉ lọc `scored`
        # là vô ích: `đâu` vẫn vào `hi[0]` qua đường này (đo: A/B ngưỡng không đổi một ký tự).
        if and_form_min > 0.0 and _emit_ratio is not None and _emit_ratio[tid] < and_form_min:
            continue
        pos0 = int(first_pos[tid])
        early_by_gidf.append((pos0, g, int(tid)))
    # Entity sau opener trước (Lý > Nhà), rồi gidf cao.
    early_by_gidf.sort(key=lambda x: (-x[0], -x[1]))
    # Chỉ opener ở pos0 (Trận/Nhà đơn) → AND opener∧đuôi giết cha đúng không có opener
    # ("Chiến thắng Bạch Đằng năm 938" không có Trận). Lấy thêm 1 late thay opener.
    opener_only = len(early_by_gidf) == 1 and int(early_by_gidf[0][0]) == 0
    if not opener_only:
        for _pos, _g, tid in early_by_gidf:
            if tid not in hi and tid not in wh_ban:
                hi.append(tid)
                break
    else:
        for _sticky, _negpos, tid in scored:
            tid = int(tid)
            if tid in late_ids and tid not in hi and tid not in wh_ban:
                hi.append(tid)
                break
    for _sticky, _negpos, tid in scored:
        tid = int(tid)
        if tid not in hi and tid not in wh_ban:
            hi.append(tid)
        if len(hi) >= 4:
            break
    if len(hi) < 2:
        hi = [int(t) for _, _, t in scored[:4] if int(t) not in wh_ban]

    # TK_045 §12.2 (ĐO trên mini): hard-AND đòi ĐÚNG token-id → `Nhà`(hoa, đầu prompt) ≠ `Ġnhà`
    # (thường, giữa câu corpus) ⇒ cha ĐÚNG bị nhân 0. Gộp lớp hoa/thường qua case_equiv_ids.
    def _rows_union(tid: int, *, weighted: bool) -> np.ndarray:
        ids = {int(tid)}
        if casefold:
            try:
                ids |= {int(x) for x in ps.case_equiv_ids(int(tid), state.vocab)}
            except Exception:  # vocab legacy / thiếu index → giữ hành vi cũ
                pass
        hit = np.zeros(bi.n_parents, dtype=np.float64)
        for t in ids:
            if not (0 <= t < bi.A.shape[0]):
                continue
            r = bi.A.getrow(t)
            if r.nnz:
                if weighted:
                    hit[r.indices] = np.maximum(hit[r.indices], r.data)
                else:
                    hit[r.indices] = 1.0
        return hit

    if len(hi) >= 2:
        hard = np.ones(bi.n_parents, dtype=np.float64)
        for tid in hi[:2]:
            hard *= _rows_union(tid, weighted=False)
        act = act * hard
        if len(hi) > 2:
            cover = np.ones(bi.n_parents, dtype=np.float64)
            for tid in hi[2:]:
                hit = _rows_union(tid, weighted=True)
                if hit.max() > 0:
                    hit = hit / (hit.max() + 1e-12)
                cover *= 0.05 + 0.95 * hit
            act = act * cover

    act = act * (1.0 + float(gamma) * act)
    if float(act.max()) <= 0:
        return bag, succ_out, conf, act

    k = min(int(top_k_parents), bi.n_parents)
    if bi.n_parents > k > 0:
        idx = np.argpartition(-act, k - 1)[:k]
        sparse = np.zeros_like(act)
        sparse[idx] = act[idx]
        act = sparse

    mass = float(act.sum())
    if mass <= 0:
        return bag, succ_out, conf, act
    top5 = np.partition(act, -min(5, act.size))[-min(5, act.size) :]
    conf = float(np.clip(act.max() / (top5.sum() + 1e-12), 0.0, 1.0))
    act = act / mass

    raw = np.asarray(bi.E.T @ act).ravel()
    if raw.shape[0] != vocab_size:
        raw = np.pad(raw, (0, max(0, vocab_size - raw.shape[0])))[:vocab_size]
    bag = np.maximum(raw, 0.0)

    w0 = int(history_ids[-1])
    prev = int(history_ids[-2]) if len(history_ids) >= 2 else None
    act_peak = float(act.max()) if act.size else 0.0
    succ_act_floor = 0.05 * act_peak  # bỏ μ-edge từ cha act ≈0 (Lê Lợi→Thăng)
    _sa = succ_arrays(bi).get(w0)
    if _sa is not None:
        _pis, _nxs, _wts = _sa
        _a = act[_pis] if _pis.size and act.size > _pis.max() else np.zeros(_pis.size)
        _m = (_a > 0) & (_a >= succ_act_floor) & (_nxs >= 0) & (_nxs < vocab_size)
        if _m.any():
            _pi_m, _nx_m, _w_m, _a_m = _pis[_m], _nxs[_m], _wts[_m], _a[_m]
            _bonus = np.ones(_pi_m.size, dtype=np.float64)
            if prev is not None and 0 <= prev < bi.A.shape[0]:
                # `bi.A[prev, pi] > 0` cho TỪNG pi — lấy cả hàng MỘT lần rồi so tập hợp
                _bonus[np.isin(_pi_m, bi.A.getrow(prev).indices)] = 1.75
            np.add.at(succ_out, _nx_m, _a_m * _w_m * float(succ_boost) * _bonus)

    # Tiếp nối chuỗi trong cha (suffix full history) — kênh chính.
    from nn_aivie.core.vocab import SPECIAL_EOS

    eos_id = state.vocab._token_to_id.get(SPECIAL_EOS)  # noqa: SLF001
    struct = cache_get(_STRUCT_MASK, state.vocab)
    if struct is None:
        from nn_aivie.hca.context_positional import is_structural_token

        struct = np.array(
            [is_structural_token(state.vocab, i) for i in range(vocab_size)],
            dtype=bool,
        )
        cache_put(_STRUCT_MASK, state.vocab, struct)
    path, lock_pi = _path_continuation_logits(
        bi,
        act,
        history_ids,
        vocab_size=vocab_size,
        eos_id=eos_id,
        struct_mask=struct,
        vocab=state.vocab,
        soft=bool(path_soft),
        gidf=ps._gidf,  # noqa: SLF001
        floor_retry=bool(path_floor_retry),
    )
    if path.max() > 0:
        succ_out = succ_out + 4.0 * path
    if lock_out is not None:
        lock_out["lock_pi"] = int(lock_pi)
        lock_out["act"] = act  # §12.14: chép số theo cha ĐANG KÍCH khi path không khoá nổi
    if lock_pi >= 0:
        row = bi.E.getrow(lock_pi)
        locked = np.zeros(vocab_size, dtype=np.float64)
        for j, c in zip(row.indices, row.data):
            if 0 <= int(j) < vocab_size:
                locked[int(j)] = float(c)
        if locked.max() > 0:
            bag = locked
        conf = max(conf, 0.85)
    return bag, succ_out, conf, act


def build_know_qk(state: GraphState, *, dim: int = 64, oja_row: bool = True) -> KnowFactors:
    know = ensure_w_know(state, oja_row=oja_row)
    n = know.shape[0]
    k = min(dim, max(1, min(n - 1, know.nnz // 4 if know.nnz else 1)))
    if k < 1 or know.nnz == 0:
        z = np.zeros((n, dim), dtype=np.float64)
        return KnowFactors(z, z.copy())
    u, s, vt = svds(know.asfptype(), k=k, which="LM")
    order = np.argsort(-s)
    u, s, vt = u[:, order], s[order], vt[order]
    root = np.sqrt(np.maximum(s, 0.0))
    wq = u * root
    wk = vt.T * root
    if wq.shape[1] < dim:
        wq = np.pad(wq, ((0, 0), (0, dim - wq.shape[1])))
        wk = np.pad(wk, ((0, 0), (0, dim - wk.shape[1])))
    return KnowFactors(wq, wk)


def ensure_know_qk(state: GraphState, *, dim: int = 64) -> KnowFactors:
    cached = cache_get(_KNOW_QK, state)
    if cached is None:
        cached = cache_put(_KNOW_QK, state, build_know_qk(state, dim=dim))
    return cached


def _positional_context_vec(
    rows: np.ndarray,
    history_ids: list[int],
    *,
    lam: float,
    ctx_len: int = 0,
) -> np.ndarray:
    """ctx = Σ λ^{T-i} row[w_i] trên TOÀN bộ history (+ bag residual) — không cắt window."""
    d = int(rows.shape[1])
    q = np.zeros(d, dtype=np.float64)
    if not history_ids:
        return q
    from nn_aivie.hca.context_positional import positional_weights

    ctx = list(history_ids) if int(ctx_len) <= 0 else list(history_ids[-int(ctx_len) :])
    weights = positional_weights(len(ctx), lam)
    n = int(rows.shape[0])
    bag = np.zeros(d, dtype=np.float64)
    for tid, w in zip(ctx, weights):
        tid = int(tid)
        if 0 <= tid < n and w > 0:
            row = rows[tid]
            q += w * row
            bag += row  # mọi token trước đó đều hiện diện (không bị λ xoá sổ)
    q = q + 0.25 * bag
    norm = float(np.linalg.norm(q))
    if norm > 1e-12:
        q /= norm
    return q


def _matvec_know_logits(
    state: GraphState,
    history_ids: list[int],
    *,
    vocab_size: int,
    lam: float,
    ctx_len: int = 0,
) -> np.ndarray:
    """logits_know = Σ_i λ_i W_know[i,·] trên full history + W_tri[prev,cur,·]."""
    out = np.zeros(vocab_size, dtype=np.float64)
    if not history_ids:
        return out
    from nn_aivie.hca.context_positional import positional_weights

    know = ensure_w_know(state, oja_row=True)
    ctx = list(history_ids) if int(ctx_len) <= 0 else list(history_ids[-int(ctx_len) :])
    weights = positional_weights(len(ctx), lam)
    for tid, w in zip(ctx, weights):
        # Sàn nhỏ: token đầu vẫn đóng góp dù λ^{T-i} nhỏ.
        ww = max(float(w), 0.05)
        row = know.getrow(int(tid))
        for j, c in zip(row.indices, row.data):
            if 0 <= int(j) < vocab_size:
                out[int(j)] += ww * float(c)
    ps = state.parents
    if ps is not None and len(history_ids) >= 1:
        ps._ensure_seq_index()  # noqa: SLF001
        cur = int(history_ids[-1])
        prev = int(history_ids[-2]) if len(history_ids) >= 2 else None
        tri = getattr(ps, "_seq_tri", {}) or {}
        bi = getattr(ps, "_seq_bi", {}) or {}
        d2 = tri.get((prev, cur)) if prev is not None else None
        if d2 and sum(d2.values()) >= 2.0:
            for j, c in d2.items():
                if 0 <= int(j) < vocab_size:
                    out[int(j)] += 2.5 * float(c)
        else:
            for j, c in (bi.get(cur) or {}).items():
                if 0 <= int(j) < vocab_size:
                    out[int(j)] += 1.5 * float(c)
    return out


def _topk_from_logits(logits: np.ndarray, k: int) -> list[int]:
    if k <= 0 or logits.size == 0:
        return []
    k = min(k, int(logits.size))
    idx = np.argpartition(-logits, k - 1)[:k]
    return [int(i) for i in idx]


def matrix_core_logits(
    state: GraphState,
    history_ids: list[int],
    *,
    config,
    spectral_stack=None,
    lock_out: dict | None = None,  # TK_045 §12.5: nhận lock_pi (cha khoá) cho chép-số
) -> np.ndarray:
    """Logits full-vocab — ưu tiên parent-conditioned; fluency hạ khi conf cao."""
    n = len(state.vocab)
    logits = np.zeros(n, dtype=np.float64)
    if not history_ids:
        return logits

    lam = float(getattr(config, "ar_ctx_lambda", 0.85))
    # Matrix core: full history — ctx_len<=0 nghĩa là không cắt.
    ctx_len = 0
    d_know = int(getattr(config, "ar_know_qk_dim", 64))

    # --- (0) cùng-cha ---
    b_par = float(getattr(config, "ar_matrix_parent_weight", 1.0))
    b_succ = float(getattr(config, "ar_matrix_parent_succ_weight", 4.5))
    gate_w = float(getattr(config, "ar_matrix_parent_gate", 0.7))
    par_bag = np.zeros(n, dtype=np.float64)
    par_succ = np.zeros(n, dtype=np.float64)
    conf = 0.0
    parent_act = np.zeros(0, dtype=np.float64)
    if (b_par > 0 or b_succ > 0 or gate_w > 0) and state.parents is not None:
        par_bag, par_succ, conf, parent_act = parent_coactivation_logits(
            state,
            history_ids,
            vocab_size=n,
            lam=lam,
            ctx_len=0,  # full history
            gamma=float(getattr(config, "ar_matrix_parent_gamma", 1.5)),
            succ_boost=float(getattr(config, "ar_matrix_parent_succ_boost", 3.0)),
            top_k_parents=int(getattr(config, "ar_matrix_parent_topk", 48)),
            gidf_hi=float(getattr(config, "ar_matrix_parent_gidf_hi", 4.0)),
            casefold=bool(getattr(config, "ar_matrix_hard_and_casefold", True)),
            path_soft=bool(getattr(config, "ar_matrix_path_soft", False)),
            path_floor_retry=bool(getattr(config, "ar_matrix_path_floor_retry", True)),
            and_form_min=float(getattr(config, "ar_matrix_and_form_min", 0.0)),
            wh_learned=bool(getattr(config, "ar_matrix_wh_learned", False)),
            ask_form_min=float(getattr(config, "ar_matrix_qa_ask_form_min", 0.8)),
            wh_veto_max=float(getattr(config, "ar_matrix_wh_veto_max", 0.0)),
            lock_out=lock_out,
            order_act=bool(getattr(config, "ar_matrix_order_act", False)),
            order_dmax=int(getattr(config, "ar_matrix_order_dmax", 4)),
            order_lam=float(getattr(config, "ar_matrix_order_lam", 0.5)),
            order_beta=float(getattr(config, "ar_matrix_order_beta", 1.5)),
            order_min_idf=float(getattr(config, "ar_matrix_order_min_idf", 2.0)),
            order_accum=str(getattr(config, "ar_matrix_order_accum", "sum")),
        )
        if b_par > 0 and par_bag.max() > 0:
            logits = logits + b_par * (par_bag / par_bag.max())
        if b_succ > 0 and par_succ.max() > 0:
            logits = logits + b_succ * (par_succ / par_succ.max())

    # --- W_skip: giao/tích đề xuất theo khoảng cách (ý tác giả; mặc định OFF) ---
    b_wskip = float(getattr(config, "ar_matrix_w_skip_weight", 0.0))
    if b_wskip > 0 and state.parents is not None:
        from nn_aivie.hca.w_skip import soft_constraint_logits

        wskip_vec = soft_constraint_logits(
            state,
            history_ids,
            d_max=int(getattr(config, "ar_matrix_w_skip_d_max", 12)),
            min_count=int(getattr(config, "ar_matrix_w_skip_min_count", 2)),
            topk=int(getattr(config, "ar_matrix_w_skip_topk", 64)),
            ctx_max=int(getattr(config, "ar_matrix_w_skip_ctx_max", 2)),
            hard_bonus=float(getattr(config, "ar_matrix_w_skip_hard_bonus", 2.0)),
            lam=float(getattr(config, "ar_sent_lr_lam", 0.85)),
            hard_gate=bool(getattr(config, "ar_matrix_w_skip_hard_gate", False)),
            gidf_min=float(getattr(config, "ar_matrix_w_skip_gidf_min", 4.0)),
        )
        if wskip_vec.shape[0] != n:
            wskip_vec = np.pad(wskip_vec, (0, max(0, n - wskip_vec.shape[0])))[:n]
        if wskip_vec.max() > 0:
            logits = logits + b_wskip * (wskip_vec / wskip_vec.max())

    # --- TK_045 P7 / P0.2: L2R⊙emit + agenda; R2L chỉ rescore cha (không thr next) ---
    b_l2r = float(getattr(config, "ar_sent_l2r_weight", 2.0))
    b_agenda = float(getattr(config, "ar_sent_agenda_weight", 1.5))
    if (
        (b_l2r > 0 or b_agenda > 0)
        and state.parents is not None
        and bool(getattr(config, "ar_sent_intra", True))
    ):
        from nn_aivie.hca.sent_intra import sent_intra_decode_channels

        if parent_act.size == 0:
            # vẫn cần act — gọi nhẹ nếu parent channel tắt
            _, _, conf2, parent_act = parent_coactivation_logits(
                state,
                history_ids,
                vocab_size=n,
                lam=lam,
                ctx_len=0,
                gamma=float(getattr(config, "ar_matrix_parent_gamma", 1.5)),
                succ_boost=float(getattr(config, "ar_matrix_parent_succ_boost", 3.0)),
                top_k_parents=int(getattr(config, "ar_matrix_parent_topk", 48)),
                gidf_hi=float(getattr(config, "ar_matrix_parent_gidf_hi", 4.0)),
                casefold=bool(getattr(config, "ar_matrix_hard_and_casefold", True)),
                path_soft=bool(getattr(config, "ar_matrix_path_soft", False)),
                path_floor_retry=bool(getattr(config, "ar_matrix_path_floor_retry", True)),
                and_form_min=float(getattr(config, "ar_matrix_and_form_min", 0.0)),
                wh_learned=bool(getattr(config, "ar_matrix_wh_learned", False)),
                ask_form_min=float(getattr(config, "ar_matrix_qa_ask_form_min", 0.8)),
                wh_veto_max=float(getattr(config, "ar_matrix_wh_veto_max", 0.0)),
                order_act=bool(getattr(config, "ar_matrix_order_act", False)),
                order_dmax=int(getattr(config, "ar_matrix_order_dmax", 4)),
                order_lam=float(getattr(config, "ar_matrix_order_lam", 0.5)),
                order_beta=float(getattr(config, "ar_matrix_order_beta", 1.5)),
                order_min_idf=float(getattr(config, "ar_matrix_order_min_idf", 2.0)),
                order_accum=str(getattr(config, "ar_matrix_order_accum", "sum")),
            )
            conf = max(conf, conf2)
        if parent_act.size == 0:
            parent_act = np.zeros(1, dtype=np.float64)
        anchors, cont, agenda, act2 = sent_intra_decode_channels(
            state,
            history_ids,
            parent_act,
            config=config,
            vocab_size=n,
        )
        if act2.size and state.parents is not None:
            # làm mới bag nhẹ từ act đã rescore (không thay path)
            bi = ensure_parent_bipartite(state)
            if act2.size == bi.n_parents and act2.max() > 0:
                a_n = act2 / float(act2.sum())
                raw = np.asarray(bi.E.T @ a_n).ravel()
                if raw.shape[0] == n and raw.max() > 0 and b_par > 0:
                    logits = logits + 0.35 * b_par * (raw / raw.max())
        if b_l2r > 0 and cont.max() > 0:
            logits = logits + b_l2r * (cont / cont.max())
        # Agenda = câu SAU. Mặc định weight=0 (đa câu = handoff ở EOS).
        # Chỉ bơm khi hết cont VÀ w0 đã là biên câu — tránh nhảy giữa chừng.
        if b_agenda > 0 and agenda.max() > 0 and float(cont.max()) <= 1e-12:
            from nn_aivie.hca.sent_intra import _bound_of as _bound_of_mc
            from nn_aivie.hca.sent_intra import _is_boundary

            w0_b = int(history_ids[-1])
            if _is_boundary(state.vocab, w0_b, _bound_of_mc(state)):
                logits = logits + b_agenda * (agenda / agenda.max())

        # --- P6.A / P7.5: h dài (hist + anchors R2L + cont L2R) · E ---
        b_h = float(getattr(config, "ar_matrix_h_weight", 0.45))
        if b_h > 0:
            from nn_aivie.hca.sent_intra import long_context_h_logits

            h_sc = long_context_h_logits(
                state, history_ids, anchors, cont, config=config
            )
            if h_sc.max() > 0:
                logits = logits + b_h * (h_sc / h_sc.max())

    # Fluency toàn cục hạ khi đã neo được ít cha (tránh lẫn thời đại / hub).
    flu = max(0.12, 1.0 - 0.9 * conf)

    # --- EOS matvec (cạnh tranh như token thường, không if-rule) ---
    # logits[eos] += β · (ctx · end_after) / ‖ ‖  — end_after chiết từ corpus.
    b_eos = float(getattr(config, "ar_matrix_eos_weight", 1.2))
    eos_id = None
    if b_eos > 0 and state.parents is not None:
        from nn_aivie.core.vocab import SPECIAL_EOS

        eos_id = state.vocab._token_to_id.get(SPECIAL_EOS)  # noqa: SLF001
        bi = ensure_parent_bipartite(state)
        if eos_id is not None and 0 <= int(eos_id) < n and bi.end_after.size:
            ctx = np.zeros(bi.end_after.shape[0], dtype=np.float64)
            for tid in history_ids:
                tid = int(tid)
                if 0 <= tid < ctx.shape[0]:
                    ctx[tid] += 1.0
            # Markov + soft full-hist: end_after[w0] + 0.25 ctx·end_after
            w0 = int(history_ids[-1])
            s = 0.0
            if 0 <= w0 < bi.end_after.shape[0]:
                s += float(bi.end_after[w0])
            s += 0.25 * float(ctx @ bi.end_after)
            if s > 0:
                # Cạnh tranh cùng thang với kênh đã chuẩn hoá ~O(1)–O(5)
                sm = float(par_succ.max()) if par_succ.max() > 0 else 1.0
                logits[int(eos_id)] += b_eos * (s / (s + sm))

    # --- (1) W_know ---
    b_know = float(getattr(config, "ar_matrix_know_weight", 3.0)) * flu
    if b_know > 0 and state.parents is not None:
        know_lv = _matvec_know_logits(
            state, history_ids, vocab_size=n, lam=lam, ctx_len=ctx_len
        )
        kqk = ensure_know_qk(state, dim=d_know)
        qk_vec = _positional_context_vec(
            kqk.wq, history_ids, lam=lam, ctx_len=ctx_len
        )
        know_lr = np.maximum(kqk.wk @ qk_vec, 0.0)
        mix = know_lv + float(getattr(config, "ar_matrix_know_lr_weight", 0.8)) * know_lr
        if mix.max() > 0:
            logits = logits + b_know * (mix / mix.max())

    # --- (2) PPMI q·k — mặc định từ SENT_L2R (P7.2c); fallback W_seq ---
    b_qk = float(getattr(config, "ar_matrix_qk_weight", 0.6)) * flu
    if b_qk > 0:
        use_l2r_qk = bool(getattr(config, "ar_matrix_qk_from_l2r", True))
        if use_l2r_qk and bool(getattr(config, "ar_sent_intra", True)):
            from nn_aivie.hca.sent_intra import ensure_sent_qk

            qk = ensure_sent_qk(
                state,
                which="l2r",
                dim=int(getattr(config, "ar_sent_qk_dim", 64)),
                config=config,
            )
        else:
            qk = ensure_directed_qk(state)
        q = _positional_context_vec(qk.wq, history_ids, lam=lam, ctx_len=ctx_len)
        qs = np.maximum(qk.wk @ q, 0.0)
        if qs.shape[0] != n:
            qs = np.pad(qs, (0, max(0, n - qs.shape[0])))[:n]
        if qs.max() > 0:
            logits = logits + b_qk * (qs / qs.max())

    # --- (2b) R2L “FFN-like” enrich (P7.2c) — neo ngữ cảnh, không thay Attn ---
    b_r2l_ffn = float(getattr(config, "ar_matrix_r2l_ffn_weight", 0.35)) * flu
    if b_r2l_ffn > 0 and bool(getattr(config, "ar_sent_intra", True)):
        from nn_aivie.hca.sent_intra import ensure_sent_qk

        rqk = ensure_sent_qk(
            state,
            which="r2l",
            dim=int(getattr(config, "ar_sent_qk_dim", 64)),
            config=config,
        )
        rq = _positional_context_vec(rqk.wq, history_ids, lam=lam, ctx_len=ctx_len)
        rs = np.maximum(rqk.wk @ rq, 0.0)
        if rs.shape[0] != n:
            rs = np.pad(rs, (0, max(0, n - rs.shape[0])))[:n]
        if rs.max() > 0:
            logits = logits + b_r2l_ffn * (rs / rs.max())

    # --- (3) spectral stack (nhẹ khi conf cao) ---
    b_stack = float(getattr(config, "ar_matrix_stack_weight", 0.9)) * flu
    if b_stack > 0:
        from nn_aivie.hca.spectral_stack import (
            ensure_spectral_stack,
            forward_spectral_stack,
            spectral_forward_kwargs,
        )

        stack = spectral_stack
        if stack is None:
            stack = ensure_spectral_stack(
                state,
                d=int(getattr(config, "ar_spectral_stack_d", 24)),
                sidecar=getattr(state, "_spectral_stack_sidecar", None),
            )
        seed = logits.copy()
        if seed.max() <= 0:
            seed = par_succ if par_succ.max() > 0 else par_bag
        neighbors = _topk_from_logits(
            seed, int(getattr(config, "ar_spectral_s_wseq_k", 96))
        )
        wseq_scores = {j: float(seed[j]) for j in neighbors if seed[j] > 0}
        emb = getattr(state, "embedding", None)
        emb_mat = getattr(emb, "matrix", None) if emb is not None else None
        kw = spectral_forward_kwargs(config, state, history_ids)
        # Full history cho stack positional (không cắt 12/48).
        kw["pos_ctx_len"] = max(len(history_ids), 1)
        kw["wseq_neighbor_ids"] = neighbors
        kw["wseq_scores"] = wseq_scores
        a_out = forward_spectral_stack(
            stack,
            history_ids=history_ids,
            candidate_ids=None,
            entity_ids=tuple(
                t for t in history_ids if state.vocab.is_surface(t)
            ),
            emb_matrix=emb_mat,
            **kw,
        )
        if a_out.shape[0] != n:
            a_out = np.pad(a_out, (0, max(0, n - a_out.shape[0])))[:n]
        if a_out.max() > 0:
            logits = logits + b_stack * (a_out / a_out.max())

    # --- (4) readout ---
    b_ro = float(getattr(config, "ar_matrix_readout_weight", 0.05)) * flu
    if b_ro > 0:
        from nn_aivie.hca.context_positional import positional_readout_base

        base = positional_readout_base(
            state,
            history_ids,
            vocab_size=n,
            lam=lam,
            ctx_len=max(len(history_ids), 1),
        )
        base = base / (
            1.0
            + float(getattr(config, "ar_oja_compete", 0.55)) * np.maximum(base, 0.0)
        )
        if base.max() > 0:
            logits = logits + b_ro * (base / base.max())

    # Gate cùng-cha
    if gate_w > 0 and (par_bag.max() > 0 or par_succ.max() > 0):
        gate = np.zeros(n, dtype=np.float64)
        if par_bag.max() > 0:
            gate = gate + par_bag / par_bag.max()
        if par_succ.max() > 0:
            gate = gate + 2.0 * (par_succ / par_succ.max())
        if gate.max() > 0:
            gate = gate / gate.max()
            floor = max(0.05, 1.0 - gate_w)
            logits = logits * (floor + gate_w * gate)

    # Anti-loop trên TOÀN bộ history đã sinh.
    if len(history_ids) > 4:
        counts: dict[int, int] = {}
        for t in history_ids:
            t = int(t)
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            if 0 <= t < n and c > 0:
                logits[t] *= 0.45 ** min(c, 8)
        if len(history_ids) >= 8:
            tail = tuple(history_ids[-4:])
            hist = history_ids[:-4]
            for i in range(len(hist) - 3):
                if tuple(hist[i : i + 4]) == tail:
                    for t in set(tail):
                        if 0 <= int(t) < n:
                            logits[int(t)] *= 0.08
                    break

    struct = cache_get(_STRUCT_MASK, state.vocab)
    if struct is None:
        from nn_aivie.hca.context_positional import is_structural_token

        struct = np.array(
            [is_structural_token(state.vocab, i) for i in range(n)], dtype=bool
        )
        cache_put(_STRUCT_MASK, state.vocab, struct)
    logits = np.asarray(logits, dtype=np.float64).copy()
    # Giữ EOS — không xoá như punct (EOS cạnh tranh argmax như LLM).
    from nn_aivie.core.vocab import SPECIAL_EOS

    eos_keep = state.vocab._token_to_id.get(SPECIAL_EOS)  # noqa: SLF001
    eos_val = float(logits[eos_keep]) if eos_keep is not None and 0 <= int(eos_keep) < n else None
    logits[struct[:n]] = 0.0
    if eos_keep is not None and eos_val is not None and 0 <= int(eos_keep) < n:
        logits[int(eos_keep)] = eos_val
    return logits

def _is_digit_tok(vocab, tid: int) -> bool:
    """Token bề mặt toàn chữ-số — lớp KÝ TỰ của token (như `is_structural_token` dùng `isalnum`),
    KHÔNG phải bảng từ tiếng Việt ⇒ Luật 0 sạch."""
    try:
        tok = vocab.token_of(int(tid)).lstrip("ĠĊ")
    except Exception:
        return False
    return bool(tok) and tok.isdigit()


def inner_digit_ids(vocab) -> np.ndarray:
    """Mọi id có bề mặt toàn chữ-số **và KHÔNG có `Ġ`** = chữ-số nối-liền vào số đang viết.

    Dùng để ĐÓNG số sau khi chép verbatim (§12.12): `Ġ1 0 1 0` + `9` → "10109". Chỉ chặn
    nhánh nối-liền; `Ġ9` (số MỚI sau dấu cách) vẫn cho qua. Lớp KÝ TỰ, không bảng-từ ⇒ Luật 0 sạch."""
    ids = getattr(vocab, "_inner_digit_ids_cache", None)
    if ids is None:
        acc = [t for t in range(len(vocab))
               if (s := vocab.token_of(t)) and s[:1] not in ("Ġ", "Ċ") and s.isdigit()]
        ids = np.asarray(acc, dtype=np.int64)
        vocab._inner_digit_ids_cache = ids
    return ids


def _digit_runs(vocab, seq) -> list[list[int]]:
    """Các CỤM chữ-số liền nhau (≥2 token) trong emit — một cụm = một con số."""
    runs: list[list[int]] = []
    cur: list[int] = []
    for t in seq:
        if _is_digit_tok(vocab, int(t)):
            cur.append(int(t))
        else:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
    if len(cur) > 1:
        runs.append(cur)
    return runs


def unattested_premise(state, prompt_ids: list[int], *, gidf_hi: float = 4.0,
                       min_tokens: int = 2, last_k: int = 6) -> bool:
    """TK_045 §12.15 — tổ hợp token ĐẶC-HIỆU của prompt có cha nào chứng thực không?

    `Nhà Trần dời đô về Thăng Long năm` → mạng trả `1010` (fact của nhà **Lý**). Cha thắng
    (act 0.508) thiếu ĐÚNG MỘT token gidf-cao: `Trần`. Nhưng ca `Trận Bạch Đằng năm` cũng
    thiếu đúng một (`Trận`) mà lại ĐÚNG ⇒ "thiếu token" một mình KHÔNG phân biệt được.

    Thứ phân biệt: **giao các cha** của mọi token gidf≥`gidf_hi` trong prompt.
    Đo mini — 0 ở đúng 2 ca bịa, ≥1 ở cả 8 ca đúng:

        ly 2 · bach 11 · bach2 5 · nq938 2 · hung 9 · thd 12 · hoalu 1 · lehoan 1
        Nhà TRẦN dời đô 0 · Ngô Quyền … quân TỐNG 0

    Giao rỗng = trí nhớ CHƯA TỪNG thấy tổ hợp này ⇒ mọi cha chỉ khớp một PHẦN ⇒ trả lời tự tin
    từ khớp-phần = BỊA. Tín hiệu là gidf + ma trận A đã học ⇒ Luật 0 sạch.
    """
    ps = getattr(state, "parents", None)
    if ps is None or not prompt_ids:
        return False
    bi = ensure_parent_bipartite(state)
    if bi.n_parents <= 0:
        return False
    from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL
    from nn_aivie.hca.context_positional import is_structural_token

    vocab = state.vocab
    # §12.18 — `Hỏi:`/`Đáp:` KHÔNG bị `is_structural_token` bắt (đo: trả False), mà
    # `build_parent_bipartite` lại cố ý loại token cấu-trúc khỏi `A` ⇒ df=0 ⇒ MỌI prompt QA
    # làm giao rỗng NGAY token đầu ⇒ cổng nổ 100% trên qa_supervised/qa_heldout (§12.17).
    # Loại chúng như `parent_coactivation_logits` vẫn làm: so INT trên special nguyên-tử
    # (TK_041 §2.2) — không phải so chuỗi ⇒ Luật 0 sạch.
    frame_ids = getattr(vocab, "_frame_label_ids_cache", None)
    if frame_ids is None:
        raw = []
        for lab in (HOI_LABEL, DAP_LABEL):
            tid = vocab._token_to_id.get(lab)  # noqa: SLF001
            if tid is not None:
                raw.append(int(tid))
        frame_ids = frozenset(raw)
        vocab._frame_label_ids_cache = frame_ids  # type: ignore[attr-defined]
    hi = [t for t in dict.fromkeys(int(x) for x in prompt_ids)
          if 0 <= t < bi.A.shape[0]
          and t not in frame_ids
          and not is_structural_token(vocab, t)
          and float(ps._gidf(t)) >= gidf_hi]        # noqa: SLF001
    # Chỉ xét MỆNH ĐỀ ĐANG VIẾT (`last_k` token đặc-hiệu cuối). Không giới hạn thì prompt dài
    # (QA có đoạn ngữ cảnh) gom hàng chục token ⇒ giao rỗng ⇒ cổng nổ bừa: đo mini 3/4 prompt
    # HỢP LỆ dài đều nổ. Tiền đề cần kiểm là cái đang được nối tiếp, không phải cả bài.
    if int(last_k) > 0:
        hi = hi[-int(last_k):]
    if len(hi) < int(min_tokens):
        return False  # quá ít token đặc-hiệu → không đủ cơ sở kết luận
    inter: set[int] | None = None
    for t in hi:
        ids = {t}
        try:
            ids |= {int(x) for x in ps.case_equiv_ids(t, vocab)}
        except Exception:
            pass
        r: set[int] = set()
        for u in ids:
            if 0 <= u < bi.A.shape[0]:
                r |= set(bi.A.getrow(u).indices.tolist())
        inter = r if inter is None else (inter & r)
        if not inter:
            return True
    return not inter


def answer_span_from_active(state, act, history_ids: list[int], *, gidf=None,
                           min_cover: float = 0.8, min_match: int = 3,
                           max_tokens: int = 32, form_min: float = 0.9,
                           gidf_hi: float = 4.0, statement: bool = True,
                           wh_learned: bool = False) -> list[int]:
    """TK_045 §12.27 — CHÉP SPAN ĐÁP của cha QA đang kích. Path không thể làm việc này.

    `_path_continuation_logits` khớp **hậu-tố history ↔ emit**. Với cha QA thì
    `build_parent_bipartite` lấy `emit = rec.answer_ids`, còn history là **câu hỏi** ⇒ hai chuỗi
    không giao nhau ⇒ suffix-match KHÔNG BAO GIỜ fire. Đo mini-có-QA: hỏi `2 cộng 3 bằng mấy?`
    thì retrieval trúng tuyệt đối (`act=0.798`, match_ids = đúng câu hỏi, emit = `2 cộng 3 bằng
    5.`) mà `path.max=0`, `lock=-1`. Không khoá ⇒ chỉ còn `bag` yếu: trường logit phẳng
    (1.53/1.34/1.33/1.31/1.30) thay vì cú khoá dứt khoát mà câu trần được hưởng (7.55).

    Cổng an toàn = **history phủ `match_ids` của cha** (câu hỏi được hỏi phải chứa nội dung câu
    hỏi đã nhớ). Chiều phủ này quan trọng: hỏi `Nhà Lý dời đô về đâu?` thì cha act đỉnh là
    "Nhà mình sắp đón Tết, dọn dẹp từ **đâu**?" — khớp theo HÌNH DẠNG câu hỏi, `match_ids` của nó
    (`mình/sắp/đón/Tết/dọn/dẹp`) KHÔNG nằm trong history ⇒ phủ thấp ⇒ chặn, không bịa.
    Tín hiệu chỉ gồm `act` đã học + fold ⇒ Luật 0 sạch.
    """
    ps = getattr(state, "parents", None)
    if ps is None or act is None or not len(act) or float(np.max(act)) <= 0:
        return []
    bi_for_ratio = ensure_parent_bipartite(state)
    pi = int(np.argmax(act))
    if not (0 <= pi < len(ps.parents)):
        return []
    rec = ps.parents[pi]
    vocab = state.vocab
    if not getattr(rec, "is_qa", False) and not statement:
        return []
    if getattr(rec, "is_qa", False) and getattr(rec, "answer_ids", ()):
        # Cha QA: `match_ids` CHÍNH LÀ câu hỏi ⇒ đòi history phủ được nó.
        mfolds = {_path_fold(vocab, int(t)) for t in rec.match_ids}
        mfolds.discard("")
        if len(mfolds) < int(min_match):
            return []  # câu hỏi nhớ được quá ngắn → phủ 100% là vô nghĩa
        hfolds = {_path_fold(vocab, int(t)) for t in history_ids}
        if len(mfolds & hfolds) / len(mfolds) < float(min_cover):
            return []  # cha nói về câu hỏi KHÁC → đừng chép đáp của nó
        src = rec.answer_ids
    else:
        # §12.28 — cha CÂU TRẦN: chiều phủ NGƯỢC LẠI. Tri thức trả lời `Nhà Lý dời đô về đâu?`
        # nằm ở câu trần "Năm 1010, nhà Lý dời đô từ Hoa Lư về Thăng Long" — nó chứa THÊM
        # (`Hoa Lư`, `Thăng Long`) chứ không nằm gọn trong câu hỏi. Nên phải hỏi ngược: nội dung
        # CÂU HỎI có nằm trong cha không. Từ chỉ HÌNH THỨC (`đâu`/`nào`, E/A thấp) bị loại khỏi
        # tập cần phủ — nếu không thì không cha nào phủ nổi.
        if not getattr(rec, "emit_ids", ()):
            return []
        # §12.43 M4c3: `emit_match_ratio` CHẾT trên v12 (WH ≈ 0.99–1.0) ⇒ tập cần phủ vẫn dính
        # từ-để-hỏi. Ưu tiên WH-slot học từ cha QA; E/A chỉ là phương án lùi (còn ăn trên mini).
        _wh = wh_slot_ids(state) if wh_learned else frozenset()
        er = None if _wh else (emit_match_ratio(bi_for_ratio) if bi_for_ratio is not None else None)
        q: set[str] = set()
        # §12.42: đừng đòi cha phủ WH-slot (nào/đâu/gì/ai) — cha trần "…năm 938" không
        # chứa "nào". Cùng luật độ dài/vị trí với wh_ban ở parent_coactivation.
        from nn_aivie.core.tokenizer import DAP_LABEL as _DAP

        _dap = vocab._token_to_id.get(_DAP)  # noqa: SLF001
        _wh_fold: set[str] = set()
        if _dap is not None and history_ids and int(history_ids[-1]) == int(_dap):
            _surf: list[tuple[int, str]] = []
            for t in history_ids:
                t = int(t)
                try:
                    s = vocab.token_of(t).lstrip("ĠĊ")
                except Exception:
                    s = ""
                if not s or s in ".!?;:,":
                    continue
                if s in ("Hỏi:", "Đáp:"):
                    continue
                _surf.append((t, s))
            if _surf:
                _ls = _surf[-1][1]
                if _ls and not _ls.isdigit() and len(_ls) <= 3:
                    _wh_fold.add(_path_fold(vocab, _surf[-1][0]))
                _fs = _surf[0][1]
                if _fs and not _fs.isdigit() and len(_fs) <= 2:
                    _wh_fold.add(_path_fold(vocab, _surf[0][0]))
            _wh_fold.discard("")
        for t in history_ids:
            t = int(t)
            if not (0 <= t < bi_for_ratio.A.shape[0]):
                continue
            # Không cha nào KHỚP bằng token này ⇒ nó không mang bằng chứng truy hồi nào ⇒ loại
            # khỏi tập cần phủ. Bắt luôn `Hỏi:`/`Đáp:` (A rỗng, E/A=inf nên lọt bộ lọc form):
            # để chúng lại thì phủ tụt 5/7=0.71 < 0.8 và cổng chặn oan (đo).
            if bi_for_ratio.A.getrow(t).nnz == 0:
                continue
            if _wh and t in _wh:
                continue
            if er is not None and er[t] < float(form_min):
                continue
            if float(ps._gidf(t)) < float(gidf_hi):  # noqa: SLF001
                continue
            f = _path_fold(vocab, t)
            if f and f not in _wh_fold:
                q.add(f)
        if len(q) < int(min_match):
            return []
        pfolds = {_path_fold(vocab, int(t)) for t in rec.emit_ids}
        if len(q & pfolds) / len(q) < float(min_cover):
            return []
        src = rec.emit_ids
    span: list[int] = []
    for t in src:
        t = int(t)
        if not (0 <= t < len(vocab)) or not vocab.is_surface(t):
            continue
        span.append(t)
        if len(span) >= int(max_tokens):
            break
    return span


def digit_run_from_active(bi: ParentBipartite, vocab, act, history_ids: list[int], nxt: int,
                          *, gidf=None, min_content: int = 3) -> list[int]:
    """TK_045 §12.14 — CHÉP-LÙI: số của cha ĐANG KÍCH khi path KHÔNG khoá được.

    Ca `Ngô Quyền đánh bại quân Nam Hán năm` (đo mini): cha nguồn là "**Năm 938,** Ngô Quyền
    đánh bại quân Nam Hán trên sông Bạch Đằng" — số đứng TRƯỚC phần khớp, và hậu-tố history
    (`… Nam Hán năm`) KHÔNG xuất hiện trong emit ⇒ `_path_continuation_logits` bó tay (lock=-1)
    ⇒ mạng sinh từng chữ-số → `năm 8 3 8`. Nhưng `act` thì KHÔNG mù: cha đúng act=0.677,
    phủ 100% content của history, emit có ĐÚNG MỘT cụm số.

    v1 (§12.5) chép nhầm ra `18110` vì chọn cha bằng **trọng số cạnh** `succ[w0]` — mù việc cha
    có nói CÙNG SỰ VIỆC không. Ở đây cổng là **độ phủ content = 100%**: mọi content token của
    history phải có mặt trong emit. Cộng thêm: emit chỉ được có MỘT cụm số (không thì chép cái
    nào?). Toàn bộ tín hiệu là act đã học + fold + lớp ký tự ⇒ Luật 0 sạch.
    """
    if not _is_digit_tok(vocab, nxt):
        return []  # mạng KHÔNG định viết số → đừng chen số vào (đo: `ly` mất "Thăng Long năm")
    if act is None or not len(act) or float(np.max(act)) <= 0:
        return []
    pi = int(np.argmax(act))
    if not (0 <= pi < len(bi.emits)):
        return []
    hidx = _path_content_index([int(t) for t in history_ids], vocab, soft=True, gidf=gidf)
    hfolds = {_path_fold(vocab, t) for _, t in hidx}
    if len(hfolds) < int(min_content):
        return []  # history quá ngắn → phủ 100% là vô nghĩa
    seq = [int(x) for x in bi.emits[pi]]
    if not hfolds <= {_path_fold(vocab, t) for t in seq}:
        return []  # cha KHÔNG nói cùng sự việc → không chép
    runs = _digit_runs(vocab, seq)
    if len(runs) != 1:
        return []  # 0 số, hoặc nhiều số → không biết chép cái nào
    run = runs[0]
    # Số đã có trong history rồi → đừng lặp lại.
    folds = [_path_fold(vocab, t) for t in run]
    h = [_path_fold(vocab, int(t)) for t in history_ids]
    if any(h[i:i + len(folds)] == folds for i in range(max(0, len(h) - len(folds) + 1))):
        return []
    return run


def digit_run_from_locked(bi: ParentBipartite, vocab, lock_pi: int, hist_tail: int,
                          nxt: int) -> list[int]:
    """TK_045 §12.5 v2 — CỤM SỐ liền sau `hist_tail` trong emit của **cha ĐANG KHOÁ** (`lock_pi`).

    v1 SAI: lấy cha từ `bi.succ[w0]` (gom MỌI cha có cạnh w0→nxt) ⇒ chép số từ cha KHÁC → "18110"
    (đo `out/verify_copy_number.log`). v2 chỉ đọc emit của cha path ĐÃ KHOÁ.
    Vì sao: BPE tách số thành chữ-số rời + mạng không có cơ quan số (TK_041 `1288`⊥`1287`) ⇒ số
    phải CHÉP từ cha, không SINH từng chữ-số (`1010`→`1 0 1 0`)."""
    if lock_pi is None or int(lock_pi) < 0 or int(lock_pi) >= len(bi.emits):
        return []
    if not _is_digit_tok(vocab, nxt):
        return []
    seq = bi.emits[int(lock_pi)]
    n = int(seq.shape[0])
    for i in range(n - 1):
        if int(seq[i]) == int(hist_tail) and int(seq[i + 1]) == int(nxt):
            run: list[int] = []
            j = i + 1
            while j < n and _is_digit_tok(vocab, int(seq[j])):
                run.append(int(seq[j]))
                j += 1
            return run if len(run) > 1 else []
    return []
