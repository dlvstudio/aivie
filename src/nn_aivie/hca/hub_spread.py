"""Lan truyền hub ý niệm trên W_skip (không cần Lbl predefine).

Khỉ → xếp đích bằng ``w · gidf · diversity`` (hạ func ``là``/``có``…);
hub → nguồn ngược (chó|mèo…); peers → thuộc_tính chung (nghịch…).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nn_aivie.core.state import GraphState
from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.hca.analogy_bridge_gen import STOP
from nn_aivie.hca.w_skip import WSkip, clear_w_skip_cache, ensure_w_skip

# Pivot / func — count cao nhưng không phải ý niệm lớp.
_FUNC = frozenset(STOP) | frozenset({
    "là", "có", "không", "của", "và", "hoặc", "hay", "thì", "mà",
    "một", "các", "những", "này", "đó", "được", "bị", "đã", "đang", "sẽ",
    "cũng", "rất", "chỉ", "còn", "vẫn", "như", "với", "cho", "trong",
    "trên", "dưới", "để", "về", "từ", "đến", "ra", "lên", "xuống",
})


def _surf(vocab: object, tid: int) -> str:
    # §195 — CACHE dict [vocab]: gọi 380k+ lần trong vòng sinh; bề mặt cố định.
    tid = int(tid)
    cache = getattr(vocab, "_surf_str_cache", None)
    if cache is None:
        cache = {}
        try:
            vocab._surf_str_cache = cache  # noqa: SLF001
        except Exception:
            return str(vocab.token_of(tid)).replace(MARK_SPACE, "").strip()
    s = cache.get(tid)
    if s is None:
        s = str(vocab.token_of(tid)).replace(MARK_SPACE, "").strip()
        cache[tid] = s
    return s


def _gidf(state: GraphState, tid: int) -> float:
    ps = state.parents
    if ps is None:
        return 0.0
    return float(ps._gidf(int(tid)))  # noqa: SLF001


def _surf_ix(index, vocab: object, tid: int) -> str:
    """Bề mặt có xét nhãn CỤM (id ảo >= len(vocab))."""
    lab = getattr(index, "label_surf", None)
    if lab and int(tid) in lab:
        return lab[int(tid)]
    return _surf(vocab, tid)


def _gidf_ix(index, state: GraphState, tid: int) -> float:
    """gidf có xét nhãn CỤM — cụm dùng `power` của ClusterNeuron."""
    lg = getattr(index, "label_gidf", None)
    if lg and int(tid) in lg:
        return float(lg[int(tid)])
    return _gidf(state, tid)


def resolve_surface_ids(vocab: object, surface: str) -> list[int]:
    """Mọi id có surface khớp (kể cả Ġ / hoa thường)."""
    want = surface.strip()
    want_l = want.lower()
    out: list[int] = []
    for i in range(len(vocab)):
        s = _surf(vocab, i)
        if s == want or s.lower() == want_l:
            out.append(i)
    return out


@dataclass
class HubHit:
    tid: int
    surface: str
    score: float
    weight: float
    gidf: float
    diversity: int
    best_d: int
    # True = tới qua PIVOT có kiểu (A·là/như·B); False = đồng hiện W_skip.
    # Hai kênh KHÔNG cùng thang đo — xem `score_hubs_from`.
    typed: bool = False


@dataclass
class PeerHit:
    tid: int
    surface: str
    weight: float
    best_d: int
    # True = tới qua PIVOT có kiểu; False = đồng hiện. Xem `reverse_peers`.
    typed: bool = False


@dataclass
class PropHit:
    tid: int
    surface: str
    support: int
    peer_surfaces: list[str]


def _aggregate_mats(ws: WSkip, d_max: int, min_count: float) -> tuple[dict[int, dict[int, tuple[float, int]]], np.ndarray]:
    """fwd[src][dst] = (max_w, best_d); diversity[dst] = #unique sources."""
    n = int(ws.mats[1].shape[0]) if ws.d_max >= 1 else 0
    diversity = np.zeros(n, dtype=np.int32)
    seen_col: list[set[int]] = [set() for _ in range(n)]
    fwd: dict[int, dict[int, tuple[float, int]]] = {}
    for d in range(1, min(int(d_max), ws.d_max) + 1):
        m = ws.mats[d]
        indptr, indices, data = m.indptr, m.indices, m.data
        for i in range(m.shape[0]):
            start, end = int(indptr[i]), int(indptr[i + 1])
            row = fwd.setdefault(i, {})
            for k in range(start, end):
                w = float(data[k])
                if w < float(min_count):
                    continue
                j = int(indices[k])
                prev = row.get(j)
                if prev is None or w > prev[0]:
                    row[j] = (w, d)
                if i not in seen_col[j]:
                    seen_col[j].add(i)
                    diversity[j] += 1
    return fwd, diversity


def _build_rev(
    fwd: dict[int, dict[int, tuple[float, int]]],
) -> dict[int, dict[int, tuple[float, int]]]:
    rev: dict[int, dict[int, tuple[float, int]]] = {}
    for src, dsts in fwd.items():
        for dst, (w, d) in dsts.items():
            slot = rev.setdefault(dst, {})
            prev = slot.get(src)
            if prev is None or w > prev[0]:
                slot[src] = (w, d)
    return rev


@dataclass
class HubSpreadIndex:
    ws: WSkip
    fwd: dict[int, dict[int, tuple[float, int]]]
    rev: dict[int, dict[int, tuple[float, int]]]
    diversity: np.ndarray
    # src → dst qua pivot func (là/như…): (count, mid_tid)
    via_pivot: dict[int, dict[int, tuple[float, int]]]
    d_max: int
    min_count: int
    # Nhãn là CỤM ISG: id ảo >= len(vocab) → bề mặt/span/gidf tra ở đây.
    label_surf: dict[int, str] = field(default_factory=dict)
    label_span: dict[int, tuple[int, ...]] = field(default_factory=dict)
    label_gidf: dict[int, float] = field(default_factory=dict)
    # đích → số CHỦ NGỮ khác nhau trỏ tới qua pivot (nền để chuẩn hoá phiếu)
    label_indeg: dict[int, int] = field(default_factory=dict)
    # đích → tập NGUỒN (đảo của via_pivot). Dùng cho kiểm KHỨ HỒI ở
    # `score_hubs_from(rank="phieu_vong_ve")`.
    via_rev: dict[int, set[int]] = field(default_factory=dict)
    # Bể cho BƯỚC ③ — như `via_pivot` nhưng đã loại CHỦ NGỮ GIẢ do trạng ngữ
    # («Đặc biệt là ngành du lịch»). ② vẫn dùng `via_pivot` đầy đủ vì nó cần bể
    # PHIẾU rộng; ③ cần thành viên SẠCH. Xem §36.4 và §37.
    via_pivot_sach: dict[int, dict[int, tuple[float, int]]] = field(
        default_factory=dict)
    # id nhãn → từ loại (bake từ từ điển). Rỗng = không lọc.
    label_pos: dict[int, str] = field(default_factory=dict)
    # id nhãn → túi TOKEN NGỮ CẢNH của những câu đã sinh ra cạnh vào nhãn đó.
    # Thông tin `via_pivot` xưa nay vứt đi (§43).
    label_ctx: dict[int, dict[int, float]] = field(default_factory=dict)
    # nhãn → nhãn HỌ HÀNG (chia nhau thành viên): {id: độ chồng lấn}. §46.
    label_kin: dict[int, dict[int, float]] = field(default_factory=dict)
    # nhãn → tập CHỈ SỐ CHA sinh ra cạnh của nó (proxy vùng tài liệu). §48.
    label_par: dict[int, set[int]] = field(default_factory=dict)
    # THUỘC TÍNH → VẬT MANG, trích từ pivot so sánh (`như`/`giống`/`tựa`). §53.
    # Loại cạnh THỨ HAI: `via_pivot` mang quan hệ PHÂN LOẠI cho ②③, cái này mang
    # quan hệ THUỘC TÍNH cho ① — bước mà trước đây không có kênh typed nào.
    via_attr: dict[int, dict[int, tuple[float, int]]] = field(default_factory=dict)
    # BỀ MẶT → id ẢO của span WS. `resolve_surface_ids` không thấy id ảo, nên
    # từ bị BPE băm mất sạch cạnh khi tra bằng bề mặt thật (TK_052 vòng 7).
    label_by_surf: dict[str, list[int]] = field(default_factory=dict)


def save_ws_spans(path, spans: set[tuple[int, ...]]) -> None:
    """Lưu span ra sidecar — bake MỘT LẦN, mọi lần dựng index chỉ nạp."""
    import json as _j
    from pathlib import Path as _P
    _P(path).parent.mkdir(parents=True, exist_ok=True)
    _P(path).write_text(
        _j.dumps([list(x) for x in sorted(spans)]), encoding="utf-8")


def load_nhan_lop(path) -> dict[str, list[str]]:
    """§128 — bảng TỪ → NHÃN LỚP, bake từ khuôn định nghĩa của từ điển.

    §127 đo `via_pivot` (học thống kê `A là B` trên toàn văn bản) cho nhãn lớp
    đúng 4,1% trên 1 718 từ; bảng này đạt 84,9% top-1 trên 8 568 từ với cùng
    chuẩn và cùng cách chấm. Dựng bằng `scripts/dung_bang_nhan_lop.py`.

    Luật 0: từ điển là dữ liệu NGOÀI dùng ở BAKE (tiền lệ `pos_map`,`ws_spans`);
    runtime chỉ đọc bảng đã bake, không có danh sách từ viết tay.
    """
    from pathlib import Path as _P

    if not path:
        return {}
    q = _P(path)
    if not q.exists():
        return {}
    try:
        import json as _j
        d = _j.loads(q.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): [str(x) for x in v] for k, v in d.items()
            if isinstance(v, list)}


def load_pos_map(path) -> dict[str, str]:
    """Bề mặt TỪ → từ loại, đọc từ `data/dict/vi_dict_raw.jsonl` (36 764 mục).

    Luật 0: từ điển là dữ liệu NGOÀI dùng ở BAKE (cùng tiền lệ TK_046 §14 và
    `ws_spans`); runtime chỉ đọc bảng đã bake, không có danh sách từ viết tay.

    Vì sao đáng dùng ở bước ②, đo trên `classrich` (§40):

        nhãn lớp chân lý           : 80% là DANH TỪ
        ứng viên cơ chế sinh ra    : danh từ 54đ/302s · động từ 5đ/10s
                                     tính từ 0đ/48s · đại từ 0đ/19s
                                     không-có-trong-từ-điển 0đ/50s
    ⇒ loại tính/đại/phụ/kết từ bỏ 119 ứng viên SAI mà KHÔNG mất ứng viên đúng nào.

    §14 kết luận nhãn từ loại «đạt ở mức TỪ, trượt ở mức âm tiết». Nay nhãn đã là
    CỤM (`thành phố`, `triều đại`) nhờ WS-span, tức đúng vùng §14 nói là ĐẠT.
    """
    import json as _j
    from pathlib import Path as _P
    q = _P(path)
    if not q.exists():
        return {}
    out: dict[str, str] = {}
    for ln in q.read_text("utf-8").splitlines():
        try:
            d = _j.loads(ln)
        except Exception:
            continue
        w = str(d.get("word") or "").strip().lower()
        pp = str(d.get("part_of_speech") or "").strip().lower()
        if w and pp and w not in out:
            out[w] = pp
    return out


def ghep_tu_tu_dien(vocab, pos_map: dict[str, str]) -> dict[int, set[int]]:
    """`a` → tập id `b` sao cho «a b» là MỘT TỪ trong từ điển. §52.

    Bước ① là quan hệ THUỘC TÍNH nên nó buộc phải chạy trên kênh đồng-hiện
    (§51.1) — và ở đó `nghịch` hút `nghịch lý`(6,7) · `nghịch hành`(5,7) ·
    `nghịch lý Fermi`(4,8) trong khi cạnh vàng `nghịch → chó` chỉ 0,95.

    §50 thử lọc theo TỈ LỆ KỀ NHAU: hỏng, vì `nghịch ngợm` và `nghịch lý` đều
    là từ ghép nên tỉ lệ không phân biệt được. Thứ phân biệt được là RANH GIỚI
    TỪ: cả hai đều có trong từ điển, nhưng chỉ MỘT trong hai là chính thuộc
    tính đang hỏi — nên loại mọi từ ghép KHÁC của cùng âm tiết đầu.

    Luật 0: từ điển là dữ liệu ngoài dùng ở BAKE (tiền lệ §14, §40).
    """
    by_surf: dict[str, list[int]] = {}
    for i in range(len(vocab)):
        by_surf.setdefault(_surf(vocab, i).lower(), []).append(i)
    out: dict[int, set[int]] = {}
    for w in pos_map:
        am = w.split()
        if len(am) != 2:
            continue
        for ia in by_surf.get(am[0], ()):
            for ib in by_surf.get(am[1], ()):
                out.setdefault(int(ia), set()).add(int(ib))
    return out


# Từ loại KHÔNG BAO GIỜ là nhãn lớp — đo được 0 đúng / 119 sai (§40).
_DBG_ATTR: list | None = None      # gắn đo tạm cho TK_052
_DBG_CLS: list | None = None       # gắn đo nhánh PHÂN LOẠI (TK_052 vòng 6)
_DBG_GOP: list | None = None       # gắn đo vòng GỘP MẢNH (TK_052 vòng 9)

POS_KHONG_PHAI_LOP = frozenset({"tính từ", "đại từ", "phụ từ", "kết từ",
                                "trợ từ", "cảm từ"})


def _bang_be_mat(lsurf: dict[int, str]) -> dict[str, list[int]]:
    """id ảo → bề mặt  ⇒  bề mặt → [id ảo]."""
    ra: dict[str, list[int]] = {}
    for t, w in lsurf.items():
        if w:
            ra.setdefault(w, []).append(int(t))
    return ra


def ids_theo_be_mat(index: "HubSpreadIndex", vocab, w: str) -> list[int]:
    """Mọi id có bề mặt `w` — GỒM CẢ id ẢO của span WS (TK_052 vòng 7).

    `resolve_surface_ids` chỉ trả id TOKEN THẬT. Nhưng từ bị BPE băm (`sóc` →
    `s`+`óc`) được span WS gộp lại thành MỘT id ẢO, và cạnh lớp nằm trên id đó.
    Tra bằng `resolve_surface_ids` thì `via_pivot['sóc']` rỗng dù cạnh
    `sóc → thú` tồn tại — đây là lý do ca `«nhanh như ___»` không nối được
    `sóc` với `cắt` qua lớp chung.
    """
    ra = list(resolve_surface_ids(vocab, w) or [])
    for t in index.label_by_surf.get(w, ()):  # id ảo
        if int(t) not in ra:
            ra.append(int(t))
    return ra


import weakref as _weakref

# §83.2 — cache theo WEAKREF, KHÔNG theo `id()`.
#
# Bản đầu là `dict[int, bool]` khoá bằng `id(vocab)`, không giữ tham chiếu và
# không bao giờ dọn. Vocab cũ bị thu gom ⇒ CPython tái dùng địa chỉ ⇒ vocab MỚI
# nhận câu trả lời CŨ. Hệ quả đúng bằng lỗi TK_052 vòng 7–9: `_ghep_bm` rẽ nhầm
# nhánh, bề mặt span thành `"s óc"`, `label_by_surf` rỗng, cạnh `sóc → thú` mất.
#
# Đã hiện hình: `test_LOI5_...` XANH khi chạy riêng, ĐỎ khi chạy cả bộ — và đỏ
# kể cả khi stash hết thay đổi đang làm, tức thuần do thứ tự chạy + gc. Một
# cache khoá bằng `id()` trên vật thu gom được là sai bất kể có test hay không.
_DBG_LOC: list[int] = []

_CO_MARK: "_weakref.WeakKeyDictionary[object, bool]" = _weakref.WeakKeyDictionary()


def _vocab_co_mark(vocab) -> bool:
    """Từ vựng này có đánh dấu ranh giới từ bằng `MARK_SPACE` không? (cache)"""
    try:
        ra = _CO_MARK.get(vocab)
    except TypeError:            # không băm/weakref được → cứ tính lại
        ra = None
    if ra is None:
        n = min(2000, len(vocab))
        ra = any(MARK_SPACE in str(vocab.token_of(i)) for i in range(n))
        try:
            _CO_MARK[vocab] = ra
        except TypeError:
            pass
    return ra


def _ghep_bm(vocab, ts) -> str:
    """Ghép bề mặt một dãy token: chỉ chèn dấu cách ở ranh giới TỪ THẬT.

    ⚠️ Chỉ áp khi từ vựng THẬT SỰ đánh dấu ranh giới bằng `MARK_SPACE`. Fixture
    trong test dựng vocab không có dấu đó, và ghép theo luật này sẽ dính liền
    (`động`+`vật` → `độngvật`) — 13 test đỏ. Không có dấu thì giữ luật cũ.
    """
    raws = [str(vocab.token_of(int(t))) for t in ts]
    # ⚠️ GUARD PHẢI HỎI TỪ VỰNG, KHÔNG HỎI DÃY NÀY — TK_052 vòng 9.
    # Bản đầu kiểm «không token NÀO trong dãy mang MARK_SPACE» rồi ghép bằng
    # dấu cách. Nhưng từ đứng ĐẦU DÒNG thì mảnh đầu cũng không mang dấu:
    # `s`+`óc` (của `sóc`) rơi vào đúng nhánh đó và cho bề mặt `"s óc"`, nên
    # `label_by_surf["sóc"]` rỗng và cạnh `sóc → thú` không tra được.
    if not _vocab_co_mark(vocab):
        return " ".join(r.replace(MARK_SPACE, "").strip() for r in raws)
    ra = ""
    for raw in raws:
        sf = raw.replace(MARK_SPACE, "").strip()
        if not sf:
            continue
        ra += (" " if (MARK_SPACE in raw and ra) else "") + sf
    return ra.strip()


def load_ws_spans(path) -> set[tuple[int, ...]] | None:
    import json as _j
    from pathlib import Path as _P
    q = _P(path)
    if not q.exists():
        return None
    return {tuple(int(t) for t in x) for x in _j.loads(q.read_text("utf-8"))}


def ws_word_spans(
    state: GraphState,
    wsm,
    *,
    max_parents: int = 0,
    only_pivot_parents: bool = True,
    pivots: set[str] | None = None,
    progress=None,
) -> set[tuple[int, ...]]:
    """Span TOKEN của các TỪ thật, tách bằng WS-CRF (TK_034, F1 0,9844).

    Vì sao không dùng cụm ISG: ISG là n-gram theo TẦN SUẤT. Đo trên
    `mini_sent_ep_qa.pkl`: **2427/4234 = 57%** cụm không phải đơn vị ngôn ngữ
    (dấu câu/EOS 28% · mở đầu bằng hư từ 15% · kết bằng hư từ 7% · số 7%) —
    `là Nguyễn`(power 33) · `của phở` · `đá và`. Neo nhãn vào đó cho ra
    `tụ vĩ` (từ «lãnh tụ vĩ đại»), `Bốn đài`, `Các tôn`.

    WS-CRF trả đúng từ: «Hồ Chí Minh | chính | là | lãnh tụ | vĩ đại».

    Luật 0: CRF chạy **offline/build** (TK_034 §đầu), runtime chỉ đọc span.
    """
    from nn_aivie.hca.context_positional import is_structural_token

    ps = state.parents
    vocab = state.vocab
    assert ps is not None
    struct_ws = [is_structural_token(vocab, i) for i in range(len(vocab))]
    out: set[tuple[int, ...]] = set()
    recs = ps.parents if not max_parents else ps.parents[:max_parents]
    # Chỉ cha CÓ pivot mới sinh được cạnh ⇒ bỏ phần còn lại. Đây là đòn cắt
    # chi phí chính: đo 40 ms/cha ⇒ 430k cha ≈ 4,8 GIỜ nếu chạy tất.
    piv_ids: set[int] = set()
    if only_pivot_parents:
        pv = pivots or {"là", "như", "gồm", "gọi"}   # `thành` đã bỏ — xem _mine_via_pivot
        piv_ids = {i for i in range(len(vocab)) if _surf(vocab, i) in pv}
    n_skip = 0
    for i_rec, rec in enumerate(recs):
        if progress and i_rec % 20000 == 0:
            progress(f"ws_spans {i_rec}/{len(recs)} · span={len(out)} · bỏ={n_skip}")
        seq = [int(t) for t in (
            rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)]
        if len(seq) < 2:
            continue
        if piv_ids and not (piv_ids & set(seq)):
            n_skip += 1
            continue
        # token → chỉ số ÂM TIẾT: gom bề mặt, âm tiết mới bắt đầu ở MARK_SPACE
        syl: list[str] = []
        si_of: list[int] = []
        for t in seq:
            raw = str(vocab.token_of(t))
            piece = raw.replace(MARK_SPACE, " ").strip()
            # Dấu câu / special KHÔNG mang MARK_SPACE nên sẽ bị dán vào âm tiết
            # trước ⇒ ra span rác `Lôi Châu . <EOS>`, `bắc ,`, `III :`.
            # Tách riêng: chúng là âm tiết của chính chúng.
            rieng = (not piece) or not any(ch.isalnum() for ch in piece)
            if not syl or raw.startswith(MARK_SPACE) or rieng or (
                syl and syl[-1] and not any(ch.isalnum() for ch in syl[-1])
            ):
                syl.append(piece)
                si_of.append(len(syl) - 1)
            else:
                syl[-1] += piece
                si_of.append(len(syl) - 1)
        if len(syl) < 2 or len(syl) > 400:
            continue
        try:
            tags = wsm.tag(syl)
        except Exception:
            continue
        if len(tags) != len(syl):
            continue
        # âm tiết → chỉ số TỪ
        w_of: list[int] = []
        cur = -1
        for tg in tags:
            if tg != "I-W" or cur < 0:
                cur += 1
            w_of.append(cur)
        # gom token theo từ
        by_word: dict[int, list[int]] = {}
        for k, t in enumerate(seq):
            by_word.setdefault(w_of[si_of[k]], []).append(t)
        for toks_w in by_word.values():
            if len(toks_w) < 2:
                continue
            surfs = [_surf(vocab, t) for t in toks_w]
            if any((not x) or not any(ch.isalnum() for ch in x) for x in surfs):
                continue                      # span dính dấu câu/EOS → bỏ
            if all(x.isdigit() for x in surfs):
                continue                      # `8 0`, `1 9 5 2` — BPE tách số
            # MỐC KHUNG (`Hỏi:` `Đáp:` `Giải nghĩa:`) — đo được ở bake v12:
            # span `Hỏi: Tàu`. `is_structural_token` KHÔNG bắt (chúng có
            # is_surface=True) và `_FUNC` cũng không — cả hai vá đầu của tôi
            # trượt. Dấu hiệu thật là kết bằng `:`; đây là kiểm CẤU TRÚC lúc
            # bake, không phải danh sách từ.
            if any(x.endswith(":") for x in surfs):
                continue
            if any(t < len(struct_ws) and struct_ws[t] for t in toks_w):
                continue
            # TỈA hư từ dẫn đầu thay vì bỏ cả span: `tại Gia Lý Học Viện` →
            # `Gia Lý Học Viện`. Nhận bằng gidf (tại 4,47 « Gia 6,06 …), KHÔNG
            # bằng danh sách — `tại` vốn không có trong `_FUNC` (đã đo).
            gs = [_gidf(state, t) for t in toks_w]
            while len(toks_w) > 2 and gs[0] < max(gs) - 1.0:
                toks_w = toks_w[1:]
                surfs = surfs[1:]
                gs = gs[1:]
            if len(toks_w) < 2:
                continue
            out.add(tuple(toks_w))
    return out


def _mine_via_pivot(
    state: GraphState,
    *,
    pivots: set[str] | None = None,
    ws_spans: set[tuple[int, ...]] | None = None,
    bien_tu_pivot: bool = True,
    tu_choi_khi_toan_dinh_tu: bool = True,
    attr_duong_lui: bool = False,
    gop_manh_tu: bool = False,
    head_gidf_q: float | None = None,
    kenh_vi_ngu: bool = False,
    lop_trang: set[str] | None = None,
    head_lookahead: int = 3,
    use_isg: bool = True,
    use_liet_ke: bool = True,
    label_gidf_q: float = 0.20,
    label_indeg_q: float = 0.0,
    chu_ngu_ket_hop_q: float = 0.0,
    pos_map: dict[str, str] | None = None,
    head_rule: str = "max_gidf",
    head_rule_trai: str = "max_gidf",
) -> tuple[dict[int, dict[int, tuple[float, int]]], dict[int, str],
           dict[int, tuple[int, ...]], dict[int, float]]:
    """Parent content triples ``A·pivot·B`` → cạnh ý niệm A→B.

    ``B`` = **trung tâm ngữ đoạn** sau pivot, KHÔNG phải token liền kề.

    Vì sao: tiếng Việt hay viết ``X là MỘT động vật`` / ``X là NHỮNG …`` ⇒ lấy
    token liền kề cho nhãn ``một``/``những``. Đo trên v12
    (`out/hub_step3_v12_isg_pw.json`): 11/15 nhãn mạnh nhất theo bậc-vào là hư
    từ — ``một(2846) vậy(1028) thế(1021) gì(918) là(732) những(664)``. Trên mini
    không lộ vì câu mẫu là ``loài chó là động vật`` (không có ``một``).

    Chọn trung tâm bằng **gidf cao nhất** trong ``head_lookahead`` token nội dung
    kế tiếp — cùng nguyên tắc `ParentStore.pick_primary` (token phân biệt nhất =
    token định danh). KHÔNG dùng danh sách hư từ viết tay: gidf tự dìm chúng và
    tự chỉnh theo corpus (Luật 0).

    ``head_lookahead=1`` ⇒ hành vi CŨ (token liền kề), giữ để A/B.

    ``head_rule`` — chọn trung tâm BÊN PHẢI pivot. Đo 2026-08-12 trên
    `classrich`: `via_pivot` chỉ phủ **18,1%** cạnh lớp có thật trong dữ liệu
    (144/797), vì max-gidf nhặt BỔ NGỮ thay vì ĐẦU LỚP::

        «Cuba là quốc gia duy nhất…»  →  via_pivot[Cuba] = ['duy nhất']
        «Nhơn là trung tâm…»          →  via_pivot[Nhơn] = ['thủ phạm']

    Cùng bệnh mà nhánh TRÁI đã ghi (`quan trọng` SAI). Ngữ đoạn danh từ tiếng
    Việt TRUNG-TÂM-ĐẦU nên gidf một mình không đủ — bổ ngữ luôn hiếm hơn đầu lớp.

    - ``max_gidf``  — hành vi CŨ, giữ làm đối chứng.
    - ``trai_nhat`` — trái nhất. Hỏng ở ``là MỘT động vật`` (lấy ``một``).
    - ``trai_du_dac_thu`` — trái nhất trong số ứng viên có gidf ≥ phân vị thấp.
      Định từ (``một``/``các``) tự rơi khỏi ngưỡng, KHÔNG cần danh sách (Luật 0).
    - ``dau_lop`` — học từ chính corpus: ứng viên nào **hay đứng NGAY SAU pivot**
      thì là đầu lớp. ``quốc gia`` đứng sát ``là`` ở Cuba/Canada/Angkor…, còn
      ``duy nhất`` gần như luôn ở sau một danh từ khác. Thuần dữ liệu.
    """
    if head_rule not in ("max_gidf", "trai_nhat", "trai_du_dac_thu", "dau_lop"):
        raise ValueError(f"head_rule lạ: {head_rule!r}")
    if head_rule_trai not in (
            "max_gidf", "phai_nhat", "phai_du_dac_thu", "dau_ngu_doan",
            "trai_dac_thu", "phai_nhat_danh_tu"):
        raise ValueError(f"head_rule_trai lạ: {head_rule_trai!r}")
    from nn_aivie.hca.context_positional import is_structural_token
    from nn_aivie.hca.sent_intra import _content_positions

    vocab = state.vocab
    ps = state.parents
    assert ps is not None
    # ⚠️ `thành` ĐÃ BỊ BỎ khỏi pivot mặc định (2026-08-11). Nó vào đây với
    # nghĩa «trở thành», nhưng trong tiếng Việt nó cũng là ÂM TIẾT của hàng
    # loạt từ thường. Đếm corpus v12:
    #   trở thành 3 516 (ĐÚNG)  vs  thành phố 2 262 · thành công 1 995 ·
    #   hoàn thành 1 466 · thành lập 1 368 · hình thành 1 297  (SAI)
    # ⇒ ~8 400 lần dùng sai / 3 500 đúng. Mỗi câu «thành phố» bị khai thác như
    # một triple phân loại — đây là nguồn hút-điểm `phố` (10/28 dòng vòng 3).
    # HAI LOẠI PIVOT, HAI LOẠI QUAN HỆ — §53.
    #   PHÂN LOẠI  `X là Y`   → trung tâm phải = ĐẦU LỚP  (`Cuba là quốc gia`)
    #   THUỘC TÍNH `P như X`  → bên phải = DANH TỪ CỤ THỂ (`nghịch ngợm như
    #                            loài CHÓ` — bearer là `chó`, KHÔNG phải `loài`)
    # Trước §53 cả hai dùng CHUNG một luật «đầu lớp», nên `như` cho `ngợm → loài`
    # và quan hệ mà bước ① cần bị VỨT (§51.1 · §52.2). Hai loại quan hệ ⇒ hai
    # bể cạnh, hai luật trích.
    piv = pivots or {"là", "như", "gồm", "gọi"}
    PIV_TT = {"như", "giống", "tựa"}
    n = len(vocab)
    struct = [is_structural_token(vocab, i) for i in range(n)]
    look = max(1, int(head_lookahead))
    out: dict[int, dict[int, tuple[float, int]]] = {}
    out_attr: dict[int, dict[int, tuple[float, int]]] = {}
    out_sach: dict[int, dict[int, tuple[float, int]]] = {}
    label_surf: dict[int, str] = {}

    def _sfm(t) -> str:
        """Bề mặt, CHỊU ĐƯỢC id ẢO của span — TK_052 vòng 8.

        Khi `gop_manh_tu` bật, `toks` chứa id ảo của span WS (từ bị BPE băm đã
        được gộp). `_sfm(…)` sẽ IndexError trên chúng.
        """
        t = int(t)
        if t in label_surf:
            return label_surf[t]
        return _surf(vocab, t) if 0 <= t < n else ""
    label_span: dict[int, tuple[int, ...]] = {}
    label_gidf: dict[int, float] = {}
    # Bậc-vào = SỐ CHỦ NGỮ KHÁC NHAU trỏ tới đích qua pivot. Dùng để chuẩn hoá
    # phiếu bầu ở `score_hubs_from(rank="phieu_chuan")`: 2 phiếu trên đích chỉ
    # có 5 chủ ngữ mạnh hơn hẳn 2 phiếu trên đích có 500.
    label_indeg: dict[int, int] = {}
    # CHỮ KÝ NGỮ CẢNH — §43. `via_pivot` xưa nay chỉ giữ (chủ ngữ, nhãn, pivot)
    # và VỨT phần còn lại của câu. §42 kết luận: thông tin cần cho ② KHÔNG nằm
    # trong biểu diễn hiện tại (8 tín hiệu × 2 dạng ghép đều ~0) ⇒ phải THÊM
    # biểu diễn, không phải xếp lại. Đây là thứ bị vứt đi.
    label_ctx: dict[int, dict[int, float]] = {}
    # VÙNG TÀI LIỆU — §48. `via_pivot` xử lý TỪNG CÂU độc lập, chưa bao giờ dùng
    # quan hệ GIỮA các câu. Không có `doc_id`/`episode` trong ckpt, nhưng cha
    # được nạp THEO THỨ TỰ TÀI LIỆU nên chỉ số cha là proxy cho cùng-tài-liệu.
    label_par: dict[int, set[int]] = {}
    buf: list[tuple[int, int, float, int, float, bool]] = []

    # --- neo nhãn bằng CỤM ISG -------------------------------------------
    # `tình thế` cắt thành `tình` là ĐỔI NGHĨA (tình yêu/tình hình), không chỉ
    # mất thông tin. Ưu tiên khớp span cụm đã học; id ảo = n + cluster_idx,
    # bề mặt tra trong `label_surf` (cùng khuôn `analogy_bridge_gen`).
    span2ci: dict[tuple[int, ...], int] = {}
    seg = getattr(ps, "seg", None)
    # WS-CRF thắng ISG khi có: ISG 57% không phải đơn vị ngôn ngữ (xem
    # `ws_word_spans`). id ảo cấp phát tiếp sau dải cụm ISG để không đụng nhau.
    # CHỈ span WS-CRF (F1 0,9844) mới được dùng làm biên TỪ ở `_bien_pivot`.
    # `span2ci` còn chứa cụm ISG — n-gram theo tần suất, 57% KHÔNG phải đơn vị
    # ngôn ngữ (`là động` là một cụm ISG hợp lệ!) — lấy nó làm biên thì pivot
    # `là` bị nuốt và cả chuỗi chết. Đo được: 16 test đỏ.
    _ws_only: set[tuple[int, ...]] = {
        tuple(int(x) for x in sp) for sp in (ws_spans or ())} if bien_tu_pivot else set()
    if ws_spans:
        base = n + (len(seg.clusters) if seg is not None else 0)
        for i, sp in enumerate(sorted(ws_spans)):
            span2ci.setdefault(sp, -(i + 1))       # âm = span WS-CRF
        _ws_off = base
    else:
        _ws_off = n
    # Có span WS-CRF thì KHÔNG trộn thêm cụm ISG: trộn = ISG vẫn cạnh tranh và
    # thắng bằng span rác (`phố quan` cắt ngang `thành phố`/`quan trọng`).
    # Vòng 3 v12 đã so nhầm (WS+ISG) với (ISG) chứ không phải (WS) với (ISG).
    if use_isg and not ws_spans and seg is not None and getattr(seg, "clusters", None):
        for ci, c in enumerate(seg.clusters):
            try:
                sp = tuple(int(t) for t in c.token_span())
            except Exception:
                continue
            if len(sp) >= 2:
                span2ci.setdefault(sp, ci)
    max_span = max((len(s) for s in span2ci), default=0)
    n_par = max(1, len(ps.parents))

    def _label_at(seq: list[int], k: int) -> tuple[int, int]:
        """(id nhãn, số token đã tiêu) tại vị trí k — cụm dài nhất trước."""
        for L in range(min(max_span, len(seq) - k), 1, -1):
            ci = span2ci.get(tuple(seq[k : k + L]))
            if ci is not None:
                vid = (_ws_off + (-ci - 1)) if ci < 0 else (n + ci)
                # GHÉP THEO MARK_SPACE, không phải bằng dấu cách — TK_052 v7.
                # `sóc` bị BPE tách thành `s`+`óc`; span WS gộp đúng chúng lại,
                # nhưng bề mặt span được ghép bằng " " nên thành `"s óc"`, và
                # `resolve_surface_ids('sóc')` KHÔNG BAO GIỜ tìm ra cạnh lớp
                # `sóc → thú` dù nó tồn tại. Cùng lỗi đã vá cho chân lý v2 ở
                # §66.3, còn ở đây thì chưa: mọi từ bị BPE băm đều mất cạnh
                # khi tra bằng bề mặt thật.
                label_surf.setdefault(vid, _ghep_bm(vocab, seq[k : k + L]))
                label_span.setdefault(vid, tuple(seq[k : k + L]))
                # gidf cụm = gidf của THÀNH PHẦN PHÂN BIỆT NHẤT.
                # KHÔNG dùng `power`: power đếm LƯỢT xuất hiện còn `_gdf` đếm
                # SỐ CHA — khác mẫu số, cụm sẽ bị dìm giả tạo (đo được: cụm
                # `động vật` 1.18 < token `động` 1.41 dù cùng phân bố).
                # df(cụm) ≤ min df(thành phần) ⇒ gidf(cụm) ≥ max gidf(thành phần).
                label_gidf.setdefault(
                    vid, max(_gidf(state, t) for t in seq[k : k + L]))
                return vid, L
        return int(seq[k]), 1

    def _gop_manh_tu(seq: list[int]) -> list[int]:
        """Gộp MẢNH BPE thành TỪ trước khi lọc nội dung — TK_052 vòng 8.

        `_content_positions` loại token không phải `is_surface`, tức loại mảnh
        đầu của từ bị BPE băm (`sóc` → `s`+`óc` ⇒ mất `s`). Span WS `s óc` vì
        thế không khớp được, chủ ngữ trích ra là `óc`, và cạnh `sóc → thú` bị
        khoá dưới bề mặt `óc` — tra bằng `sóc` không thấy.
        Ở đây gộp trước: mỗi dãy khớp span WS mà các mảnh sau KHÔNG mở đầu từ
        (không có `MARK_SPACE`) thì thay bằng id ẢO của span.
        """
        # VÔ HIỆU khi từ vựng KHÔNG đánh dấu ranh giới từ — TK_052 vòng 10.
        # Fixture test dựng vocab không có `MARK_SPACE`; ở đó không có khái
        # niệm «mảnh của một từ», nên gộp là vô nghĩa và nó đổi hành vi ① đủ
        # để CA THỬ VÀNG mất `khỉ` (2 test đỏ khi bật mặc định ở vòng 9).
        if not _vocab_co_mark(vocab):
            return _content_positions(
                seq, vocab, content_only=True, struct_mask=struct)
        ra: list[int] = []
        i = 0
        n_ = len(seq)
        _dbg = _DBG_GOP is not None and seq[:1] and int(seq[0]) in _DBG_GOP[0]
        while i < n_:
            hit = 0
            for L in range(min(max_span, n_ - i), 1, -1):
                sp = tuple(int(x) for x in seq[i:i + L])
                if sp not in span2ci:
                    if _dbg and i == 0:
                        _DBG_GOP.append(("KHONG_CO_SPAN", L, sp))
                    continue
                # chỉ gộp khi ĐÂY LÀ MỘT TỪ: các mảnh sau không mở đầu từ
                if any(MARK_SPACE in str(vocab.token_of(int(t)))
                       for t in seq[i + 1:i + L]):
                    if _dbg and i == 0:
                        _DBG_GOP.append(("MANH_SAU_MO_DAU_TU", L, sp))
                    continue
                hit = L
                break
            if hit and _dbg and i == 0:
                _DBG_GOP.append(("GOP", hit, tuple(int(x) for x in seq[:hit])))
            if hit:
                ci = span2ci[tuple(int(x) for x in seq[i:i + hit])]
                vid = (_ws_off + (-ci - 1)) if ci < 0 else (n + ci)
                label_surf.setdefault(vid, _ghep_bm(vocab, seq[i:i + hit]))
                label_span.setdefault(vid, tuple(int(x) for x in seq[i:i + hit]))
                ra.append(int(vid))
                i += hit
                continue
            t = int(seq[i])
            i += 1
            if 0 <= t < len(struct) and struct[t]:
                continue
            if not vocab.is_surface(t):
                continue
            ra.append(t)
        return ra

    def _bien_pivot(toks: list[int], ipiv: int) -> tuple[int, int]:
        """Biên của TỪ chứa pivot — `bao gồm` · `gọi là` là MỘT từ (§62).

        `gồm` trong «bao GỒM» bị lấy làm pivot, và vế trái lấy `bao` — tức
        MẢNH CỦA CHÍNH CỤM PIVOT — làm chủ ngữ. Đó là cạnh rác nặng nhất của
        ca gãy 148 875 (§61.4). Đếm trên `classrich`: **19,2% lần pivot nằm
        lọt trong một từ** (`gồm` 48,7% · `gọi` 67,8% · `là` 12,1%).

        KHÔNG bỏ những lần đó — `gọi là`/`tức là` vẫn là pivot phân loại thật.
        Chỉ dời BIÊN: quét trái dừng trước từ, quét phải bắt đầu sau từ.
        Không có span WS thì trả về chính vị trí pivot (hành vi cũ).
        """
        if not _ws_only:
            return ipiv, ipiv
        lo, hi = ipiv, ipiv
        for a in range(max(0, ipiv - 3), ipiv + 1):
            for b in range(ipiv, min(len(toks), a + 4)):
                if b > a and tuple(int(x) for x in toks[a:b + 1]) in _ws_only:
                    lo, hi = min(lo, a), max(hi, b)
        return lo, hi

    def _ung_vien_phai(toks: list[int], ipiv: int) -> list[tuple[int, int, float, int]]:
        """Ứng viên nhãn SAU pivot: (id, số token, gidf, độ lệch từ pivot)."""
        acc: list[tuple[int, int, float, int]] = []
        _, ipiv = _bien_pivot(toks, ipiv)
        k = ipiv + 1
        while k < min(ipiv + 1 + look, len(toks)):
            cs = _sfm(int(toks[k]))
            if cs in piv or cs.lower() in piv:
                break
            cand, used = _label_at(toks, k)
            if any(
                (_sfm(t) in piv or _sfm(t).lower() in piv)
                for t in toks[k : k + used]
            ):
                k += 1
                continue
            g = label_gidf.get(cand) if cand >= n else _gidf(state, cand)
            if g is None:
                g = _gidf(state, int(toks[k]))
            acc.append((cand, used, float(g), k - (ipiv + 1)))
            # k += 1 chứ KHÔNG += used: quét tham lam sẽ ăn `một động` rồi nhảy
            # qua, làm cụm đúng `động vật` không bao giờ được thử. Ứng viên
            # chồng lấn là CỐ Ý — luật chọn quyết ai thắng.
            k += 1
        return acc

    def _emit(src: int, dst: int, w: float, mid: int, *, sach: bool = True,
              chi_sach: bool = False) -> None:
        """Phát vào HAI bể — §37.

        `via_pivot` phục vụ cả ② (bể PHIẾU, cần RỘNG) lẫn ③ (THÀNH VIÊN, cần
        SẠCH). Đo ở §36.4: lọc chủ ngữ giả làm ③ sạch hơn nhưng ②top1 tụt
        35,8% → 29,4% vì bể phiếu nghèo đi. Cạnh là TÀI NGUYÊN DÙNG CHUNG, nên
        tách làm hai thay vì đánh đổi.
        """
        if not chi_sach:
            slot = out.setdefault(int(src), {})
            prev = slot.get(int(dst))
            slot[int(dst)] = (w if prev is None else prev[0] + w, int(mid))
        if sach:
            s2 = out_sach.setdefault(int(src), {})
            p2 = s2.get(int(dst))
            s2[int(dst)] = (w if p2 is None else p2[0] + w, int(mid))

    def _pos_cua(cand: int) -> str:
        if not pos_map:
            return ""
        su = (label_surf.get(int(cand)) or _sfm(int(cand))).strip().lower()
        return pos_map.get(su, "")

    def _ung_vien_trai(
        seq_raw: list[int], ipiv: int
    ) -> list[tuple[int, int, float, int]]:
        """Ứng viên CHỦ NGỮ (bên trái pivot) — không phải token kề.

        Ngữ đoạn danh từ tiếng Việt TRUNG-TÂM-ĐẦU, và giữa chủ ngữ với pivot
        thường chen trạng từ. Lấy token kề pivot cho ra rác, footprint đo được
        trong corpus:

        - ``chính là`` 3784 · ``cũng là`` 3856 · ``vẫn là`` 2148 · ``đều là``
          2648 · ``mới là`` 899 … ⇒ **~13 700 câu** mất chủ ngữ.
          «Hồ Chí Minh **chính là** lãnh tụ» → cạnh `chính → lãnh tụ`.
        - ``chẳng hạn như A, B, C`` **1 659 câu**: lớp thật (`Các tôn giáo khác`)
          nằm NGAY TRƯỚC dấu phẩy, quét ngược dừng ở dấu phẩy là mất nó.

        ⇒ Quét ngược tới ranh giới mệnh đề, **cho phép vượt MỘT dấu phẩy**, rồi
        chọn theo **gidf cao nhất** (cụm ISG ưu tiên khi hoà). gidf tự loại số
        từ (`Bốn`) và cụm dẫn ví dụ (`chẳng hạn`) vì chúng phổ biến — không cần
        danh sách từ nào (Luật 0).
        """
        ipiv, _ = _bien_pivot(seq_raw, ipiv)   # §62: dừng TRƯỚC cả từ chứa pivot
        start, n_pham = 0, 0
        for k in range(ipiv - 1, -1, -1):
            s = _sfm(int(seq_raw[k]))
            if s in piv or s.lower() in piv:
                start = k + 1
                break
            if s == ",":
                n_pham += 1
                if n_pham > 1:          # vượt tối đa MỘT dấu phẩy
                    start = k + 1
                    break
                continue
            if not s or (not any(ch.isalnum() for ch in s)):
                start = k + 1
                break
        # Gom ứng viên theo THỨ TỰ, rồi chọn CÁI ĐẦU TIÊN đủ đặc thù.
        # Ngữ đoạn danh từ tiếng Việt TRUNG-TÂM-ĐẦU: «[thành phố] quan trọng
        # khác», «[đài truyền hình] lớn nhất». Chọn max-gidf thuần lấy nhầm
        # BỔ NGỮ (`quan trọng` hiếm hơn `thành phố` ⇒ gidf cao hơn); chọn
        # trái-nhất thuần lấy nhầm ĐỊNH TỪ/SỐ TỪ (`Các`, `Bốn`).
        # ⇒ bỏ phần dẫn đầu kém đặc thù, rồi lấy TRÁI NHẤT trong phần còn lại.
        cands: list[tuple[int, int, float]] = []
        k = start
        while k < ipiv:
            s_ = _sfm(int(seq_raw[k]))
            nxt = _sfm(int(seq_raw[k + 1])) if k + 1 < len(seq_raw) else ""
            if not s_ or not any(ch.isalnum() for ch in s_) or nxt == ":":
                k += 1
                continue
            cand, used = _label_at(seq_raw, k)
            if any(
                (_sfm(t) in piv or _sfm(t).lower() in piv)
                for t in seq_raw[k : k + used]
            ):
                k += 1
                continue
            g = label_gidf.get(cand) if cand >= n else None
            if g is None:
                g = _gidf(state, cand if cand < n else int(seq_raw[k]))
            cands.append((cand, used, g, len(cands)))
            k += 1
        return cands

    def _trung_tam_trai(seq_raw: list[int], ipiv: int) -> tuple[int | None, float]:
        cands = _ung_vien_trai(seq_raw, ipiv)
        if not cands:
            return None, -1.0
        # ⚠️ Ba luật đã thử trước đây, mỗi luật sai một kiểu (đo trên fixture):
        #   trái-nhất          → `Các` / `Bốn`        (định từ, số từ)
        #   max-gidf           → `quan trọng` SAI · `đài truyền hình` ĐÚNG
        #   trái-nhất-đủ-đặc-thù → `Các` / `Bốn`      (biên 1.0 quá rộng ở fixture)
        #
        # 2026-08-12 — ĐO ĐƯỢC max-gidf TRÔI THEO QUY MÔ, và đây là điểm gãy của
        # cả chuỗi ba bước. Đường cong nhiễu trên ca thử vàng
        # (`scripts/probe_chain3_noise_curve.py`), câu `loài chó là động vật`:
        #
        #     cha=337   via_pivot[chó] = [('vật', 0.95)]   ✓
        #     cha=895   via_pivot[chó] = [('vật', 0.95)]   ✓
        #     cha=3069  via_pivot[chó] = []                ✗  cạnh vàng MẤT
        #               via_pivot[loài] = [('vật', 1.90)]
        #
        # gidf của `loài` và `chó` đổi vị trí tương đối khi corpus lớn lên ⇒ chủ
        # ngữ chuyển từ THÀNH VIÊN (`chó`) sang LƯỢNG TỪ (`loài`). Từ đó `chó`
        # mất sạch cạnh typed, bước ② rơi xuống kênh cooc, `loài cá`/`loài thú`
        # áp đảo. Đây là lý do chuỗi ĐẠT ở 895 cha và CÂM ở 3 069.
        if head_rule_trai == "max_gidf":
            best = max(cands, key=lambda c: (c[2], c[1]))
            return best[0], best[2]
        if head_rule_trai == "phai_nhat":
            # Xếp theo vị trí KẾT THÚC (`c[3] + c[1] - 1`), không phải BẮT ĐẦU.
            # Đo 2026-08-13: xếp theo vị trí bắt đầu thì trong «Lúa mì là loại»
            # token `mì` (bắt đầu muộn hơn) THẮNG cụm `Lúa mì` ⇒ chủ ngữ bị chặt
            # còn âm tiết cuối, `via_pivot['Lúa mì']` RỖNG. Đây đúng bệnh
            # `Thụy Điển → Điển` (§30) nhưng do CHÍNH luật tôi chọn ở §29 gây ra.
            # Hai ứng viên cùng điểm kết thúc ⇒ lấy cụm DÀI hơn.
            best = max(cands, key=lambda c: (c[3] + c[1] - 1, c[1]))
            return best[0], best[2]
        if head_rule_trai == "phai_du_dac_thu":
            # Ngữ đoạn chủ ngữ hay mở đầu bằng LƯỢNG TỪ (`loài`, `con`, `các`)
            # rồi mới tới thành viên. Lấy PHẢI NHẤT trong số ứng viên đủ đặc thù:
            # lượng từ phổ biến ⇒ gidf thấp ⇒ tự rơi, không cần danh sách (Luật 0).
            du = [c for c in cands if c[2] >= det_cut_trai]
            best = max(du or cands, key=lambda c: (c[3], c[1]))
            return best[0], best[2]
        if head_rule_trai == "phai_nhat_danh_tu":
            # PHẢI NHẤT trong số DANH TỪ — §54. Tiếng Việt có CẢ HAI hướng:
            #     loài chó       → đầu là `chó`  (phải)   `loài` là lượng từ
            #     loài phổ biến  → đầu là `loài` (trái)   `phổ biến` là bổ ngữ
            # Không luật VỊ TRÍ nào đúng cả hai. Thứ phân biệt được là TỪ LOẠI:
            # `chó` danh từ, `phổ biến` tính từ ⇒ lấy phải nhất trong số DANH TỪ.
            # Sửa đúng ca `[3501] Một số loài phổ BIẾN là cáo đỏ, chó sói…` —
            # nơi khuôn liệt kê phát `chó → biến` nặng 6,65 và đè cạnh vàng.
            dn = [c for c in cands
                  if _pos_cua(c[0]) in ("danh từ", "danh từ chỉ phương hướng")]
            best = max(dn or cands, key=lambda c: (c[3] + c[1] - 1, c[1]))
            return best[0], best[2]
        if head_rule_trai == "trai_dac_thu":
            # TRÁI NHẤT ĐỦ ĐẶC THÙ — §54. Ngữ đoạn danh từ tiếng Việt
            # TRUNG-TÂM-ĐẦU: «[Một số] LOÀI [phổ biến]» ⇒ đầu là `loài`.
            # `phai_nhat` lấy phải nhất ⇒ `biến` (âm tiết cuối của `phổ biến`),
            # rồi khuôn liệt kê phát `chó → biến` nặng 6,65, đè cạnh vàng
            # `chó → động` (0,95). Đây là lỗi `phai_nhat` lần thứ TƯ (§30 chân
            # lý · §33 `Lúa mì` · §50 `nghịch ngợm` · §54 `phổ biến`).
            #
            # Định từ (`Một`, `các`) rơi khỏi ngưỡng đặc thù; cụm dài thắng khi
            # hoà (`Thụy Điển` nguyên khối thay vì `Thụy`).
            du = [c for c in cands if c[2] >= det_cut_trai]
            best = min(du or cands, key=lambda c: (c[3], -c[1]))
            return best[0], best[2]
        # dau_ngu_doan — học từ corpus: ứng viên nào hay ĐỨNG ĐẦU ngữ đoạn chủ
        # ngữ thì là lượng từ/định từ, KHÔNG phải trung tâm ⇒ chọn cái ít đứng
        # đầu nhất.
        best = min(cands, key=lambda c: (headness_trai.get(c[0], 0.0), -c[1]))
        return best[0], best[2]

    def _liet_ke(seq_raw: list[int], ipiv: int, pw: float) -> int:
        """Khuôn LIỆT KÊ: ``<LỚP> là A, B và C`` — lớp bên TRÁI, thành viên PHẢI.

        Đo trên v12 (`out/hub3_chain_v12_*`, ca C008): câu train
        «Bốn đài truyền hình lớn nhất tại Hoa Kỳ **là** NBC, CBS, ABC và Fox»
        là câu PHÂN LOẠI đúng nghĩa, nhưng khuôn ``A là B`` đọc ngược đầu ⇒ sinh
        cạnh thành-viên→thành-viên (`NBC → Fox`). Khuôn này đếm được **554 lần**
        trong corpus (paragraph_units 276 · doc_add 223 · history 55) — kho câu
        phân loại đang bị bỏ phí.

        Nhận diện: sau pivot có danh sách ≥2 mục ngăn bởi `,` / `và`. Phát cạnh
        ``thành_viên → LỚP`` cho từng mục, đúng quy ước `via_pivot[member][class]`.
        Trả số cạnh đã phát (0 = không phải liệt kê).
        """
        sep = {",", "và"}
        muc: list[int] = []
        cur: list[int] = []
        _, _hi = _bien_pivot(seq_raw, ipiv)     # §62
        j = _hi + 1
        n_sep = 0
        while j < len(seq_raw) and len(muc) < 8:
            s = _sfm(int(seq_raw[j]))
            if s in sep:
                n_sep += 1
                if cur:
                    muc.append(cur[0])
                cur = []
            elif not s or s in piv or s.lower() in piv or not any(
                ch.isalnum() for ch in s
            ):
                break                      # hết mệnh đề
            else:
                cur.append(int(seq_raw[j]))
            j += 1
        if cur:
            muc.append(cur[0])
        if n_sep < 1 or len(muc) < 3:
            return 0                       # cần ≥3 mục mới chắc là liệt kê
        # MỤC LIỆT KÊ PHẢI LÀ DANH TỪ — TK_052 vòng 6.
        # Khuôn GIỐNG–LOÀI của từ điển có ĐÚNG hình dạng liệt kê:
        #   «cắt là chim ăn thịt , nhỏ hơn diều hâu , cánh dài và nhọn , bay
        #    rất nhanh»
        # `_liet_ke` thấy 4 mục [chim · nhỏ · cánh · bay] và phát NGƯỢC CHIỀU
        # `chim → cắt`, nên `via_pivot[cắt]` không bao giờ có `chim`. Đo được:
        # via_pivot[chim] CHỨA `cắt` — cạnh tồn tại, chỉ ngược đầu.
        # Liệt kê THẬT thì các mục cùng là DANH TỪ («NBC, CBS, ABC và Fox»);
        # phần khu biệt thì là tính từ/động từ. Đòi ĐA SỐ mục là danh từ.
        if pos_map:
            _dt = sum(1 for _m in muc if "danh từ" in _pos_cua(int(_m)))
            if _dt * 2 < len(muc):
                return 0
        lop, _g = _trung_tam_trai(seq_raw, ipiv)
        if lop is None:
            return 0
        for mem in muc:
            if mem != lop:
                _emit(mem, lop, pw, int(seq_raw[ipiv]))
        return len(muc)

    # CHỦ NGỮ GIẢ do TRẠNG NGỮ. Đo được ở nhãn `ngành`: `via_rev` chứa ĐỦ 13/13
    # thành viên chân lý, nhưng `reverse_peers` trả `đặc biệt`/`hiện tại`/`biệt`
    # lên trước — chúng đến từ «ĐẶC BIỆT là ngành du lịch», «HIỆN TẠI là…», tức
    # trạng ngữ bị nhận làm chủ ngữ, và vì phổ biến nên trọng số cao hơn thành
    # viên thật. Cùng loại với `đều`/`chủ yếu`/`có lẽ` mà chân lý v2 (§31) đã
    # loại bằng TỈ LỆ ĐỨNG NGAY TRƯỚC PIVOT — bộ lọc đó chưa từng áp vào cơ chế.
    # Ngưỡng theo phân vị, kèm `dem ≥ 20` (thiếu nó thì phân vị ra 1,000 và chỉ
    # loại rác hiếm — §31 đã dính đúng bẫy này).
    chu_gia: set[int] = set()
    if float(chu_ngu_ket_hop_q) > 0.0:
        d_all: dict[int, int] = {}
        d_truoc: dict[int, int] = {}
        for rec in ps.parents:
            seq0 = list(
                rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
            for t in seq0:
                d_all[int(t)] = d_all.get(int(t), 0) + 1
            for k0 in range(1, len(seq0)):
                s0 = _sfm(int(seq0[k0]))
                if s0 in piv or s0.lower() in piv:
                    d_truoc[int(seq0[k0 - 1])] = d_truoc.get(int(seq0[k0 - 1]), 0) + 1
        rs = sorted(d_truoc[t] / max(d_all.get(t, 1), 1)
                    for t in d_truoc if d_all.get(t, 0) >= 20)
        if rs:
            c0 = rs[min(len(rs) - 1, int(len(rs) * float(chu_ngu_ket_hop_q)))]
            chu_gia = {t for t in d_truoc
                       if d_all.get(t, 0) >= 20
                       and d_truoc[t] / max(d_all.get(t, 1), 1) >= c0}

    def _pivot_positions(seq: list[int]) -> tuple[list[int], list[int]]:
        toks = _content_positions(
            seq, vocab, content_only=True, struct_mask=struct,
        )
        if len(toks) < 3:
            return [], []
        ip = [i + 1 for i in range(len(toks) - 2)
              if (_sfm(int(toks[i + 1])) in piv
                  or _sfm(int(toks[i + 1])).lower() in piv)]
        return toks, ip

    # ── LƯỢT HỌC (chỉ cho hai luật cần thống kê) ─────────────────────────
    # Không dùng danh sách hư từ: cả hai ngưỡng đều rút từ chính corpus.
    det_cut = -1.0
    det_cut_trai = -1.0
    headness: dict[int, float] = {}
    headness_trai: dict[int, float] = {}
    can_phai = head_rule in ("trai_du_dac_thu", "dau_lop")
    can_trai = head_rule_trai in ("phai_du_dac_thu", "dau_ngu_doan",
                                  "trai_dac_thu")
    if can_phai or can_trai:
        gs_p: list[float] = []
        gs_t: list[float] = []
        sat_p: dict[int, int] = {}
        tong_p: dict[int, int] = {}
        sat_t: dict[int, int] = {}
        tong_t: dict[int, int] = {}
        for rec in ps.parents:
            seq = list(
                rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
            toks, ips = _pivot_positions(seq)
            for ipiv in ips:
                if can_phai:
                    for cand, _u, g, off in _ung_vien_phai(toks, ipiv):
                        gs_p.append(g)
                        tong_p[cand] = tong_p.get(cand, 0) + 1
                        if off == 0:
                            sat_p[cand] = sat_p.get(cand, 0) + 1
                if can_trai:
                    for cand, _u, g, off in _ung_vien_trai(toks, ipiv):
                        gs_t.append(g)
                        tong_t[cand] = tong_t.get(cand, 0) + 1
                        if off == 0:
                            sat_t[cand] = sat_t.get(cand, 0) + 1
        _hq = float(label_gidf_q if head_gidf_q is None else head_gidf_q)
        if gs_p:
            gs_p.sort()
            # ⚠️ NÚM RIÊNG (§64). `label_gidf_q` điều khiển BA việc: ngưỡng
            # chọn ĐẦU PHẢI (đây), ĐẦU TRÁI, và bộ LỌC CẠNH cuối hàm. §55 vặn
            # nó về 0 để tắt bộ lọc cạnh (mua ngưỡng nhiễu 3 069 → 20 316) và
            # vô tình tắt luôn hai cái đầu ⇒ mọi ứng viên «đủ đặc thù» ⇒ luật
            # trái-nhất lấy đúng ĐỊNH TỪ: 1 142 cạnh trỏ vào `một`.
            # Đây là bệnh «tài nguyên dùng chung» lần thứ TƯ (§37 · §55 · §56).
            det_cut = gs_p[min(len(gs_p) - 1, int(len(gs_p) * _hq))]
        if gs_t:
            gs_t.sort()
            det_cut_trai = gs_t[min(len(gs_t) - 1,
                                    int(len(gs_t) * _hq))]
        for c, t in tong_p.items():
            headness[c] = sat_p.get(c, 0) / float(t)
        for c, t in tong_t.items():
            headness_trai[c] = sat_t.get(c, 0) / float(t)

    def _chon_phai(
        cands: list[tuple[int, int, float, int]],
    ) -> tuple[int | None, float]:
        if not cands:
            return None, -1.0
        if head_rule == "max_gidf":
            best_b, best_g = None, -1.0
            for cand, used, g, _off in cands:
                # cụm thắng token đơn khi hoà — nguyên khối đáng giá hơn
                if g > best_g or (used > 1 and g >= best_g):
                    best_b, best_g = cand, g
            return best_b, best_g
        if head_rule == "trai_nhat":
            b = min(cands, key=lambda c: (c[3], -c[1]))
            return b[0], b[2]
        if head_rule == "trai_du_dac_thu":
            du = [c for c in cands if c[2] >= det_cut]
            # ĐƯỜNG LUI LÀ NGUỒN RÁC LỚN NHẤT — §64. Khi KHÔNG ứng viên nào đủ
            # đặc thù, luật cũ lùi về `cands` rồi lấy trái-nhất, tức lấy đúng
            # ĐỊNH TỪ mà `det_cut` vừa loại. Đo trên `classrich` với chuẩn
            # chân lý v2 (949 cặp): nhãn rác hay gặp nhất là `một` 149 ·
            # `các` 58 · `những` 45 · `1` 34 · `2` 20 — toàn định từ/số từ.
            # Từ điển gọi `một`/`các`/`những` là «danh từ» nên POS không cứu
            # được (§40 đã kết luận POS vô dụng ở ②).
            # `tu_choi_khi_toan_dinh_tu`: thà KHÔNG phát cạnh còn hơn phát cạnh
            # vào định từ — câu này đơn giản là không phân loại được.
            if not du and tu_choi_khi_toan_dinh_tu:
                return None, -1.0
            b = min(du or cands, key=lambda c: (c[3], -c[1]))
            return b[0], b[2]
        # dau_lop — ứng viên hay đứng NGAY SAU pivot nhất
        b = max(cands, key=lambda c: (headness.get(c[0], 0.0), c[1], -c[3]))
        return b[0], b[2]

    for _pi, rec in enumerate(ps.parents):
        seq = list(rec.answer_ids if (rec.is_qa and rec.answer_ids) else rec.emit_ids)
        pw = float(max(1, rec.power)) * float(getattr(rec, "priority", 1.0) or 1.0)
        if use_liet_ke:
            for j, t in enumerate(seq):
                s = _sfm(int(t))
                if s in piv or s.lower() in piv:
                    _liet_ke(seq, j, pw)
        toks = (_gop_manh_tu(seq) if gop_manh_tu else _content_positions(
            seq, vocab, content_only=True, struct_mask=struct,
        ))
        if len(toks) < 3:
            continue
        # ── LOẠI CẠNH THỨ BA: THUỘC TÍNH THEO VỊ NGỮ (§69) ──────────────
        # §68.2 đếm được kênh nuôi ① mỏng gấp 15 lần kênh nuôi ②③: `via_attr`
        # 3 383 cạnh vs `via_pivot` 50 686, vì nó CHỈ đọc pivot so sánh
        # (`như`/`giống`/`tựa`) — chỉ 5,5% câu có pivot đó.
        # Văn xuôi thật không nói «nghịch ngợm như loài chó», nó nói «chó rất
        # trung thành». Khuôn: DANH TỪ … TÍNH TỪ, cùng mệnh đề, tính từ đứng
        # SAU (tiếng Việt hậu bổ). Từ loại lấy từ từ điển đã bake — cùng tiền
        # lệ Luật 0 với `ws_spans` (§40).
        if kenh_vi_ngu and pos_map:
            _mang: int | None = None
            _mang_k = -99
            for _k in range(len(toks)):
                _t = int(toks[_k])
                _s = _sfm(_t)
                if not _s or len(_s) <= 1:
                    continue
                if _s in piv or _s.lower() in piv:
                    _mang = None            # sang mệnh đề khác
                    continue
                _cand2, _used2 = _label_at(toks, _k)
                _sf = (label_surf.get(_cand2) or _s).strip().lower()
                _p2 = pos_map.get(_sf, "")
                if "danh từ" in _p2:
                    _mang, _mang_k = int(_cand2), _k
                elif ("tính từ" in _p2 and _mang is not None
                      and _cand2 != _mang
                      # KỀ: tiếng Việt hậu bổ, tính từ đứng NGAY SAU danh từ,
                      # cho phép chen MỘT từ mức độ («chó RẤT trung thành»).
                      # Quét cả mệnh đề thì bắt cả cặp không liên quan: bảng
                      # thuộc tính ra `đã`·`nhiều`·`hơn`·`mới` (§69).
                      and _k - _mang_k <= 2
                      # ĐỦ ĐẶC THÙ — cùng ngưỡng với luật chọn đầu (§64).
                      # `đã`/`cả`/`chính` là tính từ theo từ điển nhưng phổ
                      # biến tới mức không phân biệt được gì.
                      and _gidf(state, _cand2 if _cand2 < n else _t) >= det_cut):
                    _sl = out_attr.setdefault(int(_cand2), {})
                    _pv = _sl.get(int(_mang))
                    _sl[int(_mang)] = (pw if _pv is None else _pv[0] + pw, _t)
        for i in range(len(toks) - 2):
            m = int(toks[i + 1])
            ms = _sfm(m)
            if ms not in piv and ms.lower() not in piv:
                continue
            # CHỦ NGỮ = trung tâm ngữ đoạn, KHÔNG phải token kề pivot.
            #  cho «Hồ Chí Minh CHÍNH là lãnh tụ» →  (mất chủ
            # ngữ). Đo corpus: ~13 700 câu có trạng từ chen giữa (chính là 3784
            # · cũng là 3856 · vẫn là 2148 · đều là 2648 · mới là 899).
            a_h, _ag = _trung_tam_trai(toks, i + 1)
            a = int(a_h) if a_h is not None else int(toks[i])
            # trung tâm ngữ đoạn = gidf cao nhất trong cửa sổ sau pivot,
            # nhưng không vượt qua một pivot khác (ranh giới mệnh đề).
            # Cụm ISG khớp được thì ưu tiên (nguyên khối, không cắt đôi).
            _uv = _ung_vien_phai(toks, i + 1)
            if ms in PIV_TT or ms.lower() in PIV_TT:
                if _DBG_ATTR is not None and len(toks) <= 5:
                    _DBG_ATTR.append((
                        [_sfm(int(t)) for t in toks],
                        len(_uv),
                        [_sfm(int(c[0])) if c[0] < n
                         else label_surf.get(c[0], "?") for c in _uv],
                        (lambda r: _sfm(int(r[0])) if r[0] is not None
                         and r[0] < n else (label_surf.get(r[0], "?")
                                            if r[0] is not None else None))(
                            _trung_tam_trai(toks, i + 1)),
                    ))
                # THUỘC TÍNH: bên phải lấy ứng viên PHẢI NHẤT (danh từ cụ thể),
                # bên trái là chính thuộc tính. `nghịch ngợm như loài chó` ⇒
                # `nghịch ngợm → chó`, thay vì `ngợm → loài` của luật phân loại.
                if _uv:
                    # VẬT MANG PHẢI LÀ DANH TỪ — TK_052 vòng 5. Trên văn bản
                    # thật, vế phải của `như` nhặt cả hư từ: đo được vật mang
                    # `cả`·`chỉ`·`có`, và lớp của chúng thành `dài`·`trắng`.
                    # §69 đã siết đúng cách này cho kênh vị ngữ; kênh SO SÁNH
                    # thì chưa. Không ứng viên nào là danh từ ⇒ giữ luật cũ.
                    # VÀ PHẢI ĐỦ ĐẶC THÙ — TK_052 vòng 6. Từ điển gọi `các`·
                    # `thể` là DANH TỪ (đúng bệnh §64), nên lọc từ loại không
                    # đủ: `via_attr[nhanh]` cho top-4 là `các·thể·đá·ưng`, đẩy
                    # `cắt`·`sóc`·`dơi` ra khỏi hạt giống. Dùng chính ngưỡng
                    # `det_cut` của luật chọn đầu.
                    _uvd = ([c for c in _uv
                             if "danh từ" in _pos_cua(c[0])
                             and c[2] >= det_cut] if pos_map else [])
                    # KHÔNG có ứng viên nào vừa là DANH TỪ vừa đủ đặc thù ⇒
                    # KHÔNG phát cạnh (nguyên tắc §64). Đường lui `_uvd or _uv`
                    # để lọt `các`/`thể` và chúng chiếm hạt giống của ①.
                    if pos_map and not _uvd:
                        continue
                    _b = max(_uvd or _uv,
                             key=lambda c: (c[3] + c[1] - 1, c[1]))[0]
                    _a2, _ = _trung_tam_trai(toks, i + 1)
                    # KHÔNG CÓ ĐẦU NGỮ ĐOẠN THÌ KHÔNG PHÁT CẠNH — §68.
                    # Đường lui cũ lấy token SÁT pivot, tức lấy đúng `là` của
                    # «X là như…»: `là` đứng đầu bảng thuộc tính với 245 vật
                    # mang. Cùng nguyên tắc §64: thà không phát còn hơn phát
                    # vào hư từ.
                    if _a2 is None and not attr_duong_lui:
                        continue
                    _src = int(_a2) if _a2 is not None else int(toks[i])
                    # DẤU HIỆU SO SÁNH KHÔNG PHẢI THUỘC TÍNH — §68.
                    # `chu_gia` (§37) = token đứng ngay trước pivot ở tỉ lệ rất
                    # cao; nó xưa nay CHỈ áp cho kênh phân loại. Không áp ở đây
                    # nên vế trái nhặt luôn dấu hiệu so sánh: đo trên
                    # `classrich2`, «thuộc tính» nhiều vật mang nhất là
                    #   là 245 · giống 102 · cũng 85 · tự 42 · khác 41 ·
                    #   coi 38 · tương tự 34 · chẳng 26 · hạn 30
                    # tức `giống như` · `cũng như` · `chẳng hạn như` ·
                    # `tương tự như` — toàn khuôn dẫn, không phải thuộc tính.
                    if chu_gia and int(_src) in chu_gia:
                        continue
                    # DUAL-WRITE chỉ cho thành phần LÀ TỪ NỘI DUNG. Không lọc
                    # thì cụm chứa pivot/hư từ ghi luôn cạnh theo token đó:
                    # `là` đứng đầu bảng thuộc tính với 245 vật mang (§68).
                    _tp = [int(x) for x in label_span.get(_src, ())
                           if _sfm(int(x)) not in piv
                           and _sfm(int(x)).lower() not in _FUNC
                           and len(_sfm(int(x))) > 1]
                    for _s2 in ({int(_src)} | set(_tp)):
                        _sl = out_attr.setdefault(int(_s2), {})
                        _pv = _sl.get(int(_b))
                        _sl[int(_b)] = (pw if _pv is None else _pv[0] + pw, m)
                continue
            best_b, best_g = _chon_phai(_uv)
            # DANH SÁCH LỚP TỪ ĐIỂN — §76. §75 chứng minh nhãn rác KHÔNG lọc
            # được bằng tín hiệu nội tại (sáu tín hiệu trượt), nên chuyển sang
            # hỏi NGUỒN: chỉ nhận nhãn từng được từ điển dùng làm GIỐNG trong
            # lời giải nghĩa («chó là GIA SÚC…» · «cắt là CHIM…»). Bake một
            # lần, runtime chỉ đọc bảng — cùng tiền lệ `pos_map`/`ws_spans`.
            if lop_trang is not None and best_b is not None:
                _sfb = (label_surf.get(int(best_b))
                        or _sfm(int(best_b))).strip().lower()
                if _sfb not in lop_trang:
                    best_b = None
            if _DBG_CLS is not None and _sfm(int(toks[0])) in _DBG_CLS[0]:
                _DBG_CLS.append((
                    [_sfm(int(t)) for t in toks][:12],
                    (lambda r: (_sfm(int(r[0])) if r[0] is not None
                                and r[0] < n else label_surf.get(r[0], "?")
                                if r[0] is not None else None))(
                        _trung_tam_trai(toks, i + 1)),
                    [(_sfm(int(c[0])) if c[0] < n
                      else label_surf.get(c[0], "?"), round(c[2], 2), c[3])
                     for c in _uv],
                    (_sfm(int(best_b)) if best_b is not None
                     and best_b < n else label_surf.get(best_b, "?")
                     if best_b is not None else None),
                    round(det_cut, 3),
                ))
            if best_b is None:
                continue
            # Nhãn phải ĐỦ ĐẶC THÙ. Loại `vậy`/`thế`/`gì` — chúng là HỒI CHỈ và
            # CÂU HỎI, không phải lớp. Đo v12: nhãn `vậy` hút 8/22 dòng và cho
            # y hệt một danh sách; nguồn là truyện mạng («thì ra là vậy»,
            # «Hy vọng là vậy :)») — 54 lần trong conv_stories.
            # Dùng PHÂN VỊ chứ không ngưỡng hằng: ngưỡng hằng chỉnh trên mini
            # không chuyển được sang v12 (đã dính đúng bẫy này với `bậc-vào >=3`,
            # cột đối chứng bão hoà 88%). Lọc ở vòng hai, sau khi có phân bố.
            # DUAL-WRITE: chủ ngữ là CỤM thì phát thêm từ từng token thành
            # phần — nếu chỉ có id cụm thì tra theo token đơn (`chó`) không
            # thấy gì. Cùng khuôn `analogy_bridge_gen` (dual-write token cuối).
            _bagp = label_par.setdefault(int(best_b), set())
            if len(_bagp) < 200:
                _bagp.add(int(_pi))
            for _c in toks:
                _ci = int(_c)
                if _ci == m or _ci == a or _ci == best_b:
                    continue
                if _sfm(_ci) in _FUNC:
                    continue
                _bag = label_ctx.setdefault(int(best_b), {})
                if len(_bag) < 400 or _ci in _bag:
                    _bag[_ci] = _bag.get(_ci, 0.0) + pw
            _sach = not (chu_gia and (int(a) in chu_gia
                                      or any(int(x) in chu_gia
                                             for x in label_span.get(a, ()))))
            for a_ in ({a} | set(label_span.get(a, ()))):
                buf.append((int(a_), best_b, pw, m, best_g, _sach))
            continue
            slot = out.setdefault(a, {})
            prev = slot.get(best_b)
            slot[best_b] = (pw if prev is None else prev[0] + pw, m)

    # Vòng hai — bỏ nhãn KÉM ĐẶC THÙ NHẤT theo phân vị. Loại `vậy`/`thế`/`gì`
    # (hồi chỉ + câu hỏi) mà không cần danh sách từ nào.
    if buf:
        gs = sorted(x[4] for x in buf)
        cut = gs[min(len(gs) - 1, int(len(gs) * float(label_gidf_q)))]
        # Vòng ba (`label_indeg_q > 0`) — LỌC THEO SỐ CHỦ NGỮ KHÁC NHAU.
        #
        # `via_pivot` KHÔNG nghèo: 18 903 cạnh typed trên `classrich`. Nó đầy
        # cạnh SAI LOẠI — chỉ 18,1% cạnh lớp thật lọt vào (§20.5), và nhãn
        # "typed" vì thế không bảo đảm là nhãn LỚP:
        #     buộc  → 'Trãi'   (Nguyễn Trãi là danh tướng — MÔ TẢ, không phải lớp)
        #     dòng  → 'Hoài'
        # Phân biệt được bằng chính đồ thị: LỚP nhận cạnh từ NHIỀU chủ ngữ khác
        # nhau (`quốc gia` ← Cuba, Canada, Angkor…), MÔ TẢ chỉ từ một
        # (`danh tướng` ← Trãi). Thuần dữ liệu, không danh sách từ (Luật 0).
        chu: dict[int, set[int]] = {}
        for a_, b_, _w, _m, g_, _sc in buf:
            if g_ >= cut:
                chu.setdefault(b_, set()).add(a_)
        for b_, srcs in chu.items():
            label_indeg[b_] = len(srcs)
        keep_b: set[int] | None = None
        if float(label_indeg_q) > 0.0:
            degs = sorted(len(x) for x in chu.values())
            if degs:
                dcut = degs[min(len(degs) - 1,
                                int(len(degs) * float(label_indeg_q)))]
                keep_b = {b for b, srcs in chu.items() if len(srcs) >= dcut}
        # LỌC PHÂN VỊ CHỈ ÁP CHO BỂ ③ — §56. §55 đo được hai chiều ngược nhau:
        #   tắt lọc → ngưỡng nhiễu 3 069 → ≥20 316 · classrich ②top1 15,2→19,4%
        #   tắt lọc → thước sinh tầng ② ĐÚNG 0,290 → 0,995  (HỎNG NẶNG)
        # Vì hai bộ lọc phục vụ ③ (thành viên sạch) còn ② cần bể PHIẾU rộng —
        # đúng bệnh «tài nguyên dùng chung» §37, nhưng lần này lọc chạy TRƯỚC
        # khi tách bể nên nó cắt cả hai. Nay: bể ② lấy TẤT CẢ, bể ③ lấy phần
        # qua lọc.
        for a_, b_, w_, m_, g_, sc_ in buf:
            _qua = (g_ >= cut) and (keep_b is None or b_ in keep_b)
            _emit(a_, b_, w_, m_, sach=(sc_ and _qua), chi_sach=False)
            if not _qua:
                continue
    return (out, out_sach, label_surf, label_span, label_gidf, label_indeg,
            label_ctx, label_par, out_attr)


def build_hub_spread_index(
    state: GraphState,
    *,
    d_max: int = 3,
    min_count: int = 2,
    head_lookahead: int = 3,
    use_isg: bool = True,
    use_liet_ke: bool = True,
    label_gidf_q: float = 0.20,
    label_indeg_q: float = 0.0,
    chu_ngu_ket_hop_q: float = 0.0,
    pos_map: dict[str, str] | None = None,
    head_rule: str = "max_gidf",
    head_rule_trai: str = "max_gidf",
    bien_tu_pivot: bool = True,
    tu_choi_khi_toan_dinh_tu: bool = True,
    head_gidf_q: float | None = None,
    kenh_vi_ngu: bool = False,
    attr_duong_lui: bool = False,
    lop_trang: set[str] | None = None,
    loc_dich: bool = False,          # §93 — lọc đích không thể là LỚP
    attr_w_min: float = 0.0,         # §106 — ngưỡng trọng số cạnh via_attr
    gop_manh_tu: bool = False,
    ws_model=None,
    ws_spans: set[tuple[int, ...]] | None = None,
) -> HubSpreadIndex:
    clear_w_skip_cache()
    ws = ensure_w_skip(
        state, d_max=int(d_max), min_count=int(min_count), content_only=True,
    )
    fwd, diversity = _aggregate_mats(ws, d_max=d_max, min_count=min_count)
    rev = _build_rev(fwd)
    # Ưu tiên span ĐÃ BAKE (sidecar) — tính lại tốn 2,5 phút trên v12.
    wsp = ws_spans
    if wsp is None and ws_model is not None:
        wsp = ws_word_spans(state, ws_model)
    (via, via_sach, lsurf, lspan, lgidf, lindeg, lctx,
     lpar, vattr) = _mine_via_pivot(
        state, head_lookahead=head_lookahead, use_isg=use_isg,
        use_liet_ke=use_liet_ke, label_gidf_q=label_gidf_q,
        label_indeg_q=label_indeg_q,
        chu_ngu_ket_hop_q=chu_ngu_ket_hop_q, pos_map=pos_map,
        head_rule=head_rule, head_rule_trai=head_rule_trai, ws_spans=wsp,
        bien_tu_pivot=bien_tu_pivot,
        tu_choi_khi_toan_dinh_tu=tu_choi_khi_toan_dinh_tu,
        head_gidf_q=head_gidf_q, kenh_vi_ngu=kenh_vi_ngu,
        attr_duong_lui=attr_duong_lui, lop_trang=lop_trang,
        gop_manh_tu=gop_manh_tu)
    # ── §93 LỌC ĐÍCH — bỏ cạnh «A là B» mà B không thể là LỚP ────────────
    # Đo được (§92.3): trong 223 988 cạnh `via_pivot`, chỉ 39,1% có đích là
    # danh từ; 61% không phải quan hệ phân loại gì cả — `Nước là 0`,
    # `con là 3`, `Ngọc là 2`, `Lan là Amsterdam`. Chuỗi §89 sống được chỉ
    # nhờ tầng xếp hạng nặng phía sau che đi nền hỏng này (bài học 147).
    #
    # Lọc THẬN TRỌNG, hai luật, cả hai đều không cần danh sách từ viết tay:
    #   1. đích là SỐ / DẤU CÂU / EOS / rỗng  ⇒ bỏ (không bao giờ là lớp)
    #   2. từ điển NÓI đích không phải danh từ ⇒ bỏ
    # Đích KHÔNG có trong từ điển thì GIỮ — nhãn lớp nhiều từ (`kim loại màu
    # xám`) và tên riêng làm lớp đều nằm ngoài từ điển; bỏ chúng là bỏ cả
    # phần đúng. Thà lọc thiếu còn hơn lọc nhầm.
    if loc_dich and pos_map:
        import re as _re
        _bo = 0
        for _s in list(via):
            _d = via[_s]
            for _t in list(_d):
                _su = (lsurf.get(int(_t)) or _surf(state.vocab, int(_t))).strip()
                _lo = _su.lower()
                _xau = (not _lo) or ("<EOS>" in _su) or _re.fullmatch(
                    r"[\d.,;:!?%()\[\]\"'\-–—/]+", _su) is not None
                if not _xau:
                    _p = pos_map.get(_lo)
                    _xau = bool(_p) and "danh từ" not in _p
                if _xau:
                    del _d[int(_t)]
                    _bo += 1
            if not _d:
                del via[_s]
        _DBG_LOC.append(_bo)

    # ── §106 LỌC VẬT MANG THEO TRỌNG SỐ ──────────────────────────────────
    # Đo (§106.1) trên 10 thuộc tính, nhãn thật/rác xác thực bằng khuôn
    # «A như X» tra trên toàn bộ 450 303 cha:
    #     chứng thực  n=24 · trung vị 3,40 · max 4,25
    #     KHÔNG       n=20 · trung vị 1,20 · MAX 2,40
    # Ngưỡng 2,5 ⇒ rác lọt 0/20, thật giữ 16/24: chính xác 100%, phủ 67%.
    # Rác điển hình là MẢNH TỪ GHÉP: `khoẻ như lực sĩ` tách thành `lực`+`sĩ`.
    if attr_w_min > 0:
        for _s in list(vattr):
            _d = vattr[_s]
            for _t in list(_d):
                _val = _d[_t]
                _w = float(_val[0]) if isinstance(_val, (tuple, list)) else float(_val)
                if _w < attr_w_min:
                    del _d[_t]
            if not _d:
                del vattr[_s]

    lpos: dict[int, str] = {}
    if pos_map:
        for _t in {x for d_ in via.values() for x in d_}:
            _su = (lsurf.get(int(_t)) or _surf(state.vocab, int(_t))).strip().lower()
            _p = pos_map.get(_su)
            if _p:
                lpos[int(_t)] = _p
    # HỌ HÀNG NHÃN — §46. Mỗi nhãn xưa nay là một ĐẢO: `triều đại` và
    # `vương triều` chia nhau thành viên mà điểm của chúng không liên quan gì
    # nhau. Dựng đồ thị nhãn–nhãn từ THÀNH VIÊN CHUNG rồi cho phiếu lan qua.
    # Đây là tín hiệu THÊM cấu trúc (§45.3), không phải lọc.
    lkin: dict[int, dict[int, float]] = {}
    vrev: dict[int, set[int]] = {}
    for _s, _d in via.items():
        for _t in _d:
            vrev.setdefault(int(_t), set()).add(int(_s))
    # CHỈ dựng họ hàng trên NHÃN LỚP THẬT. §46 dựng trên MỌI đích typed ⇒
    # `Đây ← hydro, bưu kiện`, phiếu lan sang rác, ②top1 41,4% → 38,9%.
    # Tiêu chí phân biệt đã có sẵn và đã chứng minh có tác dụng ở §24:
    # BẬC-VÀO (số chủ ngữ khác nhau) — lớp nhận cạnh từ nhiều chủ ngữ, mô tả
    # chỉ từ một. Ngưỡng theo phân vị, không phải hằng.
    _degs = sorted(len(v_) for v_ in vrev.values() if v_)
    _dcut = _degs[min(len(_degs) - 1, int(len(_degs) * 0.90))] if _degs else 2
    _mem = {t: set(v_) for t, v_ in vrev.items()
            if max(_dcut, 3) <= len(v_) <= 400}
    _by_src: dict[int, list[int]] = {}
    for _t, _srcs in _mem.items():
        for _sx in _srcs:
            _by_src.setdefault(_sx, []).append(_t)
    for _t, _srcs in _mem.items():
        _cnt: dict[int, int] = {}
        for _sx in _srcs:
            for _u in _by_src.get(_sx, ()):
                if _u != _t:
                    _cnt[_u] = _cnt.get(_u, 0) + 1
        if not _cnt:
            continue
        _row = {}
        for _u, _c in _cnt.items():
            _j = _c / float(min(len(_srcs), len(_mem[_u])))
            if _j >= 0.25:
                _row[_u] = _j
        if _row:
            lkin[_t] = dict(sorted(_row.items(), key=lambda kv: -kv[1])[:12])
    return HubSpreadIndex(
        ws=ws, fwd=fwd, rev=rev, diversity=diversity, via_pivot=via,
        d_max=int(d_max), min_count=int(min_count),
        label_surf=lsurf, label_span=lspan, label_gidf=lgidf,
        label_indeg=lindeg,
        via_rev=vrev,
        via_pivot_sach=via_sach,
        label_pos=lpos,
        label_ctx=lctx,
        label_kin=lkin,
        label_par=lpar,
        via_attr=vattr,
        label_by_surf=_bang_be_mat(lsurf),
    )


def score_hubs_from(
    state: GraphState,
    index: HubSpreadIndex,
    seed_ids: list[int],
    *,
    topk: int = 20,
    gidf_min: float = 4.0,
    ban_func: bool = True,
    channel: str = "typed_first",
    prefer_via_pivot: bool | None = None,
    rank: str = "diem",
    loc_pos: bool = False,
    chi_pivot_la: bool = False,
    loc_ghep_tu: float = 0.0,
    kenh_thuoc_tinh: bool = False,
    ghep_tu: dict[int, set[int]] | None = None,
    giu_ghep: set[int] | None = None,
    uu_tien_tu_du: bool = True,
    ctx_ids: list[int] | None = None,
    par_ids: list[int] | None = None,
) -> list[HubHit]:
    """Đích từ seed. `channel` quyết định cách trộn HAI KÊNH KHÁC THANG ĐO.

    - ``typed`` — cạnh có kiểu ``A·là/như·B``: đếm câu ĐỊNH NGHĨA. **Không tăng
      theo cỡ corpus** (thêm 20 000 câu không liên quan ⇒ số không đổi).
    - ``cooc`` — đồng hiện W_skip: **tăng tuyến tính** theo cỡ corpus.

    Cộng hai kênh vào một điểm (bản cũ: `boost=8.0`) là sai về đơn vị. Đo trên
    ví dụ khỉ/chó/mèo + nhiễu (`tests/test_hub_spread_typed_channel.py`)::

        nhiễu       cooc: `xóm`      typed: `động`
            0                 60,4              3,0
        2 000                902,8              3,0
       20 000              9 529,4              3,0

    Một câu đời thường (`chó nhà hàng xóm sủa to`) lặp lại đủ nhiều là chôn
    `chó là động vật` dưới 3 bậc độ lớn — hệ số nhân hằng nào cũng thua. Đây là
    lý do `hub_spread` ĐẠT trên synthetic mà CÂM trên `isg_pw` (TK_046 §18.H).

    Chế độ:

    - ``typed_first`` (mặc định): xếp hạng theo **(kênh, điểm)** — mọi hit typed
      đứng trước mọi hit cooc. Bất biến theo nhiễu, vẫn giữ cooc làm phần đệm.
    - ``typed`` : chỉ typed.
    - ``cooc``  : chỉ đồng hiện (baseline).
    - ``mixed`` : hành vi CŨ (cộng điểm, boost 8×) — giữ để A/B.
    """
    if prefer_via_pivot is not None:  # tương thích ngược
        channel = "mixed" if prefer_via_pivot else "cooc"
    if channel not in ("typed_first", "typed", "cooc", "mixed"):
        raise ValueError(f"channel lạ: {channel!r}")
    if rank not in ("diem", "phieu", "phieu_chuan", "phieu_gidf",
                    "phieu_indeg", "phieu_vong_ve", "phieu_cos",
                    "phieu_x_gidf", "phieu_x_indeg", "phieu_x_vongve",
                    "phieu_x_pos", "phieu_x_all", "phieu_x_ctx",
                    "phieu_x_ho", "phieu_x_ctx_ho", "phieu_x_doc",
                    "phieu_x_ctx_doc"):
        raise ValueError(f"rank lạ: {rank!r}")
    vocab = state.vocab
    acc: dict[int, HubHit] = {}
    phieu: dict[int, set[int]] = {}
    _sid_hien = [0]

    # BẠN GHÉP TỪ ở bước ① — §50. `nghịch` hút `nghịch lý`, `nghịch lý Fermi`,
    # `tinh nghịch`: ở ~10k cha bước ① trôi hẳn sang đó và cả chuỗi hỏng (§49.1).
    # Đây là NGƯỠNG NHIỄU thật của cơ chế, mà 13 tín hiệu ở ② không hề chạm tới.
    #
    # Tiêu chí: `b` là bạn ghép từ của `a` nếu `b` đứng NGAY SAU `a` (d=1) ở một
    # TỈ LỆ CAO trong tổng số lần `a` xuất hiện. Thuần dữ liệu, chạy được cả khi
    # chưa bake WS-span (checkpoint đường-cong-nhiễu không có span).
    # Quét d=1..2: bạn ghép từ KHÔNG chỉ đứng liền kề. `nghịch lý Fermi` cho
    # `Fermi` ở d=2; `phép nghịch đảo` cho `phép` ở d=2 phía trước. Lọc d=1 gỡ
    # được `lý` mà `Fermi`/`quỹ`/`phép` còn nguyên (đo được).
    _ghep: set[int] = set()
    if ghep_tu:
        # RANH GIỚI TỪ (§52): loại `b` nếu «seed b» là một TỪ trong từ điển,
        # TRỪ những `b` nằm trong `giu_ghep` — tức chính từ ghép của thuộc tính
        # đang hỏi (`nghịch ngợm`), thứ ta muốn GIỮ.
        for _sid in seed_ids:
            _ghep |= ghep_tu.get(int(_sid), set())
        if giu_ghep:
            _ghep -= set(giu_ghep)
    if loc_ghep_tu > 0.0 and index.ws is not None and index.ws.d_max >= 1:
        for _d in (1, 2):
            if _d > index.ws.d_max:
                break
            _md = index.ws.mats[_d]
            for _sid in seed_ids:
                _si = int(_sid)
                if not (0 <= _si < _md.shape[0]):
                    continue
                _row = _md.getrow(_si)
                _tot = float(_row.sum())
                if _tot <= 0:
                    continue
                _nguong = float(loc_ghep_tu) * (1.0 if _d == 1 else 1.5)
                for _j, _w in zip(_row.indices, _row.data):
                    if float(_w) / _tot >= _nguong:
                        _ghep.add(int(_j))

    _la_ids = {i for i in range(len(state.vocab))
               if _surf(state.vocab, i).lower() == "là"} if chi_pivot_la else set()

    def _consider(
        dst: int, w: float, d: int, *, boost: float = 1.0, typed: bool = False,
        mid: int | None = None,
    ) -> None:
        surf = _surf_ix(index, vocab, dst)
        if not surf or len(surf) <= 1:
            return
        if ban_func and (surf in _FUNC or surf.lower() in _FUNC):
            return
        g = _gidf_ix(index, state, dst)
        if g < float(gidf_min):
            return
        # CHỈ lọc khi gọi cho BƯỚC ② (`loc_pos=True`). `score_hubs_from` phục vụ
        # cả ① lẫn ②; tính từ là HẠT GIỐNG hợp lệ ở ① (`trắng` → mèo, áo) nhưng
        # KHÔNG BAO GIỜ là nhãn lớp ở ②. Bật chung làm cơ hội ① tụt 67 → 55 —
        # đúng bệnh «tài nguyên dùng chung» của §37.
        # CHỈ PIVOT `là` cho bước ② — §45. Đếm trên `classrich`:
        #     là   697 cạnh lớp / 7320 khác   8,69%
        #     như    2 / 1642                 0,12%
        #     gồm    0 /  912                 0,00%
        #     gọi    0 /  197                 0,00%
        # `như`/`gồm`/`gọi` = 27% bể ② mà chỉ sinh 2/2751 cạnh lớp.
        # KHÔNG bật cho ①: `như` là pivot của chính ví dụ gốc
        # («nghịch ngợm NHƯ loài chó») nên nó cần cho bước ① — bệnh «tài nguyên
        # dùng chung» của §37/§40.
        if _ghep and int(dst) in _ghep:
            return
        if chi_pivot_la and typed and mid is not None and int(mid) not in _la_ids:
            return
        if loc_pos and index.label_pos and typed:
            _p = index.label_pos.get(int(dst))
            if _p in POS_KHONG_PHAI_LOP:
                return
        div = int(index.diversity[dst]) if dst < len(index.diversity) else 0
        sc = float(w) * float(g) * math.log1p(float(max(div, 1))) * float(boost)
        if typed:
            phieu.setdefault(dst, set()).add(_sid_hien[0])
        prev = acc.get(dst)
        # typed thắng cooc bất kể điểm; trong cùng kênh mới so điểm.
        if prev is None or (typed, sc) > (prev.typed, prev.score):
            acc[dst] = HubHit(
                tid=int(dst), surface=surf, score=sc,
                weight=float(w), gidf=g, diversity=div, best_d=int(d),
                typed=bool(typed),
            )

    for sid in seed_ids:
        _sid_hien[0] = int(sid)
        if kenh_thuoc_tinh:
            for dst, (w, _mid) in index.via_attr.get(int(sid), {}).items():
                _consider(dst, w, d=2, boost=1.0, typed=True, mid=_mid)
        if channel != "cooc" and not kenh_thuoc_tinh:
            for dst, (w, _mid) in index.via_pivot.get(int(sid), {}).items():
                _consider(
                    dst, w, d=2,
                    boost=8.0 if channel == "mixed" else 1.0,
                    typed=channel != "mixed", mid=_mid,
                )
        if channel != "typed":
            for dst, (w, d) in index.fwd.get(int(sid), {}).items():
                _consider(dst, w, d, boost=1.0, typed=False)

    if channel == "mixed":
        return sorted(
            acc.values(), key=lambda h: (-h.score, -h.gidf, h.surface)
        )[:topk]
    if rank in ("phieu", "phieu_chuan", "phieu_gidf", "phieu_indeg",
                "phieu_vong_ve", "phieu_cos", "phieu_x_gidf", "phieu_x_indeg",
                "phieu_x_vongve", "phieu_x_pos", "phieu_x_all",
                "phieu_x_ctx", "phieu_x_ho", "phieu_x_ctx_ho",
                "phieu_x_doc", "phieu_x_ctx_doc"):
        _seeds = {int(x) for x in seed_ids}
        _ctx = {int(x) for x in (ctx_ids or [])} - _seeds
        _q = None
        if rank == "phieu_cos" and state.embedding is not None:
            _E = state.embedding.matrix
            _vs = [_E[i] for i in _seeds if 0 <= i < len(_E)]
            if _vs:
                _q = np.mean(_vs, axis=0).astype(np.float32)
                _n = float(np.linalg.norm(_q))
                _q = _q / _n if _n > 1e-9 else None

        def _cos(tid: int) -> float:
            """cosine(E[nhãn], trung bình E[hạt giống]).

            Tín hiệu TRỰC GIAO với đồng thuận-đếm: §40.4 đo được bảy tín hiệu đã
            thử đều TƯƠNG QUAN với thứ `rank="phieu"` đã dùng nên đều thừa hoặc
            hại. `E` là thứ duy nhất chuỗi chưa động tới — cosine không suy ra
            được từ số phiếu.

            Nhãn CỤM (id ảo) không có hàng trong `E` ⇒ lấy trung bình token
            thành phần.
            """
            if _q is None:
                return 0.0
            _E2 = state.embedding.matrix
            if 0 <= tid < len(_E2):
                w = _E2[tid]
            else:
                sp = index.label_span.get(int(tid))
                if not sp:
                    return 0.0
                vs = [_E2[int(x)] for x in sp if 0 <= int(x) < len(_E2)]
                if not vs:
                    return 0.0
                w = np.mean(vs, axis=0)
            nw = float(np.linalg.norm(w))
            return float(np.dot(w, _q) / nw) if nw > 1e-9 else 0.0


        def _phu(h: HubHit) -> float:
            """Phá hoà kiểu THAY `score` — giữ làm ĐỐI CHỨNG ÂM (§41.1).

            Bốn tín hiệu dùng kiểu này đều rơi về ~0–1,5% top-1 vì chúng chiếm
            chỗ của `score`. Dạng ghép ĐÚNG là NHÂN — xem `_he_so`.
            """
            if rank == "phieu_gidf":
                return float(h.gidf)
            if rank == "phieu_indeg":
                return float(index.label_indeg.get(h.tid, 0))
            if rank == "phieu_vong_ve":
                return float(len(index.via_rev.get(h.tid, frozenset()) & _seeds))
            return 0.0

        def _uu(h: HubHit) -> float:
            nv = len(phieu.get(h.tid, ())) if h.typed else 0
            if rank == "phieu" or not nv:
                return float(nv)
            # CHUẨN HOÁ theo nền: `nhà`/`nước` nhận cạnh từ hàng trăm chủ ngữ
            # nên 2 phiếu ở đó là ngẫu nhiên; `triều đại` chỉ 7 chủ ngữ nên 2
            # phiếu là tín hiệu. Đây đúng bệnh đã làm hỏng `phieu` lần đầu —
            # đếm thô thì tần suất nền nuốt (TK_046 §20.4).
            nen = float(index.label_indeg.get(h.tid, 1) or 1)
            return (nv * nv) / nen
        # ĐỒNG THUẬN quyết, CHỈ trong kênh typed. Thiết kế của tác giả: `động
        # vật` thành nhãn vì chó+mèo+khỉ CÙNG trỏ vào nó — không phải vì cạnh
        # nào nặng nhất.
        #
        # Đo trên đường cong nhiễu (`probe_chain3_noise_curve.py`) tại 3 069 cha,
        # sau khi `head_rule_trai` phục hồi cạnh vàng:
        #     mèo → [vật:0,95 · đặc:1,90]
        # `đặc` (nhiễu, MỘT hạt giống bầu) nặng gấp đôi `vật` (BA hạt giống bầu)
        # ⇒ xếp theo điểm chọn `đặc`, xếp theo phiếu chọn `vật`. Ghép với
        # `head_rule_trai` đẩy điểm gãy 895 → 3 069 cha.
        #
        # ⚠️ CHỈ trong kênh typed. Đếm phiếu trên kênh cooc thì TẦN SUẤT NỀN
        # nuốt: đo trên classrich, `thi`/`học`/`đều`/`đầu` thắng vì cái gì cũng
        # đồng hiện với cái gì (TK_046 §20.4).
        _NHAN = {"phieu_cos", "phieu_x_gidf", "phieu_x_indeg", "phieu_x_vongve",
                 "phieu_x_pos", "phieu_x_all", "phieu_x_ctx",
                 "phieu_x_ho", "phieu_x_ctx_ho", "phieu_x_doc",
                 "phieu_x_ctx_doc"}

        def _ctx_khop(tid: int) -> float:
            """Phần ngữ cảnh HIỆN TẠI khớp chữ ký của nhãn — §43.

            Nhãn `triều đại` học được chữ ký {vua, nhà, năm, trị vì…} từ chính
            các câu `X là triều đại`. Prompt nói về vua chúa thì `triều đại`
            được nâng, `khoảng`/`phần` thì không. Tín hiệu này KHÔNG suy ra được
            từ số phiếu, điểm, gidf, bậc-vào, E hay từ loại.
            """
            if not _ctx:
                return 0.0
            bag = index.label_ctx.get(int(tid))
            if not bag:
                return 0.0
            # ⚠️ ĐÃ THỬ cân theo gidf (token ngữ cảnh hiếm đáng giá hơn) —
            # TỆ HƠN: 41,4% → 40,8% (§45.2). Dạng đếm ĐỀU thắng; giữ nguyên.
            return len(_ctx & set(bag)) / float(len(_ctx))

        def _he_so(h: HubHit) -> float:
            """Hệ số NHÂN vào `score` — dạng ghép ĐÚNG (§41.4 số 48).

            Mỗi tín hiệu đưa về [0,1] rồi nhân `(1 + k)`; `score` giữ vai trò
            nền. Bảy tín hiệu ở §28/§35/§40 từng bị kết luận là vô dụng, nhưng
            §41.1 cho thấy chúng chưa bao giờ được thử ở dạng này — chúng bị
            đem THAY `score` chứ không NHÂN vào.
            """
            k = 1.0
            r = rank
            if r in ("phieu_cos", "phieu_x_all"):
                k *= 1.0 + max(_cos(int(h.tid)), 0.0)
            if r in ("phieu_x_gidf", "phieu_x_all"):
                k *= 1.0 + min(float(h.gidf) / 10.0, 1.0)
            if r in ("phieu_x_indeg", "phieu_x_all"):
                k *= 1.0 + min(
                    math.log1p(float(index.label_indeg.get(h.tid, 0))) / 5.0, 1.0)
            if r in ("phieu_x_vongve", "phieu_x_all"):
                ov = len(index.via_rev.get(h.tid, frozenset()) & _seeds)
                k *= 1.0 + (ov / max(len(_seeds), 1))
            if r in ("phieu_x_doc", "phieu_x_ctx_doc") and par_ids:
                # VÙNG TÀI LIỆU: nhãn có cạnh sinh ra từ những câu GẦN câu hiện
                # tại (cùng bài) thì đáng tin hơn. Thông tin GIỮA CÁC CÂU —
                # `via_pivot` xưa nay xử lý từng câu độc lập.
                _bp = index.label_par.get(int(h.tid))
                if _bp:
                    _g = sum(1 for x in par_ids
                             if any(abs(x - y) <= 20 for y in _bp))
                    k *= 1.0 + 2.0 * (_g / float(len(par_ids)))
            if r in ("phieu_x_ho", "phieu_x_ctx_ho"):
                # PHIẾU LAN QUA HỌ: `triều đại` được 2 phiếu, `vương triều` 1 —
                # nếu hai nhãn chia nhau thành viên thì phiếu của họ hàng cũng
                # là bằng chứng cho nhãn này.
                _th = 0.0
                for _u, _j in index.label_kin.get(int(h.tid), {}).items():
                    _th += _j * len(phieu.get(_u, ()))
                k *= 1.0 + min(_th, 3.0)
            if r in ("phieu_x_ctx", "phieu_x_all", "phieu_x_ctx_ho",
                     "phieu_x_ctx_doc"):
                k *= 1.0 + 2.0 * _ctx_khop(int(h.tid))
            if r in ("phieu_x_pos", "phieu_x_all"):
                _p = index.label_pos.get(int(h.tid))
                if _p in POS_KHONG_PHAI_LOP:
                    k *= 0.25
                elif _p == "danh từ":
                    k *= 1.5
            return k

        if rank in _NHAN:
            # NHÂN vào `score`, KHÔNG thay nó. Đo được (§41): bốn bộ phá hoà
            # trước (gidf · bậc-vào · khứ hồi · cosine) đều rơi về ~0–1,5% top-1
            # vì chúng CHIẾM CHỖ của `score` trong khoá sắp xếp — mà `score`
            # (`w·gidf·log(div)`) mới là thứ đang gánh sau khi phiếu hoà.
            # TỪ ĐỦ THẮNG MẢNH, rồi LỚP ĐÔNG THẮNG LỚP THƯA — §63.
            # Có span WS thì nhãn lớp là CỤM (`động vật`), và điểm của id-cụm
            # KHÔNG cùng thang với điểm token (đúng bệnh §18.H): đo trên toy
            # 99 362 cha, `động vật` và `cưng` HOÀ 2 phiếu nhưng `động vật`
            # score 2,7 < `cưng` 15,6 ⇒ mảnh thắng, chuỗi chết.
            #     hoà phiếu → TỪ ĐỦ    : 99 362 `cưng` → `động vật` ✓
            #     hoà tiếp  → LỚP ĐÔNG : 148 875 `cạnh tranh` → `động vật` ✓
            # `label_indeg` từng bị dùng làm BỘ LỌC ở §24/§55 và cắt mất cạnh
            # vàng; ở đây nó chỉ là khoá PHỤ sau khi phiếu đã hoà (bài học 48).
            def _tu_du(t: int) -> int:
                return 1 if len(index.label_span.get(int(t), ())) >= 2 else 0

            return sorted(
                acc.values(),
                key=lambda h: (not h.typed,
                               -(len(phieu.get(h.tid, ())) if h.typed else 0),
                               -_tu_du(h.tid) * int(uu_tien_tu_du),
                               -int(index.label_indeg.get(h.tid, 0))
                               * int(uu_tien_tu_du),
                               -h.score * _he_so(h),
                               -h.gidf, h.surface),
            )[:topk]
        return sorted(
            acc.values(),
            key=lambda h: (not h.typed, -_uu(h), -_phu(h), -h.score,
                           -h.gidf, h.surface),
        )[:topk]
    return sorted(
        acc.values(), key=lambda h: (not h.typed, -h.score, -h.gidf, h.surface)
    )[:topk]


def score_hubs_raw_compare(
    state: GraphState,
    index: HubSpreadIndex,
    seed_ids: list[int],
    *,
    topk: int = 20,
) -> list[HubHit]:
    """Baseline: chỉ theo weight (để so với là vs động/vật)."""
    vocab = state.vocab
    acc: dict[int, HubHit] = {}
    for sid in seed_ids:
        for dst, (w, d) in index.fwd.get(int(sid), {}).items():
            surf = _surf(vocab, dst)
            g = _gidf(state, dst)
            div = int(index.diversity[dst]) if dst < len(index.diversity) else 0
            prev = acc.get(dst)
            if prev is None or w > prev.weight:
                acc[dst] = HubHit(
                    tid=int(dst), surface=surf, score=float(w),
                    weight=float(w), gidf=g, diversity=div, best_d=int(d),
                )
    return sorted(acc.values(), key=lambda h: (-h.weight, h.surface))[:topk]


def bang_ke_ngu_canh(index: HubSpreadIndex, vocab,
                     ctx_ids: list[int]) -> dict[str, float]:
    """BỀ MẶT → trọng số, cho những gì ĐỨNG ĐƯỢC ngay sau một token ngữ cảnh.

    Tách ra từ `reverse_peers` (§60) để dùng lại ở đường đề cử TRỰC TIẾP (§72):
    chẩn đoán trên thước sinh v3 cho thấy **67,2% ca trượt top-5 là lỗi XẾP
    HẠNG** — cạnh tới lớp đúng CÓ SẴN nhưng bị nhãn khác của cùng chủ ngữ vượt.
    Với prompt «X là ___», nhãn nào từng thật sự đứng sau `là` trong câu train
    thì đáng tin hơn hẳn.

    Khoá theo BỀ MẶT chứ không id (Luật 0): một bề mặt có nhiều id BPE.
    """
    ra: dict[str, float] = {}
    bm: list[str] = []
    for i, t in enumerate(ctx_ids):
        s = _surf_ix(index, vocab, int(t))
        # ⚠️ TOKEN CUỐI luôn được tính dù là HƯ TỪ: với «X là ___» thì thứ
        # đứng ngay trước ô trống chính là `là`, và «cái gì hay đứng sau `là`»
        # mới là tín hiệu cần. Bỏ nó đi thì bảng kề chỉ còn «cái gì đứng sau X»
        # — đo được: tín hiệu phẳng (§72).
        cuoi = (i == len(ctx_ids) - 1)
        if s and s not in bm and (cuoi or (len(s) > 1 and s not in _FUNC)):
            bm.append(s)
    for s in bm:
        for c in (resolve_surface_ids(vocab, s) or [])[:6]:
            for dst, (w_, d_) in index.fwd.get(int(c), {}).items():
                if int(d_) > 1:
                    continue
                sd = _surf_ix(index, vocab, int(dst))
                if sd:
                    ra[sd] = max(ra.get(sd, 0.0), float(w_))
    return ra


def chon_hat_dong_thuan(
    index: HubSpreadIndex,
    vocab,
    ung_vien: list[HubHit],
    *,
    topk: int = 4,
    quet: int = 60,
    pos_map: dict[str, str] | None = None,
) -> list[HubHit]:
    """BƯỚC ① — chọn hạt theo ĐỒNG THUẬN LỚP thay vì top-k theo điểm (§66).

    Đo trên `classrich2`, 570 ca lớp thật:

        ① top-4 theo điểm : có ≥2 ĐỒNG-THÀNH-VIÊN của lớp đích →   0,0%
        ① top-60 (cả bể)  : có ≥2                              →  11,8%

    Tức thông tin CÓ trong bể, chỉ bị xếp hạng chôn — «không TÌM được», không
    phải «không BIỂU DIỄN được». Mà cơ chế phiếu ở ② CẦN ≥2 hạt cùng lớp mới
    phân biệt nổi, nên suốt §20–§65 mọi bản vá cho ② đều vá một bước có ĐẦU VÀO
    RỖNG.

    Thay vì xếp hạng từng ứng viên rời rạc, tìm NHÓM ứng viên đã cùng trỏ về
    một nhãn — đó chính là định nghĩa «cùng lớp». Cùng trực giác đồng thuận của
    §21, đặt ở ① thay vì ②. Đo được: ≥2 đồng-thành-viên 0,0% → 0,9%, ca ĐẠT
    2 → 5 trên 570.

    ⚠️ CHỈ DÙNG LÀM DỰ PHÒNG. Bật vô điều kiện thì GIẾT ca thử vàng: đồng
    thuận quét top-60 nên nó nhặt lại `nghịch đảo`/`nghịch lý` — đúng thứ §59
    đã loại — và toy gãy từ 9 983 cha trở lên (10/10 mức ĐẠT → 4/10). Luật
    đúng là luật của §59: **có ĐỦ hạt typed thì dùng typed; không đủ mới gọi
    hàm này.** Cùng trực giác, hai cách thi hành, kết quả trái ngược (bài học
    76 lần thứ hai).

    ⚠️ ĐÃ THỬ chia điểm nhóm cho `label_indeg` (để nhãn ĐẶC THÙ thắng nhãn
    chung chung): TỆ HƠN — ĐẠT 5 → 2. Nhóm lớn nhất thắng, không phạt gì.
    """
    # THÀNH VIÊN của một lớp là DANH TỪ. `trắng` kéo về [vôi, mắt, in] —
    # `in` là động từ, nó chiếm suất trong nhóm mà không bao giờ là thành viên.
    uv = ung_vien[:quet]
    if pos_map:
        _dt = [h for h in uv
               if "danh từ" in (pos_map.get(h.surface.lower().strip(), "")
                                or pos_map.get(" " + h.surface.strip(), ""))]
        if len(_dt) >= 2:
            uv = _dt
    nhan: dict[int, list[HubHit]] = {}
    for h in uv:
        for i in (resolve_surface_ids(vocab, h.surface) or [h.tid])[:3]:
            for d in index.via_pivot.get(int(i), {}):
                g = nhan.setdefault(int(d), [])
                if h not in g:
                    g.append(h)
    tot = max(nhan.values(), key=len, default=[])
    return tot[:topk] if len(tot) >= 2 else []


def reverse_peers(
    state: GraphState,
    index: HubSpreadIndex,
    hub_ids: list[int],
    *,
    exclude: set[int] | None = None,
    topk: int = 30,
    gidf_min: float = 0.0,
    channel: str = "typed_first",
    ctx_ids: list[int] | None = None,
    ke_ctx: float = 0.0,
) -> list[PeerHit]:
    """BƯỚC 3 — từ nhãn lan ngược ra THÀNH VIÊN.

    Cùng bệnh và cùng thuốc với `score_hubs_from`: trước đây trộn
    ``index.rev`` (đồng hiện W_skip) với ``via_pivot`` đảo bằng hệ số hằng
    ``×4.0`` — hai thang đo khác nhau ⇒ đồng hiện nuốt kênh typed.

    Đo trên v12 (`out/hub_step3_v12_isg_pw.json`): ``mèo`` → hub ``thú`` (đúng)
    → thành viên ``['hứng','yêu','thích','thật',…]`` = từ hay ĐỨNG GẦN ``thú``,
    không phải thứ ĐƯỢC GỌI LÀ thú.

    ``channel``: ``typed_first`` (mặc định) · ``typed`` · ``cooc`` ·
    ``mixed`` (hành vi cũ, giữ để A/B). Xem `score_hubs_from`.

    ``ctx_ids``/``ke_ctx`` — KỀ NGỮ CẢNH (§60). Trọng số typed KHÔNG phân biệt
    được ở quy mô lớn: trên toy 99 362 cha, 62/62 thành viên của nhãn ``động``
    đều nặng đúng 0,95 nên thứ tự rơi về ABC và ``khỉ`` tụt hạng 35, ngoài
    ``topk``. Thành viên nào đứng ĐƯỢC ngay sau một token ngữ cảnh thì nhân
    ``(1 + ke_ctx)`` — cùng dạng ghép với `_he_so` ở ② (§41.4). Cạnh vàng vốn
    còn nguyên; đây là thước xếp hạng, không phải cạnh mới.
    """
    if channel not in ("typed_first", "typed", "cooc", "mixed"):
        raise ValueError(f"channel lạ: {channel!r}")
    vocab = state.vocab
    excl = exclude or set()
    acc: dict[int, PeerHit] = {}
    # Bảng KỀ khoá theo BỀ MẶT, không theo id (Luật 0). Một bề mặt có nhiều id
    # BPE và `fwd` khoá theo TỪNG id — tra bằng id thô của `encode_prompt` bắt
    # hụt 3/6 cạnh kề trong ca thử vàng (§23.1, cùng bệnh, lần thứ tư).
    # Chỉ lấy token nội dung CUỐI — cái đứng ngay trước ô sắp điền («… như
    # loài ___»). KHÔNG lấy cả ngữ cảnh: THUỘC TÍNH cũng nằm trong đó, và nâng
    # thành viên từng đứng cạnh thuộc tính là giết đúng thứ ta cần — `khỉ` ra
    # được CHÍNH VÌ nó chưa bao giờ đứng cạnh `nghịch ngợm`. Rỗng thì lùi dần
    # về token trước đó.
    ke_bm: dict[str, float] = {}
    if ke_ctx and ctx_ids:
        _bm: list[str] = []
        for t in ctx_ids:
            s = _surf_ix(index, vocab, int(t))
            if s and len(s) > 1 and s not in _FUNC and s not in _bm:
                _bm.append(s)
        for s in reversed(_bm):
            for c in (resolve_surface_ids(vocab, s) or [])[:6]:
                for dst, (w_, d_) in index.fwd.get(int(c), {}).items():
                    if int(d_) > 1:
                        continue
                    sd = _surf_ix(index, vocab, int(dst))
                    if sd:
                        ke_bm[sd] = max(ke_bm.get(sd, 0.0), float(w_))
            if ke_bm:
                break

    def _add(src: int, w: float, d: int, *, typed: bool = False) -> None:
        if src in excl:
            return
        surf = _surf_ix(index, vocab, src)
        if not surf or len(surf) <= 1:
            return
        if surf in _FUNC or surf.lower() in _FUNC:
            return
        g = _gidf_ix(index, state, src)
        if g < float(gidf_min):
            return
        prev = acc.get(src)
        if prev is None or (typed, w) > (prev.typed, prev.weight):
            acc[src] = PeerHit(
                tid=int(src), surface=surf, weight=float(w), best_d=int(d),
                typed=bool(typed),
            )

    want = {int(h) for h in hub_ids}
    if channel != "typed":
        for hid in want:
            for src, (w, d) in index.rev.get(hid, {}).items():
                _add(src, w, d, typed=False)
    if channel != "cooc":
        # đảo via_pivot: ai được gán «là <hub>»
        # BƯỚC ③ đọc bể SẠCH (đã loại chủ ngữ giả do trạng ngữ); bước ② vẫn
        # dùng `via_pivot` đầy đủ vì nó cần bể PHIẾU rộng — §37. Rỗng thì rơi
        # về bể đầy đủ (tương thích ngược khi `chu_ngu_ket_hop_q=0`).
        _be = index.via_pivot_sach or index.via_pivot
        for src, dsts in _be.items():
            for hid in want:
                if hid in dsts:
                    w, _mid = dsts[hid]
                    _add(
                        src,
                        float(w) * (4.0 if channel == "mixed" else 1.0),
                        2,
                        typed=channel != "mixed",
                    )
    if channel == "mixed":
        return sorted(acc.values(), key=lambda p: (-p.weight, p.surface))[:topk]

    def _w(p: PeerHit) -> float:
        return p.weight * (1.0 + float(ke_ctx)) if p.surface in ke_bm else p.weight

    # ⚠️ TRẢ VỀ TRỌNG SỐ ĐÃ NHÂN, không chỉ dùng nó trong khoá sắp xếp — TK_052
    # vòng 2. `hub_chain_proposals` đọc `m.weight` rồi XẾP LẠI toàn cục, nên
    # thứ tự do `ke_ctx` tạo ra bị VỨT ngay khi hàm trả về. Đo trên corpus
    # TK_052: chuỗi chạy tay cho ③ ra m3 ở 73,8% bộ, mà đường sinh chỉ thu hồi
    # 29,0% — chênh lệch nằm đúng ở đây.
    ra_ = sorted(acc.values(), key=lambda p: (not p.typed, -_w(p), p.surface))
    if ke_ctx and ke_bm:
        ra_ = [PeerHit(tid=p.tid, surface=p.surface, weight=_w(p),
                       best_d=p.best_d, typed=p.typed) for p in ra_]
    return ra_[:topk]


def shared_props_from_peers(
    state: GraphState,
    index: HubSpreadIndex,
    peer_ids: list[int],
    *,
    prefix: str = "",
    min_support: int = 2,
    topk: int = 20,
    gidf_min: float = 4.0,
) -> list[PropHit]:
    """Thuộc_tính chung của peers (lọc prefix surface, vd ``nghịch``)."""
    vocab = state.vocab
    pref = prefix.strip().lower()
    support: dict[int, set[str]] = {}
    for pid in peer_ids:
        # peer có thể là id ảo CỤM (chủ ngữ dual-write) → tra qua index
        psurf = _surf_ix(index, vocab, pid)
        for dst, (_w, _d) in index.fwd.get(int(pid), {}).items():
            surf = _surf(vocab, dst)
            if not surf or len(surf) <= 1:
                continue
            if surf in _FUNC or surf.lower() in _FUNC:
                continue
            if pref and not surf.lower().startswith(pref):
                continue
            if _gidf(state, dst) < float(gidf_min):
                continue
            support.setdefault(dst, set()).add(psurf)
    hits: list[PropHit] = []
    for tid, peers in support.items():
        if len(peers) < int(min_support):
            continue
        hits.append(
            PropHit(
                tid=int(tid),
                surface=_surf(vocab, tid),
                support=len(peers),
                peer_surfaces=sorted(peers)[:8],
            )
        )
    hits.sort(key=lambda h: (-h.support, h.surface))
    return hits[:topk]


def spread_analogy(
    state: GraphState,
    index: HubSpreadIndex,
    seed_surface: str,
    *,
    prop_prefix: str = "",
    hub_topk: int = 8,
    peer_topk: int = 20,
    gidf_min: float = 4.0,
) -> dict[str, object]:
    """Một lượt: seed → hubs → peers → props."""
    seeds = resolve_surface_ids(state.vocab, seed_surface)
    raw = score_hubs_raw_compare(state, index, seeds, topk=hub_topk)
    hubs = score_hubs_from(
        state, index, seeds, topk=hub_topk, gidf_min=gidf_min, ban_func=True,
    )
    hub_ids = [h.tid for h in hubs[:3]]
    peers = reverse_peers(
        state, index, hub_ids, exclude=set(seeds), topk=peer_topk,
    )
    props = shared_props_from_peers(
        state, index, [p.tid for p in peers],
        prefix=prop_prefix, min_support=2, gidf_min=gidf_min,
    ) if prop_prefix or peers else []
    return {
        "seed": seed_surface,
        "seed_ids": seeds,
        "hubs_raw_by_weight": [
            {"Y": h.surface, "w": h.weight, "gidf": round(h.gidf, 2), "div": h.diversity, "d": h.best_d}
            for h in raw
        ],
        "hubs_scored": [
            {
                "Y": h.surface, "score": round(h.score, 2),
                "w": h.weight, "gidf": round(h.gidf, 2),
                "div": h.diversity, "d": h.best_d,
            }
            for h in hubs
        ],
        "peers": [
            {"X": p.surface, "w": p.weight, "d": p.best_d} for p in peers
        ],
        "props": [
            {
                "Z": p.surface, "support": p.support,
                "via": p.peer_surfaces,
            }
            for p in props
        ],
    }
