"""
Lõi sinh v3 — nơ-ron cha + pattern completion (docx §1), KHÔNG dạy ngữ pháp.

Suy luận:
  1. Lan xuôi: prompt (phần câu hỏi trước `Đáp:`) → chọn **nơ-ron cha** khớp nhiều con nhất
     (IDF + khớp mờ E cho paraphrase) — `ParentStore.best_match`.
  2. Lan ngược (pattern completion): dội ra phần con **còn thiếu** của cha = đáp / cả câu phát
     biểu (case "con kiến") — `ParentStore.completion`.
  3. Fallback (không cha nào khớp đủ): sinh autoregressive bằng W_seq + readout (mượt văn phong).

Chỉ dùng cạnh/cha học từ **dữ liệu thật** + frame `Đáp:` (được phép). Không Rule/Var/teaching.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any

import numpy as np

# Unit tests (conftest) set True → ép `ar_w_skip_primary=False` để giữ route matrix/parent.
_FORCE_MATRIX_FOR_TESTS: bool = False

# §TK_056 — hook kNN-LM. Nếu đặt (KnnLM.blend, lam), mỗi bước vote sẽ TRỘN phân
# bố next-token từ datastore encoder vào tập ứng viên sống. Mặc định None ⇒
# đường sinh chạy y như cũ. Đặt qua `set_knn_hook`.
_KNN_HOOK = None  # tuple(knn_lm, mode, params) | None


def set_knn_hook(knn_lm, lam: float = 0.35, widen_k: int = 0,
                 hybrid: bool = False, **params) -> None:
    """Bật/tắt can thiệp causal-encoder cho đường sinh. `knn_lm=None` để tắt.
    `hybrid=True` (§TK_061) bật SINH LAI cổng-độ-tin (causal-LM dẫn khi W_skip
    phân vân/cạn); kwargs truyền vào `hybrid_blend` (conf_lo/conf_hi/w_max/…).
    Ngược lại là kNN reorder (§TK_056); `widen_k>0` bật nới (opt-in)."""
    global _KNN_HOOK
    if knn_lm is None:
        _KNN_HOOK = None
    elif params.pop("rerank", False):
        _KNN_HOOK = (knn_lm, "rerank", {"a": float(params.pop("rerank_a", 0.6))})
    elif hybrid:
        _KNN_HOOK = (knn_lm, "hybrid", params)
    else:
        _KNN_HOOK = (knn_lm, "knn", {"lam": float(lam), "widen_k": int(widen_k)})


# §TK_083 (Pha 1c) — NEO MỨC-PROMPT: bias ứng viên về centroid embedding của
# PROMPT (chủ đề CỐ ĐỊNH). Khác kNN (bám h HIỆN TẠI đã trôi → củng cố trôi).
_ANCHOR = None  # (E_norm (V,d) float32, beta) | None


def set_anchor_hook(emb_matrix=None, beta: float = 6.0) -> None:
    """Bật/tắt neo mức-prompt. `emb_matrix=None` để tắt. `beta` = độ mạnh kéo về
    chủ đề prompt (exp(beta·cos) trên ứng viên sống)."""
    global _ANCHOR
    if emb_matrix is None:
        _ANCHOR = None
        return
    E = np.asarray(emb_matrix, dtype=np.float32)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    _ANCHOR = (En, float(beta))


def _apply_prompt_anchor(scores, out, prompt_len):
    """out[t] *= exp(beta·cos(E[t], centroid_prompt)) trên ứng viên SỐNG. Không
    hồi sinh token chết; centroid = mean embedding token prompt (chủ đề cố định)."""
    if _ANCHOR is None or prompt_len <= 0:
        return scores
    En, beta = _ANCHOR
    live = np.isfinite(scores) & (scores > 0)
    if not live.any():
        return scores
    V = En.shape[0]
    ptok = [int(t) for t in out[:prompt_len] if 0 <= int(t) < V]
    if not ptok:
        return scores
    tv = En[ptok].mean(0)
    nrm = float(np.linalg.norm(tv))
    if nrm < 1e-9:
        return scores
    tv = tv / nrm
    idx = np.where(live)[0]
    sim = En[idx] @ tv                       # cosine ứng viên vs chủ đề prompt
    out2 = scores.copy()
    out2[idx] = scores[idx] * np.exp(beta * sim)
    return out2

# §13.22 — DẠNG ĐÁP cho frame `Hỏi:…Đáp:` phụ thuộc ĐƯỜNG DECODE, không phải sở thích:
#   W_skip primary (production)  → đáp NGẮN  "Thăng Long."      — agenda phải chặt
#   matrix/parent route          → đáp ĐẦY ĐỦ "Thủ đô … là Hà Nội." — dội agenda trực tiếp
#
# Đo A/B trên mini (`Hỏi: Nhà Lý dời đô về đâu? Đáp:`):
#   cắt tiền tố BẬT  → 'Thăng Long.'                       ← §12.149, đúng
#   cắt tiền tố TẮT  → 'Đại La và là Năm tên Long kinh…'    ← salad
# Vòng phiếu W_skip phủ agenda dài rất tệ; còn route matrix thì đáp cụt mất chủ ngữ
# (`Cá voi là gì` → `là động vật…`), đó là 6 test đỏ từ `a5f4862`.
#
# ⇒ Điều kiện đúng là **route**, không phải cờ toàn cục và tuyệt đối không phải "đang test".
# `conftest` ép `ar_w_skip_primary=False` vì unit test khẳng định hợp đồng route matrix
# (docstring của chính conftest) — nên hai bên khớp nhau một cách nhất quán.
def _qa_trim_prefix(config) -> bool:
    return bool(getattr(config, "ar_w_skip_primary", True))

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.frame_bracket_ref import decode_endpoint_label
from nn_aivie.core.parent_store import ParentStore, _content_ids
from nn_aivie.core.seg_graph import SlotRef
from nn_aivie.core.state import GraphState
from nn_aivie.hca.sent_intra import _bound_of  # §13.56 — biên đơn vị HỌC ĐƯỢC
from nn_aivie.core.tokenizer import MARK_SPACE
from nn_aivie.core.vocab import SPECIAL_EOS, is_surface_token
from nn_aivie.hca.objcache import cache_get as _oc_get, cache_put as _oc_put
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.frame_bracket_gate import (
    apply_frame_bracket_gate,
    ensure_frame_brackets,
    seed_open_brackets,
    update_open_brackets,
)
from nn_aivie.hca.skip_span_gate import apply_skip_span_gate, ensure_skip_spans
from nn_aivie.hca.ar_candidates import (
    assemble_ar_scores,
    build_micro_dict,
    collect_ar_candidates,
    macro_sparse,
)
from nn_aivie.hca.readout import (
    apply_repetition_penalty,
    apply_repetition_penalty_inplace,
    mask_non_surface,
    mask_non_surface_inplace,
    multiply_scores_by_boost,
    spreading_activation_read,
)
from nn_aivie.hca.style_readout import style_boost_from_coact
from nn_aivie.hca.style_inventory import style_inject_plan
from nn_aivie.hca.style_fluency import (
    apply_soft_style_decode,
    pick_style_by_cosine,
    token_cosine,
)
from nn_aivie.hca.verify_v3 import apply_g4_verify_v3


def _split_frame(generated: list[int], vocab) -> tuple[list[int], list[int]]:
    """Tách tại `Đáp:` cuối: (q_ids = câu hỏi, answer_so_far)."""
    dap_id = vocab._token_to_id.get(DAP_LABEL)
    if dap_id is None or dap_id not in generated:
        return generated, []
    idx = len(generated) - 1 - generated[::-1].index(dap_id)
    return generated[:idx], generated[idx + 1 :]


def _completion_from_parent(ps: ParentStore, parent_idx: int) -> tuple[int, ...]:
    """TK_013 — QA dùng answer_ids; statement có thể expand slot_chain tail."""
    rec = ps.parents[parent_idx]
    if rec.is_qa and rec.answer_ids:
        return rec.answer_ids
    tail = ps.completion_tail_slots(parent_idx)
    if tail and ps.seg.clusters:
        out: list[int] = []
        for s in tail:
            out.extend(ps.seg.expand_slot(s))
        if out:
            return tuple(out)
    return ps.completion(parent_idx)


def _chi_muc_chu_dinh_nghia(state: GraphState) -> dict[int, int]:
    """token → cha có văn bản BẮT ĐẦU bằng «<token> là». Dựng 1 lần/state."""
    hit = _oc_get(_CHU_DN_CACHE, state)
    if hit is not None:
        return hit
    ps, v = state.parents, state.vocab
    la = {i for i in range(len(v))
          if v.token_of(i).lstrip(MARK_SPACE).strip().lower() == "là"}
    # token đầu → DANH SÁCH cha, để còn so được CẢ CỤM chứ không chỉ token đầu.
    ra: dict[int, list[int]] = {}
    if ps is not None:
        for idx, rec in enumerate(ps.parents):
            e = rec.emit_ids
            if len(e) >= 3 and any(int(x) in la for x in e[1:4]):
                ra.setdefault(int(e[0]), []).append(idx)
    return _oc_put(_CHU_DN_CACHE, state, ra)


def _cum_co_muc(state: GraphState, cum: list[int]) -> bool:
    """Có cha nào bắt đầu ĐÚNG bằng cả cụm `cum` (rồi tới `là`) không?

    ⚠️ Bản đầu chỉ kiểm TOKEN ĐẦU có mục định nghĩa hay không — mà `con`, `chim`,
    `cây` đều CÓ («con là người hoặc động vật thuộc thế hệ sau…»), nên guard luôn
    trả «đã tra được» và bản vá KHÔNG BAO GIỜ CHẠY. Phải so CẢ CỤM.
    """
    if not cum:
        return False
    chi = _chi_muc_chu_dinh_nghia(state)
    v = state.vocab
    for idx in chi.get(int(cum[0]), ()):
        e = list(state.parents.parents[idx].emit_ids)
        if len(e) > len(cum) and all(
            v.token_of(int(a)).lstrip(MARK_SPACE).strip().lower()
            == v.token_of(int(b)).lstrip(MARK_SPACE).strip().lower()
                for a, b in zip(e, cum)):
            return True
    return False


def _lui_ve_tu_chinh(state: GraphState, query: list[int], config) -> list[int] | None:
    """§99 — cụm chủ ngữ không có mục định nghĩa ⇒ LÙI VỀ TỪ CHÍNH.

    §98 đo được tương quan 10/10: sinh đúng KHI VÀ CHỈ KHI cụm chủ ngữ là một mục
    có sẵn trong corpus định nghĩa. `chó là` ✓ · `xe đạp là` ✓ · nhưng
    `con chó là` ✗ · `chim sẻ là` ✗ · `chó nhà là` ✗ — vì hệ KHÔNG lùi về từ chính.

    KHÔNG dùng danh sách loại từ (Luật 0). Thay vào đó: thử MỌI từ của cụm, giữ
    từ nào thật sự CÓ mục định nghĩa, rồi chọn từ ĐẶC TRƯNG NHẤT theo `_gidf` đã
    gộp ranh giới (§97). Loại từ tự rụng vì chúng chung chung:

        con chó   `con` gidf 4,4 · `chó` 6,8  ⇒ chọn `chó`
        cái bàn   `cái` 4,0      · `bàn` 5,9  ⇒ chọn `bàn`
        chim sẻ   `chim`         · `sẻ` hiếm hơn ⇒ chọn `sẻ`  (đúng con chim sẻ)

    Trả None nếu cụm ĐÃ có mục định nghĩa (không lùi) hoặc không lùi được.
    """
    ps = state.parents
    if ps is None or len(query) < 2:
        return None
    chi = _chi_muc_chu_dinh_nghia(state)
    v0 = state.vocab
    _la_i = next((i for i, x in enumerate(query)
                  if v0.token_of(int(x)).lstrip(MARK_SPACE).strip().lower() == "là"),
                 len(query))
    if _cum_co_muc(state, list(query[:_la_i])):
        return None                       # CẢ CỤM đã tra được — không đụng vào
    v = state.vocab
    ung: list[tuple[float, int]] = []
    for t in query:
        su = v.token_of(int(t)).lstrip(MARK_SPACE).strip()
        if not su or su.lower() == "là":
            continue
        for cand in {int(t), v.id_of(su)}:
            if cand in chi:
                ung.append((ps._gidf(cand), cand))   # noqa: SLF001
                break
    if not ung:
        return None
    tot = max(ung)[1]                     # ĐẶC TRƯNG NHẤT
    la = next((int(x) for x in query
               if v.token_of(int(x)).lstrip(MARK_SPACE).strip().lower() == "là"), None)
    return [tot] + ([la] if la is not None else [])


def _kenh_buoc(config, buoc: int) -> str:
    """§103 — kênh bắc cầu RIÊNG cho từng bước.

    §102.3: nỗi lo §59 (cooc kéo về BẠN GHÉP TỪ: `khôn` → `khôn ngoan`) là về
    HẠT GIỐNG ở bước ①. Cách phát biểu «a,b,c cùng đứng với x» nói về CẦU NỐI ở
    bước ②. Cờ `hub_chain_channel` áp cho CẢ HAI nên §101/§102 không tách được:
    nếu ② có lợi mà ① có hại thì chúng triệt tiêu — đúng hình dạng số liệu thấy
    được (mọi kênh xấp xỉ đối chứng).

    `hub_chain_channel_1` / `_2` ghi đè; không đặt thì rơi về `hub_chain_channel`.
    """
    rieng = getattr(config, f"hub_chain_channel_{buoc}", "") or ""
    return str(rieng or getattr(config, "hub_chain_channel", "typed_first"))


def _bien_the_hoa_thuong(vocab, query: list[int]) -> list[int] | None:
    """§86 — cùng truy vấn, lật hoa/thường CHỮ ĐẦU; None nếu không có biến thể.

    `hổ` và `Hổ` là HAI token khác nhau (§85.3). Từ điển viết thường, Wikipedia
    viết hoa, prompt người dùng viết thường ⇒ nửa kho không với tới được:
    «hổ là» cho cha đúng hạng 18, «Hổ là» cho hạng 7.
    """
    if not query:
        return None
    s = vocab.detokenize(list(query))
    i = next((k for k, c in enumerate(s) if c.isalpha()), None)
    if i is None:
        return None
    lat = (s[:i] + (s[i].lower() if s[i].isupper() else s[i].upper())
           + s[i + 1:])
    if lat == s:
        return None
    try:
        ra = [int(t) for t in vocab.encode_prompt(lat)]
    except (AttributeError, ValueError):
        return None
    if _tv_cho and _bang_tv_w > 0.0:
        _nen = max(ra.values()) if ra else 0.0
        if _nen > 0:
            for _t3 in _tv_cho:
                ra[_t3] = max(ra.get(_t3, 0.0), _bang_tv_w * _nen)

    return ra if ra and ra != list(query) else None


def _rank_matches_gop(ps, query, vocab, embedding=None, **kw) -> list[tuple[int, float]]:
    """§86 — `rank_matches` NỞ theo hoa/thường rồi trộn theo điểm CAO NHẤT.

    Không đụng bộ xếp hạng, không dựng lại chỉ mục. Đo được (§86.3, 14 prompt
    định nghĩa): hạng trung vị cha đúng 4 → 2, `hổ là` 18 → 2; chiều ngược
    (danh từ riêng) KHÔNG tụt hạng nào.
    """
    goc = ps.rank_matches(query, vocab, embedding, **kw)
    bt = _bien_the_hoa_thuong(vocab, list(query))
    if bt is None:
        return goc
    diem: dict[int, float] = {}
    for pi, sc in list(goc) + list(ps.rank_matches(bt, vocab, embedding, **kw)):
        pi = int(pi)
        if float(sc) > diem.get(pi, -1e18):
            diem[pi] = float(sc)
    top_n = int(kw.get("top_n", 8) or 8)
    return sorted(diem.items(), key=lambda kv: -kv[1])[:top_n]


def _active_parent_set(state: GraphState, query_ids: list[int], *, top_n: int = 20,
                       config=None) -> frozenset[int]:
    """TK_020 ctx-resonance — tập cha ĐANG KÍCH HOẠT bởi prompt (tính 1 lần/lượt sinh).

    Cha kích hoạt dội ngược làm mạnh cạnh ngang có provenance trong nó (ý tác giả 07-06)."""
    ps = state.parents
    if ps is None:
        return frozenset()
    # §87 — gộp hoa/thường PHẢI áp ở ĐÂY nữa. Bản §86 chỉ nối vào chỗ gọi trong
    # `generate()` (dòng ~4164) và đo được ĐÚNG BẰNG 0 thay đổi ở đầu ra: 78,9%
    # → 78,9%, bốn ca vàng giống hệt từng byte. Vì tập cha đang hoạt — thứ kênh
    # bậc-2 thật sự đọc — lấy qua hàm này, gọi `rank_matches` TRẦN.
    # Lần thứ NĂM cùng kiểu lỗi trong phiên (LOI1, §79, §81, §83, nay §87), và
    # lần này do chính bản vá §86 gây ra.
    _gop = bool(getattr(config, "retrieval_gop_hoa_thuong", False))
    try:
        if _gop:
            ranked = _rank_matches_gop(
                ps, query_ids, state.vocab, None, fuzzy_topk=0, top_n=top_n)
        else:
            ranked = ps.rank_matches(
                query_ids, state.vocab, None, fuzzy_topk=0, top_n=top_n)
    except (AttributeError, TypeError, ValueError):
        return frozenset()
    return frozenset(pi for pi, _s in ranked)


def _w_skip_primary_loop(
    state: GraphState,
    out: list[int],
    *,
    prompt_len: int,
    eos_id: int,
    max_steps: int,
    config: HCAConfig,
    rng: np.random.Generator,
    open_brackets: list[tuple[int, int]],
    agenda_ids: list[int] | None = None,
    entity_ids: tuple[int, ...] = (),
) -> list[int]:
    """TK_045 §12.131/133 — W_skip primary + ưu tiên agenda MZ (+ chống salad 132)."""
    from dataclasses import replace

    from nn_aivie.hca.matrix_core import inner_digit_ids
    from nn_aivie.hca.context_positional import is_structural_token
    from nn_aivie.hca.w_skip import (
        apply_local_repeat_penalty,
        blocked_dst_from_hist,
        content_channel_gate,
        count_boundaries,
        d1_fluency_row,
        ensure_same_sent_index,
        eos_from_end_after,
        expire_w_skip_consume,
        hist_after_last_boundary,
        mark_w_skip_consume,
        same_sentence_bonus,
        sent_open_from_w_agenda,
        sent_open_from_w_skip,
        vote_logits,
    )

    n = len(state.vocab)
    # CHUỖI BA BƯỚC — tính MỘT LẦN từ prompt (dựng index tốn ~11s trên ckpt
    # thật, không được làm mỗi bước). Chuẩn hoá về [0,1] để một đề cử nặng
    # không nuốt phân bố (§78.1: hệ số thô chạy 5×…923× trong cùng prompt).
    _chain_b = float(getattr(config, "hub_chain_boost", 0.0) or 0.0)
    _chain_w: dict[int, float] = {}
    if _chain_b > 0:
        _cw = hub_chain_proposals(state, list(out), config)
        _mx = max(_cw.values()) if _cw else 0.0
        if _mx > 0:
            _chain_w = {int(t): float(w) / float(_mx) for t, w in _cw.items()}
    # §13.54 — biên ĐƠN VỊ học được (EOS-lift trên W_seq), thay danh sách dấu câu cứng.
    # Dựng một lần cho cả vòng sinh; `ensure_boundary_signal` có cache weakref theo state.
    from nn_aivie.hca.sent_intra import _bound_of

    # §139 — bigram câu cha, dựng một lần cho cả vòng sinh (có cache trong store).
    _bi_idx: dict = {}
    if bool(getattr(config, "ar_w_skip_chain_vi_tri", False)) and state.parents is not None:
        try:
            state.parents._ensure_seq_index()   # noqa: SLF001
            _bi_idx = state.parents._seq_bi     # noqa: SLF001
        except AttributeError:
            _bi_idx = {}
    # §140 — token do CHUỖI đề cử đã phát trong câu này (một-lần-mỗi-từ).
    _da_chuoi: set[int] = set()
    # §165 — nạp cạnh REINFORCED (loại suy đã XÁC NHẬN, TK_054 §161) MỘT LẦN.
    # §143: đường sinh đọc ps.parents, KHÔNG đọc adjacency ⇒ phải tiêm tường
    # minh. Chỉ nạp khi beta>0 (mặc định 0 = tắt).
    _rf_beta = float(getattr(config, "ar_w_skip_reinforced_beta", 0.0) or 0.0)
    _rf_conf = float(getattr(config, "ar_w_skip_reinforced_conf", 0.2) or 0.2)
    _rf_edges: dict = {}
    if _rf_beta > 0.0 and getattr(state, "adjacency", None) is not None:
        from nn_aivie.core.edge_kinds import EdgeKind as _EK
        _lay = state.adjacency._layers.get(_EK.REINFORCED)  # noqa: SLF001
        if _lay is not None:
            _tmp: dict = {}
            for _i, _j, _w in _lay.items():
                _tmp.setdefault(int(_i), []).append((int(_j), float(_w)))
            _rf_edges = _tmp
    _bd_gen = _bound_of(state)
    T = float(getattr(config, "ar_w_skip_primary_T", 0.5))
    d_max = int(getattr(config, "ar_matrix_w_skip_decode_dmax", 16))
    lam = float(getattr(config, "ar_matrix_w_skip_decode_lam", 0.5))
    # 0 = full CSR row (§12.136); >0 = cắt topk (kênh phụ cũ dùng 12)
    topk = int(getattr(config, "ar_w_skip_vote_topk", 0))
    vote_agg = str(getattr(config, "ar_w_skip_vote_agg", "prod") or "prod")
    prod_window = int(getattr(config, "ar_w_skip_prod_window", 0) or 0)
    oja_row = bool(getattr(config, "ar_w_skip_oja_row", True))
    oja_deg = float(getattr(config, "ar_w_skip_oja_degree_beta", 1.0) or 0.0)
    oja_comp = float(getattr(config, "ar_w_skip_oja_compete", 0.0) or 0.0)
    oja_indeg = float(getattr(config, "ar_w_skip_oja_indeg_beta", 0.75) or 0.0)
    beta = float(getattr(config, "ar_w_skip_same_sent_beta", 1.0))
    flu_w = float(getattr(config, "ar_w_skip_fluency_weight", 1.0))
    flu_floor = float(getattr(config, "ar_w_skip_fluency_floor", 0.0))
    flu_topk = int(getattr(config, "ar_w_skip_fluency_topk", 64))
    multi_floor = float(getattr(config, "ar_w_skip_multivote_gate_floor", 0.35))
    ag_boost = float(getattr(config, "ar_w_skip_agenda_boost", 2.5))
    ag_win = int(getattr(config, "ar_w_skip_agenda_order_window", 4))
    ag_decay = float(getattr(config, "ar_w_skip_agenda_order_decay", 0.6))
    ag_strict = bool(getattr(config, "ar_w_skip_agenda_strict", True))
    nonag_scale = float(getattr(config, "ar_w_skip_nonagenda_scale", 0.05))
    # §151 — ngưỡng ĐỘ TIN tối thiểu của bước sinh; dưới ngưỡng ⇒ dừng.
    _do_tin_min = float(getattr(config, "ar_w_skip_do_tin_min", 0.0) or 0.0)
    # §83 — tín hiệu thứ tự bậc-2 của cha; 0 = TẮT (mặc định lúc thêm, chờ đo)
    _o2_w = float(getattr(config, "ar_w_skip_order2_weight", 0.0) or 0.0)
    _o2_min = float(getattr(config, "ar_seq_order2_min", 2.0) or 2.0)
    # §84 — LẤY TỪ ĐÂU. `seq_next` gộp trên CẢ 450 050 cha nên bị chuỗi phổ
    # thông thống trị: §83.3 đo được nó kéo `chó là gia súc thuộc nhóm ăn thịt`
    # thành `chó là gia đình và bạn bè` — trôi chảy hoàn hảo, tri thức bằng 0.
    # Bản cục bộ chỉ đọc chuỗi của CHA ĐANG HOẠT (`_active_parent_set`, tức
    # đúng câu nguồn mà prompt đã truy ra), nên trật tự không đánh đổi tri thức.
    _o2_cha_hoat = bool(getattr(config, "ar_w_skip_order2_cha_hoat", False))
    _o2_top_n = int(getattr(config, "ar_w_skip_order2_top_n", 20) or 20)
    _o2_bi: dict[int, dict[int, float]] = {}
    _o2_tri: dict[tuple[int, int], dict[int, float]] = {}
    if _o2_w > 0 and _o2_cha_hoat and state.parents is not None:
        _pa = _active_parent_set(state, list(out[:prompt_len]), top_n=_o2_top_n,
                                 config=config)
        for _pi in _pa:
            try:
                _rec = state.parents.parents[int(_pi)]
            except (IndexError, TypeError):
                continue
            _sq = (_rec.answer_ids
                   if (getattr(_rec, "is_qa", False) and _rec.answer_ids)
                   else _rec.emit_ids)
            _pw = float(max(1, getattr(_rec, "power", 1)))
            for _i in range(len(_sq) - 1):
                _o2_bi.setdefault(int(_sq[_i]), {})
                _o2_bi[int(_sq[_i])][int(_sq[_i + 1])] = (
                    _o2_bi[int(_sq[_i])].get(int(_sq[_i + 1]), 0.0) + _pw)
            for _i in range(len(_sq) - 2):
                _k = (int(_sq[_i]), int(_sq[_i + 1]))
                _o2_tri.setdefault(_k, {})
                _o2_tri[_k][int(_sq[_i + 2])] = (
                    _o2_tri[_k].get(int(_sq[_i + 2]), 0.0) + _pw)
    consume_edges = bool(getattr(config, "ar_w_skip_consume_edges", True))
    consume_tokens = bool(getattr(config, "ar_w_skip_consume_tokens", False))
    consume_gidf = float(getattr(config, "ar_w_skip_consume_gidf_min", 4.0) or 0.0)
    consume_edge_scale = float(getattr(config, "ar_w_skip_consume_edge_scale", 0.0) or 0.0)
    consume_rev_win = int(getattr(config, "ar_w_skip_consume_reverse_window", 4) or 0)
    consume_recover = int(getattr(config, "ar_w_skip_consume_recover_steps", 8) or 0)
    local_rep_win = int(getattr(config, "ar_w_skip_local_repeat_window", 8) or 0)
    local_rep_scale = float(getattr(config, "ar_w_skip_local_repeat_scale", 0.15) or 1.0)
    local_rep_min = int(getattr(config, "ar_w_skip_local_repeat_min_count", 1) or 1)
    local_rep_near_k = int(getattr(config, "ar_w_skip_local_repeat_near_k", 3) or 0)
    local_rep_near_sc = float(getattr(config, "ar_w_skip_local_repeat_near_scale", 0.08) or 1.0)
    local_rep_out_pref = float(
        getattr(config, "ar_w_skip_local_repeat_outsider_prefer", 0.05) or 1.0
    )
    eos_ea_beta = float(getattr(config, "ar_w_skip_eos_end_after_beta", 2.0) or 0.0)
    sent_open_beta = float(getattr(config, "ar_w_skip_sent_open_beta", 1.0) or 0.0)
    sent_open_wskip_beta = float(
        getattr(config, "ar_w_skip_sent_open_wskip_beta", 2.5) or 0.0
    )
    content_gate_on = bool(getattr(config, "ar_w_skip_content_gate", True))
    content_gate_floor = float(getattr(config, "ar_w_skip_content_gate_floor", 0.15) or 0.0)
    vote_pre_b_k = int(getattr(config, "ar_w_skip_vote_pre_boundary_k", 2) or 0)
    episode_eos_boost = float(getattr(config, "ar_w_skip_episode_eos_boost", 2.2) or 1.0)
    post_bound_dampen = float(getattr(config, "ar_w_skip_post_boundary_dampen", 0.08) or 1.0)
    # Đứng trên biên sau khi đã sinh ≥1 câu: EOS cạnh tranh mạnh (tránh salad câu 2).
    boundary_eos_beta = float(getattr(config, "ar_w_skip_boundary_eos_beta", 2.5) or 0.0)
    cfg_s = replace(config, hybrid_temperature=max(T, 1e-6), ar_sharp_gate=False)

    agenda_seq = [int(t) for t in (agenda_ids or []) if 0 <= int(t) < n]
    agenda_cap = max(32, max_steps * 4)
    if len(agenda_seq) > agenda_cap:
        agenda_seq = agenda_seq[:agenda_cap]
    consumed = [False] * len(agenda_seq)
    ent_set = {int(t) for t in entity_ids if 0 <= int(t) < n}
    # §12.144/145 — consume cạnh + tuổi để hồi; không chặn token
    blocked_edges: set[tuple[int, int, int]] = set()
    edge_age: dict[tuple[int, int, int], int] = {}
    gen_step = 0

    tok2par = par_cont = struct = None
    if beta > 0:
        tok2par, par_cont, struct = ensure_same_sent_index(state)

    for _ in range(max_steps):
        if consume_edges and consume_recover > 0:
            hold_dst: set[int] = set()
            if local_rep_win > 0:
                for t in out[prompt_len:][-local_rep_win:]:
                    tid = int(t)
                    if (
                        0 <= tid < n
                        and state.vocab.is_surface(tid)
                        and not is_structural_token(state.vocab, tid)
                    ):
                        hold_dst.add(tid)
            expire_w_skip_consume(
                blocked_edges, edge_age, step=gen_step, recover_steps=consume_recover,
                hold_dst=hold_dst,
            )
        blocked_dst = (
            blocked_dst_from_hist(
                state, out, prompt_len=prompt_len, gidf_min=consume_gidf,
            )
            if consume_tokens
            else set()
        )
        # §12.141 — không để hist câu cũ (`là`) vote sau biên
        vote_hist = hist_after_last_boundary(
            list(out), state.vocab, pre_boundary_k=vote_pre_b_k, bound=_bd_gen,
        )
        if not vote_hist:
            vote_hist = list(out)
        scores = vote_logits(
            state, list(vote_hist), n_vocab=n, d_max=d_max, lam=lam, topk=topk,
            agg=vote_agg, prod_window=prod_window,
            oja_row=oja_row, oja_degree_beta=oja_deg, oja_compete_beta=oja_comp,
            oja_indeg_beta=oja_indeg,
            blocked_edges=blocked_edges if consume_edges else None,
            blocked_dst=blocked_dst if blocked_dst else None,
            blocked_edge_scale=consume_edge_scale,
        )
        vote_raw = scores  # sum: vote>1 ⇒ multi; prod: >0 ⇒ sống qua giao
        if beta > 0 and tok2par is not None:
            # Chỉ cửa sổ gần — không quét cả prompt (tránh recirculate `Năm` đầu câu)
            hc = [
                int(t) for t in vote_hist[-12:]
                if 0 <= int(t) < n
                and not bool(struct[int(t)])
                and state.vocab.is_surface(int(t))
            ]
            bonus = same_sentence_bonus(hc, tok2par, par_cont, n, window=4, top_m=8)
            # §12.137 prod: bonus chỉ GIA CƯỜNG token đã sống qua Hadamard —
            # không được mở lại hub ngoài giao (ông/của/ngoại).
            if str(vote_agg).lower() in {"prod", "product", "hadamard", "mul"}:
                bonus = bonus * (vote_raw > 0.0)
            # §12.145 — chỉ tắt same-sent cho token trong cửa sổ local-repeat
            # (lặp xa / lần hai `đô` vẫn có thể được túi nâng nếu còn trong vote).
            if consume_edges or local_rep_win > 0:
                zw = local_rep_win if local_rep_win > 0 else 8
                recent = out[prompt_len:][-zw:]
                for t in recent:
                    if 0 <= int(t) < n:
                        bonus[int(t)] = 0.0
            if blocked_dst:
                for bi in blocked_dst:
                    if 0 <= int(bi) < n:
                        bonus[int(bi)] = 0.0
            rv = float(scores.max()) or 1.0
            rb = float(bonus.max()) or 1.0
            scores = scores + beta * rv * (bonus / rb)
        if flu_w > 0 and out:
            # Sau biên: flu từ token nội dung trước biên (`.` nnz=0 trên W_skip)
            from nn_aivie.hca.sent_intra import _is_boundary as _is_b_flu

            flu_src = int(out[-1])
            if _is_b_flu(state.vocab, flu_src, _bd_gen):
                vh = hist_after_last_boundary(
                    list(out), state.vocab, pre_boundary_k=1, bound=_bd_gen,
                )
                if vh:
                    flu_src = int(vh[-1])
            flu = d1_fluency_row(
                state, flu_src, n, topk=flu_topk, d_max=d_max,
                blocked_edges=blocked_edges if consume_edges else None,
                blocked_edge_scale=consume_edge_scale,
            )
            fm = float(flu.max())
            if fm > 0:
                # floor=0: diệt token không kề W_skip[d=1] — NHƯNG bảo vệ multi-d vote
                # (§12.136: quyết có phiếu từ Lý/Thái/Tổ/vua nhưng flu d=1 yếu/rank67)
                gate = flu_floor + flu_w * (flu / fm)
                if multi_floor > 0:
                    # sum: vote>1 ⇒ ≥2 proposer; prod: score>0 ⇒ sống qua giao Hadamard
                    if str(vote_agg).lower() in {"prod", "product", "hadamard", "mul"}:
                        multi = vote_raw > 0.0
                    else:
                        multi = vote_raw > 1.0 + 1e-9
                    gate = np.where(multi, np.maximum(gate, multi_floor), gate)
                scores = scores * gate
        # §12.145 — phạt mềm túi xáo cửa sổ gần (cho phép lặp token ở xa)
        if local_rep_win > 0 and (
            local_rep_scale < 1.0
            or local_rep_near_sc < 1.0
            or local_rep_out_pref < 1.0
        ):
            scores = apply_local_repeat_penalty(
                scores, out, prompt_len=prompt_len, n_vocab=n, vocab=state.vocab,
                window=local_rep_win, scale=local_rep_scale, min_count=local_rep_min,
                near_k=local_rep_near_k, near_scale=local_rep_near_sc,
                outsider_prefer=local_rep_out_pref,
            )
        # §12.146/P43 — dual-channel: surface ⊙ content (agenda∪entity∪prompt)
        if content_gate_on:
            cg = content_channel_gate(
                n,
                agenda_ids=agenda_seq,
                entity_ids=ent_set,
                prompt_ids=list(out[:prompt_len]),
                floor=content_gate_floor,
                vocab=state.vocab,
            )
            scores = scores * cg
        # §12.141/P42 — sau biên: boost SENT_SUCC; hạ token thuộc câu ĐÃ ĐÓNG
        if out:
            from nn_aivie.hca.sent_intra import _is_boundary

            if _is_boundary(state.vocab, int(out[-1]), _bound_of(state)):
                if post_bound_dampen < 1.0:
                    last_b = len(out) - 1
                    for t in out[:last_b]:
                        tid = int(t)
                        if 0 <= tid < n:
                            scores[tid] *= post_bound_dampen
                rv = (
                    float(np.nanmax(scores[np.isfinite(scores)]))
                    if np.isfinite(scores).any()
                    else 1.0
                )
                if not np.isfinite(rv) or rv <= 0:
                    rv = 1.0
                if sent_open_wskip_beta > 0:
                    open_ws = sent_open_from_w_skip(
                        state, list(out), n, beta=sent_open_wskip_beta,
                        d_max=min(4, d_max), lam=lam,
                    )
                    if float(open_ws.max()) > 0:
                        scores = scores + open_ws * rv
                if sent_open_beta > 0:
                    open_b = sent_open_from_w_agenda(
                        state, list(out), n, beta=sent_open_beta,
                    )
                    if float(open_b.max()) > 0:
                        scores = scores + open_b * rv
        # §116 — mặc định MỖI BƯỚC: khối agenda dưới có thể không chạy
        # (`ag_boost <= 0`), mà chỗ dìm EOS vẫn đọc `_ag_tin` ⇒ phải khai ở đây.
        _ag_tin = True
        # §12.133 — ưu tiên token agenda MZ (cửa sổ thứ tự + entity)
        if agenda_seq and ag_boost > 0:
            out_tail = out[prompt_len:]
            from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL, MARK_SPACE

            def _surf_key(tid: int) -> str:
                s = state.vocab.token_of(int(tid))
                return s.lstrip(MARK_SPACE).lstrip("Ċ")

            def _is_glue(tid: int) -> bool:
                b = _surf_key(tid)
                return (not b) or b.isdigit() or b in {
                    ".", ",", "!", "?", ":", ";", "*", "(", ")",
                    "là", "của", "ông", "ngoại", "và", "có", "không",
                    DAP_LABEL, HOI_LABEL,
                }

            prompt_keys = {_surf_key(t) for t in out[:prompt_len]}
            for i in range(len(consumed)):
                consumed[i] = False
            # §12.149 — prefix agenda trùng bề mặt prompt → coi như đã phủ
            # (tránh boost lại `Việt`/`Nhà`/`Thủ` → glue ởViệt / lậpNhà)
            i_pre = 0
            while i_pre < len(agenda_seq):
                tid = int(agenda_seq[i_pre])
                bare = _surf_key(tid)
                if (not bare) or bare in prompt_keys or _is_glue(tid):
                    consumed[i_pre] = True
                    i_pre += 1
                    continue
                break
            oi = 0
            for i, t in enumerate(agenda_seq):
                if consumed[i]:
                    continue
                if ag_strict:
                    # STRICT: khớp đúng vị trí — lệch 1 token thì dừng (không skip `về`/`Lư`)
                    if oi < len(out_tail) and int(out_tail[oi]) == int(t):
                        consumed[i] = True
                        oi += 1
                    else:
                        break
                else:
                    while oi < len(out_tail):
                        if int(out_tail[oi]) == int(t):
                            consumed[i] = True
                            oi += 1
                            break
                        oi += 1
                    else:
                        break
            pending = [i for i, c in enumerate(consumed) if not c]

            # §12.137 prod: nếu đầu agenda (bỏ glue) KHÔNG nằm trong giao W_skip
            # → agenda lệch cha (đo: prompt Tổ nhưng MZ trả Tông `, ông ngoại…`) —
            # bỏ boost/nonag bước này, để Hadamard + softmax quyết.
            agenda_ok = True
            if (
                pending
                and str(vote_agg).lower() in {"prod", "product", "hadamard", "mul"}
            ):
                head_i = next(
                    (i for i in pending if not _is_glue(int(agenda_seq[i]))),
                    None,
                )
                if head_i is not None:
                    ht = int(agenda_seq[head_i])
                    if 0 <= ht < n and float(vote_raw[ht]) <= 0.0:
                        agenda_ok = False
                        _ag_tin = False
            if agenda_ok:
                finite_sc = scores[np.isfinite(scores)]
                rv = float(np.max(finite_sc)) if finite_sc.size else 1.0
                if not np.isfinite(rv) or rv <= 0:
                    rv = 1.0
                pending_set = {int(agenda_seq[i]) for i in pending}
                for off, i in enumerate(pending[: max(1, ag_win)]):
                    t = agenda_seq[i]
                    scores[t] += ag_boost * rv * (ag_decay ** off)
                prompt_keys = {_surf_key(t) for t in out[:prompt_len]}
                pending_keys = {_surf_key(t) for t in pending_set}
                # Entity neo: bỏ bề mặt đã thuộc prompt (`Năm` trong `Năm 1010`)
                for t in ent_set:
                    if t not in out_tail and _surf_key(t) not in prompt_keys:
                        scores[t] += 0.35 * ag_boost * rv
                # Chặn recirculate Năm↔ĠNăm từ prompt (§12.134)
                ban_keys = prompt_keys - pending_keys
                if ban_keys:
                    pos_idx = np.flatnonzero(
                        np.isfinite(scores) & (scores > 0)
                    )
                    for tid in pos_idx:
                        if _surf_key(int(tid)) in ban_keys:
                            scores[int(tid)] = -np.inf
                # §12.135 — đè hub flu ngoài cửa sổ agenda (Tổ→chức vs quyết)
                if pending and nonag_scale < 1.0:
                    allow = {int(agenda_seq[i]) for i in pending[: max(1, ag_win)]}
                    pos_idx = np.flatnonzero(np.isfinite(scores) & (scores > 0))
                    for tid in pos_idx:
                        ti = int(tid)
                        if ti in allow:
                            continue
                        ts = state.vocab.token_of(ti)
                        bare = ts.lstrip(MARK_SPACE)
                        # punct / boundary giữ
                        if bare in {".", ",", "!", "?", ":", ";"} or (
                            len(bare) <= 2 and not any(ch.isalnum() for ch in bare)
                        ):
                            continue
                        if 0 <= eos_id < n and ti == int(eos_id):
                            continue
                        scores[ti] *= nonag_scale
            else:
                prompt_keys = {_surf_key(t) for t in out[:prompt_len]}
                pos_idx = np.flatnonzero(np.isfinite(scores) & (scores > 0))
                for tid in pos_idx:
                    if _surf_key(int(tid)) in prompt_keys:
                        scores[int(tid)] = -np.inf
        else:
            from nn_aivie.core.tokenizer import MARK_SPACE

            def _surf_key2(tid: int) -> str:
                s = state.vocab.token_of(int(tid))
                return s.lstrip(MARK_SPACE).lstrip("Ċ")

            prompt_keys = {_surf_key2(t) for t in out[:prompt_len]}
            pos_idx = np.flatnonzero(np.isfinite(scores) & (scores > 0))
            for tid in pos_idx:
                if _surf_key2(int(tid)) in prompt_keys:
                    scores[int(tid)] = -np.inf
        scores[inner_digit_ids(state.vocab)] = 0.0
        scores = mask_non_surface(scores, state.vocab)
        scores = apply_repetition_penalty(
            scores, out, state.vocab, prompt_len=prompt_len,
        )
        # §12.144 — cứng: đích đã consume không sống lại qua bonus/flu
        if blocked_dst:
            for bi in blocked_dst:
                if 0 <= int(bi) < n:
                    scores[int(bi)] = 0.0
        if 0 <= eos_id < n:
            pos = scores[np.isfinite(scores) & (scores > 0)]
            ref = float(np.median(pos)) if pos.size else 0.0
            # §12.141/P42 — EOS từ end_after cạnh tranh; fallback median nếu β=0
            if eos_ea_beta > 0:
                n_bound = count_boundaries(out[prompt_len:], state.vocab)
                ea = eos_from_end_after(
                    state, list(out), eos_id=eos_id, n_vocab=n,
                    beta=eos_ea_beta, ref_scale=max(ref, 1e-6),
                    boundary_count=n_bound,
                    episode_eos_boost=episode_eos_boost,
                )
                scores[eos_id] = max(ea, ref * 0.5) if ea > 0 else ref
            else:
                scores[eos_id] = ref
            # §121 — ĐÃ THỬ cho hai vế CÙNG ĐƠN VỊ và ĐÃ HOÀN NGUYÊN.
            # §120 đo được: agenda cộng điểm TƯƠNG ĐỐI (`ag_boost * rv`), EOS
            # có sàn TUYỆT ĐỐI (2,00) ⇒ nền sụp thì EOS thắng mặc định.
            # Cách thử: chặn TRẦN EOS theo nền (`min(EOS, 0,05 * rv)`), k hiệu
            # chỉnh để giữ nguyên hành vi ở nền bình thường (rv≈40 → 2,0).
            # ĐẶT SAU khối này thì trần chặn luôn EOS đã được §118 nâng hợp lệ
            # (`lúa` mất chỗ dừng: 1,15×46,9 = 53,9 bị kéo về 2,34) — lỗi «đúng
            # cơ chế SAI CHỖ ĐẶT» lần thứ MƯỜI. Đặt TRƯỚC khối này thì `lúa`
            # giữ được, `đồng` hết cụt — nhưng phần nói thêm là RÁC, và cả ba
            # thước tụt nặng trên bộ 20 prompt:
            #     TRUNG THÀNH 85,9 → 81,9 · TRẬT TỰ 97,3 → 90,1 · LẶP 7,0 → 13,4
            # Bỏ trần EOS chỉ đổi CỤT lấy LẢM NHẢM, y hệt §119. Đừng thử lại.
            # Đứng trên biên + đã đóng ≥1 câu trong phần sinh → EOS ≥ max
            if boundary_eos_beta > 0 and out:
                from nn_aivie.hca.sent_intra import _is_boundary as _is_b_eos

                # §117 — ĐÃ THỬ truyền `_bd_gen` vào CẢ BA chỗ gọi
                # `count_boundaries` và ĐÃ HOÀN NGUYÊN. Mâu thuẫn là THẬT:
                # `count_boundaries(hist, vocab, bound=None)` chạy BOOTSTRAP
                # (dấu câu cứng) còn `_is_b_eos` ngay dưới dùng tập ĐÃ HỌC
                # `_bd_gen` ⇒ ở `lúa` bước 21, out[-1] LÀ biên mà n_b_gen = 0
                # ⇒ cú đẩy bị chặn ⇒ EOS ghim 2,0 / max 46,9 ⇒ không dừng nổi.
                # NHƯNG hợp nhất theo hướng `_bd_gen` làm EOS nổ bừa: `chó là`
                # cụt còn «gia súc», `nước là` cụt ở «ao». Hợp nhất theo hướng
                # bootstrap thì `lúa` vẫn không dừng. CẢ HAI HƯỚNG ĐỀU TRƯỢT —
                # `_bd_gen` quá rộng, bootstrap quá chặt. Đừng vá lại kiểu này.
                n_b_gen = count_boundaries(out[prompt_len:], state.vocab)
                # §118 — TÁCH HAI VỊ TỪ. §117 chứng minh không hợp nhất
                # được vì «biên» đang gánh HAI việc khác nhau:
                #   ĐÃ ĐÓNG CÂU  = `n_b_gen` (bootstrap, dấu câu CỨNG)
                #   ĐƯỢC PHÉP DỪNG = `_is_b_eos` + `_bd_gen` (tập ĐÃ HỌC)
                # Cả hai vị từ đều ĐÚNG; sai là chỗ NỐI chúng bằng AND: đòi
                # `n_b_gen >= 1` tức đòi ĐÃ đóng một câu thì mới cho dừng, nên
                # câu ĐẦU TIÊN không bao giờ được kết thúc (§117 «muốn được
                # dừng thì phải ĐÃ dừng»).
                # Vế thay: đã đóng câu HOẶC agenda hết đáng tin (`_ag_tin`,
                # §116) — khi agenda không còn dẫn được thì không có lý do cấm
                # dừng. Đây là điều kiện có NỘI DUNG, không phải ngưỡng độ dài.
                # §119 — ĐÃ THỬ THẮT vế này bằng ngưỡng «agenda đã tiêu
                # thụ ≥ 0,5» và ĐÃ HOÀN NGUYÊN. Lý do thắt: §118 để `sóc là`
                # cụt ở «thú gặm» khi mới tiêu thụ 7% agenda (`lúa` dừng ở 70%,
                # `chó` ở 100%) — tách bạch rất rõ, tưởng chừng chỉ cần chặn
                # dưới ngưỡng. Nhưng ngưỡng chặn chỗ dừng ở NHIỀU prompt chứ
                # không riêng `sóc`: cả ba thước đều tụt trên bộ 20 prompt
                # (TRUNG THÀNH 85,9 → 83,3 · TRẬT TỰ 97,3 → 94,6 · LẶP 7,0 →
                # 7,2), và `sóc` chỉ đổi mẩu cụt lấy câu rác dài. Vế KHÔNG kèm
                # ngưỡng tốt hơn về tổng thể. Đừng thắt lại kiểu này.
                # §126 — ngưỡng «agenda đã tiêu thụ» ĐÃ CHẤM LẠI bằng 6 hạt
                # giống và LOẠI DỨT ĐIỂM. Mức 0,3 và 0,5 cho số Y HỆT nhau
                # (hiệu ứng BÃO HOÀ dưới 0,3 — không có mức «vừa phải»):
                #   0,0  TT 86,1±1,3 · TQ 97,2±1,8 · LẶP 9,9
                #   0,3  TT 83,7±2,4 · TQ 95,4±1,8 · LẶP 9,9
                # Chênh −2,4 trên TRUNG BÌNH 6 hạt ≈ 3× sai số chuẩn ⇒ THẬT.
                # (§119 loại bằng 1 hạt nên khi đó chưa đủ căn cứ; §125 nghi
                # oan, §126 xác nhận loại là đúng.) Đừng dựng lại ngưỡng này.
                _duoc_dung = (n_b_gen >= 1) or (not _ag_tin)
                if (
                    _is_b_eos(state.vocab, int(out[-1]), _bd_gen)
                    and _duoc_dung
                    and len(out) > prompt_len
                ):
                    finite_pos = scores[np.isfinite(scores)]
                    mx = float(np.max(finite_pos)) if finite_pos.size else ref
                    if not np.isfinite(mx) or mx <= 0:
                        mx = max(ref, 1e-6)
                    scores[eos_id] = max(
                        float(scores[eos_id]),
                        float(boundary_eos_beta) * mx,
                    )
            # §116 — CHỈ dìm EOS khi agenda còn ĐÁNG TIN. Trước đây hai bộ phận
            # bất đồng về cùng một câu hỏi: §12.137 (dòng ~630) tuyên bố «agenda
            # lệch cha» rồi bỏ CẢ boost lẫn đè `nonag_scale`, trong khi chỗ này
            # vẫn lấy «agenda còn mục chưa phát» làm lý do CẤM DỪNG. Mạng bị ép
            # nói tiếp mà không còn nguồn dẫn ⇒ bịa. Đo được 1:1 ở `lúa`: ba
            # token rác «thể giúp cơ» rơi ĐÚNG ba bước agenda_ok=False, mục chờ
            # là «bao» (đầu của `bao ngoài hạt`). `hổ`/`chó` có 0 bước như vậy.
            if agenda_seq and _ag_tin and any(not c for c in consumed):
                scores[eos_id] *= 0.15
        # ── TÍN HIỆU THỨ TỰ BẬC-2 CỦA CHA vào ĐƯỜNG PRODUCTION — §83 ────
        # `W_seq` là bigram BẬC-1 (parent_store.py §"_ensure_seq_index",
        # varlm.py §mở đầu) ⇒ mất trí nhớ sau mỗi từ ⇒ word-salad. §82 đo được:
        # mạng truy ĐÚNG câu nguồn (không một từ lạ) nhưng mất trật tự sau
        # ~8–12 token. `ParentStore.seq_next` đã có sẵn phân bố bậc-2 dựng từ
        # chuỗi cha — nhưng ĐO ĐƯỢC 0 lời gọi trong `generate()` thật (§83.1):
        # nó chỉ được nối vào `generate_autoregressive`, đường mà production
        # không bao giờ vào. Lần thứ TƯ cùng kiểu (LOI1, §79, §81, nay §83).
        #
        # ⚠️ CHỈ XẾP LẠI ứng viên đang sống (`scores > 0`), KHÔNG hồi sinh
        # token đã bị chặn: §12.144 đặt `blocked_dst` về 0 ở NGAY TRÊN với đúng
        # bất biến «đích đã consume không sống lại qua bonus/flu», và §82 đã
        # chứng minh từ đúng vẫn nằm trong túi — thiếu là THỨ HẠNG, không phải
        # ứng viên. Cộng thêm theo đúng thành ngữ đường cũ (dòng ~3282):
        # chuẩn hoá theo max của chính kênh bậc-2 rồi nhân với thang nền.
        # §115 — ĐÃ THỬ cổng «chỉ áp khi agenda cạn» (§114.3) và GỠ: đo được
        # bậc-2 nổ 0/5 prompt, tức ở MỌI bước agenda vẫn còn mục chưa phát.
        # `lúa` trượt đuôi TRONG KHI agenda còn mục đúng đang chờ ⇒ đuôi lệch
        # KHÔNG do agenda cạn. Đừng dựng lại cổng này.
        if _o2_w > 0 and state.parents is not None and out:
            _prev = int(out[-2]) if len(out) >= 2 else None
            if _o2_cha_hoat:
                # §84 — bậc-2 CỤC BỘ: tri của cha đang hoạt, lui về bi cục bộ.
                _hi = None
                if _prev is not None:
                    _d = _o2_tri.get((_prev, int(out[-1])))
                    if _d and sum(_d.values()) >= _o2_min:
                        _hi = _d
                if _hi is None:
                    _hi = _o2_bi.get(int(out[-1]))
            else:
                _hi = state.parents.seq_next(_prev, int(out[-1]), min_hi=_o2_min)
            if _hi:
                _live = scores[np.isfinite(scores) & (scores > 0)]
                if _live.size:
                    _base = float(_live.max())
                    _hmax = max(_hi.values()) or 1.0
                    for _t, _c in _hi.items():
                        _t = int(_t)
                        if not (0 <= _t < n):
                            continue
                        if np.isfinite(scores[_t]) and scores[_t] > 0:
                            scores[_t] += _o2_w * (float(_c) / _hmax) * _base
        # ── CHUỖI BA BƯỚC vào ĐƯỜNG PRODUCTION — §79 ───────────────────
        # `hub_chain_proposals` xưa nay chỉ được gọi trong
        # `generate_autoregressive`, mà đường production (`ar_w_skip_primary`
        # = True, mặc định) KHÔNG BAO GIỜ vào đó — nó chạy vòng lặp này.
        # Nên suốt từ §22 tới §78, bật `hub_chain_boost` KHÔNG đổi một token
        # nào của đầu ra thật. Cùng loại lỗi với «kênh §53 chưa bao giờ được
        # gọi» (TK_052 vòng 2): tính năng có, đo được, mà đường thật không dùng.
        if _chain_w:
            _f0 = scores[np.isfinite(scores) & (scores > 0)]
            _med = float(np.median(_f0)) if _f0.size else 0.0
            # §138 — ĐO GIÁ của việc nới §12.144. Nhánh dưới hồi sinh token
            # CHƯA CÓ trong phiếu (gồm cả đích đã tiêu thụ) ở thang TRUNG VỊ
            # (~0,03 so với đỉnh ~40) ⇒ §137 đo được nó KHÔNG BAO GIỜ thắng,
            # tức loại suy không ra được văn bản. `_hs > 0` thay thang trung vị
            # bằng `_hs × ĐỈNH`. Mặc định 0 = giữ nguyên §12.144.
            # §139 — CỔNG VỊ TRÍ. §138 đo được: cho chuỗi đủ sức thắng
            # (hồi_sinh=1,0) thì nó RA ĐƯỢC văn bản nhưng phát lặp
            # («chó là gia gia gia gia gia súc») — nó biết «từ này đáng nói»,
            # không biết «đáng nói ở CHỖ NÀO». `_seq_bi` (bigram câu cha, chính
            # thứ thước TRẬT TỰ dùng) trả lời đúng câu đó: `là → gia` có chứng
            # cứ, `gia → gia` thì không.
            _vt = bool(getattr(config, "ar_w_skip_chain_vi_tri", False))
            _ml = bool(getattr(config, "ar_w_skip_chain_mot_lan", False))
            _bi_sau: set = set()
            if _vt and out and state.parents is not None:
                _bi_sau = set(_bi_idx.get(int(out[-1]), ()) or ())
            _hs = float(getattr(config, "ar_w_skip_chain_hoi_sinh", 0.0) or 0.0)
            _dinh = float(_f0.max()) if (_hs > 0 and _f0.size) else 0.0
            _nen_hs = (_hs * _dinh) if _dinh > 0 else _med
            for _t, _w in _chain_w.items():
                if not (0 <= _t < n):
                    continue
                if _vt and _bi_sau and _t not in _bi_sau:
                    continue      # §139 — không có chứng cứ đứng SAU token vừa phát
                # §140 — MỘT LẦN MỖI TỪ. §139 đo: câu bắt đầu ĐÚNG rồi thoái hoá
                # («gia súc thuộc nhóm ăn thịt GIA GIA, vị GIA…») vì bigram một
                # bước là ràng buộc yếu trên 450k câu cha — `gia` có chứng cứ
                # đứng sau hàng trăm token nên qua cổng gần như mọi lúc. Lỗi còn
                # lại là TÁI XUẤT trong cùng câu, thứ phạt lặp ×0,25 không kham
                # nổi khi thang hồi sinh lên 1,0.
                if _ml and _t in _da_chuoi:
                    continue
                if np.isfinite(scores[_t]) and scores[_t] > 0:
                    scores[_t] = scores[_t] * (1.0 + _chain_b * _w)
                elif _nen_hs > 0:
                    scores[_t] = _nen_hs * _chain_b * _w
        # §151 — DỪNG THEO ĐỘ TIN (TK_053 §1). Đo được (§151.1): khi sinh
        # bình thường `p_top1` ∈ [0,38 ; 0,96]; lúc hết nguyên liệu nó tụt còn
        # [0,029 ; 0,082] — khoảng trống 4,6 LẦN, không chồng lấn. Tín hiệu «hết
        # nguyên liệu» có SẴN một bước TRƯỚC khi mạng phát bậy: `mặt trời là`
        # phát rác `ngôi` ở bước có p_top1 = 0,082.
        # Thay luật biên thủ công (§116–§118 tốn 4 mục, vẫn còn đuôi rác) bằng
        # MỘT đại lượng đo được từ chính phân bố điểm.
        # §156 — KHÔNG cho EOS thắng ở BƯỚC ĐẦU. `mùa xuân` sinh đúng 1 token
        # = EOS (rỗng) vì `xuân` hay đứng cuối câu trong corpus ⇒ phiếu bước 0
        # chọn EOS. Trả về rỗng thì vô dụng; mỗi lượt sinh phải ra ≥1 token.
        # Chặn EOS khi CHƯA sinh token nào (out chỉ có prompt).
        if len(out) <= prompt_len and 0 <= int(eos_id) < n:
            scores[int(eos_id)] = 0.0
        if _do_tin_min > 0.0:
            _pp = scores[np.isfinite(scores) & (scores > 0)]
            if _pp.size:
                _tong = float(_pp.sum())
                # §156 — cũng KHÔNG dừng theo độ tin ở bước đầu (chưa có gì)
                if (len(out) > prompt_len and _tong > 0
                        and float(_pp.max()) / _tong < _do_tin_min):
                    break
        # §165 — TIÊM cạnh REINFORCED: token nội dung trong ngữ cảnh gần kích
        # thích loại-suy-anh-em đã xác nhận. Scale theo `rv` (đỉnh phiếu) — bài
        # học §137: thang tuyệt đối không thắng nổi. Chỉ nâng đích ĐANG SỐNG
        # (scores>0), KHÔNG hồi sinh token chết (§137). beta nhỏ ⇒ nudge.
        # §171 — HỒI SINH THEO ĐỘ TIN (plan a). §170 tìm ra: cạnh loại suy vào
        # được W_skip nhưng vote PROD triệt tiêu cạnh MỘT-NGUỒN (natri chỉ do
        # sắt đề xuất; `và` không ⇒ tích = 0). Fix: cộng đề cử loại suy như SUM
        # (vòng qua prod), HỒI SINH cả đích chết — NHƯNG chỉ khi mạng KHÔNG CHẮC
        # (p_top1 < ngưỡng), tức lúc phiếu câu yếu, nudge liên tưởng ít phá nhất
        # VÀ hữu ích nhất. Khi mạng chắc (đang phát câu định nghĩa đúng), tắt.
        if _rf_edges and _rf_beta > 0.0:
            _pos_rf = scores[np.isfinite(scores) & (scores > 0)]
            if _pos_rf.size:
                _rv_rf = float(_pos_rf.max())
                _sum_rf = float(_pos_rf.sum())
                _conf = _rv_rf / _sum_rf if _sum_rf > 0 else 1.0
                # chỉ khi KHÔNG CHẮC (độ tin thấp)
                if _conf < _rf_conf:
                    # cửa sổ gồm token GẦN ĐÂY kể cả PROMPT: khái niệm nguồn
                    # (sắt) thường Ở TRONG prompt — dùng max(prompt_len,…) sẽ bỏ
                    # sót nó (§176). Lấy 3 token cuối bất kể ranh prompt.
                    for _ct in out[max(0, len(out) - 3):]:
                        for _j, _w in _rf_edges.get(int(_ct), ()):
                            if not (0 <= _j < n):
                                continue
                            # boost phải VƯỢT được max hiện tại khi cạnh đã-xác-
                            # nhận mạnh: ở vùng phẳng (conf thấp) rv nhỏ nên
                            # beta·rv·w không bao giờ thắng (§176). Dùng
                            # rv·(1+beta·w): cạnh w cao → >rv → nút khái niệm
                            # đã-xác-nhận nổi lên khi mạng THẬT SỰ phân vân.
                            _add = _rv_rf * (1.0 + _rf_beta * _w)
                            _cur = scores[_j] if np.isfinite(scores[_j]) else 0.0
                            # SUM: hồi sinh đích chết (set), cộng đích sống
                            scores[_j] = max(_cur, 0.0, _add)
        # §TK_056 — TRỘN kNN-LM (encoder h → datastore) vào tập ứng viên sống,
        # sau MỌI mask/penalty ⇒ tôn trọng bất biến không hồi sinh token chết.
        if _KNN_HOOK is not None and len(out) > prompt_len:
            try:
                _kl, _mode, _p = _KNN_HOOK
                if _mode == "hybrid":
                    scores = _kl.hybrid_blend(list(out), scores, n, state.vocab,
                                              eos_id=eos_id, prompt_len=prompt_len, **_p)
                elif _mode == "rerank":
                    scores = _kl.rerank(list(out), scores, n, a=_p["a"])
                else:
                    scores = _kl.blend(list(out), scores, _p["lam"], n,
                                       vocab=state.vocab, widen_k=_p["widen_k"])
            except Exception:
                pass
        # §TK_083 Pha 1c — NEO MỨC-PROMPT (sau mask/kNN, chỉ reweight ứng viên sống)
        if _ANCHOR is not None and len(out) > prompt_len:
            try:
                scores = _apply_prompt_anchor(scores, out, prompt_len)
            except Exception:
                pass
        finite = np.where(np.isfinite(scores), scores, -np.inf)
        if not np.isfinite(finite).any() or float(finite.max()) <= 0:
            break
        # §157 — GỠ khối ÉP CỨNG EOS. A/B đo (30+ prompt): bỏ nhánh này KHÔNG
        # đổi một token nào (0/20 khác) và nó chưa BAO GIỜ ép EOS khi argmax
        # chọn khác (0 lần). Đã thành MÃ CHẾT sau khi §118 (cộng điểm EOS),
        # §151 (dừng theo độ tin) và §156 (chặn EOS bước đầu) chuyển việc dừng
        # sang tầng ĐIỂM SỐ — tới lúc chọn token, nếu ở biên câu trọn thì EOS
        # đã thắng sẵn trong `scores`. Gỡ luôn phép nối hỏng `and n_b_gen >= 1`
        # (§124.5) và lời gọi `count_boundaries` thừa tại đây.
        if T <= 0:
            nxt = int(np.argmax(finite))
        else:
            nxt = _sample(finite, cfg_s, rng)
        # §140 — ghi nhận nếu token vừa chọn là đề cử của CHUỖI.
        if _chain_w and int(nxt) in _chain_w:
            _da_chuoi.add(int(nxt))
        # §12.144/145 — đánh dấu cạnh vừa dùng (+ tuổi để hồi)
        if consume_edges:
            mark_w_skip_consume(
                state, list(out), int(nxt),
                d_max=d_max,
                blocked_edges=blocked_edges,
                consume_edges=True,
                reverse_window=consume_rev_win,
                edge_age=edge_age,
                step=gen_step,
            )
        out.append(int(nxt))
        gen_step += 1
        update_open_brackets(open_brackets, int(nxt), len(out) - 1, state, out)
        if int(nxt) == int(eos_id):
            break
    return out


def _wseq_fallback(
    state: GraphState, generated: list[int], eos_id: int, max_steps: int,
    config: HCAConfig, rng: np.random.Generator,
    *,
    agenda_ids: list[int] | None = None,
    entity_ids: tuple[int, ...] = (),
) -> list[int]:
    """Autoregressive — W_skip primary (§12.133 MZ agenda) hoặc matrix."""
    out = list(generated)
    prompt_len = len(generated)
    vocab_size = len(state.vocab)
    ensure_frame_brackets(state)
    open_brackets: list[tuple[int, int]] = []
    seed_open_brackets(state, out, open_brackets)
    # §12.131/133 — W_skip primary + agenda MZ
    if bool(getattr(config, "ar_w_skip_primary", True)):
        return _w_skip_primary_loop(
            state, out,
            prompt_len=prompt_len, eos_id=eos_id, max_steps=max_steps,
            config=config, rng=rng, open_brackets=open_brackets,
            agenda_ids=agenda_ids, entity_ids=entity_ids,
        )
    # §12.34: KHÔNG tính ở đây. `active_parents` chỉ được dùng ở đường legacy bên dưới
    # (`apply_frame_bracket_gate`), mà khối matrix core `return` trước khi tới đó ⇒ ở chế độ
    # `ar_matrix_core=True` đây là công VÔ ÍCH hoàn toàn khi không primary.
    # Giá: `rank_matches` là O(số cha) — đo mini 31ms/5.664 cha ⇒ v12 (430k) ~2,4s MỖI PROMPT.
    if bool(getattr(config, "ar_matrix_core", True)):
        from nn_aivie.hca.matrix_core import matrix_core_logits
        from nn_aivie.hca.spectral_stack import ensure_spectral_stack

        stack = None
        if float(getattr(config, "ar_matrix_stack_weight", 0.0)) > 0:
            # Bật stack trong matrix core kể cả khi ar_spectral_stack=False (no_agenda probe).
            stack = ensure_spectral_stack(
                state,
                d=int(getattr(config, "ar_spectral_stack_d", 24)),
                sidecar=getattr(state, "_spectral_stack_sidecar", None),
            )
        n_handoffs = 0
        max_h = int(getattr(config, "ar_sent_agenda_handoff_max", 2))
        num_closed = False  # §12.12: vừa chép xong cụm số → số ĐÓNG ở bước này
        zone_done: set[int] = set()  # §12.29: cha đã phát qua zone-handoff
        # §12.15: tổ hợp token đặc-hiệu của PROMPT chưa cha nào chứng thực (giao rỗng) ⇒ mọi cha
        # chỉ khớp một PHẦN ⇒ trả lời tự tin = BỊA (`Nhà Trần dời đô…` → `1010` của nhà Lý).
        # Tính MỘT lần trên prompt (ổn định; history dài dần sẽ tự làm giao rỗng nếu tính lại).
        if bool(getattr(config, "ar_matrix_premise_guard", True)):
            from nn_aivie.hca.matrix_core import unattested_premise

            if unattested_premise(
                state, list(out[:prompt_len]) or list(out),
                last_k=int(getattr(config, "ar_matrix_premise_last_k", 6)),
            ):
                max_steps = 0
        # §12.27: prompt kết bằng `Đáp:` và cha QA đang kích khớp câu hỏi ⇒ CHÉP span đáp.
        # Path không làm được (emit của cha QA là answer_ids, không giao với câu hỏi).
        if bool(getattr(config, "ar_matrix_qa_answer_copy", False)) and out:
            from nn_aivie.core.tokenizer import DAP_LABEL
            from nn_aivie.hca.matrix_core import (
                answer_span_from_active, matrix_core_logits as _mcl,
            )

            _dap = state.vocab._token_to_id.get(DAP_LABEL)  # noqa: SLF001
            if _dap is not None and int(out[-1]) == int(_dap):
                _lk: dict = {}
                _mcl(state, out, config=config, spectral_stack=stack, lock_out=_lk)
                _sp = answer_span_from_active(
                    state, _lk.get("act"), out,
                    gidf=getattr(state.parents, "_gidf", None),
                    form_min=float(getattr(config, "ar_matrix_and_form_min", 0.0)) or 0.9,
                    statement=bool(getattr(config, "ar_matrix_qa_answer_statement", True)),
                    wh_learned=bool(getattr(config, "ar_matrix_wh_learned", False)),
                )
                for _t in _sp:
                    out.append(int(_t))
                    update_open_brackets(open_brackets, int(_t), len(out) - 1, state, out)
                if _sp:
                    max_steps = 0
        for _ in range(max_steps):
            w0 = out[-1]
            _lock: dict = {}
            scores = matrix_core_logits(
                state, out, config=config, spectral_stack=stack, lock_out=_lock
            )
            scores = mask_non_surface(scores, state.vocab)
            if num_closed:
                # Chép `Ġ1 0 1 0` rồi bước kế argmax ra `9` (không `Ġ`) → dính thành "10109"
                # (đo v12 §12.11). Số đã chép NGUYÊN từ cha ⇒ cấm nối thêm chữ-số vào nó.
                from nn_aivie.hca.matrix_core import inner_digit_ids

                scores[inner_digit_ids(state.vocab)] = -np.inf
                num_closed = False
            scores = apply_repetition_penalty(
                scores, out, state.vocab, prompt_len=prompt_len
            )
            # TK_045 §12.106 — KÊNH PHIẾU W_skip (đặc tả tác giả §12.105). Cộng MỀM trên cả cửa
            # sổ, chuỗi ĐẦY ĐỦ, không cổng cứng. Mục đích: phá vòng chép — §12.105b đo được
            # `argmax` cưỡng bức `copy ≈ 1.0` vì token điểm cao nhất luôn thuộc đoạn đã nhớ.
            # Cần `hybrid_temperature > 0` để có tác dụng; xem chú thích knob ở config.
            _wsw = float(getattr(config, "ar_matrix_w_skip_decode_weight", 0.0))
            if _wsw > 0.0:
                from nn_aivie.hca.w_skip import vote_logits

                _vt = vote_logits(
                    state, list(out), n_vocab=len(scores),
                    d_max=int(getattr(config, "ar_matrix_w_skip_decode_dmax", 16)),
                    lam=float(getattr(config, "ar_matrix_w_skip_decode_lam", 0.5)),
                    topk=int(getattr(config, "ar_matrix_w_skip_decode_topk", 12)),
                )
                # §12.107a — CHỮ-SỐ: `W_skip` học được `1`→`8`→`7`→`0` như bigram thường, nên
                # kênh phiếu tự chọn được đường "đánh vần số từng chữ", vượt qua đường CHÉP cụm
                # số từ cha (`digit_run_from_locked`). Đo được `digit⎵` 0.00→0.60. Số KHÔNG thuộc
                # thẩm quyền kênh này (mạng không có cơ quan số — TK_041) ⇒ trả phiếu về 0.
                from nn_aivie.hca.matrix_core import inner_digit_ids

                _vt[inner_digit_ids(state.vocab)] = 0.0
                _fin = np.isfinite(scores)
                if _fin.any() and _vt.any():
                    # Chuẩn hoá về cùng thang trước khi cộng: điểm phiếu nằm trong dải ~1..8 còn
                    # `scores` là logits kênh matrix — cộng thô thì một bên áp đảo tuỳ checkpoint.
                    _sc = scores[_fin]
                    _rng_s = float(_sc.max() - _sc.min())
                    _rng_v = float(_vt.max())
                    if _rng_v > 0 and _rng_s > 0:
                        # §12.107a/§12.110 — EOS phải được BÙ đúng bằng phần các ứng viên TIẾP TỤC
                        # được cộng. `iter_sentence_spans` cắt ở `.` nên `<EOS>` luôn rơi vào span
                        # dài 1 và bị `len(toks) < 2` bỏ ⇒ khối phiếu VÀO EOS đúng bằng 0 (đo mini)
                        # ⇒ cộng thô thì mỗi bước dìm EOS ~1 khoảng điểm. Kênh phiếu chỉ được XẾP
                        # LẠI các ứng viên tiếp tục; dừng-hay-tiếp thuộc kênh `eos`.
                        #
                        # §12.110: bản §12.107a bù theo `max(khác EOS)` — giữ đúng quyết định
                        # ARGMAX, nhưng `_sample` làm softmax trên CẢ top-k (k=12) rồi multinomial.
                        # Kênh phiếu nâng cả hạng 2…12 nên mẫu số phình ⇒ P(EOS) vẫn tụt. Đo trên
                        # v12 (§12.108): `stop` 1.00 → 0.60. `max` chỉ là giới hạn T→0 của thước
                        # đúng, là logsumexp/T. Bù theo logsumexp ⟺ giữ nguyên ĐÚNG tỉ lệ softmax
                        # của EOS, trung lập về dừng ở mọi T và mọi độ dày phiếu.
                        _T = float(getattr(config, "hybrid_temperature", 0.0) or 0.0)
                        _kk = int(getattr(config, "hybrid_top_k", 12))

                        def _cont_anchor(_s: np.ndarray) -> float:
                            _o = _s.copy()
                            if eos_id < len(_o):
                                _o[eos_id] = -np.inf
                            _o = _o[np.isfinite(_o)]
                            if _o.size == 0:
                                return float("nan")
                            if _T <= 0.0:
                                return float(_o.max())
                            _k = min(max(1, _kk), _o.size)
                            _t = np.partition(_o, -_k)[-_k:]
                            _m = float(_t.max())
                            return _m + _T * float(np.log(np.exp((_t - _m) / _T).sum()))

                        _a_pre = _cont_anchor(scores)
                        scores[_fin] = _sc + _wsw * _rng_s * (_vt[_fin] / _rng_v)
                        if eos_id < len(scores) and np.isfinite(scores[eos_id]):
                            _a_post = _cont_anchor(scores)
                            if np.isfinite(_a_pre) and np.isfinite(_a_post):
                                scores[eos_id] += _a_post - _a_pre

            # Matrix core: không frame-bracket thủ công (giống LLM — chỉ logits).
            _, ans = _split_frame(out, state.vocab)
            if eos_id < len(scores):
                pre_eos = (
                    float(scores[eos_id])
                    if np.isfinite(scores[eos_id])
                    else -np.inf
                )
                others = scores.copy()
                others[eos_id] = -np.inf
                omax = (
                    float(np.max(others[np.isfinite(others)]))
                    if np.any(np.isfinite(others))
                    else -np.inf
                )
                # Matrix path hết câu (EOS đã thắng logits) → giữ, đừng -inf.
                if pre_eos > omax and pre_eos > 0:
                    pass
                elif not ans:
                    scores[eos_id] = -np.inf
                elif state.adjacency.get_readout(w0, eos_id) >= config.hybrid_eos_threshold:
                    fm = (
                        scores[np.isfinite(scores)].max()
                        if np.any(np.isfinite(scores))
                        else 0.0
                    )
                    scores[eos_id] = fm * 1.5 + 1.0
                else:
                    scores[eos_id] = -np.inf
            if not np.any(np.isfinite(scores)):
                break
            nxt = (
                int(np.argmax(scores))
                if config.hybrid_temperature <= 0
                else _sample(scores, config, rng)
            )
            # TK_045 §12.5 v2: token kế là CHỮ-SỐ → CHÉP nguyên cụm số từ cha ĐANG KHOÁ
            # (lock_pi). Mạng không có cơ quan số (TK_041) ⇒ sinh từng chữ-số mất giá trị
            # (1010→"1 0 1 0"). v1 dùng succ[w0] → chép nhầm cha ("18110") nên đã revert.
            if nxt != eos_id and bool(getattr(config, "ar_matrix_copy_number", True)):
                from nn_aivie.hca.matrix_core import (
                    digit_run_from_locked, ensure_parent_bipartite,
                )

                _bi = ensure_parent_bipartite(state)
                _run = digit_run_from_locked(
                    _bi, state.vocab, int(_lock.get("lock_pi", -1)), int(w0), int(nxt),
                )
                if not _run and bool(
                    getattr(config, "ar_matrix_copy_number_from_act", True)
                ):
                    # §12.14 chép-lùi: path không khoá nổi (số đứng TRƯỚC phần khớp / prompt
                    # đảo trật tự) nhưng act vẫn trỏ đúng cha → chép số của cha đang kích.
                    from nn_aivie.hca.matrix_core import digit_run_from_active

                    _run = digit_run_from_active(
                        _bi, state.vocab, _lock.get("act"), out, int(nxt),
                        gidf=getattr(state.parents, "_gidf", None),
                    )
                if _run:
                    for _t in _run:
                        out.append(int(_t))
                        update_open_brackets(open_brackets, int(_t), len(out) - 1, state, out)
                    num_closed = True
                    continue

            if nxt == eos_id:
                # TK_045 P0.3: EOS local ≠ hết bài nếu episode/W_agenda còn câu sau.
                # ar_no_agenda = tắt MZ/fact retrieval — KHÔNG tắt handoff đa câu matrix.
                if not bool(getattr(config, "ar_sent_agenda_handoff", True)):
                    break
                if n_handoffs >= max_h:
                    break
                from nn_aivie.hca.sent_intra import _is_boundary as _is_bnd
                from nn_aivie.hca.sent_intra import agenda_handoff_span

                pre_eos = (
                    float(scores[eos_id])
                    if eos_id < len(scores) and np.isfinite(scores[eos_id])
                    else 0.0
                )
                span = agenda_handoff_span(
                    state,
                    out,
                    config=config,
                    eos_logit=pre_eos,
                    vocab_size=vocab_size,
                )
                if not span and bool(getattr(config, "ar_matrix_zone_handoff", False)):
                    # §12.29: episode-handoff đi theo TRẬT TỰ NGUỒN nên cạn ở ca GHÉP (prompt
                    # đòi tổ hợp, không có episode nào chứa sẵn). Lấy thêm span từ ZONE kế.
                    from nn_aivie.hca.multizone import zone_handoff_span

                    # PROMPT, không phải `out`: zone dựng từ cap-span, phần đã sinh làm nhiễu
                    # (đo: `…nhà Lý và nhà Trần Đáp: Trần` chỉ còn 1 zone `Trần`, mất `nhà Lý`).
                    _z = zone_handoff_span(
                        state, list(out[:prompt_len]) or list(out),
                        config=config, emitted_pis=zone_done,
                    )
                    if _z:
                        span, _zpi = _z
                        zone_done.add(int(_zpi))
                if not span:
                    break
                # TK_045 §12.74 — CỔNG PHỦ cho span nối tiếp. Đo trên mini, nhãn suy từ đồ thị
                # (dương = hai câu liền kề CÙNG tài liệu · âm khó = khác tài liệu NHƯNG có chia
                # fold nội dung): đòi span chia ≥3 fold `gidf≥4` với câu trước thì giữ 39,7% ca
                # đúng mà chỉ lọt 3,4% ca trôi. Hai luật hiển nhiên đều THUA: "phủ prompt" giết
                # cả chuỗi đúng (câu 3 hợp lệ không chia fold nào với prompt), "cùng episode"
                # cũng vậy (chuỗi đúng đi qua episode 62→175→87). Fold VIẾT HOA cũng thua luật
                # này ở cả hai trục (32,5%/3,8%).
                _cmin = int(getattr(config, "ar_sent_agenda_handoff_cover_min", 0))
                if _cmin > 0:
                    from nn_aivie.hca.matrix_core import _path_fold as _pf

                    _ps = state.parents

                    def _cfolds(ids):
                        _o = set()
                        for _t in ids:
                            _t = int(_t)
                            try:
                                if float(_ps._gidf(_t)) < 4.0:  # noqa: SLF001
                                    continue
                            except Exception:
                                continue
                            _f = _pf(state.vocab, _t)
                            if _f:
                                _o.add(_f)
                        return _o

                    # CÂU TRƯỚC (từ biên gần nhất) — đúng thứ đã hiệu chỉnh ngưỡng, không phải
                    # cả `out`: lấy cả `out` thì tập fold to dần nên cổng lỏng dần theo độ dài.
                    _j = len(out) - 1
                    while _j > 0 and not _is_bnd(state.vocab, int(out[_j - 1])):
                        _j -= 1
                    if len(_cfolds(out[_j:]) & _cfolds(span)) < _cmin:
                        break
                from nn_aivie.hca.sent_intra import _is_boundary

                # EOS ảo không vào surface → chèn "." + Ġ nếu cần để khỏi dính câu.
                if out and not _is_boundary(state.vocab, int(out[-1]), _bound_of(state)):
                    period_id = state.vocab._token_to_id.get(".")  # noqa: SLF001
                    if period_id is not None:
                        out.append(int(period_id))
                        update_open_brackets(
                            open_brackets, int(period_id), len(out) - 1, state, out
                        )
                first = int(span[0])
                hand_tok = state.vocab.token_of(first)
                if not hand_tok.startswith(("Ġ", "Ċ")):
                    spaced = "Ġ" + hand_tok.lstrip("Ġ")
                    alt = state.vocab._token_to_id.get(spaced)  # noqa: SLF001
                    if alt is not None:
                        span = [int(alt)] + [int(t) for t in span[1:]]
                    else:
                        gid = state.vocab._token_to_id.get("Ġ")  # noqa: SLF001
                        if gid is not None:
                            out.append(int(gid))
                            update_open_brackets(
                                open_brackets, int(gid), len(out) - 1, state, out
                            )
                for tid in span:
                    tid = int(tid)
                    if tid == eos_id:
                        break
                    out.append(tid)
                    update_open_brackets(open_brackets, tid, len(out) - 1, state, out)
                # Kết thúc câu vừa handoff — chèn "." nếu emit chưa có biên.
                if out and not _is_boundary(state.vocab, int(out[-1]), _bound_of(state)):
                    period_id = state.vocab._token_to_id.get(".")  # noqa: SLF001
                    if period_id is not None:
                        out.append(int(period_id))
                        update_open_brackets(
                            open_brackets, int(period_id), len(out) - 1, state, out
                        )
                n_handoffs += 1
                # Đã chèn đủ câu kế → dừng (tránh matrix nối dính sau dấu chấm).
                if n_handoffs >= max_h:
                    break
                continue
            out.append(nxt)
            update_open_brackets(open_brackets, nxt, len(out) - 1, state, out)
        return out

    # ---- legacy channel blend (ar_matrix_core=False) ----
    active_parents = (
        _active_parent_set(state, generated)
        if config.ar_hlink_ctx_weight > 0 else frozenset()
    )
    qk = None
    if config.ar_qk_weight > 0:
        from nn_aivie.hca.attention_qk import ensure_directed_qk

        qk = ensure_directed_qk(state)
    spectral_stack = None
    if getattr(config, "ar_spectral_stack", False):
        from nn_aivie.hca.spectral_stack import ensure_spectral_stack

        spectral_stack = ensure_spectral_stack(
            state,
            d=int(getattr(config, "ar_spectral_stack_d", 32)),
            sidecar=getattr(state, "_spectral_stack_sidecar", None),
        )
    topic_seeds = [
        (t, 1.0) for t in dict.fromkeys(generated)
        if state.vocab.is_surface(t)
    ]
    use_pos = bool(getattr(config, "ar_ctx_positional", True))
    pos_lam = float(getattr(config, "ar_ctx_lambda", 0.85))
    pos_len = int(getattr(config, "ar_ctx_len", 12))
    from nn_aivie.hca.context_positional import content_token_ids, ensure_varlm

    ensure_varlm(
        state,
        sidecar=getattr(state, "_varlm_sidecar", None),
        fallbacks=["checkpoints/network_v11.varlm.pkl"],
    )
    content_ids = []
    if getattr(config, "ar_ctx_content_gate", True) or getattr(
        config, "ar_distant_content_weight", 0.0
    ) > 0:
        content_ids = content_token_ids(generated, state.vocab)

    for _ in range(max_steps):
        w0 = out[-1]
        use_ppmi = bool(getattr(config, "ar_wseq_ppmi", True))
        skip_struct = bool(getattr(config, "ar_wseq_skip_structural", True))
        if use_pos:
            from nn_aivie.hca.context_positional import (
                positional_readout_base,
                positional_wseq_micro,
            )

            base = positional_readout_base(
                state, out, vocab_size=vocab_size, lam=pos_lam, ctx_len=pos_len,
            )
            micro = positional_wseq_micro(
                state,
                out,
                vocab_size=vocab_size,
                lam=pos_lam,
                ctx_len=pos_len,
                use_ppmi=use_ppmi,
                skip_structural=skip_struct,
            )
        else:
            base = np.zeros(vocab_size, dtype=np.float64)
            for v, w in state.adjacency.row_readout(w0).items():
                if v < vocab_size and w > 0:
                    base[v] = w
            micro = np.zeros(vocab_size, dtype=np.float64)
            from nn_aivie.hca.context_positional import (
                _add_seq_row,
                is_structural_token,
            )

            src = w0
            if skip_struct and is_structural_token(state.vocab, w0):
                for tid in reversed(out):
                    if not is_structural_token(state.vocab, tid):
                        src = int(tid)
                        break
            _add_seq_row(
                micro,
                state,
                src,
                1.0,
                vocab_size=vocab_size,
                use_ppmi=use_ppmi,
            )
            mmax = micro.max() or 1.0
            micro = micro / mmax
        nw = float(getattr(config, "ar_ngram_weight", 0.0))
        if nw > 0 and getattr(state, "varlm", None) is not None:
            from nn_aivie.hca.context_positional import ngram_varlm_micro

            micro = micro + nw * ngram_varlm_micro(
                state, out, vocab_size=vocab_size
            )
        o2w = float(getattr(config, "ar_order2_cont_weight", 0.0))
        if o2w > 0 and state.parents is not None:
            from nn_aivie.hca.context_positional import order2_micro

            micro = micro + o2w * order2_micro(
                state,
                out,
                vocab_size=vocab_size,
                min_hi=float(getattr(config, "ar_seq_order2_min", 2.0)),
            )
        dw = float(getattr(config, "ar_distant_content_weight", 0.0))
        if dw > 0 and content_ids:
            from nn_aivie.hca.context_positional import distant_content_micro

            micro = micro + dw * distant_content_micro(
                state,
                content_ids,
                vocab_size=vocab_size,
                mode=str(getattr(config, "ar_distant_mode", "coact")),
            )
        if qk is not None:
            qs = np.maximum(qk.scores(w0), 0.0)
            if qs.shape[0] != vocab_size:
                qs = np.pad(qs, (0, max(0, vocab_size - qs.shape[0])))[:vocab_size]
            qmax = qs.max() or 1.0
            micro = micro + config.ar_qk_weight * (qs / qmax)
        scores = base * (config.hybrid_micro_floor + config.hybrid_micro_weight * micro)
        if config.ar_continuation_topic_weight > 0 and topic_seeds:
            topic = spreading_activation_read(
                w0, topic_seeds, state.adjacency, config, vocab_size
            )
            tmax = topic.max() or 1.0
            scores = scores * (
                config.ar_continuation_topic_floor
                + config.ar_continuation_topic_weight * topic / tmax
            )
        if spectral_stack is not None:
            from nn_aivie.hca.spectral_stack import (
                forward_spectral_stack,
                spectral_forward_kwargs,
            )

            emb = getattr(state, "embedding", None)
            emb_mat = getattr(emb, "matrix", None) if emb is not None else None
            a_out = forward_spectral_stack(
                spectral_stack,
                history_ids=out,
                candidate_ids=None,
                entity_ids=(),
                emb_matrix=emb_mat,
                **spectral_forward_kwargs(config, state, out),
            )
            if a_out.shape[0] != vocab_size:
                a_out = np.pad(a_out, (0, max(0, vocab_size - a_out.shape[0])))[:vocab_size]
            multiply_scores_by_boost(
                scores,
                a_out,
                floor=float(getattr(config, "ar_spectral_stack_floor", 0.15)),
                weight=float(getattr(config, "ar_spectral_stack_weight", 0.55)),
            )
        if content_ids and getattr(config, "ar_ctx_content_gate", True):
            from nn_aivie.hca.context_positional import apply_content_context_gate

            apply_content_context_gate(
                scores,
                state,
                content_ids,
                floor=float(getattr(config, "ar_ctx_content_gate_floor", 0.12)),
            )
        scores = mask_non_surface(scores, state.vocab)
        scores = apply_repetition_penalty(scores, out, state.vocab, prompt_len=prompt_len)
        apply_frame_bracket_gate(
            scores, state, out, open_brackets, config, vocab_size,
            active_parents=active_parents,
        )
        _, ans = _split_frame(out, state.vocab)
        if eos_id < len(scores):
            if not ans:
                scores[eos_id] = -np.inf
            elif state.adjacency.get_readout(w0, eos_id) >= config.hybrid_eos_threshold:
                fm = scores[np.isfinite(scores)].max() if np.any(np.isfinite(scores)) else 0.0
                scores[eos_id] = fm * 1.5 + 1.0
            else:
                scores[eos_id] = -np.inf
        if not np.any(np.isfinite(scores)):
            break
        nxt = int(np.argmax(scores)) if config.hybrid_temperature <= 0 else _sample(scores, config, rng)
        if nxt == eos_id:
            break
        out.append(nxt)
        update_open_brackets(open_brackets, nxt, len(out) - 1, state, out)
    return out


def _hub_guarded_completion(
    state: GraphState, q_ids: list[int], generated: list[int], config: HCAConfig
) -> tuple[int, ...] | None | bool:
    """
    Dùng nơ-ron hub để chặn bịa fact + route novel filler sang đáp NHẮC ĐÚNG thực thể hỏi.

    Câu hỏi thuộc họ mẫu (vd "Nhà ___ thành lập năm nào") với filler LẠ (Trần, chưa nối): đáp
    hợp lệ **bắt buộc phải chứa chính slot_x** (Trần). Nhờ đó:
      - có statement/QA về đúng thực thể (vd "Thủ đô của Ý là Rome") → route tới nó (case con
        kiến, đúng thực thể);
      - chỉ có fact của filler khác ("Nhà Lý… 1009") hay câu khác khung ("Đảng… 1930") → loại
        hết → KHÔNG bịa (đáp rỗng), đúng G4 Verify.

    Trả: `False` = không thuộc họ mẫu novel (dùng luồng thường); `()` = không có đáp đúng thực
    thể → không bịa; `tuple` = đáp nhắc đúng thực thể.
    """
    ps = state.parents
    if not getattr(ps, "hubs", None):
        return False
    q_seq = _content_ids(q_ids or generated, state.vocab, skip=set())
    hit = ps.hub_for_query(tuple(q_seq))
    if hit is None:
        return False
    hub, slot_x = hit
    if slot_x in hub.fillers:
        return False  # filler đã biết → matching thường xử lý đúng, không cần guard
    ranked = ps.rank_matches(
        q_ids or generated, state.vocab, state.embedding,
        fuzzy_topk=config.hybrid_fuzzy_topk,
        fuzzy_min_sim=config.hybrid_fuzzy_min_sim,
        fuzzy_scale=config.hybrid_fuzzy_weight,
        top_n=8, qa_completion_mode=True,
    )
    for pidx, score in ranked:
        if score < config.hybrid_parent_min_score:
            break
        comp = ps.completion(pidx)
        if slot_x in comp:  # đáp nhắc đúng thực thể được hỏi
            return tuple(comp)
    return ()  # không đáp nào nhắc đúng thực thể → không bịa fact filler khác


def logits_topk_entropy(scores: np.ndarray, k: int) -> float:
    """Entropy (nats) của softmax trên top-k logits hữu hạn — P5.3/P5.4."""
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    kk = min(int(k), int(np.sum(np.isfinite(finite))))
    if kk <= 1:
        return 0.0
    top = np.argpartition(-finite, kk - 1)[:kk]
    logits = finite[top]
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def _sample(scores: np.ndarray, config: HCAConfig, rng: np.random.Generator) -> int:
    """Top-k (rồi tuỳ chọn nucleus top-p) → softmax(T) → multinomial.

    P5: nếu entropy top-k cao (phẳng) → argmax hoặc hạ T.
    """
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    k = min(int(getattr(config, "hybrid_top_k", 12)), int(np.sum(np.isfinite(finite))))
    if k <= 0:
        return int(np.argmax(scores))
    # P5.4 peakiness / entropy gate
    if bool(getattr(config, "ar_sharp_gate", True)):
        H = logits_topk_entropy(finite, k)
        h_max = float(getattr(config, "ar_sharp_entropy_max", 2.2))
        if H > h_max:
            if bool(getattr(config, "ar_sharp_fallback_argmax", True)):
                return int(np.argmax(finite))
            # hạ T rồi sample
            temp = max(
                1e-6,
                float(config.hybrid_temperature)
                * float(getattr(config, "ar_sharp_temp_scale", 0.45)),
            )
        else:
            temp = max(1e-6, float(config.hybrid_temperature))
    else:
        temp = max(1e-6, float(config.hybrid_temperature))
    top = np.argpartition(-finite, k - 1)[:k]
    top = top[np.argsort(-finite[top])]
    logits = finite[top] / temp
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    # Nucleus (top-p) trên tập đã cắt top-k.
    top_p = float(getattr(config, "hybrid_top_p", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        cdf = np.cumsum(probs)
        cut = int(np.searchsorted(cdf, top_p, side="left")) + 1
        cut = max(1, min(cut, len(probs)))
        probs = probs[:cut]
        probs /= probs.sum()
        top = top[:cut]
    return int(top[rng.choice(len(top), p=probs)])


@dataclass
class _SegmentPlan:
    """A2/A3: mỗi fact = một segment; entity register = keys cho entity attention."""

    segments: list[list[int]]
    entity_ids: tuple[int, ...]
    flat_agenda: list[int]
    # Style segs (pivot+payoff) — soft decode trừ khi hard_copy / pivot_lock.
    hard_copy_segs: frozenset[int] = field(default_factory=frozenset)
    # Pivot glue ngắn (`thế nhưng`) — luôn emit liền (tránh `tự chủ. cuộc…`).
    pivot_lock_segs: frozenset[int] = field(default_factory=frozenset)


def _surface_content_ids(ids: list[int], vocab) -> list[int]:
    out: list[int] = []
    for t in ids:
        if vocab.is_surface(t):  # TK_012 attr
            out.append(t)
    return out


def _prefer_atomic_facts(
    picked: list[int], ps: ParentStore, vocab
) -> list[int]:
    """Bỏ fact ghép dài nếu các fact ngắn khác đã bao phủ (tránh lặp compound)."""
    if len(picked) <= 1:
        return picked
    comps = [
        (pi, list(ps.completion(pi)), set(_surface_content_ids(list(ps.completion(pi)), vocab)))
        for pi in picked
    ]
    out: list[int] = []
    for pi, emit, cs in comps:
        drop = False
        if any(vocab.token_of(t) == "," for t in emit):
            subsumed = sum(
                1 for pj, _, cs2 in comps
                if pj != pi and len(cs2) >= 3 and cs2 <= cs
            )
            if subsumed >= 2:
                drop = True
        if not drop:
            for pj, _, cs2 in comps:
                if pi == pj or len(cs2) < 3:
                    continue
                jacc = len(cs & cs2) / max(1, len(cs | cs2))
                if jacc > 0.65 and len(cs) > len(cs2) + 2:
                    drop = True
                    break
        if not drop:
            out.append(pi)
    return out if out else picked


def _plan_from_multizone(
    state: GraphState,
    query_ids: list[int],
    config: HCAConfig,
    *,
    emitted_ids: list[int] | None = None,
) -> _SegmentPlan | None:
    """FACT head: zone_retrieve (cap-span) → 1 segment/zone CONTENT. None = không đủ zone."""
    if not getattr(config, "ar_multizone", False):
        return None
    # lazy — tránh vòng import multizone ↔ generate_hebbian
    from nn_aivie.hca.multizone import zone_retrieve

    ps = state.parents
    if ps is None:
        return None
    emit = emitted_ids if emitted_ids is not None else query_ids
    zones = zone_retrieve(
        state, list(query_ids),
        gidf_min=config.ar_multizone_gidf_min,
        gidf_solo=config.ar_multizone_gidf_solo,
        beta=config.ar_multizone_beta,
        facts_top=config.ar_multizone_facts_top,
        theme_weight=config.compose_theme_weight,
        cover_min=getattr(config, "ar_multizone_cover_min", 0.5),
    )
    cos_min = config.ar_multizone_cos_min
    segments: list[list[int]] = []
    ent: list[int] = []
    seen_pi: set[int] = set()
    for z in zones:
        if not z.is_content(cos_min) or not z.facts:
            continue
        ent.extend(z.tokens)
        # P44 — rerank fact trong zone: overlap nội dung prompt + gidf trung bình
        q_set = set(_content_ids(query_ids, state.vocab, skip=set()))
        ranked_facts = list(z.facts)

        def _fact_prompt_score(item: tuple[int, float, float]) -> float:
            pi, cos, w = item
            try:
                rec = ps.parents[int(pi)]
            except Exception:
                return float(cos)
            bag = set()
            for attr in ("match_ids", "emit_ids", "answer_ids"):
                ids = getattr(rec, attr, ()) or ()
                for t in ids:
                    if state.vocab.is_surface(int(t)) and int(t) in q_set:
                        bag.add(int(t))
            ov = len(bag) / max(1, len(q_set))
            return float(cos) * (0.35 + 0.65 * ov) + 0.15 * float(w)

        ranked_facts.sort(key=_fact_prompt_score, reverse=True)
        for pi, _cos, _w in ranked_facts:
            if pi in seen_pi:
                continue
            seen_pi.add(pi)
            seg = _agenda_ids_from_parent_index(
                ps, pi, emit, state.vocab, bound=_bound_of(state))
            if seg:
                segments.append(seg)
                break  # 1 segment / zone
    if not segments:
        return None
    q_content = _content_ids(query_ids, state.vocab, skip=set())
    entity_ids = tuple(dict.fromkeys(
        list(ent) + list(ps.entity_register_ids(tuple(q_content)))
    ))
    flat: list[int] = []
    for seg in segments:
        flat.extend(seg)
    return _SegmentPlan(segments, entity_ids, flat)


def _build_segment_plan(
    state: GraphState,
    query_ids: list[int],
    config: HCAConfig,
    *,
    max_facts: int = 3,
    emitted_ids: list[int] | None = None,
    trim_prefix: bool | None = None,
) -> _SegmentPlan:
    ps = state.parents
    if ps is None:
        return _SegmentPlan([], (), [])
    q_content = _content_ids(query_ids, state.vocab, skip=set())
    entity_ids = ps.entity_register_ids(tuple(q_content))
    focus = ps.top_focus_tokens(tuple(q_content), k=2, vocab=state.vocab) if q_content else ()
    segments: list[list[int]] = []

    emit = emitted_ids if emitted_ids is not None else query_ids
    qa_frame = _prompt_has_qa_frame(emit, state.vocab)

    # Multi-zone FACT head (opt-in) — nhiều entity → nhiều segment; fallback đường cũ.
    mz = _plan_from_multizone(state, query_ids, config, emitted_ids=emit)
    if mz is not None and mz.flat_agenda:
        return mz

    # FIX GROUNDING (TK_042 2026-07-21, đo local harness probe_grounding_local): đường CHÍNH
    # `rank_matches` XẾP HẠNG THEME-BLIND (coverage×priority, tiebreak power) → TRÔI entity sang
    # node prominent / va-chạm-bề-mặt: Thăng Long→Gia Long, Ngô Quyền→Đinh Tiên Hoàng, Trần Hưng
    # Đạo→Lý Chiêu Hoàng. `facts_about` có tín hiệu THEME (nơ-ron chủ-đề, trọng số theo vị-trí
    # entity trong fact) → SUBJECT-AWARE, đo local đúng 4/5 (Bạch Đằng→Trận Bạch Đằng, Thăng
    # Long→Thăng Long…). → ƯU TIÊN facts_about-theme cho truy vấn CÓ focus thực thể; rank_matches
    # (pattern / QA-completion) làm FALLBACK khi không có fact chủ-đề.
    # §12.149 — frame Hỏi/Đáp: ưu tiên cha QA (đáp) trước facts_about tường thuật.
    pi_pick: int | None = None
    seg: list[int] = []
    if qa_frame:
        ranked = ps.rank_matches(
            query_ids, state.vocab, state.embedding,
            fuzzy_topk=config.hybrid_fuzzy_topk,
            fuzzy_min_sim=config.hybrid_fuzzy_min_sim,
            fuzzy_scale=config.hybrid_fuzzy_weight,
            min_priority=config.generation_min_priority,
            qa_completion_mode=True,
            top_n=16,
        )
        ranked = [(p, s) for p, s in ranked if s >= config.hybrid_parent_min_score]
        qa_ranked = [(p, s) for p, s in ranked if getattr(ps.parents[p], "is_qa", False)]
        _tp = _qa_trim_prefix(config) if trim_prefix is None else bool(trim_prefix)
        pi_pick, seg, _ = _pick_parent_agenda(
            ps, qa_ranked or ranked, emit, state.vocab, trim_prefix=_tp,
            bound=_bound_of(state),
        )

    if (pi_pick is None or not seg) and focus and config.compose_theme_weight > 0:
        tf = _prefer_atomic_facts(
            list(ps.facts_about(
                focus, top_k=max(max_facts, 8), vocab=state.vocab,
                theme_weight=config.compose_theme_weight,
            )),
            ps, state.vocab,
        )
        for cand in tf:
            if qa_frame and not getattr(ps.parents[cand], "is_qa", False):
                continue
            cand_seg, fully = _agenda_from_parent_index(
                ps, cand, emit, state.vocab, bound=_bound_of(state),
                trim_prefix=(_qa_trim_prefix(config) if trim_prefix is None
                             else bool(trim_prefix)),
            )
            if cand_seg and not fully:
                pi_pick = cand
                seg = cand_seg
                break

    if pi_pick is None or not seg:  # không có fact chủ-đề → pattern/QA qua rank_matches
        ranked = ps.rank_matches(
            query_ids, state.vocab, state.embedding,
            fuzzy_topk=config.hybrid_fuzzy_topk,
            fuzzy_min_sim=config.hybrid_fuzzy_min_sim,
            fuzzy_scale=config.hybrid_fuzzy_weight,
            min_priority=config.generation_min_priority,
            qa_completion_mode=True,
            top_n=12,
        )
        ranked = [(p, s) for p, s in ranked if s >= config.hybrid_parent_min_score]
        if qa_frame:
            qa_ranked = [
                (p, s) for p, s in ranked if getattr(ps.parents[p], "is_qa", False)
            ]
            ranked = qa_ranked or ranked
        _tp = _qa_trim_prefix(config) if trim_prefix is None else bool(trim_prefix)
        pi_pick, seg, _ = _pick_parent_agenda(
            ps, ranked, emit, state.vocab, trim_prefix=_tp, bound=_bound_of(state),
        )

    if pi_pick is not None and seg:
        segments.append(seg)

    flat: list[int] = []
    for seg in segments:
        flat.extend(seg)
    return _SegmentPlan(segments, entity_ids, flat)


_CONV_PRIORITY_BAND = (0.25, 0.35)  # pool văn nói R3b (build_conv_pool.py: priority=0.3)
_PARAGRAPH_PRIORITY_BAND = (0.355, 0.395)  # TK_035 (build_paragraph_pool.py: priority=0.36)


def _longest_suffix_cut(
    src: list[int],
    q_folds: list[int],
    fold,
    *,
    min_k: int = 1,
) -> tuple[int, int]:
    """(cut_index, matched_len) — hậu tố dài nhất khớp query (fold-group)."""
    if not q_folds or not src:
        return -1, 0
    src_f = [fold(t) for t in src]
    best = -1
    best_k = 0
    lo = max(1, min_k)
    for k in range(min(len(q_folds), len(src)), lo - 1, -1):
        suffix = q_folds[-k:]
        for start in range(len(src_f) - k + 1):
            if src_f[start:start + k] == suffix:
                cut = start + k - 1
                if cut > best:
                    best = cut
                    best_k = k
                break
        if best >= 0:
            return best, best_k
    return -1, 0


def _query_content_folds(query_ids: list[int], vocab) -> list[int]:
    """Token nội dung (bề mặt, không dấu) → fold-group ids — dùng cho suffix-cut."""
    fold = vocab.fold_group_id
    return [
        fold(t)
        for t in query_ids
        if vocab.is_surface(t)
    ]


def _agenda_surface_tail(tail: list[int], vocab) -> list[int]:
    """Token sau index-cut — giữ dấu câu (`.`) để AR đóng câu, không lọc bỏ rồi fallback."""
    return [
        t
        for t in tail
        if vocab.is_surface(t)
    ]


def _qa_question_ids(rec, vocab) -> list[int]:
    """Phần câu hỏi của cha QA (trước Đáp:) — để đo khớp prompt."""
    from nn_aivie.core.tokenizer import DAP_LABEL

    src = list(getattr(rec, "source_ids", None) or ())
    dap = vocab._token_to_id.get(DAP_LABEL)  # noqa: SLF001
    if dap is not None and src and int(dap) in src:
        return list(src[: src.index(int(dap))])
    return list(getattr(rec, "match_ids", None) or src)


def _is_agenda_content_token(vocab, tid: int) -> bool:
    """Token có nghĩa để khớp agenda — bỏ frame / punct / Ġ trống / chữ-số."""
    from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL, MARK_SPACE

    tid = int(tid)
    if not vocab.is_surface(tid):
        return False
    tok = vocab.token_of(tid)
    if tok in (DAP_LABEL, HOI_LABEL, MARK_SPACE, "Ġ", "Ċ", ""):
        return False
    bare = tok.lstrip(MARK_SPACE).lstrip("Ċ")
    if not bare or bare.isdigit():
        return False
    # §13.56 — thay danh sách dấu câu chép tay bằng `is_structural_token`: nó xét LỚP KÝ TỰ
    # (`all(not ch.isalnum())`), tức phủ MỌI dấu câu kể cả cái không ai nghĩ tới, và không
    # phải tri thức từ vựng tiếng Việt. Danh sách cũ (11 dấu) vừa hẹp vừa phải bảo trì tay.
    from nn_aivie.hca.context_positional import is_structural_token

    if is_structural_token(vocab, tid):
        return False
    return True


def _agenda_content_folds(token_ids: list[int], vocab) -> list[int]:
    """Fold-group nội dung — không để `? Ġ Đáp:`假 khớp suffix."""
    fold = vocab.fold_group_id
    return [
        fold(int(t))
        for t in token_ids
        if _is_agenda_content_token(vocab, int(t))
    ]


def _fold_overlap_count(q_folds: list[int], token_ids: list[int], vocab) -> int:
    """Số fold query xuất hiện trong token_ids (chỉ đếm token nội dung phía bag)."""
    if not q_folds or not token_ids:
        return 0
    fold = vocab.fold_group_id
    bag = {
        fold(int(t))
        for t in token_ids
        if _is_agenda_content_token(vocab, int(t))
    }
    return sum(1 for f in q_folds if f in bag)


def _agenda_min_match(q_folds: list[int], *, min_suffix_tokens: int = 1) -> int:
    if len(q_folds) >= 2:
        return min(3, len(q_folds))
    return max(1, int(min_suffix_tokens))


# Từ chức năng — không dùng để chứng thực cha khớp topic
_AGENDA_STOP_SURFACES = frozenset({
    "trong", "thời", "kỳ", "là", "của", "và", "một", "các", "có", "được",
    "người", "này", "cho", "từ", "khi", "đến", "đã", "sẽ", "cũng", "những",
    "với", "để", "như", "vào", "ra", "lại", "nên", "bị", "bởi", "hay",
    "hoặc", "rất", "nhiều", "ít", "hơn", "nhất", "đó", "đây", "kia",
    "nhân", "dân", "ta", "tôi", "bạn", "họ", "mà", "thì", "còn", "về",
})


def _specific_prompt_surfaces(token_ids: list[int], vocab) -> set[str]:
    from nn_aivie.core.tokenizer import MARK_SPACE

    out: set[str] = set()
    for t in token_ids:
        if not _is_agenda_content_token(vocab, int(t)):
            continue
        bare = vocab.token_of(int(t)).lstrip(MARK_SPACE).lstrip("Ċ").lower()
        if bare and bare not in _AGENDA_STOP_SURFACES:
            out.add(bare)
    return out


def _parent_has_specific_overlap(
    emitted_ids: list[int],
    parent_ids: list[int],
    vocab,
    *,
    min_hit: int = 2,
) -> bool:
    """Cha phải chứa ≥min_hit bề mặt topic của prompt (không chỉ «Trong thời kỳ»)."""
    need = _specific_prompt_surfaces(emitted_ids, vocab)
    if len(need) < 2:
        return True
    from nn_aivie.core.tokenizer import MARK_SPACE

    bag = {
        vocab.token_of(int(t)).lstrip(MARK_SPACE).lstrip("Ċ").lower()
        for t in parent_ids
        if _is_agenda_content_token(vocab, int(t))
    }
    hit = sum(1 for s in need if s in bag)
    return hit >= min(int(min_hit), len(need))


def _prompt_has_qa_frame(emitted_ids: list[int], vocab) -> bool:
    from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL

    frame = {
        vocab._token_to_id.get(x)  # noqa: SLF001
        for x in (HOI_LABEL, DAP_LABEL)
    }
    frame.discard(None)
    return any(int(t) in frame for t in emitted_ids)


def _agenda_from_parent_index(
    ps: ParentStore,
    parent_idx: int,
    emitted_ids: list[int],
    vocab,
    *,
    min_suffix_tokens: int = 1,
    trim_prefix: bool = True,
    bound=None,   # §13.56 — biên ĐƠN VỊ học được; None ⇒ KHÔNG nhận mốc kết, không đoán ngầm
) -> tuple[list[int], bool]:
    """
    Agenda A3 theo INDEX trong `source_ids` — cùng logic pattern-completion.

    Returns:
        (agenda, source_fully_matched): `source_fully_matched=True` khi prompt đã khớp
        hết source (không còn token nào kể cả dấu) → caller thử rank kế. Nếu còn `.`
        hoặc nội dung → agenda = phần đó, KHÔNG fallback full emit.

    §12.149 — khớp agenda chỉ trên fold nội dung (bỏ Hỏi/Đáp/?/Ġ); cha QA chỉ
    lấy answer khi hỏi khớp; cắt suffix yếu → bỏ cha.
    """
    rec = ps.parents[parent_idx]
    q_folds = _agenda_content_folds(emitted_ids, vocab)
    need = _agenda_min_match(q_folds, min_suffix_tokens=min_suffix_tokens)
    # Cha QA: chỉ lấy answer_ids khi câu hỏi cha khớp đủ prompt
    if getattr(rec, "is_qa", False) and rec.answer_ids:
        qq_ids = _qa_question_ids(rec, vocab)
        qq_folds = _agenda_content_folds(qq_ids, vocab)
        q_ov = _fold_overlap_count(q_folds, qq_ids, vocab)
        qq_cover = float(q_ov) / float(max(1, len(qq_folds)))
        qa_frame = _prompt_has_qa_frame(emitted_ids, vocab)
        # Statement chỉ nhận QA khi phủ phần lớn câu hỏi (tránh Tổ→QA «dời đô về đâu»)
        qa_ok = q_ov >= int(need) and (
            qa_frame or qq_cover >= 0.5 or q_ov >= max(int(need), len(qq_folds) - 1)
        )
        if qa_ok:
            ans = _surface_content_ids(list(rec.answer_ids), vocab)
            if ans:
                from nn_aivie.core.tokenizer import MARK_SPACE

                def _sk(tid: int) -> str:
                    return vocab.token_of(int(tid)).lstrip(MARK_SPACE).lstrip("Ċ")

                def _ends_u(tid: int) -> bool:
                    """§13.56 — mốc KẾT ĐƠN VỊ, đọc từ P(EOS|w) chứ không so với `"."`.

                    `bound=None` ⇒ trả False: không có tín hiệu thì **không nhận** mốc nào,
                    thay vì lặng lẽ quay về so ký tự. Mọi đường gọi ở đường sinh đều truyền.
                    """
                    return bool(bound is not None and bound.ends_unit(int(tid)))

                # §13.56 — cùng lý do như `_is_agenda_content_token`: xét lớp ký tự, không
                # chép danh sách dấu câu. Nhận `tid` để hỏi được `is_structural_token`.
                from nn_aivie.hca.context_positional import is_structural_token

                def _key_ok_id(tid: int) -> bool:
                    bare = _sk(int(tid))
                    return (bool(bare) and (not bare.isdigit())
                            and not is_structural_token(vocab, int(tid)))

                # Câu đáp đầu (WH + tránh câu 2 «Vùng đất…»)
                first_sent: list[int] = []
                for t in ans:
                    first_sent.append(int(t))
                    if _ends_u(int(t)):
                        break
                ans = first_sent
                prompt_keys = {
                    _sk(t) for t in emitted_ids if _key_ok_id(t)
                }
                if qa_frame and trim_prefix:
                    # Route W_skip primary — GIỮ NGUYÊN §12.149 (đo được: cho
                    # `Đáp: Thăng Long.` sạch; đổi sang cắt-tiền-tố làm salad, §13.22).
                    last_hit = -1
                    for i, t in enumerate(ans):
                        if _key_ok_id(int(t)) and _sk(int(t)) in prompt_keys:
                            last_hit = i
                    tail = ans[last_hit + 1 :] if last_hit >= 0 else list(ans)
                else:
                    # Statement trúng QA — bỏ trùng prompt, giữ câu đầy đủ
                    tail = list(ans)
                # §13.22 — GIỮ NGUYÊN thứ tự & nội dung của `tail`; KHÔNG lọc bỏ token chỉ
                # vì nó trùng prompt.
                #
                # ⚠️ Bản cũ lọc trùng prompt trên TOÀN câu: `Cá voi là gì` + đáp
                # `Cá voi là động vật có vú lớn nhất .` ⇒ rụng `Cá voi là`, còn
                # `động vật có vú lớn nhất .` — mất chủ ngữ. Đáp đúng ngữ pháp tiếng Việt
                # **phải** nhắc lại chủ ngữ; "trùng prompt" không đồng nghĩa "thừa".
                # Phần trùng thật sự thừa (tiền tố tiếp diễn) đã bị `pre` cắt ở nhánh QA
                # phía trên — chỗ này không được cắt thêm lần nữa.
                if trim_prefix:
                    # Route W_skip primary (§12.149): agenda phải CHẶT — bỏ token trùng
                    # prompt để vòng phiếu không phải phủ lại phần đã có.
                    novel = [
                        int(t) for t in tail
                        if _ends_u(int(t))
                        or (_key_ok_id(int(t)) and _sk(int(t)) not in prompt_keys)
                    ]
                    if not novel:
                        novel = [
                            int(t) for t in ans
                            if _ends_u(int(t))
                            or (_key_ok_id(int(t)) and _sk(int(t)) not in prompt_keys)
                        ]
                else:
                    # Route matrix/parent: dội agenda khá trực tiếp ⇒ lọc trùng prompt làm
                    # RỤNG CHỦ NGỮ (`Cá voi là gì` → `là động vật…`). Đáp đúng ngữ pháp
                    # tiếng Việt phải nhắc lại chủ ngữ — "trùng prompt" ≠ "thừa".
                    novel = [
                        int(t) for t in tail
                        if _key_ok_id(int(t)) or _ends_u(int(t))
                    ]
                return (novel if novel else list(ans)), False
        if qa_frame:
            return [], True
    src = list(rec.source_ids) if rec.source_ids else []
    if src and q_folds:
        fold = vocab.fold_group_id
        cut, mk = _longest_suffix_cut(
            src, q_folds, fold, min_k=min_suffix_tokens,
        )
        if cut >= 0:
            if int(mk) < int(need):
                return [], True
            if not _parent_has_specific_overlap(emitted_ids, src, vocab):
                return [], True
            tail = src[cut + 1:]
            if not tail:
                return [], True
            return _agenda_surface_tail(tail, vocab), False
    parent_bag = list(rec.match_ids or ()) or list(rec.emit_ids or ()) or list(
        _completion_from_parent(ps, parent_idx)
    )
    if _fold_overlap_count(q_folds, parent_bag, vocab) < int(need):
        return [], True
    if not _parent_has_specific_overlap(emitted_ids, parent_bag, vocab):
        return [], True
    # Completion dump chỉ khi fact mở cùng chủ đề prompt (lead trùng) —
    # tránh «Hai Bà…» khi prompt là «nhân dân ta».
    from nn_aivie.core.tokenizer import MARK_SPACE

    def _sk2(tid: int) -> str:
        return vocab.token_of(int(tid)).lstrip(MARK_SPACE).lstrip("Ċ")

    def _lead_surf(ids: list[int], k: int = 2) -> list[str]:
        out: list[str] = []
        for t in ids:
            if not _is_agenda_content_token(vocab, int(t)):
                continue
            bare = _sk2(int(t)).lower()
            if bare:
                out.append(bare)
            if len(out) >= k:
                break
        return out

    comp = list(_completion_from_parent(ps, parent_idx))
    if not comp:
        return [], True
    prompt_lead = _lead_surf(list(emitted_ids), 2)
    comp_lead = _lead_surf(comp, 2)
    spec = _specific_prompt_surfaces(emitted_ids, vocab)
    lead_ok = bool(prompt_lead) and (
        comp_lead == prompt_lead
        or (bool(comp_lead) and set(comp_lead) <= (set(prompt_lead) | spec))
    )
    if not lead_ok:
        return [], True
    first_sent: list[int] = []
    for t in comp:
        if not vocab.is_surface(int(t)):
            continue
        first_sent.append(int(t))
        if _sk2(int(t)) == ".":
            break
    # Bỏ đúng lead trùng prompt — giữ «Quốc»/«Hà Nội» (không lọc mọi từ prompt)
    #
    # ⚠️ §13.22 — CHỈ áp cho prompt TIẾP DIỄN. Với frame `Hỏi:…Đáp:` thì completion là
    # câu ĐỘC LẬP và lead chính là CHỦ NGỮ của nó: `Hỏi: Cá voi là gì Đáp:` +
    # `Cá voi là động vật có vú lớn nhất .` bị bỏ lead ⇒ `là động vật có vú lớn nhất .`
    # — mất chủ ngữ, câu không đứng được. Đây là lần thứ BA cùng một gốc trong §12.149:
    # luật dành cho prompt tiếp diễn bị áp lên frame QA (hai chỗ kia ở §13.22 phía trên).
    n_lead = (
        0
        if (_prompt_has_qa_frame(list(emitted_ids), vocab) and not trim_prefix)
        else len(comp_lead)
    )
    seen = 0
    novel: list[int] = []
    for t in first_sent:
        bare = _sk2(int(t))
        if _is_agenda_content_token(vocab, int(t)) and bare and not bare.isdigit():
            if seen < n_lead:
                seen += 1
                continue
        if bare == ".":
            novel.append(int(t))
            continue
        if bare and not bare.isdigit():
            novel.append(int(t))
    if len(novel) < 2:
        return [], True
    return novel, False


def _agenda_ids_from_parent_index(
    ps: ParentStore,
    parent_idx: int,
    emitted_ids: list[int],
    vocab,
    *,
    min_suffix_tokens: int = 1,
    bound=None,   # §13.56
) -> list[int]:
    """Wrapper — chỉ agenda (không fully_matched); dùng khi không cần skip rank."""
    agenda, _fully = _agenda_from_parent_index(
        ps, parent_idx, emitted_ids, vocab, min_suffix_tokens=min_suffix_tokens,
        bound=bound,
    )
    return agenda


def _pick_parent_agenda(
    ps: ParentStore,
    ranked: list[tuple[int, float]],
    emitted_ids: list[int],
    vocab,
    *,
    trim_prefix: bool = True,
    bound=None,   # §13.56
) -> tuple[int | None, list[int], tuple[int, ...]]:
    """
    Chọn cha đầu tiên trong rank có phần còn lại sau index-cut.
    Cha khớp hoàn toàn (tail rỗng) → bỏ qua, thử rank kế.
  """
    for pi, _sc in ranked:
        agenda, fully = _agenda_from_parent_index(
            ps, pi, emitted_ids, vocab, trim_prefix=trim_prefix, bound=bound,
        )
        if fully:
            continue
        return pi, agenda, _completion_from_parent(ps, pi)
    return None, [], ()


def _conv_pool_indices(ps: ParentStore) -> frozenset[int]:
    """Lazy: tập idx cha thuộc POOL VĂN NÓI (two-tier — chỉ đọc ở route continuation)."""
    cached = getattr(ps, "_conv_pool_cache", None)
    if cached is not None and cached[0] == len(ps.parents):
        return cached[1]
    lo, hi = _CONV_PRIORITY_BAND
    pool = frozenset(
        i for i, rec in enumerate(ps.parents) if lo <= rec.priority <= hi
    )
    ps._conv_pool_cache = (len(ps.parents), pool)
    return pool


def _paragraph_pool_indices(ps: ParentStore) -> frozenset[int]:
    """TK_035 — pool đoạn 2–4 câu (priority≈0.36)."""
    cached = getattr(ps, "_paragraph_pool_cache", None)
    if cached is not None and cached[0] == len(ps.parents):
        return cached[1]
    lo, hi = _PARAGRAPH_PRIORITY_BAND
    pool = frozenset(
        i for i, rec in enumerate(ps.parents) if lo <= rec.priority <= hi
    )
    ps._paragraph_pool_cache = (len(ps.parents), pool)
    return pool


def _discourse_pool_indices(ps: ParentStore, *, include_paragraph: bool) -> frozenset[int]:
    pool = _conv_pool_indices(ps)
    if include_paragraph:
        pool = pool | _paragraph_pool_indices(ps)
    return pool


def _is_paragraph_parent(ps: ParentStore, pi: int) -> bool:
    lo, hi = _PARAGRAPH_PRIORITY_BAND
    return lo <= ps.parents[pi].priority <= hi


def _essay_pool_indices(ps: ParentStore, min_priority: float) -> frozenset[int]:
    """Cha essay/curated (two-tier sinh) — dùng cho pattern-completion mở lửng."""
    cached = getattr(ps, "_essay_pool_cache", None)
    key = (len(ps.parents), min_priority)
    if cached is not None and cached[0] == key:
        return cached[1]
    lo, hi = _CONV_PRIORITY_BAND
    pool = frozenset(
        i for i, rec in enumerate(ps.parents)
        if rec.priority >= min_priority and not (lo <= rec.priority <= hi)
    )
    ps._essay_pool_cache = (key, pool)
    return pool


def _pick_tail_from_ranked(
    ps: ParentStore,
    vocab,
    ranked: list[tuple[int, float]],
    q_folds: list[int],
    fold,
    *,
    min_score: float,
    min_suffix_tokens: int,
    prefer_long_paragraph: bool,
) -> list[int] | None:
    """Chọn đuôi từ ranked parents; TK_035 ưu tiên đuôi DÀI từ pool đoạn."""
    para_best: list[int] | None = None
    para_k = para_len = -1
    para_sc = -1.0
    short_best: list[int] | None = None
    short_len = 10**9
    short_sc = -1.0

    for pi, score in ranked:
        if score < min_score:
            break
        src = list(ps.parents[pi].source_ids)
        cut, matched_k = _longest_suffix_cut(
            src, q_folds, fold, min_k=min_suffix_tokens,
        )
        if cut < 0:
            continue
        tail = src[cut + 1 :]
        if len(tail) < 2 or len(_surface_content_ids(tail, vocab)) < 2:
            continue
        tlen = len(tail)
        if prefer_long_paragraph and _is_paragraph_parent(ps, pi):
            if (
                matched_k > para_k
                or (matched_k == para_k and tlen > para_len)
                or (matched_k == para_k and tlen == para_len and score > para_sc)
            ):
                para_best = tail
                para_k = matched_k
                para_len = tlen
                para_sc = score
            continue
        if tlen < short_len or (tlen == short_len and score > short_sc):
            short_best = tail
            short_len = tlen
            short_sc = score

    if prefer_long_paragraph and para_best is not None:
        return para_best
    return short_best


def _tail_from_ranked_parents(
    ps: ParentStore,
    vocab,
    ranked: list[tuple[int, float]],
    q_folds: list[int],
    fold,
    *,
    min_score: float,
    min_suffix_tokens: int,
    prefer_long_paragraph: bool = False,
) -> list[int] | None:
    return _pick_tail_from_ranked(
        ps, vocab, ranked, q_folds, fold,
        min_score=min_score,
        min_suffix_tokens=min_suffix_tokens,
        prefer_long_paragraph=prefer_long_paragraph,
    )


def _tail_from_pattern_templates(
    state: GraphState,
    q_folds: list[int],
    fold,
    *,
    min_suffix_tokens: int,
) -> list[int] | None:
    """Sidecar mẫu câu (R4b) — không qua ParentStore / phrase index."""
    templates = state.pattern_templates
    if not templates:
        return None
    best_tail: list[int] | None = None
    best_k = 0
    best_len = 10**9
    for src in templates:
        cut, matched_k = _longest_suffix_cut(list(src), q_folds, fold, min_k=min_suffix_tokens)
        if cut < 0:
            continue
        tail = src[cut + 1 :]
        if len(tail) < 2 or len(_surface_content_ids(tail, state.vocab)) < 2:
            continue
        tlen = len(tail)
        if matched_k > best_k or (matched_k == best_k and tlen < best_len):
            best_tail = list(tail)
            best_k = matched_k
            best_len = tlen
    return best_tail


def _pattern_completion_tail(
    state: GraphState,
    query_ids: list[int],
    config: HCAConfig,
    *,
    min_suffix_tokens: int,
) -> list[int] | None:
    """
    R4 — pattern-completion trên conv pool + essay tier (prompt mở không frame / trước W_seq).
    `min_suffix_tokens`: opener-route = 2 (cho phép khớp cụm bán phần); open-route = max(2, |query|).
    """
    ps = state.parents
    if ps is None:
        return None
    fold = state.vocab.fold_group_id
    q_content = [
        t for t in query_ids
        if state.vocab.is_surface(t)
    ]
    if not q_content:
        return None
    q_folds = [fold(t) for t in q_content]
    tail = _tail_from_pattern_templates(
        state, q_folds, fold, min_suffix_tokens=min_suffix_tokens,
    )
    if tail:
        return tail
    kw = dict(fuzzy_topk=0, top_n=8)
    prefer_long = config.paragraph_prefer_long_tail
    conv = _discourse_pool_indices(ps, include_paragraph=True)
    if conv:
        ranked = ps.rank_matches(
            query_ids, state.vocab, state.embedding,
            restrict_parents=conv, **kw,
        )
        min_sc = min(config.conv_pool_min_score, config.paragraph_pool_min_score)
        tail = _tail_from_ranked_parents(
            ps, state.vocab, ranked, q_folds, fold,
            min_score=min_sc,
            min_suffix_tokens=min_suffix_tokens,
            prefer_long_paragraph=prefer_long,
        )
        if tail:
            return tail
    essay = _essay_pool_indices(ps, config.generation_min_priority)
    if essay:
        ranked = ps.rank_matches(
            query_ids, state.vocab, state.embedding,
            restrict_parents=essay, **kw,
        )
        tail = _tail_from_ranked_parents(
            ps, state.vocab, ranked, q_folds, fold,
            min_score=config.pattern_completion_min_score,
            min_suffix_tokens=min_suffix_tokens,
        )
        if tail:
            return tail
    return None


def _conv_pool_tail(
    state: GraphState,
    query_ids: list[int],
    config: HCAConfig,
) -> list[int] | None:
    """
    R3b (TK_017 §14) + TK_035 — CONTINUATION trên pool văn nói / đoạn.
    First-match theo rank; đoạn ưu tiên đuôi dài (nhiều câu, dấu phẩy/chấm).
    """
    ps = state.parents
    if ps is None:
        return None
    pool = _discourse_pool_indices(ps, include_paragraph=True)
    if not pool:
        return None
    ranked = ps.rank_matches(
        query_ids, state.vocab, state.embedding,
        fuzzy_topk=0, top_n=15, restrict_parents=pool,
    )
    fold = state.vocab.fold_group_id
    q_content = [
        t for t in query_ids
        if state.vocab.is_surface(t)
    ]
    if not q_content:
        return None
    q_folds = [fold(t) for t in q_content]
    prefer_long = config.paragraph_prefer_long_tail
    min_sc = min(config.conv_pool_min_score, config.paragraph_pool_min_score)
    return _pick_tail_from_ranked(
        ps, state.vocab, ranked, q_folds, fold,
        min_score=min_sc,
        min_suffix_tokens=1,
        prefer_long_paragraph=prefer_long,
    )


def _knowledge_extra_facts_eligible(
    ps,
    q_content: tuple[int, ...],
    config: HCAConfig,
) -> bool:
    """TK_025: bồi fact khi query có THỰC THỂ phân biệt (entity_register_ids, idf/vị-trí — Luật 0).
    Thay gate WH cũ (`wh_anchor_clusters`) — không phân-loại câu-hỏi, chỉ hỏi 'có thực thể để
    truy hồi fact không'."""
    if not q_content:
        return False
    return bool(ps.entity_register_ids(tuple(q_content)))


@dataclass(frozen=True)
class ExtraFactRank:
    """Ứng viên fact bồi + điểm kích thích (TK_023 — viz/API)."""

    idx: int
    text: str
    stimulus_score: float
    rank: int
    picked: bool
    skip: str | None = None


def _knowledge_entity_span(
    ps, q_content: tuple[int, ...]
) -> tuple[tuple[int, ...], int] | None:
    """Entity trước copula/skeleton (gidf thấp nhất) + primary = token phải nhất span."""
    if len(q_content) < 2:
        return None
    gidfs = [ps._gidf(t) for t in q_content]
    split_i = min(range(len(q_content)), key=lambda i: gidfs[i])
    entity = q_content[:split_i] if split_i > 0 else q_content
    entity = tuple(t for t in entity if ps._gdf.get(t, 0) > 0)
    if not entity:
        return None
    return entity, entity[-1]


def _rank_extra_fact_candidates(
    state: GraphState,
    q_ids: list[int],
    entity: tuple[int, ...],
    primary: int,
    config: HCAConfig,
    anchor_ids: list[int] | None = None,
) -> list[tuple[int, float]]:
    ps = state.parents
    if ps is None:
        return []
    if getattr(config, "knowledge_fact_rank", "stimulus") == "stimulus":
        seed = frozenset(entity)
        pool = list(ps.facts_about(
            entity, top_k=config.knowledge_extra_facts + 6, vocab=state.vocab,
            min_priority=0.5,
            theme_weight=config.compose_theme_weight,
        ))
        # A1 (TK_024 §2 Nhánh A) — MỞ SANG GIA ĐÌNH THỰC THỂ: facts_about đòi cụm "Nhà Lý"
        # LIỀN KỀ → "Lý Công Uẩn dời đô" (chỉ có "Lý") không vào pool. theme_neuron(primary)
        # gom MỌI fact chứa primary (Lý) → vua/tướng nhà Lý lên được. Filter chứa-primary +
        # MMR ở tầng plan lo phần đúng-thực-thể. Cờ `knowledge_family_expand`.
        if getattr(config, "knowledge_family_expand", True):
            theme = ps.theme_neuron(primary, state.vocab)
            if theme is not None:
                have = set(pool)
                for pi, _w in theme.facts:
                    if pi not in have and ps.parents[pi].priority >= 0.5:
                        pool.append(pi)
                        have.add(pi)
        def _stim(pi: int) -> float:
            return ps.parent_stimulus_score(
                pi, q_ids, state.vocab,
                theme_anchor=primary,
                theme_weight=config.compose_theme_weight,
                seed_tokens=seed,
            )

        # TK_027: CỔNG NEO THỰC-THỂ — trường từ vựng của THỰC-THỂ query → fact khác domain
        # (lý-thuyết/citation/Hàn-Quốc, dot thấp) rớt. Hồ-sơ entity key CỤM title (không fold).
        # Bước 3: ƯU TIÊN neo = Entity:title CỦA QUERY (query_entity_field — sạch, độc lập primary
        # có thể nhiễm 'con cháu Lý sang Cao Ly'); fallback neo = câu đáp chính (base_out).
        if getattr(config, "knowledge_entity_resonance", True):
            co_layer = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
            afield = ps.query_entity_field(entity, co_layer, state.vocab)
            if not afield and getattr(config, "knowledge_entity_resonance_fallback", True):
                # title-match trượt ("Triều Lý") → để MẠNG tự tìm entity qua cộng-hưởng ngữ cảnh
                afield = ps.query_entity_resonance_field(entity, co_layer, state.vocab)
            if not afield and anchor_ids:
                anchor_content = tuple(_content_ids(anchor_ids, state.vocab, skip=set()))
                afield = ps.entity_anchor_field(
                    anchor_content, co_layer, state.vocab,
                    top_entities=getattr(config, "knowledge_entity_top", 6),
                )
            if afield:
                dot = {pi: ps.parent_field_dot(pi, afield) for pi in pool}
                mx = max(dot.values(), default=0.0)
                if mx > 0:
                    gate = getattr(config, "knowledge_entity_gate_frac", 0.1) * mx
                    survivors = [pi for pi in pool if dot[pi] > gate]
                    dropped = [pi for pi in pool if dot[pi] <= gate]
                    # TK_027 bước 3b: xếp survivor theo dot×stimulus — fact phải VỪA trung-tâm
                    # thực-thể query (dot) VỪA khớp câu hỏi (stimulus). "Lý Bạch"(thơ) dot THẤP với
                    # trường Nhà Lý → chìm khỏi top-K dù stimulus cao (lặp "Lý"). (cosine-margin bị
                    # probe bác: fact dynasty tốt cũng cosine thấp với query-entity.)
                    ranked = sorted(
                        survivors,
                        key=lambda pi: (-(dot[pi] * (_stim(pi) or 1.0)), -ps.parents[pi].power),
                    )
                    ranked += sorted(dropped, key=lambda pi: (-dot[pi], -_stim(pi)))
                    return [(pi, _stim(pi)) for pi in ranked]

        return sorted(
            ((pi, _stim(pi)) for pi in pool),
            key=lambda x: (-x[1], -ps.parents[x[0]].power),
        )
    return [
        (pi, 0.0)
        for pi in ps.facts_about(
            entity, top_k=config.knowledge_extra_facts + 6, vocab=state.vocab,
            min_priority=0.5,
            theme_weight=config.compose_theme_weight,
        )
    ]


def knowledge_extra_fact_plan(
    state: GraphState,
    q_ids: list[int],
    base_out: list[int],
    prompt_len: int,
    config: HCAConfig,
) -> dict[str, Any] | None:
    """Kế hoạch bồi fact: xếp theo stimulus + đánh dấu fact nào được ghép vào đáp."""
    ps = state.parents
    if ps is None or config.knowledge_extra_facts <= 0:
        return None
    q_content = tuple(_content_ids(q_ids, state.vocab, skip=set()))
    span = _knowledge_entity_span(ps, q_content)
    if span is None:
        return None
    entity, primary = span
    vocab = state.vocab
    # FIX GROUNDING (TK_042 2026-07-21): span cho primary = entity[-1] (PHẢI NHẤT) — đúng danh-từ-
    # ghép ("Mùa hè"→'hè' head-noun) nhưng SAI tên riêng VA-CHẠM ("Thăng Long"→'Long' → theme_neuron
    # ('Long') kéo Gia Long/Hạ Long vào pool → drift verbatim "Gia Long"). Phân biệt bằng cap_class:
    # entity TOÀN HOA (tên riêng đa-token) → primary = token ĐỊNH DANH (theme NHỎ nhất = đặc thù,
    # phá va-chạm; tie gidf); có token thường (danh-từ-ghép) → GIỮ rightmost. Đo local: Thăng Long
    # →'Thăng', Nhà Trần→'Trần', Bạch Đằng→'Bạch/Đằng'; Mùa hè giữ 'hè' (test cap_primary_gate).
    if len(entity) > 1 and all(vocab.cap_class(t) > 0 for t in entity):
        def _theme_sz(t: int) -> int:
            tn = ps.theme_neuron(t, vocab)
            return len(tn.facts) if tn is not None else 10**9
        primary = min(entity, key=lambda t: (_theme_sz(t), -ps._gidf(t)))
    tok = vocab.token_of
    # Nhánh C: primary danh-từ-chung viết thường ("bệnh"/"học") → query KHÔNG phải entity-knowledge
    # → KHÔNG bồi (thà đáp crisp 1 câu còn hơn vơ fact wiki lạc). cap_class = attr precompute (Luật 0.0).
    if getattr(config, "knowledge_extra_require_cap_primary", True) and vocab.cap_class(primary) == 0:
        return None
    eligible = _knowledge_extra_facts_eligible(ps, q_content, config)
    anchor_ids = base_out[prompt_len:] if base_out else None  # TK_027: neo = câu đáp CHÍNH
    ranked = (
        _rank_extra_fact_candidates(state, q_ids, entity, primary, config, anchor_ids)
        if eligible else []
    )
    fold = vocab.fold_group_id
    pf = fold(primary)
    period = vocab._token_to_id.get(".")
    ans_now = set(base_out[prompt_len:])
    picked_sets: list[frozenset[int]] = []  # fold-set các fact đã chọn (MMR)
    mmr_lambda = getattr(config, "knowledge_mmr_lambda", 0.5)
    added = 0
    facts: list[ExtraFactRank] = []
    for rank, (pi, score) in enumerate(ranked, start=1):
        comp = list(ps.completion(pi))
        text = vocab.detokenize(comp)
        cset = {t for t in comp if vocab.is_surface(t)}
        comp_folds = frozenset(fold(t) for t in comp)
        picked = False
        skip: str | None = None
        if added >= config.knowledge_extra_facts:
            skip = "limit"
        elif not cset:
            skip = "empty"
        elif pf not in comp_folds:
            # fact phải được KÍCH HOẠT bởi token thực-thể-chính (primary) — thuần token-activation,
            # KHÔNG quét/luật trên chuỗi completion (Luật 0/0.0).
            skip = "entity"
        else:
            inter = len(cset & ans_now) / max(1, len(cset))
            if inter > 0.6:
                skip = "dedup"
            else:
                # A1 MMR: phạt trùng với fact ĐÃ CHỌN → đa dạng khía cạnh (thành lập/vua/
                # văn hoá/chống giặc) thay vì top-N gần trùng.
                ov = max(
                    (len(comp_folds & ps_) / max(1, len(comp_folds | ps_)) for ps_ in picked_sets),
                    default=0.0,
                )
                if ov > mmr_lambda:
                    skip = "mmr"
            if skip is None:
                picked = True
                added += 1
                ans_now |= cset
                picked_sets.append(comp_folds)
                if period is not None:
                    ans_now.add(period)
        facts.append(
            ExtraFactRank(
                idx=pi,
                text=text,
                stimulus_score=score,
                rank=rank,
                picked=picked,
                skip=skip,
            )
        )
    return {
        "eligible": eligible,
        "entity": [tok(t) for t in entity],
        "primary": tok(primary),
        "facts": [
            {
                "idx": f.idx,
                "text": f.text,
                "score": round(f.stimulus_score, 3),
                "rank": f.rank,
                "picked": f.picked,
                "skip": f.skip,
            }
            for f in facts
        ],
    }


def _prompt_is_open_continuation(
    ids: list[int],
    vocab,
    ps=None,
    *,
    min_filler_diversity: int = 5,
) -> bool:
    """TK_025 — chuỗi ≥2 token bề mặt = ĐỦ ĐIỀU KIỆN hoàn-thành-mẫu (pattern-completion).

    WH-detector cũ (`wh_anchor_clusters`, phân-loại câu-hỏi vs mở-lửng) đã gỡ (Luật 0). Việc
    'có phải câu hỏi cần truy hồi không' giờ do RETRIEVAL quyết (best_match trúng cha → đáp;
    trượt → về pattern/W_seq). `ps`/`min_filler_diversity` giữ cho tương thích chữ ký."""
    del ps, min_filler_diversity
    content = [t for t in ids if vocab.is_surface(t)]
    return len(content) >= 2


def _opener_closer_pair_in_content(
    content: list[int],
    fb,
    opener: int,
    vocab,
) -> bool:
    """Có closer mined (opener→closer) xuất hiện trong query — nếu…thì, không phải Cách làm…"""
    row = fb.row(opener)
    if not row:
        alt = vocab._token_to_id.get(vocab.fold(opener))
        if alt is not None:
            row = fb.row(alt)
    if not row:
        return False
    return any(t in row for t in content[1:])


def _is_frame_opener_continuation(
    q_ids: list[int],
    state: GraphState,
    config: HCAConfig,
) -> bool:
    """TK_021 T2 — frame QA ruột-continuation: opener + closer trong query.

    TK_027: opener phải là HƯ-TỪ mở-mệnh-đề (nếu/tuy/không…), KHÔNG phải THỰC-THỂ query. Sau khi
    gỡ WH, `_prompt_is_open_continuation` luôn True → nhánh này over-fire cho "Lý Thái Tổ là ai"
    ("Lý" tình cờ ở lớp FB nhiễu) → route rác. Chặn bằng `entity_register_ids` (đã demote FB-opener
    → "nếu" không phải entity, "Lý"/"Nhà" thì có). Luật 0 (idf/vị-trí, không WH/danh-sách)."""
    ps = state.parents
    if ps is None:
        return False
    content = list(_content_ids(q_ids, state.vocab, skip=set()))
    if len(content) < 2:
        return False
    fold = state.vocab.fold_group_id
    reg = ps.entity_register_ids(tuple(content))
    if reg and fold(content[0]) in {fold(t) for t in reg}:
        return False  # opener LÀ thực-thể query → câu HỎI tri-thức, không phải continuation
    ensure_frame_brackets(state)
    fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    first = content[0]
    is_opener = bool(fb.row(first))
    if not is_opener:
        alt = state.vocab._token_to_id.get(state.vocab.fold(first))
        is_opener = alt is not None and bool(fb.row(alt))
        if is_opener:
            first = alt or first
    if not is_opener:
        return False
    return _opener_closer_pair_in_content(content, fb, first, state.vocab)


def _emit_tail(
    state: GraphState,
    prompt: str,
    generated: list[int],
    tail: list[int],
    *,
    max_len: int,
    config: HCAConfig,
    seed: int | None,
    eos_id: int,
    trace_out: list[dict[str, object]] | None,
    verbatim_only: bool = False,
) -> list[int]:
    """
    R4b — phát đuôi mẫu: agenda A3 (sinh token) hoặc verbatim nếu tắt / fallback cov thấp.
    """
    if not tail:
        return generated
    if verbatim_only or not config.ar_tail_as_agenda:
        return generated + tail
    agenda = _surface_content_ids(tail, state.vocab)
    if len(agenda) < 2:
        return generated + tail
    ps = state.parents
    q_content = _content_ids(generated, state.vocab, skip=set())
    entity_ids = ps.entity_register_ids(tuple(q_content)) if ps else ()
    plan = _SegmentPlan([agenda], entity_ids, list(agenda))
    out = generate_autoregressive(
        state, prompt, max_len=max_len, config=config, seed=seed,
        plan=plan, trace_out=trace_out,
    )
    if config.ar_tail_verbatim_fallback:
        gen = out[len(generated):]
        if eos_id in gen:
            gen = gen[: gen.index(eos_id)]
        gset = set(gen)
        cov = sum(1 for t in agenda if t in gset) / len(agenda)
        if cov < config.essay_ar_min_cov:
            return generated + tail
    return out


def _finish_or_pattern(
    state: GraphState,
    generated: list[int],
    eos_id: int,
    max_len: int,
    config: HCAConfig,
    rng: np.random.Generator,
    *,
    min_suffix_tokens: int,
    prompt: str = "",
    seed: int | None = None,
    trace_out: list[dict[str, object]] | None = None,
) -> list[int]:
    if config.ar_pattern_completion and _prompt_is_open_continuation(
        generated, state.vocab, state.parents,
        min_filler_diversity=config.wh_min_filler_diversity,
    ):
        tail = _pattern_completion_tail(state, generated, config, min_suffix_tokens=min_suffix_tokens)
        if tail:
            return _emit_tail(
                state, prompt or _decode_prompt(state, generated), generated, tail,
                max_len=max_len, config=config, seed=seed, eos_id=eos_id, trace_out=trace_out,
            )
    return _wseq_fallback(state, generated, eos_id, max_len, config, rng)


def _decode_prompt(state: GraphState, ids: list[int]) -> str:
    return state.vocab.detokenize(ids)


def _filter_plan_on_focus(
    state: GraphState,
    query_ids: list[int],
    plan: "_SegmentPlan",
) -> "_SegmentPlan":
    """
    R1 (TK_017 §11 F5) — gate plan FALLBACK theo focus: khi best_match trượt, `facts_about` có
    thể kéo cha LẠC (query "nếu trời mưa" → fact "Mặt trời cung cấp ánh sáng" qua token "trời")
    → agenda khoá sai "NÓI GÌ" từ đầu, gate khung không cứu được.

    Luật: segment giữ lại phải CHỨA token focus PHÂN BIỆT NHẤT của query (primary = max gidf —
    cùng nguyên tắc W3/`_match_has_focus`), so bằng fold-group (TK_012, không string-op).
    Không segment nào phủ → plan rỗng → caller rơi về `_wseq_fallback` (W_seq + bracket-gate
    dẫn) thay vì bịa fact lạc đề. Case con-kiến GIỮ (statement chứa thực thể → pass).
    """
    ps = state.parents
    if ps is None or not plan.segments:
        return plan
    q_content = _content_ids(query_ids, state.vocab, skip=set())
    focus = ps.top_focus_tokens(tuple(q_content), k=2, vocab=state.vocab) if q_content else ()
    if not focus:
        return plan
    primary = max(focus, key=ps._gidf)
    fold = state.vocab.fold_group_id
    pf = fold(primary)
    kept = [seg for seg in plan.segments if any(fold(t) == pf for t in seg)]
    if len(kept) == len(plan.segments):
        return plan
    flat: list[int] = []
    for seg in kept:
        flat.extend(seg)
    return _SegmentPlan(
        kept, plan.entity_ids, flat,
        hard_copy_segs=plan.hard_copy_segs,
        pivot_lock_segs=plan.pivot_lock_segs,
    )


def _at_boundary(state: GraphState, w0: int, config: HCAConfig) -> bool:
    """w0 có đứng ở RANH GIỚI diễn ngôn không (→ index-backbone tin trật tự NGUỒN vô điều kiện).

    **TK_041 — cái chết của so-string cuối cùng trong inference.** Đường BPE không còn dòng
    nào hỏi "token này trông như thế nào": `Hỏi:`/`Đáp:` là special NGUYÊN TỬ với id cố định
    (§2.2) → so INT; phần còn lại do `_ends_unit` (LIFT P(w0→EOS), học từ data — §0.4.2).
    Đóng nốt mục (B) bảng tuân thủ Luật 0.0 (README §0.0).

    ⚠️ **Legacy giữ NGUYÊN VĂN `endswith(":")`** — không phải vì lười, mà vì nó KHÔNG tương
    đương: nó bắt MỌI token đuôi `:` (`Toán:`, `Lớp 3:`, `Tiết 1:` — có thật trong
    `grade_lessons`), không riêng hai nhãn frame. Thay bằng so-id trên checkpoint cũ = đổi
    hành vi = baseline §7.1 trôi = mất thước để so v11. Legacy chết cùng checkpoint cũ.

    Trên BPE, `Toán:` tách thành `Toán` + `:`, và mốc ranh giới thật của nó là **`Ċ` (xuống
    dòng) — nay là token**. Tức tín hiệu chuyển từ ĐOÁN-mặt-chữ sang ĐỌC-dữ-liệu.
    """
    vocab = state.vocab
    if not vocab.is_bpe:
        w0_tok = vocab.token_of(w0)
        return (
            (not is_surface_token(w0_tok))
            or w0_tok.endswith(":")
            or _ends_unit(state, w0, config)
        )
    return (
        (not vocab.is_surface(w0))
        or w0 in _frame_label_ids(vocab)
        or _ends_unit(state, w0, config)
    )


def _frame_label_ids(vocab) -> frozenset[int]:
    """Id của nhãn frame `Hỏi:`/`Đáp:` — cache trên vocab (so int, không so chuỗi)."""
    ids = getattr(vocab, "_frame_label_ids_cache", None)
    if ids is None:
        from nn_aivie.core.tokenizer import DAP_LABEL, HOI_LABEL

        ids = frozenset(vocab.id_of(t) for t in (HOI_LABEL, DAP_LABEL))
        vocab._frame_label_ids_cache = ids
    return ids


def _ends_unit(state: GraphState, w0: int, config: HCAConfig) -> bool:
    """`w0` có phải mốc KẾT THÚC đơn vị không — đọc từ đồ thị, KHÔNG từ danh sách dấu câu.

    Tín hiệu: P(EOS | w0) trên W_seq = tỉ lệ khối lượng cạnh w0→· đổ vào EOS. Token nào
    thường kết thúc đơn vị train thì tỉ lệ này cao. Học được vì `ingest._with_eos` nối EOS
    vào cuối MỖI đơn vị (ý tác giả 2026-07-15).

    Thay `vocab.is_punct(w0)` của bản cũ (Luật 0: không đặc-cách từ vựng lúc suy luận). Hai
    tín hiệu dẫn xuất khác đã thử và THẤT BẠI — chúng không tách nổi ranh giới khỏi nội dung:
      - entropy hàng W_seq:      '.'=5.93 nhưng 'là'=5.45, '?'=3.08 < 'học'=4.39
      - tỉ lệ đứng cuối chuỗi:   '.'=0.152 nhưng ','=':'='Đáp:'=0.000 (ngang từ nội dung)
    P(→EOS) tách sạch (đo trên corpus đồ chơi 3 câu): '.'=1.000, 'là'/'Việt'/'Nam'/'ăn'=0.000.

    Checkpoint train TRƯỚC 2026-07-15 không có cạnh →EOS trên đường ingest → hàm trả False
    → at_boundary lùi về chỉ nhận mốc qua nhãn frame (đúng bằng hành vi hiện tại, không tệ
    hơn). Sau train lại thì tín hiệu bật lên. Xem TK_012 §9.5.
    """
    if config.ar_eos_boundary_min_lift <= 0:
        return False
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    row = seq.row(w0)
    if not row:
        return False
    eos_id = state.vocab.id_of(SPECIAL_EOS)
    w = row.get(eos_id, 0.0)
    if w <= 0:
        return False
    total = sum(row.values())
    if total <= 0:
        return False
    base = _eos_base_rate(state, seq, eos_id)
    if base <= 0:
        return False
    return (w / total) / base >= config.ar_eos_boundary_min_lift


def _eos_base_rate(state: GraphState, seq, eos_id: int) -> float:
    """P(EOS) nền = khối lượng cạnh ·→EOS / tổng khối lượng W_seq. Memo 1 lần/state."""
    cached = getattr(state, "_eos_base_rate_cache", None)
    if cached is not None:
        return cached
    to_eos = tot = 0.0
    for _i, j, w in seq.items():
        tot += w
        if j == eos_id:
            to_eos += w
    base = (to_eos / tot) if tot > 0 else 0.0
    state._eos_base_rate_cache = base
    return base


def _build_agenda(
    state: GraphState,
    query_ids: list[int],
    config: HCAConfig,
    *,
    max_facts: int = 3,
    multi: bool = False,
) -> list[int]:
    """
    PLAN (A1 → A3): xác định các token NỘI DUNG câu trả lời cần phủ (agenda).

    `multi=False` (mặc định): CRISP — agenda = đáp của cha khớp nhất (1 câu gọn).
    `multi=True`: MỞ — gom tối đa `max_facts` fact về thực thể trọng tâm (cả đáp QA lẫn statement)
    → A3 sinh **nhiều câu** token-by-token. Agenda là *ý cần diễn đạt*, KHÔNG phải chuỗi verbatim.

    (Không tự đoán crisp/mở từ đồ thị: tín hiệu `is_qa` không tin cậy vì nhiều fact được lưu dưới
    dạng QA — để CALLER/UI chọn mode, tránh heuristic sai.)
    """
    plan = _build_segment_plan(
        state, query_ids, config, max_facts=max_facts
    )
    return plan.flat_agenda


def _top_scores_ui(
    scores: np.ndarray, vocab, *, top_k: int = 8
) -> list[dict[str, object]]:
    """Top-k token scores cho Graph UI trace A3."""
    finite = np.where(np.isfinite(scores) & (scores > 0), scores, -np.inf)
    n = int(np.sum(np.isfinite(finite)))
    if n <= 0:
        return []
    k = min(top_k, n)
    idx = np.argpartition(-finite, k - 1)[:k]
    idx = idx[np.argsort(-finite[idx])]
    tok_of = vocab.token_of
    return [
        {"token": tok_of(int(i)), "score": round(float(scores[i]), 4)}
        for i in idx
    ]


# §83.2 — khoá qua `objcache` (id + weakref + so `is`), KHÔNG `id()` trần.
# `id()` chỉ duy nhất TRONG THỜI GIAN đối tượng còn sống: một `GraphState`
# bị thu hồi rồi state khác trùng địa chỉ sẽ được phát chỉ mục loại suy của
# CHECKPOINT CŨ — sai lặng, ngay trên đường chuỗi §79. Cùng họ với lỗi
# `_CO_MARK` ở `hub_spread.py`; `hca/objcache.py` sinh ra đúng để chặn nó.
_HUB_CHAIN_CACHE: dict = {}
_CHU_DN_CACHE: dict = {}


def hub_chain_proposals(
    state: GraphState,
    ctx_ids: list[int],
    config: "HCAConfig",
) -> dict[int, float]:
    """CHUỖI BA BƯỚC → đề cử token kế (TK_046 §22).

    ① mồi (token nội dung trong ngữ cảnh trái) → hạt giống
    ② hạt giống → NHÃN, xếp theo ĐỒNG THUẬN trong kênh typed (`rank="phieu"`)
    ③ nhãn → THÀNH VIÊN; thành viên chưa kề mồi = ứng viên «con khỉ»

    Chỉ lấy cạnh **typed** (`A·là/như·B`): kênh đồng hiện tăng tuyến tính theo cỡ
    corpus nên nó nuốt sạch cạnh phân loại (TK_046 §18.H).

    Luật 0: chỉ đọc `via_pivot` đã bake, không có từ vựng viết tay.
    """
    from nn_aivie.hca import hub_spread as _H

    if not ctx_ids:
        return {}
    idx = _oc_get(_HUB_CHAIN_CACHE, state)
    if idx is None:
        idx = _H.build_hub_spread_index(
            state, d_max=int(config.hub_chain_d_max),
            min_count=float(config.hub_chain_min_count),
            head_rule=str(config.hub_chain_head_rule),
            head_rule_trai=str(config.hub_chain_head_rule_trai),
            label_indeg_q=float(config.hub_chain_indeg_q),
            label_gidf_q=float(config.hub_chain_gidf_q),
            head_gidf_q=float(config.hub_chain_head_gidf_q),
            kenh_vi_ngu=bool(config.hub_chain_kenh_vi_ngu),
            gop_manh_tu=bool(config.hub_chain_gop_manh_tu),
            pos_map=_H.load_pos_map(config.hub_chain_pos_dict),
            chu_ngu_ket_hop_q=float(config.hub_chain_chu_gia_q),
            loc_dich=bool(getattr(config, "hub_chain_loc_dich", False)),
            attr_w_min=float(getattr(config, "hub_chain_attr_w_min", 0.0) or 0.0),
        )
        _oc_put(_HUB_CHAIN_CACHE, state, idx)

    vocab = state.vocab

    def _s(t: int) -> str:
        return _H._surf(vocab, int(t))

    ctx = set(int(t) for t in ctx_ids)
    # KỀ NGỮ CẢNH cho đường TRỰC TIẾP (§72) — 67,2% ca trượt top-5 là lỗi xếp
    # hạng giữa các nhãn của CÙNG một chủ ngữ.
    _ke = (_H.bang_ke_ngu_canh(idx, vocab, [int(t) for t in ctx_ids])
           if float(config.hub_chain_ke_truc_tiep) > 0.0 else {})
    # GOM TOKEN THÀNH TỪ TRƯỚC KHI GIEO ① — TK_052 vòng 2.
    # `encode_prompt` tách 89% tính từ thành mảnh BPE (`ngặt` → `ng`+`ặt`), nên
    # vòng lặp theo TOKEN gieo ① bằng MẢNH, và `resolve_surface_ids` chỉ tra
    # lại được mảnh đó. Chạy tay bằng cả bề mặt cho ③ ra m3 ở 71,7% bộ, đường
    # sinh chỉ 38,6%. Đây là bệnh §23.1 ở chỗ chưa ai vá: một bề mặt nhiều id.
    from nn_aivie.core.tokenizer import MARK_SPACE as _MS
    _tu: list[list] = []          # [bề mặt TỪ, [mọi token của từ đó]]
    for _t in ctx_ids:
        _raw = str(vocab.token_of(int(_t)))
        _sf = _raw.replace(_MS, "").strip()
        if not _sf or _sf.startswith("<"):
            continue
        if _MS in _raw or not _tu:
            _tu.append([_sf, [int(_t)]])
        else:
            _tu[-1][0] += _sf
            _tu[-1][1].append(int(_t))
    moi_tu = [(w, ts) for w, ts in _tu if len(w) > 1 and w not in _H._FUNC]
    moi = [int(t) for t in ctx_ids if len(_s(t)) > 1 and _s(t) not in _H._FUNC]
    # §129 — nạp bảng nhãn lớp một lần cho cả lượt gọi.
    # §136 — nạp bảng khi BẤT KỲ chỗ dùng nào bật. Bản đầu chỉ xét
    # `bang_nhan_w` nên bật `bang_tv_w` một mình thì bảng rỗng và cơ chế KHÔNG
    # NỔ (đo được 0/5 prompt đổi). Phép đếm rẻ trước A/B bắt được (bài học 192).
    _bang_nl = (_H.load_nhan_lop(getattr(config, "hub_chain_bang_nhan_lop", ""))
                if (float(getattr(config, "hub_chain_bang_nhan_w", 0.0)) > 0.0
                    or float(getattr(config, "hub_chain_bang_tv_w", 0.0)) > 0.0)
                else {})
    # bề mặt nhãn → id NHỊP của chỉ mục (đảo `label_surf`), để nhãn GHÉP
    # («kim loại», «gia súc») vào ③ nguyên vẹn thay vì bị cắt còn token đầu.
    # §132 — MỘT BỀ MẶT có NHIỀU id nhịp (bệnh §23.1, bình luận ở khối gom
    # từ ngay trên đã nêu cho HẠT GIỐNG; nó áp y hệt cho ĐÍCH). Đo được:
    # «kim loại» có 3 id (33947, 67036, 90856); mọi kim loại trỏ tới 33947 và
    # `reverse_peers(33947)` = vàng·đồng·chì·ngân, còn hai id kia RỖNG. Bản
    # `{bề mặt: id}` giữ id CUỐI gặp phải — trúng id rỗng ⇒ lớp trông như không
    # có thành viên. Phải giữ MỌI id rồi gieo ③ bằng tất cả.
    # §136 — ĐẢO CỦA BẢNG ở hạt 2 TỪ: lớp → thành viên, lấy thẳng từ từ điển
    # thay vì từ `via_rev` của corpus. §135 đo: 64,4% số từ có lớp ≥2 thành
    # viên (mốc corpus 38,5%) mà vẫn 5 293 lớp nên không mất độ cụ thể. Hạt
    # 1 từ phủ 96,6% nhưng chỉ 1 295 lớp ⇒ lớp `thú`/`cây`/`loài` vô dụng.
    _bang_tv_w = float(getattr(config, "hub_chain_bang_tv_w", 0.0) or 0.0)
    # §137 — thành viên từ bảng gom lại, TIÊM SAU theo thang của chính `ra`.
    # Bản §136 tiêm trọng số TUYỆT ĐỐI (2,0) trong khi đỉnh chuỗi là 40,8 ⇒ sau
    # chuẩn hoá theo max còn ~0,05 ⇒ không đổi được token nào (đo: 0/60 và
    # 0/14, kể cả prompt có |agenda| = 0). Cùng bất đối xứng tuyệt-đối/tương-đối
    # mà §120 gặp ở EOS: trộn hai thang thì vế tuyệt đối luôn thua.
    _tv_cho: set[int] = set()
    _lop2tu: dict[str, list[str]] = {}
    if _bang_tv_w > 0.0 and _bang_nl:
        _tmp: dict[str, set[str]] = {}
        for _bw2, _bls in _bang_nl.items():
            for _bl in _bls:
                _p = _bl.split()
                if len(_p) >= 2:
                    _tmp.setdefault(" ".join(_p[:2]), set()).add(_bw2)
        _lop2tu = {k: sorted(x) for k, x in _tmp.items() if len(x) >= 2}
    _s2sp: dict[str, list[int]] = {}
    if _bang_nl:
        for _sk, _sv in idx.label_surf.items():
            _s2sp.setdefault(str(_sv).strip().lower(), []).append(int(_sk))
    ra: dict[int, float] = {}
    # ĐỒNG THUẬN CẤP THUỘC TÍNH (§36). Trước đây các thuộc tính trong ngữ cảnh
    # được hợp nhất bằng `max` — tức UNION, không đếm ai đồng ý với ai. Đây là
    # tín hiệu độc lập DUY NHẤT còn lại sau khi §35 loại ba tín hiệu nội tại:
    # nó không suy ra được từ cạnh đang xét mà đến từ MỘT THUỘC TÍNH KHÁC.
    dong_thuan: dict[int, set[int]] = {}
    manh: set[int] = set()
    for _w, _reps in (moi_tu[-4:] if int(config.hub_chain_gom_tu) else
                      [(_s(t), [int(t)]) for t in moi[-4:]]):
        # `p` = token ĐẠI DIỆN của từ; `bien_the` gom mọi id cùng bề mặt TỪ.
        # Không tra được bề mặt ghép thì lùi về chính token của prompt — bỏ
        # hẳn thì ca thử vàng mất sạch đề cử (2 test đỏ khi tôi làm thế).
        # ⚠️ DỰ PHÒNG PHẢI LÀ MỌI MẢNH CỦA TỪ, không phải mảnh ĐẦU. Lùi về
        # mảnh đầu (`b` của `bèo`) làm ca SUY của TK_046 tụt 32,5% → 2,5%:
        # mảnh đầu hầu như không có cạnh nào.
        # HỢP của id-CẢ-TỪ và id-MẢNH. Corpus khác nhau mã hoá khác nhau: chỗ
        # thì cạnh bám vào id cả từ, chỗ thì bám vào mảnh (`èo` của `bèo`).
        # Chỉ lấy một phía thì mất phía kia — đo được cả hai chiều:
        #   chỉ cả-từ  → TK_052 70,3% · TK_046 ca SUY 2,5%  (tụt từ 32,5%)
        #   chỉ mảnh   → TK_052 44,1% · TK_046 ca SUY 32,5%
        _che_do = int(config.hub_chain_gom_tu)
        # GỒM CẢ id ẢO của span WS — TK_052 vòng 7: từ bị BPE băm (`sóc` →
        # `s`+`óc`) mang cạnh trên id ảo, `resolve_surface_ids` không thấy.
        _ca_tu = _H.ids_theo_be_mat(idx, vocab, _w)
        if _che_do == 2:                       # CHỈ id cả-từ
            _bt0 = _ca_tu or list(_reps)
        elif _che_do == 1:                     # HỢP
            _bt0 = list(dict.fromkeys(_ca_tu + list(_reps)))
        else:                                  # 0 = chỉ MẢNH (hành vi cũ)
            _bt0 = list(_reps)
        p = int(_bt0[0])
        # MỘT bề mặt có NHIỀU id BPE (`dòng` = 2880/7427/8744/25529) và
        # `via_pivot`/`fwd` khoá theo TỪNG id. `encode_prompt` trả đúng một
        # biến thể, và trên `classrich` đó thường là biến thể RỖNG CẠNH —
        # đo được: 5/10 ca chuỗi ra rỗng chỉ vì lý do này. Gom mọi biến thể
        # cùng bề mặt (Luật 0: đồng nhất theo bề mặt, không phải danh sách từ).
        bien_the = _bt0
        # NHÃN TRỰC TIẾP — §71. Chuỗi ba bước luôn đi vòng
        # (p → vật mang → nhãn → thành viên) kể cả khi `p` ĐÃ CÓ cạnh lớp của
        # chính nó. Với prompt «X là ___» thì đáp án đúng CHÍNH LÀ lớp của X,
        # và thước v3 đo được ca NHỚ chỉ đạt top-1 4,0% — thấp hơn cả ca SUY
        # (8,1%) vốn không có cạnh nào. Cơ chế đang bỏ phí thứ nó đã học.
        # §129 — NHÃN LỚP TỪ BẢNG BAKE. §127 đo nhãn của `via_pivot` đúng
        # 4,1% trên 1 718 từ (mẫu `A là B` bắt trúng câu hỏi «Nước là gì?» và
        # mệnh đề tu từ «chó là bạn»); bảng bake từ KHUÔN ĐỊNH NGHĨA đạt 84,9%
        # top-1 trên 8 568 từ với CÙNG chuẩn, CÙNG cách chấm (§128).
        # §100 đo: cấp nhãn lớp ĐÚNG thì thu hồi top-1 nhảy 4,5% → 53,3%.
        # Vào thẳng ② như một nguồn nhãn song song với `via_pivot`.
        # §136 — guard nhận CẢ HAI chỗ dùng. Bản đầu chỉ xét `bang_nhan_w`
        # nên bật `bang_tv_w` một mình thì khối này không chạy — lần thứ MƯỜI
        # MỘT kiểu «đúng cơ chế, sai cổng vào». Sửa nạp bảng thôi chưa đủ.
        if (_bang_nl and (float(getattr(config, "hub_chain_bang_nhan_w", 0.0)) > 0.0
                          or _bang_tv_w > 0.0)):
            _kb = float(config.hub_chain_bang_nhan_w)
            # Giá trị của bảng KHÔNG nằm ở việc tiêm nhãn (nhãn nhiều từ
            # không phân giải thành một token, còn từ đầu — `gia`,`kim`,`loài`
            # — thì `via_pivot` đã có sẵn), mà ở việc DÙNG nhãn để kéo
            # ĐỒNG-THÀNH-VIÊN qua `reverse_peers`: đó chính là bước ③ áp cho
            # nhãn ĐÃ BIẾT. Đo được ③ chạy tốt khi ② có nhãn tử tế
            # (`sắt là` → chì/titan/đồng/kẽm) và chỉ hỏng nơi nhãn rác
            # (`sóc là` → bạn/hàng/rơm).
            # ⚠️ Phân giải nhãn bằng `resolve_surface_ids` là SAI: nhãn nhiều
            # từ co về token ĐẦU (`kim loại`→`kim`, `loài linh`→`loài`) rồi
            # `reverse_peers` kéo về đủ thứ — đo được `sắt là` tụt từ
            # «chì/titan/đồng/kẽm» xuống «Quốc/chẳng/quá/tới». Chỉ mục ĐÃ có id
            # cho nhãn nhiều từ trong `label_surf` (id nhịp → bề mặt), nên tra
            # NGƯỢC qua đó; chỉ lùi về token đơn khi bảng không khớp nhịp nào.
            # ƯU TIÊN nhãn GHÉP: bảng liệt kê từ đầu đơn trước («kim» rồi
            # «kim loại»), mà từ đầu đơn chính là cái mơ hồ. Nếu có nhãn ghép
            # khớp nhịp thì DÙNG RIÊNG nó, bỏ hẳn mục đơn.
            # §131 — quét CẢ danh sách và ưu tiên cụm DÀI NHẤT khớp nhịp.
            # §128 dùng cửa sổ 2 từ nên nhãn bị cắt (`thú gặm`, `cây lương`);
            # đo trên chỉ mục: nhịp dài tới 6 từ, nhịp 3 TỪ phổ biến nhất
            # (28 240/79 601). Cụm dài hơn thì cụ thể hơn ⇒ lớp hẹp hơn ⇒ đồng
            # thành viên sát hơn (`thú gặm nhấm` ≻ `thú`).
            _ds = list(_bang_nl.get(_w.lower(), ()))
            _ghep = sorted((x for x in _ds if " " in x and x.lower() in _s2sp),
                           key=lambda s: -len(s.split()))[:2]
            # gieo ③ bằng tối đa 8 id (nhiều nhãn × nhiều id đồng bề mặt)
            # §136 — thành viên CÙNG LỚP lấy thẳng từ đảo-bảng (hạt 2 từ).
            if _bang_tv_w > 0.0 and _lop2tu:
                for _nl2 in _ds:
                    _p2 = _nl2.split()
                    if len(_p2) < 2:
                        continue
                    for _tv2 in _lop2tu.get(" ".join(_p2[:2]), ())[:24]:
                        if _tv2 == _w.lower():
                            continue
                        for _ti2 in _H.resolve_surface_ids(vocab, _tv2)[:1]:
                            _ti2 = int(_ti2)
                            if 0 <= _ti2 < len(vocab) and _ti2 not in ctx:
                                _tv_cho.add(_ti2)
                    break          # chỉ lớp ĐẦU đủ 2 từ, tránh trộn nhiều lớp
            _hat_nl: list[int] = []
            for _nl in (_ghep or _ds):
                _sids = _s2sp.get(_nl.lower())
                if _sids:
                    _hat_nl.extend(int(x) for x in _sids[:4])
                    continue
                if " " in _nl:
                    continue          # nhãn ghép không có nhịp ⇒ BỎ, không cắt
                for _tb in _H.resolve_surface_ids(vocab, _nl)[:2]:
                    _tb = int(_tb)
                    if 0 <= _tb < len(vocab):
                        if _tb not in ctx:
                            ra[_tb] = max(ra.get(_tb, 0.0), _kb)
                        _hat_nl.append(_tb)
            if _hat_nl:
                for _mb in _H.reverse_peers(
                        state, idx, _hat_nl[:8], topk=20, gidf_min=0.0,
                        channel="typed_first",
                        ctx_ids=[int(x) for x in ctx_ids],
                        ke_ctx=float(config.hub_chain_ke_ctx)):
                    if not _mb.typed:
                        continue
                    _t2 = int(_mb.tid)
                    if not (0 <= _t2 < len(vocab)):
                        _sp2 = idx.label_span.get(_t2)
                        if not _sp2:
                            continue
                        _t2 = int(_sp2[0])
                        manh.update(int(x) for x in _sp2[1:])
                    if 0 <= _t2 < len(vocab) and _t2 not in ctx:
                        ra[_t2] = max(ra.get(_t2, 0.0),
                                      _kb * float(_mb.weight))
        if float(config.hub_chain_nhan_truc_tiep) > 0.0:
            _k0 = float(config.hub_chain_nhan_truc_tiep)
            for _i0 in bien_the[:4]:
                for _d0, (_w0, _m0) in idx.via_pivot.get(int(_i0), {}).items():
                    _t0 = int(_d0)
                    if not (0 <= _t0 < len(vocab)):
                        _sp0 = idx.label_span.get(_t0)
                        if not _sp0:
                            continue
                        _t0 = int(_sp0[0])
                        manh.update(int(x) for x in _sp0[1:])
                    if 0 <= _t0 < len(vocab) and _t0 not in ctx:
                        _w0b = _k0 * float(_w0)
                        if _ke:
                            _sf0 = (idx.label_surf.get(int(_d0))
                                    or _H._surf(vocab, _t0))
                            if _sf0 in _ke:
                                _w0b *= 1.0 + float(config.hub_chain_ke_truc_tiep)
                        ra[_t0] = max(ra.get(_t0, 0.0), _w0b)
                    # ĐỒNG-THÀNH-VIÊN của nhãn trực tiếp — §72. Đây chính là
                    # bước ③ áp cho nhãn ĐÃ BIẾT thay vì nhãn phải suy ra:
                    # «chó là ___» thì `mèo`/`hổ` cũng là đáp án hợp lệ, và
                    # chúng nằm sẵn trong `reverse_peers` của lớp.
                    _k1 = float(config.hub_chain_tv_truc_tiep)
                    if _k1 <= 0.0:
                        continue
                    _bt1 = _H.resolve_surface_ids(vocab, _H._surf(vocab, int(_d0))
                                                  if int(_d0) < len(vocab)
                                                  else idx.label_surf.get(int(_d0), "")) or []
                    for _m1 in _H.reverse_peers(
                            state, idx, [int(_d0)] + [t for t in _bt1
                                                      if t != int(_d0)][:3],
                            topk=20, gidf_min=0.0, channel="typed_first",
                            ctx_ids=[int(t) for t in ctx_ids],
                            ke_ctx=float(config.hub_chain_ke_ctx)):
                        if not _m1.typed:
                            continue
                        _t1 = int(_m1.tid)
                        if not (0 <= _t1 < len(vocab)):
                            _sp1 = idx.label_span.get(_t1)
                            if not _sp1:
                                continue
                            _t1 = int(_sp1[0])
                            manh.update(int(x) for x in _sp1[1:])
                        if 0 <= _t1 < len(vocab) and _t1 not in ctx:
                            ra[_t1] = max(ra.get(_t1, 0.0),
                                          _k1 * float(_m1.weight))
        # KÊNH THUỘC TÍNH (§53) — TK_052 vòng 2. Bước ① là quan hệ THUỘC TÍNH
        # («nghịch ngợm» → vật mang), và §53 dựng hẳn một LOẠI CẠNH THỨ HAI cho
        # nó. Nhưng đường sinh gọi ① mà KHÔNG truyền `kenh_thuoc_tinh=True`,
        # nên suốt từ §53 tới nay nó chạy trên kênh phân loại và mù với
        # `via_attr`. Đo trên corpus TK_052: chạy tay (có kênh) ③ ra m3 ở 73,8%
        # bộ, đường sinh (không kênh) thu hồi 27,6%.
        s1 = _H.score_hubs_from(
            state, idx, bien_the[:int(config.hub_chain_n_bien_the)],
            topk=int(config.hub_chain_topk_1),
            gidf_min=0.0, kenh_thuoc_tinh=True,
            channel=_kenh_buoc(config, 1))
        # HẠT GIỐNG: nếu có hit TYPED thì CHỈ dùng typed (§59). Kênh cooc kéo
        # theo BẠN GHÉP TỪ (`nghịch` → `nghịch đảo`/`nghịch lý`) và chúng làm
        # loãng phiếu ở ②. Đo trên toy 59 769 cha:
        #     hạt=[chó, mèo, đảo, lý] → nhãn `bao`  · khỉ —
        #     hạt=[chó, mèo]          → nhãn `động` · khỉ hạng 20
        # Không có typed thì rơi về cooc như cũ (ckpt nhỏ chưa có cạnh typed).
        _cand = [h for h in s1 if h.tid not in ctx]
        # §101 — «CHỈ typed nếu có typed» (§59) đo trên TOY 59 769 cha. Trên
        # checkpoint thật, nhãn typed phần lớn là hư danh (`phần`/`nơi`), nên
        # loại kênh cooc có thể là vứt đúng tín hiệu cần. Cờ để đo lại.
        _ty = [h for h in _cand if h.typed]
        _chi_ty = _kenh_buoc(config, 1) != "cooc"
        hat = ((_ty if (_ty and _chi_ty) else _cand))[:4]
        if bool(config.hub_chain_hat_dong_thuan) and len(_ty) < 2:
            _dt = _H.chon_hat_dong_thuan(idx, vocab, _cand, topk=4)
            if _dt:
                hat = _dt
        if not hat:
            continue
        # §105 — ÉP CẦU NỐI ② PHẢI ĐẶC TRƯNG. §104 đo (n=196, kiểm chia đôi,
        # 20 hạt hoán vị): ở `gidf≥6` hub tự chứa BẠN CÙNG LỚP ở 4,6% so với
        # 1,9%±0,8 của hoán vị — vượt 20/20 hạt, 3,4σ; nửa B giữ kín cho 5,1%
        # vs 1,8%±1,1. Ở gidf=0 (mặc định cũ) chênh bằng 0: hư danh nuốt sạch.
        s2 = _H.score_hubs_from(
            state, idx, [h.tid for h in hat], topk=10,
            gidf_min=float(getattr(config, "hub_chain_gidf_min_2", 0.0) or 0.0),
            channel=_kenh_buoc(config, 2), rank=str(config.hub_chain_rank),
            loc_pos=True, ctx_ids=[int(t) for t in ctx_ids])
        cam = {h.surface for h in hat} | {_w, _s(p)}
        ok = [h for h in s2 if h.surface not in cam]
        if not ok:
            continue
        # CỘNG DỒN THAY VÌ BA ARGMAX — §70. Chuỗi cũ chọn tốt-nhất ba lần nối
        # tiếp; sai ở bước một là mất sạch. Đo được (§69.4): bể ① có ≥2
        # đồng-thành-viên ở 16,1% ca mà top-4 chỉ bắt 1,5% — 14,6 điểm nằm ở
        # BƯỚC CHỌN, và bốn luật chọn của §67.B đều trượt.
        # Cách này KHÔNG CHỌN: mọi nhãn ứng viên của ② đều được lan ngược, và
        # điểm cộng dồn lên THÀNH VIÊN. Thành viên nào được nhiều đường mạch
        # lạc cùng đề cử thì nổi lên — cùng nguyên lý đồng thuận đã ăn tiền ở
        # ② (§21) và ③ (§60), lần này áp cho CẢ ĐƯỜNG ĐI.
        _nhieu = int(config.hub_chain_cong_don)
        _n_duyet = _nhieu if _nhieu > 0 else (
            int(getattr(config, "hub_chain_hub_top", 0) or 0) or 1)
        for _oi, _nh in enumerate(ok[:_n_duyet]):
            _bt = _H.ids_theo_be_mat(idx, vocab, _nh.surface)
            _nids = [_nh.tid] + [t for t in _bt if t != _nh.tid][:3]
            # §105 — HUB LÀM ĐỀ CỬ THẲNG. §104: ở `gidf≥6` hub ② tự chứa bạn
            # cùng lớp (`cáo·chuột·cầy` → `sói`,`chồn`) — không cần ③ nở ra. Mà
            # dòng dưới lại nằm trong `_nhieu > 0`, và `hub_chain_cong_don` mặc
            # định 0 (bác ở §70) ⇒ hub ② CHƯA BAO GIỜ được phát làm đề cử.
            if bool(getattr(config, "hub_chain_hub_lam_de_cu", False)) \
                    and 0 <= _nh.tid < len(vocab):
                ra[_nh.tid] = ra.get(_nh.tid, 0.0) + float(_nh.score)
            # NHÃN cũng là đề cử: «X là ___» thì câu trả lời đúng chính là LỚP.
            if _nhieu > 0 and 0 <= _nh.tid < len(vocab):
                ra[_nh.tid] = ra.get(_nh.tid, 0.0) + float(_nh.score)
            if _nhieu <= 0:
                continue        # §105 — vẫn duyệt tiếp để gom hub làm đề cử
            for _m in _H.reverse_peers(
                    state, idx, _nids, topk=30, gidf_min=0.0,
                    channel="typed_first", ctx_ids=[int(t) for t in ctx_ids],
                    ke_ctx=float(config.hub_chain_ke_ctx)):
                if not _m.typed:
                    continue
                _t = int(_m.tid)
                if not (0 <= _t < len(vocab)):
                    _sp = idx.label_span.get(_t)
                    if not _sp:
                        continue
                    _t = int(_sp[0])
                    manh.update(int(x) for x in _sp[1:])
                if 0 <= _t < len(vocab):
                    ra[_t] = ra.get(_t, 0.0) + float(_m.weight)
                    dong_thuan.setdefault(_t, set()).add(int(p))
        if _nhieu > 0:
            continue
        # RÒ 2 (§27.2): ngay khi ② đúng lớp thì chỉ 46% mục tiêu được nâng.
        # Cùng bệnh biến-thể-BPE đã lộ ở §23.1 — nhãn `quốc gia` có nhiều id,
        # thành viên bám rải trên các id khác nhau; lấy MỘT id thì mất phần còn
        # lại. Gom mọi biến thể cùng bề mặt của NHÃN trước khi lan ngược.
        # GỘP MẢNH (§38): dual-write phát CẢ cụm LẪN token thành phần, nên một
        # thành viên chiếm 2–3 suất top-k (`Live`+`Live Blue`, `Bóng`+`Bóng đá`
        # +`đá`). Đo được: bỏ mảnh đưa ③ chính xác 25,0% → 38,9%.
        nhan_bt = _H.resolve_surface_ids(vocab, ok[0].surface) or []
        nhan_ids = [ok[0].tid] + [t for t in nhan_bt if t != ok[0].tid][:3]
        for m in _H.reverse_peers(
                state, idx, nhan_ids, topk=30, gidf_min=0.0,
                channel="typed_first", ctx_ids=[int(t) for t in ctx_ids],
                ke_ctx=float(config.hub_chain_ke_ctx)):
            if not m.typed:
                continue
            if 0 <= m.tid < len(vocab):
                ra[m.tid] = max(ra.get(m.tid, 0.0), float(m.weight))
                dong_thuan.setdefault(int(m.tid), set()).add(int(p))
                continue
            # RÒ 2a (§32.1a): 74% thành viên lớp là TỪ NHIỀU ÂM TIẾT
            # (`Sông Hồng`, `Hậu Lê`, `Sao Hoả`). `via_pivot` CÓ chúng — dual-write
            # ghi cả id CỤM lẫn token thành phần — nhưng dòng lọc cũ
            # `m.tid < len(vocab)` vứt sạch id cụm, nên ③ chỉ còn token đơn.
            # Sinh là TỰ HỒI QUY nên nâng TOKEN ĐẦU của cụm là đúng: muốn ra
            # `Sông Hồng` thì token kế phải là `Sông`.
            sp = idx.label_span.get(int(m.tid))
            if sp:
                t0 = int(sp[0])
                if 0 <= t0 < len(vocab):
                    ra[t0] = max(ra.get(t0, 0.0), float(m.weight))
                    dong_thuan.setdefault(t0, set()).add(int(p))
                    # Cụm đã phát qua TOKEN ĐẦU ⇒ các token còn lại của cụm là
                    # MẢNH, không được chiếm thêm suất. GIỮ `sp[0]` — nó chính
                    # là đích sinh tự hồi quy.
                    manh.update(int(x) for x in sp[1:])
    # Bỏ MẢNH: token đã được một cụm dài hơn đại diện thì không chiếm thêm suất.
    # `Live`+`Live Blue`, `Bóng`+`Bóng đá`+`đá` — trước đây một thành viên ăn
    # 2–3 suất top-k. Đo: ③ chính xác 25,0% → 38,9% (§38).
    if bool(config.hub_chain_bo_manh):
        for t in manh:
            if len(ra) > 1:
                ra.pop(t, None)
    if float(config.hub_chain_dong_thuan) > 0.0 and dong_thuan:
        # Nhân trọng số theo SỐ THUỘC TÍNH cùng đề cử. Một thành viên được hai
        # thuộc tính trong ngữ cảnh cùng trỏ tới thì đáng tin hơn hẳn.
        k = float(config.hub_chain_dong_thuan)
        for t, ps_ in dong_thuan.items():
            if t in ra:
                ra[t] *= (1.0 + k * (len(ps_) - 1))
    return ra


def generate_autoregressive(
    state: GraphState,
    prompt: str,
    *,
    max_len: int = 48,
    config: HCAConfig | None = None,
    seed: int | None = None,
    agenda: list[int] | None = None,
    max_facts: int = 3,
    multi: bool = False,
    plan: "_SegmentPlan | None" = None,
    trace_out: list[dict[str, object]] | None = None,
) -> list[int]:
    """
    A3 (docx §4) — sinh TỪNG TOKEN thật, bám agenda. KHÁC verbatim: mỗi bước tính điểm toàn
    vocab rồi chọn 1 token → dựng chuỗi, có thể ra câu chưa lưu.

    `a_final = (sàn + a_micro) ⊙ (sàn + a_macro) ⊙ (sàn + a_entity) ⊙ (sàn + a_discourse)`:
      - a_micro = W_seq(w0→·): token nào thường theo sau w0 (mượt cục bộ, Markov);
      - a_macro = spreading_activation từ **agenda còn thiếu**: kéo về phía các ý chưa diễn đạt;
      - a_entity = entity_attention_read (keys = entity register, giống Q·K trong attention);
      - a_discourse = W_seq sau ranh giới `.`/`,` (liên kết giữa segment).
    EOS được phép khi agenda phủ đủ + cạnh w0→EOS đủ mạnh (đóng câu tự nhiên, không cụt).

    `plan` (nếu truyền) = kế hoạch segment/entity/agenda đã dựng sẵn (do `generate()` hợp nhất
    cấp) → dùng thẳng, không dựng lại → segments ↔ agenda nhất quán. Không truyền → tự dựng
    (backward-compat cho caller cũ dùng `multi`/`agenda`).
    """
    config = config or HCAConfig()
    ensure_frame_brackets(state)
    ensure_skip_spans(state)
    _ensure_neural_lm(state, config)
    rng = np.random.default_rng(seed)
    generated = list(state.vocab.encode_prompt(prompt))
    # TK_046 §22 — chuỗi ba bước, tính MỘT LẦN từ ngữ cảnh prompt (dựng index
    # tốn ~11s trên ckpt thật, không được làm mỗi bước).
    _chain_boost = float(config.hub_chain_boost)
    _chain: dict[int, float] = (
        hub_chain_proposals(state, generated, config) if _chain_boost > 0 else {})
    if not generated:
        return []
    q_ids, _ = _split_frame(generated, state.vocab)
    if plan is None:
        # §13.24 — CHẠM tới đây nghĩa là ĐANG Ở route AR/matrix, bất kể cờ
        # `ar_w_skip_primary` nói gì: đường production (W_skip primary) rẽ sang
        # `_wseq_fallback` từ `generate()` và KHÔNG BAO GIỜ vào đây.
        # `_qa_trim_prefix(config)` đọc cờ đó nên nó nói "generate() SẼ chọn đường nào",
        # không nói "ta ĐANG ở đường nào" — sai vị ngữ. Ép theo route thực tế.
        plan = _build_segment_plan(
            state, q_ids or generated, config, max_facts=max_facts, emitted_ids=generated,
            trim_prefix=False,
        )
    # Spectral stack (opt-in): 4 tầng đồng cấu; tắt copy cứng khi bật.
    spectral_on = bool(getattr(config, "ar_spectral_stack", False))
    spectral_stack = None
    if spectral_on:
        from nn_aivie.hca.spectral_stack import ensure_spectral_stack

        spectral_stack = ensure_spectral_stack(
            state,
            d=int(getattr(config, "ar_spectral_stack_d", 32)),
            sidecar=getattr(state, "_spectral_stack_sidecar", None),
        )
    # STYLE payoff: FACT segments + pivot + snippet đời thường (opt-in).
    if getattr(config, "ar_style_payoff", False) and plan.flat_agenda:
        from nn_aivie.hca.style_payoff import append_payoff_segments, plan_style_payoff
        po = plan_style_payoff(
            state, q_ids or generated, plan.entity_ids,
            max_payoff_tokens=int(getattr(config, "ar_style_payoff_max_tokens", 28)),
        )
        if po is not None:
            segs, flat, style_segs, pivot_segs = append_payoff_segments(
                list(plan.segments), plan.entity_ids, po,
                vocab=state.vocab,
                setup_segs=int(getattr(config, "ar_style_payoff_setup_segs", 1)),
                setup_max_tokens=int(
                    getattr(config, "ar_style_payoff_setup_max_tokens", 48)
                ),
                keep_tail_facts=bool(
                    getattr(config, "ar_style_payoff_keep_tail", False)
                ),
            )
            plan = _SegmentPlan(
                segs, plan.entity_ids, flat,
                hard_copy_segs=style_segs,
                pivot_lock_segs=pivot_segs,
            )
    if agenda is None:
        agenda = plan.flat_agenda
    entity_ids = plan.entity_ids

    agenda_seq = list(agenda)
    # Budget guard: max_len nhỏ thì không cần quét agenda hàng nghìn token mỗi bước.
    agenda_cap = max(32, max_len * 4)
    if len(agenda_seq) > agenda_cap:
        agenda_seq = agenda_seq[:agenda_cap]
    consumed = [False] * len(agenda_seq)
    # chỉ số bắt đầu mỗi segment trong agenda phẳng
    seg_starts: list[int] = []
    off = 0
    for seg in plan.segments:
        seg_starts.append(off)
        off += len(seg)
    n_segments = len(plan.segments)

    def _segment_idx_for_pos(pos: int) -> int:
        idx = 0
        for si in range(len(seg_starts) - 1, -1, -1):
            if pos >= seg_starts[si]:
                return si
        return 0

    def _segment_complete(si: int) -> bool:
        if si >= len(seg_starts):
            return True
        start = seg_starts[si]
        end = seg_starts[si + 1] if si + 1 < len(seg_starts) else len(agenda_seq)
        return all(consumed[j] for j in range(start, end))

    pending_bridge: int | None = None  # (giữ biến — bridge đã bỏ TK_019)
    out = list(generated)
    open_brackets: list[tuple[int, int]] = []
    seed_open_brackets(state, out, open_brackets)
    active_parents = (
        _active_parent_set(state, q_ids or generated)
        if config.ar_hlink_ctx_weight > 0 else frozenset()
    )
    vocab_size = len(state.vocab)
    eos_id = state.vocab.id_of(SPECIAL_EOS)
    seq_layer = state.adjacency.layer(EdgeKind.SEQUENTIAL)
    no_progress = 0  # số bước liên tiếp không phủ thêm agenda (guard chống trôi)

    # STYLE head (opt-in): CO_ACT boost từ token HOW trong prompt — tính 1 lần/lượt sinh.
    style_boost: np.ndarray | None = None
    if getattr(config, "ar_style_weight", 0.0) > 0 or getattr(config, "ar_style_boundary_inject", False):
        style_boost, _style_seeds = style_boost_from_coact(
            state, q_ids or generated, vocab_size,
            gidf_min=config.ar_style_gidf_min,
            c_min=config.ar_style_c_min,
            top_k=config.ar_style_top_k,
            max_seeds=getattr(config, "ar_style_max_seeds", 2),
        )
        if float(style_boost.max()) <= 0:
            style_boost = None

    style_injects_left = (
        int(getattr(config, "ar_style_boundary_max", 2))
        if getattr(config, "ar_style_boundary_inject", False)
        else 0
    )
    # Inventory lọc + phrase plan (tránh inject `cốt` từ raw W×IDF).
    style_plan: list[tuple[int, ...]] = []
    style_queue: list[int] = []  # phần còn lại của cụm đang chèn
    style_plan_i = 0
    style_cool_left = 0  # mixture: cooldown giữa 2 lần chèn
    if style_injects_left > 0:
        if getattr(config, "ar_style_use_inventory", True):
            style_plan = style_inject_plan(
                state, q_ids or generated,
                prefer_phrase=getattr(config, "ar_style_prefer_phrase", True),
                max_items=style_injects_left,
                use_corpus=getattr(config, "ar_style_corpus_inventory", True),
                discourse_ngram=getattr(config, "ar_style_discourse_ngram", True),
            )
            # Hadamard cũng dùng inventory (không raw domain noun)
            if style_boost is not None and style_plan:
                inv_boost = np.zeros(vocab_size, dtype=np.float64)
                for unit in style_plan:
                    for j in unit:
                        if j < vocab_size:
                            inv_boost[j] = max(inv_boost[j], float(style_boost[j]) + 1.0)
                if float(inv_boost.max()) > 0:
                    style_boost = inv_boost
        elif style_boost is not None:
            order = np.argsort(-style_boost)[:8]
            style_plan = [(int(j),) for j in order if style_boost[j] > 0][:style_injects_left]

    # TK_044 P3.2 — kênh COPY (Plan B hard / Plan A Hopfield). Luật 0: chỉ đọc trọng số đã học.
    _copy_ch = getattr(config, "ar_copy_channel", "off")
    _coref_left_set: frozenset = frozenset()
    _copy_boost: float = getattr(config, "ar_copy_boost", 15.0)
    _copy_cos_thresh: float = getattr(config, "ar_copy_cos_thresh", 0.5)
    from nn_aivie.training.coref_pass import SIG_W as _SIG_W, ENT as _ENT_SENT  # 1 nguồn hằng số
    # gate entity_ids theo gidf: bỏ function word ('rất/chăm/học' lọt entity_register) khỏi sentinel-SIG
    # + boost (M4 diagnostic: entity_register quá rộng → SIG toàn sentinel → không khớp). gidf=Luật-0 OK.
    _gidf_min = getattr(config, "ar_copy_entity_gidf_min", 3.0)
    _gidf_fn = state.parents._gidf if state.parents is not None else (lambda t: 0.0)
    _entity_ids_set: frozenset = frozenset(e for e in entity_ids if _gidf_fn(e) >= _gidf_min)
    # Plan A Hopfield: K matrix (n_coref × emb_dim). DETECT bằng cosine (độc-lập-n), KHÔNG softmax-weight.
    _hopfield_K: "np.ndarray | None" = None

    if _copy_ch != "off" and getattr(state, "coref", None):
        # Plan B: exact left-SIG set
        _coref_left_set = frozenset(sig[0] for sig in state.coref)

    if _copy_ch == "hopfield" and getattr(state, "coref", None) and state.embedding is not None:
        # Plan A: build K matrix emb(source_sig_left) — Luật 0: chỉ dùng E đã học.
        # SIG-embedding = normalize(mean(E[t] cho t ∈ left, t ≠ sentinel, t < vocab_size)).
        _E = state.embedding.matrix   # (vocab_size, emb_dim)
        _hop_sigs = list(state.coref.keys())   # [(left,right), ...]
        _K_rows = []
        for sig in _hop_sigs:
            toks = [t for t in sig[0] if t != _ENT_SENT and 0 <= t < len(_E)]
            if toks:
                v = _E[toks].mean(axis=0)
                n = np.linalg.norm(v)
                _K_rows.append(v / n if n > 1e-9 else v)
            else:
                _K_rows.append(np.zeros(_E.shape[1]))
        _hopfield_K = np.array(_K_rows, dtype=np.float32)   # (n_coref, emb_dim)

    def _runtime_left_sig(pos: int) -> tuple:
        """Left-context SIG tại pos: SIG_W token trước, entity_ids → sentinel."""
        start = max(0, pos - _SIG_W)
        return tuple(_ENT_SENT if t in _entity_ids_set else t for t in out[start:pos])

    for _ in range(max_len):
        if pending_bridge is not None:
            bridge = pending_bridge
            out.append(bridge)
            update_open_brackets(open_brackets, bridge, len(out) - 1, state, out)
            pending_bridge = None
            continue

        pending = [i for i, c in enumerate(consumed) if not c]
        w0 = out[-1]
        use_sparse = getattr(config, "ar_candidate_scoring", True)

        # Pivot glue / payoff lead / hard_copy — TẮT khi spectral stack (không splice verbatim).
        in_style_seg = bool(
            pending
            and plan.hard_copy_segs
            and _segment_idx_for_pos(pending[0]) in plan.hard_copy_segs
        )
        seg_i = _segment_idx_for_pos(pending[0]) if pending else -1
        in_pivot_lock = bool(
            (not spectral_on)
            and pending
            and plan.pivot_lock_segs
            and seg_i in plan.pivot_lock_segs
            and getattr(config, "ar_style_payoff_lock_pivot", True)
        )
        lead_lock_n = int(getattr(config, "ar_style_soft_lead_lock", 6))
        in_payoff_lead = False
        if (
            (not spectral_on)
            and pending
            and in_style_seg
            and not in_pivot_lock
            and lead_lock_n > 0
            and not getattr(config, "ar_style_payoff_hard_copy", False)
        ):
            start = seg_starts[seg_i] if seg_i < len(seg_starts) else 0
            offset_in_seg = pending[0] - start
            in_payoff_lead = 0 <= offset_in_seg < lead_lock_n
        if (not spectral_on) and (
            in_pivot_lock
            or in_payoff_lead
            or (in_style_seg and getattr(config, "ar_style_payoff_hard_copy", False))
        ):
            nxt = int(agenda_seq[pending[0]])
            out.append(nxt)
            update_open_brackets(open_brackets, nxt, len(out) - 1, state, out)
            consumed[pending[0]] = True
            no_progress = 0
            continue

        soft_peakiness = None
        soft_agenda_target: int | None = None

        if use_sparse:
            micro_d = build_micro_dict(w0, out, state, seq_layer, vocab_size, config)
            macro_d = macro_sparse(
                w0, pending, agenda_seq, state.adjacency, config, vocab_size,
            )
            candidates = collect_ar_candidates(
                w0, out, pending, agenda_seq, entity_ids, open_brackets,
                state, state.adjacency, seq_layer, vocab_size, eos_id, config,
                prompt_len=len(generated),
            )
            candidates.update(micro_d)
            candidates.update(macro_d)
            if style_boost is not None:
                # đảm bảo neighbor style vào tập ứng viên sparse
                for j in np.where(style_boost > 0)[0]:
                    candidates.add(int(j))
            for unit in style_plan:
                for j in unit:
                    candidates.add(int(j))
            for j in style_queue:
                candidates.add(int(j))
            scores = assemble_ar_scores(micro_d, macro_d, candidates, vocab_size, config)
            micro = micro_d  # backbone reachable check dùng dict
        else:
            micro = np.zeros(vocab_size, dtype=np.float64)
            for v, w in seq_layer.row(w0).items():
                if v < vocab_size and w > 0:
                    micro[v] = w
            if state.parents is not None and config.ar_seq_order2_weight > 0:
                prev = out[-2] if len(out) >= 2 else None
                hi = state.parents.seq_next(prev, w0, min_hi=config.ar_seq_order2_min)
                if hi:
                    bmax = micro.max() or 1.0
                    hmax = max(hi.values()) or 1.0
                    for v, c in hi.items():
                        if v < vocab_size:
                            micro[v] += config.ar_seq_order2_weight * (c / hmax) * bmax
            varlm = getattr(state, "varlm", None)
            if varlm is not None and getattr(config, "ar_varlm_weight", 0.0) > 0:
                ctx = out
                if getattr(varlm, "use_clusters", False) and state.parents is not None:
                    ctx = state.parents.seg.encode_tokens_with_clusters(out)
                pp = varlm.scores(ctx[-varlm.max_ctx:])
                if pp:
                    bmax = micro.max() or 1.0
                    pmax = max(pp.values()) or 1.0
                    for v, p in pp.items():
                        if v < vocab_size:
                            micro[v] += config.ar_varlm_weight * (p / pmax) * bmax
            mmax = micro.max() or 1.0

            # (TK_042 G10: kênh LM nằm trong `build_micro_dict` — đường THẬT là
            #  nhánh sparse. Nhánh dense này là fallback `ar_candidate_scoring=False`.)

            if pending:
                seeds = [(agenda_seq[i], 1.0) for i in pending]
                macro = spreading_activation_read(
                    w0, seeds, state.adjacency, config, vocab_size
                )
            else:
                macro = np.zeros(vocab_size, dtype=np.float64)
            gmax = macro.max() or 1.0

            scores = (
                (config.ar_micro_floor + micro / mmax)
                * (config.ar_macro_floor + config.ar_spread_weight * macro / gmax)
            )
        if style_boost is not None:
            multiply_scores_by_boost(
                scores, style_boost,
                floor=config.ar_style_floor, weight=config.ar_style_weight,
            )
        # Spectral stack: blend a_out vào scores AR (agenda vẫn prior "NÓI GÌ").
        if spectral_stack is not None:
            from nn_aivie.hca.spectral_stack import (
                forward_spectral_stack,
                spectral_forward_kwargs,
            )

            cand_ids = None
            if use_sparse:
                cand_ids = [int(v) for v in candidates if 0 <= int(v) < vocab_size]
            # pending agenda tokens vào S để gold content không lệch cửa sổ
            pending_ids = [int(agenda_seq[i]) for i in pending[:16] if i < len(agenda_seq)]
            if cand_ids is None:
                cand_ids = pending_ids
            else:
                cand_ids = list(dict.fromkeys([*cand_ids, *pending_ids]))
            emb = getattr(state, "embedding", None)
            emb_mat = getattr(emb, "matrix", None) if emb is not None else None
            a_out = forward_spectral_stack(
                spectral_stack,
                history_ids=out,
                candidate_ids=cand_ids,
                entity_ids=entity_ids,
                emb_matrix=emb_mat,
                **spectral_forward_kwargs(config, state, out),
            )
            if a_out.shape[0] != vocab_size:
                a_out = np.pad(a_out, (0, max(0, vocab_size - a_out.shape[0])))[:vocab_size]
            multiply_scores_by_boost(
                scores,
                a_out,
                floor=float(getattr(config, "ar_spectral_stack_floor", 0.15)),
                weight=float(getattr(config, "ar_spectral_stack_weight", 0.55)),
            )
        # prior THỨ TỰ mềm (docx §2.3): boost cửa sổ agenda kế tiếp theo trật tự nguồn, giảm dần.
        # W_seq vẫn quyết fluency; đây chỉ nghiêng để chống đảo lộn trên đồ thị thưa.
        for off, i in enumerate(pending[: config.ar_order_window]):
            t = agenda_seq[i]
            if t < vocab_size:
                if use_sparse:
                    m_hit = micro.get(t, 0.0) > 0
                else:
                    m_hit = micro[t] > 0
                if m_hit or state.adjacency.get_readout(w0, t) > 0:
                    scores[t] += config.ar_agenda_boost * (config.ar_order_decay ** off)

        if use_sparse:
            mask_non_surface_inplace(scores, state.vocab)
        else:
            scores = mask_non_surface(scores, state.vocab)
        anchor_ids = list(entity_ids) + [agenda_seq[i] for i in pending[:4]]
        scores = apply_g4_verify_v3(
            scores, w0, anchor_ids, state.adjacency, state.vocab, eos_id, config=config
        )
        if use_sparse:
            apply_repetition_penalty_inplace(
                scores, out, prompt_len=len(generated),
            )
        else:
            scores = apply_repetition_penalty(
                scores, out, state.vocab, prompt_len=len(generated)
            )

        gate_tr: dict = {}
        fb_fired = apply_frame_bracket_gate(
            scores, state, out, open_brackets, config, vocab_size,
            active_parents=active_parents, gate_trace=gate_tr,
        )
        skip_tr: dict = {}
        if pending:
            apply_skip_span_gate(
                scores, state, out, pending, agenda_seq, config, vocab_size,
                active_parents=active_parents, gate_trace=skip_tr,
            )
            if skip_tr and gate_tr is not None:
                gate_tr.update(skip_tr)

        # Soft payoff only (pivot đã lock ở trên): W_seq peaky vs hub+ctx.
        if in_style_seg and pending and not in_pivot_lock:
            seq_row = {int(k): float(v) for k, v in seq_layer.row(w0).items() if v > 0}
            gidf_w0 = None
            if state.parents is not None:
                gidf_w0 = float(state.parents._gidf(w0))  # noqa: SLF001
            allow_slip = bool(getattr(config, "ar_style_soft_slip", False))
            soft_peakiness, soft_agenda_target = apply_soft_style_decode(
                scores,
                emb=state.embedding,
                w0=w0,
                out_ids=out,
                seq_row=seq_row,
                pending=pending,
                agenda_seq=agenda_seq,
                vocab_size=vocab_size,
                peak_frac_min=float(getattr(config, "ar_style_soft_peak_frac", 0.35)),
                tie_eps=float(getattr(config, "ar_style_soft_tie_eps", 0.08)),
                hub_wseq_scale=float(getattr(config, "ar_style_soft_hub_wseq", 0.35)),
                ctx_k=int(getattr(config, "ar_style_soft_ctx", 6)),
                ctx_weight=float(getattr(config, "ar_style_soft_ctx_weight", 0.6)),
                agenda_weight=float(
                    getattr(config, "ar_style_soft_agenda_weight", 0.75)
                ),
                gidf_w0=gidf_w0,
                allow_slip=allow_slip,
                fluency_weight=float(
                    getattr(config, "ar_style_soft_fluency_weight", 0.55)
                ),
            )
            if (
                allow_slip
                and soft_agenda_target is not None
                and soft_peakiness is not None
                and not soft_peakiness.peaky
            ):
                for j, i in enumerate(pending[:3]):
                    if int(agenda_seq[i]) == soft_agenda_target:
                        for skip_i in pending[:j]:
                            consumed[skip_i] = True
                        pending = [i for i, c in enumerate(consumed) if not c]
                        break

        # A3 phase-2: index backbone (docx §1) — token kế tiếp theo trật tự nguồn, nếu ĐẠT ĐƯỢC
        # từ w0 → cho vượt trội (xương sống trật tự thay cạnh vị trí R-slot mà v3 không có).
        at_boundary = _at_boundary(state, w0, config)
        style_step = False
        style_emitted_this_step = False
        if getattr(config, "ar_style_boundary_inject", False):
            inject_ids: list[int] = []
            # tiếp tục cụm phrase đang chèn
            if style_queue:
                inject_ids = [style_queue.pop(0)]
                style_step = True
            elif style_injects_left > 0 and style_plan_i < len(style_plan):
                use_mix = bool(getattr(config, "ar_style_mixture", False))
                cos_thresh = float(getattr(config, "ar_style_cos_thresh", 0.12))
                cooldown = int(getattr(config, "ar_style_mixture_cooldown", 2))
                fact_safe = bool(getattr(config, "ar_style_fact_safe", True))
                # Ranh giới CHẶT: chỉ dấu câu / xuống dòng — tránh _ends_unit lửa giữa fact.
                w0_tok = state.vocab.token_of(w0)
                w0_core = (
                    w0_tok[1:] if w0_tok.startswith(("Ġ", "Ċ")) else w0_tok
                )
                at_punct = w0_core in ".!?…" or w0_tok in {"Ċ", "Ġ."} or w0_core.endswith(".")
                if use_mix and fact_safe:
                    slot_ok = at_punct
                elif use_mix:
                    slot_ok = True
                else:
                    slot_ok = at_boundary
                if use_mix and cooldown > 0 and style_cool_left > 0:
                    slot_ok = False
                if slot_ok:
                    if use_mix:
                        right = (
                            int(agenda_seq[pending[0]])
                            if pending and pending[0] < len(agenda_seq)
                            else None
                        )
                        pick = pick_style_by_cosine(
                            state.embedding, w0, style_plan,
                            start_i=style_plan_i, cos_thresh=cos_thresh,
                            right_id=right,
                        )
                        if pick is not None and right is not None and state.embedding is not None:
                            agenda_c = token_cosine(state.embedding, w0, right)
                            margin = float(getattr(config, "ar_style_cos_margin", 0.05))
                            if pick.cosine < agenda_c + margin:
                                pick = None
                        if pick is not None:
                            for k in range(style_plan_i, len(style_plan)):
                                if style_plan[k] == pick.unit:
                                    style_plan_i = k + 1
                                    break
                            style_injects_left -= 1
                            inject_ids = [pick.head]
                            style_queue = list(pick.unit[1:])
                            style_step = True
                            style_cool_left = cooldown
                    else:
                        unit = style_plan[style_plan_i]
                        style_plan_i += 1
                        style_injects_left -= 1
                        inject_ids = [unit[0]]
                        style_queue = list(unit[1:])
                        style_step = True
            if style_step and inject_ids:
                fin = scores[np.isfinite(scores)]
                base = float(fin.max()) if fin.size else 0.0
                boost = float(getattr(config, "ar_style_boundary_boost", 25.0))
                if getattr(config, "ar_style_mixture", False) and state.embedding is not None:
                    c = token_cosine(state.embedding, w0, inject_ids[0])
                    thr = float(getattr(config, "ar_style_cos_thresh", 0.12))
                    cos_scale = max(0.0, (c - thr) / max(1e-6, 1.0 - thr))
                    boost = boost * (0.35 + 0.65 * cos_scale)
                for j in inject_ids:
                    if j < vocab_size:
                        extra = float(style_boost[j]) if style_boost is not None else 0.0
                        if np.isfinite(scores[j]):
                            scores[j] = base + boost + extra
                        else:
                            scores[j] = base + boost
                style_emitted_this_step = True
            if style_cool_left > 0 and not style_step:
                style_cool_left -= 1

        if pending and config.ar_index_backbone and not style_step:
            t = (
                int(soft_agenda_target)
                if in_style_seg and soft_agenda_target is not None
                else agenda_seq[pending[0]]
            )
            # TK_009 vòng 2: tại RANH GIỚI diễn ngôn (w0 = token frame "Đáp:" / dấu câu),
            # W_seq từ w0 không mang tín hiệu trật tự → tin trật tự NGUỒN vô điều kiện.
            # (Bug đo được: "Việc đọc sách"→"đọc Việc sách" — agenda[0] không reachable từ
            # "Đáp:" nên mất boost, token giữa nhảy lên đầu, xáo hệ thống mọi đầu segment.)
            # TK_009 vòng 3 (trace bắt tận tay): tại boundary, agenda[0] có thể bị G4 mask
            # -inf (token hiếm dưới ngưỡng grounding) → boost bị isfinite bỏ qua → "Việc đọc
            # sách"→"đọc Việc sách". Token agenda đến từ CHA ĐÃ HỌC = grounded bởi xuất xứ
            # → backbone được cứu nó khỏi -inf tại boundary (G4 vẫn giữ nguyên cho off-agenda).
            reachable = np.isfinite(scores[t]) and (
                (micro.get(t, 0.0) > 0 if use_sparse else micro[t] > 0)
                or state.adjacency.get_readout(w0, t) > 0
            )
            apply_backbone = True
            if in_style_seg and soft_peakiness is not None:
                if soft_peakiness.peaky and state.embedding is not None:
                    margin = float(getattr(config, "ar_style_soft_index_margin", 0.05))
                    agenda_c = token_cosine(state.embedding, w0, t)
                    top_c = 0.0
                    if soft_peakiness.top_ids:
                        top_c = token_cosine(
                            state.embedding, w0, soft_peakiness.top_ids[0],
                        )
                    if agenda_c + margin < top_c:
                        apply_backbone = False
                elif not soft_peakiness.peaky:
                    # hub: soft agenda boost đã làm trong apply_soft_style_decode
                    apply_backbone = False
            if t < vocab_size and apply_backbone and (at_boundary or reachable):
                finite = scores[np.isfinite(scores)]
                cur_max = float(finite.max()) if finite.size else 0.0
                base = float(scores[t]) if np.isfinite(scores[t]) else 0.0
                scores[t] = max(base, cur_max) * (1.0 + config.ar_index_margin)

        if eos_id < vocab_size:
            if pending:  # còn ý chưa phủ → cấm dừng
                scores[eos_id] = -np.inf
            elif getattr(config, "ar_no_agenda", False):
                # Không agenda: đừng force EOS ngay bước 1 — chỉ dừng khi cạnh w0→EOS đã học.
                fm = (
                    scores[np.isfinite(scores)].max()
                    if np.any(np.isfinite(scores))
                    else 0.0
                )
                if state.adjacency.get_readout(w0, eos_id) >= config.hybrid_eos_threshold:
                    scores[eos_id] = float(fm) * 1.5 + 1.0
                else:
                    scores[eos_id] = -np.inf
            else:  # agenda phủ đủ → dừng NGAY (không sinh rác sau khi hết ý)
                fm = scores[np.isfinite(scores)].max() if np.any(np.isfinite(scores)) else 0.0
                scores[eos_id] = fm + 1.0

        # TK_044 P3.2 Plan B — copy slot: nếu left-context SIG khớp cạnh coref đã học
        # → boost entity_ids tokens (không gọi string/cap_class — Luật 0).
        if _copy_ch == "hard" and _coref_left_set and _entity_ids_set:
            lsig = _runtime_left_sig(len(out))
            if lsig in _coref_left_set:
                fin = scores[np.isfinite(scores)]
                base = float(fin.max()) if fin.size else 0.0
                for eid in _entity_ids_set:
                    if eid < vocab_size:
                        scores[eid] = base + _copy_boost

        # TK_044 P3.2 Plan A — Hopfield mềm: q=emb(left_ctx); DETECT copy-slot bằng COSINE thô
        # cosine(q, copy-slot đã học) VƯỢT ngưỡng (độc-lập-n; softmax-weight chia mass cho n_coref →
        # match hoàn hảo cũng ~1/n < thresh, KHÔNG BAO GIỜ fire ở checkpoint thật). Luật 0: chỉ đọc E.
        elif _copy_ch == "hopfield" and _hopfield_K is not None and _entity_ids_set \
                and state.embedding is not None:
            _E = state.embedding.matrix
            ctx_toks = [out[i] for i in range(max(0, len(out) - _SIG_W), len(out))
                        if out[i] not in _entity_ids_set and 0 <= out[i] < len(_E)]
            if ctx_toks:
                q = _E[ctx_toks].mean(axis=0).astype(np.float32)
                q_n = np.linalg.norm(q)
                if q_n > 1e-9:
                    q /= q_n
                    cos_max = float((_hopfield_K @ q).max())   # K đã chuẩn hoá → dot = cosine
                    if cos_max > _copy_cos_thresh:
                        fin = scores[np.isfinite(scores)]
                        base = float(fin.max()) if fin.size else 0.0
                        for eid in _entity_ids_set:
                            if eid < vocab_size:
                                scores[eid] = base + _copy_boost

        if _chain:
            # Nhân với token ĐÃ có điểm; CỘNG cho token chưa có — thành viên
            # thật sự mới («con khỉ») thường nằm ngoài phân bố của đường sinh.
            # CHUẨN HOÁ VỀ [0,1] TRƯỚC KHI BƠM — §78. `_w` là trọng số THÔ của
            # đề cử và sau §71/§72 nó không còn bị chặn: đo trên `classrich2`
            # thấy hệ số nhân `(1+boost·w)` chạy từ 5× tới 923× TRONG CÙNG một
            # prompt. Một đề cử nặng vì thế nuốt cả phân bố, và thứ tự giữa các
            # đề cử bị chính độ lớn thô quyết định thay vì thứ hạng.
            _f0 = scores[np.isfinite(scores) & (scores > 0)]
            _med = float(np.median(_f0)) if _f0.size else 0.0
            _wmax = max(_chain.values()) if _chain else 1.0
            _wmax = float(_wmax) if _wmax > 0 else 1.0
            for _t, _w in _chain.items():
                _w = float(_w) / _wmax
                if not (0 <= _t < vocab_size):
                    continue
                if np.isfinite(scores[_t]) and scores[_t] > 0:
                    scores[_t] = scores[_t] * (1.0 + _chain_boost * _w)
                elif _med > 0:
                    scores[_t] = _med * _chain_boost * _w

        if not np.any(np.isfinite(scores)) or float(np.nanmax(scores)) <= 0:
            break
        nxt = (
            int(np.argmax(scores))
            if config.ar_temperature <= 0
            else _sample_ar(scores, config, rng)
        )
        if trace_out is not None:
            tok_of = state.vocab.token_of
            endpoints = (
                getattr(state.parents, "hlink_endpoints", None) or {}
                if state.parents is not None
                else {}
            )
            trace_out.append(
                {
                    "step": len(trace_out),
                    "w0": tok_of(w0),
                    "chosen": tok_of(nxt) if nxt != eos_id else SPECIAL_EOS,
                    "agenda_pending": [tok_of(agenda_seq[i]) for i in pending],
                    "frame_bracket_gate": fb_fired,
                    "gate": {
                        "opener": (
                            decode_endpoint_label(gate_tr["opener"], endpoints, state.vocab)
                            if "opener" in gate_tr
                            else None
                        ),
                        "gap": gate_tr.get("gap"),
                        "override": tok_of(gate_tr["override"]) if "override" in gate_tr else None,
                    } if gate_tr else None,
                    "top": _top_scores_ui(scores, state.vocab),
                }
            )
        if nxt == eos_id:
            break
        out.append(nxt)
        update_open_brackets(open_brackets, nxt, len(out) - 1, state, out)
        progressed = False
        for i in pending:
            if agenda_seq[i] == nxt:
                consumed[i] = True
                progressed = True
                break

        if pending:
            # style inject cố ý không phủ agenda → không tính vào patience (tránh cụt sau 2 marker)
            if style_emitted_this_step:
                no_progress = 0
            else:
                no_progress = 0 if progressed else no_progress + 1
                if no_progress >= config.ar_patience:
                    break
    return out


def _sample_ar(scores: np.ndarray, config: HCAConfig, rng: np.random.Generator) -> int:
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    k = min(
        int(getattr(config, "hybrid_top_k", 12)),
        int(np.sum(np.isfinite(finite) & (finite > 0))),
    )
    if k <= 0:
        return int(np.argmax(scores))
    top = np.argpartition(-finite, k - 1)[:k]
    top = top[np.argsort(-finite[top])]
    logits = finite[top] / max(1e-6, float(config.ar_temperature))
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    top_p = float(getattr(config, "hybrid_top_p", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        cdf = np.cumsum(probs)
        cut = int(np.searchsorted(cdf, top_p, side="left")) + 1
        cut = max(1, min(cut, len(probs)))
        probs = probs[:cut]
        probs /= probs.sum()
        top = top[:cut]
    return int(top[rng.choice(len(top), p=probs)])


def _knowledge_extra_facts(
    state: GraphState,
    q_ids: list[int],
    base_out: list[int],
    prompt_len: int,
    config: HCAConfig,
    plan_out: dict[str, Any] | None = None,
) -> list[int]:
    """
    TK_022b (tác giả 07-06: "sinh thiếu fact, chỉ câu đơn") — KNOWLEDGE mode bồi thêm K fact
    RỜI về đúng thực thể (TK_023 rank_facts_by_stimulus), mỗi fact một câu, KHÔNG bridge/đại từ.

    Fact chính vẫn A3; fact bồi = completion nguyên khối (kiến thức tra cứu — verbatim chấp
    nhận được ở tầng này; sổ nợ: A3-hoá từng fact bồi khi cần).
    """
    ps = state.parents
    if ps is None or config.knowledge_extra_facts <= 0:
        return base_out
    q_content = tuple(_content_ids(q_ids, state.vocab, skip=set()))
    if _knowledge_entity_span(ps, q_content) is None:
        return base_out
    if not _knowledge_extra_facts_eligible(ps, q_content, config):
        return base_out
    plan = knowledge_extra_fact_plan(state, q_ids, base_out, prompt_len, config)
    if plan_out is not None and plan is not None:
        plan_out.clear()
        plan_out.update(plan)
    if not plan:
        return base_out
    period = state.vocab._token_to_id.get(".")
    out = list(base_out)
    for item in plan["facts"]:
        if not item["picked"]:
            continue
        pi = item["idx"]
        # Trước đây `_strip_leading_punct_ids` bỏ dấu câu đầu completion để tránh `. .`.
        # Bỏ (Luật 0): tránh `. .` là tri thức chính tả viết tay. W_seq đã học `(., .)` yếu
        # nên mạng tự không nối hai dấu chấm; nếu nó vẫn nối thì đó là tín hiệu train thiếu,
        # không phải thứ nên vá bằng danh sách ký tự ở đường sinh.
        comp = list(ps.completion(pi))
        if not comp:
            continue
        if out and period is not None and out[-1] != period:
            out.append(period)
        out.extend(comp)
    return out


def _entity_rerank_primary(
    state: GraphState, query: list[int], config: HCAConfig
) -> int | None:
    """TK_027 bước 4 — khi câu hỏi khớp một THỰC-THỂ ĐẦY ĐỦ (Entity:Mây) và candidate PRIMARY
    SÁT NÚT (bag-of-words không tách: "Mây là gì"→Đường Ngô Mây 0.267 ≈ mây-def 0.240), re-rank
    các ứng viên gần-đỉnh theo dot với hồ-sơ thực-thể → chọn fact TRUNG-TÂM thực-thể (mây khí-tượng).
    CHỈ đụng khi sát nút → winner rõ (guard curated) KHÔNG đổi. Trả pi mới hoặc None."""
    ps = state.parents
    if ps is None:
        return None
    q_content = tuple(_content_ids(query, state.vocab, skip=set()))
    span = _knowledge_entity_span(ps, q_content)
    if span is None:
        return None
    entity, _primary = span
    co = state.adjacency.layer(EdgeKind.CO_ACTIVATION)
    afield = ps.query_entity_field(entity, co, state.vocab)
    if not afield and getattr(config, "knowledge_entity_resonance_fallback", True):
        afield = ps.query_entity_resonance_field(entity, co, state.vocab)
    if not afield:
        return None
    cands = ps.rank_matches(
        list(query), state.vocab, state.embedding,
        top_n=8, min_priority=config.generation_min_priority,
    )
    if len(cands) < 2:
        return None
    top_pi, top_score = cands[0]
    margin = getattr(config, "primary_entity_rerank_margin", 0.05)
    near = [pi for pi, s in cands if s >= top_score - margin]
    if len(near) < 2:
        return None  # winner rõ (điểm tách) → không đụng (bảo vệ guard)
    best = max(near, key=lambda pi: ps.parent_field_dot(pi, afield))
    # chỉ đổi khi ứng viên mới TRỘI hơn đỉnh cũ về entity-dot (thực sự trung-tâm hơn)
    if best == top_pi or ps.parent_field_dot(best, afield) <= ps.parent_field_dot(top_pi, afield):
        return None
    return best


def _ensure_neural_lm(state, config):
    """Nạp LM 1 lần vào `state.neural_lm` (TK_042 G10). Không có path/torch → no-op."""
    path = getattr(config, "ar_lm_path", "")
    if not path or getattr(config, "ar_lm_weight", 0.0) <= 0:
        return
    if getattr(state, "neural_lm", None) is not None:
        return
    from nn_aivie.core.neural_lm import NeuralLM

    state.neural_lm = NeuralLM.load(path, len(state.vocab))


def generate(
    state: GraphState,
    prompt: str,
    *,
    max_len: int = 48,
    config: HCAConfig | None = None,
    seed: int | None = None,
    multi: bool = False,
    max_facts: int = 3,
    trace_out: list[dict[str, object]] | None = None,
    fact_plan_out: dict[str, Any] | None = None,
) -> list[int]:
    """
    ĐIỂM VÀO SINH DUY NHẤT (hợp nhất hybrid/composed) — như LLM: sinh TỪNG TOKEN cho đến khi
    tự đóng câu (EOS nổi từ đồ thị) hoặc chạm `max_len`.

    Truy hồi nơ-ron cha chỉ để **lập agenda** (ý cần phủ) + neo entity; bề mặt token LUÔN do
    vòng A3 (`generate_autoregressive`) sinh — KHÔNG dội đáp verbatim. `multi=True` gom nhiều
    fact về thực thể trọng tâm (đáp nhiều câu). Lưới an toàn (`ar_verbatim_fallback`, chỉ plan
    crisp): A3 phủ agenda < `essay_ar_min_cov` → dội đáp cha khớp nhất verbatim (giữ đúng fact
    câu đã thuộc). Đặt cờ =False để đo "thuần A3" không lưới.
    """
    config = config or HCAConfig()
    if _FORCE_MATRIX_FOR_TESTS:
        config = _dc_replace(config, ar_w_skip_primary=False)
    rng = np.random.default_rng(seed)
    generated = list(state.vocab.encode_prompt(prompt))
    if not generated:
        return []
    max_new = max(0, int(max_len))

    def _finalize(ids: list[int]) -> list[int]:
        return ids[: len(generated) + max_new]

    eos_id = state.vocab.id_of(SPECIAL_EOS)

    # Opt-in modes — không đụng W_skip primary.
    _gen_mode = str(getattr(config, "generation_mode", "w_skip") or "w_skip")
    if _gen_mode in ("frame_da_khe", "compose_2cha", "analogy_la"):
        from pathlib import Path as _P
        root = _P(getattr(state, "_checkpoint_path", "") or ".").resolve()
        if root.is_file():
            root = root.parent.parent  # checkpoints/x.pkl → repo root guess
        if not (root / "out" / "token_labels_v12.json").exists():
            root = _P.cwd()
        if _gen_mode == "analogy_la":
            # TK_046 P98G — điền X là ? qua cầu thuộc_tính (mini OK; không cần label_map).
            from nn_aivie.hca.analogy_bridge_gen import (
                AnalogyBridgeConfig,
                generate_analogy_bridge,
            )
            acfg = AnalogyBridgeConfig()
            cont, _meta = generate_analogy_bridge(state, prompt, cfg=acfg)
        elif _gen_mode == "frame_da_khe":
            from nn_aivie.hca.frame_slot_gen import (
                FrameSlotConfig,
                generate_frame_slot,
            )
            fcfg = FrameSlotConfig(
                seed=int(seed) if seed is not None else 42,
                label_map="out/token_labels_v12.json",
                ws_model="out/models/ws_crf_uts_full",
            )
            cont, _meta = generate_frame_slot(state, prompt, cfg=fcfg, root=root)
        else:
            from nn_aivie.hca.compose_2cha import (
                ComposeConfig,
                generate_compose_2cha,
            )
            ccfg = ComposeConfig(
                seed=int(seed) if seed is not None else 42,
                label_map="out/token_labels_v12.json",
                ws_model="out/models/ws_crf_uts_full",
            )
            cont, _meta = generate_compose_2cha(
                state, prompt, cfg=ccfg, root=root)
        return _finalize(list(generated) + list(cont))

    # §12.131/133 — W_skip primary: vote + ưu tiên agenda multizone (không matrix/path).
    if bool(getattr(config, "ar_w_skip_primary", True)):
        q_ids, _ans = _split_frame(generated, state.vocab)
        query = list(q_ids) if q_ids else list(generated)
        # §99 — LÙI VỀ TỪ CHÍNH phải áp Ở ĐÂY. Đây là khối dựng agenda CHẠY
        # TRƯỚC (dòng ~4126); khối `generate()` phía dưới (~4257) chạy SAU nên vá
        # ở đó KHÔNG đổi một byte nào — đo được, lần thứ bảy cùng khuôn trong
        # phiên (LOI1, §79, §81, §83, §87, §90, nay §99).
        if getattr(config, "lui_ve_tu_chinh", False):
            _l0 = _lui_ve_tu_chinh(state, list(query), config)
            if _l0:
                query = _l0
        agenda: list[int] = []
        ents: tuple[int, ...] = ()
        if state.parents is not None and len(state.parents) > 0:
            cfg_plan = config
            if bool(getattr(config, "ar_w_skip_use_mz_agenda", True)):
                cfg_plan = _dc_replace(config, ar_multizone=True)
            plan = _build_segment_plan(
                state, query, cfg_plan,
                max_facts=int(getattr(config, "ar_multizone_facts_top", 1) or 1) + 2,
                emitted_ids=generated,
            )
            agenda = list(plan.flat_agenda)
            ents = tuple(plan.entity_ids)
        return _finalize(_wseq_fallback(
            state, generated, eos_id, max_len, config, rng,
            agenda_ids=agenda, entity_ids=ents,
        ))

    q_ids, answer_so_far = _split_frame(generated, state.vocab)
    ps = state.parents

    # Đo stack / fluency thuần: bỏ MZ/fact/pattern → W_seq continuation (+ stack nếu bật).
    if getattr(config, "ar_no_agenda", False):
        return _finalize(_wseq_fallback(state, generated, eos_id, max_len, config, rng))

    # Không có cha, hoặc đang nối tiếp GIỮA đáp → A3 thuần W_seq (không plan truy hồi).
    if ps is None or len(ps) == 0 or answer_so_far:
        return _finalize(_wseq_fallback(state, generated, eos_id, max_len, config, rng))

    # R1 (TK_017 §11 F5/G1): prompt MỞ ĐẦU bằng bracket-OPENER ("nếu/tuy/vì…" — token có
    # cạnh FRAME_BRACKET đã mine, so theo fold-group) và KHÔNG mang frame Hỏi:/Đáp: →
    # đây là câu TIẾP DIỄN (continuation), không phải câu hỏi tra cứu. Retrieval crisp với
    # câu mở match sát ngưỡng qua từ-chung ("nếu trời mưa"→cha "Mặt trời cung cấp…" 0.192,
    # đo 07-05) mà ngưỡng score KHÔNG tách được khỏi Para/Syn hợp lệ (0.121–0.139) → route
    # bằng CẤU TRÚC thay vì score: đi thẳng W_seq + bracket-gate. Luật 0 sạch (đọc cạnh
    # mined, không danh sách liên từ). Guard QA nguyên vẹn (prompt "Hỏi: …" không trigger).
    if config.ar_opener_continuation and q_ids == generated:  # không có frame Đáp:
        ensure_frame_brackets(state)
        fb = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
        first = generated[0]
        first_is_opener = bool(fb.row(first))
        if not first_is_opener:
            # biến thể fold ("Nếu" đầu câu ≡ "nếu") — thuộc-tính-token TK_012, không string-op
            alt = state.vocab._token_to_id.get(state.vocab.fold(first))
            first_is_opener = alt is not None and bool(fb.row(alt))
        if first_is_opener:
            # TK_031 (đồng bộ fix bước-2 TK_027): opener là THỰC-THỂ của prompt ("Nhà Lý dời
            # đô về" — "Nhà" trong lớp FB nhiễu) → KHÔNG phải mệnh-đề mở → bỏ route này, rơi
            # xuống R4/crisp (varlm/pattern xử đúng "Thăng Long"). entity_register demote
            # FB-opener thật ("nếu") nên "nếu trời mưa" vẫn đi lối cũ. Luật 0 (idf/vị-trí).
            content0 = list(_content_ids(generated, state.vocab, skip=set()))
            reg0 = ps.entity_register_ids(tuple(content0)) if content0 else ()
            fold0 = state.vocab.fold_group_id
            if reg0 and content0 and fold0(content0[0]) in {fold0(t) for t in reg0}:
                first_is_opener = False
        if first_is_opener:
            # R3b: pattern-completion pool văn nói TRƯỚC (mẫu thật "nếu…thì…"); trượt → W_seq
            if config.ar_conv_pool:
                tail = _conv_pool_tail(state, generated, config)
                if tail:
                    return _finalize(_emit_tail(
                        state, prompt, generated, tail,
                        max_len=max_len, config=config, seed=seed, eos_id=eos_id,
                        trace_out=trace_out,
                        verbatim_only=not config.ar_tail_as_agenda_opener,
                    ))
            return _finalize(_finish_or_pattern(
                state, generated, eos_id, max_len, config, rng,
                min_suffix_tokens=2, prompt=prompt, seed=seed, trace_out=trace_out,
            ))

    # TK_021 T2: frame QA ruột-continuation — "Hỏi: Nếu trời mưa thì ? Đáp:" không phải WH
    # → pool văn nói (WH-detector thật; reverted frame-opener route giờ an toàn).
    if (
        config.ar_opener_continuation
        and q_ids != generated
        and _is_frame_opener_continuation(q_ids, state, config)
    ):
        if config.ar_conv_pool:
            tail = _conv_pool_tail(state, q_ids, config)
            if tail:
                return _finalize(_emit_tail(
                    state, prompt, generated, tail,
                    max_len=max_len, config=config, seed=seed, eos_id=eos_id,
                    trace_out=trace_out,
                    verbatim_only=not config.ar_tail_as_agenda_opener,
                ))
        return _finalize(_finish_or_pattern(
            state, generated, eos_id, max_len, config, rng,
            min_suffix_tokens=max(2, len(_content_ids(q_ids, state.vocab, skip=set()))),
            prompt=prompt, seed=seed, trace_out=trace_out,
        ))

    # R4: prompt mở không frame, không bracket-opener → thử pattern-completion (essay+conv)
    # trước retrieval crisp / A3 (substrate order-1 không cứu được câu mở lửng).
    if config.ar_pattern_completion and q_ids == generated and _prompt_is_open_continuation(
        generated, state.vocab, ps,
        min_filler_diversity=config.wh_min_filler_diversity,
    ):
        q_content = _content_ids(generated, state.vocab, skip=set())
        min_suf = max(2, len(q_content))
        tail = _pattern_completion_tail(state, generated, config, min_suffix_tokens=min_suf)
        # TK_031: đuôi ĐÚNG-mà-NGẮN ("Nhà Lý dời đô về"→"Thăng Long", 2 tok < min_suf=5) bị
        # min_suf từ chối oan. Chấp nhận đuôi ngắn KHI VarOrderLM (tín-hiệu-học, ctx dài) XÁC
        # NHẬN token đầu của đuôi là kế-tiếp khả dĩ nhất — gate bằng đếm đã học, không rule tay.
        varlm = getattr(state, "varlm", None)
        if not tail and varlm is not None and getattr(config, "ar_varlm_weight", 0.0) > 0:
            short = _pattern_completion_tail(state, generated, config, min_suffix_tokens=2)
            if short:
                pp = varlm.scores(generated[-varlm.max_ctx:])
                if pp and short[0] == max(pp, key=pp.get):
                    tail = short
        if tail:
            return _finalize(_emit_tail(
                state, prompt, generated, tail,
                max_len=max_len, config=config, seed=seed, eos_id=eos_id,
                trace_out=trace_out,
            ))

    # Hướng 1 (TK_014): đồng bộ cờ kích thích cụm phổ biến (WH/frame) vào store cho mọi
    # rank_matches/best_match phía sau đọc chung.
    ps.match_common_clusters = config.match_common_clusters
    ensure_frame_brackets(state)
    ensure_skip_spans(state)
    _ensure_neural_lm(state, config)
    # entity_register_ids + distinct primary cần vocab + FRAME_BRACKET
    ps._vocab_ref = state.vocab
    ps._fb_layer_ref = state.adjacency.layer(EdgeKind.FRAME_BRACKET)
    ps.focus_pos_alpha = config.focus_pos_alpha
    ps.directed_bigram_bonus = getattr(config, "directed_bigram_bonus", 0.0)  # TK_029
    ps.directed_bigram_min_idf = getattr(config, "directed_bigram_min_idf", 2.0)
    ps.directed_bigram_dmax = int(getattr(config, "directed_bigram_dmax", 1))  # §12.124 ③
    ps.directed_bigram_lam = float(getattr(config, "directed_bigram_lam", 0.5))
    ps.directed_bigram_accum = str(getattr(config, "directed_bigram_accum", "max"))  # §12.124 ③
    # TK_025: WH-mode/readout KNOWLEDGE/theme-first/topic-first/slot-consistency ĐÃ GỠ (Luật 0) —
    # không set ps.wh_min_filler_diversity/topic_first/slot_consistency/readout_mode nữa.

    query = q_ids if q_ids else generated
    # §99 — LÙI VỀ TỪ CHÍNH khi cụm chủ ngữ không có mục định nghĩa (§98).
    # Chỉ đổi TRUY VẤN dùng để tra cha; `generated` giữ nguyên nên đầu ra vẫn
    # mở đầu bằng đúng chữ người dùng gõ.
    if getattr(config, "lui_ve_tu_chinh", False):
        _lui = _lui_ve_tu_chinh(state, list(query), config)
        if _lui:
            query = _lui
    verbatim: tuple[int, ...] = ()  # đáp cha khớp nhất — dùng cho lưới an toàn crisp
    plan = _SegmentPlan([], (), [])
    plan_locked = False  # True khi multizone đã dựng multi-segment

    if True:  # TK_019: multi (ghép fact) đã bỏ — luôn crisp
        # Hub guard: câu hỏi họ mẫu với filler LẠ → route đúng thực thể hoặc KHÔNG bịa.
        completion: tuple[int, ...] | None = None
        pi: int | None = None
        seg: list[int] = []
        if config.hybrid_hub_guard:
            guarded = _hub_guarded_completion(state, q_ids, generated, config)
            if guarded is not False:
                if guarded:  # đáp slot-consistent (statement/QA đúng thực thể)
                    completion = tuple(guarded)
                else:  # họ mẫu, filler lạ, không đáp đúng → không bịa (đáp rỗng)
                    return _finalize(generated)
        if completion is None:
            # Multi-zone FACT head (opt-in) trước facts_about đơn — đa entity → đa segment.
            mz = _plan_from_multizone(state, query, config, emitted_ids=generated)
            if mz is not None and mz.flat_agenda:
                plan = mz
                parts: list[int] = []
                for seg_i in mz.segments:
                    parts.extend(seg_i)
                verbatim = tuple(parts)
                plan_locked = True
        if completion is None and not plan_locked:
            # FIX GROUNDING (TK_042 2026-07-21, đo local): đây là ĐƯỜNG OUTPUT crisp thật —
            # `completion` → `verbatim` (dội ra khi A3 phủ agenda thấp). rank_matches THOÁI HOÁ cho
            # tên riêng đa-token: "Thăng Long"→8 bài sử ĐỒNG ĐIỂM 0.8922, bài đúng KHÔNG có trong
            # top → _pick_parent_agenda/best_match trôi (Gia Long). facts_about-theme subject-aware
            # TÌM ĐÚNG (Thăng Long). Ưu tiên nó cho truy vấn CÓ focus thực thể; rank_matches fallback.
            if config.compose_theme_weight > 0:
                _qc = list(dict.fromkeys(_content_ids(query, state.vocab, skip=set())))
                _focus = ps.top_focus_tokens(tuple(_qc), k=2, vocab=state.vocab) if _qc else ()
                if _focus:
                    _tf = _prefer_atomic_facts(
                        list(ps.facts_about(
                            _focus, top_k=1, vocab=state.vocab,
                            theme_weight=config.compose_theme_weight,
                        )),
                        ps, state.vocab,
                    )
                    if _tf:
                        pi = _tf[0]
                        completion = _completion_from_parent(ps, pi)
                        seg, _fully = _agenda_from_parent_index(
                            ps, pi, generated, state.vocab, bound=_bound_of(state),
                            trim_prefix=_qa_trim_prefix(config),
                        )
        if completion is None and not plan_locked:
            _rm = ps.rank_matches
            if getattr(config, "retrieval_gop_hoa_thuong", False):
                def _rm(*_a, **_k):          # noqa: E306 — bind `ps`
                    return _rank_matches_gop(ps, *_a, **_k)
            ranked = _rm(
                query, state.vocab, state.embedding,
                fuzzy_topk=config.hybrid_fuzzy_topk,
                fuzzy_min_sim=config.hybrid_fuzzy_min_sim,
                fuzzy_scale=config.hybrid_fuzzy_weight,
                min_priority=config.generation_min_priority,
                qa_completion_mode=True,
                top_n=12,
            )
            ranked = [(p, s) for p, s in ranked if s >= config.hybrid_parent_min_score]
            pi = None
            seg: list[int] = []
            if ranked:
                if getattr(config, "primary_entity_rerank", True):
                    rr = _entity_rerank_primary(state, query, config)
                    if rr is not None:
                        ranked = [(rr, s) for p, s in ranked if p == rr] + [
                            (p, s) for p, s in ranked if p != rr
                        ]
                pi, seg, completion_t = _pick_parent_agenda(
                    ps, ranked, generated, state.vocab,
                    trim_prefix=_qa_trim_prefix(config), bound=_bound_of(state),
                )
                if pi is not None:
                    completion = completion_t
            if completion is None:
                match = ps.best_match(
                    query, state.vocab, state.embedding,
                    fuzzy_topk=config.hybrid_fuzzy_topk,
                    fuzzy_min_sim=config.hybrid_fuzzy_min_sim,
                    fuzzy_scale=config.hybrid_fuzzy_weight,
                    min_score=config.hybrid_parent_min_score,
                    pool_min_priority=config.generation_min_priority,
                    qa_completion_mode=True,
                )
                if match is not None:
                    pi = match[0]
                    if getattr(config, "primary_entity_rerank", True):
                        rr = _entity_rerank_primary(state, query, config)
                        if rr is not None:
                            pi = rr
                    completion = _completion_from_parent(ps, pi)
                    seg, _fully = _agenda_from_parent_index(
                        ps, pi, generated, state.vocab, bound=_bound_of(state),
                        trim_prefix=_qa_trim_prefix(config),
                    )
        if plan_locked:
            pass  # giữ multi-segment từ multizone
        elif completion:
            verbatim = tuple(completion)
            if not seg:
                if pi is not None:
                    seg, _fully = _agenda_from_parent_index(
                        ps, pi, generated, state.vocab, bound=_bound_of(state),
                        trim_prefix=_qa_trim_prefix(config),
                    )
                else:
                    seg = _surface_content_ids(list(completion), state.vocab)
            q_content = _content_ids(query, state.vocab, skip=set())
            plan = _SegmentPlan(
                [seg] if seg else [], ps.entity_register_ids(tuple(q_content)), list(seg)
            )
        else:
            # best_match trượt → thử truy hồi theo THỰC THỂ (facts_about) trước khi buông về
            # W_seq thuần: đồ thị có thể có statement về thực thể mà khung câu hỏi không khớp cha.
            plan = _build_segment_plan(state, query, config, max_facts=max_facts)
            # R1 (TK_017): fact fallback phải PHỦ focus phân biệt nhất — không thì coi như
            # KHÔNG truy hồi được (plan rỗng → W_seq + bracket-gate dẫn, không bịa lạc đề).
            if config.ar_fallback_focus_gate:
                plan = _filter_plan_on_focus(state, query, plan)

    if not plan.flat_agenda:
        q_content = _content_ids(generated, state.vocab, skip=set())
        return _finalize(_finish_or_pattern(
            state, generated, eos_id, max_len, config, rng,
            min_suffix_tokens=max(2, len(q_content)),
            prompt=prompt, seed=seed, trace_out=trace_out,
        ))

    out = generate_autoregressive(
        state, prompt, max_len=max_len, config=config, seed=seed,
        plan=plan, trace_out=trace_out,
    )

    # Lưới an toàn (crisp): A3 phủ agenda quá thấp → dội verbatim đáp truy hồi (giữ đúng fact).
    if verbatim and config.ar_verbatim_fallback:
        gen = out[len(generated):]
        if eos_id in gen:
            gen = gen[: gen.index(eos_id)]
        gset = set(gen)
        cov = sum(1 for t in plan.flat_agenda if t in gset) / len(plan.flat_agenda)
        if cov < config.essay_ar_min_cov:
            return _finalize(generated + list(verbatim))
        # TK_019 SỔ NỢ: bỏ kênh entity/discourse làm QA gen-hy 100→86 (1 ca "mạnh hơn"→
        # "mạnh mẽ"). Đã đo 2 lưới cứu: cov=0.99 và key-token-gidf — CẢ HAI đẩy vb 14→100%
        # (máy tra cứu, phản Trụ 4) → KHÔNG lưới thêm; giữ sinh thật, giá −1 ca ghi sổ.
        if cov < config.essay_ar_min_cov:
            pass  # (đã return ở trên)
    # TK_022b — bồi thêm fact rời về thực thể (đáp nhiều câu, không đại từ). Fire cho câu
    # hỏi-thực-thể (có cụm WH đa-filler "là gì"/"là ai"…), KHÔNG chỉ "là gì" (tác giả: "Lý
    # Thái Tổ là ai" cũng thiếu fact). Continuation (q_ids==generated) không bồi.
    if config.knowledge_extra_facts > 0 and q_ids != generated:
        q_content = tuple(_content_ids(q_ids, state.vocab, skip=set()))
        if _knowledge_extra_facts_eligible(ps, q_content, config):
            out = _knowledge_extra_facts(
                state, q_ids, out, len(generated), config, plan_out=fact_plan_out,
            )
    return _finalize(out)


#
# NOTE: QA/crisp mode has been removed to focus optimization on a single generation path.
#
