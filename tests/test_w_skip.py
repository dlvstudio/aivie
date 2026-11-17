"""W_skip — xây CSR theo d + giao đề xuất (ý tác giả)."""

from __future__ import annotations

import numpy as np
import pytest

from nn_aivie.core.parent_store import ParentRecord, ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.w_skip import build_w_skip, constrain_next


def test_w_skip_intersect_toy():
    """Việt@d=2 ∩ Nam@d=1 ⊇ {vô, mạnh} theo ý tác giả."""
    words = (
        "Việt", "Nam", "vô", "địch", "mạnh", "mẽ", "Đại", "hồng", "phúc",
        "cộng", "đánh", "Mỹ", ".",
    )
    v = Vocabulary()
    for w in words:
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in words}

    def sent(*ws: str) -> tuple[int, ...]:
        return tuple(id_[w] for w in ws) + (id_["."],)

    ps = ParentStore()
    for emit in (
        sent("Việt", "Nam", "vô", "địch"),
        sent("Việt", "Nam", "mạnh", "mẽ"),
        sent("Đại", "Nam", "hồng", "phúc"),
        sent("Việt", "cộng", "đánh", "Mỹ"),
    ):
        ps.parents.append(
            ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
        )

    st = GraphState(vocab=v, parents=ps)
    wskip = build_w_skip(st, d_max=4, min_count=1, content_only=True)

    top2 = {j for j, _ in wskip.row_topk(id_["Việt"], 2, topk=20)}
    assert id_["vô"] in top2 and id_["mạnh"] in top2 and id_["đánh"] in top2

    top1 = {j for j, _ in wskip.row_topk(id_["Nam"], 1, topk=20)}
    assert id_["vô"] in top1 and id_["mạnh"] in top1 and id_["hồng"] in top1

    res = constrain_next(
        wskip, [id_["Việt"], id_["Nam"]], v, topk=20, ctx_max=4, content_only=True
    )
    hard = res["hard"]
    assert id_["vô"] in hard and id_["mạnh"] in hard
    assert id_["đánh"] not in hard
    assert id_["hồng"] not in hard

    from nn_aivie.hca.w_skip import soft_constraint_logits

    vec_gate = soft_constraint_logits(
        st, [id_["Việt"], id_["Nam"]], d_max=4, min_count=1, topk=20,
        ctx_max=4, hard_gate=True, gidf_min=0.0,
    )
    assert vec_gate[id_["vô"]] > 0 and vec_gate[id_["mạnh"]] > 0
    assert vec_gate[id_["đánh"]] == 0.0 and vec_gate[id_["hồng"]] == 0.0


def test_vote_logits_dem_phieu_theo_dac_ta():
    """§12.105/§12.106 — đúng ví dụ tác giả: `đẹp`/`vô` 2 phiếu · `xa`/`đánh` 1 phiếu.

    Khác `constrain_next` (giao + cổng cứng): cộng MỀM nên token 1 phiếu KHÔNG bị loại, chỉ
    thấp điểm hơn. Đó là chỗ §12.85 hỏng — cổng cứng trên tập giao bẩn đẩy sinh vào hub.
    """
    import numpy as np

    from nn_aivie.hca.w_skip import clear_w_skip_cache, vote_logits

    words = ("Việt", "Nam", "đẹp", "tuyệt", "vời", "Phương", "xa", "xôi",
             "vô", "địch", "cộng", "đánh", "thắng", "Mỹ", ".")
    v = Vocabulary()
    for w in words:
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in words}

    def sent(*ws: str) -> tuple[int, ...]:
        return tuple(id_[w] for w in ws) + (id_["."],)

    ps = ParentStore()
    for emit in (
        sent("Việt", "Nam", "đẹp", "tuyệt", "vời"),
        sent("Phương", "Nam", "xa", "xôi"),
        sent("Việt", "Nam", "vô", "địch"),
        sent("Việt", "cộng", "đánh", "thắng", "Mỹ"),
    ):
        ps.parents.append(
            ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
        )

    st = GraphState(vocab=v, parents=ps)
    clear_w_skip_cache()
    # λ=1.0 để phiếu là số đếm thuần, đúng ví dụ trong đặc tả
    vec = vote_logits(
        st, [id_["Việt"], id_["Nam"]], n_vocab=len(v),
        d_max=4, lam=1.0, topk=20, min_count=1,
        oja_row=False, oja_degree_beta=0.0, oja_compete_beta=0.0,
        oja_indeg_beta=0.0,
    )
    # `Việt` ở d=2 đề cử {đẹp, vô, đánh}; `Nam` ở d=1 đề cử {đẹp, vô, xa}
    assert vec[id_["đẹp"]] == 2.0
    assert vec[id_["vô"]] == 2.0
    assert vec[id_["xa"]] == 1.0
    assert vec[id_["đánh"]] == 1.0
    # cộng mềm: 1 phiếu vẫn > 0 (không bị cổng cứng loại như `constrain_next`)
    assert vec[id_["tuyệt"]] == 0.0          # không ai đề cử ở đúng khoảng cách
    assert float(np.max(vec)) == 2.0
    clear_w_skip_cache()


def test_vote_logits_prod_hop_de_xuat():
    """§12.139 — hợp đề xuất (không giao): đơn phiếu vẫn vào list."""
    from nn_aivie.hca.w_skip import clear_w_skip_cache, vote_logits

    v = Vocabulary()
    for w in ("Việt", "Nam", "đẹp", "vô", "xa", "đánh", "khác", "."):
        v.id_of(w)
    id_ = {
        w: v.id_of(w)
        for w in ("Việt", "Nam", "đẹp", "vô", "xa", "đánh", "khác", ".")
    }
    ps = ParentStore()
    for emit in (
        (id_["Việt"], id_["Nam"], id_["đẹp"], id_["."]),
        (id_["Việt"], id_["Nam"], id_["vô"], id_["."]),
        # Việt→đánh ở d=2 (qua `khác`) — Nam không đề cử đánh
        (id_["Việt"], id_["khác"], id_["đánh"], id_["."]),
        # Nam→xa ở d=1 — Việt@d=2 không đề cử xa
        (id_["Nam"], id_["xa"], id_["."]),
    ):
        ps.parents.append(
            ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
        )
    st = GraphState(vocab=v, parents=ps)
    clear_w_skip_cache()
    # Nam@d1 ∪ Việt@d2 = {đẹp,vô,xa,đánh} — hợp, không giao
    vec = vote_logits(
        st, [id_["Việt"], id_["Nam"]], n_vocab=len(v),
        d_max=4, lam=1.0, topk=20, min_count=1, agg="prod",
    )
    assert vec[id_["đẹp"]] > 0.0
    assert vec[id_["vô"]] > 0.0
    assert vec[id_["xa"]] > 0.0, "đơn phiếu Nam vẫn vào pool"
    assert vec[id_["đánh"]] > 0.0, "đơn phiếu Việt vẫn vào pool"
    assert vec[id_["đẹp"]] >= vec[id_["xa"]]
    assert vec[id_["vô"]] >= vec[id_["đánh"]]
    clear_w_skip_cache()


def test_w_skip_xuyen_cau_khong_cat_span():
    """§12.143 — `đẹp. Việt` / `tươi→Việt` xuyên chấm trong cùng emit."""
    from nn_aivie.hca.w_skip import clear_w_skip_cache

    words = ("Việt", "Nam", "tươi", "đẹp", "hùng", "cường", ".")
    v = Vocabulary()
    for w in words:
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in words}
    # Việt Nam tươi đẹp. Việt Nam hùng cường.
    emit = (
        id_["Việt"], id_["Nam"], id_["tươi"], id_["đẹp"], id_["."],
        id_["Việt"], id_["Nam"], id_["hùng"], id_["cường"], id_["."],
    )
    ps = ParentStore()
    ps.parents.append(
        ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
    )
    st = GraphState(vocab=v, parents=ps)
    clear_w_skip_cache()
    # content_only: bỏ `.` → [Việt,Nam,tươi,đẹp,Việt,Nam,hùng,cường]
    ws = build_w_skip(st, d_max=4, min_count=1, content_only=True)
    top1 = {j for j, _ in ws.row_topk(id_["đẹp"], 1, topk=20)}
    top2 = {j for j, _ in ws.row_topk(id_["tươi"], 2, topk=20)}
    top3 = {j for j, _ in ws.row_topk(id_["Nam"], 3, topk=20)}
    assert id_["Việt"] in top1, "đẹp→Việt d=1"
    assert id_["Việt"] in top2, "tươi→Việt d=2"
    assert id_["Việt"] in top3, "Nam(câu1)→Việt(câu2) d=3"
    # full chain: `.` giữ → đẹp→Việt d=2
    ws2 = build_w_skip(st, d_max=4, min_count=1, content_only=False)
    top_d2 = {j for j, _ in ws2.row_topk(id_["đẹp"], 2, topk=20)}
    assert id_["Việt"] in top_d2, "đẹp→.→Việt d=2 khi giữ punct"
    clear_w_skip_cache()


def test_w_skip_consume_canh_va_dich():
    """§12.144 — cạnh (src,d)→dst bị chặn; đích = token đã có trong hist."""
    from nn_aivie.hca.w_skip import (
        blocked_dst_from_hist,
        clear_w_skip_cache,
        mark_w_skip_consume,
        vote_logits,
    )

    words = ("A", "B", "C", "D", ".")
    v = Vocabulary()
    for w in words:
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in words}
    emit = (
        id_["A"], id_["B"], id_["C"], id_["."],
        id_["A"], id_["B"], id_["D"], id_["."],
    )
    ps = ParentStore()
    ps.parents.append(
        ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
    )
    st = GraphState(vocab=v, parents=ps)
    clear_w_skip_cache()
    hist = [id_["A"], id_["B"]]
    blocked_e: set[tuple[int, int, int]] = set()
    mark_w_skip_consume(
        st, hist, id_["C"], d_max=4, blocked_edges=blocked_e, consume_edges=True,
    )
    assert (id_["B"], 1, id_["C"]) in blocked_e
    vec = vote_logits(
        st, hist, n_vocab=len(v), d_max=4, lam=1.0, topk=0, min_count=1, agg="sum",
        oja_row=False, oja_degree_beta=0.0, oja_compete_beta=0.0, oja_indeg_beta=0.0,
        blocked_edges=blocked_e, blocked_dst=None,
    )
    assert float(vec[id_["C"]]) == 0.0
    # hist [A,B,C]: A,B,C đều trong blocked_dst (không cần biên câu)
    hist2 = [id_["A"], id_["B"], id_["C"]]
    dst = blocked_dst_from_hist(st, hist2, prompt_len=0, gidf_min=0.0)
    assert id_["A"] in dst and id_["C"] in dst
    # sau khi đã có C trong hist, vote không đề cử lại C
    vec2 = vote_logits(
        st, hist2, n_vocab=len(v), d_max=4, lam=1.0, topk=0, min_count=1, agg="sum",
        oja_row=False, oja_degree_beta=0.0, oja_compete_beta=0.0, oja_indeg_beta=0.0,
        blocked_edges=None, blocked_dst=dst,
    )
    assert float(vec2[id_["C"]]) == 0.0
    clear_w_skip_cache()


def test_w_skip_consume_khong_chan_token_khac_canh():
    """§12.144 — consume dời→đô (mọi d từ hist) không chặn kinh→đô."""
    from nn_aivie.hca.w_skip import clear_w_skip_cache, mark_w_skip_consume, vote_logits

    v = Vocabulary()
    for w in ("kinh", "đô", "dời", "."):
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in ("kinh", "đô", "dời", ".")}
    emit = (
        id_["dời"], id_["đô"], id_["."],
        id_["kinh"], id_["đô"], id_["."],
    )
    ps = ParentStore()
    ps.parents.append(
        ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
    )
    st = GraphState(vocab=v, parents=ps)
    clear_w_skip_cache()
    blocked: set[tuple[int, int, int]] = set()
    # đã dùng dời→đô — mark mọi d từ hist [dời]
    mark_w_skip_consume(
        st, [id_["dời"]], id_["đô"], d_max=4, blocked_edges=blocked, consume_edges=True,
    )
    assert (id_["dời"], 1, id_["đô"]) in blocked
    # kinh chưa nằm trong hist lúc consume → cạnh kinh→đô còn
    vec = vote_logits(
        st, [id_["kinh"]], n_vocab=len(v), d_max=4, lam=1.0, topk=0, min_count=1,
        agg="sum",
        oja_row=False, oja_degree_beta=0.0, oja_compete_beta=0.0, oja_indeg_beta=0.0,
        blocked_edges=blocked, blocked_dst=None, blocked_edge_scale=0.0,
    )
    assert float(vec[id_["đô"]]) > 0.0, "kinh→đô chưa consume vẫn đề cử đô"
    clear_w_skip_cache()


def test_w_skip_consume_chan_lai_cung_src_khac_d():
    """Consume A→C lúc hist=[A,B] chặn cả (A,d,*)→C mọi d — không chỉ d đúng vị trí."""
    from nn_aivie.hca.w_skip import clear_w_skip_cache, mark_w_skip_consume, vote_logits

    v = Vocabulary()
    for w in ("A", "B", "C", "."):
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in ("A", "B", "C", ".")}
    # A B C .  và  A x C ở d=2 trong emit khác
    ps = ParentStore()
    for emit in (
        (id_["A"], id_["B"], id_["C"], id_["."]),
        (id_["A"], id_["C"], id_["."]),  # A→C d=1 cũng có
    ):
        ps.parents.append(
            ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
        )
    st = GraphState(vocab=v, parents=ps)
    clear_w_skip_cache()
    blocked: set[tuple[int, int, int]] = set()
    mark_w_skip_consume(
        st, [id_["A"], id_["B"]], id_["C"], d_max=4,
        blocked_edges=blocked, consume_edges=True,
    )
    assert (id_["A"], 2, id_["C"]) in blocked or (id_["A"], 1, id_["C"]) in blocked
    # A không còn đề cử C
    vec = vote_logits(
        st, [id_["A"], id_["B"]], n_vocab=len(v), d_max=4, lam=1.0, topk=0,
        min_count=1, agg="sum",
        oja_row=False, oja_degree_beta=0.0, oja_compete_beta=0.0, oja_indeg_beta=0.0,
        blocked_edges=blocked, blocked_edge_scale=0.0,
    )
    assert float(vec[id_["C"]]) == 0.0
    clear_w_skip_cache()


def test_w_skip_consume_reverse_cua_so_gan():
    """§12.145 — gần/xa: trọng→quan; điểm→quan yếu; kinh→đô mạnh được giữ."""
    from nn_aivie.hca.w_skip import clear_w_skip_cache, mark_w_skip_consume, vote_logits

    v = Vocabulary()
    words = ("quan", "trọng", "kinh", "đô", "điểm", "a", "b", "c", "d", "e", ".")
    for w in words:
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in words}
    ps = ParentStore()
    for emit in (
        (id_["quan"], id_["trọng"], id_["."]),
        (id_["trọng"], id_["quan"], id_["."]),
        (id_["điểm"], id_["quan"], id_["."]),
        (id_["điểm"], id_["a"], id_["."]),
        (id_["điểm"], id_["a"], id_["."]),
        (id_["điểm"], id_["a"], id_["."]),
        (id_["đô"], id_["a"], id_["b"], id_["c"], id_["d"], id_["e"],
         id_["kinh"], id_["đô"], id_["."]),
        (id_["kinh"], id_["đô"], id_["."]),
        (id_["kinh"], id_["đô"], id_["."]),
    ):
        ps.parents.append(
            ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
        )
    st = GraphState(vocab=v, parents=ps)
    clear_w_skip_cache()
    blocked: set[tuple[int, int, int]] = set()
    mark_w_skip_consume(
        st, [id_["quan"]], id_["trọng"], d_max=4,
        blocked_edges=blocked, consume_edges=True, reverse_window=4,
    )
    assert (id_["trọng"], 1, id_["quan"]) in blocked
    blocked_p: set[tuple[int, int, int]] = set()
    far_q = [id_["quan"], id_["a"], id_["b"], id_["c"], id_["d"], id_["e"]]
    mark_w_skip_consume(
        st, far_q, id_["điểm"], d_max=4,
        blocked_edges=blocked_p, consume_edges=True, reverse_window=4,
    )
    assert (id_["điểm"], 1, id_["quan"]) in blocked_p
    blocked2: set[tuple[int, int, int]] = set()
    far = [id_["đô"], id_["a"], id_["b"], id_["c"], id_["d"], id_["e"]]
    mark_w_skip_consume(
        st, far, id_["kinh"], d_max=4,
        blocked_edges=blocked2, consume_edges=True, reverse_window=4,
    )
    assert (id_["kinh"], 1, id_["đô"]) not in blocked2
    vec2 = vote_logits(
        st, [id_["kinh"]], n_vocab=len(v), d_max=4, lam=1.0, topk=0, min_count=1,
        agg="sum",
        oja_row=False, oja_degree_beta=0.0, oja_compete_beta=0.0, oja_indeg_beta=0.0,
        blocked_edges=blocked2, blocked_edge_scale=0.0,
    )
    assert float(vec2[id_["đô"]]) > 0.0
    clear_w_skip_cache()


def test_w_skip_consume_recover_sau_n_buoc():
    """§12.145 — cạnh đã consume hồi sau recover_steps; hold_dst giữ hub gần."""
    from nn_aivie.hca.w_skip import (
        clear_w_skip_cache,
        expire_w_skip_consume,
        mark_w_skip_consume,
        vote_logits,
    )

    v = Vocabulary()
    for w in ("A", "B", "."):
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in ("A", "B", ".")}
    emit = (id_["A"], id_["B"], id_["."])
    ps = ParentStore()
    ps.parents.append(
        ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
    )
    st = GraphState(vocab=v, parents=ps)
    clear_w_skip_cache()
    blocked: set[tuple[int, int, int]] = set()
    age: dict[tuple[int, int, int], int] = {}
    mark_w_skip_consume(
        st, [id_["A"]], id_["B"], d_max=4, blocked_edges=blocked,
        consume_edges=True, edge_age=age, step=0,
    )
    assert (id_["A"], 1, id_["B"]) in blocked
    # còn trong cửa sổ gần → không hồi
    expire_w_skip_consume(
        blocked, age, step=8, recover_steps=8, hold_dst={id_["B"]},
    )
    assert (id_["A"], 1, id_["B"]) in blocked
    # ra khỏi cửa sổ → hồi
    expire_w_skip_consume(blocked, age, step=8, recover_steps=8, hold_dst=set())
    assert (id_["A"], 1, id_["B"]) not in blocked
    vec2 = vote_logits(
        st, [id_["A"]], n_vocab=len(v), d_max=4, lam=1.0, topk=0, min_count=1,
        agg="sum",
        oja_row=False, oja_degree_beta=0.0, oja_compete_beta=0.0, oja_indeg_beta=0.0,
        blocked_edges=blocked, blocked_edge_scale=0.0,
    )
    assert float(vec2[id_["B"]]) > 0.0
    clear_w_skip_cache()


def test_w_skip_local_repeat_penalty_cua_so():
    """§12.145 — token trong cửa sổ gần bị scale; ngoài cửa sổ không phạt."""
    from nn_aivie.hca.w_skip import apply_local_repeat_penalty

    v = Vocabulary()
    for w in ("quan", "mới", "."):
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in ("quan", "mới", ".")}
    n = len(v)
    scores = np.ones(n, dtype=np.float64)
    # cửa sổ 3: quan×2 + mới; tắt near để đo scale^count thuần
    out = [id_["."], id_["quan"], id_["quan"], id_["mới"]]
    got = apply_local_repeat_penalty(
        scores, out, prompt_len=1, n_vocab=n, vocab=v,
        window=3, scale=0.25, min_count=1, near_k=0, near_scale=1.0,
        outsider_prefer=1.0,
    )
    assert got[id_["quan"]] == pytest.approx(0.25 ** 2)
    assert got[id_["mới"]] == pytest.approx(0.25)
    got2 = apply_local_repeat_penalty(
        scores, out, prompt_len=1, n_vocab=n, vocab=v,
        window=1, scale=0.25, min_count=1, near_k=0, near_scale=1.0,
        outsider_prefer=1.0,
    )
    assert got2[id_["quan"]] == pytest.approx(1.0)
    assert got2[id_["mới"]] == pytest.approx(0.25)
    # near_k phạt thêm token vừa ra
    got3 = apply_local_repeat_penalty(
        scores, out, prompt_len=1, n_vocab=n, vocab=v,
        window=3, scale=1.0, min_count=1, near_k=1, near_scale=0.1,
        outsider_prefer=1.0,
    )
    assert got3[id_["mới"]] == pytest.approx(0.1)
    assert got3[id_["quan"]] == pytest.approx(1.0)
    # còn outsider → túi gần bị hạ thêm
    got4 = apply_local_repeat_penalty(
        scores, out, prompt_len=1, n_vocab=n, vocab=v,
        window=3, scale=1.0, min_count=1, near_k=0, near_scale=1.0,
        outsider_prefer=0.05,
    )
    assert got4[id_["quan"]] == pytest.approx(0.05)
    assert got4[id_["mới"]] == pytest.approx(0.05)


def test_oja_scale_row_bao_hoa_bac_cao():
    """§12.140 — hàng nnz lớn bị α(k) hạ mạnh hơn hàng hẹp (không if dấu câu)."""
    from nn_aivie.hca.w_skip import _oja_scale_row

    hub = {i: 10.0 for i in range(100)}
    narrow = {0: 10.0, 1: 10.0}
    hub_s = _oja_scale_row(hub, oja_row=True, degree_beta=1.0)
    nar_s = _oja_scale_row(narrow, oja_row=True, degree_beta=1.0)
    # Sau L2, mỗi cạnh hub nhỏ hơn; ×α(100) << ×α(2) → max cạnh hub < max cạnh hẹp
    assert max(hub_s.values()) < max(nar_s.values())


def test_vote_logits_giam_theo_khoang_cach():
    """λ<1 ⇒ đề cử từ xa nhẹ hơn đề cử gần — "khoảng cách xa thì giảm bớt" (đặc tả tác giả)."""
    from nn_aivie.hca.w_skip import clear_w_skip_cache, vote_logits

    v = Vocabulary()
    for w in ("a", "b", "c", "."):
        v.id_of(w)
    id_ = {w: v.id_of(w) for w in ("a", "b", "c", ".")}
    ps = ParentStore()
    emit = (id_["a"], id_["b"], id_["c"], id_["."])
    ps.parents.append(
        ParentRecord(emit_ids=emit, match_ids=emit[:-1], answer_ids=(), power=1)
    )
    st = GraphState(vocab=v, parents=ps)
    clear_w_skip_cache()
    # lịch sử [a, b]: `c` được `b` đề cử ở d=1 (trọng 1.0) và `a` ở d=2 (trọng λ)
    for lam in (1.0, 0.5, 0.25):
        vec = vote_logits(st, [id_["a"], id_["b"]], n_vocab=len(v),
                          d_max=4, lam=lam, topk=20, min_count=1,
                          oja_row=False, oja_degree_beta=0.0, oja_compete_beta=0.0,
                          oja_indeg_beta=0.0)
        assert abs(vec[id_["c"]] - (1.0 + lam)) < 1e-9, lam
    clear_w_skip_cache()
