"""TK_045 §13 Trụ A — tầng nơ-ron MỆNH ĐỀ.

Bất biến khoá bằng test (mỗi cái tương ứng một phát hiện của P47 §13.6):

1. **Biên đơn vị đến từ EOS-lift**, không từ danh sách dấu câu.
2. **EOS-lift MỘT MÌNH là không đủ** — nó mù với biên dưới-câu (`,`). Đây là cái vấp
   thật của P47; khoá lại để không ai "tối giản" PMI đi rồi tưởng vẫn chạy.
3. **Thung lũng PMI cắt được dưới câu** ⇒ mệnh đề ngắn hơn hẳn.
4. **Mối nối phải ĐƯỢC CHỨNG THỰC** trên W_skip — đây là bảo chứng chống salad.
5. **Tái tổ hợp KHÁC-CHA tồn tại** (kill switch M2 của P47).
"""

from __future__ import annotations

from nn_aivie.core.parent_store import ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.hca.clause_layer import (
    build_clause_layer,
    clear_clause_layer_cache,
    cut_clauses,
    ensure_boundary_signal,
    ensure_clause_layer,
    join_candidates,
)
from nn_aivie.hca.w_skip import build_w_skip
from nn_aivie.training.ingest import ingest_statements

# Mỗi cha = một ĐOẠN hai câu (không phải một câu) — bắt buộc, vì:
#  · biên dưới-câu (`,`) là thứ EOS-lift không thấy (bất biến 2);
#  · tái tổ hợp KHÁC-CHA chỉ có khi W_skip bắc qua biên câu (§12.143) và các cha
#    dùng chung token MỞ mệnh đề (`Việt` ở đây) — cha một-câu KHÔNG dựng nổi ca này.
_SENTS = [
    "Phở là món ăn nổi tiếng , nó được bán khắp nơi . Việt Nam có nhiều món ngon .",
    "Hà Nội là thủ đô , nơi đây có lịch sử lâu đời . Việt Nam tự hào về Hà Nội .",
    "Huế là cố đô , thành phố này nằm ở miền Trung . Việt Nam có nhiều cố đô .",
    "Sài Gòn là đô thị lớn , kinh tế phát triển mạnh . Việt Nam đang đổi mới .",
]


def _trained() -> GraphState:
    clear_clause_layer_cache()
    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, _SENTS, split_punct=True)
    return st


def test_eos_lift_marks_unit_end_not_content():
    """Bất biến 1 — `.` là mốc kết, từ nội dung thì không (§0.4.2)."""
    st = _trained()
    b = ensure_boundary_signal(st)
    assert b.eos_base > 0
    assert b.ends_unit(st.vocab.id_of("."))
    for t in ("là", "Việt", "Nam", "thủ"):
        assert not b.ends_unit(st.vocab.id_of(t)), f"{t!r} không phải mốc kết"


def test_eos_lift_is_BLIND_to_sub_sentence_boundary():
    """Bất biến 2 — cái VẤP của P47: `,` KHÔNG có tín hiệu EOS.

    Vì `,` không bao giờ đứng trước EOS ⇒ lift = 0 ⇒ cắt-chỉ-bằng-EOS cho ra CÂU,
    không cho ra MỆNH ĐỀ. Đây là lý do PMI phải tồn tại.
    """
    st = _trained()
    b = ensure_boundary_signal(st)
    assert not b.ends_unit(st.vocab.id_of(",")), (
        "nếu test này đỏ, EOS-lift đã đủ và PMI thành thừa — đọc lại §13.6 trước khi sửa"
    )


def test_pmi_valley_cuts_below_sentence():
    """Bất biến 3 — θ_pmi càng cao ⇒ càng nhiều mệnh đề, càng ngắn (đơn điệu).

    KHÔNG khoá một θ cụ thể: trên corpus đồ chơi mọi bigram chỉ thấy 1 lần nên PMI
    dương khắp nơi (θ=0 không cắt gì) — đó là tính chất của corpus, không phải bug.
    Vùng vận hành thật đo trên v12/mini ở §13.6.
    """
    st = _trained()
    only_sent = build_clause_layer(st, pmi_theta=None)
    layers = [build_clause_layer(st, pmi_theta=th) for th in (2.0, 4.0, 6.0)]
    counts = [len(only_sent)] + [len(x) for x in layers]
    assert counts == sorted(counts), f"không đơn điệu theo θ: {counts}"
    assert counts[-1] > counts[0], "θ cao nhất vẫn không cắt thêm mệnh đề nào"
    assert layers[-1].mean_len < only_sent.mean_len


def test_clause_fields_and_indices_consistent():
    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    assert len(layer) > 0
    for ci, cl in enumerate(layer.clauses):
        assert cl.tokens, "mệnh đề rỗng"
        assert cl.first_content in cl.tokens and cl.last_content in cl.tokens
        assert ci in layer.by_opener[cl.first_content]
        assert ci in layer.by_parent[cl.parent_idx]
        assert 0 <= cl.pos_in_parent < cl.n_in_parent
    # mỗi cha phải có ≥1 mệnh đề MỞ
    for _pi, cids in layer.by_parent.items():
        assert any(layer.clauses[c].opens_parent for c in cids)


def test_join_candidates_are_attested_only():
    """Bất biến 4 — chỉ mối nối CÓ TRÊN W_skip mới vào; đây là bảo chứng chống salad."""
    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    ws = build_w_skip(st, d_max=4, min_count=1, content_only=False)
    cl = layer.clauses[0]
    cands = join_candidates(layer, ws, cl.last_content, d_join=4)
    for opener in cands:
        assert opener in layer.by_opener
        assert any(
            ws.mats[d].getrow(cl.last_content)[0, opener] > 0
            for d in range(1, 5)
        ), "ứng viên KHÔNG có cạnh W_skip — mối nối chưa được chứng thực"


def test_cross_parent_recombination_exists():
    """Bất biến 5 — kill switch M2: phải nối được sang mệnh đề của CHA KHÁC."""
    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    ws = build_w_skip(st, d_max=4, min_count=1, content_only=False)
    cross = 0
    for cl in layer.clauses:
        for opener in join_candidates(layer, ws, cl.last_content, d_join=4):
            cross += sum(
                1 for m in layer.by_opener[opener]
                if layer.clauses[m].parent_idx != cl.parent_idx
            )
    assert cross > 0, "0 tái tổ hợp khác-cha ⇒ tầng mệnh đề chỉ phát lại cha (§13.6 M2)"


def test_cut_clauses_min_len_and_lossless():
    """`min_len` chỉ ràng buộc cắt-PMI; biên ĐƠN VỊ luôn cắt (câu ngắn vẫn phải đóng)."""
    st = _trained()
    b = ensure_boundary_signal(st)
    emit = st.parents.parents[0].emit_ids
    spans = cut_clauses(emit, b, pmi_theta=6.0, min_len=3)  # θ cao = cắt rất hăng
    for s in spans[:-1]:
        assert len(s) >= 3 or b.ends_unit(s[-1]), (
            "span ngắn hơn min_len mà KHÔNG kết ở mốc đơn vị ⇒ min_len bị bỏ qua"
        )
    assert sum(len(s) for s in spans) == len(emit), "cắt làm MẤT token"
    assert [t for s in spans for t in s] == list(emit), "cắt làm ĐỔI thứ tự token"


def test_ensure_clause_layer_caches_per_params():
    st = _trained()
    a1 = ensure_clause_layer(st, pmi_theta=0.0)
    a2 = ensure_clause_layer(st, pmi_theta=0.0)
    b1 = ensure_clause_layer(st, pmi_theta=1.0)
    assert a1 is a2, "cache không ăn"
    assert b1 is not a1, "tham số khác phải cho layer khác"


# ------------------------------------------------- Trụ C — cổng phủ (§13.10)
def test_content_folds_uses_gidf_gate():
    """Fold nội dung phải LỌC theo gidf — nếu không, hư từ lọt vào và cổng vô nghĩa."""
    from nn_aivie.hca.clause_layer import content_folds

    st = _trained()
    ids = list(st.parents.parents[0].emit_ids)
    # corpus đồ chơi có gidf THẤP hơn corpus thật ⇒ dùng ngưỡng thấp; chính chỗ này
    # chứng minh vì sao ngưỡng phải là THAM SỐ chứ không phải hằng 4.0 cứng.
    folds = content_folds(st, ids, gidf_min=1.5)
    assert folds, "không rút được fold nội dung nào"
    loose = content_folds(st, ids, gidf_min=0.0)
    assert folds <= loose and len(folds) < len(loose), "ngưỡng gidf không có tác dụng"
    assert content_folds(st, ids, gidf_min=99.0) == set(), "ngưỡng rất cao phải lọc SẠCH"


def test_gate_anchor_requires_shared_fold():
    from nn_aivie.hca.clause_layer import CoverageGate, content_folds

    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    c0 = layer.clauses[0]
    own = frozenset(content_folds(st, c0.tokens, gidf_min=1.5))
    assert own, "fixture không có fold nội dung — test vô nghĩa"
    assert CoverageGate(
        theme_folds=own, anchor_min_folds=1, gidf_min=1.5,
    ).accept(st, c0, set())
    assert not CoverageGate(
        theme_folds=frozenset({"##khong-ton-tai##"}), anchor_min_folds=1, gidf_min=1.5,
    ).accept(st, c0, set())


def test_gate_require_new_cover_is_not_vacuously_false():
    """REGRESSION §13.10 — bản đầu của tôi so với `agenda_folds` mà `covered` đã chứa sẵn
    ⇒ điều kiện LUÔN sai ⇒ loại sạch ứng viên, dead=100%. Ngữ nghĩa đúng: "mang fold MỚI
    so với những gì ĐÃ phủ"."""
    from nn_aivie.hca.clause_layer import CoverageGate, content_folds

    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    c0 = layer.clauses[0]
    cf = content_folds(st, c0.tokens, gidf_min=1.5)
    g = CoverageGate(
        theme_folds=frozenset(cf), anchor_min_folds=1, require_new_cover=True,
        gidf_min=1.5,
    )
    # chưa phủ gì → mệnh đề mang fold mới → PHẢI nhận (bản lỗi trả False ở đây)
    assert g.accept(st, c0, set())
    # đã phủ hết fold của nó → không mang gì mới → loại
    assert not g.accept(st, c0, set(cf))


def test_compose_chain_dead_when_gate_rejects_all():
    from nn_aivie.hca.clause_layer import CoverageGate, compose_chain
    from nn_aivie.hca.w_skip import build_w_skip

    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    ws = build_w_skip(st, d_max=4, min_count=1, content_only=False)
    prompt = layer.clauses[0].tokens
    # ⚠️ `gidf_min=0.0` là BẮT BUỘC ở corpus đồ chơi, không phải để cho dễ. Với mặc định
    # 4.0 thì **mọi** token của corpus 4 câu đều dưới ngưỡng ⇒ `content_folds` rỗng ⇒
    # `last=None` ⇒ chuỗi chết ở bước 0 và **cổng không bao giờ được hỏi**. Bản đầu của
    # test này chính là như vậy: nó mang tên "cổng chặn hết" mà **chưa từng chạm cổng**,
    # và vẫn xanh suốt — vì `chain2 == [] and dead` đúng vì một lý do KHÁC. Trường `stop`
    # thêm ở §13.36 phát hiện ra ngay khi vừa gắn vào. Giữ ghi chú này: một test xanh
    # không có nghĩa nó đã kiểm thứ ghi trong tên nó.
    chain, stats = compose_chain(st, layer, ws, prompt, max_clauses=3, gidf_min=0.0)
    assert stats["n_clauses"] == len(chain)
    assert stats["stop"] in {"du_max", "khong_neo", "khong_ung_vien", "loai_het"}
    blocked = CoverageGate(
        theme_folds=frozenset({"##khong-ton-tai##"}), anchor_min_folds=1, gidf_min=1.5,
    )
    chain2, stats2 = compose_chain(
        st, layer, ws, prompt, max_clauses=3, gate=blocked, gidf_min=0.0,
    )
    assert chain2 == [] and stats2["dead"], "cổng chặn hết mà chuỗi vẫn sinh"
    assert stats2["stop"] == "loai_het", f"chết KHÔNG do cổng ⇒ test vô nghĩa: {stats2}"
    assert stats2["n_gate"] > 0, "cổng loại sạch mà không đếm được lần loại nào"
    assert stats2["pool_at_stop"] > 0, "tắc vì bị LOẠI thì bể phải khác rỗng"


def test_h_clause_analytic_identity():
    """P47 §12: `H_clause = H_type + Σ_j q_j·log2(m_j)`.

    Probe dùng dạng giải tích thay vì materialize list dài BẰNG SỐ ỨNG VIÊN (v12 ~138k
    mỗi mẫu × 1500 mẫu). Test khoá đẳng thức để tối ưu không âm thầm đổi số.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(20):
        k = int(rng.integers(2, 8))
        w = rng.random(k) + 0.01
        m = rng.integers(1, 30, size=k).astype(float)
        q = w / w.sum()
        h_type = float(-(q * np.log2(q)).sum())
        analytic = h_type + float((q * np.log2(m)).sum())
        # dạng vét cạn: mỗi thành viên p = q_j/m_j
        pc = []
        for qj, mj in zip(q, m):
            pc.extend([qj / mj] * int(mj))
        pcv = np.array(pc)
        brute = float(-(pcv * np.log2(pcv)).sum())
        assert abs(analytic - brute) < 1e-9, f"{analytic} != {brute}"


def test_compose_does_not_prefer_longest_clause():
    """§13.14 REGRESSION — bộ soạn KHÔNG được ưu tiên mệnh đề dài nhất.

    P52 đo: thiên lệch `-len(tokens)` làm khối chép liên tục 22.7→51.7 token khi dữ
    liệu ×45, trong khi `novelty` (tỉ lệ) vẫn đẹp lên vì đầu ra dài ra theo. Tức bộ
    soạn tiêu vật liệu mới bằng cách CHÉP DÀI HƠN.

    Fixture phải có nhiều mệnh đề CÙNG token mở nhưng ĐỘ DÀI KHÁC HẲN — fixture chung
    của file này mọi câu dài bằng nhau nên KHÔNG bắt được thiên lệch.
    """
    import random as _rnd

    from nn_aivie.hca.clause_layer import compose_chain
    from nn_aivie.hca.w_skip import build_w_skip

    clear_clause_layer_cache()
    st = GraphState()
    st.parents = ParentStore()
    # cha ĐA CÂU: mệnh đề 2 cùng token mở `Dân` nhưng độ dài rất khác nhau
    ingest_statements(st, [
        "Vua ban chiếu . Dân nghe theo .",
        "Vua ban chiếu . Dân nghe theo lời của triều đình trong cả nước một cách "
        "nghiêm túc và đầy đủ .",
        "Vua ban chiếu . Dân nghe .",
        "Vua ban chiếu . Dân nghe theo lời quan .",
    ], split_punct=True)
    layer = build_clause_layer(st, pmi_theta=0.0)
    ws = build_w_skip(st, d_max=4, min_count=1, content_only=False)

    # nhóm token mở phải có >1 thành viên với độ dài khác nhau
    groups = {
        op: {len(layer.clauses[c].tokens) for c in cs}
        for op, cs in layer.by_opener.items()
    }
    assert any(len(v) > 1 for v in groups.values()), (
        "fixture không có nhóm cùng token mở mà khác độ dài — test vô nghĩa"
    )

    prompt = list(layer.clauses[0].tokens)  # "Vua ban chiếu ."
    picked_lens = set()
    for seed in range(30):
        chain, _ = compose_chain(
            st, layer, ws, prompt, max_clauses=3, rng=_rnd.Random(seed),
            gidf_min=0.5,
        )
        if chain:
            picked_lens.add(tuple(len(c.tokens) for c in chain))
    assert len(picked_lens) > 1, (
        "mọi seed cho cùng dãy độ dài ⇒ chọn tất định theo độ dài, thiên lệch còn đó"
    )


# --------------------------------------------------- §13.18: chuỗi hút + dính số
class _FakeBpeVocab:
    """Vocab BPE tối giản — chỉ cần `is_bpe` + `token_of` cho luật biên bề mặt."""

    is_bpe = True

    def __init__(self, toks):
        self._t = list(toks)

    def token_of(self, i):
        return self._t[int(i)]


def test_word_start_rule_is_noop_on_non_bpe_vocab():
    """REGRESSION — vocab KHÔNG phải BPE không có marker `Ġ` ⇒ luật phải NO-OP.

    Bản đầu trả False cho mọi token ⇒ lọc sạch ứng viên ⇒ chuỗi chết. Test đồ chơi đỏ
    ngay khi wire, nên bắt được; nếu chỉ chạy trên checkpoint BPE thì đã lọt.
    """
    from nn_aivie.hca.clause_layer import opener_is_word_start

    st = _trained()
    assert not getattr(st.vocab, "is_bpe", False), "fixture phải là vocab thường"
    for t in list(st.parents.parents[0].emit_ids)[:6]:
        assert opener_is_word_start(st.vocab, t), "luật phải no-op trên vocab thường"


def test_word_start_rule_blocks_glue_on_bpe():
    """`"năm 1989"` + `"000 hố va chạm"` → `"1989000"` — chấm mù bắt (§13.17 lỗi 2)."""
    from nn_aivie.hca.clause_layer import opener_is_word_start

    v = _FakeBpeVocab(["Ġnăm", "Ġ1989", "000", "Ġhố", "Ċ"])
    assert opener_is_word_start(v, 0)      # Ġnăm — mở từ
    assert opener_is_word_start(v, 1)      # Ġ1989 — mở từ (số MỚI, hợp lệ)
    assert not opener_is_word_start(v, 2)  # "000" — DÍNH vào token trước
    assert opener_is_word_start(v, 4)      # Ċ — xuống dòng cũng là mở


def test_hub_saturation_lowers_concentration():
    """§13.18 — bão hoà hub phải HẠ độc quyền của token-mở đông thành viên.

    Chẩn mini: `Ġlà` chiếm 16.8% mối nối (200 mệnh đề cùng mở), top-10 chiếm 50%.
    """
    from nn_aivie.hca.clause_layer import compose_chain, join_candidates
    from nn_aivie.hca.w_skip import build_w_skip

    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    ws = build_w_skip(st, d_max=4, min_count=1, content_only=False)
    hub = max(layer.by_opener, key=lambda j: len(layer.by_opener[j]))
    src = layer.clauses[0].last_content

    plain = join_candidates(layer, ws, src, d_join=4)
    sat = join_candidates(layer, ws, src, d_join=4, hub_beta=2.0, vocab=st.vocab)
    if hub in plain and len(plain) > 1:
        share_p = plain[hub] / sum(plain.values())
        share_s = sat[hub] / sum(sat.values())
        assert share_s < share_p, "bão hoà hub không hạ được tỉ trọng hub"
    # và nó không được làm MẤT ứng viên nào
    assert set(sat) == set(plain)
    # compose_chain vẫn chạy với mặc định mới
    chain, _ = compose_chain(st, layer, ws, layer.clauses[0].tokens,
                             max_clauses=3, gidf_min=0.5)
    assert isinstance(chain, list)


def test_pair_folds_separate_homophone_syllables():
    """§13.20 — tiếng Việt đơn âm tiết ⇒ âm tiết LẺ là đồng âm hàng loạt.

    `lục lạp` (bào quan) và `lục bát` (thể thơ) chia âm tiết `lục` ⇒ khớp fold-lẻ nói
    chúng "cùng đề". Đơn vị CẶP phải tách được hai cái đó. Đây đúng là lỗi chấm mù bắt
    (§13.19) và là thứ làm `lạc = 0.000` trở nên RỖNG.
    """
    from nn_aivie.hca.clause_layer import content_folds

    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, [
        "Quang hợp diễn ra trong lục lạp của tế bào .",
        "Truyện Kiều viết bằng thể lục bát rất mượt .",
    ], split_punct=True)
    a = list(st.parents.parents[0].emit_ids)
    b = list(st.parents.parents[1].emit_ids)

    single_a = content_folds(st, a, gidf_min=0.0)
    single_b = content_folds(st, b, gidf_min=0.0)
    assert single_a & single_b, "fixture phải có âm tiết chung (`lục`) — nếu không test vô nghĩa"

    pair_a = content_folds(st, a, gidf_min=0.0, pair=True)
    pair_b = content_folds(st, b, gidf_min=0.0, pair=True)
    assert not (pair_a & pair_b), (
        "đơn vị CẶP vẫn coi `lục lạp` ≡ `lục bát` — van chính của §13.20 không hoạt động"
    )


def test_drop_digits_removes_digit_folds():
    """Chữ số đứng đầu bảng bằng chứng 'đúng đề' (`'4'` 10 lượt) — phải bỏ được."""
    from nn_aivie.hca.clause_layer import content_folds

    st = GraphState()
    st.parents = ParentStore()
    ingest_statements(st, ["Vua lên ngôi năm 1010 tại kinh đô ."], split_punct=True)
    ids = list(st.parents.parents[0].emit_ids)
    with_d = content_folds(st, ids, gidf_min=0.0)
    no_d = content_folds(st, ids, gidf_min=0.0, drop_digits=True)
    assert no_d < with_d, "drop_digits không bỏ được fold nào"
    assert not any(any(ch.isdigit() for ch in f) for f in no_d)


def test_order_join_candidates_rescore_survives_shuffle():
    """REGRESSION §13.28 — `join_rescore` phải xếp ở tầng TOKEN-MỞ và KHÔNG bị shuffle xoá.

    Bản §13.26: `sort(rescore)` rồi ngay sau `rng.shuffle()` ⇒ NO-OP. Đầu ra vẫn "đổi" nên
    phép đo end-to-end mù với lỗi này — chấm mù vòng năm vì thế so nhiễu với nhiễu.
    Test ở tầng HÀM THUẦN, không qua `compose_chain` (fixture đồ chơi không kéo nổi chuỗi ≥2).
    """
    import random as _rnd

    from nn_aivie.hca.clause_layer import order_join_candidates

    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    openers = list(layer.by_opener)[:4]
    assert len(openers) >= 3, "fixture cần ≥3 token-mở"
    order = [(op, 1.0) for op in openers]          # điểm W_skip BẰNG NHAU ⇒ chỉ rescore quyết
    prev = layer.clauses[0]

    # rescore TẤT ĐỊNH: ưu tiên token-mở có id LỚN nhất
    def rs(_p, cand):
        return float(cand.first_content)

    got = [
        [op for op, _ in order_join_candidates(
            layer, order, prev, join_rescore=rs, rng=_rnd.Random(seed))]
        for seed in range(8)
    ]
    assert all(g == got[0] for g in got), (
        f"rescore TẤT ĐỊNH mà thứ tự token-mở vẫn đổi theo seed ⇒ bị shuffle xoá: {got[:3]}"
    )
    assert got[0] == sorted(openers, reverse=True), (
        f"không xếp theo điểm rescore ở tầng TOKEN-MỞ: {got[0]} vs {sorted(openers, reverse=True)}"
    )

    # không rescore ⇒ giữ thứ tự `order` đầu vào (không tự ý xáo tầng token-mở)
    plain = [op for op, _ in order_join_candidates(layer, order, prev, rng=_rnd.Random(0))]
    assert plain == openers




# --------------------------------------------- §13.31 — neo ĐỀ có trọng số ĐỘ HIẾM
def test_fold_gidf_map_weights_by_rarity_not_count():
    """`gidf` phải PHÂN BIỆT fold hiếm với fold chung — nếu không thì nó chỉ là `len()`.

    Đây là toàn bộ lý do §13.31 tồn tại: đếm fold thì chữ số và âm tiết chung (`ông`,
    `từ`, `dân` — §13.20) lấn át, còn nhân `gidf` thì tên riêng nặng hơn.
    """
    from nn_aivie.hca.clause_layer import fold_gidf_map

    st = _trained()
    # `Việt` có ở CẢ BỐN cha (chung); `Phở` chỉ ở một (hiếm).
    ids = [st.vocab.id_of("Việt"), st.vocab.id_of("Phở")]
    m = fold_gidf_map(st, ids, gidf_min=0.0)
    assert set(m) == {"việt", "phở"} or len(m) == 2, m
    common = m[[k for k in m if k.startswith("vi")][0]]
    rare = m[[k for k in m if k.startswith("ph")][0]]
    assert rare > common, f"fold hiếm phải nặng hơn fold chung: {rare} vs {common}"


def test_prompt_anchor_rescorer_prefers_RARE_overlap_at_EQUAL_count():
    """Bất biến §13.31 — trùng **cùng số lượng** fold thì bên trùng fold HIẾM phải thắng.
    Bản `len(cf & pf)` (đã thử) cho HOÀ ⇒ trượt test này.

    ⚠️ Bản đầu của test này đòi mạnh hơn: *1 fold hiếm thắng 2 fold chung*. **Sai yêu
    cầu**, và phép đo phân bố `gidf` thật (§13.31) là thứ bác nó: rác §13.20 (`nước`,
    `1`, `bằng`, `từ`, `ra`) nằm ở **4.0–4.2**, fold đặc trưng **bão hoà ở 9.64** ⇒ đúng
    ra **~2.4 lần trùng rác = 1 lần trùng đặc trưng**. Đòi 1 > 2 là ép một tỉ lệ mà dữ
    liệu không có. Giữ lại ghi chú này để không ai "siết" test theo trực giác lần nữa.
    """
    from nn_aivie.hca.clause_layer import build_clause_layer, prompt_anchor_rescorer

    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    rs = prompt_anchor_rescorer(
        st,
        [st.vocab.id_of("Phở"), st.vocab.id_of("Việt"), st.vocab.id_of("Nam")],
        gidf_min=0.0,
    )

    class _C:
        def __init__(self, toks):
            self.tokens = tuple(toks)

    rare = _C([st.vocab.id_of("Phở")])       # 1 fold, HIẾM (1/4 cha)
    common = _C([st.vocab.id_of("Việt")])    # 1 fold, chung (4/4 cha)
    assert rs(None, rare) > rs(None, common), (rs(None, rare), rs(None, common))
    assert layer  # tầng dựng được, giữ ràng buộc chung với các test khác


def test_prompt_anchor_rescorer_is_zero_without_overlap():
    """Không trùng gì ⇒ 0.0 — bộ xếp phải TRUNG TÍNH, không âm thầm phạt."""
    from nn_aivie.hca.clause_layer import prompt_anchor_rescorer

    st = _trained()

    class _C:
        def __init__(self, toks):
            self.tokens = tuple(toks)

    rs = prompt_anchor_rescorer(st, [st.vocab.id_of("Phở")], gidf_min=0.0)
    assert rs(None, _C([st.vocab.id_of("Huế")])) == 0.0
    assert prompt_anchor_rescorer(st, [], gidf_min=0.0)(None, _C([1, 2])) == 0.0


def test_word_pivot_strength_bat_ban_le_doi_vai():
    """§13.39 — bản lề bị buộc chặt CẢ HAI phía = đang đổi vai trong từ (`tối đa`→`đa số`).

    Bệnh gốc §13.25 đọc lại cho đúng. Đo trên mini: 10,7% mối nối SINH có min ≥ 0.05,
    so với 0,0–0,1% mối nối THẬT của người viết ⇒ đây KHÔNG phải hiện tượng của văn thật.
    """
    from nn_aivie.hca.clause_layer import (
        compose_chain,
        ensure_boundary_signal,
        word_pivot_strength,
    )
    from nn_aivie.hca.w_skip import build_w_skip

    st = _trained()
    bound = ensure_boundary_signal(st)
    layer = build_clause_layer(st, pmi_theta=0.0)
    ws = build_w_skip(st, d_max=4, min_count=1, content_only=False)

    # thiếu vế nào cũng ⇒ 0.0, KHÔNG được ném lỗi: bước 0 của chuỗi có thể không có chữ trái
    assert word_pivot_strength(bound, None, 1, 2) == 0.0
    # đối xứng của `min`: một phía yếu thì tổng thể yếu, dù phía kia mạnh
    a, b = st.vocab.id_of("Việt"), st.vocab.id_of("Nam")
    assert 0.0 <= word_pivot_strength(bound, a, b, 12345) <= 1.0

    # bộ chặn phải TẮT mặc định — không được đổi hành vi sinh khi chưa qua chấm mù
    prompt = layer.clauses[0].tokens
    ch0, s0 = compose_chain(st, layer, ws, prompt, max_clauses=3, gidf_min=0.0)
    assert s0["n_pivot"] == 0, "pivot_max=None mà vẫn lọc ⇒ mặc định đã bị đổi"
    # Bật bộ chặn. ⚠️ Đây là test ĐẤU DÂY, không phải test ngưỡng: corpus đồ chơi 4 câu
    # không có sẵn cặp bigram nào đủ mạnh cả hai phía, nên ngưỡng thật (0.05) sẽ **không
    # bao giờ** kích hoạt ở đây và test sẽ xanh mà chẳng kiểm gì — đúng cái bẫy
    # `test_compose_chain_dead_when_gate_rejects_all` đã mắc (xem ghi chú ở đó).
    # `pivot_max=0.0` ⇒ mọi ứng viên đều bị chặn, chứng minh dây đã nối. Con số thật
    # (10,7% mối nối SINH vs 0,0–0,1% mối nối THẬT) nằm ở §13.39, đo trên mini.
    ch1, s1 = compose_chain(
        st, layer, ws, prompt, max_clauses=3, gidf_min=0.0, pivot_max=0.0)
    assert s1["n_pivot"] > 0, s1
    assert len(ch1) < len(ch0) and s1["stop"] == "loai_het", s1


def test_stop_on_close_dung_o_menh_de_dong_duoc_cau():
    """§13.42 — dừng ở mệnh đề ĐÓNG ĐƯỢC CÂU, sau khi đã đủ `min_clauses`.

    Vì sao cần: §13.41 chấm mù cho `max_clauses=2` thắng 78,7%, nhưng đó là hằng số MÙ —
    37% điểm dừng rơi giữa câu và giám khảo phạt 8 ca "cụt" (`dừng ở 'hoặc các'`).
    """
    from nn_aivie.hca.clause_layer import compose_chain, ensure_boundary_signal
    from nn_aivie.hca.w_skip import build_w_skip

    st = _trained()
    bound = ensure_boundary_signal(st)
    layer = build_clause_layer(st, pmi_theta=0.0)
    ws = build_w_skip(st, d_max=4, min_count=1, content_only=False)
    prompt = layer.clauses[0].tokens

    # TẮT mặc định — không được đổi hành vi sinh khi chưa qua chấm mù
    _ch, s0 = compose_chain(st, layer, ws, prompt, max_clauses=5, gidf_min=0.0)
    assert s0["stop"] != "dong_cau", s0

    ch, s1 = compose_chain(
        st, layer, ws, prompt, max_clauses=5, gidf_min=0.0,
        stop_on_close=True, min_clauses=1)
    # bất biến THẬT của cơ chế: nếu nó dừng vì `dong_cau` thì mệnh đề cuối PHẢI đóng câu,
    # và chuỗi phải đủ `min_clauses`. Không khoá độ dài cụ thể — corpus đồ chơi không
    # bảo đảm có đủ ứng viên (đúng bài học test-xanh-vô-nghĩa §13.36).
    if s1["stop"] == "dong_cau":
        assert len(ch) >= 1
        assert bound.ends_unit(int(ch[-1].tokens[-1])), "dừng vì đóng câu mà câu chưa đóng"
    # `min_clauses` phải được tôn trọng: không bao giờ dừng-vì-đóng-câu dưới ngưỡng
    ch2, s2 = compose_chain(
        st, layer, ws, prompt, max_clauses=5, gidf_min=0.0,
        stop_on_close=True, min_clauses=3)
    assert not (s2["stop"] == "dong_cau" and len(ch2) < 3), s2


def test_join_seq_max_chi_nhan_moi_noi_KE_NHAU():
    """§13.44 — mối nối phải được chứng thực KỀ NHAU, không chỉ đồng hiện cửa sổ 8.

    Đo trên mini: 46,6% mối nối sinh đặt cạnh nhau hai token CHƯA BAO GIỜ đứng kề nhau
    trong corpus (văn thật 5,2%); bật điều kiện này thì tỷ lệ đó về 0,0%.
    """
    import math

    from nn_aivie.hca.clause_layer import compose_chain, ensure_boundary_signal
    from nn_aivie.hca.w_skip import build_w_skip

    st = _trained()
    bound = ensure_boundary_signal(st)
    layer = build_clause_layer(st, pmi_theta=0.0)
    ws = build_w_skip(st, d_max=4, min_count=1, content_only=False)
    prompt = layer.clauses[0].tokens

    # TẮT mặc định
    _ch, s0 = compose_chain(st, layer, ws, prompt, max_clauses=3, gidf_min=0.0)
    assert s0["n_seq"] == 0, "join_seq_max=None mà vẫn lọc ⇒ mặc định đã bị đổi"

    ch, _s1 = compose_chain(
        st, layer, ws, prompt, max_clauses=3, gidf_min=0.0, join_seq_max=0.0)
    # Bất biến THẬT: mọi mối nối còn lại phải được chứng thực kề nhau VÀ ≤ ngưỡng.
    # Không khoá số mệnh đề — corpus đồ chơi không bảo đảm có ứng viên (bài học §13.36).
    for a, b in zip(ch, ch[1:]):
        p = bound.pmi(int(a.tokens[-1]), int(b.tokens[0]))
        assert math.isfinite(p), "mối nối lọt qua mà cặp CHƯA từng đứng kề nhau"
        assert p <= 0.0, f"mối nối lọt qua mà nằm ngoài dải biên: pmi={p}"


def test_reject_when_no_theme_va_bay_cong_tat_am_tham():
    """§13.48 — siết `gidf_min` lên mức ĐẶC TRƯNG có thể làm cổng TẮT HẲN, âm thầm.

    `accept` chỉ xét neo khi `theme_folds` khác rỗng. Nâng ngưỡng ⇒ nhiều đề không còn
    fold nào ⇒ cổng nhận MỌI ứng viên — siết lại hoá ra mở toang, đúng ở những đề khó
    nhất (đo mini: 19,7% đề rơi vào ca đó ở ngưỡng 7,0).
    """
    from nn_aivie.hca.clause_layer import CoverageGate

    st = _trained()
    layer = build_clause_layer(st, pmi_theta=0.0)
    cl = layer.clauses[0]

    # đề KHÔNG có fold nào ở ngưỡng đang dùng
    trong = CoverageGate(theme_folds=frozenset(), anchor_min_folds=1, gidf_min=1.5)
    assert trong.accept(st, cl, set()), (
        "hành vi CŨ phải giữ nguyên: theme rỗng ⇒ cổng không xét neo (nhận)"
    )
    va = CoverageGate(theme_folds=frozenset(), anchor_min_folds=1, gidf_min=1.5,
                      reject_when_no_theme=True)
    assert not va.accept(st, cl, set()), "bật cờ mà cổng vẫn nhận ⇒ bẫy chưa được vá"

    # cờ chỉ đụng ca theme RỖNG, không đụng ca thường
    from nn_aivie.hca.clause_layer import content_folds
    own = frozenset(content_folds(st, cl.tokens, gidf_min=1.5))
    assert own, "fixture không có fold — test vô nghĩa"
    for flag in (False, True):
        g = CoverageGate(theme_folds=own, anchor_min_folds=1, gidf_min=1.5,
                         reject_when_no_theme=flag)
        assert g.accept(st, cl, set()), "cờ đổi hành vi ở ca theme KHÁC rỗng"
