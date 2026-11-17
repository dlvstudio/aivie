"""TK_045 P7 — cạnh trong câu L2R/R2L+d và W_agenda."""

from __future__ import annotations

import numpy as np

from nn_aivie.core.parent_store import AbstractNeuron, ParentStore
from nn_aivie.core.state import GraphState
from nn_aivie.data.qa import DAP_LABEL
from nn_aivie.hca.config import HCAConfig
from nn_aivie.hca.sent_intra import (
    accumulate_span_pairs,
    build_sent_intra_views,
    clear_sent_intra_cache,
    ensure_sent_intra,
    r2l_anchor_scores,
    sent_intra_decode_channels,
)
from nn_aivie.hca.matrix_core import (
    ensure_parent_bipartite,
    matrix_core_logits,
    parent_coactivation_logits,
)


def _toy_em_be_lan() -> GraphState:
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    ids = st.vocab.encode_tokens(["Em", "bé", "Lan", "."])
    ps.add_example(ids, st.vocab, dap_id=dap)
    st.parents = ps
    clear_sent_intra_cache()
    return st


def test_em_be_lan_has_six_directed_pairs():
    """Em bé Lan → 3 L2R + 3 R2L (content only)."""
    toks = [0, 1, 2]  # fake ids
    l2r, r2l = accumulate_span_pairs(toks, lam=1.0, d_max=12, weight=1.0)
    assert len(l2r) == 3
    assert len(r2l) == 3
    # distances: (0,1)=1, (1,2)=1, (0,2)=2
    ds = sorted(j - i for i, j, _ in [(0, 1, 0), (1, 2, 0), (0, 2, 0)])
    assert ds == [1, 1, 2]
    pairs_l = {(a, b) for a, b, _ in l2r}
    assert pairs_l == {(0, 1), (1, 2), (0, 2)}
    pairs_r = {(a, b) for a, b, _ in r2l}
    assert pairs_r == {(1, 0), (2, 1), (2, 0)}


def test_build_views_on_em_be_lan():
    st = _toy_em_be_lan()
    views = build_sent_intra_views(st, lam=0.85, d_max=12, content_only=True, use_gidf=False)
    assert views.n_sent_spans >= 1
    assert views.n_l2r_pairs == 3
    assert views.n_r2l_pairs == 3
    em, be, lan = st.vocab.encode_tokens(["Em", "bé", "Lan"])
    assert views.W_l2r[em, be] > 0
    assert views.W_l2r[em, lan] > 0
    assert views.W_r2l[lan, em] > 0
    assert views.W_r2l[lan, be] > 0
    # d=2 yếu hơn d=1 khi λ<1
    assert views.W_l2r[em, lan] < views.W_l2r[em, be] + 1e-9


def test_r2l_anchors_from_lan_see_em():
    st = _toy_em_be_lan()
    views = ensure_sent_intra(st, force=True)
    em, be, lan = st.vocab.encode_tokens(["Em", "bé", "Lan"])
    hist = [em, be, lan]
    anchors = r2l_anchor_scores(views, hist, vocab_size=len(st.vocab), M=2)
    assert anchors[em] > 0
    assert anchors[be] > 0


def test_agenda_from_episode_members():
    st = GraphState()
    dap = st.vocab.id_of(DAP_LABEL)
    ps = ParentStore()
    s1 = st.vocab.encode_tokens(["Ngô", "Quyền", "lập", "nước", "."])
    s2 = st.vocab.encode_tokens(["Chiến", "thắng", "Bạch", "Đằng", "."])
    ps.add_example(s1, st.vocab, dap_id=dap)
    ps.add_example(s2, st.vocab, dap_id=dap)
    ps.episodes.append(AbstractNeuron(members=((0, 1.0), (1, 1.0))))
    st.parents = ps
    clear_sent_intra_cache()
    views = build_sent_intra_views(st, use_gidf=False)
    assert views.n_agenda_edges >= 1
    nuoc = st.vocab.id_of("nước")
    chien = st.vocab.id_of("Chiến")
    assert views.W_agenda[nuoc, chien] > 0


def test_r2l_not_added_as_raw_next_in_matrix_logits():
    """Đứng ở Lan: Em có thể là neo nhưng không được đẩy thr R2L làm next thắng bé."""
    st = _toy_em_be_lan()
    em, be, lan = st.vocab.encode_tokens(["Em", "bé", "Lan"])
    cfg = HCAConfig(
        ar_matrix_core=True,
        ar_sent_intra=True,
        ar_sent_l2r_weight=3.0,
        ar_sent_agenda_weight=0.0,
        ar_matrix_qk_weight=0.0,
        ar_matrix_stack_weight=0.0,
        ar_matrix_know_weight=0.0,
        ar_matrix_readout_weight=0.0,
        ar_sent_lr_use_gidf=False,
    )
    clear_sent_intra_cache()
    ensure_parent_bipartite(st)
    hist = [em, be, lan]
    _bag, _succ, _conf, act = parent_coactivation_logits(
        st, hist, vocab_size=len(st.vocab)
    )
    anchors, cont, _ag, _ = sent_intra_decode_channels(
        st, hist, act, config=cfg, vocab_size=len(st.vocab)
    )
    assert anchors[em] > 0
    # Sau hết emit, remaining mask rỗng → cont phải ~0 (không hub)
    assert float(cont.max()) == 0.0 or cont[em] <= cont.max() * 1e-6


def test_matrix_core_runs_with_sent_intra():
    st = _toy_em_be_lan()
    em, be, lan = st.vocab.encode_tokens(["Em", "bé", "Lan"])
    cfg = HCAConfig(
        ar_matrix_core=True,
        ar_sent_intra=True,
        ar_matrix_qk_weight=0.0,
        ar_matrix_stack_weight=0.0,
        ar_matrix_know_weight=0.0,
        ar_matrix_readout_weight=0.0,
    )
    clear_sent_intra_cache()
    logits = matrix_core_logits(st, [em, be], config=cfg)
    assert logits.shape[0] == len(st.vocab)
    assert np.isfinite(logits).all()


def test_ensure_sent_qk_from_l2r_r2l():
    from nn_aivie.hca.sent_intra import ensure_sent_qk, ppmi_from_csr

    st = _toy_em_be_lan()
    clear_sent_intra_cache()
    views = ensure_sent_intra(st, force=True)
    ppmi = ppmi_from_csr(views.W_l2r)
    assert ppmi.nnz > 0
    qk_l = ensure_sent_qk(st, which="l2r", dim=8)
    qk_r = ensure_sent_qk(st, which="r2l", dim=8)
    assert qk_l.wq.shape[0] == len(st.vocab)
    assert qk_r.wq.shape[1] == 8
    cfg = HCAConfig(
        ar_matrix_core=True,
        ar_sent_intra=True,
        ar_matrix_qk_from_l2r=True,
        ar_matrix_qk_weight=0.5,
        ar_matrix_r2l_ffn_weight=0.3,
        ar_matrix_stack_weight=0.0,
        ar_matrix_know_weight=0.0,
        ar_matrix_readout_weight=0.0,
        ar_sent_qk_dim=8,
    )
    em, be, _lan = st.vocab.encode_tokens(["Em", "bé", "Lan"])
    logits = matrix_core_logits(st, [em, be], config=cfg)
    assert np.isfinite(logits).all()


def test_sent_qk_sidecar_roundtrip(tmp_path):
    from nn_aivie.hca.sent_intra import (
        ensure_sent_qk,
        load_sent_qk_sidecar,
        save_sent_qk_sidecar,
    )

    st = _toy_em_be_lan()
    clear_sent_intra_cache()
    path = tmp_path / "toy.sent_qk.npz"
    save_sent_qk_sidecar(st, path, dim=8)
    clear_sent_intra_cache()
    assert load_sent_qk_sidecar(st, path, dim=8)
    qk = ensure_sent_qk(st, which="l2r", dim=8)
    assert qk.wq.shape == (len(st.vocab), 8)


def test_long_context_h_logits_finite():
    from nn_aivie.hca.sent_intra import long_context_h_logits, r2l_anchor_scores

    st = _toy_em_be_lan()
    n = len(st.vocab)

    class _E:
        matrix = np.random.default_rng(0).normal(size=(n, 8))

    st.embedding = _E()  # type: ignore[assignment]
    em, be, lan = st.vocab.encode_tokens(["Em", "bé", "Lan"])
    views = ensure_sent_intra(st, force=True)
    anchors = r2l_anchor_scores(views, [em, be, lan], vocab_size=n, M=2)
    cont = np.zeros(n)
    cont[lan] = 1.0
    sc = long_context_h_logits(st, [em, be], anchors, cont, config=HCAConfig())
    assert sc.shape[0] == n
    assert np.isfinite(sc).all()
