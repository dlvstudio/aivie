"""Unit: soft path bỏ hư-từ `ở` khi khớp emit (TK_045 §12.2c)."""

from __future__ import annotations

import numpy as np

from nn_aivie.core.vocab import Vocabulary
from nn_aivie.hca.matrix_core import ParentBipartite, _path_continuation_logits
from scipy.sparse import csr_matrix


def _vocab_with(*surfaces: str) -> Vocabulary:
    v = Vocabulary()
    for s in surfaces:
        v.id_of(s)
    return v


def test_soft_path_skips_function_word_o():
    # hist: Lê Lợi khởi nghĩa ở   emit: Lê Lợi khởi nghĩa Lam Sơn
    words = ["Lê", "Lợi", "khởi", "nghĩa", "ở", "Lam", "Sơn"]
    v = _vocab_with(*words)
    ids = {w: v.id_of(w) for w in words}
    hist = [ids["Lê"], ids["Lợi"], ids["khởi"], ids["nghĩa"], ids["ở"]]
    emit = np.array(
        [ids["Lê"], ids["Lợi"], ids["khởi"], ids["nghĩa"], ids["Lam"], ids["Sơn"]],
        dtype=np.int32,
    )
    n = len(v)
    bi = ParentBipartite(
        A=csr_matrix((n, 1)),
        E=csr_matrix((1, n)),
        n_parents=1,
        succ={},
        emits=[emit],
        end_after=np.zeros(n),
        cont_after=np.zeros(n),
    )
    act = np.array([1.0])

    exact, _ = _path_continuation_logits(
        bi, act, hist, vocab_size=n, vocab=v, soft=False
    )
    soft, _ = _path_continuation_logits(
        bi, act, hist, vocab_size=n, vocab=v, soft=True
    )
    # Exact chết vì đuôi `ở` ≠ `Lam`; soft bỏ `ở` → next = Lam
    assert float(exact[ids["Lam"]]) == 0.0
    assert float(soft[ids["Lam"]]) > 0.0


def test_soft_path_keeps_content_nam():
    # Không bỏ `năm` — sai emit "…Đằng của" không được khớp bằng cách skip năm
    words = ["Trận", "Bạch", "Đằng", "năm", "của", "Ngô"]
    v = _vocab_with(*words)
    ids = {w: v.id_of(w) for w in words}
    hist = [ids["Trận"], ids["Bạch"], ids["Đằng"], ids["năm"]]
    emit = np.array(
        [ids["Trận"], ids["Bạch"], ids["Đằng"], ids["của"], ids["Ngô"]],
        dtype=np.int32,
    )
    n = len(v)
    bi = ParentBipartite(
        A=csr_matrix((n, 1)),
        E=csr_matrix((1, n)),
        n_parents=1,
        succ={},
        emits=[emit],
        end_after=np.zeros(n),
        cont_after=np.zeros(n),
    )
    act = np.array([1.0])
    soft, _ = _path_continuation_logits(
        bi, act, hist, vocab_size=n, vocab=v, soft=True
    )
    # `năm` là content → không match `của`; không được đẩy `của`
    assert float(soft[ids["của"]]) == 0.0
