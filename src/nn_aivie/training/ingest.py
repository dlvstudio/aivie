"""
Nạp dữ liệu thật (câu fact / cặp QA) vào GraphState theo đúng kiến trúc v3.

Mỗi câu/QA → (a) co-occurrence G1 + W_seq (học cạnh A, E, fluency) và (b) **một nơ-ron cha**
(pattern completion). QA thêm bind đáp G2. Dùng cho ingest HF (Wikipedia/ViQuAD/…) sau khi đã
qua cổng chất lượng `data/quality.py`.
"""

from __future__ import annotations

from nn_aivie.core.edge_kinds import EdgeKind
from nn_aivie.core.state import GraphState
from nn_aivie.core.vocab import SPECIAL_EOS
from nn_aivie.data.qa import render_qa
from nn_aivie.teaching.align import tokenize, tokenize_punct
from nn_aivie.training.g1_batch import begin_g1_batch, flush_g1_batch, g1_and_wseq

_g1_and_wseq = g1_and_wseq


def _with_eos(toks: list[str]) -> list[str]:
    """Nối EOS vào CUỐI mỗi đơn vị train — mốc "hết đơn vị" phải LÀ DỮ LIỆU.

    (Ý tác giả 2026-07-15.) Trước đây chỉ `g1_passive` (đường lesson cũ) nối EOS; đường nạp
    corpus thật (`ingest_*` → `train_v3`) thì KHÔNG → `W_seq['.'][EOS] = 0` → mạng không có
    gì để học "đây là điểm kết". Hệ quả: mọi cơ chế cần khái niệm "ranh giới diễn ngôn" phải
    ĐOÁN từ danh sách dấu câu viết tay (`at_boundary` TK_009) — vi phạm Luật 0, và đo được
    là không thể thay bằng tín hiệu dẫn xuất: entropy W_seq cho `là`=5.45 > `,`=4.31; tỉ lệ
    đứng-cuối-chuỗi cho `,`=`:`=`Đáp:`=0.000, ngang hệt từ nội dung (TK_012 §9.5).

    EOS đi vào `ids` như token BÌNH THƯỜNG (không đặc cách): W_seq học `. → EOS`, cha giữ nó
    trong `source_ids`. Nó KHÔNG vào `match_ids` vì `is_surface_token(SPECIAL_EOS)` = False,
    nên không nhiễu retrieval; `decode_surface` cũng lọc nó khỏi output.
    """
    return [*toks, SPECIAL_EOS]


def encode_content(vocab, text: str, *, legacy_tok=None, lookup_only: bool = False) -> list[int]:
    """Text → ids, KHÔNG nối EOS. Cho bag-of-token (hồ sơ Entity:, sense từ điển).

    Cùng cửa với `encode_unit` — nếu đường này tự tokenize lấy thì id nó tra ra sẽ KHÁC id
    mà train nạp (bug #3: `_token_to_id.get("Nội")` trúng biến thể không-Ġ), và trên vocab
    BPE `encode_tokens` còn **ĐẺ id mới ngoài dải bề mặt** → phá tính ĐÓNG (§2.4).

    `lookup_only` (legacy): bỏ qua token chưa có thay vì thêm vào vocab — `ingest_entities`
    khai thác hồ sơ thực thể bằng `_token_to_id.get`, KHÔNG được phép làm vocab mọc lúc mine.
    Trên BPE thì vô nghĩa (mọi token đã có sẵn) nên nhánh này chỉ giữ hành vi legacy.
    """
    if vocab.tokenizer is not None:
        return vocab.encode_text(text)
    toks = (legacy_tok or tokenize)(text)
    if lookup_only:
        return [i for i in (vocab._token_to_id.get(t) for t in toks) if i is not None]
    return vocab.encode_tokens(toks)


def encode_unit(state: GraphState, text: str, *, legacy_tok=None) -> list[int]:
    """Text đơn-vị-train → ids + EOS. **CỬA NẠP DUY NHẤT** (TK_041 §4.3).

    Trước TK_041 mỗi đường nạp tự chọn tokenizer (`tok = tokenize_punct if split_punct else
    tokenize`) → 3 đường độc lập, và đã sót 2 lần trong đợt EOS (README §0.4.2): sót
    `pipeline.py` suýt bake 7.7 giờ ra checkpoint không mốc kết; sót `build_varlm.py` →
    verify FAIL. Cờ `split_punct` tồn tại CHỈ vì có hai tokenizer đối chọi — nay một cửa
    thì nó vô nghĩa.

    `legacy_tok` chỉ dùng khi vocab chưa có tokenizer (checkpoint/test tiền-TK_041).
    """
    v = state.vocab
    return [*encode_content(v, text, legacy_tok=legacy_tok), v.id_of(SPECIAL_EOS)]


def ingest_statements(
    state: GraphState,
    sentences: list[str],
    *,
    eta: float = 0.05,
    priority: float = 0.6,
    split_punct: bool = False,
    logger=None,
    log_every: int = 2000,
) -> dict[str, int]:
    """Câu fact → co-occurrence + W_seq + nơ-ron cha (statement). `priority`<1 = ngoài miền.

    `split_punct=True` (R0): tách dấu câu (`,`/`.`) thành token riêng — cần cho A2/discourse
    (học `, → nó`) khi nạp dữ liệu thật/HF có dấu dính từ (`ra,`).

    `logger` (FileProgressLog) → ghi tiến độ mỗi `log_every` câu (biết đang nạp đến đâu).
    """
    from nn_aivie.data.qa import DAP_LABEL

    tok = tokenize_punct if split_punct else tokenize
    dap_id = state.vocab.id_of(DAP_LABEL)
    total = len(sentences)
    n = 0
    ps = state.parents
    # NẠP HÀNG LOẠT: defer_index + begin/end_batch — KHÔNG rebuild index O(#parents) MỖI câu
    # (80k câu × 144k parents = treo). Rebuild 1 lần cuối. (Fix TK_027; lợi cả train_v3.)
    if ps is not None:
        begin_g1_batch(state)
        ps.begin_ingest_batch()
    for k, text in enumerate(sentences):
        ids = encode_unit(state, text, legacy_tok=tok)
        if len(ids) >= 3:  # ≥2 token nội dung + EOS
            g1_and_wseq(state, ids, eta)
            if ps is not None:
                ps.add_example(ids, state.vocab, dap_id=dap_id, priority=priority,
                               defer_index=True)
            n += 1
        if logger is not None and (k + 1) % log_every == 0:
            logger.progress("ingest_statements", k + 1, total, extra=f"parents+={n}")
    if ps is not None:
        flush_g1_batch(state)
        ps.finalize_batch_ingest(state.vocab, logger=logger)
    return {"statements": n}


def ingest_qa(
    state: GraphState,
    pairs: list[tuple[str, str]],
    *,
    eta: float = 0.05,
    priority: float = 0.6,
    split_punct: bool = False,
    logger=None,
    log_every: int = 2000,
) -> dict[str, int]:
    """Cặp (hỏi, đáp) → frame `Hỏi:/Đáp:` → co-occurrence + W_seq + đáp bind + nơ-ron cha QA.

    `split_punct=True` (R0): tách dấu câu thành token riêng (dữ liệu thật/HF).
    `logger` → ghi tiến độ mỗi `log_every` cặp.
    """
    from nn_aivie.data.qa import DAP_LABEL

    tok = tokenize_punct if split_punct else tokenize
    dap_id = state.vocab.id_of(DAP_LABEL)
    total = len(pairs)
    n = 0
    ps = state.parents
    if ps is not None:
        ps.begin_ingest_batch()
    for k, (question, answer) in enumerate(pairs):
        prompt, _ = render_qa(question, answer)  # "Hỏi: q Đáp: a"
        ids = encode_unit(state, prompt, legacy_tok=tok)
        if len(ids) >= 4 and dap_id in ids:
            _g1_and_wseq(state, ids, eta)
            # củng cố đường đáp (REINFORCED) như G2 — gồm cả `→ EOS`: đáp KẾT THÚC ở đâu
            # cũng là thứ phải học, không phải thứ suy ra từ dấu chấm.
            # Tìm mốc trên ID, không trên chuỗi: `Đáp:` là special NGUYÊN TỬ (§2.2) nên
            # `dap_id in ids` là phép so int — BPE không thể tách nó thành `Đáp`+`:`.
            di = ids.index(dap_id)
            ans_ids = ids[di + 1 :]
            for a, b in zip(ans_ids, ans_ids[1:]):
                state.adjacency.add(a, b, eta * 1.2, EdgeKind.REINFORCED)
            if state.parents is not None:
                state.parents.add_example(ids, state.vocab, dap_id=dap_id, priority=priority)
            n += 1
        if logger is not None and (k + 1) % log_every == 0:
            logger.progress("ingest_qa", k + 1, total, extra=f"parents+={n}")
    # TK_036 S3 — incremental skip ledger lúc train; materialize cuối batch
    if ps is not None:
        ps.seg.end_batch(ps.parents)
        ps.rebuild_hierarchical_index()
        ps.flush_skip_ledger(state.vocab, logger=logger)
    return {"qa": n}
