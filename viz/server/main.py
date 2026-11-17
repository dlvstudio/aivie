"""FastAPI server for AIVIE Graph UI (TK_001)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from nn_aivie.core.checkpoint_io import load_checkpoint
from nn_aivie.core.content_cache import ContentCache
from nn_aivie.core.paths import project_root, viz_default_checkpoint
from nn_aivie.core.state import GraphState
from nn_aivie.teaching.align import tokenize_punct
from nn_aivie.viz.export_graph import (
    export_graph_json,
    list_rule_names,
    subgraph_for_tokens,
)
from nn_aivie.viz.export_graph_v3 import completion_graph, list_parents
from nn_aivie.viz.export_hlinks import hlinks_info, hlinks_of
from nn_aivie.viz.export_embedding import (
    embedding_compare,
    embedding_info,
    embedding_lookup,
    embedding_search,
    embedding_senses,
)

CHECKPOINT = Path(os.environ.get("AIVIE_CHECKPOINT", str(viz_default_checkpoint())))
_state: GraphState | None = None
_cache: ContentCache | None = None

VIZ_ROOT = project_root() / "viz"
STATIC_DIR = VIZ_ROOT / "web" / "dist"
FALLBACK_HTML = VIZ_ROOT / "server" / "fallback.html"


_load_lock = __import__("threading").Lock()


def get_state() -> GraphState:
    global _state, _cache
    if _state is None:
        # TK_016: lock chống RACE — 2 request đồng thời cùng thấy None → double-load 2.5GB
        # → thrash memory (đo được: log load bị đôi). Chỉ load MỘT lần.
        with _load_lock:
            if _state is None:
                if not CHECKPOINT.exists():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Checkpoint not found: {CHECKPOINT}",
                    )
                _state, _cache = load_checkpoint(CHECKPOINT)
    return _state


def get_cache() -> ContentCache:
    """Content-cache (nhãn debug/UI). TK_012: chỉ viz đọc — không phải inference."""
    get_state()
    return _cache if _cache is not None else ContentCache()


_encoder = None
_encoder_tried = False
_knn = None
_knn_tried = False


def get_encoder():
    """§192 — encoder ngữ cảnh (lazy, 1 lần). None nếu thiếu torch/checkpoint."""
    global _encoder, _encoder_tried
    if not _encoder_tried:
        _encoder_tried = True
        try:
            from nn_aivie.hca.ctx_encoder import load_ctx_encoder
            _encoder = load_ctx_encoder()
            print(f"[encoder] {'nạp OK' if _encoder else 'không có → dùng cosine tĩnh'}")
        except Exception as e:  # noqa: BLE001
            print(f"[encoder] lỗi nạp ({e}) → cosine tĩnh")
            _encoder = None
    return _encoder


def get_knn():
    """§TK_056 — kNN-LM (lazy, 1 lần): datastore encoder cho đường sinh. None nếu
    thiếu torch/datastore/encoder → sinh chạy như cũ."""
    global _knn, _knn_tried
    if not _knn_tried:
        _knn_tried = True
        try:
            from pathlib import Path
            from nn_aivie.hca.ctx_encoder import load_ctx_encoder
            from nn_aivie.hca.knn_lm import load_knn_lm
            # §TK_059 — ƯU TIÊN encoder CAUSAL + datastore causal (kNN-LM bits
            # 7.39 vs 8.44 masked, ~2× perplexity). QA vẫn dùng masked bidirectional
            # (get_encoder). Rơi về masked nếu thiếu file causal.
            # ưu tiên d512MIX (d512 warm-start + 174M token fw2+viwiki+prose, văn
            # phong Wikipedia mượt nhất) > DEEP (d384 curated, grounded hơn) > ...
            cands = [
                ("d512MIX CLEAN", "checkpoints/ctx_encoder_causal_d512mix.pt",
                 "checkpoints/net_ls_concept_conf.datastore_d512mix_clean.npz"),
                ("d512MIX 174M", "checkpoints/ctx_encoder_causal_d512mix.pt",
                 "checkpoints/net_ls_concept_conf.datastore_d512mix.npz"),
                ("DEEP 12t ctx128", "checkpoints/ctx_encoder_causal_deep.pt",
                 "checkpoints/net_ls_concept_conf.datastore_deep.npz"),
                ("LONG ctx128", "checkpoints/ctx_encoder_causal_long.pt",
                 "checkpoints/net_ls_concept_conf.datastore_long.npz"),
                ("XL d384", "checkpoints/ctx_encoder_causal_xl.pt",
                 "checkpoints/net_ls_concept_conf.datastore_xl.npz"),
                ("BIG d256", "checkpoints/ctx_encoder_causal_big.pt",
                 "checkpoints/net_ls_concept_conf.datastore_big.npz"),
                ("causal d128", "checkpoints/ctx_encoder_causal.pt",
                 "checkpoints/net_ls_concept_conf.datastore_causal.npz"),
            ]
            _knn = None
            for tag, ep, dp in cands:
                if Path(ep).exists() and Path(dp).exists():
                    _knn = load_knn_lm(load_ctx_encoder(ep), dp)
                    if _knn:
                        print(f"[knn] nạp OK ({tag})"); break
            if _knn is None:
                _knn = load_knn_lm(get_encoder())
                print(f"[knn] {'nạp OK (masked)' if _knn else 'không có → vote thuần'}")
        except Exception as e:  # noqa: BLE001
            print(f"[knn] lỗi nạp ({e}) → vote thuần")
            _knn = None
    return _knn


app = FastAPI(title="AIVIE Graph UI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_exists": CHECKPOINT.exists(),
        "ui": "react" if STATIC_DIR.exists() else "fallback",
    }


@app.get("/api/rules")
def rules() -> dict:
    state = get_state()
    return {"rules": list_rule_names(state)}


import re as _re

# §183 — cầu nối QA→STATEMENT. Định nghĩa lưu dạng phát biểu ("Sắt là kim
# loại…"), không phải QA. `generate()` thô chỉ nối token CỤC BỘ nên "Hỏi: Sắt là
# gì? Đáp:" không truy được định nghĩa (ra "có bốn"). Truy-hồi-theo-thực-thể đúng
# nghĩa nằm ở completion_graph (fact/hub) nhưng treo trên 450k cha. Cầu nối rẻ:
# bóc thực thể X từ "…X là gì?…" và sinh từ "X là".
_QA_DINH_NGHIA = _re.compile(
    r"^\s*(?:hỏi\s*:\s*)?(.+?)\s+là\s+(?:cái\s+)?gì\s*\?*\s*(?:đáp\s*:?.*)?$",
    _re.IGNORECASE | _re.DOTALL,
)


def _qa_entity(prompt: str) -> str | None:
    """Thực thể X nếu prompt là câu hỏi định nghĩa "X là gì?", ngược lại None."""
    m = _QA_DINH_NGHIA.match(prompt)
    if not m:
        return None
    ent = m.group(1).strip().strip("\"'“”")
    return ent if ent and len(ent) <= 40 else None


def _qa_bridge(prompt: str) -> str | None:
    """Câu hỏi định nghĩa "X là gì?" → prompt sinh "X là" (fallback khi không
    tra được câu định nghĩa đã lưu)."""
    ent = _qa_entity(prompt)
    return f"{ent} là" if ent else None


def _lm_led_generate(enc, prompt_ids, max_len, rep_pen=3.5, recent=24,
                     knn=None, ground_a=0.0, eos_id=None,
                     temp=0.7, top_p=0.9, seed=0):
    """§TK_085 ĐẢO VAI — causal LM DẪN sinh (mạch-lạc, đo được 83%), kNN datastore
    SẠCH làm lớp KIỂM/neo-sự-thật (log-space nudge). LM lo trôi-chảy, đếm/kNN lo
    grounding — ngược với đường /api/gen cũ (đếm dẫn → trôi).

    Decoding: rep-penalty + CHẶN lặp bigram + nucleus (top_p) softmax(temp)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ids = [int(t) for t in prompt_ids]
    seen_bigrams = set(zip(ids, ids[1:]))
    for _ in range(int(max_len)):
        lo = enc.causal_logits_last(ids)
        if lo is None:
            break
        lo = lo.astype(np.float64).copy()
        for t in set(ids[-recent:]):                 # rep-penalty token gần
            if 0 <= t < lo.shape[0]:
                lo[t] -= rep_pen
        last = ids[-1] if ids else None              # CHẶN lặp bigram đã gặp
        if last is not None:
            for (a, b) in seen_bigrams:
                if a == last and 0 <= b < lo.shape[0]:
                    lo[b] -= 6.0
        if knn is not None and ground_a > 0:          # KIỂM: nudge theo datastore sạch
            P = knn.next_dist(ids, lo.shape[0])
            if P is not None:
                lo = lo + ground_a * np.log(P + 1e-8)
        # nucleus top_p trên softmax(temp)
        lo = lo / max(temp, 1e-6)
        lo -= lo.max()
        p = np.exp(lo); p /= p.sum()
        order = np.argsort(p)[::-1]
        cdf = np.cumsum(p[order])
        keep = order[:max(1, int(np.searchsorted(cdf, top_p)) + 1)]
        pk = p[keep] / p[keep].sum()
        nx = int(rng.choice(keep, p=pk))
        if eos_id is not None and nx == int(eos_id):
            break
        if ids:
            seen_bigrams.add((ids[-1], nx))
        ids.append(nx)
    return ids


@app.get("/api/gen")
def gen_light(prompt: str, max_len: int = 24, knn: int = 1, knn_lambda: float = 0.35,
              hybrid: int = 0, extend: int = 0, extend_min: int = 6,
              rerank: int = 0, rerank_a: float = 0.6,
              anchor: int = 0, anchor_beta: float = 6.0,
              lm_led: int = 1, ground_a: float = 0.5, lm_temp: float = 0.5) -> dict:
    """§182 — SINH nhẹ: gọi thẳng generate() (cấu hình mặc định, §171 loại suy
    active), bỏ máy móc fact/essay/multi của /api/v3/complete (quá nặng trên
    checkpoint 450k cha). W_skip cache sau lần đầu → các lần sau nhanh.

    §183 — QA định nghĩa ("X là gì?") tự bắc cầu sang statement "X là"."""
    from nn_aivie.hca.config import HCAConfig
    from nn_aivie.hca.generate_hebbian import generate as _gen

    state = get_state()
    ml = min(64, max(1, int(max_len)))
    t0 = time.perf_counter()

    def _grounded(text: str, pidx: int, kind: str) -> dict:
        return {
            "prompt": prompt, "answer": text, "tail": text, "generated": text,
            "grounded": True, "answer_kind": kind, "source_parent": pidx,
            "meta": {"elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)},
        }

    # §185 — QA định nghĩa "X là gì?": TRA câu định nghĩa đã lưu (grounded).
    ent = _qa_entity(prompt)
    if ent is not None:
        hit = state.parents.define_entity(state.vocab, ent)
        if hit is not None:
            return _grounded(hit[0], hit[1], "definition")

    # §186/§192 — câu hỏi KHÁC DẠNG: truy-hồi fact, tách nghĩa bằng encoder ngữ
    # cảnh (nếu nạp được) — nếu không, rơi về cosine tĩnh §188b.
    if "?" in prompt or _QA_DINH_NGHIA.match(prompt):
        hit = state.parents.answer_fact(
            state.vocab, list(state.vocab.encode_prompt(prompt)),
            embedding=state.embedding, encoder=get_encoder(),
        )
        if hit is not None:
            return _grounded(hit[0], hit[1], "fact")

    # fallback: sinh (có cầu nối "X là" nếu là câu hỏi định nghĩa)
    gen_prompt = _qa_bridge(prompt) or prompt
    # §TK_085 ĐẢO VAI — LM DẪN (kNN datastore sạch KIỂM). Bỏ đường W_skip-dẫn.
    if lm_led and get_knn() is not None:
        enc = get_knn()._enc
        eos = state.vocab._token_to_id.get("</s>") or state.vocab._token_to_id.get("<eos>")
        pids = list(state.vocab.encode_prompt(gen_prompt))
        ids = _lm_led_generate(enc, pids, ml, knn=(get_knn() if ground_a > 0 else None),
                               ground_a=ground_a, eos_id=eos, temp=lm_temp)
        n = len(pids)
        return {"prompt": prompt, "gen_prompt": gen_prompt, "lm_led": True,
                "grounded": False, "ground_a": ground_a, "lm_temp": lm_temp,
                "generated": state.vocab.detokenize(ids),
                "tail": state.vocab.detokenize(ids[n:]),
                "meta": {"elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}}
    # §TK_056 — bật kNN-LM (encoder h → datastore) cho đường sinh nếu có.
    from nn_aivie.hca.generate_hebbian import set_knn_hook, set_anchor_hook
    knn_on = bool(knn) and get_knn() is not None
    # §TK_083 Pha 1c — neo mức-prompt (độc lập kNN; dùng embedding tĩnh của mạng)
    if anchor:
        set_anchor_hook(state.embedding.matrix, beta=anchor_beta)
    # §TK_061 — hybrid opt-in (sinh lai cổng-độ-tin): an toàn (facts giữ) nhưng
    # lợi trôi-chảy còn kín vì causal-LM d=128 yếu. Mặc định TẮT; bật &hybrid=1.
    if knn_on and rerank:
        set_knn_hook(get_knn(), rerank=True, rerank_a=rerank_a)   # §TK_082 Pha 1b
    elif knn_on and hybrid:
        set_knn_hook(get_knn(), hybrid=True)
    elif knn_on:
        set_knn_hook(get_knn(), lam=knn_lambda)
    try:
        ids = [int(x) for x in _gen(state, gen_prompt, max_len=ml, config=HCAConfig())]
    finally:
        set_knn_hook(None)
        set_anchor_hook(None)
    n = len(state.vocab.encode_prompt(gen_prompt))
    # §TK_061-v2 — BÀN GIAO SPAN: W_skip cụt/ngắn → causal-LM nối cho mượt.
    # W_skip dài (định nghĩa grounded) không đụng ⇒ facts an toàn.
    extended = False
    if extend and get_knn() is not None and (len(ids) - n) < int(extend_min):
        eos = state.vocab._token_to_id.get("</s>") or state.vocab._token_to_id.get("<eos>")
        # §TK_065 — chống-drift: dừng ở dấu câu khi độ-tin tụt (kết đoạn sạch,
        # ≥1 câu trọn, cắt đuôi rác). anchor tắt (đo được: thất thường).
        ids = get_knn().lm_extend(ids, eos_id=eos, max_new=ml, vocab=state.vocab,
                                  conf_stop=0.12)
        extended = True
    return {
        "prompt": prompt,
        "gen_prompt": gen_prompt,
        "bridged": gen_prompt != prompt,
        "grounded": False,
        "knn": knn_on,
        "hybrid": bool(knn_on and hybrid),
        "extended": extended,
        "knn_lambda": (None if hybrid else knn_lambda) if knn_on else None,
        "generated": state.vocab.detokenize(ids),
        "tail": state.vocab.detokenize(ids[n:]),
        "meta": {"elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)},
    }


@app.get("/api/graph/token")
def graph_token(
    token: str,
    kinds: str = "analogy,reinforced",
    min_weight: float = 0.0,
    max_edges: int = 400,
    hops: int = 1,
) -> dict:
    """§180 — subgraph quanh TOKEN để xem cạnh LOẠI SUY (analogy/reinforced).

    Ví dụ: /api/graph/token?token=sắt  → sắt→thiếc/kẽm/... (kind analogy|reinforced).
    Nhiều token: token=sắt,kẽm. `kinds` lọc loại cạnh (mặc định analogy+reinforced).
    """
    from nn_aivie.core.edge_kinds import EdgeKind

    state = get_state()
    toks = [t.strip() for t in token.split(",") if t.strip()]
    try:
        kind_list = [EdgeKind(k.strip()) for k in kinds.split(",") if k.strip()]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"kind không hợp lệ: {e}") from e
    return subgraph_for_tokens(
        state, toks, kinds=kind_list,
        min_weight=min_weight, max_edges=max_edges, hops=hops,
    )


@app.get("/api/graph")
def graph(
    rule: str = "vi_QA_hoi_dap",
    rules: str | None = None,
    min_weight: float = 0.15,
    max_edges: int = 800,
    hops: int = 2,
    mode: str = "structural",
) -> dict:
    state = get_state()
    names = [r.strip() for r in rules.split(",")] if rules else [rule]
    return export_graph_json(
        state,
        names,
        min_weight=min_weight,
        max_edges=max_edges,
        hops=hops,
        mode=mode,
        checkpoint=str(CHECKPOINT),
    )


# TK_019: /api/trace (Inference Lab v2) đã xóa — trace A3 nằm trong /api/v3/complete.

# --- v3: nơ-ron cha + pattern completion (thay đồ thị rule cũ) ---


class CompleteRequest(BaseModel):
    prompt: str
    top_candidates: int = Field(default=6, ge=1, le=16)
    max_len: int = Field(default=48, ge=1, le=64)
    mode: str = Field(default="autoregressive", pattern="^autoregressive$")
    multi: bool = True
    max_facts: int = Field(default=3, ge=1, le=8)


@app.get("/api/v3/parents")
def v3_parents(query: str = "", limit: int = 60) -> dict:
    return list_parents(get_state(), query=query, limit=limit)


@app.post("/api/v3/complete")
def v3_complete(body: CompleteRequest) -> dict:
    t0 = time.perf_counter()
    out = completion_graph(
        get_state(),
        body.prompt,
        top_candidates=body.top_candidates,
        max_len=body.max_len,
        mode=body.mode,  # type: ignore[arg-type]
        multi=body.multi,
        max_facts=body.max_facts,
        cache=get_cache(),
    )
    out.setdefault("meta", {})["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


# --- embedding E (PPMI+SVD) ---


@app.get("/api/assoc/related")
def assoc_related(phrase: str, topk: int = 10) -> dict:
    """TK_008 (2): cạnh liên tưởng CỤM↔CỤM học từ đồng-episode (phrase_assoc)."""
    st = get_state()
    ps = st.parents
    assoc = getattr(ps, "_phrase_assoc", None) if ps is not None else None
    # Tokenize + fold bằng chính cơ chế của mạng, KHÔNG strip ký tự bằng tay: bản cũ là
    # `w.strip(".,;:!?…\"'()[]").lower()` — chép nguyên `vocab.FOLD_STRIP` + `.lower()` từ
    # `Vocabulary._ensure`, tức bản sao tay của một thứ vocab đã tính sẵn lúc build. Lệch
    # một ký tự là UI tra ra kết quả khác mạng. `related_phrases(..., vocab=)` tự dịch
    # bề-mặt → fold-group-id (TK_012 Phase 3).
    words = [w for w in tokenize_punct(phrase.strip()) if w]
    if not assoc or len(words) < 2:
        return {"phrase": phrase, "available": bool(assoc), "pairs": [],
                "n_total": sum(len(v) for v in (assoc or {}).values()) // 2}
    bg = (st.vocab.fold_of(words[0]), st.vocab.fold_of(words[1]))
    pairs = [
        {"phrase": " ".join(b), "pmi": round(w, 2)}
        for b, w in ps.related_phrases(bg, topk=topk)
    ]
    return {"phrase": " ".join(bg), "available": True, "pairs": pairs,
            "n_total": sum(len(v) for v in assoc.values()) // 2}


@app.get("/api/embedding/info")
def embedding_meta() -> dict:
    return embedding_info(get_state())


@app.get("/api/embedding/lookup")
def embedding_nearest(
    token: str,
    topk: int = 16,
    min_sim: float = 0.0,
) -> dict:
    if not token.strip():
        raise HTTPException(status_code=400, detail="token required")
    try:
        return embedding_lookup(
            get_state(), token, topk=min(topk, 32), min_sim=min_sim
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/embedding/compare")
def embedding_pair(a: str, b: str) -> dict:
    if not a.strip() or not b.strip():
        raise HTTPException(status_code=400, detail="a and b required")
    try:
        return embedding_compare(get_state(), a, b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/embedding/search")
def embedding_vocab_search(q: str = "", limit: int = 40) -> dict:
    return embedding_search(get_state(), q, limit=min(limit, 80))


@app.get("/api/hlinks/info")
def hlinks_meta() -> dict:
    """TK_020 — thống kê cạnh ngang XA (FRAME_BRACKET + meta)."""
    return hlinks_info(get_state())


@app.get("/api/hlinks")
def hlinks(token: str, topk: int = 12) -> dict:
    """TK_020 — cạnh ngang từ opener: closer/PMI/power/gap-hist/ví dụ cha nguồn."""
    if not token.strip():
        raise HTTPException(status_code=400, detail="token required")
    return hlinks_of(get_state(), token, topk=min(topk, 24))


@app.get("/api/embedding/senses")
def embedding_word_senses(word: str, topk: int = 8) -> dict:
    """TK_016 — tách nghĩa: token đơn (blur) vs từng sense-neuron (hồ sơ hàng xóm riêng)."""
    if not word.strip():
        raise HTTPException(status_code=400, detail="word required")
    try:
        return embedding_senses(get_state(), word, topk=min(topk, 20))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


LOAI_SUY_HTML = VIZ_ROOT / "server" / "loai_suy.html"
PROMPT_HTML = VIZ_ROOT / "server" / "prompt.html"


@app.get("/loaisuy", response_model=None)
def loai_suy_view():
    """§181 — trang xem cạnh LOẠI SUY quanh một token (force-graph tối giản)."""
    if LOAI_SUY_HTML.exists():
        return FileResponse(LOAI_SUY_HTML)
    raise HTTPException(status_code=404, detail="loai_suy.html không tồn tại")


@app.get("/prompt", response_model=None)
def prompt_view():
    """§182 — trang SINH: nhập prompt → /api/v3/complete (thay tab Inference cũ
    gọi /api/trace đã xóa). Sinh dùng cấu hình mặc định (§171 loại suy active)."""
    if PROMPT_HTML.exists():
        return FileResponse(PROMPT_HTML)
    raise HTTPException(status_code=404, detail="prompt.html không tồn tại")


CHAT_HTML = VIZ_ROOT / "server" / "chat.html"

# ── §TK_080 — TRỢ LÝ hỏi-đáp có nguồn: retrieval BM25F (của ta) + QLoRA-3B (ta) ──
_assistant = None
_assistant_lock = __import__("threading").Lock()


def get_assistant():
    """Lazy-load: nạp BM25F (285k viwiki) + model C lần đầu được hỏi (~15-20s)."""
    global _assistant
    if _assistant is None:
        with _assistant_lock:
            if _assistant is None:
                import sys as _sys
                _sys.path.insert(0, str(project_root() / "scripts"))
                from assistant_ours import Assistant
                _assistant = Assistant()
    return _assistant


class ChatReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(3, ge=1, le=6)


@app.post("/api/assistant/chat")
def assistant_chat(req: ChatReq):
    """Hỏi → truy top-k nguồn → C trả lời có căn cứ → verify grounded_ratio."""
    r = get_assistant().answer(req.query, k=req.k)
    return {"answer": r["answer"], "grounded": r["grounded"], "sources": r["src"]}


@app.get("/chat", response_model=None)
def chat_view():
    """§TK_080 — trang hỏi-đáp tương tác, dùng /api/assistant/chat."""
    if CHAT_HTML.exists():
        return FileResponse(CHAT_HTML)
    raise HTTPException(status_code=404, detail="chat.html không tồn tại")


MAMBA_HTML = VIZ_ROOT / "server" / "mamba.html"

# ── §TK_093 — DEMO SINH-CHỮ Mamba/hybrid (kiến-trúc TỰ-XÂY, base LM 9.9M đồ-chơi) ──
_mamba = None
_mamba_lock = __import__("threading").Lock()


def get_mamba():
    """Lazy-load vocab + checkpoint Mamba & hybrid (nhỏ, ~1s)."""
    global _mamba
    if _mamba is None:
        with _mamba_lock:
            if _mamba is None:
                import sys as _sys, torch
                _sys.path.insert(0, str(project_root() / "scripts"))
                from gen_mamba_text import load_model
                from nn_aivie.core.tokenizer import BPETokenizer
                from nn_aivie.core.vocab import Vocabulary
                dev = "mps" if torch.backends.mps.is_available() else "cpu"
                vocab = Vocabulary(BPETokenizer.load(str(project_root() / "checkpoints/bpe_v1.json")))
                V = 32002
                models = {}
                for name, ck in [("hybrid", "checkpoints/hybrid_d256_2k.pt"),
                                 ("mamba", "checkpoints/mamba_d256_2k.pt")]:
                    p = project_root() / ck
                    if p.exists():
                        m, tag, _c = load_model(str(p), V, dev)
                        models[name] = (m, tag)
                _mamba = (vocab, models, dev, V)
    return _mamba


class MambaReq(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=200)
    model: str = Field("hybrid")
    max_len: int = Field(45, ge=5, le=80)
    temp: float = Field(0.8, ge=0.1, le=1.5)


@app.post("/api/mamba/gen")
def mamba_gen(req: MambaReq):
    """Sinh NỐI-TIẾP bằng Mamba/hybrid TỰ-XÂY (base LM đồ-chơi — mượt nhưng word-salad)."""
    vocab, models, dev, V = get_mamba()      # thêm scripts/ vào path TRƯỚC khi import gen
    from gen_mamba_text import gen
    if not models:
        raise HTTPException(status_code=404, detail="thiếu checkpoint mamba/hybrid")
    name = req.model if req.model in models else next(iter(models))
    m, tag = models[name]
    out = gen(m, vocab, req.prompt, V, dev, n=req.max_len, temp=req.temp)
    return {"prompt": req.prompt, "model": tag, "tail": out}


@app.get("/mamba", response_model=None)
def mamba_view():
    """§TK_093 — trang demo sinh-chữ Mamba/hybrid tự-xây."""
    if MAMBA_HTML.exists():
        return FileResponse(MAMBA_HTML)
    raise HTTPException(status_code=404, detail="mamba.html không tồn tại")


@app.get("/", response_model=None)
def index():
    """SPA index — React build hoặc fallback HTML."""
    react_index = STATIC_DIR / "index.html"
    if react_index.exists():
        return FileResponse(react_index)
    if FALLBACK_HTML.exists():
        return FileResponse(FALLBACK_HTML)
    return HTMLResponse(
        "<h1>AIVIE Graph UI</h1>"
        "<p>Chạy <code>cd viz/web && npm run build</code> hoặc thêm fallback.html.</p>"
        "<p>API: <a href='/api/health'>/api/health</a></p>",
        status_code=200,
    )


def _mount_static_assets() -> None:
    """Mount React assets sau API routes (Starlette: mount / cuối cùng)."""
    assets = STATIC_DIR / "assets"
    if assets.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")


_mount_static_assets()


@app.on_event("startup")
def _warm_checkpoint() -> None:
    """Nạp checkpoint + lazy index (skip spans, frame brackets, seq) lúc khởi động."""
    import threading

    def _work() -> None:
        try:
            st = get_state()
            from nn_aivie.hca.frame_bracket_gate import ensure_frame_brackets
            from nn_aivie.hca.skip_span_gate import ensure_skip_spans

            ensure_frame_brackets(st)
            ensure_skip_spans(st)
            if st.parents is not None:
                st.parents._ensure_seq_index()
            print("[warm] checkpoint + AR indices ready")
            # §197 — PRE-WARM đường sinh + encoder để request ĐẦU không chờ ~114s.
            # Encoder (torch) cho QA; một generate nhỏ dựng W_skip + agenda.
            get_encoder()
            get_knn()  # §TK_056 — nạp datastore 186MB lên MPS trước
            from nn_aivie.hca.config import HCAConfig
            from nn_aivie.hca.generate_hebbian import generate as _g, set_knn_hook
            if get_knn() is not None:
                set_knn_hook(get_knn(), lam=0.35)
            _g(st, "sắt và", max_len=4, config=HCAConfig())
            set_knn_hook(None)
            print("[warm] W_skip + encoder + kNN ready (đường sinh sẵn sàng)")
        except Exception as exc:
            print(f"[warm] failed: {exc}")

    threading.Thread(target=_work, daemon=True).start()


def main() -> None:
    import uvicorn

    ui = "React" if STATIC_DIR.exists() else "fallback HTML"
    print(f"AIVIE Graph UI → http://127.0.0.1:8765  ({ui})")
    if not STATIC_DIR.exists():
        print("  Tip: cd viz/web && npm run build  — để dùng force-graph đầy đủ")
    # reload=True (dev) làm loader ghi log/pyc trong project dir → uvicorn RESTART worker giữa lúc
    # load checkpoint LỚN (v6 142k parents) → load không bao giờ xong. Mặc định TẮT; bật khi sửa code.
    uvicorn.run(
        "viz.server.main:app",
        host="127.0.0.1",
        port=8765,
        reload=os.environ.get("AIVIE_RELOAD") == "1",
    )


if __name__ == "__main__":
    main()
