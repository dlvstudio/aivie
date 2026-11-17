#!/usr/bin/env python3
"""
Verify checkpoint sau train — thay đổi 2026-07-15/16 có THẬT SỰ vào checkpoint không?

Vì sao cần: cả ba thay đổi lớn đợt này đều HỎNG TRONG IM LẶNG nếu sót — không lỗi, không
cảnh báo, chỉ là kết quả kém đi mà không ai biết:
  - `_with_eos` sót đường `pipeline.py` → checkpoint không có cạnh →EOS → at_boundary mù
  - `resync`/`_is_match_token` sót → `_gdf['.']`=0 → gidf kịch trần → dấu chấm thành neo
  - varlm sidecar build TRƯỚC checkpoint → token-id lệch vocab → load OK nhưng điểm sai

Đã cắn thật 2 lần: (a) eval 50 phút quét `varlm=0/0.35/0.5` ra kết quả GIỐNG TỪNG BYTE vì
sidecar thiếu — báo cáo kết luận nhầm "varlm không tạo khác biệt"; (b) `_with_eos` thêm vào
`ingest.py` nhưng `train.py` đi lối `pipeline.py`, suýt phí 7.7 giờ bake.

Chạy:
  PYTHONPATH=src python -u scripts/verify_checkpoint.py checkpoints/network_v10.pkl

Exit 0 = tất cả PASS. Exit 1 = có FAIL (in rõ mục nào).
"""

from __future__ import annotations

import io
import contextlib
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.edge_kinds import EdgeKind  # noqa: E402
from nn_aivie.core.varlm import varlm_sidecar_path  # noqa: E402
from nn_aivie.core.vocab import SPECIAL_EOS  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca.generate_hebbian import _ends_unit  # noqa: E402

CLUSTER_OFFSET = 20_000_000
_t0 = time.time()
_fails: list[str] = []


def log(msg: str) -> None:
    print(f"[{time.time() - _t0:7.1f}s] {msg}", flush=True)


def check(ok: bool, name: str, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    log(f"  [{tag}] {name}: {detail}")
    if not ok:
        _fails.append(name)


def _verify_bpe(state, vocab, ps) -> None:
    """TK_041 §7.2 — bảy mục cho đáy BPE. Mỗi mục là một lớp lỗi ĐÃ BIẾT, không phải nghi lễ."""
    import json as _json
    from pathlib import Path as _P

    from nn_aivie.core.tokenizer import MARK_NL, MARK_SPACE, SPECIAL_UNK, BPETokenizer

    tok = vocab.tokenizer

    # 11. Roundtrip vô-tổn-hao trên corpus THẬT (không phải mẫu đồ chơi trong test).
    sample: list[str] = []
    for name in ("grade_lessons.jsonl", "history_articles.jsonl", "qa_supervised.jsonl"):
        p = _ROOT / "data/text_corpus" / name
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    o = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                t = o if isinstance(o, str) else (o.get("text") if isinstance(o, dict) else None)
                if t:
                    sample.append(t)
                if len(sample) >= 1000:
                    break
    from nn_aivie.core.tokenizer import normalize as _nfc

    bad = [t for t in sample if tok.decode(tok.encode(t)) != _nfc(t)]
    check(
        not bad,
        "roundtrip byte-exact",
        f"{len(sample)-len(bad)}/{len(sample)} đơn vị — hỏng = tokenizer ĐANG ĐỐT dữ liệu",
    )

    # 12. Merge table trên đĩa khớp cái checkpoint mang theo. Lệch = mọi id trỏ sai token,
    #     KHÔNG crash, chỉ ra rác lặng lẽ — đúng lớp lỗi phải chết ồn ào.
    art = _ROOT / "checkpoints/bpe_v1.json"
    if art.exists():
        disk = BPETokenizer.load(art).fingerprint()
        check(
            disk == tok.fingerprint(),
            "tokenizer khớp artifact",
            f"checkpoint={tok.fingerprint()} bpe_v1.json={disk}",
        )

    # 13. Vocab bề mặt ĐÓNG: không id bề mặt nào lọt lên dải cấu trúc.
    n_surface = sum(1 for i in range(len(vocab)) if vocab.is_surface(i))
    check(
        n_surface == len(tok) - 2,  # trừ <UNK>, <EOS>
        "vocab bề mặt đóng",
        f"{n_surface:,} bề mặt / {len(tok):,} tokenizer / {len(vocab):,} tổng "
        f"({len(vocab)-len(tok):,} nơ-ron cấu trúc)",
    )

    # 14. <UNK> chết — byte-fallback khiến nó BẤT KHẢ. Còn thì tokenizer có đường rò.
    unk = vocab._token_to_id.get(SPECIAL_UNK)
    check(
        unk is None or ps._gdf.get(unk, 0) == 0,
        "<UNK> không xuất hiện",
        f"gdf={ps._gdf.get(unk, 0)} — byte-fallback ⇒ phải = 0",
    )

    # 15. Không token nào trộn chữ số với non-digit (§3.2 — cưỡng chế ở pre-token).
    #
    # `isdecimal`, KHÔNG `isdigit` — bản đầu dùng `isdigit` và mục này TRƯỢT với
    # ['Ġkm²','Ġm³','m²'] : `'²'.isdigit()` = True nhưng regex `\d` (= Nd) KHÔNG khớp nó.
    # Tokenizer đúng, check sai: `²` là ký hiệu đơn vị (km², m³), không phải chữ số tham gia
    # phép toán — gộp `km²` là hành vi MONG MUỐN. Check phải dùng đúng khái niệm "chữ số"
    # mà pre-token regex dùng, nếu không nó bắt oan chính cái nó bảo vệ.
    mixed = [
        t
        for t in tok.tokens
        if not t.startswith("<0x")
        and t not in tok.specials
        and any(c.isdecimal() for c in t)
        and any(not c.isdecimal() and c != MARK_SPACE for c in t)
    ]
    check(
        not mixed,
        "chữ số không dính chữ",
        f"{len(mixed)} token trộn{': ' + str(mixed[:5]) if mixed else ''} — "
        f"`1287`/`1288` phải CHIA nơ-ron chữ số",
    )

    # 16. Khoảng trắng + xuống dòng THẬT SỰ vào mạng (không phải chỉ có trong tokenizer).
    nl = [i for t, i in vocab._token_to_id.items() if MARK_NL in t]
    sp = [i for t, i in vocab._token_to_id.items() if t.startswith(MARK_SPACE)]
    nl_df = sum(ps._gdf.get(i, 0) for i in nl)
    sp_df = sum(ps._gdf.get(i, 0) for i in sp)
    check(
        sp_df > 0,
        "khoảng trắng là token",
        f"{len(sp):,} loại Ġ*, tổng gdf={sp_df:,}",
    )
    check(
        nl_df > 0,
        "xuống dòng là token",
        f"{len(nl):,} loại chứa Ċ, tổng gdf={nl_df:,} — =0 nghĩa là `\\n` vẫn bị "
        f"`normalize_ws` đốt lúc nạp (dùng `normalize_ws_keep_newlines`)",
    )


def main() -> int:
    ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/network_v10.pkl")
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    mb = ckpt.stat().st_size / 1024 / 1024 if ckpt.exists() else 0
    log(f"load {ckpt.name} ({mb:,.0f} MB) … — checkpoint lớn mất 5–15 phút "
        f"(resync_match_ids quét MỌI cha lúc load)")
    with contextlib.redirect_stdout(io.StringIO()):
        state, _ = load_checkpoint(ckpt)
    ps, vocab, cfg = state.parents, state.vocab, HCAConfig()
    eos = vocab.id_of(SPECIAL_EOS)
    log(f"loaded: cha={len(ps.parents):,} vocab={len(vocab):,}")

    # ---- 1. EOS nối vào cuối mỗi đơn vị train (ingest._with_eos + pipeline) ----
    sample = ps.parents[:2000]
    n_eos = sum(1 for r in sample if r.source_ids and r.source_ids[-1] == eos)
    check(
        n_eos >= 0.95 * len(sample),
        "EOS cuối source_ids",
        f"{n_eos}/{len(sample)} cha — nếu ~0 thì `_with_eos` KHÔNG chạy trên đường train này",
    )

    # ---- 2. EOS KHÔNG được vào match_ids (nếu vào → mọi cha đều có nó → nhiễu retrieval) ----
    n_bad = sum(1 for r in sample if eos in r.match_ids)
    check(n_bad == 0, "EOS ngoài match_ids", f"{n_bad}/{len(sample)} cha có EOS trong match_ids")

    # ---- 3. Dấu câu ĐƯỢC ĐẾM vào _gdf → IDF tự dìm nó ----
    dot = vocab._token_to_id.get(".")
    if dot is None:
        check(False, "vocab có token '.'", "KHÔNG — checkpoint tiền-R0? (tokenize không tách)")
    else:
        gdf_dot, gidf_dot = ps._gdf.get(dot, 0), ps._gidf(dot)
        check(
            gdf_dot > 0 and gidf_dot < 2.0,
            "dấu câu vào thống kê",
            f"'.' gdf={gdf_dot:,} gidf={gidf_dot:.3f} — gidf>7 nghĩa là dấu câu VẮNG khỏi "
            f"_gdf → nó thành token phân biệt NHẤT corpus (bug)",
        )
        for t in ("Việt", "là"):
            i = vocab._token_to_id.get(t)
            if i is not None:
                log(f"         (đối chiếu {t!r}: gdf={ps._gdf.get(i,0):,} gidf={ps._gidf(i):.3f})")

    # ---- 4. Mốc ranh giới HỌC ĐƯỢC từ EOS (at_boundary — thay is_punct) ----
    seq = state.adjacency.layer(EdgeKind.SEQUENTIAL)

    from nn_aivie.hca.generate_hebbian import _eos_base_rate  # noqa: E402

    _base = _eos_base_rate(state, seq, eos)
    log(f"  P(EOS) nền = {_base:.5f} (≈ số đơn vị / tổng token — lift chia cho số này)")

    def _lift(i: int) -> float:
        row = seq.row(i)
        tot = sum(row.values())
        if not tot or _base <= 0:
            return 0.0
        return (row.get(eos, 0.0) / tot) / _base

    def p_eos(tok: str) -> float | None:
        i = vocab._token_to_id.get(tok)
        if i is None:
            return None
        row = seq.row(i)
        tot = sum(row.values())
        return (row.get(eos, 0.0) / tot) if tot else 0.0

    if dot is not None:
        ok_dot = _ends_unit(state, dot, cfg)
        bad_content = [
            t for t in ("là", "của", "và", "Việt")
            if (i := vocab._token_to_id.get(t)) is not None and _ends_unit(state, i, cfg)
        ]
        check(
            ok_dot and not bad_content,
            "mốc kết học từ EOS",
            f"'.' P(→EOS)={p_eos('.'):.3f} lift={_lift(dot):.1f} (ngưỡng {cfg.ar_eos_boundary_min_lift}) ends_unit={ok_dot}; "
            f"từ nội dung nhận nhầm làm mốc: {bad_content or 'không'}",
        )

    # ---- 5. Bake: FRAME_BRACKET + endpoints + skip spans ----
    fb = len(state.adjacency.layer(EdgeKind.FRAME_BRACKET))
    meta = len(getattr(ps, "hlink_meta", None) or {})
    eps = len(getattr(ps, "hlink_endpoints", None) or {})
    spans = len(getattr(ps, "skip_spans", None) or [])
    check(
        fb > 0 and meta > 0 and eps > 0,
        "bake FRAME_BRACKET",
        f"edges={fb:,} meta={meta:,} endpoints={eps:,} — endpoints=0 nghĩa là checkpoint "
        f"KẸT (bake báo skipped, runtime bị guard chặn) → chạy lại với --bake",
    )
    check(spans > 0, "bake skip_spans", f"{spans:,}")

    # ---- 6. VarLM sidecar khớp vocab của CHÍNH checkpoint này ----
    vpath = varlm_sidecar_path(ckpt)
    v = getattr(state, "varlm", None)
    if v is None:
        check(
            False,
            "varlm sidecar",
            f"KHÔNG nạp — runtime tìm {vpath.name}. Thiếu = no-op IM LẶNG "
            f"(`else: state.varlm = None`, không log). Build: scripts/build_varlm.py",
        )
    else:
        log(f"  varlm: max_ctx={v.max_ctx} lam={v.lam} use_clusters={v.use_clusters} "
            f"n_tokens={v.n_tokens:,}")
        log(f"         ctx mỗi bậc: {v.stats()}")
        n_out = 0
        n_eos_ctx = 0
        for tab in v.tables:
            for ctx, nxt in tab.items():
                if eos in nxt:
                    n_eos_ctx += 1
                for t_ in (*ctx, *nxt.keys()):
                    if t_ < CLUSTER_OFFSET and t_ >= len(vocab):
                        n_out += 1
        check(
            n_out == 0,
            "varlm khớp vocab",
            f"{n_out:,} token-id NGOÀI vocab — sidecar build từ checkpoint KHÁC → load OK "
            f"nhưng điểm SAI (im lặng). Build lại SAU khi train xong.",
        )
        check(
            n_eos_ctx > 0,
            "varlm học mốc kết",
            f"{n_eos_ctx:,} ctx có EOS làm token kế",
        )
        if v.use_clusters:
            n_cl = sum(
                1 for tab in v.tables for ctx in tab for t_ in ctx if t_ >= CLUSTER_OFFSET
            )
            check(n_cl > 0, "varlm cluster-aware", f"{n_cl:,} ctx-element là cluster id")

    # ---- 11–17. TK_041 §7.2 — đáy BPE (bỏ qua trên checkpoint legacy) ----
    if not vocab.is_bpe:
        log("  (checkpoint LEGACY — bỏ qua mục TK_041; vocab .split()+FOLD_STRIP)")
    else:
        _verify_bpe(state, vocab, ps)

    log("")
    if _fails:
        log(f"❌ {len(_fails)} MỤC TRƯỢT: {', '.join(_fails)}")
        return 1
    log("✅ TẤT CẢ PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
