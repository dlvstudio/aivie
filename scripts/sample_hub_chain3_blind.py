#!/usr/bin/env python3
"""Mẫu CHẤM MÙ cho chuỗi BA BƯỚC đầy đủ (thiết kế của tác giả).

Chuỗi đúng — điểm xuất phát là THUỘC TÍNH, không phải một từ đơn lẻ::

    ① nghịch ngợm → {chó, mèo}        thuộc tính CHỌN RA tập hạt giống
    ② {chó, mèo}  → động vật          tập → nhãn CHUNG (typed, cụm ISG)
    ③ động vật    → {chó, mèo, KHỈ}   nhãn → thành viên, `khỉ` là MỚI

⚠️ Vòng đo trước (`sample_hub_step3_blind.py`) **bỏ mất bước ①**: nó xuất phát
từ một seed đơn lẻ, nên không có ràng buộc thuộc tính nào giữ tập hạt giống
mạch lạc. Kết quả 10,7% của vòng đó KHÔNG kiểm định thiết kế này.

Câu hỏi chấm khác hẳn vòng trước — không phải "hub có phải nhãn không" mà:

    **Phần tử MỚI ở ③ có thật sự mang thuộc tính ① không?**

Đó mới là định nghĩa "sáng tạo" trong ví dụ gốc: `khỉ` chưa từng đứng cạnh
`nghịch ngợm`, nhưng suy ra được và ĐÚNG.

    PYTHONPATH=src python scripts/sample_hub_chain3_blind.py \
      --checkpoint checkpoints/network_v12_subst_isg_pw.pkl
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.hca.hub_spread import (  # noqa: E402
    _FUNC,
    _gidf,
    _surf,
    _surf_ix,
    build_hub_spread_index,
    load_ws_spans,
    reverse_peers,
    score_hubs_from,
)

T0 = time.monotonic()
_LOG: str | None = None


def log(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824
    line = f"[{time.monotonic() - T0:7.1f}s RSS {rss:5.2f}GB] {msg}"
    print(line, flush=True)
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def main() -> int:
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/network_v12_subst_isg_pw.pkl")
    ap.add_argument("--n", type=int, default=30, help="số THUỘC TÍNH mỗi nhánh")
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--n-hat-giong", type=int, default=6, help="|S| ở bước ①")
    ap.add_argument("--n-moi", type=int, default=5, help="số phần tử mới in ra")
    ap.add_argument("--prop-gidf-min", type=float, default=4.0)
    # ① thuộc tính → hạt giống cần TẦM XA hơn ②③: trong «nghịch ngợm như loài
    # chó» thì `chó` ở d=4, d_max=3 cắt mất nó (đo được trên ví dụ tác giả).
    ap.add_argument("--d-max", type=int, default=6)
    # MẶC ĐỊNH 0: thứ MANG thuộc tính hoàn toàn có thể là từ phổ biến (`chó`).
    # Lọc hạt giống bằng gidf là sai nguyên tắc — nhiễu do bỏ phiếu ở ② chặn.
    ap.add_argument("--hat-gidf-min", type=float, default=0.0)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--out", default="out/hub3_chain_samples.txt")
    ap.add_argument("--out-key", default="out/hub3_chain_key.json")
    ap.add_argument("--out-meta", default="out/hub3_chain_meta.json")
    ap.add_argument("--log", default="")
    # Sidecar span TỪ (WS-CRF) — mặc định tự tìm cạnh checkpoint.
    # ① LỌC THUỘC TÍNH bằng từ điển WS-CRF (26 244 mục). Đo được: 31/40 thuộc
    # tính lấy mẫu trên classrich là MẢNH BPE của tên ngoại lai (`imb`←Rimbaud,
    # `Es`←Essonne, `Klux`, `Westberg`…). Với thuộc tính vô nghĩa thì KHÔNG có
    # câu trả lời đúng nào tồn tại ⇒ dòng tự động `hong` bất kể cơ chế.
    # Đây là chọn KHUNG LẤY MẪU của phép đo (đo trên từ tiếng Việt thật),
    # không phải luật của cơ chế.
    # Bộ thuộc tính CHỌN TAY — bỏ qua lấy mẫu ngẫu nhiên. Vì sao cần: sau khi
    # lọc mảnh BPE, thuộc tính lấy mẫu vẫn phần lớn là DANH TỪ (`Sông`, `Quân`,
    # `tim`), mà câu hỏi chấm «từ MỚI có mang thuộc tính «Sông» không» là VÔ
    # NGHĨA ⇒ dòng tự động `hong` bất kể cơ chế. Thiết kế gốc giả định thuộc
    # tính kiểu `nghịch ngợm` (TÍNH TỪ).
    # ① QUAN HỆ BỔ NGHĨA thay ĐỒNG HIỆN. Đo được (bộ 48 tính từ chọn tay):
    #   «mỏng» → quyển, hơn, dịu, dàng   ← `mỏng manh dịu dàng`, không phải thứ MỎNG
    #   «mát»  → sản, phổ, Đức, chùm     ← không có gì mát
    #   «khô»  → mưa, sa, mạc, Gobi      ← ✅ chỉ ca này đúng, vì `sa mạc` ĐỨNG NGAY TRƯỚC
    # Tiếng Việt đặt tính từ SAU danh từ (`sa mạc khô`, `nước sạch`) ⇒ thứ MANG
    # thuộc tính là danh từ LIỀN TRƯỚC (d=1), không phải mọi thứ trong d<=6.
    ap.add_argument("--cho-ngan-mach", action="store_true",
                    help="cho phép nhãn == thuộc tính (hành vi CŨ, để A/B)")
    ap.add_argument("--seed-mode", choices=("bonghia", "cooc"), default="bonghia",
                    help="bonghia = danh từ liền trước (d=1) · cooc = đồng hiện d<=6 (cũ)")
    ap.add_argument("--props", default="",
                    help="danh sách thuộc tính cách nhau bởi dấu phẩy (bỏ qua lấy mẫu)")
    ap.add_argument("--prop-dict", default="out/models/ws_crf_uts_full/dictionary_lower.txt",
                    help="lọc thuộc tính theo từ điển; 'none' = tắt")
    ap.add_argument("--ws-spans", default="auto",
                    help="auto | <đường dẫn .ws_spans.json> | none")
    args = ap.parse_args()
    _LOG = args.log or "out/hub3_chain.log"
    Path(_LOG).parent.mkdir(parents=True, exist_ok=True)

    gc.disable()
    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()
    v = state.vocab
    log(f"nạp xong · vocab={len(v)} parents={len(state.parents)}")

    wsp = None
    if args.ws_spans != "none":
        sc = (args.ws_spans if args.ws_spans != "auto"
              else str(Path(args.checkpoint).with_suffix("")) + ".ws_spans.json")
        wsp = load_ws_spans(_ROOT / sc)
        if wsp is None:
            raise SystemExit(
                f"⛔ thiếu sidecar {sc}. Bake trước: "
                f"scripts/bake_ws_spans.py --checkpoint {args.checkpoint}\n"
                f"   (dùng --ws-spans none để chạy KHÔNG có span WS-CRF — đối chứng)")
        log(f"sidecar WS-CRF: {len(wsp)} span TỪ ({sc})")
    else:
        log("⚠️ CHẠY KHÔNG CÓ span WS-CRF (đối chứng, neo bằng cụm ISG)")

    idx = build_hub_spread_index(
        state, d_max=args.d_max, min_count=args.min_count, ws_spans=wsp)
    log(f"index xong · cạnh typed={sum(len(d) for d in idx.via_pivot.values())}"
        f" · nhãn CỤM={len(idx.label_surf)}")

    # Ứng viên THUỘC TÍNH: token nội dung, gidf đủ cao, có ≥3 láng giềng.
    # Token cùng CỤM với thuộc tính (vd `ngợm` ⊂ `nghịch ngợm`) — không được
    # làm hạt giống: đó là nửa kia của chính từ, không phải thứ MANG thuộc tính.
    cung_cum: dict[int, set[int]] = {}
    _seg = getattr(state.parents, "seg", None)
    if _seg is not None and getattr(_seg, "clusters", None):
        for c in _seg.clusters:
            sp = tuple(int(t) for t in c.token_span())
            for t in sp:
                cung_cum.setdefault(t, set()).update(sp)

    tu_dien: set[str] = set()
    if args.prop_dict != "none":
        dp = _ROOT / args.prop_dict
        if not dp.exists():
            raise SystemExit(f"⛔ thiếu từ điển {dp} (dùng --prop-dict none để tắt)")
        tu_dien = {w.strip().lower() for w in dp.read_text("utf-8").splitlines() if w.strip()}
        log(f"lọc thuộc tính theo từ điển: {len(tu_dien)} mục")
    props = [
        t for t, row in idx.fwd.items()
        if len(row) >= 3 and _gidf(state, t) >= args.prop_gidf_min
        and (s := _surf(v, t)) and len(s) > 1
        and s not in _FUNC and s.lower() not in _FUNC
        and (not tu_dien or s.lower() in tu_dien)
    ]
    rng = random.Random(args.seed)
    if args.props.strip():
        want = [w.strip() for w in args.props.split(",") if w.strip()]
        by_surf: dict[str, list[int]] = {}
        for t in idx.fwd:
            by_surf.setdefault(_surf(v, t), []).append(int(t))
        props, thieu = [], []
        for w in want:
            ids = by_surf.get(w) or by_surf.get(w.capitalize()) or []
            # chọn id có NHIỀU láng giềng nhất (biến thể Ġ vs không)
            ids = sorted(ids, key=lambda t: -len(idx.fwd.get(t, {})))
            if ids and len(idx.fwd.get(ids[0], {})) >= 3:
                props.append(ids[0])
            else:
                thieu.append(w)
        log(f"CHỌN TAY: {len(props)}/{len(want)} thuộc tính có đủ dữ liệu")
        if thieu:
            log(f"  thiếu (bỏ qua): {thieu}")
    else:
        rng.shuffle(props)
        props = props[: args.n]
    log(f"{len(props)} thuộc tính đóng băng")

    rows: list[dict] = []
    for ch in ("typed_first", "cooc"):
        for p in props:
            # ① thuộc tính → tập hạt giống
            if args.seed_mode == "bonghia":
                # danh từ LIỀN TRƯỚC tính từ (d=1) — quan hệ bổ nghĩa
                nb = sorted(
                    ((t, wd) for t, wd in idx.rev.get(p, {}).items() if wd[1] == 1),
                    key=lambda kv: -kv[1][0],
                )
            else:
                nb = sorted(idx.fwd.get(p, {}).items(), key=lambda kv: -kv[1][0])
            hat = []
            ban = cung_cum.get(p, set()) | {p}
            for t, _w in nb:
                s = _surf(v, t)
                if not s or len(s) <= 1 or s in _FUNC or s.lower() in _FUNC:
                    continue
                if int(t) in ban:          # nửa kia của chính từ
                    continue
                if _gidf(state, int(t)) < args.hat_gidf_min:
                    continue
                # neo bằng span TỪ: `mạc` là đuôi của `sa mạc` ⇒ dùng cả cụm
                tt = int(t)
                if wsp:
                    for sp in wsp:
                        if len(sp) >= 2 and sp[-1] == tt:
                            tt = sp[-1]     # giữ id token, ghi cụm vào nhãn hiển thị
                            break
                hat.append(tt)
                if len(hat) >= args.n_hat_giong:
                    break
            if len(hat) < 2:
                continue
            # ② tập hạt giống → nhãn CHUNG (bỏ phiếu, không lấy 1 seed)
            phieu: dict[int, float] = {}
            for t in hat:
                for h in score_hubs_from(
                    state, idx, [t], topk=3, gidf_min=0.0,
                    ban_func=True, channel=ch,
                ):
                    phieu[h.tid] = phieu.get(h.tid, 0.0) + 1.0
            if not phieu:
                continue
            # 🔴 CHẶN NGẮN MẠCH: nhãn KHÔNG được là chính thuộc tính, cũng
            # không được là một hạt giống. Đo được (vòng bổ-nghĩa): 22/32 dòng
            # có nhãn == thuộc tính, và 16/16 ca "không hỏng" đều là ca đó —
            # chuỗi sập từ BA bước xuống HAI:
            #   thiết kế: thuộc tính → thành viên → LỚP → thành viên MỚI
            #   thực tế : thuộc tính → thành viên → thuộc tính → thứ khác
            # Không có trừu tượng hoá nào; thước cũ thưởng cho việc BỎ bước ②.
            if not args.cho_ngan_mach:
                cam = set(hat) | {int(p)}
                if wsp:
                    for sp in wsp:
                        if int(p) in sp:
                            cam.update(int(x) for x in sp)
                phieu = {k: w for k, w in phieu.items() if int(k) not in cam}
                if not phieu:
                    continue
            hub = max(phieu.items(), key=lambda kv: (kv[1], kv[0]))
            if hub[1] < 2:      # nhãn phải được ≥2 hạt giống cùng bầu
                continue
            hub_id = hub[0]
            # ③ nhãn → thành viên; MỚI = không có trong tập hạt giống
            peers = reverse_peers(
                state, idx, [hub_id], exclude=set(hat), topk=30, channel=ch)
            moi = [p_ for p_ in peers if p_.tid not in set(hat)][: args.n_moi]
            if not moi:
                continue
            rows.append({
                "nhanh": ch,
                "thuoc_tinh": _surf(v, p),
                "hat_giong": [_surf(v, t) for t in hat],
                "nhan": _surf_ix(idx, v, hub_id),
                "n_bau": int(hub[1]),
                "moi": [p_.surface for p_ in moi],
            })
    rng.shuffle(rows)

    key, meta, lines = {}, {}, [
        "# CHẤM MÙ — chuỗi BA BƯỚC. Câu hỏi DUY NHẤT cho mỗi dòng:",
        "#   Các từ ở [MỚI] có mang THUỘC TÍNH đầu dòng không?",
        "#   dung = phần lớn từ MỚI mang thuộc tính đó",
        "#   tam  = ít nhất một từ MỚI mang, số còn lại không sai trắng trợn",
        "#   hong = không từ MỚI nào mang thuộc tính đó",
        "# (nhãn giữa dòng chỉ để tham khảo, KHÔNG chấm nhãn)",
        "",
    ]
    for i, r in enumerate(rows):
        rid = f"C{i:03d}"
        key[rid] = r["nhanh"]
        meta[rid] = r
        lines.append(
            f"[{rid}]  thuộc tính «{r['thuoc_tinh']}»"
            f"  ·  hạt giống: {', '.join(r['hat_giong'])}"
            f"  ·  nhãn: {r['nhan']} ({r['n_bau']} phiếu)"
            f"  ·  MỚI: {', '.join(r['moi'])}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_key).write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    # ⚠️ meta CHỨA nhãn nhánh → cùng phía với key, không mở khi chấm.
    Path(args.out_meta).write_text(
        json.dumps({"checkpoint": args.checkpoint, "n_prop": len(props),
                    "n_rows": len(rows), "rows": meta},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{len(rows)} dòng → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
