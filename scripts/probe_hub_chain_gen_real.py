"""TK_046 §22 trên DỮ LIỆU THẬT — chuỗi có nâng được thành viên MỚI trong phân bố sinh?

§22 đo trên ckpt tí hon (5–3 069 cha) và thấy `khỉ` h1560 → h22. Memory
`long-range-cooccurrence-negative` đã ghi: ckpt nhỏ từng đánh lừa NGƯỢC DẤU.
Script này chạy đúng phép đo đó trên `classrich` (35 757 cha).

Ca đo lấy từ chính dữ liệu, không bịa: đào lớp `X là C`, rồi với mỗi thuộc tính
`P` chạm ≥2 thành viên, tập «con khỉ» = thành viên của `C` **chưa từng đồng hiện
với P**. Thước: hạng của các «con khỉ» đó trong phân bố token kế, tắt vs bật.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/probe_hub_chain_gen_real.py \
        --checkpoint checkpoints/classrich.pkl --n 25
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

import numpy as np  # noqa: E402

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import generate_hebbian as G  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402
from nn_aivie.hca import generate_hebbian as G2  # noqa: E402,F401

_DET = frozenset({"một", "các", "những", "mọi", "mỗi"})
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:9.1f}s] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="out/hub_chain_gen_real.json")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--boost", type=float, default=4.0)
    ap.add_argument("--ke-ctx", type=float, default=-1.0,
                    help="KỀ ngữ cảnh ở ③ (§60); <0 = giữ mặc định HCAConfig")
    ap.add_argument("--prompt-cau", action="store_true",
                    help="prompt = TIỀN TỐ CÂU THẬT chứa P (nhiều từ nội dung) "
                         "thay vì mỗi từ P. Đồng thuận cấp thuộc tính (§36) chỉ "
                         "kích hoạt được khi ngữ cảnh có >1 thuộc tính.")
    ap.add_argument("--chan-ly", default="", help="JSON build_chan_ly_v2 (TK_046 §31)")
    ap.add_argument("--cases-rong", type=int, default=0,
                    help="dựng bộ ca theo §26: MẪU RỘNG token nội dung df∈[5,2000], "
                         "k=10 láng giềng, giữ ca có ≥2 láng giềng cùng lớp. "
                         "Bộ ca §20 đã bị §26 bác vì thiên về token hiếm.")
    ap.add_argument("--head-gidf-q", type=float, default=-1.0,
                    help="§64: ngưỡng «đủ đặc thù» cho luật CHỌN ĐẦU, tách "
                         "khỏi bộ lọc cạnh; <0 = giữ mặc định HCAConfig")
    ap.add_argument("--ws-spans", default="")
    ap.add_argument("--p-tu-dien", default="",
                    help="từ điển (jsonl) — buộc THUỘC TÍNH P và HEAD lớp phải "
                         "là TỪ THẬT. Không có nó, bộ ca đầy mảnh BPE của tên "
                         "riêng (`Le`←Lecomte · `St` · `Tr`) và thước chấm cơ "
                         "chế trên rác. TK_046 §60.")
    ap.add_argument("--cases-from", default="",
                    help="JSON của probe_chain3_known_labels — tách HAI NHÁNH: "
                         "ca mà bước ② ĐÃ khớp lớp chung vs ca chưa khớp. "
                         "Phân định «thước sinh phẳng vì ② thưa» khỏi «vì chọn ca kém».")
    args = ap.parse_args()

    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(args.checkpoint)
    v, ps = state.vocab, state.parents
    n = len(v)
    raw = [str(v.token_of(i)) for i in range(n)]
    S = [r.replace(MARK_SPACE, "").strip() for r in raw]
    BAD = {i for i in range(n) if re.match(r"^<.*>$", S[i]) or S[i].endswith(":")}
    LA = {i for i in range(n) if S[i] == "là"}
    DET = {i for i in range(n) if S[i].lower() in _DET}
    FUNC = {i for i in range(n)
            if S[i].lower() in H._FUNC or len(S[i]) <= 1
            or MARK_SPACE not in raw[i]} | BAD

    log("đào lớp + bảng đồng hiện …")
    cls: dict[str, set[str]] = defaultdict(set)
    tok_par: dict[str, set[int]] = defaultdict(set)
    par_tok: list[frozenset[str]] = []
    for pi, rec in enumerate(ps.parents):
        seq = list(rec.emit_ids)
        ts = {S[t] for t in seq if t not in FUNC}
        par_tok.append(frozenset(ts))
        for s_ in ts:
            tok_par[s_].add(pi)
        for k, t in enumerate(seq):
            if t not in LA or k == 0 or k + 1 >= len(seq):
                continue
            mem = seq[k - 1]
            if mem in FUNC or mem in DET:
                continue
            j = k + 1
            while j < len(seq) and seq[j] in DET:
                j += 1
            if j >= len(seq) or seq[j] in FUNC:
                continue
            head = S[seq[j]]
            if j + 1 < len(seq) and seq[j + 1] not in FUNC:
                head += " " + S[seq[j + 1]]
            cls[head].add(S[mem])
    if args.chan_ly:
        big = {c: set(m) for c, m in
               json.loads(Path(args.chan_ly).read_text())["lop"].items()}
        log(f"  CHÂN LÝ v2: {len(big)} lớp")
    else:
        big = {c: m for c, m in cls.items() if len(m) >= 3}
    log(f"  lớp ≥3 thành viên = {len(big)}")

    if args.cases_rong:
        log(f"dựng bộ ca §26 (mẫu rộng, n={args.cases_rong}) …")
        wsp = H.load_ws_spans(args.ws_spans) if args.ws_spans else None
        idx = H.build_hub_spread_index(
            state, d_max=6, min_count=0.5, head_rule="trai_du_dac_thu",
            head_rule_trai="phai_du_dac_thu", label_indeg_q=0.8, ws_spans=wsp)
        dfc: dict[str, int] = defaultdict(int)
        for pt in par_tok:
            for s_ in pt:
                dfc[s_] += 1
        m2c: dict[str, set[str]] = defaultdict(set)
        for c, ms in big.items():
            for m in ms:
                m2c[m].add(c)
        ung = sorted((c, s_) for s_, c in dfc.items() if 5 <= c <= 2000)
        ung.sort(key=lambda x: -x[0])
        cases = []
        for _c, P in ung:
            if len(cases) >= args.n:
                break
            ids = H.resolve_surface_ids(v, P)
            if not ids:
                continue
            s1 = H.score_hubs_from(state, idx, ids[:4], topk=40,
                                   gidf_min=0.0, channel="typed_first")
            lang = [h.surface for h in s1 if h.surface != P][:10]
            votes: dict[str, int] = defaultdict(int)
            for x in lang:
                for c2 in m2c.get(x, ()):
                    votes[c2] += 1
            chung = [c2 for c2, z in votes.items() if z >= 2]
            if not chung:
                continue
            lop = chung[0]
            gan = set()
            for pi in list(tok_par.get(P, ()))[:300]:
                gan |= par_tok[pi]
            khi = sorted(big[lop] - gan - set(lang))
            if not khi:
                continue
            cases.append({"P": P, "lop": lop, "hat": lang[:6], "khi": khi[:8],
                          "n_tv": len(big[lop]), "nhanh": "§26"})
        log(f"  ca = {len(cases)}")
    elif args.cases_from:
        log(f"lấy ca từ {args.cases_from} — tách hai nhánh …")
        src = json.loads(Path(args.cases_from).read_text())
        cases = []
        for r in src["ca"]:
            khop = r.get("②_khop_lop_chung") or []
            lop = khop[0] if khop else r["lop"]
            members = big.get(lop) or big.get(lop.split()[0]) or set()
            if not members:
                continue
            gan = set()
            for pi in list(tok_par.get(r["P"], ()))[:300]:
                gan |= par_tok[pi]
            khi = sorted(members - gan - set(r.get("①_ra") or []))
            if not khi:
                continue
            cases.append({"P": r["P"], "lop": lop, "hat": (r.get("①_ra") or [])[:6],
                          "khi": khi[:8], "n_tv": len(members),
                          "nhanh": "②_KHỚP" if khop else "②_chưa"})
        log(f"  ca = {len(cases)}  (khớp={sum(1 for c in cases if c['nhanh']=='②_KHỚP')})")
    else:
      log("chọn ca §20 (P chạm ≥2 nhưng ≤½ thành viên) …")
      cases = []
      per: dict[str, int] = defaultdict(int)
      for c, members in sorted(big.items(), key=lambda kv: -len(kv[1])):
          hit: dict[str, set[str]] = defaultdict(set)
          for m in members:
              seen: set[str] = set()
              for pi in list(tok_par.get(m, ()))[:300]:
                  seen |= par_tok[pi]
              seen -= {m, c}
              for p in seen:
                  hit[p].add(m)
          for p, touched in hit.items():
              if p in members or len(touched) < 2 or len(touched) > 0.5 * len(members):
                  continue
              khi = sorted(members - touched)
              if not khi or per[c] >= 2:
                  continue
              per[c] += 1
              cases.append({"P": p, "lop": c, "hat": sorted(touched)[:6],
                            "khi": khi[:8], "n_tv": len(members)})
              if len(cases) >= args.n:
                  break
          if len(cases) >= args.n:
              break
    if args.p_tu_dien:
        pm_ = H.load_pos_map(args.p_tu_dien)
        tu = {k.replace(MARK_SPACE, "").strip() for k in pm_} if pm_ else set()
        log(f"  từ điển = {len(tu)} từ")
        truoc = len(cases)
        cases = [c for c in cases
                 if c["P"] in tu and c["lop"].split()[0] in tu]
        log(f"  lọc TỪ THẬT: {truoc} → {len(cases)} ca")
    log(f"  ca đem đo = {len(cases)}")

    tien_to: dict[str, str] = {}
    if args.prompt_cau:
        log("dựng tiền tố câu cho từng P …")
        can = {c["P"] for c in cases}
        for rec in ps.parents:
            if not can:
                break
            seq = list(rec.emit_ids)
            surf = [S[t] for t in seq]
            for k, w in enumerate(surf):
                if w in can and 3 <= k <= 14:
                    txt = " ".join(x for x in surf[:k + 1]
                                   if any(ch.isalnum() for ch in x))
                    if len(txt.split()) >= 4:
                        tien_to[w] = txt
                        can.discard(w)
                        break
        log(f"  có tiền tố cho {len(tien_to)}/{len(cases)} ca")

    # bắt vector điểm qua trace_out (không sửa src)
    BAT: dict = {}
    _orig = G._top_scores_ui

    def _spy(sc, vocab, *, top_k=8):
        BAT.setdefault("s", np.array(sc, copy=True))
        return _orig(sc, vocab, top_k=top_k)

    def diem(prompt: str, boost: float):
        BAT.clear()
        G._top_scores_ui = _spy
        cfg = HCAConfig()
        cfg.hub_chain_boost = boost
        if args.head_gidf_q >= 0:
            cfg.hub_chain_head_gidf_q = args.head_gidf_q
        if args.ke_ctx >= 0:
            cfg.hub_chain_ke_ctx = args.ke_ctx
        try:
            G.generate_autoregressive(
                state, prompt, max_len=1, config=cfg, seed=0, trace_out=[])
        except Exception as e:  # noqa: BLE001
            log(f"     ✗ sinh lỗi: {type(e).__name__}: {e}")
        finally:
            G._top_scores_ui = _orig
        return BAT.get("s")

    def hang(sc, surfaces: list[str]) -> list[int]:
        if sc is None:
            return []
        fin = np.where(np.isfinite(sc) & (sc > 0), sc, -np.inf)
        order = np.argsort(-fin)
        rk = {int(i): r for r, i in enumerate(order, 1)}
        out = []
        for w in surfaces:
            b = None
            # Sinh TỰ HỒI QUY: mục tiêu đa âm tiết (`Sông Hồng`) được coi là
            # với tới khi TOKEN ĐẦU (`Sông`) lên hạng. Thước cũ tra cả cụm ⇒
            # `resolve_surface_ids` trả rỗng ⇒ 74% mục tiêu bị loại im lặng
            # khỏi phép đếm (TK_046 §32.1a).
            ids_w = H.resolve_surface_ids(v, w)
            if not ids_w and w.split():
                ids_w = H.resolve_surface_ids(v, w.split()[0])
            for i in ids_w:
                if np.isfinite(fin[i]) and (b is None or fin[i] > fin[b]):
                    b = i
            out.append(rk[b] if b is not None else 0)   # 0 = ngoài phân bố
        return out

    rows: list[dict] = []
    for ci, cse in enumerate(cases):
        log(f"ca {ci+1}/{len(cases)}: P={cse['P']!r} lớp={cse['lop']!r}")
        pr = tien_to.get(cse["P"], cse["P"])
        s0 = diem(pr, 0.0)
        s1 = diem(pr, args.boost)
        h0, h1 = hang(s0, cse["khi"]), hang(s1, cse["khi"])
        # RÒ 2 đo lại trên chân lý v2: phân tầng theo ĐẦU RA của ② (hợp lệ —
        # xem TK_046 §27.1), tức đề cử của chuỗi có nằm trong lớp ĐÚNG không.
        cfg2 = HCAConfig()
        cfg2.hub_chain_boost = args.boost
        if args.ke_ctx >= 0:
            cfg2.hub_chain_ke_ctx = args.ke_ctx
        de = G.hub_chain_proposals(state, list(v.encode_prompt(pr)), cfg2)
        ms = {H._surf(v, t) for t in de}
        r = dict(cse)
        r["n_de_cu"] = len(de)
        r["②_dung"] = bool(ms & big.get(cse["lop"], set()))
        r["nhanh"] = "② ĐÚNG" if r["②_dung"] else "② SAI"
        r["hang_tat"], r["hang_bat"] = h0, h1
        cap = [(a, b) for a, b in zip(h0, h1) if a and b]
        r["len"] = sum(1 for a, b in cap if b < a)
        r["xuong"] = sum(1 for a, b in cap if b > a)
        r["vao_moi"] = sum(1 for a, b in zip(h0, h1) if not a and b)
        rows.append(r)
        log(f"   khỉ={cse['khi'][:4]} tắt={h0[:4]} bật={h1[:4]}"
            f"  lên={r['len']} xuống={r['xuong']} mới={r['vao_moi']}")

    tot_len = sum(r["len"] for r in rows)
    tot_xuong = sum(r["xuong"] for r in rows)
    tot_moi = sum(r["vao_moi"] for r in rows)
    ty = [b / a for r in rows for a, b in zip(r["hang_tat"], r["hang_bat"])
          if a and b and b < a]
    out = {"checkpoint": args.checkpoint, "boost": args.boost, "n_ca": len(rows),
           "len": tot_len, "xuong": tot_xuong, "vao_moi": tot_moi,
           "ty_le_hang_trung_vi": round(statistics.median(ty), 3) if ty else None,
           "ca": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log("═══ TỔNG ═══")
    for nh in sorted({r.get("nhanh","—") for r in rows}):
        sub=[r for r in rows if r.get("nhanh","—")==nh]
        L=sum(r["len"] for r in sub); X=sum(r["xuong"] for r in sub)
        M=sum(r["vao_moi"] for r in sub)
        t=[b/a for r in sub for a,b in zip(r["hang_tat"],r["hang_bat"]) if a and b and b<a]
        log(f"  [{nh}] n={len(sub)}  lên={L} xuống={X} mới={M}"
            f"  tỉ lệ={statistics.median(t):.3f}" if t else
            f"  [{nh}] n={len(sub)}  lên={L} xuống={X} mới={M}  tỉ lệ=—")
    log(f"  «con khỉ» LÊN hạng   : {tot_len}")
    log(f"  «con khỉ» XUỐNG hạng : {tot_xuong}")
    log(f"  vào phân bố lần đầu  : {tot_moi}")
    if ty:
        log(f"  tỉ lệ hạng (bật/tắt) trung vị: {statistics.median(ty):.3f}"
            f"  (<1 = tốt hơn)")
    log(f"✓ → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
