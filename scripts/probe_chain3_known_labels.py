"""VIỆC 3 — chạy chuỗi ba bước với TỪ VỰNG NHÃN ĐÃ BIẾT.

Đường 1 cho thấy cơ chế KHÔNG hỏng (ca thử vàng chó/mèo/khỉ chạy).
Đường 3 cho thấy dữ liệu KHÔNG thiếu lớp (164 lớp có ≥3 thành viên, sạch:
`triều đại` → Hồ/Lê/Lý/Nguyễn/Trần).

Vậy nút thắt nằm ở ĐƯỜNG NỐI. Script này cắt đúng một biến: **bước ② không
phải tự phát hiện nhãn nữa** — ứng viên nhãn bị giới hạn vào 164 lớp đã đào.
Mọi bước khác vẫn do cơ chế làm.

  ✓ ② chạy được ⇒ nút thắt ĐÚNG là khâu PHÁT HIỆN NHÃN (vá bằng cụm/WS-span).
  ✗ ② vẫn câm  ⇒ nút thắt ở chỗ khác, phải tìm tiếp.

Thước nằm TRONG đồ thị (memory `blocking-gate-measurement-design`): mỗi bước
báo đạt/trượt riêng, không gộp thành một số.

Chạy::

    PYTHONPATH=src python scripts/probe_chain3_known_labels.py \
        --checkpoint checkpoints/classrich.pkl \
        --ws-spans checkpoints/classrich.ws_spans.json \
        --n 40 --out out/chain3_known_labels.json
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402

_DET = frozenset({"một", "các", "những", "mọi", "mỗi"})
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:9.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ws-spans", default="")
    ap.add_argument("--out", default="out/chain3_known_labels.json")
    ap.add_argument("--n", type=int, default=40, help="số ca (P, lớp) đem đo")
    ap.add_argument("--d-max", type=int, default=3)
    ap.add_argument("--min-count", type=float, default=2.0)
    ap.add_argument("--gidf-min", type=float, default=4.0)
    ap.add_argument("--channel", default="typed_first")
    ap.add_argument("--seed-universe", choices=("tu_do", "lop"), default="tu_do",
                    help="`lop` = ÉP bước ① chỉ lấy hạt giống là THÀNH VIÊN của "
                         "lớp đã đào (phép đối xứng với việc giới hạn bước ②)")
    ap.add_argument("--seed-topk", type=int, default=10,
                    help="lấy sâu bao nhiêu ở ① TRƯỚC khi lọc theo vũ trụ")
    ap.add_argument("--head-rule", default="max_gidf",
                    help="luật chọn trung tâm SAU pivot — xem `_mine_via_pivot`")
    ap.add_argument("--head-rule-trai", default="max_gidf")
    ap.add_argument("--indeg-q", type=float, default=0.0)
    ap.add_argument("--hub-rank", choices=("diem", "phieu"), default="diem",
                    help="`diem` = hành vi hiện tại của `score_hubs_from` (giữ "
                         "điểm LỚN NHẤT trên mỗi đích, không đếm hạt giống nào "
                         "bầu). `phieu` = xếp theo SỐ HẠT GIỐNG cùng trỏ tới, "
                         "đúng thiết kế `chó+mèo+khỉ → động vật`.")
    args = ap.parse_args()

    log(f"nạp {args.checkpoint} …")
    state, _ = load_checkpoint(args.checkpoint)
    vocab, ps = state.vocab, state.parents
    n = len(vocab)
    raw = [str(vocab.token_of(i)) for i in range(n)]
    S = [r.replace(MARK_SPACE, "").strip() for r in raw]
    BAD = {i for i in range(n) if re.match(r"^<.*>$", S[i]) or S[i].endswith(":")}
    LA = {i for i in range(n) if S[i] == "là"}
    DET = {i for i in range(n) if S[i].lower() in _DET}
    FUNC = {i for i in range(n)
            if S[i].lower() in H._FUNC or len(S[i]) <= 1
            or MARK_SPACE not in raw[i]} | BAD

    # ── A. đào lớp (giống đường 3, đã vá lỗi S[vị_trí]) ───────────────────
    log("đào lớp `X là C` …")
    cls: dict[str, set[str]] = defaultdict(set)
    for rec in ps.parents:
        seq = list(rec.emit_ids)
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
    big = {c: m for c, m in cls.items() if len(m) >= 3}
    # từ vựng nhãn: cả cụm ĐẦY ĐỦ lẫn token ĐẦU (hub_spread có thể chỉ ra 1 token)
    lab_full = set(big)
    lab_head = {c.split()[0] for c in big}
    all_mem: set[str] = set().union(*big.values()) if big else set()
    mem2cls: dict[str, set[str]] = defaultdict(set)
    for c, ms in big.items():
        for m in ms:
            mem2cls[m].add(c)
    log(f"  lớp ≥3 thành viên = {len(big)} · nhãn-đầu riêng = {len(lab_head)}"
        f" · thành viên riêng = {len(all_mem)}")

    # ── B. token nào ở cha nào (để chọn thuộc tính có hình dạng đúng) ─────
    log("bảng token → cha …")
    tok_par: dict[str, set[int]] = defaultdict(set)
    par_tok: list[frozenset[str]] = []
    for pi, rec in enumerate(ps.parents):
        ts = {S[t] for t in rec.emit_ids if t not in FUNC}
        par_tok.append(frozenset(ts))
        for s in ts:
            tok_par[s].add(pi)

    # ── C. chọn ca (P, lớp): P chạm ≥2 nhưng ≤½ thành viên ────────────────
    log("chọn ca đo …")
    cases: list[dict] = []
    for c, members in sorted(big.items(), key=lambda kv: -len(kv[1])):
        prop_hit: dict[str, set[str]] = defaultdict(set)
        for m in members:
            seen: set[str] = set()
            for pi in list(tok_par.get(m, ()))[:300]:
                seen |= par_tok[pi]
            seen -= {m, c}
            for p in seen:
                prop_hit[p].add(m)
        for p, touched in prop_hit.items():
            if p in members or len(touched) < 2:
                continue
            if len(touched) > 0.5 * len(members):
                continue
            unseen = members - touched
            if not unseen:
                continue
            cases.append({
                "P": p, "lop": c,
                "hat_giong_dung": sorted(touched)[:8],
                "thanh_vien_moi_dung": sorted(unseen)[:8],
                "n_thanh_vien": len(members),
            })
    # đa dạng: mỗi lớp tối đa 2 ca
    per: dict[str, int] = defaultdict(int)
    picked: list[dict] = []
    for cse in cases:
        if per[cse["lop"]] >= 2:
            continue
        per[cse["lop"]] += 1
        picked.append(cse)
        if len(picked) >= args.n:
            break
    log(f"  ca ứng viên = {len(cases)} · đem đo = {len(picked)}")

    # ── D. dựng index cơ chế ──────────────────────────────────────────────
    wsp = None
    if args.ws_spans and Path(args.ws_spans).exists():
        wsp = H.load_ws_spans(args.ws_spans)
        log(f"  ws_spans sidecar = {len(wsp) if wsp else 0}")
    log(f"dựng index (d_max={args.d_max} min_count={args.min_count} "
        f"head_rule={args.head_rule}) …")
    idx = H.build_hub_spread_index(
        state, d_max=args.d_max, min_count=args.min_count,
        head_rule=args.head_rule, head_rule_trai=args.head_rule_trai,
        label_indeg_q=args.indeg_q, ws_spans=wsp)
    log("  xong")

    def ids_of(surface: str) -> list[int]:
        return H.resolve_surface_ids(vocab, surface)

    # ── E. chạy chuỗi, chấm TỪNG BƯỚC ─────────────────────────────────────
    rows: list[dict] = []
    tally = {"①_co_hat": 0, "①_co_2_hat": 0, "①_dung_hat": 0,
             "①_≥2_cung_lop": 0, "②_ra_nhan": 0,
             "②_dung_lop": 0, "②_khop_lop_chung": 0, "③_co_tv": 0, "③_co_moi": 0, "③_moi_that": 0}
    for ci, cse in enumerate(picked):
        log(f"ca {ci + 1}/{len(picked)}: P={cse['P']!r} lớp={cse['lop']!r}")
        pid = ids_of(cse["P"])
        if not pid:
            continue
        r: dict = dict(cse)

        # ① P → hạt giống
        s1 = H.score_hubs_from(state, idx, pid[:3], topk=args.seed_topk,
                               gidf_min=args.gidf_min, channel=args.channel)
        cand = [h for h in s1 if h.surface != cse["P"]]
        r["①_truoc_loc"] = [h.surface for h in cand[:8]]
        if args.seed_universe == "lop":
            cand = [h for h in cand if h.surface in all_mem]
        hat = cand[:4]
        r["①_ra"] = [h.surface for h in hat]
        if hat:
            tally["①_co_hat"] += 1
        if len(hat) >= 2:
            tally["①_co_2_hat"] += 1
        trung = set(r["①_ra"]) & set(cse["hat_giong_dung"])
        r["①_trung_hat_dung"] = sorted(trung)
        if trung:
            tally["①_dung_hat"] += 1
        # có ≥2 hạt giống CÙNG một lớp không? (điều kiện cần của bước ②)
        votes: dict[str, list[str]] = defaultdict(list)
        for h in hat:
            for c in mem2cls.get(h.surface, ()):
                votes[c].append(h.surface)
        shared = {c: vs for c, vs in votes.items() if len(vs) >= 2}
        r["①_lop_chung"] = {c: vs for c, vs in list(shared.items())[:4]}
        if shared:
            tally["①_≥2_cung_lop"] += 1

        # ② hạt giống → nhãn, ỨNG VIÊN GIỚI HẠN vào lớp đã đào
        if hat:
            cam = {h.surface for h in hat} | {cse["P"]}
            s2 = H.score_hubs_from(state, idx, [h.tid for h in hat], topk=60,
                                   gidf_min=args.gidf_min, channel=args.channel)
            if args.hub_rank == "phieu":
                # Đếm hạt giống nào bầu — `score_hubs_from` KHÔNG làm việc này.
                phieu: dict[str, set[str]] = defaultdict(set)
                tot_best: dict[str, object] = {}
                for h in hat:
                    for x in H.score_hubs_from(
                            state, idx, [h.tid], topk=60,
                            gidf_min=args.gidf_min, channel=args.channel):
                        phieu[x.surface].add(h.surface)
                        cur = tot_best.get(x.surface)
                        if cur is None or (x.typed, x.score) > (cur.typed, cur.score):
                            tot_best[x.surface] = x
                s2 = sorted(tot_best.values(),
                            key=lambda x: (-len(phieu[x.surface]),
                                           not x.typed, -x.score))
                r["②_phieu"] = {k: sorted(v) for k, v in
                                sorted(phieu.items(),
                                       key=lambda kv: -len(kv[1]))[:5]}
            ok = [h for h in s2
                  if h.surface not in cam
                  and (h.surface in lab_full or h.surface in lab_head)]
            r["②_ra_tu_do"] = [h.surface for h in s2[:6]]
            r["②_ra_gioi_han"] = [h.surface for h in ok[:6]]
            if ok:
                tally["②_ra_nhan"] += 1
                nhan = ok[0]
                r["②_nhan"] = nhan.surface
                r["②_dung"] = nhan.surface in (cse["lop"], cse["lop"].split()[0])
                if r["②_dung"]:
                    tally["②_dung_lop"] += 1
                # THƯỚC ĐÚNG: nhãn không buộc phải là lớp `C` sinh ra `P` —
                # hạt giống ① lấy tự do nên có thể thuộc lớp khác. Cái phải hỏi
                # là: nhãn có phải MỘT TRONG các lớp mà ≥2 hạt giống cùng thuộc?
                khop = [c for c in shared
                        if nhan.surface in (c, c.split()[0])]
                r["②_khop_lop_chung"] = khop
                if khop:
                    tally["②_khop_lop_chung"] += 1
                # ③ nhãn → thành viên
                mem = H.reverse_peers(state, idx, [nhan.tid], topk=15,
                                      gidf_min=0.0, channel=args.channel)
                r["③_ra"] = [m.surface for m in mem[:10]]
                if mem:
                    tally["③_co_tv"] += 1
                pset = {x.surface for x in s1} | {cse["P"]}
                moi = [m.surface for m in mem if m.surface not in pset
                       and m.surface not in cam]
                r["③_moi"] = moi[:8]
                if moi:
                    tally["③_co_moi"] += 1
                # MỚI THẬT = chưa nối với THUỘC TÍNH trong W_skip (tiêu chí của
                # tác giả: `khỉ` chưa từng đứng cạnh `nghịch ngợm`), VÀ là thành
                # viên có thật của lớp đã đào — không phải mảnh trôi nổi.
                near_p: set[str] = set()
                for p_ in pid[:3]:
                    for t_ in idx.fwd.get(int(p_), {}):
                        near_p.add(H._surf_ix(idx, vocab, t_))
                that = [s_ for s_ in moi
                        if s_ not in near_p and s_ in big.get(nhan.surface, set())]
                r["③_moi_that"] = that[:6]
                if that:
                    tally["③_moi_that"] += 1
        rows.append(r)

    out = {"checkpoint": args.checkpoint, "channel": args.channel,
           "head_rule": args.head_rule,
           "min_count": args.min_count, "n_ca": len(rows),
           "n_lop_da_dao": len(big), "tally": tally, "ca": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log("═══ TỔNG ═══")
    for k, val in tally.items():
        log(f"  {k:16} {val:3}/{len(rows)}")
    log(f"✓ → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
