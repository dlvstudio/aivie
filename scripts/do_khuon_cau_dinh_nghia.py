"""ĐƯỜNG 3 — rác cạnh lớp đến từ KHUÔN CÂU nào?

§61.3 đếm được: câu dạng vàng («X là loài/lớp/động vật…») chỉ chiếm 2,1% trong
~9 500 câu có `là` trên `classrich`; phần còn lại cũng sinh cạnh `X là Y` và
chúng đông hơn. Nhưng «đông hơn» chưa đủ để vá — cần biết CHÚNG LÀ KHUÔN GÌ.

Cách đo: lấy 949 cặp (thành viên → lớp) của chân lý v2 làm chuẩn. Với mỗi
thành viên chuẩn, cơ chế đang gán cho nó những nhãn nào? Nhãn nào ĐÚNG chuẩn,
nhãn nào rác — và câu sinh ra nhãn rác đó mở đầu bằng khuôn nào sau `là`?

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/do_khuon_cau_dinh_nghia.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
gc.disable()

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.core.tokenizer import MARK_SPACE  # noqa: E402
from nn_aivie.hca import hub_spread as H  # noqa: E402
from nn_aivie.hca.config import HCAConfig  # noqa: E402

T0 = time.time()


def main() -> int:
    ck = "checkpoints/classrich.pkl"
    cfg = HCAConfig()
    state, _ = load_checkpoint(ck)
    v, ps = state.vocab, state.parents
    S = [str(v.token_of(i)).replace(MARK_SPACE, "").strip() for i in range(len(v))]
    wsp = H.load_ws_spans("checkpoints/classrich.ws_spans.json")
    pm = H.load_pos_map(cfg.hub_chain_pos_dict)
    hq = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # §64 head_gidf_q
    print(f"  cờ: head_gidf_q={hq}")
    idx = H.build_hub_spread_index(
        state, d_max=cfg.hub_chain_d_max, min_count=cfg.hub_chain_min_count,
        ws_spans=wsp, head_gidf_q=hq,
        head_rule=cfg.hub_chain_head_rule,
        head_rule_trai=cfg.hub_chain_head_rule_trai,
        label_gidf_q=cfg.hub_chain_gidf_q, label_indeg_q=cfg.hub_chain_indeg_q,
        chu_ngu_ket_hop_q=cfg.hub_chain_chu_gia_q, pos_map=pm)
    lop = json.load(open("out/chan_ly_v2.json"))["lop"]
    print(f"\n### {ck} · {len(ps.parents)} cha · chuẩn {sum(len(x) for x in lop.values())} cặp\n")

    def surf(t: int) -> str:
        return idx.label_surf.get(int(t)) or H._surf(v, int(t))

    # thành viên chuẩn → tập lớp chuẩn
    chuan: dict[str, set[str]] = defaultdict(set)
    for c, ms in lop.items():
        for m in ms:
            chuan[m].add(c)

    be = idx.via_pivot
    n_dung = n_rac = n_tv = 0
    rac_nhan: Counter[str] = Counter()
    for m, cs in chuan.items():
        ids = H.resolve_surface_ids(v, m) or []
        nhan = {surf(d) for i in ids for d in be.get(int(i), {})}
        if not nhan:
            continue
        n_tv += 1
        for x in nhan:
            if x in cs or any(x == c.split()[0] for c in cs):
                n_dung += 1
            else:
                n_rac += 1
                rac_nhan[x] += 1
    print(f"  thành viên chuẩn CÓ cạnh: {n_tv}")
    print(f"  nhãn ĐÚNG chuẩn: {n_dung}   ·   nhãn RÁC: {n_rac}"
          f"   ⇒ độ sạch {100*n_dung/max(1,n_dung+n_rac):.1f}%\n")
    print(f"  20 nhãn RÁC hay gặp nhất:\n     {rac_nhan.most_common(20)}\n")

    # KHUÔN: hai token ngay sau `là` trong câu sinh ra nhãn rác đó
    LA = {i for i in range(len(v)) if S[i] == "là"}
    khuon_rac: Counter[str] = Counter()
    khuon_dung: Counter[str] = Counter()
    top_rac = {x for x, _ in rac_nhan.most_common(40)}
    moi_lop = set(lop)
    for rec in ps.parents:
        seq = [int(t) for t in rec.emit_ids]
        for k, t in enumerate(seq):
            if t not in LA or k + 1 >= len(seq):
                continue
            kh = " ".join(S[x] for x in seq[k + 1:k + 3] if S[x])
            sau = {S[x] for x in seq[k + 1:k + 6]}
            if sau & top_rac:
                khuon_rac[kh] += 1
            if sau & moi_lop:
                khuon_dung[kh] += 1
    print("  KHUÔN sau `là` — câu chứa nhãn RÁC vs câu chứa lớp CHUẨN:")
    print(f"     {'khuôn':<18} {'rác':>6} {'chuẩn':>7}")
    for kh, n in khuon_rac.most_common(15):
        print(f"     {kh:<18} {n:>6} {khuon_dung.get(kh, 0):>7}")
    print(f"\n[{time.time() - T0:.0f}s] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
