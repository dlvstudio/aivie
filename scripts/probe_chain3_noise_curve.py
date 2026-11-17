"""ĐƯỜNG CONG NHIỄU — ca thử vàng chịu được bao nhiêu câu thật trước khi gãy?

Hai điểm dữ liệu đã có, và giữa chúng là một biến CHƯA AI ĐO:

    5 câu sạch, không nhiễu   → ĐẠT (`khỉ` ra)
    35 757 cha thật           → 0

Script này lấp khoảng giữa: giữ nguyên 5 câu vàng, trộn dần câu thật vào, tìm
ĐIỂM GÃY. Nó trả lời trực tiếp câu «bổ dữ liệu có cứu được không»:

  • gãy ở vài trăm câu  ⇒ bổ dữ liệu VÔ ÍCH, phải sửa cơ chế trước.
  • trụ tới hàng chục nghìn ⇒ bổ dữ liệu là đường ĐÚNG, và biết cần bao nhiêu.

Thước là câu hỏi nhị phân của chính tác giả — «`khỉ` có ra không, ở hạng mấy» —
nên không cần chấm mù.

⚠️ `min_count` giữ 0,5 ở MỌI mức để cô lập biến nhiễu: ở min_count mặc định,
cạnh vàng (w = 1×0,95) bị lỗi trọng-số-vs-số-lần cắt sạch (TK_046 §20.2a) và ta
sẽ đo nhầm bug đó thay vì đo nhiễu.

Chạy::

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/probe_chain3_noise_curve.py \
        --muc 0,100,300,1000,3000,10000,30000
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

VANG = [
    "loài chó là động vật.",
    "loài mèo là động vật.",
    "loài khỉ là động vật.",
    "nghịch ngợm như loài chó.",
    "nghịch ngợm như loài mèo.",
]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:9.1f}s] {msg}", flush=True)


def _cau(pool_path: Path, can: int) -> list[str]:
    """Tách câu từ corpus thật — 1 câu = 1 dòng = 1 cha (kiểm soát đếm chính xác)."""
    out: list[str] = []
    with pool_path.open() as f:
        for line in f:
            try:
                txt = json.loads(line)
            except Exception:
                continue
            if not isinstance(txt, str):
                continue
            for c in re.split(r"(?<=[.!?])\s+", txt.replace("\n", " ")):
                c = c.strip()
                if 20 <= len(c) <= 300:
                    out.append(c)
                    if len(out) >= can:
                        return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/text_corpus/paragraph_units.jsonl")
    ap.add_argument("--muc", default="0,100,300,1000,3000,10000,30000")
    ap.add_argument("--out", default="out/chain3_noise_curve.json")
    ap.add_argument("--min-count", type=float, default=0.5)
    ap.add_argument("--d-max", type=int, default=6)
    ap.add_argument("--tmp", default="out/noise_curve")
    args = ap.parse_args()

    muc = [int(x) for x in args.muc.split(",") if x.strip()]
    tmp = ROOT / args.tmp
    tmp.mkdir(parents=True, exist_ok=True)

    log(f"lấy tối đa {max(muc)} câu nhiễu từ {args.pool} …")
    pool = _cau(ROOT / args.pool, max(muc))
    log(f"  có {len(pool)} câu")

    import gc

    gc.disable()
    from nn_aivie.core.checkpoint_io import load_checkpoint
    from nn_aivie.core.tokenizer import MARK_SPACE
    from nn_aivie.hca import hub_spread as H

    ket: list[dict] = []
    for n_nhieu in muc:
        n_nhieu = min(n_nhieu, len(pool))
        ds = tmp / f"toy_noise_{n_nhieu}.jsonl"
        ck = tmp / f"toy_noise_{n_nhieu}.pkl"
        with ds.open("w") as f:
            for s in VANG:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            for s in pool[:n_nhieu]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        log(f"── nhiễu={n_nhieu}: train …")
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "scripts/train.py", "--dataset", str(ds),
             "--priority", "0.95", "--out", str(ck), "--bake"],
            cwd=ROOT, capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": "src"},
        )
        if r.returncode != 0:
            log(f"   ✗ train lỗi: {r.stderr[-400:]}")
            continue
        t_train = time.time() - t0

        state, _ = load_checkpoint(str(ck))
        v = state.vocab

        def surf(t: int) -> str:
            return str(v.token_of(int(t))).replace(MARK_SPACE, "").strip()

        idx = H.build_hub_spread_index(
            state, d_max=args.d_max, min_count=args.min_count, use_isg=False)
        nghich = H.resolve_surface_ids(v, "nghịch")
        row: dict = {"n_nhieu": n_nhieu, "n_cha": len(state.parents.parents),
                     "giay_train": round(t_train, 1)}
        if not nghich:
            row["loi"] = "không thấy token `nghịch`"
            ket.append(row)
            continue

        for ch in ("typed_first", "cooc"):
            s1 = H.score_hubs_from(state, idx, nghich, topk=12,
                                   gidf_min=0.0, channel=ch)
            hat = [h for h in s1 if h.surface not in ("ngợm", "như")][:4]
            r1 = [h.surface for h in hat]
            sub: dict = {"①": r1}
            # `khỉ` có tới thẳng từ `nghịch` không? (nếu có thì không còn là MỚI)
            sub["khỉ_ke_nghịch"] = any(h.surface == "khỉ" for h in s1)
            if hat:
                s2 = H.score_hubs_from(state, idx, [h.tid for h in hat],
                                       topk=20, gidf_min=0.0, channel=ch)
                cam = {h.surface for h in hat} | {"nghịch"}
                ok = [h for h in s2 if h.surface not in cam]
                sub["②"] = [h.surface for h in ok[:5]]
                if ok:
                    nhan = ok[0]
                    sub["nhãn"] = nhan.surface
                    mem = H.reverse_peers(state, idx, [nhan.tid], topk=30,
                                          gidf_min=0.0, channel=ch)
                    ms = [m.surface for m in mem]
                    sub["③"] = ms[:8]
                    sub["khỉ_hang"] = (ms.index("khỉ") + 1) if "khỉ" in ms else 0
                    sub["chó_mèo"] = ("chó" in ms, "mèo" in ms)
            row[ch] = sub
        ket.append(row)
        tf = row.get("typed_first", {})
        log(f"   cha={row['n_cha']} · nhãn={tf.get('nhãn')!r}"
            f" · khỉ hạng={tf.get('khỉ_hang', 0)}"
            f" · ①={tf.get('①')}  ({t_train:.0f}s train)")

    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.out).write_text(json.dumps(ket, ensure_ascii=False, indent=1))
    log("═══ ĐƯỜNG CONG ═══")
    log(f"  {'nhiễu':>7} {'cha':>7}  {'nhãn':<12} {'khỉ hạng':>9}  {'chó/mèo':>8}")
    for r_ in ket:
        tf = r_.get("typed_first", {})
        h = tf.get("khỉ_hang", 0)
        log(f"  {r_['n_nhieu']:>7} {r_.get('n_cha', 0):>7}  "
            f"{str(tf.get('nhãn')):<12} {(h or '—'):>9}  {str(tf.get('chó_mèo')):>8}")
    log(f"✓ → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
