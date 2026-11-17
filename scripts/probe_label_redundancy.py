#!/usr/bin/env python
"""P97 (TK_046 §16) — **SÀNG NHÃN**: nhãn nào `E` đã biết rồi, nhãn nào thật sự mới?

§15 đo xong: nhãn **từ loại** chỉ cho **1,03×** trên `tach_lat` — không mua gì. Kết luận đã
đăng ký trước là *nhãn từ loại **dư thừa** với thống kê láng giềng*. Nhưng đó là kết luận rút
từ **một** loại nhãn. Trước khi xây thêm bất kỳ nhãn nào (loại từ, phong cách, vai cú pháp),
cần một phép sàng **tổng quát** trả lời đúng câu hỏi đó cho **mọi** nhãn ứng viên.

## Phép sàng

`E` = SVD của đồng hiện có hướng ⇒ nó **là** "thống kê láng giềng" ở dạng nén. Vậy:

    dự đoán nhãn của token `t` bằng bỏ phiếu của 15 láng giềng gần nhất trong `E`

- đoán **đúng cao** ⇒ `E` **đã biết** nhãn đó ⇒ nhãn **dư thừa**, xây thêm không mua gì;
- đoán **kém** ⇒ nhãn mang thông tin `E` **không** có ⇒ đáng xây.

**Độ dư thừa** = phần **khoảng trống** mà `E` lấp được, không phải độ chính xác trần:

    dư thừa = (đúng − sàn đa số) / (1 − sàn đa số)

Phải chuẩn hoá theo sàn đa số: nhãn nào cũng có một lớp trội (`danh từ` 44%), nên "đúng 60%"
tự nó **không nói gì** — đúng bài học §13.65 (bão hoà) và §13.35 (mẫu số).

## Ba nhãn + MỘT ĐỐI CHỨNG bắt buộc

| | |
|---|---|
| `tu_loai` | 8 nhãn từ điển (§14) |
| `phong_cach` | `khẩu ngữ` / `cũ` / `phương ngữ` / `văn chương` / … — 34,8% mục từ điển có, **chưa ai dùng** |
| `ngau_nhien` | **đối chứng**: xáo nhãn từ loại giữa các token, giữ nguyên phân bố lớp |

⚠️ Không có cột `ngau_nhien` thì phép đo mù: nếu một nhãn **bốc ngẫu nhiên** cũng ra độ dư thừa
cao thì thang đo hỏng, và mọi kết luận ở trên đều rỗng.

## CỔNG G-sang (nêu TRƯỚC)

1. **≥500 token** mỗi loại nhãn.
2. **Đối chứng phải ≈ 0**: `ngau_nhien` dư thừa < **0,10**. Trượt ⇒ thang đo hỏng, không đọc
   tiếp.
3. Một nhãn **ĐÁNG XÂY** khi dư thừa < **0,40** — `E` lấp chưa tới 40% khoảng trống.

## Dự đoán ghi TRƯỚC (Claude)

- `tu_loai` dư thừa **0,50–0,75** ⇒ **không đáng xây thêm**, và nó **giải thích** con số 1,03×
  ở §15.
- `phong_cach` dư thừa **0,20–0,45** — thấp hơn, vì phong cách là tính chất của **văn bản gốc**
  chứ không của **láng giềng cục bộ**. Nhưng mini là kho lịch sử/bách khoa, gần như thuần một
  giọng, nên có thể **thiếu tương phản** và số sẽ khó đọc; nếu `phong_cach` phủ quá lệch tôi ghi
  rõ là **chưa phân giải được**, không kết luận.
- **Tự sửa lời khuyên của chính tôi**: tôi đã khuyên tác giả thêm **loại từ** (`con`/`cái`).
  Phép sàng này cho thấy khuyên như vậy là **sai** — loại từ **chính là một token láng giềng**,
  nên `framesL` của tách lát **đã dùng nó rồi**. Không cần đo cũng biết nó dư thừa **theo cấu
  tạo**. Ghi lại để không lặp.

Chạy: `PYTHONPATH=src python scripts/probe_label_redundancy.py`
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nn_aivie.core.checkpoint_io import load_checkpoint  # noqa: E402
from nn_aivie.util.progress import get_run_logger  # noqa: E402

GATE_MIN_TOK = 500
GATE_CONTROL_MAX = 0.10
GATE_WORTH_MAX = 0.40
GATE_POSCTRL_MIN = 0.30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/mini_sent_ep_qa.pkl")
    ap.add_argument("--map", default="out/token_labels_mini.json")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/probe_label_redundancy.json")
    args = ap.parse_args()

    logger = get_run_logger("probe_label_redundancy")
    t0 = time.time()
    gc.disable()
    state, _ = load_checkpoint(str(_ROOT / args.checkpoint))
    gc.enable()

    mp = json.loads((_ROOT / args.map).read_text(encoding="utf-8"))
    E = np.asarray(state.embedding.matrix, dtype=np.float32)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    rng = random.Random(args.seed)

    # đếm lượt để cân theo tần suất dùng thật
    occ: Counter[int] = Counter()
    for rec in state.parents.parents:
        occ.update(int(t) for t in rec.emit_ids)

    def dominant(d: dict[str, float]) -> str | None:
        if not d:
            return None
        best = max(d.values())
        top = sorted(k for k, x in d.items() if x == best)
        return top[0] if len(top) == 1 else None   # đồng hạng ⇒ bỏ, đừng phá vỡ bừa

    sets: dict[str, dict[int, str]] = {}
    for key in ("tu_loai", "phong_cach"):
        lab: dict[int, str] = {}
        for ts, d in (mp.get(key) or {}).items():
            t = int(ts)
            if 0 <= t < En.shape[0]:
                g = dominant(d)
                if g:
                    lab[t] = g
        sets[key] = lab
    # đối chứng ÂM: xáo nhãn từ loại giữa các token, GIỮ NGUYÊN phân bố lớp
    base = sets["tu_loai"]
    toks = list(base)
    vals = [base[t] for t in toks]
    rng.shuffle(vals)
    sets["ngau_nhien"] = dict(zip(toks, vals))

    # đối chứng DƯƠNG = **TRẦN ĐƯỜNG ỐNG**, không phải bằng chứng ngữ nghĩa.
    #
    # Hai bản trước đều hỏng, mỗi bản một kiểu: "láng giềng TRÁI trội nhất" trượt (−0,13) vì
    # nó hỏi tính chất *lớp* mà `E` chỉ mã hoá *ngữ cảnh* (`vua → Tông, ngôi, Lý, Thánh`);
    # "đồng hiện nhiều nhất" thì **suy biến về một lớp duy nhất** (sàn 100%) vì token rác đồng
    # hiện với mọi thứ (§13.20). Cả hai đều là lỗi CHỌN ĐỐI CHỨNG, không phải `E` hỏng.
    #
    # Câu hỏi còn lại thật ra hẹp hơn: *đường ống kNN có chạy không?* Nhãn = **cụm của chính
    # `E`**. Dự đoán nó từ láng giềng `E` là **vòng tròn có chủ ý** — nó cho **TRẦN**, cặp với
    # sàn `ngau_nhien`. `tu_loai` nằm ở đâu giữa hai mốc đó mới là thứ đọc được. Không dùng
    # cột này để nói gì về ngữ nghĩa.
    lab_ids = np.array(sorted(base), dtype=np.int64)
    Vb = En[lab_ids]
    cseed = np.random.default_rng(args.seed).choice(len(lab_ids), 8, replace=False)
    C = Vb[cseed].copy()
    for _ in range(15):
        asg = (Vb @ C.T).argmax(axis=1)
        for q in range(8):
            m = asg == q
            if m.any():
                C[q] = Vb[m].mean(axis=0)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    sets["tran_duong_ong"] = {int(t): f"K{int(q)}" for t, q in zip(lab_ids, asg)}

    out: dict[str, dict] = {}
    print(f"\n{'nhãn':<14}{'token':>8}{'sàn đa số':>11}{'E đoán đúng':>13}"
          f"{'DƯ THỪA':>10}{'đáng xây':>10}")
    for key, lab in sets.items():
        ids = np.array(sorted(lab), dtype=np.int64)
        if len(ids) < 2:
            continue
        y = np.array([lab[int(t)] for t in ids])
        w = np.array([max(occ.get(int(t), 0), 1) for t in ids], dtype=np.float64)
        cls = sorted(set(y.tolist()))
        cidx = {c: i for i, c in enumerate(cls)}
        yi = np.array([cidx[c] for c in y])

        V = En[ids]                              # (m, 96)
        S = V @ V.T                              # (m, m) — mini đủ nhỏ để làm thẳng
        np.fill_diagonal(S, -np.inf)
        k = min(args.k, len(ids) - 1)
        nb = np.argpartition(-S, k - 1, axis=1)[:, :k]

        votes = np.zeros((len(ids), len(cls)))
        for r in range(len(ids)):
            for c_ in nb[r]:
                votes[r, yi[c_]] += max(float(S[r, c_]), 0.0)
        pred = votes.argmax(axis=1)
        acc = float((pred == yi) @ w / w.sum())

        cw: dict[int, float] = defaultdict(float)
        for i_, ww in zip(yi, w):
            cw[int(i_)] += ww
        maj = max(cw.values()) / w.sum()
        red = (acc - maj) / max(1.0 - maj, 1e-9)
        worth = red < GATE_WORTH_MAX
        out[key] = {"n_token": int(len(ids)), "n_lop": len(cls), "san_da_so": maj,
                    "E_dung": acc, "du_thua": red, "dang_xay": bool(worth)}
        print(f"{key:<14}{len(ids):>8,}{maj:>11.1%}{acc:>13.1%}{red:>10.2f}"
              f"{('✅' if worth else '❌'):>9}")

    ctrl = out.get("ngau_nhien", {}).get("du_thua", 1.0)
    pos_c = out.get("tran_duong_ong", {}).get("du_thua", 0.0)
    g1 = all(o["n_token"] >= GATE_MIN_TOK for o in out.values())
    g2 = ctrl < GATE_CONTROL_MAX
    g3 = pos_c >= GATE_POSCTRL_MIN
    print(f"\nđối chứng ÂM  ngau_nhien       dư thừa = {ctrl:+.2f} (phải < {GATE_CONTROL_MAX})")
    print(f"TRẦN đường ống tran_duong_ong  dư thừa = {pos_c:+.2f} "
          f"(phải ≥ {GATE_POSCTRL_MIN})")
    print(f"CỔNG G-sang: (1) n {'✅' if g1 else '❌'} · (2) sàn {'✅' if g2 else '❌'} "
          f"· (3) trần {'✅' if g3 else '❌'} ⇒ "
          f"{'ĐỌC ĐƯỢC' if (g1 and g2 and g3) else 'KHÔNG ĐỌC ĐƯỢC'}")
    if not g3:
        print("⛔ đường ống kNN không đạt cả TRẦN vòng-tròn ⇒ thang đo hỏng; 'dư thừa thấp'\n"
              "   ở trên KHÔNG đọc được là 'nhãn mang thông tin mới'.")
    if not g2:
        print("⛔ nhãn xáo ngẫu nhiên vẫn đoán được ⇒ thang đo hỏng")

    (_ROOT / args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "k": args.k, "ket_qua": out,
         "gate": {"min_tok": GATE_MIN_TOK, "control_max": GATE_CONTROL_MAX,
                  "worth_max": GATE_WORTH_MAX},
         "doc_duoc": bool(g1 and g2 and g3), "seconds": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    logger.log(f"→ {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
