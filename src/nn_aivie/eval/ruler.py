r"""TK_045 §13 Trụ E — CÂY THƯỚC cho sinh sáng tạo.

## Vì sao trụ này đứng TRƯỚC mọi trụ khác

Riêng TK_045 đã có **năm** lần kết luận sai vì thước: `log2(rank)` gọi là bits (v4 Trụ 1) ·
`copy=1.00` coi là đỉnh (§12.112) · nhãn gold sai **hai** lần (§12.66/§12.68) · G1 đo nhầm
thứ (§12.110) · tín hiệu n=30 không lặp (§12.113). Và riêng §13 tôi đã hỏng cổng **ba** lần
(§13.6 · §13.9 · §13.10). Thước sai không làm chậm dự án — nó làm dự án **đi sai hướng mà
vẫn thấy số đẹp**.

## ⚠️ HAI TRỤC NÀY CHỈ CÓ NGHĨA KHI ĐỌC THÀNH CẶP

| | novelty THẤP | novelty CAO |
|---|---|---|
| **attestation CAO** | **CHÉP** — trôi chảy, vô dụng | ✅ **ĐÍCH** — soạn mới từ mảnh đã chứng thực |
| **attestation THẤP** | (hiếm — hỏng gì đó) | **SALAD** — mới mà vô nghĩa |

Đọc `attestation` một mình là **tái phạm đúng lỗi §12.112**: bộ chép verbatim đạt
attestation = 1.00 **tầm thường**. Đọc `novelty` một mình thì salad thắng tuyệt đối.
Vì vậy API này **luôn trả cả hai** và `cell()` gọi tên ô — không có hàm nào trả một trục.

## Ba trục

* **`attestation`** — tỉ lệ n-gram sinh ra CÓ trong corpus. Đây là **bảo chứng chống salad
  bằng cấu trúc** của §13 Trụ A: nếu mọi span đều được chứng thực thì "không salad" là
  **định lý**, không phải hy vọng. Đo được ⇒ tuyên bố đó kiểm chứng được.
* **`novelty`** — 1 − (đoạn LIÊN TỤC dài nhất trùng một cha) / |sinh|. Cố ý dùng đoạn
  **liên tục**, khác `copy` của `eval_gen_broad` vốn dùng LCS (dãy CON, không liên tục):
  LCS phạt cả việc dùng lại từ vựng đúng chỗ, còn cái ta muốn bắt là **chép nguyên khối**.
* **`lạc` / `pmi_cross`** — trôi đề; dùng lại `eval_gen_broad` / `metrics_pmi`, không định
  nghĩa lại.

## Cỡ mẫu — vì sao n=30 KHÔNG đủ (§12.113)

`min_pairs_for()` trả số cặp cần để một chênh lệch tỉ lệ thắng nhất định vượt nhiễu. Với
so-sánh-CẶP (mỗi prompt một cặp A/B, người chấm chọn bên tốt hơn), phát hiện chênh lệch
+15pp quanh 50% ở mức 95% cần **~85 cặp**; +10pp cần **~190**. §12.113 chấm n=30 rồi thấy
"tín hiệu +0.50 không lặp" — **đúng như số học tiên đoán**, không phải xui.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


# ------------------------------------------------------------------ chứng thực
class NgramIndex:
    """Tập n-gram của corpus (từ `emit_ids` của cha). Dựng một lần, hỏi nhiều lần."""

    def __init__(self, emits, *, orders=(2, 3)):
        self.orders = tuple(int(o) for o in orders)
        self._sets: dict[int, set[tuple[int, ...]]] = {o: set() for o in self.orders}
        for e in emits:
            seq = [int(t) for t in e]
            for o in self.orders:
                s = self._sets[o]
                for i in range(len(seq) - o + 1):
                    s.add(tuple(seq[i: i + o]))

    def attested(self, seq, order: int) -> tuple[int, int]:
        """(số n-gram có trong corpus, tổng n-gram) của `seq` ở bậc `order`."""
        seq = [int(t) for t in seq]
        s = self._sets.get(int(order))
        if s is None or len(seq) < order:
            return 0, 0
        tot = len(seq) - order + 1
        hit = sum(
            1 for i in range(tot) if tuple(seq[i: i + order]) in s
        )
        return hit, tot


def attestation_rate(index: NgramIndex, seq, *, order: int = 2) -> float:
    """Tỉ lệ n-gram sinh ra đã được chứng thực. 1.0 = mọi mối nối đều có thật.

    ⚠️ **KHÔNG đọc một mình** — bộ chép verbatim đạt 1.00 tầm thường. Xem `cell()`.
    """
    hit, tot = index.attested(seq, order)
    return (hit / tot) if tot else float("nan")


# --------------------------------------------------------------------- mới lạ
def _longest_common_run(a, b) -> int:
    """Đoạn LIÊN TỤC chung dài nhất (không phải LCS dãy-con)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


class CopySpanIndex:
    """Đoạn liên tục dài nhất mà một chuỗi sinh chia với BẤT KỲ cha nào."""

    def __init__(self, emits, *, max_len: int = 96, topk: int = 40):
        self.emits = [[int(t) for t in e][:max_len] for e in emits]
        self.topk = int(topk)
        self.inv: dict[int, list[int]] = defaultdict(list)
        for pi, e in enumerate(self.emits):
            for t in set(e):
                self.inv[t].append(pi)

    def longest_run(self, seq) -> tuple[int, int]:
        """(độ dài đoạn liên tục dài nhất, chỉ số cha). Chỉ xét cha chia nhiều token nhất."""
        seq = [int(t) for t in seq]
        if not seq:
            return 0, -1
        cnt: dict[int, int] = defaultdict(int)
        for t in set(seq):
            for pi in self.inv.get(t, ()):
                cnt[pi] += 1
        if not cnt:
            return 0, -1
        best, bpi = 0, -1
        for pi in sorted(cnt, key=lambda p: -cnt[p])[: self.topk]:
            r = _longest_common_run(seq, self.emits[pi])
            if r > best:
                best, bpi = r, pi
        return best, bpi


def novelty_score(index: CopySpanIndex, seq) -> float:
    """1 − (đoạn liên tục dài nhất trùng một cha)/|sinh|. 0 = chép nguyên khối.

    ⚠️ **KHÔNG đọc một mình** — salad đạt novelty ≈ 1.0 tầm thường. Xem `cell()`.
    """
    seq = [int(t) for t in seq]
    if not seq:
        return float("nan")
    run, _pi = index.longest_run(seq)
    return 1.0 - (run / len(seq))


def copy_run_tokens(index: CopySpanIndex, seq) -> int:
    """Đoạn chép liên tục dài nhất, tính bằng TOKEN (tuyệt đối, không chuẩn hoá).

    Cặp bắt buộc của `novelty_score`: tỉ lệ có thể đẹp lên chỉ vì đầu ra dài ra (§13.13).
    """
    run, _pi = index.longest_run([int(t) for t in seq])
    return int(run)


# ------------------------------------------------------------------- ô kết hợp
@dataclass
class RulerScores:
    attestation: float
    novelty: float
    n_tokens: int
    lac: float = float("nan")
    pmi_cross: float = float("nan")
    # ⚠️ Trục TUYỆT ĐỐI, thêm sau P52 (§13.13). `novelty` là TỈ LỆ `1 − run/len`, nên nó
    # đẹp lên chỉ vì đầu ra DÀI RA. Đo được: dữ liệu ×45 ⇒ novelty 0.545→0.737 (tỉ lệ tốt
    # hơn) NHƯNG khối chép liên tục 22.7→**51.7 token** (tệ hơn 2.3×). Một trích dẫn
    # nguyên văn 52 token thì không "sáng tạo" dù tỉ lệ có đẹp.
    # ⇒ `novelty` KHÔNG được đọc mà thiếu `copy_run`.
    copy_run: int = -1

    def cell(self, *, att_hi: float = 0.90, nov_hi: float = 0.50) -> str:
        """Gọi TÊN ô trong bảng 2×2 — buộc người đọc thấy cả hai trục cùng lúc."""
        if self.n_tokens < 4:
            return "quá-ngắn"
        a = self.attestation >= att_hi
        n = self.novelty >= nov_hi
        if a and n:
            return "ĐÍCH"
        if a and not n:
            return "CHÉP"
        if (not a) and n:
            return "SALAD"
        return "hỏng"


def score_generation(
    ngram: NgramIndex,
    copyspan: CopySpanIndex,
    cont_ids,
    *,
    order: int = 2,
    lac: float = float("nan"),
    pmi_cross: float = float("nan"),
) -> RulerScores:
    cont = [int(t) for t in cont_ids]
    run = copy_run_tokens(copyspan, cont) if cont else -1
    return RulerScores(
        attestation=attestation_rate(ngram, cont, order=order),
        novelty=(1.0 - run / len(cont)) if cont else float("nan"),
        n_tokens=len(cont),
        lac=lac,
        pmi_cross=pmi_cross,
        copy_run=run,
    )


# ------------------------------------------------------------------- cỡ mẫu
def min_pairs_for(
    delta_pp: float, *, conf_z: float = 1.96, power_z: float = 0.8416
) -> int:
    """Số cặp A/B tối thiểu để **PHÁT HIỆN ĐƯỢC** chênh lệch `delta_pp` (điểm %).

    So-sánh-CẶP quanh p=0.5: `n ≥ (z_α + z_β)² / (4δ²)`, δ = delta_pp/100.
    Mặc định α=5% hai phía (z=1.96), **power 80%** (z=0.8416):

        +28pp → 26 cặp · +15pp → **88** · +10pp → **197** · +5pp → **785**

    ⚠️ Bỏ số hạng power (chỉ `(z_α/2δ)²`) cho ra **43** cặp cho +15pp — đó là ngưỡng
    *"nếu may mắn quan sát đúng độ lớn thật thì CI vừa đủ tách khỏi 50%"*, tức **power ≈ 50%
    = tung đồng xu**. Chính tôi viết nhầm công thức đó lần đầu; test bắt được.

    Đây là lý do **số học** của §12.113: n=30 chỉ phát hiện nổi chênh lệch **≥ ~26pp**, nên
    *"tín hiệu +0.50 không lặp"* là kết cục **tiên đoán được**, không phải xui.
    """
    d = float(delta_pp) / 100.0
    if d <= 0:
        raise ValueError("delta_pp phải > 0")
    return int(math.ceil((float(conf_z) + float(power_z)) ** 2 / (4.0 * d * d)))


def win_rate_ci(wins: int, n: int, *, conf_z: float = 1.96) -> tuple[float, float, float]:
    """(tỉ lệ thắng, cận dưới, cận trên) — Wald. `n` bỏ hoà thì truyền n đã trừ hoà."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = wins / n
    half = conf_z * math.sqrt(max(p * (1 - p), 1e-12) / n)
    return p, max(0.0, p - half), min(1.0, p + half)
