# TK_020 — KẾT NỐI NGANG: kiểm kê + tham số KHOẢNG CÁCH (gap)

| Field | Value |
|-------|-------|
| **ID** | TK_020 |
| **Status** | 🟢 PROBE DƯƠNG (Claude 2026-07-06) — gap-prior thắng cả baseline lẫn broadcast; chờ tác giả duyệt bước nâng cấp cạnh |
| **Nguồn** | Tác giả hỏi: mạng tận dụng kết nối ngang chưa/thế nào? bao nhiêu? có power chưa? **khoảng cách giữa nơ-ron có quan trọng không?** |
| **Checkpoint đo** | `network_v4_full.pkl` |

## 1. KIỂM KÊ KẾT NỐI NGANG (đo `scripts/probe_horizontal_links.py`)

| Lớp | Số cạnh | Power? | Gap? | Đang tận dụng ở đâu |
| :--- | ---: | :---: | :---: | :--- |
| W_seq (SEQUENTIAL) | 270 711 | ẩn trong weight (eta×count) | luôn =1 (kề) | kênh micro A3 + `_wseq_fallback` + nguồn chiết q·k (R2) |
| ISG seg_h (TK_013) | 848 818 | ✅ `bigram_count` tường minh (top: có→thể 1818, Việt→Nam 1349) | =1 (kề, theo index slot) | matching (query_match_units qua cluster) + completion (expand slot) |
| **FRAME_BRACKET (D3)** | 35 205 | ❌ chỉ PMI — count bị VỨT lúc mine | ❌ gap bị VỨT lúc mine | bracket-gate (Δ−0.052) + opener-routing R1 + chọn data R3b |
| fact_links (t1, TK_010) | 3 448 | có position | thứ tự fact | essay t1 agenda |
| REINFORCED (G2) | 48 482 | ẩn trong weight | — | readout đường đáp |
| CO_ACTIVATION | 4 644 938 | ẩn (đếm cửa sổ) | mất (window 12 phẳng) | readout + spreading + PPMI/E |

→ Trả lời: **CÓ, đang tận dụng nhiều tầng** — nhưng lớp ngang-XA (FRAME_BRACKET) là lớp DUY NHẤT
**không mang power lẫn gap** (thông tin bị vứt lúc mine) — đúng lỗ tác giả chỉ.

## 2. KHOẢNG CÁCH — phân bố THẬT (quét source_ids 64 573 cha)

| Cặp | n | mode gap | P(mode) | P(±2 quanh mode) |
| :--- | ---: | :---: | :---: | :---: |
| càng→càng | 159 | **2** | **50%** | **64%** — RẤT NHỌN (khuôn "càng X càng Y") |
| nếu→thì | 95 | 8 | 17% | 49% |
| tuy→nhưng | 63 | 3 | 17% | 46% |
| không→mà | 550 | 7 | 11% | 44% |
| vì→nên | 53 | 7 | 11% | 40% |

→ Gap KHÔNG phải hằng số, cũng KHÔNG uniform — là **PHÂN BỐ đặc trưng từng cặp** (mệnh đề co
giãn theo nội dung). 40–64% khối lượng nằm trong ±2 quanh mode ⇒ dùng làm **PRIOR MỀM** đúng
hơn gate cứng.

## 3. GAP CÓ QUAN TRỌNG KHÔNG? — ĐO A/B trên thước (n=2860, prefix-only, KHÔNG oracle)

| Cách bơm closer | bits ↓ | top1 |
| :--- | :---: | :---: |
| base-LM (không bơm) | 4.028 | 22.1% |
| broadcast decay đều (0.9^d) | 3.939 | 22.4% |
| **GAP-PRIOR: boost × P_pair(gap=d)** | **3.926** | **22.6%** ← top1 CAO NHẤT từng đo |

**Trả lời: CÓ.** Gap-prior thắng cả baseline (Δ−0.102) lẫn broadcast, bằng đúng mức q·k blend
(3.926) qua một cơ chế hoàn toàn khác — và về nguyên lý nó là **cây cầu prefix-only** giữa 2 cực
đã đo trước đây (broadcast mù vị trí vs oracle −0.94 nhìn gold). Nó cũng chính là điều kiện để
sau này cho gate **quyền override mạnh** tại vị trí đóng khung mà không nhiễu (TK_017 §11 R2).

Ghi chú trung thực: (a) Δ so broadcast khiêm tốn (−0.012 bits / +0.2pp) trên held-out essay —
kỳ vọng cắn mạnh hơn ở văn khung dày; (b) broadcast lần này ĐÃ hơi dương (khác +0.45 cũ) vì
mine hiện tại có lọc bất-đối-xứng + PMI 1.3 (cạnh sạch hơn nhiều so bản đo D3 đầu).

## 4. ĐỀ XUẤT (chờ duyệt) — nâng CẠNH NGANG XA thành cạnh đầy đủ 3 tham số

1. **Cạnh ngang = (PMI, POWER, GAP_HIST)**: `mine_frame_brackets` giữ lại `count` (power) +
   histogram gap (đang có sẵn trong vòng quét — hiện bị vứt; giữ = gần miễn phí). Serialize
   như layer phụ (dict pair→(count, {gap:count})). Probe hiện phải quét lại 11 phút — mine-time
   1 pass là chỗ đúng.
2. **Gate R2 = FB_weight × P_pair(gap)** thay broadcast trong `frame_bracket_gate` (wired path
   + `_wseq_fallback`) — cờ `ar_gap_prior=True`, ablate về broadcast.
3. Nấc sau: gap-prior đủ sạch → thử **override mạnh tại đỉnh P(gap)** (additive/top-K) — điều
   broadcast không dám làm vì nhiễu (TK_017 §10 fix #2).
4. Xa hơn (ý mở): W_seq/co-act cũng có thể mang gap (co-act hiện đập phẳng cửa sổ 12) —
   nhưng đo từng bước, không cắm phẳng (định luật 1).

## 5. Files probe

`scripts/probe_horizontal_links.py` (kiểm kê + hist theo cặp), probe gap-prior inline
(`out/probe_gap_prior.log`).

---

## 6. IMPLEMENTED (duyệt + ý bổ sung PROVENANCE của tác giả, 2026-07-06)

### Ý tác giả bổ sung (đã vào kiến trúc)

> Cạnh ngang phát hiện trong câu/cụm nào thì phải NỐI tới nơ-ron câu/cụm đó; inference kích
> hoạt cha ("nếu trời mưa thì chúng ta sẽ ở nhà") → dội ngược làm mạnh cạnh + biết cạnh mạnh
> trong NGỮ CẢNH nào → ưu tiên đúng next-token ("chúng").

### Code

| File | Thay đổi |
|------|----------|
| `training/frame_bracket.py` | mine giữ **meta per-edge**: `power` (count), `gaps` (histogram), `prov` (parent_idx, pos, gap — cap 50/pair) → `ps.hlink_meta` |
| `core/parent_store.py` | serialize/deserialize `hlink_meta` (to/from_payload) |
| `hca/frame_bracket_gate.py` | `ensure` re-mine nếu checkpoint thiếu meta (clear layer chống PMI double); `apply_frame_bracket_gate`: boost = PMI × **P_pair(gap=d)** (`ar_gap_prior`) × **(1+ctx_w nếu prov ∈ cha-đang-kích-hoạt)** (`ar_hlink_ctx_weight`) |
| `generate_hebbian.py` | `_active_parent_set` (rank_matches top-20, 1 lần/lượt sinh) → truyền vào gate ở `_wseq_fallback` + AR loop |
| `tests/test_frame_bracket_gate.py` | +3 test (power/gap/prov; gap-prior+ctx boost; roundtrip) — **138 pass toàn suite** |

### ĐO (network_v4_full; mine mới trên 64.5k cha gồm pool văn nói → **48 448 cạnh** +meta)

Ruler qua gate ĐÚNG-như-wire (1-opener-gần-nhất; khác probe §3 dùng sum-mọi-opener):

| Cấu hình | bits ↓ | top1 |
| :--- | :---: | :---: |
| OFF | 4.028 | 22.1% |
| broadcast | 3.978 | 22.3% |
| gap-prior | 3.980 | 22.4% |
| **gap + CTX-resonance (full)** | **3.976** | **22.4%** |

Guard đọc tay: QA + 4 continuation **GIỮ NGUYÊN TOÀN BỘ**.

**Đọc trung thực:** trên wire-path 1-opener, gap+ctx chỉ nhỉnh nhẹ (−0.002 bits, +0.1pp) so
broadcast — vì 1-opener vốn đã ít nhiễu (khác probe sum-opener nơi gap thắng rõ −0.012).
**Giá trị chính của đợt này là KIẾN TRÚC**: cạnh ngang giờ là thực thể đầy đủ
`(PMI, power, gap_hist, provenance)` serialize được; ctx-resonance là máy sẵn chờ dữ liệu khung
dày hơn + nấc override (§4.3). Flags ON theo luật 3.5, giá ghi trên.

### Sổ nợ TK_020

- ctx-resonance tại RULER chậm (rank_matches mỗi text; trong generate thực = 1 lần/prompt, OK).
- Checkpoint `network_v4_full.pkl` đã RE-SAVE với meta persist (khỏi re-mine 72s/process; load 34s).

---

## 7. OVERRIDE + GRAPH UI/API (2026-07-06, tiếp)

- **§4.3 OVERRIDE implemented**: điều kiện KÉP — `P(gap=d) ≥ ar_gap_override_min_p (0.25)` VÀ
  ctx-hit (provenance ∈ cha kích hoạt) → closer được ÉP thắng (×1+margin, như index-backbone).
  Fire hẹp+tự tin — chỗ broadcast không dám. Cờ `ar_gap_override_margin` (0=tắt). Test toy:
  đối thủ áp đảo ×100 vẫn bị closer ép thắng KHI và CHỈ KHI đủ 2 điều kiện. **139 test pass.**
- **Trace A3**: mỗi bước ghi `gate: {opener, gap, override}` trong trace_out (UI đọc được).
- **API mới**: `GET /api/hlinks?token=nếu` (closer/PMI/power/mode_gap+%/gap-hist/3 ví dụ cha
  nguồn), `GET /api/hlinks/info` (48 448 edges, 48 448 with_meta). `viz/export_hlinks.py`.
- **UI**: tab Embedding E → panel **“Cạnh ngang xa (TK_020)”** — gõ opener → lưới thẻ closer
  (PMI · power · gap d:count · ví dụ « cha nguồn »). `npm run build` xong (index-Du1ykV6r.js).
- Verify sống: `nếu→thì` PMI 2.25 **power 89** mode_gap 8 + ví dụ "« …nếu chúng quả quyết đến
  tấn công , thì nhà… »"; sinh API `nếu trời mưa thì` → "…đây quả là một trường tai nàn đáng
  sợ ." (2.4s).
