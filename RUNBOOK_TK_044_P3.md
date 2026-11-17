# RUNBOOK TK_044 P3 — cắm binding vào ĐỒ THỊ THẬT (cho M4)

**Mục tiêu:** đưa induction-head Hebbian (đã chứng minh P0/P2/P1 trên synthetic, 100%/6-seed) vào
`generate_autoregressive` của graph thật — **phép thử QUYẾT ĐỊNH**: binding có sống với BPE token-vụn
+ nhiễu + đa nghĩa của corpus thật không, hay chạm tường. Nếu sống → mở generation copy-biến (nền cho
văn phong P4) trên nền Luật-0 sạch. Nếu gãy → biết gãy ở đâu (đo, không đoán).

**Nền:** POC `scripts/poc_hebbian_binding.py` (P0), `poc_binding_stress.py` (P2),
`poc_binding_continuation.py` (P1). Đọc TK_044 §3–5.6 trước. Cơ chế = attention {Q,K=E(category),
V=token identity (copy), routing=cạnh coreference} — Hopfield=attention, trọng số tích-ngoài Hebbian,
KHÔNG backprop (Luật 0 chỉ cấm backprop-attention).

---

## 0. Tiền đề & kỷ luật

- **Luật 0:** không hardcode từ-vựng/nghĩa tiếng Việt. Coref + role-backoff học Hebbian (đếm-1-lần)
  lúc ingest. Nhận thực thể MỚI qua NGỮ CẢNH (W_seq/entity-register), KHÔNG qua bảng tên.
- **Luật 0.0:** quyết định sinh KHÔNG đọc content-cache. Copy dùng token-id + cạnh, không đọc text.
- **Measure-first:** mỗi nấc một số đo + cổng. Không nấc nào xanh thì DỪNG, báo số, không đắp tiếp.
- **Observability (bắt buộc):** mọi pass dài (ingest coref, A/B) ghi progress ra file `out/*.log`
  (gồm lúc nạp checkpoint). Xem [[observability-logging]].
- **Nạp checkpoint:** `gc.disable()` quanh `GraphState.load` (1300× — [[checkpoint-load-gc]]).
- Checkpoint dùng: graph v11 mới nhất (cùng cái grounding-fix đã chạy). Ghi rõ đường dẫn + fingerprint
  vào log.

## 1. Bản đồ POC → graph thật (KHÔNG code lại — dùng cấu trúc CÓ SẴN)

| POC (numpy) | Graph thật | File |
|---|---|---|
| `category = cluster E(PPMI+SVD)` / centroid-cosine | E đã có: PPMI trên `CO_ACTIVATION` → SVD | `core/embedding.py` |
| `binds` (thực thể trong prompt) | `ps.entity_register_ids(q_content)` | `parent_store.py:1628` |
| `slot_after_is_name` (anchor dự-đoán-tên) | W_seq = `EdgeKind.SEQUENTIAL` (bigram a→b) | `edge_kinds.py`, đọc row-normalize |
| `a_entity` attention (Q·K entity) — ĐÃ CÓ | `entity_attention_read` trong A3 | `generate_hebbian.py:1288` |
| **cạnh coreference** (MỚI) | thêm `EdgeKind.COREFERENCE`, add Hebbian lúc ingest | `edge_kinds.py` + `hebb.py` mẫu `update_from_coactivation` |
| **op copy-slot** (MỚI) | kênh trong `generate_autoregressive` cạnh a_entity | `generate_hebbian.py:1268` |
| **role-backoff** `after_role[sig]` (MỚI) | bảng {slot-signature → token cấu-trúc kế}, đọc khi w0 là thực-thể vừa-copy (OOV → W_seq rỗng) | mới, lưu trong state |

**3 bài học P2/P1 = bản đồ chỗ dễ gãy trên graph:** (a) **category-purity** — token hi-df kề tên
(dấu câu/đại từ) lọt cụm → lọc theo TẦN SUẤT (df), không nghĩa; (b) **nhận-OOV-qua-ngữ-cảnh** —
thực thể mới không có E, bind qua anchor; (c) **signature slot đủ RỘNG** bắt verb-frame; (d) **đại từ**
= outlier df-cao ở vị-trí-tên, mask khỏi co-occurrence + mượn cạnh coref của tên. Graph E sạch hơn
synthetic (data lớn) nên (a) nhẹ hơn, nhưng BPE token-vụn + nhiễu là thật → ĐO.

---

## 2. Nấc P3.0 — HARNESS A/B TRƯỚC (đo binding graph HIỆN TẠI = ~0)

**Không đụng lõi.** Dựng thước đã, để mọi thay đổi sau có số so.

1. Tạo `data/sft/bind_probe.jsonl` (~40–60 mục): thực-thể MỚI (chưa trong graph) điền vào khuôn
   đã học — single (`"{X} … , {X} …"`) + crossed (`"{X} tặng {Y} …, {Y} … {X}"`) + đại từ
   (`"{X} …, nó …"`). X/Y = tên/danh-từ KHÔNG xuất hiện trong corpus train (để đo binding thật,
   không phải nhớ). Gold = token(s) phải copy ở mỗi slot.
2. Viết `scripts/probe_binding_graph.py`: nạp graph (gc.disable + log), với mỗi prompt gọi
   `generate_autoregressive`, chấm **slot-copy** (token sinh ở vị-trí-slot == gold?) + **drift**
   (có copy nhầm thực-thể train nào không). In bảng single/crossed/đại-từ %. Ghi `out/probe_bind_P30.log`.
3. **Kỳ vọng nền ≈ 0** (graph chép/mosaic phát thực-thể TRAIN, không bind biến mới) — xác nhận baseline,
   giống P0 baseline-bigram. **Cổng:** harness chạy + số nền ổn định 2 lần → sang P3.1.

## 3. Nấc P3.1 — cạnh COREFERENCE (Hebbian, lúc ingest)

1. Thêm `EdgeKind.COREFERENCE` vào `edge_kinds.py` (đọc kênh RIÊNG, priority 0 — KHÔNG cộng readout
   structural, giống SEQUENTIAL/SENSE).
2. Pass ingest (đặt cạnh chỗ populate W_seq/CO_ACT): mỗi câu, lấy các **occurrence thực-thể** (span
   E-category, nhận qua entity-register/context), tính **slot-signature** = W-token cấu-trúc trái (SIG
   rộng, tên→`<ENT>`); với mỗi CẶP occurrence đếm Hebbian "cùng filler trong câu" → `adjacency.add(sig_a,
   sig_b, eta, COREFERENCE)`. **Purity:** bỏ token df>0.5·N khỏi vai-tên; đại từ (outlier df-cao vị-trí-tên)
   → mask + đánh dấu biến-coref. Tie-break TẤT ĐỊNH (đừng lặp qua set chuỗi — [[variable-binding-poc]]:
   PYTHONHASHSEED làm chập chờn).
3. Đồng thời build **`after_role[sig]`** = Counter token cấu-trúc theo sau mỗi slot-signature (role-tracking).
   Lưu vào state (serialize cùng checkpoint).
4. **Đo (chưa đụng sinh):** dump vài cạnh coref học được — có khớp khuôn không (single: comma-slot↔chủ-ngữ;
   crossed: `,↔tặng`=Y, `<S>↔ơn`=X)? Log `out/coref_edges_P31.log`. **Cổng:** cạnh đúng khuôn xuất hiện
   (đọc tay 10 cạnh) → P3.2. Sai → dừng, xem category/signature (bài học P2 #a/#c).

## 4. Nấc P3.2 — kênh COPY-SLOT trong A3

1. Trong `generate_autoregressive`, mỗi bước: phát hiện **copy-slot** = w0 (anchor) mà W_seq dự-đoán
   **E-category-thực-thể nhiều-filler** (đa dạng ≥ ngưỡng) — giống `slot_after_is_name`. Tận dụng
   `a_entity`/`entity_attention_read` sẵn có (đã là Q·K entity).
2. Ở copy-slot: signature hiện tại → cạnh `COREFERENCE` → antecedent slot → `entity_register_ids` cấp
   filler → **copy** token-id đó (V = identity). Đây là 1 kênh MỚI nhân vào `a_final` (hoặc override khi
   copy-mass vượt sàn), KHÔNG thay W_seq fluency.
3. **role-backoff:** khi w0 là thực-thể VỪA copy (OOV → W_seq(w0) rỗng/yếu) → next = argmax
   `after_role[last_sig]` (token cấu-trúc theo vai), tránh tắc. Xen 3 kênh: copy-slot → role-backoff →
   W_seq (đúng thứ tự POC P1).
4. **A/B:** chạy lại `probe_binding_graph.py` (P3.0 harness) BẬT/TẮT kênh copy (flag config). Log
   `out/probe_bind_P32.log`. **Cổng:** single-copy ↑ rõ so nền ~0 + **drift không tăng** + **guard
   essays Δbpb ≤ +0.05** (không hại fluency/không quên). Đạt → P3.3. Không → báo số, xem gãy đâu.

## 5. Nấc P3.3 — crossed + đại từ + regression

1. Bật crossed (cạnh coref chéo) + đại-từ (mượn cạnh coref của tên). Đo cả 3 loại trên harness.
2. **Regression bắt buộc:** `pytest -m "not slow"` (4s — [[running-tests]]) + probe grounding cũ
   (`probe_retrieval_compose_v11`) KHÔNG tụt (binding không được phá grounding đã vá) + guard essays.
3. **Cổng chốt P3:** crossed-copy > nền + đại-từ giải đúng tiền-ngữ > nền + 0 regression grounding/tests
   + guard sạch. Log `out/probe_bind_P33.log` + cập nhật TK_044 §8.

---

## 6. Trần trung thực (ghi trước, kẻo tự lừa)

- Synthetic 100% KHÔNG bảo đảm graph thật 100%. Kỳ vọng thực tế: **single-copy sống** (dễ nhất), crossed
  + đại từ **có thể tụt** do BPE token-vụn/đa nghĩa làm category+signature nhiễu — đó là SỐ cần đo, không
  phải thất bại. Nấc nào tụt = chỉ đúng chỗ E/signature cần khỏe hơn.
- Binding token-copy ≠ văn phong (P4). P3 là NỀN, không tự cho style.
- **Kill-criteria:** nếu single-copy cũng ~0 sau khi cạnh coref đã đúng khuôn (P3.1 xanh) → cơ chế không
  chuyển được sang BPE thật; DỪNG, viết negative vào TK_044 §7, KHÔNG đắp heuristic che.
- Đừng để harness "tự khớp": thực-thể probe phải NGOÀI corpus train (nếu không = đo trí nhớ, không phải bind).

## 7. Thứ tự chạy (checklist)

```
[ ] P3.0  bind_probe.jsonl + probe_binding_graph.py → số nền ~0 (out/probe_bind_P30.log)
[ ] P3.1  EdgeKind.COREFERENCE + ingest pass + after_role → dump cạnh đúng khuôn (out/coref_edges_P31.log)
[ ] P3.2  kênh copy-slot + role-backoff trong A3 → A/B single (out/probe_bind_P32.log) · cổng: single↑, drift↓, guard sạch
[ ] P3.3  crossed + đại từ + regression (pytest, grounding, guard) → cổng chốt (out/probe_bind_P33.log)
[ ] cập nhật TK_044 §6/§8 + memory [[variable-binding-poc]] với SỐ THẬT
```

Mọi nấc: dán nguyên stdout/log vào báo cáo, đừng tóm tắt số.
