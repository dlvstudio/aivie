# TK_003 — Anchor họ-người phá chọn đại từ + refresh gold composition (2 việc tách từ TK_002)

| Field | Value |
|-------|-------|
| **ID** | TK_003 |
| **Title** | Anaphor: anchor là họ/tên người → tín hiệu vị ngữ sai; Gold: cập nhật ca curated-swap |
| **Status** | **DEFERRED 2026-07-02** — tác giả hoãn: sẽ có giải pháp nối câu riêng sau này; không làm gate/anchor tiếp |
| **Repo** | `nn-aivie`, nhánh `cursor2` |
| **Nguồn** | Chẩn đoán trong TK_002 mục 7 (2026-07-02) |
| **Ưu tiên** | Thấp hơn T2 essay — làm khi quay lại chất lượng anaphor |

---

## 1. Vấn đề A — anchor họ-người (chẩn đoán ĐÃ ĐO, chưa có fix đạt)

`best_anaphor_for_entity` neo vào token thực thể phân biệt nhất — với "Nhà Trần"/"Lê Lợi",
anchor là **họ/tên người wiki** ("Trần", "Lợi"). Successor profile của họ-người = **tên đệm
viết hoa** (Trần→{Văn .09, Quốc .08, Hưng .06, "," .04, Trọng .04} — đo trên big_a2), KHÔNG
phải vị ngữ chủ ngữ:

- Gate `subj_overlap` (top-5 successor của anchor) cut đại từ đúng ("họ" ov=0.025 < 0.06);
- ứng viên "pass" còn lại là token tình cờ chia successor với TÊN ("lý" pass vì "lý Thái…";
  "sau" pass qua dấu "," trong top-5) → rác thay "họ".

**3 biến thể đã thử & ĐO TỆ HƠN (đừng lặp lại):**
| Biến thể | Kết quả composition big_a2 (top) |
|---|---|
| ent_top chỉ giữ successor viết thường | 83→69 (rác "hình/sống/sản") |
| + loại dấu câu khỏi ent_top | 72; mechanism ra "lý/sinh/lần" |
| Bản giữ lại (TK_002): gate bản-83 + ứng viên phải viết thường | **83** ✓, anaphor 50% |

**Hướng cho lần sau (chưa thử):**
1. **Phát hiện anchor họ-người bằng cap-share**: nếu ≥50% khối lượng successor của anchor là
   token viết hoa → anchor không mang tín hiệu vị ngữ → **trả None sớm** (giữ chủ ngữ, an
   toàn) thay vì để rác thắng. Thuần chính tả (Luật 0). Rủi ro cần đo: mất "họ" ở ca đang
   đúng (Nhà Lý/Nhà Trần → mechanism hiện ra "họ"/"sau" tùy anchor).
2. **Anchor bằng CỤM thay token**: successor của cụm "Nhà Trần"/"Lê Lợi" (qua Re-Pair phrase
   hoặc last-token-of-mention) là vị ngữ thật ("Nhà Trần đánh/thay…") — đúng gốc hơn mọi gate.
   Cần W_seq mức cụm hoặc tra successor tại các vị trí mention trong cha.
3. Treo pref đại từ lên **ThemeNeuron/t1** khi P2-lần-2 khả thi (host đã có).

**Acceptance:** composition big_a2 on_topic ≥ 83 (không tụt), anaphor mechanism ≥ 50% và
KHÔNG còn pick ngoài nhóm đại từ thật trên 13 ca gold (nó/họ/chúng/ông/đó/đây/giữ-chủ-ngữ);
"sản/các/lý/sau/hình" = fail. grade_v3 nguyên vẹn.

## 2. Vấn đề B — gold composition 2 ca lỗi thời sau case-equiv

`case_equiv_ids` (TK_002) làm focus "Trái Đất"/"Mặt Trời" truy được **fact curated** ("Trái
đất quay quanh mặt trời", "Mặt trời cung cấp ánh sáng và nhiệt") — đáp thật TỐT LÊN (on_topic
100% cả 2 ca) nhưng gold n=13 khóa theo wiki fact cũ → cov đo 33%/0% oan (kéo tổng 69→64).

**Việc:** cập nhật `data/grade_03/qa/composition.grade_v3_big_a2.jsonl` 2 ca này — group ý
viết theo **fact tồn tại trong checkpoint mà hệ NÊN trả** (curated ưu tiên; giữ mức khó tương
đương, không hạ chuẩn để ăn điểm — mỗi ca vẫn ≥3 group). Ghi rõ trong nhật ký v4 rằng đây là
sửa-gold-vì-đáp-đúng-hơn, kèm before/after. Kỳ vọng cov về ≥69.

## 3. Vấn đề C (ghi nhận) — "sản"/"các" trên đường composed

Cùng gốc vấn đề A (anchor "Nguyễn"). Fix A xong thì đo lại; nếu còn → nằm ở tường POS
(inventory đại từ sạch — P2 lần 2, §7.3 docs).
