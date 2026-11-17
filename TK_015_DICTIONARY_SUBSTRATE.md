# TK_015 — Từ điển làm substrate có nhãn (POS/nghĩa/đồng-trái nghĩa + khung xa)

| Field | Value |
|-------|-------|
| **ID** | TK_015 |
| **Status** | 🟡 SPEC + **bước 0 XONG** (thước LM không bão hoà, Claude 2026-07-05) — chờ tác giả duyệt D1+ |
| **Nguồn** | Chỉ đạo tác giả 2026-07-05: nạp từ điển tiếng Việt thật (mỗi từ có POS, từ đơn/đa nghĩa, đồng/trái nghĩa, định nghĩa, ví dụ) làm data train → nhiều điểm NEO; dùng nhiều nơ-ron để neo thay vì fact; link ngang KHÔNG kề nhau (`nếu→thì`, `Hỏi:→Đáp:`). Hoàn toàn dựa mạng + data, KHÔNG hardcode rule. |
| **Doc** | [`docs/study/08_DICTIONARY_LABELED_SUBSTRATE.md`](docs/study/08_DICTIONARY_LABELED_SUBSTRATE.md) (thiết kế trọn vẹn, D1–D4) |
| **Checkpoint** | probe C0 trên `grade_v3_big_a2_t3_isg.pkl`; guard `grade_v3.pkl` |

## 1. Tổng quan

Substrate cấu trúc rỗng (v4 W2): máy sinh chỉ có `W_seq` bigram + `ar_index_backbone`
ép-verbatim → sinh neo 1 fact, ghép không mượt, đa-nghĩa nuốt nghĩa, synonym trượt. Từ điển
= nguồn **NHÃN cấu trúc** (POS, lớp từ, đồng/trái nghĩa, ví dụ = khung ngữ pháp), nạp qua
đúng pipeline train hiện có → dựng lại substrate W2 bảo là rỗng. **Luật 0 sạch tuyệt đối**
(data train, không rule); kỷ luật duy nhất = Luật 0.0 (inference đọc neo như nơ-ron, không
`if pos==VERB`).

## 2. Hiện trạng chi tiết

- **A3 sinh** ([`generate_hebbian.py:477-524`](src/nn_aivie/hca/generate_hebbian.py)): 4 kênh
  Hadamard (micro=W_seq bigram, macro=spread agenda, discourse, entity) + `ar_index_backbone`
  ép token agenda theo thứ tự nguồn. Không có Q/K có hướng; "attention" duy nhất =
  `entity_attention_read` (readout ĐỐI XỨNG — doc 03 §1 nói đó chỉ là fuzzy cosine).
- **ISG** ([`seg_graph.py::_register_bigram`](src/nn_aivie/core/seg_graph.py)): chỉ promote
  bigram **liền kề**. Không có cạnh ngang xa.
- **Tường "cần POS"** lặp trong nhật ký: anaphor P2 âm tính (v4 §7.3), q·k nhiễu function-token
  (doc 07), synonym 50/17/33% (v4 §0.5.6), đa-nghĩa T0 âm tính (v4 Trụ 7 T0).
- **Đã có sẵn dùng lại được:** `add_example` (ingest), `EdgeKind` (thêm layer), sense/anchor
  cùng loài cluster/theme/hub, `content_cache` (TK_012) cho text mục từ.

## 3. Đề xuất giải pháp

Bốn trụ (chi tiết doc 08 §3), thứ tự đòn bẩy **0 → D1 → (D2 ‖ D3) → D4**:

- **D1 (từ điển = neo):** parser mục từ → chùm neo `{lemma, sense, POS, attr, SYN, ANT, DEF,
  EX}`; hub POS gom hàng nghìn từ. Gỡ tường anaphor (pronoun-gate `pos==PRONOUN`) + synonym
  (cạnh SYN tường minh) + đa-nghĩa (sense-neuron riêng hồ sơ).
- **D2 (Q/K có hướng — doc 03):** SVD PPMI-có-hướng của W_seq/fact_links → Wq/Wk; head tách
  theo POS (function-token riêng). Thay `ar_index_backbone`.
- **D3 (link ngang xa — ý ⑦):** cạnh `FRAME_BRACKET` = PMI có hướng cặp không-kề trong cha,
  gate POS (đầu = liên-từ/tiểu-từ/frame). `nếu→thì`, `Hỏi:→Đáp:`.
- **D4 (sinh = đọc stack neo — doc 05):** H_slow/mid/fast; agenda-câu chỉ là 1 neo trong nhiều.

## 4. Đề xuất thay đổi cụ thể

| File | Thay đổi |
|------|----------|
| `scripts/ingest_dictionary.py` (mới) | parser mục từ semi-structured → `add_example` (def/ex) + cạnh SYN/ANT/POS/HAS_ATTR/DEF/EX |
| `core/edge_kinds.py` | `EdgeKind.{SYNONYM, ANTONYM, POS, HAS_ATTR, DEF, EX, FRAME_BRACKET}` |
| `core/parent_store.py` | `SenseNeuron`/`AnchorNeuron(kind=pos/attr)`; serialize; `senses_of(lemma)` |
| `core/seg_graph.py` (D3) | quét cặp không-kề gate POS → promote `FRAME_BRACKET` |
| `hca/attention_qk.py` (mới, D2) | SVD có hướng → Wq/Wk view; head theo POS/layer |
| `hca/generate_hebbian.py` (D4) | stack neo H_slow/mid/fast thay `ar_index_backbone` |
| `hca/config.py` | cờ ablate `dict_*` / `qk_*` / `frame_bracket_*` / `anchor_stack_*` (mỗi kênh =0 tắt) |
| `eval/v4.py` (bước 0) | mở held-out generation + metric nghiêm (điều kiện tiên quyết) |
| `data/quality.py` | gate nhiễu crawl từ điển |

## 5. Lộ trình đo (acceptance)

**Bước 0 (BẮT BUỘC trước mọi thứ):** held-out generation ~200 câu + IDF-metric + `verbatim_rate`.

**C0 (probe rẻ, mở D1):** nạp tay ~15 mục (gồm "là","chân","đông") vào `_t3`; đếm attach;
đọc tay sense tách nghĩa. Acceptance: ≥70% mục hút ≥3 cha/ví dụ đúng nghĩa; ≥1 từ đa nghĩa
tách ≥2 sense có hàng-xóm khác rõ; mục "chết" ghi rõ.

**D1→D4:** mỗi bước A/B, giữ nếp "đo dương mới giữ, âm tính vào sổ nợ (không tắt)" (v4 §3.5).
Guard: `grade_v3` + ISG không tụt; retrieval crisp-QA đóng băng.

## 5b. KẾT QUẢ ĐO — bước 0 (2026-07-05, Claude)

**Vấn đề W1:** held-out hiện tại QA/Para/Syn = **7/8/6 dòng → bão hoà** (mọi cải tiến đọc ra
"trung tính 100/100"). `bits_per_byte` cũ (`eval/metrics.py`) là **proxy GIẢ**: p=0.85 nếu
rank-1 else 0.05 (hit-rate nhị phân đội lốt bits) + gọi `generate()` mỗi vị trí (chậm).

**Làm:** thước **KHÔNG bão hoà** `next_token_lm` (`eval/metrics.py`) — log2(rank) token gold
trong phân bố base-LM (W_seq ⊙ readout, KHÔNG agenda) trên corpus held-out thật. + MRR + top-k.
Cùng họ đo A0 doc 03 → tái dùng cho D2. Runner: `scripts/eval_lm_ruler.py`. Held-out =
200 câu `essays.jsonl` (vi-alpaca) **không nằm trong** `big_a2` (build trước essay _t2), persist
`data/eval_v4/heldout_lm.jsonl` (ổn định qua các lần chạy).

| Checkpoint (cùng 200 câu held-out) | bits ↓ | MRR ↑ | top-1 | top-5 | top-10 | vị trí |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `grade_v3_big_a2.pkl` (KHÔNG train essays) | 5.041 | 0.241 | 18.2% | 29.7% | 37.5% | 2814 |
| `grade_v3_big_a2_t3_isg.pkl` (CÓ train essays) | **4.028** | 0.300 | **22.1%** | 38.3% | 47.6% | 2860 |

**Xác thực ruler:** checkpoint CÓ train essays đạt Δbits **≈ −1.0** / Δtop-1 **+3.9pp** trên
held-out essay → thước **nhạy + đúng hướng** (train liên quan hơn ⇒ bits thấp hơn), và
**top-1 chỉ 18–22% = KHÔNG bão hoà** (đối lập QA 100%). Đây là cây thước để theo DELTA qua
D1–D4. `python -m pytest tests/test_eval_v4.py` = 9 pass.

**Sổ nợ bước 0:** (a) leakage — số tuyệt đối sạch cần reserved-shard lúc train (TK_011 perf);
hiện dùng theo DELTA trên corpus cố định (đủ). (b) chưa mở held-out QA/Para/Syn có nhãn lớn
(cần gold — data task, Cursor/tác giả). (c) chưa wire vào `scripts/eval_v4.py` (đang standalone).

## 5c. KẾT QUẢ ĐO — C0 probe (2026-07-05, Claude)

Xác nhận 2 bức tường D1/D3 nhắm **CÓ THẬT + thấy được** trên `grade_v3_big_a2_t3_isg.pkl`
(read-only, `scripts/probe_c0_walls.py`) — điều kiện mở D1/D3 (đo trước khi build).

**Tường 1 — ĐA NGHĨA BLUR (D1 sense-neuron nhắm) — CONFIRMED, 2 kiểu hỏng:**
- *Nghĩa trội NUỐT:* `đông` → hàng xóm CHỈ hướng {bắc,nam,tây,phía,vùng,miền} — mất sạch
  nghĩa "đông đúc/mùa đông/đông lạnh". `là` → chỉ {gọi,một,trong,của,có} (copula); `chạy` →
  chỉ {trốn,đuổi,nhanh} (run), mất "vận hành".
- *Nghĩa TRỘN:* `đá` → {bóng,Sheffield,Moyes}(đá bóng) + {vôi,thạch,băng}(đá/khoáng) lẫn nhau.
- **Vật liệu cure tách được (số chốt):** `sim(đá,bóng)=0.75`, `sim(đá,vôi)=0.74` — cả 2 nghĩa
  ôm sát "đá" — **NHƯNG `sim(bóng,vôi)=0.28`** → vector "đá" là trung bình BỊ TRỘN của 2 cụm
  tách rõ. Sense-neuron (mean-pool ví dụ riêng mỗi nghĩa từ từ điển) sẽ rơi vào cụm bóng vs
  cụm vôi = tách. (Tương tự `chạy`: sim(trốn,nhanh)=0.12.)

**Tường 2 — THIẾU CẠNH XA (D3 FRAME_BRACKET nhắm) — CONFIRMED sạch:** cặp khung không-kề có
`seq(a→b)=0.00` (bigram kề = 0, KHÔNG BAO GIỜ liền nhau) trong khi `co` (đồng-hiện cửa sổ) cao:

| Cặp | seq (kề) | co (cửa sổ) | emb_sim |
| :--- | :---: | :---: | :---: |
| nếu→thì | 0.00 | 5.25 | 0.60 |
| vì→nên | 0.00 | 5.70 | 0.70 |
| Hỏi:→Đáp: | 0.00 | 20.28 | 1.00 |
| càng→càng | 0.00 | 21.90 | 1.00 |

Substrate **biết** chúng đồng-hiện (co cao, đối xứng) nhưng kênh sinh dùng `W_seq` (=seq) =
**0.00** → lúc sinh, sau "nếu" KHÔNG có tín hiệu nào kéo tới "thì". Đúng gốc "ghép/chuyển không
mượt". D3 = vật chất hoá cạnh XA có hướng cho đúng ràng buộc này.

**Verdict C0: ĐÈN XANH cho D1 (sense-neuron) + D3 (frame bracket)** — bệnh thật, khu trú được,
vật liệu cure tách được. Acceptance C0 (≥1 từ đa nghĩa tách ≥2 cụm) đạt (đá, chạy).

## 5d. KẾT QUẢ ĐO — D1 (2026-07-05, Claude) — CƠ CHẾ DƯƠNG, FLAT-DUMP ÂM TÍNH

**Làm:** `training/ingest_dictionary.py` + `EdgeKind.{SENSE,POS_ANCHOR,SYNONYM,ANTONYM,LEXATTR}`
+ prefix vocab `Sense:`/`POS:` + `add_example(defer_index=)` (vá perf: index rebuild/câu = O(#cha)
→ chết; batch begin/end) + `concept_meanpool` 1-pass (vá O(#concept×#cạnh)). Nạp **36 764 nghĩa /
26 245 từ** (HF `tsdocode/vietnamese-dictionary`) vào `_t3_isg` → `_dict.pkl`: vocab 34.5k→**77.5k**,
parents 44.6k→**115.5k**, 14 POS-hub. `scripts/{fetch_vi_dictionary,build_dict_checkpoint,
probe_d1_senses}.py`.

**① Cơ chế TÁCH NGHĨA — DƯƠNG (đúng dự đoán C0):** sense-neuron nhận vector RIÊNG mỗi nghĩa
(Jaccard neighbor≈0). Vài nghĩa SẠCH + đúng: `cân#3`→{1/16, gram, kilogram, lạng, 37,8} (cân
nặng ✓✓), `đá#1`→{hòn, tảng} (khoáng), `đông#1`→{bắc, tây}(hướng), `cổ#3`→{áo, giày}(cổ áo). ⇒
substrate CÓ THỂ biểu diễn nghĩa tách rời — phá tường C0 ở mức substrate.

**② Chất lượng — DATA-LIMITED (âm tính trung thực):** ĐA SỐ sense-neuron NHIỄU/THƯA — nhiều nghĩa
0–1 hàng xóm, hoặc dính danh-từ-riêng/số wiki ngẫu nhiên (`đông#3`→Catholique/Isabelle). Gốc: từ
điển cho **1 định nghĩa + ~1 ví dụ/nghĩa** → sense-node quá ít co-activation → PPMI bị token hiếm
ngẫu nhiên chi phối. Jaccard≈0 **KHÔNG** phải "tách hoàn hảo" — phần lớn vì neighbor-set quá nhỏ
(đã cảnh báo doc 08 §6). split_punct giảm nhiễu dấu-câu nhưng không cứu độ thưa.

**③ Ruler (essay held-out) — ÂM TÍNH:** flat-dump từ điển làm **TỆ ĐI**:

| Checkpoint | bits ↓ | top-1 | top-5 |
| :--- | :---: | :---: | :---: |
| `_t3_isg` (baseline) | **4.028** | **22.1%** | **38.3%** |
| `_dict` (flat-dump 36k nghĩa) | 4.400 | 20.0% | 33.3% |

**Chẩn đoán = LẶP LẠI bài học README 0.5.8** ("dump data ngoài miền thô làm TỤT benchmark trong
miền"): nhồi 70k câu định-nghĩa (miền từ-điển) + gấp đôi vocab vào substrate SINH → pha loãng
next-token essay (miền khác) + tăng cạnh tranh ứng viên.

**VERDICT D1 — REDIRECT KIẾN TRÚC (giá trị thật):** KHÔNG flat-dump từ điển làm nơ-ron cha trong
substrate sinh. Từ điển phải là **tầng NEO/ĐỊNH TUYẾN riêng, bind vào corpus**:
1. sense-neuron **bind bằng resonance vào occurrence trong corpus** (nhiều ví dụ thật) thay vì chỉ
   1-2 ví dụ từ điển → hết thưa (máy `bind_fact_links`/resonance đã có — doc 06 ②).
2. định nghĩa/ví dụ từ điển KHÔNG vào pool retrieval sinh essay (two-tier như 0.5.10) — tránh dilution.
3. lever chưa test: **POS-hub cho anaphora/grammar** (14 hub đã tạo) + **cạnh SYN/ANT** (tsdocode
   THIẾU → cần WordNet/ViCon) — đo riêng, không qua ruler-essay.

Guard: 83 test lõi (embedding/parent/seg/eval/vocab) pass sau vá. `_dict.pkl` = artifact probe.

## 6. Sổ nợ / phase sau

- **Routing sense per-occurrence** — substrate có, runtime disambiguation là việc riêng (D2 q·k
  ngữ cảnh / spreading). Không hứa ở D1.
- **Độ phủ từ điển** = trần synonym mới (nạp thêm data, không nới luật).
- **Vocab phình** nếu tách sense mọi từ → chỉ tách từ "đa nghĩa".
- **Nguồn từ điển** — cần chốt (HF / Wiktionary VI / crawl); parser theo format.
- **D2 function-token** — doc 07 §8 cảnh báo q·k nhiễu; probe token-đầu-đáp trước khi build.
</content>
