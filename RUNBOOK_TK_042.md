# RUNBOOK TK_042 — chạy trên máy mạnh

> Máy dev hiện tại là **MacBook Intel 8GB** (TK_040 M0 ghi *"24GB"* — **SAI**, đã cắn 2 lần).
> Ba probe dưới đây **treo hoặc quá chậm** ở đó. Chúng độc lập nhau, chạy thứ tự nào cũng được.
>
> **Kết quả cần mang về: dán nguyên văn stdout.** Đừng tóm tắt — mấy bảng này để **đọc tay**,
> và cả ticket này đã bị nhiều lần đo sai vì tin tóm tắt thay vì đọc số.

---

## 📍 TRẠNG THÁI HIỆN TẠI (2026-07-20) — đọc TRƯỚC

| | việc | trạng thái |
|---|---|---|
| ✅ | **G19c** (BƯỚC 13): 42M v10 = **5.843** > 11.4M **5.594**, gap +0.962 | XONG → **bức tường DATA, chốt bằng số** (thêm tham số không giúp) |
| ✅ | QA held-out `data/sft/qa_heldout.jsonl` (278: 250 viquad_val + 28 grade), decode polish, SFT recipe | XONG (SFT recipe lành: guard +0.001; nhưng sinh rác vì base yếu) |
| 🔄 | **BƯỚC 10c — 42M trên v11 180M, eval qa_heldout** — `out/run_step10c_v11.sh` | **M4 ĐANG CHẠY (đường LM)** ~9h. Câu quyết định: thêm DATA có nâng QA không |
| ⏳ | **BƯỚC 16 CulturaX** v12 ~750M — `out/run_culturax.sh` | CHỜ 10c thắng rồi mới chạy |
| ✅ | **BƯỚC 10d — v11 + repeat WEIGHTS** — `out/run_10d_v11_weighted.sh` | **WEIGHTING ĂN**: qa_heldout answer-only v11w **4.821** < v10 **4.939** < v11-uniform **5.791**. Thêm data GIÚP khi có trọng số. gap +0.307 (vs uniform +1.464) |
| 🔄 | **SFT trên v11w** (base tốt hơn → QA sinh khá hơn?) — `out/run_sft_v11w.sh` | **CHỜ CHẠY** — replay dùng `train_v11_weighted.yaml` |
| ⏳ | **CulturaX v12** giờ ĐÃ justified (weighting cứu → data lớn hơn+weighting) | sau SFT-v11w xác nhận hướng |
| ⏳ | SFT lại trên base mạnh (v11/v12) | CHỜ base tốt hơn |
| ⏳ | Probe sinh tự do đồ thị — `scripts/eval_free_gen.py` | ĐỘC LẬP (đường đồ thị), chạy lúc nào cũng được |
| 🔄 | **GROUNDING — vá trôi entity (đường ĐỒ THỊ)** → xem **`RUNBOOK_GROUNDING.md`** | Fix pushed (efc55b0+6ee491e, 4/5 local); **M4 chờ confirm scale = `retrieval_compose`, KHÔNG probe_grounding (hang)** |

**Đường tới hạn hiện tại = LM data-scale:** 10c (đang chạy) → nếu thắng → CulturaX → SFT. Đọc
**bits/token qa_heldout answer-only** (không phải essays) là số quyết định. Chi tiết mỗi bước dưới.

---

## 🎯 BƯỚC 13 (G19c) — ✅ XONG (5.843 > 5.594, bức tường DATA) — PHÉP SO SẠCH 42M vs 11.4M trên v10

Sau HAI vòng confounded (lần 1 bug Muon LR; lần 2 sai corpus viwiki), đây là lần đầu so
**apples-to-apples**: 42M v2+Muon trên **corpus v10 y hệt** 11.4M đã dùng (19.7M token), cùng
held-out essays. Biến khác chỉ còn: kích thước (42M vs 11.4M) + kiến trúc (v2 vs v1).

**Kỷ luật: bf16 CHƯA test — không nhét vào run kết luận khi chưa smoke.** Làm 2 bước:

```bash
cd <repo> && git pull && export PYTHONPATH=src

# BƯỚC A — smoke bf16 (~5 phút): kiểm bf16 chạy + loss GIẢM + tok/s nhanh hơn fp32.
python -u scripts/train_neural_lm.py --arch v2 --opt muon --bf16 --batch 48 \
  --manifest config/train_v10.yaml --d-model 512 --layers 8 --heads 8 --ctx 512 \
  --steps 200 --eval-every 0 --log-file out/smoke_bf16.log 2>&1 | tail -20
#   ✓ nếu loss tụt đều (10→8→7…) + không NaN ⇒ bf16 OK, dùng --bf16 ở bước B.
#   ✗ nếu NaN/loss nổ/lỗi ⇒ BỎ --bf16 ở bước B (chạy fp32, chậm hơn nhưng sạch).

# BƯỚC B — phép so thật (~2.5-3h bf16 / ~5h fp32):
python -u scripts/train_neural_lm.py --arch v2 --opt muon --bf16 --batch 48 \
  --manifest config/train_v10.yaml --d-model 512 --layers 8 --heads 8 --ctx 512 \
  --steps 15000 --eval-every 500 --save out/lm_v2_v10.pt --log-file out/lm_v2_v10.log
git add -f out/lm_v2_v10.log out/smoke_bf16.log && git commit -m "TK_042: G19c so sạch v10" && git push
```

**Đọc kết quả (điều kiện phân định GHI TRƯỚC):**

| eval TỐT NHẤT vs **5.594** (11.4M) | gap train/eval | ⇒ kết luận |
|---|---|---|
| **thắng rõ** (< 5.4) | ≥ 0.5 | 42M hơn 11.4M trên cùng data ⇒ **kích thước có đất**, tiếp data quý (CulturaX) |
| **≈ 5.594** (± 0.1) | ≥ 0.5 | 19.7M là **bức tường DATA** — cả hai bão hoà; muốn tiến phải thêm data quý, không phải tham số |
| **thua** (> 5.7) | ≥ 0.5 | 42M **overfit** 19.7M (thừa tham số cho data ít) ⇒ vẫn là bức tường data |
| bất kỳ | **< 0.5** | UNDERFIT — chưa kết luận, tăng step |

Dự đoán (ghi trước để đối chiếu): nhiều khả năng **≈ hoặc thua** — vì 19.7M nuôi tối ưu ~1M
tham số (Chinchilla), 11.4M đã thừa, 42M càng thừa. Nếu đúng ⇒ chốt "bức tường là DATA" bằng
SỐ, và bước kế là **data quý từ HF (CulturaX lọc) + sampling weights**, không phải tham số.

---

## 🔴 KẾT QUẢ G19c (2026-07-20) — BỨC TƯỜNG DATA, CHỐT BẰNG SỐ

42M v10 30k = **5.843 bits/token** (gap train/eval **+0.962**) **THUA** 11.4M (**5.594**) →
đúng ô *"overfit 19.7M ⇒ bức tường DATA"* của BƯỚC 13. Trúng dự đoán ghi trước. **Hệ quả: thêm
tham số KHÔNG giúp; đòn bẩy là DATA.** SFT thử trên base này → guard sạch (+0.001, replay ăn)
nhưng bits-đáp chỉ −0.329 và **sinh RÁC** (lặp + bịa vi-alpaca) = base yếu lộ ra, KHÔNG phải
lỗi recipe. → Miền đích tác giả chốt = **QA**. Hai việc mở khoá chất lượng sinh:

## ⚡ BƯỚC 14a — QA HELD-OUT (thước miền đích) + DECODE POLISH

Essays (vi-alpaca) đo NHẦM miền QA (v11 wiki 6.510 trên essays = lệch miền). Dựng thước QA
cố định (ViQuAD **test** — disjoint theo split + grade val closed-book, dedup leakage pretrain):

```bash
cd <repo> && git pull && export PYTHONPATH=src
python -u scripts/build_qa_heldout.py 2>&1 | tail -5   # → data/sft/qa_heldout.jsonl (~280)
```

**Decode polish đã cắm** (`NeuralLM.generate`: repetition-penalty + no-repeat-ngram + top-p —
§7.17d/G15, argmax thuần gây loop). SFT read-by-hand mặc định anti-loop (`--gen-rep-penalty 1.2
--gen-no-repeat 3`); muốn đa dạng thì `--gen-temp 0.8 --gen-top-p 0.9`. Đây là **son phấn cắt
loop nhìn thấy ngay**, KHÔNG thêm tri thức — thuốc thật là data (BƯỚC 10c/CulturaX).

→ Từ nay data-scale experiment eval trên **CẢ HAI**: `--holdout data/real_vi/essays.jsonl` (nối
mốc 5.594) VÀ `--holdout data/sft/qa_heldout.jsonl` (miền đích QA). Số QA là số quyết định.

## ⚡ PROBE — SINH TỰ DO đồ thị (không bám agenda) — `scripts/eval_free_gen.py`

Câu hỏi tác giả: "đồ thị sinh tốt vì bám agenda — thử KHÔNG bám xem". Tách CHÉP vs SINH thật:
cùng prompt `Hỏi:`, quét agenda ĐẦY → TẮT (`ar_index_backbone=False` + `ar_agenda_boost=0` +
`ar_spread_weight=0`) + `ar_temperature` 0→1.0. Đo `copy%` (LCS corpus) + `rep4%` (loop) + đọc tay.

```bash
# M4 (đồ thị 2GB → 20GB RAM; dev 8GB không load nổi):
python -u scripts/eval_free_gen.py --checkpoint checkpoints/network_v11.pkl 2>&1 | tee out/free_gen.txt
```
Đọc: agenda-TẮT còn mạch lạc + copy% tụt ⇒ SINH thật (tin "đồ thị tốt"). Sụp (loop/trôi/bịa) +
agenda-ĐẦY copy% cao ⇒ chất lượng = CHÉP (khớp §7.13/§0.5; §7.12 nghiêng về vế này).

## ⚡ BƯỚC 10c (THUỐC THẬT) — 42M trên v11 180M, eval qa_heldout — `out/run_step10c_v11.sh`

Đòn bẩy DUY NHẤT sau khi G19c chốt bức tường DATA. Corpus v11 (v10 + viwiki 160M ≈ 180M token,
đã sẵn) — 9× data ⇒ ít lặp epoch ⇒ bớt overfit (nút +0.962 của G19c). Held-out = **qa_heldout**
(miền đích) drives best-checkpoint. Driver tự chạy BẢNG so 2×2 cuối cùng.

```bash
cd <repo> && git pull && export PYTHONPATH=src
python -u scripts/build_qa_heldout.py     # nếu chưa có data/sft/qa_heldout.jsonl
bash out/run_step10c_v11.sh               # ~9h + bảng so; log out/lm_v2_v11_qa.log, out/eval_10c_table.txt
git add -f out/lm_v2_v11_qa.log out/eval_10c_table.txt && git commit -m "TK_042: 10c v11 QA" && git push
```

**Bảng đọc (bits/token trên qa_heldout — thấp hơn = QA tốt hơn):**

| | qa_heldout | ⇒ |
|---|---|---|
| v11 **< rõ** v10 | | thêm data NÂNG QA ⇒ scale tiếp (CulturaX vi) + right-size model |
| v11 ≈ v10 | | 180M chưa đủ / viwiki lệch miền QA ⇒ cần data QA-hơn (không phải wiki thô) + weighted sampling |
| v11 **> v10** | | wiki thô pha loãng miền QA ⇒ hạ trọng số viwiki, ưu tiên conv/QA |

Đọc tay 6 mẫu sinh QA (decode polish) ở cuối `eval_10c_table.txt` — bits mù với bịa (G15).

## ⚡ BƯỚC 16 (SAU 10c, NẾU v11 thắng) — CulturaX: base fluent thật — `out/run_culturax.sh`

CulturaX vi = **55.4 tỷ token** (uonlp/CulturaX, mc4+OSCAR lọc sẵn) — nguồn thật đưa base lên
fluent. Lấy lát ~500M. **GATED** — làm 1 lần trước:

```bash
huggingface-cli login                                    # token từ HF settings
# → mở huggingface.co/datasets/uonlp/CulturaX bấm "Agree and access"
cd <repo> && git pull && export PYTHONPATH=src && pip install datasets
python -u scripts/ingest_culturax.py --max-tokens 500000000   # → culturax_vi.jsonl (~2GB, KHÔNG push)
bash out/run_culturax.sh                                       # mặc định 60k step (~17h)
git add -f out/lm_v2_v12_qa.log out/eval_v12_table.txt && git commit -m "TK_042: CulturaX v12" && git push
```

**⚠️ ĐIỂM MẤU CHỐT — token SEEN ≠ corpus size:** Chinchilla 42M cần ~840M token SEEN = steps×16×512.
30k=245M seen (0.3 epoch của corpus lớn = CHƯA vắt data) · **60k=490M (~17h, mặc định)** · 100k=819M
(~29h, gần Chinchilla — `STEPS=100000 bash out/run_culturax.sh`). Corpus lớn CHỈ giúp nếu train ĐỦ LÂU.

**`repeat` = sampling weight** (`config/train_v12.yaml`): CulturaX 500M nhấn chìm QA 34k (0.007%) nếu
không nâng → base fluent nhưng QUÊN khung Hỏi/Đáp. Đã nâng qa/grade ×10–15. **TUNE theo qa_heldout.**

Bảng `eval_v12_table.txt`: **v10(19.7M) < v11(180M) < v12(~750M)** trên qa_heldout? Xuống đều ⇒ data
là đòn bẩy đúng, scale tiếp / SFT lại trên v12. Phẳng ⇒ chạm trần khác (model nhỏ / cần train lâu hơn).

## ⚡ BƯỚC 14 — SFT trên data mở chất lượng (SAU khi G19c có checkpoint)

Nguồn đã khảo sát + đọc mẫu (2026-07-20): **vi-alpaca 50k** (sinh trực tiếp tiếng Việt,
KHÔNG phải dịch) + **UIT-ViQuAD2.0 28k** (QA người viết, wiki VI — giữ context trong prompt,
bỏ is_impossible). LOẠI OpenOrca-Viet/Bactrian-X (translationese). ⚠️ held-out `essays.jsonl`
rút TỪ vi-alpaca → ingest lọc leakage theo instruction, log phải in `LOẠI n trùng held-out`
với n>0 — n=0 là filter hỏng.

```bash
cd <repo> && git pull && export PYTHONPATH=src
# 1) kéo data (REST /rows, không cần pip datasets; ~78k cặp, vài phút):
python -u scripts/ingest_sft_vi.py 2>&1 | tee out/ingest_sft.log
# 2) SFT (loss CHỈ trên phần đáp sau `Đáp:`; replay chống quên mỗi 4 step; ~1-2h):
python -u scripts/sft_neural_lm.py --ckpt out/lm_v2_v10_fp32.pt \
  --save out/lm_v2_sft.pt --log-file out/sft_neural_lm.log
git add -f out/sft_neural_lm.log out/ingest_sft.log && git commit -m "TK_042: SFT" && git push
```

**Điều kiện phân định (GHI TRƯỚC — verdict in sẵn cuối log):**

| thước | ngưỡng |
|---|---|
| bits-đáp sft-val TRƯỚC→SAU | giảm ≥ 0.5 bits ⇒ SFT ăn |
| guard essays bpb (CÙNG held-out mốc 5.594) | Δ ≤ +0.05 sạch · +0.05–0.2 cảnh báo · > +0.2 QUÊN (chữ ký G18) ⇒ tăng replay/hạ lr, không đắp núm quá 1 vòng |
| ĐỌC TAY 12 mẫu greedy cuối log | bits MÙ với "bịa mạch lạc" (G15) — phải đọc, dán nguyên văn |

---

## ✅ B5 XONG (2026-07-18, M4) — `network_v11.pkl`

```
tổng 8.91h · train 4.19h · save+bake 4.72h
vocab BPE 32,002 (fingerprint 1582dad40906e69f) · parents=65,098 (cùng corpus v10)
FRAME_BRACKET edges=1,815,908 — đã bake, khớp quy mô v10 (1,812,272)
```

Checkpoint ~2GB, **không** đi theo git, ở lại M4 (`checkpoints/network_v11.pkl`). Log đầy đủ đã
push: [out/train_v11_full.log](out/train_v11_full.log).

## ⚡ BƯỚC TIẾP THEO — làm trên M4

### ✅ BƯỚC 1 ĐÃ XONG — `v11.txt` (2026-07-18)

```
TỔNG: đáp TB 30.5 tok · copy TB 6% · 5-gram MỚI 0/216 = 0%
```

**BPE KHÔNG đổi kết luận §0.5** — `0/216 = 0%`, đúng bằng không, y hệt `v10.pkl` (`0/276=0%`
tiền-BPE). Mạng khảm bất kể tokenizer. `median cha` 54→58 (khớp +7.8% token của BPE — memory dự
án). Không có cảnh báo `FRAME_BRACKET gate TẮT` — bake ổn.

Outlier `'Nhà Lý'`/`'Trần Hưng Đạo'` đổi triệu chứng so với checkpoint histwiki cũ: KHÔNG còn
chữ vô nghĩa (word-salad), mà dừng sau đúng 2 token ở dấu đóng ngoặc (`'".'`, `').'`)  — **chưa
điều tra**, khác hẳn outlier cũ, ghi lại để không lẫn.

So tokenizer công bằng (§7 dòng ~170, `v10.pkl` vs `v11.pkl` — CÙNG corpus CÙNG manifest, chỉ
khác tokenizer): cả hai đều `MỚI≈0%`, `copy TB` gần nhau (5% vs 6%) — **BPE không thay đổi mức
độ khảm**, chỉ đổi cách chia token.

### ✅ BƯỚC 2 (G15 v2) ĐÃ XONG — `knn_scale.txt` (2026-07-18)

```
TỔNG (trộn lẫn, KHÔNG dùng số này) 433/1152 = 38%
tách theo λ:  λ=1.0 98.6%  ·  λ=0.5 34.7%  ·  λ=0.2 12.2%  ·  λ=0.0 8.0%
```

`λ=1.0` chiếm gần hết tổng nhưng là **giả tạo** — vòng lặp thoái hoá (`'nhà thơ, nhà thơ, nhà
thơ…'`) tạo 5-gram "mới" chỉ vì corpus thật không ai lặp một cụm 10 lần liên tiếp. **Sửa lại
tuyên bố trước đó "vòng lặp biến mất hoàn toàn"** — đọc toàn văn thì ở `λ=0.5` vòng lặp **vẫn
còn**, chỉ thô hơn (lặp cả mệnh đề: *"Đình Ninh Xá, xã Bắc An"* lặp 3 lần gần nguyên văn).

**Phát hiện thật: có ĐÁNH ĐỔI, không phải lời giải sạch.** λ cao → mới hơn thống kê nhưng dễ lặp
hỏng; λ thấp → mạch lạc hơn hẳn (đọc như văn xuôi lịch sử thật) nhưng gần chép nguyên. Không có
điểm nào trong dải λ vừa mới thật vừa mạch lạc cùng lúc — chi tiết TK_042 §7.17b.

### ⚡ BƯỚC 3 (MỚI, G15 v3) — LẤY MẪU thay ARGMAX

Toàn bộ G13→G15 đến giờ dùng `argmax` — TẤT ĐỊNH, luôn chọn token cao nhất. Nhưng câu hỏi mở ra
TK_042 mô tả **lấy mẫu ngẫu nhiên** (*"lúc thì sinh 'chó', lúc thì sinh 'cún'"*) — chưa từng thử
đúng thứ được hỏi. Cũng có thể argmax CHÍNH LÀ nguyên nhân vòng lặp (chọn cao nhất → đẩy xác
suất token tương tự lên cao nhất lượt sau → lặp mãi) — lấy mẫu có nhiễu thường tự phá được.

Đã vá `scripts/probe_knn_scale.py` thêm `--mode sample`: rút mẫu trực tiếp từ phân bố **đã
trộn** `λ·P_LM + (1-λ)·P_kNN`, chạy nhiều seed/prompt để QUAN SÁT TRỰC TIẾP biến thiên.

```bash
cd <repo> && git pull && export PYTHONPATH=src
python -u scripts/probe_knn_scale.py --mode sample --seeds 3 --sample-lams 0.5,0.3 2>&1 | tee out/knn_sample.txt
git add -f out/knn_sample.txt scripts/probe_knn_scale.py && git commit -m "TK_042: G15 v3 — lấy mẫu thay argmax" && git push
```

### ✅ BƯỚC 3 (G15 v3, sample temp=1.0) ĐÃ XONG — biến thiên ✅ chất lượng SỤP ❌

`out/knn_sample.txt` (đã có): 3/3 seed khác nhau MỌI prompt (biến thiên đạt), vòng lặp biến
mất — NHƯNG đọc tay phần lớn **vô nghĩa/bịa** (*"vùng thuộc địa Ontario"*, token rác `∩Es2`).
`MỚI 93%` là **bẫy** — thước đo "chưa từng có" không phân biệt "sáng tạo" với "bịa". Chi tiết
TK_042 §7.17c. Nguyên nhân: temp=1.0 rút mẫu từ phân bố phẳng của LM yếu ≈ bốc ngẫu nhiên, token
rác đuôi bị bốc trúng.

### ⚡ BƯỚC 4 (MỚI, G15 v4) — QUÉT temp thấp + top-k cắt đuôi rác

Top-k/top-p **cắt đuôi phân bố** — chính chỗ token rác nấp (mỗi cái xác suất cực nhỏ, nhưng
tổng đuôi trên 32k token đủ để temp=1 bốc trúng). Thêm thước tự động `rep4%` (tỷ lệ lặp nội bộ)
để bắt "hỏng do lặp" không cần đọc tay. **Cả `MỚI%` lẫn `rep4%` vẫn MÙ với "bịa mạch lạc"** —
cột toàn văn để đọc tay tính đúng.

```bash
cd <repo> && git pull && export PYTHONPATH=src
python -u scripts/probe_knn_scale.py --mode sample --sample-lams 0.4 \
    --sample-temps 0.3,0.5,0.7 --top-k 40 --seeds 2 2>&1 | tee out/knn_sweep.txt
git add -f out/knn_sweep.txt scripts/probe_knn_scale.py && git commit -m "TK_042: G15 v4 — quét temp+top-k" && git push
```

Script in **bảng tóm tắt** `λ · temp · MỚI% · rep4% · seed-khác%` rồi toàn văn. Đọc bảng:

| dấu hiệu | nghĩa |
|---|---|
| `rep4%` **thấp** + `MỚI%` vừa phải + seed-khác cao | ứng viên tốt — **rồi đọc tay** cột toàn văn xem có đúng sự thật không |
| `rep4%` **cao** | vẫn lặp — temp quá thấp / phân bố quá nhọn |
| `MỚI%` ~90%+ (như temp=1.0 cũ) | nghi bịa — top-k chưa cắt đủ rác, hoặc LM tự bịa từ hợp lệ |
| **không ô nào** vừa rep-thấp vừa mới+đúng | củng cố *"LM 11.4M là trần"* → chuyển sang **bits/byte** |

### ✅ BƯỚC 4 (G15 v4) ĐÃ XONG — top-k cắt rác ✅ nhưng "bịa mạch lạc" ❌

`out/knn_sweep.txt` (đã có): top-k=40 cắt sạch rác token + lặp (`rep4` 0-8%) — thắng so temp=1.0.
NHƯNG đọc tay: văn trôi chảy mà **sai sự thật** (Nhà Trần→"ông Hồ đọc Tuyên ngôn"; Lý Thánh Tông
"mất năm 400"). Token ĐẦU đúng (kNN neo được) nhưng sau vài token TRÔI. **Đã quét HẾT núm giải
mã — không núm nào sửa được, vì là giới hạn NĂNG LỰC LM.** Chi tiết TK_042 §7.17d.

### ⚡ BƯỚC 5 (MỚI, G16) — bits/byte HELD-OUT: thước CHỐT bằng số

Hết đọc tay. bits/byte = cross-entropy thật, phạt CẢ lặp LẪN bịa. Trả lời câu DUY NHẤT:
`bits/byte(LM+kNN) < bits/byte(LM thuần)?` — kNN có cải thiện perplexity thật không.

```bash
cd <repo> && git pull && export PYTHONPATH=src
python -u scripts/eval_bits_knn.py 2>&1 | tee out/bits_knn.txt
git add -f out/bits_knn.txt scripts/eval_bits_knn.py && git commit -m "TK_042: G16 — bits/byte held-out" && git push
```

Chỉ cần `g7_best_model.pt` + `bpe_v1.json` + `history_articles.jsonl` (KHÔNG cần checkpoint đồ
thị). Script tách held-out khỏi datastore (bài `i%10==0` → held-out) để tránh leakage, quét
`λ∈{1.0,0.7,0.5,0.4,0.25}` (λ=0 bỏ vì -log2(0)=∞), mặc định 2500 token (CPU).

| kết quả | nghĩa |
|---|---|
| λ<1.0 cho bits/byte **thấp hơn** λ=1.0 | kNN **cải thiện** perplexity thật — đáng giữ, không phải đồ trang trí |
| λ=1.0 (LM thuần) **thấp nhất** | kNN KHÔNG giúp density — xác nhận trần bằng CON SỐ, bỏ kNN cho density |

⚠️ bits/byte chỉ đo **density**, KHÔNG đo "sinh đúng sự thật". kNN cải thiện bits/byte mà SINH
vẫn trôi chủ đề (§7.17d) là hoàn toàn có thể — hai câu hỏi khác nhau. Nếu cả hai đều tệ ⇒ điểm
dừng rõ ràng để quyết định **train LM lớn hơn 11.4M** (đòn bẩy duy nhất còn lại).

### ✅ BƯỚC 5 (G16) ĐÃ XONG — kNN cải thiện +3.9% @ λ=0.7, U-curve, căng thẳng liều (§7.17e)

### ⚡ BƯỚC 6 (MỚI, G17) — "SPREADING HỌC ĐƯỢC + XẾP CHỒNG ĐƯỢC" = attention thiên vị đồ thị

Yêu cầu tác giả: *"làm cho spreading activation học được và xếp chồng được thử xem"*. Dạng đo
được rẻ nhất: A/B hai transformer **giống hệt từng batch**, nhánh B cộng vào logit attention
`α_l · (Ê_tᵢ·Ê_tⱼ)` — Ê = E của `network_v11.pkl` (PPMI+SVD, chính hình học spreading dùng),
đóng băng; `α_l` = cổng học được mỗi tầng. Xếp chồng L tầng = spreading nhiều vòng có cổng.

```bash
cd <repo> && git pull && export PYTHONPATH=src

# một lần: rút E (32002×96) từ checkpoint 2GB ra .npy nhỏ (~12MB) — nạp có gc.disable, ~60s
python -u scripts/probe_graph_attention.py --extract-e checkpoints/network_v11.pkl

# A/B (cùng seed, cùng thứ tự batch; d=128 L=2 — probe tín hiệu, không phải SOTA)
python -u scripts/probe_graph_attention.py --steps 4000 2>&1 | tee out/graph_attn.txt
git add -f out/graph_attn.txt && git commit -m "TK_042: G17 — graph-prior attention A/B" && git push
```

**Điều kiện phân định (ghi trước):**

| kết quả | nghĩa |
|---|---|
| GRAPH < PLAIN rõ (bits/tok held-out) + α giữ xa 0 | hình học đồ thị là **prior có ích** — đường mới nhấc trần LM nhỏ, đáng chạy bản to |
| Δ≈0 hoặc α→0 | backprop **tự học hết** — đồ thị không đóng góp ở tầng ĐỊNH TUYẾN (vẫn có giá trị ở tầng BỘ NHỚ — G16). **Âm tính trung thực, khép nhánh, không đắp núm.** |

### ✅ BƯỚC 6 (G17) ĐÃ XONG — Δ −0.06% (hoà), α tăng nhưng thừa — khép nhánh định tuyến (§7.18a)

### ⚡ BƯỚC 7 (MỚI, G18) — TẦNG NƠ-RON TRI THỨC: tri thức tham số, MỌC được, sinh = lan truyền

Yêu cầu tác giả: *"kNN-LM bản chất là RAG, tôi không muốn — phải có backprop + NƠ-RON MỚI mang
tri thức + sinh bản chất là lan truyền kích thích"*. G18 = memory layer (Meta 2024) ở quy mô
probe: N=4096 nơ-ron (key,value), top-32 NỔ mỗi bước (lan truyền thưa theo nội dung), backprop,
và **MỌC** 20 nơ-ron mới cho 20 fact hư cấu sau khi train xong.

```bash
cd <repo> && git pull && export PYTHONPATH=src
# cần out/E_v11.npy (đã có từ G17). ~15-20 phút cho 3 nhánh trên M4:
python -u scripts/probe_knowledge_neurons.py --steps 3000 2>&1 | tee out/knowledge_neurons.txt
git add -f out/knowledge_neurons.txt && git commit -m "TK_042: G18 — knowledge-neuron layer" && git push
```

3 nhánh cùng backbone d=128 L=2, cùng thứ tự batch: `ffn` (thường) · `mem_rand` (key ngẫu
nhiên) · `mem_graph` (key gieo từ đơn vị corpus = cha đồ thị). **Bài thi quyết định — mọc-
không-quên**, thứ cả FFN lẫn kNN-LM đều không có:

| kết quả | nghĩa |
|---|---|
| nhánh mem: `Δfact ≤ −2.0` VÀ `|Δquên| < 0.05` | **PASS** — tri thức tham số, mọc được, không quên: kiến trúc giao được thứ đã hứa |
| nhánh ffn full-FT: Δfact âm nhưng Δquên **dương** | quên thảm hoạ — nền so sánh, làm nổi giá trị sparse-update |
| nhánh mem: Δquên lớn | slot mới NỔ NHẦM trên ngữ cảnh cũ — key không đủ chọn lọc — âm tính trung thực |
| `mem_rand` ≈ `mem_graph` (held-out) | gieo mầm đồ thị thừa (bài học G17 lặp lại) — ghi thật, không cứu vớt |

### ✅ BƯỚC 7 (G18, grow-mode=fact) ĐÃ XONG — RỚT cả 2 tiêu chí, chẩn đoán = SAI HẠT ĐỘ (§7.19a)

Δfact −0.4 (cần ≤−2.0), Δquên +0.13-0.19 (cần <0.05). Key CHẠY (20/32 slot nổ là slot mới),
value CHẾT (1 vector tĩnh/fact không mã hoá nổi ~25 dự đoán theo vị trí → sinh '.<EOS>').
FFN full-FT đối chứng: học được (2/3 probe đúng @ 6000 step) nhưng quên thảm hoạ (+13 bits).

### ⚡ BƯỚC 8 (G18b) — mọc THEO VỊ TRÍ (hạt độ datastore kNN-LM)

```bash
cd <repo> && git pull && export PYTHONPATH=src
# --grow-mode position giờ là MẶC ĐỊNH: 1 nơ-ron/vị-trí (~500 slot cho 20 fact),
# key = q của chính mô hình tại vị trí đó — hệt công thức datastore (h_t, next) của kNN-LM
python -u scripts/probe_knowledge_neurons.py --steps 3000 2>&1 | tee out/knowledge_neurons_pos.txt
git add -f out/knowledge_neurons_pos.txt && git commit -m "TK_042: G18b — grow theo vị trí" && git push
```

Điều kiện phân định **GIỮ NGUYÊN**: PASS = `Δfact ≤ −2.0` VÀ `|Δquên| < 0.05`. Rủi ro mới phải
đo: ~500 nơ-ron mọc (thay vì 20) ⇒ nguy cơ nổ nhầm trên văn cũ TĂNG ⇒ **Δquên là thước quyết
định**. Nếu PASS ⇒ kiến trúc giao được lời hứa, và "nơ-ron tri thức" đúng nghĩa = datastore-
vị-trí có gradient trong mạng (lần hội tụ thứ 4 — §7.19a). Nếu Δquên nổ ⇒ cần key chọn lọc
hơn/ngưỡng nổ — ghi rồi tính tiếp, KHÔNG đắp núm quá một vòng.

### ✅ BƯỚC 8 (G18b) ĐÃ XONG — BÁC chẩn đoán hạt độ: Δfact −0.37 ≈ cũ, cùng chữ ký '.<EOS>' (§7.19b)

### ⚡ BƯỚC 9 (G18c) — VÁ LỖI ĐO, VÒNG CHÓT

Soi lại probe lộ lỗi thiết kế đo (§7.19c): capture/eval dùng fact ĐỨNG MỘT MÌNH vị trí 0, còn
train mọc dùng cửa sổ cắt từ dòng nối ở offset ngẫu nhiên ⇒ `q` lệch pha với key đã lưu ⇒
nơ-ron đúng KHÔNG nổ lúc train ⇒ gradient rơi nhầm chỗ, value học ra trung bình `'.'`/EOS —
giải thích chữ ký hỏng Y HỆT ở cả hai chế độ mọc. Đã vá: train mọc trên TỪNG FACT ĐỨNG RIÊNG
từ vị trí 0 (khớp capture & eval), áp cả nhánh ffn-FT.

```bash
cd <repo> && git pull && export PYTHONPATH=src
python -u scripts/probe_knowledge_neurons.py --steps 3000 2>&1 | tee out/knowledge_neurons_c.txt
git add -f out/knowledge_neurons_c.txt && git commit -m "TK_042: G18c — vá lỗi đo, vòng chót" && git push
```

**Kỷ luật ghi trước: đây là VÒNG CHÓT của G18.** PASS = `Δfact ≤ −2.0` VÀ `|Δquên| < 0.05`
(không đổi). Rớt nữa ⇒ khép nhánh dứt khoát: "bộ nhớ tham số mọc được không giao hàng ở quy mô
probe; kNN-LM phi-tham-số vẫn là câu trả lời thực dụng."

### ✅ BƯỚC 9 (G18c) ĐÃ XONG — RỚT, NHÁNH KHÉP DỨT KHOÁT (§7.19d)

### ⚡ BƯỚC 10 (G19) — TRAIN LM LỚN HƠN: v2 42M (RoPE+RMSNorm+SwiGLU+Muon) + arm memory-from-scratch

Đòn bẩy cuối cùng của ticket (§7.17e), giờ mới chạy. Kiến trúc v2 trong `neural_lm.py`
(`build_model_v2` — một nguồn duy nhất), train qua `train_neural_lm.py --arch v2`:

* **RoPE** thay bảng vị trí học được (checkpoint v2 = dict {cfg, state_dict} — `NeuralLM.load`
  tự nhận, `nh` đọc từ cfg vì không suy được từ shape)
* **RMSNorm + SwiGLU** (hid = 2/3·4d làm tròn bội 64 — giữ tham số ≈ FFN cũ)
* **Orthogonal init** mọi ma trận 2D attention/FFN — các head là lát hàng của cùng ma trận ⇒
  trực giao TỪNG ĐÔI tại t=0 (câu hỏi tác giả); **Muon** giữ update đẳng hướng trong train
  (đó chính là cơ chế ruột của nó — Newton-Schulz trực giao hoá momentum); cuối train in
  **chẩn đoán head-sim** (đo-không-ép)
* **Muon lai**: ma trận ẩn 2D đi Muon (lr 0.02, phẳng), embedding/norm/bảng memory đi AdamW
  (OneCycle) — chuẩn thực hành speedrun/Kimi
* **Arm memory-from-scratch** (`--mem-slots 16384`): thay FFN block 5/8 bằng MemLayer train
  CÙNG backbone từ đầu — đúng điều kiện sống duy nhất G18 §7.19d chỉ ra

```bash
cd <repo> && git pull && export PYTHONPATH=src

# ARM 1 — v2 thuần, ~42M (ước ~3-4h M4; 10k step × 16×512 = 82M tok ≈ 4.2 epoch):
python -u scripts/train_neural_lm.py --arch v2 --opt muon \
  --d-model 512 --layers 8 --heads 8 --ctx 512 --batch 16 \
  --steps 10000 --eval-every 500 --save out/lm_v2_42m.pt --log-file out/lm_v2_42m.log

# ARM 2 — v2 + memory from-scratch, ~49M (chạy TUẦN TỰ sau arm 1):
python -u scripts/train_neural_lm.py --arch v2 --opt muon --mem-slots 16384 \
  --d-model 512 --layers 8 --heads 8 --ctx 512 --batch 16 \
  --steps 10000 --eval-every 500 --save out/lm_v2_mem.pt --log-file out/lm_v2_mem.log

git add -f out/lm_v2_42m.log out/lm_v2_mem.log && git commit -m "TK_042: G19 — train v2 42M + mem arm" && git push
# (2 file .pt ~170/200MB KHÔNG push — ở lại M4)
```

**⚠️ G19 LẦN 1 (2026-07-19) HỎNG + BỊ NHIỄU — xem §7.20a.** best 6.479 bits/tok (tệ hơn mốc
5.594), sinh lặp thoái hoá. Hai nhiễu: (1) thiếu train 3× (10k step vs G7 30k); (2) **lỗi cấu
hình: Muon LR để phẳng** — đã vá anneal cosine. Head-sim 0.005 (trực giao ✅, câu hỏi tác giả).
⚠️ 2 file .pt 365MB bị commit làm ĐỨT fetch — đã gitignore `out/*.pt`.

### ⚡ BƯỚC 10b (G19 lần 2) — CHẠY LẠI arm 1 với Muon LR anneal + 30k step (khớp token G7)

```bash
cd <repo> && git pull && export PYTHONPATH=src
# 30k step = khớp token-budget G7 (loại nhiễu thiếu-train); Muon LR giờ anneal cosine (đã vá).
# ~9h trên M4 (3.06h/10k đo được). Chạy qua đêm.
python -u scripts/train_neural_lm.py --arch v2 --opt muon \
  --d-model 512 --layers 8 --heads 8 --ctx 512 --batch 16 \
  --steps 30000 --eval-every 1000 --save out/lm_v2_42m.pt --log-file out/lm_v2_42m_r2.log
git add -f out/lm_v2_42m_r2.log && git commit -m "TK_042: G19 lần 2 — Muon anneal 30k" && git push
# .pt KHÔNG push (gitignore). Arm mem: HOÃN tới khi arm 1 sạch.
```

Đọc: eval TỐT NHẤT (không phải cuối) so **5.594**. Nếu thắng rõ + gap train/eval ≥ 0.5 ⇒ tham
số còn đất, tiếp G16/G15 với LM mới. Nếu vẫn thua + gap ≥ 0.5 (đã vắt data) ⇒ **giờ mới** chốt
bức tường DATA bằng số. Nếu gap < 0.5 ⇒ vẫn underfit, chưa kết luận.

### ⚡ BƯỚC 10c — GỠ NÚT DATA (viwiki) rồi rerun trên corpus v11 (LÀM TRƯỚC 10b)

Số học: 42M cần ~840M token, corpus 19.7M thiếu 40×. Thêm viwiki (~170M) TRƯỚC khi rerun —
nếu không, rerun 30k vẫn thiếu data.

```bash
cd <repo> && git pull && export PYTHONPATH=src
pip install datasets                      # cho ingest_viwiki (M4 có net)
python -u scripts/ingest_viwiki.py --max-tokens 160000000    # → data/text_corpus/viwiki.jsonl
git add -f data/text_corpus/viwiki.jsonl && git commit -m "TK_042: viwiki corpus" && git push
#   ⚠️ viwiki.jsonl có thể ~500MB — nếu push ĐỨT như .pt (§7.20a), giữ ở M4, đừng push;
#      chỉ push log. Corpus KHÔNG cần về máy dev.

# rerun 30k trên corpus v11 (= v10 + viwiki), Muon anneal:
python -u scripts/train_neural_lm.py --arch v2 --opt muon --manifest config/train_v11.yaml \
  --d-model 512 --layers 8 --heads 8 --ctx 512 --batch 16 \
  --steps 30000 --eval-every 1000 --save out/lm_v2_v11.pt --log-file out/lm_v2_v11.log
git add -f out/lm_v2_v11.log && git commit -m "TK_042: G19 lần 2 — v11 corpus + Muon anneal 30k" && git push
```

⚠️ 30k step trên corpus ~9× to hơn: mỗi step vẫn 1 batch (thời gian/step không đổi ~1.1s) nên
vẫn ~9h. Corpus to hơn = mỗi epoch nhiều token hơn = ÍT lặp epoch hơn = tốt (bớt overfit).

### ⚡ BƯỚC 11 (G20, §7.21) — MoS head — CHỈ khi 10c xong mà bits VẪN kẹt + nút ở OUTPUT

Đã code sẵn (`--mos-k`), CHƯA chạy thực. **Điều kiện bật:** viwiki + rerun sạch rồi mà bits vẫn
> 5.594 VÀ chẩn đoán chỉ vào output (không depth). A/B như G17: cùng mọi thứ, chỉ thêm `--mos-k
15`. Nếu bits cải thiện rõ ⇒ softmax-bottleneck là thật ở mô hình này. Nếu ≈ ⇒ nút không ở
output, khép. **KHÔNG chạy G20 trước 10c** — sẽ tối ưu chỗ không đau (§7.21).

```bash
python -u scripts/train_neural_lm.py --arch v2 --opt muon --manifest config/train_v11.yaml \
  --mos-k 15 --d-model 512 --layers 8 --heads 8 --ctx 512 --batch 16 \
  --steps 30000 --eval-every 1000 --save out/lm_v2_mos.pt --log-file out/lm_v2_mos.log
```

### ⚡ BƯỚC 12 (§7.22) — KV CACHE: sinh nhanh, đầu ra GIỐNG HỆT (độc lập, chạy bất cứ lúc nào)

Câu hỏi tác giả "chạy runtime / sinh nhanh". KV cache = prefill prompt một lần, decode O(1)/
token thay vì O(ctx). Đã code (`NeuralLM.generate_cached`, v2). Cần MỘT checkpoint v2 bất kỳ
(dù chưa train tốt — chỉ đo cơ chế). Không phụ thuộc data/rerun.

```bash
cd <repo> && git pull && export PYTHONPATH=src
python -u scripts/probe_kv_cache.py --ckpt out/lm_v2_42m.pt 2>&1 | tee out/kv_cache.txt
git add -f out/kv_cache.txt && git commit -m "TK_042: G21 KV cache" && git push
```

Probe làm ĐÚNG-ĐẮN trước tốc độ: bước 1 khẳng định greedy có-cache == không-cache TỪNG TOKEN
(lệch ⇒ exit 1, nghi RoPE offset). Chỉ khi giống hệt mới đo tokens/giây. Kỳ vọng 5-20× nhanh
hơn với đầu ra y hệt. **Nếu bước 1 fail** ⇒ dán nguyên văn (vị trí token lệch đã in sẵn).

⚠️ Đường v2 **chưa chạy thử** (máy dev không có torch) — lỗi thì dán nguyên văn traceback, đó
là thứ cần sửa. ⚠️ Các probe kNN cũ (`probe_knn_scale.py` `_hidden` viết theo ruột v1, có
`m.pos`) sẽ cần vá khi chuyển sang v2 — việc SAU khi có checkpoint.

**Điều kiện phân định (ghi trước):**

| kết quả | nghĩa |
|---|---|
| v2-42M bits/token < **5.594** rõ rệt (mốc 11.4M cũ, cùng corpus/held-out) | tham số còn đất — tiếp tục kế hoạch: vá probe kNN → chạy lại G16 (λ sweep) + G15 v4 (sinh) với LM mới; kỳ vọng trôi chủ đề giảm, căng thẳng λ nới |
| v2-42M **không thắng** 5.594 | **bức tường là DATA** (19.7M token), chốt bằng số — hướng tiếp là thêm corpus, không phải thêm tham số |
| arm mem ≈ arm thuần (bits) | memory from-scratch chưa có lợi ở cỡ này (Meta thấy lợi ở hàng tỷ) — ghi thật, không cứu |
| arm mem **thắng** arm thuần | "nơ-ron tri thức" sống lại đúng cách — mở lại bài thi mọc-không-quên trên nền NÀY |
| `gap train/eval < 0.5` ở dòng cuối log | UNDERFIT — chưa kết luận gì, tăng step |

---

FFN-FT sau vá lỗi đo: học HOÀN HẢO (fact → 0.057 bits, 3/3 probe đúng từng chữ) nhưng quên
sạch (+14.2) ⇒ **phép đo giờ chuẩn**. Memory vẫn đứng im Δfact −0.38 qua CẢ BA vòng (hạt độ,
phân bố train đều đã thử) ⇒ thất bại KIẾN TRÚC. Tam giác đánh đổi chốt: tham-số-không-bảo-vệ
= học-mà-quên · tham-số-đóng-băng = không-quên-không-học · **phi-tham-số (kNN-LM) = giao cả
hai**. Trạng thái ticket: kNN-LM làm bộ nhớ + LM lớn hơn làm năng lực — G18 là bằng chứng định
lượng vì sao phân công đó bắt buộc ở quy mô hiện tại.

---

## ⚡ CHẠY TẤT CẢ TRÊN M4 — làm đúng thứ tự này (B5, đã xong ở trên — giữ lại để tham khảo)

```bash
cd <repo> && git pull && export PYTHONPATH=src && mkdir -p out

# ── BƯỚC 1: copy 3 file KHÔNG có trong git từ máy 8GB sang (chỉ làm 1 lần) ──
#   Chạy TỪ máy 8GB (đổi <m4> thành host/đường của M4):
#     scp checkpoints/network_v10_histwiki.pkl \
#         checkpoints/network_v10_histwiki.content.pkl \
#         checkpoints/network_v10_histwiki_paragraph.pkl \
#         checkpoints/network_v10_histwiki_paragraph.content.pkl \
#         out/g7_best_model.pt  <m4>:<repo>/checkpoints/   # .pt để vào out/
#   (network_v10.pkl 1.94GB thì M4 đã có sẵn — KHÔNG cần copy)

# ── BƯỚC 2: XÁC NHẬN file có thật, ĐÚNG kích thước — đừng tin, phải thấy ──
ls -la checkpoints/network_v10.pkl                     # ~1.94GB
ls -la checkpoints/network_v10_histwiki.pkl            # ~366MB
ls -la checkpoints/network_v10_histwiki_paragraph.pkl  # ~419MB
ls -la out/g7_best_model.pt                            # ~45MB

# ── BƯỚC 3: BAKE hai checkpoint histwiki TRƯỚC khi sinh (~15 phút MỖI cái) ──
#   Lần chạy 2026-07-17 TREO ngay khi sinh prompt 1 trên cả hai — vì bracket gate
#   ở trạng thái NỬA VỜI (thiếu layer/hlink_meta/hlink_endpoints, 151k/176k cha >
#   cap 5000 nên KHÔNG tự mine lúc runtime — xem cảnh báo RuntimeWarning trong log).
#   Bake là việc NẶNG MỘT LẦN, không phải việc nhẹ lặp mỗi lần sinh — PHẢI làm trước:
python -u scripts/bake_checkpoint_indices.py checkpoints/network_v10_histwiki.pkl           --in-place
python -u scripts/bake_checkpoint_indices.py checkpoints/network_v10_histwiki_paragraph.pkl --in-place
#   (network_v10.pkl 65k cha < cap 5000 KHÔNG cần bake — gate tự mine được, đã CHẠY XONG)

# ── BƯỚC 4: chạy 3 probe còn lại (mỗi cái ghi log riêng + tee ra .txt để push về) ──
python -u scripts/probe_granularity.py checkpoints/network_v10_histwiki.pkl           out/hw_A.log 2>&1 | tee out/hw_A.txt
python -u scripts/probe_granularity.py checkpoints/network_v10_histwiki_paragraph.pkl out/hw_B.log 2>&1 | tee out/hw_B.txt
python -u scripts/probe_knn_compete.py                                                              2>&1 | tee out/knn_compete2.txt

# ── BƯỚC 5: push kết quả về ──
git add -f out/hw_A.txt out/hw_B.txt out/knn_compete2.txt && git commit -m "TK_042: probe outputs từ M4 (sau bake)" && git push
```

✅ **`network_v10.pkl` ĐÃ XONG** (chạy trước lúc chưa bake, không cần lặp lại) — kết quả:
`5-gram MỚI 0/276 = 0%` — xem §1 dưới, đây là câu trả lời trực tiếp cho câu hỏi *"mạng khảm
hay sinh thật"*.

**Đọc gì:** `v10.txt` trả lời *khảm hay sinh* (cột `MỚI%`); `hw_A` vs `hw_B` trả lời *cha-đoạn
giúp hay hại* (so `MỚI%` + `đáp TB` hai file). Cách đọc chi tiết ở §1–§2 dưới.

⚠️ `g7_best_model.pt` cũng KHÔNG có trong git (out/ bị ignore) → BƯỚC 1 đã copy. Thiếu nó thì
probe 4 (knn) không chạy.

---

## 0. Chi tiết — chuẩn bị

```bash
cd <repo>
git pull
export PYTHONPATH=src
mkdir -p out
```

**⚠️ `probe_knn_*` cần `torch`.** Trên Intel Mac torch kẹt ở 2.2.2, xung đột numpy 2.2.6 ⇒
đã dùng venv riêng. Máy mạnh (Apple Silicon/Linux) thì `pip install torch` là xong.

### 🔴 CHECKPOINT KHÔNG ĐI THEO GIT — phải COPY TAY

`.gitignore` có `checkpoints/*.pkl` ⇒ **`git clone` KHÔNG mang checkpoint theo**. Bản runbook
đầu bỏ qua chuyện này và **hậu quả đã xảy ra thật**: máy mạnh chạy nhầm `network_v10.pkl`
(65 098 cha) thay vì `network_v10_histwiki.pkl` (151 430 cha), rồi chạy nhầm
`network_v10.varlm.pkl` — thứ đó là **sidecar, KHÔNG phải checkpoint** → `KeyError:
'vocab_tokens'`. **Không nhánh nào của thí nghiệm granularity được chạy.**

**Trước khi chạy, kiểm tra bằng lệnh — đừng tin là có:**

```bash
ls -la checkpoints/network_v10_histwiki.pkl checkpoints/network_v10_histwiki_paragraph.pkl
# THIẾU → scp từ máy 8GB sang. KHÔNG có thì probe 1 KHÔNG chạy được. Đừng thay bằng
# checkpoint khác cho có: phép so cần ĐÚNG HAI file này vì chúng chỉ khác nhau granularity.
```

⚠️ `checkpoints/network_v10.varlm.pkl` **lọt vào git** dù bị gitignore (chắc `add -f` lúc nào
đó) ⇒ nó là một trong vài `.pkl` `clone` về có sẵn ⇒ **rất dễ bị tưởng là checkpoint**. Nó
không phải. Nợ vệ sinh repo, nên gỡ khỏi git.

| file | dùng cho | trong git? | ghi chú |
|---|---|---|---|
| `checkpoints/network_v10_histwiki.pkl` | probe 1 | ❌ **copy tay** | 349MB · 151 430 cha · vocab 113 572 · median 19 |
| `checkpoints/network_v10_histwiki_paragraph.pkl` | probe 1 | ❌ **copy tay** | 400MB · 176 430 cha |
| `out/g7_best_model.pt` | probe 2, 3 | ❌ **copy tay** | LM 11.4M, 45MB |
| `checkpoints/bpe_v1.json` | probe 2, 3 | ✅ | fingerprint `1582dad40906e69f` |

---

## 0.5. ✅ ĐÃ CHẠY XONG 2026-07-17 — `network_v10.pkl` (1.94GB, 65 098 cha, MỚI NHẤT)

Không phải phần của phép so granularity (đó là §1, dùng 2 file histwiki riêng) — đây là câu
hỏi **gốc** của tác giả: *"mạng đang khảm hay sinh thật?"*, đo trên checkpoint mới nhất/to nhất,
chính là cái đẻ ra đầu ra bị phàn nàn *"vẫn khá là verbatim copy"*.

```
TỔNG: đáp TB 38.5 tok · copy TB 5% · 5-gram MỚI 0/276 = 0%
```

**`0/276` — ĐÚNG BẰNG KHÔNG,** không phải "gần 0". Trong 276 cụm 5-token sinh ra qua 8 prompt,
**không một cụm nào** chưa từng xuất hiện trong 65 098 cha. `copy%` thấp (2-8%) từng gây hiểu
lầm là tín hiệu tốt — nhưng nó chỉ so với **một** cha (rank_matches top-1); độ mới 5-gram so
với **toàn corpus** và kết luận dứt khoát: **mạng KHẢM — ghép mảnh chép nguyên xi từ NHIỀU cha
khác nhau — không sinh gì mới.** `copy%` thấp không mâu thuẫn với khảm, nó CHÍNH LÀ dấu hiệu
của khảm (mỗi mảnh đến từ cha khác nên LCS-với-một-cha ngắn, dù mọi mảnh đều bị chép).

Đọc tay xác nhận trôi chủ đề giữa các mảnh chép: *'Hai Bà Trưng'* (năm 40) →
*' . Khuôn viên cổ kính của Viện Đại học Đông Dương…'* (1906).

⇒ Đây là bằng chứng cứng nhất từ trước đến giờ trong TK_042 cho lời phàn nàn gốc của tác giả.
Không phải giả thuyết, không phải mini-checkpoint — đo trên checkpoint production 1.94GB.

---

## 1. Granularity — cha-ĐOẠN giúp hay hại? (**~30–60 phút**, RAM ~4GB)

Trả lời câu hỏi tác giả 2026-07-17: *"chia đoạn dài thành nơ-ron câu, token nối vào câu thay vì
vào đoạn — có giúp gì không?"* **Hai checkpoint có sẵn CHÍNH LÀ hai nhánh** của thí nghiệm đó.
Biến độc lập đúng một: có cha-đoạn hay không.

```bash
python -u scripts/probe_granularity.py checkpoints/network_v10_histwiki.pkl           out/gran_A.log 2>&1 | tee out/gran_A.txt
python -u scripts/probe_granularity.py checkpoints/network_v10_histwiki_paragraph.pkl out/gran_B.log 2>&1 | tee out/gran_B.txt
```

Chạy **tuần tự** nếu RAM < 16GB. `tail -f out/gran_A.log` để xem tiến độ từng prompt.

**Đọc:**

| | nghĩa |
|---|---|
| `_paragraph` đáp **dài hơn** + `copy%` **cao hơn** | cha-đoạn **kéo dài đoạn bị chép** ⇒ đề xuất tầng thứ bậc **có lý** |
| hai nhánh **ngang nhau** | cha-đoạn **không bao giờ thắng truy hồi** ⇒ xây tầng thứ bậc là **dã tràng** |
| `_paragraph` **tốt hơn** | đề xuất đi **ngược** hướng |

**Nhánh A đã có một phần** (nạp xong, chưa sinh xong):

```
cha=151,430 · dài>60tok=490 (0.3%) · median=19
```

Median **19 token** ⇒ nhánh "chỉ cha-câu" đã ngắn sẵn. Nếu B ra phân bố tương tự **ở chỗ cha
THẮNG**, nghiêng mạnh về ô "dã tràng".

### Thước thứ hai: **độ mới 5-gram** (thêm 2026-07-17 — thước `copy%` một mình KHÔNG đủ)

Lần chạy nhầm trên `network_v10.pkl` (65 098 cha) ra `copy TB 5%` — **ngược hẳn** dự đoán
`≈100%` viết sẵn ở runbook này. Nhưng **KHÔNG kết luận được**, vì `copy%` đo LCS với **đúng một
cha** (cha `rank_matches` xếp hạng 1 cho *prompt*), trong khi bộ sinh dùng agenda/spreading và
có thể lấy từ cha khác:

| mạng thật sự làm gì | `copy%` hiện ra |
|---|---|
| sinh THẬT | ~5% |
| **KHẢM** — ghép mảnh 3-5 token từ hàng chục cha | **~5%** |

Hai giả thuyết → **cùng một số** ⇒ thước cũ mù. Và "khảm" vẫn đúng lời phàn nàn gốc của tác giả
(*"bám vào các đoạn văn đã train"*), chỉ là bám nhiều đoạn cùng lúc.

⇒ probe giờ dựng index **mọi 5-gram của toàn corpus** rồi hỏi: bao nhiêu % 5-gram trong đầu ra
**chưa từng xuất hiện ở BẤT KỲ cha nào**. Không quan tâm mảnh đến từ đâu.

| | nghĩa |
|---|---|
| `MỚI ≈ 0%` | mọi cụm 5 token đều chép từ corpus ⇒ **khảm**, không sinh gì mới |
| `MỚI` cao | ghép ra chuỗi **chưa từng có** ⇒ sinh thật — **đọc tay** xem có nghĩa hay word-salad |

`copy%` ≈ 100% sẽ xác nhận §7.13c (*"verbatim LÀ đặc tả"*) **trên checkpoint THẬT** lần đầu —
§7.11–7.13 đều đo trên mini 1 294 cha, thứ sổ tay dự án cảnh báo *"ckpt nhỏ đánh lừa NGƯỢC DẤU"*.

**⚠️ Hai caveat phải nhớ khi kết luận:**
1. `FRAME_BRACKET` gate **TẮT** cả hai nhánh (151k cha > cap 5000, chưa bake) ⇒ mạng **thiếu một
   cổng**. Phép so vẫn có kiểm soát (cùng tắt), nhưng kết luận phải ghi kèm.
   Muốn bật: `python scripts/bake_checkpoint_indices.py <ckpt> --in-place`
2. Vocab **113 572 = từ-mức CŨ**, không phải BPE 32k. Hai checkpoint có **trước** TK_041
   ⇒ kết luận nói về mạng **tiền-BPE**.

---

## 2. ✅ G14 — ĐÃ CHẠY XONG 2026-07-17 (`out/knn_compete.txt`) — Q2 ĐẬU, Q1 RỚT

```bash
python -u scripts/probe_knn_compete.py 2>&1 | tee out/knn_compete.txt   # đã chạy, kết quả dưới
```

```
             sổ  #dòng          λ=1.0            λ=0.5          λ=0.2          λ=0.0
                          (LM thuần)                                     (kNN thuần)
 n=1  (G13 gốc)     16  ' đô thị chính…'  ' Huế. Huế nằm…' ' Huế. Huế…'  ' Huế. Huế…'
+6 HARD cùng khuôn 118  ' đô thị chính…'  ' Huế. Huế nằm…' ' Huế. Huế…'  ' Huế. Huế…'
   +6 HARD +5 SOFT 193  ' đô thị chính…'  ' Huế. Huế nằm…' ' Huế. Huế…'  ' Huế. Huế…'

cos=0.8690  TARGET → ' Huế'      ← thắng RÕ
cos=0.8198  HARD   → ' Bắc'
cos=0.8167  HARD   → ' Tokyo'
```

**✅ Q2 ĐẬU — truy hồi là THẬT.** Nhồi 6 câu **cùng khuôn** *"Thủ đô X là Y"* (16 → 193 dòng),
kNN **vẫn** chọn đúng, khoảng cách rõ (0.869 vs 0.820). `h` **có** mã hoá *"Việt Nam"* khác
*"Pháp"*/*"Nhật Bản"* dù ngữ cảnh chỉ khác một tên nước. Lo *"G13 là ảo giác n=1"* → **SAI**.
Đối chứng sạch: cột λ=1.0 giống nhau cả 3 hàng.

**❌ Q1 RỚT — LM là HÀNH KHÁCH.** λ=0.5 ≡ λ=0.2 ≡ λ=0.0 **giống nhau từng byte**. λ=0.0 là kNN
thuần ⇒ ở **mọi** mức trộn, LM **không đóng góp gì**. Đây là **băng ghi âm phát lại**, và nó
phát đúng câu trong sổ, **nguyên văn**.

**Vì sao (là sự thật về TỈ LỆ TỰ TIN, không phải "trộn hỏng"):** LM 11.4M ở 5.594 bits/token có
phân bố **rất phẳng** — token mạnh nhất chỉ ~0.1 trên 32k. kNN với `temp=0.02` gần **one-hot**
(~0.65 dồn vào "Huế"). Trộn 50/50: kNN góp `0.5×0.65=0.33`, LM góp `0.5×0.1=0.05`. **kNN thắng
mà không cần cố.** Hễ kNN có một hàng xóm tự tin, nó **độc tài**, bất kể λ.

🔴 **`temp=0.02` là DO TÔI CHỌN** — tôi vặn một núm rồi để nó quyết định câu trả lời. kNN-LM
đàng hoàng phải chỉnh `temp` **và** λ trên held-out bằng **bits/byte** (`scripts/eval_bits.py`),
không phải bằng cách ngắm vài dòng sinh rồi gật. **G13/G14 vẫn CHƯA đo bits/byte lần nào** —
sinh đúng token ≠ mô hình tốt hơn. Đó là việc tiếp theo, không phải kết luận đã có.

**Vì sao cần:** G13 (§7.15) khoe *"datastore ghi Huế → mạng sinh Huế ⇒ đồ thị lái được LM"*.
Hai lỗ hổng, **tôi tự tạo cả hai**:
* datastore **một câu** ⇒ top-8 hàng xóm đều từ 20 dòng của chính câu đó ⇒ **kNN không thể sai**
  ⇒ phép đo **không hề kiểm tra truy hồi**
* output `" Huế. Huế nằm bên bờ sôn…"` **chính là câu trong sổ, chép nguyên văn** ⇒ G13 khoe một
  cơ chế **verbatim copy** trong đúng ticket sinh ra để **giết** verbatim copy

Probe này nhồi thêm **6 câu HARD cùng khuôn** *"Thủ đô X là Y"* (Paris/Tokyo/Bangkok/Seoul/Bắc
Kinh/Viêng Chăn) ⇒ `h` gần trùng ⇒ ép kNN phân biệt bằng *"Việt Nam"* vs *"Pháp"*.

**Đọc:**

| | nghĩa |
|---|---|
| **Q1** λ=0.0 (kNN thuần) ≡ λ=0.5 | LM là **hành khách**, toàn bộ là băng phát lại ❌ |
| **Q2** thêm HARD **vẫn ra "Huế"** | truy hồi **THẬT**, G13 sống ✅ |
| **Q2** ra **"Paris"/"Tokyo"** | kNN không phân biệt nổi ⇒ G13 là **ảo giác n=1** ❌ |

Cột `λ=1.0` **phải giống nhau mọi hàng** (LM thuần, sổ không dùng) — đó là **đối chứng**. Nếu
khác ⇒ probe hỏng, mọi kết luận vứt.

Ngã ở 12 câu thì khỏi bàn 151k cha.

---

## 3. B5 — train graph v11 với BPE + BAKE (**~8h**, chạy trên M4)

**✅ ĐÃ XONG 2026-07-18 — xem số liệu thật ở đầu file.** Giữ phần dưới để tham khảo cách build
(lệnh/ước lượng gốc) — bước TIẾP THEO (dùng `network_v11.pkl`) ở mục **"⚡ BƯỚC TIẾP THEO"** đầu
file, không phải ở đây.

### Vì sao chỉ cần chạy lại ĐÚNG manifest v10, không cần viết gì mới

`network_v10.pkl` build **2026-07-15**, `--bpe` mặc định được thêm vào `scripts/train.py`
**2026-07-17** (TK_041) — **v10 build TRƯỚC khi có default đó** ⇒ nó dùng vocab **LEGACY**
(126,518 token, `.split()+FOLD_STRIP`), không phải BPE. Giờ `--bpe` mặc định **đã là**
`checkpoints/bpe_v1.json` ⇒ chạy lại **ĐÚNG SAME manifest**, không đổi gì khác, là ra graph BPE:

```bash
cd <repo> && git pull && export PYTHONPATH=src
ls -la data/text_corpus/*.jsonl   # xác nhận 8 file, tổng 65,160 dòng — PHẢI khớp bảng dưới
ls -la checkpoints/bpe_v1.json    # fingerprint PHẢI là 1582dad40906e69f (kiểm bằng script train — tự log ra)

nohup python -u scripts/train.py \
  --manifest config/train_v10.yaml \
  --out checkpoints/network_v11.pkl \
  --bake \
  > out/train_v11_full.log 2>&1 &

echo "PID: $!"
tail -f out/train_v11_full.log   # theo dõi — log ghi TỪNG bước, TỪNG dataset, TỪNG batch
```

**`--bake` là bắt buộc** — đây chính là thứ thiếu ở `histwiki.pkl`/`histwiki_paragraph.pkl` khiến
sinh bị treo (§0.5 caveat). `v10.pkl` xây ĐÚNG cách: `--bake` mine FRAME_BRACKET (1.8M edge)
**lúc build**, nên `v10.txt` không hề có cảnh báo `FRAME_BRACKET gate TẮT`. Thiếu cờ này = phải
chạy `bake_checkpoint_indices.py` bù sau, tốn thêm ~15-30 phút riêng và dễ quên.

### Thời gian / dung lượng (đo thật từ `train_v10_full.log`, dùng làm mốc)

```
tổng      27,786s = 7.72h
  trong đó save+bake (FRAME_BRACKET 1.8M edge) một mình  = 14,314s = 3.98h  (hơn NỬA tổng thời gian)
checkpoint ra                                             = 1.94GB
```

BPE làm chuỗi dài hơn **+7.8%** token (memory dự án, đo lúc TK_041) ⇒ ước v11 build **~8.3h**,
checkpoint **~2.0-2.1GB**. Corpus **không đổi** — cùng 8 file, cùng 65,160 dòng, cùng thứ tự
priority trong `config/train_v10.yaml` (không cần file manifest riêng cho v11).

### Sau khi xong

```bash
git add -f out/train_v11_full.log && git commit -m "TK_042 B5: train log v11 (BPE+bake)" && git push
# checkpoint 2GB KHÔNG push — giữ trên M4, dùng scp khi cần chạy probe từ đây
```

`network_v11.pkl` sẽ là checkpoint **BPE + full-graph + đã bake** ĐẦU TIÊN — thay thế mọi
checkpoint pre-BPE (`v10`, `histwiki`, `histwiki_paragraph`) làm nền cho mọi probe granularity/
G13/G14/bits-byte về sau. So sánh tokenizer công bằng (§7 dòng 170: *"hai dòng KHÔNG so được vì
khác corpus train"*) chỉ có ý nghĩa khi cả hai phía đều chạy trên graph **cùng corpus, khác chỉ
tokenizer** — B5 xong mới đo được.

---

## 4. 🔴 Phát hiện phụ — `gc.disable()` khi nạp checkpoint (**1300×**, cần ticket riêng)

Đo trên Intel 8GB, `network_v10_histwiki.pkl` (349MB, 151 430 cha):

```
KHÔNG gc.disable() → 73+ PHÚT vẫn CHƯA qua nổi bước `unpickle`
CÓ    gc.disable() → unpickle 3.4s · load_checkpoint TRỌN VẸN 60.2s
```

**Cơ chế:** unpickle dựng hàng chục triệu object Python nhỏ. GC gen-2 bị trigger liên tục và
**mỗi lần duyệt LẠI toàn bộ heap đã dựng** — trong khi chúng chắc chắn còn sống, không có gì để
thu ⇒ O(n²).

**Chữ ký nhận bệnh:** RSS **đứng im** (không dựng thêm) + CPU **99%** (GC đi bộ trên heap) +
stack vẫn trong `_pickle`. Ba dấu hiệu cùng lúc.

**`grep -rn "gc.disable" src scripts` → KHÔNG AI CẢ.** Mọi lần nạp checkpoint trong dự án —
mọi probe, mọi eval, mọi lần train nối tiếp — đã trả giá này **trong im lặng, từ đầu**. Đúng họ
lỗi im lặng TK_041 §5.2: không crash, không test đỏ, kết quả vẫn **đúng** — chỉ chậm 1300×, mà
chậm thì bị đổ cho *"máy yếu"*.

**Sửa đúng chỗ:** `gc.disable()`/`gc.enable()` (hoặc `gc.freeze()`) quanh unpickle trong
`GraphState.load` — [state.py:69](src/nn_aivie/core/state.py:69) — chứ không phải vá từng probe
như hiện tại.

**Đổi cách đọc TK_040:** lý do *"checkpoint là object Python nên nạp quá chậm"* vừa bốc hơi phần
lớn — **GC** mới là thủ phạm, không phải format. CSR vẫn đáng làm (RAM), nhưng luận cứ *tốc độ
nạp* thì không còn.

---

## 5. Sổ nợ quan sát được — 2 lần tự bịt mắt trong CÙNG một script

Ghi lại vì cùng một bài học, mắc hai lần cách nhau 20 phút:

1. Bọc `load_checkpoint` trong `contextlib.redirect_stdout(io.StringIO())` cho bảng đầu ra sạch
   ⇒ nuốt sạch progress. **`GraphState.load` đã có sẵn `logger=FileProgressLog` ghi từng pha ra
   file** — đúng quy ước dự án. Có **hai** đường để nhìn, bỏ cả hai, rồi ngồi `sample` stack +
   `lsof` offset suốt 73 phút để đoán lại thứ script sẵn sàng nói thẳng.
2. Vá xong đèn pha *nạp*, để nguyên pha *sinh*: gom 8 dòng vào list rồi in **một lần ở cuối**
   ⇒ 26 phút không biết đang ở prompt 1 hay 7.

Cả hai đã vá trong `scripts/probe_granularity.py`. **Quy tắc: process dài ⇒ log ra file, TỪNG
BƯỚC, kể cả bước nạp.**

Thêm một lần đoán sai đáng ghi: tôi hô *"thrashing"* dựa trên `Pages free: 4000` (=16MB), bị bác
sau 20 giây — swap **đứng im** 1061.5M. Quên `inactive` 2.1GB là RAM macOS thu hồi được.
`vm_stat` free ≠ RAM khả dụng.
