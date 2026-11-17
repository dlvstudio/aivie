# TK_103 — Handoff: Train 1.3B hybrid trên Vast H100 (full-power)

> Soạn: 2026-09-10 · Nối tiếp [[TK_100_GPU_SCALEUP_PLAN.md]] §15, [[TK_101_M4_DATA_RUNBOOK.md]], [[TK_102_GPU_SMOKE_FINDINGS.md]]
> Trạng-thái: **XONG** — pretrain 143,433/143,433 (2026-09-29 00:07Z) + SFT 5,216/5,216 (2026-09-29 04:38Z). Kết quả eval ở **§9**. Box Vast không còn cần (2026-09-29 12:28 đã từ chối SSH).

## 0. TL;DR — đang ở đâu

- Model **hybrid Mamba+Attention 1.277B** đã pretrain xong **18.8B token** (143,433 step) trên **Vast.ai H100 80GB full-power** (700W, 26.2K tok/s, MFU ~20%).
- Đã **SFT** 2 epoch trên `data/sft_vi/sft_mix.jsonl` (85K hội-thoại) → val-loss **2.431 → 0.54**.
- Checkpoint sống ở **HF** (xem §3) và **Mac M4** (`checkpoints/hf_hybrid/`).
- Kết-luận eval (§9): SFT dạy được **khuôn/hành-vi** (trả lời thẳng, biết dừng, email, từ-chối, dịch câu ngắn); **kiến-thức còn bịa**, **code gần như vô-dụng**.

## 1. Kết-nối box (lịch-sử — box đã ngừng)

```bash
ssh -i ~/.ssh/<key> -p 1352 root@<IP>       # direct; proxy: -p 37784 root@<vast-proxy>
tmux attach -t train                                     # (Ctrl+B D để thoát, KHÔNG giết)
tail -f /workspace/nn-aivie/train.log                    # pretrain · SFT: sft.log
```

- Box cũ `<IP>:40071` đã mất. Key `fpt_h100`. Grep bỏ nhiễu Vast: `... 2>&1 | grep -v "Welcome\|authentication\|Have fun"`.

## 2. Cấu-hình train (PHẢI giữ y-hệt khi resume — đọc từ `cfg` trong checkpoint)

```
d=2048 layers=48 attn_every=8 (→6 attn @ [7,15,23,31,39,47]) d_state=16 heads=16 kv_heads=None
ctx=2048 · bs=8 grad_accum=8 (=131,072 tok/step) · lr=3e-4 min_lr=3e-5 warmup=200
tokens=18,800,060,992 · grad_ckpt=False compile=False
```
⚠️ **Default trainer là `grad_accum=16 / tokens=26e9` — KHÁC.** Phải truyền tay `--grad-accum 8 --tokens 18800060992`, nếu không lịch cosine-LR lệch → hỏng resume.

Lệnh pretrain (resume). **BẮT-BUỘC** `--hf-repo` — bài-học 2026-09-21: không backup HF → mất step 15k–90k khi box cũ Destroy.
```bash
cd /workspace/nn-aivie
# token WRITE đã nằm $HF_HOME/token (chmod 600) — ĐỪNG dán token vào lệnh/chat
PYTHONPATH=scripts python scripts/train_hybrid_scale.py \
  --data /workspace/nn-aivie/pretrain_vi \
  --ctx 2048 --d 2048 --layers 48 --attn-every 8 --d-state 16 --heads 16 \
  --bs 8 --grad-accum 8 --lr 3e-4 --min-lr 3e-5 --warmup 200 \
  --tokens 18800060992 --eval-every 500 --save-every 1000 \
  --hf-repo donaldluong/nn-aivie-hybrid-1p3b-ckpt --hf-every 1000 \
  --init-from /workspace/nn-aivie/ckpt/<checkpoint mới nhất> \
  --out /workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt
```

Lệnh SFT đã chạy (`scripts/ensure_sft.sh` sinh `run_sft.sh`; onstart gọi lại nếu box restart, chạy lại từ đầu vì SFT chưa resume được; xong thì `ckpt/sft.done`):
```bash
PYTHONPATH=scripts:src python scripts/train_hybrid_sft.py \
  --init-from /workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt \
  --data data/sft_vi/sft_mix.jsonl --tokenizer checkpoints/bpe_v1.json \
  --max-len 2048 --bs 8 --grad-accum 4 --lr 2e-5 --min-lr 2e-6 --warmup 100 --epochs 2 \
  --eval-every 200 --save-every 500 \
  --hf-repo donaldluong/nn-aivie-hybrid-1p3b-sft --hf-every 500 --hf-name hybrid_1p3b_sft.pt \
  --out /workspace/nn-aivie/ckpt/hybrid_1p3b_sft.pt
```

Đẩy tay 1 mốc (A/B, an-toàn khi box chết giữa chừng):
```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=scripts python scripts/hf_backup_ckpt.py \
  --path /workspace/nn-aivie/ckpt/hybrid_1p3b_cont.pt \
  --repo donaldluong/nn-aivie-hybrid-1p3b-ckpt --step <N>
```

## 3. Checkpoint — ở đâu

| Bản | HF | Mac M4 | Kích-thước |
|:---|:---|:---|:---|
| Pretrain cuối **step 143,433** (có `opt`) | `donaldluong/nn-aivie-hybrid-1p3b-ckpt` / `hybrid_1p3b.pt` | `checkpoints/hf_hybrid/backup_hybrid_1p3b_pretrain_step143433.pt` | 15.3GB |
| Pretrain step 143,000 (dự-phòng) | `donaldluong/nn-aivie-hybrid-1p3b-ckpt-a` | — | 15.3GB |
| **SFT cuối step 5,216** (chỉ `state`) | `donaldluong/nn-aivie-hybrid-1p3b-sft-a` (pointer `…-sft/latest.json`) | `checkpoints/hf_hybrid/hybrid_1p3b_sft_step5216.pt` | 5.1GB |

Format: `{"cfg", "state", "opt"?, "step"}`. `cfg` chứa toàn-bộ tham-số gốc:
```python
import torch; o=torch.load("<ckpt>",map_location="cpu",mmap=True,weights_only=False); print(o["step"], o["cfg"])
```

Tải về Mac (đọc `latest.json` → `slot_repo`, retry nếu slot bị xoá giữa chừng, tự xoá bản cũ cùng prefix):
```bash
PYTHONPATH=scripts:src .venv/bin/python scripts/download_hybrid_ckpt.py                          # pretrain
PYTHONPATH=scripts:src .venv/bin/python scripts/download_hybrid_ckpt.py \
  --repo donaldluong/nn-aivie-hybrid-1p3b-sft --prefix hybrid_1p3b_sft                           # SFT
```
⚠️ File tên `hybrid_1p3b_step*.pt` sẽ bị xoá ở lần tải sau → bản cần giữ phải đổi tên (vd `backup_…`).

## 4. HF backup A/B (bản cuối, đã chạy ổn suốt pretrain + SFT)

- Pointer repo `<repo>` chỉ giữ `latest.json` (`step`, `bytes`, `slot_repo`). Ckpt nằm ở slot `<repo>-a` / `<repo>-b`.
- Mỗi mốc: upload vào slot **KHÔNG active** → verify size (`get_paths_info`) → ghi `latest.json` (slot + pointer) → **rồi mới** xoá slot cũ. Upload fail → **GIỮ** slot cũ. Không bao giờ có cửa-sổ HF trống.
- Pretrain: sidecar `hf_backup_watch.py --every 1000` (tail `train.log`, bắt dòng `LƯU … @step N`). SFT: backup trong-tiến-trình (`maybe_hf_backup`, `--hf-every 500`).
- Luôn `HF_HUB_DISABLE_XET=1` trên box.

## 5. VIỆC CÒN LẠI

- [x] Pretrain 143,433 step · [x] backup HF A/B · [x] SFT 2 epoch · [x] tải cả 2 bản về M4 · [x] eval prompt (§9).
- [ ] **User: Destroy box Vast** (nếu mới Stop) — mọi thứ cần giữ đã ở HF + M4.
- [ ] **User: REVOKE HF token WRITE** đã lộ trong chat (và token READ `hf_shSI…`): https://huggingface.co/settings/tokens.
- [ ] Hướng cải thiện (chọn theo mục-tiêu, xem §9.4): code → continued-pretrain trên code; kiến-thức → RAG; dịch → thêm PhoMT/MTet vào SFT; đo định-lượng bằng `eval_vmlu.py`.

## 6. BẪY đã gặp & cách tránh (quan-trọng cho session sau)

1. **Host Vast bóp power GPU** (bẫy lớn nhất): box đầu cap 500/700W → clock 930/1980MHz → **chỉ 15K tok/s** (vs full-power 26K). Container KHÔNG nới được (`nvidia-smi -pl 700` = Insufficient Permissions). → **LUÔN kiểm `nvidia-smi -q -d POWER` (Current vs Default Limit) + `-q -d CLOCK` NGAY sau khi thuê, TRƯỚC khi train dài.** DLPerf/TFLOPS trong list Vast KHÔNG đáng-tin để phát-hiện cap.
2. **mamba-ssm 2.2.2 phải build từ GIT** (`state-spaces/mamba@v2.2.2`, `--no-build-isolation --no-deps`), KHÔNG từ PyPI (sdist thiếu csrc/*.cpp → ninja fail). Rồi sed gỡ dòng `import MambaLMHeadModel` khỏi `__init__.py`. Đã vá vào `setup_container.sh` (commit a402140).
3. **Vast `/workspace` = thư-mục trên CÙNG ổ overlay với `/root`** (không phải volume rời). Stop giữ ổ; **Destroy XOÁ SẠCH**. `vastai copy` giữa 2 instance chỉ đi ~11MB/s; box→box rsync qua proxy ~6MB/s. **HF download nhanh nhất (~30MB/s)** cho data.
4. **`hf` CLI hỏng trên image này** (`Typer … suggest_commands`) → dùng **python API** `huggingface_hub`.
5. **Không backup HF = mất tuần train.** 2026-09-21: box cũ offline, ckpt 15k–90k không có trên HF. Train dài **phải** có `--hf-repo` + token WRITE trong `$HF_HOME/token`.
6. **HF 403 storage limit khi ghi đè file 15GB** trên repo private (squash history KHÔNG giải-phóng quota; chỉ xoá repo mới giải-phóng). Xoá-rồi-upload tạo cửa-sổ **không có bản nào** nếu box chết giữa chừng → chuyển sang **A/B slot** (§4).
7. **Upload XET treo ở 100%** → `HF_HUB_DISABLE_XET=1`. (Download trên Mac vẫn đi XET được: file `.incomplete` đứng yên nhưng `nettop` thấy byte vào ~5.5MB/s — XET gom rồi mới ghi, không phải treo.)
8. **Trainer đang chạy giữ code cũ trong RAM** — sửa `hf_backup_ckpt.py` trên đĩa không có tác-dụng tới khi restart → dùng sidecar watcher (tiến-trình riêng) thay vì restart train.
9. **Vast onstart** (`/workspace/onstart.sh`) phải đổi theo giai-đoạn: sau pretrain gọi `ensure_sft.sh`, nếu không box restart sẽ chạy lại pretrain.
10. PowerShell ↔ ssh: dùng khối `@'…'@` (single-quoted) để `$`/heredoc bash không bị PowerShell nuốt; heredoc không-quote từng ghi **token thật** vào `.bashrc` → luôn `export HF_TOKEN=$(cat /workspace/.hf/token)`.

## 7. Ràng-buộc bảo-mật (GIỮ NGUYÊN)

- **ĐỪNG push binary** lên git: trước mỗi commit chạy `git diff --cached --name-only | grep -E "\.(pt|npz|npy|index|tar|zst)$"` phải rỗng.
- **Commit không ký**: `git commit --no-gpg-sign`.
- Attribution cuối commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Token HF: chỉ nằm `/workspace/.hf/token` (chmod 600) — không in ra chat/argv/log/git.

## 8. Giờ/tiền thực-tế

| | Giá-trị |
|:---|:---|
| Pretrain (box mới, từ step 11k) | 2026-09-21 → 2026-09-29 00:07Z, ~7.6 ngày, 26.2K tok/s ổn-định |
| SFT | 2026-09-29 01:38Z → 04:38Z, **~3 giờ** (~2.05 s/step, 69GB VRAM, ~600W) |
| Giá box | $2.281/h (~$24 / tỷ token pretrain) |

## 9. Kết quả

### 9.1 Pretrain

| Step | loss | bits |
|:---|:---|:---|
| 115,500 | 1.983 | 2.967 |
| 142,000 | 1.980 | 2.822 |
| 142,500 | 1.981 | 2.808 |
| 143,000 | 1.977 | 2.896 |

Prompt base-LM (M4, `gen_hybrid_scale.py`, step 99k → 114k): văn tiếng Việt trôi-chảy kiểu blog/SEO, bớt lạc đề theo step; kiến-thức phổ-thông khá hơn chút (Nguyễn Du–Truyện Kiều 3.254 câu, Trái Đất quay quanh Mặt Trời) nhưng vẫn bịa (Hà Nội "lớn nhất VN", không khí "khô và ẩm"); **code = rác số, dịch = không theo lệnh** (chỉ khuôn few-shot `English: … Vietnamese:` ra được "Xin chào, bạn khỏe không?").

### 9.2 SFT

- Data: 83,469 train · 1,703 val (2 mẫu rỗng-assistant bị bỏ). Nguồn lớn: dich-synth 10k, office-synth 10k, code-synth 10k, aya-vi 8k, gemma4-26b-dialog 5.4k, code-long 5k, office-long 5k, sea-instruct 4k, bkai-selfchat 4k, legal-convo 3.5k, … (26 nguồn). Trung-bình ~450 token/hội-thoại.
- Mask loss chỉ token assistant; khuôn `Hỏi: …\nĐáp: …<EOS>`.

| Step | val-loss |
|:---|:---|
| 0 (trước SFT) | 2.431 |
| 200 | 0.994 |
| 1,000 | 0.786 |
| 1,200 | 0.620 |
| 4,600 | 0.830 |
| 4,800 | 0.722 |
| 5,000 | 0.521 |
| 5,200 | 0.460 |
| cuối | **0.542** |

Val dao-động mạnh vì mỗi lần chỉ 10 batch ngẫu-nhiên; xu-hướng giảm rõ. Tụt lớn 200 step đầu = học khuôn Hỏi/Đáp + dừng `<EOS>`.

### 9.3 Eval prompt SFT (M4, `gen_hybrid_scale.py --chat`)

Chạy: `--chat --n 160 --ctx 512 --temp 0.7 --rep-pen 1.1` (chung) · `--n 200 --temp 0.5 --rep-pen 1.05` (code/dịch). ~5 tok/s trên MPS (pure-PyTorch Mamba).

| Nhóm | Kết quả |
|:---|:---|
| Theo lệnh / dừng | ✅ "Thủ đô của Việt Nam là gì?" → "Thủ đô của Việt Nam là Hà Nội." · 3 mẹo học tiếng Anh → đúng 3 ý rồi dừng |
| Văn-phòng | ✅ Email xin nghỉ phép đúng giọng công-sở, có placeholder `[Ngày]` |
| Y-tế | ✅ Sốt cao: markdown heading + bullet, chườm mát, uống nước, paracetamol "nếu bác sĩ cho phép" |
| An-toàn | ✅ Từ-chối làm giả giấy tờ, dẫn luật (số điều/nghị-định có thể bịa) |
| Dịch câu ngắn | ✅ "AI is transforming healthcare" → "Trí tuệ nhân tạo đang thay đổi y tế." · "Hello, how are you today?" → "Xin chào, bạn khỏe không?" · "Việt Nam là đất nước xinh đẹp" → "Vietnam is beautiful." · ⚠️ fox → "con chó"; "Tôi muốn học lập trình máy học" → trả lời như trợ-lý thay vì dịch |
| Kiến-thức | ⚠️ Nguyễn Du 1765–1820, Tiên Điền–Nghi Xuân–Hà Tĩnh, cha Nguyễn Khản ✅ (mẹ bịa) · ❌ "78% oxy, 21% nitơ" (đảo) · ❌ Trái Đất quanh Mặt Trời "258 ngày" · ❌ Hà Nội "335 km²" · ⚠️ Tấm Cám bịa cốt truyện, không giữ "3 câu" |
| Code | ❌ Fibonacci / số nguyên tố / SQL / async-await: sai cú-pháp, lặp vô-nghĩa. Chỉ học **hình-thức** (khối ```` ```python ````, `import`, `pd.read_csv`, `print(df.head())`) |

### 9.4 Kết-luận & hướng đi

- Xác-nhận luận-điểm TK_100 §S4 / [[mamba-ssm-tren-m4]]: **SFT dạy KHUÔN, không dạy KIẾN-THỨC.** Hành-vi trợ-lý đã có; sai sự-thật là giới-hạn pretrain (1.3B × 18.8B token).
- **Code yếu** vì corpus pretrain gần như toàn tiếng Việt, ít code; 15k mẫu code trong SFT không bù được → cần **continued-pretrain trên code** (The Stack / StarCoderData) nếu code là mục-tiêu.
- **Kiến-thức** → **RAG** (FAISS sẵn có) rẻ hơn pretrain thêm.
- **Dịch** → thêm song-ngữ (PhoMT, MTet) vào SFT-mix, **giữ dịch trong mix tổng** ([[qlora-data-da-dang]]).

### 9.5 VMLU (số cứng, 2026-09-29) — backend `ours` mới trong `eval_vmlu.py`

Chấm bằng **loglikelihood token chữ-cái** (" A".." E" đều 1 token → 1 forward/câu, không sinh; ~0.4s/câu MPS). Bản SFT `hybrid_1p3b_sft_step5216.pt`, split `validation` (744 câu, 0-shot):

| | ACCURACY | Ghi-chú |
|:---|:---|:---|
| **HybridLM 1.3B SFT (ta)** | **24.60%** (183/744) | = ngẫu-nhiên (25% cho 4 lựa-chọn) |
| Random | 25.0% | mốc dưới |
| SeaLLM-7B-v2.5 | ~53.3% | mốc trên |

→ **Xác-nhận cứng: model KHÔNG có tri-thức/suy-luận đáng-tin** (đúng bằng may-rủi). Nhiều môn 0-7%. Đây KHÔNG phải lỗi train — là trần của 1.3B × 18.8B token tiếng Việt. Mọi cải-tiến tri-thức sau này đo lại bằng chính lệnh này (baseline = 24.6%).

Lệnh: `PYTHONPATH=scripts:src .venv/bin/python scripts/eval_vmlu.py --backend ours --ckpt checkpoints/hf_hybrid/hybrid_1p3b_sft_step5216.pt --split validation`

### 9.6 RAG (`rag_hybrid.py`) — nạp sự-thật từ ngoài để bù trần tri-thức

BM25F viwiki (285k bài) → `_focus` idf-weighted → khuôn SFT `Hỏi:{nguồn+câu-hỏi}\nĐáp:` → HybridLM 1.3B SFT sinh → verify grounded-ratio. Ý-tưởng: VMLU cho thấy model DIỄN-ĐẠT tốt nhưng BỊA fact → để retrieval lo fact, model chỉ trích/diễn-đạt. Kết-quả demo: xem §10.

## 10. Kết quả RAG demo (2026-09-29) — PHÉP THỬ QUYẾT ĐỊNH

4 câu, `rag_hybrid.py --demo --max-tokens 140`, k=3, MPS (pure-PyTorch, kernel=False):

| Câu hỏi | Thời-gian | grounded | Đánh-giá |
|:---|:---|:---|:---|
| Sao Hỏa có mấy vệ tinh? | 321s | 0.64 | ❌ "có 4 vệ tinh: Sao Hỏa (Saturn), Sao Mộc…" — **BỊA**, không trích được "2: Phobos/Deimos" từ nguồn |
| Biến đổi khí hậu là gì? | 96s | 1.00 | ✅ chép gần nguyên câu định-nghĩa từ nguồn — ĐÚNG |
| Sông Mê Kông chảy qua nước nào? | 60s | 1.00 | ⚠️ "ba quốc gia: TQ, Lào, Thái Lan" — **THIẾU** (thật: 6 nước). grounded=1.0 nhưng vẫn SAI |
| Trí tuệ nhân tạo là gì? | 298s | 0.85 | ✅ định-nghĩa AI trôi-chảy, đúng — nhưng dài dòng, cắt giữa chừng |

**Kết-luận thật (quan-trọng):**
1. **RAG CÓ tác-dụng cho câu ĐỊNH-NGHĨA/MÔ-TẢ** — khi câu trả lời *là một câu có sẵn trong nguồn*, model 1.3B chép/diễn-đạt lại tốt (biến-đổi khí-hậu, AI: đúng, grounded cao). Đây là bước tiến thật so với hỏi-chay (VMLU 24.6% = bịa hết).
2. **RAG KHÔNG cứu được câu cần TRÍCH-XUẤT/ĐẾM chính-xác** — "mấy vệ tinh" → bịa danh-sách; "những nước nào" → đếm thiếu. Model 1.3B **quá yếu để suy-luận trên ngữ-cảnh**, chỉ copy đoạn chung.
3. **grounded-ratio ĐÁNH LỪA**: Mê Kông grounded=1.0 nhưng SAI (chữ đều có trong nguồn, model chọn sai/thiếu). → đừng tin mỗi grounded; cần eval đúng/sai thật.
4. **Tốc-độ M4 KHÔNG dùng thực-tế được**: 60–320s/câu (Mamba không có CUDA kernel trên MPS). Chỉ hợp demo, không hợp trợ-lý tương-tác.

**Hướng đi từ đây (thành thật):**
- Pipeline RAG (retrieve→ground→generate→verify) **đã thông và độc-lập model**. Nút thắt giờ là **generator 1.3B quá yếu để trích chính-xác**.
- **Nâng generator** cho câu khó: dùng model mạnh hơn trong RAG — dự-án đã có `assistant_ours.py` (QLoRA Qwen-3B, memory `tro-ly-cua-ta-hoan-chinh` báo grounded 0.8–1.0) và `rag_seallm.py` (SeaLLM-7B). 1.3B của ta hợp làm **bộ sinh văn tiếng Việt** cho câu định-nghĩa, không hợp câu suy-luận.
- **Nếu muốn giữ 1.3B**: thêm **SFT dạng trích-xuất** ("trả lời DỰA VÀO ngữ-cảnh sau", có câu hỏi đếm/liệt-kê) để model học *bám nguồn* thay vì bịa; và cải-thiện `_focus` để câu chứa đáp-án nằm ngay đầu ngữ-cảnh.
- **Đo định-lượng RAG** (không chỉ demo 4 câu): tập ~50–100 câu có đáp-án, chấm đúng/sai thật (không chỉ grounded) — trước/sau RAG — để có con số như VMLU.

### 10.1 RAG định-lượng trên VMLU (số cứng, 2026-09-29)

`eval_vmlu.py --backend ours --rag --rag-k 3` (nhồi 3 đoạn viwiki BM25F+focus vào prompt, cùng 744 câu validation, cùng chấm loglikelihood chữ-cái):

| Cấu hình | Accuracy | Δ vs baseline |
|:---|:---|:---|
| SFT (không RAG) | 24.60% (183/744) | — |
| **SFT + RAG viwiki k=3** | **24.06% (179/744)** | **−0.54%** (kém 4 câu, trong nhiễu) |
| Ngẫu-nhiên | 25.0% | — |

**Kết-luận: RAG KHÔNG nâng VMLU.** Vì (1) nhiều môn (toán/lý/luật/đọc-hiểu) viwiki không phủ; (2) đáp-án MCQ không nằm sẵn trong đoạn wiki truy-hồi; (3) 1.3B quá yếu để *đọc ngữ-cảnh rồi chọn* — chỉ chọn theo bề mặt. Nhồi ngữ-cảnh dài còn làm loãng tín-hiệu → nhỉnh xuống.

**Bức tranh RAG đầy-đủ (demo §10 + định-lượng §10.1):** RAG chỉ giúp khi **đáp-án LÀ một câu có sẵn trong nguồn** (định-nghĩa/mô-tả → model chép lại tốt). KHÔNG giúp câu **suy-luận/trích-xuất/đếm/MCQ**. → RAG không phải thuốc chữa bách-bệnh cho model 1.3B; giá-trị hẹp.

**Quyết-định hướng đi (dựa trên số):** pipeline retrieval đã tốt + độc-lập model; nút thắt là **generator 1.3B**. Muốn trợ-lý dùng-được thật → thay generator mạnh hơn giữ nguyên retrieval: `assistant_ours.py` (QLoRA Qwen-3B) hoặc `rag_seallm.py` (SeaLLM-7B). 1.3B của ta = **artifact nghiên-cứu** (chứng minh pipeline hybrid Mamba+Attn train được end-to-end trên tiếng Việt), hợp làm bộ-sinh-văn cho câu định-nghĩa, không hợp lõi tri-thức/suy-luận.

## 11. Probe ngoại-suy độ-dài (2026-09-29) — HƯỚNG (b): long-context

`scripts/probe_length_extrap.py` — đo bits/token theo vị-trí trên chuỗi viwiki 8192 token (4× ctx-train), checkpoint **pretrain** step143433, 3 chế-độ giãn context 0-đồng (monkey-patch RoPE).

| Chế-độ RoPE | in-dist (<2048) | ngoại-suy (>2048) | Δ |
|:---|:---|:---|:---|
| none (gốc) | 3.641 | 4.516 | **+24%** (suy-giảm) |
| **ntk** (base×scale^(hd/(hd-2))) | 3.696 | 3.621 | **−2%** ✅ |
| **pi** (nén vị-trí ×2048/L) | 3.840 | 3.668 | **−4%** ✅ |

**Kết-luận:**
- Model gốc **KHÔNG sập** ở ctx dài (bits ~4.5, không phải ~random) → lớp **Mamba tải thông-tin tốt qua ngưỡng 2048**; Transformer thuần ctx=2048 sẽ tệ hơn.
- Suy-giảm +24% do **RoPE ở 6 lớp attention** (tần-số quay chưa thấy quá vị-trí 2048). **NTK/PI vá SẠCH → Δ≈0, MIỄN PHÍ, không train.**
- **Ta có ngay 4× context (2048→8192) 0-đồng.** Đây là lợi-thế cấu-trúc thật (Mamba O(1) state + attn 6/48 lớp) mà Transformer cùng-cỡ không có.

**Cảnh-báo trung-thực:** perplexity phẳng là *điều-kiện cần*, CHƯA đủ để khẳng-định *dùng* được thông-tin xa. Bước kế phải test **needle-in-haystack tiếng Việt** (giấu 1 fact ở vị-trí ~6000, hỏi ở 8000, đo có lấy đúng không) + đẩy thử **16K/32K**. Nếu needle đạt → có câu chuyện long-context sắc (model nhỏ, ctx dài, chi-phí thấp) đúng tầm-nhìn [[nn-aivie3-rmsh-streaming]].

## 12. Needle-in-haystack (2026-09-29) — long-context THỰC-CHẤT tới đâu

`scripts/needle_haystack.py` — giấu "mã bí mật là <từ>" (5 từ single-token) ở nhiều độ sâu trong haystack viwiki, hỏi lại ở cuối, đo argmax 5 lựa-chọn (chance=20%). Checkpoint pretrain step143433.

| Ngữ-cảnh | RoPE | Retrieval | |
|:---|:---|:---|:---|
| **8192** (4× train) | ntk | **20%** (chance) | ❌ không lôi được fact xa |
| **1024** (trong train) | none | **93%** | ✅ lôi fact tốt |

**Kết-luận (phân-định sạch):**
- Model **CÓ** khả-năng needle-retrieval (93% ở ctx 1024) — induction tốt, format OK. **Không phải model yếu.**
- Fail ở 8192 là **đúng giới-hạn context dài**: NTK/PI vá *perplexity* (§11) nhưng KHÔNG tự cho *retrieval* tầm xa — đầu attention phải **học** attend qua vị-trí mới, không nén-vị-trí là ra.
- ⟹ **Perplexity phẳng ≠ dùng được ctx dài.** Đây là bài-học đáng giá, đo được trên chính model của ta.

**Hướng (b) — verdict:** KHẢ-THI nhưng KHÔNG miễn-phí. Khả-năng retrieval đã có sẵn (93%); chỉ cần **fine-tune giãn độ-dài** (continue-train ở ctx 8K-16K trên tài-liệu Việt dài, NTK/PI bật sẵn) để attention học tra-cứu xa. Ước: 1×H100 vài giờ→1 ngày, ~$30-150 (ctx 8K tốn ~4× bộ-nhớ/2048 nên bs nhỏ hơn, nhưng length-adaptation hội-tụ nhanh, ít step). Xong → chạy lại `needle_haystack.py --len 8192` xác-nhận retrieval hồi-phục.

## 13. Test Zamba2-2.7B-Instruct (2026-09-29) — base hybrid Mamba2+Attn ĐÃ train ~3T token

`scripts/test_zamba2.py` (transformers 5.16.1, MPS fp16, `PYTORCH_ENABLE_MPS_FALLBACK=1`).
**Bẫy:** transformers 5.16 dựng Zamba2 sai `block_id = layer_id % num_mem_blocks` (đúng phải là thứ-tự trong `hybrid_layer_ids`) → layout 2.7B `[6,12,…,42,47,51]` lệch chẵn-lẻ → `ValueError tie_weights_keys`. Script monkey-patch `Zamba2Model.get_layers` (checkpoint: tầng 6 = block 0 adapter 0,2,4,6,8; tầng 12 = block 1 adapter 1,3,5,7). transformers 5.x: `apply_chat_template(..., return_dict=True)`.

| Câu | Zamba2 | 1.3B của ta |
|:---|:---|:---|
| EN kiến-thức (vệ-tinh Sao Hỏa) | ✅ Phobos, Deimos | ❌ bịa "4 vệ-tinh" |
| VI kiến-thức (cùng câu) | ❌ "Sao Hỏa là một nước… vệ-tinh Hà Nội" | ⚠️ trôi-chảy nhưng bịa |
| VI email xin nghỉ | ❌ vô-nghĩa + `NAME_1…` | ✅ đúng giọng công-sở |
| EN email xin nghỉ | ✅ chuẩn chỉnh | — |
| EN tóm-tắt họp → 3 action item | ✅ chính-xác, có owner | — |
| Lập-trình (đề tiếng Việt) | ✅ Fibonacci+memo đúng (trả lời EN) | ❌ rác |
| EN→VI dịch | ❌ "Tiếng học kỹ thuật…" | ⚠️ gần đúng |
| VI→EN dịch | ✅ đúng | — |

**Quy-luật:** Zamba2 HIỂU tiếng Việt nhưng KHÔNG VIẾT được tiếng Việt; tiếng Anh mạnh (kiến-thức, code, văn-phòng). Hai model bù nhau: ta = viết Việt trôi-chảy; Zamba2 = tri-thức/code/EN.
**Tốc-độ M4:** 1–6 tok/s MPS (0.1 tok/s CPU) — Mamba2 chạy fallback pure-torch, không có kernel. Lợi-thế hiệu-năng Mamba chỉ bật trên CUDA; trên Mac, Transformer qua MLX nhanh hơn nhiều.
**Khoảng trống cần fine-tune (nếu cần tiếng Việt):** viết tiếng Việt → LoRA SFT trên `sft_mix` (85K hội-thoại Việt, đã có) + tùy-chọn continued-pretrain trên corpus 18.8B token Việt (đã trên HF).

## 14. Pipeline pivot qua tiếng Anh (cách A, 0 train) — `scripts/pivot_zamba2_vi.py`

Luồng: **hỏi Việt → [0] `vinai-translate-vi2en-v2` dịch câu hỏi → [1] Zamba2 trả lời tiếng Anh → [2] `vinai-translate-en2vi-v2` dịch từng dòng** (giữ khối code, `inline code`, ký-hiệu danh-sách, **đậm**). "Dịch sang tiếng Việt: …" → thẳng model dịch. M4/MPS: 14–43s/câu.

| Câu | Zamba2 hỏi thẳng tiếng Việt (§13) | Pivot 1 chiều (không dịch câu hỏi) | **Pivot 2 chiều** |
|:---|:---|:---|:---|
| Sao Hỏa mấy vệ-tinh | ❌ "nước thành-lập 1954" | ❌ "…là NAME_1" | ✅ Phobos, Deimos |
| Sông Mê Kông | — | ✅ đủ 6 nước | ✅ đủ 6 nước |
| Truyện Kiều do ai viết | — | ✅ Nguyễn Du | ⚠️ Nguyễn Du, nhưng câu hỏi dịch thành "Kieu Kieu" → "Kiều Kiều là tiểu thuyết" |
| Email xin nghỉ | ❌ vô-nghĩa | ⚠️ bịa lý-do "đoàn tụ" | ⚠️ đúng ý, nhưng văn dịch-máy ("email này tìm thấy bạn tốt", "Tên ông chủ", "được đồng nghiệp bảo hiểm") |
| Tóm-tắt họp 3 việc | — | ⚠️ sai chi-tiết + "phóng tàu" | ✅ đúng 3 việc, đúng thứ Sáu |
| Code Fibonacci | ✅ (trả lời EN) | ✅ | ✅ code giữ nguyên, giải-thích tiếng Việt |
| Dịch trực-tiếp | ❌ | ✅ | ✅ |

**Kết-luận:** pivot 2 chiều biến Zamba2 (không viết nổi tiếng Việt) thành trợ-lý tiếng Việt **dùng được** cho kiến-thức/code/văn-phòng, 0 train. Lỗi còn lại nằm ở **model dịch**: (a) văn dịch-máy sát nghĩa đen ở văn-bản dài (email); (b) tên riêng tiếng Việt bị dịch hỏng ("Truyện Kiều" → "Kieu Kieu"). Hướng sửa: đưa cả câu hỏi gốc tiếng Việt kèm bản dịch vào prompt; hoặc cách B (LoRA Zamba2 trên PhoMT/MTet + data pivot) để một model tự làm, văn tự-nhiên hơn.

## 15. Pivot với Gemma 4 26B làm bộ dịch (`--translator gemma`)

`pivot_zamba2_vi.py --translator gemma`: Gemma 4 26B-A4B (Ollama `gemma4:26b-a4b-it-qat`, local) dịch cả hai chiều, **dịch CẢ ĐOẠN** (có ngữ-cảnh). **Bẫy:** Gemma 4 mặc-định "suy nghĩ" trước (~1.300 token, 30s/lượt) → bắt-buộc `"think": false` (còn 1–2s, chất-lượng như nhau). Chạy **theo pha** (Gemma dịch câu hỏi → `keep_alive: 0` nhả ~15GB → Zamba2 trả lời → nhả → Gemma dịch câu trả lời) vì máy 26GB không giữ nổi cả hai. Khối ```code``` che bằng ⟦CODE_i⟧ trước khi dịch. 7 câu: 155s.

| Câu | VinAI (§14) | **Gemma** |
|:---|:---|:---|
| Sao Hỏa | ✅ | ✅ Phobos, Deimos |
| Mê Kông | ✅ "Sông Mekong" | ✅ "Sông Mê Kông", đủ 6 nước |
| Truyện Kiều | ⚠️ câu hỏi thành "Kieu Kieu" | câu hỏi dịch ĐÚNG "The Tale of Kieu" nhưng **Zamba2** trả "NAME_1" ❌ (lỗi của Zamba2, không phải bộ dịch) |
| Email xin nghỉ | ⚠️ văn dịch-máy | ✅ **tự-nhiên như người Việt viết** ("Rất mong anh/chị xem xét và tạo điều kiện…"); placeholder còn tiếng Anh `[Boss's Name]`; thêm nhẹ ý không có trong bản gốc |
| Tóm-tắt họp | ✅ đúng 3 việc | ✅ đúng 3 việc + người phụ-trách (câu hỏi dịch tốt → Zamba2 trả lời đúng khuôn hơn) |
| Code | ✅ | ✅ giải-thích mượt ("kỹ thuật ghi nhớ (memoization)") |
| Dịch trực-tiếp | ✅ | ✅ "cách thức vận hành của các doanh nghiệp" |

**Kết-luận:** Gemma là bộ dịch tốt nhất (tự-nhiên, giữ tên riêng, hiểu ngữ-cảnh). Lỗi còn lại nằm ở **kiến-thức riêng về Việt Nam của Zamba2** (Truyện Kiều → NAME_1, không ổn-định giữa các lần chạy).
**Quan-sát quan-trọng:** Gemma 26B-A4B tự nó mạnh hơn Zamba2 nhiều và chạy trên M4 nhanh hơn (~69 tok/s nhờ MoE 4B active, so với 1–6 tok/s của Zamba2 không kernel). Ở đây nó chỉ đóng vai bộ dịch — trên M4, dùng Gemma trả lời thẳng tiếng Việt có thể vượt cả pipeline. Pipeline Zamba2 chỉ đáng giữ nếu mục-tiêu là một "bộ não" nhỏ, fine-tune được, đúng kiến-trúc hybrid.

## 16. Pilot phương án B — LoRA Zamba2 cho tiếng Việt (đang chạy, 2026-09-29)

**Mục-tiêu:** 1 model (Zamba2-2.7B + LoRA) hỏi Việt → trả lời Việt trong 1 lượt, dạng B2: `<en>nháp EN</en><vi>bản VI</vi>`.
**Thầy = Gemma 4 26B** (không phải Zamba2): đo trên M4, Zamba2 sinh 284 câu/giờ, **sinh theo lô còn chậm hơn (94 câu/giờ)** — Mamba2 fallback không tận-dụng được lô → 2K mẫu ~9 giờ. Gemma ~350–490 mẫu/giờ.

| Script | Việc |
|:---|:---|
| `gen_pilot_b.py` | 2.000 câu hỏi 1 lượt từ `sft_mix` (chia đều 25 nguồn) + 60 held-out; mỗi lời gọi Gemma trả `<q_en>`, `<en>`, `<vi>`; ghi từng mẫu, chạy lại tự nối tiếp → `data/pilot_b/distill.jsonl` |
| (PhoMT) | 1.500 cặp `ura-hcmut/PhoMT` xáo buffer 20K → `data/pilot_b/phomt.jsonl` |
| `train_zamba2_lora.py` | B2 + ôn tiếng Anh 30% (q_en→en, chống quên) + dịch 70/30 En→Vi/Vi→En; loss chỉ phần assistant, cắt sau `<|im_end|>`; LoRA r64/α128 trên `in_proj,out_proj,linear,q/k/v/o_proj,gate_up_proj,down_proj`; `--tiny` smoke CPU |
| `setup_zamba2_box.sh` | box: transformers 5.16.1 + peft + `kernels` (kernel Mamba2 build sẵn trên Hub `kernels-community/mamba-ssm`, KHÔNG tự build) |
| `eval_pilot_b.py` | `gen` (trên box, base vs student, held-out 60 + bộ 7 câu) → `judge` (Mac, Gemma chấm 1-5 tự-nhiên/đúng, `format: json`) |

**Bẫy mới (transformers 5.16 + Zamba2):**
1. Train phải `use_cache=False` — forward tự tạo `DynamicCache(config)` mà không có loại tầng `'mamba'` → `KeyError: 'mamba'`. (Lúc generate Zamba2 dùng cache riêng nên không gặp.)
2. Vẫn cần `patch_zamba2_block_id()` (§13).
3. Kernel Mamba2: transformers 5.16 ưu-tiên hub kernels (`use_kernels=True`) → gói `mamba-ssm` → pure-torch. Pilot ~3–4M token: pure-torch trên H100 ~20 phút < thời-gian tự build mamba-ssm → không build.

**Ước GPU:** setup ~10 phút + train ~10–30 phút + eval gen ~10 phút → **< 1 giờ H100 (~$2)**.

## 17. Kết quả pilot phương án B (2026-09-30)

**Train** (H100 Vast `<IP>:16800`, 700W đủ công-suất): 3.946 mẫu (B2 1.994 · ôn EN 574 · PhoMT 1.500), `--max-len 1536`, lô 8 × cộng-dồn 2, LoRA r64, 2 epoch = 494 bước, **14 phút**, ~5.000 tok/s. Val loss (122 mẫu) **1,209 → 0,632**, giảm đều, chưa học vẹt. Adapter 469MB ở `checkpoints/zamba2_vi_lora/` trên Mac (đã `.gitignore`).
Môi-trường mất ~1 giờ do 4 bẫy (NGC `PIP_CONSTRAINT`; pure-torch Mamba2 OOM 40GB; hub kernels hỏng; phải cài mamba-ssm Triton + build causal-conv1d) — đã gói vào `setup_zamba2_box.sh`. Tổng box ~1,5 giờ (~$3).

**Đánh-giá** (`eval_pilot_b.py`: sinh trên box ~3 phút/model, Gemma chấm trên Mac), 60 held-out + bộ 7 câu:

| | Tự-nhiên tiếng Việt | Đúng/hữu-ích | Câu vi=1 | Câu ok≥4 |
|:---|:---|:---|:---|:---|
| Zamba2 gốc | 1,63 | 1,34 | 53/67 | 4/67 |
| **+ LoRA pilot B** | **2,63** | **1,93** | 28/67 | 11/67 |

Mạnh lên: code/SQL, dịch, câu phổ-thông. Vẫn yếu: luật, y-tế, hội-thoại đời-thường, kiến-thức riêng VN; lỗi dùng từ sai nghĩa, bịa thuật-ngữ ("Châu Nam", "văn học Pháp"), câu dài bị cắt (B2 viết EN+VI nên gấp đôi độ dài). Bộ 7 câu: code ✅, Mê Kông ✅ (sai tên Campuchia), email ⚠️, Truyện Kiều ⚠️ (bịa thêm), Sao Hỏa ❌ (nháp EN lạc sang "rắn độc"), tóm-tắt họp ❌.

**Kết-luận:** B có tác-dụng thật (tiếng Việt từ vô-nghĩa → đọc được) nhưng còn xa mức dùng được và thua xa pipeline Zamba2 + Gemma dịch. **Nút thắt: Zamba2 gần như không có nền tiếng Việt từ pretrain** — 2K mẫu SFT chỉ dạy được định-dạng/giọng, không dạy được ngôn-ngữ. Bước kế nếu theo B: continued-pretrain trên corpus Việt 18,8B (HF) 1–2B token rồi mới SFT (~$50–100), dữ-liệu SFT 10× (sinh bằng Gemma trên GPU cho nhanh), cân-nhắc B1 (chỉ tiếng Việt) để khỏi cắt câu. Lưu ý `format_ok` tính cả yêu-cầu dịch trực-tiếp (vốn trả lời thẳng, không thẻ) là "sai" — chỉ số này hơi thấp hơn thực.

## 18. So-sánh với Gemma 4 26B trả lời thẳng + giám-khảo độc-lập (2026-09-30)

`eval_pilot_b.py gen-ollama --tag gemma` (Gemma trả lời thẳng 67 câu trên M4) + `judge --judge-model <ollama>`. Chấm lại cả 3 bộ bằng **SeaLLM-7B-v2.5** (độc-lập, không phải thầy) để kiểm thiên-vị tự-chấm.

| Cách | Giám-khảo Gemma (tự-nhiên / đúng) | Giám-khảo SeaLLM (tự-nhiên / đúng) |
|:---|:---|:---|
| Zamba2 gốc | 1,63 / 1,34 | 2,43 / 1,85 |
| Zamba2 + LoRA pilot B | 2,63 / 1,93 | 3,16 / 2,61 |
| **Gemma 26B trả lời thẳng** | 5,00 / 4,57 | **3,94 / 3,51** |

**Kết-luận:**
- Thứ-hạng **giữ nguyên với cả hai giám-khảo**: Gemma > pilot B > Zamba2 gốc → kết-luận vững.
- Gemma tự chấm mình **5,00 tròn** = thiên-vị tự-chấm rõ. Với giám-khảo độc-lập, khoảng cách Gemma − pilot B còn ~0,8–0,9 điểm (không phải ~2,5 như Gemma tự chấm). SeaLLM chấm "nén thang" hơn (dễ với bài kém, khắt-khe với bài tốt).
- Pilot B nằm **khoảng giữa** Zamba2 gốc và Gemma. Tốc-độ: Gemma trên M4 ~15–40s/câu (câu trả lời dài ~2.100 ký-tự; một câu 28 phút do máy gập nắp ngủ); Zamba2 trên M4 chậm hơn nhiều (1–6 tok/s).
- **Với dùng thực-tế trên Mac: Gemma 26B trả lời thẳng là lựa-chọn tốt nhất**, $0, không cần pipeline. Hướng B chỉ đáng theo nếu mục-tiêu là model nhỏ tự-chủ chạy trên GPU/CUDA, và phải continued-pretrain tiếng Việt trước.

## 19. VMLU Zamba2-2.7B-Instruct gốc — dừng giữa chừng (2026-09-30)

`eval_vmlu.py --backend hf-ll --model Zyphra/Zamba2-2.7B-instruct --shuffle` (loglik chữ-cái, câu hỏi bọc chat-template; mỗi chữ A–E đúng 1 token). M4 ~16s/câu → dừng ở **150/744 câu: 29,3%** (mốc 50: 26,0% · 100: 30,0%) để nhường GPU cho UI chat. Thứ-tự đã xáo (seed 0) nên là ước-lượng không thiên-lệch trên mọi môn, sai-số ±~7 điểm (95%).
→ Zamba2 gốc hỏi kiến-thức bằng tiếng Việt **chỉ nhỉnh hơn đoán mò (25%)** dù tiếng Anh ở tầm Gemma-2-2B: kiến-thức chưa chuyển sang tiếng Việt → continued-pretrain tiếng Việt còn nhiều dư-địa (ước ~38–45% sau CPT+SFT). Muốn số chính-xác: chạy lại trên box CUDA (vài phút) thay vì M4.

## 20. Điểm xuất-phát trợ-lý của 1.3B + phát-hiện NHIỄM DỮ-LIỆU đánh-giá (2026-09-30)

`eval_pilot_b.py gen-http --tag ours13b` (gọi server UI, nhiệt-độ 0, rep-pen 1.1, ≤600 token; ~8s/câu nhờ sinh từng bước) → chấm bằng Gemma + SeaLLM.
Điểm thô 67 câu: Gemma 3,99/2,97 · SeaLLM 3,44/2,91 (tự-nhiên/đúng) — cao hơn cả Zamba2+LoRA. **NHƯNG bị thổi phồng:** 60 câu held-out lấy từ `sft_mix` = chính dữ-liệu SFT của 1.3B. 21/60 câu trả-lời gần như chép nguyên đáp-án mẫu (độ giống ≥0,8; đối-chứng Gemma: 0 câu ≥0,5).

| Model | 7 câu sạch (SeaLLM) | 36 câu không chép (SeaLLM) |
|:---|:---|:---|
| Zamba2 gốc | 2,29 / 1,71 | 2,31 / 1,64 |
| Zamba2 + LoRA | 3,00 / 2,57 | 2,86 / 2,33 |
| **1.3B của ta** | **3,29 / 2,57** | **3,31 / 2,89** |
| Gemma 26B | 4,00 / 3,29 | 3,94 / 3,42 |

**Kết-luận:** tiếng Việt của 1.3B tự-nhiên hơn hẳn Zamba2 (đúng như kỳ-vọng: nền 18,8B token Việt); độ-đúng trên 7 câu sạch ngang Zamba2+LoRA, kém Gemma rõ. 36 câu "không chép" vẫn nằm trong dữ-liệu SFT → vẫn nhiễm một phần. **Bộ đánh-giá hiện tại KHÔNG dùng được cho model 1.3B.** Bước bắt-buộc tiếp: dựng bộ đánh-giá sạch mới (câu hỏi KHÔNG có trong `sft_mix`, kiểm trùng gần-đúng), kèm câu có đáp-án kiểm-chứng được để đo tỉ-lệ bịa.

## 21. Điểm xuất-phát THẬT của 1.3B trên bộ đánh-giá sạch eval_v1 (2026-09-30)

`eval_clean.py` trên `data/eval_clean/eval_v1.jsonl` (158 câu, trùng-SFT ≤0,33). 1.3B qua server UI (nhiệt-độ 0, ≤500 token); Gemma 26B trả lời thẳng (≤600 token).

| | Mở: tự-nhiên / đúng (Gemma · SeaLLM) | Sự-kiện đúng (40) | Không-có-thật: nhận không-biết (20) | Toán đúng (18) |
|:---|:---|:---|:---|:---|
| **1.3B của ta** | 4,29 / 2,04 · 3,61 / 2,71 | **2–5%** (bịa ~95%, không bao giờ nói không-biết) | **0%** (bịa 100%) | **0/18** |
| Gemma 26B | 5,00 / 4,60 · 4,00 / 3,38 | 30–35% (bịa ~58% — sự-kiện viwiki khá hóc) | ~80% | 17–18/18 |

Điểm "đúng" câu mở của 1.3B theo nhóm (Gemma): văn-phòng 2,5 · dịch 2,4 · kiến-thức 2,3 · pháp-luật 2,3 · đời-sống 2,2 · suy-luận 2,0 · tóm-tắt 1,6 · **lập-trình 1,0**.

**Kiểm giám-khảo bằng thước đo khách-quan** (đáp-án chuẩn có trong câu trả-lời; đáp-số cuối; cụm "không biết"): khớp giám-khảo **Gemma** (1.3B: sự-kiện 2/40, không-biết 0/20, toán 0/18). **SeaLLM chấm quá dễ ở phần phân-loại** (nói 1.3B nhận không-biết 90%, toán 22%) → chỉ dùng SeaLLM làm ý-kiến thứ hai cho câu MỞ; phần có đáp-án dùng Gemma + kiểm khách-quan.

**Kết-luận — thứ-tự ưu-tiên cải-tiến rõ ràng:**
1. **Trung-thực**: 1.3B không bao giờ nói "không biết" (0/20 thực-thể bịa) → dạy nhận không-biết (SFT/DPO) là việc số 1, dễ đo.
2. **Suy-luận/toán**: 0/18, không tính toán gì → mốc cực thấp, mọi cải-tiến (dạy viết nháp, bầu chọn nhiều lần, GRPO) đều đo được ngay.
3. **Lập-trình**: 1,0/5 → kho pretrain gần như không có code.
4. Tiếng Việt tự-nhiên là điểm mạnh thật (4,29 theo Gemma — cao hơn mọi model trừ Gemma).

**Lỗi bộ đề cần sửa ở v2:** vài câu "không có thật" thực ra có thật hoặc mơ-hồ (pin nước muối, máy chiếu không dùng phim, tên người phổ-biến "Nguyễn Ngọc Diệp"); sự-kiện viwiki hơi hóc (Gemma chỉ 35%).
