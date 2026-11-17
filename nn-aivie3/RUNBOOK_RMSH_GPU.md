# RUNBOOK — nn-aivie3 scale trên GPU CUDA (mamba-ssm kernel)

> Nối hạ-tầng TK_100/TK_101. Mục-tiêu: biến các cơ-chế đã-chứng-minh (M0→M5, thuần-PyTorch
> trên M4) thành model **dùng-được** bằng **kernel `mamba-ssm` trên CUDA** (nhanh ~10-50×
> vòng-lặp scan MPS) + **data thật**. Máy Mac chỉ smoke fallback; train thật ở box GPU.

## 0. Vì sao phải lên CUDA
Nghẽn đo thật ở M3/M3.5: **selective-scan Mamba thuần-PyTorch trên MPS chậm** khi
token/frame cao (36 token/frame → train >10 phút). Kernel CUDA (`selective_scan_cuda` +
`causal_conv1d`) xoá nghẽn này. Code đã sẵn đường kernel:
`src/nn_aivie3/model.py:make_mamba()` + `HybridBackbone(use_kernel=True)` — **tự bật khi
`torch.cuda.is_available()`**, tự fallback thuần-PyTorch ở nơi khác.

## 1. Dựng box (1 lệnh, tái-dùng TK_100)
Chọn template **NGC PyTorch** (`nvcr.io/nvidia/pytorch:24.xx-py3`) để có `nvcc` (bắt-buộc
build kernel). Rồi:
```bash
git clone <repo> && cd nn-aivie
bash scripts/setup_container.sh          # cài mamba-ssm 2.2.2 + causal-conv1d 1.4.0 (đã ghim)
python -c "import mamba_ssm; print('kernel OK')"
```
> ⚠️ Bẫy đã biết (setup_container.sh §1): `causal-conv1d>=1.5` đổi API C++ → hỏng forward
> mamba-ssm 2.2.2. **Phải** ghim `causal-conv1d==1.4.0 build từ GitHub gốc** (PyPI thiếu
> csrc/*.cpp). Script đã làm sẵn.

## 2. Data thật (nút-thắt — build trên M4 rồi upload)
Định-dạng shard: thư-mục `shard_*.npz`, mỗi shard:
- `img`: uint8 `[N, 3, H, W]` (H=W=`--img`, vd 224)
- `cap`: int16 `[N, L]` caption token (`--cap-len`, tokenize bằng **BPE của ta**
  `src/nn_aivie/core/vocab.py` — CÙNG vocab với LM để weight-tie)

**Script build sẵn:** `nn-aivie3/scripts/build_rmsh_data.py` → shard đúng định-dạng, tokenize
caption bằng **BPE của ta** (`checkpoints/bpe_v1.json`, vocab 32002 — đã kiểm tại-chỗ M4:
build synthetic → train_rmsh_scale đọc được). Nguồn: `synthetic` (kiểm pipeline), `local:<dir>`
(captions.jsonl đã-dịch-vi + ảnh), `hf:<dataset>` (cần `datasets`+`Pillow`).
```bash
PYTHONPATH=src python nn-aivie3/scripts/build_rmsh_data.py --source local:data/flickr_vi \
    --img 224 --cap-len 32 --out /workspace/data/rmsh_vi
```
Nguồn ảnh-văn tiếng Việt (TK_100 §14 tinh-thần): COCO/LLaVA-Instruct **dịch→vi** (Ollama SeaLLM
[[rag-seallm-huong-di]] / PhoMT / MTet — dịch TRƯỚC, đưa vào `local:`), caption wiki-commons,
screen-record tự-nhãn bằng teacher-VLM. **Build trên M4** → **upload qua HF Hub / bundle**
(`gpu_bundle.tgz`): TK_101 đã đo **FPT serverless CHẶN ssh/rsync** (mọi cổng timeout) → KHÔNG rsync thẳng.

## 3. Smoke ĐO tok/s TRƯỚC khi đốt GPU (bắt-buộc)
```bash
PYTHONPATH=src python nn-aivie3/scripts/train_rmsh_scale.py --data /workspace/data/rmsh_vi \
    --d 1024 --layers 24 --attn-every 6 --img 224 --patch 16 --nq 32 \
    --bs 32 --grad-accum 8 --cap-len 32 --steps 10 --grad-ckpt --save-every 0
```
Kỳ-vọng log: `Mamba=CUDA-kernel(mamba-ssm)` (KHÔNG được thấy "kernel THIẾU!"). Ghi tok/s.

## 4. Train thật
```bash
PYTHONPATH=src python nn-aivie3/scripts/train_rmsh_scale.py --data /workspace/data/rmsh_vi \
    --d 1024 --layers 24 --attn-every 6 --img 224 --patch 16 --nq 32 \
    --bs 32 --grad-accum 8 --lr 3e-4 --tokens 8e9 --grad-ckpt --compile \
    --out /workspace/checkpoints/rmsh_mini.pt
```
- **Cỡ:** `mini` d1024×L24 (~vài trăm M) hợp suy-luận M4 int4 (TK3_000 §6). Tier `nano`
  (d512×L16) cho robot/Jetson; `plus` (d2048×L32) khi cần đẹp.
- **3 giai-đoạn (TK3_000 §5):** S1 căn ảnh-văn (đông backbone, train VFE+proj) → S2 streaming
  video/temporal → S3 hội-thoại SFT. Script này là khung S1/S3 (ảnh+caption); video = mở
  `cap`→chuỗi-frame sau.
- Resume: `--init-from /workspace/checkpoints/rmsh_mini.pt`. Hết phiên: **STOP container**
  (ngừng phí GPU), `/workspace` giữ nguyên → start lại + `--init-from`.

## 5. Serve realtime (sau train) — về edge/M4
- Train (scan) trên CUDA dùng kernel; **serve streaming** (perceive O(1)/decide) dùng
  `HybridBackbone(use_kernel=False)` → `MambaStep` thuần-PyTorch (đúng, đủ nhanh khi phục-vụ
  một luồng). Nạp weight kernel→pure-PyTorch: cùng `state_dict` (kernel `mamba_ssm.Mamba` và
  `MambaBlock` KHÁC layout → cần map, hoặc train bản pure cho serve; **TODO nối
  `mamba_ssm.Mamba.step()` cho serve CUDA**).
- Distill + quantize int4 (spec-decode TK_058) cho M4/robot — TK3_000 §6.

## 6. Trạng-thái code
- ✅ `train_rmsh_scale.py` chạy (smoke MPS fallback: RMSH 6.2M, grad-ckpt, tok/s log).
- ✅ backbone kernel-aware (`use_kernel` tự bật trên CUDA), grad-ckpt.
- 🔴 chưa chạy CUDA thật (máy Mac); cần box + data ảnh-caption. `mamba_ssm.step()` cho serve = TODO.
