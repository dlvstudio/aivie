# RUNBOOK — Train causal-LM trên GPU CUDA (phá trần Mac M4/MPS)

**Tại sao:** máy M4 24GB (MPS) đụng trần cứng: `d ≤ 384`, `ctx ≤ 128`, model ~33M
tham số → sinh grounded tốt nhưng **trôi-nghĩa** ở prompt mở. Đã đo & loại trừ mọi
mẹo local (anchor, kNN-ground, sâu 12 tầng, lọc data). Trôi-nghĩa là
**model-size-bound** → cần GPU rời để scale d/ctx/tầng.

Training **self-contained**: chỉ cần `torch + numpy + 1 script + file data npz`.
KHÔNG cần cả repo (repo chỉ cần cho build-data & serving, làm trên Mac).

---

## 0. Chọn GPU (thuê cloud: RunPod / Vast.ai / Lambda)

| GPU | VRAM | Config chạy được | Giá tham khảo |
|-----|------|------------------|----------------|
| RTX 4090 | 24GB | d=768, 12 tầng, ctx=256, bs=48 | ~$0.4/h |
| A100 40GB | 40GB | d=1024, 16 tầng, ctx=512, bs=64 | ~$1.3/h |
| A100 80GB | 80GB | d=1280, 24 tầng, ctx=512, bs=96 | ~$1.8/h |

Một lần train ~30k bước ở A100 với AMP: **~1-2 giờ** (rẻ hơn nhiều so 3h/run trên MPS).

---

## 1. Trên Mac — gói file cần chuyển (nhỏ, ~60-160MB)

```bash
cd ~/Documents/learn/nn-aivie
# Chỉ cần 2 thứ: script train + data đoạn đã đóng gói.
tar czf gpu_bundle.tgz \
  scripts/train_ctx_encoder_causal.py \
  checkpoints/enc_data_long.npz          # 58MB — đoạn 128-token đã pack
# (tuỳ chọn) muốn train ctx dài hơn 128 thì cần build lại data trên máy CÓ repo:
#   PYTHONPATH=src python scripts/build_enc_data_long.py --maxlen 512 --out checkpoints/enc_data_long512.npz
#   rồi thêm enc_data_long512.npz vào tar.
```

Chuyển `gpu_bundle.tgz` lên box GPU (scp / giao diện web của provider).

> **Ngữ cảnh > 128:** `enc_data_long.npz` đang pack ở maxlen=128. Muốn ctx 256/512,
> build lại `enc_data_longNNN.npz` trên Mac (cần repo + `net_ls_concept_conf.pkl`),
> vì bước đóng-gói-đoạn đọc `parents` từ checkpoint. Trainer TỰ nhận maxlen từ
> `X.shape[1]` của file data — không cần sửa code.

---

## 2. Trên box GPU — setup (1 lần)

```bash
tar xzf gpu_bundle.tgz
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy          # torch bản CUDA (mặc định trên box GPU đã có CUDA)
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. Train ở QUY MÔ LỚN (điểm mấu chốt — dùng `--amp`)

```bash
# A100 40GB — bước nhảy thật: d=1024, 16 tầng, ctx=128 (data hiện có), AMP
PYTHONPATH=. python train_ctx_encoder_causal.py \
  --data enc_data_long.npz \
  --d 1024 --layers 16 --heads 16 --bs 64 \
  --warmup 3000 --steps 40000 --eval-every 2000 --lr 3e-4 --amp \
  --clean-mask NONE \
  --out ctx_encoder_causal_gpu.pt 2>&1 | tee train_gpu.log
```

Cờ quan trọng:
- `--amp` : mixed-precision (chỉ CUDA) → nhanh 2-3× + nửa VRAM. **Luôn bật trên CUDA.**
- `--warmup 3000` : BẮT BUỘC với model sâu (không có → hội tụ tệ, đã đo).
- `--d --layers --heads --bs` : phóng to theo VRAM (bảng mục 0).
- `--data enc_data_longNNN.npz` : file quyết định ctx (maxlen tự đọc từ data).

Điều chỉnh khi thiếu VRAM (`CUDA out of memory`): hạ `--bs` trước (÷2), rồi `--d`.

Checkpoint lưu mỗi `--eval-every` bước → tắt ngang chỉ mất phần cuối.
Theo dõi: `held-out bits` giảm đều là tốt (mốc Mac: 12t/ctx128 = **5.29**;
GPU lớn nên xuống **< 4.5** = nhảy bậc chất lượng).

---

## 4. (Tuỳ chọn) Build datastore ngay trên GPU

`dung_datastore.py` cũng self-contained (torch+numpy). Thêm nó vào bundle nếu muốn:

```bash
python dung_datastore.py \
  --encoder ctx_encoder_causal_gpu.pt \
  --data enc_data_long.npz --clean NONE \
  --ra datastore_gpu.npz --sents 15000
```

---

## 5. Mang về Mac + gắn vào server

```bash
# scp checkpoint (và datastore nếu build trên GPU) về Mac:
scp box_gpu:~/ctx_encoder_causal_gpu.pt   checkpoints/
scp box_gpu:~/datastore_gpu.npz           checkpoints/net_ls_concept_conf.datastore_gpu.npz
# nếu CHƯA build datastore trên GPU, build trên Mac (nhanh, MPS ~30s):
PYTHONPATH=src python scripts/dung_datastore.py \
  --encoder checkpoints/ctx_encoder_causal_gpu.pt \
  --data checkpoints/enc_data_long.npz --clean NONE \
  --ra checkpoints/net_ls_concept_conf.datastore_gpu.npz --sents 15000
```

Trỏ server ưu tiên model GPU — sửa `viz/server/main.py`, hàm `get_knn()`, thêm
lên ĐẦU list `cands`:

```python
("GPU", "checkpoints/ctx_encoder_causal_gpu.pt",
 "checkpoints/net_ls_concept_conf.datastore_gpu.npz"),
```

Khởi động lại server (chờ ~5 phút warm — xem `bake-va-warm-server`):

```bash
PYTHONPATH=src:. .venv/bin/python -m viz.server.main
```

> **Lưu ý suy-luận trên Mac:** model lớn (d≥768) chạy SINH trên MPS sẽ chậm hơn
> (mỗi bước 1 forward). Nếu quá chậm, hạ `max_new`/`extend_min` hoặc phục vụ luôn
> từ box GPU. `ctx_encoder.py` tự đọc `cfg.d`/`cfg.maxlen` từ checkpoint → nạp
> drop-in, không cần sửa (đã có `pos.num_embeddings` cho maxlen thật).

---

## 6. Xa hơn (nếu vẫn muốn tiến)

- **Ngữ cảnh 512+**: build `enc_data_long512.npz` (Mac) → train `--data` đó. Đây là
  đòn bẩy mạnh nhất cho mạch-lạc-dài (đã chứng minh ở bước 48→128).
- **Data sạch hơn**: `scripts/build_clean_prose.py` lọc truyện tu tiên (~40% corpus)
  → prose factual. Trên GPU, model đủ lớn có thể vừa sạch vừa giữ ctx dài (Mac phải
  đánh đổi vì đoạn ngắn lại).
- **Vốn từ / tokenizer**: hiện V=32002. Không cần đổi để scale.

**Tóm tắt 1 dòng:** copy `train_ctx_encoder_causal.py` + `enc_data_long.npz` lên
A100, chạy lệnh mục 3 với `--amp`, mang `.pt` về, gắn server. Đó là đường tới
LLM-quality mà máy này không đi được.
