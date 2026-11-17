# TK_102 — GPU box: 2 bug thật lộ ra khi smoke-test + benchmark throughput thật

> Soạn: 2026-09-06 (Claude trên GPU box, sau khi nhận data 18.8B token từ M4 qua HF Hub)
> Nối tiếp [[TK_100_GPU_SCALEUP_PLAN.md]] §15, [[TK_101_M4_DATA_RUNBOOK.md]]
> Mục đích: báo cho M4 phân-tích thêm — 2 bug này ảnh-hưởng trực-tiếp tới độ tin-cậy của
> `train_hybrid_scale.py`, không phải vấn-đề môi-trường.

## 1. Bug: `causal-conv1d` version không khớp API `mamba_ssm 2.2.2`

**Triệu-chứng:** `from mamba_ssm import Mamba` import OK, nhưng **forward pass thật** (smoke-test
10 step trên data thật) crash: `TypeError: causal_conv1d_fwd(): incompatible function arguments`.

**Nguyên-nhân:** `causal-conv1d>=1.5.0` (kể cả `1.7.0` — bản đang cài, wheel cũ ở
`/workspace/wheels`) đổi signature C++ của `causal_conv1d_fwd`: nhận thêm tensor `out` truyền
sẵn, hàm trả `None` (ghi in-place). `mamba_ssm 2.2.2` (`selective_scan_interface.py`) viết cho
API **cũ hơn**: gọi 6-7 tham-số, hàm **trả về Tensor**. Hai bên lệch → lỗi ngay lần forward đầu.

**Fix đã áp-dụng:**
- Gỡ `causal_conv1d-1.7.0` wheel cũ, build lại **`causal-conv1d==1.4.0`** — nhưng PyPI sdist bản
  này **thiếu file `csrc/*.cpp`** (`ninja: error: ... missing and no known rule to make it`) →
  phải build từ **GitHub gốc** `git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0`.
- Wheel mới lưu lại `/workspace/wheels/causal_conv1d-1.4.0-cp312-cp312-linux_x86_64.whl`
  (`reinit.sh` không cần sửa — glob `*.whl` tự nhặt đúng file).
- `scripts/setup_container.sh` đã sửa để lần dựng box MỚI không dính lại: ghim đúng
  `causal-conv1d==1.4.0` từ git source + `mamba-ssm==2.2.2` (trước đó không ghim version) + thêm
  luôn sed-patch `MambaLMHeadModel` (trước đây chỉ có trong `reinit.sh`, script dựng box gốc thiếu).

**Câu hỏi cho M4:** nếu sau này muốn tận-dụng optimizations mới của causal-conv1d ≥1.5 (vd.
`initial_states`/`return_final_states` cho continuous-batching inference), sẽ cần *nâng cấp*
`mamba_ssm` lên bản hỗ-trợ API mới tương-ứng — không nên tự sửa lẻ-tẻ `selective_scan_interface.py`
vì còn backward pass (`causal_conv1d_bwd`) cũng đổi signature tương-tự.

## 2. Bug: Python late-binding closure trong `--grad-ckpt` (train_hybrid_scale.py)

**Triệu-chứng:** Bật `--grad-ckpt` (dù có/không `--compile`) → crash ngay bước đầu:
`torch.utils.checkpoint.CheckpointError: Recomputed values ... have different metadata`, shape
hoàn-toàn khác nhau giữa saved/recomputed (vd. saved `[2048,16384]` vs recomputed `[2048,2048]`).

**Nguyên-nhân (đã xác-định, không phải lỗi kernel Mamba):**

```python
# HybridLM.forward(), trước khi sửa:
for blk, nm in zip(self.blocks, self.norms):
    if self.grad_ckpt and self.training:
        x = x + torch.utils.checkpoint.checkpoint(lambda t: blk(nm(t)), x, use_reentrant=False)
```

`use_reentrant=False` **hoãn recompute tới lúc `backward()`** (không giống reentrant-mode tính lại
ngay trong forward). Nhưng `lambda t: blk(nm(t))` đóng biến `blk`/`nm` **theo tham-chiếu** — vòng
lặp Python không tạo scope riêng mỗi iteration, nên tới lúc backward chạy (SAU KHI cả 48 layer đã
forward xong), `blk`/`nm` trong closure đã trỏ vào **layer cuối cùng** của vòng lặp. Mọi lần
recompute (bất kể checkpoint đó thuộc layer nào) đều gọi nhầm layer cuối → shape lệch, gradient sai
hoàn-toàn (nếu code không có safety-check `check_recomputed_tensors_match` thì sẽ **âm-thầm train
sai**, không crash — nguy-hiểm hơn cả).

**Fix:** default-argument binding để mỗi lambda giữ đúng tham-chiếu tại thời-điểm tạo:
```python
x = x + torch.utils.checkpoint.checkpoint(
    lambda t, blk=blk, nm=nm: blk(nm(t)), x, use_reentrant=False)
```
Đã verify: bits giảm đúng quỹ-đạo giống baseline không grad-ckpt (15.7→13.3 sau 10 step).

## 3. Benchmark throughput thật (H100, model 1.277B, ctx=2048, data thật 18.8B token)

| Config | tok/s (steady-state) | MFU | VRAM peak | Ghi chú |
|---|---|---|---|---|
| baseline, bs=8×ga8 | ~24-26K | ~19-21% | 64.4GB/80GB (79%) | nhanh nhất |
| `--grad-ckpt`, bs=8 | ~16.4K | — | thấp hơn (chưa đo lại sau fix) | chậm hơn ~35%, không cần (dư VRAM) |
| `--compile`, bs=8 | ~21.4K | — | — | **chậm hơn baseline** — kernel CUDA Mamba/causal-conv1d không trace được, torch.compile gãy graph liên-tục quanh nó (`Graph break due to unsupported builtin causal_conv1d_cuda...`), overhead tracing > lợi-ích fuse |
| cả hai, bs=8 | ~17.5K | — | — | |
| baseline, bs=10×ga8 | ~25-26.5K | ~19-21% | 74.7GB/80GB (92%) | nhỉnh hơn bs=8 chút, margin VRAM hẹp hơn cho chạy dài không giám-sát |
| baseline, bs=12×ga8 | — | — | **OOM** (cần 79GB+, tried to allocate thêm 2.93GiB lúc 77GB đã dùng) | vượt trần |

**Kết-luận:** `--grad-ckpt`/`--compile` **không cần thiết** ở scale 1.3B/ctx=2048/bs=8 — VRAM dư
(64GB/80GB), và cả hai cờ chỉ làm chậm do đặc-thù kernel CUDA custom (Mamba) không tương-thích tốt
với `torch.compile`, còn `--grad-ckpt` đánh-đổi tốc-độ lấy VRAM vốn không thiếu. **Khuyến-nghị dùng
baseline (không cờ), bs=8** — an-toàn VRAM hơn bs=10 cho chạy nhiều ngày không giám-sát, chênh
tok/s không đáng kể (~24-26K vs ~25-26.5K).

## 4. Thực-tế ETA/chi-phí lệch so kế-hoạch

Plan §15 giả-định ~1 tuần/~$500 cho 26B token. Đo thật trên H100 (baseline, không cờ):

- **18.8B token ÷ ~25K tok/s ≈ 209 giờ ≈ 8.7 ngày**
- **8.7 ngày × ~78.592đ/giờ (~$3.02) ≈ ~$630** — cao hơn ~25% so ngân-sách gốc.

MFU ~19-21% là mức hợp-lý cho kiến-trúc hybrid Mamba+Attention (attn chỉ 1/8 layer, batch chưa
lớn) — không phải cấu-hình sai, chỉ là giả-định ban-đầu trong TK_100 §15 lạc-quan hơn thực-tế đo
được. Đang chờ quyết-định: chạy đủ 18.8B token (chấp-nhận ~$630/8.7 ngày), hay điều-chỉnh mục-tiêu.

## 5. File đã sửa (commit kèm theo)

- `scripts/train_hybrid_scale.py` — fix closure bug (mục 2)
- `scripts/setup_container.sh` — ghim đúng version causal-conv1d/mamba-ssm + sed-patch (mục 1)
- Wheel mới: `/workspace/wheels/causal_conv1d-1.4.0-*.whl` (không vào git, theo đúng nguyên-tắc
  KHÔNG push binary — chỉ tồn-tại trên `/workspace` của box)
