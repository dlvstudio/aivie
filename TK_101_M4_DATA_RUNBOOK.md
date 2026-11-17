# TK_101 — Runbook: build data pretrain trên M4 + đẩy sang GPU box

> Soạn: 2026-09-05 · Nối tiếp [[TK_100_GPU_SCALEUP_PLAN.md]] §15
> Trạng thái: **RUNBOOK vận-hành** (không phải kế-hoạch) — dùng khi build data trên máy M4 (Mac, CPU/MPS)
> thay vì (hoặc bổ-sung) build trực tiếp trên GPU box.

## 0. Vì sao cần đường này

Build trực tiếp trên GPU box (DC, mạng nhanh) tưởng sẽ nhanh hơn, nhưng đo thật trên FPT H100
container cho thấy **nút-thắt là bước đọc/giải-mã parquet streaming từ HuggingFace ở
main-process (đơn-luồng)**, không phải CPU tokenize (đã song-song 127 worker). Tốc độ tụt dần
từ ~700K tok/s (burst đầu) xuống **~195-300K tok/s** sau vài phút → ETA 18B token thực tế
**~17-25h**, không phải ~7h ước-tính ban-đầu. Nếu M4 đã có sẵn corpus tương-đương/lớn hơn (build
trước đó, không bị giới-hạn bởi băng-thông DC tại thời-điểm build), **dùng lại + upload** rẻ hơn
nhiều so với build lại từ đầu trên box.

## 1. Build (hoặc xác-nhận đã có) data trên M4

```bash
cd nn-aivie && export PYTHONPATH=$(pwd)/src
python scripts/build_pretrain_pipeline.py \
  --sources fineweb2 wiki --no-dedup \
  --limit 100000000 --max-tokens 18e9 --shard-tokens 400000000 \
  --out data/pretrain_vi
```

**Bẫy đã gặp:** `--limit` mặc-định chỉ **5000** doc/nguồn (giá-trị dành cho smoke-test) — nếu quên
set lớn, pipeline dừng ở vài chục M token dù `--max-tokens` chưa đạt. Luôn set `--limit` lớn hơn
nhiều so với số doc cần để `--max-tokens` mới là điều-kiện dừng thật.

Output: thư-mục `data/pretrain_vi/` chứa `shard_00000.npy … shard_NNNNN.npy` (uint16, mỗi shard
~`--shard-tokens` token) + `manifest.json`.

## 2. Verify trước khi upload (bắt-buộc)

```bash
python -c "import json; m=json.load(open('data/pretrain_vi/manifest.json')); print(m['V'], m['EOS'], m['dtype'], m['total_tokens'], m['num_shards'])"
```

- `V` phải khớp đúng **cùng file** `checkpoints/bpe_v1.json` đang dùng trên GPU box (vocab size
  32002 tại thời-điểm soạn). **Khác tokenizer → token id sai lệch, train ra rác** — nếu nghi-ngờ,
  so-sánh checksum: `shasum -a 256 checkpoints/bpe_v1.json` trên cả hai máy phải giống nhau.
- `dtype` phải là `uint16` (V ≤ 65535).
- `total_tokens` — đối-chiếu mục-tiêu Chinchilla (~18-26B token, xem TK_100 §15).

## 3. Upload sang GPU box

> ⚠️ **CẬP-NHẬT (Claude-Mac, 2026-09-05): rsync-qua-SSH KHÔNG dùng được trên FPT serverless.**
> Đã kiểm-chứng từ Mac: **mọi cổng của box đều timeout** — `tcp-endpoint.serverless.fptcloud.com`
> chỉ mở **443 (HTTPS)**; cổng SSH forward (vd 37457) và IP trực-tiếp (109.237.64.x) đều
> **không route** (portquiz cổng-cao từ Mac thì thông → lỗi phía FPT, không phải firewall nhà).
> → **Bỏ §3a (rsync/SSH). Dùng §3b (HuggingFace Hub qua HTTPS).**

### 3a. rsync qua SSH — ❌ KHÔNG chạy trên FPT (giữ để tham-khảo nếu đổi nhà cung-cấp có SSH thật)
```bash
rsync -avz --progress --partial -e "ssh -p <PORT>" \
  data/pretrain_vi/ root@<IP>:/workspace/data/pretrain_vi_m4/
```
Chỉ dùng được nơi TCP endpoint/SSH thực-sự route (Vast/RunPod/VM có public IP). FPT serverless: không.

### 3b. HuggingFace Hub (HTTPS) — ✅ ĐƯỜNG DÙNG ĐƯỢC
Cả Mac lẫn box đều ra được `huggingface.co:443`. Đẩy data (đã nén `pretrain_vi.tar.zst` ~24GB)
lên **dataset private**, box tải về (mạng DC nhanh).

**Trên Mac** (một lần: `huggingface-cli login` nhập token; upload resume-được):
```bash
PYTHONPATH=src .venv/bin/huggingface-cli upload donaldluong/nn-aivie-pretrain-vi \
  data/pretrain_vi.tar.zst pretrain_vi.tar.zst --repo-type dataset --private
```
(~2,3h @ 23Mbps, đứt thì chạy lại — có resume. Off-clock, không tốn GPU.)

**✅ TRẠNG-THÁI (2026-09-06): ĐÃ UPLOAD XONG** — file `pretrain_vi.tar.zst` (24.13GB, 18.8B token)
đã có trên `donaldluong/nn-aivie-pretrain-vi` (private, đã verify). Box chỉ việc tải về.

**Trên box** (dùng `hf`, không phải `huggingface-cli` đã deprecated):
```bash
hf auth login          # dán HF token read-scope của donaldluong  (hoặc: export HF_TOKEN=<token>)
hf download donaldluong/nn-aivie-pretrain-vi pretrain_vi.tar.zst --repo-type dataset --local-dir /workspace/data
tar -I zstd -xf /workspace/data/pretrain_vi.tar.zst -C /workspace/data   # ra data/pretrain_vi/
# verify: python -c "import json;m=json.load(open('/workspace/data/pretrain_vi/manifest.json'));print(m['total_tokens'],m['num_shards'],m['V'])"
#   → phải là: 18800060992 47 32002
```
Rồi `--data /workspace/data/pretrain_vi` cho train_hybrid_scale.py.

**So-sánh:** build lại trên box ~17-25h + ~$60 GPU (đo thật, network-bound) · HF Hub ~2,3h upload
(off-box, $0) + tải nhanh trên box → **HF Hub thắng rõ**.

## 4. Trên box: xác-nhận nhận đủ + trỏ training vào data

```bash
python -c "import json; m=json.load(open('/workspace/data/pretrain_vi_m4/manifest.json')); print(m)"
du -sh /workspace/data/pretrain_vi_m4
```

So khớp `total_tokens`, `num_shards`, `V`, `EOS` với giá-trị đo trên M4 (bước 2). Nếu khớp, dùng
thẳng `--data /workspace/data/pretrain_vi_m4` cho `scripts/train_hybrid_scale.py` (không cần đổi
tên hay merge vào `pretrain_vi/` trừ khi muốn gộp thêm nguồn khác).

## 5. Troubleshooting

| Triệu-chứng | Nguyên-nhân thường gặp | Xử-lý |
|:---|:---|:---|
| `ssh: connect to host … port 22: Connection timed out` | Cloud NAT port khác 22 | Xem console nhà cung-cấp lấy port public thật |
| rsync đứt giữa chừng | Mạng M4↔DC không ổn-định | Chạy lại **cùng lệnh** (có `--partial`) để resume |
| `V` khác giữa M4 và box | Hai máy dùng bản `bpe_v1.json` khác nhau | Copy đúng 1 file tokenizer dùng chung cho cả build lẫn train |
| Tổng dung-lượng sau rsync lệch so M4 | Transfer chưa xong hoặc lỗi một phần | So `du -sh` + số shard hai bên; rsync lại nếu thiếu |
