#!/usr/bin/env bash
# §TK_103 · Pilot phương án B — dựng môi-trường train LoRA Zamba2 trên box GPU (Vast H100, image NGC pytorch 25.03).
#   Chạy trên box:  cd /workspace/nn-aivie && bash scripts/setup_zamba2_box.sh
#   Data pilot (data/pilot_b/*.jsonl, vài MB) đẩy từ Mac lên trước (COPYFILE_DISABLE=1 tar … để khỏi rác ._*);
#   adapter kéo về sau khi train.
#
# 4 BẪY đã gặp 2026-09-30 (mất ~1 giờ GPU) — script này tránh sẵn:
#  1. Image NGC đặt PIP_CONSTRAINT=/etc/pip/constraint.txt → transformers 5.16.1 "ResolutionImpossible".
#     Bỏ constraint (PIP_CONSTRAINT=) an-toàn: torch/numpy của NVIDIA KHÔNG bị đụng (đã dry-run kiểm).
#  2. Không có kernel Mamba2 → transformers rơi về pure-torch: mamba2_chunk_scan tạo tensor G=C×B
#     40GB (lô 8) / 20GB (lô 4) → OOM kể cả lô 2 khi train. PHẢI có kernel.
#  3. Hub kernels (`use_kernels=True`, kernels-community/mamba-ssm) KHÔNG cứu được: cần đúng kernels 0.16.x
#     (pip cài 0.17 / sót dist-info 0.15.2; kéo sigstore→cryptography mà Debian cài không gỡ được), và
#     cài xong hàm quét vẫn rơi về pure-torch (không có bản build khớp torch 2.7.0a0 NGC). → BỎ hướng này.
#  4. mamba-ssm: cài MAMBA_SKIP_CUDA_BUILD (kernel quét là Triton, không cần biên-dịch) + vá __init__.py
#     (bản gốc import selective_scan_cuda → lỗi) + BUILD causal-conv1d 1.4.0 (đường nhanh Zamba2 gọi
#     mamba_split_conv1d_scan_combined → cần causal_conv1d_cuda). Kết-quả: lô 8 × 1536 token chạy, ~5.000 tok/s.
set -uo pipefail
echo "== [0] GPU: kiểm power-cap TRƯỚC (bẫy TK_103: host bóp 500/700W → chậm ~40%) =="
nvidia-smi --query-gpu=name,power.limit,power.default_limit,clocks.max.sm --format=csv

echo "== [1] thư-viện Python (bỏ PIP_CONSTRAINT của NGC) =="
PIP_CONSTRAINT= pip install -q "transformers==5.16.1" peft accelerate sentencepiece
python scripts/patch_zamba2_7b.py      # TK_105: Zamba2-7B nạp lỗi nếu không vá (xem script)

echo "== [2] kernel Mamba2: mamba-ssm (Triton, không biên-dịch) + causal-conv1d 1.4.0 (biên-dịch ~5 phút) =="
MAMBA_SKIP_CUDA_BUILD=TRUE PIP_CONSTRAINT= pip install -q --no-build-isolation --no-deps \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.2"
MJDIR="$(python -c 'import importlib.util as u; print(u.find_spec("mamba_ssm").submodule_search_locations[0])')"
cat > "$MJDIR/__init__.py" <<'EOF'
__version__ = "2.2.2"
# vá TK_103 §17: cài MAMBA_SKIP_CUDA_BUILD → không có selective_scan_cuda; kernel quét Mamba2 là Triton.
# Xuất thẳng tên để transformers.resolve_internal_import tìm thấy (thử mamba_ssm.<tên> trước).
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
EOF
# causal-conv1d 1.4.0 (≥1.5 đổi API C++, lệch mamba-ssm 2.2.2); build từ GIT (sdist PyPI thiếu csrc).
MAX_JOBS=32 PIP_CONSTRAINT= pip install -q --no-build-isolation --no-deps \
  "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0"
python -W ignore -c "import causal_conv1d_cuda; from mamba_ssm import mamba_chunk_scan_combined; print('✓ kernel Mamba2 + causal_conv1d OK')"

echo "== [3] tải trước Zamba2 + kiểm dữ liệu =="
HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download as s; s('Zyphra/Zamba2-2.7B-instruct')"
wc -l data/pilot_b/distill.jsonl data/pilot_b/phomt.jsonl data/pilot_b/heldout.jsonl

echo "== XONG. Train (setsid nohup để không chết khi rớt SSH), ~13 phút cho 2 epoch pilot: =="
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=scripts setsid nohup python -W ignore \\"
echo "    scripts/train_zamba2_lora.py --epochs 2 --max-len 1536 --bs 8 --grad-accum 2 --eval-every 50 \\"
echo "    --out ckpt/zamba2_vi_lora > train_lora.log 2>&1 < /dev/null &"
