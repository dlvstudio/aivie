#!/usr/bin/env bash
# §TK_100 · M1 — DỰNG MÔI-TRƯỜNG train trên GPU Container FPT (1×H100) trong 1 lệnh.
#   Cài: deps train (mamba-ssm kernel + datasets…) · Claude Code (Node) · thư-mục bền.
#   Chạy trên box:  git clone <repo> && cd nn-aivie && bash scripts/setup_container.sh
#   Ý-tưởng: mọi thứ CẦN-GIỮ đặt dưới /workspace (persistent disk) → sống qua stop/restart.
set -euo pipefail
WS="${WORKSPACE:-/workspace}"                      # persistent disk mount
echo "== [setup] workspace bền = $WS =="
mkdir -p "$WS/data" "$WS/checkpoints" "$WS/.claude" "$WS/.hf"

# ── 0. Kiểm CUDA/torch (điều-kiện để build mamba-ssm kernel) ───────────────────────
echo "== [0] kiểm CUDA/torch =="
command -v nvcc >/dev/null && nvcc --version | tail -1 || {
  echo "⚠️  KHÔNG thấy nvcc (CUDA-devel). mamba-ssm sẽ KHÔNG build được."
  echo "    → đổi template sang NGC PyTorch (nvcr.io/nvidia/pytorch:24.xx-py3) rồi chạy lại."; }
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda)" || \
  echo "⚠️  torch chưa có — cài trong image PyTorch/NGC."

# ── 1. Deps train ─────────────────────────────────────────────────────────────────
echo "== [1] cài deps Python (đường CUDA nhanh cho Mamba) =="
pip install -q --upgrade pip
pip install -q datasets numpy huggingface_hub
apt-get install -y -qq zstd >/dev/null 2>&1 || echo "⚠️ zstd cài lỗi (cần để giải-nén data upload từ HF Hub, xem TK_101)"
# mamba-ssm + causal-conv1d biên-dịch kernel CUDA (cần nvcc); --no-build-isolation nếu vấp
# LƯU Ý (đo thật 2026-09-06): causal-conv1d>=1.5.0 (kể cả 1.7.0 mới nhất) đổi API C++
# causal_conv1d_fwd (nhận thêm tensor `out`, trả None) — KHÔNG khớp mamba_ssm 2.2.2 (kỳ-vọng
# hàm trả về Tensor, kiểu API cũ) → forward pass lỗi TypeError dù `import mamba_ssm` vẫn OK.
# Ghim đúng 1.4.0 build từ GitHub gốc (PyPI sdist bản này THIẾU csrc/*.cpp, build local sẽ lỗi
# "missing and no known rule to make it" — phải lấy từ git, không phải PyPI).
pip install -q --no-build-isolation --no-deps \
  "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0" || \
  echo "⚠️ causal-conv1d lỗi — thử lại: PIP_CONSTRAINT= pip install --no-build-isolation --no-deps 'causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0'"
# mamba-ssm: PHẢI build từ GIT, KHÔNG từ PyPI. sdist PyPI 2.2.2 THIẾU csrc/selective_scan/*.cpp
# → ninja lỗi "missing and no known rule to make it" (đo thật Vast H100 2026-09-09). Repo git có đủ csrc.
pip install -q --no-build-isolation --no-deps \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.2" || \
  echo "⚠️ mamba-ssm build lỗi — xem log; nếu OOM lúc compile thử: MAX_JOBS=8 pip install ... (giảm luồng ninja)"
# transformers 5.x bỏ GreedySearchDecoderOnlyOutput → dòng 'import MambaLMHeadModel' trong __init__.py gốc
# kéo transformers.generation làm CRASH import (ta chỉ cần khối Mamba lõi). Gỡ dòng đó.
# Tìm __init__ bằng find_spec (KHÔNG import — vì import đang lỗi TRƯỚC khi vá).
MJDIR="$(python -c 'import importlib.util as u,sys; s=u.find_spec("mamba_ssm"); print(s.submodule_search_locations[0]) if s and s.submodule_search_locations else sys.exit(1)' 2>/dev/null)"
[ -n "$MJDIR" ] && sed -i '/mixer_seq_simple import MambaLMHeadModel/d' "$MJDIR/__init__.py" 2>/dev/null && \
  echo "✓ đã gỡ dòng import MambaLMHeadModel khỏi $MJDIR/__init__.py" || true
python -c "import mamba_ssm; print('✓ mamba-ssm OK (đường kernel bật)')" 2>/dev/null || \
  echo "⚠️ mamba-ssm chưa import được → trainer sẽ rơi về fallback thuần-PyTorch (CHẬM, đừng train dài)."

# ── 2. Cache HF + config bền trên /workspace (khỏi login/tải lại sau restart) ──────
echo "== [2] trỏ cache HF + Claude config sang persistent disk =="
{
  echo "export HF_HOME=$WS/.hf"
  echo "export HF_DATASETS_CACHE=$WS/.hf/datasets"
  echo "export CLAUDE_CONFIG_DIR=$WS/.claude"
  echo "[ -f $WS/.hf/token ] && export HF_TOKEN=\$(cat $WS/.hf/token)"
} >> ~/.bashrc
export HF_HOME="$WS/.hf" CLAUDE_CONFIG_DIR="$WS/.claude"
if [ -f "$WS/.hf/token" ]; then
  export HF_TOKEN="$(cat "$WS/.hf/token")"
  echo "✓ HF_TOKEN nạp từ $WS/.hf/token (WRITE — backup ckpt). chmod 600 nếu chưa."
else
  echo "⚠️ CHƯA có $WS/.hf/token — train dài PHẢI có token WRITE (bài-học TK_103:"
  echo "    Destroy box = mất ckpt). Tạo token WRITE trên huggingface.co/settings/tokens,"
  echo "    scp vào $WS/.hf/token rồi chmod 600. ĐỪNG dán token vào chat/git/log."
fi

# ── 3. Claude Code (Node 20 + CLI) ────────────────────────────────────────────────
echo "== [3] cài Node + Claude Code =="
if ! command -v node >/dev/null; then
  if command -v apt-get >/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs
  else
    echo "⚠️ không có apt — cài Node thủ-công (nvm) rồi chạy lại phần này."
  fi
fi
command -v node >/dev/null && npm install -g @anthropic-ai/claude-code && \
  echo "✓ Claude Code cài xong — chạy 'claude' để đăng-nhập (thuê-bao: mở URL trên máy anh; hoặc export ANTHROPIC_API_KEY)."

echo
echo "== [XONG] Bước kế: =="
echo "  • Upload data đã build (M4) vào  $WS/data/pretrain_vi/"
echo "  • Smoke 5-10 step ĐO tok/s thật TRƯỚC khi train dài:"
echo "      PYTHONPATH=src python scripts/train_hybrid_scale.py --data $WS/data/pretrain_vi \\"
echo "         --ctx 2048 --d 2048 --layers 48 --attn-every 8 --steps 10 --eval-every 5 \\"
echo "         --out $WS/checkpoints/hybrid_1p3b.pt"
echo "  • Train thật: bỏ --steps, thêm --tokens 18e9 (KHÔNG --grad-ckpt/--compile — TK_102 đo CHẬM hơn"
echo "      trên H100; baseline bs=8 grad-accum=8 nhanh nhất). Resume: --init-from <ckpt>."
echo "  • Hết phiên: STOP instance (ngừng phí GPU); ổ giữ nguyên → start lại + --init-from."
echo "    ⚠️ Vast: DESTROY xoá sạch ổ. Train dài BẮT-BUỘC:"
echo "        --hf-repo donaldluong/nn-aivie-hybrid-1p3b-ckpt --hf-every 5000"
