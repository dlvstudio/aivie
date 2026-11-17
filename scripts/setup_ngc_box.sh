#!/bin/bash
# Box Vast image NGC pytorch 25.03 (python hệ thống): cài kernel mamba-ssm 2.2.2 + causal-conv1d 1.4.0 (build từ git) + hf CLI.
# Dùng cho P2 (train nền) và P3 (SFT/expert). ~10–20 phút. Xong in SETUP_XONG.
#   bash scripts/setup_ngc_box.sh > /workspace/setup_p3.log 2>&1
cd /workspace/nn-aivie
export PIP_CONSTRAINT= MAX_JOBS=64                     # NGC ghim torch cho mọi pip → bỏ ràng buộc
set -x
apt-get install -y -qq zstd >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq zstd)
pip install -q -U "huggingface_hub[cli]" pyflakes || exit 1
pip install -q --no-build-isolation --no-deps "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0" || exit 1
pip install -q --no-build-isolation --no-deps "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.2" || exit 1
MJDIR="$(python -c "import importlib.util as u; print(u.find_spec(\"mamba_ssm\").submodule_search_locations[0])")"
sed -i "/mixer_seq_simple import MambaLMHeadModel/d" "$MJDIR/__init__.py"      # tránh kéo transformers
python -W ignore -c "
import torch, causal_conv1d_cuda, selective_scan_cuda
from mamba_ssm import Mamba
m=Mamba(d_model=256).cuda().bfloat16(); x=torch.randn(2,128,256,device=\"cuda\",dtype=torch.bfloat16); y=m(x); y.sum().backward()
print(\"KERNEL OK\", tuple(y.shape))" || exit 1
PYTHONPATH=scripts:src python -c "import train_hybrid_scale as t; print(\"HAVE_KERNEL\", t.HAVE_KERNEL)"
which hf; hf version 2>/dev/null
echo SETUP_XONG
