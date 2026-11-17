#!/bin/bash
# TK_108 C2b phần A (CPU — chạy song song khi GPU đang học C4 lần 2): tải 5 nguồn code đa ngôn ngữ (giấy phép cho thương mại;
# Magicoder dùng theo phép user đã xin) → lọc top 10 ngôn ngữ → kiểm cú pháp tree-sitter / sqlglot + SQL gretel chạy thật
# SQLite → khử trùng + bỏ tên HumanEval → data/c2b/<lang>.jsonl → HF. Trạng thái → /workspace/p3_status.log (tiền tố "C2b").
# Dùng: (setsid nohup bash scripts/box_run_c2b.sh > /workspace/c2b_run.log 2>&1 &)
S=/workspace/p3_status.log
log() { echo "$(date '+%F %T') $*" >> $S; }
cd /workspace/nn-aivie
export PYTHONPATH=scripts:src HF_HUB_DISABLE_XET=1
EV=/workspace/evalenv/bin/python; R=/workspace/c2b_raw
PIP_CONSTRAINT= $EV -m pip -q install tree-sitter-language-pack sqlglot pyarrow > /workspace/c2b_pip.log 2>&1 \
  || { log "LỖI C2b: cài tree-sitter/sqlglot"; exit 1; }
log "C2b A bắt đầu: tải Ling-Coder-SFT · OpenCoder stage1 · Magicoder · CommitPackFT · gretel text-to-SQL"
mkdir -p $R data/c2b
dl() { hf download "$@" --repo-type dataset > /dev/null 2>> /workspace/c2b_dl.log || log "LỖI C2b: tải $1"; }
dl gretelai/synthetic_text_to_sql --local-dir $R/gretel --include "*.parquet" &
dl ise-uiuc/Magicoder-OSS-Instruct-75K --local-dir $R/magic &
# hf download: mỗi mẫu cần một --include riêng (nhiều mẫu sau một --include bị hiểu là tên file)
dl bigcode/commitpackft --local-dir $R/commit $(for l in javascript typescript html css python sql java "c#" php go shell; do echo --include "data/$l/*"; done) &
dl OpenCoder-LLM/opc-sft-stage1 --local-dir $R/opc --include "data/largescale_diverse_instruct-*" --include "data/realuser_instruct-*" &
dl inclusionAI/Ling-Coder-SFT --local-dir $R/ling &
wait
log "C2b A tải xong · $(du -sh $R | cut -f1)"
$EV - <<'PY' > /workspace/c2b_he.log 2>&1
import re
from datasets import load_dataset
names = set()
for r in load_dataset("openai/openai_humaneval", split="test"):
    names.add(r["entry_point"])
open("data/c2b/humaneval_names.txt", "w").write("\n".join(sorted(names)))
print(len(names))
PY
$EV -u scripts/c2b_multilang.py extract --raw $R --per-lang 40000 --workers 24 > /workspace/c2b_extract.log 2>&1 \
  || { log "LỖI C2b: lọc (c2b_extract.log)"; exit 1; }
log "$(grep '###' /workspace/c2b_extract.log | sed 's/### //')"
for f in data/c2b/*.jsonl data/c2b/stats.json; do hf upload donaldluong/nn-aivie-2p6b-p3 $f $f > /dev/null 2>&1; done
log "C2b A xong — đã đẩy HF nn-aivie-2p6b-p3/data/c2b/ (phần B cần GPU: thầy viết test chạy thật + T-SQL trên SQL Server Express + dịch tiếng Việt)"
