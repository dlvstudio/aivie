#!/bin/bash
# Gom kết quả + dữ liệu từ box H100 về Mac (chạy lại được nhiều lần — rsync chỉ kéo phần mới)
cd "$(dirname "$0")/.."
R="rsync -a --partial -e 'ssh -o ConnectTimeout=30 -i ~/.ssh/<key> -p 16500'"
H=root@<IP>:/workspace
eval $R "$H/nn-aivie/data/office/raw.jsonl" data/office/
eval $R "$H/nn-aivie/data/code_tb/" data/code_tb/
eval $R "$H/nn-aivie/data/code_sft/vi_code.jsonl" data/code_sft/
eval $R --include="'*.json'" --exclude="'*.pt'" "$H/nn-aivie/ckpt/expert_code/" out/box_results/expert_code/
eval $R "$H/nn-aivie/out/" --include="'evalclean_*'" --include="'zceil*'" --include="'office_eval*'" --exclude="'*'" out/
eval $R "$H/'*.log'" "$H/'*.txt'" out/box_results/logs/ 2>/dev/null
echo "$(date) gom xong"
