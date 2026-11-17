#!/bin/bash
# Điều-phối tuần-tự theo ngân-sách bộ-nhớ (tránh tràn 80GB).
cd /workspace/nn-aivie
w(){ while [ ! -f "$1" ]; do sleep 30; done; }
GT="python -W ignore -u curriculum/giaotiep/train_gt.py --steps 8000 --dialog-repeat 3"
w out/curriculum/rpq_d384_s0.json
echo "$(date) gt2_big" >> /workspace/orch.log
$GT --tag gt2_big --d 768 --layers 12 --heads 12 > out/curriculum/gt/gt2_big.log 2>&1 &
w out/curriculum/gt/gt2_base.json
echo "$(date) gt2_grow" >> /workspace/orch.log
$GT --tag gt2_grow --d 512 --layers 8 --heads 8 --grow 4,4,6,8,8 > out/curriculum/gt/gt2_grow.log 2>&1 &
w out/curriculum/gt/gt2_big.json; w out/curriculum/gt/gt2_grow.json; w out/curriculum/rpq12k_s0.json
echo "$(date) evals" >> /workspace/orch.log
python -W ignore -u curriculum/eval_deep.py rpq12k_s0 rpq_d384_s0 > out/curriculum/deep_rpq2.log 2>&1
for t in gt2_base gt2_big gt2_grow; do python -W ignore -u curriculum/giaotiep/eval_gt.py --tag $t > out/curriculum/gt/${t}_eval.log 2>&1; done
for t in gt_vi gt_base gt_big gt_grow; do python -W ignore -u curriculum/giaotiep/eval_gt.py --tag $t --tok data/giaotiep/v1/tok.json > out/curriculum/gt/${t}_eval.log 2>&1; done
echo "$(date) XONG" >> /workspace/orch.log
