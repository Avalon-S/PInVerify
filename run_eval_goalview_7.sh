#!/bin/bash

NUM_JOBS=7

mkdir -p logs

for i in $(seq 0 $((NUM_JOBS - 1))); do
  echo "Launching job $i / $NUM_JOBS"
  CUDA_VISIBLE_DEVICES=0 nohup python eval_goalview.py \
    --exp_name eval_pin_goalview \
    --config configs/models/pin/pin_hm3d_v1.yaml \
    --num_jobs $NUM_JOBS \
    --job_index $i \
    --dump_location results/eval_owl_nocat/job_$i \
    --save_snapshots \
    > logs/job_eval_goalview_$i.raw.log 2>&1 &
done

# All workers share one experiment name; results land under results/pin/
# pkill -f eval_goalview.py
# rm -rf "${PIN_RESULT_DIR:-./pin_result}"
# ./run_eval_goalview_7.sh