#!/bin/bash

NUM_JOBS=12
CONFIG="configs/models/pin/pin_hm3d_v1.yaml"
EXP_NAME="oracle_eval_distributed_train"
SAVE_VIDEO=true

mkdir -p logs_distributed

for i in $(seq 0 $((NUM_JOBS - 1))); do
  echo "Launching job $i / $NUM_JOBS"
  CUDA_VISIBLE_DEVICES=0 nohup python distributeed_pin_eval.py \
    --config "$CONFIG" \
    --exp_name "$EXP_NAME" \
    --num_jobs $NUM_JOBS \
    --job_index $i \
    --save_video $SAVE_VIDEO \
    > logs_distributed/oracle_eval_distributed_job_$i.raw.log 2>&1 &
done

# Stop everything:
# pkill -f oracle_eval_parallel.py

# Example:
# CONFIG="configs/models/pin/pin_hm3d_v1.yaml"
# EXP_NAME="oracle_eval_spf_baseline"
# SAVE_VIDEO=false
