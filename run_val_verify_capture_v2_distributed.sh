#!/bin/bash

NUM_JOBS=12

for i in $(seq 0 $((NUM_JOBS - 1))); do
  echo "Launching job $i / $NUM_JOBS"
  CUDA_VISIBLE_DEVICES=0 nohup python val_verify_capture_v2_distributed.py \
    --output_root "${PIN_CAPTURE_ROOT:-./captures}/pin_capture_FT0_RGB_v2" \
    --num_jobs $NUM_JOBS \
    --job_index $i \
    --seed 43 \
    --save_vis \
    > logs_distributed/val_verify_capture_v2_distributed_job_$i.raw.log 2>&1 &
done

# pkill -f val_verify_capture_v2_distributed.py
# rm -rf "${PIN_CAPTURE_ROOT:-./captures}/pin_capture_FT0_RGB_v2"
# ./run_val_verify_capture_v2_distributed.sh

#     --save_vis \

# --output_root <capture root>/pin_capture_RGB_v2
# --output_root <capture root>/pin_capture_FT_RGB_v2
# --output_root <capture root>/pin_capture_FT2_RGB_v2