#!/bin/bash

# ==============================================================================
# Evaluation over the train split, with a lower worker count to avoid OOM
# three workers, so large scenes still fit in memory
# ==============================================================================

NUM_JOBS=3
mkdir -p logs

# split is fixed to train
EXTRA_ARGS="habitat.dataset.split=train"

# further overrides can be appended, for example:
# EXTRA_ARGS="habitat.dataset.split=train habitat.environment.max_episode_steps=500"

echo "Launching Train Evaluation with $NUM_JOBS parallel jobs..."
echo "Extra Args: $EXTRA_ARGS"

for i in $(seq 0 $((NUM_JOBS - 1))); do
    echo "Launching job $i / $NUM_JOBS"
    
    # nohup in the background, output discarded
    CUDA_VISIBLE_DEVICES=0 nohup python eval_goalview.py \
    --exp_name eval_pin_goalview \
    --config configs/models/pin/pin_hm3d_v1.yaml \
    --num_jobs $NUM_JOBS \
    --job_index $i \
    --dump_location results/train/eval_pin_goalview/job_$i \
    --save_snapshots \
    $EXTRA_ARGS \
    > /dev/null 2>&1 &
    
    # stagger the starts to avoid an IO spike
    sleep 5
done

echo "Done! Check logs in logs/job_eval_goalview_train_*.raw.log"
