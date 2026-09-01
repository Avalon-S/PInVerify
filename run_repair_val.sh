#!/bin/bash

# ==============================================================================
# End-to-end repair run over the validation set
# ==============================================================================

# 1. build the match pool
echo "Step 1: Extracting Match Pool..."
python extract_match_pool.py \
    --dataset_dir "data/datasets/pin/hm3d/v1/val/content" \
    --good_episodes "results/val/eval_pin_goalview/good_episodes.json" \
    --output "results/val/repair/match_pool.json"

# 2. write the repaired dataset to a temporary directory first
TEMP_REPAIRED_DIR="data/datasets/pin/hm3d/v1/val/content_repaired_temp"
ORIGINAL_DIR="data/datasets/pin/hm3d/v1/val/content"
BACKUP_DIR="data/datasets/pin/hm3d/v1/val/content_backup"

echo "Step 2: Generating Repaired Dataset to temp dir..."
python generate_repaired_dataset.py \
    --original_dir "$ORIGINAL_DIR" \
    --bad_episodes "results/val/eval_pin_goalview/bad_episodes.json" \
    --match_pool "results/val/repair/match_pool.json" \
    --output_dir "$TEMP_REPAIRED_DIR"

# 2.5 Swap Datasets
echo "Step 2.5: Swapping Datasets..."
if [ -d "$BACKUP_DIR" ]; then
    echo "Backup dir $BACKUP_DIR already exists. Assuming original data is already backed up."
    # An existing backup means this ran before; do not overwrite the pristine one.
    # content/ is whatever is being replaced: last round's repair, or a restored original.
    # The original backup is never touched, so repeated runs stay safe.
    
    
    
    
    
    if [ -d "$ORIGINAL_DIR" ]; then
        mv "$ORIGINAL_DIR" "${ORIGINAL_DIR}_old_$(date +%s)"
    fi
else
    # first run: rename content to content_backup
    echo "Backing up original content to $BACKUP_DIR"
    mv "$ORIGINAL_DIR" "$BACKUP_DIR"
fi

# move the repaired directory into place
echo "Moving repaired data to $ORIGINAL_DIR"
mv "$TEMP_REPAIRED_DIR" "$ORIGINAL_DIR"


# 3. verify, re-running only the repaired episodes
echo "Step 3: Verifying Repairs..."
# The dataset now sits at the standard path, so no data_path override is needed.


# Passing bad_episodes.json to --filter-episodes restricts the run
# to the episodes that previously failed.

NUM_JOBS=4
mkdir -p logs

for i in $(seq 0 $((NUM_JOBS - 1))); do
    echo "Launching verification job $i / $NUM_JOBS"
    
    CUDA_VISIBLE_DEVICES=0 nohup python eval_goalview.py \
    --exp_name verify_repair_val \
    --config configs/models/pin/pin_hm3d_v1.yaml \
    --num_jobs $NUM_JOBS \
    --job_index $i \
    --dump_location results/val/repair/verification/job_$i \
    --save_snapshots \
    --filter-episodes "results/val/eval_pin_goalview/bad_episodes.json" \
    habitat.dataset.split="val" \
    > logs/verify_repair_val_$i.log 2>&1 &
done

echo "Verification launched! Check logs/verify_repair_val_*.log"
