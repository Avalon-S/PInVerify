#!/bin/bash
# stop on the first error
set -e

# ================= Configuration =================

# the dataset, modified in place
TRAIN_CONTENT_DIR="data/datasets/pin/hm3d/v1/train/content"

# the initial bad-episode list, input to round 1
# this directory must hold bad_episodes.json and good_episodes.json
INITIAL_RESULT_DIR="${PIN_RESULT_DIR:-./pin_result}/train/eval_train_pin_goalview"
CURRENT_BAD_EPS="$INITIAL_RESULT_DIR/bad_episodes.json"
GOOD_EPS="$INITIAL_RESULT_DIR/good_episodes.json"

# working directory for intermediates and reports
WORK_ROOT="results/train/repair_loop"
MATCH_POOL="$WORK_ROOT/match_pool.json"

# parallelism
NUM_JOBS=4
GPUS="0"  # all workers share this GPU

mkdir -p "$WORK_ROOT"

echo "========================================================"
echo "PIN-v2 Train Set Iterative Repair Script"
echo "Work Root: $WORK_ROOT"
echo "Initial Bad List: $CURRENT_BAD_EPS"
echo "========================================================"

# ================= Step 0: build the match pool, once =================
if [ ! -f "$MATCH_POOL" ]; then
    echo "[Init] Extracting match pool from good episodes..."
    python extract_match_pool.py \
        --dataset_dir "$TRAIN_CONTENT_DIR" \
        --good_episodes "$GOOD_EPS" \
        --output "$MATCH_POOL"
    echo "[Init] Match pool saved to $MATCH_POOL"
else
    echo "[Init] Match pool exists, skipping extraction."
fi

# ================= Iteration =================
MAX_ROUNDS=20  # round limit

# resume from a given round: bash run_train_repair_loop.sh 2
START_ROUND=${1:-1}

for ((round=START_ROUND; round<=MAX_ROUNDS; round++)); do
    echo ""
    echo "========================================================"
    echo "              STARTING REPAIR ROUND $round "
    echo "========================================================"
    
    # this round's output directory
    ROUND_DIR="$WORK_ROOT/round_${round}"
    mkdir -p "$ROUND_DIR"
    
    VERIFY_NAME="verify_round_${round}"
    RAW_OUTPUT="$ROUND_DIR/verify_raw"
    RESULT_SUBDIR="$RAW_OUTPUT/train/$VERIFY_NAME"
    REPORT_DIR="$ROUND_DIR/report"
    
    # ========== resume detection ==========
    # has this round's report already been written
    if [ -f "$REPORT_DIR/bad_episodes.json" ]; then
        echo "[Round $round] SKIP: Report already exists at $REPORT_DIR/bad_episodes.json"
        echo "                 Using existing report for next round."
        CURRENT_BAD_EPS="$REPORT_DIR/bad_episodes.json"
        continue
    fi
    
    # has verification finished, i.e. is there a gate_summary.json
    VERIFICATION_DONE=false
    if [ -d "$RESULT_SUBDIR" ]; then
        SUMMARY_COUNT=$(find "$RESULT_SUBDIR" -name "*_gate_summary.json" 2>/dev/null | wc -l)
        if [ "$SUMMARY_COUNT" -ge "$NUM_JOBS" ]; then
            echo "[Round $round] RESUME: Verification already done ($SUMMARY_COUNT summaries found)."
            echo "                 Skipping to analysis phase..."
            VERIFICATION_DONE=true
        fi
    fi
    
    # count the bad episodes by counting episode_id occurrences
    if [ ! -f "$CURRENT_BAD_EPS" ]; then
        echo "Error: Bad episodes file not found: $CURRENT_BAD_EPS"
        exit 1
    fi
    BAD_COUNT=$(grep -o '"episode_id":' "$CURRENT_BAD_EPS" | wc -l)
    echo "[Round $round] Target bad episodes count: $BAD_COUNT"
    
    if [ "$BAD_COUNT" -eq 0 ]; then
        echo "🎉🎉 CONGRATULATIONS! Zero bad episodes remaining!"
        echo "Dataset is fully repaired."
        break
    fi

    # ========== repair and verify, skippable on resume ==========
    if [ "$VERIFICATION_DONE" = false ]; then
        # 1. snapshot the current dataset
        #    named content_v0 for the initial state, content_v1 after round 1, and so on
        #    the snapshot is taken before this round's repair
        BACKUP_NAME="${TRAIN_CONTENT_DIR}_v$((round-1))"
        if [ ! -d "$BACKUP_NAME" ]; then
            echo "[Round $round] Backing up dataset to $BACKUP_NAME..."
            cp -r "$TRAIN_CONTENT_DIR" "$BACKUP_NAME"
        else
            echo "[Round $round] Backup $BACKUP_NAME already exists."
        fi

        # 2. repair, into a temporary dataset
        echo "[Round $round] Generating repaired dataset (using generate_repaired_dataset.py)..."
        python generate_repaired_dataset.py \
            --original_dir "$TRAIN_CONTENT_DIR" \
            --bad_episodes "$CURRENT_BAD_EPS" \
            --match_pool "$MATCH_POOL" \
            --output_dir "${TRAIN_CONTENT_DIR}_temp"

        # 3. swap it in
        echo "[Round $round] Swapping dataset content..."
        rm -rf "$TRAIN_CONTENT_DIR"
        mv "${TRAIN_CONTENT_DIR}_temp" "$TRAIN_CONTENT_DIR"
        echo "[Round $round] Content replaced."

        # 4. verify in parallel
        #    with fewer workers when fewer episodes remain than workers
        if [ "$BAD_COUNT" -lt "$NUM_JOBS" ]; then
            ACTUAL_JOBS=$BAD_COUNT
            if [ "$ACTUAL_JOBS" -lt 1 ]; then
                ACTUAL_JOBS=1
            fi
            echo "[Round $round] Notice: Only $BAD_COUNT episodes, reducing jobs to $ACTUAL_JOBS"
        else
            ACTUAL_JOBS=$NUM_JOBS
        fi
        
        echo "[Round $round] Running validation on repaired episodes ($ACTUAL_JOBS jobs)..."
        echo "             Output: $RAW_OUTPUT"
        
        pids=()
        for ((i=0; i<ACTUAL_JOBS; i++)); do
            # --filter-episodes restricts verification to the episodes repaired this round
            # stdout is discarded, stderr kept, so the terminal stays readable
            CUDA_VISIBLE_DEVICES=$GPUS python eval_goalview.py \
                --exp_name "$VERIFY_NAME" \
                --config configs/models/pin/pin_hm3d_v1.yaml \
                --num_jobs $ACTUAL_JOBS \
                --job_index $i \
                --output_root "$RAW_OUTPUT" \
                --filter-episodes "$CURRENT_BAD_EPS" \
                habitat.dataset.split="train" > /dev/null 2>&1 &
            
            pids+=($!)
            echo "  - Started Job $i (PID ${pids[-1]})"
        done

        # wait for the workers
        echo "[Round $round] Waiting for jobs to finish..."
        for pid in "${pids[@]}"; do
            wait $pid
        done
        echo "[Round $round] All validation jobs finished."
    fi  # end of VERIFICATION_DONE check

    # 5. write the report and refresh the bad list
    echo "[Round $round] Analyzing results..."
    
    # results land in RAW_OUTPUT/train/VERIFY_NAME
    RESULT_SUBDIR="$RAW_OUTPUT/train/$VERIFY_NAME"
    REPORT_DIR="$ROUND_DIR/report"
    
    python classify_abnormal_episodes.py \
        --input_dir "$RESULT_SUBDIR" \
        --output_dir "$REPORT_DIR"

    # 6. archive the jsonl for inspection
    mkdir -p "$ROUND_DIR/jsonl_logs"
    cp "$RESULT_SUBDIR"/*.jsonl "$ROUND_DIR/jsonl_logs/" 2>/dev/null || true
    echo "[Round $round] JSONL logs archived."
    
    # 7. prepare the next round's bad list
    NEW_BAD_EPS="$REPORT_DIR/bad_episodes.json"
    
    if [ ! -f "$NEW_BAD_EPS" ]; then
        echo "Error: Report generation failed, $NEW_BAD_EPS not found."
        exit 1
    fi

    NEW_BAD_COUNT=$(grep -o '"episode_id":' "$NEW_BAD_EPS" | wc -l)
    REPAIRED_COUNT=$((BAD_COUNT - NEW_BAD_COUNT))
    
    echo "--------------------------------------------------------"
    echo "Round $round Summary:"
    echo "  - Start Bad:    $BAD_COUNT"
    echo "  - End Bad:      $NEW_BAD_COUNT"
    echo "  - Successfully Repaired: $REPAIRED_COUNT"
    echo "  - Report:       $REPORT_DIR"
    echo "--------------------------------------------------------"
    
    # advance
    CURRENT_BAD_EPS="$NEW_BAD_EPS"
    
    # retry logic
    if [ "$NEW_BAD_COUNT" -eq "$BAD_COUNT" ]; then
        echo "⚠️  Stagnation: Bad episode count remained the same ($BAD_COUNT)."
        echo "   Action: Proceeding to next round. The script will try a DIFFERENT random selection"
        echo "           from the Match Pool for these remaining episodes."
    elif [ "$NEW_BAD_COUNT" -gt "$BAD_COUNT" ]; then
        echo "⚠️  Warning: Bad episode count INCREASED ($BAD_COUNT -> $NEW_BAD_COUNT). This is unexpected but we will continue."
    else
        echo "✅  Progress: Successfully repaired $REPAIRED_COUNT episodes this round."
    fi
done

echo "Repair loop finished."
