#!/bin/bash
# ============================================================
# PIN capture pipeline: top-level driver
#
# Usage:
#   ./pipeline/run_pipeline.sh                          # capture only, assumes content/ is ready
#   ./pipeline/run_pipeline.sh --init                   # full run: navigation eval, init, then capture
#   ./pipeline/run_pipeline.sh --init --skip-nav-eval   # skip the navigation evaluation
#   ./pipeline/run_pipeline.sh --dry-run                # print the steps without running them
#   ./pipeline/run_pipeline.sh --start_ft <N>           # resume from capture round N
#   ./pipeline/run_pipeline.sh --no-save-vis            # capture without saving overview visualizations
#
# Phases:
#   Phase 0 (--init): navigation eval -> build content_verify -> keep good episodes
#   Phase 1: iterative repair loop
#     Round 0: capture everything -> analyze -> repair -> next round
#     Round N>0: capture only the failures, merge last round's successes, analyze, repair
#     at the round limit: discard episodes that still fail
#   Phase 2: drop episodes below threshold, then verify
#   Phase 3: write the report
# ============================================================

set -euo pipefail

# ---- read settings from config.py ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# parse the Python config
eval "$(python3 -c "
from pipeline.config import *
print(f'DATA_ROOT=\"{DATA_ROOT}\"')
print(f'CONTENT_DIR=\"{CONTENT_DIR}\"')
print(f'CONTENT_VERIFY=\"{CONTENT_VERIFY}\"')
print(f'CONTENT_BACKUP=\"{CONTENT_BACKUP}\"')
print(f'CAPTURE_ROOT=\"{CAPTURE_ROOT}\"')
print(f'RESULTS_DIR=\"{RESULTS_DIR}\"')
print(f'NUM_JOBS={NUM_JOBS}')
print(f'CAPTURE_SCRIPT=\"{CAPTURE_SCRIPT}\"')
print(f'CAPTURE_CONFIG=\"{CAPTURE_CONFIG}\"')
print(f'SEED_BASE={SEED_BASE}')
print(f'MAX_ITERATIONS={MAX_ITERATIONS}')
print(f'LOG_DIR=\"{LOG_DIR}\"')
print(f'NAV_EVAL_SCRIPT=\"{NAV_EVAL_SCRIPT}\"')
print(f'NAV_EVAL_NUM_JOBS={NAV_EVAL_NUM_JOBS}')
print(f'NAV_EVAL_EXP_NAME=\"{NAV_EVAL_EXP_NAME}\"')
")"

# ---- defaults ----
DO_INIT=false
DRY_RUN=false
RESULTS_FILE=""
START_FT=""
SAVE_VIS="--save_vis"
SKIP_NAV_EVAL=false

# ---- parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --init)           DO_INIT=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --results_file)   RESULTS_FILE="$2"; shift 2 ;;
        --start_ft)       START_FT="$2"; shift 2 ;;
        --no-save-vis)    SAVE_VIS=""; shift ;;
        --skip-nav-eval)  SKIP_NAV_EVAL=true; shift ;;
        *)                echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- helpers ----
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_or_dry() {
    if $DRY_RUN; then
        log "[DRY-RUN] $*"
    else
        "$@"
    fi
}

count_episodes() {
    local dir="$1"
    python3 -c "
import os, gzip, json
d = '$dir'
total = 0
if os.path.isdir(d):
    for f in os.listdir(d):
        if f.endswith('.json.gz'):
            with gzip.open(os.path.join(d, f), 'rt') as fp:
                data = json.load(fp)
                total += len(data.get('episodes', []))
print(total)
"
}

calc_actual_jobs() {
    local ep_count="$1"
    local max_jobs="$2"
    if [ "$ep_count" -le 0 ]; then
        echo 1
    elif [ "$ep_count" -lt "$max_jobs" ]; then
        echo "$ep_count"
    else
        echo "$max_jobs"
    fi
}

# pick verify_dir: content_verify if present, otherwise content/
resolve_verify_dir() {
    if [ -d "$CONTENT_VERIFY" ]; then
        echo "$CONTENT_VERIFY"
    else
        echo "$CONTENT_DIR"
    fi
}

# ============================================================
# Phase 0: one-time initialization, requires --init
# ============================================================
if $DO_INIT; then
    log "===== Phase 0: Initialization ====="

    # Step 0a: build content_verify
    if [ ! -d "$CONTENT_VERIFY" ] || [ -z "$(ls -A "$CONTENT_VERIFY" 2>/dev/null)" ]; then
        log "Building content_verify from $CONTENT_DIR ..."
        run_or_dry python3 -m pipeline.init_verify_dataset \
            --input_dir "$CONTENT_DIR" \
            --output_dir "$CONTENT_VERIFY"
    else
        log "content_verify already exists, skipping init_verify_dataset."
    fi

    # Step 0b: navigation evaluation
    if [ -z "$RESULTS_FILE" ] && ! $SKIP_NAV_EVAL; then
        NAV_EVAL_DIR="${RESULTS_DIR}/${NAV_EVAL_EXP_NAME}"
        NAV_MERGED_FILE="${NAV_EVAL_DIR}/all_results.jsonl"

        if [ -f "$NAV_MERGED_FILE" ]; then
            log "Step 0b: Nav eval results already exist: $NAV_MERGED_FILE"
            RESULTS_FILE="$NAV_MERGED_FILE"
        else
            log "Step 0b: Running Oracle navigation evaluation ($NAV_EVAL_NUM_JOBS workers)..."

            if ! $DRY_RUN; then
                mkdir -p "$NAV_EVAL_DIR"
                mkdir -p "$LOG_DIR"

                PIDS=()
                for (( JOB=0; JOB<NAV_EVAL_NUM_JOBS; JOB++ )); do
                    JOB_LOG="${LOG_DIR}/nav_eval_job${JOB}.log"
                    log "  Starting nav eval worker $JOB -> $JOB_LOG"
                    CUDA_VISIBLE_DEVICES=0 python3 "$NAV_EVAL_SCRIPT" \
                        --config "$CAPTURE_CONFIG" \
                        --exp_name "$NAV_EVAL_EXP_NAME" \
                        --results_dir "$RESULTS_DIR" \
                        --num_jobs "$NAV_EVAL_NUM_JOBS" \
                        --job_index "$JOB" \
                        > "$JOB_LOG" 2>&1 &
                    PIDS+=($!)
                done

                log "  All $NAV_EVAL_NUM_JOBS nav eval workers launched. PIDs: ${PIDS[*]}"
                log "  Waiting for all workers to complete..."

                FAILED_JOBS=0
                for PID in "${PIDS[@]}"; do
                    if ! wait "$PID"; then
                        FAILED_JOBS=$((FAILED_JOBS + 1))
                        log "  WARN: Nav eval worker PID $PID exited with error."
                    fi
                done

                if [ "$FAILED_JOBS" -eq "$NAV_EVAL_NUM_JOBS" ]; then
                    log "  ERROR: ALL nav eval workers failed!"
                    exit 1
                elif [ "$FAILED_JOBS" -gt 0 ]; then
                    log "  WARN: $FAILED_JOBS/$NAV_EVAL_NUM_JOBS worker(s) failed."
                else
                    log "  All nav eval workers completed successfully."
                fi

                log "  Merging nav eval results..."
                MERGE_OUTPUT=$(python3 -m pipeline.merge_nav_results \
                    --input_dir "$NAV_EVAL_DIR" \
                    --output_file "$NAV_MERGED_FILE" 2>&1)
                echo "$MERGE_OUTPUT"

                if [ ! -f "$NAV_MERGED_FILE" ]; then
                    log "  ERROR: Merged results not created: $NAV_MERGED_FILE"
                    exit 1
                fi

                if [ -f "${NAV_EVAL_DIR}/nav_eval_summary.json" ]; then
                    cp "${NAV_EVAL_DIR}/nav_eval_summary.json" "$LOG_DIR/nav_eval_summary.json"
                fi

                RESULTS_FILE="$NAV_MERGED_FILE"
            else
                log "[DRY-RUN] Would launch $NAV_EVAL_NUM_JOBS nav eval workers"
            fi
        fi
    elif [ -n "$RESULTS_FILE" ]; then
        log "Step 0b: Using provided results file: $RESULTS_FILE"
    else
        log "Step 0b: Nav eval skipped (--skip-nav-eval)."
    fi

    # Step 0c: keep the good episodes
    if [ -n "$RESULTS_FILE" ]; then
        log "Step 0c: Filtering good episodes..."
        run_or_dry python3 -m pipeline.filter_good_episodes \
            --results_file "$RESULTS_FILE" \
            --verify_dir "$CONTENT_VERIFY" \
            --output_dir "$CONTENT_DIR"
    else
        log "WARN: No results file, skipping filter. Using content/ as-is."
    fi

    log "===== Phase 0 complete ====="
fi

# ============================================================
# Phase 1: iterative repair loop
# ============================================================
log ""
log "===== Phase 1: Iterative Capture & Repair ====="

# decide which capture round to start from
if [ -n "$START_FT" ]; then
    FT_INDEX=$START_FT
else
    FT_INDEX=$(python3 -c "from pipeline.config import detect_next_ft_index; print(detect_next_ft_index())")
fi
log "Starting from FT index: $FT_INDEX"

VERIFY_DIR=$(resolve_verify_dir)
log "Verify dir: $VERIFY_DIR"

mkdir -p "$CONTENT_BACKUP"
mkdir -p "$LOG_DIR"

PREV_CAPTURE_DIR=""

for (( ROUND=0; ROUND<MAX_ITERATIONS; ROUND++ )); do
    CAPTURE_DIR=$(python3 -c "from pipeline.config import capture_dir_path; print(capture_dir_path($FT_INDEX))")
    BACKUP_DIR="${CONTENT_BACKUP}/content_v${FT_INDEX}"

    log ""
    log "============================================"
    log "  Round $ROUND / $MAX_ITERATIONS  (FT$FT_INDEX)"
    log "  Capture dir: $CAPTURE_DIR"
    log "============================================"

    # ---- Step A: back up the current content/ ----
    if [ -d "$CONTENT_DIR" ]; then
        if [ ! -d "$BACKUP_DIR" ]; then
            log "Step A: Backing up content/ -> $BACKUP_DIR"
            if ! $DRY_RUN; then
                cp -r "$CONTENT_DIR" "$BACKUP_DIR"
            else
                log "[DRY-RUN] cp -r $CONTENT_DIR $BACKUP_DIR"
            fi
        else
            log "Step A: Backup already exists, skipping."
        fi
    else
        log "ERROR: content/ does not exist: $CONTENT_DIR"
        exit 1
    fi

    # ---- Step B: launch capture ----
    EP_COUNT=$(count_episodes "$CONTENT_DIR")
    ACTUAL_JOBS=$(calc_actual_jobs "$EP_COUNT" "$NUM_JOBS")

    if [ $ROUND -eq 0 ]; then
        log "Step B: Full capture ($EP_COUNT episodes, $ACTUAL_JOBS workers)..."
    else
        log "Step B: Incremental capture ($EP_COUNT bad episodes, $ACTUAL_JOBS workers)..."
    fi

    if [ "$EP_COUNT" -le 0 ]; then
        log "  WARN: No episodes in content/. Skipping capture."
    elif ! $DRY_RUN; then
        mkdir -p "$CAPTURE_DIR"

        PIDS=()
        for (( JOB=0; JOB<ACTUAL_JOBS; JOB++ )); do
            JOB_LOG="${LOG_DIR}/capture_ft${FT_INDEX}_job${JOB}.log"
            log "  Starting job $JOB -> $JOB_LOG"
            CUDA_VISIBLE_DEVICES=0 python3 "$CAPTURE_SCRIPT" \
                --config "$CAPTURE_CONFIG" \
                --output_root "$CAPTURE_DIR" \
                --num_jobs "$ACTUAL_JOBS" \
                --job_index "$JOB" \
                --seed "$SEED_BASE" \
                $SAVE_VIS \
                > "$JOB_LOG" 2>&1 &
            PIDS+=($!)
        done

        log "  All $ACTUAL_JOBS jobs launched. PIDs: ${PIDS[*]}"
        log "  Waiting for all jobs to complete..."

        FAILED_JOBS=0
        for PID in "${PIDS[@]}"; do
            if ! wait "$PID"; then
                FAILED_JOBS=$((FAILED_JOBS + 1))
                log "  WARN: Job PID $PID exited with error."
            fi
        done

        if [ "$FAILED_JOBS" -gt 0 ]; then
            log "  WARN: $FAILED_JOBS job(s) failed. Continuing..."
        else
            log "  All $ACTUAL_JOBS jobs completed successfully."
        fi
    else
        log "[DRY-RUN] Would launch $ACTUAL_JOBS capture jobs to $CAPTURE_DIR"
    fi

    # ---- Step B2: incremental mode, merge last round's successes ----
    if [ $ROUND -gt 0 ] && [ -n "$PREV_CAPTURE_DIR" ]; then
        log "Step B2: Merging successful episodes from $PREV_CAPTURE_DIR ..."
        if ! $DRY_RUN; then
            python3 -m pipeline.merge_captures \
                --prev_dir "$PREV_CAPTURE_DIR" \
                --new_dir "$CAPTURE_DIR"
        else
            log "[DRY-RUN] Would merge $PREV_CAPTURE_DIR -> $CAPTURE_DIR"
        fi
    fi

    # ---- Step C: analyze the captures ----
    log "Step C: Analyzing captures..."

    if ! $DRY_RUN; then
        ANALYZE_OUTPUT=$(python3 -m pipeline.analyze_captures \
            --capture_dir "$CAPTURE_DIR" \
            --verify_dir "$VERIFY_DIR" 2>&1)
        echo "$ANALYZE_OUTPUT"

        RESULT_LINE=$(echo "$ANALYZE_OUTPUT" | grep "^PIPELINE_RESULT:" | tail -1)
        if [ -z "$RESULT_LINE" ]; then
            log "ERROR: Could not parse analysis result."
            exit 1
        fi
        RESULT_JSON="${RESULT_LINE#PIPELINE_RESULT:}"

        FAIL_COUNT=$(echo "$RESULT_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['fail_count'])")
        SUCCESS_COUNT=$(echo "$RESULT_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['success_count'])")
        TOTAL_COUNT=$(echo "$RESULT_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_episodes'])")

        log "  Result: $SUCCESS_COUNT/$TOTAL_COUNT success, $FAIL_COUNT failed"

        # copy this round's stats into the central log directory
        if [ -f "$CAPTURE_DIR/round_summary.json" ]; then
            cp "$CAPTURE_DIR/round_summary.json" \
               "$LOG_DIR/round_${ROUND}_ft${FT_INDEX}_summary.json"
        fi
    else
        log "[DRY-RUN] Would analyze $CAPTURE_DIR"
        FAIL_COUNT=1
    fi

    # ---- Step D: check for convergence ----
    if [ "$FAIL_COUNT" -eq 0 ]; then
        log ""
        log "ALL EPISODES PASSED! Converged at round $ROUND (FT$FT_INDEX)."
        break
    fi

    # the final round does not repair
    if [ $((ROUND + 1)) -ge "$MAX_ITERATIONS" ]; then
        log ""
        log "Reached MAX_ITERATIONS=$MAX_ITERATIONS. $FAIL_COUNT episodes still failing."
        log "  These will be discarded before final capture."
        break
    fi

    # ---- Step E: repair the failing episodes (incremental mode) ----
    log "Step E: Repairing failed episodes (incremental mode)..."

    # Round > 0: read from content_full, keeping positions fixed in earlier rounds
    # Round 0: read from verify_dir, which holds all 2,510 original episodes.
    #   Episodes filtered out in Phase 0 are repaired too: once they get a new goal_position they re-enter the pipeline.
    CONTENT_FULL="${CONTENT_DIR}_full"
    if [ $ROUND -gt 0 ] && [ -d "$CONTENT_FULL" ]; then
        REPAIR_SOURCE="$CONTENT_FULL"
    else
        REPAIR_SOURCE="$VERIFY_DIR"
    fi
    log "  Repair source: $REPAIR_SOURCE"

    if ! $DRY_RUN; then
        REPAIR_OUTPUT=$(python3 -m pipeline.repair_episodes \
            --incremental \
            --success_ids "$CAPTURE_DIR/success_object_ids_by_scene.json" \
            --good_positions "$CAPTURE_DIR/good_goal_positions.json" \
            --verify_dir "$REPAIR_SOURCE" \
            --output_dir "$CONTENT_DIR" 2>&1)
        echo "$REPAIR_OUTPUT"

        REPAIR_LINE=$(echo "$REPAIR_OUTPUT" | grep "^PIPELINE_RESULT:" | tail -1)
        if [ -n "$REPAIR_LINE" ]; then
            REPAIR_JSON="${REPAIR_LINE#PIPELINE_RESULT:}"
            MODIFIED=$(echo "$REPAIR_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['modified'])")
            SKIPPED=$(echo "$REPAIR_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['skipped_no_good_pos'])")

            log "  Modified: $MODIFIED, Skipped (no good pos): $SKIPPED"

            if [ "$MODIFIED" -eq 0 ]; then
                log ""
                log "No episodes could be repaired. Stopping."
                break
            fi
        fi
    else
        log "[DRY-RUN] Would repair episodes using $CAPTURE_DIR results"
    fi

    # remember this capture dir so the next round can merge it
    PREV_CAPTURE_DIR="$CAPTURE_DIR"
    FT_INDEX=$((FT_INDEX + 1))
done

# ============================================================
# Phase 1.5: discard episodes that still fail
# ============================================================
# restore content/ to the full set; in incremental mode it holds only the failures
CONTENT_FULL="${CONTENT_DIR}_full"
if [ -d "$CONTENT_FULL" ]; then
    log ""
    log "Restoring content/ from full backup..."
    if ! $DRY_RUN; then
        rm -rf "$CONTENT_DIR"
        mv "$CONTENT_FULL" "$CONTENT_DIR"
    else
        log "[DRY-RUN] Would restore $CONTENT_FULL -> $CONTENT_DIR"
    fi
fi

# drop whatever still failed in the last round
LAST_CAPTURE_DIR=$(python3 -c "from pipeline.config import capture_dir_path; print(capture_dir_path($FT_INDEX))")
if ! $DRY_RUN && [ -f "$LAST_CAPTURE_DIR/success_object_ids_by_scene.json" ]; then
    LAST_FAIL=$(echo "$RESULT_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('fail_count',0))" 2>/dev/null || echo "0")
    if [ "$LAST_FAIL" -gt 0 ]; then
        log ""
        log "Discarding $LAST_FAIL unfixable episodes from content/..."
        DISCARD_OUTPUT=$(python3 -m pipeline.discard_bad_episodes \
            --success_ids "$LAST_CAPTURE_DIR/success_object_ids_by_scene.json" \
            --content_dir "$CONTENT_DIR" 2>&1)
        echo "$DISCARD_OUTPUT"

        DISCARD_LINE=$(echo "$DISCARD_OUTPUT" | grep "^PIPELINE_RESULT:" | tail -1)
        if [ -n "$DISCARD_LINE" ]; then
            DISCARD_JSON="${DISCARD_LINE#PIPELINE_RESULT:}"
            REMOVED=$(echo "$DISCARD_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['removed'])")
            REMAINING=$(echo "$DISCARD_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_after'])")
            log "  Removed: $REMOVED, Remaining: $REMAINING"
        fi
    fi
fi

# ============================================================
# Phase 2: final cleanup and verification
# ============================================================
# Phase 1 already captured everything with --save_vis (RGB, mask, overview).
# The merged directory from Phase 1's last round becomes the final dataset;
# only the directories of still-failing episodes need removing.
log ""
log "===== Phase 2: Final Cleanup & Verification ====="

FINAL_CAPTURE_DIR="$LAST_CAPTURE_DIR"
log "Using Phase 1 merged capture as final: $FINAL_CAPTURE_DIR"

# clean up failed episode directories, if any
if ! $DRY_RUN && [ -f "$FINAL_CAPTURE_DIR/failed_episodes.json" ]; then
    log "Cleaning up failed episodes from final capture..."

    CLEANUP_OUTPUT=$(python3 -m pipeline.cleanup_capture_dir \
        --capture_dir "$FINAL_CAPTURE_DIR" 2>&1)
    echo "$CLEANUP_OUTPUT"

    CLEANUP_LINE=$(echo "$CLEANUP_OUTPUT" | grep "^PIPELINE_RESULT:" | tail -1)
    if [ -n "$CLEANUP_LINE" ]; then
        CLEANUP_JSON="${CLEANUP_LINE#PIPELINE_RESULT:}"
        CLEANUP_REMOVED=$(echo "$CLEANUP_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['removed'])")
        log "  Removed $CLEANUP_REMOVED failed episode directories from capture"
    fi
fi

# final analysis pass
log "Analyzing final capture..."
if ! $DRY_RUN; then
    FINAL_ANALYZE=$(python3 -m pipeline.analyze_captures \
        --capture_dir "$FINAL_CAPTURE_DIR" \
        --verify_dir "$CONTENT_DIR" 2>&1)
    echo "$FINAL_ANALYZE"

    FINAL_RESULT_LINE=$(echo "$FINAL_ANALYZE" | grep "^PIPELINE_RESULT:" | tail -1)
    if [ -n "$FINAL_RESULT_LINE" ]; then
        FINAL_JSON="${FINAL_RESULT_LINE#PIPELINE_RESULT:}"
        FINAL_SUCCESS=$(echo "$FINAL_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['success_count'])")
        FINAL_FAIL=$(echo "$FINAL_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['fail_count'])")
        FINAL_TOTAL=$(echo "$FINAL_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_episodes'])")
        log "  Final result: $FINAL_SUCCESS/$FINAL_TOTAL success, $FINAL_FAIL failed"
    fi

    if [ -f "$FINAL_CAPTURE_DIR/round_summary.json" ]; then
        cp "$FINAL_CAPTURE_DIR/round_summary.json" \
           "$LOG_DIR/final_ft${FT_INDEX}_summary.json"
    fi
else
    log "[DRY-RUN] Would cleanup and analyze $FINAL_CAPTURE_DIR"
fi

log "===== Phase 2 complete ====="

# ============================================================
# Phase 3: write the final report
# ============================================================
log ""
log "===== Phase 3: Generating Final Report ====="

TOTAL_ROUNDS=$((ROUND + 1))

if ! $DRY_RUN; then
    python3 -m pipeline.generate_report \
        --log_dir "$LOG_DIR" \
        --final_capture_dir "$FINAL_CAPTURE_DIR" \
        --final_ft_index "$FT_INDEX" \
        --total_rounds "$TOTAL_ROUNDS" \
        --content_dir "$CONTENT_DIR"
else
    log "[DRY-RUN] Would generate final report"
fi

log ""
log "===== Pipeline Complete ====="
log "  Final dataset: $FINAL_CAPTURE_DIR"
log "  Report: $LOG_DIR/pipeline_report.json"
log "============================="
