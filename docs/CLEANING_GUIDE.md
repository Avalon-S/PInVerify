# PIN-v2 cleaning guide

The operational side of the workflow described in
[REPAIR_REPORT.md](REPAIR_REPORT.md).

## Environment

```bash
conda activate pin
cd pin
```

Habitat-sim is needed for steps 1 and 3, so those want a machine with a GPU.
Step 2 is pure CPU.

---

## The gates

| Gate | Field | Rejected when | Meaning |
|---|---|---|---|
| G1 | `is_cross_floor` | `== True` | path height spread above 0.25 m |
| G2 | `episode_mask_visible` | `== False` | the target was never visible |
| G3 | `height_diff` | `< 0` or `> 1.6 m` | implausible target height |

---

## Step 1: collect per-episode metrics

**Option A, the launcher (7 parallel workers):**

```bash
./run_eval_goalview_7.sh
```

**Option B, by hand:**

```bash
# single process, a quick check
python eval_goalview.py \
    --config configs/models/pin/pin_hm3d_v1.yaml \
    --split val \
    --num-procs 1 \
    --max-episodes 100

# the full run
python eval_goalview.py \
    --config configs/models/pin/pin_hm3d_v1.yaml \
    --split val \
    --num-procs 8
```

For the training split use `run_eval_goalview_train.sh`, which drops to three
workers so large scenes do not run out of memory.

**Output:** JSONL files under `results/`.

---

## Step 2: classify the abnormal episodes

CPU only, no simulator needed.

```bash
python classify_abnormal_episodes.py
```

Set `input_dir` and `output_dir` at the bottom of the script first.

**Output:**

| File | Contents |
|---|---|
| `good_episodes.json` | passed all three gates |
| `bad_episodes.json` | failed at least one |
| `fail_g1_only.json` | failed G1 alone |
| `fail_g2_only.json` | failed G2 alone |
| `fail_g3_only.json` | failed G3 alone |
| `fail_g1_g2.json` | failed both G1 and G2 |
| ... | the remaining combinations |
| `statistics_report.json` | the full breakdown |

`merge_gate_summaries_notebook.py` runs the same aggregation with plots, either
as a script or pasted into a Jupyter cell.

---

## Step 3: penetration detection (optional)

```bash
python detect_penetration.py \
    --config configs/models/pin/pin_hm3d_v1.yaml \
    --split val \
    --save-photos
```

**Output:** `penetration_results/`, including photographs of each flagged object
when `--save-photos` is given.

---

## Step 4: repair

Once `bad_episodes.json` and `good_episodes.json` exist:

```bash
# validation split, one pass
./run_repair_val.sh

# training split, the full iterative loop
./run_train_repair_loop.sh          # or: ./run_train_repair_loop.sh 3 to resume at round 3
```

The loop backs up the dataset before each round (`content_v0`, `content_v1`, and
so on), repairs, swaps the dataset in, re-verifies only the episodes it touched,
and stops when nothing fails. It resumes from where it left off if interrupted.

---

## Notes

- Semantic floor detection is unavailable at Habitat runtime: HM3D's
  `semantic_annotations()` returns an empty object, so G1 falls back to the OVON
  height threshold of 0.25 m. `semantic_floor_detector.py` implements the
  semantic route for scenes where the annotations do load.
- The repair loop modifies the dataset in place, which is why every round takes a
  snapshot first.
