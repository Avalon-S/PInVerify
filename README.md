# PInVerify: PIN benchmark repairs

Defects found in the upstream [PIN](https://github.com/aimagelab/pin) benchmark
while building the PInVerify dataset, and the code that detects and repairs them.

> **Not part of the PInVerify paper.** The paper builds on PInNED as published
> and does not report these repairs, so there is no corresponding section to look
> for. This branch exists because the findings are useful to anyone else working
> with PIN.

The benchmark itself lives on [`main`](https://github.com/Avalon-S/PInVerify);
the capture pipeline lives on `data-collection`.

> **Environment.** Same habitat stack as `data-collection`, and incompatible with
> `main`: Python 3.9, torch 1.13.1+cu117, habitat and habitat-sim 0.2.3,
> numpy 1.23, transformers 4.38.2. One conda environment per branch.

## What is wrong with the upstream episodes

Running an oracle agent over every PIN episode surfaced three failure modes that
have nothing to do with the navigation policy:

| Gate | Failure | How it is detected |
|---|---|---|
| **G1** | The episode requires crossing floors or climbing stairs, though PIN is defined as single-floor navigation | Height spread above 0.25 m, measured both on the planned shortest path and on the walked trajectory |
| **G2** | The target is never visible along the route: too small, occluded, or in a dead angle | The semantic mask never clears the visibility threshold in any frame |
| **G3** | The target floats above reach or sits below the floor | `goal.y - agent.y` outside [0, 1.6] m |

A fourth check, run separately, finds objects intersecting scene geometry through
Habitat's contact points.

The two G1 checks are not redundant: the path check catches a plan that goes
upstairs, the trajectory check catches an agent that wandered onto a staircase
the plan did not anticipate.

## Repairing rather than discarding

A failing episode keeps its object category and scene, and receives a start and
goal sampled from episodes that passed. Distractors are migrated piece by piece:
their identity (category and object id) comes from the original episode so the
scene's semantic composition is unchanged, while their positions come from a
verified-good episode. The repaired episode therefore holds exactly the same
object list, placed where the geometry actually works.

Repair is iterative, because a substituted pair still has to hold up under
verification. On the training split it converged in a handful of rounds:

| Round | Bad in | Repaired |
|---|---|---|
| 1 | 232,740 | 226,220 |
| 2 | 6,520 | 6,225 |
| ... | ... | ... |
| N | 0 | |

## Relationship to upstream PIN

This is an overlay on a [PIN](https://github.com/aimagelab/pin) checkout, not a
standalone repository, so the ~100 MB of upstream third-party code is not
duplicated here. Two upstream files were modified; they ship under
`upstream_patches/` at their original paths.

```bash
git clone https://github.com/aimagelab/pin
cd pin
conda env create -f environment.yml && conda activate pin
pip install -r requirements.txt

git clone -b benchmark-fix https://github.com/Avalon-S/PInVerify.git /tmp/pv-fix
cp /tmp/pv-fix/*.py /tmp/pv-fix/*.sh .
cp -r /tmp/pv-fix/upstream_patches/* .
```

## Paths

```bash
export PIN_RESULT_DIR=./pin_result       # evaluation output
export PIN_CONTENT_DIR=./data/datasets/pin/hm3d/v1/val/content
export PIN_OUTPUT_DIR=./output_dir       # videos and raw run output
```

Defaults are repository-relative; nothing points at the machine this was built on.

## Running it

Full procedure in [docs/CLEANING_GUIDE.md](docs/CLEANING_GUIDE.md). In short:

```bash
# 1. measure every episode (needs a GPU)
./run_eval_goalview_7.sh                 # val, 7 workers
./run_eval_goalview_train.sh             # train, 3 workers to stay inside memory

# 2. classify what failed (CPU only)
python classify_abnormal_episodes.py

# 3. optional: object penetration
python detect_penetration.py --config configs/models/pin/pin_hm3d_v1.yaml \
    --split val --save-photos

# 4. repair
./run_repair_val.sh                      # val, one pass
./run_train_repair_loop.sh               # train, iterate to convergence
```

The training loop snapshots the dataset before every round (`content_v0`,
`content_v1`, ...), repairs, swaps it in, re-verifies only the episodes it
touched, and resumes from where it stopped if interrupted.

## Modules

| File | Role |
|---|---|
| `eval_goalview.py` | Oracle run over the val split, applying the three gates per episode |
| `eval_goalview_train.py` | The same for the train split, with lower parallelism |
| `classify_abnormal_episodes.py` | Sorts episodes into good, bad, and each failing gate combination |
| `detect_penetration.py` | Contact-point test for objects intersecting scene geometry |
| `semantic_floor_detector.py` | Floor assignment from HM3D semantic annotations, when they load |
| `extract_match_pool.py` | Collects verified (start, goal) pairs into the repair pool |
| `generate_repaired_dataset.py` | Substitutes start, goal and distractor geometry for failing episodes |
| `merge_gate_summaries_notebook.py` | Merges per-worker summaries and plots them |
| `count_val_episodes.py`, `inspect_dataset.py`, `inspect_missing_ep.py`, `analyze_scene_distribution.py` | Small inspection utilities |
| `plot_metrics.py` | Metric comparison plots |

## A caveat on semantic floors

HM3D's `semantic_annotations()` returns an empty object at Habitat runtime, so
G1 cannot use real floor heights and falls back to the OVON threshold of 0.25 m.
`semantic_floor_detector.py` implements the annotation-based route for scenes
where the data does load, and degrades to the same threshold when it does not.

## License

MIT, matching upstream PIN. Scene and object assets keep their own terms.
