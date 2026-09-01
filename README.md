# PInVerify: capture pipeline

This branch holds the pipeline that turned the [PInNED](https://arxiv.org/abs/2410.18195)
navigation dataset into the **PInVerify** verification benchmark: object injection,
6-sector viewpoint capture, mask and reachability annotation, an iterative repair
loop, and the render audit reported in the paper's appendix.

The benchmark itself, the agents, and the evaluation live on the
[`main`](https://github.com/Avalon-S/PInVerify) branch. Nothing here is needed to
*use* the benchmark; this is how it was built.

> **Different environment from `main`.** This branch runs on the habitat stack
> (Python 3.9, `numpy<1.24`, `transformers==4.38.2`), which cannot coexist with the
> Qwen3-VL stack on `main`. Keep one conda environment per branch.

## Relationship to upstream PIN

This is not a standalone repository. It is the first-party layer of a
[PIN](https://github.com/aimagelab/pin) fork, published as an overlay so the 100 MB
of upstream third-party code is not duplicated here. Of the ~910 upstream files, this
work modified five; those are shipped separately under `upstream_patches/`.

```bash
# 1. Set up PIN and its habitat environment as upstream documents
git clone https://github.com/aimagelab/pin
cd pin
conda env create -f environment.yml && conda activate pin
pip install -r requirements.txt

# 2. Overlay this branch
git clone -b data-collection https://github.com/Avalon-S/PInVerify.git /tmp/pv-capture
cp -r /tmp/pv-capture/{pipeline,*.py,*.sh} .

# 3. Apply the upstream patches (they keep their original paths)
cp -r /tmp/pv-capture/upstream_patches/* .
```

`upstream_patches/` contains:

| File | Change |
|---|---|
| `habitat/core/env.py` | Load object templates once per scene instead of once per reset, which otherwise accumulates duplicate handles |
| `habitat/datasets/pin/pin.py` | Keep the JSON `episode_id` instead of renumbering, so incremental capture rounds do not collide |
| `habitat/config/habitat/task/pin.yaml` | Add the `look_down` action |
| `pin_eval.py` | Face the goal and tilt before the snapshot |
| `utils/wandb_logger.py` | Logging adjusted for sharded runs |

## Paths

Three environment variables control where everything lives. The defaults are
repository-relative, so nothing points at the machine this was built on:

```bash
export PIN_DATA_ROOT=./data/datasets/pin/hm3d/v1/val   # content/, content_verify/
export PIN_CAPTURE_ROOT=./captures                      # capture output
export PIN_RESULTS_DIR=./results/pin/val                # navigation evaluation results
```

All other settings, including the per-category mask-area thresholds, are in
[`pipeline/config.py`](pipeline/config.py).

## Running the pipeline

```bash
# Full run, starting from the raw PIN episodes
bash pipeline/run_pipeline.sh --init

# Capture only, when content/ is already prepared
bash pipeline/run_pipeline.sh

# Useful flags
bash pipeline/run_pipeline.sh --dry-run          # print the steps without running them
bash pipeline/run_pipeline.sh --start_ft 3       # resume from capture round 3
bash pipeline/run_pipeline.sh --no-save-vis      # skip the overview images
```

### What it does

**Phase 0 (`--init`, once).** Runs an oracle navigation evaluation over the raw
episodes, flattens them into `content_verify/` with one episode per object and no
distractors, then keeps the episodes the navigator can actually reach under a
height constraint.

**Phase 1 (iterative).** Each round captures the current episode set, analyzes the
result, and repairs the failures by moving their goal to a position that succeeded
for another object in the same scene. Round 0 captures everything; later rounds
capture only the failures and merge the previous round's successes by hardlink.
The loop runs until it converges or hits the round limit.

**Phase 1.5.** Episodes still failing after the last round are discarded from
`content/`.

**Phase 2.** Deletes the capture directories of failed episodes and re-verifies.

**Phase 3.** Writes a JSON report plus a readable summary.

### Acceptance criteria

An episode is kept when it has at least 6 navigable viewpoints inside the frustum
and at least 3 whose mask area clears the threshold for its category. Thresholds are
per-category and tiered by object size, from tiny objects such as keys and watches up
to backpacks and laptops. A viewpoint that is reachable but whose mask falls below
the threshold becomes a **trap view**, which the benchmark treats as a distinct
failure mode rather than discarding.

## Modules

| File | Role |
|---|---|
| `val_verify_capture_v2_distributed.py` | The capture worker: places the agent on the sector ring, checks navigability and frustum, renders RGB and mask, writes `meta.json` |
| `env.py` | Object injection and rendering setup |
| `distributed_pin_eval.py` | Sharded oracle navigation evaluation, used in Phase 0 |
| `pipeline/init_verify_dataset.py` | Phase 0: flatten to one episode per object |
| `pipeline/filter_good_episodes.py` | Phase 0: keep the reachable episodes |
| `pipeline/analyze_captures.py` | Per-round analysis, failure lists, per-scene progress |
| `pipeline/repair_episodes.py` | Move failing goals to known-good positions |
| `pipeline/merge_captures.py` | Hardlink the previous round's successes into the new round |
| `pipeline/discard_bad_episodes.py` | Drop episodes that never converged |
| `pipeline/cleanup_capture_dir.py` | Remove the directories of failed episodes |
| `pipeline/check_object_consistency.py` | Diagnostic: catch object_id and goal-position mixups from a bad repair |
| `pipeline/generate_report.py` | Final report |
| `extract_dataset_stats.py` | Dataset statistics over the finished captures |
| `generate_dataset_figures.py` | Produces the paper's appendix capture figures and tables |
| `scan_render_colors.py` | The render audit, see below |
| `fix_object_rendering.py` | The rejected emissive-channel alternative, kept because the paper reports its outcome |

## Render audit

One object out of 338 rendered as solid red: the flat shader reads only
`baseColorFactor` and discarded its texture. `scan_render_colors.py` found it by
averaging object pixels across every captured viewpoint and flagging channel-ratio
outliers, over 2,504 episodes. The 36 affected episodes were dropped, 1.4% of the set.

An emissive-channel workaround was tried first and did not recover the texture,
which is why the final dataset keeps PIN's original injection approach and simply
removes the anomaly. Full write-up in [`docs/RENDER_AUDIT.md`](docs/RENDER_AUDIT.md).

## Citation

If you use the capture pipeline, please cite both PInVerify and PInNED. See the
[`main`](https://github.com/Avalon-S/PInVerify) branch for the BibTeX entries.

## License

MIT, matching upstream PIN. The scene and object assets keep their own terms:
HM3D and Objaverse-XL through PInNED.
