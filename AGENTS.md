# Notes for coding agents

Read this before changing anything. It is the short list of things that are not
obvious from the code and that have caused real, silent breakage here. Human
documentation lives in [README.md](README.md) and [docs/](docs/); this file only
covers what an agent is likely to get wrong.

## What the repository is

An offline benchmark for **Active Instance Verification**: the agent has already
navigated near a candidate object and must decide whether it matches a
fine-grained description, choosing viewpoints around it to settle the question.

There is no simulator at evaluation time. Every viewpoint was rendered ahead of
time in Habitat and is read off disk, so a run is deterministic given a fixed
model. Anything that looks like navigation is a lookup into the capture set.

Three branches, three different worlds. `main` is the benchmark. `data-collection`
is the Habitat capture pipeline; `benchmark-fix` is the upstream PIN repairs.
The latter two need Python 3.9 and habitat 0.2.3 and share nothing with `main`'s
environment. Do not try to make them coexist.

## Traps

**There are two sector numbering systems, and they are not the same numbers.**

*On disk*, `sector_index` in `meta.json` and `navigable_sectors` in the index are
absolute world bearings on a twelve-slot circle. Only every other slot is used,
so the ids are `0, 2, 4, 6, 8, 10`, and slot `s` covers `[30s, 30s+30)` degrees
of `atan2(cz - gz, cx - gx)` around the goal.

*At runtime*, [`pver/envs/env.py`](pver/envs/env.py) ignores those and builds its
own map: it takes `captures[0]` as the reference direction and bins every other
capture by its angle relative to that, giving virtual ids `0..5`. **Virtual
sector 0 is wherever the first capture happens to be, not world bearing 0**, so
the frame is different in every episode.

Mixing the two agrees on sector 0 and diverges after, which is why it survives
the first case you test.

Directions never go through either numbering. `front-left` and the rest resolve
geometrically: add the offset to the current bearing, then search unvisited
sectors for the capture whose actual bearing is closest to the target and within
half a sector span. So `(current_sector + offset) % 6` is not a direction, even
though the virtual ids are evenly spaced enough to make it look like one. If you
need a direction, go through the coordinates.

**Habitat's forward axis is -Z.** The top-down visualisation negates it
(`goal_z = -goal_pos[2]`) while the bearing math above does not. A plot can
therefore look correct while the geometry behind it is mirrored, so do not use
the visualisation alone to confirm a change to direction resolution.

**The index jsonl wins over `meta.json`.** The index carries
`target_object_id` (what is actually in the scene) and `query_object_id` (what
the description refers to); they differ for negative pairs, and that difference
is the whole task. `meta.json` carries `object_id`, meaning the target. A plain
`ep_data.update(full_meta)` silently introduces `object_id` and destroys the
distinction, which turns every negative pair into a positive one. The merge in
[`pver/envs/env.py`](pver/envs/env.py) only fills keys the index did not set;
keep it that way. Policies read `query_object_id` first and fall back second.

**A trap view is not a missing view.** A capture with `navigable: true` and
`mask_meets_threshold: false` is reachable but shows nothing useful. In GT bbox
mode this must pass the full image through with `detection_confidence = 0.1`, not
fall back to the detector. Silently substituting DINO there makes GT mode stop
being an oracle and the GT column stops meaning anything.

**Two distinct navigation failures, both costing a step.** A trap view (arrived,
target not visible) and an unreachable direction (no capture in that wedge). Both
are surfaced to the model through the `{visibility_warning}` placeholder that all
three navigation prompts carry. Removing that placeholder does not error; it just
makes the agent blind to its own failures.

**`configs/prompts/` is the core asset.** The prompt templates are the substance
of the method, not scaffolding. Do not reword, reformat, or "improve" them.
Changing a prompt invalidates every number in the paper.

## Reproducing the paper

[`configs/agent/README.md`](configs/agent/README.md) maps every table row to the
config that produced it, with the expected accuracy. Use it. Config names are
close enough to each other to be easy to confuse, and the differences matter:
every published number uses an `*_adaptive_*` config, meaning `attr_majority`
fusion with convergence detection inside a 6-step budget. A config without
`adaptive` in its name is a fixed-3-step ablation and will not reproduce anything.

Table 4 is the training-free main table, 5 is cross-model, 6 is the trained
agents. Appendix Table 18 is the embedding sweep, Table 23 the full trained
results.

GSPO is `rlhf_type=grpo` with `importance_sampling_level=sequence`. That is the
only difference from GRPO; both train on the same `rl_data_v2.jsonl`.

## Environments

`pv_bench` (torch 2.1.2, transformers 4.52.4) runs everything, including
evaluating a trained adapter. `pv_train` (torch 2.2.0, transformers 4.57.0,
ms-swift 3.12.6) exists only because ms-swift needs a newer transformers than
`requirements.txt` pins. Train in one, evaluate in the other.
[docs/INSTALL.md](docs/INSTALL.md) has both.

## Before you claim something works

The benchmark is a measurement instrument, so a change that runs is not a change
that is correct. Both failure modes above produce complete runs with wrong
numbers. If you touch direction resolution, pair loading, fusion, or the trap
view path, evaluate on `pv_index_50.jsonl` and compare against the expected
accuracy in [`configs/agent/README.md`](configs/agent/README.md) before saying it
is fine.

The code is published as it was when it produced the paper's numbers. It has not
been re-run end to end in a clean environment, so paths and ports will need
adjusting for yours. That is expected; a path that does not match your machine is
not a bug to report.
