# Evaluation Guide

This page covers running the training-free evaluation matrix and reproducing
Table 4 (training-free main table) and Table 5 (cross-model comparison) of the
paper. Trained agents are covered in [TRAINING.md](TRAINING.md).

## Prerequisites

1. Dataset at `./data/pv_dataset/` (see [DATASET.md](DATASET.md))
2. Qwen3-VL-4B (or 8B) weights at `./models/Qwen3-VL-4B-Instruct`
3. For `method.bbox_mode=dino`: a Grounding DINO server, started from inside a
   GroundingDINO checkout with `servers/run_groundingdino_server.py`
4. `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118`

## Quickest path: one configuration on a smoke split

```bash
# Start the Qwen3-VL server on one GPU. The model path is the MODEL_PATH
# constant at the top of the script; the batched server used by the multi-GPU
# launchers takes --model on the command line instead.
python servers/run_qwen3_server.py --port 12182 &

# Evaluate MV-Attr + LLM-NBV on 50 episodes with GT boxes
python scripts/evaluate.py \
  --config configs/agent/multi_view_attr_adaptive_llm.yaml \
  dataset.index_file=pv_index_50.jsonl \
  method.bbox_mode=gt \
  output.root=./outputs/smoke
```

`metrics.json` lands under `output.root` and carries overall / Pos / Neg_Same /
Neg_Diff accuracy, ASD, navigation-failure counts, and a per-category breakdown.

`pv_index_{50,100,500,1000}.jsonl` are progressively larger subsets of the same
test pool; `pv_index_all.jsonl` is the full 3,000-episode split every paper
number uses.

## Multi-GPU full sweep (Table 4)

```bash
# Boot 4 Qwen + GDINO server pairs (one per GPU) in tmux
bash scripts/start_multigpu_servers.sh 4

# Run the 9 training-free configs on the full split, DINO detection
bash run_all.sh

# Tear down
bash scripts/manage_servers_multigpu.sh stop
```

`run_all.sh` calls `scripts/evaluate_multigpu_dynamic.py`, which schedules
per-episode work across the servers with dynamic load balancing, so a slow GPU
does not hold up the others.

**Budget the time.** On 8 RTX 3090s the full 9-config main table takes about 17
hours of wall time, roughly 153 GPU-hours, and the heaviest single config
(MV-Attr + LLM-NBV) accounts for 31 of those. Adding `BBOX_MODES=gt` roughly
doubles it. See the hardware notes in [INSTALL.md](INSTALL.md); newer GPUs help a
lot here. Start with `bash run_all.sh 50` to confirm the pipeline works before
committing to the full split.

Knobs, all overridable from the environment:

| Variable | Default | Effect |
|---|---|---|
| `AGENT_SET` | `main` | `main` = Table 4's 9 configs, `embedding` = CLIP/SigLIP2, `all` = both |
| `BBOX_MODES` | `dino` | Space-separated; `gt` reproduces the Sec. 6.4 GT-box ablations |
| `NUM_GPUS` / `GPU_IDS` / `BASE_PORTS` | 4 / `0,1,2,3` / `12182,...` | Server topology |
| `OUT_BASE` | `./outputs/<agent_set>_<split>` | Where runs land |
| `SAVE_VIZ` | `false` | `true` also writes per-episode `episode.json`, needed by the figure scripts |

The positional argument is the split: `bash run_all.sh 50` for the smoke split,
no argument (or `3000`, or `all`) for the full one.

## Agent config matrix

Every config that produces a paper table row, with its expected accuracy, is
listed in [`configs/agent/README.md`](../configs/agent/README.md). In short:

| Table | Configs | Runner |
|---|---|---|
| Table 4, training-free | `single_view_{attr,direct,merged}` + `multi_view_{attr,direct}_adaptive_{random,fps,llm}` | `bash run_all.sh` |
| Table 5, embedding baselines | `{clip,siglip2}_single_view_merged` and the appendix sweep | `AGENT_SET=embedding bash run_all.sh` |
| Table 5, other backbones | the Table 4 configs against a different server | `start_multigpu_servers_sensenova.sh`, or the 8B weights |
| Table 6, trained | `trained_e2e.yaml` plus a LoRA adapter | `scripts/eval_trained.sh` |

Single-view configs carry no `adaptive` suffix because they take exactly one
step and never navigate. Earlier fixed-budget multi-view variants that no table
uses are not published.

## Detection mode

| `method.bbox_mode` | Description |
|---|---|
| `dino` | Grounding DINO detection, deployment-realistic, used for the main tables |
| `gt`   | Ground-truth boxes, the oracle upper bound used in the Sec. 6.4 ablations |

Toggle per run:

```bash
python scripts/evaluate.py --config <cfg> method.bbox_mode=dino
```

## Aggregation

```bash
# Tabulate every metrics.json under a run root
python scripts/summarize_all_agents.py --output_base ./outputs/main_all

# Comparison table; directories are positional arguments
python scripts/compare_metrics.py ./outputs/main_all
python scripts/compare_metrics.py ./outputs/main_all --breakdown --sort accuracy
python scripts/compare_metrics.py ./outputs/main_all --csv table4.csv
python scripts/compare_metrics.py ./outputs/main_all --mode dino --category
```

## Figures

The plotting scripts read run outputs from paths hardcoded at the top of each
file (`./outputs/qwen3_vl_4b/...` and `./outputs/trained/...`). Either edit those
constants or produce runs where the scripts expect them:

```bash
OUT_BASE=./outputs/qwen3_vl_4b SAVE_VIZ=true bash run_all.sh

python scripts/plot_per_category.py        # per-category accuracy, reads metrics.json
python scripts/plot_nbv_polar_dino.py      # NBV direction polar plots, reads episode.json
python scripts/plot_case_study.py          # one qualitative episode, reads episode.json
python scripts/plot_dataset_overview.py    # dataset statistics, reads the dataset directly
```

## Statistical notes

The paper reports a 95% binomial CI of about ±1.3 pp at p=0.85 on n=3,000.
Episodes are not i.i.d. (71 unique instances, roughly 42 episodes each), so
differences below 2 pp sit within the cluster CI. For close comparisons prefer an
instance-level paired bootstrap or McNemar's test.

## Reference numbers

The training-free reference points on the full 3,000-episode split, Qwen3-VL-4B
with DINO detection, are in
[`configs/agent/README.md`](../configs/agent/README.md). A run that lands within
about 1 pp of those is behaving correctly.

On a 50-episode smoke split, expect swings of several points in either direction.
Use it to check the pipeline runs end to end, not to compare methods.
