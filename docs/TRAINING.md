# Training Guide

This page covers LoRA fine-tuning the Qwen3-VL-4B end-to-end agent: **SFT** then
**DPO / GRPO / GSPO**. The trained agents correspond to Table 6 of the paper.

## Two training-data variants

The repository ships two parallel pipelines. They differ only in how the
chain-of-thought in the SFT targets is written, and they map to different parts
of the paper:

| Variant | Data | Scripts | Adapter output | Paper |
|---|---|---|---|---|
| **Generic-CoT** | `sft_data_v2.jsonl` | `run_sft.sh`, `run_grpo.sh`, `run_gspo.sh` | `./outputs/training/*_v2*` | Table 6 (main) |
| **Specific-CoT** | `sft_data_v3.jsonl` | `run_sft_v3.sh`, `run_grpo_v3.sh`, `run_gspo_v3.sh`, `run_dpo.sh` | `./outputs/training/{sft,grpo,gspo,dpo}` | Appendix F (Table 23) |

Both RL stages train on the same `rl_data_v2.jsonl`; only the SFT adapter they
start from differs.

Headline numbers on the 3,000-episode test split, DINO detection:

| | Generic-CoT | Specific-CoT |
|---|---|---|
| SFT | 0.848 | 0.858 |
| SFT + GRPO | 0.853 | 0.855 |
| SFT + GSPO | **0.856** | 0.851 |
| SFT + DPO (200 / 400) | not run | 0.859 / 0.860 |

The paper's headline trained result (0.856 overall, 0.745 Pos, ASD 1.62,
NavFail 9.1%) is **Generic-CoT GSPO**, so reproduce Table 6 with `run_sft.sh`
followed by `run_gspo.sh`.

## Prerequisites

1. Dataset at `./data/pv_dataset/` with `train_sft/` and `train_rl/` populated
   (including the `crops*` directories, which hold the pre-cut object crops)
2. Qwen3-VL-4B base weights at `./models/Qwen3-VL-4B-Instruct`
3. The `pv_train` environment (torch 2.2.0+cu118, transformers 4.57.0,
   ms-swift 3.12.6), built in [INSTALL.md §4](INSTALL.md#4-pv_train-optional).
   It is separate from the `pv_bench` evaluation environment because ms-swift
   requires a newer transformers than `requirements.txt` pins. Train in
   `pv_train`, evaluate the resulting adapter in `pv_bench`.
4. At least 4 GPUs with 24 GB is comfortable; single-GPU works with a reduced
   batch size

## Stage 1: training data

The dataset release already contains the generated training files, so **you can
skip straight to Stage 2**:

```
data/pv_dataset/train_sft/sft_data_v2.jsonl     Generic-CoT SFT targets
data/pv_dataset/train_sft/sft_data_v3.jsonl     Specific-CoT SFT targets
data/pv_dataset/train_rl/rl_data_v2.jsonl       GRPO / GSPO prompts
data/pv_dataset/train_rl/dpo_data_v3.jsonl      DPO preference pairs
```

Regenerate them only if you change the prompt format or the sampling policy.
The generators take explicit paths (no defaults pointing at a private machine):

```bash
# Generic-CoT SFT data
python training/prepare_sft_data.py \
  --data-root ./data/pv_dataset \
  --train-dir ./data/pv_dataset/train_sft \
  --index-file ./data/pv_dataset/train_sft/pv_train_sft_index.jsonl \
  --desc-db ./data/pv_dataset/object_descriptions_with_category.json \
  --output ./data/pv_dataset/train_sft/sft_data_v2.jsonl \
  --crop-dir ./data/pv_dataset/train_sft/crops

# Specific-CoT SFT data
python training/prepare_sft_data_v3.py \
  --data-root ./data/pv_dataset \
  --train-dir ./data/pv_dataset/train_sft \
  --index-file ./data/pv_dataset/train_sft/pv_train_sft_index.jsonl \
  --desc-db ./data/pv_dataset/object_descriptions_with_category.json \
  --output ./data/pv_dataset/train_sft/sft_data_v3.jsonl \
  --crop-dir ./data/pv_dataset/train_sft/crops_v3

# RL prompts (GRPO / GSPO)
python training/prepare_rl_data.py \
  --data-root ./data/pv_dataset \
  --train-dir ./data/pv_dataset/train_rl \
  --index-file ./data/pv_dataset/train_rl/pv_train_rl_index.jsonl \
  --desc-db ./data/pv_dataset/object_descriptions_with_category.json \
  --output ./data/pv_dataset/train_rl/rl_data_v2.jsonl \
  --crop-dir ./data/pv_dataset/train_rl/crops_rl

# DPO preference pairs
python training/prepare_dpo_data.py \
  --data-root ./data/pv_dataset \
  --train-dir ./data/pv_dataset/train_rl \
  --index-file ./data/pv_dataset/train_rl/pv_train_rl_index.jsonl \
  --desc-db ./data/pv_dataset/object_descriptions_with_category.json \
  --output ./data/pv_dataset/train_rl/dpo_data_v3.jsonl \
  --crop-dir ./data/pv_dataset/train_rl/crops_dpo
```

`--attr-cache` and `--category-cache` are optional; pass the dataset's
`attr_cache.json` / `category_cache.json` to avoid re-querying the LLM for
attribute decompositions.

Supporting utilities in `training/`:

- `sample_train_episodes.py` samples non-overlapping SFT/RL episode sets from the PInNED parts
- `prepare_train_index.py` builds the episode index by scene/position
- `build_train_attr_split.py` partitions training attributes
- `build_train_distractors_map.py` builds the distractor sampling map
- `count_train_positions.py` sanity-checks position coverage
- `reward.py` is the reward function used by the RL stages

## Stage 2: SFT

```bash
# Generic-CoT (paper main table)
bash training/run_sft.sh          # -> ./outputs/training/sft_v2/

# Specific-CoT (appendix)
bash training/run_sft_v3.sh       # -> ./outputs/training/sft/
```

Both invoke ms-swift with LoRA rank 16, alpha 32, lr 1e-4, 3 epochs,
`max_length` 2048, bf16, batch size 2 per device. Edit `CUDA_VISIBLE_DEVICES`
and `NPROC_PER_NODE` inside the script for a different GPU count.

The RL scripts expect the chosen SFT adapter at `<output>/best_adapter`, so
symlink or copy the checkpoint you want to continue from:

```bash
ln -s checkpoint-XXXX ./outputs/training/sft_v2/best_adapter
```

## Stage 3: preference / RL alignment

```bash
# Generic-CoT
bash training/run_grpo.sh         # -> ./outputs/training/grpo_v2_from_sft/
bash training/run_gspo.sh         # -> ./outputs/training/gspo_v2_from_sft/   (paper best)

# Specific-CoT
bash training/run_grpo_v3.sh      # -> ./outputs/training/grpo/
bash training/run_gspo_v3.sh      # -> ./outputs/training/gspo/
bash training/run_dpo.sh          # -> ./outputs/training/dpo/
```

`run_grpo.sh` / `run_gspo.sh` read a `MODE` variable: the default `sft`
continues from the SFT adapter (what the paper does), `MODE=scratch` trains RL
directly on the base model into `*_from_scratch/`.

## Evaluating a trained checkpoint

```bash
ADAPTER=./outputs/training/gspo_v2_from_sft bash scripts/start_multigpu_servers_lora.sh 4
bash scripts/eval_trained.sh gspo_v2
```

`scripts/eval_trained.sh` drives `configs/agent/trained_e2e.yaml` through the
multi-GPU evaluator and loads the LoRA adapter via a dedicated server
(`servers/run_qwen_batched_lora.py`). It evaluates both GT and DINO detection by
default; `BBOX_MODES=dino` restricts it.

## Released checkpoints

All adapters live in one repository, `Avalon-S/PInVerify-Qwen3VL-4B`, under
subdirectories named after the training variant:

| Subdirectory | Stage | Source run | DINO | GT |
|---|---|---|---|---|
| `generic-cot/sft` | SFT | `run_sft.sh` | 0.848 | 0.877 |
| `generic-cot/grpo` | SFT+GRPO | `run_grpo.sh` | 0.853 | 0.887 |
| **`generic-cot/gspo`** | SFT+GSPO | `run_gspo.sh` | **0.856** | **0.889** |
| `specific-cot/sft` | SFT | `run_sft_v3.sh` | 0.858 | 0.884 |
| `specific-cot/dpo-200` | SFT+DPO | `run_dpo.sh` @200 | 0.859 | 0.881 |
| `specific-cot/dpo-400` | SFT+DPO | `run_dpo.sh` @400 | 0.860 | 0.884 |
| `specific-cot/grpo` | SFT+GRPO | `run_grpo_v3.sh` | 0.855 | 0.884 |
| `specific-cot/gspo` | SFT+GSPO | `run_gspo_v3.sh` | 0.851 | 0.889 |

Each subdirectory keeps its `args.json` and `trainer_state.json`, so both the
ms-swift configuration that produced it and its loss curve are recoverable. Paths
inside `args.json` and `adapter_config.json` were rewritten to their
repository-relative form on upload, since the originals pointed at the training
machine.

Download one and point the LoRA server at it:

```bash
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B \
    --include "generic-cot/gspo/*" --local-dir ./models/pinverify

ADAPTER=./models/pinverify/generic-cot/gspo bash scripts/start_multigpu_servers_lora.sh 4
```

## Reward design

The RL stages use a multi-component reward (`training/reward.py`):

- `r_format`: output adheres to `<think>...</think><answer>...</answer>`
- `r_answer`: final YES/NO matches ground truth
- `r_action`: the chosen navigation direction visits a sector the metadata marks visible
- `r_step_penalty`: small per-step penalty to encourage stopping

`r_action` implicitly encodes an FPS-ranked best-sector preference. For a clean
NBV-agnostic reward ablation, set the FPS weight in `reward.py` to 0 and re-run.

## Wall-clock reference

Rough figures from the authors' runs, bf16 + LoRA:

| Stage | 4x RTX 3090 | 1x RTX 3090 |
|---|---|---|
| SFT   | ~3 hours | ~10 hours |
| GSPO  | ~6 hours | ~22 hours |
| DPO   | ~1 hour  | ~3.5 hours |
