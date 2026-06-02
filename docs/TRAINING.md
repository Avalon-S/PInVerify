# Training Guide

This page covers LoRA fine-tuning the Qwen3-VL-4B end-to-end agent: **SFT** → **DPO / GRPO / GSPO**. The trained agents correspond to Table 6 of the paper.

## Prerequisites

1. Dataset at `./data/pv_dataset/` (with `train_sft/` and `train_rl/` populated)
2. Qwen3-VL-4B base weights at `./models/Qwen3-VL-4B-Instruct`
3. `pip install ms-swift` (training framework)
4. ≥4 GPUs (24 GB+) recommended; single-GPU is possible with reduced batch size

## Stage 1: prepare training data

```bash
# Build training data jsonl for SFT
python training/prepare_sft_data_v3.py \
  --episodes ./data/pv_dataset/pin_capture/train_sft \
  --image_gt ./data/pv_dataset/image_gt \
  --output ./data/pv_dataset/train_sft/sft_data_v3.jsonl

# Build training data for DPO
python training/prepare_dpo_data.py --output ./data/pv_dataset/train_sft/dpo_data.jsonl

# Build training data for RL (GRPO / GSPO)
python training/prepare_rl_data.py --output ./data/pv_dataset/train_rl/rl_data.jsonl
```

Supporting utilities live in `training/`:

- `sample_train_episodes.py` — sample non-overlapping SFT/RL episode sets from PInNED parts
- `prepare_train_index.py` — build episode index by scene/position
- `build_train_attr_split.py` — partition training attributes
- `build_train_distractors_map.py` — distractor sampling map
- `count_train_positions.py` — sanity check on position coverage
- `reward.py` — reward model for RL stages

## Stage 2: SFT

```bash
bash training/run_sft_v3.sh
# Output: ./outputs/training/sft/
```

`run_sft_v3.sh` invokes ms-swift with LoRA on Qwen3-VL-4B. Defaults to 4 GPUs; edit `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` inside the script for single-GPU.

## Stage 3 (option A): DPO

```bash
bash training/run_dpo.sh
# Output: ./outputs/training/dpo/
```

## Stage 3 (option B): GRPO

```bash
bash training/run_grpo_v3.sh
# Output: ./outputs/training/grpo/
```

## Stage 3 (option C): GSPO  (paper's best — 85.6% Overall)

```bash
bash training/run_gspo_v3.sh
# Output: ./outputs/training/gspo/
```

## Evaluating a trained checkpoint

```bash
bash scripts/eval_trained.sh gspo_v3
# Equivalent: python scripts/evaluate.py --config configs/agent/trained_gspo.yaml ...
```

`scripts/eval_trained.sh` wraps the standard evaluator and loads the LoRA adapter via a dedicated server (`servers/run_qwen_batched_lora.py`).

## Released checkpoints

Each variant ships as a standalone Hugging Face model repository (so HF download stats track per-variant):

| Variant | HF Repo |
|---|---|
| SFT-only           | [`Avalon-S/PInVerify-Qwen3VL-4B-SFT`](https://huggingface.co/Avalon-S/PInVerify-Qwen3VL-4B-SFT) |
| SFT + DPO (200)    | [`Avalon-S/PInVerify-Qwen3VL-4B-SFT-DPO-200`](https://huggingface.co/Avalon-S/PInVerify-Qwen3VL-4B-SFT-DPO-200) |
| SFT + DPO (400)    | [`Avalon-S/PInVerify-Qwen3VL-4B-SFT-DPO-400`](https://huggingface.co/Avalon-S/PInVerify-Qwen3VL-4B-SFT-DPO-400) |
| SFT + GRPO         | [`Avalon-S/PInVerify-Qwen3VL-4B-SFT-GRPO`](https://huggingface.co/Avalon-S/PInVerify-Qwen3VL-4B-SFT-GRPO) |
| **SFT + GSPO** ⭐  | [`Avalon-S/PInVerify-Qwen3VL-4B-SFT-GSPO`](https://huggingface.co/Avalon-S/PInVerify-Qwen3VL-4B-SFT-GSPO) — paper-best (85.6%) |

Download one (or all) and point the LoRA server at it:

```bash
# Paper-best checkpoint
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B-SFT-GSPO --local-dir ./models/pinverify/sft_gspo

ADAPTER=./models/pinverify/sft_gspo \
  bash scripts/start_multigpu_servers_lora.sh 4
```

## Reward design summary

The RL stages (GRPO / GSPO) use a multi-component reward (`training/reward.py`):

- **r_format** — output adheres to `<think>...</think><answer>...</answer>`
- **r_answer** — final YES/NO matches ground truth
- **r_action** — chosen NAV direction visits a sector that was visible per metadata
- **r_step_penalty** — small penalty per step to encourage stopping

The paper observes that `r_action` implicitly encodes an FPS-ranked best-sector preference; if you want a clean ablation of NBV-agnostic reward, set the FPS weight in `reward.py` to 0 and re-run.

## Wall-clock reference

| Stage | 4× RTX 3090 | 1× RTX 3090 |
|---|---|---|
| SFT (1 epoch on ~15k pairs) | ~3 hours | ~10 hours |
| GSPO (RL fine-tune)        | ~6 hours | ~22 hours |
| DPO (400 pairs)            | ~1 hour  | ~3.5 hours |

All times measured with bf16, LoRA rank 64, batch size 4 per GPU.
