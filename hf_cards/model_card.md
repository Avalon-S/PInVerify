---
license: mit
library_name: peft
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - active-instance-verification
  - embodied-ai
  - lora
  - sft
  - dpo
  - grpo
  - gspo
  - reinforcement-learning
  - multi-view
  - verification
  - vision-language
  - qwen3-vl
datasets:
  - Avalon-S/PInVerify
language:
  - en
pipeline_tag: visual-question-answering
---

# PInVerify Qwen3-VL-4B adapters

Every LoRA adapter reported in the FMEA @ CVPR 2026 paper
**"PInVerify: An Offline Embodied Benchmark for Active Instance Verification"**,
in one repository. All are adapters for
[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

| | |
|---|---|
| **Paper** | [arXiv:2605.30639](https://arxiv.org/abs/2605.30639) |
| **Code** | [github.com/Avalon-S/PInVerify](https://github.com/Avalon-S/PInVerify) |
| **Dataset** | [Avalon-S/PInVerify](https://huggingface.co/datasets/Avalon-S/PInVerify) |
| **Base model** | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |

## The task

The agent has already navigated near a candidate object and must decide whether
it matches a fine-grained description, choosing viewpoints around it to settle
the question. These adapters collapse the modular pipeline (attribute
decomposition, per-view verification, tracker, next-best-view) into a single
model that navigates and answers end to end.

## Which adapter to use

`generic-cot/gspo` is the paper's headline result. Take that one unless you have
a reason not to.

The two families differ only in how the chain-of-thought in the SFT targets was
written. **Generic-CoT** is the paper's main table; **Specific-CoT** is the
appendix variant. Both RL stages train on the same data, so within a family only
the SFT starting point differs.

```
generic-cot/          Table 6, the main trained results
├── sft/              SFT only
├── grpo/             SFT then GRPO
└── gspo/             SFT then GSPO      <- paper best
specific-cot/         Appendix F, Table 23
├── sft/
├── dpo-200/          SFT then DPO, 200 steps
├── dpo-400/          SFT then DPO, 400 steps
├── grpo/
└── gspo/
```

## Results on the PInVerify test split (3,000 episodes)

Overall accuracy, with positive-pair accuracy in parentheses. Grounding DINO is
the deployment-realistic setting; GT boxes are the oracle upper bound.

| Adapter | DINO | GT |
|---|---|---|
| *(base model, no fine-tuning)* | 0.706 (0.146) | 0.710 (0.161) |
| `generic-cot/sft` | 0.848 (0.759) | 0.877 (0.828) |
| `generic-cot/grpo` | 0.853 (0.736) | 0.887 (0.806) |
| **`generic-cot/gspo`** | **0.856** (0.745) | **0.889** (0.813) |
| `specific-cot/sft` | 0.858 (0.697) | 0.884 (0.761) |
| `specific-cot/dpo-200` | 0.859 (0.700) | 0.881 (0.756) |
| `specific-cot/dpo-400` | 0.860 (0.665) | 0.884 (0.729) |
| `specific-cot/grpo` | 0.855 (0.793) | 0.884 (0.847) |
| `specific-cot/gspo` | 0.851 (0.796) | 0.889 (0.813) |

The 95% binomial confidence interval at p=0.85 on n=3,000 is about ±1.3 pp, so
most of the differences between fine-tuned variants sit inside the noise. The
gap that matters is against the un-tuned base model, which fails on positives
almost entirely (0.146) because of a strong "no match" bias.

Training-free comparison: the best modular agent reaches 0.850 overall with 0.596
on positives. The trained agents trade calibration for confirmation, gaining
about 15 pp on positives while giving up ground on same-category rejection, and
they are more efficient (1.6 steps to decision versus 2.3, navigation failures
9% versus 28%).

## Usage

```bash
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B \
    --include "generic-cot/gspo/*" --local-dir ./models/pinverify

ADAPTER=./models/pinverify/generic-cot/gspo \
    bash scripts/start_multigpu_servers_lora.sh 4

bash scripts/eval_trained.sh gspo
```

Evaluation runs through `configs/agent/trained_e2e.yaml`, which drives the
end-to-end agent. `scripts/eval_trained.sh` covers both detection modes by
default.

## Training

4× NVIDIA RTX 3090, bf16, [ms-swift](https://github.com/modelscope/ms-swift).
LoRA rank 16, alpha 32, applied to all linear layers of the language model; the
vision encoder is frozen.

| Stage | Data | Starts from | Steps | LR |
|---|---|---|---|---|
| SFT | `sft_data_v2` (Generic) or `sft_data_v3` (Specific), 21,931 samples | base model | 686 | 1e-4 |
| DPO | `dpo_data_v3`, 13,273 preference pairs | SFT | 200 / 400 | 5e-7 |
| GRPO | `rl_data_v2` | SFT | 500 | 1e-6 |
| GSPO | `rl_data_v2` | SFT | 500 | 1e-6 |

GSPO differs from GRPO only in using sequence-level importance ratios
([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)); in ms-swift terms,
`rlhf_type=grpo` with `importance_sampling_level=sequence`.

The RL reward has four components: output format, final answer, whether the
chosen navigation direction lands on a sector the metadata marks visible, and a
per-step penalty that pushes the agent to stop. See `training/reward.py`.

Reproduce with `training/run_sft.sh` then `training/run_gspo.sh` for the Generic
family, or the `*_v3.sh` scripts for Specific. Each adapter directory keeps its
`args.json` and `trainer_state.json`, so the run configuration and the loss curve
are both recoverable.

Two edits were made on the way here. Absolute paths from the training machine
were rewritten to their repository-relative equivalents, in `args.json` and in
`adapter_config.json`, where `base_model_name_or_path` would otherwise point at a
directory that exists on no other machine. And `training_args.bin`, a pickle that
duplicates `args.json`, was dropped along with the `README.md` that peft
generates.

## Citation

```bibtex
@misc{jiang2026pinverifyofflineembodiedbenchmark,
      title={PInVerify: An Offline Embodied Benchmark for Active Instance Verification},
      author={Yuhang Jiang},
      year={2026},
      eprint={2605.30639},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.30639},
}
```

## License

MIT for the adapters. The base model and the dataset keep their own licenses;
the benchmark data inherits PInNED's terms.
