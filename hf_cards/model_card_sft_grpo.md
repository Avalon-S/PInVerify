---
license: mit
library_name: peft
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - active-instance-verification
  - embodied-ai
  - lora
  - grpo
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

# PInVerify Qwen3-VL-4B — SFT + GRPO

LoRA adapter for [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), built by **SFT** followed by **Group Relative Policy Optimization (GRPO)** on the [PInVerify](https://huggingface.co/datasets/Avalon-S/PInVerify) RL training pool.

| | |
|---|---|
| **Paper** | [arXiv:2605.30639](https://arxiv.org/abs/2605.30639) |
| **Code** | [github.com/Avalon-S/PInVerify](https://github.com/Avalon-S/PInVerify) |
| **Dataset** | [Avalon-S/PInVerify](https://huggingface.co/datasets/Avalon-S/PInVerify) |
| **Base model** | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| **Predecessor** | [PInVerify-Qwen3VL-4B-SFT](https://huggingface.co/Avalon-S/PInVerify-Qwen3VL-4B-SFT) |
| **Training stage** | SFT + GRPO |

## Results on PInVerify val (3,000 episodes, Grounding DINO detection)

| Overall | Pos | NegSame | NegDiff | ASD |
|---|---|---|---|---|
| 0.853 | 0.736 | 0.838 | 0.985 | 1.61 |

Full breakdown in Table 6 / Appendix F of the paper.

## Usage

```bash
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B-SFT-GRPO --local-dir ./models/sft_grpo

ADAPTER=./models/sft_grpo bash scripts/start_multigpu_servers_lora.sh 4

python scripts/evaluate.py --config configs/agent/trained_grpo.yaml \
  dataset.root=./data/pv_dataset \
  method.bbox_mode=dino
```

## Training

- 4× RTX 3090, bf16, LoRA rank 64
- ~6 hours wall-clock
- Multi-component reward: `r_format + r_answer + r_action - r_step_penalty` (see `training/reward.py`)
- Reproduce: `bash training/run_grpo_v3.sh`

## Citation

```bibtex
@inproceedings{jiang2026pinverify,
  title         = {PInVerify: An Offline Embodied Benchmark for Active Instance Verification},
  author        = {Jiang, Yuhang},
  booktitle     = {Foundation Models Meet Embodied Agents (FMEA) Workshop at CVPR},
  year          = {2026},
  note          = {Poster},
  eprint        = {2605.30639},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

MIT (this adapter). Base model and dataset retain their own licenses.
