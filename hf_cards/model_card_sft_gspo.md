---
license: mit
library_name: peft
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - active-instance-verification
  - embodied-ai
  - lora
  - gspo
  - reinforcement-learning
  - multi-view
  - verification
  - vision-language
  - qwen3-vl
  - paper-best
datasets:
  - Avalon-S/PInVerify
language:
  - en
pipeline_tag: visual-question-answering
---

# PInVerify Qwen3-VL-4B — SFT + GSPO ⭐ paper best

LoRA adapter for [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), built by **SFT** followed by **Group Sequence Policy Optimization (GSPO)** on the [PInVerify](https://huggingface.co/datasets/Avalon-S/PInVerify) RL training pool. **This is the paper-best checkpoint (Overall 85.6% / 88.9% with GT detection).**

| | |
|---|---|
| **Paper** | [arXiv:2605.30639](https://arxiv.org/abs/2605.30639) |
| **Code** | [github.com/Avalon-S/PInVerify](https://github.com/Avalon-S/PInVerify) |
| **Dataset** | [Avalon-S/PInVerify](https://huggingface.co/datasets/Avalon-S/PInVerify) |
| **Base model** | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| **Predecessor** | [PInVerify-Qwen3VL-4B-SFT](https://huggingface.co/Avalon-S/PInVerify-Qwen3VL-4B-SFT) |
| **Training stage** | SFT + GSPO |

## Results on PInVerify val (3,000 episodes)

| Detection | Overall | Pos | NegSame | NegDiff | ASD |
|---|---|---|---|---|---|
| Grounding DINO | **0.856** | 0.745 | 0.839 | 0.985 | 1.62 |
| GT bbox        | **0.889** | 0.813 | 0.864 | 0.991 | 1.65 |

Full breakdown in Table 6 / Appendix F of the paper.

## Usage

```bash
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B-SFT-GSPO --local-dir ./models/sft_gspo

ADAPTER=./models/sft_gspo bash scripts/start_multigpu_servers_lora.sh 4

python scripts/evaluate.py --config configs/agent/trained_gspo.yaml \
  dataset.root=./data/pv_dataset \
  method.bbox_mode=dino
```

## Training

- 4× RTX 3090, bf16, LoRA rank 64
- ~6 hours wall-clock
- Reward: multi-component (`r_format + r_answer + r_action - r_step_penalty`), `r_action` weighted by FPS-ranked best-sector metadata; see `training/reward.py`
- GSPO uses sequence-level importance ratios ([Zheng et al., 2025](https://arxiv.org/abs/2507.18071))
- Reproduce: `bash training/run_gspo_v3.sh`

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
