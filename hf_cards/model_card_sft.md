---
license: mit
library_name: peft
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - active-instance-verification
  - embodied-ai
  - lora
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

# PInVerify Qwen3-VL-4B — SFT

LoRA adapter for [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), fine-tuned with **Supervised Fine-Tuning** on the [PInVerify](https://huggingface.co/datasets/Avalon-S/PInVerify) training pool for the Active Instance Verification (AIV) task.

| | |
|---|---|
| **Paper** | [arXiv:2605.30639](https://arxiv.org/abs/2605.30639) |
| **Code** | [github.com/Avalon-S/PInVerify](https://github.com/Avalon-S/PInVerify) |
| **Dataset** | [Avalon-S/PInVerify](https://huggingface.co/datasets/Avalon-S/PInVerify) |
| **Base model** | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| **Training stage** | SFT (LoRA rank 64, ms-swift) |

## Results on PInVerify val (3,000 episodes, Grounding DINO detection)

| Overall | Pos | NegSame | NegDiff | ASD |
|---|---|---|---|---|
| 0.848 | 0.759 | 0.814 | 0.971 | 1.96 |

Full breakdown in Table 6 / Appendix F of the paper.

## Usage

```bash
# 1. Download
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B-SFT --local-dir ./models/sft

# 2. Serve the LoRA-loaded base model
ADAPTER=./models/sft bash scripts/start_multigpu_servers_lora.sh 4

# 3. Evaluate
python scripts/evaluate.py --config configs/agent/trained_sft.yaml \
  dataset.root=./data/pv_dataset \
  method.bbox_mode=dino
```

## Training

- 4× RTX 3090, bf16, LoRA rank 64, batch size 4 / GPU
- ~3 hours wall-clock (1 epoch on ~15K SFT pairs)
- Training data: `train_sft/sft_data_v3.jsonl` from the [PInVerify dataset](https://huggingface.co/datasets/Avalon-S/PInVerify)
- Reproduce: `bash training/run_sft_v3.sh`

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

MIT (this adapter). The base model and dataset have their own licenses — see their respective pages.
