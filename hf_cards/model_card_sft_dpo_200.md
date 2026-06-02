---
license: mit
library_name: peft
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - active-instance-verification
  - embodied-ai
  - lora
  - dpo
  - preference-optimization
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

# PInVerify Qwen3-VL-4B — SFT + DPO (200 pairs)

LoRA adapter for [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), built by **SFT** followed by **Direct Preference Optimization** on **200 preference pairs** from the [PInVerify](https://huggingface.co/datasets/Avalon-S/PInVerify) training pool.

| | |
|---|---|
| **Paper** | [arXiv:2605.30639](https://arxiv.org/abs/2605.30639) |
| **Code** | [github.com/Avalon-S/PInVerify](https://github.com/Avalon-S/PInVerify) |
| **Dataset** | [Avalon-S/PInVerify](https://huggingface.co/datasets/Avalon-S/PInVerify) |
| **Base model** | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| **Predecessor** | [PInVerify-Qwen3VL-4B-SFT](https://huggingface.co/Avalon-S/PInVerify-Qwen3VL-4B-SFT) |
| **Training stage** | SFT + DPO (200 pairs) |

## Results on PInVerify val

See **Appendix F** of the paper for the full DPO-200 breakdown (Overall / Pos / NegSame / NegDiff / ASD under both Grounding DINO and GT detection).

## Usage

```bash
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B-SFT-DPO-200 --local-dir ./models/sft_dpo_200

ADAPTER=./models/sft_dpo_200 bash scripts/start_multigpu_servers_lora.sh 4

python scripts/evaluate.py --config configs/agent/trained_e2e.yaml \
  dataset.root=./data/pv_dataset \
  method.bbox_mode=dino
```

## Training

- 4× RTX 3090, bf16, LoRA rank 64
- ~30 min wall-clock (200 preference pairs)
- Initialize from the SFT adapter, then run DPO
- Reproduce: `bash training/run_dpo.sh` (with `DPO_SIZE=200`)

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
