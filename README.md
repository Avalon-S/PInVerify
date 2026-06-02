# PInVerify: An Offline Embodied Benchmark for Active Instance Verification

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2605.30639-red)](https://arxiv.org/abs/2605.30639)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://avalon-s.github.io/PInVerify)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FMEA @ CVPR 2026](https://img.shields.io/badge/FMEA-CVPR%202026-blueviolet)](https://github.com/Avalon-S/PInVerify)

Official code release for the FMEA @ CVPR 2026 paper
**"PInVerify: An Offline Embodied Benchmark for Active Instance Verification."**

PInVerify introduces **Active Instance Verification (AIV)**: the agent has already navigated near a candidate object and must now actively select viewpoints around it to decide whether it matches a fine-grained natural-language description. We provide a 3,000-episode offline benchmark with a 6-sector navigation topology, plus reference training-free and LoRA-fine-tuned MLLM agents.

---

## Repository Structure

```
PInVerify/
├── pver/                 Core package: env, policies, tracker, fusion, NBV, eval, viz
├── configs/
│   ├── agent/            21 agent configs covering the paper's evaluation matrix
│   └── prompts/          9 prompt templates (extract / verify / category / merge / nav)
├── scripts/              Evaluation entry points, cache builders, figure scripts
├── training/             SFT + DPO/GRPO/GSPO data prep and training shells
├── servers/              VLM / detector server wrappers (Qwen3-VL, CLIP, SenseNova-SI)
├── data/examples/        Tiny episode samples for sanity checks
├── docs/                 DATASET / EVALUATION / TRAINING / ARCHITECTURE
├── hf_cards/             Hugging Face dataset + model cards (pre-authored)
├── project-page/         Static project website (auto-deployed to GitHub Pages)
├── run_all.sh            Master multi-GPU evaluation runner
└── runner.py             Batch evaluation harness
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Avalon-S/PInVerify.git
cd PInVerify
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

Optional external dependencies (install if you intend to use the corresponding baseline):

- **Grounding DINO** — clone and install from [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). PInVerify calls it through `servers/run_qwen_batched.py` via an HTTP wrapper.
- **SenseNova-SI** — clone [SenseNova/SenseNova-SI](https://github.com/SenseNova/SenseNova-SI) into `./SenseNova-SI` (or set `SENSENOVA_PATH=<dir>`).
- **ms-swift** — required for LoRA training; install via `pip install ms-swift`.

### 2. Download the dataset

```bash
# Hugging Face Datasets
pip install huggingface_hub
huggingface-cli download Avalon-S/PInVerify-dataset --repo-type dataset --local-dir ./data/pv_dataset
```

Expected layout after download:

```
data/pv_dataset/
├── pin_capture/          6-sector multi-view captures (val + train_sft + train_rl)
├── image_gt/             Ground-truth bounding-box masks
├── train_sft/            SFT training jsonl + cached crops
├── train_rl/             RL training trajectories
├── category_cache.json
├── attr_cache.json
├── merge_cache.json
└── object_descriptions_with_category.json
```

See [docs/DATASET.md](docs/DATASET.md) for the full data spec.

### 3. (Optional) Download trained checkpoints

Each fine-tuned variant lives in its own Hugging Face model repository:

| Variant | HF Repo | Overall (DINO) |
|---|---|---|
| SFT          | `Avalon-S/PInVerify-Qwen3VL-4B-SFT`          | 84.8 |
| SFT + DPO-200| `Avalon-S/PInVerify-Qwen3VL-4B-SFT-DPO-200`  | see paper |
| SFT + DPO-400| `Avalon-S/PInVerify-Qwen3VL-4B-SFT-DPO-400`  | 86.0 |
| SFT + GRPO   | `Avalon-S/PInVerify-Qwen3VL-4B-SFT-GRPO`     | 85.3 |
| **SFT + GSPO** ⭐ | `Avalon-S/PInVerify-Qwen3VL-4B-SFT-GSPO`     | **85.6** |

```bash
# Download the paper-best checkpoint
huggingface-cli download Avalon-S/PInVerify-Qwen3VL-4B-SFT-GSPO --local-dir ./models/pinverify/sft_gspo
# Or download all five at once:
for v in SFT SFT-DPO-200 SFT-DPO-400 SFT-GRPO SFT-GSPO; do
  huggingface-cli download "Avalon-S/PInVerify-Qwen3VL-4B-$v" --local-dir "./models/pinverify/${v,,}"
done
```

### 4. Run one configuration on the 50-episode smoke split

```bash
# Start a single Qwen3-VL-4B server (one GPU)
python servers/run_qwen3_server.py --port 12182 --model ./models/Qwen3-VL-4B-Instruct

# In another shell: run MV-Attr+LLM-NBV on 50 episodes
python scripts/evaluate.py \
  --config configs/agent/multi_view_attr_llm.yaml \
  +start_idx=0 +end_idx=50 \
  dataset.index_file=pv_index_sectors6_50.jsonl \
  method.bbox_mode=gt
```

Results land in `./outputs/<run_name>/metrics.json`.

---

## Reproducing the Paper Results

### Main training-free table (Table 5)

```bash
# 4-GPU dynamic evaluation across all 18 training-free configs × {GT, DINO}
bash run_all.sh 3000   # full 3,000-episode split (~74 GPU-hours total)
```

Aggregation:

```bash
python scripts/summarize_all_agents.py --root ./outputs/sectors6_3000
python scripts/compare_metrics.py --root ./outputs/sectors6_3000
```

### Trained agent table (Table 6)

```bash
# Re-train (skip if using released checkpoints)
bash training/run_sft_v3.sh
bash training/run_gspo_v3.sh

# Evaluate trained models
bash scripts/eval_trained.sh sft_v3
bash scripts/eval_trained.sh gspo_v3
```

### Paper figures

```bash
python scripts/generate_report_figs.py     # main plots
python scripts/plot_per_category.py        # per-category accuracy
python scripts/plot_nbv_polar_dino.py      # NBV direction polar plots
python scripts/plot_case_study.py          # qualitative case studies
```

See [docs/EVALUATION.md](docs/EVALUATION.md) and [docs/TRAINING.md](docs/TRAINING.md) for the full reproduction guide.

---

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

---

## Acknowledgements

PInVerify is built on top of [PInNED (Barsellotti et al., NeurIPS 2024)](https://arxiv.org/abs/2410.18195) for the HM3D scenes, Objaverse-XL object pool, and instance descriptions. We thank the authors for releasing their work.

## License

Released under the [MIT License](LICENSE). The benchmark data inherits PInNED's terms; please consult its license for dataset-use conditions.

## Contact

Yuhang Jiang — [jyhtjtj@gmail.com](mailto:jyhtjtj@gmail.com) — [avalon-s.github.io](https://avalon-s.github.io/)
