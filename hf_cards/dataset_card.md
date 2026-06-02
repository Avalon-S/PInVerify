---
license: mit
task_categories:
  - visual-question-answering
  - image-classification
language:
  - en
size_categories:
  - 1K<n<10K
source_datasets:
  - extended|other
tags:
  - active-instance-verification
  - embodied-ai
  - multi-view
  - habitat
  - hm3d
  - objaverse
  - benchmark
  - fine-grained
  - vision-language
pretty_name: PInVerify — Active Instance Verification Benchmark
---

# PInVerify Dataset

**An offline embodied benchmark for Active Instance Verification (AIV).**

| | |
|---|---|
| **Paper** | [arXiv:2605.30639](https://arxiv.org/abs/2605.30639) |
| **Code** | [github.com/Avalon-S/PInVerify](https://github.com/Avalon-S/PInVerify) |
| **Project page** | [avalon-s.github.io/PInVerify](https://avalon-s.github.io/PInVerify) |
| **Venue** | FMEA Workshop @ CVPR 2026 (Poster) |

## Overview

Embodied agents that navigate to a target object don't always reach the **right instance** — subtle attribute differences ("white floral" vs. "white striped") demand close-range, multi-view inspection. PInVerify isolates this post-arrival verification step as a standalone task and ships a 3,000-episode evaluation benchmark, plus training pools for SFT and RL fine-tuning.

## Statistics

| Split | Episodes | Pair types (positive : neg_same : neg_diff) |
|---|---|---|
| `val`         | 3,000  | 1,000 : 1,000 : 1,000 |
| `train_sft`   | ~15K pairs    | balanced |
| `train_rl`    | ~5K trajectories | balanced |

- **18 object categories** (PInNED-inherited)
- **71 unique evaluation instances** across **35 HM3D scenes**
- **6 sectors per episode** (far/near rings)
- Trap-view + unreachable-sector annotations

## Episode structure

Each episode in `pin_capture/val/<episode_id>/`:

```
meta.json              Episode metadata (target_object_id, query_object_id, pair_type, description)
captures/sector_{0..5}/
  rgb.png
  depth.png
  mask.png
  gt_bbox.json
```

`meta.json` key fields:

| Field | Meaning |
|---|---|
| `target_object_id` | Object actually placed in the scene |
| `query_object_id`  | Object the description corresponds to (`==target` for positives) |
| `pair_type`        | `positive` / `neg_same` / `neg_diff` |
| `description`      | Fine-grained instance description |
| `sector_layout[i].navigable` | Whether sector `i` is reachable |
| `sector_layout[i].mask_meets_threshold` | False = trap view (navigable but uninformative) |

## How to use

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Avalon-S/PInVerify", repo_type="dataset", local_dir="./data/pv_dataset")
```

For the full evaluation pipeline see [github.com/Avalon-S/PInVerify](https://github.com/Avalon-S/PInVerify):

```bash
python scripts/evaluate.py \
  --config configs/agent/multi_view_attr_llm.yaml \
  dataset.root=./data/pv_dataset
```

## License & attribution

- **Annotations / protocol** (sector graph, pair split, trap-view tags): MIT (this repository)
- **Visual substrate** (HM3D scenes, Objaverse-XL objects, language descriptions): inherited from [PInNED (Barsellotti et al., NeurIPS 2024)](https://arxiv.org/abs/2410.18195); follow PInNED's terms.
- **Citation**: see below.

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

## Contact

Yuhang Jiang — [jyhtjtj@gmail.com](mailto:jyhtjtj@gmail.com) — [avalon-s.github.io](https://avalon-s.github.io/)
