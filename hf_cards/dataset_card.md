---
license: mit
task_categories:
  - visual-question-answering
  - image-classification
language:
  - en
size_categories:
  - 10K<n<100K
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

| Split | Index file | Pairs | positive : neg_same : neg_diff |
|---|---|---|---|
| `val` (test) | `val/pv_index_all.jsonl` | 3,000 | 1,000 : 1,000 : 1,000 |
| `train_sft`  | `train_sft/pv_train_sft_index.jsonl` | 15,225 | 5,075 : 5,075 : 5,075 |
| `train_rl`   | `train_rl/pv_train_rl_index.jsonl` | 15,225 | 5,075 : 5,075 : 5,075 |

Smaller evaluation subsets ship alongside the full split:
`pv_index_{50,100,500,1000}.jsonl`. The 50-episode one is the smoke test.

- **18 object categories** (PInNED-inherited)
- **71 unique evaluation instances** across **35 HM3D scenes**
- **6 sectors per episode**, captured on a far and a near ring
- Trap-view + unreachable-sector annotations

## Episode structure

Captures are grouped by scene, then by episode:

```
pin_capture/val/<scene_id>/<episode_id>/
  meta.json              Capture-level metadata (see below)
  overview.png           Top-down overview of the sector ring
  rgb/rgb_s<N>_<far|near>.png
  mask/mask_s<N>_<far|near>.png
```

`<N>` is the sector angle index and `far`/`near` is the capture ring, so a
6-sector episode has up to 12 viewpoints; sectors that were not reachable during
capture are simply absent.

The verification pair (which instance the description belongs to, and whether it
matches) lives in the **index jsonl**, not in `meta.json`. One index row:

| Field | Meaning |
|---|---|
| `episode_path` | Path to the episode directory, relative to the dataset root |
| `target_object_id` | Object actually placed in the scene |
| `query_object_id` | Object the description corresponds to (equal to target for positives) |
| `pair_type` | `positive` / `neg_same` / `neg_diff` |
| `label` | Ground-truth verification answer |
| `navigable_sectors` / `valid_start_sectors` | Sector reachability, and legal starting sectors |
| `n_navigable` / `n_mask_visible` | Reachable sector count, and how many of those actually show the target |

`meta.json` carries the capture geometry: camera intrinsics, per-viewpoint
`camera_to_world`, sector index and radius, and per-capture
`mask_meets_threshold`. A capture with `navigable: true` but
`mask_meets_threshold: false` is a **trap view**: the agent can go there and
still learn nothing.

Object descriptions are in `object_descriptions_with_category.json`, keyed by
object id. The `*_cache.json` files are precomputed attribute decompositions,
category predictions, and merged descriptions, so a run does not have to
re-query the LLM for them.

## How to use

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Avalon-S/PInVerify", repo_type="dataset", local_dir="./data/pv_dataset")
```

For the full evaluation pipeline see [github.com/Avalon-S/PInVerify](https://github.com/Avalon-S/PInVerify):

```bash
python scripts/evaluate.py \
  --config configs/agent/multi_view_attr_adaptive_llm.yaml \
  dataset.root=./data/pv_dataset \
  dataset.index_file=pv_index_50.jsonl
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
