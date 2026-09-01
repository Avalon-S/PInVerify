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
pretty_name: "PInVerify: Active Instance Verification Benchmark"
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

An agent that navigates to a target object does not always arrive at the right instance. Telling "white floral" from "white striped" takes a close look from more than one viewpoint, which is a separate problem from getting there. PInVerify isolates that post-arrival step as a task of its own: a 3,000-episode evaluation benchmark, with training pools for SFT and RL.

## Statistics

| Split | Index file | Pairs | positive : neg_same : neg_diff |
|---|---|---|---|
| `val` (paper) | `val/pv_index_all.jsonl` | 3,000 | 1,000 : 1,000 : 1,000 |
| `val` (full) | `val/pv_index_all_7455.jsonl` | 7,455 | 2,485 : 2,485 : 2,485 |
| `train_sft`  | `train_sft/pv_train_sft_index.jsonl` | 15,225 | 5,075 : 5,075 : 5,075 |
| `train_rl`   | `train_rl/pv_train_rl_index.jsonl` | 15,225 | 5,075 : 5,075 : 5,075 |

Every number in the paper uses the 3,000-pair index, which draws on 1,000 of the
2,485 captured episodes. All 2,485 are here, so `pv_index_all_7455.jsonl` runs as
well; it just costs two and a half times as much compute, which is why the paper
does not report it.

Smaller subsets ship alongside for development: `pv_index_{50,100,500,1000}.jsonl`,
the 50-episode one being the smoke test.

The two training pools were sampled from a larger capture pool that is not
published. Nothing in the paper uses more than what is here. If you need the
larger pool for something, open an issue on the
[GitHub repository](https://github.com/Avalon-S/PInVerify/issues).

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

The test split is stored file by file, so it can be browsed on the Hub. The two
training pools hold 168,000 files between them, well past the Hub's per-repository
recommendation, so each of their scenes is one archive instead:

```
pin_capture/train_sft/<scene_id>.tar
pin_capture/train_rl/<scene_id>.tar
```

Unpacking an archive where it sits reproduces the same
`<scene_id>/<episode_id>/` layout as the test split. The crops directories and
every index file stay loose either way.

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

<details>
<summary><b>One index row, verbatim</b></summary>

A `neg_same` pair: the imagery holds a laptop, the description belongs to a
*different* laptop, so the answer is 0. Three of the six sectors were reachable
and all three show the target.

```json
{
  "episode_path": "pin_capture/val/00800-TEEsavR23oF/25",
  "scene": "00800-TEEsavR23oF",
  "episode": "25",
  "meta_path": "pin_capture/val/00800-TEEsavR23oF/25/meta.json",
  "rgb_dir": "pin_capture/val/00800-TEEsavR23oF/25/rgb",
  "depth_dir": "pin_capture/val/00800-TEEsavR23oF/25/depth",
  "target_object_id": "8dac2731fff9431399c01ee114e5e002",
  "target_object_category": "laptop",
  "query_object_id": "6495988c6c044c76a2fc9f9278543c16",
  "query_object_category": "laptop",
  "pair_type": "neg_same",
  "label": 0,
  "valid_start_sectors": [0, 2, 4],
  "navigable_sectors": [0, 2, 4],
  "n_navigable": 3,
  "n_mask_visible": 3
}
```

`target_object_id` is what is rendered in the scene; `query_object_id` is what
the description refers to. For a `positive` pair they are equal. Keeping them
apart is the entire task, which is why nothing is allowed to overwrite them.

</details>

`meta.json` carries the capture geometry: camera intrinsics, per-viewpoint
`camera_to_world`, sector index and radius, and per-capture
`mask_meets_threshold`. A capture with `navigable: true` but
`mask_meets_threshold: false` is a **trap view**: the agent can go there and
still learn nothing.

<details>
<summary><b>meta.json for the same episode, abridged</b></summary>

Episode level. `n_sectors` is 12 and `sector_order` takes every other slot, which
is where the six sectors come from and why the ids are even numbers:

```json
{
  "scene_key": "00800-TEEsavR23oF",
  "episode_id": "25",
  "object_id": "8dac2731fff9431399c01ee114e5e002",
  "object_category": "laptop",
  "goal_position_nominal": [0.0749, 0.8203, -8.4031],
  "camera_intrinsics": {
    "width": 360, "height": 640,
    "hfov_deg": 42.0, "vfov_deg": 68.62,
    "K": [[468.916, 0, 179.5], [0, 468.916, 319.5], [0, 0, 1]]
  },
  "n_sectors": 12,
  "sector_order": [0, 2, 4, 6, 8, 10],
  "ranges": {"near": [0.9, 1.2], "far": [1.4, 1.7]},
  "category_mask_threshold": 500,
  "sensor_mount_offset_y": 1.31,
  "episode_result": {
    "navigable_count": 6, "valid_mask_count": 5,
    "total_viewpoints": 12, "success": true
  },
  "viewpoints": ["... all 12 candidate poses ..."],
  "captures": ["... the 6 that were reachable ..."]
}
```

One entry of `captures`. `camera_to_world` and `world_to_camera` are 4x4 and
omitted here for length:

```json
{
  "tag": "s0_far",
  "sector_index": 0,
  "angle_deg_range": [0.0, 30.0],
  "range_label": "far",
  "radius_m": 1.5976,
  "navigable": true,
  "in_frustum": true,
  "has_mask": true,
  "mask_area_px": 5218,
  "mask_bbox_xyxy": [107, 186, 253, 293],
  "mask_area_fraction": 0.02265,
  "mask_meets_threshold": true,
  "category_threshold_used": 500,
  "rgb": "rgb/rgb_s0_far.png",
  "mask_raw_path": "mask/mask_s0_far.png",
  "camera_position": [1.4619, 1.4734, -7.6102],
  "camera_rotation_quat_wxyz": [0.8355, -0.2239, 0.4847, 0.1299],
  "camera_fwd_xz": [-0.7518, -0.4298],
  "goal_position": [0.0749, 0.8203, -8.4031],
  "action": "look_down"
}
```

`sector_index` is a world bearing, not a position in the array: slot `s` covers
`[30s, 30s+30)` degrees of `atan2(cz - gz, cx - gx)` around the goal.

The sixth capture of this episode is a **trap view**. It is reachable and the
target is technically in frame, but at 326 pixels it falls under the category
threshold of 500, so the agent can navigate there and learn nothing:

```json
{
  "tag": "s4_far",
  "sector_index": 4,
  "angle_deg_range": [120.0, 150.0],
  "range_label": "far",
  "navigable": true,
  "has_mask": true,
  "mask_area_px": 326,
  "mask_meets_threshold": false
}
```

</details>

Object descriptions are in `object_descriptions_with_category.json`, keyed by
object id. The `*_cache.json` files are precomputed attribute decompositions,
category predictions, and merged descriptions, so a run does not have to
re-query the LLM for them.

## How to use

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Avalon-S/PInVerify", repo_type="dataset", local_dir="./data/pv_dataset")
```

Evaluation needs only the test split, which arrives ready to use. To train, unpack
the two pools first:

```bash
cd ./data/pv_dataset/pin_capture
for pool in train_sft train_rl; do
    (cd $pool && for f in *.tar; do tar -xf "$f" && rm "$f"; done)
done
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

## Contact

Yuhang Jiang, [jyhtjtj@gmail.com](mailto:jyhtjtj@gmail.com), [avalon-s.github.io](https://avalon-s.github.io/)
