# PInVerify Dataset Guide

PInVerify is derived from [PInNED](https://arxiv.org/abs/2410.18195): 18 object
categories, 71 evaluation instances, 35 HM3D scenes. PInVerify adds the 6-sector
capture topology, ground-truth bounding boxes, trap-view and unreachable-sector
annotations, and the positive / neg_same / neg_diff pair split.

## Download

```bash
huggingface-cli download Avalon-S/PInVerify --repo-type dataset \
  --local-dir ./data/pv_dataset
```

The test split arrives ready to use. The two training pools ship as one archive
per scene, because loose they would be 168,000 files, well past what a single Hub
repository should hold. Unpack them before training:

```bash
cd ./data/pv_dataset/pin_capture
for pool in train_sft train_rl; do
    (cd $pool && for f in *.tar; do tar -xf "$f" && rm "$f"; done)
done
```

Unpacking in place gives the `<scene>/<episode>/` layout below.

## Directory layout

```
data/pv_dataset/
├── pin_capture/                            Multi-view captures, grouped by scene
│   ├── val/<scene>/<episode>/              evaluation split, loose files
│   ├── train_sft/<scene>/<episode>/        SFT pool, shipped as <scene>.tar
│   └── train_rl/<scene>/<episode>/         RL pool, shipped as <scene>.tar
├── image_gt/<category>/                    Ground-truth bounding-box masks
├── val/
│   └── pv_index_{50,100,500,1000,all,all_7455}.jsonl
├── train_sft/
│   ├── pv_train_sft_index.jsonl            15,225 pairs
│   ├── sft_data_v2.jsonl                   Generic-CoT SFT targets (paper main table)
│   ├── sft_data_v3.jsonl                   Specific-CoT SFT targets (appendix)
│   └── crops/, crops_v3/                   pre-cut object crops
├── train_rl/
│   ├── pv_train_rl_index.jsonl             15,225 pairs
│   ├── rl_data_v2.jsonl                    GRPO / GSPO prompts
│   ├── dpo_data_v3.jsonl                   DPO preference pairs
│   └── crops_rl/, crops_dpo/
├── category_cache.json                     object id to coarse category
├── attr_cache.json                         cached attribute decompositions
├── merge_cache.json                        cached merged descriptions
└── object_descriptions_with_category.json  description database
```

The `*_cache.json` files exist so a run does not have to re-query the LLM for
attribute decomposition, category prediction, and description merging.

## Episode structure

Captures are grouped by scene, then by episode:

```
pin_capture/val/<scene_id>/<episode_id>/
  meta.json              Capture geometry and per-viewpoint annotations
  overview.png           Top-down overview of the sector ring
  rgb/rgb_s<N>_<far|near>.png
  mask/mask_s<N>_<far|near>.png
```

`<N>` is the sector angle index and `far` / `near` is the capture ring, so a
6-sector episode has up to 12 viewpoints. Sectors that were not reachable during
capture are simply absent, which is what makes a navigation target unreachable
at evaluation time.

`meta.json` holds the capture side: camera intrinsics, `sector_order`, `ranges`,
`category_mask_threshold`, and a `captures` array. Per capture:

| Field | Meaning |
|---|---|
| `rgb` | Path to the frame, relative to the episode directory |
| `sector_index`, `range_label`, `radius_m` | Where the viewpoint sits on the ring |
| `camera_to_world`, `camera_position` | Full pose |
| `navigable` | Whether the agent can reach this viewpoint |
| `has_mask`, `mask_area_px`, `mask_bbox_xyxy` | Ground-truth target extent in frame |
| `mask_meets_threshold` | False means a **trap view**: reachable, but the target is not usefully visible |

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

## Indices

The verification pair lives in the index, not in `meta.json`. Index files sit
under `val/` and the two training pool directories.

| File | Episodes | Purpose |
|---|---|---|
| `val/pv_index_50.jsonl` | 50 | smoke test |
| `val/pv_index_100.jsonl` | 100 | quick check |
| `val/pv_index_500.jsonl` | 500 | mid-size sweep |
| `val/pv_index_1000.jsonl` | 1,000 | large sweep |
| `val/pv_index_all.jsonl` | 3,000 | **the paper protocol** |
| `val/pv_index_all_7455.jsonl` | 7,455 | every pair over the same captures |
| `train_sft/pv_train_sft_index.jsonl` | 15,225 pairs | SFT pool |
| `train_rl/pv_train_rl_index.jsonl` | 15,225 pairs | RL pool |

Override at runtime with `dataset.index_file=pv_index_500.jsonl`, or through
`run_all.sh`, whose argument is the part after `pv_index_`: `bash run_all.sh
all_7455`.

The 3,000-pair protocol draws on 1,000 of the 2,485 captured episodes; the full
index uses all of them. Both run identically, the full one at roughly two and a
half times the cost, which is why the paper reports the 3,000-pair subset.

The two training pools were sampled from a larger capture pool that is not
published. Nothing in the paper uses more than what ships here; if you need the
larger pool, open an issue.

One index row:

| Field | Meaning |
|---|---|
| `episode_path` | Episode directory, relative to the dataset root |
| `scene`, `episode` | Scene id and episode id |
| `target_object_id` | Object actually placed in the scene, the one in the imagery |
| `query_object_id` | Object the description refers to; equal to target for positives |
| `target_object_category`, `query_object_category` | Their coarse categories |
| `pair_type` | `positive` / `neg_same` / `neg_diff` |
| `label` | 1 if the pair matches, 0 otherwise |
| `navigable_sectors`, `valid_start_sectors` | Reachable sectors, and legal starting sectors |
| `n_navigable`, `n_mask_visible` | How many sectors are reachable, and how many of those show the target |

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

## Pair construction

The evaluation split is a 1:1:1 stratification:

- **positive** (1,000): `query_object_id == target_object_id`
- **neg_same** (1,000): query and target share a category but are different instances
- **neg_diff** (1,000): query and target are in different categories

The split separates calibration bias (neg_same, where fine attributes decide the
answer) from category recognition (neg_diff, which any competent model should
reject). The training pools use the same 1:1:1 ratio, 5,075 pairs each.

## Examples in `data/examples/`

Three small files for sanity checks without the full download:

- `index.jsonl` three real index rows, one of each pair type
- `episode.json` an evaluation **output**, not a dataset file: the transcript,
  per-step tracker state, prediction and label produced by a run
- `episode-clip.json` the same for a CLIP baseline run

The episode paths in these files are relative to `./data/pv_dataset`.

## License & attribution

The visual substrate (HM3D scenes), object pool (Objaverse-XL), and language
descriptions are inherited from PInNED, and use of the dataset is bound by
PInNED's terms. The PInVerify additions (sector annotations, pair split,
evaluation protocol) are released under MIT.
