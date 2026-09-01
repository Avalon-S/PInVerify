# PInVerify Dataset Guide

PInVerify is derived from [PInNED](https://arxiv.org/abs/2410.18195): 18 object
categories, 71 evaluation instances, 35 HM3D scenes. PInVerify adds the 6-sector
capture topology, ground-truth bounding boxes, trap-view and unreachable-sector
annotations, and the positive / neg_same / neg_diff pair split.

## Download

> Not on Hugging Face yet. The command lands here once the repo is public; see
> the [release status](../README.md#release-status) table.

```bash
huggingface-cli download Avalon-S/PInVerify --repo-type dataset \
  --local-dir ./data/pv_dataset
```

## Directory layout

```
data/pv_dataset/
├── pin_capture/                            Multi-view captures, grouped by scene
│   ├── val/<scene>/<episode>/              evaluation split
│   ├── train_sft/<scene>/<episode>/        SFT pool
│   └── train_rl/<scene>/<episode>/         RL pool
├── image_gt/<category>/                    Ground-truth bounding-box masks
├── val/
│   └── pv_index_{50,100,500,1000,all}.jsonl
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
| `train_sft/pv_train_sft_index.jsonl` | 15,225 pairs | SFT pool |
| `train_rl/pv_train_rl_index.jsonl` | 15,225 pairs | RL pool |

Override at runtime with `dataset.index_file=pv_index_500.jsonl`.

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
