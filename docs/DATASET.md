# PInVerify Dataset Guide

The PInVerify dataset is derived from [PInNED](https://arxiv.org/abs/2410.18195): 18 object categories, 71 evaluation instances, 35 HM3D scenes. PInVerify adds the 6-sector capture topology, GT bounding boxes, trap-view annotations, unreachable-sector annotations, and the positive / neg_same / neg_diff pair split.

## Download

```bash
huggingface-cli download Avalon-S/PInVerify-dataset --repo-type dataset --local-dir ./data/pv_dataset
```

The dataset is ~1.4 GB total.

## Directory layout

```
data/pv_dataset/
├── pin_capture/                          Multi-view captures
│   ├── val/                              3,000-episode evaluation split
│   ├── train_sft/                        SFT pool
│   └── train_rl/                         RL pool
├── image_gt/                             Ground-truth bounding-box masks
├── train_sft/                            SFT training jsonl (sft_data_v3.jsonl)
├── train_rl/                             RL training jsonl
├── category_cache.json                   Object → category mapping
├── attr_cache.json                       Cached attribute extractions
├── merge_cache.json                      Cached description merging results
└── object_descriptions_with_category.json  Full description database (~110 KB)
```

## Episode structure

Each episode in `pin_capture/val/<episode_id>/` contains:

```
meta.json              Episode metadata: target instance, sector layout, pair type, query description
captures/              6 sectors × {RGB, depth, mask, GT bbox}
  sector_0/
    rgb.png
    depth.png
    mask.png
    gt_bbox.json
  ...
  sector_5/
```

Key fields in `meta.json`:

| Field | Meaning |
|---|---|
| `target_object_id` | Object that was actually placed in the scene |
| `query_object_id` | Object the language description corresponds to |
| `pair_type` | `positive` (target == query) / `neg_same` (same category) / `neg_diff` (different category) |
| `description` | Fine-grained instance description (3 variants per object) |
| `sector_layout[i].navigable` | Whether sector `i` is reachable |
| `sector_layout[i].mask_meets_threshold` | Whether target is visible from sector `i` (False = trap view) |

## Pair construction

Episodes come in 1:1:1 ratio:

- **positive (1,000)**: `query_object_id == target_object_id`
- **neg_same (1,000)**: query and target share category but are different instances
- **neg_diff (1,000)**: query and target are in different categories

This stratification separates calibration bias (neg_same) from category recognition (neg_diff).

## Indices

PInVerify ships JSONL index files that enumerate episodes for evaluation:

| File | Purpose |
|---|---|
| `pv_index_sectors6_50.jsonl` | 50-episode smoke test (~15 min on 1× RTX 3090) |
| `pv_index_sectors6_500.jsonl` | 500-episode quick eval (~74 GPU-hours all configs) |
| `pv_index_sectors6_3000.jsonl` | Full 3,000-episode protocol used in the paper |

Override at runtime: `dataset.index_file=pv_index_sectors6_500.jsonl`.

## Example episode (in `data/examples/`)

For sanity checks without downloading the full dataset, three tiny example files ship in `data/examples/`:

- `index.jsonl` — single-episode index entry
- `episode.json` — full episode structure
- `episode-clip.json` — CLIP-compatible reformatting

## License & attribution

The visual substrate (HM3D scenes), object pool (Objaverse-XL), and language descriptions are inherited from PInNED. Use of the dataset is bound by PInNED's terms. PInVerify additions (sector annotations, pair split, evaluation protocol) are released under MIT.
