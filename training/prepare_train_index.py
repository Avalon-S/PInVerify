#!/usr/bin/env python3
"""
Generate pair-based training index (positive + neg_same + neg_diff) for SFT/RL sets.

Reuses pver/data/builder.py (NegativeSampler) and
scripts/build_distractors_map.py (build_map) for auto-building distractors.

Usage:
    # Auto-build distractors map from training episodes + generate pairs (recommended)
    python training/prepare_train_index.py \
        --data-root ./data/pv_dataset \
        --train-dir ./data/pv_dataset/pin_capture/train_sft \
        --output ./data/pv_dataset/train_sft/pv_train_sft_index.jsonl

    # With explicit distractors map
    python training/prepare_train_index.py \
        --data-root ./data/pv_dataset \
        --train-dir ./data/pv_dataset/pin_capture/train_sft \
        --distractors-map /path/to/object_goal_distractors_map.json \
        --output ./data/pv_dataset/train_sft/pv_train_sft_index.jsonl

    # Just build raw index (no pairing)
    python training/prepare_train_index.py \
        --data-root ./data/pv_dataset \
        --train-dir ./data/pv_dataset/pin_capture/train_sft \
        --save-raw ./data/pv_dataset/train_sft/pv_index_raw.jsonl \
        --raw-only
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pver.data.builder import NegativeSampler


def build_raw_index(train_dir, data_root=None):
    """
    Scan train_dir/<scene>/<episode>/meta.json and build raw episode index.

    Unlike builder.py's IndexBuilder (which assumes pin_capture/<split>/ layout),
    this directly scans a flat <scene>/<episode>/ structure.
    """
    train_path = Path(train_dir)
    episodes = []
    stats = {"total": 0, "skipped_no_valid_start": 0, "errors": 0}

    for scene_dir in sorted(train_path.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_key = scene_dir.name

        for ep_dir in sorted(scene_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            meta_path = ep_dir / "meta.json"
            if not meta_path.exists():
                continue

            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                object_id = meta.get("object_id")
                object_cat = meta.get("object_category")
                episode_id = str(meta.get("episode_id", ep_dir.name))

                # episode_path relative to data_root (if provided) or absolute
                if data_root:
                    try:
                        episode_path = str(ep_dir.relative_to(data_root)).replace("\\", "/")
                    except ValueError:
                        episode_path = str(ep_dir).replace("\\", "/")
                else:
                    episode_path = str(ep_dir).replace("\\", "/")

                # Parse sector info from viewpoints/captures
                viewpoints = meta.get("viewpoints") or meta.get("captures", [])
                valid_start_sectors = set()
                navigable_sectors = set()

                for vp in viewpoints:
                    tag = vp.get("tag", "")
                    # Parse sector index from tag
                    sector_idx = vp.get("sector_index", -1)
                    if sector_idx < 0 and tag.startswith("s"):
                        try:
                            sector_idx = int(tag.split("_")[0][1:])
                        except (ValueError, IndexError):
                            sector_idx = -1

                    navigable = vp.get("navigable", False)
                    has_mask = vp.get("mask_meets_threshold", False)
                    rgb = vp.get("rgb") or vp.get("rgb_filename")

                    if navigable and rgb:
                        navigable_sectors.add(sector_idx)
                    if has_mask:
                        valid_start_sectors.add(sector_idx)

                if not valid_start_sectors:
                    stats["skipped_no_valid_start"] += 1
                    continue

                episode_entry = {
                    "episode_path": episode_path,
                    "scene": scene_key,
                    "episode": episode_id,
                    "meta_path": episode_path + "/meta.json",
                    "rgb_dir": episode_path + "/rgb",
                    "target_object_id": object_id,
                    "target_object_category": object_cat,
                    "valid_start_sectors": sorted(valid_start_sectors),
                    "navigable_sectors": sorted(navigable_sectors),
                    "n_navigable": len(navigable_sectors),
                    "n_mask_visible": len(valid_start_sectors),
                }

                episodes.append(episode_entry)
                stats["total"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"  [ERROR] {meta_path}: {e}")

    return episodes, stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate pair-based training index from sampled episodes")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root for relative paths (optional)")
    parser.add_argument("--train-dir", required=True,
                        help="Training episode directory (e.g. ./data/train_sft)")
    parser.add_argument("--distractors-map", required=True,
                        help="Path to object_goal_distractors_map.json")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Step 1: Build raw episode index
    print(f"Scanning {args.train_dir} ...")
    episodes, stats = build_raw_index(args.train_dir, args.data_root)
    print(f"  Found {stats['total']} valid episodes")
    if stats["skipped_no_valid_start"]:
        print(f"  Skipped {stats['skipped_no_valid_start']} (no valid start sector)")
    if stats["errors"]:
        print(f"  Errors: {stats['errors']}")

    # Save temporary raw index
    tmp_path = args.output + ".raw.tmp"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    # Step 2: Generate pairs using NegativeSampler
    print(f"\nGenerating pairs...")
    sampler = NegativeSampler(args.distractors_map)
    sampler.augment_index(tmp_path, args.output)

    # Clean up temp file
    os.remove(tmp_path)

    # Report
    with open(args.output, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    n_total = len(lines)
    from collections import Counter
    pair_types = Counter()
    for line in lines:
        ep = json.loads(line)
        pair_types[ep.get("pair_type", "unknown")] += 1

    print(f"\nSaved {n_total} pairs to {args.output}")
    print(f"  Distribution: {dict(pair_types)}")


if __name__ == "__main__":
    main()
