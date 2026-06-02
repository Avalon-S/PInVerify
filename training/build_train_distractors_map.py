#!/usr/bin/env python3
"""
build_train_distractors_map.py
-------------------------------
从原始 PIN 数据集 content_raw 的 .json.gz 文件中提取 distractor 关系。

每个 episode 有:
  - goals[0]: target object (object_id + object_category)
  - distractors: 干扰物列表 (object_id + object_category)

本脚本扫描所有 episode，为每个 target object 收集全部 distractors 并去重。
输出格式与 val 的 object_goal_distractors_map.json 完全一致。

用法:
    python training/build_train_distractors_map.py \
        --src ./data/data/datasets/pin/hm3d/v1/train/yuanben/content_raw \
        --output ./data/pv_dataset/train/object_goal_distractors_map.json
"""

import os
import sys
import json
import gzip
import argparse
from collections import defaultdict


def build_map(src_dir):
    """Extract distractor relationships from PIN content_raw .json.gz files."""

    # Per target: collect unique distractors
    obj_cat_map = {}           # obj_id → category (for all objects seen)
    obj_distractors = defaultdict(set)  # target_id → set of (distractor_id, distractor_cat)

    gz_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".json.gz"))
    print(f"Found {len(gz_files)} .json.gz files")

    n_episodes = 0
    n_skipped = 0

    for i, fname in enumerate(gz_files):
        path = os.path.join(src_dir, fname)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        episodes = data.get("episodes", [])

        for ep in episodes:
            # Target object
            target_id = ep.get("object_id")
            goals = ep.get("goals", [])
            if not goals or not target_id:
                n_skipped += 1
                continue

            target_cat = goals[0].get("object_category")
            if not target_cat:
                n_skipped += 1
                continue

            obj_cat_map[target_id] = target_cat
            n_episodes += 1

            # Distractors
            for d in ep.get("distractors", []):
                d_id = d.get("object_id")
                d_cat = d.get("object_category")
                if d_id and d_cat:
                    obj_cat_map[d_id] = d_cat
                    obj_distractors[target_id].add((d_id, d_cat))

        if (i + 1) % 20 == 0 or (i + 1) == len(gz_files):
            print(f"  [{i+1}/{len(gz_files)}] {n_episodes} episodes processed")

    print(f"\nTotal: {n_episodes} episodes, {n_skipped} skipped")
    print(f"Unique target objects: {len(obj_distractors)}")
    print(f"Unique objects (all): {len(obj_cat_map)}")

    # Build output map (same format as val)
    distractors_map = {}
    for obj_id in sorted(obj_distractors.keys()):
        distractor_set = obj_distractors[obj_id]
        distractors = sorted(
            [{"object_id": d_id, "object_category": d_cat} for d_id, d_cat in distractor_set],
            key=lambda x: x["object_id"],
        )
        distractors_map[obj_id] = {
            "object_category": obj_cat_map[obj_id],
            "distractors": distractors,
        }

    # Stats
    n_same = sum(
        len([d for d in v["distractors"] if d["object_category"] == v["object_category"]])
        for v in distractors_map.values()
    )
    n_diff = sum(
        len([d for d in v["distractors"] if d["object_category"] != v["object_category"]])
        for v in distractors_map.values()
    )
    n_no_same = sum(
        1 for v in distractors_map.values()
        if not any(d["object_category"] == v["object_category"] for d in v["distractors"])
    )

    # Category distribution
    cat_counts = defaultdict(int)
    for v in distractors_map.values():
        cat_counts[v["object_category"]] += 1

    print(f"\nDistractors map: {len(distractors_map)} target objects")
    print(f"  Total same-cat distractors: {n_same}")
    print(f"  Total diff-cat distractors: {n_diff}")
    if n_no_same:
        print(f"  WARNING: {n_no_same} objects have NO same-category distractors")

    print(f"\nCategories ({len(cat_counts)}):")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt} objects")

    return distractors_map


def main():
    parser = argparse.ArgumentParser(
        description="Build distractors map from original PIN content_raw")
    parser.add_argument("--src", required=True,
                        help="Path to content_raw directory with .json.gz files")
    parser.add_argument("--output", required=True,
                        help="Output path for distractors map JSON")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"ERROR: Directory not found: {args.src}")
        sys.exit(1)

    print(f"=== Build Training Distractors Map ===")
    print(f"Source: {args.src}")
    print(f"Output: {args.output}")
    print()

    distractors_map = build_map(args.src)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(distractors_map, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
