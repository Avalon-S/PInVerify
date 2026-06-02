#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_distractors_map.py
-------------------------------
从 raw episode index 构建 distractor 映射表。
每个 target object 关联同类 (neg_same) 和异类 (neg_diff) distractor。

前置条件:
  已生成 pv_index_raw.jsonl (用 IndexBuilder)

用法:
  python scripts/build_distractors_map.py \
      --index ./data/pv_dataset/train/pv_index_raw.jsonl \
      --output ./data/pv_dataset/train/object_goal_distractors_map.json \
      --seed 42
"""

import os
import sys
import json
import random
import argparse
from collections import defaultdict


def build_map(index_path, max_same=10, max_diff_per_cat=2, seed=42):
    """Build distractors map from raw episode index.

    For each target_object_id:
      - same_category: ALL other objects of the same category
      - diff_category: up to max_diff_per_cat objects per other category
    """
    random.seed(seed)

    # Step 1: Collect unique (object_id, category) pairs
    obj_to_cat = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            obj_id = ep["target_object_id"]
            obj_cat = ep["target_object_category"]
            obj_to_cat[obj_id] = obj_cat

    print(f"Found {len(obj_to_cat)} unique objects")

    # Step 2: Group by category
    cat_to_objs = defaultdict(list)
    for obj_id, cat in obj_to_cat.items():
        cat_to_objs[cat].append(obj_id)

    print(f"Categories ({len(cat_to_objs)}):")
    for cat, objs in sorted(cat_to_objs.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(objs)} objects")

    # Step 3: Build distractors for each object
    distractors_map = {}
    all_cats = sorted(cat_to_objs.keys())

    for obj_id, obj_cat in obj_to_cat.items():
        distractors = []

        # Same category distractors (exclude self)
        same_pool = [oid for oid in cat_to_objs[obj_cat] if oid != obj_id]
        if len(same_pool) > max_same:
            same_selected = random.sample(same_pool, max_same)
        else:
            same_selected = same_pool

        for d_id in same_selected:
            distractors.append({
                "object_id": d_id,
                "object_category": obj_cat,
            })

        # Different category distractors
        for other_cat in all_cats:
            if other_cat == obj_cat:
                continue
            pool = cat_to_objs[other_cat]
            n = min(max_diff_per_cat, len(pool))
            if n > 0:
                selected = random.sample(pool, n)
                for d_id in selected:
                    distractors.append({
                        "object_id": d_id,
                        "object_category": other_cat,
                    })

        distractors_map[obj_id] = {
            "object_category": obj_cat,
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

    print(f"\nDistractors map: {len(distractors_map)} objects")
    print(f"  Total same-cat distractors: {n_same}")
    print(f"  Total diff-cat distractors: {n_diff}")
    if n_no_same:
        print(f"  WARNING: {n_no_same} objects have NO same-category distractors (singleton categories)")

    return distractors_map


def main():
    parser = argparse.ArgumentParser(description="Build distractors map for negative sampling")
    parser.add_argument("--index", required=True,
                        help="Path to raw episode index (pv_index_raw.jsonl)")
    parser.add_argument("--output", required=True,
                        help="Output path for distractors map JSON")
    parser.add_argument("--max-same", type=int, default=10,
                        help="Max same-category distractors per object (default: 10)")
    parser.add_argument("--max-diff-per-cat", type=int, default=2,
                        help="Max diff-category distractors per other category (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(f"ERROR: Index file not found: {args.index}")
        sys.exit(1)

    print(f"=== Build Distractors Map ===")
    print(f"Index: {args.index}")
    print(f"Output: {args.output}")
    print()

    distractors_map = build_map(
        args.index,
        max_same=args.max_same,
        max_diff_per_cat=args.max_diff_per_cat,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(distractors_map, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
