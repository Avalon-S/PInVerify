#!/usr/bin/env python3
"""
Count unique injection positions per scene in the training set.

Usage:
    python scripts/count_train_positions.py ./data/pv_dataset/pin_capture/train
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def round_pos(pos, decimals=1):
    """Round position to merge nearby locations (default 0.1m tolerance)."""
    return tuple(round(x, decimals) for x in pos)


def main():
    parser = argparse.ArgumentParser(description="Count unique injection positions per scene")
    parser.add_argument("train_dir", help="Path to pin_capture/train directory")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    if not train_dir.is_dir():
        print(f"Error: {train_dir} is not a directory")
        return

    # scene_key -> set of unique goal positions
    scene_positions = defaultdict(set)
    # scene_key -> list of (episode_id, object_id, category, goal_pos)
    scene_episodes = defaultdict(list)
    total_episodes = 0
    errors = 0

    for scene_dir in sorted(train_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_key = scene_dir.name

        for ep_dir in sorted(scene_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
            if not ep_dir.is_dir():
                continue
            meta_path = ep_dir / "meta.json"
            if not meta_path.exists():
                continue

            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                goal_pos = meta.get("goal_position_nominal")
                obj_id = meta.get("object_id", "?")
                category = meta.get("object_category", "?")

                if goal_pos:
                    pos_key = round_pos(goal_pos)
                    scene_positions[scene_key].add(pos_key)
                    scene_episodes[scene_key].append({
                        "episode": ep_dir.name,
                        "object_id": obj_id,
                        "category": category,
                        "goal_pos": pos_key,
                    })
                    total_episodes += 1
            except Exception as e:
                errors += 1
                print(f"  [ERROR] {meta_path}: {e}")

    # Print results
    print(f"\n{'='*70}")
    print(f"Training Set Injection Position Statistics")
    print(f"{'='*70}")
    print(f"Total scenes: {len(scene_positions)}")
    print(f"Total episodes: {total_episodes}")
    print(f"Errors: {errors}")
    print()

    total_unique = 0
    print(f"{'Scene':<30} {'Episodes':>10} {'Unique Pos':>12} {'Objects':>10}")
    print(f"{'-'*30} {'-'*10} {'-'*12} {'-'*10}")

    for scene_key in sorted(scene_positions.keys()):
        n_eps = len(scene_episodes[scene_key])
        n_unique = len(scene_positions[scene_key])
        # Count unique object IDs
        obj_ids = set(ep["object_id"] for ep in scene_episodes[scene_key])
        n_objs = len(obj_ids)
        total_unique += n_unique
        print(f"{scene_key:<30} {n_eps:>10} {n_unique:>12} {n_objs:>10}")

    print(f"{'-'*30} {'-'*10} {'-'*12} {'-'*10}")
    print(f"{'TOTAL':<30} {total_episodes:>10} {total_unique:>12}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Avg episodes per scene:  {total_episodes / len(scene_positions):.1f}")
    print(f"  Avg unique pos per scene: {total_unique / len(scene_positions):.1f}")
    print(f"  Avg episodes per position: {total_episodes / total_unique:.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
