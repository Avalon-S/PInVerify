#!/usr/bin/env python3
"""
Sample two non-overlapping sets of training episodes (SFT + RL) from part directories.

Constraints:
- Each set: 145 scenes x 35 episodes = 5075 episodes
- No overlap between SFT and RL sets
- Same position (0.1m tolerance) OK, but different target object_id required
- Episodes are deduplicated by (round_pos, object_id)

Usage:
    python training/sample_train_episodes.py \
        --parts ./data/part_00 ./data/part_01 ... \
        --dst ./data \
        --per_scene 35

    # Dry run
    python training/sample_train_episodes.py \
        --parts ./data/part_00 ... \
        --dst ./data \
        --per_scene 35 \
        --dry_run
"""

import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path


def round_pos(pos, decimals=1):
    """Round position to merge nearby locations (0.1m tolerance)."""
    return tuple(round(x, decimals) for x in pos)


def load_episode_info(ep_dir):
    """Load episode metadata for deduplication."""
    meta_path = ep_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        goal_pos = meta.get("goal_position_nominal")
        object_id = meta.get("object_id", "unknown")
        if goal_pos is None:
            return None
        return {
            "ep_id": ep_dir.name,
            "ep_path": str(ep_dir),
            "object_id": object_id,
            "goal_pos": goal_pos,
            "pos_key": round_pos(goal_pos),
        }
    except Exception:
        return None


def sample_two_sets(episodes, per_scene, scene_key):
    """
    Sample two non-overlapping sets from episodes for a single scene.

    Strategy:
    1. Deduplicate by (round_pos, object_id)
    2. Shuffle and split into two halves
    3. If not enough for both sets, log warning and take what's available

    Returns: (sft_episodes, rl_episodes)
    """
    need_total = per_scene * 2

    # Deduplicate by (round_pos, object_id)
    seen = set()
    unique_eps = []
    for ep in episodes:
        key = (ep["pos_key"], ep["object_id"])
        if key not in seen:
            seen.add(key)
            unique_eps.append(ep)

    random.shuffle(unique_eps)

    if len(unique_eps) < need_total:
        # Not enough unique episodes, take what we can
        half = len(unique_eps) // 2
        sft = unique_eps[:half]
        rl = unique_eps[half:]
    else:
        # Sample need_total and split
        chosen = unique_eps[:need_total]
        sft = chosen[:per_scene]
        rl = chosen[per_scene:]

    return sft, rl


def main():
    parser = argparse.ArgumentParser(
        description="Sample two non-overlapping training episode sets (SFT + RL)")
    parser.add_argument("--parts", nargs="+", required=True,
                        help="Part directories (e.g. ./data/part_00 ...)")
    parser.add_argument("--dst", required=True,
                        help="Destination root (e.g. ./data)")
    parser.add_argument("--per_scene", type=int, default=35,
                        help="Number of episodes per scene per set (default: 35)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show plan without copying")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. Scan all parts, collect scene -> list of episode info
    scene_episodes = defaultdict(list)

    print(f"Scanning {len(args.parts)} part(s)...")
    for part_dir in sorted(args.parts):
        part_path = Path(part_dir)
        if not part_path.is_dir():
            print(f"  [WARN] Not a directory: {part_dir}")
            continue

        for scene_dir in sorted(part_path.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_key = scene_dir.name

            for ep_dir in sorted(scene_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                info = load_episode_info(ep_dir)
                if info is not None:
                    scene_episodes[scene_key].append(info)

    # 2. Summary
    total_scenes = len(scene_episodes)
    total_episodes = sum(len(eps) for eps in scene_episodes.values())
    print(f"\nFound {total_scenes} scenes, {total_episodes} total episodes")

    # 3. Sample two non-overlapping sets
    sampled_sft = {}
    sampled_rl = {}
    total_sft = 0
    total_rl = 0
    short_scenes = []

    for scene_key in sorted(scene_episodes.keys()):
        eps = scene_episodes[scene_key]
        sft, rl = sample_two_sets(eps, args.per_scene, scene_key)
        sampled_sft[scene_key] = sft
        sampled_rl[scene_key] = rl
        total_sft += len(sft)
        total_rl += len(rl)

        if len(sft) < args.per_scene or len(rl) < args.per_scene:
            short_scenes.append((scene_key, len(sft), len(rl), len(eps)))

    print(f"\nSampling plan:")
    print(f"  Per scene target: {args.per_scene}")
    print(f"  SFT total: {total_sft}")
    print(f"  RL total:  {total_rl}")
    if short_scenes:
        print(f"  Scenes with fewer than {args.per_scene} per set ({len(short_scenes)}):")
        for sk, n_sft, n_rl, n_total in short_scenes:
            print(f"    {sk}: SFT={n_sft}, RL={n_rl} (total unique={n_total})")

    if args.dry_run:
        print(f"\n[DRY RUN] Would copy:")
        print(f"  SFT: {total_sft} episodes -> {args.dst}/train_sft/")
        print(f"  RL:  {total_rl} episodes -> {args.dst}/train_rl/")
        for scene_key in sorted(sampled_sft.keys()):
            n_sft = len(sampled_sft[scene_key])
            n_rl = len(sampled_rl[scene_key])
            print(f"  {scene_key}: SFT={n_sft}, RL={n_rl}")
        return

    # 4. Copy both sets
    for set_name, sampled, total in [("train_sft", sampled_sft, total_sft),
                                      ("train_rl", sampled_rl, total_rl)]:
        dst_root = Path(args.dst) / set_name
        dst_root.mkdir(parents=True, exist_ok=True)

        print(f"\nCopying {total} episodes to {dst_root} ...")
        start_time = time.time()
        copied = 0
        errors = 0

        for scene_key in sorted(sampled.keys()):
            scene_dst = dst_root / scene_key
            scene_dst.mkdir(exist_ok=True)

            for ep in sampled[scene_key]:
                ep_dst_path = scene_dst / ep["ep_id"]
                if ep_dst_path.exists():
                    copied += 1
                    continue  # Resume support

                try:
                    shutil.copytree(ep["ep_path"], str(ep_dst_path))
                    copied += 1
                except Exception as e:
                    errors += 1
                    print(f"  [ERROR] {ep['ep_path']} -> {ep_dst_path}: {e}")

            elapsed = time.time() - start_time
            print(f"  {scene_key}: {len(sampled[scene_key])} eps "
                  f"({copied}/{total}, {elapsed:.0f}s)")

        elapsed = time.time() - start_time
        print(f"\n{set_name}: Copied {copied} episodes in {elapsed:.1f}s (errors: {errors})")

    print("\nDone!")


if __name__ == "__main__":
    main()
