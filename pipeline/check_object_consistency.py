#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic: check object_id consistency across a capture directory.

Checks:
  1. object_id in meta.json matches the one in the dataset JSON
  2. no two object_ids in a scene share a goal_position, which would signal a bad repair swap
  3. distribution of goal_position_mode (runtime vs canonical)

Usage:
    python -m pipeline.check_object_consistency \
        --capture_dir <capture root>/pin_capture_FT9_RGB_v2 \
        --content_dir <data root>/content_full
"""

import os
import sys
import json
import gzip
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def safe_load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        s = f.read()
    for bad in ("Infinity", "-Infinity", "NaN"):
        s = s.replace(bad, "null")
    return json.loads(s)


def load_dataset_index(content_dir):
    """Index (scene_key, episode_id) -> object_id from the content directory."""
    index = {}
    if not content_dir or not os.path.isdir(content_dir):
        return index

    for fname in sorted(os.listdir(content_dir)):
        if not fname.endswith(".json.gz"):
            continue
        scene_key = os.path.splitext(os.path.splitext(fname)[0])[0]
        fpath = os.path.join(content_dir, fname)
        try:
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to read {fpath}: {e}")
            continue

        for ep in data.get("episodes", []):
            ep_id = str(ep.get("episode_id", ""))
            # prefer the episode-level object_id, fall back to goals[0]
            oid = ep.get("object_id")
            if oid is None:
                goals = ep.get("goals", [])
                if goals and isinstance(goals[0], dict):
                    oid = goals[0].get("object_id")
            if oid is not None:
                index[(scene_key, ep_id)] = str(oid)

    return index


def round_pos(pos, nd=3):
    return tuple(round(float(x), nd) for x in pos[:3])


def check_consistency(capture_dir, content_dir=None):
    dataset_index = load_dataset_index(content_dir)
    print(f"[check] Dataset index loaded: {len(dataset_index)} episodes")

    total = 0
    mismatches = []
    position_mode_counts = defaultdict(int)
    # scene -> { rounded_pos -> set of object_ids }
    scene_pos_oids = defaultdict(lambda: defaultdict(set))
    # scene -> { object_id -> set of rounded_pos }
    scene_oid_positions = defaultdict(lambda: defaultdict(set))

    for scene_key in sorted(os.listdir(capture_dir)):
        scene_dir = os.path.join(capture_dir, scene_key)
        if not os.path.isdir(scene_dir):
            continue

        for ep_id in os.listdir(scene_dir):
            ep_dir = os.path.join(scene_dir, ep_id)
            if not os.path.isdir(ep_dir):
                continue
            meta_path = os.path.join(ep_dir, "meta.json")
            if not os.path.exists(meta_path):
                continue

            total += 1
            try:
                meta = safe_load_json(meta_path)
            except Exception:
                continue

            meta_oid = meta.get("object_id")
            meta_cat = meta.get("object_category", "unknown")
            goal_pos = meta.get("goal_position_nominal")

            # check 1: object_id agreement between meta.json and the dataset JSON
            dataset_oid = dataset_index.get((scene_key, str(ep_id)))
            if dataset_oid and meta_oid and str(meta_oid) != str(dataset_oid):
                mismatches.append({
                    "scene_key": scene_key,
                    "episode_id": ep_id,
                    "meta_object_id": meta_oid,
                    "dataset_object_id": dataset_oid,
                    "category": meta_cat,
                })

            # check 2: goal_position_mode distribution
            viewpoints = meta.get("viewpoints", []) or meta.get("captures", [])
            for vp in viewpoints:
                if isinstance(vp, dict):
                    mode = vp.get("goal_position_mode", "unknown")
                    position_mode_counts[mode] += 1

            # check 3: distinct object_ids sharing a goal_position within one scene
            if goal_pos and meta_oid:
                rpos = round_pos(goal_pos)
                scene_pos_oids[scene_key][rpos].add(str(meta_oid))
                scene_oid_positions[scene_key][str(meta_oid)].add(rpos)

    # ---- report ----
    print(f"\n{'='*60}")
    print(f"[check] Capture dir: {capture_dir}")
    print(f"[check] Total episodes checked: {total}")

    # 1) object_id mismatches
    print(f"\n--- object_id mismatches (meta vs dataset) ---")
    if mismatches:
        print(f"  Found {len(mismatches)} mismatches!")
        for m in mismatches[:20]:
            print(f"  {m['scene_key']}/{m['episode_id']}: "
                  f"meta={m['meta_object_id'][:12]}... "
                  f"dataset={m['dataset_object_id'][:12]}... "
                  f"({m['category']})")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
    else:
        print(f"  No mismatches found (meta.json object_id == dataset object_id)")

    # 2) position mode counts
    print(f"\n--- goal_position_mode distribution ---")
    for mode, cnt in sorted(position_mode_counts.items()):
        print(f"  {mode}: {cnt}")
    if position_mode_counts.get("runtime", 0) == 0:
        print(f"  [WARN] ALL viewpoints use canonical mode — "
              f"get_target_object_info() never found the injected object!")

    # 3) distinct object_ids on the same goal_position
    print(f"\n--- shared goal_position across different object_ids ---")
    shared_count = 0
    for scene_key, pos_map in sorted(scene_pos_oids.items()):
        for rpos, oids in pos_map.items():
            if len(oids) > 1:
                shared_count += 1
                if shared_count <= 15:
                    oid_strs = [f"{o[:12]}..." for o in sorted(oids)]
                    print(f"  {scene_key} pos={rpos}: {len(oids)} object_ids: {oid_strs}")
    if shared_count == 0:
        print(f"  No shared positions found")
    else:
        print(f"  Total: {shared_count} positions shared by different object_ids")
        print(f"  (This is expected if repair replaced positions with cross-object good positions)")

    # 4) one object_id placed at several positions in the same scene
    multi_pos_count = 0
    for scene_key, oid_map in scene_oid_positions.items():
        for oid, positions in oid_map.items():
            if len(positions) > 1:
                multi_pos_count += 1
    if multi_pos_count > 0:
        print(f"\n  [INFO] {multi_pos_count} object_ids have multiple different "
              f"goal_positions in the same scene (normal for repaired episodes)")

    print(f"\n{'='*60}")

    return {
        "total": total,
        "mismatches": len(mismatches),
        "position_modes": dict(position_mode_counts),
        "shared_positions": shared_count,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Check object_id consistency in capture results")
    ap.add_argument("--capture_dir", type=str, required=True,
                    help="Capture directory to check (e.g. pin_capture_FT9_RGB_v2)")
    ap.add_argument("--content_dir", type=str, default=None,
                    help="Content directory for cross-reference (content_full or content_verify)")
    return ap.parse_args()


def main():
    args = parse_args()
    check_consistency(args.capture_dir, args.content_dir)


if __name__ == "__main__":
    main()
