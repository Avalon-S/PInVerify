#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0: build the verification dataset (content_verify).
Flattens the original content/*.json.gz to one episode per object, dropping distractors,
and writes content_verify/*.json.gz.

Usage:
    python -m pipeline.init_verify_dataset
    python -m pipeline.init_verify_dataset --input_dir <path> --output_dir <path>
"""

import os
import sys
import json
import gzip
import math
import random
import argparse

# allow running straight from the repository root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.config import CONTENT_DIR, CONTENT_VERIFY, MIN_START_DIST


# ============================================================
# IO helpers
# ============================================================

def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_json_gz(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def euclidean_distance(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


# ============================================================
# Core logic
# ============================================================

def process_one_file(data):
    """
    Flatten one scene file:
    - every object becomes the goal once, whether it was a goal or a distractor
    - no distractors are injected (distractors=[])
    - pick a start at least MIN_START_DIST away from the goal
    """
    episodes = data.get("episodes", [])
    if not episodes:
        return {
            "category_to_mp3d_category_id": data.get("category_to_mp3d_category_id", {}),
            "category_to_task_category_id": data.get("category_to_task_category_id", {}),
            "episodes": [],
        }

    # gather every instance in this scene
    objs_in_scene = {}
    for ep in episodes:
        for o in ep.get("goals", []):
            objs_in_scene[o["object_id"]] = {
                "object_category": o["object_category"],
                "position": o["position"],
            }
        for o in ep.get("distractors", []):
            objs_in_scene[o["object_id"]] = {
                "object_category": o["object_category"],
                "position": o["position"],
            }

    # gather the existing start_position / start_rotation pairs
    start_pool = [(ep["start_position"], ep["start_rotation"]) for ep in episodes]
    if not start_pool:
        start_pool = [([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])]

    scene_id = episodes[0]["scene_id"]
    scene_dataset_config = episodes[0]["scene_dataset_config"]

    new_eps = []
    eid = 0
    for goal_id, meta in objs_in_scene.items():
        cat = meta["object_category"]
        gpos = meta["position"]

        candidates = [(sp, sr) for sp, sr in start_pool
                       if euclidean_distance(sp, gpos) >= MIN_START_DIST]
        if candidates:
            start_position, start_rotation = random.choice(candidates)
        else:
            start_position, start_rotation = start_pool[0]

        ep = {
            "episode_id": str(eid),
            "scene_id": scene_id,
            "start_position": start_position,
            "start_rotation": start_rotation,
            "info": {"geodesic_distance": None},
            "start_room": None,
            "shortest_paths": [[]],
            "goals": [{
                "object_category": cat,
                "object_id": goal_id,
                "position": gpos,
                "radius": None,
                "room_id": None,
                "room_name": None,
            }],
            "distractors": [],
            "scene_dataset_config": scene_dataset_config,
            "object_category": cat,
            "object_id": goal_id,
        }
        new_eps.append(ep)
        eid += 1

    return {
        "category_to_mp3d_category_id": data.get("category_to_mp3d_category_id", {}),
        "category_to_task_category_id": data.get("category_to_task_category_id", {}),
        "episodes": new_eps,
    }


def build_verify_dataset(input_dir: str, output_dir: str):
    """Flatten every .json.gz under input_dir into output_dir."""
    random.seed(0)
    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.endswith(".json.gz")]
    print(f"[init_verify] Found {len(input_files)} scene files in {input_dir}")

    total_objects = 0
    for fname in sorted(input_files):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        data = load_json_gz(in_path)
        processed = process_one_file(data)
        save_json_gz(processed, out_path)
        n = len(processed["episodes"])
        total_objects += n
        print(f"  {fname}: {n} episodes")

    print(f"[init_verify] Done. Total episodes: {total_objects}")
    print(f"[init_verify] Output: {output_dir}")
    return total_objects


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Build content_verify from original content")
    ap.add_argument("--input_dir", type=str, default=CONTENT_DIR)
    ap.add_argument("--output_dir", type=str, default=CONTENT_VERIFY)
    return ap.parse_args()


def main():
    args = parse_args()
    build_verify_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
