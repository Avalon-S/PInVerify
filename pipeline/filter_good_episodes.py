#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1, run once: filter the navigation results into the initial good-episode set.
Supersedes the earlier episode-statistics and good-episode scripts.

Steps:
  1. read the navigation results (results.json / .jsonl)
  2. keep success=1 records that also satisfy 0 <= goal_y - final_y <= 1.6
  3. keep only those episodes from content_verify/, merging in final_position and rotation
  4. write the result to content/ as the initial working set

Usage:
    python -m pipeline.filter_good_episodes --results_file <path>
"""

import os
import sys
import json
import gzip
import math
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.config import (
    CONTENT_DIR, CONTENT_VERIFY,
    HEIGHT_FILTER_MIN, HEIGHT_FILTER_MAX,
)


# ============================================================
# IO helpers
# ============================================================

def load_json_or_jsonl(path):
    """Accepts .json (an array) or .jsonl (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    data = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return data


def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_json_gz(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def scene_key_from_scene_id(scene_id: str) -> str:
    parts = scene_id.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return os.path.splitext(os.path.basename(scene_id))[0]


def get_y(pos):
    if pos is None:
        return None
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            val = float(pos[1])
            return val if math.isfinite(val) else None
        except Exception:
            return None
    if isinstance(pos, dict) and "y" in pos:
        try:
            val = float(pos["y"])
            return val if math.isfinite(val) else None
        except Exception:
            return None
    return None


def to_float_vec3(v):
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        return None
    try:
        return [float(v[0]), float(v[1]), float(v[2])]
    except Exception:
        return None


def to_float_quat(q):
    if isinstance(q, dict):
        for keys in (("w", "x", "y", "z"), ("qw", "qx", "qy", "qz")):
            if all(k in q for k in keys):
                try:
                    return [float(q[k]) for k in keys]
                except Exception:
                    return None
        return None
    if isinstance(q, (list, tuple)) and len(q) >= 4:
        try:
            return [float(q[i]) for i in range(4)]
        except Exception:
            return None
    return None


# ============================================================
# Core logic
# ============================================================

def filter_success_records(records, h_min=HEIGHT_FILTER_MIN, h_max=HEIGHT_FILTER_MAX):
    """
    Select records with success=1 that also satisfy the height constraint.
    Height constraint: h_min <= (goal_y - final_y) <= h_max
    """
    filtered = []
    total = len(records)
    success_count = 0

    for r in records:
        s = r.get("success", 0)
        try:
            succ = int(s) if not isinstance(s, bool) else (1 if s else 0)
        except Exception:
            succ = 0
        if succ != 1:
            continue
        success_count += 1

        fy = get_y(r.get("final_position"))
        gy = get_y(r.get("goal_position"))
        if fy is None or gy is None:
            continue
        height_diff = gy - fy
        if h_min <= height_diff <= h_max:
            filtered.append(r)

    print(f"[filter] Total records: {total}")
    print(f"[filter] Success (raw): {success_count}")
    print(f"[filter] After height filter [{h_min}, {h_max}]: {len(filtered)}")
    return filtered


def build_good_episodes(success_records, verify_dir, output_dir):
    """
    Take the episodes to keep from success_records, read the matching scene
    files out of verify_dir, filter them, merge in final_position and rotation,
    and write to output_dir.
    """
    # build the lookup
    wanted_ids_by_scene = defaultdict(set)
    final_pos_by_scene = defaultdict(dict)
    final_rot_by_scene = defaultdict(dict)

    for r in success_records:
        scene_id = r.get("scene_id")
        ep_id = r.get("episode_id")
        if scene_id is None or ep_id is None:
            continue
        scene_key = scene_key_from_scene_id(scene_id)
        ep_id_str = str(ep_id)
        wanted_ids_by_scene[scene_key].add(ep_id_str)

        fp = to_float_vec3(r.get("final_position"))
        if fp is not None:
            final_pos_by_scene[scene_key][ep_id_str] = fp

        fr = to_float_quat(r.get("final_rotation"))
        if fr is not None:
            final_rot_by_scene[scene_key][ep_id_str] = fr

    os.makedirs(output_dir, exist_ok=True)
    total_in = 0
    total_out = 0
    kept_files = 0

    for scene_key, wanted_ids in wanted_ids_by_scene.items():
        in_file = os.path.join(verify_dir, f"{scene_key}.json.gz")
        out_file = os.path.join(output_dir, f"{scene_key}.json.gz")

        if not os.path.exists(in_file):
            print(f"  [WARN] Missing source for scene '{scene_key}': {in_file}")
            continue

        data = load_json_gz(in_file)
        episodes = data.get("episodes", [])
        total_in += len(episodes)

        kept = []
        for ep in episodes:
            old_id_str = str(ep.get("episode_id"))
            if old_id_str not in wanted_ids:
                continue

            fp = final_pos_by_scene.get(scene_key, {}).get(old_id_str)
            fr = final_rot_by_scene.get(scene_key, {}).get(old_id_str)

            if fp is not None:
                ep.setdefault("info", {})
                ep["info"]["final_position"] = fp
            if fr is not None:
                ep.setdefault("info", {})
                ep["info"]["final_rotation"] = fr

            kept.append(ep)

        # renumber episode_id
        for new_id, ep in enumerate(kept):
            ep["episode_id"] = str(new_id)

        out_data = {
            "category_to_mp3d_category_id": data.get("category_to_mp3d_category_id", {}),
            "category_to_task_category_id": data.get("category_to_task_category_id", {}),
            "episodes": kept,
        }

        save_json_gz(out_data, out_file)
        total_out += len(kept)
        kept_files += 1

    print(f"[filter] Scenes written: {kept_files}")
    print(f"[filter] Episodes in source: {total_in}")
    print(f"[filter] Episodes kept: {total_out}")
    print(f"[filter] Output: {output_dir}")
    return total_out


def filter_and_export(results_file, verify_dir, output_dir):
    """Full pass: read results, filter them, write the good episodes."""
    records = load_json_or_jsonl(results_file)
    success_records = filter_success_records(records)
    return build_good_episodes(success_records, verify_dir, output_dir)


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Filter nav results and build initial good episodes")
    ap.add_argument("--results_file", type=str, required=True,
                    help="Path to navigation evaluation results file (.json or .jsonl)")
    ap.add_argument("--verify_dir", type=str, default=CONTENT_VERIFY)
    ap.add_argument("--output_dir", type=str, default=CONTENT_DIR)
    return ap.parse_args()


def main():
    args = parse_args()
    filter_and_export(args.results_file, args.verify_dir, args.output_dir)


if __name__ == "__main__":
    main()
