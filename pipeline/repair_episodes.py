#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3, run every round: repair the goal placement of failing episodes.
For each failing object_id, pick a position with success_ratio == 1.0 from the
same scene's good_goal_positions and use it as the new goal position.

Incremental mode (--incremental):
  output_dir (content/)     -> writes only the failing episodes, for the incremental capture
  output_dir + "_full"      -> writes the full set: successes untouched, failures repaired

Non-incremental mode:
  output_dir (content/)     -> writes the full set, the older behaviour

Usage:
    python -m pipeline.repair_episodes \
        --success_ids <capture_dir>/success_object_ids_by_scene.json \
        --good_positions <capture_dir>/good_goal_positions.json

    # incremental mode
    python -m pipeline.repair_episodes --incremental \
        --success_ids <capture_dir>/success_object_ids_by_scene.json \
        --good_positions <capture_dir>/good_goal_positions.json
"""

import os
import sys
import re
import json
import gzip
import random
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.config import CONTENT_DIR, CONTENT_VERIFY, REQUIRE_PERFECT

SCENE_KEY_RE = re.compile(r"hm3d/val/([^/]+)/", re.IGNORECASE)


# ============================================================
# IO helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json_gz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_scene_key(scene_id: str) -> str:
    m = SCENE_KEY_RE.search(scene_id.replace("\\", "/"))
    if m:
        return m.group(1)
    parts = scene_id.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return os.path.splitext(os.path.basename(scene_id))[0]


# ============================================================
# Core logic
# ============================================================

def build_good_pos_index(good_pos_raw, require_perfect=REQUIRE_PERFECT):
    """
    Reshape good_goal_positions.json into {scene_key: [[x, y, z], ...]},
    keeping only success_ratio == 1.0 entries when require_perfect is set.
    """
    out = {}
    for scene_key, items in good_pos_raw.items():
        lst = []
        for it in items:
            if not isinstance(it, dict):
                continue
            pos = it.get("goal_position")
            ratio = it.get("success_ratio", 0.0)
            if not isinstance(pos, list) or len(pos) < 3:
                continue
            if require_perfect and float(ratio) != 1.0:
                continue
            lst.append([float(pos[0]), float(pos[1]), float(pos[2])])
        if lst:
            out[scene_key] = lst
    return out


def repair(verify_dir, output_dir, success_ids_file, good_positions_file,
           require_perfect=REQUIRE_PERFECT, incremental=False):
    """
    Repair the failing episodes in content_verify:
    - object_ids that already succeeded are left as they are
    - failing object_ids get their goal position replaced by a known-good one

    With incremental=True:
      output_dir       -> writes only the failing episodes
      output_dir_full  -> writes the full set, kept as a backup

    With incremental=False:
      output_dir       -> writes the full set, the older behaviour

    Returns: {"modified": int, "skipped_no_good_pos": int, "kept": int, "total": int}
    """
    # read the list of successful object_ids
    success_oids_by_scene = load_json(success_ids_file)
    success_oids_by_scene = {
        sk: set(map(str, (v or []))) for sk, v in success_oids_by_scene.items()
    }

    # read the known-good positions
    good_pos_raw = load_json(good_positions_file)
    good_pos_index = build_good_pos_index(good_pos_raw, require_perfect)

    # incremental mode: clear output_dir so it holds only the failing episodes
    if incremental and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    full_output_dir = output_dir + "_full" if incremental else None
    if full_output_dir:
        os.makedirs(full_output_dir, exist_ok=True)

    total_episodes = 0
    episodes_modified = 0
    episodes_skipped_no_goodpos = 0
    episodes_kept = 0
    files_processed = 0

    for name in sorted(os.listdir(verify_dir)):
        if not name.endswith(".json.gz"):
            continue
        in_path = os.path.join(verify_dir, name)

        try:
            data = read_json_gz(in_path)
        except Exception as e:
            print(f"  [WARN] Failed to read {in_path}: {e}")
            continue

        eps = data.get("episodes", [])
        if not isinstance(eps, list):
            eps = []

        bad_eps = []   # failing episodes after repair
        all_eps = []   # the full set: successes plus repaired failures

        for ep in eps:
            total_episodes += 1
            scene_id = ep.get("scene_id", "")
            scene_key = extract_scene_key(scene_id)

            goals = ep.get("goals", [])
            if not goals or not isinstance(goals, list):
                episodes_kept += 1
                all_eps.append(ep)
                continue
            g0 = goals[0]
            # use the episode-level object_id, matching analyze_captures,
            # falling back to goals[0].object_id
            object_id = ep.get("object_id") or g0.get("object_id", None)
            object_id_str = None if object_id is None else str(object_id)

            success_set = success_oids_by_scene.get(scene_key, set())

            if object_id_str in success_set:
                # already successful: keep as is
                episodes_kept += 1
                all_eps.append(ep)
                continue

            # failed: swap in a new position
            good_list = good_pos_index.get(scene_key, [])
            if not good_list:
                episodes_skipped_no_goodpos += 1
                all_eps.append(ep)
                continue

            new_pos = random.choice(good_list)
            g0["position"] = [float(new_pos[0]), float(new_pos[1]), float(new_pos[2])]
            episodes_modified += 1

            bad_eps.append(ep)
            all_eps.append(ep)

        if incremental:
            # incremental mode: output_dir gets only the failing episodes
            if bad_eps:
                bad_data = {
                    "category_to_mp3d_category_id": data.get("category_to_mp3d_category_id", {}),
                    "category_to_task_category_id": data.get("category_to_task_category_id", {}),
                    "episodes": bad_eps,
                }
                write_json_gz(os.path.join(output_dir, name), bad_data)
            # otherwise write nothing: this scene has no failures to recapture

            # the full output always gets everything
            full_data = dict(data)
            full_data["episodes"] = all_eps
            write_json_gz(os.path.join(full_output_dir, name), full_data)
        else:
            # non-incremental mode: output_dir gets everything
            out_data = dict(data)
            out_data["episodes"] = all_eps
            write_json_gz(os.path.join(output_dir, name), out_data)

        files_processed += 1

    result = {
        "total": total_episodes,
        "modified": episodes_modified,
        "skipped_no_good_pos": episodes_skipped_no_goodpos,
        "kept": episodes_kept,
        "incremental": incremental,
    }

    print(f"\n[repair] Files processed: {files_processed}")
    print(f"[repair] Total episodes: {total_episodes}")
    print(f"[repair] Already successful (kept): {episodes_kept}")
    print(f"[repair] Modified (position replaced): {episodes_modified}")
    print(f"[repair] Skipped (no good positions): {episodes_skipped_no_goodpos}")
    if incremental:
        print(f"[repair] Incremental output (bad only): {output_dir}")
        print(f"[repair] Full output (all episodes): {full_output_dir}")
    else:
        print(f"[repair] Output: {output_dir}")

    # emit JSON for the shell driver
    print(f"PIPELINE_RESULT:{json.dumps(result)}")
    return result


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Repair failed episodes by replacing goal positions")
    ap.add_argument("--success_ids", type=str, required=True,
                    help="Path to success_object_ids_by_scene.json")
    ap.add_argument("--good_positions", type=str, required=True,
                    help="Path to good_goal_positions.json")
    ap.add_argument("--verify_dir", type=str, default=CONTENT_VERIFY)
    ap.add_argument("--output_dir", type=str, default=CONTENT_DIR)
    ap.add_argument("--incremental", action="store_true",
                    help="Incremental mode: output_dir gets only bad episodes, "
                         "output_dir_full gets all episodes")
    return ap.parse_args()


def main():
    args = parse_args()
    repair(
        verify_dir=args.verify_dir,
        output_dir=args.output_dir,
        success_ids_file=args.success_ids,
        good_positions_file=args.good_positions,
        incremental=args.incremental,
    )


if __name__ == "__main__":
    main()
