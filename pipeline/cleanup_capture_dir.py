#!/usr/bin/env python3
"""Phase 2 cleanup: delete episode directories that did not meet the threshold.

Reads failed_episodes.json from analyze_captures and removes each
capture_dir/{scene_key}/{episode_id}/ directory, then prints stats for the shell driver.
"""

import argparse
import json
import os
import shutil


def cleanup_capture_dir(capture_dir: str) -> dict:
    failed_path = os.path.join(capture_dir, "failed_episodes.json")
    if not os.path.isfile(failed_path):
        print(f"[cleanup] No failed_episodes.json found in {capture_dir}")
        return {"removed": 0, "not_found": 0}

    with open(failed_path, "r", encoding="utf-8") as f:
        failed_list = json.load(f)

    if not failed_list:
        print(f"[cleanup] failed_episodes.json is empty — nothing to clean")
        return {"removed": 0, "not_found": 0}

    removed = 0
    not_found = 0

    for item in failed_list:
        scene_key = item.get("scene_key", "")
        episode_id = str(item.get("episode_id", ""))
        ep_dir = os.path.join(capture_dir, scene_key, episode_id)

        if os.path.isdir(ep_dir):
            shutil.rmtree(ep_dir)
            removed += 1
        else:
            not_found += 1

    # drop scene directories left empty
    scenes_removed = 0
    for scene_name in os.listdir(capture_dir):
        scene_path = os.path.join(capture_dir, scene_name)
        if not os.path.isdir(scene_path):
            continue
        # skip non-scene entries such as the json files at this level
        try:
            remaining = os.listdir(scene_path)
        except OSError:
            continue
        if not remaining:
            os.rmdir(scene_path)
            scenes_removed += 1

    print(f"[cleanup] Capture dir: {capture_dir}")
    print(f"[cleanup] Failed episodes in list: {len(failed_list)}")
    print(f"[cleanup] Directories removed: {removed}")
    print(f"[cleanup] Not found (already missing): {not_found}")
    if scenes_removed:
        print(f"[cleanup] Empty scene dirs removed: {scenes_removed}")

    result = {
        "removed": removed,
        "not_found": not_found,
        "total_failed": len(failed_list),
    }
    print(f"PIPELINE_RESULT:{json.dumps(result)}")
    return result


def parse_args():
    ap = argparse.ArgumentParser(
        description="Delete failed episode directories from final capture"
    )
    ap.add_argument("--capture_dir", type=str, required=True,
                    help="Path to the final capture directory")
    return ap.parse_args()


def main():
    args = parse_args()
    cleanup_capture_dir(args.capture_dir)


if __name__ == "__main__":
    main()
