#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove episodes that failed capture from content/.

Reads success_object_ids_by_scene.json, keeps only the successful episodes,
and rewrites content/*.json.gz in place.

Usage:
    python -m pipeline.discard_bad_episodes \
        --success_ids <capture_dir>/success_object_ids_by_scene.json \
        --content_dir <path>
"""

import os
import sys
import json
import gzip
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.config import CONTENT_DIR


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json_gz(path, obj):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def discard_bad_episodes(success_ids_file, content_dir):
    """
    Remove every episode from content_dir that is not in success_ids.
    The .json.gz files are rewritten in place.
    Scene files left with no successful episode are deleted.

    Returns: {"total_before": int, "total_after": int, "removed": int}
    """
    success_oids_by_scene = load_json(success_ids_file)
    success_oids_by_scene = {
        sk: set(map(str, (v or []))) for sk, v in success_oids_by_scene.items()
    }

    total_before = 0
    total_after = 0
    files_removed = 0

    for name in sorted(os.listdir(content_dir)):
        if not name.endswith(".json.gz"):
            continue
        fpath = os.path.join(content_dir, name)
        scene_key = os.path.splitext(os.path.splitext(name)[0])[0]

        try:
            data = read_json_gz(fpath)
        except Exception as e:
            print(f"  [WARN] Failed to read {fpath}: {e}")
            continue

        eps = data.get("episodes", [])
        if not isinstance(eps, list):
            eps = []
        total_before += len(eps)

        success_set = success_oids_by_scene.get(scene_key, set())

        kept = []
        for ep in eps:
            goals = ep.get("goals", [])
            if not goals:
                continue
            # use the episode-level object_id, matching analyze_captures,
            # falling back to goals[0].object_id
            oid = ep.get("object_id") or goals[0].get("object_id", "")
            oid = str(oid) if oid is not None else ""
            if oid in success_set:
                kept.append(ep)

        # episode_id is never renumbered, so the per-episode seed stays stable

        total_after += len(kept)

        if not kept:
            # no successful episode in this scene, drop the file
            os.remove(fpath)
            files_removed += 1
        else:
            out_data = {
                "category_to_mp3d_category_id": data.get("category_to_mp3d_category_id", {}),
                "category_to_task_category_id": data.get("category_to_task_category_id", {}),
                "episodes": kept,
            }
            write_json_gz(fpath, out_data)

    removed = total_before - total_after
    result = {
        "total_before": total_before,
        "total_after": total_after,
        "removed": removed,
        "files_removed": files_removed,
    }

    print(f"\n[discard] Content dir: {content_dir}")
    print(f"[discard] Episodes before: {total_before}")
    print(f"[discard] Episodes after:  {total_after}")
    print(f"[discard] Removed:         {removed}")
    print(f"[discard] Scene files removed: {files_removed}")
    print(f"PIPELINE_RESULT:{json.dumps(result)}")
    return result


def parse_args():
    ap = argparse.ArgumentParser(description="Remove failed episodes from content/")
    ap.add_argument("--success_ids", type=str, required=True,
                    help="Path to success_object_ids_by_scene.json")
    ap.add_argument("--content_dir", type=str, default=CONTENT_DIR)
    return ap.parse_args()


def main():
    args = parse_args()
    discard_bad_episodes(args.success_ids, args.content_dir)


if __name__ == "__main__":
    main()
