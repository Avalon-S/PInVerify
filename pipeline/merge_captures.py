#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental capture helper: merge the previous round's successful episodes into the new capture directory.

Only directories whose meta.json has episode_result.success == True are copied.
Anything already in new_capture_dir, freshly captured this round, is left alone.

On the same filesystem this hardlinks instead of copying, which costs no I/O.

Usage:
    python -m pipeline.merge_captures \
        --prev_dir <FT{N} capture dir> \
        --new_dir  <FT{N+1} capture dir>
"""

import os
import sys
import json
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def is_episode_successful(meta_path: str) -> bool:
    """Read meta.json to decide whether the episode succeeded."""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        episode_result = meta.get("episode_result", {})
        return bool(episode_result.get("success", False))
    except Exception:
        return False


def _hardlink_tree(src: str, dst: str):
    """Clone a directory tree with hardlinks, roughly 100x faster than copytree on one filesystem."""
    os.makedirs(dst, exist_ok=True)
    for entry in os.scandir(src):
        d = os.path.join(dst, entry.name)
        if entry.is_dir(follow_symlinks=False):
            _hardlink_tree(entry.path, d)
        else:
            os.link(entry.path, d)


def _same_filesystem(p1: str, p2: str) -> bool:
    try:
        return os.stat(p1).st_dev == os.stat(p2).st_dev
    except Exception:
        return False


def merge_captures(prev_capture_dir: str, new_capture_dir: str):
    """
    Merge successful episode directories from prev_capture_dir into new_capture_dir.
    Directories already present in new_capture_dir are skipped.
    Non-directory entries, such as the .json stat files, are skipped.
    """
    if not os.path.isdir(prev_capture_dir):
        print(f"[merge] WARN: prev_capture_dir not found: {prev_capture_dir}")
        return 0

    use_hardlink = _same_filesystem(prev_capture_dir, new_capture_dir)
    method = "hardlink" if use_hardlink else "copy"
    print(f"[merge] Using {method} method")

    copied = 0
    skipped_exists = 0
    skipped_fail = 0

    for scene_key in sorted(os.listdir(prev_capture_dir)):
        scene_dir = os.path.join(prev_capture_dir, scene_key)
        if not os.path.isdir(scene_dir):
            continue

        for ep_id in os.listdir(scene_dir):
            ep_dir = os.path.join(scene_dir, ep_id)
            if not os.path.isdir(ep_dir):
                continue

            meta_path = os.path.join(ep_dir, "meta.json")
            if not os.path.exists(meta_path):
                continue

            if not is_episode_successful(meta_path):
                skipped_fail += 1
                continue

            dest = os.path.join(new_capture_dir, scene_key, ep_id)
            if os.path.exists(dest):
                skipped_exists += 1
                continue

            if use_hardlink:
                try:
                    _hardlink_tree(ep_dir, dest)
                except OSError:
                    # fallback (e.g. cross-device)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copytree(ep_dir, dest)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copytree(ep_dir, dest)
            copied += 1

    print(f"\n[merge] Source: {prev_capture_dir}")
    print(f"[merge] Destination: {new_capture_dir}")
    print(f"[merge] Merged: {copied} successful episodes ({method})")
    print(f"[merge] Skipped (already exists): {skipped_exists}")
    print(f"[merge] Skipped (not successful): {skipped_fail}")
    return copied


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge successful episodes from previous capture round")
    ap.add_argument("--prev_dir", type=str, required=True,
                    help="Previous round's capture directory (FT{N})")
    ap.add_argument("--new_dir", type=str, required=True,
                    help="New round's capture directory (FT{N+1})")
    return ap.parse_args()


def main():
    args = parse_args()
    merge_captures(args.prev_dir, args.new_dir)


if __name__ == "__main__":
    main()
