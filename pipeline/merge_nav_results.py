#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the per-worker navigation jsonl files into one and print a summary.

Usage:
    python -m pipeline.merge_nav_results \
        --input_dir results/pin/val/pipeline_nav_eval \
        --output_file results/pin/val/pipeline_nav_eval/all_results.jsonl
"""

import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.config import RESULTS_DIR, NAV_EVAL_EXP_NAME


def merge_jsonl_files(input_dir: str, output_file: str) -> dict:
    """
    Scan input_dir for *_results.jsonl, merge them, and write output_file.
    Returns the summary.
    """
    jsonl_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith("_results.jsonl") and f != os.path.basename(output_file)
    ])

    if not jsonl_files:
        print(f"[merge_nav] ERROR: No *_results.jsonl files found in {input_dir}")
        return {"total": 0, "success": 0, "files_merged": 0}

    all_records = []
    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # write the merged file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # aggregate
    total = len(all_records)
    success_count = sum(1 for r in all_records if _is_success(r))
    has_final_pos = sum(1 for r in all_records if r.get("final_position") is not None)

    # per-scene counts
    scene_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in all_records:
        scene_id = r.get("scene_id", "unknown")
        # pull out scene_key
        parts = scene_id.replace("\\", "/").split("/")
        scene_key = parts[-2] if len(parts) >= 2 else scene_id
        scene_stats[scene_key]["total"] += 1
        if _is_success(r):
            scene_stats[scene_key]["success"] += 1

    summary = {
        "total_episodes": total,
        "success_count": success_count,
        "success_rate": round(success_count / max(total, 1), 4),
        "has_final_position": has_final_pos,
        "missing_final_position": total - has_final_pos,
        "files_merged": len(jsonl_files),
        "scenes": len(scene_stats),
        "per_scene": {k: v for k, v in sorted(scene_stats.items())},
    }

    # write the summary
    summary_path = os.path.join(input_dir, "nav_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # print
    print(f"\n[merge_nav] Input dir: {input_dir}")
    print(f"[merge_nav] Files merged: {len(jsonl_files)}")
    for fp in jsonl_files:
        print(f"  - {os.path.basename(fp)}")
    print(f"[merge_nav] Total episodes: {total}")
    print(f"[merge_nav] Success: {success_count} ({summary['success_rate']*100:.1f}%)")
    print(f"[merge_nav] Has final_position: {has_final_pos}")
    if has_final_pos < total:
        print(f"[merge_nav] WARN: {total - has_final_pos} episodes missing final_position")
    print(f"[merge_nav] Scenes: {len(scene_stats)}")
    print(f"[merge_nav] Output: {output_file}")
    print(f"[merge_nav] Summary: {summary_path}")

    # for the shell driver to parse
    print(f"PIPELINE_RESULT:{json.dumps(summary)}")
    return summary


def _is_success(record: dict) -> bool:
    s = record.get("success", 0)
    if isinstance(s, bool):
        return s
    try:
        return int(s) == 1
    except (ValueError, TypeError):
        return False


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge navigation evaluation results from multiple workers")
    default_dir = os.path.join(RESULTS_DIR, NAV_EVAL_EXP_NAME)
    ap.add_argument("--input_dir", type=str, default=default_dir,
                    help="Directory containing worker *_results.jsonl files")
    ap.add_argument("--output_file", type=str, default=None,
                    help="Output merged jsonl path (default: input_dir/all_results.jsonl)")
    return ap.parse_args()


def main():
    args = parse_args()
    output = args.output_file or os.path.join(args.input_dir, "all_results.jsonl")
    merge_jsonl_files(args.input_dir, output)


if __name__ == "__main__":
    main()
