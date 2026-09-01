#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2, run every round: analyze the capture results.
Supersedes the earlier viewpoint-coverage and dataset-result scripts,
adding canonical cross-referencing, a failure list, and per-scene progress.

Outputs, written under capture_dir:
  - good_goal_positions.json
  - success_object_ids_by_scene.json
  - failed_episodes.json
  - round_summary.json

Usage:
    python -m pipeline.analyze_captures --capture_dir <path>
    python -m pipeline.analyze_captures --capture_dir <path> --verify_dir <path>
"""

import os
import sys
import json
import gzip
import argparse
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.config import (
    CONTENT_VERIFY, MIN_NAVIGABLE_VIEWPOINTS, MIN_VALID_MASK_VIEWPOINTS,
    ROUND_DECIMALS,
)


# ============================================================
# IO helpers
# ============================================================

def safe_load_json(p):
    """Read JSON tolerantly, replacing Infinity/NaN with null."""
    with open(p, "r", encoding="utf-8") as f:
        s = f.read()
    for bad in ("Infinity", "-Infinity", "NaN"):
        s = s.replace(bad, "null")
    return json.loads(s)


def round_pos_key(pos, nd=ROUND_DECIMALS):
    x, y, z = pos[:3]
    return (round(float(x), nd), round(float(y), nd), round(float(z), nd))


def describe_counts(arr):
    if not arr:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0}
    a = sorted(arr)
    n = len(a)
    mean = sum(a) / n
    median = a[n // 2] if n % 2 == 1 else (a[n // 2 - 1] + a[n // 2]) / 2
    return {"count": n, "mean": mean, "median": median, "min": a[0], "max": a[-1]}


# ============================================================
# Core logic
# ============================================================

def load_canonical_index(verify_dir):
    """
    Read content_verify/*.json.gz and index the canonical episode information.
    Returns:
      canon_info: {(scene_key, episode_id): {goal_position, object_id, object_category}}
      total_episodes_per_scene: {scene_key: int}
    """
    canon_info = {}
    total_episodes_per_scene = {}

    if not os.path.isdir(verify_dir):
        print(f"[analyze] WARN: verify_dir not found: {verify_dir}")
        return canon_info, total_episodes_per_scene

    for fname in sorted(os.listdir(verify_dir)):
        if not fname.endswith(".json.gz"):
            continue
        scene_key = os.path.splitext(os.path.splitext(fname)[0])[0]
        fpath = os.path.join(verify_dir, fname)

        try:
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to read {fpath}: {e}")
            continue

        eps_list = data.get("episodes", []) or []
        total_episodes_per_scene[scene_key] = len(eps_list)

        for ep in eps_list:
            ep_id = str(ep.get("episode_id"))
            goal_pos = None
            goals = ep.get("goals")
            if isinstance(goals, list) and goals:
                g0 = goals[0]
                if isinstance(g0, dict):
                    pos = g0.get("position")
                    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                        goal_pos = pos

            oid = ep.get("object_id")
            cat = ep.get("object_category") or ep.get("category")

            canon_info[(scene_key, ep_id)] = {
                "goal_position": goal_pos,
                "object_id": oid,
                "object_category": cat,
            }

    return canon_info, total_episodes_per_scene


def analyze(capture_dir, verify_dir=CONTENT_VERIFY):
    """
    Analyze the capture results under capture_dir.

    Returns a dict:
      total_episodes, success_count, fail_count,
      good_positions_file, success_ids_file, failed_episodes_file,
      scene_completion: {scene_key: {done, total, ratio}}
    """
    canon_info, total_episodes_per_scene = load_canonical_index(verify_dir)

    # accumulators
    total_eps = 0
    success_eps = 0
    per_scene = defaultdict(lambda: {"eps": 0, "success": 0})
    per_category = defaultdict(lambda: {"total": 0, "success": 0})
    object_ids_all = set()
    object_ids_success = set()
    navigable_counts_all = []
    valid_mask_counts_all = []
    goal_stats = defaultdict(lambda: defaultdict(lambda: {"pos": None, "total": 0, "success": 0}))
    success_object_ids_by_scene = defaultdict(set)
    failed_eps_list = []
    fail_reason_counts = Counter()

    if not os.path.isdir(capture_dir):
        print(f"[analyze] ERROR: capture_dir not found: {capture_dir}")
        return {
            "total_episodes": 0, "success_count": 0, "fail_count": 0,
            "good_positions_file": None, "success_ids_file": None,
            "failed_episodes_file": None, "scene_completion": {},
        }

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

            total_eps += 1
            per_scene[scene_key]["eps"] += 1

            try:
                meta = safe_load_json(meta_path)
            except Exception:
                continue

            # cross-reference against the canonical episode
            meta_ep_id = str(meta.get("episode_id", ep_id))
            cinfo = canon_info.get((scene_key, meta_ep_id), {})
            oid = cinfo.get("object_id") or meta.get("object_id")
            cat = cinfo.get("object_category") or meta.get("object_category") or "unknown"

            per_category[cat]["total"] += 1
            if oid is not None:
                object_ids_all.add(str(oid))

            # ---- success test ----
            episode_result = meta.get("episode_result")
            if episode_result is not None:
                # new format: read episode_result, already computed by the capture script
                is_success = bool(episode_result.get("success", False))
                nav_count = int(episode_result.get("navigable_count", 0))
                mask_count = int(episode_result.get("valid_mask_count", 0))
            else:
                # old format: fall back to capture count and sector coverage
                caps = meta.get("captures", []) or []
                if not isinstance(caps, list):
                    caps = []
                num_caps = len(caps)
                sector_ids = set()
                for c in caps:
                    si = c.get("sector_index")
                    if isinstance(si, int):
                        sector_ids.add(si)
                covered = len(sector_ids)
                is_success = (num_caps >= 8) and (covered >= 4)
                nav_count = num_caps
                mask_count = covered  # approximation for the old format

            navigable_counts_all.append(nav_count)
            valid_mask_counts_all.append(mask_count)

            if is_success:
                success_eps += 1
                per_scene[scene_key]["success"] += 1
                per_category[cat]["success"] += 1
                if oid is not None:
                    object_ids_success.add(str(oid))
                    success_object_ids_by_scene[scene_key].add(str(oid))
            else:
                reason = []
                if episode_result is not None:
                    if nav_count < MIN_NAVIGABLE_VIEWPOINTS:
                        reason.append(f"navigable_lt_{MIN_NAVIGABLE_VIEWPOINTS}")
                    if mask_count < MIN_VALID_MASK_VIEWPOINTS:
                        reason.append(f"valid_mask_lt_{MIN_VALID_MASK_VIEWPOINTS}")
                else:
                    if nav_count < 8:
                        reason.append("num_lt_8")
                    if mask_count < 4:
                        reason.append("covered_lt_4")

                if len(reason) == 2:
                    fail_reason_counts["both"] += 1
                elif reason:
                    fail_reason_counts[reason[0]] += 1

                failed_eps_list.append({
                    "scene_key": scene_key,
                    "episode_id": meta_ep_id,
                    "object_id": oid,
                    "navigable_count": nav_count,
                    "valid_mask_count": mask_count,
                    "fail_reasons": reason,
                })

            # position statistics
            goal_pos = meta.get("goal_position_nominal") or meta.get("goal_position")
            if (not isinstance(goal_pos, (list, tuple))) or (len(goal_pos) < 3):
                caps = meta.get("captures", []) or []
                if caps and isinstance(caps[0], dict):
                    cand = caps[0].get("goal_position")
                    if isinstance(cand, (list, tuple)) and len(cand) >= 3:
                        goal_pos = cand
            if (not isinstance(goal_pos, (list, tuple))) or (len(goal_pos) < 3):
                goal_pos = cinfo.get("goal_position")

            if isinstance(goal_pos, (list, tuple)) and len(goal_pos) >= 3:
                rounded_key = round_pos_key(goal_pos)
                st = goal_stats[scene_key][rounded_key]
                if st["pos"] is None:
                    st["pos"] = [float(rounded_key[0]), float(rounded_key[1]), float(rounded_key[2])]
                st["total"] += 1
                if is_success:
                    st["success"] += 1

    # ---- report ----
    overall_ratio = (success_eps / total_eps) if total_eps else 0.0
    fail_count = total_eps - success_eps

    print(f"\n{'='*60}")
    print(f"[analyze] Capture dir: {capture_dir}")
    print(f"[analyze] Episode total: {total_eps}")
    print(f"[analyze] Success (nav>={MIN_NAVIGABLE_VIEWPOINTS}, "
          f"mask>={MIN_VALID_MASK_VIEWPOINTS}): {success_eps}")
    print(f"[analyze] Failed: {fail_count}")
    print(f"[analyze] Success ratio: {overall_ratio:.2%}")
    print(f"[analyze] Unique object_ids: all={len(object_ids_all)}, "
          f"success={len(object_ids_success)}")

    # success rate per scene, top 10
    rank_scene = sorted(
        per_scene.items(),
        key=lambda kv: (kv[1]["success"] / kv[1]["eps"] if kv[1]["eps"] else 0),
    )
    print(f"\n  Bottom-10 scenes by success ratio:")
    for sc, st in rank_scene[:10]:
        eps, suc = st["eps"], st["success"]
        ratio = (suc / eps) if eps else 0.0
        print(f"    {sc}: {suc}/{eps} = {ratio:.2%}")

    # success rate per category
    rank_cat = sorted(
        per_category.items(),
        key=lambda kv: (kv[1]["success"] / kv[1]["total"] if kv[1]["total"] else 0),
        reverse=True,
    )
    print(f"\n  Top-15 categories by success ratio:")
    for c_name, st in rank_cat[:15]:
        t, s = st["total"], st["success"]
        print(f"    {c_name:<20} {s}/{t} = {(s/t if t else 0):.2%}")

    # navigable and mask coverage
    desc_nav = describe_counts(navigable_counts_all)
    print(f"\n  Navigable viewpoints (all): N={desc_nav['count']}, "
          f"mean={desc_nav['mean']:.2f}, median={desc_nav['median']:.2f}, "
          f"min={desc_nav['min']}, max={desc_nav['max']}")

    desc_mask = describe_counts(valid_mask_counts_all)
    print(f"  Valid mask viewpoints (all): N={desc_mask['count']}, "
          f"mean={desc_mask['mean']:.2f}, median={desc_mask['median']:.2f}, "
          f"min={desc_mask['min']}, max={desc_mask['max']}")

    # failure reasons
    if fail_reason_counts:
        print(f"\n  Failure reasons: {dict(fail_reason_counts)}")

    # ---- per-scene progress ----
    scene_completion = {}
    all_scene_keys = sorted(set(total_episodes_per_scene.keys()) | set(per_scene.keys()))
    incomplete_count = 0
    for sc_key in all_scene_keys:
        total_required = total_episodes_per_scene.get(sc_key, 0)
        done = per_scene.get(sc_key, {}).get("eps", 0)
        ratio = (done / total_required) if total_required else 0.0
        scene_completion[sc_key] = {"done": done, "total": total_required, "ratio": ratio}
        if total_required and done < total_required:
            incomplete_count += 1

    if incomplete_count > 0:
        print(f"\n  Incomplete scenes: {incomplete_count}")
    else:
        print(f"\n  All scenes complete.")

    # ---- write outputs ----
    # 1) good_goal_positions.json
    export_positions = {}
    for sc_key, pos_map in goal_stats.items():
        items = []
        for rounded_key, st in pos_map.items():
            ratio = (st["success"] / st["total"]) if st["total"] else 0.0
            items.append({
                "goal_position": st["pos"],
                "count": st["total"],
                "success": st["success"],
                "success_ratio": round(ratio, 6),
            })
        if items:
            items.sort(key=lambda d: (d["success_ratio"], d["success"], d["count"]), reverse=True)
            export_positions[sc_key] = items

    good_pos_out = os.path.join(capture_dir, "good_goal_positions.json")
    with open(good_pos_out, "w", encoding="utf-8") as f:
        json.dump(export_positions, f, ensure_ascii=False, indent=2)
    print(f"\n  Exported: {good_pos_out} ({sum(len(v) for v in export_positions.values())} positions)")

    # 2) success_object_ids_by_scene.json
    export_oids = {
        sc_key: sorted(list(s))
        for sc_key, s in success_object_ids_by_scene.items()
        if s
    }
    success_ids_out = os.path.join(capture_dir, "success_object_ids_by_scene.json")
    with open(success_ids_out, "w", encoding="utf-8") as f:
        json.dump(export_oids, f, ensure_ascii=False, indent=2)
    print(f"  Exported: {success_ids_out} ({len(export_oids)} scenes)")

    # 3) failed_episodes.json
    failed_out = os.path.join(capture_dir, "failed_episodes.json")
    with open(failed_out, "w", encoding="utf-8") as f:
        json.dump(failed_eps_list, f, ensure_ascii=False, indent=2)
    print(f"  Exported: {failed_out} ({len(failed_eps_list)} failed episodes)")

    # 4) round_summary.json
    summary = {
        "timestamp": datetime.now().isoformat(),
        "capture_dir": capture_dir,
        "total_episodes": total_eps,
        "success_count": success_eps,
        "fail_count": fail_count,
        "success_ratio": round(overall_ratio, 6),
        "unique_object_ids_all": len(object_ids_all),
        "unique_object_ids_success": len(object_ids_success),
        "failure_reasons": dict(fail_reason_counts),
        "per_category_summary": {
            cat: {"total": st["total"], "success": st["success"]}
            for cat, st in per_category.items()
        },
        "incomplete_scenes": incomplete_count,
    }
    summary_path = os.path.join(capture_dir, "round_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Exported: {summary_path}")

    print(f"{'='*60}\n")

    return {
        "total_episodes": total_eps,
        "success_count": success_eps,
        "fail_count": fail_count,
        "good_positions_file": good_pos_out,
        "success_ids_file": success_ids_out,
        "failed_episodes_file": failed_out,
        "scene_completion": scene_completion,
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Analyze capture results and export statistics")
    ap.add_argument("--capture_dir", type=str, required=True,
                    help="Root of capture output (scene_key/episode_id/meta.json)")
    ap.add_argument("--verify_dir", type=str, default=CONTENT_VERIFY,
                    help="Path to content_verify for canonical cross-reference")
    return ap.parse_args()


def main():
    args = parse_args()
    result = analyze(args.capture_dir, args.verify_dir)
    # emit the summary as JSON so the shell driver can parse it
    summary = {
        "total_episodes": result["total_episodes"],
        "success_count": result["success_count"],
        "fail_count": result["fail_count"],
    }
    print(f"PIPELINE_RESULT:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
