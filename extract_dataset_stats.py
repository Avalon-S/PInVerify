#!/usr/bin/env python3
"""
Extract dataset statistics from captured PIN data.
Supports single/multiple dirs, and offline merge of per-part JSONs.

Usage:
    # single split
    python extract_dataset_stats.py --data_dir <capture root>/val --output val_stats.json

    # two parts on the same server
    python extract_dataset_stats.py \
        --data_dir <capture root>/part_00 <capture root>/part_01 \
        --output parts_00_01_stats.json

    # merge previously saved JSONs on local machine (zero precision loss)
    python extract_dataset_stats.py \
        --merge parts_00_01_stats.json parts_02_03_stats.json ... \
        --output train_stats.json
"""

import os
import json
import argparse
import math
from collections import defaultdict


# =====================================================================
#  Core: scan directories and collect raw per-episode numbers
# =====================================================================

def collect_raw(data_dirs):
    """Scan capture directories and return raw per-episode data lists."""
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    # per-episode scalars
    navigable_counts = []       # navigable viewpoints (out of 12)
    valid_mask_counts = []      # mask-passing viewpoints (out of 12)
    navigable_sectors = []      # navigable sectors (out of 6)
    visible_sectors = []        # visible sectors (out of 6)
    trap_sectors = []           # trap sectors (out of 6)
    far_navigable = []          # far navigable viewpoints (out of 6)
    near_navigable = []         # near navigable viewpoints (out of 6)
    far_visible = []            # far visible viewpoints (out of 6)
    near_visible = []           # near visible viewpoints (out of 6)

    # per-viewpoint mask areas
    mask_areas_all = []
    mask_areas_far = []
    mask_areas_near = []

    # per-episode category tag
    episode_categories = []

    # per-episode object_id (for unique counting)
    category_objects = defaultdict(set)

    # per-scene episode count
    scene_episode_counts = {}

    for data_dir in data_dirs:
        scenes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        for scene_key in scenes:
            scene_dir = os.path.join(data_dir, scene_key)
            ep_dirs = sorted([
                d for d in os.listdir(scene_dir)
                if os.path.isdir(os.path.join(scene_dir, d))
            ])

            scene_ep_count = 0
            for ep_id in ep_dirs:
                meta_path = os.path.join(scene_dir, ep_id, "meta.json")
                if not os.path.isfile(meta_path):
                    continue

                with open(meta_path, "r") as f:
                    meta = json.load(f)

                scene_ep_count += 1

                # category
                cat = meta.get("object_category", "unknown")
                episode_categories.append(cat)
                oid = meta.get("object_id", "")
                if oid:
                    category_objects[cat].add(oid)

                # episode_result
                ep_result = meta.get("episode_result", {})
                navigable_counts.append(ep_result.get("navigable_count", 0))
                valid_mask_counts.append(ep_result.get("valid_mask_count", 0))

                # viewpoint-level analysis
                viewpoints = meta.get("viewpoints", [])
                sectors_map = defaultdict(list)
                ep_fn, ep_nn, ep_fv, ep_nv = 0, 0, 0, 0

                for vp in viewpoints:
                    sid = vp.get("sector_index", -1)
                    sectors_map[sid].append(vp)

                    is_nav = vp.get("navigable", False)
                    is_vis = is_nav and vp.get("mask_meets_threshold", False)
                    rlabel = vp.get("range_label", "")

                    if rlabel == "far":
                        ep_fn += int(is_nav)
                        ep_fv += int(is_vis)
                    elif rlabel == "near":
                        ep_nn += int(is_nav)
                        ep_nv += int(is_vis)

                    if vp.get("has_mask", False) and is_nav:
                        area = vp.get("mask_area_px", 0)
                        mask_areas_all.append(area)
                        if rlabel == "far":
                            mask_areas_far.append(area)
                        elif rlabel == "near":
                            mask_areas_near.append(area)

                far_navigable.append(ep_fn)
                near_navigable.append(ep_nn)
                far_visible.append(ep_fv)
                near_visible.append(ep_nv)

                # sector-level
                n_nav_sec, n_vis_sec, n_trap_sec = 0, 0, 0
                for sid, vps in sectors_map.items():
                    any_nav = any(v.get("navigable", False) for v in vps)
                    any_vis = any(
                        v.get("navigable", False) and v.get("mask_meets_threshold", False)
                        for v in vps
                    )
                    if any_nav:
                        n_nav_sec += 1
                        if any_vis:
                            n_vis_sec += 1
                        else:
                            n_trap_sec += 1

                navigable_sectors.append(n_nav_sec)
                visible_sectors.append(n_vis_sec)
                trap_sectors.append(n_trap_sec)

            if scene_ep_count > 0:
                scene_episode_counts[scene_key] = scene_ep_count

    # convert sets to sorted lists for JSON
    cat_objects_json = {k: sorted(v) for k, v in category_objects.items()}

    return {
        "episode_categories": episode_categories,
        "category_objects": cat_objects_json,
        "scene_episode_counts": scene_episode_counts,
        "navigable_counts": navigable_counts,
        "valid_mask_counts": valid_mask_counts,
        "navigable_sectors": navigable_sectors,
        "visible_sectors": visible_sectors,
        "trap_sectors": trap_sectors,
        "far_navigable": far_navigable,
        "near_navigable": near_navigable,
        "far_visible": far_visible,
        "near_visible": near_visible,
        "mask_areas_all": mask_areas_all,
        "mask_areas_far": mask_areas_far,
        "mask_areas_near": mask_areas_near,
    }


# =====================================================================
#  Merge raw data from multiple JSON files
# =====================================================================

def merge_raw(raw_list):
    """Merge multiple raw dicts into one."""
    merged = {
        "episode_categories": [],
        "category_objects": defaultdict(set),
        "cat_mask_areas": defaultdict(list),
        "scene_episode_counts": {},
        "navigable_counts": [],
        "valid_mask_counts": [],
        "navigable_sectors": [],
        "visible_sectors": [],
        "trap_sectors": [],
        "far_navigable": [],
        "near_navigable": [],
        "far_visible": [],
        "near_visible": [],
        "mask_areas_all": [],
        "mask_areas_far": [],
        "mask_areas_near": [],
    }

    list_keys = [
        "episode_categories",
        "navigable_counts", "valid_mask_counts",
        "navigable_sectors", "visible_sectors", "trap_sectors",
        "far_navigable", "near_navigable", "far_visible", "near_visible",
        "mask_areas_all", "mask_areas_far", "mask_areas_near",
    ]

    for raw in raw_list:
        for k in list_keys:
            merged[k].extend(raw.get(k, []))
        # merge scene counts
        for scene, cnt in raw.get("scene_episode_counts", {}).items():
            merged["scene_episode_counts"][scene] = (
                merged["scene_episode_counts"].get(scene, 0) + cnt
            )
        # merge category objects
        for cat, oids in raw.get("category_objects", {}).items():
            if isinstance(oids, list):
                merged["category_objects"][cat].update(oids)
            else:
                merged["category_objects"][cat].update(oids)
        # merge per-category mask areas
        for cat, areas in raw.get("cat_mask_areas", {}).items():
            merged["cat_mask_areas"][cat].extend(areas)

    # convert sets back to sorted lists
    merged["category_objects"] = {
        k: sorted(v) for k, v in merged["category_objects"].items()
    }
    # convert defaultdict to dict
    merged["cat_mask_areas"] = dict(merged["cat_mask_areas"])
    return merged


# =====================================================================
#  Compute statistics from raw data
# =====================================================================

def compute_stats(raw, data_label=""):
    """Compute all statistics from raw per-episode data."""

    episode_categories = raw["episode_categories"]
    category_objects = raw["category_objects"]
    scene_episode_counts = raw["scene_episode_counts"]

    # helper functions
    def safe_avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    def safe_median(lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    def safe_std(lst):
        if len(lst) < 2:
            return 0.0
        avg = sum(lst) / len(lst)
        var = sum((x - avg) ** 2 for x in lst) / len(lst)
        return round(math.sqrt(var), 2)

    def int_distribution(lst):
        d = defaultdict(int)
        for v in lst:
            d[v] += 1
        return {str(k): v for k, v in sorted(d.items())}

    def histogram(lst, bin_edges):
        bins = [0] * len(bin_edges)
        for v in lst:
            placed = False
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= v < bin_edges[i + 1]:
                    bins[i] += 1
                    placed = True
                    break
            if not placed:
                bins[-1] += 1
        result = []
        for i in range(len(bin_edges) - 1):
            result.append({"range": f"{bin_edges[i]}-{bin_edges[i+1]}", "count": bins[i]})
        result.append({"range": f"{bin_edges[-1]}+", "count": bins[-1]})
        return result

    def pctiles(lst):
        if not lst:
            return {}
        s = sorted(lst)
        n = len(s)
        return {
            "p5": s[int(n * 0.05)],
            "p25": s[int(n * 0.25)],
            "p50": s[int(n * 0.50)],
            "p75": s[int(n * 0.75)],
            "p95": s[int(n * 0.95)],
        }

    mask_bins = [0, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    # global lists
    nav_c = raw["navigable_counts"]
    mask_c = raw["valid_mask_counts"]
    nav_s = raw["navigable_sectors"]
    vis_s = raw["visible_sectors"]
    trap_s = raw["trap_sectors"]
    fn = raw["far_navigable"]
    nn = raw["near_navigable"]
    fv = raw["far_visible"]
    nv = raw["near_visible"]
    ma_all = raw["mask_areas_all"]
    ma_far = raw["mask_areas_far"]
    ma_near = raw["mask_areas_near"]

    # per-category indices
    cat_indices = defaultdict(list)
    for i, cat in enumerate(episode_categories):
        cat_indices[cat].append(i)

    def sub(lst, idxs):
        return [lst[i] for i in idxs]

    # per-category stats
    per_category = {}
    for cat in sorted(cat_indices.keys()):
        idxs = cat_indices[cat]
        # collect mask areas for this category from raw lists
        # (mask areas are per-viewpoint, not per-episode, so we need a different approach)
        # We'll compute from the per-episode data: filter mask_areas by category
        # Unfortunately mask_areas_all is per-viewpoint not per-episode, so we can't index by episode.
        # Instead, keep category mask areas in the raw data approach separately.
        # For now, use a category-level collection approach.
        per_category[cat] = {
            "episodes": len(idxs),
            "unique_objects": len(category_objects.get(cat, [])),
            "navigable_sectors": {
                "avg": safe_avg(sub(nav_s, idxs)),
                "distribution": int_distribution(sub(nav_s, idxs)),
            },
            "visible_sectors": {
                "avg": safe_avg(sub(vis_s, idxs)),
                "distribution": int_distribution(sub(vis_s, idxs)),
            },
            "trap_sectors": {
                "avg": safe_avg(sub(trap_s, idxs)),
                "distribution": int_distribution(sub(trap_s, idxs)),
            },
            "navigable_viewpoints": {
                "avg": safe_avg(sub(nav_c, idxs)),
                "distribution": int_distribution(sub(nav_c, idxs)),
            },
            "valid_mask_viewpoints": {
                "avg": safe_avg(sub(mask_c, idxs)),
                "distribution": int_distribution(sub(mask_c, idxs)),
            },
        }

    # per-category mask areas — stored separately in raw as cat_mask_areas
    cat_mask_areas = raw.get("cat_mask_areas", {})
    for cat in per_category:
        areas = cat_mask_areas.get(cat, [])
        per_category[cat]["mask_area_px"] = {
            "avg": safe_avg(areas),
            "median": safe_median(areas),
            "std": safe_std(areas),
            "percentiles": pctiles(areas),
            "histogram": histogram(areas, mask_bins),
        }

    total_unique = set()
    for oids in category_objects.values():
        total_unique.update(oids)

    stats = {
        "data_label": data_label,
        "summary": {
            "total_scenes": len(scene_episode_counts),
            "total_episodes": len(episode_categories),
            "total_unique_objects": len(total_unique),
            "total_categories": len(cat_indices),
        },
        "viewpoint_stats": {
            "navigable_viewpoints_per_episode": {
                "avg": safe_avg(nav_c), "std": safe_std(nav_c),
                "distribution": int_distribution(nav_c),
            },
            "valid_mask_viewpoints_per_episode": {
                "avg": safe_avg(mask_c), "std": safe_std(mask_c),
                "distribution": int_distribution(mask_c),
            },
            "navigable_sectors_per_episode": {
                "avg": safe_avg(nav_s), "std": safe_std(nav_s),
                "distribution": int_distribution(nav_s),
            },
            "visible_sectors_per_episode": {
                "avg": safe_avg(vis_s), "std": safe_std(vis_s),
                "distribution": int_distribution(vis_s),
            },
            "trap_sectors_per_episode": {
                "avg": safe_avg(trap_s), "std": safe_std(trap_s),
                "distribution": int_distribution(trap_s),
            },
        },
        "range_stats": {
            "far": {
                "avg_navigable_per_episode": safe_avg(fn),
                "avg_visible_per_episode": safe_avg(fv),
                "navigable_distribution": int_distribution(fn),
                "visible_distribution": int_distribution(fv),
            },
            "near": {
                "avg_navigable_per_episode": safe_avg(nn),
                "avg_visible_per_episode": safe_avg(nv),
                "navigable_distribution": int_distribution(nn),
                "visible_distribution": int_distribution(nv),
            },
        },
        "mask_area_stats": {
            "all": {
                "count": len(ma_all), "avg": safe_avg(ma_all),
                "median": safe_median(ma_all), "std": safe_std(ma_all),
                "min": min(ma_all) if ma_all else 0,
                "max": max(ma_all) if ma_all else 0,
                "percentiles": pctiles(ma_all),
                "histogram": histogram(ma_all, mask_bins),
            },
            "far": {
                "count": len(ma_far), "avg": safe_avg(ma_far),
                "median": safe_median(ma_far),
                "percentiles": pctiles(ma_far),
                "histogram": histogram(ma_far, mask_bins),
            },
            "near": {
                "count": len(ma_near), "avg": safe_avg(ma_near),
                "median": safe_median(ma_near),
                "percentiles": pctiles(ma_near),
                "histogram": histogram(ma_near, mask_bins),
            },
        },
        "per_category": per_category,
        "per_scene": scene_episode_counts,
    }

    return stats


# =====================================================================
#  Extended collect_raw: also collect per-category mask areas
# =====================================================================

def collect_raw_full(data_dirs):
    """collect_raw + per-category mask areas for accurate merge."""
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    raw = collect_raw(data_dirs)

    # second pass: collect per-category mask areas
    # (we reconstruct from episode_categories + viewpoints)
    # Actually, let's do it in a single pass by re-scanning.
    # Better: integrate into collect_raw. But to avoid rewriting,
    # we'll do it inline here by re-scanning.
    cat_mask_areas = defaultdict(list)

    for data_dir in data_dirs:
        scenes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        for scene_key in scenes:
            scene_dir = os.path.join(data_dir, scene_key)
            ep_dirs = sorted([
                d for d in os.listdir(scene_dir)
                if os.path.isdir(os.path.join(scene_dir, d))
            ])
            for ep_id in ep_dirs:
                meta_path = os.path.join(scene_dir, ep_id, "meta.json")
                if not os.path.isfile(meta_path):
                    continue
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                cat = meta.get("object_category", "unknown")
                for vp in meta.get("viewpoints", []):
                    if vp.get("has_mask", False) and vp.get("navigable", False):
                        cat_mask_areas[cat].append(vp.get("mask_area_px", 0))

    raw["cat_mask_areas"] = dict(cat_mask_areas)
    return raw


# =====================================================================
#  Main
# =====================================================================

def print_summary(stats):
    s = stats["summary"]
    v = stats["viewpoint_stats"]
    r = stats["range_stats"]
    m = stats["mask_area_stats"]
    print(f"\n{'='*60}")
    print(f"  Scenes:      {s['total_scenes']}")
    print(f"  Episodes:    {s['total_episodes']}")
    print(f"  Objects:     {s['total_unique_objects']}")
    print(f"  Categories:  {s['total_categories']}")
    print(f"{'='*60}")
    print(f"  Viewpoints (per episode, out of 12):")
    print(f"    Navigable:   avg={v['navigable_viewpoints_per_episode']['avg']}, "
          f"std={v['navigable_viewpoints_per_episode']['std']}")
    print(f"    Valid mask:  avg={v['valid_mask_viewpoints_per_episode']['avg']}, "
          f"std={v['valid_mask_viewpoints_per_episode']['std']}")
    print(f"  Sectors (per episode, out of 6):")
    print(f"    Navigable:   avg={v['navigable_sectors_per_episode']['avg']}, "
          f"std={v['navigable_sectors_per_episode']['std']}")
    print(f"    Visible:     avg={v['visible_sectors_per_episode']['avg']}, "
          f"std={v['visible_sectors_per_episode']['std']}")
    print(f"    Trap:        avg={v['trap_sectors_per_episode']['avg']}, "
          f"std={v['trap_sectors_per_episode']['std']}")
    print(f"  Range breakdown (per episode, out of 6 each):")
    print(f"    Far  navigable={r['far']['avg_navigable_per_episode']}, "
          f"visible={r['far']['avg_visible_per_episode']}")
    print(f"    Near navigable={r['near']['avg_navigable_per_episode']}, "
          f"visible={r['near']['avg_visible_per_episode']}")
    print(f"  Mask area (px):")
    print(f"    All:  avg={m['all']['avg']}, median={m['all']['median']}, "
          f"p5={m['all']['percentiles'].get('p5','')}, p95={m['all']['percentiles'].get('p95','')}")
    print(f"    Far:  avg={m['far']['avg']}, median={m['far']['median']}")
    print(f"    Near: avg={m['near']['avg']}, median={m['near']['median']}")
    print(f"{'='*60}")
    print(f"\nPer-category:")
    for cat, info in stats["per_category"].items():
        print(f"  {cat:16s}: {info['episodes']:4d} eps, {info['unique_objects']:3d} objs, "
              f"nav_sec={info['navigable_sectors']['avg']:.1f}, "
              f"vis_sec={info['visible_sectors']['avg']:.1f}, "
              f"trap={info['trap_sectors']['avg']:.1f}, "
              f"mask_avg={info['mask_area_px']['avg']:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Extract PIN dataset statistics")
    parser.add_argument("--data_dir", nargs="+", default=None,
                        help="One or more root capture directories")
    parser.add_argument("--merge", nargs="+", default=None,
                        help="Merge previously saved raw JSON files (no server needed)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    if args.merge and args.data_dir:
        parser.error("Use --data_dir OR --merge, not both")
    if not args.merge and not args.data_dir:
        parser.error("Provide --data_dir or --merge")

    # --- merge mode ---
    if args.merge:
        print(f"Merging {len(args.merge)} JSON files ...")
        raw_list = []
        for path in args.merge:
            with open(path, "r") as f:
                data = json.load(f)
            raw_list.append(data["raw"])
            print(f"  {path}: {len(data['raw']['episode_categories'])} episodes")

        merged = merge_raw(raw_list)
        label = "merged"
        if args.output is None:
            args.output = "merged_stats.json"

    # --- scan mode ---
    else:
        print(f"Scanning {len(args.data_dir)} directories ...")
        for d in args.data_dir:
            print(f"  {d}")
        merged = collect_raw_full(args.data_dir)
        label = ", ".join(args.data_dir)
        if args.output is None:
            if len(args.data_dir) == 1:
                split_name = os.path.basename(args.data_dir[0].rstrip("/\\"))
                args.output = f"{split_name}_stats.json"
            else:
                args.output = "combined_stats.json"

    stats = compute_stats(merged, data_label=label)

    # save both stats and raw data (raw enables future merging)
    output_data = {
        "stats": stats,
        "raw": merged,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print_summary(stats)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
