#!/usr/bin/env python3
"""
Scan a capture directory for objects whose rendered color looks wrong.

Method: read RGB and mask, take the object's pixels, average their RGB.
Flags red-dominant (R >> G, B), blue-dominant, and very dark objects.

Usage:
    python3 scan_render_colors.py \
        --capture_dir <capture root>/pin_capture_FT9_RGB_v2 \
        --sample 0          # 0 scans everything

    # restrict to one category
    python3 scan_render_colors.py \
        --capture_dir <capture root>/pin_capture_FT9_RGB_v2 \
        --category ball
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


TARGET_SEMANTIC_ID = 100000
MIN_MASK_PIXELS = 50  # minimum object pixels before color is analyzed


def analyze_episode(ep_dir):
    """Analyze the rendered object color for one episode."""
    ep_dir = Path(ep_dir)
    meta_path = ep_dir / "meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    object_id = meta.get("object_id", "unknown")
    object_category = meta.get("object_category", "unknown")
    episode_id = meta.get("episode_id", "?")
    scene_key = meta.get("scene_key", "?")

    rgb_dir = ep_dir / "rgb"
    mask_dir = ep_dir / "mask"

    if not rgb_dir.exists():
        return None

    # collect the object color at every viewpoint
    all_pixels = []
    viewpoint_colors = []

    from PIL import Image

    for rgb_file in sorted(rgb_dir.glob("rgb_*.png")):
        tag = rgb_file.stem.replace("rgb_", "")
        mask_file = mask_dir / f"mask_{tag}.png" if mask_dir.exists() else None

        try:
            rgb_img = np.array(Image.open(rgb_file))
        except Exception:
            continue

        # try to read the mask
        mask = None
        if mask_file and mask_file.exists():
            try:
                mask_img = np.array(Image.open(mask_file))
                # the mask may be single- or multi-channel
                if mask_img.ndim == 3:
                    mask = mask_img[:, :, 0] > 127
                else:
                    mask = mask_img > 127
            except Exception:
                pass

        if mask is None:
            # no mask file, skip this viewpoint
            continue

        mask_pixels = int(mask.sum())
        if mask_pixels < MIN_MASK_PIXELS:
            continue

        # take the object pixels, RGB only, dropping alpha
        obj_rgb = rgb_img[mask][:, :3] if rgb_img.shape[-1] >= 3 else rgb_img[mask]
        all_pixels.append(obj_rgb)

        # mean color for this viewpoint
        mean_rgb = obj_rgb.mean(axis=0)
        viewpoint_colors.append({
            "tag": tag,
            "mask_pixels": mask_pixels,
            "mean_r": float(mean_rgb[0]),
            "mean_g": float(mean_rgb[1]),
            "mean_b": float(mean_rgb[2]),
        })

    if not all_pixels:
        return None

    # pool the pixels across viewpoints
    combined = np.concatenate(all_pixels, axis=0)
    overall_mean = combined.mean(axis=0).astype(float)
    overall_std = combined.std(axis=0).astype(float)

    r, g, b = overall_mean[0], overall_mean[1], overall_mean[2]

    # --- anomaly tests ---
    anomalies = []

    # red-dominant: R well above both G and B
    if r > 120 and r > g * 1.8 and r > b * 1.8:
        anomalies.append(f"RED_DOMINANT (R={r:.0f}, G={g:.0f}, B={b:.0f})")

    # blue-dominant
    if b > 120 and b > r * 1.8 and b > g * 1.8:
        anomalies.append(f"BLUE_DOMINANT (R={r:.0f}, G={g:.0f}, B={b:.0f})")

    # green-dominant
    if g > 120 and g > r * 1.8 and g > b * 1.8:
        anomalies.append(f"GREEN_DOMINANT (R={r:.0f}, G={g:.0f}, B={b:.0f})")

    # very dark, which can mean the object did not render at all
    brightness = (r + g + b) / 3
    if brightness < 20:
        anomalies.append(f"VERY_DARK (avg={brightness:.0f})")

    # flat grey with no channel spread, which can mean a lost material
    color_range = max(r, g, b) - min(r, g, b)
    if brightness > 50 and color_range < 10:
        # grey is not necessarily wrong; some objects really are grey
        pass

    return {
        "episode_id": episode_id,
        "scene_key": scene_key,
        "object_id": object_id,
        "object_category": object_category,
        "num_viewpoints_analyzed": len(viewpoint_colors),
        "total_pixels": int(combined.shape[0]),
        "mean_rgb": [float(r), float(g), float(b)],
        "std_rgb": [float(overall_std[0]), float(overall_std[1]), float(overall_std[2])],
        "anomalies": anomalies,
        "is_anomalous": len(anomalies) > 0,
        "ep_dir": str(ep_dir),
    }


def scan_capture_dir(capture_dir, sample=0, category_filter=None):
    """Scan a whole capture directory."""
    capture_dir = Path(capture_dir)

    # gather every episode directory
    episode_dirs = []
    for scene_dir in sorted(capture_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for ep_dir in sorted(scene_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            meta_path = ep_dir / "meta.json"
            if meta_path.exists():
                # apply the category filter, if any
                if category_filter:
                    try:
                        with open(meta_path) as f:
                            m = json.load(f)
                        if m.get("object_category", "").lower() != category_filter.lower():
                            continue
                    except Exception:
                        continue
                episode_dirs.append(ep_dir)

    if sample > 0:
        import random
        random.shuffle(episode_dirs)
        episode_dirs = episode_dirs[:sample]

    total = len(episode_dirs)
    print(f"Scanning {total} episodes in {capture_dir}")
    if category_filter:
        print(f"  Category filter: {category_filter}")

    results = []
    anomalous = []
    per_category = defaultdict(lambda: {"total": 0, "anomalous": 0, "oids": set()})
    per_object_id = defaultdict(lambda: {"anomalous_count": 0, "total_count": 0,
                                          "category": "", "mean_rs": []})

    for idx, ep_dir in enumerate(episode_dirs):
        if (idx + 1) % 200 == 0:
            print(f"  ... {idx + 1}/{total}")

        result = analyze_episode(ep_dir)
        if result is None:
            continue

        results.append(result)
        cat = result["object_category"]
        oid = result["object_id"]
        per_category[cat]["total"] += 1
        per_category[cat]["oids"].add(oid)
        per_object_id[oid]["total_count"] += 1
        per_object_id[oid]["category"] = cat
        per_object_id[oid]["mean_rs"].append(result["mean_rgb"][0])

        if result["is_anomalous"]:
            anomalous.append(result)
            per_category[cat]["anomalous"] += 1
            per_object_id[oid]["anomalous_count"] += 1

    # ===================== REPORT =====================
    print(f"\n{'='*70}")
    print(f"COLOR SCAN REPORT")
    print(f"{'='*70}")
    print(f"Episodes analyzed: {len(results)}")
    print(f"Anomalous episodes: {len(anomalous)}")
    if len(results) > 0:
        print(f"Anomaly rate: {len(anomalous)/len(results)*100:.1f}%")

    # --- Per category ---
    print(f"\n--- Per category ---")
    for cat in sorted(per_category.keys()):
        info = per_category[cat]
        anom = info["anomalous"]
        tot = info["total"]
        n_oids = len(info["oids"])
        status = "OK" if anom == 0 else f"!!! {anom} ANOMALOUS"
        print(f"  {cat:15s}: {tot:4d} episodes, {n_oids:3d} objects  [{status}]")

    # --- Anomalous objects ---
    if anomalous:
        print(f"\n--- Anomalous episodes ({len(anomalous)}) ---")
        # Group by object_id
        by_oid = defaultdict(list)
        for r in anomalous:
            by_oid[r["object_id"]].append(r)

        print(f"  Unique anomalous object_ids: {len(by_oid)}")
        for oid, episodes in sorted(by_oid.items()):
            cat = episodes[0]["object_category"]
            rgb_strs = [f"({r['mean_rgb'][0]:.0f},{r['mean_rgb'][1]:.0f},{r['mean_rgb'][2]:.0f})"
                        for r in episodes[:3]]
            reasons = set()
            for r in episodes:
                for a in r["anomalies"]:
                    reasons.add(a.split("(")[0].strip())
            print(f"\n  {oid} ({cat}) — {len(episodes)} anomalous episodes")
            print(f"    Reasons: {', '.join(reasons)}")
            print(f"    Sample colors: {', '.join(rgb_strs)}")
            print(f"    Sample path: {episodes[0]['ep_dir']}")
    else:
        print(f"\n  No anomalous episodes found! All objects appear to render correctly.")

    # --- Object-level summary ---
    print(f"\n--- Object-level color summary (top 10 reddest) ---")
    oid_avg_r = []
    for oid, info in per_object_id.items():
        if info["mean_rs"]:
            avg_r = sum(info["mean_rs"]) / len(info["mean_rs"])
            oid_avg_r.append((oid, info["category"], avg_r, info["anomalous_count"],
                             info["total_count"]))
    oid_avg_r.sort(key=lambda x: -x[2])  # Sort by average R descending

    for oid, cat, avg_r, anom_count, total_count in oid_avg_r[:10]:
        flag = " <<< ANOMALOUS" if anom_count > 0 else ""
        print(f"  {oid[:16]}... ({cat:15s}): avg_R={avg_r:6.1f}, "
              f"anom={anom_count}/{total_count}{flag}")

    # --- JSON output ---
    output = {
        "total_analyzed": len(results),
        "anomalous_count": len(anomalous),
        "anomalous_object_ids": list(set(r["object_id"] for r in anomalous)),
        "anomalous_episodes": [
            {
                "episode_id": r["episode_id"],
                "scene_key": r["scene_key"],
                "object_id": r["object_id"],
                "object_category": r["object_category"],
                "mean_rgb": r["mean_rgb"],
                "anomalies": r["anomalies"],
            }
            for r in anomalous
        ],
        "per_category": {
            cat: {"total": info["total"], "anomalous": info["anomalous"]}
            for cat, info in per_category.items()
        },
    }
    out_path = capture_dir / "color_scan_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON report saved: {out_path}")

    return results, anomalous


def main():
    ap = argparse.ArgumentParser(description="Scan rendered object colors for anomalies")
    ap.add_argument("--capture_dir", required=True,
                    help="Path to capture directory (e.g. pin_capture_FT9_RGB_v2)")
    ap.add_argument("--sample", type=int, default=0,
                    help="Random sample N episodes (0 = all)")
    ap.add_argument("--category", type=str, default=None,
                    help="Only scan this category")
    args = ap.parse_args()

    if not os.path.isdir(args.capture_dir):
        print(f"[ERROR] Directory not found: {args.capture_dir}")
        sys.exit(1)

    scan_capture_dir(args.capture_dir, args.sample, args.category)


if __name__ == "__main__":
    main()
