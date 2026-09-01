#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# Classify abnormal episodes and tabulate which gate combinations fail.
# 
# Gates:
#   G1: is_cross_floor == False, stays on one floor
#   G2: episode_mask_visible == True, the target was seen
#   G3: 0 <= height_diff <= 1.6, plausible height
#
# A good episode passes all three.
# 

import os
import json
import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# font setup for the plots
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# OVON threshold
SINGLE_FLOOR_THRESHOLD = 0.25


def load_metrics_map(root_dir: str):
    """Load per-episode metrics from the *_results.jsonl files."""
    pattern = os.path.join(root_dir, "*_results.jsonl")
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        print(f"[WARN] No *_results.jsonl found under: {root_dir}")
        return {}
    
    metrics_map = {}
    for rf in result_files:
        with open(rf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except:
                    continue
                scene_id = obj.get("scene_id")
                episode_id = obj.get("episode_id")
                if scene_id is None or episode_id is None:
                    continue
                key = (scene_id, str(episode_id))
                metrics_map[key] = obj
    return metrics_map


def classify_abnormal_episodes(root_dir: str, output_dir: str = None, show_plots: bool = True):
    """
    Classify abnormal episodes and tabulate the failing gate combinations.
    
    The three gates:
      G1: is_cross_floor == False, stays on one floor
      G2: episode_mask_visible == True, the target was seen
      G3: 0 <= height_diff <= 1.6, plausible height
    
    A good episode passes all three.
    """
    if output_dir is None:
        output_dir = root_dir
    
    # every *_gate_summary.json, skipping the merged_ ones
    pattern = os.path.join(root_dir, "*_gate_summary.json")
    summary_files = sorted(glob.glob(pattern))
    summary_files = [f for f in summary_files if not os.path.basename(f).startswith("merged")]
    
    if not summary_files:
        raise FileNotFoundError(f"No *_gate_summary.json found under: {root_dir}")
    
    print(f"Found {len(summary_files)} summary files")
    
    # load the metrics map
    metrics_map = load_metrics_map(root_dir)
    
    # gather the episodes
    all_episodes = []
    for path in summary_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        eps = data.get("episodes", [])
        all_episodes.extend(eps)
    
    print(f"Total episodes: {len(all_episodes)}")
    
    # counter over failing gate combinations
    # keyed by (G1_fail, G2_fail, G3_fail)
    gate_combinations = defaultdict(list)
    
    # counters
    height_diffs = []
    path_height_ranges = []
    traj_height_ranges = []
    
    # classification output
    good_episodes = []
    bad_episodes = []
    
    for ep in all_episodes:
        scene_id = ep.get("scene_id")
        episode_id = str(ep.get("episode_id"))
        key = (scene_id, episode_id)
        
        # goal_position and final_position come from the metrics map
        m = metrics_map.get(key)
        goal_y = None
        final_y = None
        height_diff = None
        
        if m:
            goal_pos = m.get("goal_position")
            final_pos = m.get("final_position")
            if isinstance(goal_pos, (list, tuple)) and len(goal_pos) >= 2:
                goal_y = float(goal_pos[1])
            if isinstance(final_pos, (list, tuple)) and len(final_pos) >= 2:
                final_y = float(final_pos[1])
            if goal_y is not None and final_y is not None:
                height_diff = goal_y - final_y
                height_diffs.append(height_diff)
        
        # record the height spread
        path_hr = ep.get("path_height_range", 0.0)
        traj_hr = ep.get("trajectory_height_range", ep.get("height_range", 0.0))
        if path_hr:
            path_height_ranges.append(path_hr)
        if traj_hr:
            traj_height_ranges.append(traj_hr)
        
        # ========================================
        # evaluate the three gates
        # ========================================
        
        # G1: single floor
        is_cross_floor = ep.get("is_cross_floor", False)
        g1_pass = not is_cross_floor
        
        # G2: target seen
        mask_visible = ep.get("episode_mask_visible", False)
        g2_pass = mask_visible
        
        # G3: plausible height
        if height_diff is not None:
            g3_pass = (height_diff >= 0) and (height_diff <= 1.6)
        else:
            g3_pass = True  # no data, assume it passes
        
        # basic fields
        episode_info = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "video_rgb": ep.get("video_rgb"),
            "video_mask": ep.get("video_mask"),
            "snapshot_rgb": ep.get("snapshot_rgb"),
            "snapshot_mask": ep.get("snapshot_mask"),
            "goal_y": goal_y,
            "final_y": final_y,
            "height_diff": height_diff,
            "success": ep.get("success"),
            "spl": ep.get("spl"),
            # gate outcomes
            "g1_pass": g1_pass,
            "g2_pass": g2_pass,
            "g3_pass": g3_pass,
            "is_cross_floor": is_cross_floor,
            "episode_mask_visible": mask_visible,
            "path_height_range": path_hr,
            "trajectory_height_range": traj_hr,
        }
        
        # good or bad
        is_good = g1_pass and g2_pass and g3_pass
        
        if is_good:
            good_episodes.append(episode_info)
        else:
            # record why it failed
            fail_gates = []
            if not g1_pass:
                fail_gates.append("G1")
            if not g2_pass:
                fail_gates.append("G2")
            if not g3_pass:
                fail_gates.append("G3")
            
            episode_info["fail_gates"] = fail_gates
            episode_info["fail_pattern"] = "+".join(fail_gates)
            bad_episodes.append(episode_info)
            
            # record the combination
            combo_key = (not g1_pass, not g2_pass, not g3_pass)
            gate_combinations[combo_key].append(episode_info)
    
    # write JSON
    os.makedirs(output_dir, exist_ok=True)
    
    def save_json(data, filename, description):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "description": description,
                "count": len(data),
                "episodes": data
            }, f, ensure_ascii=False, indent=2)
        print(f"  {filename}: {len(data)} episodes")
        return filepath
    
    print(f"\n{'='*60}")
    print(f"Saving results to: {output_dir}")
    print(f"{'='*60}")
    
    save_json(good_episodes, "good_episodes.json", 
              "Good episodes (G1 AND G2 AND G3 all pass)")
    
    save_json(bad_episodes, "bad_episodes.json",
              "Bad episodes (at least one gate failed)")
    
    # ========================================
    # failing-combination statistics
    # ========================================
    num_eps = len(all_episodes)
    
    print(f"\n{'='*60}")
    print(f"Gate Failure Distribution")
    print(f"{'='*60}")
    print(f"  G1: is_cross_floor == False (single floor)")
    print(f"  G2: episode_mask_visible == True")
    print(f"  G3: 0 <= height_diff <= 1.6m")
    print(f"{'='*60}")
    
    # names for the combinations
    combo_names = {
        (True, False, False): "G1 only",
        (False, True, False): "G2 only",
        (False, False, True): "G3 only",
        (True, True, False): "G1+G2",
        (True, False, True): "G1+G3",
        (False, True, True): "G2+G3",
        (True, True, True): "G1+G2+G3",
    }
    
    combo_stats = {}
    for combo_key, name in combo_names.items():
        eps = gate_combinations.get(combo_key, [])
        count = len(eps)
        pct = count / num_eps * 100 if num_eps > 0 else 0
        combo_stats[name] = {"count": count, "percentage": pct}
        if count > 0:
            print(f"  {name:12s}: {count:4d} ({pct:5.2f}%)")
    
    # write the combination stats
    for combo_key, name in combo_names.items():
        eps = gate_combinations.get(combo_key, [])
        if eps:
            filename = f"fail_{name.replace('+', '_').replace(' ', '_').lower()}.json"
            save_json(eps, filename, f"Episodes failing {name}")
    
    # overall statistics
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Total episodes:     {num_eps}")
    print(f"  Good episodes:      {len(good_episodes):4d} ({len(good_episodes)/num_eps*100:.2f}%)")
    print(f"  Bad episodes:       {len(bad_episodes):4d} ({len(bad_episodes)/num_eps*100:.2f}%)")
    
    # per-gate failure rates
    g1_fail = sum(1 for ep in bad_episodes if not ep.get("g1_pass", True))
    g2_fail = sum(1 for ep in bad_episodes if not ep.get("g2_pass", True))
    g3_fail = sum(1 for ep in bad_episodes if not ep.get("g3_pass", True))
    
    print(f"\n  Gate failure rates (in bad episodes):")
    print(f"    G1 (cross-floor): {g1_fail:4d}")
    print(f"    G2 (invisible):   {g2_fail:4d}")
    print(f"    G3 (height):      {g3_fail:4d}")
    
    # write the report
    stats_report = {
        "total_episodes": num_eps,
        "good_episodes": len(good_episodes),
        "bad_episodes": len(bad_episodes),
        "good_rate": len(good_episodes) / num_eps * 100 if num_eps > 0 else 0,
        "bad_rate": len(bad_episodes) / num_eps * 100 if num_eps > 0 else 0,
        "gate_definitions": {
            "G1": "is_cross_floor == False (single floor navigation)",
            "G2": "episode_mask_visible == True (target visible)",
            "G3": "0 <= height_diff <= 1.6m (normal height)",
        },
        "gate_failure_counts": {
            "G1": g1_fail,
            "G2": g2_fail,
            "G3": g3_fail,
        },
        "combination_stats": combo_stats,
    }
    
    with open(os.path.join(output_dir, "statistics_report.json"), "w", encoding="utf-8") as f:
        json.dump(stats_report, f, ensure_ascii=False, indent=2)
    print(f"\n  statistics_report.json saved")
    
    # =========================================================
    # plots
    # =========================================================
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PIN-v2 Episode Classification (3 Gates)', fontsize=14, fontweight='bold')
        
        # 1. good vs bad, pie
        ax1 = axes[0, 0]
        good_count = len(good_episodes)
        bad_count = len(bad_episodes)
        labels = [f'Good\n({good_count})', f'Bad\n({bad_count})']
        sizes = [good_count, bad_count]
        colors = ['#27ae60', '#e74c3c']
        explode = (0.05, 0)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode,
               startangle=90, shadow=True, textprops={'fontsize': 11})
        ax1.set_title('Good vs Bad Episodes')
        
        # 2. failing combinations
        ax2 = axes[0, 1]
        combo_labels = []
        combo_counts = []
        for name in ["G1 only", "G2 only", "G3 only", "G1+G2", "G1+G3", "G2+G3", "G1+G2+G3"]:
            count = combo_stats.get(name, {}).get("count", 0)
            if count > 0:
                combo_labels.append(name)
                combo_counts.append(count)
        
        if combo_counts:
            colors = plt.cm.Set3(np.linspace(0, 1, len(combo_counts)))
            bars = ax2.bar(combo_labels, combo_counts, color=colors, edgecolor='black')
            ax2.set_ylabel('Episode Count')
            ax2.set_title('Gate Failure Combinations')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            for bar, count in zip(bars, combo_counts):
                pct = count / num_eps * 100
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)
        
        # 3. height difference
        ax3 = axes[1, 0]
        if height_diffs:
            ax3.hist(height_diffs, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
            ax3.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Ground (0m)')
            ax3.axvline(x=1.6, color='red', linestyle='--', linewidth=2, label='Max (1.6m)')
            ax3.axvspan(-10, 0, alpha=0.1, color='red')
            ax3.axvspan(1.6, 10, alpha=0.1, color='red')
            ax3.set_xlabel('Height Diff (goal_y - final_y) [m]')
            ax3.set_ylabel('Episode Count')
            ax3.set_title('G3: Height Difference Distribution')
            ax3.set_xlim(-3, 5)
            ax3.legend(loc='upper right')
        
        # 4. path height spread
        ax4 = axes[1, 1]
        height_ranges = traj_height_ranges if traj_height_ranges else path_height_ranges
        if height_ranges:
            ax4.hist(height_ranges, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
            ax4.axvline(x=SINGLE_FLOOR_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold ({SINGLE_FLOOR_THRESHOLD}m)')
            ax4.set_xlabel('Height Range [m]')
            ax4.set_ylabel('Episode Count')
            ax4.set_title('G1: Trajectory Height Range Distribution')
            ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    return stats_report


# ========================================
# entry point, takes command-line arguments
# ========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify abnormal episodes based on 3 gates")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing *_gate_summary.json files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for reports (defaults to input_dir)")
    parser.add_argument("--show_plots", action="store_true", default=False,
                        help="Show matplotlib plots")
    args = parser.parse_args()
    
    output = args.output_dir if args.output_dir else args.input_dir
    classify_abnormal_episodes(args.input_dir, output, show_plots=args.show_plots)


