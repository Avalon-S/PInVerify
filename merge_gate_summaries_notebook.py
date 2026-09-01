#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# Paste the body below into a Jupyter cell, or run this file directly.
# Merges the per-worker gate summaries and plots them, OVON-style cross-floor detection included.
# 

import os
import json
import glob
import math
import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# font setup for the plots
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# OVON threshold
SINGLE_FLOOR_THRESHOLD = 0.25  # metres


def load_metrics_map(root_dir: str):
    """Load per-episode metrics from the *_results.jsonl files."""
    pattern = os.path.join(root_dir, "*_results.jsonl")
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        print(f"[WARN] No *_results.jsonl found under: {root_dir}")
        return {}
    
    metrics_map = {}
    print(f"Found {len(result_files)} results.jsonl files:")
    for rf in result_files:
        print("  -", rf)
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


def merge_gate_summaries(root_dir: str, output_path: str, show_plots: bool = True):
    """Merge the gate summaries and plot them."""
    
    # every *_gate_summary.json, skipping the merged_ ones
    pattern = os.path.join(root_dir, "*_gate_summary.json")
    summary_files = sorted(glob.glob(pattern))
    summary_files = [f for f in summary_files if not os.path.basename(f).startswith("merged")]

    if not summary_files:
        raise FileNotFoundError(f"No *_gate_summary.json found under: {root_dir}")

    print(f"Found {len(summary_files)} summary files:")
    for f in summary_files:
        print("  -", f)

    # load the metrics map
    metrics_map = load_metrics_map(root_dir)

    all_episodes = []
    for path in summary_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        eps = data.get("episodes", [])
        all_episodes.extend(eps)

    # overall counters
    num_eps_total = len(all_episodes)
    num_nav_success = sum(1 for r in all_episodes if float(r.get("success", 0.0)) > 0.5)
    num_attempted = sum(1 for r in all_episodes if bool(r.get("gate_attempted", False)))
    num_gate1 = sum(1 for r in all_episodes if bool(r.get("gate1_pass", False)))
    num_gate2 = sum(1 for r in all_episodes if bool(r.get("gate2_pass", False)))
    num_gate3 = sum(1 for r in all_episodes if bool(r.get("gate3_pass", False)))
    num_mask_visible_eps = sum(1 for r in all_episodes if bool(r.get("episode_mask_visible", False)))

    nav_success_rate = float(num_nav_success) / float(num_eps_total) if num_eps_total > 0 else 0.0
    gate1_rate = float(num_gate1) / float(num_attempted) if num_attempted > 0 else 0.0
    gate2_rate = float(num_gate2) / float(num_attempted) if num_attempted > 0 else 0.0
    gate3_rate = float(num_gate3) / float(num_attempted) if num_attempted > 0 else 0.0
    mask_visible_rate = float(num_mask_visible_eps) / float(num_eps_total) if num_eps_total > 0 else 0.0

    # OVON cross-floor statistics
    num_cross_floor = sum(1 for r in all_episodes if r.get("is_cross_floor", False))
    num_cross_floor_path = sum(1 for r in all_episodes if r.get("is_cross_floor_path", False))
    num_cross_floor_traj = sum(1 for r in all_episodes if r.get("is_cross_floor_trajectory", False))
    
    cross_floor_rate = float(num_cross_floor) / float(num_eps_total) if num_eps_total > 0 else 0.0
    
    # grouped by reason
    cross_floor_reasons = defaultdict(int)
    for r in all_episodes:
        if r.get("is_cross_floor", False):
            reason = r.get("cross_floor_reason", "unknown")
            cross_floor_reasons[reason] += 1

    # height spread distribution
    path_height_ranges = [r.get("path_height_range", 0.0) for r in all_episodes if r.get("path_height_range") is not None]
    traj_height_ranges = [r.get("trajectory_height_range", 0.0) for r in all_episodes if r.get("trajectory_height_range") is not None]
    # tolerate older records
    if not traj_height_ranges:
        traj_height_ranges = [r.get("height_range", 0.0) for r in all_episodes if r.get("height_range") is not None]

    # height statistics
    goal_heights = []
    height_diffs = []
    num_height_abnormal = 0
    num_height_and_mask_invisible = 0
    num_height_or_mask_invisible = 0

    for ep in all_episodes:
        scene_id = ep.get("scene_id")
        episode_id = str(ep.get("episode_id"))
        key = (scene_id, episode_id)
        m = metrics_map.get(key)
        if not m:
            continue

        goal_pos = m.get("goal_position")
        final_pos = m.get("final_position")

        if not (isinstance(goal_pos, (list, tuple)) and len(goal_pos) >= 2 and
                isinstance(final_pos, (list, tuple)) and len(final_pos) >= 2):
            continue

        goal_y = float(goal_pos[1])
        final_y = float(final_pos[1])
        height_diff = goal_y - final_y

        goal_heights.append(goal_y)
        height_diffs.append(height_diff)

        height_abnormal = (height_diff > 1.6) or (height_diff < 0)
        mask_visible = bool(ep.get("episode_mask_visible", False))
        mask_invisible = not mask_visible

        if height_abnormal:
            num_height_abnormal += 1
        if height_abnormal and mask_invisible:
            num_height_and_mask_invisible += 1
        if height_abnormal or mask_invisible:
            num_height_or_mask_invisible += 1

    # =========================================================
    # print the summary
    # =========================================================
    print(f"\n{'='*60}")
    print(f"📁 Merged summary saved to: {output_path}")
    print(f"{'='*60}")
    
    print(f"\n📊 Basic Stats:")
    print(f"  Total episodes: {num_eps_total}")
    print(f"  Nav success rate: {nav_success_rate:.2%}")
    print(f"  Gate1/2/3 success: {gate1_rate:.2%}, {gate2_rate:.2%}, {gate3_rate:.2%}")
    print(f"  Mask visible: {num_mask_visible_eps} ({mask_visible_rate:.2%})")

    print(f"\n🏢 Cross-Floor Stats (OVON-style, threshold={SINGLE_FLOOR_THRESHOLD}m):")
    print(f"  Cross-floor episodes (total): {num_cross_floor} ({cross_floor_rate:.2%})")
    print(f"    - Path-based detection: {num_cross_floor_path}")
    print(f"    - Trajectory-based detection: {num_cross_floor_traj}")
    if cross_floor_reasons:
        print(f"  Reasons breakdown:")
        for reason, count in sorted(cross_floor_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")

    print(f"\n📏 Height Abnormal Stats:")
    print(f"  Height-abnormal episodes: {num_height_abnormal} ({num_height_abnormal/num_eps_total*100:.1f}%)")
    print(f"  Height-abnormal & mask-invisible (∩): {num_height_and_mask_invisible}")
    print(f"  Height-abnormal OR mask-invisible (∪): {num_height_or_mask_invisible}")
    
    # estimate the good episodes
    all_abnormal_keys = set()
    for ep in all_episodes:
        scene_id = ep.get("scene_id")
        episode_id = str(ep.get("episode_id"))
        key = (scene_id, episode_id)
        m = metrics_map.get(key)
        
        is_cross_floor = ep.get("is_cross_floor", False)
        mask_invisible = not ep.get("episode_mask_visible", False)
        
        height_abnormal = False
        if m:
            goal_pos = m.get("goal_position")
            final_pos = m.get("final_position")
            if isinstance(goal_pos, (list, tuple)) and isinstance(final_pos, (list, tuple)):
                height_diff = goal_pos[1] - final_pos[1]
                height_abnormal = (height_diff > 1.6) or (height_diff < 0)
        
        if is_cross_floor or mask_invisible or height_abnormal:
            all_abnormal_keys.add(key)
    
    good_eps = num_eps_total - len(all_abnormal_keys)
    print(f"\n🎯 Good Episodes Estimate:")
    print(f"  Total abnormal (cross-floor OR mask-invisible OR height-abnormal): {len(all_abnormal_keys)}")
    print(f"  Good episodes: {good_eps} ({good_eps/num_eps_total*100:.1f}%)")

    # write JSON
    merged = {
        "num_episodes_total": num_eps_total,
        "nav_success_rate": nav_success_rate,
        "gate1_success_rate": gate1_rate,
        "gate2_success_rate": gate2_rate,
        "gate3_success_rate": gate3_rate,
        "mask_visible_rate": mask_visible_rate,
        "cross_floor_stats": {
            "num_cross_floor": num_cross_floor,
            "cross_floor_rate": cross_floor_rate,
            "num_cross_floor_path": num_cross_floor_path,
            "num_cross_floor_trajectory": num_cross_floor_traj,
            "threshold_used": SINGLE_FLOOR_THRESHOLD,
        },
        "height_stats": {
            "num_height_abnormal": num_height_abnormal,
            "num_height_or_mask_invisible": num_height_or_mask_invisible,
        },
        "good_episodes_estimate": good_eps,
        "episodes": all_episodes,
        "generated_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # =========================================================
    # plots
    # =========================================================
    if show_plots:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PIN Dataset Statistics Analysis (OVON-style)', fontsize=14, fontweight='bold')
        
        # 1. gate pass rates
        ax1 = axes[0, 0]
        gates = ['Gate1', 'Gate2', 'Gate3']
        rates = [gate1_rate * 100, gate2_rate * 100, gate3_rate * 100]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax1.bar(gates, rates, color=colors, edgecolor='black')
        ax1.set_ylabel('Pass Rate (%)')
        ax1.set_title('Gate Pass Rates')
        ax1.set_ylim(0, 100)
        for bar, rate in zip(bars, rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 2. episode classification
        ax2 = axes[0, 1]
        mask_invisible = num_eps_total - num_mask_visible_eps
        labels = ['Good', 'Height Abnormal', 'Mask Invisible', 'Cross Floor']
        sizes = [good_eps, num_height_abnormal, mask_invisible, num_cross_floor]
        sizes = [max(0, s) for s in sizes]
        colors = ['#27ae60', '#e74c3c', '#3498db', '#f39c12']
        if sum(sizes) > 0:
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, shadow=True)
        ax2.set_title('Episode Quality Distribution')
        
        # 3. cross-floor detection comparison
        ax3 = axes[0, 2]
        labels = ['Path\nDetection', 'Trajectory\nDetection', 'Total\n(Union)']
        counts = [num_cross_floor_path, num_cross_floor_traj, num_cross_floor]
        colors = ['#9b59b6', '#3498db', '#e74c3c']
        bars = ax3.bar(labels, counts, color=colors, edgecolor='black')
        ax3.set_ylabel('Episode Count')
        ax3.set_title('OVON Cross-Floor Detection')
        for bar, count in zip(bars, counts):
            pct = count / num_eps_total * 100 if num_eps_total > 0 else 0
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        # 4. height difference histogram
        ax4 = axes[1, 0]
        if height_diffs:
            ax4.hist(height_diffs, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
            ax4.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Ground level')
            ax4.axvline(x=1.6, color='red', linestyle='--', linewidth=2, label='Max visible (1.6m)')
            ax4.set_xlabel('Height Diff (goal_y - final_y) [m]')
            ax4.set_ylabel('Episode Count')
            ax4.set_title('Height Difference Distribution')
            ax4.legend()
        
        # 5. trajectory and path height spread
        ax5 = axes[1, 1]
        if traj_height_ranges:
            ax5.hist(traj_height_ranges, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
            ax5.axvline(x=SINGLE_FLOOR_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                       label=f'OVON threshold ({SINGLE_FLOOR_THRESHOLD}m)')
            ax5.set_xlabel('Height Range [m]')
            ax5.set_ylabel('Episode Count')
            ax5.set_title('Trajectory Height Range Distribution')
            ax5.legend()
        
        # 6. anomaly summary
        ax6 = axes[1, 2]
        categories = ['Height\nAbnormal', 'Mask\nInvisible', 'Cross\nFloor', 'Union\n(All Bad)']
        counts = [num_height_abnormal, mask_invisible, num_cross_floor, len(all_abnormal_keys)]
        colors = ['#e74c3c', '#3498db', '#f39c12', '#2c3e50']
        bars = ax6.bar(categories, counts, color=colors, edgecolor='black')
        ax6.set_ylabel('Episode Count')
        ax6.set_title('Abnormal Episode Breakdown')
        for bar, count in zip(bars, counts):
            pct = count / num_eps_total * 100 if num_eps_total > 0 else 0
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    return merged


# ========================================
# Set the paths below, then run
# ========================================
if __name__ == "__main__":
    input_dir = "./pin_result/val/eval_pin_goalview"
    output_path = "./pin_result/val/eval_pin_goalview/merged_gate_summary.json"
    merge_gate_summaries(input_dir, output_path, show_plots=True)
