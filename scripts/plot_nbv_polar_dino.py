#!/usr/bin/env python
"""
Generate NBV direction polar charts from DINO episode data.

Produces nbv_per_step.pdf with 4 rows x 3 cols:
  Row 1: Step 1 — Random, FPS, LLM
  Row 2: Step 1 — SFT, GRPO, GSPO
  Row 3: Step 2 — Random, FPS, LLM
  Row 4: Step 2 — SFT, GRPO, GSPO

Usage:
  python scripts/plot_nbv_polar_dino.py
  python scripts/plot_nbv_polar_dino.py --output report_figs/nbv_per_step.pdf
"""

import argparse
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

# 6 directions (excluding "front" = STOP)
DIRECTIONS = ["front-left", "front-right", "back-right", "back", "back-left"]
DIR_LABELS = ["front-\nleft", "front-\nright", "back-\nright", "back", "back-\nleft"]

# Angles for polar plot (evenly spaced, starting from top-left)
# front-left=120°, front-right=60°, back-right=330°, back=270°, back-left=210°
# In radians, measured from top (90° = pi/2)
ANGLES_DEG = [120, 60, 330, 270, 210]
ANGLES_RAD = [np.deg2rad(a) for a in ANGLES_DEG]

# Method configs: (label, color, ep_dir)
BASE_TF = "./outputs/qwen3_vl_4b"
BASE_TR = "./outputs/trained"

METHODS = [
    ("Random", "#808080", os.path.join(BASE_TF, "multi_view_attr_adaptive_random_dino_all")),
    ("FPS",    "#4878CF", os.path.join(BASE_TF, "multi_view_attr_adaptive_fps_dino_all")),
    ("LLM",   "#D65F5F", os.path.join(BASE_TF, "multi_view_attr_adaptive_llm_dino_all")),
    ("SFT",   "#6ACC65", os.path.join(BASE_TR, "trained_sft_v2_all/trained_sft_v2_dino_all")),
    ("GRPO",  "#FF7F0E", os.path.join(BASE_TR, "trained_grpo_v2_all/trained_grpo_v2_dino_all")),
    ("GSPO",  "#B47CC7", os.path.join(BASE_TR, "trained_gspo_v2_all/trained_gspo_v2_dino_all")),
]


def extract_direction(nav_rel: str) -> str:
    """Extract clean direction from nav_rel string like 'back-right (Random)'."""
    direction = nav_rel.split("(")[0].strip().lower()
    if direction in set(DIRECTIONS) | {"front"}:
        return direction
    for d in sorted(set(DIRECTIONS) | {"front"}, key=len, reverse=True):
        if d in nav_rel.lower():
            return d
    return "unknown"


def collect_distributions(ep_dir: str):
    """
    Scan episode.json files, return per-step direction counts.
    Returns: {step_num: {direction: count}}, where step_num is 1-indexed.
    """
    step_counts = defaultdict(lambda: defaultdict(int))
    files = glob.glob(os.path.join(ep_dir, "**", "episode.json"), recursive=True)

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                ep = json.load(fh)
        except Exception:
            continue

        for step_rec in ep.get("transcript", []):
            action = step_rec.get("action", {})
            nav_rel = action.get("nav_rel", "")
            if not nav_rel or nav_rel.strip() == "front":
                continue  # STOP decision, not a navigation action

            direction = extract_direction(nav_rel)
            if direction in ("front", "unknown", ""):
                continue  # Skip hallucinated "front" or unknown

            step_num = step_rec.get("step", 1)
            step_counts[step_num][direction] += 1

    return dict(step_counts)


def make_polar_subplot(ax, counts, color, title, show_labels=True):
    """Draw a single polar bar chart on the given axes."""
    total = sum(counts.get(d, 0) for d in DIRECTIONS)
    if total == 0:
        ax.set_title(f"{title}\n(n=0)", fontsize=11, fontweight="bold", color=color, pad=15)
        return

    values = [counts.get(d, 0) / total * 100 for d in DIRECTIONS]
    angles = ANGLES_RAD.copy()

    # Bar width (each bar spans ~50 degrees)
    width = np.deg2rad(50)

    bars = ax.bar(angles, values, width=width, bottom=0,
                  color=color, alpha=0.7, edgecolor="white", linewidth=1.2)

    # Add percentage labels
    for angle, val, bar in zip(angles, values, bars):
        if val > 3:  # Only label bars > 3%
            # Position label outside the bar
            r = bar.get_height() + 1.5
            ha = "center"
            ax.text(angle, r, f"{val:.1f}%", ha=ha, va="center",
                    fontsize=7.5, fontweight="bold")

    # Dashed circle for uniform baseline (20%)
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, [20] * 100, "--", color="gray", linewidth=0.8, alpha=0.6)

    # Small "front" marker at top (90°)
    ax.annotate("front", xy=(np.deg2rad(90), 2), fontsize=6, ha="center",
                va="center", color="gray", alpha=0.7)

    ax.set_title(f"{title}\n(n={total})", fontsize=11, fontweight="bold",
                 color=color, pad=15)

    # Configure polar axes
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 40)
    ax.set_thetagrids(ANGLES_DEG, DIR_LABELS if show_labels else [""] * 5,
                      fontsize=7)
    ax.set_yticklabels([])  # Hide radial tick labels
    ax.set_rticks([10, 20, 30, 40])
    ax.yaxis.set_tick_params(labelsize=6)
    # Show radial gridlines with light labels
    for r in [10, 20, 30, 40]:
        ax.text(np.deg2rad(90), r, f"{r}", fontsize=5, ha="center", va="bottom",
                color="gray", alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None,
                        help="Output path (default: saves to both ECCV and thesis)")
    args = parser.parse_args()

    # Collect data for all methods
    print("Collecting DINO NBV distributions...")
    all_data = {}
    for name, color, ep_dir in METHODS:
        print(f"  {name}: scanning {ep_dir}...")
        all_data[name] = collect_distributions(ep_dir)
        for step in sorted(all_data[name].keys()):
            total = sum(all_data[name][step].values())
            print(f"    Step {step}: n={total}")

    # Create figure: 4 rows x 3 cols
    fig = plt.figure(figsize=(14, 18))

    # Row titles
    row_configs = [
        ("Step 1 — First NBV Choice",  1, [0, 1, 2]),   # Random, FPS, LLM
        ("",                            1, [3, 4, 5]),   # SFT, GRPO, GSPO
        ("Step 2 — Second NBV Choice",  2, [0, 1, 2]),   # Random, FPS, LLM
        ("",                            2, [3, 4, 5]),   # SFT, GRPO, GSPO
    ]

    for row_idx, (row_title, step, method_indices) in enumerate(row_configs):
        for col_idx, mi in enumerate(method_indices):
            name, color, _ = METHODS[mi]
            ax_idx = row_idx * 3 + col_idx + 1
            ax = fig.add_subplot(4, 3, ax_idx, projection="polar")

            counts = all_data[name].get(step, {})
            make_polar_subplot(ax, counts, color, name)

    # Add section titles between row groups using figure coordinates
    fig.text(0.5, 0.98, "Step 1 — First NBV Choice",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.50, "Step 2 — Second NBV Choice",
             ha="center", va="top", fontsize=14, fontweight="bold")

    # Add note about dashed circle
    fig.text(0.95, 0.01, "Dashed circle = uniform baseline (20%)",
             ha="right", va="bottom", fontsize=8, fontstyle="italic", color="gray")

    plt.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=3.0)

    # Save per-step figure
    if args.output:
        out_paths = [args.output]
    else:
        out_paths = [
            "./figures/nbv_per_step.pdf",
            "./figures/nbv_per_step.pdf",
        ]

    for out_path in out_paths:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close()

    # ── Generate overall distribution figure (nbv_direction_distributions.pdf) ──
    print("\nGenerating overall distribution figure...")
    fig2 = plt.figure(figsize=(14, 9))

    # Row 1: Training-free, Row 2: Trained
    fig2.text(0.5, 0.98, "Training-free (Qwen3-VL-4B Attr + DINO)",
              ha="center", va="top", fontsize=13, fontweight="bold", fontstyle="italic")
    fig2.text(0.5, 0.50, "Trained Agents (DINO detection)",
              ha="center", va="top", fontsize=13, fontweight="bold", fontstyle="italic")

    row_methods = [
        [0, 1, 2],  # Random, FPS, LLM
        [3, 4, 5],  # SFT, GRPO, GSPO
    ]

    for row_idx, method_indices in enumerate(row_methods):
        for col_idx, mi in enumerate(method_indices):
            name, color, _ = METHODS[mi]
            ax_idx = row_idx * 3 + col_idx + 1
            ax = fig2.add_subplot(2, 3, ax_idx, projection="polar")

            # Aggregate all steps
            overall = defaultdict(int)
            for step_data in all_data[name].values():
                for d, c in step_data.items():
                    overall[d] += c
            make_polar_subplot(ax, dict(overall), color, name)

    fig2.text(0.95, 0.01, "Dashed circle = uniform baseline (20%)",
              ha="right", va="bottom", fontsize=8, fontstyle="italic", color="gray")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=3.0)

    overall_paths = [
        "./figures/nbv_direction_distributions.pdf",
        "./figures/nbv_direction_distributions.pdf",
    ]
    for out_path in overall_paths:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig2.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close()

    print("Done!")


if __name__ == "__main__":
    main()
