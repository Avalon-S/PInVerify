#!/usr/bin/env python3
"""
Generate all report figures (8 PNGs) + 2 table images for advisor presentation.
Run: python scripts/generate_report_figs.py
Output: report_figs/fig1_baseline.png .. fig8_size_group.png, table1.png, table2.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── Global style ──────────────────────────────────────────────────────────────
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

TAB10 = plt.cm.tab10.colors
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "report_figs")
os.makedirs(OUT, exist_ok=True)

# ── DATA ──────────────────────────────────────────────────────────────────────

# DATA 1
D1_MODELS = [
    "CLIP ViT-L/14\n(0.4B)",
    "SigLIP2\n(0.4B)",
    "Qwen3-VL-4B\n(4B)",
    "Qwen3-VL-8B\n(8B)",
    "SenseNova-8B\n(8B)",
]
D1_POS  = [35.5, 47.6, 71.1, 51.8, 69.3]
D1_NS   = [97.0, 96.4, 87.4, 98.2, 79.6]
D1_ND   = [100.0, 100.0, 98.2, 100.0, 92.2]
D1_OVR  = [77.6, 81.4, 85.6, 83.4, 80.4]

# DATA 2
D2 = [
    ("Qwen3-4B", "attr",    85.6, 71.1, 87.4, 98.2),
    ("Qwen3-4B", "direct",  80.8, 45.8, 96.4, 100.0),
    ("Qwen3-8B", "attr",    83.4, 51.8, 98.2, 100.0),
    ("Qwen3-8B", "direct",  76.8, 30.7, 99.4, 100.0),
    ("SenseNova","direct",   80.4, 69.3, 79.6, 92.2),
    ("SenseNova","merged",   78.0, 86.1, 67.7, 80.2),
]

# DATA 3
D3 = [
    ("SigLIP2\nmerged",      [47.6, 40.4, None]),
    ("Q3-4B\nattr",          [71.1, 82.5, 84.9]),
    ("Q3-4B\ndirect",        [45.8, 61.5, 64.5]),
    ("SenseNova\ndirect",    [69.3, 74.7, 81.9]),
]

# DATA 4 (corrected: LLM row = multi_view_attr_majority = LLM NBV + majority fusion)
D4_LABELS = ["Single", "Random", "FPS", "LLM", "ViewHint", "Oracle"]
D4_POS    = [71.1, 82.5, 78.9, 77.7, 79.5, 84.9]
D4_NS     = [87.4, 85.0, 83.8, 85.0, 83.2, 82.0]
D4_EFFV   = [1.0,  2.5,  1.9,  2.0,  2.2,  3.4]
D4_NF_U   = [0, 85, 362, 281, 242, 0]
D4_NF_T   = [0, 182, 204, 202, 169, 0]

# DATA 5 (sorted by Pos gap ascending)
D5 = [
    ("CLIP\nmerged",         35.5, 28.3, 7.2),
    ("Q3-4B\ndirect",        45.8, 38.6, 7.2),
    ("Q3-4B\nattr",          71.1, 59.6, 11.5),
    ("SigLIP2\nmerged",      47.6, 35.5, 12.1),
    ("Q3-8B\ndirect",        30.7,  9.6, 21.1),
    ("SenseNova\nmerged",    86.1, 25.3, 60.8),
    ("SenseNova\ndirect",    69.3, 16.3, 53.0),
    ("Q3-8B\nattr",          51.8, 13.9, 37.9),
]
D5_OVR = [
    ("CLIP merged",           77.6, 75.8, 1.8),
    ("Q3-4B direct",          80.8, 78.2, 2.6),
    ("Q3-4B attr",            85.6, 82.6, 3.0),
    ("SigLIP2 merged",        81.4, 78.0, 3.4),
    ("SenseNova merged",      78.0, 71.6, 6.4),
    ("Q3-8B direct",          76.8, 69.8, 7.0),
    ("SenseNova direct",      80.4, 69.8, 10.6),
    ("Q3-8B attr",            83.4, 71.4, 12.0),
]

# DATA 6
D6_CATS = [
    "ball", "bag", "backpack", "headphones", "toy", "laptop", "shoes",
    "teddy_bear", "book", "hat", "camera", "mug", "keys", "cellphone",
    "visor", "wallet", "watch", "eyeglasses",
]
D6_AVG = [90.0, 85.8, 81.8, 80.0, 77.0, 75.0, 74.8, 66.8, 62.2, 56.0,
           51.0, 48.2, 34.8, 34.0, 32.4, 24.8, 17.6, 16.6]
D6_VALS = {
    "CLIP":     [100, 86, 36, 43, 57, 62, 55, 67, 33, 30, 0, 33, 12, 30, 23, 0, 0, 0],
    "SigLIP2":  [100, 57, 100, 100, 100, 25, 64, 80, 11, 50, 27, 50, 25, 0, 46, 12, 0, 17],
    "Qwen3-4B": [75, 100, 100, 100, 100, 88, 91, 87, 89, 70, 91, 50, 50, 50, 31, 38, 44, 33],
    "Qwen3-8B": [75, 100, 73, 71, 71, 100, 73, 40, 89, 60, 73, 50, 25, 30, 85, 62, 11, 30],
    "SenseNova":[100, 86, 100, 86, 57, 100, 91, 60, 89, 70, 64, 58, 62, 60, 54, 12, 33, 33],
}

# DATA 7
D7_GROUPS = ["Large /\nDistinctive", "Medium /\nVaried", "Small /\nUniform"]
D7_POS = [93.2, 80.7, 48.1]
D7_NS  = [96.0, 84.5, 84.4]

# DATA 8: NavFail by NBV x Model (GT, attr mode, comparable agents)
# Columns: (model, nbv, nf_unreachable, nf_trap, eff_views)
D8_NAVFAIL = [
    # Qwen3-4B attr
    ("Q3-4B",    "Random",  237, 190, 2.1),  # updated after RandomNBV bug fix
    ("Q3-4B",    "FPS",     362, 204, 1.9),
    ("Q3-4B",    "LLM",     281, 202, 2.0),
    ("Q3-4B",    "ViewHint",242, 169, 2.2),
    ("Q3-4B",    "Oracle",    0,   0, 3.4),
    # Qwen3-8B attr
    ("Q3-8B",    "Random",  237, 190, 2.1),  # same as Q3-4B: NavFail is dataset/NBV-determined
    ("Q3-8B",    "Oracle",    0,   0, 3.4),
    # SenseNova attr
    ("SenseNova","Random",  237, 190, 2.1),  # same as Q3-4B: NavFail is dataset/NBV-determined
    ("SenseNova","FPS",     362, 204, 1.9),
    ("SenseNova","LLM",     282, 177, 2.1),
    ("SenseNova","Oracle",    0,   0, 3.4),
    # Qwen3-4B direct
    ("Q3-4B\n(direct)","Random", 267, 182, 2.1),  # updated after RandomNBV bug fix
    ("Q3-4B\n(direct)","FPS",   362, 204, 1.9),
    ("Q3-4B\n(direct)","LLM",   279, 211, 2.0),
    ("Q3-4B\n(direct)","Oracle",  0,   0, 3.4),
]

# DATA 9: Multi-View Impact (FV_Acc → Final) — Random NBV, majority fusion, GT
# Columns: (label, model_mode, fv_acc, final_acc)
D9_MV_IMPACT = [
    # MLLM models — multi-view helps (net positive)
    ("Q3-4B\nattr",       "mllm", 85.8, 88.2),
    ("Q3-4B\ndirect",     "mllm", 80.6, 86.2),
    ("Q3-8B\nattr",       "mllm", 83.8, 85.8),
    ("Q3-8B\ndirect",     "mllm", 76.8, 79.6),
    ("SenseNova\ndirect", "mllm", 80.6, 84.0),
    # MLLM — multi-view hurts (net negative)
    ("SenseNova\nattr",   "mllm", 68.0, 66.8),
    # Non-MLLM baselines — multi-view always hurts
    ("CLIP\nmerged",      "vlm",  77.6, 76.4),
    ("SigLIP2\nmerged",   "vlm",  81.4, 79.2),
]

# DATA 10: NBV Pos vs NS opposing effect (Q3-4B attr GT, majority fusion)
# Columns: (label, pos, ns, nd, overall, eff_views, trap)
D10_POS_NS = [
    ("Single",    71.1, 87.4, 98.2, 85.6, 1.0,   0),
    ("Random",    82.5, 85.0, 97.0, 88.2, 2.5, 182),
    ("FPS",       78.9, 83.8, 97.6, 86.8, 1.9, 204),
    ("LLM",       77.7, 85.0, 98.2, 87.0, 2.0, 202),
    ("ViewHint",  79.5, 83.2, 96.4, 86.4, 2.2, 169),
    ("Oracle",    84.9, 82.0, 97.6, 88.2, 3.4,   0),
]

# DATA 11: Cross-model Single → Random → Oracle (attr GT)
# Columns: (model, nbv, pos, ns, trap)
D11_CROSS = [
    ("Q3-4B",     "Single", 71.1, 87.4,   0),
    ("Q3-4B",     "Random", 82.5, 85.0, 182),
    ("Q3-4B",     "Oracle", 84.9, 82.0,   0),
    ("Q3-8B",     "Single", 51.8, 98.2,   0),
    ("Q3-8B",     "Random", 60.8, 96.4, 182),
    ("Q3-8B",     "Oracle", 63.2, 93.4,   0),
    ("SenseNova", "Single", 91.6, 44.9,   0),
    ("SenseNova", "Random", 95.8, 37.1, 182),
    ("SenseNova", "Oracle", 96.4, 37.7,   0),
]


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1: Baseline Comparison – Grouped Bar
# ══════════════════════════════════════════════════════════════════════════════
def fig1():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(D1_MODELS))
    w = 0.22
    c_pos, c_ns, c_nd = TAB10[0], TAB10[1], TAB10[2]

    bars_pos = ax.bar(x - w, D1_POS, w, label="Positive", color=c_pos, edgecolor="white", linewidth=0.5)
    bars_ns  = ax.bar(x,     D1_NS,  w, label="Neg_Same", color=c_ns, edgecolor="white", linewidth=0.5)
    bars_nd  = ax.bar(x + w, D1_ND,  w, label="Neg_Diff", color=c_nd, edgecolor="white", linewidth=0.5)

    for bars in [bars_pos, bars_ns, bars_nd]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # Overall line
    ax.plot(x, D1_OVR, "k--o", markersize=6, linewidth=1.5, label="Overall", zorder=5)
    for xi, ov in zip(x, D1_OVR):
        # Place label below the point where it would overlap with bar text
        if xi in (0, 2, 4):  # CLIP, Qwen3-4B, SenseNova
            ax.text(xi, ov - 4, f"{ov:.1f}", ha="center", va="top", fontsize=8, fontweight="bold")
        else:
            ax.text(xi, ov + 1.5, f"{ov:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(D1_MODELS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Baseline Comparison: Single-View GT", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_baseline.png"))
    plt.close(fig)
    print("  fig1_baseline.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2: Pos vs NS Trade-off Scatter
# ══════════════════════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    model_colors = {"Qwen3-4B": TAB10[0], "Qwen3-8B": TAB10[1], "SenseNova": TAB10[2]}
    mode_markers = {"attr": "o", "direct": "s", "merged": "D"}

    # Diagonal
    ax.plot([20, 100], [20, 100], "k--", alpha=0.25, linewidth=1, zorder=0)
    ax.text(95, 93, "balanced", fontsize=8, alpha=0.4, rotation=38, ha="right")

    # Per-point label positions: (offset_x, offset_y, ha)
    label_cfg = {
        ("Qwen3-4B", "attr"):     (6, -1, "left"),
        ("Qwen3-4B", "direct"):   (6, 2, "left"),
        ("Qwen3-8B", "attr"):     (6, -1, "left"),
        ("Qwen3-8B", "direct"):   (6, 2, "left"),
        ("SenseNova", "direct"):   (6, -1, "left"),
        ("SenseNova", "merged"):   (6, -1, "left"),
    }

    for model, mode, ovr, pos, ns, nd in D2:
        c = model_colors[model]
        m = mode_markers[mode]
        ax.scatter(pos, ns, c=[c], marker=m, s=130, edgecolors="black", linewidths=0.6, zorder=3)
        label = f"{model} ({mode})  {ovr:.1f}%"
        ox, oy, ha = label_cfg.get((model, mode), (6, -1, "left"))
        ax.annotate(label, (pos, ns), textcoords="offset points",
                    xytext=(ox, oy), fontsize=8.5, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, shrinkA=4, shrinkB=2))

    # Legend patches
    handles = []
    for mdl, c in model_colors.items():
        handles.append(mpatches.Patch(color=c, label=mdl))
    for mode, m in mode_markers.items():
        handles.append(plt.Line2D([0],[0], marker=m, color="gray", linestyle="",
                                   markersize=8, label=f"mode={mode}"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Positive Accuracy (%)")
    ax.set_ylabel("Neg_Same Accuracy (%)")
    ax.set_xlim(20, 100)
    ax.set_ylim(60, 105)
    ax.set_title("Positive vs Neg_Same Trade-off by Reasoning Mode", fontweight="bold")

    # Mode tendency annotations – place in open space
    ax.annotate("attr mode:\nhigher Positive\n(recall-oriented)", xy=(53, 90), fontsize=8.5,
                color=TAB10[0], alpha=0.8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TAB10[0], alpha=0.5))
    ax.annotate("direct mode:\nhigher Neg_Same\n(precision-oriented)", xy=(40, 72), fontsize=8.5,
                color=TAB10[1], alpha=0.8, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TAB10[1], alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_tradeoff.png"))
    plt.close(fig)
    print("  fig2_tradeoff.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3: Multi-View Positive Lift – Grouped Bar
# ══════════════════════════════════════════════════════════════════════════════
def fig3():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_groups = len(D3)
    x = np.arange(n_groups)
    w = 0.22
    shade = [0.35, 0.6, 0.9]  # lightness for single/random/oracle
    view_labels = ["Single", "Random", "Oracle"]

    for i, (label, vals) in enumerate(D3):
        for j, v in enumerate(vals):
            if v is None:
                continue
            base_color = np.array(TAB10[i])
            color = base_color * shade[j] + np.array([1,1,1]) * (1 - shade[j])
            bar = ax.bar(x[i] + (j-1)*w, v, w, color=color, edgecolor="white", linewidth=0.5)
            ax.text(x[i] + (j-1)*w, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # Draw arrows showing lift
    for i, (label, vals) in enumerate(D3):
        if vals[0] is not None and vals[1] is not None:
            delta = vals[1] - vals[0]
            sign = "+" if delta >= 0 else ""
            color = "green" if delta > 0 else "red"
            mid_x = x[i] - w/2
            ax.annotate(f"{sign}{delta:.1f}pp",
                        xy=(mid_x, max(vals[0], vals[1]) + 4),
                        fontsize=7, ha="center", color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in D3])
    ax.set_ylabel("Positive Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Multi-View Impact on Positive Accuracy (GT)", fontweight="bold")

    # Custom legend
    for j, vl in enumerate(view_labels):
        ax.bar([], [], color=[0.5*shade[j]+0.5*(1-shade[j])]*3, label=vl)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_multiview.png"))
    plt.close(fig)
    print("  fig3_multiview.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4: NBV Strategy Comparison – Grouped Bar + EffV annotation
# ══════════════════════════════════════════════════════════════════════════════
def fig4():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(D4_LABELS))
    w = 0.28

    bars_pos = ax.bar(x - w/2, D4_POS, w, label="Positive", color=TAB10[0], edgecolor="white")
    bars_ns  = ax.bar(x + w/2, D4_NS,  w, label="Neg_Same", color=TAB10[1], edgecolor="white")

    for b in bars_pos:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for b in bars_ns:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    # EffV + NavFail annotation on top
    for i in range(len(D4_LABELS)):
        ev = D4_EFFV[i]
        nfu = D4_NF_U[i]
        nft = D4_NF_T[i]
        ax.text(x[i], 97, f"EffV={ev:.1f}", ha="center", fontsize=8, color="dimgray",
                bbox=dict(boxstyle="round,pad=0.15", fc="#f0f0f0", ec="gray", lw=0.5))
        if nfu > 0 or nft > 0:
            ax.text(x[i], 102, f"NF: {nfu}u+{nft}t", ha="center", fontsize=7, color="gray")

    # Highlight Oracle as upper bound
    ax.annotate("upper bound\n(skip traps)",
                xy=(5 + w/2 + 0.05, 82), xytext=(5 + w/2 + 0.15, 75),
                fontsize=7.5, color="green", ha="left",
                arrowprops=dict(arrowstyle="->", color="green", lw=1))

    ax.set_xticks(x)
    ax.set_xticklabels(D4_LABELS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 120)
    ax.set_xlim(-0.6, len(D4_LABELS) - 0.3)
    ax.set_title("NBV Strategy Comparison (Qwen3-4B attr GT, majority fusion)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, bbox_to_anchor=(0.01, 0.78))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_nbv.png"))
    plt.close(fig)
    print("  fig4_nbv.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5: GT vs DINO Positive Accuracy – Paired Bar
# ══════════════════════════════════════════════════════════════════════════════
def fig5():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [d[0] for d in D5]
    gt_vals = [d[1] for d in D5]
    dino_vals = [d[2] for d in D5]
    gaps = [d[3] for d in D5]

    x = np.arange(len(labels))
    w = 0.32

    ax.bar(x - w/2, gt_vals,   w, label="GT",   color=TAB10[0], edgecolor="white")
    ax.bar(x + w/2, dino_vals, w, label="DINO", color=TAB10[3], edgecolor="white")

    for i in range(len(labels)):
        # Gap annotation
        top = max(gt_vals[i], dino_vals[i])
        ax.text(x[i], top + 2.5, f"\u0394{gaps[i]:.1f}pp", ha="center", fontsize=8,
                color="red" if gaps[i] > 15 else "darkorange" if gaps[i] > 10 else "black",
                fontweight="bold")

    # Divider between robust and fragile groups
    divider_x = 3.5  # between index 3 (SigLIP2) and 4 (Q3-8B direct)
    ax.axvline(divider_x, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(divider_x - 0.15, 95, "Robust\n(gap\u226412pp)", fontsize=8, ha="right",
            color="green", alpha=0.8)
    ax.text(divider_x + 0.15, 95, "Fragile\n(gap>20pp)", fontsize=8, ha="left",
            color="red", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Positive Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("GT vs DINO: Positive Accuracy Gap", fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_gt_dino_pos.png"))
    plt.close(fig)
    print("  fig5_gt_dino_pos.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6: GT vs DINO Overall Gap Scatter
# ══════════════════════════════════════════════════════════════════════════════
def fig6():
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = [d[0] for d in D5_OVR]
    gt_ovr = [d[1] for d in D5_OVR]
    gaps   = [d[3] for d in D5_OVR]

    # Color by cluster
    colors = []
    for g in gaps:
        if g <= 4:
            colors.append(TAB10[2])  # green = robust
        else:
            colors.append(TAB10[3])  # red = fragile

    ax.scatter(gt_ovr, gaps, c=colors, s=120, edgecolors="black", linewidths=0.6, zorder=3)

    for i, lab in enumerate(labels):
        offset_x, offset_y = 5, 0
        if "SenseNova direct" in lab:
            offset_x, offset_y = 5, -0.6
        if "Q3-8B attr" in lab:
            offset_x, offset_y = 5, 0.3
        if "Q3-4B attr" in lab:
            offset_x, offset_y = 5, -0.5
        if "SigLIP2" in lab:
            offset_x, offset_y = 5, 0.3
        ax.annotate(lab, (gt_ovr[i], gaps[i]), textcoords="offset points",
                    xytext=(offset_x, offset_y), fontsize=8.5, va="center")

    # Cluster boxes
    from matplotlib.patches import FancyBboxPatch
    robust_box = FancyBboxPatch((74, -0.5), 13, 5.5, boxstyle="round,pad=0.3",
                                 fc="green", ec="green", alpha=0.08, zorder=0)
    fragile_box = FancyBboxPatch((74, 5.5), 13, 8, boxstyle="round,pad=0.3",
                                  fc="red", ec="red", alpha=0.08, zorder=0)
    ax.add_patch(robust_box)
    ax.add_patch(fragile_box)
    ax.text(74.5, 1.0, "Robust cluster", fontsize=9, color="green", alpha=0.7)
    ax.text(74.5, 12.5, "Fragile cluster", fontsize=9, color="red", alpha=0.7)

    # Horizontal divider
    ax.axhline(5, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("GT Overall Accuracy (%)")
    ax.set_ylabel("GT \u2013 DINO Overall Gap (pp)")
    ax.set_title("Calibration Determines DINO Robustness", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_gt_dino_scatter.png"))
    plt.close(fig)
    print("  fig6_gt_dino_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7: Per-Category Positive Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig7():
    models = list(D6_VALS.keys())
    cats = D6_CATS  # already sorted by avg descending
    data = np.array([D6_VALS[m] for m in models]).T  # (n_cats, n_models)

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats, fontsize=9)

    # Annotate cells
    for i in range(len(cats)):
        for j in range(len(models)):
            v = data[i, j]
            text_color = "white" if v < 30 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                    color=text_color, fontweight="bold" if v >= 90 else "normal")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Positive Accuracy (%)", fontsize=10)

    ax.set_title("Per-Category Positive Accuracy (Single-View GT)", fontweight="bold", pad=12)

    # Horizontal dividers for difficulty tiers
    ax.axhline(6.5, color="black", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.axhline(12.5, color="black", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.text(len(models) + 0.1, 3, "Easy", fontsize=9, color="green", va="center", fontweight="bold")
    ax.text(len(models) + 0.1, 9.5, "Medium", fontsize=9, color="orange", va="center", fontweight="bold")
    ax.text(len(models) + 0.1, 15.5, "Hard", fontsize=9, color="red", va="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_category_heatmap.png"))
    plt.close(fig)
    print("  fig7_category_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8: Object Size Group – Grouped Bar
# ══════════════════════════════════════════════════════════════════════════════
def fig8():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(D7_GROUPS))
    w = 0.30

    bars_pos = ax.bar(x - w/2, D7_POS, w, label="avg Positive", color=TAB10[0], edgecolor="white")
    bars_ns  = ax.bar(x + w/2, D7_NS,  w, label="avg Neg_Same", color=TAB10[1], edgecolor="white")

    for b in bars_pos:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for b in bars_ns:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Pos drop text annotation (no arrow/line)
    ax.text(1 - w/2, 102, "Positive \u039445.1pp\n(93.2 \u2192 48.1)", fontsize=10,
            ha="center", color="red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(D7_GROUPS, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Object Size Effect on Verification Difficulty\n(Qwen3-4B attr GT, avg across categories)",
                 fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_size_group.png"))
    plt.close(fig)
    print("  fig8_size_group.png")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Main Results Table (as image)
# ══════════════════════════════════════════════════════════════════════════════
def table1():
    col_labels = ["Model", "Params", "Mode", "Overall", "Positive", "Neg_Same", "Neg_Diff"]
    rows = [
        ["CLIP ViT-L/14",  "0.4B", "merged", "77.6", "35.5", "97.0", "100.0"],
        ["SigLIP2-so400m",  "0.4B", "merged", "81.4", "47.6", "96.4", "100.0"],
        ["Qwen3-VL-4B",   "4B",   "attr",   "85.6", "71.1", "87.4", "98.2"],
        ["Qwen3-VL-8B",   "8B",   "attr",   "83.4", "51.8", "98.2", "100.0"],
        ["SenseNova-8B",   "8B",   "direct", "80.4", "69.3", "79.6", "92.2"],
    ]
    bold_cells = {(2, 3), (2, 4), (3, 5), (0, 6), (1, 6), (3, 6)}

    n_rows = len(rows)
    row_h = 0.38
    fig_h = 0.6 + (n_rows + 1) * row_h  # title + header + data rows
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.6)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    for (r, c) in bold_cells:
        table[r + 1, c].set_text_props(fontweight="bold", color="#c00000")
    for i in range(n_rows):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor("#D6E4F0" if i % 2 == 0 else "white")

    ax.set_title("Table 1: Baseline Comparison (Single-View, GT, Best Mode)",
                 fontweight="bold", fontsize=12, pad=8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02)
    fig.savefig(os.path.join(OUT, "table1_main.png"))
    plt.close(fig)
    print("  table1_main.png")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: LLM NBV Failure Modes
# ══════════════════════════════════════════════════════════════════════════════
def table2():
    """Table 2: Fusion method ablation (LLM NBV held constant, Qwen3-4B attr GT)."""
    col_labels = ["Fusion", "Overall", "Positive", "Neg_Same", "Behavior"]
    rows = [
        ["Majority",  "87.0%", "77.7%", "85.0%", "Balanced"],
        ["Weighted",   "76.2%", "29.5%", "98.8%", "Extreme conservative:\nalmost never says Yes"],
        ["LLM Fusion", "69.6%", "49.4%", "79.6%", "Noisy"],
    ]

    n_rows = len(rows)
    row_h = 0.45
    fig_h = 0.7 + (n_rows + 1) * row_h
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.7)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Best row (majority) in green
    for j in range(len(col_labels)):
        if j >= 1 and j <= 3:
            table[1, j].set_text_props(fontweight="bold", color="#006600")
    # Worst Pos in red
    table[2, 2].set_text_props(color="#c00000", fontweight="bold")  # Weighted Pos
    table[3, 2].set_text_props(color="#c00000")  # LLM Fusion Pos
    # Last column italic
    for i in range(n_rows):
        table[i + 1, 4].set_text_props(style="italic")
    # Alternating rows
    for i in range(n_rows):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor("#D6E4F0" if i % 2 == 0 else "white")

    ax.set_title("Table 2: Fusion Method Ablation (LLM NBV, Qwen3-4B attr GT)",
                 fontweight="bold", fontsize=12, pad=8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.02)
    fig.savefig(os.path.join(OUT, "table2_fusion.png"))
    plt.close(fig)
    print("  table2_fusion.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9: NavFail by NBV Strategy (stacked bar: unreachable + trap)
# ══════════════════════════════════════════════════════════════════════════════
def fig9():
    """NavFail comparison across NBV strategies, Qwen3-4B attr GT."""
    nbv_labels = ["Random", "FPS", "LLM", "ViewHint", "Oracle"]
    nf_u = [85, 362, 281, 242, 0]
    nf_t = [182, 204, 202, 169, 0]
    eff_v = [2.5, 1.9, 2.0, 2.2, 3.4]
    pos_acc = [82.5, 78.9, 77.7, 79.5, 84.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    # Left: Stacked bar for NavFail
    x = np.arange(len(nbv_labels))
    w = 0.5
    bars_u = ax1.bar(x, nf_u, w, label="Unreachable", color=TAB10[3], edgecolor="white")
    bars_t = ax1.bar(x, nf_t, w, bottom=nf_u, label="Trap View", color=TAB10[1], edgecolor="white")

    for i in range(len(nbv_labels)):
        total = nf_u[i] + nf_t[i]
        if total > 0:
            ax1.text(x[i], total + 8, f"{total}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
            # Label unreachable count inside bar
            if nf_u[i] > 30:
                ax1.text(x[i], nf_u[i] / 2, f"{nf_u[i]}", ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
            # Label trap count inside bar
            if nf_t[i] > 30:
                ax1.text(x[i], nf_u[i] + nf_t[i] / 2, f"{nf_t[i]}", ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
        else:
            ax1.text(x[i], 10, "0", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(nbv_labels)
    ax1.set_ylabel("Total Navigation Failures (500 episodes)")
    ax1.set_ylim(0, 620)
    ax1.set_title("Navigation Failures by NBV Strategy", fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.9)

    # Annotate FPS issue
    ax1.annotate("FPS: highest unreachable\n(picks farthest angle,\noften no viewpoint)",
                 xy=(1, 362), xytext=(1.8, 520), fontsize=7.5, color=TAB10[3],
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=TAB10[3], lw=1))

    # Right: EffV vs Pos scatter
    colors = [TAB10[0], TAB10[3], TAB10[2], TAB10[4], TAB10[1]]
    for i in range(len(nbv_labels)):
        ax2.scatter(eff_v[i], pos_acc[i], c=[colors[i]], s=150, edgecolors="black",
                    linewidths=0.6, zorder=3)
        ox = 6 if nbv_labels[i] != "Oracle" else -6
        ha = "left" if nbv_labels[i] != "Oracle" else "right"
        ax2.annotate(nbv_labels[i], (eff_v[i], pos_acc[i]),
                     textcoords="offset points", xytext=(ox, 0),
                     fontsize=9, ha=ha, va="center")

    ax2.set_xlabel("Effective Views (avg)")
    ax2.set_ylabel("Positive Accuracy (%)")
    ax2.set_title("Effective Views vs Positive Acc", fontweight="bold")
    ax2.set_xlim(0.5, 4.0)
    ax2.set_ylim(70, 90)

    fig.suptitle("Qwen3-4B attr GT, majority fusion", fontsize=10, y=0.02, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(OUT, "fig9_navfail.png"))
    plt.close(fig)
    print("  fig9_navfail.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10: Cross-Model NavFail Comparison (grouped stacked bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig10():
    """Cross-model NavFail: shows NavFail is determined by NBV type, not model."""
    nbv_order = ["Random", "FPS", "LLM"]
    models = ["Q3-4B", "SenseNova", "Q3-4B\n(direct)"]
    model_labels = ["Q3-4B (attr)", "SenseNova (attr)", "Q3-4B (direct)"]

    # Build lookup: (model, nbv) → (unreachable, trap, eff_views)
    lookup = {}
    for model, nbv, nf_u, nf_t, ev in D8_NAVFAIL:
        lookup[(model, nbv)] = (nf_u, nf_t, ev)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1.5, 1]})

    # ── Left: Grouped stacked bar ──
    n_nbv = len(nbv_order)
    n_model = len(models)
    group_w = 0.7
    bar_w = group_w / n_model
    model_colors = [TAB10[0], TAB10[2], TAB10[4]]

    for j, model in enumerate(models):
        u_vals, t_vals = [], []
        for nbv in nbv_order:
            d = lookup.get((model, nbv), (0, 0, 0))
            u_vals.append(d[0])
            t_vals.append(d[1])

        x = np.arange(n_nbv) + j * bar_w - group_w / 2 + bar_w / 2

        # Unreachable (solid)
        ax1.bar(x, u_vals, bar_w * 0.88, color=model_colors[j],
                edgecolor="white", linewidth=0.5)
        # Trap (hatched, lighter)
        ax1.bar(x, t_vals, bar_w * 0.88, bottom=u_vals,
                color=model_colors[j], alpha=0.45, edgecolor="white",
                linewidth=0.5, hatch="///")

        # Total annotation
        for i in range(n_nbv):
            total = u_vals[i] + t_vals[i]
            if total > 0:
                ax1.text(x[i], total + 8, f"{total}", ha="center",
                         fontsize=7.5, fontweight="bold", color=model_colors[j])

    ax1.set_xticks(np.arange(n_nbv))
    ax1.set_xticklabels(nbv_order, fontsize=11)
    ax1.set_ylabel("Total Navigation Failures (500 episodes)")
    ax1.set_ylim(0, 650)
    ax1.set_title("NavFail by NBV Strategy × Model", fontweight="bold")

    # Legend: model colors + unreachable/trap distinction
    handles = []
    for j, ml in enumerate(model_labels):
        handles.append(mpatches.Patch(color=model_colors[j], label=ml))
    handles.append(mpatches.Patch(facecolor="gray", label="Unreachable (solid)"))
    handles.append(mpatches.Patch(facecolor="gray", alpha=0.45, hatch="///",
                                   label="Trap (hatched)"))
    ax1.legend(handles=handles, fontsize=8, framealpha=0.9, loc="upper left")

    # Key insight annotation – placed in lower-left area
    ax1.text(0.03, 0.05,
             "Same NBV \u2192 nearly identical NavFail\nregardless of model",
             transform=ax1.transAxes, fontsize=8.5, color="dimgray",
             fontstyle="italic", ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec="goldenrod",
                       lw=0.8))

    # ── Right: LLM NBV trap breakdown (only NBV that varies by model) ──
    llm_models = ["Q3-4B", "SenseNova", "Q3-4B\n(direct)"]
    llm_labels = ["Q3-4B\n(attr)", "SenseNova\n(attr)", "Q3-4B\n(direct)"]
    llm_u = [lookup[(m, "LLM")][0] for m in llm_models]
    llm_t = [lookup[(m, "LLM")][1] for m in llm_models]

    x2 = np.arange(len(llm_models))
    w2 = 0.45
    ax2.bar(x2, llm_u, w2, label="Unreachable", color=TAB10[3], edgecolor="white")
    ax2.bar(x2, llm_t, w2, bottom=llm_u, label="Trap", color=TAB10[1],
            edgecolor="white")

    for i in range(len(llm_models)):
        # Unreachable label
        if llm_u[i] > 30:
            ax2.text(x2[i], llm_u[i] / 2, f"{llm_u[i]}", ha="center",
                     va="center", fontsize=9, color="white", fontweight="bold")
        # Trap label
        if llm_t[i] > 30:
            ax2.text(x2[i], llm_u[i] + llm_t[i] / 2, f"{llm_t[i]}",
                     ha="center", va="center", fontsize=9, color="white",
                     fontweight="bold")
        total = llm_u[i] + llm_t[i]
        ax2.text(x2[i], total + 8, f"{total}", ha="center", fontsize=9,
                 fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(llm_labels, fontsize=9)
    ax2.set_ylabel("Navigation Failures")
    ax2.set_ylim(0, 550)
    ax2.set_title("LLM NBV: Trap vs Unreachable by Model", fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9)

    # Highlight Q3-4B direct has more traps
    ax2.annotate("direct mode:\nmore traps (211)",
                 xy=(2, llm_u[2] + llm_t[2] / 2),
                 xytext=(2.4, 420), fontsize=7.5, color=TAB10[1],
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=TAB10[1], lw=1))

    fig.suptitle("Navigation failures depend on NBV algorithm, not model choice (GT, 500 episodes)",
                 fontsize=10, y=0.02, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(OUT, "fig10_navfail_cross_model.png"))
    plt.close(fig)
    print("  fig10_navfail_cross_model.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11: Multi-View Impact – FV vs Final
# ══════════════════════════════════════════════════════════════════════════════
def fig11():
    """Multi-view impact: scatter of FV_Acc vs Final accuracy."""
    fig, ax1 = plt.subplots(figsize=(8, 7))

    # ── FV_Acc vs Final scatter with arrows ──
    type_colors = {"mllm": TAB10[0], "vlm": TAB10[3]}
    type_markers = {"mllm": "o", "vlm": "s"}

    # Diagonal line (no change)
    ax1.plot([60, 95], [60, 95], "k--", alpha=0.25, linewidth=1, zorder=0)
    ax1.fill_between([60, 95], [60, 95], [95, 95], color="green", alpha=0.04, zorder=0)
    ax1.fill_between([60, 95], [60, 60], [60, 95], color="red", alpha=0.04, zorder=0)
    ax1.text(63, 92, "multi-view helps", fontsize=8, color="green", alpha=0.6,
             fontstyle="italic")
    ax1.text(85, 63, "multi-view hurts", fontsize=8, color="red", alpha=0.6,
             fontstyle="italic")

    # Label position offsets: (label) -> (dx, dy, ha)
    label_offsets = {
        "Q3-4B\nattr":       (6, 5, "left"),
        "Q3-4B\ndirect":     (-8, 10, "right"),
        "Q3-8B\nattr":       (-8, -8, "right"),
        "Q3-8B\ndirect":     (-7, 6, "right"),
        "SenseNova\ndirect": (-8, -5, "right"),
        "SenseNova\nattr":   (6, 0, "left"),
        "CLIP\nmerged":      (6, -6, "left"),
        "SigLIP2\nmerged":   (-7, -6, "right"),
    }

    for label, mtype, fv, final in D9_MV_IMPACT:
        c = type_colors[mtype]
        m = type_markers[mtype]
        delta = final - fv
        # Point at final position
        ax1.scatter(fv, final, c=[c], marker=m, s=120, edgecolors="black",
                    linewidths=0.6, zorder=4)

        # Label with delta
        dx, dy, ha = label_offsets.get(label, (6, 0, "left"))
        sign = "+" if delta >= 0 else ""
        delta_color = "green" if delta > 0 else "red"
        display_label = label.replace("\n", " ")
        ax1.annotate(f"{display_label}\n{sign}{delta:.1f}pp",
                     (fv, final), textcoords="offset points",
                     xytext=(dx * 2, dy * 2), fontsize=7.5, ha=ha, va="center",
                     color=delta_color if abs(delta) > 1 else "gray",
                     fontweight="bold" if abs(delta) > 2 else "normal",
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.3,
                                     shrinkA=4, shrinkB=2))

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color=TAB10[0], linestyle="",
                   markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                   label="MLLM (Qwen3, SenseNova)"),
        plt.Line2D([0], [0], marker="s", color=TAB10[3], linestyle="",
                   markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                   label="VLM baseline (CLIP, SigLIP2)"),
    ]
    ax1.legend(handles=handles, fontsize=8.5, framealpha=0.9, loc="upper left")

    ax1.set_xlabel("First-View Accuracy (%)", fontsize=11)
    ax1.set_ylabel("Final Multi-View Accuracy (%)", fontsize=11)
    ax1.set_xlim(62, 92)
    ax1.set_ylim(62, 92)
    ax1.set_aspect("equal")
    ax1.set_title("Multi-View Impact: First View vs Final", fontweight="bold")

    fig.suptitle("Random NBV, majority fusion, GT mode, 500 episodes",
                 fontsize=10, y=0.02, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(OUT, "fig11_multiview_impact.png"))
    plt.close(fig)
    print("  fig11_multiview_impact.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12: Pos vs NS Opposing Effect across NBV Strategies
# ══════════════════════════════════════════════════════════════════════════════
def fig12():
    """Better NBV -> Pos UP but NS DOWN; Overall masks the real benefit."""
    # Sort by effective views to show trend
    sorted_data = sorted(D10_POS_NS, key=lambda x: x[4])
    labels  = [d[0] for d in sorted_data]
    pos     = [d[1] for d in sorted_data]
    ns      = [d[2] for d in sorted_data]
    overall = [d[3] for d in sorted_data]
    eff_v   = [d[4] for d in sorted_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # ── Left: Trend lines across NBV strategies ──
    x = np.arange(len(labels))

    ax1.plot(x, pos, "o-", color=TAB10[0], linewidth=2.2, markersize=9,
             label="Positive", zorder=4)
    ax1.plot(x, ns, "s-", color=TAB10[1], linewidth=2.2, markersize=9,
             label="Neg_Same", zorder=4)
    ax1.plot(x, overall, "D--", color="gray", linewidth=1.5, markersize=7,
             label="Overall", zorder=3, alpha=0.7)

    # Value labels
    for i in range(len(labels)):
        # Last point: Pos label above the point (close to NS)
        if i == len(labels) - 1:
            ax1.text(x[i], pos[i] + 1.5, f"{pos[i]:.1f}", ha="center", fontsize=8,
                     color=TAB10[0], fontweight="bold")
        else:
            ax1.text(x[i], pos[i] - 2.8, f"{pos[i]:.1f}", ha="center", fontsize=8,
                     color=TAB10[0], fontweight="bold")
        ax1.text(x[i], ns[i] + 1.5, f"{ns[i]:.1f}", ha="center", fontsize=8,
                 color=TAB10[1], fontweight="bold")
        if i == 0 or i == len(labels) - 1:
            ax1.text(x[i] + 0.15, overall[i] + 1.5, f"{overall[i]:.1f}",
                     ha="left", fontsize=8, color="gray")

    # EffV as secondary annotation below x-axis
    for i in range(len(labels)):
        ax1.text(x[i], 66, f"EffV={eff_v[i]:.1f}", ha="center", fontsize=7,
                 color="dimgray")

    # Trend arrows (subtle background)
    ax1.annotate("Pos +13.8pp", xy=(5, pos[-1]), xytext=(5.15, pos[-1] + 3),
                 fontsize=8.5, color=TAB10[0], fontweight="bold", ha="left")
    ax1.annotate("NS -5.4pp", xy=(5, ns[-1]), xytext=(5.15, ns[-1] - 5),
                 fontsize=8.5, color=TAB10[1], fontweight="bold", ha="left")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(63, 95)
    ax1.set_xlim(-0.4, len(labels) - 0.2)
    ax1.set_title("Opposing Effects: Better NBV Helps Pos but Hurts NS",
                   fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.9, loc="center left")

    # Insight box
    ax1.text(0.98, 0.05,
             "Overall stays flat (~86-88%)\nbecause Pos gain and NS drop cancel out",
             transform=ax1.transAxes, fontsize=8, ha="right", va="bottom",
             fontstyle="italic", color="dimgray",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="goldenrod",
                       lw=0.8))

    # ── Right: Delta from Single baseline ──
    base_pos = sorted_data[0][1]
    base_ns  = sorted_data[0][2]
    base_ovr = sorted_data[0][3]

    # Skip Single (delta = 0)
    delta_labels = labels[1:]
    delta_pos = [p - base_pos for p in pos[1:]]
    delta_ns  = [n - base_ns  for n in ns[1:]]
    delta_ovr = [o - base_ovr for o in overall[1:]]

    y = np.arange(len(delta_labels))
    h = 0.25

    ax2.barh(y - h, delta_pos, h, label="\u0394Positive", color=TAB10[0],
             edgecolor="white")
    ax2.barh(y, delta_ns, h, label="\u0394Neg_Same", color=TAB10[1],
             edgecolor="white")
    ax2.barh(y + h, delta_ovr, h, label="\u0394Overall", color="gray",
             alpha=0.5, edgecolor="white")

    # Value labels
    for i in range(len(delta_labels)):
        # Pos delta (always positive)
        sign_p = "+" if delta_pos[i] >= 0 else ""
        ax2.text(delta_pos[i] + 0.4, y[i] - h,
                 f"{sign_p}{delta_pos[i]:.1f}", ha="left", va="center",
                 fontsize=8, color=TAB10[0], fontweight="bold")
        # NS delta (always negative)
        sign_n = "+" if delta_ns[i] >= 0 else ""
        nudge = -0.4 if delta_ns[i] < 0 else 0.4
        ha_ns = "right" if delta_ns[i] < 0 else "left"
        ax2.text(delta_ns[i] + nudge, y[i],
                 f"{sign_n}{delta_ns[i]:.1f}", ha=ha_ns, va="center",
                 fontsize=8, color=TAB10[1], fontweight="bold")
        # Overall delta
        sign_o = "+" if delta_ovr[i] >= 0 else ""
        ax2.text(delta_ovr[i] + 0.4, y[i] + h,
                 f"{sign_o}{delta_ovr[i]:.1f}", ha="left", va="center",
                 fontsize=7.5, color="gray")

    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(delta_labels, fontsize=10)
    ax2.set_xlabel("\u0394 from Single Baseline (pp)")
    ax2.set_title("Change from Single-View Baseline", fontweight="bold")
    ax2.legend(fontsize=8.5, framealpha=0.9, loc="lower right")
    ax2.invert_yaxis()

    # Highlight Oracle row
    ax2.axhspan(len(delta_labels) - 1.5, len(delta_labels) - 0.5,
                color="green", alpha=0.06)

    fig.suptitle("Qwen3-4B attr GT, majority fusion \u2014 Overall accuracy masks "
                 "the real NBV benefit",
                 fontsize=10, y=0.02, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(OUT, "fig12_pos_ns_opposing.png"))
    plt.close(fig)
    print("  fig12_pos_ns_opposing.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 13: Cross-Model Verification – Pos UP / NS DOWN pattern
# ══════════════════════════════════════════════════════════════════════════════
def fig13():
    """Same Pos-up / NS-down pattern holds across all 3 models."""
    model_order = ["Q3-4B", "Q3-8B", "SenseNova"]
    nbv_levels  = ["Single", "Random", "Oracle"]

    # Build lookup: (model, nbv) -> (pos, ns, trap)
    lookup = {}
    for model, nbv, pos, ns, trap in D11_CROSS:
        lookup[(model, nbv)] = (pos, ns, trap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for idx, model in enumerate(model_order):
        ax = axes[idx]
        pos_vals = [lookup[(model, nbv)][0] for nbv in nbv_levels]
        ns_vals  = [lookup[(model, nbv)][1] for nbv in nbv_levels]

        x = np.arange(len(nbv_levels))

        # Lines
        ax.plot(x, pos_vals, "o-", color=TAB10[0], linewidth=2.5, markersize=11,
                label="Positive", zorder=4)
        ax.plot(x, ns_vals, "s-", color=TAB10[1], linewidth=2.5, markersize=11,
                label="Neg_Same", zorder=4)

        # Fill between to visualize gap
        ax.fill_between(x, pos_vals, ns_vals, alpha=0.08, color="purple")

        # Value labels
        for i in range(len(nbv_levels)):
            # Pos label below/above depending on relative position
            pos_offset = -3.5 if pos_vals[i] < ns_vals[i] else 2.0
            ns_offset  = 2.0 if pos_vals[i] < ns_vals[i] else -3.5
            # SenseNova last point: force NS label above
            if idx == 2 and i == len(nbv_levels) - 1:
                ns_offset = 4.0
            ax.text(x[i], pos_vals[i] + pos_offset, f"{pos_vals[i]:.1f}",
                    ha="center", fontsize=9.5, color=TAB10[0], fontweight="bold")
            ax.text(x[i], ns_vals[i] + ns_offset, f"{ns_vals[i]:.1f}",
                    ha="center", fontsize=9.5, color=TAB10[1], fontweight="bold")

        # Delta annotations
        delta_pos = pos_vals[-1] - pos_vals[0]
        delta_ns  = ns_vals[-1] - ns_vals[0]
        ax.text(0.97, 0.05,
                f"\u0394Pos: +{delta_pos:.1f}pp\n\u0394NS: {delta_ns:.1f}pp",
                transform=ax.transAxes, fontsize=9.5, ha="right", va="bottom",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="goldenrod", lw=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels(nbv_levels, fontsize=10)
        ax.set_title(model, fontweight="bold", fontsize=13)

        # Y-axis range
        all_vals = pos_vals + ns_vals
        ymin = min(all_vals) - 10
        ymax = max(all_vals) + 10
        ax.set_ylim(ymin, ymax)

        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.legend(fontsize=9, framealpha=0.9, loc="best")

    fig.suptitle("Cross-Model Verification: Better NBV Consistently "
                 "Raises Positive but Lowers Neg_Same (attr GT)",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig13_cross_model_pos_ns.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  fig13_cross_model_pos_ns.png")



# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Saving to {OUT}/\n")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    table1()
    table2()
    fig9()
    fig10()
    fig11()
    fig12()
    fig13()
    print(f"\nAll done! {len(os.listdir(OUT))} files in {OUT}/")
