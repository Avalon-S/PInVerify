#!/usr/bin/env python3
"""
Generate publication-quality PDF figures for PInVerify dataset statistics.
Outputs to both thesis and ECCV paper image directories.

Usage:
    python generate_dataset_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# Academic color palette (colorblind-friendly, muted tones)
COLORS = {
    "train": "#4C72B0",
    "val":   "#DD8452",
    "nav":   "#55A868",
    "vis":   "#C44E52",
    "trap":  "#8172B3",
    "far":   "#4C72B0",
    "near":  "#DD8452",
    "bar1":  "#4C72B0",
    "bar2":  "#DD8452",
    "bar3":  "#55A868",
}

# Category display order (by typical object size: large → small)
CAT_ORDER = [
    "teddy_bear", "backpack", "laptop", "bag", "shoes",
    "hat", "book", "toy", "ball", "headphones",
    "camera", "visor", "mug", "cellphone", "wallet",
    "eyeglasses", "watch", "keys",
]

CAT_LABELS = {
    "teddy_bear": "Teddy Bear",
    "backpack": "Backpack",
    "laptop": "Laptop",
    "bag": "Bag",
    "shoes": "Shoes",
    "hat": "Hat",
    "book": "Book",
    "toy": "Toy",
    "ball": "Ball",
    "headphones": "Headphones",
    "camera": "Camera",
    "visor": "Visor",
    "mug": "Mug",
    "cellphone": "Cellphone",
    "wallet": "Wallet",
    "eyeglasses": "Eyeglasses",
    "watch": "Watch",
    "keys": "Keys",
}


# ── Data loading ───────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))

def load_stats(name):
    path = os.path.join(BASE, "dataset_details", name)
    with open(path, "r") as f:
        data = json.load(f)
    return data["stats"], data["raw"]


train_stats, train_raw = load_stats("train_stats.json")
val_stats, val_raw = load_stats("val_stats.json")


# ── Output dirs ────────────────────────────────────────────────────────

THESIS_DIR = os.path.join(BASE, "Paper_Writing",
    "_Draft__Yuhang_Jiang_Master_Thesis__EITM", "images", "appendix_capture")
ECCV_DIR = os.path.join(BASE, "Paper_Writing",
    "Yuhang_PInVerify", "images", "supplement")

os.makedirs(THESIS_DIR, exist_ok=True)
os.makedirs(ECCV_DIR, exist_ok=True)


def save_fig(fig, name):
    """Save figure as PDF to both paper dirs."""
    for d in [THESIS_DIR, ECCV_DIR]:
        path = os.path.join(d, f"{name}.pdf")
        fig.savefig(path, format="pdf")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
#  Figure 1: Per-Category Episode Distribution (Train vs Val)
# =====================================================================

def fig_category_episodes():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    cats = CAT_ORDER
    labels = [CAT_LABELS[c] for c in cats]
    train_counts = [train_stats["per_category"].get(c, {}).get("episodes", 0) for c in cats]
    val_counts = [val_stats["per_category"].get(c, {}).get("episodes", 0) for c in cats]

    y = np.arange(len(cats))
    h = 0.35

    ax.barh(y + h/2, train_counts, h, label="Train", color=COLORS["train"], edgecolor="white", linewidth=0.3)
    ax.barh(y - h/2, val_counts, h, label="Val", color=COLORS["val"], edgecolor="white", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of Episodes")
    ax.set_title("Episode Distribution by Object Category")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.invert_yaxis()

    # add count labels on bars
    for i, (tv, vv) in enumerate(zip(train_counts, val_counts)):
        ax.text(tv + 30, i + h/2, str(tv), va="center", fontsize=6.5, color="#333")
        ax.text(vv + 30, i - h/2, str(vv), va="center", fontsize=6.5, color="#333")

    fig.tight_layout()
    save_fig(fig, "category_episode_distribution")


# =====================================================================
#  Figure 2: Navigable / Visible / Trap Sectors Distribution
# =====================================================================

def fig_sector_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)

    titles = ["Navigable Sectors", "Visible Sectors", "Trap Sectors"]
    keys = ["navigable_sectors_per_episode", "visible_sectors_per_episode", "trap_sectors_per_episode"]

    for ax, title, key in zip(axes, titles, keys):
        train_dist = train_stats["viewpoint_stats"][key]["distribution"]
        val_dist = val_stats["viewpoint_stats"][key]["distribution"]

        all_keys = sorted(set(list(train_dist.keys()) + list(val_dist.keys())), key=int)
        x = np.arange(len(all_keys))
        w = 0.35

        train_vals = [train_dist.get(k, 0) for k in all_keys]
        val_vals = [val_dist.get(k, 0) for k in all_keys]

        # normalize to percentage
        train_total = sum(train_vals) or 1
        val_total = sum(val_vals) or 1
        train_pct = [v / train_total * 100 for v in train_vals]
        val_pct = [v / val_total * 100 for v in val_vals]

        ax.bar(x - w/2, train_pct, w, label="Train", color=COLORS["train"], edgecolor="white", linewidth=0.3)
        ax.bar(x + w/2, val_pct, w, label="Val", color=COLORS["val"], edgecolor="white", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(all_keys)
        ax.set_xlabel("Count (out of 6)")
        ax.set_title(title, fontsize=9.5)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    axes[0].set_ylabel("Fraction of Episodes")
    axes[0].legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    save_fig(fig, "sector_distribution")


# =====================================================================
#  Figure 3: Mask Area Histogram (All / Far / Near)
# =====================================================================

def fig_mask_area_histogram():
    fig, ax = plt.subplots(figsize=(5, 3))

    bins = [0, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    bin_labels = ["0-\n100", "100-\n200", "200-\n500", "500-\n1k", "1k-\n2k",
                  "2k-\n5k", "5k-\n10k", "10k-\n20k", "20k-\n50k", "50k+"]

    def get_hist_counts(stats):
        h = stats["mask_area_stats"]["all"]["histogram"]
        return [item["count"] for item in h]

    def get_hist_pct(stats):
        counts = get_hist_counts(stats)
        total = sum(counts) or 1
        return [c / total * 100 for c in counts]

    train_pct = get_hist_pct(train_stats)
    val_pct = get_hist_pct(val_stats)

    x = np.arange(len(bin_labels))
    w = 0.35

    ax.bar(x - w/2, train_pct, w, label="Train", color=COLORS["train"], edgecolor="white", linewidth=0.3)
    ax.bar(x + w/2, val_pct, w, label="Val", color=COLORS["val"], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=7)
    ax.set_xlabel("Mask Area (pixels)")
    ax.set_ylabel("Fraction of Viewpoints")
    ax.set_title("Target Object Mask Area Distribution")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "mask_area_histogram")


# =====================================================================
#  Figure 4: Mask Area — Far vs Near comparison
# =====================================================================

def fig_mask_area_far_near():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    bin_labels = ["0-\n100", "100-\n200", "200-\n500", "500-\n1k", "1k-\n2k",
                  "2k-\n5k", "5k-\n10k", "10k-\n20k", "20k-\n50k", "50k+"]

    for ax, stats, title_suffix in zip(axes,
                                        [train_stats, val_stats],
                                        ["Train", "Val"]):
        far_h = stats["mask_area_stats"]["far"]["histogram"]
        near_h = stats["mask_area_stats"]["near"]["histogram"]

        far_counts = [item["count"] for item in far_h]
        near_counts = [item["count"] for item in near_h]
        far_total = sum(far_counts) or 1
        near_total = sum(near_counts) or 1
        far_pct = [c / far_total * 100 for c in far_counts]
        near_pct = [c / near_total * 100 for c in near_counts]

        x = np.arange(len(bin_labels))
        w = 0.35

        ax.bar(x - w/2, far_pct, w, label="Far (1.4\u20131.7 m)", color=COLORS["far"], edgecolor="white", linewidth=0.3)
        ax.bar(x + w/2, near_pct, w, label="Near (0.9\u20131.2 m)", color=COLORS["near"], edgecolor="white", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=6.5)
        ax.set_xlabel("Mask Area (pixels)")
        ax.set_title(f"Far vs Near ({title_suffix})", fontsize=9.5)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.legend(fontsize=7, framealpha=0.9)

    axes[0].set_ylabel("Fraction of Viewpoints")

    fig.tight_layout()
    save_fig(fig, "mask_area_far_vs_near")


# =====================================================================
#  Figure 5: Per-Category Mask Area (box-plot style using percentiles)
# =====================================================================

def _draw_boxplot(ax, stats_data, cats, offset, color, label):
    """Draw box plots (whiskers=p5-p95, box=p25-p75, line=median) for one split."""
    for i, cat in enumerate(cats):
        info = stats_data["per_category"].get(cat, {}).get("mask_area_px", {})
        pct = info.get("percentiles", {})
        if not pct:
            continue
        p5  = pct.get("p5", 0)
        p25 = pct.get("p25", 0)
        p50 = pct.get("p50", 0)
        p75 = pct.get("p75", 0)
        p95 = pct.get("p95", 0)

        y = i + offset
        box_w = 0.18

        # whiskers (p5-p95)
        ax.plot([p5, p95], [y, y], color="#777", linewidth=0.6, zorder=1)
        ax.plot([p5, p5], [y - 0.08, y + 0.08], color="#777", linewidth=0.6, zorder=1)
        ax.plot([p95, p95], [y - 0.08, y + 0.08], color="#777", linewidth=0.6, zorder=1)

        # box
        rect = plt.Rectangle((p25, y - box_w/2), p75 - p25, box_w,
                              facecolor=color, alpha=0.7, edgecolor="#333",
                              linewidth=0.5, zorder=2)
        ax.add_patch(rect)

        # median line
        ax.plot([p50, p50], [y - box_w/2, y + box_w/2], color="#C44E52",
                linewidth=1.0, zorder=3)


def fig_category_mask_area():
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    cats = CAT_ORDER
    labels = [CAT_LABELS[c] for c in cats]
    y_pos = np.arange(len(cats))

    _draw_boxplot(ax, train_stats, cats, offset=-0.12, color=COLORS["train"], label="Train")
    _draw_boxplot(ax, val_stats, cats, offset=+0.12, color=COLORS["val"], label="Val")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mask Area (pixels)")
    ax.set_title("Per-Category Object Visibility")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS["train"], alpha=0.7, edgecolor="#333", label="Train IQR"),
        Patch(facecolor=COLORS["val"], alpha=0.7, edgecolor="#333", label="Val IQR"),
        Line2D([0], [0], color="#C44E52", linewidth=1.0, label="Median"),
        Line2D([0], [0], color="#777", linewidth=0.6, label="p5\u2013p95"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "category_mask_area_boxplot")


# =====================================================================
#  Figure 6: Per-Category Sector Quality (navigable / visible / trap)
# =====================================================================

def fig_category_sector_quality():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)

    cats = CAT_ORDER
    labels = [CAT_LABELS[c] for c in cats]
    y = np.arange(len(cats))
    h = 0.25

    for ax, stats, title_suffix in zip(axes,
                                        [train_stats, val_stats],
                                        ["Train", "Val"]):
        nav_vals = [stats["per_category"].get(c, {}).get("navigable_sectors", {}).get("avg", 0) for c in cats]
        vis_vals = [stats["per_category"].get(c, {}).get("visible_sectors", {}).get("avg", 0) for c in cats]
        trap_vals = [stats["per_category"].get(c, {}).get("trap_sectors", {}).get("avg", 0) for c in cats]

        ax.barh(y + h, nav_vals, h, label="Navigable", color=COLORS["nav"], edgecolor="white", linewidth=0.3)
        ax.barh(y, vis_vals, h, label="Visible", color=COLORS["vis"], edgecolor="white", linewidth=0.3)
        ax.barh(y - h, trap_vals, h, label="Trap", color=COLORS["trap"], edgecolor="white", linewidth=0.3)

        ax.set_xlabel("Avg Sectors per Episode (out of 6)")
        ax.set_title(f"Sector Quality ({title_suffix})", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        ax.set_xlim(0, 6.2)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()

    fig.tight_layout()
    save_fig(fig, "category_sector_quality")


# =====================================================================
#  Table: Dataset Summary (LaTeX tabular, saved as .tex)
# =====================================================================

def gen_summary_table():
    ts = train_stats["summary"]
    vs = val_stats["summary"]
    tv = train_stats["viewpoint_stats"]
    vv = val_stats["viewpoint_stats"]

    tex = r"""\begin{table}[htbp]
\centering
\caption{PInVerify capture dataset statistics.}
\label{tab:capture_stats}
\begin{tabular}{l r r}
\toprule
\textbf{Property} & \textbf{Train} & \textbf{Val} \\
\midrule
Scenes & """ + f"{ts['total_scenes']}" + r" & " + f"{vs['total_scenes']}" + r""" \\
Episodes & """ + f"{ts['total_episodes']:,}" + r" & " + f"{vs['total_episodes']:,}" + r""" \\
Unique objects & """ + f"{ts['total_unique_objects']}" + r" & " + f"{vs['total_unique_objects']}" + r""" \\
Object categories & """ + f"{ts['total_categories']}" + r" & " + f"{vs['total_categories']}" + r""" \\
\midrule
Avg navigable sectors / ep & """ + f"{tv['navigable_sectors_per_episode']['avg']}" + r" & " + f"{vv['navigable_sectors_per_episode']['avg']}" + r""" \\
Avg visible sectors / ep & """ + f"{tv['visible_sectors_per_episode']['avg']}" + r" & " + f"{vv['visible_sectors_per_episode']['avg']}" + r""" \\
Avg trap sectors / ep & """ + f"{tv['trap_sectors_per_episode']['avg']}" + r" & " + f"{vv['trap_sectors_per_episode']['avg']}" + r""" \\
\midrule
Avg navigable viewpoints / ep & """ + f"{tv['navigable_viewpoints_per_episode']['avg']}" + r" & " + f"{vv['navigable_viewpoints_per_episode']['avg']}" + r""" \\
Avg valid-mask viewpoints / ep & """ + f"{tv['valid_mask_viewpoints_per_episode']['avg']}" + r" & " + f"{vv['valid_mask_viewpoints_per_episode']['avg']}" + r""" \\
\midrule
Median mask area (px) & """ + f"{int(train_stats['mask_area_stats']['all']['median']):,}" + r" & " + f"{int(val_stats['mask_area_stats']['all']['median']):,}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    for d in [THESIS_DIR, ECCV_DIR]:
        path = os.path.join(d, "capture_stats_table.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"  Saved: {path}")


# =====================================================================
#  Table: Per-Category Breakdown (LaTeX)
# =====================================================================

def gen_category_table():
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-category statistics of the capture dataset (train split).}",
        r"\label{tab:per_category_stats}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\small",
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Eps} & \textbf{Objs} & \textbf{Nav} & \textbf{Vis} & \textbf{Trap} & \textbf{Mask (med)} \\",
        r"\midrule",
    ]

    for cat in CAT_ORDER:
        info = train_stats["per_category"].get(cat, {})
        eps = info.get("episodes", 0)
        objs = info.get("unique_objects", 0)
        nav = info.get("navigable_sectors", {}).get("avg", 0)
        vis = info.get("visible_sectors", {}).get("avg", 0)
        trap = info.get("trap_sectors", {}).get("avg", 0)
        med = info.get("mask_area_px", {}).get("median", 0)
        label = CAT_LABELS[cat]
        tex_lines.append(
            f"    {label} & {eps:,} & {objs} & {nav:.1f} & {vis:.1f} & {trap:.1f} & {int(med):,} \\\\"
        )

    # totals
    ts = train_stats["summary"]
    tv = train_stats["viewpoint_stats"]
    med_all = int(train_stats["mask_area_stats"]["all"]["median"])
    tex_lines.append(r"\midrule")
    tex_lines.append(
        f"    \\textbf{{Total / Avg}} & {ts['total_episodes']:,} & {ts['total_unique_objects']} "
        f"& {tv['navigable_sectors_per_episode']['avg']:.1f} "
        f"& {tv['visible_sectors_per_episode']['avg']:.1f} "
        f"& {tv['trap_sectors_per_episode']['avg']:.1f} "
        f"& {med_all:,} \\\\"
    )

    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(tex_lines) + "\n"
    for d in [THESIS_DIR, ECCV_DIR]:
        path = os.path.join(d, "per_category_table.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"  Saved: {path}")


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    print("Generating figures ...\n")
    fig_category_episodes()
    fig_sector_distribution()
    fig_mask_area_histogram()
    fig_mask_area_far_near()
    fig_category_mask_area()
    fig_category_sector_quality()

    print("\nGenerating LaTeX tables ...\n")
    gen_summary_table()
    gen_category_table()

    print("\nDone!")
