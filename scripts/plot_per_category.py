"""
Generate per-category accuracy visualization for the thesis.
Produces two figures:
1. Grouped bar chart: per-category accuracy for key methods
2. Heatmap: per-category × pair-type accuracy for best TF vs best Trained
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Config ──────────────────────────────────────────────────────────────
OUT_DIR = "./figures"

METHODS = {
    "SV-Attr\n(4B, DINO)":   "./outputs/qwen3_vl_4b/single_view_attr_dino_all/metrics.json",
    "MV-Attr+LLM\n(4B, DINO)": "./outputs/qwen3_vl_4b/multi_view_attr_adaptive_llm_dino_all/metrics.json",
    "SFT+GSPO\n(4B, GT)":    "./outputs/trained/trained_gspo_v2_all/trained_gspo_v2_gt_all/metrics.json",
    "SFT+GSPO\n(4B, DINO)":  "./outputs/trained/trained_gspo_v2_all/trained_gspo_v2_dino_all/metrics.json",
}

# Category order: large objects → small objects (consistent with appendix)
CAT_ORDER = [
    "teddy_bear", "backpack", "laptop", "bag", "shoes", "hat",
    "ball", "toy", "book", "headphones", "mug", "camera",
    "cellphone", "visor", "wallet", "eyeglasses", "watch", "keys"
]
CAT_LABELS = [c.replace("_", " ").title() for c in CAT_ORDER]

# ── Load data ───────────────────────────────────────────────────────────
data = {}
for name, path in METHODS.items():
    with open(path) as f:
        m = json.load(f)
    pc = m["per_category"]
    data[name] = {
        "overall": [pc[c]["accuracy"] for c in CAT_ORDER],
        "pos":     [pc[c]["positive"]["accuracy"] for c in CAT_ORDER],
        "negs":    [pc[c]["neg_same"]["accuracy"] for c in CAT_ORDER],
        "negd":    [pc[c]["neg_diff"]["accuracy"] for c in CAT_ORDER],
        "total":   [pc[c]["total"] for c in CAT_ORDER],
    }

# ── Figure 1: Grouped bar chart ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

n_cats = len(CAT_ORDER)
n_methods = len(METHODS)
bar_width = 0.18
x = np.arange(n_cats)

colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]
method_names = list(METHODS.keys())

for i, name in enumerate(method_names):
    offset = (i - (n_methods - 1) / 2) * bar_width
    bars = ax.bar(x + offset, data[name]["overall"], bar_width,
                  label=name, color=colors[i], edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(CAT_LABELS, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0.45, 1.05)
ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

# Add overall accuracy as horizontal dashed lines
overall_accs = {}
for name, path in METHODS.items():
    with open(path) as f:
        m = json.load(f)
    overall_accs[name] = m["accuracy"]

ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
ax.set_title("Per-Category Accuracy Across Methods", fontsize=12, pad=10)

# Add a subtle grid
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/per_category_accuracy.pdf", bbox_inches="tight", dpi=300)
print(f"Saved: {OUT_DIR}/per_category_accuracy.pdf")
plt.close()

# ── Figure 2: Heatmap (pair-type × category) ───────────────────────────
# Compare best TF (MV-Attr+LLM) vs best Trained (SFT+GSPO GT)
fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

heatmap_methods = [
    ("MV-Attr+LLM\n(4B, DINO)", "Training-Free Best: MV-Attr + LLM (DINO)"),
    ("SFT+GSPO\n(4B, GT)", "Trained Best: SFT + GSPO (GT)"),
]

# Custom red-yellow-green colormap
cmap = LinearSegmentedColormap.from_list("rg",
    [(0.0, "#d73027"), (0.5, "#fee08b"), (0.75, "#a6d96a"), (1.0, "#1a9850")])

for ax_idx, (method_key, title) in enumerate(heatmap_methods):
    ax = axes[ax_idx]
    d = data[method_key]

    matrix = np.array([d["pos"], d["negs"], d["negd"]])  # 3 × 18
    row_labels = ["Positive", "Neg_Same", "Neg_Diff"]

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.4, vmax=1.0)

    # Annotate cells
    for i in range(3):
        for j in range(n_cats):
            val = matrix[i, j]
            text_color = "white" if val < 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold")

    ax.set_yticks(range(3))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)

axes[-1].set_xticks(range(n_cats))
axes[-1].set_xticklabels(CAT_LABELS, rotation=45, ha="right", fontsize=8)

# Colorbar — place on the right side with enough padding to avoid overlap
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
cbar.set_label("Accuracy", fontsize=9)
fig.savefig(f"{OUT_DIR}/per_category_heatmap.pdf", bbox_inches="tight", dpi=300)
print(f"Saved: {OUT_DIR}/per_category_heatmap.pdf")
plt.close()

print("Done!")
