"""
Generate dataset overview figures for the thesis:
1. Object instance gallery (2 pages): table-style layout with images + descriptions
2. Capture environment: multi-view captures with RGB|Mask pairs + topdown overview
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np

OUT_DIR = "./figures"
IMAGE_GT = "./data/pv_dataset/image_gt"
PIN_CAPTURE = "./data/pv_dataset/pin_capture"

with open("./data/pv_dataset/object_descriptions_with_category.json") as f:
    OBJ_DESCS = json.load(f)


# ── All 18 categories, split into two pages ──────────────────────────────
# Pick visually diverse and interesting objects for each category
GALLERY_PAGE1 = [
    ("backpack",   "2ad86321197a49feb54b7726743d7fd0"),  # brown leather kanken
    ("bag",        "19842492eaac4381b4a67f84371e945f"),  # brown suitcase stickers
    ("ball",       "0a96f1f19afc432bb22c3d74da546338"),  # beach ball
    ("book",       "1dbf9e29eb344f54987e2d8d4d47568b"),  # stack of books
    ("camera",     "228d92307870440496e23b95b6c5b2a4"),  # black nikon
    ("cellphone",  "03c75adf5d9a426a9435d20cc76090be"),  # black phone
    ("eyeglasses", "00d1cb5aa82745228a3b764c97f867de"),  # gold aviator
    ("hat",        "01aa135a058e4d9396c234294d1691ea"),  # dark blue cap
    ("headphones", "0d6f7320eda243278919828ba6cd2619"),  # black headphones
]

GALLERY_PAGE2 = [
    ("keys",       "1e2aec13d81b4bb689b3a0b73e6e4b3d"),  # keys
    ("laptop",     "0cd6c12cab7d4c2fb25a6ad7df5af5b8"),  # laptop
    ("mug",        "04d0553202d34b299bc0bf43025b6ef8"),  # mug
    ("shoes",      "04e7db2b38c34f49960e04c6b4b5bdb0"),  # shoes
    ("teddy_bear", "0d14b7dc31e741359424ce73e9a98e85"),  # teddy bear
    ("toy",        "086e0e2df5424d92837de8df97dce479"),  # toy
    ("visor",      "1a5d89e23d4a45cfbf4bbf10e4bfdee5"),  # visor
    ("wallet",     "08ed2dbdb6784d8d98c7f6dca2e53571"),  # wallet
    ("watch",      "0b56c7f8e43d4f82867f8eefc37e37b5"),  # watch
]


def _get_first_valid_oid(category):
    """Get first object ID that has images and descriptions."""
    gt_dir = os.path.join(IMAGE_GT, category)
    if not os.path.isdir(gt_dir):
        return None
    for fn in sorted(os.listdir(gt_dir)):
        if fn.endswith("_0.png"):
            oid = fn.replace("_0.png", "")
            if oid in OBJ_DESCS:
                return oid
    return None


def plot_gallery_page(objects, out_name, page_label=""):
    """
    Clean gallery: horizontal rules only, 3 images left + descriptions right.
    No headers, no category column, no vertical lines.
    """
    # Validate and fallback for missing objects
    valid_objects = []
    for cat, oid in objects:
        img_path = os.path.join(IMAGE_GT, cat, f"{oid}_0.png")
        if os.path.isfile(img_path) and oid in OBJ_DESCS:
            valid_objects.append((cat, oid))
        else:
            fallback = _get_first_valid_oid(cat)
            if fallback:
                valid_objects.append((cat, fallback))
                print(f"  Fallback for {cat}: {oid[:8]}... -> {fallback[:8]}...")

    n_rows = len(valid_objects)
    row_height = 1.45
    fig_height = n_rows * row_height
    fig = plt.figure(figsize=(14, fig_height))

    # Layout constants (figure-fraction coords)
    margin_l, margin_r = 0.01, 0.99
    margin_t, margin_b = 0.995, 0.005
    total_h = margin_t - margin_b
    row_h = total_h / n_rows

    # Image zone: 3 images take ~35% width
    img_zone_r = margin_l + 0.35 * (margin_r - margin_l)
    img_w_each = (img_zone_r - margin_l) / 3
    # Description zone starts after images
    desc_x = img_zone_r + 0.015

    # Top rule
    fig.add_artist(plt.Line2D([margin_l, margin_r], [margin_t, margin_t],
                   transform=fig.transFigure, color='#555555', linewidth=0.8))

    for row, (cat, oid) in enumerate(valid_objects):
        descs = OBJ_DESCS.get(oid, {}).get("descriptions", ["?", "?", "?"])
        row_top = margin_t - row * row_h
        row_bot = row_top - row_h

        # Bottom rule for this row
        fig.add_artist(plt.Line2D([margin_l, margin_r], [row_bot, row_bot],
                       transform=fig.transFigure, color='#555555', linewidth=0.8))

        # 3 GT images
        img_pad = 0.003
        for col in range(3):
            img_path = os.path.join(IMAGE_GT, cat, f"{oid}_{col}.png")
            if not os.path.isfile(img_path):
                continue
            img = Image.open(img_path)

            ax_l = margin_l + col * img_w_each + img_pad
            ax_b = row_bot + img_pad
            ax_w = img_w_each - 2 * img_pad
            ax_h = row_h - 2 * img_pad

            ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])
            ax.imshow(np.array(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Descriptions: 3 lines, last one bold
        desc_top = row_top - 0.008
        desc_spacing = row_h / 3.5
        for i, d in enumerate(descs):
            wrapped = textwrap.fill(d, width=85)
            weight = 'bold' if i == len(descs) - 1 else 'normal'
            y = desc_top - i * desc_spacing
            fig.text(desc_x, y, wrapped, ha='left', va='top',
                     fontsize=8.5, fontstyle='italic', fontweight=weight,
                     color='#111111', transform=fig.transFigure,
                     linespacing=1.25)

    fig.savefig(f"{OUT_DIR}/{out_name}",
                bbox_inches="tight", pad_inches=0.02, dpi=200)
    print(f"Saved {out_name}")
    plt.close()


# ── Figure 2: Capture Views ──────────────────────────────────────────────
# Layout: topdown on left, then rows of (RGB | Mask) pairs, 3 pairs per row

def plot_capture_views():
    """
    Show offline capture setup for two episodes:
    - One with full visibility
    - One with partial visibility (trap views)
    Each episode: topdown overview + grid of RGB|Mask pairs.
    """
    # Pick diverse categories: one full visibility, one with traps
    episodes = [
        ("val/00800-TEEsavR23oF/6",  "Full Visibility"),      # camera, 7 caps all visible
        ("val/00800-TEEsavR23oF/14", "Partial Visibility"),    # eyeglasses, 6 caps 3 visible
    ]

    for ep_idx, (ep_rel, subtitle) in enumerate(episodes):
        ep_dir = os.path.join(PIN_CAPTURE, ep_rel)
        meta_path = os.path.join(ep_dir, "meta.json")

        with open(meta_path) as f:
            meta = json.load(f)

        captures = meta.get("captures", meta.get("viewpoints", []))
        cat = meta.get("object_category", "?").replace("_", " ").title()
        oid = meta.get("object_id", "?")
        descs = OBJ_DESCS.get(oid, {}).get("descriptions", [])

        # Sort by sector
        captures_sorted = sorted(captures, key=lambda c: c.get("sector_index", 0))
        n_caps = len(captures_sorted)

        # Layout: 3 pairs per row (RGB | Mask side by side)
        n_pairs_per_row = 3
        n_pair_rows = (n_caps + n_pairs_per_row - 1) // n_pairs_per_row

        # Figure: left = topdown, right = grid of RGB|Mask pairs
        fig_width = 14
        fig_height = 2.5 * n_pair_rows + 1.5
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Outer: [topdown | pairs grid] — top lowered to make room for 3 description lines
        outer = gridspec.GridSpec(1, 2, width_ratios=[1.2, 3.5], wspace=0.08,
                                  left=0.02, right=0.98, top=0.84, bottom=0.02)

        # Topdown overview
        ax_ov = fig.add_subplot(outer[0, 0])
        overview_path = os.path.join(ep_dir, "overview.png")
        if os.path.isfile(overview_path):
            ov_img = Image.open(overview_path)
            ax_ov.imshow(np.array(ov_img))
        ax_ov.set_xticks([])
        ax_ov.set_yticks([])
        ax_ov.set_title("Top-Down Overview", fontsize=9, fontweight='bold', pad=4)
        for spine in ax_ov.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(1)

        # Pairs grid: n_pair_rows rows, n_pairs_per_row * 2 cols (RGB + Mask)
        inner = gridspec.GridSpecFromSubplotSpec(
            n_pair_rows, n_pairs_per_row * 2,
            subplot_spec=outer[0, 1],
            wspace=0.03, hspace=0.15
        )

        for ci, cap in enumerate(captures_sorted):
            tag = cap.get("tag", f"s{cap.get('sector_index', '?')}")
            visible = cap.get("mask_meets_threshold", True)
            rgb_rel = cap.get("rgb", "")
            rgb_path = os.path.join(ep_dir, rgb_rel)
            mask_rel = cap.get("mask_raw_path") or cap.get("mask", "") or ""
            mask_path = os.path.join(ep_dir, mask_rel) if mask_rel else ""

            pair_row = ci // n_pairs_per_row
            pair_col = ci % n_pairs_per_row
            rgb_col_idx = pair_col * 2
            mask_col_idx = pair_col * 2 + 1

            border_color = '#2ca02c' if visible else '#d62728'
            vis_label = "Visible" if visible else "Trap"

            # RGB image
            ax_rgb = fig.add_subplot(inner[pair_row, rgb_col_idx])
            if os.path.isfile(rgb_path):
                rgb_img = Image.open(rgb_path)
                ax_rgb.imshow(np.array(rgb_img))
            ax_rgb.set_xticks([])
            ax_rgb.set_yticks([])
            for spine in ax_rgb.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)
            ax_rgb.set_title(f"{tag} ({vis_label})", fontsize=6.5, pad=2,
                           color=border_color, fontweight='bold')

            # Mask image
            ax_mask = fig.add_subplot(inner[pair_row, mask_col_idx])
            if mask_path and os.path.isfile(mask_path):
                mask_img = Image.open(mask_path)
                ax_mask.imshow(np.array(mask_img), cmap='gray')
            ax_mask.set_xticks([])
            ax_mask.set_yticks([])
            for spine in ax_mask.spines.values():
                spine.set_edgecolor('#999999')
                spine.set_linewidth(0.5)

        # Title with all 3 descriptions
        n_visible = sum(1 for c in captures_sorted if c.get('mask_meets_threshold'))
        title_line = f"{subtitle} ({n_visible}/{n_caps} views) — {cat}"
        fig.text(0.5, 0.98, title_line, ha='center', va='top',
                fontsize=10, fontweight='bold', transform=fig.transFigure)
        for di, d in enumerate(descs[:3]):
            if len(d) > 90:
                d = d[:87] + "..."
            fig.text(0.5, 0.955 - di * 0.025, f'D{di+1}: "{d}"',
                    ha='center', va='top', fontsize=8, fontstyle='italic',
                    color='#333333', transform=fig.transFigure)

        out_name = f"dataset_capture_{ep_idx+1}.pdf"
        fig.savefig(f"{OUT_DIR}/{out_name}",
                    bbox_inches="tight", pad_inches=0.05, dpi=200)
        print(f"Saved {out_name}")
        plt.close()


if __name__ == "__main__":
    print("Generating instance gallery page 1...")
    plot_gallery_page(GALLERY_PAGE1, "dataset_instance_gallery_1.pdf")
    print("Generating instance gallery page 2...")
    plot_gallery_page(GALLERY_PAGE2, "dataset_instance_gallery_2.pdf")
    print("Generating capture views...")
    plot_capture_views()
    print("Done!")
