"""
Generate qualitative case study figures for the thesis.
Layout: left = top-down trajectory, right = key steps (crop + attribute table).
"""
import re
import json
import base64
import os
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import numpy as np

OUT_DIR = "./figures"
BASE = "./outputs/run_50/qwen3_vl_4B_50/multi_view_attr_adaptive_llm_dino_50"


def extract_images(html_path):
    """Extract (scene, crop) pairs per step from vis.html."""
    with open(html_path, encoding='utf-8') as f:
        content = f.read()
    raw = re.findall(r'data:image/(jpeg|png);base64,([A-Za-z0-9+/=]+)', content)
    images = []
    for fmt, data in raw:
        img = Image.open(BytesIO(base64.b64decode(data)))
        images.append(img)
    pairs = []
    for i in range(0, len(images) - 1, 2):
        img_a, img_b = images[i], images[i + 1]
        if img_a.size[0] < img_a.size[1]:
            scene, crop = img_a, img_b
        else:
            scene, crop = img_b, img_a
        pairs.append((scene, crop))
    return pairs


def load_episode(ep_dir):
    """Load episode.json and extract per-step info."""
    with open(os.path.join(ep_dir, 'episode.json'), encoding='utf-8') as f:
        ep = json.load(f)
    steps = []
    for t in ep['transcript']:
        act = t.get('action', {})
        steps.append({
            'step': t['step'],
            'sector': act.get('_current_abs_sector', '?'),
            'prediction': act.get('_step_prediction', '?'),
            'decision': act.get('decision', '?'),
            'nav': act.get('nav_rel', '').split(' (')[0],
            'conf': act.get('_detection_confidence', 1.0),
            'attrs': act.get('_per_attribute_results', []),
        })
    return ep, steps


# Color scheme
C_MATCH = '#2ca02c'
C_CONTRA = '#d62728'
C_MISS = '#888888'
C_NAVFAIL = '#ff7f0e'


def _attr_color(status):
    if status == 'Matched':
        return C_MATCH
    elif status in ('Contradictory', 'Contradicted'):
        return C_CONTRA
    return C_MISS


def _pred_color(pred):
    if pred == 'Yes':
        return C_MATCH
    elif pred == 'No':
        return C_CONTRA
    return C_MISS


def _short_action(s):
    dec = s['decision']
    if dec == 'Unsure':
        return f"MOVE {s['nav']}"
    elif dec == 'Yes':
        return "STOP: Match"
    elif dec == 'No':
        return "STOP: Mismatch"
    return dec


def _set_border(ax, color, lw=3, ls='-'):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_linestyle(ls)


def plot_case(ep_dir, show_indices, title, out_name, nav_fail_steps=None):
    """
    Generate a case study figure.
    Layout: Left col = top-down map. Right cols = one per selected step:
      top = crop image, bottom = attribute verification table.
    """
    ep, steps = load_episode(ep_dir)
    pairs = extract_images(os.path.join(ep_dir, "vis.html"))
    topdown_path = os.path.join(ep_dir, "topdown_debug.png")
    nav_fail_steps = nav_fail_steps or set()

    n_show = len(show_indices)

    # Layout: 2 rows × (1 + n_show) cols
    # Row 0 = crop images (topdown spans both rows)
    # Row 1 = attribute tables
    fig = plt.figure(figsize=(5.0 + 3.8 * n_show, 7.5))
    fig.subplots_adjust(top=0.78)

    # Use a unified grid: 2 rows, 1+n_show cols
    gs = gridspec.GridSpec(2, 1 + n_show,
                           width_ratios=[1.8] + [1] * n_show,
                           height_ratios=[1.8, 2.2],
                           wspace=0.12, hspace=0.05,
                           top=0.78, bottom=0.02)

    # ── Left: Top-down map (spans both rows, anchored to top) ──
    ax_td = fig.add_subplot(gs[:, 0])
    if os.path.isfile(topdown_path):
        td_img = Image.open(topdown_path)
        td_arr = np.array(td_img)
        ax_td.imshow(td_arr)
    ax_td.set_xticks([])
    ax_td.set_yticks([])
    ax_td.set_anchor('N')  # Align to top
    _set_border(ax_td, '#333333', lw=1)
    ax_td.set_title("Agent Trajectory", fontsize=10, fontweight='bold', pad=6)

    for col, si in enumerate(show_indices):
        scene, crop = pairs[si]
        s = steps[si]
        pred = s['prediction']
        color = _pred_color(pred)
        is_nf = si in nav_fail_steps

        # ── Top: Crop image ──
        ax_crop = fig.add_subplot(gs[0, col + 1])
        ax_crop.imshow(np.array(crop))
        ax_crop.set_xticks([])
        ax_crop.set_yticks([])

        if is_nf:
            _set_border(ax_crop, C_NAVFAIL, lw=4, ls='--')
        else:
            _set_border(ax_crop, color, lw=3)

        action_str = _short_action(s)
        nf_tag = " [NAV FAIL]" if is_nf else ""
        title_weight = 'bold' if s['decision'] != 'Unsure' else 'normal'
        ax_crop.set_title(
            f"Step {s['step']}: {s['sector']}{nf_tag}\n{action_str}",
            fontsize=8.5, pad=4, fontweight=title_weight,
            color=C_NAVFAIL if is_nf else 'black'
        )

        # ── Bottom: Attribute table ──
        ax_tbl = fig.add_subplot(gs[1, col + 1])
        ax_tbl.axis('off')

        attrs = s['attrs']
        if attrs:
            # Build table data
            col_labels = ['Attribute', 'Expected', 'Obs.', 'Tracker']
            cell_text = []
            cell_colors = []
            for a in attrs:
                name = a['name'].replace('_', ' ')
                if len(name) > 12:
                    name = name[:11] + '.'
                expected = a['expected']
                if len(expected) > 14:
                    expected = expected[:13] + '.'
                obs = a['observed']
                acc = a['accumulated_status']

                row = [name, expected, obs[:5], acc[:5]]
                cell_text.append(row)

                # Color coding for obs and acc columns
                row_colors = ['white', 'white',
                              _attr_color(obs), _attr_color(acc)]
                cell_colors.append(row_colors)

            table = ax_tbl.table(
                cellText=cell_text,
                colLabels=col_labels,
                cellLoc='center',
                loc='upper center',
                colWidths=[0.30, 0.35, 0.17, 0.18],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.0, 1.3)

            # Style header
            for j in range(len(col_labels)):
                cell = table[0, j]
                cell.set_facecolor('#333333')
                cell.set_text_props(color='white', fontweight='bold')

            # Style data cells
            for i in range(len(cell_text)):
                for j in range(len(col_labels)):
                    cell = table[i + 1, j]
                    bg = cell_colors[i][j]
                    if bg != 'white':
                        cell.set_facecolor(bg)
                        cell.set_text_props(color='white', fontweight='bold')
                    else:
                        cell.set_facecolor('#f8f8f8' if i % 2 == 0 else 'white')
        else:
            ax_tbl.text(0.5, 0.5, "No attribute data", ha='center', va='center',
                       fontsize=8, color='gray')

    # ── Suptitle ──
    descs = ep.get('query_descriptions', [])

    result = ep.get('prediction', '?')
    label = ep.get('label', '?')
    correct = 'correct' if ep.get('is_correct') else 'incorrect'
    result_color = C_MATCH if ep.get('is_correct') else C_CONTRA

    fig.suptitle(title, fontsize=12, y=0.99, fontweight='bold')

    # Show all query descriptions
    desc_lines = []
    for i, d in enumerate(descs):
        if len(d) > 85:
            d = d[:82] + "..."
        desc_lines.append(f"D{i+1}: \"{d}\"")
    desc_block = "\n".join(desc_lines) if desc_lines else "No descriptions"

    fig.text(0.5, 0.95, "Query Descriptions:", ha='center', fontsize=9,
             fontweight='bold')
    fig.text(0.5, 0.94, desc_block, ha='center', fontsize=7.5,
             fontstyle='italic', va='top', linespacing=1.4)

    # Result line
    n_desc_lines = len(desc_lines)
    result_y = 0.94 - n_desc_lines * 0.025
    fig.text(0.5, result_y, f'Result: {result} (GT: {"Match" if label == 1 else "Mismatch"}) — {correct}',
             ha='center', fontsize=9, color=result_color, fontweight='bold')

    fig.savefig(f"{OUT_DIR}/{out_name}", bbox_inches="tight", pad_inches=0.05, dpi=200)
    print(f"Saved {out_name}")
    plt.close()
    return ep


# ── Cases ───────────────────────────────────────────────────────────────

def plot_case_a():
    # Mug: 6 steps, No→No→No→Yes→Yes→Yes, flip at step 4
    ep_dir = os.path.join(BASE, "positive/correct/00876-mv2HUxq3B53/35")
    plot_case(
        ep_dir,
        show_indices=[0, 2, 3],  # step 1 (No), step 3 (No), step 4 (Yes flip)
        title="(a) Multi-View Success with Belief Flip",
        out_name="case_study_success.pdf"
    )


def plot_case_b():
    # Eyeglasses: 6 steps, all No, nav failure at step 3 (sector stuck at back)
    ep_dir = os.path.join(BASE, "positive/wrong/00832-qyAac8rV8Zk/14")
    ep, steps = load_episode(ep_dir)
    # Detect nav failures
    nav_fails = set()
    for i in range(1, len(steps)):
        if (steps[i]['sector'] == steps[i-1]['sector'] and
                steps[i-1]['decision'] == 'Unsure'):
            nav_fails.add(i)

    plot_case(
        ep_dir,
        show_indices=[0, 2, len(steps) - 1],  # step 1, step 3 (nav fail), final
        title="(b) Navigation Failure Leading to Incorrect Rejection",
        out_name="case_study_navfail.pdf",
        nav_fail_steps=nav_fails
    )


def plot_case_c():
    # Camera: 4 steps, all No, red/grey query vs black Nikon
    ep_dir = os.path.join(BASE, "neg_same/correct/00862-LT9Jq6dN3Ea/41")
    plot_case(
        ep_dir,
        show_indices=[0, 1, 3],  # step 1, step 2, step 4 (final rejection)
        title="(c) Correct Rejection of Same-Category Distractor",
        out_name="case_study_rejection.pdf"
    )


if __name__ == "__main__":
    plot_case_a()
    plot_case_b()
    plot_case_c()
    print("All case study figures generated!")
