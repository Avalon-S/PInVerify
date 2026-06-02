#!/usr/bin/env python3
"""
Compare metrics across multiple agent runs.

Scans output directories for metrics.json (new format) and batch_summary.json (old format),
extracts key fields, and generates a comparison table.

Usage:
    # Scan a specific output base directory
    python scripts/compare_metrics.py ./outputs/ablation_v1_skip

    # Scan multiple directories
    python scripts/compare_metrics.py ./outputs/ablation_v1_skip ./outputs/ablation_v1_skip_nbv

    # Scan all outputs
    python scripts/compare_metrics.py ./outputs

    # Export to CSV
    python scripts/compare_metrics.py ./outputs --csv results_table.csv

    # Sort by accuracy (descending)
    python scripts/compare_metrics.py ./outputs --sort accuracy

    # Filter by mode
    python scripts/compare_metrics.py ./outputs --mode gt
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# Agent name → (NBV, Fusion) mapping for display clarity
# Format: "nbv/fusion", e.g. "fps/maj" = FarthestPoint NBV + Majority fusion
_AGENT_CONFIG_MAP = {
    # Single-view (no NBV/fusion)
    "single_view_direct":               "-",
    "single_view_attr":                 "-",
    "single_view_merged":               "-",
    # Direct variants
    "multi_view_direct_fps":            "fps/maj",
    "multi_view_direct_llm":            "llm/maj",
    "multi_view_direct_random":         "random/maj",
    "multi_view_direct_oracle":         "oracle/maj",
    # Attr variants
    "multi_view_attr_fps":              "fps/maj",
    "multi_view_attr_llm":              "llm/maj",
    "multi_view_attr_random":           "random/maj",
    "multi_view_attr_viewhint":         "vh/maj",
    "multi_view_attr_oracle":           "oracle/maj",
    # Skip variants
    "multi_view_attr_v1_skip":          "llm/maj+skip",
    "multi_view_attr_v1_skip_3":        "llm/maj+skip",
    "multi_view_attr_v1_skip_fps":      "fps/maj+skip",
    "multi_view_attr_v1_skip_random":   "random/maj+skip",
    # CLIP baselines (desc avg)
    "clip_single_view":                       "-",
    "clip_multi_view_random":                 "random/-",
    "clip_multi_view_fps":                    "fps/-",
    "clip_multi_view_adaptive_random":        "random/astop",
    "clip_multi_view_adaptive_fps":           "fps/astop",
    # CLIP baselines (desc merged)
    "clip_single_view_merged":                "-",
    "clip_multi_view_random_merged":          "random/-",
    "clip_multi_view_fps_merged":             "fps/-",
    "clip_multi_view_adaptive_random_merged": "random/astop",
    "clip_multi_view_adaptive_fps_merged":    "fps/astop",
    # SigLIP2 baselines (desc avg)
    "siglip2_single_view":                    "-",
    "siglip2_multi_view_random":              "random/-",
    "siglip2_multi_view_fps":                 "fps/-",
    "siglip2_multi_view_adaptive_random":     "random/astop",
    "siglip2_multi_view_adaptive_fps":        "fps/astop",
    # SigLIP2 baselines (desc merged)
    "siglip2_single_view_merged":                  "-",
    "siglip2_multi_view_random_merged":            "random/-",
    "siglip2_multi_view_fps_merged":               "fps/-",
    "siglip2_multi_view_adaptive_random_merged":   "random/astop",
    "siglip2_multi_view_adaptive_fps_merged":      "fps/astop",
    # V2 Component-level attr (ablation vs V1 atomic)
    "single_view_attr_v2":              "-",
    "multi_view_attr_v2_random":        "random/maj",
    "multi_view_attr_v2_fps":           "fps/maj",
    "multi_view_attr_v2_skip":          "llm/maj+skip",
    "multi_view_attr_v2_skip_random":   "random/maj+skip",
    # LoRA fine-tuned (direct mode only)
    "lora_single_view_direct":          "-",
    "lora_multi_view_direct_random":    "random/maj",
    "lora_multi_view_direct_fps":       "fps/maj",
    # Trained end-to-end (no separate NBV/Fusion)
    "trained_e2e":                        "e2e/-",
    "trained_sft":                        "e2e/-",
    "trained_grpo_v1":                    "e2e/-",
    "trained_grpo_v2":                    "e2e/-",
    "trained_gspo_v2":                    "e2e/-",
    "trained_base":                       "e2e/-",
    # Adaptive stopping (attr_majority fusion, max_steps=6)
    "multi_view_attr_adaptive_random":   "random/amaj",
    "multi_view_attr_adaptive_fps":      "fps/amaj",
    "multi_view_attr_adaptive_llm":      "llm/amaj",
    "multi_view_direct_adaptive_random": "random/amaj",
    "multi_view_direct_adaptive_fps":    "fps/amaj",
    "multi_view_direct_adaptive_llm":    "llm/amaj",
    # Visibility-weighted adaptive stopping (vis_weighted fusion, max_steps=6)
    "multi_view_attr_adaptive_vis_random":  "random/vis_w",
    "multi_view_direct_adaptive_vis_random": "random/vis_w",
}


def _build_config_tag_from_info(config_info):
    """Build 'nbv/fusion' tag from config_info dict stored in metrics.json."""
    if not config_info:
        return None
    nbv = config_info.get("nbv_type", "none")
    fusion = config_info.get("fusion_type", "none")
    # Shorten known values
    nbv_short = {"farthest": "fps", "viewhint": "vh", "evidence_gap": "evid",
                 "none": "-"}.get(nbv, nbv)
    fusion_short = {"majority": "maj", "weighted": "wt", "llm": "llmf",
                    "attr_majority": "amaj", "vis_weighted": "vis_w",
                    "none": "-"}.get(fusion, fusion)
    if nbv_short == "-" and fusion_short == "-":
        return "-"
    return f"{nbv_short}/{fusion_short}"


def _get_agent_config_tag(agent_name, metrics_config_info=None):
    """Get NBV/Fusion config tag. Prefer metrics.json data, fall back to hardcoded map."""
    # Trained e2e agents have no meaningful NBV/fusion — always use hardcoded map
    if agent_name.startswith("trained_"):
        return _AGENT_CONFIG_MAP.get(agent_name, "e2e/-")
    tag = _build_config_tag_from_info(metrics_config_info)
    if tag is not None:
        return tag
    return _AGENT_CONFIG_MAP.get(agent_name, "?")


def find_metrics_files(base_dirs):
    """Recursively find metrics.json and batch_summary.json files."""
    # Skip worker/chunk subdirectories from multi-GPU runs
    _SKIP_PATTERNS = re.compile(r'(worker_\d+|_gpu\d+)$')

    entries = []
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"[WARN] Directory not found: {base_dir}")
            continue

        # Find metrics.json (new format from evaluate.py)
        for mf in base_path.rglob("metrics.json"):
            if _SKIP_PATTERNS.search(mf.parent.name):
                continue
            entries.append(("metrics", mf))

        # Find batch_summary.json (old format)
        for mf in base_path.rglob("batch_summary.json"):
            if _SKIP_PATTERNS.search(mf.parent.name):
                continue
            entries.append(("batch_summary", mf))

    return entries


def parse_agent_info_from_path(metrics_path):
    """
    Infer agent name, bbox mode, and split from directory path.

    Expected patterns:
        .../ablation_xxx/{agent}_{mode}_{split}/metrics.json
        .../50_samples/{agent}/batch_summary.json
    """
    parts = metrics_path.parts
    parent_dir = metrics_path.parent.name  # e.g. "multi_view_attr_v1_skip_gt_50"

    # Try to parse: {agent}_{mode}_{split}
    # mode is "gt" or "dino", split is a number or "all"
    match = re.match(r'^(.+)_(gt|dino)_(\d+|all)$', parent_dir)
    if match:
        return {
            "agent": match.group(1),
            "mode": match.group(2),
            "split": match.group(3),
            "run_dir": str(metrics_path.parent),
        }

    # Try: {N}_samples/{agent}/
    grandparent = metrics_path.parent.parent.name if len(parts) > 2 else ""
    split_match = re.match(r'^(\d+)_samples$', grandparent)
    if split_match:
        return {
            "agent": parent_dir,
            "mode": "unknown",
            "split": split_match.group(1),
            "run_dir": str(metrics_path.parent),
        }

    # Fallback: use parent dir name
    return {
        "agent": parent_dir,
        "mode": "unknown",
        "split": "?",
        "run_dir": str(metrics_path.parent),
    }


def extract_metrics(fmt, filepath):
    """Extract standardized metrics from either format."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Failed to read {filepath}: {e}")
        return None

    if fmt == "metrics":
        # New format from evaluate.py
        ppt = data.get("per_pair_type", {})
        nav = data.get("nav_stats", {})
        diag = data.get("diagnostic_stats", {})
        return {
            "accuracy": data.get("accuracy", 0),
            "asd": data.get("asd", 0),
            "total": data.get("total_episodes", 0),
            "correct": data.get("correct_count", 0),
            "wrong": data.get("wrong_count", 0),
            "pos_acc": ppt.get("positive", {}).get("accuracy", None),
            "pos_n": ppt.get("positive", {}).get("total", 0),
            "neg_same_acc": ppt.get("neg_same", {}).get("accuracy", None),
            "neg_same_n": ppt.get("neg_same", {}).get("total", 0),
            "neg_diff_acc": ppt.get("neg_diff", {}).get("accuracy", None),
            "neg_diff_n": ppt.get("neg_diff", {}).get("total", 0),
            "nav_fail_total": nav.get("total_nav_failures", 0),
            "nav_fail_unreachable": nav.get("nav_fail_unreachable", 0),
            "nav_fail_trap": nav.get("nav_fail_trap", 0),
            "nav_fail_ep_rate": nav.get("nav_failure_rate_per_episode", 0),
            "nav_fail_step_rate": nav.get("nav_failure_rate_per_step", 0),
            # Diagnostic stats
            "fv_acc": diag.get("first_view_accuracy", None),
            "avg_eff_views": diag.get("avg_effective_views", None),
            # Config info (from metrics.json if available)
            "_config_info": data.get("config_info", None),
            # Per-pair-type breakdown (new fields from metrics.py)
            "_per_pair_type": ppt,
            # Per-category breakdown
            "_per_category": data.get("per_category", {}),
        }
    elif fmt == "batch_summary":
        # Old format
        summary = data.get("summary", {})
        by_type = summary.get("by_type", {})
        return {
            "accuracy": summary.get("accuracy", 0),
            "asd": 1,  # old format doesn't track steps
            "total": summary.get("num_pairs", 0),
            "correct": summary.get("confusion", {}).get("tp", 0) + summary.get("confusion", {}).get("tn", 0),
            "wrong": summary.get("confusion", {}).get("fp", 0) + summary.get("confusion", {}).get("fn", 0),
            "pos_acc": by_type.get("positive", {}).get("acc", None),
            "pos_n": by_type.get("positive", {}).get("n", 0),
            "neg_same_acc": by_type.get("neg_same", {}).get("acc", None),
            "neg_same_n": by_type.get("neg_same", {}).get("n", 0),
            "neg_diff_acc": by_type.get("neg_diff", {}).get("acc", None),
            "neg_diff_n": by_type.get("neg_diff", {}).get("n", 0),
            "nav_fail_total": 0,
            "nav_fail_unreachable": 0,
            "nav_fail_trap": 0,
            "nav_fail_ep_rate": 0,
            "nav_fail_step_rate": 0,
            "fv_acc": None,
            "avg_eff_views": None,
        }
    return None


def fmt_pct(val, width=7):
    """Format a percentage value."""
    if val is None:
        return "-".center(width)
    return f"{val:.1%}".rjust(width)


def fmt_pct_n(acc, n, width=12):
    """Format accuracy with count, e.g. '88.2% (17)'."""
    if acc is None:
        return "-".center(width)
    return f"{acc:.1%} ({n})".rjust(width)


def print_table(rows, sort_key=None, filter_mode=None):
    """Print a formatted comparison table."""
    if filter_mode:
        rows = [r for r in rows if r["info"]["mode"] == filter_mode]

    if not rows:
        print("No results found.")
        return

    if sort_key:
        reverse = sort_key in ("accuracy", "pos_acc", "neg_same_acc", "neg_diff_acc", "fv_acc")
        rows.sort(key=lambda r: r["metrics"].get(sort_key, 0) or 0, reverse=reverse)
    else:
        # Default sort: agent name, then mode
        rows.sort(key=lambda r: (r["info"]["agent"], r["info"]["mode"]))

    # Check which optional columns have data
    has_nav_data = any(r["metrics"].get("nav_fail_total", 0) > 0 for r in rows)
    has_diag_data = any(r["metrics"].get("fv_acc") is not None for r in rows)

    # Calculate column widths
    agent_w = max(5, max(len(r["info"]["agent"]) for r in rows)) + 1
    agent_w = min(agent_w, 40)  # cap width

    # Check if any agent has a known config tag
    has_config_tags = any(
        _get_agent_config_tag(r["info"]["agent"], r["metrics"].get("_config_info")) != "?"
        for r in rows
    )
    config_w = 15  # width for "NBV/Fusion" column

    # Build dynamic column list
    sep_parts = [
        "-" * (agent_w + 2), "-" * 8,
    ]
    header_parts = [
        f" {'Agent':<{agent_w}} ",
        f" {'Mode':<6} ",
    ]
    if has_config_tags:
        sep_parts.append("-" * (config_w + 2))
        header_parts.append(f" {'NBV/Fusion':<{config_w}} ")
    sep_parts.extend([
        "-" * 7, "-" * 10, "-" * 7,
        "-" * 14, "-" * 14, "-" * 14,
    ])
    header_parts.extend([
        f" {'N':>5} ",
        f" {'Overall':>8} ",
        f" {'ASD':>5} ",
        f" {'Positive':>12} ",
        f" {'Neg_Same':>12} ",
        f" {'Neg_Diff':>12} ",
    ])

    if has_nav_data:
        sep_parts.append("-" * 19)
        header_parts.append(f" {'NavFail(rate)':>17} ")
    if has_diag_data:
        sep_parts.extend(["-" * 9, "-" * 7])
        header_parts.extend([
            f" {'FV_Acc':>7} ",
            f" {'EffV':>5} ",
        ])

    sep = "+" + "+".join(sep_parts) + "+"

    print()
    print(sep)
    print("|" + "|".join(header_parts) + "|")
    print(sep)

    prev_agent = None
    for r in rows:
        info = r["info"]
        m = r["metrics"]

        agent_name = info["agent"]
        if len(agent_name) > agent_w:
            agent_name = agent_name[:agent_w - 2] + ".."

        # Add separator between different agents
        if prev_agent is not None and info["agent"] != prev_agent:
            print(sep)
        prev_agent = info["agent"]

        row_parts = [
            f" {agent_name:<{agent_w}} ",
            f" {info['mode']:<6} ",
        ]
        if has_config_tags:
            tag = _get_agent_config_tag(info["agent"], m.get("_config_info"))
            row_parts.append(f" {tag:<{config_w}} ")
        row_parts.extend([
            f" {m['total']:>5} ",
            f" {fmt_pct(m['accuracy'], 8)} ",
            f" {m['asd']:>5.2f} ",
            f" {fmt_pct_n(m['pos_acc'], m['pos_n'], 12)} ",
            f" {fmt_pct_n(m['neg_same_acc'], m['neg_same_n'], 12)} ",
            f" {fmt_pct_n(m['neg_diff_acc'], m['neg_diff_n'], 12)} ",
        ])

        if has_nav_data:
            nf = m.get("nav_fail_total", 0)
            nf_unreach = m.get("nav_fail_unreachable", 0)
            nf_trap = m.get("nav_fail_trap", 0)
            nf_step_rate = m.get("nav_fail_step_rate", 0)
            if nf > 0:
                nf_str = f"{nf_unreach}u+{nf_trap}t({nf_step_rate:.0%})"
            else:
                nf_str = "-"
            row_parts.append(f" {nf_str:>17} ")

        if has_diag_data:
            fv = m.get("fv_acc")
            fv_str = f"{fv:.1%}" if fv is not None else "-"
            eff = m.get("avg_eff_views")
            eff_str = f"{eff:.2f}" if eff is not None else "-"
            row_parts.extend([
                f" {fv_str:>7} ",
                f" {eff_str:>5} ",
            ])

        print("|" + "|".join(row_parts) + "|")

    print(sep)
    print(f"  Total: {len(rows)} runs")
    if has_nav_data:
        print("  NavFail: {unreachable}u+{trap_view}t({step_rate}) — u=unreachable, t=trap_view, rate=per-step failure rate")
    if has_config_tags:
        print("  NBV: fps=FarthestPoint llm=LLM random=Random oracle=Oracle vh=ViewHint")
        print("  Fusion: maj=Majority wt=Weighted llmf=LLM amaj=AttrMajority(adaptive) vis_w=VisWeighted(adaptive) astop=ScoreAdaptive(CLIP) +skip=SkipVerified +co=Cooccur")
    if has_diag_data:
        print(f"  FV_Acc=first-view accuracy | EffV=avg effective views")
    print()


def print_breakdown_table(rows, filter_mode=None):
    """Print per-pair-type breakdown of NavFail, FV_Acc, EffV."""
    if filter_mode:
        rows = [r for r in rows if r["info"]["mode"] == filter_mode]

    # Only show rows that have per_pair_type breakdown data
    rows = [r for r in rows if r["metrics"].get("_per_pair_type")]
    if not rows:
        print("\nNo per-pair-type breakdown data available.")
        print("(Re-run evaluate.py to regenerate metrics.json with breakdown stats)")
        return

    rows.sort(key=lambda r: (r["info"]["agent"], r["info"]["mode"]))

    PT_KEYS = ["positive", "neg_same", "neg_diff"]

    agent_w = max(5, max(len(r["info"]["agent"]) for r in rows)) + 1
    agent_w = min(agent_w, 34)

    # Build header
    # NavFail: pos/ns/nd, FV_Acc: pos/ns/nd, EffV: pos/ns/nd, F->C: pos/ns/nd, F->W: pos/ns/nd
    sub_w = 10  # width per sub-column
    group_w = sub_w * 3 + 4  # 3 sub-cols + 2 separators + 2 padding

    sep_parts = ["-" * (agent_w + 2), "-" * 8]
    header1_parts = [f" {'Agent':<{agent_w}} ", f" {'Mode':<6} "]
    header2_parts = [" " * (agent_w + 2), " " * 8]

    groups = [
        ("ASD", "asd"),
        ("NavFail (u+t)", "nav_fail"),
        ("FV_Acc", "fv_acc"),
        ("EffV", "eff_v"),
    ]

    for g_label, _ in groups:
        sep_parts.append("-" * (group_w + 2))
        header1_parts.append(f" {g_label:^{group_w}} ")
        subs = "  ".join(f"{'pos':>{sub_w-2}}" if i == 0
                         else f"{'ns':>{sub_w-2}}" if i == 1
                         else f"{'nd':>{sub_w-2}}"
                         for i in range(3))
        header2_parts.append(f" {subs:^{group_w}} ")

    sep = "+" + "+".join(sep_parts) + "+"

    print("\n  Per-Pair-Type Breakdown (pos=positive, ns=neg_same, nd=neg_diff)")
    print(sep)
    print("|" + "|".join(header1_parts) + "|")
    print("|" + "|".join(header2_parts) + "|")
    print(sep)

    prev_agent = None
    for r in rows:
        info = r["info"]
        m = r["metrics"]
        ppt = m.get("_per_pair_type", {})

        agent_name = info["agent"]
        if len(agent_name) > agent_w:
            agent_name = agent_name[:agent_w - 2] + ".."

        if prev_agent is not None and info["agent"] != prev_agent:
            print(sep)
        prev_agent = info["agent"]

        row_parts = [f" {agent_name:<{agent_w}} ", f" {info['mode']:<6} "]

        for _, g_key in groups:
            vals = []
            for pt in PT_KEYS:
                pt_data = ppt.get(pt, {})
                if g_key == "asd":
                    v = pt_data.get("asd")
                    s = f"{v:.2f}" if v is not None else "-"
                elif g_key == "nav_fail":
                    u = pt_data.get("nav_fail_unreachable", 0)
                    t = pt_data.get("nav_fail_trap", 0)
                    s = f"{u}u+{t}t" if (u + t) > 0 else "-"
                elif g_key == "fv_acc":
                    v = pt_data.get("first_view_accuracy")
                    s = f"{v:.1%}" if v is not None else "-"
                elif g_key == "eff_v":
                    v = pt_data.get("avg_effective_views")
                    s = f"{v:.2f}" if v is not None else "-"
                else:
                    s = "-"
                vals.append(f"{s:>{sub_w - 2}}")
            cell = "  ".join(vals)
            row_parts.append(f" {cell:^{group_w}} ")

        print("|" + "|".join(row_parts) + "|")

    print(sep)
    print("  NavFail: navigation failures ({unreachable}u + {trap_view}t)")
    print("  FV_Acc: first-view accuracy (if agent stopped after step 1)")
    print("  EffV: avg effective views (steps minus nav failures)")
    print()


def print_category_table(rows, filter_mode=None):
    """Print per-category accuracy breakdown by pair_type for each agent run."""
    if filter_mode:
        rows = [r for r in rows if r["info"]["mode"] == filter_mode]

    # Only show rows that have per_category data
    rows = [r for r in rows if r["metrics"].get("_per_category")]
    if not rows:
        print("\nNo per-category data available.")
        print("(Re-run evaluate.py or --recompute to regenerate metrics with category info)")
        return

    rows.sort(key=lambda r: (r["info"]["agent"], r["info"]["mode"]))

    # Collect all categories across all runs
    all_cats = set()
    for r in rows:
        all_cats.update(r["metrics"]["_per_category"].keys())
    all_cats = sorted(all_cats)

    if not all_cats:
        print("\nNo category data found.")
        return

    PT_KEYS = ["positive", "neg_same", "neg_diff"]

    # Print one table per agent run
    for r in rows:
        info = r["info"]
        m = r["metrics"]
        per_cat = m.get("_per_category", {})

        label = f"{info['agent']}  mode={info['mode']}"
        print(f"\n  Category Breakdown: {label}")

        # Column widths
        cat_w = max(10, max((len(c) for c in all_cats), default=10)) + 1
        cat_w = min(cat_w, 25)

        # Header: Category | N | Overall | pos acc(n) | ns acc(n) | nd acc(n)
        acc_w = 12
        sep_parts = [
            "-" * (cat_w + 2),
            "-" * 7,
            "-" * 9,
            "-" * (acc_w + 2),
            "-" * (acc_w + 2),
            "-" * (acc_w + 2),
        ]
        sep = "+" + "+".join(sep_parts) + "+"

        header = "|".join([
            f" {'Category':<{cat_w}} ",
            f" {'N':>5} ",
            f" {'Overall':>7} ",
            f" {'Positive':>{acc_w}} ",
            f" {'Neg_Same':>{acc_w}} ",
            f" {'Neg_Diff':>{acc_w}} ",
        ])
        print(sep)
        print("|" + header + "|")
        print(sep)

        for cat in all_cats:
            cd = per_cat.get(cat)
            if not cd:
                continue

            cat_display = cat if len(cat) <= cat_w else cat[:cat_w - 2] + ".."
            overall_acc = cd.get("accuracy", 0)
            total_n = cd.get("total", 0)

            pt_cells = []
            for pt in PT_KEYS:
                pt_data = cd.get(pt, {})
                pt_acc = pt_data.get("accuracy")
                pt_n = pt_data.get("total", 0)
                if pt_acc is not None and pt_n > 0:
                    pt_cells.append(f"{pt_acc:.0%} ({pt_n})".rjust(acc_w))
                else:
                    pt_cells.append("-".center(acc_w))

            row = "|".join([
                f" {cat_display:<{cat_w}} ",
                f" {total_n:>5} ",
                f" {overall_acc:.1%}".rjust(8) + " ",
                f" {pt_cells[0]} ",
                f" {pt_cells[1]} ",
                f" {pt_cells[2]} ",
            ])
            print("|" + row + "|")

        print(sep)
    print()


def export_csv(rows, csv_path, filter_mode=None):
    """Export results to CSV."""
    if filter_mode:
        rows = [r for r in rows if r["info"]["mode"] == filter_mode]

    rows.sort(key=lambda r: (r["info"]["agent"], r["info"]["mode"]))

    headers = [
        "agent", "mode", "nbv_fusion", "split", "total", "accuracy", "asd",
        "correct", "wrong",
        "pos_acc", "pos_n", "pos_asd",
        "neg_same_acc", "neg_same_n", "neg_same_asd",
        "neg_diff_acc", "neg_diff_n", "neg_diff_asd",
        "nav_fail_total", "nav_fail_unreachable", "nav_fail_trap",
        "fv_acc", "avg_eff_views",
        "run_dir"
    ]

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            info = r["info"]
            m = r["metrics"]
            vals = [
                info["agent"],
                info["mode"],
                _get_agent_config_tag(info["agent"], m.get("_config_info")),
                info["split"],
                str(m["total"]),
                f"{m['accuracy']:.4f}" if m["accuracy"] is not None else "",
                f"{m['asd']:.2f}",
                str(m["correct"]),
                str(m["wrong"]),
                f"{m['pos_acc']:.4f}" if m["pos_acc"] is not None else "",
                str(m["pos_n"]),
                f"{m.get('_per_pair_type', {}).get('positive', {}).get('asd', '')}" if m.get("_per_pair_type", {}).get("positive", {}).get("asd") is not None else "",
                f"{m['neg_same_acc']:.4f}" if m["neg_same_acc"] is not None else "",
                str(m["neg_same_n"]),
                f"{m.get('_per_pair_type', {}).get('neg_same', {}).get('asd', '')}" if m.get("_per_pair_type", {}).get("neg_same", {}).get("asd") is not None else "",
                f"{m['neg_diff_acc']:.4f}" if m["neg_diff_acc"] is not None else "",
                str(m["neg_diff_n"]),
                f"{m.get('_per_pair_type', {}).get('neg_diff', {}).get('asd', '')}" if m.get("_per_pair_type", {}).get("neg_diff", {}).get("asd") is not None else "",
                str(m.get("nav_fail_total", 0)),
                str(m.get("nav_fail_unreachable", 0)),
                str(m.get("nav_fail_trap", 0)),
                f"{m['fv_acc']:.4f}" if m.get("fv_acc") is not None else "",
                f"{m['avg_eff_views']:.2f}" if m.get("avg_eff_views") is not None else "",
                info["run_dir"],
            ]
            f.write(",".join(vals) + "\n")

    print(f"Exported {len(rows)} rows to {csv_path}")


def _load_category_map(index_file):
    """Load object_id → category mapping from JSONL index file."""
    obj_cat = {}  # object_id → category
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tid = rec.get("target_object_id")
                tcat = rec.get("target_object_category")
                if tid and tcat:
                    obj_cat[tid] = tcat
                qid = rec.get("query_object_id")
                qcat = rec.get("query_object_category")
                if qid and qcat:
                    obj_cat[qid] = qcat
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARN] Failed to load index file {index_file}: {e}")
    return obj_cat


def _backfill_category_fields(all_results, obj_cat_map):
    """Backfill target_object_category and query_object_category from index mapping."""
    patched = 0
    for r in all_results:
        if r.get("target_object_category") and r["target_object_category"] != "unknown":
            continue
        tid = r.get("target_object_id")
        qid = r.get("object_id")  # query object stored as object_id in older results
        if tid and tid in obj_cat_map:
            r["target_object_category"] = obj_cat_map[tid]
        if qid and qid in obj_cat_map:
            r["query_object_category"] = obj_cat_map[qid]
        elif tid and tid in obj_cat_map:
            # For positive pairs, query = target
            r["query_object_category"] = obj_cat_map[tid]
        patched += 1
    return patched


def _backfill_diagnostic_fields(all_results):
    """Backfill first_view_prediction and effective_views from transcript.

    Older evaluate.py versions didn't compute these fields. This reconstructs
    them from the saved transcript data so --recompute produces full breakdowns.
    """
    patched = 0

    for r in all_results:
        # Skip if already has diagnostic fields
        if r.get("first_view_prediction") is not None and r.get("effective_views") is not None:
            continue

        transcript = r.get("transcript", [])

        # effective_views
        if "effective_views" not in r:
            steps = r.get("steps", len(transcript))
            nf_u = r.get("nav_fail_unreachable", 0)
            nf_t = r.get("nav_fail_trap", 0)
            r["effective_views"] = steps - nf_u - nf_t

        # first_view_prediction from transcript
        if "first_view_prediction" not in r:
            step_preds = [
                rec.get("action", {}).get("_step_prediction")
                for rec in transcript
            ]
            step_preds = [p for p in step_preds if p is not None]
            if step_preds:
                r["first_view_prediction"] = step_preds[0]
            else:
                r["first_view_prediction"] = r.get("prediction", "No")

        patched += 1

    return patched


def recompute_metrics(entries, index_file=None):
    """Recompute metrics.json from results.json for all found run directories."""
    # Ensure project root is on sys.path so pver can be imported
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from pver.eval.metrics import calculate_metrics

    # Load category map from index file if provided
    obj_cat_map = _load_category_map(index_file) if index_file else {}
    if obj_cat_map:
        print(f"[CATEGORY] Loaded {len(obj_cat_map)} object→category mappings from {index_file}")

    # Collect unique directories that have results.json
    dirs_seen = set()
    for _, filepath in entries:
        run_dir = filepath.parent
        if run_dir in dirs_seen:
            continue
        dirs_seen.add(run_dir)

        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not all_results:
            continue

        # Backfill diagnostic fields from transcript for older results
        patched = _backfill_diagnostic_fields(all_results)

        # Backfill category fields from index if available
        cat_patched = 0
        if obj_cat_map:
            cat_patched = _backfill_category_fields(all_results, obj_cat_map)

        if patched or cat_patched:
            # Save backfilled results.json
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)

        # Preserve existing config_info if metrics.json already exists
        config_info = None
        old_metrics_file = run_dir / "metrics.json"
        if old_metrics_file.exists():
            try:
                with open(old_metrics_file, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                config_info = old_data.get("config_info")
            except (json.JSONDecodeError, OSError):
                pass

        metrics = calculate_metrics(all_results)
        if config_info:
            metrics["config_info"] = config_info

        with open(old_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        patch_parts = []
        if patched:
            patch_parts.append(f"diag={patched}")
        if cat_patched:
            patch_parts.append(f"cat={cat_patched}")
        patch_msg = f" (backfilled {', '.join(patch_parts)})" if patch_parts else ""
        print(f"[RECOMPUTE] {run_dir.name}: {len(all_results)} episodes -> metrics.json updated{patch_msg}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare metrics across agent runs")
    parser.add_argument("dirs", nargs="+", help="Output directories to scan")
    parser.add_argument("--csv", type=str, default=None, help="Export to CSV file")
    parser.add_argument("--sort", type=str, default=None,
                        choices=["accuracy", "asd", "pos_acc", "neg_same_acc", "neg_diff_acc", "fv_acc", "agent"],
                        help="Sort by field")
    parser.add_argument("--mode", type=str, default=None, choices=["gt", "dino"],
                        help="Filter by bbox mode")
    parser.add_argument("--breakdown", action="store_true",
                        help="Show per-pair-type breakdown of NavFail/FV_Acc/EffV")
    parser.add_argument("--category", action="store_true",
                        help="Show per-category accuracy breakdown by pair_type")
    parser.add_argument("--index-file", type=str, default=None,
                        help="JSONL index file for backfilling category info in old results")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute metrics.json from results.json (updates breakdown stats)")
    args = parser.parse_args()

    # Find all metrics files
    entries = find_metrics_files(args.dirs)

    if not entries:
        print(f"No metrics files found in: {args.dirs}")
        sys.exit(1)

    # Deduplicate: if both metrics.json and batch_summary.json exist in same dir, prefer metrics.json
    seen_dirs = {}
    for fmt, path in entries:
        d = str(path.parent)
        if d in seen_dirs:
            # Prefer metrics.json over batch_summary.json
            if fmt == "metrics":
                seen_dirs[d] = (fmt, path)
        else:
            seen_dirs[d] = (fmt, path)

    entries = list(seen_dirs.values())

    # Recompute metrics from results.json if requested
    if args.recompute:
        recompute_metrics(entries, index_file=args.index_file)
        # Re-scan after recompute
        entries = find_metrics_files(args.dirs)
        seen_dirs = {}
        for fmt, path in entries:
            d = str(path.parent)
            if d in seen_dirs:
                if fmt == "metrics":
                    seen_dirs[d] = (fmt, path)
            else:
                seen_dirs[d] = (fmt, path)
        entries = list(seen_dirs.values())

    # Parse and collect
    rows = []
    for fmt, filepath in entries:
        info = parse_agent_info_from_path(filepath)
        metrics = extract_metrics(fmt, filepath)
        if metrics:
            rows.append({"info": info, "metrics": metrics})

    if not rows:
        print("No valid metrics found.")
        sys.exit(1)

    # Print table
    print_table(rows, sort_key=args.sort, filter_mode=args.mode)

    # Print per-pair-type breakdown if requested
    if args.breakdown:
        print_breakdown_table(rows, filter_mode=args.mode)

    # Print per-category breakdown if requested
    if args.category:
        print_category_table(rows, filter_mode=args.mode)

    # Export CSV if requested
    if args.csv:
        export_csv(rows, args.csv, filter_mode=args.mode)


if __name__ == "__main__":
    main()
