#!/usr/bin/env python
"""Analyze prediction bias across models/agents.

Computes Yes-rate, precision, recall, and F1 per pair-type
to reveal whether a model is "guessing" or genuinely discriminating.

Usage:
    python scripts/analyze_prediction_bias.py <results_dir1> [results_dir2] ...
    python scripts/analyze_prediction_bias.py outputs/4B_run outputs/8B_run
"""
import json, sys, os
from collections import defaultdict
from pathlib import Path


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze_run(results):
    """Compute prediction bias statistics."""
    pred_map = {"Yes": 1, "No": 0, "Unsure": 0}
    stats = {
        "total": 0,
        "yes_pred": 0,  # How many times model said Yes
        "no_pred": 0,
        "by_type": defaultdict(lambda: {
            "total": 0, "correct": 0, "yes_pred": 0, "no_pred": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0
        })
    }

    for ep in results:
        label = ep.get("label", 1)
        pair_type = ep.get("pair_type", "unknown")
        pred = ep.get("prediction", "No")
        pred_val = pred_map.get(pred, 0)
        is_correct = (pred_val == label)
        is_yes = (pred_val == 1)

        stats["total"] += 1
        if is_yes:
            stats["yes_pred"] += 1
        else:
            stats["no_pred"] += 1

        bt = stats["by_type"][pair_type]
        bt["total"] += 1
        if is_correct:
            bt["correct"] += 1
        if is_yes:
            bt["yes_pred"] += 1
        else:
            bt["no_pred"] += 1

        # Confusion matrix (Yes=positive class)
        if label == 1 and is_yes:
            bt["tp"] += 1
        elif label == 0 and is_yes:
            bt["fp"] += 1
        elif label == 0 and not is_yes:
            bt["tn"] += 1
        elif label == 1 and not is_yes:
            bt["fn"] += 1

    return stats


def find_runs(dirs):
    """Find all results.json files in given directories."""
    runs = []
    for d in dirs:
        p = Path(d)
        if p.is_file() and p.name == "results.json":
            runs.append(p)
        else:
            for rj in sorted(p.rglob("results.json")):
                # Skip worker subdirectories (multigpu splits)
                if "worker_" in str(rj.parent.name):
                    continue
                runs.append(rj)
    return runs


def extract_run_info(path):
    """Extract agent name, mode, split from directory name."""
    dirname = path.parent.name
    parts = dirname.rsplit("_", 2)
    if len(parts) >= 3:
        agent = parts[0]
        mode = parts[1]
        split = parts[2]
        return agent, mode, split
    return dirname, "?", "?"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_prediction_bias.py <dir1> [dir2] ...")
        sys.exit(1)

    runs = find_runs(sys.argv[1:])
    if not runs:
        print("No results.json found.")
        sys.exit(1)

    print(f"\n=== PREDICTION BIAS ANALYSIS ===")
    print(f"Found {len(runs)} runs\n")

    # Header
    print(f"{'Agent':<35s} {'Mode':<6s} {'N':>5s}  {'Yes%':>6s}  "
          f"{'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  "
          f"{'Pos_Y%':>7s} {'NS_Y%':>7s} {'ND_Y%':>7s}")
    print("-" * 120)

    for rj in runs:
        agent, mode, split = extract_run_info(rj)
        results = load_results(rj)
        stats = analyze_run(results)

        total = stats["total"]
        yes_rate = stats["yes_pred"] / total * 100 if total else 0

        # Overall accuracy
        correct = sum(bt["correct"] for bt in stats["by_type"].values())
        acc = correct / total * 100 if total else 0

        # Precision, Recall, F1 (Yes = positive class)
        tp = sum(bt["tp"] for bt in stats["by_type"].values())
        fp = sum(bt["fp"] for bt in stats["by_type"].values())
        fn = sum(bt["fn"] for bt in stats["by_type"].values())
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Per-type Yes rate
        pos_yes = stats["by_type"]["positive"]["yes_pred"]
        pos_total = stats["by_type"]["positive"]["total"]
        pos_yes_rate = pos_yes / pos_total * 100 if pos_total else 0

        ns_yes = stats["by_type"]["neg_same"]["yes_pred"]
        ns_total = stats["by_type"]["neg_same"]["total"]
        ns_yes_rate = ns_yes / ns_total * 100 if ns_total else 0

        nd_yes = stats["by_type"]["neg_diff"]["yes_pred"]
        nd_total = stats["by_type"]["neg_diff"]["total"]
        nd_yes_rate = nd_yes / nd_total * 100 if nd_total else 0

        print(f"{agent:<35s} {mode:<6s} {total:>5d}  {yes_rate:>5.1f}%  "
              f"{acc:>5.1f}%  {precision:>5.1f}%  {recall:>5.1f}%  {f1:>5.1f}%  "
              f"{pos_yes_rate:>6.1f}% {ns_yes_rate:>6.1f}% {nd_yes_rate:>6.1f}%")

    print()
    print("  Yes% = overall Yes prediction rate (model bias)")
    print("  Prec = of all Yes predictions, how many are truly positive")
    print("  Rec  = of all positive cases, how many predicted Yes")
    print("  Pos_Y% = Yes rate on positive (=Recall), NS_Y% = Yes rate on neg_same (=FP rate), ND_Y% = Yes rate on neg_diff")
    print()


if __name__ == "__main__":
    main()
