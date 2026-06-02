#!/usr/bin/env python
"""Cross-model episode-level comparison.

Pairs up episodes from two runs (e.g. 4B vs 8B) on the same dataset
and cross-tabulates their predictions to reveal agreement/disagreement patterns.

Usage:
    python scripts/cross_model_compare.py <results_A.json> <results_B.json> [--name_a 4B] [--name_b 8B]

Example:
    python scripts/cross_model_compare.py outputs/4B/single_view_attr_gt_500/results.json \
                                          outputs/8B/single_view_attr_gt_500/results.json \
                                          --name_a 4B --name_b 8B
"""
import json, sys, argparse
from collections import defaultdict


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_key(ep):
    """Build a unique episode key for cross-run matching."""
    return (ep.get("scene_id", ""), ep.get("episode_id", ""), ep.get("object_id", ""))


def main():
    parser = argparse.ArgumentParser(description="Cross-model episode comparison")
    parser.add_argument("results_a", help="Path to model A results.json")
    parser.add_argument("results_b", help="Path to model B results.json")
    parser.add_argument("--name_a", default="Model_A", help="Display name for model A")
    parser.add_argument("--name_b", default="Model_B", help="Display name for model B")
    args = parser.parse_args()

    ra = load_results(args.results_a)
    rb = load_results(args.results_b)
    na, nb = args.name_a, args.name_b

    # Index by key
    idx_a = {make_key(ep): ep for ep in ra}
    idx_b = {make_key(ep): ep for ep in rb}

    matched_keys = sorted(set(idx_a.keys()) & set(idx_b.keys()))
    only_a = set(idx_a.keys()) - set(idx_b.keys())
    only_b = set(idx_b.keys()) - set(idx_a.keys())

    print(f"\n=== CROSS-MODEL COMPARISON: {na} vs {nb} ===")
    print(f"{na}: {len(ra)} episodes  |  {nb}: {len(rb)} episodes  |  Matched: {len(matched_keys)}")
    if only_a:
        print(f"  Only in {na}: {len(only_a)}")
    if only_b:
        print(f"  Only in {nb}: {len(only_b)}")

    pred_map = {"Yes": 1, "No": 0, "Unsure": 0}

    # ── 1. Overall cross-tabulation ──
    # Cells: (A_pred, B_pred) → count, correct_A, correct_B
    cross = defaultdict(lambda: {"n": 0, "a_correct": 0, "b_correct": 0})
    # Per pair_type
    cross_by_type = defaultdict(lambda: defaultdict(lambda: {"n": 0, "a_correct": 0, "b_correct": 0}))

    for key in matched_keys:
        ea, eb = idx_a[key], idx_b[key]
        label = ea.get("label", 1)
        pair_type = ea.get("pair_type", "unknown")
        pa = ea.get("prediction", "No")
        pb = eb.get("prediction", "No")
        va = pred_map.get(pa, 0)
        vb = pred_map.get(pb, 0)
        ca = int(va == label)
        cb = int(vb == label)

        cell = f"{pa}/{pb}"
        cross[cell]["n"] += 1
        cross[cell]["a_correct"] += ca
        cross[cell]["b_correct"] += cb

        cross_by_type[pair_type][cell]["n"] += 1
        cross_by_type[pair_type][cell]["a_correct"] += ca
        cross_by_type[pair_type][cell]["b_correct"] += cb

    # ── Print cross table ──
    print(f"\n{'─'*60}")
    print(f"  PREDICTION CROSS-TABLE (overall, N={len(matched_keys)})")
    print(f"{'─'*60}")
    print(f"  {'':>20s}  {nb}=Yes    {nb}=No")
    print(f"  {na+'=Yes':>20s}  {cross['Yes/Yes']['n']:>6d}    {cross['Yes/No']['n']:>6d}")
    print(f"  {na+'=No':>20s}  {cross['No/Yes']['n']:>6d}    {cross['No/No']['n']:>6d}")

    # Agreement rate
    agree = cross["Yes/Yes"]["n"] + cross["No/No"]["n"]
    print(f"\n  Agreement: {agree}/{len(matched_keys)} ({agree/len(matched_keys)*100:.1f}%)")

    # ── 2. Disagreement analysis ──
    print(f"\n{'─'*60}")
    print(f"  DISAGREEMENT ANALYSIS")
    print(f"{'─'*60}")

    # A=Yes, B=No
    aY_bN = cross["Yes/No"]
    if aY_bN["n"] > 0:
        print(f"\n  {na}=Yes, {nb}=No  ({aY_bN['n']} episodes):")
        print(f"    {na} correct: {aY_bN['a_correct']}/{aY_bN['n']} ({aY_bN['a_correct']/aY_bN['n']*100:.1f}%)")
        print(f"    {nb} correct: {aY_bN['b_correct']}/{aY_bN['n']} ({aY_bN['b_correct']/aY_bN['n']*100:.1f}%)")

    # A=No, B=Yes
    aN_bY = cross["No/Yes"]
    if aN_bY["n"] > 0:
        print(f"\n  {na}=No, {nb}=Yes  ({aN_bY['n']} episodes):")
        print(f"    {na} correct: {aN_bY['a_correct']}/{aN_bY['n']} ({aN_bY['a_correct']/aN_bY['n']*100:.1f}%)")
        print(f"    {nb} correct: {aN_bY['b_correct']}/{aN_bY['n']} ({aN_bY['b_correct']/aN_bY['n']*100:.1f}%)")

    # ── 3. Per pair_type breakdown ──
    for pt in ["positive", "neg_same", "neg_diff"]:
        ct = cross_by_type.get(pt, {})
        total = sum(v["n"] for v in ct.values())
        if total == 0:
            continue

        print(f"\n{'─'*60}")
        print(f"  PAIR TYPE: {pt} (N={total})")
        print(f"{'─'*60}")
        print(f"  {'':>20s}  {nb}=Yes    {nb}=No")
        print(f"  {na+'=Yes':>20s}  {ct.get('Yes/Yes',{}).get('n',0):>6d}    {ct.get('Yes/No',{}).get('n',0):>6d}")
        print(f"  {na+'=No':>20s}  {ct.get('No/Yes',{}).get('n',0):>6d}    {ct.get('No/No',{}).get('n',0):>6d}")

        # Disagreement detail for this type
        aY_bN = ct.get("Yes/No", {"n": 0, "a_correct": 0, "b_correct": 0})
        aN_bY = ct.get("No/Yes", {"n": 0, "a_correct": 0, "b_correct": 0})

        if aY_bN["n"] > 0:
            print(f"    {na}=Yes,{nb}=No: {aY_bN['n']} eps → "
                  f"{na} correct {aY_bN['a_correct']}/{aY_bN['n']} "
                  f"({aY_bN['a_correct']/aY_bN['n']*100:.0f}%), "
                  f"{nb} correct {aY_bN['b_correct']}/{aY_bN['n']} "
                  f"({aY_bN['b_correct']/aY_bN['n']*100:.0f}%)")
        if aN_bY["n"] > 0:
            print(f"    {na}=No,{nb}=Yes: {aN_bY['n']} eps → "
                  f"{na} correct {aN_bY['a_correct']}/{aN_bY['n']} "
                  f"({aN_bY['a_correct']/aN_bY['n']*100:.0f}%), "
                  f"{nb} correct {aN_bY['b_correct']}/{aN_bY['n']} "
                  f"({aN_bY['b_correct']/aN_bY['n']*100:.0f}%)")

    # ── 4. Summary verdict ──
    print(f"\n{'─'*60}")
    print(f"  SUMMARY")
    print(f"{'─'*60}")
    total_a_correct = sum(v["a_correct"] for v in cross.values())
    total_b_correct = sum(v["b_correct"] for v in cross.values())
    n = len(matched_keys)
    print(f"  {na} accuracy: {total_a_correct}/{n} ({total_a_correct/n*100:.1f}%)")
    print(f"  {nb} accuracy: {total_b_correct}/{n} ({total_b_correct/n*100:.1f}%)")

    # On disagreements: who wins?
    disagree_n = aY_bN_total = aN_bY_total = 0
    a_wins = b_wins = both_wrong = 0
    for cell, v in cross.items():
        parts = cell.split("/")
        if parts[0] != parts[1]:  # disagreement
            disagree_n += v["n"]
            a_wins += v["a_correct"] - min(v["a_correct"], v["b_correct"])  # A right, B wrong
            b_wins += v["b_correct"] - min(v["a_correct"], v["b_correct"])  # B right, A wrong

    # Simpler: count per episode
    a_wins = b_wins = both_wrong_disagree = 0
    for key in matched_keys:
        ea, eb = idx_a[key], idx_b[key]
        label = ea.get("label", 1)
        pa = pred_map.get(ea.get("prediction", "No"), 0)
        pb = pred_map.get(eb.get("prediction", "No"), 0)
        if pa == pb:
            continue
        ca, cb = (pa == label), (pb == label)
        if ca and not cb:
            a_wins += 1
        elif cb and not ca:
            b_wins += 1
        else:
            both_wrong_disagree += 1

    print(f"\n  Disagreements: {a_wins + b_wins + both_wrong_disagree} episodes")
    if a_wins + b_wins + both_wrong_disagree > 0:
        print(f"    {na} right, {nb} wrong: {a_wins}")
        print(f"    {nb} right, {na} wrong: {b_wins}")
        print(f"    Both wrong (different): {both_wrong_disagree}")
    print()


if __name__ == "__main__":
    main()
