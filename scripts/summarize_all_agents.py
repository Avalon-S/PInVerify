#!/usr/bin/env python3
"""
Summarize results from all agents after multi-GPU evaluation.

Generates comparison tables and saves to summary report.

Usage:
    python scripts/summarize_all_agents.py --output_base ./outputs/sectors6_500_dynamic
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_metrics(metrics_file: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not os.path.exists(metrics_file):
        return None

    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_agent_name(dirname: str) -> tuple:
    """
    Extract agent name and mode from directory name.

    Examples:
        multi_view_attr_llm_dino_500 -> (multi_view_attr_llm, dino)
        single_view_direct_gt_500 -> (single_view_direct, gt)
    """
    parts = dirname.rsplit('_', 2)  # Split from right: name, mode, limit
    if len(parts) >= 2:
        agent_name = parts[0]
        mode = parts[1]  # dino or gt
        return agent_name, mode
    return dirname, 'unknown'


def main():
    parser = argparse.ArgumentParser(description="Summarize all agent results")
    parser.add_argument("--output_base", type=str, required=True,
                       help="Base output directory containing all agent results")
    args = parser.parse_args()

    output_base = args.output_base

    if not os.path.exists(output_base):
        print(f"Error: Output directory not found: {output_base}")
        return 1

    print(f"\n{'='*80}")
    print("Collecting results from all agents...")
    print(f"{'='*80}\n")

    # Collect all metrics
    results = {}  # {agent_name: {mode: metrics}}

    for entry in os.listdir(output_base):
        entry_path = os.path.join(output_base, entry)
        if not os.path.isdir(entry_path):
            continue

        metrics_file = os.path.join(entry_path, "metrics.json")
        metrics = load_metrics(metrics_file)

        if metrics is None:
            print(f"[SKIP] No metrics found: {entry}")
            continue

        agent_name, mode = extract_agent_name(entry)

        if agent_name not in results:
            results[agent_name] = {}

        results[agent_name][mode] = metrics
        print(f"[LOADED] {agent_name} ({mode})")

    if not results:
        print("\nNo results found!")
        return 1

    print(f"\nTotal agents: {len(results)}")
    print(f"{'='*80}\n")

    # Generate summary table
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("Multi-Agent Evaluation Summary")
    summary_lines.append("="*80)
    summary_lines.append("")

    # Header
    header = f"{'Agent':<40} | {'DINO SR':>8} | {'DINO AvgR':>8} | {'GT SR':>8} | {'GT AvgR':>8}"
    summary_lines.append(header)
    summary_lines.append("-"*80)

    # Sort agents by name
    sorted_agents = sorted(results.keys())

    for agent_name in sorted_agents:
        modes = results[agent_name]

        dino_sr = modes.get('dino', {}).get('success_rate', 0.0)
        dino_avg_r = modes.get('dino', {}).get('avg_reward', 0.0)
        gt_sr = modes.get('gt', {}).get('success_rate', 0.0)
        gt_avg_r = modes.get('gt', {}).get('avg_reward', 0.0)

        line = f"{agent_name:<40} | {dino_sr:>8.4f} | {dino_avg_r:>8.4f} | {gt_sr:>8.4f} | {gt_avg_r:>8.4f}"
        summary_lines.append(line)

    summary_lines.append("="*80)
    summary_lines.append("")
    summary_lines.append("Metrics Legend:")
    summary_lines.append("  SR (Success Rate): Percentage of successful episodes")
    summary_lines.append("  AvgR (Average Reward): Mean reward across all episodes")
    summary_lines.append("  DINO: Using GroundingDINO detector")
    summary_lines.append("  GT: Using ground-truth bounding boxes")
    summary_lines.append("="*80)

    # Print to console
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save to file
    summary_file = os.path.join(output_base, "summary_report.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\nSummary report saved to: {summary_file}")

    # Also save as JSON for programmatic access
    summary_json = {
        'agents': {}
    }

    for agent_name in sorted_agents:
        summary_json['agents'][agent_name] = results[agent_name]

    summary_json_file = os.path.join(output_base, "summary_report.json")
    with open(summary_json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print(f"Summary JSON saved to: {summary_json_file}")
    print(f"\n{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
