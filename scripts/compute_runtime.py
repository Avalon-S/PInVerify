#!/usr/bin/env python3
"""Compute experiment runtime from file modification timestamps.

For each experiment directory under outputs/, scans all files,
finds the earliest and latest mtime, and reports the difference
as the wall-clock runtime.

If worker_X subdirectories exist, reports:
  - wall-clock time (earliest to latest across all workers)
  - num_workers
  - estimated single-GPU time = wall_clock × num_workers

If no worker dirs, treats as single-GPU (num_workers=1).

Usage (on server):
    python scripts/compute_runtime.py --root /path/to/outputs

    # Only specific model:
    python scripts/compute_runtime.py --root /path/to/outputs --model qwen3_vl_4b

    # Output as LaTeX table row:
    python scripts/compute_runtime.py --root /path/to/outputs --latex
"""

import argparse
import os
import re
from pathlib import Path
from datetime import timedelta


def scan_timestamps(directory: Path):
    """Scan all files in directory, return (earliest_mtime, latest_mtime, file_count)."""
    earliest = float("inf")
    latest = float("-inf")
    count = 0

    for root, _dirs, files in os.walk(directory):
        for f in files:
            fpath = Path(root) / f
            try:
                mtime = fpath.stat().st_mtime
                if mtime < earliest:
                    earliest = mtime
                if mtime > latest:
                    latest = mtime
                count += 1
            except OSError:
                continue

    if count == 0:
        return None, None, 0
    return earliest, latest, count


def count_workers(exp_dir: Path) -> int:
    """Count worker_X subdirectories. Returns 0 if none found."""
    worker_dirs = [
        d for d in exp_dir.iterdir()
        if d.is_dir() and re.match(r"worker_\d+", d.name)
    ]
    return len(worker_dirs)


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable string."""
    td = timedelta(seconds=seconds)
    total_secs = int(td.total_seconds())
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"


def format_minutes(seconds: float) -> str:
    """Format seconds as decimal minutes for LaTeX."""
    return f"{seconds / 60:.1f}"


def main():
    parser = argparse.ArgumentParser(description="Compute experiment runtimes from file timestamps")
    parser.add_argument("--root", type=str, required=True, help="Path to outputs/ directory")
    parser.add_argument("--model", type=str, default=None, help="Only scan this model subdirectory")
    parser.add_argument("--latex", action="store_true", help="Output as LaTeX table rows")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: {root} does not exist")
        return

    # Collect all model dirs
    if args.model:
        model_dirs = [root / args.model]
    else:
        model_dirs = sorted([
            d for d in root.iterdir()
            if d.is_dir() and not d.name.endswith(".zip")
        ])

    results = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        # Each subdirectory is an experiment
        exp_dirs = sorted([
            d for d in model_dir.iterdir()
            if d.is_dir()
        ])

        for exp_dir in exp_dirs:
            earliest, latest, count = scan_timestamps(exp_dir)
            if earliest is None:
                continue

            wall_clock = latest - earliest
            n_workers = count_workers(exp_dir)
            if n_workers == 0:
                n_workers = 1  # single GPU, no worker dirs

            single_gpu = wall_clock * n_workers

            results.append({
                "model": model_name,
                "experiment": exp_dir.name,
                "wall_clock_s": wall_clock,
                "wall_clock_str": format_duration(wall_clock),
                "n_workers": n_workers,
                "single_gpu_s": single_gpu,
                "single_gpu_str": format_duration(single_gpu),
                "file_count": count,
            })

    # Print results
    if args.latex:
        print("% Model & Experiment & Wall-Clock & GPUs & Single-GPU Time \\\\")
        for r in results:
            exp_escaped = r["experiment"].replace("_", r"\_")
            print(
                f"{r['model']} & {exp_escaped} & "
                f"{r['wall_clock_str']} & {r['n_workers']} & "
                f"{r['single_gpu_str']} \\\\"
            )
    else:
        # Pretty table
        header = (
            f"{'Model':<40} {'Experiment':<45} "
            f"{'Wall-Clock':>12} {'GPUs':>5} {'1-GPU Time':>12} {'Files':>8}"
        )
        print(header)
        print("-" * len(header))
        current_model = None
        for r in results:
            model_display = r["model"] if r["model"] != current_model else ""
            current_model = r["model"]
            print(
                f"{model_display:<40} {r['experiment']:<45} "
                f"{r['wall_clock_str']:>12} {r['n_workers']:>5} "
                f"{r['single_gpu_str']:>12} {r['file_count']:>8}"
            )
        print("-" * len(header))

        # Summary: total single-GPU time per model
        print("\nPer-model total single-GPU time:")
        from collections import defaultdict
        model_totals = defaultdict(float)
        for r in results:
            model_totals[r["model"]] += r["single_gpu_s"]
        for model, total in sorted(model_totals.items()):
            print(f"  {model:<40} {format_duration(total):>12}")


if __name__ == "__main__":
    main()
