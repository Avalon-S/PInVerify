"""
Analyze NBV direction output distribution by strategy (Random, FPS, LLM).

For each strategy, shows:
  - Step 1 direction distribution (first navigation decision)
  - Step 2 direction distribution (second navigation decision)
  - Overall distribution (all nav steps combined)

Usage:
  # Auto-scan all subdirs under a base directory (single-view episodes skipped automatically):
  python scripts/analyze_nbv_distribution.py --base_dir outputs/500_samples
  python scripts/analyze_nbv_distribution.py --base_dir outputs/500_samples --output report_figs/nbv_dist.png

  # From a single results JSON (may contain mixed strategies):
  python scripts/analyze_nbv_distribution.py --results results.json

  # From a directory of episode.json files:
  python scripts/analyze_nbv_distribution.py --ep_dir outputs/500_samples/qwen3vl_4b_mv_random

  # From multiple result files (one per strategy):
  python scripts/analyze_nbv_distribution.py \
      --results results_random.json results_fps.json results_llm.json

  # Save plot:
  python scripts/analyze_nbv_distribution.py --results results.json --output report_figs/nbv_dist.png
"""

import argparse
import json
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Direction labels (excluding "front" = stay in place)
ALL_DIRS = ["front-left", "back-left", "back", "back-right", "front-right"]
DIR_SHORT = {
    "front-left":  "FL",
    "back-left":   "BL",
    "back":        "B",
    "back-right":  "BR",
    "front-right": "FR",
}


def detect_strategy(nav_rel: str, nbv_debug: dict) -> str:
    """Identify NBV strategy from nav_rel label and _nbv_debug fields."""
    if "(Random)" in nav_rel:
        return "Random"
    if "(MLLM-Reasoning)" in nav_rel:
        # Could be LLMBasedNBV or LLMViewHintNBV
        if nbv_debug.get("target_attributes") is not None:
            return "ViewHint"
        return "LLM"
    mode = nbv_debug.get("mode", "")
    if mode == "farthest_point" or mode == "fps_angular":
        return "FPS"
    if mode == "oracle":
        return "Oracle"
    if mode == "evidence_gap_nbv":
        return "EvidenceGap"
    if mode == "random":
        return "Random"
    return "Unknown"


def extract_direction(nav_rel: str) -> str:
    """Extract clean direction name from nav_rel string."""
    # e.g. "back (Random)" -> "back", "front-left (MLLM-Reasoning)" -> "front-left"
    direction = nav_rel.split("(")[0].strip().lower()
    # Normalize
    valid = set(ALL_DIRS) | {"front"}
    if direction in valid:
        return direction
    # Fallback: search for known direction strings
    for d in sorted(valid, key=len, reverse=True):
        if d in nav_rel.lower():
            return d
    return "unknown"


def load_transcripts_from_file(path: str) -> list:
    """Load episode transcripts from a results JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # Handle dict wrapper
    if "results" in data:
        return data["results"]
    return []


def is_multiview_episode(episode: dict) -> bool:
    """
    Return True if this episode has at least one real navigation step
    (nav_rel is not 'front' and not empty). Single-view episodes are skipped.
    """
    for step_rec in episode.get("transcript", []):
        nav_rel = step_rec.get("action", {}).get("nav_rel", "")
        direction = extract_direction(nav_rel)
        if nav_rel and direction not in ("front", "unknown", ""):
            return True
    return False


def _inc(dist: dict, strategy: str, bucket, key: str, n: int = 1):
    """Safely increment dist[strategy][bucket][key] by n, initializing as needed."""
    s = dist.setdefault(strategy, {})
    b = s.setdefault(bucket, {})
    b[key] = b.get(key, 0) + n


def accumulate_episode(ep: dict, dist: dict, strategy_suffix: str = ""):
    """
    Accumulate one episode's nav decisions into dist in-place.
    dist format: {strategy: {step_num|"all": {direction: count}}}
    strategy_suffix: appended to detected strategy name, e.g. " (adap)" or " (fixed)"
    """
    for step_rec in ep.get("transcript", []):
        step_num = step_rec.get("step", 0)
        action = step_rec.get("action", {})
        nav_rel = action.get("nav_rel", "")
        nbv_debug = action.get("_nbv_debug", {})
        info = step_rec.get("info", {})

        if not nav_rel:
            continue
        # Bare "front" (no strategy suffix) = agent made Yes/No decision, not a
        # navigation action. Real LLM hallucinations would be "front (MLLM-Reasoning)".
        if nav_rel.strip() == "front":
            continue
        direction = extract_direction(nav_rel)
        strategy = detect_strategy(nav_rel, nbv_debug) + strategy_suffix

        if direction == "front":
            # LLM hallucination: outputting "front" (stay in place) as a nav decision
            _inc(dist, strategy, f"front_{step_num}", "front")
            _inc(dist, strategy, "front_all", "front")
            _inc(dist, strategy, f"total_{step_num}", "total")
            _inc(dist, strategy, "total_all", "total")
            continue
        if direction in ("unknown", ""):
            _inc(dist, strategy, f"unknown_{step_num}", "unknown")
            _inc(dist, strategy, "unknown_all", "unknown")
            _inc(dist, strategy, f"total_{step_num}", "total")
            _inc(dist, strategy, "total_all", "total")
            continue

        _inc(dist, strategy, step_num, direction)
        _inc(dist, strategy, "all", direction)
        _inc(dist, strategy, f"total_{step_num}", "total")
        _inc(dist, strategy, "total_all", "total")

        if info.get("navigation_failed"):
            _inc(dist, strategy, f"navfail_{step_num}", "navfail")
            _inc(dist, strategy, "navfail_all", "navfail")


def _suffix_for_dir(dirname: str) -> str:
    """Return strategy suffix based on directory name. 'adaptive' dirs get ' (adap)'."""
    name = os.path.basename(dirname).lower()
    if "adaptive" in name:
        return " (adap)"
    return ""


def stream_dir_into_dist(ep_dir: str, dist: dict, min_episodes: int = 500,
                          strategy_suffix: str = "") -> int:
    """
    Stream episode.json files from ep_dir one at a time into dist.
    Returns number of multi-view episodes accumulated, or 0 if below min_episodes.
    Single-view episodes are skipped.
    Does a fast pre-count pass first to decide whether to process the dir at all.
    strategy_suffix: appended to detected strategy name, e.g. " (adap)"
    """
    ep_jsons = sorted(glob.glob(os.path.join(ep_dir, "**", "episode.json"), recursive=True))

    # Fast pre-count: how many are multi-view?
    mv_count = 0
    sv_count = 0
    for path in ep_jsons:
        try:
            with open(path, "r", encoding="utf-8") as f:
                ep = json.load(f)
            if "transcript" not in ep:
                continue
            if is_multiview_episode(ep):
                mv_count += 1
            else:
                sv_count += 1
        except Exception:
            pass

    if mv_count < min_episodes:
        reason = f"only {mv_count} multi-view episodes (< {min_episodes})"
        if sv_count:
            reason += f", {sv_count} single-view skipped"
        print(f"  -> skipped ({reason})")
        return 0

    # Full pass: accumulate into dist
    accumulated = 0
    for path in ep_jsons:
        try:
            with open(path, "r", encoding="utf-8") as f:
                ep = json.load(f)
            if "transcript" not in ep or not is_multiview_episode(ep):
                continue
            accumulate_episode(ep, dist, strategy_suffix=strategy_suffix)
            accumulated += 1
        except Exception as e:
            print(f"\n  [WARN] Failed to process {path}: {e}")

    suffix_note = f" [suffix='{strategy_suffix}']" if strategy_suffix else ""
    if sv_count:
        print(f"  -> {accumulated} multi-view episodes ({sv_count} single-view skipped){suffix_note}")
    else:
        print(f"  -> {accumulated} multi-view episodes{suffix_note}")
    return accumulated


def stream_base_dir_into_dist(base_dir: str, dist: dict, min_episodes: int = 500):
    """
    Auto-scan all immediate subdirectories under base_dir.
    Each subdir with >= min_episodes multi-view episodes is processed.
    Episodes are streamed one by one to avoid OOM.
    Adaptive dirs (name contains 'adaptive') get suffix ' (adap)' to separate them
    from fixed-budget dirs in the distribution plots.
    """
    subdirs = sorted([
        d for d in glob.glob(os.path.join(base_dir, "*"))
        if os.path.isdir(d)
    ])
    if not subdirs:
        print(f"  No subdirs found under {base_dir}")
        return

    total = 0
    for subdir in subdirs:
        name = os.path.basename(subdir)
        suffix = _suffix_for_dir(subdir)
        print(f"  [{name}]", end=" ", flush=True)
        n = stream_dir_into_dist(subdir, dist, min_episodes=min_episodes,
                                  strategy_suffix=suffix)
        total += n
    print(f"\n  Total multi-view episodes accumulated: {total}")


def process_episodes(episodes: list) -> dict:
    """Process a pre-loaded list of episodes (used for --results / --ep_dir modes)."""
    dist = {}
    for ep in episodes:
        accumulate_episode(ep, dist)
    return dist


def print_distribution(dist: dict):
    """Print distribution tables to stdout."""
    strategies = sorted(dist.keys())

    # Dynamically find all numeric step numbers present across all strategies
    all_step_nums = set()
    for strat_data in dist.values():
        for key in strat_data:
            if isinstance(key, int):
                all_step_nums.add(key)
    step_labels = sorted(all_step_nums) + ["all"]

    for strat in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strat}")
        print(f"{'='*60}")
        for step in step_labels:
            counts = dist[strat].get(step, {})
            total = sum(v for k, v in counts.items() if k != "navfail")
            # Hallucination counts (checked independently of valid direction counts)
            total_key = f"total_{step}" if step != "all" else "total_all"
            grand_total = dist[strat].get(total_key, {}).get("total", 0)
            front_key = f"front_{step}" if step != "all" else "front_all"
            unk_key   = f"unknown_{step}" if step != "all" else "unknown_all"
            n_front = dist[strat].get(front_key, {}).get("front", 0)
            n_unk   = dist[strat].get(unk_key,   {}).get("unknown", 0)
            # Skip step only if there is truly nothing to show
            if total == 0 and not n_front and not n_unk:
                continue
            label = f"Step {step}" if step != "all" else "Overall"
            display_n = grand_total if grand_total else total
            print(f"\n  {label} (n={display_n}):")
            if total > 0:
                for d in ALL_DIRS:
                    c = counts.get(d, 0)
                    pct = 100 * c / total
                    bar = "#" * int(pct / 2)
                    print(f"    {DIR_SHORT[d]:2s} ({d:12s}): {c:4d}  {pct:5.1f}%  {bar}")
                # NavFail
                nf_key = f"navfail_{step}" if step != "all" else "navfail_all"
                nf = dist[strat].get(nf_key, {}).get("navfail", 0)
                if nf:
                    nf_pct = 100 * nf / total
                    print(f"    NavFail:              {nf:4d}  {nf_pct:5.1f}%")
            # Always print hallucinations if present (even when total==0)
            if n_front or n_unk:
                f_pct = 100 * n_front / grand_total if grand_total else 0
                u_pct = 100 * n_unk   / grand_total if grand_total else 0
                print(f"    [Halluc] front:       {n_front:4d}  {f_pct:5.1f}%  "
                      f"unknown: {n_unk:4d}  {u_pct:5.1f}%  "
                      f"(of {grand_total} total nav outputs)")


def plot_distribution(dist: dict, output_path: str = None):
    """
    Plot direction distributions as grouped bar charts.
    3 columns: Step 1 / Step 2 / Overall
    One group of bars per direction.
    """
    strategies = sorted(k for k in dist.keys() if not k.startswith("Unknown"))
    step_labels = [1, 2, "all"]
    step_titles = ["Step 1 (1st Navigation)", "Step 2 (2nd Navigation)", "Overall"]

    # Color palette per strategy
    COLORS = {
        "Random":  "#4878CF",
        "FPS":     "#6ACC65",
        "LLM":     "#D65F5F",
        "ViewHint":"#B47CC7",
        "Oracle":  "#C4AD66",
    }
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    n_dirs = len(ALL_DIRS)
    n_strats = len(strategies)
    bar_width = 0.8 / max(n_strats, 1)
    x = np.arange(n_dirs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("NBV Direction Output Distribution by Strategy", fontsize=14, fontweight="bold")

    for col, (step, title) in enumerate(zip(step_labels, step_titles)):
        ax = axes[col]

        for si, strat in enumerate(strategies):
            counts = dist[strat].get(step, {})
            total = sum(v for k, v in counts.items() if k != "navfail")
            if total == 0:
                continue
            proportions = [counts.get(d, 0) / total for d in ALL_DIRS]
            offset = (si - n_strats / 2 + 0.5) * bar_width
            base_strat = strat.split(" (")[0]  # strip " (adap)" etc. for color lookup
            color = COLORS.get(base_strat, default_colors[si % len(default_colors)])
            bars = ax.bar(x + offset, proportions, width=bar_width,
                          label=strat, color=color, alpha=0.85, edgecolor="white")

            # Add value labels on tall bars
            for bar, prop in zip(bars, proportions):
                if prop > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{prop:.0%}", ha="center", va="bottom", fontsize=7)

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([DIR_SHORT[d] for d in ALL_DIRS], fontsize=10)
        ax.set_xlabel("Direction (from agent's viewpoint)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, 0.6)
        ax.axhline(1 / n_dirs, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Uniform ({1/n_dirs:.0%})")
        if col == 0:
            ax.set_ylabel("Proportion of nav decisions", fontsize=9)
        if col == 2:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved plot to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze NBV direction distributions")
    parser.add_argument("--base_dir", default=None,
                        help="Auto-scan all subdirs (e.g. outputs/500_samples). "
                             "Single-view episodes skipped automatically.")
    parser.add_argument("--results", nargs="+", help="Path(s) to results JSON file(s)")
    parser.add_argument("--ep_dir", nargs="+", help="Path(s) to episode output directory")
    parser.add_argument("--output", default=None, help="Output PNG path for the plot")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting, print table only")
    parser.add_argument("--min_episodes", type=int, default=100,
                        help="Min multi-view episodes required to include a subdir "
                             "(default: 100; lower for adaptive-stopping agents that "
                             "terminate early for many episodes)")
    args = parser.parse_args()

    # dist is built incrementally (streaming) to avoid OOM
    dist = {}
    any_loaded = False

    if args.base_dir:
        print(f"Auto-scanning: {args.base_dir}")
        stream_base_dir_into_dist(args.base_dir, dist, min_episodes=args.min_episodes)
        any_loaded = bool(dist)

    if args.results:
        for p in args.results:
            print(f"Loading: {p}")
            eps = load_transcripts_from_file(p)
            mv_eps = [e for e in eps if is_multiview_episode(e)]
            skipped = len(eps) - len(mv_eps)
            print(f"  -> {len(mv_eps)} multi-view episodes ({skipped} single-view skipped)")
            for ep in mv_eps:
                accumulate_episode(ep, dist)
            any_loaded = any_loaded or bool(mv_eps)

    if args.ep_dir:
        for d in args.ep_dir:
            suffix = _suffix_for_dir(d)
            print(f"  [{os.path.basename(d)}]", end=" ", flush=True)
            n = stream_dir_into_dist(d, dist, min_episodes=0,  # no min threshold for explicit dirs
                                      strategy_suffix=suffix)
            any_loaded = any_loaded or n > 0

    if not any_loaded:
        print("No episodes loaded. Use --base_dir, --results, or --ep_dir.")
        return

    print_distribution(dist)

    if not args.no_plot:
        output_path = args.output or "report_figs/nbv_direction_dist.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plot_distribution(dist, output_path)


if __name__ == "__main__":
    main()
