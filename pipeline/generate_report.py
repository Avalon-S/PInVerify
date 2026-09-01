#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: write the pipeline's final report.

Usage:
    python -m pipeline.generate_report \
        --log_dir logs_pipeline \
        --final_capture_dir <capture root>/pin_capture_FT5_RGB_v2 \
        --final_ft_index 5 \
        --total_rounds 3 \
        --content_dir <data root>/content
"""

import argparse
import glob
import json
import os


def generate_report(
    log_dir: str,
    final_capture_dir: str,
    final_ft_index: int,
    total_rounds: int,
    content_dir: str,
):
    report = {
        "pipeline_summary": {
            "total_repair_rounds": total_rounds,
            "final_ft_index": final_ft_index,
            "final_capture_dir": final_capture_dir,
            "content_dir": content_dir,
        },
        "round_history": [],
        "final_result": None,
    }

    # ---- collect each round's summary ----
    for f in sorted(glob.glob(os.path.join(log_dir, "round_*_summary.json"))):
        try:
            with open(f) as fp:
                s = json.load(fp)
            report["round_history"].append({
                "file": os.path.basename(f),
                "total": s.get("total_episodes"),
                "success": s.get("success_count"),
                "fail": s.get("fail_count"),
                "ratio": s.get("success_ratio"),
                "failure_reasons": s.get("failure_reasons", {}),
            })
        except Exception:
            pass

    # ---- final summary ----
    final_candidates = sorted(
        glob.glob(os.path.join(log_dir, "final_*_summary.json"))
    )
    if final_candidates:
        try:
            with open(final_candidates[-1]) as fp:
                fs = json.load(fp)
            report["final_result"] = {
                "total": fs.get("total_episodes"),
                "success": fs.get("success_count"),
                "fail": fs.get("fail_count"),
                "ratio": fs.get("success_ratio"),
                "per_category": fs.get("per_category_summary", {}),
                "failure_reasons": fs.get("failure_reasons", {}),
            }
        except Exception:
            pass

    # ---- write the JSON report ----
    report_path = os.path.join(log_dir, "pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    # ---- print a readable summary ----
    print()
    print("=" * 60)
    print("  PIN Dataset Pipeline - Final Report")
    print("=" * 60)
    print(f"  Repair rounds:  {total_rounds}")
    print(f"  Final dataset:  {final_capture_dir}")
    print()

    if report["round_history"]:
        print("  Round History:")
        for rh in report["round_history"]:
            r = rh.get("ratio") or 0
            print(f"    {rh['file']}: "
                  f"{rh['success']}/{rh['total']} = {r*100:.1f}%")
        print()

    fr = report.get("final_result")
    if fr:
        r = fr.get("ratio") or 0
        print(f"  FINAL: {fr['success']}/{fr['total']} = {r*100:.1f}% success")

        if fr.get("fail", 0) > 0:
            print(f"  Remaining failures: {fr['fail']}")
            if fr.get("failure_reasons"):
                for reason, cnt in fr["failure_reasons"].items():
                    print(f"    - {reason}: {cnt}")
        print()

        if fr.get("per_category"):
            print("  Per-category:")
            cats = sorted(
                fr["per_category"].items(),
                key=lambda x: x[1].get("success", 0) / max(x[1].get("total", 1), 1),
            )
            for cat, info in cats:
                t = info.get("total", 0)
                s = info.get("success", 0)
                pct = s / t * 100 if t > 0 else 0
                print(f"    {cat:20s}: {s}/{t} = {pct:.1f}%")

    print("=" * 60)
    print(f"\n  Report saved: {report_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate pipeline final report")
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--final_capture_dir", required=True)
    ap.add_argument("--final_ft_index", type=int, required=True)
    ap.add_argument("--total_rounds", type=int, required=True)
    ap.add_argument("--content_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    generate_report(
        log_dir=args.log_dir,
        final_capture_dir=args.final_capture_dir,
        final_ft_index=args.final_ft_index,
        total_rounds=args.total_rounds,
        content_dir=args.content_dir,
    )


if __name__ == "__main__":
    main()
