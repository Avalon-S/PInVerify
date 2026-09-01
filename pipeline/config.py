#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central pipeline configuration: every path, threshold and constant lives here.
Other modules pull it in with `from pipeline.config import *` or `import pipeline.config as cfg`.
"""

import os

# ======================== Paths ========================
DATA_ROOT       = os.environ.get("PIN_DATA_ROOT", "./data/datasets/pin/hm3d/v1/val")
CONTENT_DIR     = os.path.join(DATA_ROOT, "content")           # active dataset, rewritten each round
CONTENT_VERIFY  = os.path.join(DATA_ROOT, "content_verify")    # immutable reference copy
CONTENT_BACKUP  = os.path.join(DATA_ROOT, "content_backups")   # versioned backups

CAPTURE_ROOT    = os.environ.get("PIN_CAPTURE_ROOT", "./captures")   # parent directory for capture output
RESULTS_DIR     = os.environ.get("PIN_RESULTS_DIR", "./results/pin/val")
LOG_DIR         = "logs_pipeline"

# ======================== Capture naming ========================
CAPTURE_PREFIX  = "pin_capture_FT"
CAPTURE_SUFFIX  = "_RGB_v2"
# e.g. pin_capture_FT0_RGB_v2, pin_capture_FT1_RGB_v2, ...

# ======================== Navigation evaluation ========================
NAV_EVAL_SCRIPT = "distributeed_pin_eval.py"
NAV_EVAL_NUM_JOBS = 12
NAV_EVAL_EXP_NAME = "pipeline_nav_eval"         # subdirectory holding navigation results

# ======================== Capture parameters ========================
NUM_JOBS        = 12
CAPTURE_SCRIPT  = "val_verify_capture_v2_distributed.py"
CAPTURE_CONFIG  = "configs/models/pin/pin_hm3d_v1.yaml"
SEED_BASE       = 43

# ======================== Capture acceptance thresholds ========================
MIN_NAVIGABLE_VIEWPOINTS  = 6    # an episode needs at least 6 usable viewpoints (navigable and in frustum)
MIN_VALID_MASK_VIEWPOINTS = 3    # at least 3 viewpoints whose mask area clears the category threshold

# Per-category mask-area thresholds in pixels, for 360x640 frames at 42 deg HFOV
CATEGORY_MASK_THRESHOLDS = {
    # Tier S: tiny objects
    "keys":        100,
    "watch":       100,
    # Tier A: small objects
    "eyeglasses":  150,
    "wallet":      150,
    "cellphone":   150,
    "visor":       150,
    "camera":      150,
    "mug":         150,
    # Tier B: medium objects
    "toy":         300,
    "ball":        300,
    "headphones":  300,
    "hat":         300,
    "book":        300,
    "shoes":       300,
    # Tier C: large objects
    "backpack":    500,
    "bag":         500,
    "laptop":      500,
    "teddy bear":  500,
}
DEFAULT_MASK_THRESHOLD = 200

# ======================== Analysis ========================
ROUND_DECIMALS  = 4        # rounding used when aggregating goal coordinates

# ======================== Repair ========================
REQUIRE_PERFECT = True     # only reuse positions with success_ratio == 1.0

# ======================== Navigation filter (one-off) ========================
HEIGHT_FILTER_MIN = 0.0
HEIGHT_FILTER_MAX = 1.6
MIN_START_DIST    = 2.0

# ======================== Pipeline control ========================
MAX_ITERATIONS  = 10


# ======================== Helpers ========================
def capture_dir_name(ft_index: int) -> str:
    """Directory name for capture round ft_index, e.g. pin_capture_FT3_RGB_v2."""
    return f"{CAPTURE_PREFIX}{ft_index}{CAPTURE_SUFFIX}"


def capture_dir_path(ft_index: int) -> str:
    """Full path to capture round ft_index."""
    return os.path.join(CAPTURE_ROOT, capture_dir_name(ft_index))


def backup_dir_path(version: int) -> str:
    """Full path to backup version, e.g. content_backups/content_v3."""
    return os.path.join(CONTENT_BACKUP, f"content_v{version}")


def nav_eval_results_dir() -> str:
    """Directory holding navigation evaluation results."""
    return os.path.join(RESULTS_DIR, NAV_EVAL_EXP_NAME)


def nav_eval_merged_file() -> str:
    """Path to the merged navigation results file."""
    return os.path.join(nav_eval_results_dir(), "all_results.jsonl")


def detect_next_ft_index() -> int:
    """Scan CAPTURE_ROOT for the highest existing FT{N} round and return N+1."""
    max_n = -1
    if os.path.isdir(CAPTURE_ROOT):
        for name in os.listdir(CAPTURE_ROOT):
            if name.startswith(CAPTURE_PREFIX) and name.endswith(CAPTURE_SUFFIX):
                middle = name[len(CAPTURE_PREFIX):-len(CAPTURE_SUFFIX)]
                try:
                    n = int(middle)
                    if n > max_n:
                        max_n = n
                except ValueError:
                    pass
    return max_n + 1
