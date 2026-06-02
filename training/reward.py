#!/usr/bin/env python3
"""
GRPO/GSPO reward functions for ms-swift.

Must follow ms-swift ORM plugin pattern:
  - Inherit from swift.plugin.orm.ORM
  - Implement __call__(self, completions, solution, **kwargs) -> List[float]
  - Register in module-level `orms` dict

v3 changes:
- Soft verification reward with partial credit (was binary 0/1):
    Unsure↔Yes: 0.3-0.4, Unsure↔No: 0.2-0.3, Yes↔No: 0.0
- Premature stop penalty: 0.3 → 0.1 (symmetric with over-exploration)
- MOVE without direction: 0.5 → 0.3
- Weights unchanged: 0.5 verification + 0.4 action + 0.1 format

Usage:
    swift rlhf --external_plugins training/reward.py --reward_funcs pv_reward pv_format
"""

import json
import re
from typing import List

from swift.plugin.orm import ORM
from swift.plugin.orm import orms as _swift_orms


# ---- Parsing helpers ----

def parse_answer_block(text):
    """Extract structured fields from <answer>...</answer> block."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not match:
        return {}
    block = match.group(1).strip()

    result = {}
    for line in block.split("\n"):
        line = line.strip()
        if ":" in line:
            key, val = line.split(":", 1)
            result[key.strip().lower()] = val.strip()
    return result


def parse_verification(text):
    """Extract verification result (Yes/No/Unsure)."""
    answer = parse_answer_block(text)
    verify = answer.get("verification", "").strip()
    if verify.lower().startswith("yes"):
        return "Yes"
    elif verify.lower().startswith("no"):
        return "No"
    elif verify.lower().startswith("unsure"):
        return "Unsure"
    return "Unknown"


def parse_action(text):
    """Extract action (STOP or MOVE <direction>)."""
    answer = parse_answer_block(text)
    action_str = answer.get("action", "").strip()
    if action_str.upper().startswith("STOP"):
        return "STOP"
    elif action_str.upper().startswith("MOVE"):
        return action_str
    return "Unknown"


def compute_verify_reward(pred, gt_label):
    """Soft verification reward with partial credit.

    Principles:
    - Exact match = 1.0, opposite (Yes↔No) = 0.0
    - Unsure is "safe" — never catastrophically wrong
    - Cautious (Unsure) when should be Yes (0.4) > when should be No (0.2)
      because exploring a correct target is useful, exploring a wrong one is wasted
    - Overconfident (Yes/No) when should be Unsure: 0.3 (premature decision risk)
    """
    if pred == gt_label:
        return 1.0
    _PARTIAL = {
        ("Unsure", "Yes"):  0.4,  # Cautious on positive — safe, will confirm with more views
        ("Unsure", "No"):   0.2,  # Cautious on negative — wasted exploration on wrong object
        ("Yes",    "Unsure"): 0.3,  # Overconfident match — premature stop risk
        ("No",     "Unsure"): 0.3,  # Overconfident reject — premature stop risk
        # Yes↔No: 0.0 (completely wrong direction)
    }
    return _PARTIAL.get((pred, gt_label), 0.0)


def compute_action_reward(action_pred, gt):
    """Compute action quality reward with 4-tier navigation incentives.

    STOP rewards:
      - Correct stop:       1.0
      - Premature stop:     0.1 (dangerous — insufficient evidence)
      - Over-exploration:   0.1 (wasteful — already had enough evidence)

    MOVE rewards (when GT says MOVE):
      - MOVE to best sector:      1.0 (visible + navigable + FPS-informative)
      - MOVE to other vis+nav:    0.7 (safe but less informative)
      - MOVE to navigable-only:   0.3 (can go but won't see target = trap view)
      - MOVE to unreachable:      0.0 (critical failure — not navigable)
    """
    gt_action = gt.get("action", "STOP").upper()
    gt_visible = set(s.lower() for s in gt.get("visible", []))
    gt_navigable = set(s.lower() for s in gt.get("navigable", []))
    gt_best = set(s.lower() for s in gt.get("best_sectors", []))

    if action_pred.upper().startswith("STOP"):
        if gt_action.startswith("STOP"):
            return 1.0   # Correct to stop
        else:
            return 0.1   # Premature stop — high risk of wrong final answer
    elif action_pred.upper().startswith("MOVE"):
        parts = action_pred.split(None, 1)
        move_target = parts[1].strip().lower() if len(parts) > 1 else ""

        if gt_action.startswith("STOP"):
            return 0.1   # Over-exploration — wastes steps but answer likely correct
        else:
            # Correct to move — check WHERE (4-tier)
            if move_target and move_target not in gt_navigable:
                return 0.0   # Unreachable (critical failure)
            elif move_target and gt_best and move_target in gt_best:
                return 1.0   # Best: visible + navigable + FPS-informative
            elif move_target and move_target in gt_visible:
                return 0.7   # Visible + navigable but not optimally informative
            elif move_target and move_target in gt_navigable:
                return 0.3   # Navigable-only (trap view risk)
            elif move_target:
                return 0.0   # Unknown direction name
            return 0.3   # MOVE without valid direction (right idea, bad format)
    return 0.0


# ---- ORM classes ----

class PVReward(ORM):
    """Combined reward: verification (0.5) + action (0.4) + format (0.1).

    v3: soft verification with partial credit, stricter premature stop penalty.
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, sol in zip(completions, solution):
            gt = json.loads(sol) if isinstance(sol, str) else sol
            text = str(completion)

            verify_pred = parse_verification(text)
            action_pred = parse_action(text)

            # 1. Verification (weight 0.5) — soft partial credit
            r_verify = compute_verify_reward(verify_pred, gt.get("label", "No"))

            # 2. Action quality (weight 0.4) — 4-tier with strict premature stop
            r_action = compute_action_reward(action_pred, gt)

            # 3. Format correctness (weight 0.1)
            has_think = "<think>" in text and "</think>" in text
            has_answer = "<answer>" in text and "</answer>" in text
            r_format = 1.0 if (has_think and has_answer) else (0.5 if has_answer else 0.0)

            reward = 0.5 * r_verify + 0.4 * r_action + 0.1 * r_format
            rewards.append(reward)

        return rewards


class PVFormat(ORM):
    """Format-only reward: checks <think>...</think><answer>...</answer> structure."""

    def __call__(self, completions, **kwargs) -> List[float]:
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        return [1.0 if re.search(pattern, c, re.DOTALL) else 0.0 for c in completions]


# Register directly into swift's global orms dict
_swift_orms["pv_reward"] = PVReward
_swift_orms["pv_format"] = PVFormat
