#!/usr/bin/env python3
"""
Generate SFT training data v3 with dual-image input + CoT reasoning.

Each training sample = one conversation turn:
- User: dual image (scene RGB + GT bbox crop) + unified prompt
- Assistant: <think>CoT</think><answer>structured output</answer>

v3 changes (single controlled variable vs v2):
- neg_same CoT uses concrete attribute comparison with target object's actual
  descriptions instead of generic "observed something different".
  e.g., 'target should be "red", but this object appears "blue"'
- Everything else (ratios, multi-step logic, positive CoT) is identical to v2.

Usage:
    python training/prepare_sft_data_v3.py \
        --data-root ./data/pv_dataset \
        --train-dir ./data/pv_dataset/train_sft \
        --index-file ./data/pv_dataset/train_sft/pv_train_sft_index.jsonl \
        --desc-db ./data/pv_dataset/train/object_descriptions_with_category.json \
        --attr-cache ./data/train/attr_cache_train.json \
        --category-cache ./data/pv_dataset/train/category_cache.json \
        --output ./data/pv_dataset/train_sft/sft_data_v3.jsonl \
        --crop-dir ./data/train/crops_v3
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Direction names matching env.py convention (6 sectors)
SECTOR_NAMES = ["front", "front-left", "back-left", "back", "back-right", "front-right"]

SYSTEM_PROMPT = (
    "You are an embodied agent navigating around an object to verify whether it matches "
    "a target description. Each step you see two images: a full scene view and a close-up "
    "of the detected object. You must verify object attributes and decide your next action."
)

USER_TEMPLATE = """<image><image>
You are an embodied agent verifying whether a detected object matches a target description.

Target: "{query_description}"
Category (must match): {query_category}
Current sector: {sector_name} ({angle}°)
Visited sectors: [{visited_list}]
Remaining budget: {remaining} steps
Available sectors: [{available_sectors}]

From the scene image (Image 1) and the object close-up (Image 2):
1. Does this object match the target description? Check each attribute.
2. Your action: STOP (if confident) or MOVE <sector> (if need more views)"""


def crop_with_gt_bbox(img_path, bbox_xyxy, pad=3, min_size=512):
    """Crop image using GT bbox, matching mllm_policy._crop_image."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    x1 = max(0, min(w, bbox_xyxy[0] - pad))
    y1 = max(0, min(h, bbox_xyxy[1] - pad))
    x2 = max(0, min(w, bbox_xyxy[2] + pad))
    y2 = max(0, min(h, bbox_xyxy[3] + pad))
    if x2 <= x1 or y2 <= y1:
        if min(w, h) < min_size:
            s = min_size / min(w, h)
            img = img.resize((int(w * s + 0.5), int(h * s + 0.5)), Image.BICUBIC)
        return img
    crop = img.crop((x1, y1, x2, y2))
    cw, ch = crop.size
    if min(cw, ch) < min_size:
        s = min_size / min(cw, ch)
        crop = crop.resize((int(cw * s + 0.5), int(ch * s + 0.5)), Image.BICUBIC)
    return crop


def load_desc_db(path):
    with open(path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    for obj_id, entry in db.items():
        if isinstance(entry, list):
            db[obj_id] = {"descriptions": entry, "object_category": "object"}
        elif isinstance(entry, dict) and "descriptions" not in entry:
            descs = entry.get("description", [])
            if isinstance(descs, str):
                descs = [descs]
            entry["descriptions"] = descs
    return db


def get_viewpoints_by_sector(meta):
    """Parse viewpoints from meta.json, group by sector index."""
    viewpoints = meta.get("viewpoints") or meta.get("captures", [])
    by_sector = {}
    for vp in viewpoints:
        tag = vp.get("tag", "")
        sector_idx = vp.get("sector_index", -1)
        if sector_idx < 0 and tag.startswith("s"):
            try:
                sector_idx = int(tag.split("_")[0][1:])
            except (ValueError, IndexError):
                continue
        if sector_idx < 0:
            continue
        by_sector.setdefault(sector_idx, []).append(vp)
    return by_sector


def get_navigable_sectors(viewpoints_by_sector):
    """Get list of navigable sector indices."""
    navigable = []
    for sec_idx, vps in viewpoints_by_sector.items():
        for vp in vps:
            if vp.get("navigable", False) and (vp.get("rgb") or vp.get("rgb_filename")):
                navigable.append(sec_idx)
                break
    return sorted(navigable)


def get_visible_sectors(viewpoints_by_sector):
    """Get sectors where mask_meets_threshold is True."""
    visible = []
    for sec_idx, vps in viewpoints_by_sector.items():
        for vp in vps:
            if vp.get("mask_meets_threshold", False):
                visible.append(sec_idx)
                break
    return sorted(visible)


def pick_viewpoint(vps, view_priority="far"):
    """Pick best viewpoint from a sector's viewpoints list."""
    for vp in vps:
        vtype = vp.get("view_type", "")
        if not vtype:
            tag = vp.get("tag", "")
            vtype = "far" if "far" in tag else "near"
        if vtype == view_priority:
            return vp
    return vps[0] if vps else None


def sector_to_angle(sector_idx, n_sectors=12):
    """Convert sector index to approximate angle in degrees."""
    return (sector_idx * 360 / n_sectors) % 360


def sector_to_direction(sector_idx, n_sectors=12):
    """Map a 12-sector index to a 6-direction name (absolute)."""
    step = max(1, n_sectors // 6)
    dir_idx = (sector_idx // step) % 6
    return SECTOR_NAMES[dir_idx]


def abs_to_relative_dir(current_sector, target_sector, n_sectors=12):
    """Get relative direction name from current to target sector.

    Agent always perceives its current position as 'front'.
    Other sectors are named relative to the agent's facing direction.
    """
    step = max(1, n_sectors // 6)
    cur_dir = (current_sector // step) % 6
    tgt_dir = (target_sector // step) % 6
    offset = (tgt_dir - cur_dir) % 6
    return SECTOR_NAMES[offset]


# ---- CoT builders ----

def build_cot_positive(attrs, available_dirs):
    """Build CoT for positive pair — all attributes confirmed, STOP."""
    lines = []
    if available_dirs:
        lines.append(f"[Navigation] Available unvisited sectors: {', '.join(available_dirs)}. "
                     f"No further exploration needed.")
    else:
        lines.append("[Navigation] No unvisited sectors remaining.")
    lines.append("")

    lines.append("[Verification]")
    for a in attrs:
        name = a.get("name", "attribute")
        evidence = a.get("evidence_phrase", "")
        lines.append(f"- {name}: Yes — \"{evidence}\" is consistent with what is observed.")
    lines.append("")
    lines.append("All attributes match the target description.")
    lines.append("")
    lines.append("[Decision] All attributes verified as matching. Confident this is the target. "
                 "Stopping with Yes.")
    return "\n".join(lines)


def build_cot_positive_unsure(attrs, available_dirs, step_idx, next_sector_name):
    """Build CoT for positive pair where agent decides to explore more."""
    lines = []
    if available_dirs:
        lines.append(f"[Navigation] Available unvisited sectors: {', '.join(available_dirs)}.")
    else:
        lines.append("[Navigation] No unvisited sectors remaining.")
    lines.append("")

    lines.append("[Verification]")
    n_checked = max(1, min(len(attrs), step_idx + len(attrs) // 2))
    for i, a in enumerate(attrs):
        name = a.get("name", "attribute")
        evidence = a.get("evidence_phrase", "")
        if i < n_checked:
            lines.append(f"- {name}: Yes — \"{evidence}\" seems consistent.")
        else:
            lines.append(f"- {name}: Unsure — cannot confirm \"{evidence}\" from this angle.")
    lines.append("")
    lines.append("Some attributes are not fully confirmed from this view.")
    lines.append("")
    lines.append(f"[Decision] Not all attributes verified. Moving to {next_sector_name} "
                 f"for a better view.")
    return "\n".join(lines)


def build_cot_trap_view(available_dirs, next_sector_name):
    """Build CoT for trap view step — target not visible, must MOVE."""
    lines = []
    if available_dirs:
        lines.append(f"[Navigation] Available unvisited sectors: {', '.join(available_dirs)}.")
    else:
        lines.append("[Navigation] No unvisited sectors remaining.")
    lines.append("")
    lines.append("[Verification]")
    lines.append("- Cannot verify any attributes — target object is not clearly visible "
                 "from this angle.")
    lines.append("")
    lines.append(f"[Decision] Object not visible. Moving to {next_sector_name} "
                 f"for a better view.")
    return "\n".join(lines)


def build_cot_trap_view_stop(available_dirs):
    """Build CoT for trap view when no more sectors to explore — forced STOP."""
    lines = []
    lines.append("[Navigation] No unvisited sectors with potential views remaining.")
    lines.append("")
    lines.append("[Verification]")
    lines.append("- Cannot verify attributes from this angle. No better viewpoints available.")
    lines.append("")
    lines.append("[Decision] No more exploration options. Based on previous observations, "
                 "stopping with best estimate.")
    return "\n".join(lines)


def build_cot_neg_same(attrs, available_dirs, mismatch_attr, target_attrs_map=None):
    """Build CoT for neg_same pair — found mismatching attribute, STOP.

    v3: Uses target object's actual attributes for concrete comparison
    when target_attrs_map is provided.
    """
    lines = []
    if available_dirs:
        lines.append(f"[Navigation] Available unvisited sectors: {', '.join(available_dirs)}.")
    else:
        lines.append("[Navigation] No unvisited sectors remaining.")
    lines.append("")

    lines.append("[Verification]")
    for a in attrs:
        name = a.get("name", "attribute")
        evidence = a.get("evidence_phrase", "")
        if name == mismatch_attr:
            # v3: concrete comparison using target object's actual attribute
            target_evidence = (target_attrs_map or {}).get(name, "")
            # Fallback: if no exact name match, use any target attribute
            if not target_evidence and target_attrs_map:
                target_evidence = next(iter(target_attrs_map.values()), "")
            if target_evidence:
                lines.append(
                    f"- {name}: No — target should be \"{evidence}\", "
                    f"but this object appears \"{target_evidence}\".")
            else:
                lines.append(
                    f"- {name}: No — expected \"{evidence}\" "
                    f"but observed something different.")
        else:
            lines.append(f"- {name}: Unsure — partially visible, cannot confirm \"{evidence}\".")
    lines.append("")
    lines.append(f"Attribute \"{mismatch_attr}\" does not match. This is not the target.")
    lines.append("")
    lines.append("[Decision] Found mismatching attribute. Confident this is NOT the target. "
                 "Stopping with No.")
    return "\n".join(lines)


def build_cot_neg_same_unsure(attrs, available_dirs, next_sector_name):
    """Build CoT for neg_same where agent is uncertain and explores more."""
    lines = []
    if available_dirs:
        lines.append(f"[Navigation] Available unvisited sectors: {', '.join(available_dirs)}.")
    else:
        lines.append("[Navigation] No unvisited sectors remaining.")
    lines.append("")

    lines.append("[Verification]")
    for a in attrs:
        name = a.get("name", "attribute")
        evidence = a.get("evidence_phrase", "")
        lines.append(f"- {name}: Unsure — cannot clearly confirm \"{evidence}\" from this view.")
    lines.append("")
    lines.append("No clear mismatch found yet, but attributes are not fully confirmed.")
    lines.append("")
    lines.append(f"[Decision] Need more views to make a confident determination. "
                 f"Moving to {next_sector_name}.")
    return "\n".join(lines)


def build_cot_neg_diff(query_category, target_category, available_dirs,
                       target_desc_summary=None):
    """Build CoT for neg_diff pair — category mismatch, immediate STOP.

    v3: Includes target object's actual description for concrete evidence.
    """
    lines = []
    if available_dirs:
        lines.append(f"[Navigation] Available unvisited sectors: {', '.join(available_dirs)}.")
    else:
        lines.append("[Navigation] No unvisited sectors remaining.")
    lines.append("")

    lines.append("[Verification]")
    lines.append(f"- Category check: No — target should be a {query_category}, "
                 f"but the object in the scene is a {target_category}.")
    if target_desc_summary:
        lines.append(f"- Appearance: This object appears to be \"{target_desc_summary}\", "
                     f"which does not match the target description at all.")
    lines.append("")
    lines.append(f"Category mismatch: expected {query_category}, got {target_category}.")
    lines.append("")
    lines.append("[Decision] Category does not match. This is definitely NOT the target. "
                 "Stopping with No.")
    return "\n".join(lines)


def build_answer(verification, action):
    """Build structured <answer> block. No navigation field."""
    return f"verification: {verification}\naction: {action}"


def process_episode(ep, meta, desc_db, attr_cache, category_cache,
                    ep_abs_path, crop_dir, view_priority="far"):
    """Generate training samples for one episode pair."""
    samples = []

    target_obj_id = ep["target_object_id"]
    query_obj_id = ep.get("query_object_id", target_obj_id)
    query_cat = ep.get("query_object_category", ep.get("target_object_category", "object"))
    target_cat = ep.get("target_object_category", "object")
    label = ep["label"]
    pair_type = ep.get("pair_type", "positive")

    # Get query description
    if query_obj_id not in desc_db:
        return []
    descs_entry = desc_db[query_obj_id]
    descs = descs_entry.get("descriptions", [])
    if not descs:
        return []
    # Use all descriptions (complementary attributes across descriptions)
    query_desc = "; ".join(d for d in descs if d)

    # Get attributes for CoT
    attrs = []
    if query_obj_id in attr_cache:
        attrs = attr_cache[query_obj_id].get("attributes", [])
    if not attrs:
        attrs = [{"name": f"desc_{i+1}", "evidence_phrase": d}
                 for i, d in enumerate(descs[:3]) if d]

    # Parse viewpoints
    vps_by_sector = get_viewpoints_by_sector(meta)
    navigable_sectors = get_navigable_sectors(vps_by_sector)
    visible_sectors = get_visible_sectors(vps_by_sector)

    if not visible_sectors:
        return []

    n_sectors = meta.get("n_sectors", 12)
    max_steps = 6

    def get_available_dirs(visited_set, current_sector):
        """All unvisited directions in RELATIVE terms from current position."""
        visited_rel = set(abs_to_relative_dir(current_sector, s, n_sectors)
                         for s in visited_set)
        return [d for d in SECTOR_NAMES if d not in visited_rel]

    def make_step_sample(sector_idx, visited_set, remaining, cot_text,
                         verification, action_text, step_history=None, is_trap=False):
        """Create one training sample for a single step."""
        vps = vps_by_sector.get(sector_idx, [])
        vp = pick_viewpoint(vps, view_priority)
        if vp is None:
            return None

        rgb_rel = vp.get("rgb") or vp.get("rgb_filename", "")
        if not rgb_rel:
            return None
        rgb_abs = os.path.join(ep_abs_path, rgb_rel.lstrip("./"))
        if not os.path.exists(rgb_abs):
            return None

        # Crop from GT bbox
        bbox = vp.get("mask_bbox_xyxy")
        vp_is_trap = not vp.get("mask_meets_threshold", False)

        crop_filename = f"{target_obj_id}_{sector_idx}_{pair_type}.jpg"
        crop_path = os.path.join(crop_dir, crop_filename)

        if bbox and not vp_is_trap:
            if not os.path.exists(crop_path):
                try:
                    crop_img = crop_with_gt_bbox(rgb_abs, bbox)
                    crop_img.save(crop_path, "JPEG", quality=95)
                except Exception:
                    return None
        else:
            # Trap view or no bbox: use full image as crop
            if not os.path.exists(crop_path):
                try:
                    img = Image.open(rgb_abs).convert("RGB")
                    w, h = img.size
                    if min(w, h) < 512:
                        s = 512.0 / min(w, h)
                        img = img.resize((int(w * s + 0.5), int(h * s + 0.5)), Image.BICUBIC)
                    img.save(crop_path, "JPEG", quality=95)
                except Exception:
                    return None

        # Agent always perceives its current position as "front" (0°)
        sector_name = "front"
        angle = 0
        # Visited sectors in relative terms from current position
        visited_names = [abs_to_relative_dir(sector_idx, s, n_sectors)
                         for s in sorted(visited_set)]
        visited_list = ", ".join(visited_names) if visited_names else "none"

        available_dirs = get_available_dirs(visited_set, sector_idx)
        available_str = ", ".join(available_dirs) if available_dirs else "none"

        user_text = USER_TEMPLATE.format(
            query_description=query_desc,
            query_category=query_cat,
            sector_name=sector_name,
            angle=angle,
            visited_list=visited_list,
            remaining=remaining,
            available_sectors=available_str,
        )

        # Visibility warning for trap views
        if is_trap or vp_is_trap:
            user_text += "\nWarning: Target object is not clearly visible from this angle."

        # Step history for multi-step context
        # Directions are converted to be relative to CURRENT position (sector_idx)
        if step_history:
            history_lines = []
            for h in step_history:
                rel_pos = abs_to_relative_dir(sector_idx, h['abs_sector'], n_sectors)
                rel_target = abs_to_relative_dir(sector_idx, h['abs_target'], n_sectors)
                history_lines.append(
                    f"- Step {h['step']} ({rel_pos}): "
                    f"Verification={h['verification']}, Action=MOVE {rel_target}"
                )
            user_text += "\n\nPrevious observations:\n" + "\n".join(history_lines)

        assistant_text = (f"<think>\n{cot_text}\n</think>\n\n"
                         f"<answer>\n{build_answer(verification, action_text)}\n</answer>")

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "images": [rgb_abs, crop_path],
        }

    # ---- Generate samples based on pair_type ----

    visible_nav = [s for s in visible_sectors if s in navigable_sectors]
    trap_nav = [s for s in navigable_sectors if s not in visible_sectors]

    if pair_type == "positive":
        # 20% chance to start at trap view for trap recovery training
        if trap_nav and random.random() < 0.2:
            start_sector = random.choice(trap_nav)
        elif visible_nav:
            start_sector = random.choice(visible_nav)
        else:
            return []

        visited = {start_sector}
        current = start_sector
        step_history = []

        for step_idx in range(max_steps):
            remaining = max_steps - step_idx - 1
            is_trap = current not in visible_sectors
            available_dirs = get_available_dirs(visited, current)
            unvisited_vis = [s for s in visible_sectors
                           if s not in visited and s in navigable_sectors]

            if is_trap:
                # Trap view: can't verify, must MOVE if possible
                if unvisited_vis and remaining > 0:
                    next_sector = random.choice(unvisited_vis)
                    next_dir = abs_to_relative_dir(current, next_sector, n_sectors)
                    cot = build_cot_trap_view(available_dirs, next_dir)
                    sample = make_step_sample(
                        current, visited, remaining, cot, "Unsure",
                        f"MOVE {next_dir}", step_history, is_trap=True)
                    if sample:
                        samples.append(sample)
                    step_history.append({
                        "step": step_idx + 1,
                        "abs_sector": current,
                        "abs_target": next_sector,
                        "verification": "Unsure",
                    })
                    visited.add(next_sector)
                    current = next_sector
                    continue
                else:
                    # No more visible sectors — forced STOP
                    cot = build_cot_trap_view_stop(available_dirs)
                    sample = make_step_sample(
                        current, visited, remaining, cot, "Yes", "STOP",
                        step_history, is_trap=True)
                    if sample:
                        samples.append(sample)
                    break

            # Normal visible view
            if remaining == 0 or not unvisited_vis:
                # No budget or no more sectors: STOP with Yes
                cot = build_cot_positive(attrs, available_dirs)
                sample = make_step_sample(
                    current, visited, remaining, cot, "Yes", "STOP", step_history)
                if sample:
                    samples.append(sample)
                break

            # Decide: MOVE or STOP (increasing stop probability per step)
            stop_prob = 0.2 + 0.15 * step_idx  # 0.20, 0.35, 0.50, 0.65, 0.80
            if random.random() < stop_prob:
                cot = build_cot_positive(attrs, available_dirs)
                sample = make_step_sample(
                    current, visited, remaining, cot, "Yes", "STOP", step_history)
                if sample:
                    samples.append(sample)
                break
            else:
                next_sector = random.choice(unvisited_vis)
                next_dir = abs_to_relative_dir(current, next_sector, n_sectors)
                cot = build_cot_positive_unsure(
                    attrs, available_dirs, step_idx, next_dir)
                sample = make_step_sample(
                    current, visited, remaining, cot, "Unsure",
                    f"MOVE {next_dir}", step_history)
                if sample:
                    samples.append(sample)
                step_history.append({
                    "step": step_idx + 1,
                    "abs_sector": current,
                    "abs_target": next_sector,
                    "verification": "Unsure",
                })
                visited.add(next_sector)
                current = next_sector

    elif pair_type == "neg_same":
        if not visible_nav:
            return []
        start_sector = random.choice(visible_nav)
        visited = {start_sector}
        step_history = []
        mismatch_attr = random.choice(attrs)["name"] if attrs else "appearance"

        # v3: Build target object's attribute map for concrete comparison
        target_attrs_map = {}
        if target_obj_id in attr_cache:
            for ta in attr_cache[target_obj_id].get("attributes", []):
                ta_name = ta.get("name", "")
                ta_evidence = ta.get("evidence_phrase", "")
                if ta_name and ta_evidence:
                    target_attrs_map[ta_name] = ta_evidence
        # Fallback: use target object's raw descriptions if no attr_cache match
        if not target_attrs_map and target_obj_id in desc_db:
            target_descs = desc_db[target_obj_id].get("descriptions", [])
            for i, d in enumerate(target_descs[:3]):
                if d:
                    target_attrs_map[f"desc_{i+1}"] = d

        unvisited_vis = [s for s in visible_sectors
                        if s not in visited and s in navigable_sectors]

        # 50% chance of 2-step: Unsure→MOVE then No→STOP
        if unvisited_vis and random.random() < 0.5:
            available_dirs = get_available_dirs(visited, start_sector)
            next_sector = random.choice(unvisited_vis)
            next_dir = abs_to_relative_dir(start_sector, next_sector, n_sectors)

            cot = build_cot_neg_same_unsure(attrs, available_dirs, next_dir)
            sample = make_step_sample(
                start_sector, visited, max_steps - 1, cot, "Unsure",
                f"MOVE {next_dir}", step_history)
            if sample:
                samples.append(sample)

            step_history.append({
                "step": 1,
                "abs_sector": start_sector,
                "abs_target": next_sector,
                "verification": "Unsure",
            })
            visited.add(next_sector)

            # Step 1: found mismatch, STOP
            available_dirs_2 = get_available_dirs(visited, next_sector)
            cot2 = build_cot_neg_same(attrs, available_dirs_2, mismatch_attr,
                                      target_attrs_map)
            sample2 = make_step_sample(
                next_sector, visited, max_steps - 2, cot2, "No", "STOP",
                step_history)
            if sample2:
                samples.append(sample2)
        else:
            # Single step: found mismatch immediately, STOP
            available_dirs = get_available_dirs(visited, start_sector)
            cot = build_cot_neg_same(attrs, available_dirs, mismatch_attr,
                                     target_attrs_map)
            sample = make_step_sample(
                start_sector, visited, max_steps - 1, cot, "No", "STOP",
                step_history)
            if sample:
                samples.append(sample)

    elif pair_type == "neg_diff":
        if visible_nav:
            start_sector = random.choice(visible_nav)
        elif navigable_sectors:
            start_sector = random.choice(navigable_sectors)
        else:
            return []

        visited = {start_sector}
        available_dirs = get_available_dirs(visited, start_sector)

        # v3: Get target object's description for concrete comparison
        target_desc_summary = None
        if target_obj_id in desc_db:
            target_descs = desc_db[target_obj_id].get("descriptions", [])
            if target_descs:
                target_desc_summary = target_descs[0]

        cot = build_cot_neg_diff(query_cat, target_cat, available_dirs,
                                 target_desc_summary)
        sample = make_step_sample(
            start_sector, visited, max_steps - 1, cot, "No", "STOP")
        if sample:
            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (for resolving relative paths)")
    parser.add_argument("--train-dir", required=True,
                        help="Training episode directory")
    parser.add_argument("--index-file", required=True,
                        help="Paired index JSONL (with pair_type/label)")
    parser.add_argument("--desc-db", required=True,
                        help="Path to object_descriptions_with_category.json")
    parser.add_argument("--attr-cache", default=None,
                        help="Path to attr_cache_train.json (optional)")
    parser.add_argument("--category-cache", default=None,
                        help="Path to category_cache.json (optional)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--crop-dir", required=True,
                        help="Directory to save cropped images")
    parser.add_argument("--view-priority", default="far",
                        choices=["far", "near"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load resources
    desc_db = load_desc_db(args.desc_db)
    print(f"Loaded {len(desc_db)} objects from desc_db")

    attr_cache = {}
    if args.attr_cache and os.path.exists(args.attr_cache):
        with open(args.attr_cache, 'r', encoding='utf-8') as f:
            attr_cache = json.load(f)
        print(f"Loaded {len(attr_cache)} objects from attr_cache")

    category_cache = {}
    if args.category_cache and os.path.exists(args.category_cache):
        with open(args.category_cache, 'r', encoding='utf-8') as f:
            category_cache = json.load(f)
        print(f"Loaded {len(category_cache)} categories from cache")

    # Load index
    with open(args.index_file, 'r', encoding='utf-8') as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(pairs)} pairs from index")

    os.makedirs(args.crop_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Process each pair
    all_samples = []
    stats = Counter()

    for ep in tqdm(pairs, desc="Processing pairs"):
        episode_path = ep["episode_path"]

        if args.data_root:
            ep_abs = os.path.join(args.data_root, episode_path)
        else:
            ep_abs = episode_path

        meta_path = os.path.join(ep_abs, "meta.json")
        if not os.path.exists(meta_path):
            stats["skip_no_meta"] += 1
            continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        ep_samples = process_episode(ep, meta, desc_db, attr_cache, category_cache,
                                     ep_abs, args.crop_dir, args.view_priority)

        all_samples.extend(ep_samples)
        stats[f"pairs_{ep.get('pair_type', 'unknown')}"] += 1
        stats[f"samples_{ep.get('pair_type', 'unknown')}"] += len(ep_samples)

    # Shuffle
    random.shuffle(all_samples)

    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n=== SFT Data Statistics ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Output: {args.output}")
    print(f"  Crops: {args.crop_dir}")


if __name__ == "__main__":
    main()
