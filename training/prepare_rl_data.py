#!/usr/bin/env python3
"""
Generate RL (GRPO/GSPO) prompt data — only user prompts, no assistant response.

Each sample contains:
- messages: [system, user] (no assistant — model generates its own response)
- images: [scene_rgb, crop]
- solution: JSON string with GT for reward computation

v2 changes:
- Removed navigation question from prompt (matches SFT v2)
- Solution includes "visible", "navigable", and FPS-ranked "best_sectors"
- best_sectors = unvisited visible dirs furthest from visited (informative NBV)
- Multi-step: generates prompts at different exploration stages
  - Fresh start prompts (should MOVE for positive, STOP for negative)
  - Mid-exploration prompts (should STOP after sufficient views)
- Step history and visibility warning for multi-step context

Usage:
    python training/prepare_rl_data.py \
        --data-root ./data/pv_dataset \
        --train-dir ./data/train_rl \
        --index-file ./data/train_rl/pv_train_rl_index.jsonl \
        --desc-db ./data/pv_dataset/train/object_descriptions_with_category.json \
        --output ./data/train/rl_data.jsonl \
        --crop-dir ./data/train/crops_rl
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

SECTOR_NAMES = ["front", "front-left", "back-left", "back", "back-right", "front-right"]

DIRECTION_ANGLES = {
    "front": 0, "front-left": 60, "back-left": 120,
    "back": 180, "back-right": 240, "front-right": 300,
}


def angular_distance(dir_a, dir_b):
    """Compute angular distance between two directions (0-180°)."""
    a = DIRECTION_ANGLES.get(dir_a, 0)
    b = DIRECTION_ANGLES.get(dir_b, 0)
    diff = abs(a - b)
    return min(diff, 360 - diff)


def compute_best_sectors(visited_dirs, unvisited_visible_dirs):
    """FPS-like ranking: visible directions furthest from visited.

    Returns top-tier directions (within 30° of best min-distance).
    """
    if not visited_dirs or not unvisited_visible_dirs:
        return list(unvisited_visible_dirs)

    scored = []
    for d in unvisited_visible_dirs:
        min_dist = min(angular_distance(d, v) for v in visited_dirs)
        scored.append((min_dist, d))

    scored.sort(reverse=True)
    if not scored:
        return []

    best_dist = scored[0][0]
    return [d for dist, d in scored if dist >= best_dist - 30]

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
    return db


def sector_to_angle(sector_idx, n_sectors=12):
    return (sector_idx * 360 / n_sectors) % 360


def sector_to_direction(sector_idx, n_sectors=12):
    """Map a sector index to a 6-direction name (absolute)."""
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


def get_viewpoints_by_sector(meta):
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
        if sector_idx >= 0:
            by_sector.setdefault(sector_idx, []).append(vp)
    return by_sector


def get_navigable_sectors(vps_by_sector):
    navigable = []
    for sec_idx, vps in vps_by_sector.items():
        for vp in vps:
            if vp.get("navigable", False) and (vp.get("rgb") or vp.get("rgb_filename")):
                navigable.append(sec_idx)
                break
    return sorted(navigable)


def get_visible_sectors(vps_by_sector):
    visible = []
    for sec_idx, vps in vps_by_sector.items():
        for vp in vps:
            if vp.get("mask_meets_threshold", False):
                visible.append(sec_idx)
                break
    return sorted(visible)


def pick_viewpoint(vps, view_priority="far"):
    for vp in vps:
        vtype = vp.get("view_type", "")
        if not vtype:
            tag = vp.get("tag", "")
            vtype = "far" if "far" in tag else "near"
        if vtype == view_priority:
            return vp
    return vps[0] if vps else None


def make_rl_sample(sector_idx, visited_set, remaining, gt_label, gt_action,
                   step_history, is_trap, query_desc, query_cat,
                   navigable_sectors, visible_sectors, n_sectors,
                   vps_by_sector, ep_abs, crop_dir, target_obj_id, pair_type,
                   view_priority="far"):
    """Create one RL sample at a given exploration state."""
    vps = vps_by_sector.get(sector_idx, [])
    vp = pick_viewpoint(vps, view_priority)
    if vp is None:
        return None

    rgb_rel = vp.get("rgb") or vp.get("rgb_filename", "")
    if not rgb_rel:
        return None
    rgb_abs = os.path.join(ep_abs, rgb_rel.lstrip("./"))
    if not os.path.isfile(rgb_abs):
        return None

    # Crop
    bbox = vp.get("mask_bbox_xyxy")
    vp_is_trap = not vp.get("mask_meets_threshold", False)
    crop_filename = f"rl_{target_obj_id}_{sector_idx}_{pair_type}.jpg"
    crop_path = os.path.join(crop_dir, crop_filename)

    if bbox and not vp_is_trap:
        if not os.path.exists(crop_path):
            try:
                crop_img = crop_with_gt_bbox(rgb_abs, bbox)
                crop_img.save(crop_path, "JPEG", quality=95)
            except Exception:
                return None
    else:
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

    # Build prompt — all directions are RELATIVE to current position
    # Agent always perceives its current position as "front" (0°)
    sector_name = "front"
    angle = 0
    visited_names = [abs_to_relative_dir(sector_idx, s, n_sectors)
                     for s in sorted(visited_set)]
    visited_list = ", ".join(visited_names) if visited_names else "none"

    # All unvisited directions in relative terms (agent decides navigability itself)
    visited_rel_set = set(abs_to_relative_dir(sector_idx, s, n_sectors)
                          for s in visited_set)
    available_dirs = [d for d in SECTOR_NAMES if d not in visited_rel_set]
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

    # Visibility warning
    if is_trap or vp_is_trap:
        user_text += "\nWarning: Target object is not clearly visible from this angle."

    # Step history — directions converted to be relative to CURRENT position
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

    # GT solution — all directions are RELATIVE to current position
    visible_rel = sorted(set(abs_to_relative_dir(sector_idx, s, n_sectors)
                             for s in visible_sectors))
    navigable_rel = sorted(set(abs_to_relative_dir(sector_idx, s, n_sectors)
                               for s in navigable_sectors))

    # FPS-ranked best sectors (most informative unvisited visible+navigable)
    navigable_rel_set = set(navigable_rel)
    unvisited_visible_nav = [d for d in visible_rel
                             if d not in visited_rel_set and d in navigable_rel_set]
    best = compute_best_sectors(list(visited_rel_set), unvisited_visible_nav)

    solution = {
        "visible": visible_rel,
        "navigable": navigable_rel,
        "best_sectors": best,
        "label": gt_label,
        "action": gt_action,
        "pair_type": pair_type,
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "images": [rgb_abs, crop_path],
        "solution": json.dumps(solution),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate RL prompt data")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--index-file", required=True)
    parser.add_argument("--desc-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-dir", required=True)
    parser.add_argument("--view-priority", default="far", choices=["far", "near"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    desc_db = load_desc_db(args.desc_db)
    print(f"Loaded {len(desc_db)} objects from desc_db")

    with open(args.index_file, 'r', encoding='utf-8') as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(pairs)} pairs from index")

    os.makedirs(args.crop_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_samples = []
    stats = Counter()
    max_steps = 6

    for ep in tqdm(pairs, desc="Processing RL pairs"):
        episode_path = ep["episode_path"]
        target_obj_id = ep["target_object_id"]
        query_obj_id = ep.get("query_object_id", target_obj_id)
        query_cat = ep.get("query_object_category", "object")
        label = ep["label"]
        pair_type = ep.get("pair_type", "positive")

        if query_obj_id not in desc_db:
            stats["skip_no_desc"] += 1
            continue
        descs = desc_db[query_obj_id].get("descriptions", [])
        if not descs:
            stats["skip_empty_desc"] += 1
            continue
        # Use all descriptions (complementary attributes across descriptions)
        query_desc = "; ".join(d for d in descs if d)

        ep_abs = os.path.join(args.data_root, episode_path) if args.data_root else episode_path
        meta_path = os.path.join(ep_abs, "meta.json")
        if not os.path.exists(meta_path):
            stats["skip_no_meta"] += 1
            continue
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        vps_by_sector = get_viewpoints_by_sector(meta)
        navigable_sectors = get_navigable_sectors(vps_by_sector)
        visible_sectors = get_visible_sectors(vps_by_sector)
        n_sectors = meta.get("n_sectors", 12)

        if not visible_sectors:
            stats["skip_no_visible"] += 1
            continue

        visible_nav = [s for s in visible_sectors if s in navigable_sectors]
        if not visible_nav:
            stats["skip_no_visible_nav"] += 1
            continue

        # Common kwargs for make_rl_sample
        common = dict(
            query_desc=query_desc, query_cat=query_cat,
            navigable_sectors=navigable_sectors, visible_sectors=visible_sectors,
            n_sectors=n_sectors, vps_by_sector=vps_by_sector,
            ep_abs=ep_abs, crop_dir=args.crop_dir,
            target_obj_id=target_obj_id, pair_type=pair_type,
            view_priority=args.view_priority,
        )

        if pair_type == "positive":
            random.shuffle(visible_nav)

            # --- Sample 1: Fresh start — should MOVE if possible ---
            start = visible_nav[0]
            visited_1 = {start}
            unvisited_vis = [s for s in visible_nav if s not in visited_1]

            if unvisited_vis:
                best_target = random.choice(unvisited_vis)
                best_dir = abs_to_relative_dir(start, best_target, n_sectors)
                sample = make_rl_sample(
                    start, visited_1, max_steps - 1,
                    gt_label="Unsure", gt_action=f"MOVE {best_dir}",
                    step_history=[], is_trap=False, **common)
                if sample:
                    all_samples.append(sample)
                    stats["pos_move"] += 1

            # --- Sample 2: After exploration — should STOP ---
            n_visited = min(random.randint(2, 4), len(visible_nav))
            visited_list = visible_nav[:n_visited]
            visited_2 = set(visited_list)
            current_2 = visited_list[-1]
            remaining_2 = max_steps - n_visited

            # Build synthetic step history (absolute sectors, rendered relative at prompt time)
            history_2 = []
            for i, s in enumerate(visited_list[:-1]):
                next_s = visited_list[i + 1]
                history_2.append({
                    "step": i + 1,
                    "abs_sector": s,
                    "abs_target": next_s,
                    "verification": "Unsure",
                })

            sample2 = make_rl_sample(
                current_2, visited_2, remaining_2,
                gt_label="Yes", gt_action="STOP",
                step_history=history_2, is_trap=False, **common)
            if sample2:
                all_samples.append(sample2)
                stats["pos_stop"] += 1

        elif pair_type == "neg_same":
            start = random.choice(visible_nav)
            visited = {start}

            # 30% chance: MOVE first (unsure), then STOP with No
            unvisited_vis = [s for s in visible_nav if s not in visited]
            if unvisited_vis and random.random() < 0.3:
                target = random.choice(unvisited_vis)
                target_dir = abs_to_relative_dir(start, target, n_sectors)

                # Sample 1: should MOVE
                sample = make_rl_sample(
                    start, visited, max_steps - 1,
                    gt_label="Unsure", gt_action=f"MOVE {target_dir}",
                    step_history=[], is_trap=False, **common)
                if sample:
                    all_samples.append(sample)
                    stats["ns_move"] += 1

                # Sample 2: should STOP with No
                visited_2 = visited | {target}
                history = [{
                    "step": 1,
                    "abs_sector": start,
                    "abs_target": target,
                    "verification": "Unsure",
                }]
                sample2 = make_rl_sample(
                    target, visited_2, max_steps - 2,
                    gt_label="No", gt_action="STOP",
                    step_history=history, is_trap=False, **common)
                if sample2:
                    all_samples.append(sample2)
                    stats["ns_stop"] += 1
            else:
                # Single step: STOP with No
                sample = make_rl_sample(
                    start, visited, max_steps - 1,
                    gt_label="No", gt_action="STOP",
                    step_history=[], is_trap=False, **common)
                if sample:
                    all_samples.append(sample)
                    stats["ns_stop"] += 1

        elif pair_type == "neg_diff":
            start = random.choice(visible_nav)
            visited = {start}

            sample = make_rl_sample(
                start, visited, max_steps - 1,
                gt_label="No", gt_action="STOP",
                step_history=[], is_trap=False, **common)
            if sample:
                all_samples.append(sample)
                stats["nd_stop"] += 1

    # Shuffle
    random.shuffle(all_samples)

    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n=== RL Data Statistics ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
