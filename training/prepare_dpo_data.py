#!/usr/bin/env python3
"""
Generate DPO preference pair data for Qwen3-VL-4B.

Each sample contains:
- messages: [system, user] (prompt only)
- images: [scene_rgb, crop]
- chosen: correct assistant response
- rejected: plausible but incorrect assistant response

Focus: fix neg_same false-positive bias from SFT.
- neg_same pairs: chosen=No (correct rejection), rejected=Yes (false accept)
- positive pairs: chosen=Yes (correct accept), rejected=No (false reject)
  (included to prevent over-correction toward always saying No)

Uses same concrete attribute comparison as SFT v3 for neg_same chosen CoT.

Usage:
    python training/prepare_dpo_data.py \
        --data-root ./data/pv_dataset \
        --train-dir ./data/pv_dataset/train_rl \
        --index-file ./data/pv_dataset/train_rl/pv_train_rl_index.jsonl \
        --desc-db ./data/pv_dataset/train/object_descriptions_with_category.json \
        --attr-cache ./data/train/attr_cache_train.json \
        --output ./data/pv_dataset/train_rl/dpo_data_v3.jsonl \
        --crop-dir ./data/train/crops_dpo
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


# ---- Image helpers ----

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
        elif isinstance(entry, dict) and "descriptions" not in entry:
            descs = entry.get("description", [])
            if isinstance(descs, str):
                descs = [descs]
            entry["descriptions"] = descs
    return db


# ---- Viewpoint helpers ----

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


def abs_to_relative_dir(current_sector, target_sector, n_sectors=12):
    step = max(1, n_sectors // 6)
    cur_dir = (current_sector // step) % 6
    tgt_dir = (target_sector // step) % 6
    offset = (tgt_dir - cur_dir) % 6
    return SECTOR_NAMES[offset]


# ---- CoT builders ----

def build_answer(verification, action):
    return f"verification: {verification}\naction: {action}"


def build_cot_positive_correct(attrs, available_dirs):
    """Chosen CoT for positive: all attributes match → Yes, STOP."""
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


def build_cot_positive_rejected(attrs, available_dirs):
    """Rejected CoT for positive: wrongly rejects a correct match → No, STOP."""
    lines = []
    if available_dirs:
        lines.append(f"[Navigation] Available unvisited sectors: {', '.join(available_dirs)}.")
    else:
        lines.append("[Navigation] No unvisited sectors remaining.")
    lines.append("")
    lines.append("[Verification]")
    # Pick a random attr to "doubt"
    doubt_idx = random.randint(0, max(0, len(attrs) - 1))
    for i, a in enumerate(attrs):
        name = a.get("name", "attribute")
        evidence = a.get("evidence_phrase", "")
        if i == doubt_idx:
            lines.append(f"- {name}: No — expected \"{evidence}\" but observed something different.")
        else:
            lines.append(f"- {name}: Unsure — cannot fully confirm \"{evidence}\" from this view.")
    lines.append("")
    lines.append(f"Found mismatching attribute. This is not the target.")
    lines.append("")
    lines.append("[Decision] Found mismatching attribute. Confident this is NOT the target. "
                 "Stopping with No.")
    return "\n".join(lines)


def build_cot_neg_same_correct(attrs, available_dirs, mismatch_attr, target_attrs_map=None):
    """Chosen CoT for neg_same: correctly identifies mismatch → No, STOP.

    Uses concrete comparison with target object's actual attributes (v3).
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


def build_cot_neg_same_rejected(attrs, available_dirs):
    """Rejected CoT for neg_same: wrongly accepts a non-match → Yes, STOP.

    This is the false-positive pattern we want the model to avoid.
    """
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


def build_cot_neg_diff_correct(query_cat, target_cat, available_dirs,
                               target_desc_summary=None):
    """Chosen CoT for neg_diff: correctly identifies category mismatch → No, STOP.

    v3: Includes target object's actual description for concrete evidence.
    """
    lines = []
    if available_dirs:
        lines.append(f"[Navigation] Available unvisited sectors: {', '.join(available_dirs)}.")
    else:
        lines.append("[Navigation] No unvisited sectors remaining.")
    lines.append("")
    lines.append("[Verification]")
    lines.append(f"- Category check: No — target should be a {query_cat}, "
                 f"but the object in the scene is a {target_cat}.")
    if target_desc_summary:
        lines.append(f"- Appearance: This object appears to be \"{target_desc_summary}\", "
                     f"which does not match the target description at all.")
    lines.append("")
    lines.append(f"Category mismatch: expected {query_cat}, got {target_cat}.")
    lines.append("")
    lines.append("[Decision] Category does not match. This is definitely NOT the target. "
                 "Stopping with No.")
    return "\n".join(lines)


def build_cot_neg_diff_rejected(query_cat, attrs, available_dirs):
    """Rejected CoT for neg_diff: wrongly accepts despite category mismatch → Yes, STOP."""
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
        lines.append(f"- {name}: Yes — \"{evidence}\" seems consistent.")
    lines.append("")
    lines.append("All attributes appear to match the target description.")
    lines.append("")
    lines.append("[Decision] All attributes verified as matching. Confident this is the target. "
                 "Stopping with Yes.")
    return "\n".join(lines)


# ---- Main processing ----

def process_episode(ep, meta, desc_db, attr_cache, crop_dir, view_priority="far"):
    """Generate DPO preference pairs for one episode."""
    target_obj_id = ep["target_object_id"]
    query_obj_id = ep.get("query_object_id", target_obj_id)
    query_cat = ep.get("query_object_category", ep.get("target_object_category", "object"))
    target_cat = ep.get("target_object_category", "object")
    pair_type = ep.get("pair_type", "positive")

    # Get query description
    if query_obj_id not in desc_db:
        return []
    descs_entry = desc_db[query_obj_id]
    descs = descs_entry.get("descriptions", [])
    if not descs:
        return []
    query_desc = "; ".join(d for d in descs if d)

    # Get query attributes for CoT
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
    visible_nav = [s for s in visible_sectors if s in navigable_sectors]
    if not visible_nav:
        return []

    # Pick a visible+navigable starting sector
    start_sector = random.choice(visible_nav)
    visited = {start_sector}

    # Get viewpoint and images
    vps = vps_by_sector.get(start_sector, [])
    vp = pick_viewpoint(vps, view_priority)
    if vp is None:
        return []

    ep_abs = ep.get("_ep_abs", "")
    rgb_rel = vp.get("rgb") or vp.get("rgb_filename", "")
    if not rgb_rel:
        return []
    rgb_abs = os.path.join(ep_abs, rgb_rel.lstrip("./"))
    if not os.path.isfile(rgb_abs):
        return []

    # Crop image
    bbox = vp.get("mask_bbox_xyxy")
    vp_is_trap = not vp.get("mask_meets_threshold", False)
    crop_filename = f"dpo_{target_obj_id}_{start_sector}_{pair_type}.jpg"
    crop_path = os.path.join(crop_dir, crop_filename)

    if bbox and not vp_is_trap:
        if not os.path.exists(crop_path):
            try:
                crop_img = crop_with_gt_bbox(rgb_abs, bbox)
                crop_img.save(crop_path, "JPEG", quality=95)
            except Exception:
                return []
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
                return []

    # Build user prompt
    sector_name = "front"
    angle = 0
    visited_names = [abs_to_relative_dir(start_sector, s, n_sectors)
                     for s in sorted(visited)]
    visited_list = ", ".join(visited_names) if visited_names else "none"
    visited_rel_set = set(abs_to_relative_dir(start_sector, s, n_sectors)
                          for s in visited)
    available_dirs = [d for d in SECTOR_NAMES if d not in visited_rel_set]
    available_str = ", ".join(available_dirs) if available_dirs else "none"

    user_text = USER_TEMPLATE.format(
        query_description=query_desc,
        query_category=query_cat,
        sector_name=sector_name,
        angle=angle,
        visited_list=visited_list,
        remaining=max_steps - 1,
        available_sectors=available_str,
    )

    if vp_is_trap:
        user_text += "\nWarning: Target object is not clearly visible from this angle."

    # Build chosen and rejected responses
    samples = []

    if pair_type == "positive":
        chosen_cot = build_cot_positive_correct(attrs, available_dirs)
        chosen_text = (f"<think>\n{chosen_cot}\n</think>\n\n"
                       f"<answer>\n{build_answer('Yes', 'STOP')}\n</answer>")

        rejected_cot = build_cot_positive_rejected(attrs, available_dirs)
        rejected_text = (f"<think>\n{rejected_cot}\n</think>\n\n"
                         f"<answer>\n{build_answer('No', 'STOP')}\n</answer>")

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": chosen_text},
            ],
            "images": [rgb_abs, crop_path],
            "rejected_response": rejected_text,
        })

    elif pair_type == "neg_same":
        # Build target attrs map for concrete comparison (v3)
        target_attrs_map = {}
        if target_obj_id in attr_cache:
            for ta in attr_cache[target_obj_id].get("attributes", []):
                ta_name = ta.get("name", "")
                ta_evidence = ta.get("evidence_phrase", "")
                if ta_name and ta_evidence:
                    target_attrs_map[ta_name] = ta_evidence
        if not target_attrs_map and target_obj_id in desc_db:
            target_descs = desc_db[target_obj_id].get("descriptions", [])
            for i, d in enumerate(target_descs[:3]):
                if d:
                    target_attrs_map[f"desc_{i+1}"] = d

        mismatch_attr = random.choice(attrs)["name"] if attrs else "appearance"

        chosen_cot = build_cot_neg_same_correct(
            attrs, available_dirs, mismatch_attr, target_attrs_map)
        chosen_text = (f"<think>\n{chosen_cot}\n</think>\n\n"
                       f"<answer>\n{build_answer('No', 'STOP')}\n</answer>")

        rejected_cot = build_cot_neg_same_rejected(attrs, available_dirs)
        rejected_text = (f"<think>\n{rejected_cot}\n</think>\n\n"
                         f"<answer>\n{build_answer('Yes', 'STOP')}\n</answer>")

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": chosen_text},
            ],
            "images": [rgb_abs, crop_path],
            "rejected_response": rejected_text,
        })

    elif pair_type == "neg_diff":
        # v3: Get target object's description for concrete comparison
        target_desc_summary = None
        if target_obj_id in desc_db:
            target_descs = desc_db[target_obj_id].get("descriptions", [])
            if target_descs:
                target_desc_summary = target_descs[0]

        chosen_cot = build_cot_neg_diff_correct(query_cat, target_cat, available_dirs,
                                                target_desc_summary)
        chosen_text = (f"<think>\n{chosen_cot}\n</think>\n\n"
                       f"<answer>\n{build_answer('No', 'STOP')}\n</answer>")

        rejected_cot = build_cot_neg_diff_rejected(query_cat, attrs, available_dirs)
        rejected_text = (f"<think>\n{rejected_cot}\n</think>\n\n"
                         f"<answer>\n{build_answer('Yes', 'STOP')}\n</answer>")

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": chosen_text},
            ],
            "images": [rgb_abs, crop_path],
            "rejected_response": rejected_text,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pair data")
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
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--crop-dir", required=True,
                        help="Directory to save cropped images")
    parser.add_argument("--view-priority", default="far",
                        choices=["far", "near"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    desc_db = load_desc_db(args.desc_db)
    print(f"Loaded {len(desc_db)} objects from desc_db")

    attr_cache = {}
    if args.attr_cache and os.path.exists(args.attr_cache):
        with open(args.attr_cache, 'r', encoding='utf-8') as f:
            attr_cache = json.load(f)
        print(f"Loaded {len(attr_cache)} objects from attr_cache")

    with open(args.index_file, 'r', encoding='utf-8') as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(pairs)} pairs from index")

    os.makedirs(args.crop_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_samples = []
    stats = Counter()

    for ep in tqdm(pairs, desc="Processing DPO pairs"):
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

        # Pass ep_abs through ep dict for process_episode
        ep["_ep_abs"] = ep_abs

        ep_samples = process_episode(ep, meta, desc_db, attr_cache,
                                     args.crop_dir, args.view_priority)

        all_samples.extend(ep_samples)
        pair_type = ep.get("pair_type", "unknown")
        stats[f"pairs_{pair_type}"] += 1
        stats[f"samples_{pair_type}"] += len(ep_samples)

    random.shuffle(all_samples)

    with open(args.output, 'w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n=== DPO Data Statistics ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
