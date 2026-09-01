#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset.py
---------------------------------
Data loading for PInVerify.
- Reads .jsonl / .json / .json.gz index formats
- Entry points: load_pairs / load_desc_db / resolve_episode_abs_dir
- Works with the val/<scene>/<episode>/meta.json layout
"""

import os, json, gzip, random
from typing import Any, Dict, List, Optional

# ===== Default paths (the runner can override them) =====
DEFAULT_DATASET_ROOT   = "./data/pv_dataset"
DEFAULT_CAPTURE_SUBDIR = "pin_capture"
DEFAULT_SPLIT          = "val"
DEFAULT_INDEX          = "pv_index_all.jsonl"
DEFAULT_DESC_DB        = "object_descriptions_with_category.json"


# ====== Basic readers ======
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[dataset][WARN] Failed to parse line: {e}")
                continue


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_gz(path: str) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ====== Main entry: load the index ======
def load_pairs(index_path: str, mode: str = "all", num: int = 200, seed: int = 0) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []

    if index_path.endswith(".jsonl"):
        pairs = list(read_jsonl(index_path))
    elif index_path.endswith(".json"):
        data = load_json(index_path)
        if isinstance(data, dict) and "episodes" in data:
            pairs = data["episodes"]
        elif isinstance(data, list):
            pairs = data
        else:
            raise ValueError(f"Unrecognized JSON format in {index_path}")
    elif index_path.endswith(".json.gz"):
        data = load_json_gz(index_path)
        if isinstance(data, dict) and "episodes" in data:
            pairs = data["episodes"]
        elif isinstance(data, list):
            pairs = data
        else:
            raise ValueError(f"Unrecognized JSON.GZ format in {index_path}")
    else:
        pairs = list(read_jsonl(index_path))

    if mode == "random" and num < len(pairs):
        random.seed(seed)
        pairs = random.sample(pairs, num)

    return pairs


# ====== Description database ======
def load_desc_db(path: str) -> Dict[str, Any]:
    if path.endswith(".gz"):
        return load_json_gz(path)
    else:
        return load_json(path)


# ====== Episode loading ======
def resolve_episode_abs_dir(dataset_root: str, capture_subdir: str, episode_rel: str) -> str:
    return os.path.join(dataset_root, capture_subdir, episode_rel)


def _abs_path(ep_root: str, rel_or_abs: Optional[str]) -> Optional[str]:
    if not rel_or_abs:
        return None
    p = rel_or_abs
    if os.path.isabs(p):
        return p
    return os.path.join(ep_root, p.lstrip("./"))


def load_episode_from_root(ep_root: str) -> Dict[str, Any]:
    """
    Load meta.json from an episode root and add absolute paths per frame:
      - captures[i]['rgb_path']   = abs(ep_root / captures[i]['rgb'])
      - captures[i]['depth_path'] = abs(ep_root / captures[i]['depth']) (when present)
    The original 'rgb' / 'depth' fields are left untouched.
    """
    meta_path = os.path.join(ep_root, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"[dataset] meta.json not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    captures = meta.get("captures")
    if not isinstance(captures, list):
        raise TypeError(f"[dataset] 'captures' should be a list in {meta_path}")

    for cap in captures:
        # follows the capture layout: cap['rgb'] / cap['depth']
        cap["rgb_path"] = _abs_path(ep_root, cap.get("rgb"))
        if "depth" in cap:
            cap["depth_path"] = _abs_path(ep_root, cap.get("depth"))

    meta["captures"] = captures
    meta["episode_root"] = ep_root
    return meta


# ====== Helpers ======
def get_descs_for_object(desc_db: Dict[str, Any], object_id: str, pad_to: int = 3) -> List[str]:
    if not object_id or object_id not in desc_db:
        return [""] * pad_to
    obj_entry = desc_db[object_id]
    descs = obj_entry.get("descriptions") or obj_entry.get("description") or []
    if isinstance(descs, str):
        descs = [descs]
    if len(descs) < pad_to and len(descs) > 0:
        descs = (descs * ((pad_to + len(descs) - 1) // len(descs)))[:pad_to]
    elif len(descs) == 0:
        descs = [""] * pad_to
    return descs


# ====== JSON dump (handy when debugging) ======
def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ====== Self-test entry point ======
if __name__ == "__main__":
    print("[dataset] self-test:")
    pairs = load_pairs(DEFAULT_INDEX, mode="all")
    print(f"Loaded {len(pairs)} pairs (example):")
    if pairs:
        print(pairs[0])
