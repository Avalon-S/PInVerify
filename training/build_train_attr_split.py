#!/usr/bin/env python3
"""
Build attribute extraction cache for training set objects.

Wrapper around scripts/build_attr_cache.py logic, adapted for training set.
Requires a running Qwen-Text server.

Usage:
    python training/build_train_attr_split.py \
        --desc-db ./data/pv_dataset/train/object_descriptions_with_category.json \
        --category-cache ./data/pv_dataset/train/category_cache.json \
        --index-file ./data/train_sft/pv_train_sft_index.jsonl \
        --output ./data/train/attr_cache_train.json \
        --server-url http://127.0.0.1:12182/qwen-text \
        --resume
"""

import argparse
import json
import os
import re
import sys
import time
import requests
from pathlib import Path
from omegaconf import OmegaConf

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROMPT_YAML = os.path.join(os.path.dirname(__file__), "../configs/prompts/extract_v1.yaml")


def call_qwen_text(server_url, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.post(server_url, json={"prompt": prompt}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"text": ""}


def parse_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return {"attributes": []}


def filter_and_standardize(attr_spec):
    """Filter invalid attributes and standardize names."""
    invalid_values = {"unknown", "unspecified", "not mentioned", "n/a", "none", ""}
    filtered = []
    for a in attr_spec.get("attributes", []):
        evidence = str(a.get("evidence_phrase", "")).strip().lower()
        if evidence and evidence not in invalid_values:
            filtered.append(a)
    attr_spec["attributes"] = filtered

    name_counts = {}
    for a in attr_spec["attributes"]:
        base_name = a.get("name", "attr")
        if base_name in name_counts:
            name_counts[base_name] += 1
            a["name"] = f"{base_name}_{name_counts[base_name]}"
        else:
            name_counts[base_name] = 1

    if any(c > 1 for c in name_counts.values()):
        for a in attr_spec["attributes"]:
            base = a.get("name", "attr").rsplit("_", 1)[0] if "_" in a.get("name", "") else a.get("name", "")
            if name_counts.get(base, 1) > 1 and a["name"] == base:
                a["name"] = f"{base}_1"

    return attr_spec


def load_index_objects(index_file):
    obj_ids = set()
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for key in ("target_object_id", "query_object_id"):
                if rec.get(key):
                    obj_ids.add(rec[key])
    return obj_ids


def main():
    parser = argparse.ArgumentParser(
        description="Build attribute cache for training set")
    parser.add_argument("--desc-db", required=True,
                        help="Path to object_descriptions_with_category.json")
    parser.add_argument("--category-cache", required=True,
                        help="Path to category_cache.json")
    parser.add_argument("--index-file", default=None,
                        help="JSONL index file to filter objects (optional)")
    parser.add_argument("--output", required=True,
                        help="Output attr_cache JSON path")
    parser.add_argument("--server-url", default="http://127.0.0.1:12182/qwen-text")
    parser.add_argument("--max-attrs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load prompt template
    prompt_cfg = OmegaConf.load(PROMPT_YAML)
    template = prompt_cfg.template

    # Load desc_db
    with open(args.desc_db, 'r', encoding='utf-8') as f:
        desc_db = json.load(f)
    print(f"Loaded {len(desc_db)} objects from desc_db")

    # Load category cache
    with open(args.category_cache, 'r', encoding='utf-8') as f:
        category_cache = json.load(f)
    print(f"Loaded {len(category_cache)} categories from cache")

    # Filter by index
    if args.index_file and os.path.exists(args.index_file):
        index_objs = load_index_objects(args.index_file)
        desc_db = {k: v for k, v in desc_db.items() if k in index_objs}
        print(f"Filtered to {len(desc_db)} objects from index")

    # Resume
    cache = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Resuming: {len(cache)} existing entries")

    total = len(desc_db)
    processed = 0
    skipped = 0
    start = time.time()

    for i, (obj_id, obj_data) in enumerate(desc_db.items()):
        if obj_id in cache:
            skipped += 1
            continue

        descs = obj_data.get("descriptions", ["", "", ""])
        desc1 = descs[0] if len(descs) > 0 else ""
        desc2 = descs[1] if len(descs) > 1 else desc1
        desc3 = descs[2] if len(descs) > 2 else desc1
        if not desc1:
            skipped += 1
            continue

        target_class = category_cache.get(obj_id, {}).get("pred_coarse", "object")

        prompt = template.format(
            class_text=target_class,
            desc1=desc1, desc2=desc2, desc3=desc3,
            max_attrs=args.max_attrs
        )

        res = call_qwen_text(args.server_url, prompt)
        attr_spec = parse_json(res.get("text", "{}"))
        attr_spec = filter_and_standardize(attr_spec)

        cache[obj_id] = {
            "attributes": attr_spec.get("attributes", []),
            "target_class": target_class,
            "raw": res.get("text", ""),
            "max_attrs": args.max_attrs,
        }
        processed += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{total}] processed={processed}, skipped={skipped}, "
                  f"rate={processed/elapsed:.1f}/s")

        if processed % 100 == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    # Final save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"\nDone! {processed} processed, {skipped} skipped in {elapsed:.1f}s")
    print(f"Cache saved to: {args.output}")


if __name__ == "__main__":
    main()
