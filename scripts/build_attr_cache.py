#!/usr/bin/env python3
"""
Build attribute extraction cache by running LLM inference on all objects.

Caches the Qwen-Text attribute extraction results so they don't need to be
recomputed every episode. Used by MLLM attr-mode agents.

Cache format: {object_id: {"attributes": [...], "target_class": "...", "raw": "...", "max_attrs": 8}}

Usage:
    python scripts/build_attr_cache.py \
        --desc_db ./data/pv_dataset/object_descriptions_with_category.json \
        --category_cache ./data/pv_dataset/category_cache.json \
        --output ./data/pv_dataset/attr_cache.json

    # Resume from partial cache
    python scripts/build_attr_cache.py \
        --desc_db /path/to/desc_db.json \
        --category_cache /path/to/category_cache.json \
        --output /path/to/attr_cache.json \
        --resume

    # Only process objects in a specific index
    python scripts/build_attr_cache.py \
        --desc_db /path/to/desc_db.json \
        --category_cache /path/to/category_cache.json \
        --output /path/to/attr_cache.json \
        --index_file pv_index_500.jsonl
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


PROMPT_YAML = os.path.join(os.path.dirname(__file__), "../configs/prompts/extract_v1.yaml")


def call_qwen_text(server_url: str, prompt: str, max_retries: int = 3) -> dict:
    """Call Qwen-Text server with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(server_url, json={"prompt": prompt}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [RETRY] attempt {attempt + 1} failed: {e}, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] all {max_retries} attempts failed: {e}")
                return {"text": ""}


def parse_json(text):
    """Parse JSON from LLM response, stripping markdown fences."""
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except:
        return {"attributes": []}


def filter_and_standardize(attr_spec):
    """Filter invalid attributes and standardize names (same logic as mllm_policy.py)."""
    invalid_values = {"unknown", "unspecified", "not mentioned", "n/a", "none", ""}
    filtered_attrs = []
    for a in attr_spec.get("attributes", []):
        evidence = str(a.get("evidence_phrase", "")).strip().lower()
        if evidence and evidence not in invalid_values:
            filtered_attrs.append(a)
    attr_spec["attributes"] = filtered_attrs

    # Standardize duplicate names
    name_counts = {}
    for a in attr_spec.get("attributes", []):
        base_name = a.get("name", "attr")
        if base_name in name_counts:
            name_counts[base_name] += 1
            a["name"] = f"{base_name}_{name_counts[base_name]}"
        else:
            name_counts[base_name] = 1
            a["_original_name"] = base_name

    if any(c > 1 for c in name_counts.values()):
        for a in attr_spec.get("attributes", []):
            orig = a.get("_original_name")
            if orig and name_counts.get(orig, 1) > 1:
                if a["name"] == orig:
                    a["name"] = f"{orig}_1"

    # Clean up temp keys
    for a in attr_spec.get("attributes", []):
        a.pop("_original_name", None)

    return attr_spec


def load_index_objects(index_file: str) -> set:
    """Load unique object IDs from JSONL index file."""
    obj_ids = set()
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("target_object_id")
            qid = rec.get("query_object_id")
            if tid:
                obj_ids.add(tid)
            if qid:
                obj_ids.add(qid)
    return obj_ids


def main():
    parser = argparse.ArgumentParser(description="Build attribute extraction cache via LLM inference")
    parser.add_argument("--desc_db", required=True, help="Path to object_descriptions_with_category.json")
    parser.add_argument("--category_cache", required=True, help="Path to category_cache.json")
    parser.add_argument("--output", required=True, help="Output cache JSON path")
    parser.add_argument("--server_url", default="http://127.0.0.1:12182/qwen-text",
                        help="Qwen-Text server URL")
    parser.add_argument("--index_file", default=None,
                        help="Optional JSONL index file to filter objects")
    parser.add_argument("--max_attrs", type=int, default=8,
                        help="Maximum number of attributes to extract (default: 8)")
    parser.add_argument("--prompt_yaml", default=None,
                        help="Path to extract prompt YAML (default: configs/prompts/extract_v1.yaml)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing cache (skip already cached objects)")
    args = parser.parse_args()

    # Load prompt template from YAML
    prompt_yaml_path = args.prompt_yaml or PROMPT_YAML
    print(f"Loading prompt template: {prompt_yaml_path}")
    prompt_cfg = OmegaConf.load(prompt_yaml_path)
    extract_template = prompt_cfg.template

    # Load descriptions database
    print(f"Loading desc_db: {args.desc_db}")
    with open(args.desc_db, 'r', encoding='utf-8') as f:
        desc_db = json.load(f)
    print(f"  Total objects in desc_db: {len(desc_db)}")

    # Load category cache
    print(f"Loading category cache: {args.category_cache}")
    with open(args.category_cache, 'r', encoding='utf-8') as f:
        category_cache = json.load(f)
    print(f"  Total categories cached: {len(category_cache)}")

    # Filter by index file if provided
    if args.index_file:
        index_path = args.index_file
        if not os.path.isabs(index_path):
            dataset_root = str(Path(args.desc_db).parent.parent)
            candidate = os.path.join(dataset_root, "val", index_path)
            if os.path.exists(candidate):
                index_path = candidate
        print(f"Filtering by index file: {index_path}")
        index_objects = load_index_objects(index_path)
        print(f"  Unique objects in index: {len(index_objects)}")
        filtered_db = {k: v for k, v in desc_db.items() if k in index_objects}
        print(f"  Objects after filtering: {len(filtered_db)}")
        desc_db = filtered_db

    # Load existing cache for resume
    cache = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Resuming from existing cache: {len(cache)} entries")

    # Process each object
    total = len(desc_db)
    skipped = 0
    processed = 0
    errors = 0

    print(f"\nProcessing {total} objects...")
    print(f"Server: {args.server_url}")
    print(f"Max attributes: {args.max_attrs}")
    print()

    start_time = time.time()

    for i, (obj_id, obj_data) in enumerate(desc_db.items()):
        # Skip if already cached
        if obj_id in cache:
            skipped += 1
            continue

        descs = obj_data.get("descriptions", ["", "", ""])
        desc1 = descs[0] if len(descs) > 0 else ""
        desc2 = descs[1] if len(descs) > 1 else desc1
        desc3 = descs[2] if len(descs) > 2 else desc1

        if not desc1:
            print(f"  [SKIP] {obj_id}: no descriptions")
            skipped += 1
            continue

        # Get target class from category cache
        target_class = category_cache.get(obj_id, {}).get("pred_coarse", "object")

        prompt = extract_template.format(
            class_text=target_class,
            desc1=desc1, desc2=desc2, desc3=desc3,
            max_attrs=args.max_attrs
        )

        res = call_qwen_text(args.server_url, prompt)
        raw_text = res.get("text", "{}")

        # Parse and filter
        attr_spec = parse_json(raw_text)
        attr_spec = filter_and_standardize(attr_spec)

        attrs = attr_spec.get("attributes", [])
        if not attrs:
            errors += 1
            print(f"  [WARN] {obj_id}: no valid attributes extracted")

        cache[obj_id] = {
            "attributes": attrs,
            "target_class": target_class,
            "raw": raw_text,
            "max_attrs": args.max_attrs
        }
        processed += 1

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  [{i + 1}/{total}] processed={processed}, skipped={skipped}, "
                  f"errors={errors}, rate={rate:.1f} obj/s")

        # Periodic save (every 100 objects)
        if processed % 100 == 0:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    # Final save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\nDone! Cache saved to: {args.output}")
    print(f"  Total: {total}, Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # Stats
    all_attr_counts = [len(v.get("attributes", [])) for v in cache.values()]
    if all_attr_counts:
        avg_attrs = sum(all_attr_counts) / len(all_attr_counts)
        print(f"  Avg attributes per object: {avg_attrs:.1f}")
        print(f"  Min/Max: {min(all_attr_counts)}/{max(all_attr_counts)}")


if __name__ == "__main__":
    main()
