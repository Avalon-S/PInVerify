#!/bin/bash
# Build all caches (category, merge, attr) for the PV benchmark.
#
# Usage:
#   bash scripts/build_all_caches.sh                           # Build all 3 caches
#   bash scripts/build_all_caches.sh /path/to/desc_db.json     # Custom desc_db path
#
# Prerequisites:
#   - Qwen-Text server running at http://127.0.0.1:12182/qwen-text
#
# Output (all in same directory as desc_db):
#   - category_cache.json   (object_id -> category)
#   - merge_cache.json      (object_id -> merged description)
#   - attr_cache.json       (object_id -> extracted attributes)

set -e
export PYTHONPATH=$PYTHONPATH:.

DESC_DB="${1:-./data/pv_dataset/object_descriptions_with_category.json}"
CACHE_DIR="$(dirname "$DESC_DB")"
SERVER_URL="${2:-http://127.0.0.1:12182/qwen-text}"

echo "========================================"
echo "Building all caches"
echo "  desc_db:   $DESC_DB"
echo "  cache_dir: $CACHE_DIR"
echo "  server:    $SERVER_URL"
echo "========================================"

# 1. Category Cache
echo ""
echo "===== [1/3] Building category_cache.json ====="
python scripts/build_category_cache.py \
    --desc_db "$DESC_DB" \
    --output "$CACHE_DIR/category_cache.json" \
    --server_url "$SERVER_URL" \
    --resume

# 2. Merge Cache
echo ""
echo "===== [2/3] Building merge_cache.json ====="
python scripts/build_merge_cache.py \
    --desc_db "$DESC_DB" \
    --output "$CACHE_DIR/merge_cache.json" \
    --server_url "$SERVER_URL" \
    --resume

# 3. Attribute Extraction Cache (depends on category_cache)
echo ""
echo "===== [3/3] Building attr_cache.json ====="
python scripts/build_attr_cache.py \
    --desc_db "$DESC_DB" \
    --category_cache "$CACHE_DIR/category_cache.json" \
    --output "$CACHE_DIR/attr_cache.json" \
    --server_url "$SERVER_URL" \
    --resume

echo ""
echo "========================================"
echo "All caches built successfully!"
echo "  $CACHE_DIR/category_cache.json"
echo "  $CACHE_DIR/merge_cache.json"
echo "  $CACHE_DIR/attr_cache.json"
echo "========================================"
