#!/bin/bash
# Extract all shard tar files with dedup renaming.
# Each shard gets a unique prefix so all files can coexist in one output dir.
#
# Usage: bash extract_all_shards.sh
#   Edit SHARD_DIR and OUT_DIR below.

set -euo pipefail

# --- Configure these ---
SHARD_DIR="/mnt/node03_tmpdata0/data/movies"
OUT_DIR="/mnt/node03_tmpdata0/data/movies/extracted"
PY_SCRIPT="$(cd "$(dirname "$0")" && pwd)/extract_tar_dedup.py"

source /mnt/node02_tmpdata0/miniconda3/etc/profile.d/conda.sh
conda activate voxcpm2

mkdir -p "$OUT_DIR"

echo "=============================================="
echo " Extracting all shards with dedup renaming"
echo " Shard dir: $SHARD_DIR"
echo " Output:    $OUT_DIR"
echo "=============================================="

for tar_file in "$SHARD_DIR"/shard-00000*.tar; do
    if [ ! -f "$tar_file" ]; then
        echo "No shard files found in $SHARD_DIR"
        exit 1
    fi

    shard_name=$(basename "$tar_file" .tar)
    # Prefix like "s000_", "s001_", ...
    idx=$(echo "$shard_name" | grep -oP '\d+$')
    prefix="s${idx}_"

    echo ""
    echo "--- Processing: $shard_name (prefix=$prefix) ---"

    python3 "$PY_SCRIPT" "$tar_file" "$OUT_DIR" --prefix "$prefix"
done

echo ""
echo "=============================================="
echo " All shards extracted."
echo " Output: $OUT_DIR"
echo " File count: $(find "$OUT_DIR" -type f | wc -l)"
echo "=============================================="
