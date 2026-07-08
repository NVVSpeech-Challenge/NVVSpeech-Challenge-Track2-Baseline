#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 只添加单独类别（不包括组合类别）
python scripts/add_all_category_tokens.py \
    --pretrained_path VoxCPM2/ckpt/base \
    --output_path VoxCPM2/ckpt/base_nv \
    --data_path NV_data/staging/merged_train_data.jsonl \
    --no_combo