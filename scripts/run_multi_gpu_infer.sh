#!/bin/bash
# ==============================================================================
# Multi-GPU parallel batch inference for NVV evaluation
# Usage:
#   bash scripts/run_multi_gpu_infer.sh \
#       --ckpt_dir /path/to/ckpt \
#       --input_json outputs/eval/prepared_data_zh_alltags.json \
#       --output_dir outputs/eval/audios_zh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
CKPT_DIR=""
INPUT_JSON=""
OUTPUT_DIR=""
NUM_GPUS=8
CFG_VALUE=2.0
INFERENCE_TIMESTEPS=10
MAX_LEN=600
CONDA_ENV="voxcpm2"
CONDA_PATH="/mnt/node02_tmpdata0/miniconda3"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt_dir) CKPT_DIR="$2"; shift 2 ;;
        --input_json) INPUT_JSON="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --cfg_value) CFG_VALUE="$2"; shift 2 ;;
        --inference_timesteps) INFERENCE_TIMESTEPS="$2"; shift 2 ;;
        --max_len) MAX_LEN="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$CKPT_DIR" ] || [ -z "$INPUT_JSON" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --ckpt_dir <path> --input_json <path> --output_dir <path>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Split dataset into N shards
echo ">>> Splitting dataset into ${NUM_GPUS} shards..."
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

TOTAL=$(python3 -c "import json; print(len(json.load(open('$INPUT_JSON'))))")
SHARD_SIZE=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
echo "    Total: $TOTAL samples, ~$SHARD_SIZE per GPU"

SHARD_DIR="$OUTPUT_DIR/.shards"
mkdir -p "$SHARD_DIR"

python3 -c "
import json, os
with open('$INPUT_JSON') as f:
    data = json.load(f)
n = $NUM_GPUS
size = $SHARD_SIZE
for i in range(n):
    shard = data[i*size:(i+1)*size]
    path = os.path.join('$SHARD_DIR', f'shard_{i}.json')
    with open(path, 'w') as f:
        json.dump(shard, f, ensure_ascii=False, indent=2)
    print(f'  Shard {i}: {len(shard)} samples -> {path}')
"

echo ""
echo ">>> Launching ${NUM_GPUS} GPU workers..."

# Launch one process per GPU
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    SHARD_JSON="$SHARD_DIR/shard_${i}.json"
    SHARD_OUT="$OUTPUT_DIR"

    if [ ! -f "$SHARD_JSON" ]; then
        continue
    fi

    echo "  GPU $i: starting worker..."
    (
        source "$CONDA_PATH/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        export CUDA_VISIBLE_DEVICES=$i
        export TORCHDYNAMO_DISABLE=1
        python3 "$PROJ_DIR/scripts/batch_infer_nvv.py" \
            --ckpt_dir "$CKPT_DIR" \
            --input_json "$SHARD_JSON" \
            --output_dir "$SHARD_OUT" \
            --cfg_value "$CFG_VALUE" \
            --inference_timesteps "$INFERENCE_TIMESTEPS" \
            --max_len "$MAX_LEN" \
            > "$OUTPUT_DIR/gpu_${i}.log" 2>&1
        echo "GPU $i: DONE" >> "$OUTPUT_DIR/gpu_${i}.log"
    ) &
    PIDS+=($!)
done

echo "    Waiting for ${#PIDS[@]} workers to finish..."

# Wait for all
FAILED=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAILED=$((FAILED + 1))
done

echo ""
echo ">>> All workers finished (failed: $FAILED)"

# Count results
TOTAL_DONE=$(ls "$OUTPUT_DIR"/*.wav 2>/dev/null | wc -l)
echo "    Total audio files: $TOTAL_DONE / $TOTAL"

# Cleanup shards
rm -rf "$SHARD_DIR"

if [ "$TOTAL_DONE" -ge "$TOTAL" ]; then
    echo ">>> Success! All samples generated."
else
    echo ">>> WARNING: Missing $((TOTAL - TOTAL_DONE)) files."
fi
