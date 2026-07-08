#!/bin/bash
# Launch 8-GPU parallel inference: GPU 0-3 for ZH, GPU 4-7 for EN
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
CKPT="/mnt/node01_tmpdata0/hujingbin/workspace/codes/VoxCPM2/ckpt/exp/checkpoints/finetune_all/step_0023000"

source /mnt/node02_tmpdata0/miniconda3/etc/profile.d/conda.sh
conda activate voxcpm2

# Clean and recreate output dirs
rm -rf "$PROJ_DIR/outputs/eval/audios_zh" "$PROJ_DIR/outputs/eval/audios_en"
mkdir -p "$PROJ_DIR/outputs/eval/audios_zh" "$PROJ_DIR/outputs/eval/audios_en"

# Split data into 8 shards per language
python3 -c "
import json, os
for lang in ['zh', 'en']:
    path = os.path.join('$PROJ_DIR', f'outputs/eval/prepared_data_{lang}_16tags.json')
    with open(path) as f:
        data = json.load(f)
    n, size = 8, (len(data) + 7) // 8
    for i in range(n):
        shard = data[i*size:(i+1)*size]
        if not shard: continue
        out = os.path.join('$PROJ_DIR', f'outputs/eval/.s_{lang}_{i}.json')
        with open(out, 'w') as f:
            json.dump(shard, f, ensure_ascii=False)
    print(f'{lang.upper()}: {len(data)} -> {n} shards')
"

# Launch 8 processes
PIDS=()
for GPU in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU python3 "$PROJ_DIR/scripts/batch_infer_nvv.py" \
        --ckpt_dir "$CKPT" \
        --input_json "$PROJ_DIR/outputs/eval/.s_zh_${GPU}.json" \
        --output_dir "$PROJ_DIR/outputs/eval/audios_zh" \
        --cfg_value 2.0 --inference_timesteps 10 --max_len 600 \
        > "$PROJ_DIR/outputs/eval/.log_zh_gpu${GPU}.txt" 2>&1 &
    PIDS+=($!)
    echo "GPU $GPU: ZH worker (pid $!)"
done

for GPU in 4 5 6 7; do
    SID=$((GPU - 4))
    CUDA_VISIBLE_DEVICES=$GPU python3 "$PROJ_DIR/scripts/batch_infer_nvv.py" \
        --ckpt_dir "$CKPT" \
        --input_json "$PROJ_DIR/outputs/eval/.s_en_${SID}.json" \
        --output_dir "$PROJ_DIR/outputs/eval/audios_en" \
        --cfg_value 2.0 --inference_timesteps 10 --max_len 600 \
        > "$PROJ_DIR/outputs/eval/.log_en_gpu${GPU}.txt" 2>&1 &
    PIDS+=($!)
    echo "GPU $GPU: EN worker (pid $!)"
done

echo "All 8 workers launched. Waiting..."
for pid in "${PIDS[@]}"; do wait $pid; done

# Cleanup
rm -f "$PROJ_DIR/outputs/eval/.s_"*.json

echo ""
echo "=== Done ==="
echo "ZH: $(ls "$PROJ_DIR/outputs/eval/audios_zh/"*.wav 2>/dev/null | wc -l) files"
echo "EN: $(ls "$PROJ_DIR/outputs/eval/audios_en/"*.wav 2>/dev/null | wc -l) files"
