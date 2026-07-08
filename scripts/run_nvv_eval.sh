#!/bin/bash
# ==============================================================================
# NVV-SuperBench Objective Evaluation Pipeline for VoxCPM
#
# Runs all 4 objective metrics:
#   1. WER/CER — intelligibility (Chinese CER via paraformer-zh)
#   2. DNSMOS — speech quality (SIG/BAK/OVRL)
#   3. CLAP Score — audio-text semantic alignment
#   4. NVV Precision/Recall/F1 — NVV controllability (requires Gemini API)
#
# Usage:
#   bash scripts/run_nvv_eval.sh [--full] [--skip-infer] [--max-samples N]
# ==============================================================================

set -euo pipefail

# --- Conda environment ---
source /mnt/node02_tmpdata0/miniconda3/etc/profile.d/conda.sh
conda activate voxcpm2

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

CKPT_DIR="/mnt/node01_tmpdata0/hujingbin/workspace/codes/VoxCPM2/ckpt/exp/checkpoints/finetune_all/step_0023000"
BENCH_DIR="/mnt/node01_tmpdata0/hujingbin/workspace/codes/Bench/NVV-SuperBench-main"
BENCHMARK_JSON="$BENCH_DIR/dataset/nvbench_data_zh.json"
MAPPING_JSON="$PROJ_DIR/docs/nvv_en2zh_mapping.json"
MAPPING_FWD_JSON="$PROJ_DIR/docs/nvv_label_mapping.json"

OUT_ROOT="$PROJ_DIR/outputs/nvv_eval"
PREPARED_JSON="$OUT_ROOT/prepared_data.json"
AUDIOS_DIR="$OUT_ROOT/audios"
RESULTS_DIR="$OUT_ROOT/results"

# GPU to use for inference
INFER_GPU=1
CFG_VALUE=2.0
INFERENCE_TIMESTEPS=10
MAX_LEN=600

# --- Flags ---
FULL_EVAL=false
SKIP_INFER=false
MAX_SAMPLES=5  # default: 5 per NVV type for quick test

while [[ $# -gt 0 ]]; do
    case $1 in
        --full) FULL_EVAL=true; shift ;;
        --skip-infer) SKIP_INFER=true; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_ROOT" "$AUDIOS_DIR" "$RESULTS_DIR"

echo "============================================================"
echo " NVV-SuperBench Objective Evaluation for VoxCPM"
echo "============================================================"
echo "  Full eval:       $FULL_EVAL"
echo "  Skip inference:  $SKIP_INFER"
echo "  Max per type:    $MAX_SAMPLES (0=all)"
echo "  Checkpoint:      $CKPT_DIR"
echo "  Output root:     $OUT_ROOT"
echo "============================================================"

# ============================================================
# Step 1: Prepare Data
# ============================================================
echo ""
echo ">>> Step 1: Preparing benchmark data..."

python3 "$SCRIPT_DIR/prepare_nvv_benchmark.py" \
    --benchmark-json "$BENCHMARK_JSON" \
    --mapping-json "$MAPPING_JSON" \
    --output-json "$PREPARED_JSON" \
    --max-samples-per-type "$MAX_SAMPLES"

NUM_SAMPLES=$(python3 -c "import json; print(len(json.load(open('$PREPARED_JSON'))))")
echo "    Prepared $NUM_SAMPLES samples"

# ============================================================
# Step 2: Batch Inference
# ============================================================
if [ "$SKIP_INFER" = false ]; then
    echo ""
    echo ">>> Step 2: Running batch inference..."

    CUDA_VISIBLE_DEVICES=$INFER_GPU python3 "$SCRIPT_DIR/batch_infer_nvv.py" \
        --ckpt_dir "$CKPT_DIR" \
        --input_json "$PREPARED_JSON" \
        --output_dir "$AUDIOS_DIR" \
        --cfg_value "$CFG_VALUE" \
        --inference_timesteps "$INFERENCE_TIMESTEPS" \
        --max_len "$MAX_LEN"

    NUM_GENERATED=$(ls "$AUDIOS_DIR"/*.wav 2>/dev/null | wc -l)
    echo "    Generated $NUM_GENERATED audio files"
else
    echo ""
    echo ">>> Step 2: Skipping inference (--skip-infer)"
    NUM_GENERATED=$(ls "$AUDIOS_DIR"/*.wav 2>/dev/null | wc -l)
    echo "    Found $NUM_GENERATED existing audio files"
fi

# ============================================================
# Step 3: Prepare reference JSON for WER/CER evaluation
# ============================================================
echo ""
echo ">>> Step 3: Preparing reference text JSON for CER evaluation..."

# The WER/CER eval needs {id, text} format with clean text (no tags)
REF_JSON="$OUT_ROOT/ref_texts_zh.json"
python3 -c "
import json
with open('$PREPARED_JSON') as f:
    data = json.load(f)
ref = [{'id': item['id'], 'text': item['text']} for item in data]
with open('$REF_JSON', 'w') as f:
    json.dump(ref, f, ensure_ascii=False, indent=2)
print(f'Written {len(ref)} entries to $REF_JSON')
"

# ============================================================
# Step 4: WER/CER Evaluation
# ============================================================
echo ""
echo ">>> Step 4: Running CER evaluation (Chinese, paraformer-zh)..."

cd "$BENCH_DIR/evaluation/objective_evaluation/wer_cer"

bash eval_wer.sh "$REF_JSON" "$AUDIOS_DIR" "zh" 1 2>&1 | tail -5

if [ -f "$AUDIOS_DIR/wav_res_ref_text.wer" ]; then
    echo ""
    echo "--- CER Result ---"
    grep "^CER:" "$AUDIOS_DIR/wav_res_ref_text.wer" 2>/dev/null || \
    grep "^WER:" "$AUDIOS_DIR/wav_res_ref_text.wer" 2>/dev/null || \
    tail -3 "$AUDIOS_DIR/wav_res_ref_text.wer"
    cp "$AUDIOS_DIR/wav_res_ref_text.wer" "$RESULTS_DIR/"
fi

# ============================================================
# Step 5: DNSMOS Evaluation
# ============================================================
echo ""
echo ">>> Step 5: Running DNSMOS P.835 evaluation..."

cd "$BENCH_DIR/evaluation/objective_evaluation/dnsmos"

bash eval_dnsmos.sh "$AUDIOS_DIR" 2>&1 | tail -10

if [ -f "$AUDIOS_DIR/dnsmos.csv" ]; then
    cp "$AUDIOS_DIR/dnsmos.csv" "$RESULTS_DIR/"
fi
if [ -f "$AUDIOS_DIR/dnsmos_summary.log" ]; then
    echo ""
    echo "--- DNSMOS Summary ---"
    cat "$AUDIOS_DIR/dnsmos_summary.log"
    cp "$AUDIOS_DIR/dnsmos_summary.log" "$RESULTS_DIR/"
fi

# ============================================================
# Step 6: CLAP Score Evaluation
# ============================================================
echo ""
echo ">>> Step 6: Running CLAP Score evaluation..."

# CLAP needs caption_with_nvb as the text key
cd "$BENCH_DIR/evaluation/objective_evaluation/clap_score"

# Prepare CLAP input: the prepared JSON already has caption_with_nvb
CLAP_INPUT="$OUT_ROOT/clap_input.json"
python3 -c "
import json
with open('$PREPARED_JSON') as f:
    data = json.load(f)
clap_data = [{'id': item['id'], 'text': item['text'], 'caption_with_nvb': item.get('caption_with_nvb', '')} for item in data]
with open('$CLAP_INPUT', 'w') as f:
    json.dump(clap_data, f, ensure_ascii=False, indent=2)
print(f'Written {len(clap_data)} entries')
"

CLAP_OUT="$RESULTS_DIR/clap_results.csv"
python3 eval_clap.py \
    --audios-dir "$AUDIOS_DIR" \
    --clap-texts-json "$CLAP_INPUT" \
    --clap-text-key "caption_with_nvb" \
    --out "$CLAP_OUT" \
    --device cuda \
    --gpu-ids "$INFER_GPU" \
    --run-clap \
    --checkpoint-every 20 \
    --auto-detect-id-suffixes 2>&1 | tail -10

if [ -f "$CLAP_OUT" ]; then
    echo ""
    echo "--- CLAP Score ---"
    grep "AVERAGE" "$CLAP_OUT" 2>/dev/null || tail -3 "$CLAP_OUT"
fi

# ============================================================
# Step 7: NVV Precision/Recall/F1 (requires Gemini API key)
# ============================================================
echo ""
echo ">>> Step 7: NVV Precision/Recall/F1..."
echo "    NOTE: This metric requires GEMINI_API_KEY."
echo "    Set export GEMINI_API_KEY=<key> and run manually:"
echo ""
echo "    cd $BENCH_DIR/evaluation/objective_evaluation/nvv_precision_recall_f1_distance"
echo "    export GEMINI_API_KEY=<your_key>"
echo "    export GT_ZH=$BENCHMARK_JSON"
echo "    # Edit run_predict.sh AUDIO_DIRS to point to: $AUDIOS_DIR"
echo "    bash run_predict.sh"
echo "    bash run_eval.sh"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo " Evaluation Complete!"
echo "============================================================"
echo "  Results directory: $RESULTS_DIR"
echo ""
echo "  Metrics collected:"
echo "    [✓] CER:      $RESULTS_DIR/wav_res_ref_text.wer"
echo "    [✓] DNSMOS:   $RESULTS_DIR/dnsmos.csv"
echo "    [✓] CLAP:     $RESULTS_DIR/clap_results.csv"
echo "    [ ] NVV F1:   (requires Gemini API key)"
echo ""
echo "  Audios: $AUDIOS_DIR ($NUM_GENERATED files)"
echo "============================================================"
