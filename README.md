# NVVSpeech Challenge @ ISCSLP 2026 — Track 2 Baseline

Official baseline for **Track 2: Controllable Speech Generation with Nonverbal Vocalizations** of the NVVSpeech Challenge @ ISCSLP 2026.

This model introduces 16 non-verbal (non-speech) tags during speech synthesis and generates the corresponding non-verbal vocalizations accordingly. For example, "[laugh] Tell me your name, please. Tell me your name." The model supports both Chinese and English.

---

## Quickstart

The fine-tuned baseline checkpoint is available on Hugging Face Hub: https://huggingface.co/NVVSpeech-Challenge/NVVSpeech-Challenge-Track2-Baseline

```bash
# 1. Download the fine-tuned baseline model from Hugging Face 
cd NVVSpeech-Challenge-Track2-Baseline
python -c "from huggingface_hub import snapshot_download; snapshot_download('NVVSpeech-Challenge/NVVSpeech-Challenge-Track2-Baseline', local_dir='./nv_baseline_ckpt')"

# 2. Install dependencies
pip install voxcpm

# 3. Run train 
bash finetune.sh

# 4. Run inference
bash infer.sh

```

---

## 1. Requirements

- Python **3.10+**
- PyTorch **2.12** (see `requirements.txt`)

---

## 2. Model

**VoxCPM2** (2B params), fully fine-tuned for 10 epochs using speech data annotated with 16 non-verbal tags, enabling the model to synthesize the corresponding non-verbal vocalizations.

| **Setting**                | **Value**                                             |
| -------------------------- | ----------------------------------------------------- |
| Base model                 | VoxCPM2 (2B)                                          |
| GPUs                       | 8 × NVIDIA A800 (80 GB)                               |
| Training strategy          | Full fine-tuning                                      |
| Training epochs            | 10                                                    |
| Training data              | Speech dataset annotated with 16 non-verbal (NV) tags |
| Optimizer                  | AdamW                                                 |
| Learning rate              | 1e-5                                      |
| Weight decay               | 0.01                                                  |
| Learning rate warmup       | 100 steps                                             |
| Batch size per GPU         | 1                                                     |
| Gradient accumulation      | 16                                                    |
| Effective batch size       | 128 (8 GPUs × 1 × 16)                                 |
| Gradient clipping          | 1.0                                                   |
| Maximum sequence length    | 4096 tokens                                           |
| Audio input sampling rate  | 16 kHz                                                |
| Audio output sampling rate | 48 kHz                                                |


The 16 canonical NVV tags:

```text
# English
[breath] [sniff] [laugh] [cry] [cough] [throat clearing]
[sneeze] [sigh] [gasp] [snore] [yawn] [hum] [moan] [hiss]
[lipsmack] [burp]

# 中文
[呼吸] [吸鼻] [笑声] [哭声] [咳嗽] [清嗓]
[打喷嚏] [叹气] [倒吸气] [打鼾] [哈欠] [哼唱] [呻吟] [嘶声]
[咂嘴] [不满]
```

---

## 3. File format (train dataset)

Files must be encoded in UTF-8 and use **JSON Lines (JSONL)** format: one JSON object per line.

```json
{"audio": "/path/to/audio.wav", "text": "So you can fit quite a bit of stuff in there and lay it out and have a lot of room for a lot of magic to happen. Man, I've got to get to work. [laugh]"}
```

| Field | Type | Description |
|---|---|---|
| `audio` | string | Absolute path to the audio file (`.wav`, `.mp3`, `.flac`, etc.) |
| `text` | string | Text with `[tag]` markers.|

---

## 4. Train

### LoRA fine-tuning (parameter-efficient, recommended)

### Single GPU

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train_voxcpm_finetune.py \  
--config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml
```

### Multi-GPU

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=29501 scripts/train_voxcpm_finetune.py \  
--config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml
```

### Full fine-tuning

### Single GPU

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train_voxcpm_finetune.py \
--config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml
```

### Multi-GPU

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=29501 scripts/train_voxcpm_finetune.py \
--config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml
```

---

## 5. Inference

The test labels are subject to the official version. Here, we use Chinese as an example.

### LoRA fine-tuning Infer

```bash
python scripts/test_voxcpm_lora_infer.py \
    --lora_ckpt nv_baseline_ckpt_lora \
    --text "[笑声]告诉我你名字吧，告诉我名字的话。" \
    --output outputs/output_laugh_lora.wav
```

### Full fine-tuning Infer

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir nv_baseline_ckpt \
    --text "[笑声]告诉我你名字吧，告诉我名字的话。" \
    --output outputs/output_laugh.wav
```

---

## 6. Baseline Results

Scores computed via Gemini 2.5 Pro LALM evaluation.

**Test method**: The baseline model is trained with Chinese NVV tags. All samples (both Chinese and English) are synthesized by mapping English NVV tags to their corresponding Chinese equivalents (e.g., `<laugh>` → `[笑声]`) before inference. The generated audio is then evaluated via the LALM-based protocol.

### Component Scores (1–5 scale, mean over all samples)

| Component | Weight | ZH | EN |
|---|---|---|---|
| NVV Accuracy (A) | 30% | 4.52 | 4.37 |
| NVV Perceptual Effect (P) | 25% | 4.41 | 3.90 |
| Overall Naturalness (N) | 15% | 3.73 | 3.37 |
| Overall Quality (Q) | 15% | 3.03 | 3.08 |
| Overall Expression (E) | 15% | 4.21 | 3.74 |

### Track 2 Score

$$\text{Track2Score} = 100 \times (0.30A + 0.25P + 0.15N + 0.15Q + 0.15E)$$

| Language | Track2Score |
|---|---|
| ZH | **77.62** |
| EN | **70.30** |
| **Bilingual Final** | **73.96** |
