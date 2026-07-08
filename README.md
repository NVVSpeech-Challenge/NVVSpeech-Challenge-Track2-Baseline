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

**Test method**: The baseline model is trained with Chinese NVV tags. All samples (both Chinese and English) are synthesized by mapping English NVV tags to their corresponding Chinese equivalents (e.g., `<laugh>` → `[笑声]`) before inference. The generated audio is then evaluated via the official LALM-based protocol.

### Component Scores (1–5 scale, mean over all samples)

| Component | Weight | ZH | EN |
|---|---|---|---|
| NVV Accuracy (A) | 30% | 4.59 | 4.38 |
| NVV Perceptual Effect (P) | 25% | 4.50 | 3.88 |
| Overall Naturalness (N) | 15% | 3.87 | 3.42 |
| Overall Quality (Q) | 15% | 3.03 | 3.14 |
| Overall Expression (E) | 15% | 4.30 | 3.75 |

### Track 2 Score

$$\text{Track2Score} = 100 \times (0.30A + 0.25P + 0.15N + 0.15Q + 0.15E)$$

| Language | Track2Score |
|---|---|
| ZH | **79.56** |
| EN | **70.78** |
| **Bilingual Final** | **75.17** |

### Per-Tag Breakdown (ZH — 50 samples each)

| NVV Tag | A | P | N | Q | E | Score |
|---|---|---|---|---|---|---|
| [哈欠] yawn | 4.96 | 4.94 | 4.22 | 3.14 | 4.80 | 88.67 |
| [叹气] sigh | 5.00 | 4.94 | 4.18 | 3.14 | 4.62 | 88.15 |
| [呻吟] moan | 4.92 | 4.76 | 4.14 | 2.98 | 4.66 | 85.82 |
| [嘶声] hiss | 4.86 | 4.82 | 3.98 | 2.84 | 4.66 | 84.62 |
| [咂嘴] lipsmack | 4.80 | 4.90 | 3.92 | 3.08 | 4.30 | 84.00 |
| [吸鼻] sniff | 4.82 | 4.72 | 3.86 | 3.00 | 4.32 | 82.58 |
| [呼吸] breath | 4.96 | 4.72 | 3.66 | 2.90 | 4.24 | 82.20 |
| [打鼾] snore | 4.60 | 4.46 | 3.96 | 3.08 | 4.38 | 80.20 |
| [清嗓] throat clearing | 4.76 | 4.68 | 3.90 | 2.90 | 3.92 | 80.15 |
| [哭声] cry | 4.66 | 4.26 | 3.72 | 3.02 | 4.46 | 78.58 |
| [哼唱] hum | 4.46 | 4.44 | 3.86 | 3.10 | 3.94 | 77.07 |
| [咳嗽] cough | 4.28 | 4.42 | 3.74 | 3.04 | 4.06 | 75.38 |
| [惊讶] gasp | 3.98 | 4.28 | 3.64 | 3.18 | 4.36 | 73.52 |
| [打喷嚏] sneeze | 4.38 | 3.90 | 3.82 | 3.00 | 4.12 | 73.25 |
| [笑声] laugh | 4.34 | 3.84 | 3.66 | 3.18 | 3.98 | 72.12 |
| [不满] burp | 3.60 | 3.94 | 3.70 | 2.94 | 4.04 | 66.68 |

### Per-Tag Breakdown (EN — 50 samples each)

| NVV Tag | A | P | N | Q | E | Score |
|---|---|---|---|---|---|---|
| [哈欠] yawn | 5.00 | 4.42 | 3.58 | 3.18 | 3.96 | 80.32 |
| [咂嘴] lipsmack | 4.70 | 4.42 | 3.58 | 3.16 | 4.00 | 78.15 |
| [呼吸] breath | 4.68 | 4.58 | 3.52 | 3.08 | 3.90 | 78.10 |
| [叹气] sigh | 4.76 | 4.38 | 3.58 | 3.06 | 3.94 | 77.75 |
| [嘶声] hiss | 4.74 | 4.36 | 3.48 | 3.18 | 3.92 | 77.48 |
| [哼唱] hum | 4.48 | 4.26 | 3.72 | 3.14 | 3.84 | 75.35 |
| [呻吟] moan | 4.60 | 4.08 | 3.50 | 3.20 | 4.00 | 75.12 |
| [吸鼻] sniff | 4.40 | 4.30 | 3.52 | 3.14 | 3.72 | 73.80 |
| [打喷嚏] sneeze | 4.68 | 3.58 | 3.42 | 3.18 | 3.74 | 71.25 |
| [惊讶] gasp | 4.08 | 3.90 | 3.32 | 3.24 | 3.60 | 68.08 |
| [打鼾] snore | 4.12 | 3.48 | 3.44 | 3.16 | 3.74 | 66.42 |
| [不满] burp | 3.50 | 2.90 | 3.28 | 3.14 | 3.46 | 56.43 |
| [咳嗽] cough | 4.22 | 3.46 | 3.30 | 3.16 | 3.66 | 66.22 |
| [清嗓] throat clearing | 4.28 | 3.58 | 2.98 | 2.88 | 3.22 | 63.52 |
| [笑声] laugh | 4.00 | 3.20 | 3.24 | 3.12 | 3.68 | 62.65 |
| [哭声] cry | 3.90 | 3.26 | 3.18 | 3.16 | 3.58 | 61.82 |

> **Note**: NVV tags are mapped to Chinese (e.g., `<laugh>` → `[笑声]`) before synthesis, as the baseline model is trained with Chinese NVV tags. Results obtained using the VoxCPM2 baseline model (full fine-tuning). The LALM evaluation follows the official Track 2 protocol with Gemini 2.5 Pro as the judge. For the final leaderboard, top-ranked systems will additionally undergo human listening tests.
