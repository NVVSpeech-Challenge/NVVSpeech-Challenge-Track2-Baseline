#!/bin/bash
# VoxCPM fine-tuning script for multi-GPU training

# LoRA fine-tuning (parameter-efficient, recommended)
# torchrun --nproc_per_node=8 --master_port=29501 scripts/train_voxcpm_finetune.py \
#     --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml

# Full fine-tuning 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=29501 scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml