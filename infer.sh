export CUDA_VISIBLE_DEVICES=0

python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir nv_baseline_ckpt \
    --text "[笑声]告诉我你名字吧，告诉我名字的话。" \
    --output outputs/output_laugh.wav