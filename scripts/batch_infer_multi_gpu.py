#!/usr/bin/env python3
"""Multi-GPU batch inference using multiprocessing — one process per GPU."""
import argparse, json, os, sys, time
from pathlib import Path
from multiprocessing import Process

import soundfile as sf


def worker(gpu_id: int, shard_data: list, output_dir: str, args):
    """Run inference on one GPU with a shard of data."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TORCHDYNAMO_DISABLE'] = '1'

    # Import torch & model AFTER setting CUDA_VISIBLE_DEVICES
    import torch
    from voxcpm.core import VoxCPM

    print(f"[GPU {gpu_id}] Loading model from {args.ckpt_dir}...")
    model = VoxCPM.from_pretrained(
        hf_model_id=args.ckpt_dir,
        load_denoiser=False,
        optimize=False,
    )
    sample_rate = model.tts_model.sample_rate
    device = torch.cuda.current_device()
    print(f"[GPU {gpu_id}] Model loaded on cuda:{device}, sr={sample_rate}, samples={len(shard_data)}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track existing for resume
    existing = {f.stem for f in out_dir.glob('*.wav')}

    success = skipped = failed = 0
    t0 = time.time()

    for i, item in enumerate(shard_data):
        sid = item['id']
        out_path = out_dir / f'{sid}.wav'

        if sid in existing:
            skipped += 1
            continue

        text = item['text_with_mark_voxcpm']

        try:
            audio_np = model.generate(
                text=text,
                prompt_wav_path=None,
                prompt_text=None,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                max_len=args.max_len,
                normalize=args.normalize,
                denoise=False,
            )
            sf.write(str(out_path), audio_np, sample_rate)
            success += 1

        except Exception as e:
            failed += 1
            print(f"[GPU {gpu_id}] [{i+1}/{len(shard_data)}] {sid} FAILED: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (len(shard_data) - i - 1)
            print(f"[GPU {gpu_id}] [{i+1}/{len(shard_data)}] ok={success} skip={skipped} fail={failed} | "
                  f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"[GPU {gpu_id}] DONE: {success} ok, {skipped} skip, {failed} fail | {elapsed/60:.1f}min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--cfg_value', type=float, default=2.0)
    parser.add_argument('--inference_timesteps', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=600)
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    total = len(data)
    n_gpu = args.num_gpus
    shard_size = (total + n_gpu - 1) // n_gpu

    print(f"Total: {total} samples, {n_gpu} GPUs, ~{shard_size} per GPU")

    processes = []
    for i in range(n_gpu):
        shard = data[i * shard_size:(i + 1) * shard_size]
        if not shard:
            continue
        p = Process(target=worker, args=(i, shard, args.output_dir, args))
        p.start()
        processes.append(p)
        print(f"  GPU {i}: {len(shard)} samples")

    for p in processes:
        p.join()

    # Count results
    out_dir = Path(args.output_dir)
    done = len(list(out_dir.glob('*.wav')))
    print(f"\n{'='*60}")
    print(f"All done! {done}/{total} files generated in {args.output_dir}")


if __name__ == '__main__':
    main()
