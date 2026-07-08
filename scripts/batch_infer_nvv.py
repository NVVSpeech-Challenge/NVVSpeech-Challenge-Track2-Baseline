#!/usr/bin/env python3
"""
Batch inference for NVV-SuperBench evaluation.

Reads prepared JSON, runs VoxCPM on each sample using text_with_mark_voxcpm,
saves audio files to output directory.
"""
import argparse
import json
import sys
import time
import os
from pathlib import Path

import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser("VoxCPM batch inference for NVV-SuperBench")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Checkpoint directory")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Prepared JSON from prepare_nvv_benchmark.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated audio files")
    parser.add_argument("--cfg_value", type=float, default=2.0)
    parser.add_argument("--inference_timesteps", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=600)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index (for resume)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End index (-1 = all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    with open(args.input_json) as f:
        data = json.load(f)

    total = len(data)
    end_idx = total if args.end_idx < 0 else min(args.end_idx, total)
    samples = data[args.start_idx:end_idx]
    print(f"Processing {len(samples)} samples ({args.start_idx}–{end_idx-1} of {total})")

    # Create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track already-done samples for resume
    existing = set()
    for f in out_dir.glob("*.wav"):
        existing.add(f.stem)

    # Import VoxCPM (lazy import to avoid loading when just checking)
    print(f"Loading model from {args.ckpt_dir}...")
    from voxcpm.core import VoxCPM
    model = VoxCPM.from_pretrained(
        hf_model_id=args.ckpt_dir,
        load_denoiser=False,
        optimize=False,
    )

    sample_rate = model.tts_model.sample_rate
    print(f"Model loaded. Sample rate: {sample_rate}")

    success = 0
    skipped = 0
    failed = 0
    total_duration = 0.0

    for i, item in enumerate(samples):
        sid = item["id"]
        out_path = out_dir / f"{sid}.wav"

        if sid in existing:
            skipped += 1
            continue

        text = item["text_with_mark_voxcpm"]
        nv_type = item["non_verbal_events"][0]

        try:
            t_start = time.time()
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
            elapsed = time.time() - t_start

            sf.write(str(out_path), audio_np, sample_rate)
            duration = len(audio_np) / sample_rate
            total_duration += duration
            success += 1

            eta = (elapsed / (i + 1)) * (len(samples) - i - 1)
            print(f"[{i+1}/{len(samples)}] {sid} | nvv={nv_type:<20} | "
                  f"dur={duration:.1f}s | time={elapsed:.1f}s | "
                  f"ETA={eta/60:.1f}min | text={text[:50]}...")

        except Exception as e:
            failed += 1
            print(f"[{i+1}/{len(samples)}] {sid} | FAILED: {e}")

    # Summary
    total_time = total_duration / 3600 if sample_rate else 0
    print(f"\n{'='*60}")
    print(f"Batch inference complete!")
    print(f"  Success: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"  Total audio duration: {total_duration/3600:.2f} hours")
    print(f"  Output dir: {out_dir}")


if __name__ == "__main__":
    main()
