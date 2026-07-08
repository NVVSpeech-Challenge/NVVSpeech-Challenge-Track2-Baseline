#!/usr/bin/env python3
"""
Add ALL non-verbal sound category tokens to VoxCPM vocabulary.

This script analyzes the training data and adds ALL categories (including
combined categories like [呼吸],[笑声]) as special tokens.

Usage:
    python scripts/add_all_category_tokens.py \
        --pretrained_path /path/to/VoxCPM-0.5B \
        --output_path /path/to/VoxCPM-all-categories \
        --data_path /path/to/merged_train_data.jsonl

Copyright 2025 OpenBMB
Licensed under the Apache License, Version 2.0
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
import json
from collections import Counter
from typing import List, Dict, Tuple


def analyze_categories(data_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Analyze all categories from training data.

    Args:
        data_path: Path to JSONL training data

    Returns:
        Tuple of (single_categories, combined_categories) with counts
    """
    categories = Counter()

    print(f"Reading data from: {data_path}", file=sys.stderr)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                cat = data.get('category', '')
                if cat:
                    categories[cat] += 1
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON at line {line_num}", file=sys.stderr)
                continue

            if line_num % 50000 == 0:
                print(f"  Processed {line_num:,} lines...", file=sys.stderr)

    print(f"Total lines processed: {line_num:,}", file=sys.stderr)
    print(f"Total categories found: {len(categories):,}", file=sys.stderr)

    # Separate single and combined categories
    single_cats = {k: v for k, v in categories.items() if ',' not in k}
    combo_cats = {k: v for k, v in categories.items() if ',' in k}

    print(f"  - Single categories: {len(single_cats)}", file=sys.stderr)
    print(f"  - Combined categories: {len(combo_cats):,}", file=sys.stderr)

    return single_cats, combo_cats


def generate_token_list(
    single_cats: Dict[str, int],
    combo_cats: Dict[str, int],
    min_count: int = 1,
    include_single: bool = True,
    include_combo: bool = True,
    add_prefix_format: bool = True,
) -> List[str]:
    """Generate list of tokens to add.

    Args:
        single_cats: Single categories with counts
        combo_cats: Combined categories with counts
        min_count: Minimum count to include a category
        include_single: Whether to include single categories
        include_combo: Whether to include combined categories
        add_prefix_format: Whether to also add prefix format tokens

    Returns:
        List of tokens to add
    """
    tokens = []

    # Add single categories
    if include_single:
        for cat, count in sorted(single_cats.items(), key=lambda x: -x[1]):
            if count >= min_count:
                tokens.append(cat)
                if add_prefix_format:
                    # [笑声] -> <|笑声|>
                    prefix = "<|" + cat[1:-1] + "|>"
                    tokens.append(prefix)

    # Add combined categories
    if include_combo:
        for cat, count in sorted(combo_cats.items(), key=lambda x: -x[1]):
            if count >= min_count:
                tokens.append(cat)
                if add_prefix_format:
                    # [呼吸],[笑声] -> <|呼吸|笑声|>
                    # Extract content between brackets
                    parts = cat.replace('[', '').replace(']', '').split(',')
                    prefix = "<|" + "|".join(parts) + "|>"
                    tokens.append(prefix)

    return tokens


def save_category_report(
    single_cats: Dict[str, int],
    combo_cats: Dict[str, int],
    output_path: str,
    min_count: int = 1,
):
    """Save category analysis report to file.

    Args:
        single_cats: Single categories with counts
        combo_cats: Combined categories with counts
        output_path: Path to save report
        min_count: Minimum count filter used
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "category_analysis.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VoxCPM 非语言声音类别分析报告\n")
        f.write("=" * 80 + "\n\n")

        total_single = sum(single_cats.values())
        total_combo = sum(combo_cats.values())

        f.write(f"总数据量: {total_single + total_combo:,} 条\n")
        f.write(f"类别总数: {len(single_cats) + len(combo_cats):,} 个\n")
        f.write(f"  - 单独类别: {len(single_cats)} 个 ({total_single:,} 条)\n")
        f.write(f"  - 组合类别: {len(combo_cats):,} 个 ({total_combo:,} 条)\n")
        f.write(f"最小出现次数过滤: {min_count}\n\n")

        # Single categories
        f.write("=" * 80 + "\n")
        f.write("一、单独类别\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'类别':<15} {'数据量':>10} {'占比':>10}\n")
        f.write("-" * 40 + "\n")
        for cat, count in sorted(single_cats.items(), key=lambda x: -x[1]):
            if count >= min_count:
                pct = count / total_single * 100
                f.write(f"{cat:<15} {count:>10,} {pct:>9.1f}%\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'合计':<15} {total_single:>10,}\n\n")

        # Combined categories
        f.write("=" * 80 + "\n")
        f.write(f"二、组合类别 (共 {len(combo_cats):,} 个，显示前 100)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'类别':<35} {'数据量':>10}\n")
        f.write("-" * 50 + "\n")
        for i, (cat, count) in enumerate(sorted(combo_cats.items(), key=lambda x: -x[1])):
            if count >= min_count and i < 100:
                f.write(f"{cat:<35} {count:>10,}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'组合类别合计':<35} {total_combo:>10,}\n")

    print(f"Category report saved to: {report_path}", file=sys.stderr)


def add_tokens_to_model(
    pretrained_path: str,
    output_path: str,
    tokens_to_add: List[str],
    device: str = "cpu",
):
    """Add tokens to model and save.

    Args:
        pretrained_path: Path to pretrained model
        output_path: Path to save expanded model
        tokens_to_add: List of tokens to add
        device: Device to load model on
    """
    import torch
    import torch.nn as nn
    from transformers import LlamaTokenizerFast
    from voxcpm.model import VoxCPMModel, VoxCPM2Model

    if not tokens_to_add:
        print("No new tokens to add.", file=sys.stderr)
        return

    print(f"\nTotal tokens to add: {len(tokens_to_add)}", file=sys.stderr)

    # Step 1: Load and update tokenizer
    print("Loading tokenizer...", file=sys.stderr)
    tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_path)
    original_vocab_size = len(tokenizer)

    # Filter out existing tokens
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in tokens_to_add if t not in existing_tokens]

    if not new_tokens:
        print("All tokens already exist. Nothing to do.", file=sys.stderr)
        return

    print(f"New tokens to add: {len(new_tokens)}", file=sys.stderr)

    # Add special tokens
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": new_tokens
    })
    new_vocab_size = len(tokenizer)

    print(f"Vocabulary: {original_vocab_size} -> {new_vocab_size} (+{num_added})", file=sys.stderr)

    # Step 2: Save tokenizer
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved tokenizer to: {output_dir}", file=sys.stderr)

    # Step 3: Update config
    with open(Path(pretrained_path) / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    config["lm_config"]["vocab_size"] = new_vocab_size
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("Updated config.json", file=sys.stderr)

    # Step 4: Load model and resize embeddings
    print("Loading model...", file=sys.stderr)
    with open(Path(pretrained_path) / "config.json", "r", encoding="utf-8") as f:
        arch = json.load(f).get("architecture", "voxcpm").lower()

    model_cls = VoxCPM2Model if arch == "voxcpm2" else VoxCPMModel
    model = model_cls.from_local(
        pretrained_path,
        optimize=False,
        training=False,
        device=device,
    )

    # Resize embeddings
    print(f"Resizing embeddings to {new_vocab_size}...", file=sys.stderr)
    old_embedding = model.base_lm.embed_tokens
    old_num, embed_dim = old_embedding.weight.shape

    new_embedding = nn.Embedding(new_vocab_size, embed_dim)
    new_embedding = new_embedding.to(dtype=old_embedding.weight.dtype, device=old_embedding.weight.device)

    with torch.no_grad():
        new_embedding.weight[:old_num] = old_embedding.weight[:old_num]
        # Initialize new tokens with normal distribution
        if new_vocab_size > old_num:
            new_embedding.weight[old_num:].normal_(mean=0.0, std=0.02)

    model.base_lm.embed_tokens = new_embedding
    model.config.lm_config.vocab_size = new_vocab_size
    model.base_lm.vocab_size = new_vocab_size

    # Step 5: Save model weights
    print("Saving model weights...", file=sys.stderr)
    state_dict = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("audio_vae.")
    }

    try:
        from safetensors.torch import save_file
        save_file(state_dict, output_dir / "model.safetensors")
        print("Saved model.safetensors", file=sys.stderr)
    except ImportError:
        torch.save({"state_dict": state_dict}, output_dir / "pytorch_model.bin")
        print("Saved pytorch_model.bin", file=sys.stderr)

    # Step 6: Copy AudioVAE
    import shutil
    for fname in ["audiovae.pth", "audiovae.safetensors"]:
        src = Path(pretrained_path) / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}", file=sys.stderr)

    print(f"\nDone! Expanded model saved to: {output_path}", file=sys.stderr)
    print(f"Final vocabulary size: {new_vocab_size}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Add ALL non-verbal sound category tokens to VoxCPM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add all categories (single + combined)
  python scripts/add_all_category_tokens.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-all-categories \\
      --data_path dataset/merged_train_data.jsonl

  # Add only categories with >= 10 occurrences
  python scripts/add_all_category_tokens.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-categories \\
      --data_path dataset/merged_train_data.jsonl \\
      --min_count 10

  # Add only single categories (no combined)
  python scripts/add_all_category_tokens.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-single \\
      --data_path dataset/merged_train_data.jsonl \\
      --no_combo

  # Add without prefix format (only [笑声], not <|笑声|>)
  python scripts/add_all_category_tokens.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-categories \\
      --data_path dataset/merged_train_data.jsonl \\
      --no_prefix
        """,
    )

    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to pretrained VoxCPM model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save expanded model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSONL training data",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Minimum count to include a category (default: 1)",
    )
    parser.add_argument(
        "--no_combo",
        action="store_true",
        help="Don't add combined categories",
    )
    parser.add_argument(
        "--no_prefix",
        action="store_true",
        help="Don't add prefix format tokens (e.g., <|笑声|>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (default: cpu)",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only generate report, don't add tokens",
    )

    args = parser.parse_args()

    # Analyze categories
    print("=" * 60, file=sys.stderr)
    print("Analyzing categories from training data...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    single_cats, combo_cats = analyze_categories(args.data_path)

    # Save report
    save_category_report(single_cats, combo_cats, args.output_path, args.min_count)

    if args.report_only:
        print("\nReport only mode. Exiting.", file=sys.stderr)
        return

    # Generate token list
    print("\n" + "=" * 60, file=sys.stderr)
    print("Generating token list...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    tokens = generate_token_list(
        single_cats,
        combo_cats,
        min_count=args.min_count,
        include_single=True,
        include_combo=not args.no_combo,
        add_prefix_format=not args.no_prefix,
    )

    # Add tokens to model
    print("\n" + "=" * 60, file=sys.stderr)
    print("Adding tokens to model...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    add_tokens_to_model(
        args.pretrained_path,
        args.output_path,
        tokens,
        args.device,
    )


if __name__ == "__main__":
    main()
