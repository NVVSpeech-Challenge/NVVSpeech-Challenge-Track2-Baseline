#!/usr/bin/env python3
"""
Add non-verbal sound category tokens to VoxCPM vocabulary.

This script adds all non-verbal sound categories found in the training data
(e.g., [笑声], [呼吸], [咳嗽]) as special tokens to the model vocabulary.

Usage:
    python scripts/add_nonverbal_tokens.py \
        --pretrained_path /path/to/VoxCPM-0.5B \
        --output_path /path/to/VoxCPM-nonverbal

Copyright 2025 OpenBMB
Licensed under the Apache License, Version 2.0
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
import json
from typing import List, Set

# All non-verbal sound categories found in merged_train_data.jsonl
NONVERBAL_CATEGORIES = [
    "[笑声]",
    "[呼吸]",
    "[惊讶]",
    "[叹气]",
    "[咳嗽]",
    "[确认]",
    "[疑问]",
    "[迟疑]",
    "[吸鼻]",
    "[哭声]",
    "[清嗓]",
    "[不满]",
    "[倒吸气]",
    "[哈欠]",
    "[嘘声]",
    "[呻吟]",
    "[吹口哨]",
    "[打喷嚏]",
    "[咂嘴]",
    "[嘶声]",
    "[哼唱]",
    "[打鼾]",
    "[鼓掌]",
]


def load_existing_special_tokens(pretrained_path: str) -> Set[str]:
    """Load existing special tokens from tokenizer."""
    from transformers import LlamaTokenizerFast

    tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_path)
    existing_tokens = set(tokenizer.additional_special_tokens)
    print(f"Existing special tokens: {len(existing_tokens)}", file=sys.stderr)
    return existing_tokens


def get_new_tokens(
    categories: List[str],
    existing_tokens: Set[str],
    include_prefix: bool = True,
) -> List[str]:
    """Get list of new tokens to add.

    Args:
        categories: List of category strings (e.g., ["[笑声]", "[呼吸]"])
        existing_tokens: Set of already existing special tokens
        include_prefix: Whether to also add tokens with special prefix format

    Returns:
        List of new tokens to add
    """
    new_tokens = []

    for cat in categories:
        # Add the category as-is (e.g., "[笑声]")
        if cat not in existing_tokens:
            new_tokens.append(cat)

        # Optionally add a special prefix format (e.g., "<|笑声|>")
        if include_prefix:
            # Convert [笑声] to <|笑声|>
            prefix_token = "<|" + cat[1:-1] + "|>"
            if prefix_token not in existing_tokens and prefix_token not in new_tokens:
                new_tokens.append(prefix_token)

    return new_tokens


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

    print(f"\nAdding {len(tokens_to_add)} tokens:", file=sys.stderr)
    for token in tokens_to_add:
        print(f"  - {token}", file=sys.stderr)

    # Step 1: Load and update tokenizer
    print("\nLoading tokenizer...", file=sys.stderr)
    tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_path)
    original_vocab_size = len(tokenizer)

    # Add special tokens
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": tokens_to_add
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
    print(f"Updated config.json", file=sys.stderr)

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


def main():
    parser = argparse.ArgumentParser(
        description="Add non-verbal sound category tokens to VoxCPM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add all non-verbal categories
  python scripts/add_nonverbal_tokens.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-nonverbal

  # Add specific categories only
  python scripts/add_nonverbal_tokens.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-laugh \\
      --categories "[笑声]"

  # Add without prefix format
  python scripts/add_nonverbal_tokens.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-nonverbal \\
      --no_prefix

Available non-verbal categories:
  [笑声] [呼吸] [惊讶] [叹气] [咳嗽] [确认] [疑问] [迟疑]
  [吸鼻] [哭声] [清嗓] [不满] [倒吸气] [哈欠] [嘘声] [呻吟]
  [吹口哨] [打喷嚏] [咂嘴] [嘶声] [哼唱] [打鼾] [鼓掌]
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
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific categories to add (default: all)",
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

    args = parser.parse_args()

    # Determine categories to add
    if args.categories:
        # Validate categories
        invalid = [c for c in args.categories if c not in NONVERBAL_CATEGORIES]
        if invalid:
            print(f"Error: Invalid categories: {invalid}", file=sys.stderr)
            print(f"Valid categories: {NONVERBAL_CATEGORIES}", file=sys.stderr)
            sys.exit(1)
        categories = args.categories
    else:
        categories = NONVERBAL_CATEGORIES

    print(f"Categories to add: {len(categories)}", file=sys.stderr)

    # Load existing tokens
    existing_tokens = load_existing_special_tokens(args.pretrained_path)

    # Get new tokens
    new_tokens = get_new_tokens(
        categories,
        existing_tokens,
        include_prefix=not args.no_prefix,
    )

    if not new_tokens:
        print("All tokens already exist. Nothing to do.", file=sys.stderr)
        sys.exit(0)

    # Add tokens to model
    add_tokens_to_model(
        args.pretrained_path,
        args.output_path,
        new_tokens,
        args.device,
    )


if __name__ == "__main__":
    main()
