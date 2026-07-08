#!/usr/bin/env python3
"""
Expand VoxCPM vocabulary with custom special tokens.

This script expands the tokenizer vocabulary by adding special tokens
(e.g., emotion tags, speaker tags) and/or regular vocabulary tokens,
then resizes the model's embedding layer accordingly.

Usage:
    python scripts/expand_vocabulary.py \
        --pretrained_path /path/to/VoxCPM-0.5B \
        --output_path /path/to/VoxCPM-expanded \
        --special_tokens "<|emotion_happy|>" "<|emotion_sad|>" "<|speaker_start|>" \
        --regular_tokens "新词1" "新词2"

Copyright 2025 OpenBMB
Licensed under the Apache License, Version 2.0
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
import json
import shutil
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import LlamaTokenizerFast

from voxcpm.model import VoxCPMModel, VoxCPM2Model


def load_tokenizer(pretrained_path: str) -> LlamaTokenizerFast:
    """Load tokenizer from pretrained model path."""
    print(f"Loading tokenizer from: {pretrained_path}", file=sys.stderr)
    tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_path)
    print(f"Original vocabulary size: {len(tokenizer)}", file=sys.stderr)
    return tokenizer


def add_special_tokens(
    tokenizer: LlamaTokenizerFast,
    special_tokens: List[str],
) -> int:
    """Add special tokens to tokenizer.

    Args:
        tokenizer: The tokenizer to modify
        special_tokens: List of special token strings (e.g., ["<|emotion_happy|>", "<|speaker_start|>"])

    Returns:
        Number of tokens actually added (may be less if some already exist)
    """
    if not special_tokens:
        return 0

    # Filter out tokens that already exist
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in special_tokens if t not in existing_tokens]

    if not new_tokens:
        print("All special tokens already exist in vocabulary", file=sys.stderr)
        return 0

    print(f"Adding {len(new_tokens)} special tokens: {new_tokens}", file=sys.stderr)

    # Use add_special_tokens for tokens with special semantics
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": new_tokens
    })

    print(f"Successfully added {num_added} special tokens", file=sys.stderr)
    return num_added


def add_regular_tokens(
    tokenizer: LlamaTokenizerFast,
    regular_tokens: List[str],
) -> int:
    """Add regular vocabulary tokens to tokenizer.

    Args:
        tokenizer: The tokenizer to modify
        regular_tokens: List of token strings to add

    Returns:
        Number of tokens actually added
    """
    if not regular_tokens:
        return 0

    # Filter out tokens that already exist
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in regular_tokens if t not in existing_tokens]

    if not new_tokens:
        print("All regular tokens already exist in vocabulary", file=sys.stderr)
        return 0

    print(f"Adding {len(new_tokens)} regular tokens: {new_tokens}", file=sys.stderr)

    num_added = tokenizer.add_tokens(new_tokens)

    print(f"Successfully added {num_added} regular tokens", file=sys.stderr)
    return num_added


def save_tokenizer(tokenizer: LlamaTokenizerFast, output_path: str):
    """Save updated tokenizer to output directory."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenizer to: {output_dir}", file=sys.stderr)
    tokenizer.save_pretrained(output_dir)

    # Also save special_tokens_map.json if it exists
    special_tokens_map = {
        "additional_special_tokens": list(tokenizer.additional_special_tokens)
    }
    with open(output_dir / "custom_special_tokens.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)


def update_model_config(
    pretrained_path: str,
    output_path: str,
    new_vocab_size: int,
):
    """Update model config.json with new vocabulary size.

    Args:
        pretrained_path: Path to original pretrained model
        output_path: Path to save updated config
        new_vocab_size: New vocabulary size
    """
    config_path = Path(pretrained_path) / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Update vocab_size in lm_config
    if "lm_config" in config:
        old_vocab_size = config["lm_config"].get("vocab_size", 0)
        config["lm_config"]["vocab_size"] = new_vocab_size
        print(f"Updated lm_config.vocab_size: {old_vocab_size} -> {new_vocab_size}", file=sys.stderr)
    else:
        print("Warning: lm_config not found in config.json", file=sys.stderr)

    # Save updated config
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Saved updated config to: {output_dir / 'config.json'}", file=sys.stderr)


def resize_token_embeddings(embedding: nn.Embedding, new_num_embeddings: int) -> nn.Embedding:
    """Resize an embedding layer to a new number of embeddings.

    Args:
        embedding: Original embedding layer
        new_num_embeddings: New number of embeddings

    Returns:
        Resized embedding layer
    """
    old_num_embeddings, embedding_dim = embedding.weight.shape

    if new_num_embeddings == old_num_embeddings:
        return embedding

    # Create new embedding layer
    new_embedding = nn.Embedding(new_num_embeddings, embedding_dim)
    new_embedding = new_embedding.to(dtype=embedding.weight.dtype, device=embedding.weight.device)

    # Copy existing embeddings
    num_to_copy = min(old_num_embeddings, new_num_embeddings)
    with torch.no_grad():
        new_embedding.weight[:num_to_copy] = embedding.weight[:num_to_copy]

    # Initialize new embeddings with normal distribution
    if new_num_embeddings > old_num_embeddings:
        with torch.no_grad():
            std = 0.02
            new_embedding.weight[old_num_embeddings:].normal_(mean=0.0, std=std)

    print(f"Resized embedding: {old_num_embeddings} -> {new_num_embeddings} "
          f"(copied {num_to_copy}, initialized {new_num_embeddings - old_num_embeddings})", file=sys.stderr)

    return new_embedding


def resize_model_embeddings(
    pretrained_path: str,
    new_vocab_size: int,
    device: str = "cpu",
) -> dict:
    """Load model and resize embedding layers.

    Args:
        pretrained_path: Path to pretrained model
        new_vocab_size: New vocabulary size
        device: Device to load model on

    Returns:
        Updated model state dict (excluding audio_vae)
    """
    print(f"Loading model from: {pretrained_path}", file=sys.stderr)

    # Detect architecture
    with open(Path(pretrained_path) / "config.json", "r", encoding="utf-8") as f:
        arch = json.load(f).get("architecture", "voxcpm").lower()

    model_cls = VoxCPM2Model if arch == "voxcpm2" else VoxCPMModel
    print(f"Detected architecture: {arch} -> {model_cls.__name__}", file=sys.stderr)

    # Load model
    model = model_cls.from_local(
        pretrained_path,
        optimize=False,
        training=False,
        device=device,
    )

    # Resize embeddings
    print(f"Resizing embeddings to: {new_vocab_size}", file=sys.stderr)
    model.base_lm.embed_tokens = resize_token_embeddings(
        model.base_lm.embed_tokens,
        new_vocab_size
    )

    # Update config
    model.config.lm_config.vocab_size = new_vocab_size
    model.base_lm.vocab_size = new_vocab_size

    # Get state dict (exclude audio_vae)
    state_dict = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("audio_vae.")
    }

    return state_dict


def save_model_weights(state_dict: dict, output_path: str):
    """Save model weights to output directory.

    Args:
        state_dict: Model state dict to save
        output_path: Directory to save weights
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file
        save_path = output_dir / "model.safetensors"
        save_file(state_dict, save_path)
        print(f"Saved model weights to: {save_path} (safetensors format)", file=sys.stderr)
    except ImportError:
        save_path = output_dir / "pytorch_model.bin"
        torch.save({"state_dict": state_dict}, save_path)
        print(f"Saved model weights to: {save_path} (pytorch format)", file=sys.stderr)


def copy_audio_vae(pretrained_path: str, output_path: str):
    """Copy AudioVAE weights from pretrained model to output directory."""
    output_dir = Path(output_path)
    pretrained_dir = Path(pretrained_path)

    vae_files = ["audiovae.pth", "audiovae.safetensors"]
    for fname in vae_files:
        src = pretrained_dir / fname
        if src.exists():
            dst = output_dir / fname
            shutil.copy2(src, dst)
            print(f"Copied {fname} to: {dst}", file=sys.stderr)


def expand_vocabulary(
    pretrained_path: str,
    output_path: str,
    special_tokens: Optional[List[str]] = None,
    regular_tokens: Optional[List[str]] = None,
    device: str = "cpu",
):
    """Main function to expand vocabulary and save updated model.

    Args:
        pretrained_path: Path to original pretrained model
        output_path: Path to save expanded model
        special_tokens: List of special tokens to add
        regular_tokens: List of regular tokens to add
        device: Device to load model on
    """
    if not special_tokens and not regular_tokens:
        print("Error: No tokens to add. Use --special_tokens or --regular_tokens", file=sys.stderr)
        sys.exit(1)

    # Step 1: Load and update tokenizer
    tokenizer = load_tokenizer(pretrained_path)
    original_vocab_size = len(tokenizer)

    num_special = add_special_tokens(tokenizer, special_tokens or [])
    num_regular = add_regular_tokens(tokenizer, regular_tokens or [])
    new_vocab_size = len(tokenizer)

    if new_vocab_size == original_vocab_size:
        print("No new tokens were added. Exiting.", file=sys.stderr)
        sys.exit(0)

    print(f"\nVocabulary expansion: {original_vocab_size} -> {new_vocab_size} "
          f"(+{new_vocab_size - original_vocab_size})", file=sys.stderr)
    print(f"  - Special tokens added: {num_special}", file=sys.stderr)
    print(f"  - Regular tokens added: {num_regular}", file=sys.stderr)

    # Step 2: Save updated tokenizer
    save_tokenizer(tokenizer, output_path)

    # Step 3: Update model config
    update_model_config(pretrained_path, output_path, new_vocab_size)

    # Step 4: Resize model embeddings and save weights
    state_dict = resize_model_embeddings(pretrained_path, new_vocab_size, device)
    save_model_weights(state_dict, output_path)

    # Step 5: Copy AudioVAE weights
    copy_audio_vae(pretrained_path, output_path)

    print(f"\nExpansion complete! Expanded model saved to: {output_path}", file=sys.stderr)
    print(f"\nTo use the expanded model, update your training config:", file=sys.stderr)
    print(f"  pretrained_path: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Expand VoxCPM vocabulary with custom tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add emotion control tokens
  python scripts/expand_vocabulary.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-expanded \\
      --special_tokens "<|emotion_happy|>" "<|emotion_sad|>" "<|emotion_angry|>"

  # Add speaker tokens
  python scripts/expand_vocabulary.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-expanded \\
      --special_tokens "<|speaker_start|>" "<|speaker_end|>"

  # Add both special and regular tokens
  python scripts/expand_vocabulary.py \\
      --pretrained_path /path/to/VoxCPM-0.5B \\
      --output_path /path/to/VoxCPM-expanded \\
      --special_tokens "<|emotion_happy|>" "<|emotion_sad|>" \\
      --regular_tokens "自定义词1" "自定义词2"
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
        "--special_tokens",
        type=str,
        nargs="+",
        default=[],
        help="Special tokens to add (e.g., '<|emotion_happy|>' '<|speaker_start|>')",
    )
    parser.add_argument(
        "--regular_tokens",
        type=str,
        nargs="+",
        default=[],
        help="Regular vocabulary tokens to add",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (default: cpu)",
    )

    args = parser.parse_args()

    expand_vocabulary(
        pretrained_path=args.pretrained_path,
        output_path=args.output_path,
        special_tokens=args.special_tokens,
        regular_tokens=args.regular_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
