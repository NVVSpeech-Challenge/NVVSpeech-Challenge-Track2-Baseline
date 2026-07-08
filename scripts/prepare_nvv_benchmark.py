#!/usr/bin/env python3
"""
Prepare NVV-SuperBench data for VoxCPM inference.

Converts NVV-SuperBench text_with_mark (English tags like <laugh>)
to VoxCPM format (Chinese tags like [笑声]) using the reverse mapping.
"""
import json
import re
import argparse
from pathlib import Path
from collections import Counter


def load_mapping(mapping_path):
    with open(mapping_path) as f:
        return json.load(f)


def convert_tags(text_with_mark, en2zh):
    """Convert <english_tag> to [中文标签] for VoxCPM."""
    # Replace <tag> with [中文]
    def replacer(match):
        en_tag = match.group(1).strip()
        zh_tag = en2zh.get(en_tag)
        if zh_tag:
            return f"[{zh_tag}]"
        else:
            print(f"  Warning: No mapping for tag '{en_tag}', keeping as-is")
            return match.group(0)

    return re.sub(r'<([^>]+)>', replacer, text_with_mark)


def main():
    parser = argparse.ArgumentParser(description="Prepare NVV-SuperBench data for VoxCPM inference")
    parser.add_argument("--benchmark-json", required=True, help="Path to nvbench_data_zh.json")
    parser.add_argument("--mapping-json", required=True, help="Path to nvv_en2zh_mapping.json")
    parser.add_argument("--output-json", required=True, help="Output JSON for VoxCPM inference")
    parser.add_argument("--max-samples-per-type", type=int, default=0,
                        help="Max samples per NVV type (0=all, for quick testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    en2zh = load_mapping(args.mapping_json)

    with open(args.benchmark_json) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from benchmark")

    # Convert each sample
    results = []
    tag_stats = Counter()
    unmapped_tags = set()

    for item in data:
        original_mark = item["text_with_mark"]
        voxcpm_text = convert_tags(original_mark, en2zh)

        # Track stats
        tags = re.findall(r'<([^>]+)>', original_mark)
        for t in tags:
            tag_stats[t] += 1
            if t not in en2zh:
                unmapped_tags.add(t)

        results.append({
            "id": item["id"],
            "text": item["text"],
            "text_with_mark_original": original_mark,
            "text_with_mark_voxcpm": voxcpm_text,
            "non_verbal_events": item["non_verbal_events"],
            "caption_with_nvb": item.get("caption_with_nvb", ""),
        })

    if unmapped_tags:
        print(f"\n⚠ Unmapped tags found: {unmapped_tags}")

    # Sample if needed
    if args.max_samples_per_type > 0:
        import random
        random.seed(args.seed)
        # Group by first NVV type
        by_type = {}
        for r in results:
            nv_type = r["non_verbal_events"][0]
            by_type.setdefault(nv_type, []).append(r)

        sampled = []
        for nv_type, items in sorted(by_type.items()):
            n = min(args.max_samples_per_type, len(items))
            sampled.extend(random.sample(items, n))
            print(f"  {nv_type}: {n}/{len(items)} samples")

        results = sampled
        print(f"Sampled {len(results)} total samples")

    # Save
    with open(args.output_json, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} prepared samples to {args.output_json}")
    print(f"\nNVV tag distribution in benchmark ({len(tag_stats)} types):")
    for tag, count in tag_stats.most_common(20):
        print(f"  {tag}: {count}")
    if len(tag_stats) > 20:
        print(f"  ... and {len(tag_stats) - 20} more types")


if __name__ == "__main__":
    main()
