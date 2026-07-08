#!/usr/bin/env python3
"""
Extract tar files with duplicate filenames by adding occurrence counter suffixes.

Usage:
    python3 extract_tar_dedup.py <tar_path> <output_dir> [--prefix PREFIX]

Each file is renamed as: {prefix}{base}__{counter:04d}{ext}
"""
import tarfile
import os
import sys
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Extract tar with dedup renaming")
    parser.add_argument("tar_path", help="Path to tar file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--prefix", default="", help="Optional prefix for all output files")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts, don't extract")
    args = parser.parse_args()

    tar_path = args.tar_path
    out_dir = args.output_dir
    prefix = args.prefix

    os.makedirs(out_dir, exist_ok=True)

    # First pass: count occurrences per filename
    counter = defaultdict(int)
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                counter[member.name] += 1

    total = sum(counter.values())
    unique = len(counter)
    dups = {k: v for k, v in counter.items() if v > 1}

    print(f"Tar:      {tar_path}")
    print(f"Output:   {out_dir}")
    print(f"Files:    {total} total, {unique} unique names, {len(dups)} names with duplicates")
    if dups:
        worst = max(dups, key=dups.get)
        print(f"Max dup:  {worst} × {dups[worst]}")

    if args.dry_run:
        return

    # Second pass: extract with renaming
    seen = defaultdict(int)
    extracted = 0
    skipped = 0

    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            name = member.name
            seen[name] += 1
            cnt = counter[name]

            if cnt == 1:
                out_name = f"{prefix}{name}"
            else:
                base, ext = os.path.splitext(name)
                out_name = f"{prefix}{base}__{seen[name]-1:04d}{ext}"

            out_path = os.path.join(out_dir, out_name)

            # Skip if already exists (resume support)
            if os.path.exists(out_path):
                skipped += 1
                continue

            # Extract the file object
            fobj = tar.extractfile(member)
            if fobj is None:
                print(f"  [SKIP] Cannot extract: {name}")
                skipped += 1
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wb') as out_f:
                out_f.write(fobj.read())

            extracted += 1
            if extracted % 1000 == 0:
                print(f"  ... {extracted}/{total} extracted")

    print(f"Done: {extracted} extracted, {skipped} skipped (already exist)")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
