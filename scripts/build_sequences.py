"""
Build temporal sequence index from preprocessed patches.

Groups patches by spatial location (row, col) and glacier,
then creates sequences of T consecutive timesteps paired
with a target patch H years in the future.

Output: data/processed/sequences/<glacier>_sequences.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PATCHES_DIR   = Path("data/processed/patches")
SEQUENCES_DIR = Path("data/processed/sequences")
SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN  = 4     # T input timesteps (reduced from 8 - we have few years per glacier)
HORIZONS = [1, 2, 3]   # predict 1, 2, 3 years ahead

GLACIERS = ["aletsch", "gangotri", "grey", "columbia", "athabasca"]


def build_sequences_for_glacier(name: str) -> list:
    patches_dir = PATCHES_DIR / name

    if not patches_dir.exists():
        print(f"  No patches found for {name}")
        return []

    # Load all patch filenames
    patch_files = sorted(patches_dir.glob("*.npz"))
    print(f"  Total patches: {len(patch_files)}")

    # Group by spatial location (row, col)
    location_groups = defaultdict(list)

    for f in patch_files:
        # Filename: <scene_id>_r<row>_c<col>.npz
        parts = f.stem.split("_")
        # Find row/col parts
        row_col = None
        for i, p in enumerate(parts):
            if p.startswith("r") and i + 1 < len(parts) and parts[i+1].startswith("c"):
                row_col = f"{p}_{parts[i+1]}"
                break

        if row_col is None:
            # Fallback: use last two parts
            row_col = f"{parts[-2]}_{parts[-1]}"

        location_groups[row_col].append(f)

    print(f"  Unique locations: {len(location_groups)}")

    # Build sequences per location
    sequences = []

    for loc, files in location_groups.items():
        # Sort by year (encoded in filename)
        def extract_year(f):
            parts = f.stem.split("_")
            for p in parts:
                if len(p) == 8 and p.isdigit():
                    return int(p[:4])
            return 0

        files = sorted(files, key=extract_year)
        years = [extract_year(f) for f in files]

        # Need at least SEQ_LEN + max(HORIZONS) files
        min_files = SEQ_LEN + max(HORIZONS)
        if len(files) < min_files:
            continue

        # Sliding window
        for i in range(len(files) - SEQ_LEN):
            input_files = files[i: i + SEQ_LEN]
            input_years = years[i: i + SEQ_LEN]

            for horizon in HORIZONS:
                target_idx = i + SEQ_LEN + horizon - 1
                if target_idx >= len(files):
                    continue

                target_file = files[target_idx]
                target_year = years[target_idx]

                sequences.append({
                    "glacier":      name,
                    "location":     loc,
                    "input_paths":  [str(f) for f in input_files],
                    "input_years":  input_years,
                    "target_path":  str(target_file),
                    "target_year":  target_year,
                    "horizon":      horizon,
                })

    return sequences


#  Build for all glaciers 
all_sequences = []

for name in GLACIERS:
    print(f"\n{name.upper()}")
    seqs = build_sequences_for_glacier(name)
    all_sequences.extend(seqs)
    print(f"  Sequences: {len(seqs)}")

#  Split train / val / test 
# Temporal split - test = most recent years
TEST_YEARS  = {2022, 2023}
VAL_YEARS   = {2016, 2017}

train_seqs = [s for s in all_sequences if s["target_year"] not in TEST_YEARS | VAL_YEARS]
val_seqs   = [s for s in all_sequences if s["target_year"] in VAL_YEARS]
test_seqs  = [s for s in all_sequences if s["target_year"] in TEST_YEARS]

print(f"\n{'='*40}")
print(f"SEQUENCE SUMMARY")
print(f"{'='*40}")
print(f"  Total:  {len(all_sequences)}")
print(f"  Train:  {len(train_seqs)}")
print(f"  Val:    {len(val_seqs)}")
print(f"  Test:   {len(test_seqs)}")

#  Save 
for split_name, split_seqs in [
    ("train", train_seqs),
    ("val",   val_seqs),
    ("test",  test_seqs),
    ("all",   all_sequences),
]:
    out_path = SEQUENCES_DIR / f"{split_name}_sequences.json"
    with open(out_path, "w") as f:
        json.dump(split_seqs, f, indent=2)
    print(f"  Saved: {out_path.name} ({len(split_seqs)} sequences)")

print("\nDone.")