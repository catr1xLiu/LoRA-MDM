#!/usr/bin/env python3
"""
Fixes the "Divide by Zero" error in LoRA-MDM training by patching Std.npy.

The Issue:
    Some features (like root local velocity) are mathematically constant (0.0) 
    across the entire dataset due to canonicalization.
    This results in Std=0.0. During training, normalization divides by Std, 
    causing NaNs.

The Fix:
    We find any Std < epsilon and set it to 1.0. 
    (x - mean) / 1.0 is safe and preserves the 0 value.
"""

import numpy as np
import argparse
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing Std.npy and Mean.npy")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    std_file = data_path / "Std.npy"
    mean_file = data_path / "Mean.npy"

    if not std_file.exists():
        print(f"Error: Could not find {std_file}")
        return

    print(f"Processing {std_file}...")
    
    # Load
    std = np.load(std_file)
    mean = np.load(mean_file) if mean_file.exists() else None
    
    # Check for zeros
    # We use a small epsilon because floating point math is weird
    epsilon = 1e-5
    zero_indices = np.where(std < epsilon)[0]

    if len(zero_indices) > 0:
        print(f"Found {len(zero_indices)} features with near-zero variance.")
        print(f"Indices: {zero_indices}")
        print("Values before fix:", std[zero_indices])

        # Create backup
        backup_path = std_file.with_suffix(".npy.bak")
        shutil.copy(std_file, backup_path)
        print(f"Backup saved to {backup_path}")

        # FIX: Replace 0 with 1.0
        # Why 1.0? Because (Val - Mean) / 1.0 = (0 - 0) / 1 = 0.
        # If we used epsilon, we might get (0 - 1e-9) / 1e-9 = 1.0 (noise amplification).
        std[zero_indices] = 1.0
        
        # Save back
        np.save(std_file, std)
        print("Fixed Std.npy saved.")
        
        # Verify Mean just in case (Mean should be 0 for these too usually)
        if mean is not None:
             print("Corresponding Means:", mean[zero_indices])
    else:
        print("No zero-variance features found. Std.npy is already safe.")

if __name__ == "__main__":
    main()