"""
Quick validation script for VanCriekingeDataset.
Run from the repo root:
    python -m data_loaders.vancriekinge.test_dataset
"""

import numpy as np
import torch
from collections import Counter

from data_loaders.vancriekinge.dataset import VanCriekingeDataset


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    styles = ["Child", "YoungAdult", "MiddleAged", "Elderly"]

    # ------------------------------------------------------------------ #
    section("1. Dataset instantiation")
    ds = VanCriekingeDataset(styles=styles, mode="train")
    print(f"  len(dataset)      : {len(ds)}")
    print(f"  nfeats            : {ds.nfeats}")
    print(f"  max_motion_length : {ds.max_motion_length}")
    print(f"  pointer           : {ds.pointer}")

    # ------------------------------------------------------------------ #
    section("2. Style distribution")
    style_counts = Counter(v["style"] for v in ds.data_dict.values())
    for style in styles:
        print(f"  {style:<15}: {style_counts.get(style, 0)} clips")

    # ------------------------------------------------------------------ #
    section("3. Token assignment")
    token_map = {
        ds.styles[i]: ds.tokens[i] for i in range(len(ds.styles))
    }
    print(f"  {token_map}")

    # ------------------------------------------------------------------ #
    section("4. Mean / Std stats")
    mean, std = ds.get_mean_std()
    print(f"  mean shape : {mean.shape}, range [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  std  shape : {std.shape},  range [{std.min():.3f}, {std.max():.3f}]")

    # ------------------------------------------------------------------ #
    section("5. Single sample")
    sample = ds[0]
    print(f"  keys       : {list(sample.keys())}")
    print(f"  inp shape  : {sample['inp'].shape}  (expect [263, 1, 196])")
    print(f"  inp dtype  : {sample['inp'].dtype}")
    print(f"  lengths    : {sample['lengths']}")
    print(f"  text       : {sample['text']}")
    print(f"  action     : {sample['action']}")
    print(f"  action_text: {sample['action_text']}")
    print(f"  style      : {sample['style']}")

    assert sample["inp"].shape == (263, 1, 196), "Unexpected inp shape"
    assert isinstance(sample["text"], str) and "style" in sample["text"]
    assert sample["lengths"] > 0

    # ------------------------------------------------------------------ #
    section("6. No T-pose (trial_index=0) clips")
    tpose_found = any("_0" in name.split("_SUBJ")[-1] for name in ds.name_list
                      if name.count("_") >= 2 and name.split("_")[2] == "0")
    # More robust check: trial_name ending in _0
    tpose_found = any(
        name.rsplit("_", 1)[-1] == "0"
        for name in ds.name_list
    )
    print(f"  T-pose clips in dataset: {tpose_found}  (expect False)")
    assert not tpose_found, "T-pose calibration clips should be excluded"

    # ------------------------------------------------------------------ #
    section("7. Motion length bounds")
    lengths = [ds.data_dict[n]["length"] for n in ds.name_list]
    print(f"  min raw length : {min(lengths)}")
    print(f"  max raw length : {max(lengths)}")
    assert min(lengths) >= ds.min_motion_length, "Clip below min_motion_length found"
    assert max(lengths) <= 121, "Unexpected motion length"

    # ------------------------------------------------------------------ #
    section("8. inv_transform round-trip")
    motion_norm = sample["inp"].squeeze(1).T.numpy()  # [196, 263]
    mean, std = ds.get_mean_std()
    motion_raw = ds.inv_transform(motion_norm)
    motion_renorm = ds.transform(motion_raw)
    diff = np.abs(motion_norm - motion_renorm).max()
    print(f"  max round-trip error: {diff:.2e}  (expect ~0)")
    assert diff < 1e-5, "inv_transform/transform round-trip failed"

    # ------------------------------------------------------------------ #
    section("9. reset_max_len")
    ds.reset_max_len(80)
    print(f"  pointer after reset_max_len(80): {ds.pointer}")
    print(f"  new len(dataset): {len(ds)}")
    assert ds.pointer > 0

    # ------------------------------------------------------------------ #
    section("10. Multiple samples — shape consistency")
    ds2 = VanCriekingeDataset(styles=styles, mode="train")
    for i in range(min(10, len(ds2))):
        s = ds2[i]
        assert s["inp"].shape == (263, 1, 196), f"Bad shape at index {i}"
    print(f"  All 10 samples have correct shape [263, 1, 196]")

    print("\n✓ All checks passed.\n")


if __name__ == "__main__":
    main()
