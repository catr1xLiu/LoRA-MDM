"""
Velocity-to-Age Mapping Utility (Rank-Based / Histogram Equalization)

This script computes average root velocity and maps it to age using RANKING.
This guarantees a Uniform Distribution of ages, preventing "clumping" at the old age.

Usage:
    python age_velocity_mapping.py --dataset 100style --output_dir ./velocity_age_cache
    python age_velocity_mapping.py --dataset humanml --output_dir ./velocity_age_cache
"""

import numpy as np
import os
import pickle
import argparse
from os.path import join as pjoin
from tqdm import tqdm
import codecs as cs
import matplotlib.pyplot as plt
from scipy.stats import rankdata 

def compute_root_velocity(motion: np.ndarray, denormalize: bool = False, 
                          mean: np.ndarray = None, std: np.ndarray = None) -> float:
    """Compute average root horizontal velocity magnitude."""
    # Root linear velocity is at indices 1 and 2 (XZ plane)
    root_vel_x = motion[:, 1]
    root_vel_z = motion[:, 2]
    # Compute per-frame velocity magnitude
    vel_magnitude = np.sqrt(root_vel_x**2 + root_vel_z**2)
    return np.mean(vel_magnitude)

def get_rank_based_ages(velocities_dict, age_min=0.18, age_max=0.90):
    """
    Map velocities to ages based on their RANK in the dataset.
    Fastest -> Youngest (age_min)
    Slowest -> Oldest (age_max)
    """
    names = list(velocities_dict.keys())
    vals = np.array(list(velocities_dict.values()))
    
    # 1. Compute Rank (0.0 to 1.0)
    # method='average' handles ties
    # We want High Velocity -> Rank 1.0 -> Young (age_min)
    # Low Velocity -> Rank 0.0 -> Old (age_max)
    
    # rankdata gives 1..N. We divide by N to get 0..1
    ranks = rankdata(vals, method='average') / len(vals)
    
    # 2. Map Rank to Age (Inverse)
    # Rank 1.0 (Fastest) should be age_min
    # Rank 0.0 (Slowest) should be age_max
    # Formula: Age = Max - Rank * (Max - Min)
    mapped_ages = age_max - (ranks * (age_max - age_min))
    
    return {name: age for name, age in zip(names, mapped_ages)}

def process_dataset(dataset_name, data_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup paths
    if dataset_name == '100style':
        motion_dir = pjoin(data_root, "new_joint_vecs-001")
        split_file = pjoin(data_root, "train_100STYLE_Full.txt")
        # Handle the folder name weirdness
        if not os.path.exists(motion_dir):
            motion_dir = pjoin(data_root, "new_joint_vecs")
    else: # humanml
        motion_dir = pjoin(data_root, "new_joint_vecs")
        split_file = pjoin(data_root, "train.txt")

    # Load ID list
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
            
    print(f"Processing {len(id_list)} files from {dataset_name}...")
    
    velocities = {}
    
    for name in tqdm(id_list, desc="Computing velocities"):
        try:
            motion_path = pjoin(motion_dir, name + ".npy")
            if not os.path.exists(motion_path):
                continue
            
            motion = np.load(motion_path)
            velocities[name] = compute_root_velocity(motion)
        except Exception:
            continue
            
    # Compute Rank-Based Ages
    print(f"Computing rank-based ages for {len(velocities)} clips...")
    ages = get_rank_based_ages(velocities)
    
    # Save
    stats = {
        'min_vel': np.min(list(velocities.values())),
        'max_vel': np.max(list(velocities.values())),
        'age_min': 0.18,
        'age_max': 0.90
    }
    
    output_data = {
        'velocities': velocities,
        'ages': ages,
        'stats': stats
    }
    
    pkl_name = "100style_velocity_ages.pkl" if dataset_name == '100style' else "humanml_velocity_ages.pkl"
    output_path = pjoin(output_dir, pkl_name)
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
        
    print(f"Saved to {output_path}")
    create_visualization(velocities, ages, output_dir, dataset_name)

def create_visualization(velocities, ages, output_dir, dataset_name):
    vel_array = np.array(list(velocities.values()))
    age_array = np.array(list(ages.values()))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity Hist
    axes[0].hist(vel_array, bins=50, alpha=0.7)
    axes[0].set_title(f'{dataset_name} Velocities (Original)')
    axes[0].set_xlabel('Root Velocity')
    
    # Age Hist (Should be flat/uniform now!)
    axes[1].hist(age_array, bins=50, color='orange', alpha=0.7)
    axes[1].set_title(f'{dataset_name} Ages (Rank-Mapped)')
    axes[1].set_xlabel('Age')
    
    plt.savefig(pjoin(output_dir, f"{dataset_name}_dist.png"))
    print(f"Saved plot to {output_dir}/{dataset_name}_dist.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="both")
    parser.add_argument("--output_dir", type=str, default="./velocity_age_cache/")
    parser.add_argument("--style_root", type=str, default="./dataset/100STYLE-SMPL/")
    parser.add_argument("--humanml_root", type=str, default="./dataset/HumanML3D/")
    args = parser.parse_args()
    
    if args.dataset in ["100style", "both"]:
        process_dataset('100style', args.style_root, args.output_dir)
    if args.dataset in ["humanml", "both"]:
        process_dataset('humanml', args.humanml_root, args.output_dir)