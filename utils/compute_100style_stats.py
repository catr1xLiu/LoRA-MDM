import numpy as np
import os
from tqdm import tqdm

dataset_root = "./dataset/100STYLE-SMPL"
motion_dir = os.path.join(dataset_root, "new_joint_vecs-001")

def compute_stats():
    if not os.path.exists(motion_dir):
        fallback = os.path.join(dataset_root, "new_joint_vecs")
        if os.path.exists(fallback):
            print(f"Missing '{motion_dir}', using '{fallback}' instead.")
            motion_dir_used = fallback
        else:
            print(f"Error: Could not find motion directory at: {motion_dir}")
            return
    else:
        motion_dir_used = motion_dir

    files = [f for f in os.listdir(motion_dir_used) if f.endswith('.npy')]
    
    if len(files) == 0:
        print("Error: No .npy files found.")
        return

    all_motions = []
    for f in tqdm(files):
        path = os.path.join(motion_dir_used, f)
        try:
            data = np.load(path)
            all_motions.append(data)
        except Exception as e:
            print(f"Skipping corrupt file {f}: {e}")

    all_data = np.concatenate(all_motions, axis=0)

    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)

    std[std < 1e-8] = 1e-8

    mean_path = os.path.join(dataset_root, "Mean.npy")
    std_path = os.path.join(dataset_root, "Std.npy")

    np.save(mean_path, mean)
    np.save(std_path, std)

    print(f"Saved Mean.npy to {mean_path}")
    print(f"Saved Std.npy to {std_path}")

if __name__ == "__main__":
    compute_stats()
