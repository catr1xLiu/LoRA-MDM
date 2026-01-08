import os

# 1. Setup Paths (Adjust 'dataset_root' if your folder is named differently)
dataset_root = "./dataset/100STYLE-SMPL" 
dict_path = os.path.join(dataset_root, "100STYLE_name_dict.txt")
output_split_path = os.path.join(dataset_root, "train_100STYLE_Full.txt")
motion_dir_standard = os.path.join(dataset_root, "new_joint_vecs")
motion_dir_weird = os.path.join(dataset_root, "new_joint_vecs-001")

def build_split_file():
    if not os.path.exists(dict_path):
        print(f"Error: {dict_path} not found.")
        print("   Please download '100STYLE_name_dict.txt' from the dataset drive and place it in the folder.")
        return

    valid_files = []
    with open(dict_path, 'r') as f:
        for line in f:
            # Parse lines like: "68_10 68_10 68_Walking_10.bvh"
            parts = line.strip().split(" ")
            if len(parts) >= 1:
                filename_no_ext = parts[0]
                valid_files.append(filename_no_ext)

    with open(output_split_path, 'w') as f:
        for name in valid_files:
            f.write(f"{name}\n")
            

def fix_folder_name():
    # The code expects 'new_joint_vecs-001', but standard processing makes 'new_joint_vecs'
    if os.path.exists(motion_dir_standard) and not os.path.exists(motion_dir_weird):
        print(f"Code expects '{motion_dir_weird}', but found '{motion_dir_standard}'")
        print("   Renaming folder to match the code's expectation...")
        os.rename(motion_dir_standard, motion_dir_weird)
    elif not os.path.exists(motion_dir_standard) and not os.path.exists(motion_dir_weird):
        print(f"Error: Could not find motion vectors. Make sure you processed the dataset first!")

if __name__ == "__main__":
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset folder '{dataset_root}' does not exist.")
        print("   If you named it '100style', please rename it to '100STYLE-SMPL' to match the hardcoded path in the library.")
    else:
        fix_folder_name()
        build_split_file()
