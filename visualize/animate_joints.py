"""
Visualize motion joints as a 3D matplotlib animation.

Supported input formats (auto-detected by shape):
  results.npy      -- dict with 'motion' [N, 22, 3, T]  (batch_age_generation output)
  new_joints       -- plain array [T, 22, 3]             (HumanML3D ground-truth xyz)
  new_joint_vecs   -- plain array [T, 263]               (HumanML3D / VanCriekinge raw feature vector)

Usage:
    python -m visualize.animate_joints --npy save/out/results.npy --rep 1
    python -m visualize.animate_joints --npy dataset/HumanML3D/new_joints/000000.npy
    python -m visualize.animate_joints --npy dataset/HumanML3D/new_joint_vecs/000000.npy
    python -m visualize.animate_joints --npy dataset/VanCriekinge/motion/SUBJ01_SUBJ1_1_humanml3d_22joints.npy
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from data_loaders.humanml.scripts.motion_process import recover_from_ric


JOINT_NAMES = [
    'pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee',
    'spine2', 'l_ankle', 'r_ankle', 'spine3', 'l_foot', 'r_foot',
    'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
    'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
]

BONES = [
    # spine / torso
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # left arm
    (9, 13), (13, 16), (16, 18), (18, 20),
    # right arm
    (9, 14), (14, 17), (17, 19), (19, 21),
]


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_results_npy(path, rep_idx):
    """results.npy dict  →  joints [T, 22, 3], title str, num_reps int."""
    data = np.load(path, allow_pickle=True).item()
    motion = data['motion']       # [N, 22, 3, T]
    lengths = data['lengths']
    texts = data['text']
    num_reps = motion.shape[0]

    if rep_idx >= num_reps:
        raise ValueError(f"rep_idx {rep_idx} out of range (0..{num_reps - 1})")

    T = lengths[rep_idx]
    joints = motion[rep_idx, :, :, :T]   # [22, 3, T]
    joints = joints.transpose(2, 0, 1)   # [T, 22, 3]
    title = texts[rep_idx % len(texts)]
    return joints, title, num_reps


def load_new_joints(path):
    """new_joints array [T, 22, 3]  →  joints [T, 22, 3], title str."""
    joints = np.load(path).astype(np.float32)
    assert joints.ndim == 3 and joints.shape[1] == 22 and joints.shape[2] == 3, \
        f"Expected [T, 22, 3], got {joints.shape}"
    return joints, os.path.basename(path)


def load_vec263(path):
    """
    Raw HumanML3D 263-D feature vector [T, 263]  →  joints [T, 22, 3].
    Works for both HumanML3D new_joint_vecs and VanCriekinge motion files
    (both store un-normalised feature vectors in the same format).
    """
    data = np.load(path).astype(np.float32)
    assert data.ndim == 2 and data.shape[1] == 263, \
        f"Expected [T, 263], got {data.shape}"
    tensor = torch.from_numpy(data).unsqueeze(0)  # [1, T, 263]
    xyz = recover_from_ric(tensor, 22)             # [1, T, 22, 3]
    return xyz.squeeze(0).numpy(), os.path.basename(path)  # [T, 22, 3]


def auto_load(path, rep_idx):
    """
    Detect format from the array shape and return (joints [T, 22, 3], title, fmt_label).
    """
    raw = np.load(path, allow_pickle=True)

    # results.npy: np.load returns a 0-d object array wrapping a dict
    if raw.ndim == 0:
        joints, title, num_reps = load_results_npy(path, rep_idx)
        return joints, f"{title}\n(rep {rep_idx}/{num_reps - 1})", "results.npy"

    arr = raw
    if arr.ndim == 3 and arr.shape[1] == 22 and arr.shape[2] == 3:
        joints, title = load_new_joints(path)
        return joints, title, "new_joints [T,22,3]"

    if arr.ndim == 2 and arr.shape[1] == 263:
        joints, title = load_vec263(path)
        return joints, title, "vec263 [T,263]"

    raise ValueError(
        f"Unrecognised array shape {arr.shape}. "
        "Expected a results.npy dict, [T,22,3] new_joints, or [T,263] feature vector."
    )


# ── Animation ────────────────────────────────────────────────────────────────

def animate(npy_path, rep_idx, show_labels=False):
    joints, title, fmt = auto_load(npy_path, rep_idx)
    print(f"Format detected: {fmt}  |  Frames: {len(joints)}")

    T = len(joints)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title, fontsize=9, wrap=True)

    pad = 0.1
    x_min, x_max = joints[:, :, 0].min() - pad, joints[:, :, 0].max() + pad
    y_min, y_max = joints[:, :, 1].min() - pad, joints[:, :, 1].max() + pad
    z_min, z_max = joints[:, :, 2].min() - pad, joints[:, :, 2].max() + pad

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    x_mid, y_mid, z_mid = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2

    joint_scatter = ax.scatter([], [], [], s=20, c='royalblue', zorder=3)
    bone_lines = [ax.plot([], [], [], lw=2, c='steelblue')[0] for _ in BONES]

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y (up)'); ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=-70)

    frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=8)

    label_texts = []
    if show_labels:
        pos0 = joints[0]
        for i, name in enumerate(JOINT_NAMES):
            t = ax.text(pos0[i, 0], pos0[i, 1], pos0[i, 2], name, fontsize=7, fontweight='bold', color='dimgray')
            label_texts.append(t)

    def update(frame_idx):
        pos = joints[frame_idx]  # [22, 3]
        joint_scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        for line, (i, j) in zip(bone_lines, BONES):
            line.set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])
            line.set_3d_properties([pos[i, 2], pos[j, 2]])
        frame_text.set_text(f'frame {frame_idx + 1}/{T}')
        for t, (x, y, z) in zip(label_texts, pos):
            t.set_position_3d((x, y, z))
        return [joint_scatter, frame_text] + bone_lines + label_texts

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 / 20, blit=False
    )
    plt.tight_layout()
    plt.show()
    return ani


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', type=str, required=True,
                        help='Path to results.npy, a new_joints .npy, or a new_joint_vecs .npy')
    parser.add_argument('--rep', type=int, default=0,
                        help='Repetition index (only used for results.npy format)')
    parser.add_argument('--labels', action='store_true',
                        help='Show joint name labels on each joint')
    args = parser.parse_args()
    animate(args.npy, args.rep, show_labels=args.labels)


if __name__ == '__main__':
    main()
