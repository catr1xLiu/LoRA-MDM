"""
Visualize motion joints from a results.npy file as a 3D matplotlib animation.

Usage:
    python -m visualize.animate_joints --npy save/out/results.npy
    python -m visualize.animate_joints --npy save/out/results.npy --rep 1
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# SMPL 22-joint kinematic chain (parent -> child pairs)
KINEMATIC_CHAIN = [
    # spine
    (0, 1), (1, 2), (2, 3), (3, 12),          # pelvis -> spine1 -> spine2 -> spine3 -> neck
    (12, 15),                                   # neck -> head
    # left arm
    (3, 13), (13, 14), (14, 16), (16, 18), (18, 20),
    # right arm
    (3, 14 - 1), (12, 17), (17, 19), (19, 21),  # will redefine cleanly below
    # left leg
    (0, 1),  # placeholder — redefined below
]

# Clean definition using standard SMPL joint indices:
# 0:pelvis 1:l_hip 2:r_hip 3:spine1 4:l_knee 5:r_knee 6:spine2
# 7:l_ankle 8:r_ankle 9:spine3 10:l_foot 11:r_foot 12:neck
# 13:l_collar 14:r_collar 15:head 16:l_shoulder 17:r_shoulder
# 18:l_elbow 19:r_elbow 20:l_wrist 21:r_wrist
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


def load_motion(npy_path, rep_idx):
    data = np.load(npy_path, allow_pickle=True).item()
    motion = data['motion']          # (reps, 22, 3, T)
    lengths = data['lengths']
    texts = data['text']
    num_reps = motion.shape[0]

    if rep_idx >= num_reps:
        raise ValueError(f"rep_idx {rep_idx} out of range (0..{num_reps - 1})")

    T = lengths[rep_idx]
    joints = motion[rep_idx, :, :, :T]   # (22, 3, T)
    joints = joints.transpose(2, 0, 1)   # (T, 22, 3)  — x=joints[0], y=up, z=depth
    text = texts[rep_idx % len(texts)]
    return joints, text, num_reps


def animate(npy_path, rep_idx):
    joints, text, num_reps = load_motion(npy_path, rep_idx)
    T, num_joints, _ = joints.shape

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f"{text}\n(rep {rep_idx}/{num_reps - 1})", fontsize=9, wrap=True)

    pad = 0.1
    x_min, x_max = joints[:, :, 0].min() - pad, joints[:, :, 0].max() + pad
    y_min, y_max = joints[:, :, 1].min() - pad, joints[:, :, 1].max() + pad
    z_min, z_max = joints[:, :, 2].min() - pad, joints[:, :, 2].max() + pad

    # Force equal spatial range on all axes for 1:1:1 aspect ratio
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    joint_scatter = ax.scatter([], [], [], s=20, c='royalblue', zorder=3)
    bone_lines = [ax.plot([], [], [], lw=2, c='steelblue')[0] for _ in BONES]

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y (up)')
    ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=-70)

    frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=8)

    def update(frame_idx):
        pos = joints[frame_idx]  # (22, 3)
        joint_scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

        for line, (i, j) in zip(bone_lines, BONES):
            xs = [pos[i, 0], pos[j, 0]]
            ys = [pos[i, 1], pos[j, 1]]
            zs = [pos[i, 2], pos[j, 2]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

        frame_text.set_text(f'frame {frame_idx + 1}/{T}')
        return [joint_scatter, frame_text] + bone_lines

    interval_ms = 1000 / 20  # 20 fps
    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=interval_ms, blit=False
    )

    plt.tight_layout()
    plt.show()
    return ani


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', type=str, default='save/out/results.npy')
    parser.add_argument('--rep', type=int, default=0, help='Repetition index to display')
    args = parser.parse_args()
    animate(args.npy, args.rep)


if __name__ == '__main__':
    main()
