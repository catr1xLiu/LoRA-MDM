"""
Diagnostic: compare VanCriekinge 263-dim HumanML3D-format motion files against
the HumanML3D training distribution (walking-only subset) and against 100STYLE-SMPL.

Goal: identify why a rank-5 LoRA fine-tuned on VC produces "on drugs" jittery
motion at inference, despite normalisation already being fixed (HumanML3D stats
are used consistently in training + inference).

Hypothesis under test: the VC .npy files themselves live on a systematically
shifted manifold relative to the base MDM's training distribution, because the
VC -> HumanML3D-263 conversion has a bug (fps, root extraction, or floor contact).

Applies the SAME sample filtering that VanCriekingeDataset uses:
  - skip clips with trial_index == 0 (T-pose)
  - skip clips with length < 40 frames

Outputs:
  - Stdout: section-wise and per-dim numerical summary
  - docs/Sessions/assets/2026-04-10-vc-manifold/*.png : diagnostic plots
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HML_STATS = os.path.join(REPO, "dataset", "HumanML3D")
VC_MOTION = os.path.join(REPO, "dataset", "VanCriekinge", "motion")
VC_META = os.path.join(REPO, "dataset", "VanCriekinge", "metadata")
HML_MOTION = os.path.join(REPO, "dataset", "HumanML3D", "new_joint_vecs")
HML_TEXT = os.path.join(REPO, "dataset", "HumanML3D", "texts")
HML_TRAIN = os.path.join(REPO, "dataset", "HumanML3D", "train.txt")
STYLE_MOTION = os.path.join(REPO, "dataset", "100STYLE-SMPL", "new_joint_vecs-001")
STYLE_SPLIT = os.path.join(REPO, "dataset", "100STYLE-SMPL", "train_100STYLE_Full.txt")
STYLE_DICT = os.path.join(REPO, "dataset", "100STYLE-SMPL", "100STYLE_name_dict.txt")

OUT_DIR = os.path.join(REPO, "docs", "Sessions", "assets", "2026-04-10-vc-manifold")
os.makedirs(OUT_DIR, exist_ok=True)

# ----- HumanML3D 263-dim layout -----
#   [0]      root_rot_vel        (1)    angular velocity around Y
#   [1:3]    root_lin_vel_xz     (2)    per-frame root XZ displacement (root frame)
#   [3:4]    root_y              (1)    root height
#   [4:67]   ric_data            (63)   21 joints * 3, root-local positions
#   [67:193] rot_data            (126)  21 joints * 6D continuous rotation
#   [193:259] local_vel          (66)   22 joints * 3 per-frame displacement
#   [259:263] foot_contact       (4)
SECTIONS = [
    ("root_rot_vel",    0, 1),
    ("root_lin_vel_xz", 1, 3),
    ("root_y",          3, 4),
    ("ric_data",        4, 67),
    ("rot_data",        67, 193),
    ("local_vel",       193, 259),
    ("foot_contact",    259, 263),
]

MIN_LEN = 40  # match VanCriekingeDataset.min_motion_length


def load_hml_stats():
    return (
        np.load(os.path.join(HML_STATS, "Mean.npy")),
        np.load(os.path.join(HML_STATS, "Std.npy")),
    )


# ---------- dataset loaders (filtered) ----------


def _parse_vc_filename(filename):
    stem = filename.replace("_humanml3d_22joints.npy", "")
    parts = stem.split("_")
    return parts[0], "_".join(parts[1:3])


def load_vc_clips():
    """Matches VanCriekingeDataset filtering: trial_index != 0, length >= 40."""
    clips = []
    skipped_short = 0
    skipped_t_pose = 0
    skipped_no_meta = 0
    for fn in sorted(os.listdir(VC_MOTION)):
        if not fn.endswith(".npy"):
            continue
        _, trial = _parse_vc_filename(fn)
        meta_path = os.path.join(VC_META, f"{trial}_metadata.json")
        if not os.path.exists(meta_path):
            skipped_no_meta += 1
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["trial_index"] == 0:
            skipped_t_pose += 1
            continue
        m = np.load(os.path.join(VC_MOTION, fn))
        if m.ndim != 2 or m.shape[1] != 263:
            print(f"WARN: {fn} has shape {m.shape}, skipping")
            continue
        if len(m) < MIN_LEN:
            skipped_short += 1
            continue
        clips.append((fn, m, meta.get("age")))
    print(f"VC: loaded {len(clips)} clips  (skipped {skipped_t_pose} T-pose, "
          f"{skipped_short} short, {skipped_no_meta} missing-meta)")
    return clips


def load_hml_walking_clips(limit=1000):
    """HumanML3D train samples whose text caption mentions 'walk'."""
    with open(HML_TRAIN) as f:
        ids = [l.strip() for l in f if l.strip()]
    clips = []
    for i in ids:
        txt_path = os.path.join(HML_TEXT, f"{i}.txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path) as f:
            s = f.read().lower()
        if "walk" not in s:
            continue
        m_path = os.path.join(HML_MOTION, f"{i}.npy")
        if not os.path.exists(m_path):
            continue
        m = np.load(m_path)
        if m.ndim != 2 or m.shape[1] != 263 or len(m) < MIN_LEN:
            continue
        clips.append((i, m, None))
        if len(clips) >= limit:
            break
    print(f"HumanML3D walking: loaded {len(clips)} clips")
    return clips


def load_style_walking_clips(limit=1000):
    """100STYLE-SMPL clips of motion type FW (forward walking) — closest match to VC."""
    # 100STYLE name dict format: "name  style_motiontype_cutidx  label ..."
    walking_names = set()
    if not os.path.exists(STYLE_DICT):
        print(f"WARN: {STYLE_DICT} not found, loading all 100STYLE clips")
        walking_names = None
    else:
        with open(STYLE_DICT) as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) < 3:
                    continue
                name = parts[0]
                mtype_tag = parts[1].split("_")[1] if "_" in parts[1] else ""
                if mtype_tag == "FW":
                    walking_names.add(name)

    with open(STYLE_SPLIT) as f:
        ids = [l.strip() for l in f if l.strip()]

    clips = []
    for i in ids:
        if walking_names is not None and i not in walking_names:
            continue
        m_path = os.path.join(STYLE_MOTION, f"{i}.npy")
        if not os.path.exists(m_path):
            continue
        m = np.load(m_path)
        if m.ndim != 2 or m.shape[1] != 263 or len(m) < MIN_LEN:
            continue
        clips.append((i, m, None))
        if len(clips) >= limit:
            break
    print(f"100STYLE (FW): loaded {len(clips)} clips")
    return clips


# ---------- analysis helpers ----------


def concat_normalised(clips, mean, std):
    frames = np.concatenate([m for _, m, _ in clips], axis=0)
    return (frames - mean) / std


def section_summary(label, norm):
    print(f"\n----- {label} (n_frames={len(norm):,}) -----")
    print(f"{'section':<22} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    for name, lo, hi in SECTIONS:
        s = norm[:, lo:hi]
        print(f"{name:<22} {s.mean():>+10.3f} {s.std():>10.3f} {s.min():>+10.2f} {s.max():>+10.2f}")


# ---------- plots ----------


def plot_perdim_mean(norms, title, path):
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = {"VanCriekinge": "#d62728", "HumanML3D (walking)": "#2ca02c", "100STYLE (FW)": "#1f77b4"}
    for label, norm in norms.items():
        ax.plot(norm.mean(axis=0), label=label, color=colors.get(label, None), linewidth=1.0)
    ax.axhline(0, color="k", linewidth=0.5)
    y_top = max(n.mean(axis=0).max() for n in norms.values())
    y_bot = min(n.mean(axis=0).min() for n in norms.values())
    label_y = y_top + 0.08 * (y_top - y_bot)
    for name, lo, hi in SECTIONS:
        ax.axvline(hi - 0.5, color="grey", linestyle="--", alpha=0.3)
        ax.text((lo + hi) / 2, label_y, name, ha="center", va="bottom", fontsize=8, alpha=0.7)
    ax.set_ylim(y_bot - 0.1 * (y_top - y_bot), label_y + 0.15 * (y_top - y_bot))
    ax.set_xlabel("dim index (0..262)")
    ax.set_ylabel("mean in HumanML3D-normalised space")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(-1, 263)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_perdim_std(norms, title, path):
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = {"VanCriekinge": "#d62728", "HumanML3D (walking)": "#2ca02c", "100STYLE (FW)": "#1f77b4"}
    for label, norm in norms.items():
        ax.plot(norm.std(axis=0), label=label, color=colors.get(label, None), linewidth=1.0)
    for _, _, hi in SECTIONS:
        ax.axvline(hi - 0.5, color="grey", linestyle="--", alpha=0.3)
    ax.axhline(1.0, color="k", linewidth=0.5, linestyle=":")
    ax.set_xlabel("dim index (0..262)")
    ax.set_ylabel("std in HumanML3D-normalised space")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(-1, 263)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_section_histograms(norms, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()
    colors = {"VanCriekinge": "#d62728", "HumanML3D (walking)": "#2ca02c", "100STYLE (FW)": "#1f77b4"}
    for ax, (name, lo, hi) in zip(axes, SECTIONS):
        for label, norm in norms.items():
            vals = norm[:, lo:hi].ravel()
            ax.hist(vals, bins=100, density=True, alpha=0.45,
                    label=label, color=colors.get(label, None),
                    range=(-5, 5))
        ax.axvline(0, color="k", linewidth=0.4)
        ax.set_title(f"{name}  (dims {lo}:{hi})")
        ax.set_xlabel("normalised value")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
    # hide extra axes
    for ax in axes[len(SECTIONS):]:
        ax.axis("off")
    fig.suptitle("Per-section distribution of normalised features\n(all values in HumanML3D-normalised space)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_joint_local_vel_mean(norms, path):
    """Per-joint mean of the local_vel block (22 joints * 3 components)."""
    # HumanML3D joint order (22):
    # 0 pelvis, 1 lhip, 2 rhip, 3 spine1, 4 lknee, 5 rknee, 6 spine2,
    # 7 lankle, 8 rankle, 9 spine3, 10 ltoe, 11 rtoe, 12 neck, 13 lclav,
    # 14 rclav, 15 head, 16 lsh, 17 rsh, 18 lelb, 19 relb, 20 lwrist, 21 rwrist
    JOINTS = ["pelvis","lhip","rhip","sp1","lknee","rknee","sp2","lank","rank",
              "sp3","ltoe","rtoe","neck","lclav","rclav","head","lsh","rsh",
              "lelb","relb","lwri","rwri"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    colors = {"VanCriekinge": "#d62728", "HumanML3D (walking)": "#2ca02c", "100STYLE (FW)": "#1f77b4"}
    axis_names = ["X (forward-ish)", "Y (vertical)", "Z (lateral)"]
    x = np.arange(22)
    width = 0.27
    for row, ax_idx in enumerate(range(3)):
        ax = axes[row]
        for k, (label, norm) in enumerate(norms.items()):
            lv = norm[:, 193:259].reshape(-1, 22, 3)[:, :, ax_idx]
            means = lv.mean(axis=0)
            ax.bar(x + (k - 1) * width, means, width, label=label, color=colors.get(label, None))
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(f"mean (norm.)")
        ax.set_title(f"local_vel[{ax_idx}] — {axis_names[row]}")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(JOINTS, rotation=45, ha="right")
    fig.suptitle("Per-joint mean of local_vel block (HumanML3D-normalised)\n— uniform offset across joints would indicate a conversion bug, not locomotion", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_root_traj_samples(vc_clips, hml_walking, style_walking, path):
    """Raw (denormalised) root_y and root_lin_vel for a handful of clips."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    datasets = [
        ("VanCriekinge", vc_clips[:8], "#d62728"),
        ("HumanML3D walking", hml_walking[:8], "#2ca02c"),
        ("100STYLE FW", style_walking[:8], "#1f77b4"),
    ]
    for col, (label, clips, color) in enumerate(datasets):
        ax_y = axes[0, col]
        ax_v = axes[1, col]
        for _, m, _ in clips:
            # raw root_y (dim 3) — these files ARE raw, not yet normalised
            ax_y.plot(m[:, 3], color=color, alpha=0.55, linewidth=0.9)
            vxz = np.linalg.norm(m[:, 1:3], axis=1)
            ax_v.plot(vxz, color=color, alpha=0.55, linewidth=0.9)
        ax_y.set_title(f"{label}  root_y (raw, dim 3)")
        ax_y.set_xlabel("frame")
        ax_y.set_ylabel("root_y (raw units)")
        ax_y.axhline(0, color="k", linewidth=0.4)
        ax_v.set_title(f"{label}  |root_lin_vel_xz|  (raw)")
        ax_v.set_xlabel("frame")
        ax_v.set_ylabel("per-frame displacement (raw)")
        ax_v.axhline(0, color="k", linewidth=0.4)
    # Share y-ranges across columns per row so magnitudes are comparable
    for row in range(2):
        lo = min(a.get_ylim()[0] for a in axes[row])
        hi = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(lo, hi)
    fig.suptitle("Raw root-Y and root linear-velocity magnitude across sample clips\n(matched y-axes per row — VC should sit in the same range if the conversion is correct)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_velocity_magnitudes(norms, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"VanCriekinge": "#d62728", "HumanML3D (walking)": "#2ca02c", "100STYLE (FW)": "#1f77b4"}
    for label, norm in norms.items():
        lv = norm[:, 193:259].reshape(-1, 22, 3)
        mag = np.linalg.norm(lv, axis=2).mean(axis=1)  # per-frame avg joint velocity magnitude
        ax.hist(mag, bins=100, density=True, alpha=0.5, label=label, color=colors.get(label, None),
                range=(0, 6))
    ax.set_xlabel("per-frame mean joint-velocity magnitude (normalised units)")
    ax.set_ylabel("density")
    ax.set_title("Distribution of per-frame mean joint velocity magnitude\n(VC being shifted right ⇒ velocities are inflated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_raw_root_y_hist(vc_clips, hml_walking, style_walking, mean, path):
    """Raw root_y values (dim 3) across datasets — check floor contact."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, clips, color in [
        ("VanCriekinge", vc_clips, "#d62728"),
        ("HumanML3D walking", hml_walking, "#2ca02c"),
        ("100STYLE FW", style_walking, "#1f77b4"),
    ]:
        vals = np.concatenate([m[:, 3] for _, m, _ in clips])
        ax.hist(vals, bins=100, density=True, alpha=0.5, label=f"{label} (median={np.median(vals):.2f})", color=color, range=(-0.5, 2.0))
    ax.axvline(mean[3], color="k", linestyle="--", linewidth=0.8, label=f"HumanML3D global mean ({mean[3]:.2f})")
    ax.set_xlabel("raw root_y (dim 3)")
    ax.set_ylabel("density")
    ax.set_title("Distribution of raw root_y  —  does the character stand on the floor?\n(If VC is consistently higher than HumanML3D walking, the floor-contact step was skipped)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_raw_root_vel_hist(vc_clips, hml_walking, style_walking, path):
    """Raw |root_lin_vel_xz| per frame."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, clips, color in [
        ("VanCriekinge", vc_clips, "#d62728"),
        ("HumanML3D walking", hml_walking, "#2ca02c"),
        ("100STYLE FW", style_walking, "#1f77b4"),
    ]:
        vals = np.concatenate([np.linalg.norm(m[:, 1:3], axis=1) for _, m, _ in clips])
        ax.hist(vals, bins=100, density=True, alpha=0.5,
                label=f"{label} (median={np.median(vals):.3f})", color=color, range=(0, 0.4))
    ax.set_xlabel("raw per-frame root XZ displacement")
    ax.set_ylabel("density")
    ax.set_title("Raw |root_lin_vel_xz| per frame  —  per-frame forward displacement\n(If VC's median is ~2x HumanML3D's, the fps / dt is wrong)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def trajectory_alignment_stats(clips, label):
    """Per-clip: measure how well the dominant walking direction aligns with +Z.

    Uses raw root_lin_vel_xz (dims 1-2) from the 263-dim vector.
    For each clip, accumulate the mean (vx, vz) over all frames, then compute
    the angle between that vector and +Z.  A perfectly aligned dataset has all
    angles near 0°.

    Returns array of per-clip angles in degrees.
    """
    angles = []
    for _, m, _ in clips:
        vx = m[:, 1]   # root_lin_vel_x  (canonical X)
        vz = m[:, 2]   # root_lin_vel_z  (canonical Z = forward)
        mean_vx = float(np.nanmean(vx))
        mean_vz = float(np.nanmean(vz))
        mag = np.sqrt(mean_vx**2 + mean_vz**2)
        if mag < 1e-6:
            continue   # nearly stationary clip — skip
        angle = np.degrees(np.arctan2(abs(mean_vx), abs(mean_vz)))  # deviation from Z
        angles.append(angle)
    angles = np.array(angles)
    print(f"\n  {label} trajectory Z-alignment ({len(angles)} moving clips):")
    print(f"    angle from +Z:  mean={angles.mean():.2f}°  median={np.median(angles):.2f}°"
          f"  p90={np.percentile(angles,90):.2f}°  max={angles.max():.2f}°")
    print(f"    clips within 10°: {(angles < 10).mean()*100:.1f}%   "
          f"within 20°: {(angles < 20).mean()*100:.1f}%   "
          f"within 45°: {(angles < 45).mean()*100:.1f}%")
    return angles


def plot_trajectory_alignment(datasets, path):
    """Histogram of per-clip angular deviation from +Z for each dataset."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"VanCriekinge": "#d62728", "HumanML3D (walking)": "#2ca02c", "100STYLE (FW)": "#1f77b4"}
    bins = np.linspace(0, 90, 46)
    for label, angles in datasets.items():
        ax.hist(angles, bins=bins, density=True, alpha=0.5,
                label=f"{label} (median={np.median(angles):.1f}°)",
                color=colors.get(label))
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Per-clip angular deviation of mean velocity from canonical +Z (degrees)")
    ax.set_ylabel("density")
    ax.set_title("Trajectory Z-axis alignment\n"
                 "0° = all forward motion in +Z (perfect); 90° = motion entirely in +X (misaligned)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def top_shifted_dims(norm, k=15):
    m = norm.mean(axis=0)
    order = np.argsort(np.abs(m))[::-1][:k]
    return [(int(i), float(m[i]), float(norm[:, i].std())) for i in order]


# ---------- angular velocity (fps diagnostic) ----------

# HumanML3D's rot_data (dims 67:193) stores 21 joints (skipping pelvis) × 6D
# continuous rotation.  This is pose, not velocity — we compute angular velocity
# between consecutive frames and compare across datasets.
#
# Assumed frame rate: HumanML3D = 20 fps.  VC is converted *into* this layout,
# so if the conversion is correct it should also be 20 fps.  If VC's angular
# velocities are uniformly inflated by factor K across *all* joints, the
# conversion's fps is K× too high.
ASSUMED_FPS = 20

ROTDATA_JOINTS = [
    "lhip", "rhip", "sp1", "lknee", "rknee", "sp2", "lank", "rank",
    "sp3", "ltoe", "rtoe", "neck", "lclav", "rclav", "head", "lsh", "rsh",
    "lelb", "relb", "lwri", "rwri",
]  # 21 joints, pelvis excluded


def sixd_to_matrix(x):
    """Convert 6D continuous rotation (Zhou et al. 2019) to rotation matrix.
    x shape (..., 6)  ->  R shape (..., 3, 3)."""
    a1 = x[..., :3]
    a2 = x[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    dot = (b1 * a2).sum(axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def per_joint_angular_velocity(clips, fps=ASSUMED_FPS):
    """For each of the 21 rot_data joints, return a flat array of frame-to-frame
    angular velocity magnitudes (rad/s) concatenated across all clips.

    Returns: list of 21 np.ndarray, one per joint.
    """
    accum = [[] for _ in range(21)]
    for _, m, _ in clips:
        if len(m) < 2:
            continue
        rot6 = m[:, 67:193].reshape(len(m), 21, 6)          # (T, 21, 6)
        R = sixd_to_matrix(rot6)                             # (T, 21, 3, 3)
        # Relative rotation R_rel = R_{t+1} @ R_t.T
        R_rel = np.einsum("tjab,tjcb->tjac", R[1:], R[:-1])  # (T-1, 21, 3, 3)
        trace = np.einsum("tjii->tj", R_rel)                 # (T-1, 21)
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)                         # rad per frame
        rad_per_s = theta * fps                              # rad / s
        for j in range(21):
            accum[j].append(rad_per_s[:, j])
    return [np.concatenate(a) if a else np.array([]) for a in accum]


def print_angular_velocity_report(results):
    """results: dict[label -> list of 21 np.ndarray of rad/s]"""
    labels = list(results.keys())
    print(f"\n\n===== JOINT ANGULAR VELOCITY (rad/s, assumed fps={ASSUMED_FPS}) =====")
    header = f"{'joint':<8}"
    for label in labels:
        header += f"  {label[:14]:>14} (med / p95)"
    print(header)
    for j in range(21):
        row = f"{ROTDATA_JOINTS[j]:<8}"
        for label in labels:
            v = results[label][j]
            if len(v) == 0:
                row += f"  {'n/a':>30}"
            else:
                row += f"  {np.median(v):>7.3f} / {np.percentile(v, 95):>7.3f}   "
        print(row)

    # Per-dataset global summary
    print(f"\n{'overall':<8}")
    for label in labels:
        allv = np.concatenate(results[label]) if results[label] else np.array([])
        if len(allv):
            print(f"  {label:<22}  median={np.median(allv):.3f} rad/s   "
                  f"p95={np.percentile(allv, 95):.3f} rad/s   "
                  f"mean={allv.mean():.3f} rad/s")

    # Cross-dataset ratio: VC vs HML walking (smoking gun for fps)
    if "VanCriekinge" in results and "HumanML3D (walking)" in results:
        print("\n  Per-joint VC/HML median ratio "
              "(uniform ratio across joints ⇒ fps error; non-uniform ⇒ real motion diff):")
        ratios = []
        for j in range(21):
            vc_med = np.median(results["VanCriekinge"][j]) if len(results["VanCriekinge"][j]) else np.nan
            hml_med = np.median(results["HumanML3D (walking)"][j]) if len(results["HumanML3D (walking)"][j]) else np.nan
            r = vc_med / hml_med if hml_med > 1e-6 else np.nan
            ratios.append(r)
            print(f"    {ROTDATA_JOINTS[j]:<8}  VC={vc_med:.3f}   HML={hml_med:.3f}   ratio={r:.2f}x")
        finite = np.array([r for r in ratios if np.isfinite(r)])
        if len(finite):
            print(f"\n    ratio summary: median={np.median(finite):.2f}x  "
                  f"std={finite.std():.2f}  min={finite.min():.2f}x  max={finite.max():.2f}x")
            print(f"    (low std ⇒ uniform inflation ⇒ fps suspect; "
                  f"high std ⇒ motion-specific)")


def plot_per_joint_angular_velocity(results, path):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"VanCriekinge": "#d62728", "HumanML3D (walking)": "#2ca02c", "100STYLE (FW)": "#1f77b4"}
    x = np.arange(21)
    width = 0.27
    for k, (label, per_joint) in enumerate(results.items()):
        meds = np.array([np.median(v) if len(v) else 0 for v in per_joint])
        ax.bar(x + (k - 1) * width, meds, width, label=label, color=colors.get(label))
    ax.set_xticks(x)
    ax.set_xticklabels(ROTDATA_JOINTS, rotation=45, ha="right")
    ax.set_ylabel(f"median joint angular velocity (rad/s @ {ASSUMED_FPS} fps)")
    ax.set_title("Per-joint median angular velocity magnitude\n"
                 "Uniform VC/HML ratio across joints ⇒ fps scaling bug; "
                 "non-uniform ⇒ real motion difference")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_angular_velocity_distribution(results, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"VanCriekinge": "#d62728", "HumanML3D (walking)": "#2ca02c", "100STYLE (FW)": "#1f77b4"}
    for label, per_joint in results.items():
        allv = np.concatenate(per_joint) if per_joint else np.array([])
        if len(allv) == 0:
            continue
        ax.hist(allv, bins=120, density=True, alpha=0.5, range=(0, 15),
                label=f"{label} (median={np.median(allv):.2f} rad/s)",
                color=colors.get(label))
    ax.set_xlabel(f"per-frame joint angular velocity (rad/s @ {ASSUMED_FPS} fps)")
    ax.set_ylabel("density")
    ax.set_title("Distribution of frame-to-frame joint angular velocity\n"
                 "If VC is uniformly shifted right, the conversion's fps is too high")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def section_of(dim):
    for name, lo, hi in SECTIONS:
        if lo <= dim < hi:
            return name
    return "?"


def main():
    mean, std = load_hml_stats()
    vc = load_vc_clips()
    hml_w = load_hml_walking_clips(limit=1500)
    style_w = load_style_walking_clips(limit=1500)

    if not vc:
        print("ERROR: no VC clips loaded")
        sys.exit(1)

    vc_norm = concat_normalised(vc, mean, std)
    hml_norm = concat_normalised(hml_w, mean, std)
    style_norm = concat_normalised(style_w, mean, std) if style_w else None

    norms = {
        "VanCriekinge": vc_norm,
        "HumanML3D (walking)": hml_norm,
    }
    if style_norm is not None:
        norms["100STYLE (FW)"] = style_norm

    # -------- numerical summary --------
    for label, n in norms.items():
        section_summary(label, n)

    print("\n\n===== TOP-15 MOST SHIFTED DIMS IN VC =====")
    for idx, m, s in top_shifted_dims(vc_norm, 15):
        hml_mean = hml_norm[:, idx].mean() if hml_norm is not None else float("nan")
        print(f"  dim {idx:3d} [{section_of(idx):<15}]  VC mean={m:+.3f}  VC std={s:.3f}   HML walking mean={hml_mean:+.3f}")

    print("\n\n===== PER-JOINT LOCAL_VEL MEAN (VC vs HML walking) =====")
    lv_vc = vc_norm[:, 193:259].reshape(-1, 22, 3)
    lv_hml = hml_norm[:, 193:259].reshape(-1, 22, 3)
    JOINTS = ["pelvis","lhip","rhip","sp1","lknee","rknee","sp2","lank","rank",
              "sp3","ltoe","rtoe","neck","lclav","rclav","head","lsh","rsh",
              "lelb","relb","lwri","rwri"]
    print(f"{'joint':<8}  {'VC.x':>8} {'VC.y':>8} {'VC.z':>8}  |  {'HML.x':>8} {'HML.y':>8} {'HML.z':>8}")
    for j in range(22):
        vm = lv_vc[:, j, :].mean(axis=0)
        hm = lv_hml[:, j, :].mean(axis=0)
        print(f"{JOINTS[j]:<8}  {vm[0]:>+8.3f} {vm[1]:>+8.3f} {vm[2]:>+8.3f}  |  {hm[0]:>+8.3f} {hm[1]:>+8.3f} {hm[2]:>+8.3f}")

    # Raw scale comparison — median per-frame root displacement
    print("\n\n===== RAW SCALE COMPARISON =====")
    for label, clips in [("VanCriekinge", vc), ("HumanML3D walking", hml_w), ("100STYLE FW", style_w)]:
        if not clips:
            continue
        all_rootv = np.concatenate([np.linalg.norm(m[:, 1:3], axis=1) for _, m, _ in clips])
        all_rooty = np.concatenate([m[:, 3] for _, m, _ in clips])
        all_lv = np.concatenate([np.linalg.norm(m[:, 193:259].reshape(-1, 22, 3), axis=2).mean(axis=1) for _, m, _ in clips])
        print(f"{label:<22}  median |root_lin_vel_xz|={np.median(all_rootv):.4f}   "
              f"median root_y={np.median(all_rooty):+.3f}   "
              f"median per-frame mean joint-vel mag={np.median(all_lv):.4f}")

    # -------- trajectory Z-axis alignment --------
    print("\n\n===== TRAJECTORY Z-AXIS ALIGNMENT =====")
    align_datasets = {}
    for label, clips in [("VanCriekinge", vc), ("HumanML3D (walking)", hml_w), ("100STYLE (FW)", style_w)]:
        if clips:
            align_datasets[label] = trajectory_alignment_stats(clips, label)

    # -------- plots --------
    print(f"\nSaving plots to {OUT_DIR}")
    plot_perdim_mean(norms, "Per-dim mean in HumanML3D-normalised space (VC vs HumanML3D walking vs 100STYLE FW)",
                     os.path.join(OUT_DIR, "01_perdim_mean.png"))
    plot_perdim_std(norms, "Per-dim std in HumanML3D-normalised space",
                    os.path.join(OUT_DIR, "02_perdim_std.png"))
    plot_section_histograms(norms, os.path.join(OUT_DIR, "03_section_histograms.png"))
    plot_joint_local_vel_mean(norms, os.path.join(OUT_DIR, "04_per_joint_local_vel_mean.png"))
    plot_root_traj_samples(vc, hml_w, style_w,
                           os.path.join(OUT_DIR, "05_sample_root_trajectories.png"))
    plot_velocity_magnitudes(norms, os.path.join(OUT_DIR, "06_velocity_magnitude_dist.png"))
    plot_raw_root_y_hist(vc, hml_w, style_w, mean, os.path.join(OUT_DIR, "07_raw_root_y_hist.png"))
    plot_raw_root_vel_hist(vc, hml_w, style_w, os.path.join(OUT_DIR, "08_raw_root_vel_hist.png"))
    plot_trajectory_alignment(align_datasets, os.path.join(OUT_DIR, "12_trajectory_alignment.png"))

    # -------- angular velocity diagnostic (fps test) --------
    angvel_results = {"VanCriekinge": per_joint_angular_velocity(vc)}
    if hml_w:
        angvel_results["HumanML3D (walking)"] = per_joint_angular_velocity(hml_w)
    if style_w:
        angvel_results["100STYLE (FW)"] = per_joint_angular_velocity(style_w)
    print_angular_velocity_report(angvel_results)
    plot_per_joint_angular_velocity(angvel_results, os.path.join(OUT_DIR, "13_per_joint_angvel.png"))
    plot_angular_velocity_distribution(angvel_results, os.path.join(OUT_DIR, "14_angvel_distribution.png"))

    print("Done.")


if __name__ == "__main__":
    main()
