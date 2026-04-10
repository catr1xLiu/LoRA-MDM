import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

METADATA_ROOT = os.path.join(
    os.path.dirname(__file__), "../../dataset/VanCriekinge/metadata"
)

MOTION_DIR = os.path.join(
    os.path.dirname(__file__), "../../dataset/VanCriekinge/motion"
)

HML3D_STATS_DIR = os.path.join(os.path.dirname(__file__), "../../dataset/HumanML3D")


def age_to_group(age: int) -> str:
    if age < 35:
        return "Young"
    if age < 60:
        return "MiddleAge"
    return "Elderly"


def parse_filename(filename):
    """
    Parse subject_id and trial_name from npy filename.
    e.g. 'SUBJ01_SUBJ1_1_humanml3d_22joints.npy' -> ('SUBJ01', 'SUBJ1_1')
    """
    stem = filename.replace("_humanml3d_22joints.npy", "")
    parts = stem.split("_")
    subject_id = parts[0]
    trial_name = "_".join(parts[1:3])
    return subject_id, trial_name


class VanCriekingeDataset(Dataset):

    def __init__(self, styles, mode, motion_type_to_exclude=()):
        assert styles is not None

        del motion_type_to_exclude  # VanCriekinge has only walking; no motion types to exclude
        self.styles = styles
        self.mode = mode
        self.tokens = ["sks", "hta", "oue", "asar", "nips"]
        assert len(self.styles) <= len(self.tokens)

        self.max_motion_length = 196
        self.min_motion_length = 40
        self.unit_length = 4
        self.pointer = 0

        self.mean = np.load(os.path.join(HML3D_STATS_DIR, "Mean.npy"))
        self.std = np.load(os.path.join(HML3D_STATS_DIR, "Std.npy"))

        data_dict = {}
        new_name_list = []
        length_list = []

        npy_files = sorted(f for f in os.listdir(MOTION_DIR) if f.endswith(".npy"))

        for filename in npy_files:
            subject_id, trial_name = parse_filename(filename)

            json_path = os.path.join(METADATA_ROOT, f"{trial_name}_metadata.json")
            if not os.path.exists(json_path):
                print(
                    "Warning: Skipping %s as no metadata json file found in %s."
                    % (filename, json_path)
                )
                continue

            with open(json_path, "r") as f:
                meta = json.load(f)

            if meta["trial_index"] == 0:
                continue

            age_group = age_to_group(meta["age"])
            if age_group not in styles:
                print(
                    "Warning: Skipping %s as its age group %s is not in the specified styles."
                    % (filename, age_group)
                )
                continue

            motion = np.load(os.path.join(MOTION_DIR, filename))
            if len(motion) < self.min_motion_length:
                print(
                    "Info: Skipping %s as its frame length %s is under the minimum threshold (%s frames)."
                    % (filename, len(motion), self.min_motion_length)
                )
                continue

            token = self.tokens[self.styles.index(age_group)]
            clip_id = filename.replace("_humanml3d_22joints.npy", "")

            data_dict[clip_id] = {
                "motion": motion,
                "length": len(motion),
                "style": age_group,
                "token": token,
                "age": meta["age"],
                "subject_id": subject_id,
            }
            new_name_list.append(clip_id)
            length_list.append(len(motion))

        assert len(new_name_list) > 0, f"No samples found for styles: {styles}"

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        self.data_dict = data_dict
        self.name_list = name_list
        self.length_arr = np.array(length_list)
        self.nfeats = 263

        style_counts = {}
        for v in data_dict.values():
            style_counts[v["style"]] = style_counts.get(v["style"], 0) + 1
        print(f"VanCriekingeDataset loaded {len(name_list)} clips: {style_counts}")

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self, data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion = data["motion"]
        m_length = data["length"]
        token = data["token"]
        style = data["style"]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        m_length = min(self.max_motion_length, m_length)
        start = random.randint(0, len(motion) - m_length)
        motion = motion[start : start + m_length]

        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )

        return {
            "inp": torch.tensor(motion.T).float().unsqueeze(1),  # [263, 1, 196]
            "lengths": m_length,
            "text": f"A person is walking forward in {token} style.",
            "action": torch.tensor(0),
            "action_text": "walking",
            "style": style,
        }
