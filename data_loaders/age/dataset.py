import os
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin
import codecs as cs
import random

class AgeMotionDataset(Dataset):
    def __init__(self, data_root, split='train', max_motion_length=196):
        self.data_root = data_root
        self.max_motion_length = max_motion_length
        self.unit_length = 4 
        
        self.motion_dir = pjoin(data_root, split, "motions")
        self.text_dir = pjoin(data_root, split, "texts")
        self.age_dir = pjoin(data_root, split, "ages")
        self.split_file = pjoin(data_root, f"{split}.txt")
        
        # Load Mean/Std
        mean_path = pjoin(data_root, "Mean.npy")
        std_path = pjoin(data_root, "Std.npy")
        try:
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
        except FileNotFoundError:
            print(f"Warning: Mean/Std not found at {mean_path}. Using default.")
            self.mean = 0
            self.std = 1
        
        self.id_list = []
        if os.path.exists(self.split_file):
            with cs.open(self.split_file, 'r') as f:
                for line in f.readlines():
                    self.id_list.append(line.strip())
        
        self.data_dict = {}
        # We will rebuild name_list and length_list after sorting
        
        print(f"Loading VanCriekinge Dataset from {self.split_file}...")
        
        n_standing_skipped = 0
        n_walking_short = 0
        n_success = 0
        
        for name in self.id_list:
            if '_0_' in name:
                n_standing_skipped += 1
                continue

            try:
                motion_path = pjoin(self.motion_dir, name + '.npy')
                if not os.path.exists(motion_path): continue
                
                motion = np.load(motion_path)
                
                # Allow slightly shorter clips (e.g. 10 frames)
                if len(motion) < 10: 
                    n_walking_short += 1
                    continue
                    
                text_path = pjoin(self.text_dir, name + '.txt')
                if not os.path.exists(text_path): 
                    text_data = "an adult is walking"
                else:
                    with open(text_path, 'r') as f: text_data = f.read().strip()
                
                age_path = pjoin(self.age_dir, name + '.txt')
                if not os.path.exists(age_path):
                    age_path = pjoin(self.age_dir, name)
                    if not os.path.exists(age_path): continue
                        
                with open(age_path, 'r') as f:
                    content = f.read().strip()
                    if not content: continue
                    age_raw = float(content.split()[0])
                
                self.data_dict[name] = {
                    'motion': motion,
                    'text': text_data,
                    'age': age_raw / 100.0,
                    'length': len(motion)
                }
                n_success += 1
                
            except Exception as e:
                print(f"Failed to load {name}: {e}")

        print(f"--- DATASET REPORT ---")
        print(f"Successfully Loaded:   {n_success}")

        # 1. Get all names
        all_names = list(self.data_dict.keys())
        
        # 2. Sort them based on motion length
        all_names.sort(key=lambda x: self.data_dict[x]['length'])
        
        # 3. Rebuild the lists in sorted order
        self.name_list = all_names
        self.length_list = [self.data_dict[x]['length'] for x in all_names]
        self.length_arr = np.array(self.length_list)
        
        self.pointer = 0
        
        if len(self.data_dict) > 0:
            self.nfeats = list(self.data_dict.values())[0]['motion'].shape[1]
        else:
            self.nfeats = 263

        # Call reset_max_len to initialize self.max_length, but we override the pointer logic
        self.reset_max_len(1)

    def reset_max_len(self, length):
        # OVERRIDE: Do NOT filter out short motions.
        # Standard MDM uses: self.pointer = np.searchsorted(self.length_arr, length)
        # We force pointer to 0 so we keep ALL data.
        self.pointer = 0 
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def get_mean_std(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        
        motion, m_length, caption, age = data['motion'], data['length'], data['text'], data['age']

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
            
        frame_idx = random.randint(0, len(motion) - m_length)
        motion = motion[frame_idx:frame_idx+m_length]

        motion = (motion - self.mean) / self.std

        # PADDING LOGIC: This saves us.
        # If the clip is shorter than the target (196), we zero-pad it.
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            ], axis=0)

        return {
            'inp': torch.tensor(motion.T).float().unsqueeze(1),
            'text': caption,
            'lengths': m_length,
            'tokens': None,
            'age': age
        }