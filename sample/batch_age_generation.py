#!/usr/bin/env python3
"""
Batch motion generation for all three age groups using the Age LoRA adapter.
Loads base model + LoRA once; generates all clips without reloading from disk.
Supports arbitrary --num_samples split into --batch_size chunks.

Usage:
    python -m sample.batch_age_generation \
        --model_path save/mdm/model000500000.pt \
        --lora_path  save/lora/Age/model000004004.pt \
        --output_dir save/samples_age \
        --num_samples 30 \
        --batch_size  10

Output layout:
    <output_dir>/Young/results.npy
    <output_dir>/MiddleAge/results.npy
    <output_dir>/Elderly/results.npy
"""

import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from utils.fixseed import fixseed
from utils.parser_util import (
    add_base_options, add_lora_options, add_sampling_options,
    add_generate_options, parse_and_load_from_model,
)
from utils.model_util import create_model_and_diffusion, load_saved_model, load_lora_to_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from diffusion.respace import SpacedDiffusion


# Token assignment must match the style order used during LoRA training:
#   --styles Young MiddleAge Elderly  =>  indices 0, 1, 2  =>  sks, hta, oue
AGE_GROUPS = [
    ("Young",     "sks"),
    ("MiddleAge", "hta"),
    ("Elderly",   "oue"),
]

MAX_FRAMES = 196
FPS = 20


def build_args():
    parser = ArgumentParser()
    add_base_options(parser)     # --seed, --batch_size, --device, --cuda
    add_lora_options(parser)     # --lora_finetune, --lora_rank, --styles, ...
    add_sampling_options(parser) # --model_path, --lora_path, --output_dir, --num_samples, --guidance_param
    add_generate_options(parser) # --motion_length
    args = parse_and_load_from_model(parser)
    # Force settings that are fixed for this generation script
    args.dataset = 'vancriekinge'
    args.styles = ['Young', 'MiddleAge', 'Elderly']
    args.lora_finetune = True
    return args


def run_batch(model, diffusion, texts, n_frames, device, guidance_param):
    """Run one diffusion forward pass. Returns (raw_sample [bs,263,1,T], lengths [bs])."""
    bs = len(texts)
    collate_args = [
        {'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames, 'text': t}
        for t in texts
    ]
    _, model_kwargs = collate(collate_args)
    if guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(bs, device=device) * guidance_param

    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            (bs, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    return sample.cpu(), model_kwargs['y']['lengths'].cpu().numpy()


def decode_to_xyz(raw_samples, inv_transform_fn, model):
    """
    Decode batched raw model output to XYZ skeleton positions.

    raw_samples : [N, 263, 1, T]  (hml_vec format, normalised)
    returns     : [N, 22, 3, T]   (xyz positions in metres)
    """
    # Denormalise: [N, 263, 1, T] -> permute -> [N, 1, T, 263]
    inv = inv_transform_fn(raw_samples.permute(0, 2, 3, 1)).float()
    # HumanML3D RIC -> [N, 1, T, 22, 3]
    xyz = recover_from_ric(inv, 22)
    # -> [N, T, 22, 3] -> permute -> [N, 22, 3, T]
    xyz = xyz.view(-1, *xyz.shape[2:]).permute(0, 2, 3, 1)
    # SMPL forward kinematics (consistent with generate.py)
    xyz = model.rot2xyz(
        x=xyz, mask=None, pose_rep='xyz', glob=True, translation=True,
        jointstype='smpl', vertstrans=True, betas=None, beta=0,
        glob_rot=None, get_rotations_back=False,
    )
    return xyz


def main():
    args = build_args()
    fixseed(args.seed)

    n_frames = min(MAX_FRAMES, int(args.motion_length * FPS))
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Abort early if any output file already exists — never overwrite
    for age_group, _ in AGE_GROUPS:
        npy_path = os.path.join(args.output_dir, age_group, 'results.npy')
        if os.path.exists(npy_path):
            raise FileExistsError(
                f"{npy_path} already exists. "
                "Use a different --output_dir or remove existing files first."
            )

    # Dataset — used for inv_transform and model construction (same HumanML3D stats)
    print("Loading dataset...")
    data = get_dataset_loader(
        name='vancriekinge',
        batch_size=1,
        num_frames=MAX_FRAMES,
        split='test',
        hml_mode='text_only',
        styles=tuple(args.styles),
    )
    inv_transform = data.dataset.inv_transform

    # Build model and load all weights ONCE
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data, DiffusionClass=SpacedDiffusion)

    print(f"Loading base model from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    model.add_LoRA_adapters()
    print(f"Loading LoRA adapter from [{args.lora_path}]...")
    load_lora_to_model(model, args.lora_path)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # __getattr__ forwards rot2xyz

    model.to(device)
    model.eval()

    # ── Generation loop ─────────────────────────────────────────────────────
    total_clips = args.num_samples * len(AGE_GROUPS)
    with tqdm(total=total_clips, unit="clip", ncols=80) as pbar:
        for age_group, token in AGE_GROUPS:
            pbar.set_description(f"{age_group:10s} [{token}]")

            output_dir = os.path.join(args.output_dir, age_group)
            os.makedirs(output_dir, exist_ok=True)

            prompt = f"A person is walking in {token} style."
            all_raw: list[torch.Tensor] = []
            all_lengths: list[np.ndarray] = []

            remaining = args.num_samples
            while remaining > 0:
                bs = min(args.batch_size, remaining)
                raw, lengths = run_batch(
                    model, diffusion, [prompt] * bs, n_frames, device, args.guidance_param
                )
                all_raw.append(raw)
                all_lengths.append(lengths)
                remaining -= bs
                pbar.update(bs)

            # Decode all clips for this age group in one pass
            all_raw_t = torch.cat(all_raw, dim=0)           # [N, 263, 1, T]
            all_lengths_arr = np.concatenate(all_lengths)   # [N]
            xyz = decode_to_xyz(all_raw_t, inv_transform, model)  # [N, 22, 3, T]

            npy_path = os.path.join(output_dir, 'results.npy')
            np.save(npy_path, {
                'motion':          xyz.cpu().numpy(),          # [N, 22, 3, T]
                'text':            [prompt] * args.num_samples,
                'lengths':         all_lengths_arr,
                'num_samples':     args.num_samples,
                'num_repetitions': 1,
            })
            tqdm.write(f"  Saved {args.num_samples} clips -> {npy_path}")

    print(f"\nDone. Results saved under {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
