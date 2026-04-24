#!/usr/bin/env python3
"""
Batch motion generation with multi-GPU support.

LoRA rank is auto-detected from the LoRA checkpoint's args.json.
Outputs one batch_{NNNN}.npy per batch (exactly batch_size clips each).

Usage:
    python -m sample.batch_age_generation \
        --model_path save/mdm/model000500000.pt \
        --lora_path  save/lora/AgeEncoderR5D100/model000004023.pt \
        --lora_finetune \
        --token oue \
        --num_samples 20 \
        --batch_size 10 \
        --n_gpu 2 \
        --output_dir save/samples/elderly

Output layout:
    <output_dir>/batch_0000.npy   # batch_size clips
    <output_dir>/batch_0001.npy   # batch_size clips
    ...
"""

import json
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from argparse import ArgumentParser
from tqdm import tqdm

from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from diffusion.respace import SpacedDiffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_lora_to_model, load_saved_model
from utils.parser_util import (
    add_base_options,
    add_generate_options,
    add_sampling_options,
    parse_and_load_from_model,
)

MAX_FRAMES = 196
FPS = 20


def _add_lora_options(parser):
    group = parser.add_argument_group("lora")
    group.add_argument("--lora_finetune", action="store_true")
    group.add_argument(
        "--lora_rank", default=None, type=int,
        help="Auto-detected from the LoRA checkpoint's args.json; rarely needed."
    )
    group.add_argument("--lora_layer", default=-100, type=int)
    group.add_argument("--no_lora_q", action="store_true")
    group.add_argument("--lora_ff", action="store_true")
    # --styles intentionally omitted: token is passed via --token instead.


def _add_script_options(parser):
    group = parser.add_argument_group("batch_script")
    group.add_argument(
        "--token", required=True, type=str,
        help="Style token embedded in the prompt, e.g. 'oue'."
    )
    group.add_argument(
        "--n_gpu", default=1, type=int,
        help="Number of GPUs to use in parallel (GPUs device..device+n_gpu-1)."
    )


def build_args():
    parser = ArgumentParser()
    add_base_options(parser)
    _add_lora_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    _add_script_options(parser)
    args = parse_and_load_from_model(parser)

    args.styles = []  # not used; token replaces style-based dispatch

    if args.lora_path:
        args.lora_finetune = True

    if args.lora_finetune:
        lora_args_path = os.path.join(os.path.dirname(args.lora_path), "args.json")
        if not os.path.exists(lora_args_path):
            raise FileNotFoundError(
                f"Cannot auto-detect lora_rank: {lora_args_path} not found. "
                "Pass --lora_rank explicitly or ensure args.json exists next to the LoRA checkpoint."
            )
        with open(lora_args_path) as f:
            lora_meta = json.load(f)
        if "lora_rank" not in lora_meta:
            raise KeyError(f"'lora_rank' not found in {lora_args_path}.")
        args.lora_rank = lora_meta["lora_rank"]
        print(f"Auto-detected lora_rank={args.lora_rank} from {lora_args_path}")
    elif args.lora_rank is None:
        args.lora_rank = 5

    return args


def _decode_to_xyz(raw_samples, inv_transform_fn, model):
    """raw_samples: [N, 263, 1, T] → [N, 22, 3, T]"""
    inv = inv_transform_fn(raw_samples.permute(0, 2, 3, 1)).float()
    xyz = recover_from_ric(inv, 22)
    xyz = xyz.view(-1, *xyz.shape[2:]).permute(0, 2, 3, 1)
    xyz = model.rot2xyz(
        x=xyz, mask=None, pose_rep="xyz", glob=True, translation=True,
        jointstype="smpl", vertstrans=True, betas=None, beta=0,
        glob_rot=None, get_rotations_back=False,
    )
    return xyz


def _worker(rank, args, prompt, all_chunks, n_frames, output_dir):
    """One process per GPU. Generates and writes each assigned batch to disk."""
    gpu_id = args.device + rank
    dist_util.setup_dist(gpu_id)
    device = dist_util.dev()
    fixseed(args.seed + rank)

    my_batches = all_chunks[rank]

    # args.dataset is shadowed from the base MDM's args.json → 'humanml'.
    # HumanML3D and VanCriekinge share the same inv_transform stats, so this is correct.
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=1,
        num_frames=MAX_FRAMES,
        split="test",
        hml_mode="text_only",
        styles=(),
    )
    inv_transform = data.dataset.inv_transform

    model, diffusion = create_model_and_diffusion(args, data, DiffusionClass=SpacedDiffusion)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.lora_finetune:
        model.add_LoRA_adapters()
        load_lora_to_model(model, args.lora_path)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)

    model.to(device).eval()

    texts = [prompt] * args.batch_size
    collate_args = [
        {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames, "text": t}
        for t in texts
    ]
    _, model_kwargs = collate(collate_args)
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=device) * args.guidance_param
        )
    lengths = model_kwargs["y"]["lengths"].cpu().numpy()

    shape = (args.batch_size, model.njoints, model.nfeats, n_frames)
    n_diffusion_steps = diffusion.num_timesteps

    bar_fmt = "{desc}: {n:.2f}/{total:.0f} batches [{elapsed}<{remaining}]"
    with tqdm(
        total=len(my_batches),
        desc=f"GPU {gpu_id}",
        position=rank,
        leave=True,
        bar_format=bar_fmt,
    ) as pbar:
        for local_i, batch_idx in enumerate(my_batches):
            with torch.no_grad():
                for step_i, out in enumerate(diffusion.p_sample_loop_progressive(
                    model,
                    shape,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                    init_image=None,
                    progress=False,
                    const_noise=False,
                )):
                    pbar.n = local_i + (step_i + 1) / n_diffusion_steps
                    pbar.refresh()

            assert out is not None  # p_sample_loop_progressive always yields ≥1 step
            raw = out["sample"].cpu()
            xyz = _decode_to_xyz(raw, inv_transform, model)

            out_path = os.path.join(output_dir, f"batch_{batch_idx:04d}.npy")
            np.save(
                out_path,
                {
                    "motion": xyz.cpu().numpy(),  # [B, 22, 3, T]
                    "text": texts,
                    "lengths": lengths,
                    "batch_idx": batch_idx,
                },
            )

            pbar.n = local_i + 1.0
            pbar.refresh()


def main():
    args = build_args()
    fixseed(args.seed)

    n_frames = min(MAX_FRAMES, int(args.motion_length * FPS))

    template = args.text_prompt if args.text_prompt else "A person is walking forward in {token} style."
    prompt = template.replace("{token}", args.token)

    assert args.num_samples % args.batch_size == 0, (
        f"--num_samples ({args.num_samples}) must be divisible by "
        f"--batch_size ({args.batch_size})."
    )
    total_batches = args.num_samples // args.batch_size

    available_gpus = torch.cuda.device_count()
    assert available_gpus >= args.device + args.n_gpu, (
        f"Requested GPUs {args.device}–{args.device + args.n_gpu - 1} "
        f"but only {available_gpus} CUDA device(s) available."
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(total_batches):
        p = os.path.join(args.output_dir, f"batch_{i:04d}.npy")
        if os.path.exists(p):
            raise FileExistsError(
                f"{p} already exists. Use a different --output_dir or remove existing files."
            )

    # Round-robin: GPU rank r handles batches r, r+n_gpu, r+2*n_gpu, ...
    all_chunks = [list(range(r, total_batches, args.n_gpu)) for r in range(args.n_gpu)]

    print(
        f"Generating {args.num_samples} clips ({total_batches} batches × {args.batch_size}) "
        f"across {args.n_gpu} GPU(s).\nPrompt: '{prompt}'"
    )

    mp.spawn(
        _worker,
        args=(args, prompt, all_chunks, n_frames, args.output_dir),
        nprocs=args.n_gpu,
        join=True,
    )

    print(f"\nDone. {total_batches} batch file(s) saved to {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
