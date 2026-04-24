# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of the original **LoRA-MDM** ("Dance Like a Chicken: Low-Rank Stylization for Human Motion Diffusion", arXiv:2503.19557) extended for **age-conditioned motion generation** research. The primary research goal is to train LoRA adapters on the **Van Criekinge (VC) gait dataset** — a clinical motion capture dataset spanning ages 20–95 — to generate age-conditioned walking motions. The companion data processing pipeline lives at `../LoRA-MDM-Age-Dataset/` (VC Pipeline v2).

The base approach: a MDM (Motion Diffusion Model) is pre-trained on HumanML3D, then lightweight LoRA adapters (~33K params at rank 5) are fine-tuned per style/age-group using the 100STYLE or VanCriekinge datasets.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate lora_mdm
```

Required data (download separately):
- HumanML3D dataset → `dataset/HumanML3D/`
- 100STYLE-SMPL dataset → `dataset/100STYLE-SMPL/`
- VanCriekinge processed motion → `dataset/VanCriekinge/` (from VC Pipeline v2)
- SMPL body models: `bash prepare/download_smpl_files.sh`
- GloVe embeddings: `bash prepare/download_glove.sh`
- T2M evaluators: `bash prepare/download_t2m_evaluators.sh`

## Key Commands

**Train base MDM:**
```bash
python -m train.train_mdm \
  --save_dir save/mdm \
  --dataset humanml \
  --diffusion_steps 1000 \
  --arch trans_dec \
  --text_encoder_type bert \
  --mask_frames
```

**Train LoRA adapter on 100STYLE:**
```bash
python -m train.train_mdm \
  --save_dir save/lora/Chicken \
  --num_steps 4000 \
  --diffusion_steps 1000 \
  --dataset 100style \
  --arch trans_dec \
  --text_encoder_type bert \
  --starting_checkpoint save/mdm/model000500000.pt \
  --styles Chicken \
  --lora_finetune \
  --mask_frames \
  --lambda_prior_preserv 0.25
```

**Train LoRA adapter on VanCriekinge (age groups):**
```bash
python -m train.train_mdm \
  --save_dir save/lora/AgeEncoder \
  --num_steps 4000 \
  --diffusion_steps 1000 \
  --dataset vancriekinge \
  --arch trans_dec \
  --text_encoder_type bert \
  --starting_checkpoint save/mdm/model000500000.pt \
  --styles Young MiddleAge Elderly \
  --lora_finetune \
  --mask_frames \
  --lambda_prior_preserv 0.25
```

**Generate from text prompt:**
```bash
python -m sample.generate \
  --lora_finetune \
  --model_path save/mdm/model000500000.pt \
  --lora_path save/lora/Chicken/model000004000.pt \
  --styles Chicken \
  --text_prompt "A person is walking in sks style." \
  --output_dir save/out/
```

**Generate from test set:**
```bash
python -m sample.generate \
  --lora_finetune \
  --model_path save/mdm/model000500000.pt \
  --lora_path save/lora/Chicken/model000004000.pt \
  --styles Chicken \
  --num_samples 10 \
  --num_repetitions 3 \
  --output_dir save/out/
```

**Evaluate:**
```bash
python -m eval.eval_lora_mdm \
  --model_path save/mdm/model000500000.pt \
  --lora_path save/lora/Chicken/model000004000.pt \
  --lora_finetune \
  --lora_rank 5 \
  --classifier_style_group All
```

**Render SMPL mesh from generated output:**
```bash
python -m visualize.render_mesh --input_path /path/to/sample##_rep##.mp4
```

**Animate joints (stick figure):**
```bash
python -m visualize.animate_joints --input_path /path/to/results.npy
```

## Argument Shadowing — Critical Gotchas

`generate_args()` and `evaluation_parser()` in `utils/parser_util.py` call `parse_and_load_from_model()`, which reads `args.json` from the checkpoint directory and **silently overwrites** CLI flags for three argument groups: `dataset`, `model`, and `diffusion`. This means many flags you pass on the command line are ignored during inference/eval. Here is the precise breakdown:

### Overwritten at inference (loaded from `args.json`, CLI value ignored)

| Group | Args overwritten |
|-------|-----------------|
| `model` | `--arch`, `--text_encoder_type`, `--emb_trans_dec`, `--layers`, `--latent_dim`, `--cond_mask_prob`, `--mask_frames`, `--lambda_rcxyz`, `--lambda_vel`, `--lambda_fc`, `--lambda_prior_preserv`, `--unconstrained`, `--emb_before_mask`, `--pos_embed_max_len`, `--use_ema` |
| `dataset` | `--dataset`, `--data_dir` |
| `diffusion` | `--noise_schedule`, `--diffusion_steps`, `--sigma_small` |

Note: `--diffusion_steps` prints a warning if overwritten but still overwrites.

### NOT overwritten — must match training manually

The `lora` group is **not** in the override list. These args must be passed explicitly and must match the values used during training, or you will get silent shape mismatches when loading the LoRA checkpoint:

| Arg | Risk if wrong |
|-----|--------------|
| `--lora_rank` | **Critical** — if rank differs from training, `load_lora()` will error on a shape mismatch or silently load garbage weights. Always check the `args.json` of the LoRA checkpoint to confirm the rank. |
| `--lora_finetune` | Must be passed at inference; not inferred from checkpoint. Missing this flag means the LoRA adapter is never loaded. |
| `--styles` | Must match the style list used during training. Used to assign token strings. |
| `--no_lora_q` | If the model was trained with `--no_lora_q`, inference must also use it; otherwise the LoRA layer structure doesn't match the checkpoint. |
| `--lora_ff` | Same — must match training if feed-forward LoRA layers were enabled. |
| `--lora_layer` | Must match training. Default is −100 (all layers). |

**Practical rule:** before running inference on a LoRA checkpoint, read the `args.json` next to it and copy the LoRA-group args verbatim.

### `--lora_path` is required

Since the `normalization-fix` branch, `--lora_path` is mandatory whenever `--lora_finetune` is set. There is no longer a fallback auto-discovery from `--styles`. Omitting it raises a `ValueError` immediately.

## Architecture

### Data Flow

```
Text prompt
  → CLIP or BERT encoder → latent embedding
  → MDM Transformer (conditioned denoising) + LoRA adapters
  → 263-dim HML motion vector (nframes × njoints × nfeats)
  → recover_from_ric() → XYZ joint positions (22 joints × 3)
  → SMPL rotation → mesh / stick figure animation
```

### Motion Representation

Motions use **HML vec format**: 263-dimensional vectors encoding joint rotations + velocities in relative inverted coordinates (RIC). The post-processing chain converts this to 3D XYZ positions via `data_loaders/humanml/utils/paramUtil.py` and `model/rotation2xyz.py`.

### MDM Model (`model/mdm.py`)

Transformer-based denoiser:
- Input: noisy motion `(batch, njoints, nfeats, nframes)` + diffusion timestep + text condition
- **trans_enc**: Encoder-only (motion + condition tokens processed jointly)
- **trans_dec**: Decoder with cross-attention (motion as query, text as key/value) — primary architecture used in this project
- Text encoders: frozen CLIP ViT-B/32 (512d) or DistilBERT (768d→latent_dim), set by `--text_encoder_type`

### LoRA Integration (`lora_pytorch/`, `utils/model_util.py`)

LoRA adapters are injected into Transformer linear layers. Key functions in `utils/model_util.py`:
- `create_model_and_diffusion()` — builds base model
- `load_model_wo_clip()` — loads base checkpoint
- `load_lora_to_model()` — injects LoRA and loads adapter weights (requires an explicit `.pt` path)
- `load_lora()` / `save_lora()` — serialize/deserialize only the LoRA parameters

All base model parameters are frozen during LoRA training; only `lora_A` and `lora_B` matrices are updated.

### Dataset Loaders (`data_loaders/`)

- `get_data.py` — factory returning the right DataLoader from `--dataset`
- `humanml/data/dataset.py` — HumanML3D text-motion pairs (train/val/test)
- `style/dataset.py` — 100STYLE dataset
- `vancriekinge/dataset.py` — Van Criekinge gait dataset; maps age to group (`Young` <35, `MiddleAge` <60, `Elderly` ≥60); skips T-pose trials (`trial_index == 0`); skips clips <40 frames
- `tensors.py` — custom collate functions for variable-length motion sequences

### VanCriekinge Dataset (`data_loaders/vancriekinge/dataset.py`)

Data lives at `dataset/VanCriekinge/motion/` (`.npy` files) and `dataset/VanCriekinge/metadata/` (per-trial JSON). The loader:
- Assigns one of five token strings (`sks`, `hta`, `oue`, `asar`, `nips`) to each age group in the order they appear in `--styles`
- Hard-codes text prompt as `"A person is walking forward in {token} style."`
- Normalises using HumanML3D stats (`dataset/HumanML3D/Mean.npy`, `Std.npy`), same as 100STYLE

### Training Loop (`train/training_loop.py`)

- `--lora_finetune`: freezes base weights, trains only LoRA adapters
- `--lambda_prior_preserv`: weight for prior preservation loss (mixes HumanML3D batches alongside style batches). Set to 0.0 for vanilla fine-tuning, 0.25 for evaluation runs.
- `--mask_frames`: masks padding frames in the loss — always use this
- Checkpoints every 50K steps by default (`--save_interval`)

### Diffusion (`diffusion/gaussian_diffusion.py`)

Cosine noise schedule, 1000 training timesteps. Classifier-free guidance in `model/cfg_sampler.py` runs the model twice (conditioned + unconditioned) and blends with scale `--guidance_param` (default 2.5).

## Dataset Normalization

**All datasets** (HumanML3D, 100STYLE, VanCriekinge) are normalized with the same `dataset/HumanML3D/Mean.npy` and `dataset/HumanML3D/Std.npy`. This is intentional: the base MDM was pre-trained in HumanML3D-normalised space, and inference always denormalises with those same stats. Using any other stats causes a train/inference distribution mismatch that produces corrupted motion output. The `normalization-fix` branch corrected `data_loaders/style/dataset.py` which had been using dataset-specific stats (`dataset/100STYLE-SMPL/Mean.npy`).

## Current Research Status — Session Reports

Session reports are in `docs/Sessions/`. Two key investigations from 2026-04-10:

### 1. Normalization Bug Fix (`docs/Sessions/2026-04-10-normalization-investigation.md`)

**Problem:** 100STYLE LoRA adapters trained on `main` produced random/broken motion. VanCriekinge adapters produced "dancing" artifacts despite correct normalization.

**Root cause:** `data_loaders/style/dataset.py` used `dataset/100STYLE-SMPL/Mean.npy` (max diff in Std: 347) instead of the HumanML3D stats used at inference, creating contradictory gradient signals with the prior preservation loss.

**Fixes applied on `normalization-fix` branch:**
1. `data_loaders/style/dataset.py` → now uses HumanML3D stats
2. `sample/generate.py` → removed implicit LoRA auto-discovery; `--lora_path` is now required
3. `utils/parser_util.py` → added `ValueError` when `--lora_finetune` is set without `--lora_path`
4. `utils/model_util.py` → `load_lora_to_model` no longer accepts a style name string; callers must pass an explicit `.pt` path

After this fix, 100STYLE LoRA training produces clean motion.

### 2. VanCriekinge Data Quality Diagnosis (`docs/Sessions/2026-04-10-vc-manifold-diagnosis.md`)

VC LoRA adapters still produce "on drugs" jittery output even after the normalization fix. A systematic distributional analysis (see `docs/Sessions/diagnose_vc_manifold.py`) found three root causes:

**Critical — Pelvis Y drift:** 43% of VC clips (184/426) have the pelvis rising by >10 cm across the clip; worst case is +1.13 m over 3 seconds. This is physically impossible for ground-level walking and is a data corruption artifact from the VC→HumanML3D conversion pipeline in `../LoRA-MDM-Age-Dataset/`. The converter likely lacks a per-frame floor-contact re-anchoring step. The LoRA learns "move the pelvis upward" as the dominant signal, which inference reproduces as incoherent motion.

**High — Walking speed is a 3σ outlier:** VC median speed is 1.26 m/s (realistic healthy adult walking), but HumanML3D's "walking" text distribution has a median of only 0.43 m/s. VC walking sits at the 97th percentile of the full HumanML3D training distribution. With rank-5 capacity and prior preservation fighting back, the adapter cannot bridge this gap.

**Secondary:** foot contact flags are mostly inactive (median −0.78 vs −0.26 in HumanML3D) because the reconstructed feet rarely touch the floor — a downstream consequence of the pelvis drift.

**Recommended next steps before retraining any VC LoRA:**
1. Filter out drifting clips: reject any clip where `abs(root_y[-1] - root_y[0]) > 0.05 m` or where the lowest reconstructed foot exceeds 0.1 m at any point (~40% of clips)
2. Fix the conversion pipeline in `../LoRA-MDM-Age-Dataset/` — check pelvis extraction and floor-anchoring in `3_export_humanml3d.py`
3. After the drift fix, if speed mismatch still causes training friction: increase `--lora_rank` to 16–32, or lower `--lambda_prior_preserv` to ~0.3

Diagnostic plots are in `docs/Sessions/assets/2026-04-10-vc-manifold/`. Plots 10 (`10_root_y_drift.png`) and 11 (`11_feet_levitation.png`) are the key evidence.

## Important File Locations

| Purpose | Path |
|---------|------|
| Model architecture | `model/mdm.py` |
| LoRA wrapper | `lora_pytorch/lora.py` |
| Model creation/loading | `utils/model_util.py` |
| All CLI arguments (with shadowing logic) | `utils/parser_util.py` |
| Training loop | `train/training_loop.py` |
| Generation script | `sample/generate.py` |
| Evaluation pipeline | `eval/eval_lora_mdm.py` |
| VanCriekinge dataset loader | `data_loaders/vancriekinge/dataset.py` |
| Style (100STYLE) dataset loader | `data_loaders/style/dataset.py` |
| HumanML3D loader | `data_loaders/humanml/data/dataset.py` |
| Motion post-processing | `data_loaders/humanml/utils/paramUtil.py` |
| VC data processing pipeline | `../LoRA-MDM-Age-Dataset/` |

## Output Format

`sample/generate.py` saves:
- `results.npy` — raw motion array `(nsamples, nrep, njoints, 3, nframes)`
- `sample##_rep##.mp4` — stick figure animation per sample/repetition
- `prompt_*.txt` / `lengths_*.npy` — associated metadata
