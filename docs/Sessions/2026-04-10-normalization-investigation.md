# Normalization Investigation — 2026-04-10

## Background

Traced the normalization pipeline end-to-end for LoRA fine-tuning and inference to identify why re-trained LoRA adapters produced degraded motion quality compared to the downloaded official checkpoints.

---

## Observations

### 1. VanCriekinge (VC) LoRA — trained with `main` branch (HumanML3D stats, current)
Motion is disrupted. Walking forward becomes dancing forward. Standing still becomes dancing with strange random movements. Style is completely overridden by artifacts.

### 2. 100style LoRA — trained with `main` branch (100STYLE-SMPL stats, original official code)
Motion is significantly worse. The human behaves randomly regardless of the input prompt. Standing still produces standing with slight shaking. Any other motion type (walking, running) collapses into random noise movement.

### 3. 100style LoRA — trained on `normalization-fix` branch (HumanML3D stats)
Motion is completely normal. All motion types behave as expected. Style is correctly applied without artifacts.

---

## Root Cause

The official `StyleMotionDataset` (`data_loaders/style/dataset.py`) loaded its normalisation statistics from the dataset-specific files:

```
dataset/100STYLE-SMPL/Mean.npy
dataset/100STYLE-SMPL/Std.npy
```

However, the base MDM model was pre-trained on HumanML3D data, and inference always denormalises using:

```
dataset/HumanML3D/Mean.npy
dataset/HumanML3D/Std.npy
```

The two stat files are numerically distinct (max absolute difference in Std: **347**). This created a mismatch between the space the LoRA was trained in and the space inference assumed. The prior preservation loss (which uses HumanML3D-normalised data) was also operating in a different space than the style loss, giving the LoRA conflicting gradient signals.

The VanCriekinge dataset (`data_loaders/vancriekinge/dataset.py`) was written correctly from the start — it explicitly loads from `dataset/HumanML3D/` — so its normalisation is consistent with both the base model and inference. The degraded VC results in observation 1 are therefore caused by a separate issue unrelated to normalisation.

---

## Bugfixes Applied (`normalization-fix` branch)

### 1. `data_loaders/style/dataset.py` — normalisation source corrected
Changed `StyleMotionDataset` to load mean/std from `dataset/HumanML3D/` instead of `dataset/100STYLE-SMPL/`, matching `VanCriekingeDataset` and the inference `inv_transform`.

```python
# Before
mean = np.load(path + "/Mean.npy")
std  = np.load(path + "/Std.npy")

# After
hml3d_stats_dir = os.path.join(os.path.dirname(__file__), "../../dataset/HumanML3D")
mean = np.load(os.path.join(hml3d_stats_dir, "Mean.npy"))
std  = np.load(os.path.join(hml3d_stats_dir, "Std.npy"))
```

### 2. `sample/generate.py` — removed implicit LoRA path discovery
Removed the fallback that used `args.styles[0]` as a search key to auto-discover a LoRA checkpoint when `--lora_path` was not specified. `--lora_path` is now required whenever `--lora_finetune` is set.

### 3. `utils/parser_util.py` — explicit validation
Added a `ValueError` in `generate_args()` when `--lora_finetune` is set without `--lora_path`, and corrected the copy-paste help text on `--lora_path`.

### 4. `utils/model_util.py` — removed hidden string-to-path conversion
`load_lora_to_model` no longer accepts a style name string and silently calls `find_lora_path` internally. All callers must now pass an explicit `.pt` file path. `find_lora_path` is retained as a standalone utility for the evaluation script.
