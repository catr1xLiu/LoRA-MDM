"""
Validation Script: Test Age Conditioning by Generating at Different Ages

This script:
1. Loads a trained model with age conditioning
2. Generates motions with the same text prompt but different age values
3. Measures the root velocity of generated motions
4. Verifies that age=young produces faster motion than age=old

Usage:
    python validate_age_conditioning.py \
        --model_path save/lora_age_test/model000500.pt \
        --output_dir save/age_validation/ \
        --text_prompt "A person is walking in sks style."
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join as pjoin


def compute_root_velocity(motion: np.ndarray) -> float:
    """
    Compute average root horizontal velocity magnitude.
    
    Args:
        motion: Motion array of shape (seq_len, 263) or (263, 1, seq_len) 
                depending on format
    
    Returns:
        Average velocity magnitude
    """
    # Handle different input shapes
    if len(motion.shape) == 3:
        # Shape is (features, 1, seq_len) - from model output
        motion = motion.squeeze(1).T  # -> (seq_len, features)
    
    # Root linear velocity is at indices 1 and 2
    root_vel_x = motion[:, 1]
    root_vel_z = motion[:, 2]
    vel_magnitude = np.sqrt(root_vel_x**2 + root_vel_z**2)
    return np.mean(vel_magnitude)


def generate_with_age(model, diffusion, text_prompt: str, age: float, 
                      num_frames: int = 196, device: str = 'cuda', guidance_scale: float = 2.5):
    """
    Generate a motion sequence conditioned on a specific age.
    """
    model.eval()
    
    # Prepare conditioning
    model_kwargs = {
        'y': {
            'text': [text_prompt],
            'lengths': torch.tensor([num_frames], device=device),
            'mask': torch.ones(1, 1, 1, num_frames, device=device),
            'age': torch.tensor([[age]], device=device),
            # FIX: Add the guidance scale tensor
            'scale': torch.ones(1, device=device) * guidance_scale
        }
    }
    
    # Generate
    with torch.no_grad():
        shape = (1, model.njoints, model.nfeats, num_frames)
        sample = diffusion.p_sample_loop(
            model,
            shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    
    return sample.squeeze(0).cpu().numpy()


def run_validation(args):
    """Main validation routine."""
    
    # Import necessary modules
    sys.path.insert(0, '.')
    import json
    
    # Import load_lora_to_model explicitly
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip, load_lora_to_model
    from model.cfg_sampler import ClassifierFreeSampleModel
    
    print("="*60)
    print("Age Conditioning Validation")
    print("="*60)
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load model arguments from the training run's args.json
    print(f"\nLoading model configuration...")
    model_dir = os.path.dirname(args.model_path)
    args_path = os.path.join(model_dir, 'args.json')
    
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Could not find args.json at {args_path}")
        
    with open(args_path, 'r') as f:
        model_args_dict = json.load(f)
    
    # Create Namespace
    model_args = argparse.Namespace(**model_args_dict)
    model_args.device = args.device
    model_args.model_path = args.model_path
    
    if not hasattr(model_args, 'cond_mode'):
        model_args.cond_mode = 'text,age'

    # Mock Data Loader for initialization
    class MockData:
        class MockDataset:
            num_actions = 1
        dataset = MockDataset()

    # 2. Initialize Base Model
    print("Initializing base model...")
    model, diffusion = create_model_and_diffusion(model_args, MockData())

    # 3. Handle LoRA Loading Sequence
    if getattr(model_args, 'lora_finetune', False):
        print(f"Detected LoRA configuration.")
        
        # A. Load Base Model Weights (Pre-LoRA)
        # We need to find the base checkpoint used to start training
        if hasattr(model_args, 'starting_checkpoint') and model_args.starting_checkpoint:
            print(f"Loading base model weights from: {model_args.starting_checkpoint}")
            # Handle relative paths: assume starting_checkpoint is relative to project root
            # if the file isn't found, try to assume it's absolute or check current dir
            if os.path.exists(model_args.starting_checkpoint):
                base_ckpt_path = model_args.starting_checkpoint
            else:
                # Fallback: maybe the user provided path was relative to the training script location?
                # Let's try to just trust the path or fail informatively
                raise FileNotFoundError(f"Could not find base checkpoint: {model_args.starting_checkpoint}. "
                                      "Please ensure this file exists relative to where you are running the script.")
            
            # NEW
            print(f"Loading base checkpoint to CPU first to save VRAM...")
            base_state_dict = torch.load(base_ckpt_path, map_location='cpu')
            load_model_wo_clip(model, base_state_dict)
        else:
            print("WARNING: No starting_checkpoint found in args.json. Model might be uninitialized!")

        # B. Add LoRA Adapters (This changes the model architecture)
        print("Adding LoRA adapters...")
        model.add_LoRA_adapters()
        
        # C. Load the Trained LoRA Weights
        print(f"Loading trained LoRA weights from: {args.model_path}")
        # Use load_lora_to_model which is designed for this specific task
        load_lora_to_model(model, args.model_path)
        
    else:
        # Standard loading for non-LoRA models
        print(f"Loading weights from {args.model_path}...")
        state_dict = torch.load(args.model_path, map_location=args.device)
        load_model_wo_clip(model, state_dict)

    model.to(args.device)
    model.eval()
    
    # If using classifier-free guidance
    if args.guidance_param > 1:
        model = ClassifierFreeSampleModel(model)
        
    # Define age values to test
    age_values = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    age_labels = ['Young (0.25)', 'Young-Mid (0.35)', 'Mid (0.45)', 
                  'Mid-Old (0.55)', 'Old-Mid (0.65)', 'Old (0.75)', 'Old (0.85)']
    
    # Generate samples
    print(f"\nGenerating samples for prompt: '{args.text_prompt}'")
    print(f"Testing {len(age_values)} age values...")
    
    results = []
    
    for age, label in zip(age_values, age_labels):
        print(f"\n  Generating with {label}...")
        
        velocities = []
        for rep in range(args.num_repetitions):
            motion = generate_with_age(
                model, diffusion, args.text_prompt, age,
                num_frames=args.num_frames, device=args.device
            )
            
            vel = compute_root_velocity(motion)
            velocities.append(vel)
            
            # Save sample
            sample_path = pjoin(args.output_dir, f"sample_age{age:.2f}_rep{rep}.npy")
            np.save(sample_path, motion)
        
        avg_vel = np.mean(velocities)
        std_vel = np.std(velocities)
        
        results.append({
            'age': age,
            'label': label,
            'mean_velocity': avg_vel,
            'std_velocity': std_vel,
            'velocities': velocities
        })
        
        print(f"    Velocity: {avg_vel:.4f} ± {std_vel:.4f}")
    
    # Analyze results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nPrompt: '{args.text_prompt}'")
    print(f"Repetitions per age: {args.num_repetitions}")
    print("\nAge → Velocity mapping:")
    print("-" * 40)
    
    for r in results:
        print(f"  {r['label']:20s}: vel = {r['mean_velocity']:.4f} ± {r['std_velocity']:.4f}")
    
    # Check if the expected correlation holds
    ages = [r['age'] for r in results]
    vels = [r['mean_velocity'] for r in results]
    
    # Correlation coefficient
    correlation = np.corrcoef(ages, vels)[0, 1]
    print(f"\nCorrelation (age vs velocity): {correlation:.4f}")
    
    if correlation < -0.5:
        print("✓ SUCCESS: Strong negative correlation detected!")
        print("  Higher age produces slower motion as expected.")
    elif correlation < 0:
        print("⚠ PARTIAL SUCCESS: Weak negative correlation detected.")
        print("  The trend is correct but may need more training.")
    else:
        print("✗ FAILURE: No negative correlation detected.")
        print("  The age conditioning may not be working properly.")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    ax1 = axes[0]
    x_pos = np.arange(len(results))
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(results)))
    bars = ax1.bar(x_pos, vels, yerr=[r['std_velocity'] for r in results],
                   capsize=5, color=colors, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([r['label'].replace(' ', '\n') for r in results], fontsize=8)
    ax1.set_ylabel('Average Root Velocity')
    ax1.set_title(f'Velocity by Age Condition\nCorrelation: {correlation:.3f}')
    
    # Scatter plot with trend line
    ax2 = axes[1]
    ax2.scatter(ages, vels, s=100, c=colors, edgecolors='black', zorder=5)
    
    # Add trend line
    z = np.polyfit(ages, vels, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ages), max(ages), 100)
    ax2.plot(x_line, p(x_line), 'r--', label=f'Trend (slope={z[0]:.3f})', zorder=1)
    
    ax2.set_xlabel('Age Condition')
    ax2.set_ylabel('Average Root Velocity')
    ax2.set_title('Age vs Velocity Relationship')
    ax2.legend()
    
    plt.tight_layout()
    fig_path = pjoin(args.output_dir, 'age_velocity_validation.png')
    plt.savefig(fig_path, dpi=150)
    print(f"\nVisualization saved to: {fig_path}")
    plt.close()
    
    # Save detailed results
    results_path = pjoin(args.output_dir, 'validation_results.npy')
    np.save(results_path, {
        'prompt': args.text_prompt,
        'results': results,
        'correlation': correlation
    })
    print(f"Results saved to: {results_path}")
    
    return correlation


def main():
    parser = argparse.ArgumentParser(description="Validate age conditioning in LoRA-MDM")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./save/age_validation/",
                        help="Directory to save validation results")
    parser.add_argument("--text_prompt", type=str, 
                        default="A person is walking in sks style.",
                        help="Text prompt for generation")
    parser.add_argument("--num_frames", type=int, default=196,
                        help="Number of frames to generate")
    parser.add_argument("--num_repetitions", type=int, default=5,
                        help="Number of samples per age value")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--guidance_param", type=float, default=2.5,
                        help="Classifier-free guidance scale")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    run_validation(args)


if __name__ == "__main__":
    main()