"""
Two-Phase Checkpoint Search: Coarse sampling then dense around best region
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join as pjoin
import glob
import json
from typing import List, Dict

from validate_age_conditioning import run_validation


def find_checkpoints(checkpoint_dir: str, pattern: str = "model*.pt") -> List[Dict]:
    """Find all checkpoint files and return with step numbers."""
    checkpoint_paths = glob.glob(pjoin(checkpoint_dir, pattern))
    
    checkpoints = []
    for path in checkpoint_paths:
        basename = os.path.basename(path)
        try:
            step = int(basename.replace("model", "").replace(".pt", "").lstrip('0') or '0')
            checkpoints.append({'path': path, 'step': step, 'basename': basename})
        except:
            continue
    
    checkpoints.sort(key=lambda x: x['step'])
    return checkpoints


def run_checkpoint_validation(checkpoint_info, args):
    """Run validation on a single checkpoint."""
    checkpoint_args = argparse.Namespace(
        model_path=checkpoint_info['path'],
        output_dir=pjoin(args.output_dir, f"step_{checkpoint_info['step']}"),
        text_prompt=args.text_prompt,
        num_frames=args.num_frames,
        num_repetitions=args.num_repetitions,
        device=args.device,
        guidance_param=args.guidance_param
    )
    
    os.makedirs(checkpoint_args.output_dir, exist_ok=True)
    
    try:
        correlation = run_validation(checkpoint_args)
        return {
            'step': checkpoint_info['step'],
            'basename': checkpoint_info['basename'],
            'path': checkpoint_info['path'],
            'correlation': correlation,
            'success': True
        }
    except Exception as e:
        print(f"✗ Failed: {e}")
        return {
            'step': checkpoint_info['step'],
            'basename': checkpoint_info['basename'],
            'path': checkpoint_info['path'],
            'correlation': None,
            'success': False,
            'error': str(e)
        }


def two_phase_search(args):
    """
    Phase 1: Sample every Nth checkpoint (coarse)
    Phase 2: Densely sample around the best region (fine)
    """
    
    print("="*80)
    print("TWO-PHASE CHECKPOINT SEARCH")
    print("="*80)
    
    all_checkpoints = find_checkpoints(args.checkpoint_dir)
    
    if not all_checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    
    print(f"\nFound {len(all_checkpoints)} total checkpoints")
    print(f"Steps: {all_checkpoints[0]['step']} to {all_checkpoints[-1]['step']}")
    
    # ============= PHASE 1: COARSE SAMPLING =============
    print("\n" + "="*80)
    print("PHASE 1: COARSE SAMPLING")
    print("="*80)
    
    # Sample every 4th checkpoint for coarse search
    coarse_indices = list(range(0, len(all_checkpoints), 4))
    # Always include the last checkpoint
    if coarse_indices[-1] != len(all_checkpoints) - 1:
        coarse_indices.append(len(all_checkpoints) - 1)
    
    coarse_checkpoints = [all_checkpoints[i] for i in coarse_indices]
    
    print(f"\nTesting {len(coarse_checkpoints)} checkpoints (every 4th):")
    for cp in coarse_checkpoints:
        print(f"  - Step {cp['step']}")
    
    coarse_results = []
    for i, cp in enumerate(coarse_checkpoints):
        print(f"\n[Phase 1: {i+1}/{len(coarse_checkpoints)}] Testing step {cp['step']}...")
        result = run_checkpoint_validation(cp, args)
        coarse_results.append(result)
        
        if result['success']:
            print(f"✓ Correlation: {result['correlation']:.4f}")
    
    # Find best region from coarse search
    valid_coarse = [r for r in coarse_results if r['success']]
    
    if not valid_coarse:
        print("\n✗ No successful validations in coarse phase!")
        return
    
    best_coarse = min(valid_coarse, key=lambda x: x['correlation'])
    best_coarse_idx = next(i for i, cp in enumerate(all_checkpoints) 
                           if cp['step'] == best_coarse['step'])
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print(f"\nBest from coarse search:")
    print(f"  Step: {best_coarse['step']}")
    print(f"  Correlation: {best_coarse['correlation']:.4f}")
    
    # ============= PHASE 2: FINE SAMPLING =============
    print("\n" + "="*80)
    print("PHASE 2: FINE SAMPLING AROUND BEST REGION")
    print("="*80)
    
    # Define the window to search densely (±2 checkpoints on each side)
    window_size = 2
    start_idx = max(0, best_coarse_idx - window_size)
    end_idx = min(len(all_checkpoints), best_coarse_idx + window_size + 1)
    
    # Get checkpoints in this window that we haven't tested yet
    tested_steps = {r['step'] for r in coarse_results}
    fine_checkpoints = [cp for cp in all_checkpoints[start_idx:end_idx] 
                        if cp['step'] not in tested_steps]
    
    print(f"\nSearching window: steps {all_checkpoints[start_idx]['step']} to {all_checkpoints[end_idx-1]['step']}")
    print(f"Testing {len(fine_checkpoints)} additional checkpoints:")
    for cp in fine_checkpoints:
        print(f"  - Step {cp['step']}")
    
    fine_results = []
    for i, cp in enumerate(fine_checkpoints):
        print(f"\n[Phase 2: {i+1}/{len(fine_checkpoints)}] Testing step {cp['step']}...")
        result = run_checkpoint_validation(cp, args)
        fine_results.append(result)
        
        if result['success']:
            print(f"✓ Correlation: {result['correlation']:.4f}")
    
    # ============= FINAL ANALYSIS =============
    all_results = coarse_results + fine_results
    valid_results = [r for r in all_results if r['success']]
    valid_results.sort(key=lambda x: x['step'])
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    best_overall = min(valid_results, key=lambda x: x['correlation'])
    
    print(f"\n🏆 BEST CHECKPOINT FOUND:")
    print(f"   Step: {best_overall['step']}")
    print(f"   File: {best_overall['basename']}")
    print(f"   Correlation: {best_overall['correlation']:.4f}")
    print(f"   Path: {best_overall['path']}")
    
    print(f"\n📊 TOP 5 CHECKPOINTS:")
    top5 = sorted(valid_results, key=lambda x: x['correlation'])[:5]
    for i, r in enumerate(top5, 1):
        print(f"   {i}. Step {r['step']:6d}: correlation = {r['correlation']:.4f}")
    
    print(f"\n📈 Total validations run: {len(valid_results)}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    steps = [r['step'] for r in valid_results]
    correlations = [r['correlation'] for r in valid_results]
    
    # Mark coarse vs fine samples
    coarse_steps = [r['step'] for r in coarse_results if r['success']]
    coarse_corrs = [r['correlation'] for r in coarse_results if r['success']]
    fine_steps = [r['step'] for r in fine_results if r['success']]
    fine_corrs = [r['correlation'] for r in fine_results if r['success']]
    
    ax.plot(steps, correlations, 'k--', alpha=0.3, linewidth=1)
    ax.scatter(coarse_steps, coarse_corrs, s=100, c='blue', marker='o', 
               label='Phase 1 (coarse)', zorder=5, edgecolors='black', linewidths=1.5)
    ax.scatter(fine_steps, fine_corrs, s=100, c='orange', marker='s',
               label='Phase 2 (fine)', zorder=5, edgecolors='black', linewidths=1.5)
    ax.scatter([best_overall['step']], [best_overall['correlation']], 
               s=300, c='green', marker='*', label='Best', zorder=10,
               edgecolors='black', linewidths=2)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Target (-0.5)')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Correlation (age vs velocity)', fontsize=12)
    ax.set_title('Two-Phase Checkpoint Search Results', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = pjoin(args.output_dir, 'two_phase_search_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 Results plot saved to: {plot_path}")
    plt.close()
    
    # Save summary
    summary = {
        'best_checkpoint': best_overall,
        'top_5': top5,
        'all_results': all_results,
        'phase_1_count': len(coarse_results),
        'phase_2_count': len(fine_results),
        'total_validations': len(valid_results)
    }
    
    summary_path = pjoin(args.output_dir, 'search_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to: {summary_path}")
    
    return best_overall


def main():
    parser = argparse.ArgumentParser(description="Two-phase checkpoint search")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for validation results")
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    two_phase_search(args)


if __name__ == "__main__":
    main()