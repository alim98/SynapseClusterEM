#!/usr/bin/env python
"""
Verify Data Processing Script

This script:
1. Loads and processes synapse data
2. Applies global normalization if requested
3. Creates the dataset
4. Saves center slice visualizations for verification
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from datetime import datetime

# Import from the package level
from synapse_analysis.data import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data,
    apply_global_normalization,
    SynapseDataset
)

def parse_args():
    parser = argparse.ArgumentParser(description="Process data and verify with center slice visualizations")
    
    # Data paths
    parser.add_argument('--raw_base_dir', type=str, required=True, help='Base directory for raw data')
    parser.add_argument('--seg_base_dir', type=str, required=True, help='Base directory for segmentation data')
    parser.add_argument('--add_mask_base_dir', type=str, required=True, help='Base directory for additional mask data')
    parser.add_argument('--excel_dir', type=str, required=True, help='Directory containing Excel files')
    parser.add_argument('--output_dir', type=str, default='outputs/data_verification', help='Directory to save outputs')
    
    # Processing parameters
    parser.add_argument('--bbox_names', type=str, nargs='+', default=['bbox1'], help='Bounding box names')
    parser.add_argument('--use_global_norm', action='store_true', help='Use global normalization')
    parser.add_argument('--global_stats_path', type=str, help='Path to existing global stats (optional)')
    parser.add_argument('--num_samples_for_stats', type=int, default=1000, help='Number of samples for global stats')
    parser.add_argument('--segmentation_type', type=int, default=1, help='Segmentation type')
    parser.add_argument('--subvol_size', type=int, default=80, help='Size of subvolume cube')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha value for segmentation overlay')
    
    # Visualization parameters
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--sample_indices', type=int, nargs='*', help='Specific sample indices to visualize')
    
    return parser.parse_args()

def save_center_slice_visualization(tensor, metadata, output_path, sample_idx, denormalize=True):
    """
    Save visualization of the center slice from a 3D tensor.
    
    Args:
        tensor: Input tensor [C, D, H, W] or [D, H, W]
        metadata: Dictionary containing sample metadata
        output_path: Path to save the visualization
        sample_idx: Sample index for naming
        denormalize: Whether to denormalize the tensor
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    print(f"   - Initial tensor shape: {tensor.shape}, type: {type(tensor)}")
    
    # For our data, tensor is in format [16, 1, 80, 80]
    # Take the middle frame from the sequence
    middle_frame_idx = tensor.shape[0] // 2
    middle_frame = tensor[middle_frame_idx, 0].numpy()  # [80, 80]
    print(f"   - Using middle frame {middle_frame_idx} from sequence of {tensor.shape[0]}")
    
    # Create a multi-panel figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original normalized image
    axs[0, 0].imshow(middle_frame, cmap='gray')
    axs[0, 0].set_title('Normalized Data')
    axs[0, 0].set_xlabel(f'Min: {middle_frame.min():.3f}, Max: {middle_frame.max():.3f}')
    
    # Denormalized image with ImageNet stats
    denorm = middle_frame * 0.229 + 0.485
    axs[0, 1].imshow(denorm, cmap='gray', vmin=0, vmax=1)
    axs[0, 1].set_title('Denormalized (ImageNet)')
    axs[0, 1].set_xlabel(f'Min: {denorm.min():.3f}, Max: {denorm.max():.3f}')
    
    # Enhanced contrast
    p_low, p_high = np.percentile(denorm, [2, 98])
    enhanced = np.clip((denorm - p_low) / (p_high - p_low), 0, 1)
    axs[1, 0].imshow(enhanced, cmap='gray')
    axs[1, 0].set_title('Enhanced Contrast')
    axs[1, 0].set_xlabel(f'Range: {p_low:.3f} to {p_high:.3f}')
    
    # Another frame for comparison
    first_frame = tensor[0, 0].numpy()
    first_frame_denorm = first_frame * 0.229 + 0.485
    axs[1, 1].imshow(first_frame_denorm, cmap='gray', vmin=0, vmax=1)
    axs[1, 1].set_title(f'First Frame')
    
    plt.tight_layout()
    
    # Add metadata as text
    bbox_name = metadata.get('bbox_name', 'unknown')
    coords = f"Center: ({metadata.get('central_coord_1', '?')}, {metadata.get('central_coord_2', '?')}, {metadata.get('central_coord_3', '?')})"
    fig.suptitle(f"Sample {sample_idx} from {bbox_name}\n{coords}", fontsize=16, y=1.02)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_path, f"sample_{sample_idx}_{bbox_name}_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading synapse data...")
    synapse_df = load_synapse_data(args.bbox_names, args.excel_dir)
    print(f"   Loaded {len(synapse_df)} synapse entries")
    
    print("\n2. Loading volume data...")
    print(f"   Loading from: {args.raw_base_dir}, {args.seg_base_dir}, {args.add_mask_base_dir}")
    vol_data_dict = load_all_volumes(
        args.bbox_names,
        args.raw_base_dir,
        args.seg_base_dir,
        args.add_mask_base_dir
    )
    print(f"   Loaded volumes for {len(vol_data_dict)} bounding boxes")
    print(f"   Bounding boxes: {list(vol_data_dict.keys())}")
    
    # Handle global normalization
    global_stats = None
    if args.use_global_norm:
        print("\n3. Setting up global normalization...")
        if args.global_stats_path and os.path.exists(args.global_stats_path):
            print(f"   Loading global statistics from {args.global_stats_path}")
            with open(args.global_stats_path, 'r') as f:
                global_stats = json.load(f)
        else:
            print("   Calculating global statistics...")
            from synapse_analysis.data.data_loader import calculate_global_stats
            global_stats = calculate_global_stats(
                raw_base_dir=args.raw_base_dir,
                seg_base_dir=args.seg_base_dir,
                add_mask_base_dir=args.add_mask_base_dir,
                excel_dir=args.excel_dir,
                segmentation_types=[args.segmentation_type],
                bbox_names=args.bbox_names,
                num_samples=args.num_samples_for_stats
            )
            
            # Save global stats
            stats_path = output_dir / 'global_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(global_stats, f)
            print(f"   Saved global statistics to {stats_path}")
    
    print("\n4. Creating dataset...")
    processor = Synapse3DProcessor(
        size=(args.subvol_size, args.subvol_size),
        mean=global_stats['mean'] if global_stats else (0.485,),
        std=global_stats['std'] if global_stats else (0.229,)
    )
    print(f"   Created processor with size={args.subvol_size}")
    
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=synapse_df,
        processor=processor,
        segmentation_type=args.segmentation_type,
        subvol_size=args.subvol_size,
        alpha=args.alpha
    )
    print(f"   Dataset created with {len(dataset)} samples")
    
    print("\n5. Saving visualizations...")
    vis_output_dir = output_dir / 'visualizations'
    vis_output_dir.mkdir(exist_ok=True)
    print(f"   Output directory: {vis_output_dir}")
    
    # Determine which samples to visualize
    if args.sample_indices:
        sample_indices = args.sample_indices
    else:
        sample_indices = np.random.choice(
            len(dataset), 
            size=min(args.num_samples, len(dataset)), 
            replace=False
        )
    print(f"   Will visualize samples: {sample_indices}")
    
    for idx in sample_indices:
        print(f"   Processing sample {idx}...")
        try:
            sample, metadata, bbox_name = dataset[idx]
            print(f"   - Sample shape: {sample.shape}, type: {type(sample)}")
            print(f"   - Metadata keys: {list(metadata.keys())}")
            print(f"   - Bbox name: {bbox_name}")
            
            # Add global stats to metadata if available
            if global_stats is not None:
                metadata['global_stats'] = global_stats
            
            save_path = save_center_slice_visualization(
                tensor=sample,
                metadata=metadata,
                output_path=vis_output_dir,
                sample_idx=idx
            )
            print(f"   Saved visualization to {save_path}")
        except Exception as e:
            print(f"   ERROR with sample {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nData processing and verification complete!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main() 