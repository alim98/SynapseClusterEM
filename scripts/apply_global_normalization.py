#!/usr/bin/env python
"""
Apply Global Normalization

This script demonstrates how to use the global normalization function
as an independent step in the data processing pipeline.

It shows how to:
1. Calculate global statistics from volumes
2. Apply global normalization to tensors
3. Save normalized data for later use
"""

import os
import argparse
import numpy as np
import torch
import json
from pathlib import Path

# Import the specific functions directly from the module
from synapse_analysis.data.data_loader import (
    Synapse3DProcessor,
    load_all_volumes
)

# Define the apply_global_normalization function here since it can't be imported
def apply_global_normalization(tensor_data, global_stats):
    """
    Apply global normalization to tensor data.
    
    Args:
        tensor_data: Input tensor data to normalize [B, C, ...] or [C, ...]
        global_stats: Dictionary containing 'mean' and 'std' values
        
    Returns:
        Normalized tensor data
    """
    # Convert to tensor if numpy array
    if isinstance(tensor_data, np.ndarray):
        tensor_data = torch.from_numpy(tensor_data)
    
    # Get global mean and std
    mean = torch.tensor(global_stats['mean'])
    std = torch.tensor(global_stats['std'])
    
    # Reshape mean and std based on input dimensions
    if tensor_data.dim() == 5:  # [B, C, D, H, W]
        mean = mean.view(1, -1, 1, 1, 1)
        std = std.view(1, -1, 1, 1, 1)
    elif tensor_data.dim() == 4:  # [B, C, H, W]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif tensor_data.dim() == 3:  # [C, H, W]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    
    # Apply normalization
    normalized_data = (tensor_data - mean) / std
    
    return normalized_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply global normalization to data"
    )
    
    # Data paths
    parser.add_argument(
        '--raw_base_dir', 
        type=str, 
        default=r"C:\Users\alim9\Documents\SynapseClusterEM\data\7_bboxes_plus_seg\raw",
        help='Base directory for raw data'
    )
    parser.add_argument(
        '--seg_base_dir', 
        type=str, 
        default=r"C:\Users\alim9\Documents\SynapseClusterEM\data\7_bboxes_plus_seg\seg",
        help='Base directory for segmentation data'
    )
    parser.add_argument(
        '--add_mask_base_dir', 
        type=str, 
        default=r"C:\Users\alim9\Documents\SynapseClusterEM\data\vesicle_cloud__syn_interface__mitochondria_annotation",
        help='Base directory for additional mask data'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs/normalized_data',
        help='Directory to save output files'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--bbox_names', 
        type=str, 
        nargs='+', 
        default=['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7'],
        help='Bounding box names to include'
    )
    
    # Normalization options
    parser.add_argument(
        '--stats_file', 
        type=str, 
        default=None,
        help='Path to existing global stats JSON file (if not provided, will calculate from data)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from:")
    print(f"  Raw data: {args.raw_base_dir}")
    print(f"  Segmentation data: {args.seg_base_dir}")
    print(f"  Additional mask data: {args.add_mask_base_dir}")
    
    # Load data for all bboxes
    vol_data_dict = load_all_volumes(
        args.bbox_names,
        args.raw_base_dir,
        args.seg_base_dir,
        args.add_mask_base_dir
    )
    
    if not vol_data_dict:
        print(f"Error: Could not load data for {args.bbox_names}")
        return
    
    # Get or calculate global statistics
    if args.stats_file and os.path.exists(args.stats_file):
        print(f"Loading global statistics from {args.stats_file}")
        with open(args.stats_file, 'r') as f:
            global_stats = json.load(f)
    else:
        print("Calculating global statistics from volumes...")
        global_stats = Synapse3DProcessor.calculate_global_stats_from_volumes(vol_data_dict)
        
        # Save global stats for reference
        stats_file = output_dir / 'global_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(global_stats, f)
        print(f"  Global stats saved to {stats_file}")
    
    print(f"Global statistics:")
    print(f"  Mean: {global_stats['mean']}")
    print(f"  Std: {global_stats['std']}")
    
    # Process each volume with global normalization
    print("Applying global normalization to volumes...")
    normalized_volumes = {}
    
    for bbox_name, (raw_vol, seg_vol, add_mask_vol) in vol_data_dict.items():
        print(f"  Processing {bbox_name}...")
        
        # Convert raw volume to tensor
        raw_tensor = torch.from_numpy(raw_vol.astype(np.float32))
        
        # Apply global normalization
        normalized_tensor = apply_global_normalization(raw_tensor, global_stats)
        
        # Convert back to numpy for storage
        normalized_vol = normalized_tensor.numpy()
        
        # Store normalized volume
        normalized_volumes[bbox_name] = normalized_vol
        
        # Save a sample slice for visualization
        center_slice = normalized_vol.shape[0] // 2
        sample_slice = normalized_vol[center_slice]
        
        # Rescale to [0, 255] for image saving
        sample_slice = ((sample_slice - sample_slice.min()) / 
                       (sample_slice.max() - sample_slice.min()) * 255).astype(np.uint8)
        
        # Save as numpy array
        slice_file = output_dir / f"{bbox_name}_normalized_slice.npy"
        np.save(slice_file, sample_slice)
        
        # Save metadata about the normalization
        meta_file = output_dir / f"{bbox_name}_normalization_info.json"
        with open(meta_file, 'w') as f:
            json.dump({
                "bbox_name": bbox_name,
                "global_mean": global_stats['mean'],
                "global_std": global_stats['std'],
                "original_shape": raw_vol.shape,
                "normalized_shape": normalized_vol.shape,
                "center_slice_idx": center_slice
            }, f, indent=2)
    
    print(f"Normalized data saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main() 