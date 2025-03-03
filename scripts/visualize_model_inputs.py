#!/usr/bin/env python
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import imageio

from synapse_analysis.data.data_loader import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data
)
from synapse_analysis.data.dataset import SynapseDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize exact model inputs for validation")
    
    # Data paths
    parser.add_argument('--raw_base_dir', type=str, required=True, help='Base directory for raw data')
    parser.add_argument('--seg_base_dir', type=str, required=True, help='Base directory for segmentation data')
    parser.add_argument('--add_mask_base_dir', type=str, required=True, help='Base directory for additional mask data')
    parser.add_argument('--excel_dir', type=str, required=True, help='Directory containing Excel files')
    parser.add_argument('--output_dir', type=str, default='outputs/model_inputs', help='Directory to save output files')
    
    # Sample selection
    parser.add_argument('--bbox_names', type=str, nargs='+', default=['bbox1'], help='Bounding box names')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--sample_indices', type=int, nargs='*', help='Specific sample indices to visualize (optional)')
    
    # Visualization parameters
    parser.add_argument('--segmentation_type', type=int, default=1, help='Segmentation type (1 for presynapse)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for blending')
    
    return parser.parse_args()

def visualize_model_input(tensor, output_path, sample_idx, frame_idx=None, bbox_name="", denormalize=True):
    """
    Visualize a single model input tensor.
    
    Args:
        tensor: Input tensor [C, H, W]
        output_path: Path to save the visualization
        sample_idx: Sample index for naming
        frame_idx: Frame index (if None, assumed to be center frame)
        bbox_name: Bounding box name
        denormalize: Whether to denormalize the tensor
    """
    # Convert tensor to numpy and squeeze channel dimension
    if isinstance(tensor, torch.Tensor):
        img = tensor.squeeze(0).cpu().numpy()  # Remove channel dim
    else:
        img = tensor.squeeze(0)  # Remove channel dim
    
    # Denormalize if needed (assuming standard normalization)
    if denormalize and img.min() < 0:
        mean, std = 0.485, 0.229  # Standard normalization values
        img = img * std + mean
    
    # Clip values to [0, 1] range
    img = np.clip(img, 0, 1)
    
    # Create figure
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    
    frame_info = f"frame_{frame_idx}" if frame_idx is not None else "center_frame"
    plt.title(f"{bbox_name} - Sample {sample_idx} - {frame_info}")
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return img

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    vol_data_dict = load_all_volumes(
        args.bbox_names,
        args.raw_base_dir,
        args.seg_base_dir,
        args.add_mask_base_dir
    )
    
    if not vol_data_dict:
        print(f"Error: Could not load data for {args.bbox_names}")
        return
    
    synapse_df = load_synapse_data(args.bbox_names, args.excel_dir)
    
    # Initialize processor - this is the exact same processor used for model input
    processor = Synapse3DProcessor(size=(80, 80))
    
    # Create dataset with the specified segmentation type and alpha
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=synapse_df,
        processor=processor,
        segmentation_type=args.segmentation_type,
        alpha=args.alpha
    )
    
    # Determine which samples to visualize
    if args.sample_indices:
        sample_indices = args.sample_indices
    else:
        # Randomly select samples if specific indices not provided
        sample_indices = np.random.choice(
            len(dataset), 
            min(args.num_samples, len(dataset)), 
            replace=False
        ).tolist()
    
    print(f"Visualizing {len(sample_indices)} samples with segmentation type {args.segmentation_type} and alpha {args.alpha}")
    
    # Process and visualize each sample
    for i, sample_idx in enumerate(sample_indices):
        # Get the sample from the dataset - this is exactly what the model would receive
        inputs, syn_info, bbox_name = dataset[sample_idx]
        
        # inputs shape: [num_frames, 1, H, W]
        num_frames = inputs.shape[0]
        center_frame_idx = num_frames // 2
        
        # Visualize center frame
        center_frame = inputs[center_frame_idx]
        output_path = output_dir / f"{bbox_name}_sample{sample_idx}_center_frame.png"
        visualize_model_input(
            center_frame, 
            output_path, 
            sample_idx, 
            center_frame_idx, 
            bbox_name
        )
        
        # Create a GIF of all frames to show the 3D nature
        frames = []
        for frame_idx in range(num_frames):
            frame = inputs[frame_idx]
            # Convert tensor to numpy and denormalize
            if isinstance(frame, torch.Tensor):
                img = frame.squeeze(0).cpu().numpy()
            else:
                img = frame.squeeze(0)
                
            # Denormalize if needed
            if img.min() < 0:
                mean, std = 0.485, 0.229
                img = img * std + mean
                
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            frames.append(img)
        
        # Save as GIF
        gif_path = output_dir / f"{bbox_name}_sample{sample_idx}_all_frames.gif"
        imageio.mimsave(gif_path, frames, fps=5, loop=0)
        
        # Print sample information
        print(f"\nSample {i+1}/{len(sample_indices)} - {bbox_name} - Index {sample_idx}:")
        for key, value in syn_info.items():
            if key in ['central_coord_1', 'central_coord_2', 'central_coord_3', 
                      'side_1_coord_1', 'side_1_coord_2', 'side_1_coord_3',
                      'side_2_coord_1', 'side_2_coord_2', 'side_2_coord_3',
                      'Var1', 'bbox_name']:
                print(f"  {key}: {value}")
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main() 