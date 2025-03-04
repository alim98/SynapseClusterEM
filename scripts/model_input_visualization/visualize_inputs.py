#!/usr/bin/env python
"""
Model Input Visualization Tool

This script visualizes the exact inputs that the neural network model receives during training
and inference. It extracts samples from the dataset, processes them exactly as they would be
processed for the model, and saves visualizations of the center slices and animated GIFs of
the 3D volumes.

This tool is useful for validating model inputs and understanding what features the model
is learning from.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import imageio
import json

from synapse_analysis.data.data_loader import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data
)
from synapse_analysis.data.dataset import SynapseDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize exact model inputs for validation and understanding"
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
        '--excel_dir', 
        type=str, 
        default=r"C:\Users\alim9\Documents\SynapseClusterEM\data\7_bboxes_plus_seg",
        help='Directory containing Excel files'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='outputs/model_input_visualization',
        help='Directory to save output files'
    )
    
    # Sample selection
    parser.add_argument(
        '--bbox_names', 
        type=str, 
        nargs='+', 
        default=['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7'],
        help='Bounding box names to include'
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=100,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--sample_indices', 
        type=int, 
        nargs='*',
        help='Specific sample indices to visualize (optional)'
    )
    
    # Visualization parameters
    parser.add_argument(
        '--segmentation_type', 
        type=int, 
        default=1,
        help='Segmentation type (0=raw, 1=presynapse, 2=postsynapse, 3=both sides, 4=vesicles+cleft)'
    )
    parser.add_argument(
        '--alpha', 
        type=float, 
        default=1.0,
        help='Alpha value for blending (0.0-1.0)'
    )
    parser.add_argument(
        '--fixed_gray_value', 
        type=float, 
        default=0.6,
        help='Fixed gray value for non-segmented regions (0.0-1.0)'
    )
    parser.add_argument(
        '--use_global_norm', 
        action='store_true',
        help='Use global normalization for visualization'
    )
    
    return parser.parse_args()


def visualize_tensor(tensor, output_path, sample_idx, frame_idx=None, bbox_name="", 
                    denormalize=True, fixed_gray_value=0.6, mask=None):
    """
    Visualize a single model input tensor.
    
    Args:
        tensor: Input tensor [C, H, W]
        output_path: Path to save the visualization
        sample_idx: Sample index for naming
        frame_idx: Frame index (if None, assumed to be center frame)
        bbox_name: Bounding box name
        denormalize: Whether to denormalize the tensor
        fixed_gray_value: Fixed gray value for non-segmented regions
        mask: Optional mask to apply fixed gray value (if None, visualize as is)
    
    Returns:
        Numpy array of the visualized image
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
    
    # Apply fixed gray value to non-segmented regions if mask is provided
    if mask is not None:
        # Create a copy to avoid modifying the original
        img_copy = img.copy()
        # Set non-segmented regions to fixed gray value
        img_copy[~mask] = fixed_gray_value
        img = img_copy
    
    # Clip values to [0, 1] range
    img = np.clip(img, 0, 1)
    
    # Create figure
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)  # Force consistent scale
    
    # Extract bbox number for the title
    bbox_num = bbox_name.replace("bbox", "")
    
    frame_info = f"frame_{frame_idx}" if frame_idx is not None else "center_frame"
    plt.title(f"Bbox {bbox_num} - Sample {sample_idx} - {frame_info}")
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return img


def create_gif_from_tensor(tensor_frames, output_path, fps=5, denormalize=True, 
                          fixed_gray_value=0.6, masks=None):
    """
    Create a GIF animation from a list of tensor frames.
    
    Args:
        tensor_frames: List of tensor frames
        output_path: Path to save the GIF
        fps: Frames per second
        denormalize: Whether to denormalize the tensors
        fixed_gray_value: Fixed gray value for non-segmented regions
        masks: Optional list of masks to apply fixed gray value
    """
    frames = []
    for i, frame in enumerate(tensor_frames):
        # Convert tensor to numpy
        if isinstance(frame, torch.Tensor):
            img = frame.squeeze(0).cpu().numpy()
        else:
            img = frame.squeeze(0)
            
        # Denormalize if needed
        if denormalize and img.min() < 0:
            mean, std = 0.485, 0.229
            img = img * std + mean
        
        # Apply fixed gray value to non-segmented regions if masks are provided
        if masks is not None:
            mask = masks[i] if i < len(masks) else masks[-1]
            img_copy = img.copy()
            img_copy[~mask] = fixed_gray_value
            img = img_copy
            
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        frames.append(img)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)


def extract_masks_from_dataset(dataset, sample_idx):
    """
    Extract segmentation masks from the dataset for a specific sample.
    This is used to apply consistent gray values in visualization.
    
    Args:
        dataset: SynapseDataset instance
        sample_idx: Sample index
        
    Returns:
        List of masks for each frame
    """
    # Get sample information
    syn_info = dataset.synapse_df.iloc[sample_idx]
    bbox_name = syn_info['bbox_name']
    
    # Get raw volumes
    raw_vol, seg_vol, add_mask_vol = dataset.vol_data_dict.get(bbox_name, (None, None, None))
    if raw_vol is None:
        return None
    
    # Get coordinates
    central_coord = (int(syn_info['central_coord_1']), 
                    int(syn_info['central_coord_2']), 
                    int(syn_info['central_coord_3']))
    side1_coord = (int(syn_info['side_1_coord_1']), 
                  int(syn_info['side_1_coord_2']), 
                  int(syn_info['side_1_coord_3']))
    side2_coord = (int(syn_info['side_2_coord_1']), 
                  int(syn_info['side_2_coord_2']), 
                  int(syn_info['side_2_coord_3']))
    
    # Import here to avoid circular imports
    from synapse_analysis.utils.processing import get_bbox_labels, create_segmented_cube
    
    # Get segmentation masks based on segmentation type
    labels = get_bbox_labels(bbox_name)
    
    # Calculate bounds
    subvolume_size = 80  # Same as in dataset
    half_size = subvolume_size // 2
    cx, cy, cz = central_coord
    x_start = max(cx - half_size, 0)
    x_end = min(cx + half_size, raw_vol.shape[2])
    y_start = max(cy - half_size, 0)
    y_end = min(cy + half_size, raw_vol.shape[1])
    z_start = max(cz - half_size, 0)
    z_end = min(cz + half_size, raw_vol.shape[0])
    
    # Create segment masks
    def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
        x1, y1, z1 = s1_coord
        x2, y2, z2 = s2_coord
        seg_id_1 = segmentation_volume[z1, y1, x1]
        seg_id_2 = segmentation_volume[z2, y2, x2]
        mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
        mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
        return mask_1, mask_2
    
    mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)
    
    # Get vesicle mask
    vesicle_full_mask = (add_mask_vol == labels['vesicle_label'])
    
    # Determine pre-synapse side
    overlap_side1 = np.sum(np.logical_and(mask_1_full, vesicle_full_mask))
    overlap_side2 = np.sum(np.logical_and(mask_2_full, vesicle_full_mask))
    presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
    
    # Create combined mask based on segmentation type
    segmentation_type = dataset.segmentation_type
    if segmentation_type == 0:  # Raw data
        combined_mask_full = np.ones_like(add_mask_vol, dtype=bool)
    elif segmentation_type == 1:  # Presynapse
        combined_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
    elif segmentation_type == 2:  # Postsynapse
        combined_mask_full = mask_2_full if presynapse_side == 1 else mask_1_full
    elif segmentation_type == 3:  # Both sides
        combined_mask_full = np.logical_or(mask_1_full, mask_2_full)
    elif segmentation_type in [4, 9]:  # Vesicles + Cleft
        from synapse_analysis.utils.processing import get_closest_component_mask
        vesicle_closest = get_closest_component_mask(
            vesicle_full_mask, z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == labels['cleft_label']), z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        cleft_closest2 = get_closest_component_mask(
            (add_mask_vol == labels['cleft_label2']), z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        combined_mask_full = np.logical_or(vesicle_closest, np.logical_or(cleft_closest, cleft_closest2))
    else:
        combined_mask_full = np.ones_like(add_mask_vol, dtype=bool)
    
    # Extract and process subvolume mask
    sub_combined_mask = combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Padding if needed
    pad_z = subvolume_size - sub_combined_mask.shape[0]
    pad_y = subvolume_size - sub_combined_mask.shape[1]
    pad_x = subvolume_size - sub_combined_mask.shape[2]
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        sub_combined_mask = np.pad(sub_combined_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), 
                                  mode='constant', constant_values=False)
    
    sub_combined_mask = sub_combined_mask[:subvolume_size, :subvolume_size, :subvolume_size]
    
    # Create list of masks for each frame
    masks = []
    for z in range(sub_combined_mask.shape[0]):
        masks.append(sub_combined_mask[z])
    
    return masks


def main():
    """Main function to run the visualization process."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from:")
    print(f"  Raw data: {args.raw_base_dir}")
    print(f"  Segmentation data: {args.seg_base_dir}")
    print(f"  Additional mask data: {args.add_mask_base_dir}")
    print(f"  Excel data: {args.excel_dir}")
    print(f"  Using fixed gray value: {args.fixed_gray_value}")
    if args.use_global_norm:
        print(f"  Using global normalization: Yes")
    
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
    
    synapse_df = load_synapse_data(args.bbox_names, args.excel_dir)
    
    # # Initialize processor with global normalization if requested
    # if args.use_global_norm:
    #     # Calculate global statistics from volumes
    #     # global_stats = Synapse3DProcessor.calculate_global_stats_from_volumes(vol_data_dict)
        
    #     # Save global stats for reference
    #     # stats_file = output_dir / 'global_stats.json'
    #     # with open(stats_file, 'w') as f:
    #     #     json.dump(global_stats, f)
    #     # print(f"  Global stats saved to {stats_file}")
    #     # print(f"  Global mean: {global_stats['mean']}")
    #     # print(f"  Global std: {global_stats['std']}")
        
    #     # Create processor with global normalization
    #     processor = Synapse3DProcessor(
    #         size=(80, 80),
    #         apply_global_norm=True,
    #         global_stats=global_stats
    #     )
    # else:
    #     # Use default processor without global normalization
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
        total_samples = min(args.num_samples, len(dataset))
        sample_indices = np.random.choice(
            len(dataset), 
            total_samples, 
            replace=False
        ).tolist()
    
    print(f"Visualizing {len(sample_indices)} samples with segmentation type {args.segmentation_type} and alpha {args.alpha}")
    
    # Create a summary file
    summary_path = output_dir / "samples_summary.txt"
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"Summary of {len(sample_indices)} model input samples\n")
        summary_file.write(f"Segmentation type: {args.segmentation_type}, Alpha: {args.alpha}\n")
        summary_file.write(f"Fixed gray value: {args.fixed_gray_value}\n")
        summary_file.write(f"Global normalization: {'Yes' if args.use_global_norm else 'No'}\n")
        if args.use_global_norm:
            summary_file.write(f"Global mean: {global_stats['mean']}\n")
            summary_file.write(f"Global std: {global_stats['std']}\n")
        summary_file.write("\n")
        
        # Process and visualize each sample
        for i, sample_idx in enumerate(sample_indices):
            # Get the sample from the dataset - this is exactly what the model would receive
            inputs, syn_info, bbox_name = dataset[sample_idx]
            
            # Extract masks for consistent visualization if needed
            masks = None
            if args.fixed_gray_value is not None:
                try:
                    masks = extract_masks_from_dataset(dataset, sample_idx)
                except Exception as e:
                    print(f"\nWarning: Could not extract masks for sample {sample_idx}: {e}")
            
            # inputs shape: [num_frames, 1, H, W]
            num_frames = inputs.shape[0]
            center_frame_idx = num_frames // 2
            
            # Visualize center frame
            center_frame = inputs[center_frame_idx]
            output_path = output_dir / f"{bbox_name}_sample{sample_idx}_center_frame.png"
            
            # Get center frame mask if available
            center_mask = None
            if masks is not None and center_frame_idx < len(masks):
                center_mask = masks[center_frame_idx]
                
            visualize_tensor(
                center_frame, 
                output_path, 
                sample_idx, 
                center_frame_idx, 
                bbox_name,
                fixed_gray_value=args.fixed_gray_value,
                mask=center_mask
            )
            
            # Create a GIF of all frames to show the 3D nature
            gif_path = output_dir / f"{bbox_name}_sample{sample_idx}_all_frames.gif"
            create_gif_from_tensor(
                inputs, 
                gif_path, 
                fixed_gray_value=args.fixed_gray_value,
                masks=masks
            )
            
            # Write sample information to summary file
            summary_file.write(f"Sample {i+1}/{len(sample_indices)} - {bbox_name} - Index {sample_idx}:\n")
            for key, value in syn_info.items():
                if key in ['central_coord_1', 'central_coord_2', 'central_coord_3', 
                          'side_1_coord_1', 'side_1_coord_2', 'side_1_coord_3',
                          'side_2_coord_1', 'side_2_coord_2', 'side_2_coord_3',
                          'Var1', 'bbox_name']:
                    summary_file.write(f"  {key}: {value}\n")
            summary_file.write("\n")
            
            # Print progress
            print(f"\rProcessing sample {i+1}/{len(sample_indices)} - {bbox_name} - Index {sample_idx}", end="")
    
    print(f"\n\nAll visualizations saved to {output_dir}")
    print(f"Summary file created at {summary_path}")


if __name__ == "__main__":
    main() 