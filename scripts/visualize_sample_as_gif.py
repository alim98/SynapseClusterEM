#!/usr/bin/env python
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from PIL import Image
import torch

from synapse_analysis.data.data_loader import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data
)
from synapse_analysis.data.dataset import SynapseDataset
from synapse_analysis.utils.processing import create_segmented_cube

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a sample with segmentation type 1 (presynapse) as a GIF")
    
    # Data paths
    parser.add_argument('--raw_base_dir', type=str, required=True, help='Base directory for raw data')
    parser.add_argument('--seg_base_dir', type=str, required=True, help='Base directory for segmentation data')
    parser.add_argument('--add_mask_base_dir', type=str, required=True, help='Base directory for additional mask data')
    parser.add_argument('--excel_dir', type=str, required=True, help='Directory containing Excel files')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output files')
    
    # Sample selection
    parser.add_argument('--bbox_name', type=str, default='bbox1', help='Bounding box name')
    parser.add_argument('--sample_index', type=int, default=0, help='Index of the sample in the dataset')
    
    # Visualization parameters
    parser.add_argument('--segmentation_type', type=int, default=1, help='Segmentation type (1 for presynapse)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for blending')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the GIF')
    parser.add_argument('--gray_value', type=float, default=0.6, help='Fixed gray value for non-segmented regions (0-1)')
    
    return parser.parse_args()

def create_gif(frames, output_path, fps=10):
    """Create a GIF from a list of frames."""
    # Convert frames to uint8 if they are float
    if frames[0].dtype == np.float32 or frames[0].dtype == np.float64:
        frames = [(frame * 255).astype(np.uint8) for frame in frames]
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to {output_path}")

def create_custom_segmented_cube(raw_vol, seg_vol, add_mask_vol, central_coord, side1_coord, side2_coord, 
                               segmentation_type, subvolume_size=80, alpha=1.0, bbox_name="", gray_value=0.6):
    """
    Create a segmented cube with a fixed gray color for non-segmented regions.
    This is a simplified version that ensures we use a fixed gray value.
    """
    from synapse_analysis.utils.processing import get_bbox_labels, get_closest_component_mask
    
    labels = get_bbox_labels(bbox_name)
    
    # Calculate bounds
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
    
    # For segmentation type 1 (presynapse)
    if segmentation_type == 1:
        # Get vesicle mask to determine presynapse side
        vesicle_full_mask = (add_mask_vol == labels['vesicle_label'])
        vesicle_mask = get_closest_component_mask(
            vesicle_full_mask, z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        
        # Determine pre-synapse side
        overlap_side1 = np.sum(np.logical_and(mask_1_full, vesicle_mask))
        overlap_side2 = np.sum(np.logical_and(mask_2_full, vesicle_mask))
        presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
        
        # Set the mask for presynapse
        combined_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
    else:
        # Default to showing everything
        combined_mask_full = np.ones_like(raw_vol, dtype=bool)
    
    # Extract and process subvolume
    sub_raw = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
    sub_combined_mask = combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Padding if needed
    pad_z = subvolume_size - sub_raw.shape[0]
    pad_y = subvolume_size - sub_raw.shape[1]
    pad_x = subvolume_size - sub_raw.shape[2]
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        sub_raw = np.pad(sub_raw, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
        sub_combined_mask = np.pad(sub_combined_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)
    
    sub_raw = sub_raw[:subvolume_size, :subvolume_size, :subvolume_size]
    sub_combined_mask = sub_combined_mask[:subvolume_size, :subvolume_size, :subvolume_size]
    
    # Convert to float32 and normalize
    normalized = sub_raw.astype(np.float32)
    mins = np.min(normalized, axis=(1, 2), keepdims=True)
    maxs = np.max(normalized, axis=(1, 2), keepdims=True)
    ranges = np.where(maxs > mins, maxs - mins, 1.0)
    normalized = (normalized - mins) / ranges
    
    # Create the output array with fixed gray value for non-segmented regions
    result = np.zeros_like(normalized)
    
    # Set non-segmented regions to fixed gray value
    result[~sub_combined_mask] = gray_value
    
    # Set segmented regions to original normalized values
    result[sub_combined_mask] = normalized[sub_combined_mask]
    
    # Transpose dimensions for visualization
    overlaid_cube = np.transpose(result, (1, 2, 0))
    
    return overlaid_cube

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    bbox_names = [args.bbox_name]
    vol_data_dict = load_all_volumes(
        bbox_names,
        args.raw_base_dir,
        args.seg_base_dir,
        args.add_mask_base_dir
    )
    
    if not vol_data_dict:
        print(f"Error: Could not load data for {args.bbox_name}")
        return
    
    synapse_df = load_synapse_data(bbox_names, args.excel_dir)
    
    # Initialize processor
    processor = Synapse3DProcessor(size=(80, 80))
    
    # Check if the dataset has any samples
    if len(synapse_df) == 0:
        print(f"Error: No samples found for {args.bbox_name}")
        return
    
    # Get sample information
    if args.sample_index >= len(synapse_df):
        print(f"Error: Sample index {args.sample_index} is out of range. Dataset has {len(synapse_df)} samples.")
        return
    
    syn_info = synapse_df.iloc[args.sample_index]
    bbox_name = syn_info['bbox_name']
    raw_vol, seg_vol, add_mask_vol = vol_data_dict.get(bbox_name, (None, None, None))
    
    if raw_vol is None:
        print(f"Error: Could not load volume data for {bbox_name}")
        return
    
    # Extract coordinates
    central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
    side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
    side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))
    
    # Create custom segmented cube with fixed gray value
    overlaid_cube = create_custom_segmented_cube(
        raw_vol=raw_vol,
        seg_vol=seg_vol,
        add_mask_vol=add_mask_vol,
        central_coord=central_coord,
        side1_coord=side1_coord,
        side2_coord=side2_coord,
        segmentation_type=args.segmentation_type,
        subvolume_size=80,
        alpha=args.alpha,
        bbox_name=bbox_name,
        gray_value=args.gray_value
    )
    
    # Extract frames
    frames = [overlaid_cube[..., z] for z in range(overlaid_cube.shape[2])]
    
    # Create GIF
    gif_path = output_dir / f"sample_{args.sample_index}_seg{args.segmentation_type}_alpha{args.alpha}_gray{args.gray_value}.gif"
    create_gif(frames, gif_path, fps=args.fps)
    
    # Also save a few sample frames as images for reference
    for i in range(0, len(frames), len(frames)//5):
        frame = frames[i]
        plt.figure(figsize=(8, 8))
        plt.imshow(frame, cmap='gray')
        plt.title(f"Frame {i}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{args.sample_index}_seg{args.segmentation_type}_alpha{args.alpha}_gray{args.gray_value}_frame{i}.png")
        plt.close()
    
    # Print sample information
    print(f"Sample information:")
    for key, value in syn_info.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 