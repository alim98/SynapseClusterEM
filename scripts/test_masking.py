#!/usr/bin/env python
"""
Test Masking Script

This script tests different masking techniques to help fix the issue with 
rendering gray in unmasked regions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Import from the package level
from synapse_analysis.data import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data,
    SynapseDataset
)
from synapse_analysis.utils.processing import create_segmented_cube

def create_segmented_cube_alternative(raw_vol, seg_vol, add_mask_vol, 
                                     central_coord, side1_coord, side2_coord,
                                     segmentation_type, subvolume_size=80, alpha=0.3,
                                     bbox_name=""):
    """
    Alternative implementation of create_segmented_cube with debug code.
    This is an exact copy of the original function with added debugging steps.
    """
    # Import the original function to use most of its code
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
    
    # Get vesicle mask
    vesicle_full_mask = (add_mask_vol == labels['vesicle_label'])
    vesicle_mask = get_closest_component_mask(
        vesicle_full_mask, z_start, z_end, y_start, y_end, x_start, x_end, central_coord
    )
    
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
    
    # Determine pre-synapse side
    overlap_side1 = np.sum(np.logical_and(mask_1_full, vesicle_mask))
    overlap_side2 = np.sum(np.logical_and(mask_2_full, vesicle_mask))
    presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
    
    # Create combined mask based on segmentation type
    if segmentation_type == 0:  # Raw data
        combined_mask_full = np.ones_like(add_mask_vol, dtype=bool)
    elif segmentation_type == 1:  # Presynapse
        combined_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
    elif segmentation_type == 2:  # Postsynapse
        combined_mask_full = mask_2_full if presynapse_side == 1 else mask_1_full
    elif segmentation_type == 3:  # Both sides
        combined_mask_full = np.logical_or(mask_1_full, mask_2_full)
    elif segmentation_type in [4, 9]:  # Vesicles + Cleft
        vesicle_closest = get_closest_component_mask(
            (add_mask_vol == labels['vesicle_label']), z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        cleft_closest = get_closest_component_mask(
            (add_mask_vol == labels['cleft_label']), z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        cleft_closest2 = get_closest_component_mask(
            (add_mask_vol == labels['cleft_label2']), z_start, z_end, y_start, y_end, x_start, x_end, central_coord
        )
        combined_mask_full = np.logical_or(vesicle_closest, np.logical_or(cleft_closest, cleft_closest2))
    else:
        raise ValueError(f"Unsupported segmentation type: {segmentation_type}")
    
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
    gray_color = 0.6
    
    # Store all variations for debugging
    results = {}
    
    # Create RGB version and apply masking
    raw_rgb = np.repeat(normalized[..., np.newaxis], 3, axis=-1)
    mask_factor = sub_combined_mask[..., np.newaxis]
    
    # Original blending
    if alpha < 1:
        blended_part = alpha * gray_color + (1 - alpha) * raw_rgb
    else:
        blended_part = gray_color * (1 - mask_factor) + raw_rgb * mask_factor
    
    overlaid_image_original = raw_rgb * mask_factor + (1 - mask_factor) * blended_part
    results['original'] = np.transpose(overlaid_image_original, (1, 2, 3, 0))
    
    # Alternative 1: Directly use gray for non-masked regions
    overlay_alt1 = raw_rgb.copy()
    gray_rgb = np.ones_like(raw_rgb) * gray_color
    overlay_alt1 = raw_rgb * mask_factor + gray_rgb * (1 - mask_factor)
    results['direct_gray'] = np.transpose(overlay_alt1, (1, 2, 3, 0))
    
    # Alternative 2: First try - fix the formula
    if alpha < 1:
        blended_part_alt2 = alpha * gray_color + (1 - alpha) * raw_rgb
        overlay_alt2 = raw_rgb * mask_factor + blended_part_alt2 * (1 - mask_factor)
    else:
        overlay_alt2 = raw_rgb * mask_factor + gray_color * (1 - mask_factor)
    results['fixed_blending'] = np.transpose(overlay_alt2, (1, 2, 3, 0))
    
    # Return all results for comparison
    return results

def main():
    # Define paths
    raw_base_dir = 'data/7_bboxes_plus_seg/raw'
    seg_base_dir = 'data/7_bboxes_plus_seg/seg'
    add_mask_base_dir = 'data/vesicle_cloud__syn_interface__mitochondria_annotation'
    excel_dir = 'data'
    bbox_names = ['bbox1']
    output_dir = Path('outputs/masking_tests')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading synapse data...")
    synapse_df = load_synapse_data(bbox_names, excel_dir)
    print(f"   Loaded {len(synapse_df)} synapse entries")
    
    print("\n2. Loading volume data...")
    vol_data_dict = load_all_volumes(
        bbox_names,
        raw_base_dir,
        seg_base_dir,
        add_mask_base_dir
    )
    print(f"   Loaded volumes for {len(vol_data_dict)} bounding boxes")
    
    # Extract the raw volumes for manual processing
    bbox_name = bbox_names[0]
    raw_vol, seg_vol, add_mask_vol = vol_data_dict[bbox_name]
    
    print("\n3. Testing with first sample...")
    sample_idx = 0
    syn_info = synapse_df.iloc[sample_idx]
    central_coord = (int(syn_info['central_coord_1']), 
                    int(syn_info['central_coord_2']), 
                    int(syn_info['central_coord_3']))
    side1_coord = (int(syn_info['side_1_coord_1']), 
                  int(syn_info['side_1_coord_2']), 
                  int(syn_info['side_1_coord_3']))
    side2_coord = (int(syn_info['side_2_coord_1']), 
                  int(syn_info['side_2_coord_2']), 
                  int(syn_info['side_2_coord_3']))
    
    segmentation_type = 1  # Presynaptic
    subvol_size = 80
    alpha = 0.3
    
    print(f"   Coordinates: Center={central_coord}, Side1={side1_coord}, Side2={side2_coord}")
    
    # Create different versions of segmented cubes for comparison
    results = create_segmented_cube_alternative(
        raw_vol=raw_vol,
        seg_vol=seg_vol,
        add_mask_vol=add_mask_vol,
        central_coord=central_coord,
        side1_coord=side1_coord,
        side2_coord=side2_coord,
        segmentation_type=segmentation_type,
        subvolume_size=subvol_size,
        alpha=alpha,
        bbox_name=bbox_name
    )
    
    # Also get the original version
    original_cube = create_segmented_cube(
        raw_vol=raw_vol,
        seg_vol=seg_vol,
        add_mask_vol=add_mask_vol,
        central_coord=central_coord,
        side1_coord=side1_coord,
        side2_coord=side2_coord,
        segmentation_type=segmentation_type,
        subvolume_size=subvol_size,
        alpha=alpha,
        bbox_name=bbox_name
    )
    
    # Visualize middle slices of each version
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original (current implementation)
    middle_z = original_cube.shape[0] // 2
    axs[0, 0].imshow(original_cube[middle_z])
    axs[0, 0].set_title("Current Implementation")
    
    # Alternative 1: Direct gray overlay
    middle_z = results['direct_gray'].shape[0] // 2
    axs[0, 1].imshow(results['direct_gray'][middle_z])
    axs[0, 1].set_title("Alternative 1: Direct Gray")
    
    # Alternative 2: Fixed blending
    middle_z = results['fixed_blending'].shape[0] // 2
    axs[1, 0].imshow(results['fixed_blending'][middle_z])
    axs[1, 0].set_title("Alternative 2: Fixed Blending")
    
    # Original approach from results
    middle_z = results['original'].shape[0] // 2
    axs[1, 1].imshow(results['original'][middle_z])
    axs[1, 1].set_title("Original from Alternative")
    
    # Save figure
    plt.tight_layout()
    fig.suptitle(f"Masking Method Comparison for Sample {sample_idx} from {bbox_name}\n"
                f"Center: {central_coord}", fontsize=16, y=1.02)
    
    save_path = output_dir / f"masking_comparison_{bbox_name}_{sample_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved comparison visualization to {save_path}")
    print("\nTest complete!")

if __name__ == "__main__":
    main() 