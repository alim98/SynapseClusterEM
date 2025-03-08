"""
Script to demonstrate and fix the issue with grayscale values changing across slices
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from synapse import config, SynapseDataset, Synapse3DProcessor
from inference import load_and_prepare_data
from synapse.data.dataloader import normalize_cube_globally

def main():
    # Create output directories
    os.makedirs("results/original_slices", exist_ok=True)
    os.makedirs("results/fixed_slices", exist_ok=True)
    
    print("Loading data...")
    vol_data_dict, syn_df = load_and_prepare_data(config)
    
    # Create two different processors
    processor_original = Synapse3DProcessor(size=config.size)
    processor_fixed = Synapse3DProcessor(size=config.size)
    
    # Set normalization settings
    if hasattr(processor_fixed, 'normalize_volume'):
        processor_fixed.normalize_volume = True
    
    # Create two datasets - one with per-slice normalization, one with volume-wide normalization
    dataset_original = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor_original,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha,
        normalize_across_volume=False  # Original behavior - per-slice normalization
    )
    
    dataset_fixed = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor_fixed,
        segmentation_type=config.segmentation_type,
        alpha=config.alpha,
        normalize_across_volume=True  # Fixed behavior - volume-wide normalization
    )
    
    # Process a sample
    sample_idx = 0
    print(f"Processing sample {sample_idx}...")
    
    # Get data from both datasets
    pixel_values_original, _, bbox_name = dataset_original[sample_idx]
    pixel_values_fixed, _, _ = dataset_fixed[sample_idx]
    
    # Direct fix using the normalize_cube_globally function
    # This can be used on any existing cube with inconsistent grayscale values
    data_loader = dataset_original.data_loader
    if data_loader is None:
        from synapse import SynapseDataLoader
        data_loader = SynapseDataLoader("", "", "")
    
    # Get original sample for direct normalization
    raw_vol, seg_vol, add_mask_vol = vol_data_dict.get(bbox_name, (None, None, None))
    
    if raw_vol is not None:
        # Get coordinates from dataframe
        sample_info = syn_df[syn_df['bbox_name'] == bbox_name].iloc[0]
        central_coord = (
            int(sample_info['central_coord_1']), 
            int(sample_info['central_coord_2']), 
            int(sample_info['central_coord_3'])
        )
        side1_coord = (
            int(sample_info['side_1_coord_1']), 
            int(sample_info['side_1_coord_2']), 
            int(sample_info['side_1_coord_3'])
        )
        side2_coord = (
            int(sample_info['side_2_coord_1']), 
            int(sample_info['side_2_coord_2']), 
            int(sample_info['side_2_coord_3'])
        )
        
        # Create cube with original behavior (per-slice normalization)
        original_cube = data_loader.create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=config.segmentation_type,
            subvolume_size=config.subvol_size,
            alpha=config.alpha,
            bbox_name=bbox_name,
            normalize_across_volume=False
        )
        
        # Apply global normalization to fix the issue
        fixed_cube = normalize_cube_globally(original_cube)
        
        # Display and save slices to demonstrate the difference
        slice_indices = [0, 5, 10, 15]
        
        # Create comparison figure
        fig, axes = plt.subplots(len(slice_indices), 2, figsize=(10, len(slice_indices) * 4))
        
        for i, slice_idx in enumerate(slice_indices):
            # Original slice with per-slice normalization
            orig_slice = original_cube[:, :, :, slice_idx]
            axes[i, 0].imshow(orig_slice)
            axes[i, 0].set_title(f"Original (Slice {slice_idx})")
            axes[i, 0].axis('off')
            
            # Fixed slice with global normalization
            fixed_slice = fixed_cube[:, :, :, slice_idx]
            axes[i, 1].imshow(fixed_slice)
            axes[i, 1].set_title(f"Fixed (Slice {slice_idx})")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig("results/grayscale_comparison.png", dpi=300)
        plt.close()
        
        print(f"Saved comparison image to results/grayscale_comparison.png")
        
        # Save individual slices for detailed inspection
        for slice_idx in range(16):
            # Save original slice
            slice_fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(original_cube[:, :, :, slice_idx])
            ax.set_title(f"Original (Slice {slice_idx})")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"results/original_slices/slice_{slice_idx}.png", dpi=200)
            plt.close()
            
            # Save fixed slice
            slice_fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(fixed_cube[:, :, :, slice_idx])
            ax.set_title(f"Fixed (Slice {slice_idx})")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"results/fixed_slices/slice_{slice_idx}.png", dpi=200)
            plt.close()
        
        print(f"Saved individual slices to results/original_slices/ and results/fixed_slices/")

if __name__ == "__main__":
    main() 