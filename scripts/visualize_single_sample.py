#!/usr/bin/env python
"""
Visualize Single Sample Script

This script loads a single sample from the dataset and visualizes it in multiple ways
to help diagnose visualization issues.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import imageio.v2 as iio
import glob

# Import from the package level
from synapse_analysis.data import (
    Synapse3DProcessor,
    load_all_volumes,
    load_synapse_data,
    SynapseDataset
)

def main():
    # Define paths
    raw_base_dir = 'data/7_bboxes_plus_seg/raw'
    seg_base_dir = 'data/7_bboxes_plus_seg/seg'
    add_mask_base_dir = 'data/vesicle_cloud__syn_interface__mitochondria_annotation'
    excel_dir = 'data/7_bboxes_plus_seg'
    bbox_names = ['bbox1']
    output_dir = Path('outputs/single_sample_visualization')
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
    
    # Check if we have raw data
    if len(vol_data_dict) == 0:
        print("No volumes were loaded. Checking raw data files directly...")
        raw_dir = os.path.join(raw_base_dir, 'bbox1')
        raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
        print(f"Found {len(raw_tif_files)} raw TIF files in {raw_dir}")
        
        if len(raw_tif_files) > 0:
            # Load a sample raw image directly
            sample_img = iio.imread(raw_tif_files[len(raw_tif_files)//2])
            plt.figure(figsize=(10, 10))
            plt.imshow(sample_img, cmap='gray')
            plt.title(f"Raw Image (min={sample_img.min()}, max={sample_img.max()})")
            plt.colorbar()
            plt.savefig(output_dir / "raw_sample_direct.png")
            plt.close()
            print(f"Saved direct raw sample visualization to {output_dir / 'raw_sample_direct.png'}")
    
    # If we have volume data, proceed with dataset creation
    if len(vol_data_dict) > 0:
        print("\n3. Creating dataset...")
        # Create processor with no normalization for diagnostic purposes
        processor = Synapse3DProcessor(
            size=(80, 80),
            mean=(0.0,),
            std=(1.0,),
            apply_global_norm=False
        )
        
        dataset = SynapseDataset(
            vol_data_dict=vol_data_dict,
            synapse_df=synapse_df,
            processor=processor,
            segmentation_type=9,
            subvol_size=80,
            num_frames=80,
            alpha=1.0
        )
        print(f"   Dataset created with {len(dataset)} samples")
        
        print("\n4. Visualizing a single sample...")
        sample_idx = 0  # Use the first sample
        sample_data = dataset[sample_idx]
        
        # Unpack the sample data
        sample = sample_data[0]  # First element is the volume tensor
        metadata = sample_data[1]  # Second element is synapse info
        bbox_name = sample_data[2]  # Third element is bbox_name
        
        print(f"   Sample shape: {sample.shape}")
        print(f"   Metadata: {metadata}")
        print(f"   Bbox name: {bbox_name}")
        
        # Take the middle frame from the sequence
        middle_frame_idx = sample.shape[1] // 2
        middle_frame = sample[:, middle_frame_idx, :, :]
        
        # Create a multi-panel figure
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot each channel
        channel_names = ['Raw', 'Segmentation', 'Mask'] if sample.shape[0] == 3 else [f'Channel {j}' for j in range(sample.shape[0])]
        
        for j in range(min(3, sample.shape[0])):
            # Original normalized image
            axs[0, j].imshow(middle_frame[j], cmap='gray')
            axs[0, j].set_title(f'{channel_names[j]} (Normalized)')
            axs[0, j].set_xlabel(f'Min: {middle_frame[j].min():.3f}, Max: {middle_frame[j].max():.3f}')
            
            # Enhanced contrast
            p_low, p_high = np.percentile(middle_frame[j].numpy(), [2, 98])
            enhanced = np.clip((middle_frame[j].numpy() - p_low) / (p_high - p_low), 0, 1)
            axs[1, j].imshow(enhanced, cmap='gray')
            axs[1, j].set_title(f'{channel_names[j]} (Enhanced)')
            axs[1, j].set_xlabel(f'Range: {p_low:.3f} to {p_high:.3f}')
        
        plt.tight_layout()
        fig.suptitle(f"Sample {sample_idx} from {bbox_name}\n"
                    f"Synapse ID: {metadata.get('Var1', 'Unknown')}",
                    fontsize=16, y=1.02)
        
        # Save the figure
        save_path = output_dir / f"sample_{sample_idx}_{bbox_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved visualization to {save_path}")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main() 