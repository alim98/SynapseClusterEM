"""
cleft Size Visualization Module

This module provides functions to visualize cleft cloud sizes in UMAP plots
and analyze the relationship between cleft sizes and clusters.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from glob import glob
from scipy import stats
from scipy.stats import gaussian_kde
from umap import UMAP
import statsmodels.api as sm
from skimage import measure  # Import for component analysis
import argparse
import torch
import tifffile
from typing import Tuple, Dict, List
from synapse import config

# Import from existing modules
from newdl.dataloader2 import SynapseDataLoader
from newdl.dataset2 import SynapseDataset

def get_cleft_label(bbox_name):
    """
    Determine cleft label based on bbox name.
    
    Args:
        bbox_name: Name of the bounding box 
    
    Returns:
        tuple: (cleft_label, cleft_label2) The labels to use for this bbox
    """
    # Use the mapping from config file if available
    if hasattr(config, 'CLEFT_LABEL_MAP') and bbox_name in config.CLEFT_LABEL_MAP:
        return config.CLEFT_LABEL_MAP[bbox_name]
    
    # Fallback to default mapping if not in config
    if hasattr(config, 'CLEFT_LABEL_MAP') and 'default' in config.CLEFT_LABEL_MAP:
        return config.CLEFT_LABEL_MAP['default']
    
    # Original implementation as further fallback
    bbox_num = bbox_name.replace("bbox", "").strip()
    if bbox_num in {'2', '5',}:
        cleft_label2 = 4
        cleft_label = 2
    elif bbox_num == '7':
        cleft_label2 = 3
        cleft_label = 4
    elif bbox_num == '4':
        cleft_label2 = 4
        cleft_label = 1
    elif bbox_num == '3':
        cleft_label2 = 8
        cleft_label = 9
    else:
        cleft_label = 7
        cleft_label2 = 7
    return cleft_label, cleft_label2

def calculate_cleft_cloud_size(row, vol_data_dict, subvol_size):
    """
    Calculate the cleft cloud mask size for a given synapse row.
    
    Args:
        row: Row from the synapse dataframe
        vol_data_dict: Dictionary containing volume data
        subvol_size: Subvolume size
        
    Returns:
        float: cleft cloud size as percentage of total 80×80×80 raw image
    """
    bbox_name = row['bbox_name']
    if bbox_name not in vol_data_dict:
        return 0  # Handle missing data

    add_mask_vol = vol_data_dict[bbox_name][2]  # Get cleft segmentation volume
    cleft_label, cleft_label2 = get_cleft_label(bbox_name)

    # Extract coordinates from the dataframe (x, y, z)
    cx, cy, cz = (
        int(row['central_coord_1']),
        int(row['central_coord_2']),
        int(row['central_coord_3'])
    )

    # Calculate subvolume bounds
    half_size = subvol_size // 2
    x_start = max(cx - half_size, 0)
    x_end = min(cx + half_size, add_mask_vol.shape[2])
    y_start = max(cy - half_size, 0)
    y_end = min(cy + half_size, add_mask_vol.shape[1])
    z_start = max(cz - half_size, 0)
    z_end = min(cz + half_size, add_mask_vol.shape[0])

    # Generate full cleft mask
    cleft_full_mask = (add_mask_vol == cleft_label) | (add_mask_vol == cleft_label2)
    
    # Extract subvolume mask
    cleft_mask = cleft_full_mask[z_start:z_end, y_start:y_end, x_start:x_end]

    # Count total cleft pixels
    total_cleft_pixels = np.sum(cleft_mask)
    
    # Calculate percentage based on the standard 80×80×80 volume size
    # regardless of actual subvolume boundaries
    standard_volume_size = 80 * 80 * 80
    cleft_size_percent = (total_cleft_pixels / standard_volume_size) * 100
    
    return cleft_size_percent  # Return as percentage of standard 80×80×80 volume

def calculate_cleft_slice_sizes(cleft_mask):
    """
    Calculate the size of cleft mask for each slice of the 3D volume
    
    Args:
        cleft_mask: 3D numpy array with the cleft mask
        
    Returns:
        dict: Dictionary with slice indices as keys and pixel counts as values
    """
    slice_sizes = {}
    
    # Calculate pixels per z-slice (axial)
    for z in range(cleft_mask.shape[0]):
        slice_sizes[f'z_{z}'] = np.sum(cleft_mask[z, :, :])
    
    # Calculate pixels per y-slice (coronal)
    for y in range(cleft_mask.shape[1]):
        slice_sizes[f'y_{y}'] = np.sum(cleft_mask[:, y, :])
    
    # Calculate pixels per x-slice (sagittal)
    for x in range(cleft_mask.shape[2]):
        slice_sizes[f'x_{x}'] = np.sum(cleft_mask[:, :, x])
        
    return slice_sizes

def find_max_cleft_slices(cleft_mask):
    """
    Find the slices with maximum cleft pixels in each dimension
    
    Args:
        cleft_mask: 3D numpy array with the cleft mask
        
    Returns:
        dict: Dictionary with max slice indices for each dimension
    """
    # For z dimension (axial)
    z_sums = [np.sum(cleft_mask[z, :, :]) for z in range(cleft_mask.shape[0])]
    max_z = np.argmax(z_sums)
    
    # For y dimension (coronal)
    y_sums = [np.sum(cleft_mask[:, y, :]) for y in range(cleft_mask.shape[1])]
    max_y = np.argmax(y_sums)
    
    # For x dimension (sagittal)
    x_sums = [np.sum(cleft_mask[:, :, x]) for x in range(cleft_mask.shape[2])]
    max_x = np.argmax(x_sums)
    
    return {
        'max_z_slice': max_z,
        'max_z_value': z_sums[max_z],
        'max_y_slice': max_y,
        'max_y_value': y_sums[max_y],
        'max_x_slice': max_x,
        'max_x_value': x_sums[max_x]
    }

def compute_cleft_cloud_sizes(syn_df, vol_data_dict, args, output_dir):
    """
    Compute cleft cloud size for each synapse and save to CSV.
    
    Args:
        syn_df: Synapse dataframe
        vol_data_dict: Dictionary containing volume data
        args: Arguments containing configuration
        output_dir: Directory to save results
        
    Returns:
        pandas.DataFrame: Synapse dataframe with cleft cloud sizes (as percentage of 80×80×80 raw image)
    """
    print("Calculating cleft cloud sizes (as % of 80×80×80 raw image)...")
    syn_df['cleft_cloud_size'] = syn_df.apply(
        lambda row: calculate_cleft_cloud_size(row, vol_data_dict, args.subvol_size),
        axis=1
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each bbox's dataframe with the new column
    for bbox in args.bbox_name:
        bbox_df = syn_df[syn_df['bbox_name'] == bbox]
        if not bbox_df.empty:
            output_path = os.path.join(output_dir, f"{bbox}.csv")
            bbox_df.to_csv(output_path, index=False)
            print(f"Saved {output_path} with cleft cloud sizes (as % of 80×80×80 raw image).")
            
    return syn_df

def visualize_3d_volumes_with_max_slices(syn_df, vol_data_dict, output_dir, sample_indices=None):
    """
    Visualize 3D volumes and their max cleft slices for selected samples
    
    Args:
        syn_df: Synapse dataframe
        vol_data_dict: Dictionary containing volume data
        output_dir: Directory to save visualizations
        sample_indices: List of indices to visualize (default: first 4 in dataframe)
    
    Returns:
        None
    """
    if sample_indices is None:
        # Default to first 4 samples if not specified
        sample_indices = list(range(min(4, len(syn_df))))
    
    # Ensure output directory exists
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    for idx in sample_indices:
        if idx >= len(syn_df):
            print(f"Index {idx} out of range for dataframe with {len(syn_df)} rows")
            continue
            
        row = syn_df.iloc[idx]
        bbox_name = row['bbox_name']
        
        if bbox_name not in vol_data_dict:
            print(f"Data for bbox {bbox_name} not found in volume data dictionary")
            continue
            
        # Get raw and segmentation volumes
        raw_vol = vol_data_dict[bbox_name][0]
        add_mask_vol = vol_data_dict[bbox_name][2]
        
        cleft_label, cleft_label2 = get_cleft_label(bbox_name)
        
        # Extract coordinates 
        cx, cy, cz = (
            int(row['central_coord_1']),
            int(row['central_coord_2']),
            int(row['central_coord_3'])
        )
        
        # Calculate subvolume bounds with 80x80x80 size
        half_size = 40  # Half of 80
        x_start = max(cx - half_size, 0)
        x_end = min(cx + half_size, add_mask_vol.shape[2])
        y_start = max(cy - half_size, 0)
        y_end = min(cy + half_size, add_mask_vol.shape[1])
        z_start = max(cz - half_size, 0)
        z_end = min(cz + half_size, add_mask_vol.shape[0])
        
        # Extract subvolumes
        raw_subvol = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Create cleft mask
        cleft_full_mask = (add_mask_vol == cleft_label) | (add_mask_vol == cleft_label2)
        cleft_mask = cleft_full_mask[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Find max slices
        max_slices = find_max_cleft_slices(cleft_mask)
        
        # Create figure with subplots for max slices in each dimension
        fig = go.Figure()
        
        # Add 3D volume visualization
        fig = go.Figure(data=go.Volume(
            x=np.arange(cleft_mask.shape[2]),
            y=np.arange(cleft_mask.shape[1]),
            z=np.arange(cleft_mask.shape[0]),
            value=cleft_mask.flatten(),
            opacity=0.2,
            surface_count=20,
            colorscale='Reds',
            name="Cleft Cloud"
        ))
        
        # Save 3D visualization
        fig.update_layout(
            title=f"3D Cleft Visualization - {bbox_name} (ID: {row.name})",
            width=800,
            height=800
        )
        fig.write_html(os.path.join(viz_dir, f"{bbox_name}_id{row.name}_3d.html"))
        
        # Create a separate figure for max slices
        fig_slices = go.Figure(
            data=[
                go.Heatmap(
                    z=raw_subvol[max_slices['max_z_slice'], :, :],
                    colorscale='Gray',
                    showscale=False,
                ),
                go.Contour(
                    z=cleft_mask[max_slices['max_z_slice'], :, :].astype(float),
                    colorscale='Reds',
                    showscale=False,
                    opacity=0.5,
                    contours=dict(start=0.5, end=1, size=0.5),
                )
            ],
            layout=go.Layout(
                title=f"Max Z-Slice ({max_slices['max_z_slice']}) - Pixels: {max_slices['max_z_value']}",
                width=500,
                height=500
            )
        )
        fig_slices.write_html(os.path.join(viz_dir, f"{bbox_name}_id{row.name}_max_z_slice.html"))
        
        # Create a similar figure for max Y slice
        fig_y = go.Figure(
            data=[
                go.Heatmap(
                    z=raw_subvol[:, max_slices['max_y_slice'], :].T,
                    colorscale='Gray',
                    showscale=False,
                ),
                go.Contour(
                    z=cleft_mask[:, max_slices['max_y_slice'], :].T.astype(float),
                    colorscale='Reds',
                    showscale=False,
                    opacity=0.5,
                    contours=dict(start=0.5, end=1, size=0.5),
                )
            ],
            layout=go.Layout(
                title=f"Max Y-Slice ({max_slices['max_y_slice']}) - Pixels: {max_slices['max_y_value']}",
                width=500,
                height=500
            )
        )
        fig_y.write_html(os.path.join(viz_dir, f"{bbox_name}_id{row.name}_max_y_slice.html"))
        
        # Create a similar figure for max X slice
        fig_x = go.Figure(
            data=[
                go.Heatmap(
                    z=raw_subvol[:, :, max_slices['max_x_slice']],
                    colorscale='Gray',
                    showscale=False,
                ),
                go.Contour(
                    z=cleft_mask[:, :, max_slices['max_x_slice']].astype(float),
                    colorscale='Reds',
                    showscale=False,
                    opacity=0.5,
                    contours=dict(start=0.5, end=1, size=0.5),
                )
            ],
            layout=go.Layout(
                title=f"Max X-Slice ({max_slices['max_x_slice']}) - Pixels: {max_slices['max_x_value']}",
                width=500,
                height=500
            )
        )
        fig_x.write_html(os.path.join(viz_dir, f"{bbox_name}_id{row.name}_max_x_slice.html"))
        
        print(f"Saved visualizations for sample {idx} (bbox: {bbox_name}, ID: {row.name})")


def load_data_with_existing_dataloader(args):
    """
    Load data using the existing SynapseDataLoader class.
    
    Args:
        args: Arguments containing data paths and configuration
        
    Returns:
        tuple: (synapse dataframe, volume data dictionary)
    """
    print(f"Loading data from {args.data_root}...")
    
    # Initialize the dataloader from the existing class
    data_loader = SynapseDataLoader(
        raw_base_dir=args.raw_base_dir,
        seg_base_dir=args.seg_base_dir,
        add_mask_base_dir=args.add_mask_base_dir
    )
    
    # Dictionary to store volume data
    vol_data_dict = {}
    
    # List to collect all dataframes
    all_dfs = []
    
    for bbox in args.bbox_name:
        print(f"Processing {bbox}...")
        
        # Load volumes using the existing loader
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox)
        if raw_vol is not None:
            vol_data_dict[bbox] = (raw_vol, seg_vol, add_mask_vol)
        else:
            print(f"Warning: Volumes not found for {bbox}")
            continue
            
        # Load annotation CSV
        if args.excel_file:
            try:
                excel_path = os.path.join(args.excel_file, f"{bbox}.xlsx")
                if os.path.exists(excel_path):
                    df = pd.read_excel(excel_path)
                    df['bbox_name'] = bbox  # Add bbox name to dataframe
                    all_dfs.append(df)
                else:
                    print(f"Warning: Excel annotations not found for {bbox}")
            except Exception as e:
                print(f"Error loading Excel file for {bbox}: {e}")
        else:
            csv_path = os.path.join(args.data_root, "annotations", f"{bbox}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['bbox_name'] = bbox  # Add bbox name to dataframe
                all_dfs.append(df)
            else:
                print(f"Warning: CSV annotations not found for {bbox}")
            
    # Combine all dataframes
    if all_dfs:
        syn_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Loaded {len(syn_df)} synapse annotations across {len(args.bbox_name)} bboxes")
    else:
        syn_df = pd.DataFrame()
        print("Warning: No annotations found")
        
    return syn_df, vol_data_dict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cleft Size Visualization')
    
    # Data paths - use existing config if available
    parser.add_argument('--raw_base_dir', type=str, default=config.raw_base_dir,
                        help='Base directory for raw volumes')
    parser.add_argument('--seg_base_dir', type=str, default=config.seg_base_dir,
                        help='Base directory for segmentation volumes')
    parser.add_argument('--add_mask_base_dir', type=str, default=config.add_mask_base_dir,
                        help='Base directory for additional mask volumes')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for data if not using base_dirs')
    parser.add_argument('--output_dir', type=str, default='results/cleft_analysis',
                        help='Directory to save outputs')
    
    # Excel file location
    parser.add_argument('--excel_file', type=str, default=config.excel_file,
                        help='Directory containing excel files with synapse annotations')
    
    # Data selection
    parser.add_argument('--bbox_name', type=str, nargs='+', default=config.bbox_name,
                        help='Bounding box names to process')
    
    # Analysis parameters
    parser.add_argument('--subvol_size', type=int, default=config.subvol_size,
                        help='Size of subvolume for analysis')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                        help='Sample indices to visualize (default: first 4)')
    
    return parser.parse_args()


def main():
    """Main function to run the analysis."""
    # Parse arguments
    args = parse_args()
    args.output_dir = 'results/cleft_size_analysis'
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data using existing dataloader
    syn_df, vol_data_dict = load_data_with_existing_dataloader(args)
    
    if syn_df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Compute cleft cloud sizes
    syn_df = compute_cleft_cloud_sizes(syn_df, vol_data_dict, args, args.output_dir)
    
    # Visualize selected samples
    visualize_3d_volumes_with_max_slices(
        syn_df, 
        vol_data_dict, 
        args.output_dir, 
        args.sample_indices
    )
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
