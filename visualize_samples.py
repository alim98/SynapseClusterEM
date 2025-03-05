#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to visualize random synapse samples.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import random
import imageio.v3 as imageio
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import json
import argparse

# Import necessary functions from your existing code
from synapse_analysis.data.data_loader import load_synapse_data, load_all_volumes, GlobalNormalizationCalculator
from synapse_analysis.utils.processing import create_segmented_cube

# Configuration - using your provided paths
RAW_BASE_DIR = "data/7_bboxes_plus_seg/raw"
SEG_BASE_DIR = "data/7_bboxes_plus_seg/seg"
ADD_MASK_BASE_DIR = "data/vesicle_cloud__syn_interface__mitochondria_annotation"
EXCEL_DIR = "data/7_bboxes_plus_seg"
OUTPUT_DIR = "outputs/visualizations"
BBOX_NAMES = ["bbox1", "bbox2"]
SEGMENTATION_TYPE = 1
ALPHA = 0.1
SUBVOL_SIZE = 80
NUM_SAMPLES = 5
CREATE_VIDEOS = True  # Set to True to create videos
VIDEO_FPS = 10  # Frames per second for the video
VIDEO_SKIP_FRAMES = 1  # Skip frames to make video smaller (1 = use every frame, 2 = use every other frame, etc.)
USE_GLOBAL_NORM = False  # Whether to use global normalization
GLOBAL_STATS_PATH = "global_stats.json"  # Path to global normalization stats

def normalize_slice(slice_img, global_stats=None):
    """
    Normalize a slice for visualization.
    
    Args:
        slice_img: 2D slice to normalize
        global_stats: Optional global stats for normalization
    
    Returns:
        Normalized slice
    """
    # If global stats are provided, use them for normalization
    if global_stats is not None:
        # Convert to float if needed
        if slice_img.dtype != np.float32 and slice_img.dtype != np.float64:
            slice_img = slice_img.astype(np.float32)
        
        # Apply global normalization
        mean = global_stats['mean'][0]
        std = global_stats['std'][0]
        normalized = (slice_img - mean) / std
        
        # Scale to 0-255 range for visualization
        min_val = normalized.min()
        max_val = normalized.max()
        if max_val > min_val:
            normalized = 255.0 * (normalized - min_val) / (max_val - min_val)
        
        return normalized
    
    # Otherwise, use standard normalization
    # Create a mask for the gray regions (where value is exactly 128.0)
    gray_mask = np.isclose(slice_img, 128.0)
    normalized_slice = slice_img.copy()
    
    # Replace the gray value with a special value temporarily
    special_value = -1000
    normalized_slice[gray_mask] = special_value
    
    # Normalize the non-gray parts
    non_gray_mask = ~gray_mask
    if np.any(non_gray_mask):
        min_val = np.min(normalized_slice[non_gray_mask])
        max_val = np.max(normalized_slice[non_gray_mask])
        
        # Avoid division by zero
        if max_val > min_val:
            # Scale the non-gray parts to 0-255
            normalized_slice[non_gray_mask] = 255.0 * (normalized_slice[non_gray_mask] - min_val) / (max_val - min_val)
    
    # Set the gray parts to a consistent value (128)
    normalized_slice[gray_mask] = 128.0
    
    return normalized_slice

def visualize_center_slice(volume, title=None, filename=None, output_dir=None, global_stats=None):
    """
    Visualize the center slice of a 3D volume.
    
    Args:
        volume: 3D or 4D numpy array representing the volume
        title: Title for the visualization
        filename: Filename to save the visualization
        output_dir: Directory to save the visualization
        global_stats: Optional global stats for normalization
    """
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Handle different volume shapes
    if volume.ndim == 4:  # Shape like (80, 80, 1, 80)
        # For 4D volumes, extract the center slice from the last dimension
        center_idx = volume.shape[3] // 2
        center_slice = volume[:, :, 0, center_idx]  # Extract (H, W) slice
    elif volume.ndim == 3:  # Shape like (80, 80, 80)
        # For 3D volumes, extract the center slice from the first dimension
        center_idx = volume.shape[0] // 2
        center_slice = volume[center_idx]
    else:
        raise ValueError(f"Unsupported volume shape: {volume.shape}")
    
    # Print debug info
    print(f"Visualizing slice with shape: {center_slice.shape}")
    
    # Normalize the slice
    normalized_slice = normalize_slice(center_slice, global_stats)
    
    # Create figure and plot
    plt.figure(figsize=(8, 8))
    plt.imshow(normalized_slice, cmap='gray', vmin=0, vmax=255)
    if title:
        plt.title(title)
    plt.axis('off')
    
    # Save the visualization if output_dir is provided
    if output_dir and filename:
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()

def create_video(volume, title=None, filename=None, output_dir=None, fps=10, skip_frames=1, global_stats=None):
    """
    Create a video flipping through the slices of a 3D volume.
    
    Args:
        volume: 3D or 4D numpy array representing the volume
        title: Title for the video frames
        filename: Filename to save the video
        output_dir: Directory to save the video
        fps: Frames per second for the video
        skip_frames: Skip frames to make video smaller
        global_stats: Optional global stats for normalization
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Handle different volume shapes
    if volume.ndim == 4:  # Shape like (80, 80, 1, 80)
        depth = volume.shape[3]
        frames = []
        
        for slice_idx in range(0, depth, skip_frames):
            slice_img = volume[:, :, 0, slice_idx]  # Extract (H, W) slice
            
            # Normalize the slice
            normalized_slice = normalize_slice(slice_img, global_stats)
            
            # Create a figure with the slice
            fig = Figure(figsize=(8, 8), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.imshow(normalized_slice, cmap='gray', vmin=0, vmax=255)
            
            if title:
                ax.set_title(f"{title} - Slice {slice_idx+1}/{depth}")
            
            ax.axis('off')
            fig.tight_layout(pad=0)
            
            # Convert figure to image - using the updated method for newer matplotlib versions
            canvas.draw()
            width, height = fig.canvas.get_width_height()
            
            # Use buffer_rgba instead of tostring_rgb
            buf = canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype='uint8').reshape(height, width, 4)
            # Convert RGBA to RGB
            img = img[:, :, :3]
            
            frames.append(img)
            plt.close(fig)
    
    elif volume.ndim == 3:  # Shape like (80, 80, 80)
        depth = volume.shape[0]
        frames = []
        
        for slice_idx in range(0, depth, skip_frames):
            slice_img = volume[slice_idx]
            
            # Normalize the slice
            normalized_slice = normalize_slice(slice_img, global_stats)
            
            # Create a figure with the slice
            fig = Figure(figsize=(8, 8), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.imshow(normalized_slice, cmap='gray', vmin=0, vmax=255)
            
            if title:
                ax.set_title(f"{title} - Slice {slice_idx+1}/{depth}")
            
            ax.axis('off')
            fig.tight_layout(pad=0)
            
            # Convert figure to image - using the updated method for newer matplotlib versions
            canvas.draw()
            width, height = fig.canvas.get_width_height()
            
            # Use buffer_rgba instead of tostring_rgb
            buf = canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype='uint8').reshape(height, width, 4)
            # Convert RGBA to RGB
            img = img[:, :, :3]
            
            frames.append(img)
            plt.close(fig)
    
    else:
        raise ValueError(f"Unsupported volume shape: {volume.shape}")
    
    # Save the video if output_dir and filename are provided
    if output_dir and filename and frames:
        save_path = os.path.join(output_dir, filename)
        # For videos, need to use cv2 since imageio.v3 imwrite doesn't support fps
        # First change extension from mp4 to png if necessary
        base_path = os.path.splitext(save_path)[0]
        
        # Save individual frames as PNG files
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = f"{base_path}_frame_{i:03d}.png"
            imageio.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        # Use OpenCV to create the video from PNG files
        if frame_paths:
            # Get dimensions from the first frame
            first_frame = cv2.imread(frame_paths[0])
            height, width, layers = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            # Add frames to video
            for frame_path in frame_paths:
                video.write(cv2.imread(frame_path))
            
            # Release video writer
            video.release()
            
            # Clean up temporary frame files
            for frame_path in frame_paths:
                os.remove(frame_path)
            
            print(f"Saved video to {save_path}")

def visualize_grid(volumes, titles, output_dir, filename="grid.png", global_stats=None):
    """
    Visualize multiple volumes in a grid layout.
    
    Args:
        volumes: List of 3D or 4D volumes to visualize
        titles: List of titles for each volume
        output_dir: Directory to save the visualization
        filename: Filename to save the visualization
        global_stats: Optional global stats for normalization
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    num_volumes = len(volumes)
    if num_volumes == 0:
        return
    
    # Determine grid dimensions
    cols = min(4, num_volumes)
    rows = (num_volumes + cols - 1) // cols
    
    # Setup the figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, (volume, title) in enumerate(zip(volumes, titles)):
        if i >= len(axes):
            break
            
        # Handle different volume shapes
        if volume.ndim == 4:  # Shape like (80, 80, 1, 80)
            # For 4D volumes, extract the center slice from the last dimension
            center_idx = volume.shape[3] // 2
            center_slice = volume[:, :, 0, center_idx]  # Extract (H, W) slice
        elif volume.ndim == 3:  # Shape like (80, 80, 80)
            # For 3D volumes, extract the center slice from the first dimension
            center_idx = volume.shape[0] // 2
            center_slice = volume[center_idx]
        else:
            print(f"Warning: Unsupported volume shape: {volume.shape}. Skipping.")
            continue
        
        # Normalize the slice
        normalized_slice = normalize_slice(center_slice, global_stats)
        
        # Display the slice
        axes[i].imshow(normalized_slice, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(volumes), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    if output_dir:
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid visualization to {save_path}")
    
    plt.close()

def main():
    """Main function to visualize random samples."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize random synapse samples')
    parser.add_argument('--use_global_norm', action='store_true', help='Use global normalization')
    parser.add_argument('--global_stats_path', type=str, default=GLOBAL_STATS_PATH, 
                        help='Path to global normalization stats file')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES,
                        help='Number of random samples to visualize')
    args = parser.parse_args()
    
    # Update configuration from arguments
    output_dir = Path(args.output_dir)
    num_samples = args.num_samples
    use_global_norm = args.use_global_norm
    global_stats_path = args.global_stats_path
    
    # Load or compute global normalization stats if needed
    global_stats = None
    if use_global_norm:
        print(f"Using global normalization with stats from {global_stats_path}")
        calculator = GlobalNormalizationCalculator(output_file=global_stats_path)
        
        # Calculate stats if file doesn't exist
        if not os.path.exists(global_stats_path):
            print(f"Computing global normalization stats...")
            calculator.compute_from_volumes(
                bbox_names=BBOX_NAMES,
                raw_base_dir=RAW_BASE_DIR,
                seg_base_dir=SEG_BASE_DIR,
                add_mask_base_dir=ADD_MASK_BASE_DIR
            )
        else:
            # Load existing stats
            calculator.load_stats()
        
        global_stats = calculator.stats
        print(f"Global stats: mean={global_stats['mean'][0]:.2f}, std={global_stats['std'][0]:.2f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a directory for videos if needed
    video_dir = output_dir / "videos"
    if CREATE_VIDEOS:
        os.makedirs(video_dir, exist_ok=True)
    
    print(f"Loading synapse data from {EXCEL_DIR}...")
    synapse_data = load_synapse_data(BBOX_NAMES, EXCEL_DIR)
    if len(synapse_data) == 0:
        print("No synapse data found. Check excel_dir path.")
        return
    
    # Print column names for debugging
    print(f"DataFrame columns: {synapse_data.columns.tolist()}")
    
    print(f"Loading volumes...")
    volumes = load_all_volumes(
        BBOX_NAMES, 
        RAW_BASE_DIR, 
        SEG_BASE_DIR, 
        ADD_MASK_BASE_DIR
    )
    if len(volumes) == 0:
        print("No volumes loaded. Check data paths.")
        return
    
    # Select random samples
    num_samples = min(num_samples, len(synapse_data))
    sample_indices = np.random.choice(len(synapse_data), num_samples, replace=False)
    
    print(f"Visualizing {num_samples} random samples...")
    
    # Store volumes and titles for grid visualization
    segmented_cubes = []
    titles = []
    
    for i, idx in enumerate(sample_indices):
        syn_info = synapse_data.iloc[idx]
        
        # Get information about this synapse
        bbox_name = syn_info['bbox_name']
        
        # Use the correct column names for coordinates
        # In case the actual column names are different
        if 'central_coord_1' in syn_info:
            # Original format
            center_x = int(syn_info['central_coord_1'])
            center_y = int(syn_info['central_coord_2'])
            center_z = int(syn_info['central_coord_3'])
            
            # Handle non-integer synapse IDs
            if 'Var1' in syn_info:
                try:
                    syn_id = int(syn_info['Var1'])
                except (ValueError, TypeError):
                    syn_id = str(syn_info['Var1'])
            else:
                syn_id = f"sample_{idx}"
        elif 'x' in syn_info:
            # New format
            center_x = int(syn_info['x'])
            center_y = int(syn_info['y'])
            center_z = int(syn_info['z'])
            
            # Handle non-integer synapse IDs
            if 'synapse_id' in syn_info:
                try:
                    syn_id = int(syn_info['synapse_id'])
                except (ValueError, TypeError):
                    syn_id = str(syn_info['synapse_id'])
            else:
                syn_id = f"sample_{idx}"
        else:
            # Default to first three numeric columns as coordinates
            numeric_cols = [col for col in syn_info.index if isinstance(syn_info[col], (int, float))][:3]
            if len(numeric_cols) >= 3:
                center_x = int(syn_info[numeric_cols[0]])
                center_y = int(syn_info[numeric_cols[1]])
                center_z = int(syn_info[numeric_cols[2]])
                syn_id = f"sample_{idx}"
            else:
                print(f"Warning: Could not determine coordinates for sample {i+1}. Skipping.")
                continue
        
        # Get the volume data
        if bbox_name not in volumes:
            print(f"Warning: {bbox_name} not found in loaded volumes. Skipping.")
            continue
            
        raw_vol, seg_vol, add_mask_vol = volumes[bbox_name]
        
        # Create a segmented cube around the center coordinates
        try:
            # Convert center coordinates to format expected by create_segmented_cube
            # create_segmented_cube expects (z,y,x) format for coordinates
            central_coord = (center_z, center_y, center_x)
            
            # Create dummy side coordinates by offsetting from center
            side1_coord = (center_z - 10, center_y, center_x)
            side2_coord = (center_z, center_y - 10, center_x)
            
            segmented_cube = create_segmented_cube(
                raw_vol=raw_vol,
                seg_vol=seg_vol,
                add_mask_vol=add_mask_vol,
                central_coord=central_coord,
                side1_coord=side1_coord,
                side2_coord=side2_coord,
                segmentation_type=SEGMENTATION_TYPE,
                subvolume_size=SUBVOL_SIZE,
                alpha=ALPHA
            )
            
            # Save this for the grid visualization
            segmented_cubes.append(segmented_cube)
            titles.append(f"ID: {syn_id}")
            
            # Visualize center slice
            title = f"Synapse ID: {syn_id} in {bbox_name}"
            filename = f"synapse_{syn_id}_slice.png"
            visualize_center_slice(segmented_cube, title, filename, output_dir, global_stats)
            
            # Create video if requested
            if CREATE_VIDEOS:
                video_filename = f"synapse_{syn_id}_volume.mp4"
                create_video(segmented_cube, title, video_filename, video_dir, VIDEO_FPS, VIDEO_SKIP_FRAMES, global_stats)
                
            print(f"Processed sample {i+1}/{num_samples}: Synapse {syn_id}")
        except Exception as e:
            print(f"Error processing sample {i+1}/{num_samples}: {e}")
    
    # Create grid visualization of all samples
    if segmented_cubes:
        grid_filename = "synapse_grid.png"
        if use_global_norm:
            grid_filename = "synapse_grid_global_norm.png"
        visualize_grid(segmented_cubes, titles, output_dir, grid_filename, global_stats)
    
    print(f"Visualization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 