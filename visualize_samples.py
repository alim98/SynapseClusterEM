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

# Import necessary functions from your existing code
from synapse_analysis.data.data_loader import load_synapse_data, load_all_volumes
from synapse_analysis.utils.processing import create_segmented_cube

# Configuration - using your provided paths
RAW_BASE_DIR = "data/7_bboxes_plus_seg/raw"
SEG_BASE_DIR = "data/7_bboxes_plus_seg/seg"
ADD_MASK_BASE_DIR = "data/vesicle_cloud__syn_interface__mitochondria_annotation"
EXCEL_DIR = "data/7_bboxes_plus_seg"
OUTPUT_DIR = "outputs/visualizations"
BBOX_NAMES = ["bbox1", "bbox2","bbox3", "bbox4","bbox5", "bbox6","bbox7"]
SEGMENTATION_TYPE = 1
ALPHA = 0.8
SUBVOL_SIZE = 80
NUM_SAMPLES = 10
CREATE_VIDEOS = True  # Set to True to create videos
VIDEO_FPS = 10  # Frames per second for the video
VIDEO_SKIP_FRAMES = 1  # Skip frames to make video smaller (1 = use every frame, 2 = use every other frame, etc.)

def normalize_slice(slice_img):
    """Normalize a slice for visualization."""
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

def visualize_center_slice(volume, title=None, filename=None, output_dir=None):
    """Simple function to visualize the center slice of a 3D volume."""
    # Extract center slice
    if volume.ndim == 4:  # [H, W, D, C]
        center_slice_idx = volume.shape[2] // 2
        center_slice = volume[:, :, center_slice_idx, :]
        slice_img = center_slice[:, :, 0]
    elif volume.ndim == 3:  # [H, W, D]
        center_slice_idx = volume.shape[2] // 2
        slice_img = volume[:, :, center_slice_idx]
    else:
        raise ValueError(f"Unsupported volume shape: {volume.shape}")
    
    # Normalize the slice
    normalized_slice = normalize_slice(slice_img)
    
    # Create figure and plot
    plt.figure(figsize=(8, 8))
    plt.imshow(normalized_slice, cmap='gray', vmin=0, vmax=255)
    if title:
        plt.title(title)
    plt.axis('off')
    
    # Save the visualization if output_dir is provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    # Close the figure instead of showing it
    plt.close()

def create_video(volume, title=None, filename=None, output_dir=None, fps=10, skip_frames=1):
    """Create a video from a 3D volume by showing slices sequentially."""
    if output_dir is None or filename is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, filename)
    
    # # Print debug information about the volume
    # print(f"[DEBUG] Volume stats for {title}:")
    # print(f"[DEBUG] - Shape: {volume.shape}")
    # print(f"[DEBUG] - Min value: {np.min(volume)}")
    # print(f"[DEBUG] - Max value: {np.max(volume)}")
    # print(f"[DEBUG] - Mean value: {np.mean(volume)}")
    # print(f"[DEBUG] - Std dev: {np.std(volume)}")
    
    # For 4D volumes [H, W, C, D], we want to iterate through all D slices (depth)
    if volume.ndim == 4:  # [H, W, C, D]
        num_slices = volume.shape[3]  # Fourth dimension is depth
        height, width = volume.shape[0], volume.shape[1]
        
        # Use a subset of slices to make the video smaller
        slice_indices = range(0, num_slices, skip_frames)
        print(f"[DEBUG] Creating video with {len(slice_indices)} frames from {num_slices} total slices")
        
        # Create a temporary directory for frames
        temp_dir = os.path.join(output_dir, "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate frames
        frame_files = []
        for i, slice_idx in enumerate(slice_indices):
            # Extract the slice at this depth index
            slice_img = volume[:, :, 0, slice_idx]  # Get the slice at depth index slice_idx
            
            # Normalize the slice
            normalized_slice = normalize_slice(slice_img)
            
            # Convert to uint8
            frame_img = normalized_slice.astype(np.uint8)
            
            # Save frame to temporary file
            frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
            
            # Create a figure with title
            plt.figure(figsize=(8, 8))
            plt.imshow(frame_img, cmap='gray', vmin=0, vmax=255)
            if title:
                plt.title(f"{title} - Slice {slice_idx+1}/{num_slices}")
            plt.axis('off')
            plt.tight_layout()
            
            # Save the frame
            plt.savefig(frame_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            frame_files.append(frame_file)
    
    elif volume.ndim == 3:  # [H, W, D]
        num_slices = volume.shape[2]  # Third dimension is depth
        height, width = volume.shape[0], volume.shape[1]
        
        # Use a subset of slices to make the video smaller
        slice_indices = range(0, num_slices, skip_frames)
        print(f"[DEBUG] Creating video with {len(slice_indices)} frames from {num_slices} total slices")
        
        # Create a temporary directory for frames
        temp_dir = os.path.join(output_dir, "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate frames
        frame_files = []
        for i, slice_idx in enumerate(slice_indices):
            # Extract the slice at this depth index
            slice_img = volume[:, :, slice_idx]  # Get the slice at depth index slice_idx
            
            # Normalize the slice
            normalized_slice = normalize_slice(slice_img)
            
            # Convert to uint8
            frame_img = normalized_slice.astype(np.uint8)
            
            # Save frame to temporary file
            frame_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
            
            # Create a figure with title
            plt.figure(figsize=(8, 8))
            plt.imshow(frame_img, cmap='gray', vmin=0, vmax=255)
            if title:
                plt.title(f"{title} - Slice {slice_idx+1}/{num_slices}")
            plt.axis('off')
            plt.tight_layout()
            
            # Save the frame
            plt.savefig(frame_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            frame_files.append(frame_file)
    else:
        raise ValueError(f"Unsupported volume shape: {volume.shape}")
    
    # Create video using OpenCV
    if frame_files:
        # Read the first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            print(f"Error: Could not read frame file {frame_files[0]}")
            # Clean up temporary files
            try:
                for frame_file in frame_files:
                    if os.path.exists(frame_file):
                        try:
                            os.remove(frame_file)
                        except Exception as e:
                            print(f"Warning: Could not remove temporary file {frame_file}: {e}")
            
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")
            return
            
        height, width, _ = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Add frames to video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                video_writer.write(frame)
        
        # Release video writer
        video_writer.release()
        
        print(f"Saved video to {video_path} with {len(frame_files)} frames")
    
    # Clean up temporary files
    try:
        for frame_file in frame_files:
            if os.path.exists(frame_file):
                try:
                    os.remove(frame_file)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {frame_file}: {e}")
        
        # Try to remove the temporary directory
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")
                # If directory is not empty, try to remove all files
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except Exception:
                        pass
                # Try again to remove the directory
                try:
                    os.rmdir(temp_dir)
                except Exception:
                    print(f"Warning: Could not remove temporary directory after cleanup attempt")
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def visualize_grid(volumes, titles, output_dir, filename="grid.png"):
    """Visualize multiple volumes in a grid."""
    n_samples = len(volumes)
    if n_samples == 0:
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each volume
    for i, (volume, title) in enumerate(zip(volumes, titles)):
        if i >= len(axes):
            break
        
        # Extract center slice
        if volume.ndim == 4:  # [H, W, D, C]
            center_slice_idx = volume.shape[2] // 2
            center_slice = volume[:, :, center_slice_idx, :]
            slice_img = center_slice[:, :, 0]
        elif volume.ndim == 3:  # [H, W, D]
            center_slice_idx = volume.shape[2] // 2
            slice_img = volume[:, :, center_slice_idx]
        else:
            continue
        
        # Normalize the slice
        normalized_slice = normalize_slice(slice_img)
        
        # Plot
        axes[i].imshow(normalized_slice, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the grid
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid visualization to {save_path}")
    
    # Show the grid
    plt.show()

def main():
    """Main function to visualize random samples."""
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
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
    num_samples = min(NUM_SAMPLES, len(synapse_data))
    sample_indices = np.random.choice(len(synapse_data), num_samples, replace=False)
    
    print(f"Visualizing {num_samples} random samples...")
    
    # Store volumes and titles for grid visualization
    segmented_cubes = []
    titles = []
    
    for i, idx in enumerate(sample_indices):
        syn_info = synapse_data.iloc[idx]
        bbox_name = syn_info['bbox_name']
        
        if bbox_name not in volumes:
            continue
            
        raw_vol, seg_vol, add_mask_vol = volumes[bbox_name]
        
        central_coord = (int(syn_info['central_coord_1']), 
                       int(syn_info['central_coord_2']), 
                       int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), 
                      int(syn_info['side_1_coord_2']), 
                      int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), 
                      int(syn_info['side_2_coord_2']), 
                      int(syn_info['side_2_coord_3']))
        
        # Create segmented cube
        segmented_cube = create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=SEGMENTATION_TYPE,
            subvolume_size=SUBVOL_SIZE,
            alpha=ALPHA,
            bbox_name=bbox_name
        )
        
        # Save individual visualization
        title = f"Sample {i+1}: {bbox_name}, Synapse {syn_info['Var1']}"
        filename = f"sample_{i+1}_{bbox_name}_synapse_{syn_info['Var1']}.png"
        
        visualize_center_slice(
            volume=segmented_cube,
            title=title,
            filename=filename,
            output_dir=output_dir
        )
        
        print(f"Visualized sample {i+1}")
        
        # Create video if requested
        if CREATE_VIDEOS:
            video_filename = f"sample_{i+1}_{bbox_name}_synapse_{syn_info['Var1']}.mp4"
            create_video(
                volume=segmented_cube,
                title=f"Sample {i+1}: {bbox_name}, Synapse {syn_info['Var1']}",
                filename=video_filename,
                output_dir=video_dir,
                fps=VIDEO_FPS,
                skip_frames=VIDEO_SKIP_FRAMES
            )
            print(f"Created video for sample {i+1}")
        
        # Store for grid visualization
        segmented_cubes.append(segmented_cube)
        titles.append(title)
    
    # Create a grid visualization of all samples
    if segmented_cubes:
        grid_filename = f"synapse_grid_segtype_{SEGMENTATION_TYPE}_alpha_{ALPHA:.1f}.png"
        visualize_grid(
            volumes=segmented_cubes,
            titles=titles,
            output_dir=output_dir,
            filename=grid_filename
        )

if __name__ == "__main__":
    main() 