import os
import numpy as np
import pandas as pd
import torch
import imageio
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Import from the reorganized modules
from synapse import (
    SynapseDataLoader, 
    Synapse3DProcessor, 
    SynapseDataset, 
    config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sample_fig_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sample_fig")

def create_gif_from_volume(volume, output_path, fps=10, loop=0):
    """
    Create a GIF from a 3D volume by stacking slices along the z-axis.
    
    Args:
        volume (numpy.ndarray): 3D volume with shape (z, y, x)
        output_path (str): Path to save the GIF
        fps (int): Frames per second
        loop (int): Number of loops (0 for infinite)
    """
    
    # Convert PyTorch tensor to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().detach().numpy()
    
    # If volume has more than 3 dimensions, squeeze it
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    
    # Apply global normalization to ensure consistent grayscale values
    normalized_volume = volume
    
    # Scale to 8-bit for GIF
    volume_8bit = (normalized_volume * 255).astype(np.uint8)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GIF
    with imageio.get_writer(output_path, mode='I', fps=fps, loop=loop) as writer:
        for i in range(volume_8bit.shape[0]):
            writer.append_data(volume_8bit[i])
    
    logger.info(f"GIF saved to {output_path}")

def save_center_slice_image(volume, output_path, consistent_gray=True):
    """
    Save a center slice of a 3D volume as an image with consistent gray values.
    
    Args:
        volume (numpy.ndarray): 3D volume with shape (z, y, x) or (z, c, y, x)
        output_path (str): Path to save the image
        consistent_gray (bool): Whether to enforce consistent gray levels
    """
    # Get center slice
    if len(volume.shape) == 4:  # (z, c, y, x)
        center_slice = volume[volume.shape[0] // 2, 0]
    else:  # (z, y, x)
        center_slice = volume[volume.shape[0] // 2]
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create the figure with controlled normalization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use fixed vmin and vmax to prevent matplotlib's auto-scaling
    if consistent_gray:
        ax.imshow(center_slice, cmap='gray', vmin=0, vmax=1)
    else:
        # For comparison, you can also save with matplotlib's auto-scaling
        ax.imshow(center_slice, cmap='gray')
    
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    
    logger.info(f"Center slice image saved to {output_path}")

def visualize_specific_sample(dataset, syn_df, bbox_name, var1, save_gifs_dir, segmentation_type):
    """
    Visualize a specific sample from the dataset.
    
    Args:
        dataset (SynapseDataset): Dataset containing the samples
        syn_df (pandas.DataFrame): DataFrame with synapse information
        bbox_name (str): Name of the bounding box
        var1 (int): Index of the sample in the bounding box
        save_gifs_dir (str): Directory to save the GIFs
        segmentation_type (int): Type of segmentation to use
    """
    # Filter the dataframe for the specific bbox_name
    bbox_df = syn_df[syn_df['bbox_name'] == bbox_name]
    
    if var1 >= len(bbox_df):
        logger.error(f"Sample index {var1} out of range for bbox {bbox_name} (max: {len(bbox_df)-1})")
        return
    
    # Get the sample
    sample_idx = bbox_df.index[var1]
    pixel_values, syn_info, _ = dataset[sample_idx]
    
    # Convert to numpy array
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.squeeze().numpy()
    else:
        pixel_values = pixel_values.squeeze()
    
    # Create output directory
    os.makedirs(save_gifs_dir, exist_ok=True)
    
    # Save as GIF
    output_path = os.path.join(
        save_gifs_dir, 
        f"{bbox_name}_sample{var1}_seg{segmentation_type}.gif"
    )
    create_gif_from_volume(pixel_values, output_path)
    
    # Also save the center slice as an image with consistent gray levels
    slice_output_path = os.path.join(
        save_gifs_dir,
        f"{bbox_name}_sample{var1}_seg{segmentation_type}_center_slice.png" 
    )
    save_center_slice_image(pixel_values, slice_output_path)
    
    logger.info(f"Visualized sample {var1} from {bbox_name} with segmentation type {segmentation_type}")
    
    return output_path

def visualize_all_samples_from_bboxes(dataset, syn_df, bbox_names, save_gifs_dir, segmentation_type, limit=None):
    """
    Visualize all samples from specified bounding boxes.
    
    Args:
        dataset (SynapseDataset): Dataset containing the samples
        syn_df (pandas.DataFrame): DataFrame with synapse information
        bbox_names (list): List of bounding box names
        save_gifs_dir (str): Directory to save the GIFs
        segmentation_type (int): Type of segmentation to use
        limit (int, optional): Maximum number of samples per bounding box
    """
    all_paths = []
    
    for bbox_name in bbox_names:
        # Filter the dataframe for the specific bbox_name
        bbox_df = syn_df[syn_df['bbox_name'] == bbox_name]
        
        # Determine how many samples to visualize
        num_samples = len(bbox_df)
        if limit is not None and limit < num_samples:
            num_samples = limit
        
        logger.info(f"Visualizing {num_samples} samples from {bbox_name}")
        
        # Visualize each sample
        for i in range(num_samples):
            output_path = visualize_specific_sample(
                dataset, syn_df, bbox_name, i, save_gifs_dir, segmentation_type
            )
            if output_path:
                all_paths.append(output_path)
    
    return all_paths

def main():
    """Main function to run the visualization."""
    # Initialize and parse configuration
    config.parse_args()
    
    # Define parameters
    bbox_names = config.bbox_name
    segmentation_type = config.segmentation_type
    alpha = config.alpha
    
    # Create output directory
    save_gifs_dir = os.path.join(config.output_dir, "gifs")
    os.makedirs(save_gifs_dir, exist_ok=True)
    
    logger.info(f"Processing bounding boxes: {bbox_names}")
    logger.info(f"Using segmentation type: {segmentation_type}")
    logger.info(f"Alpha value: {alpha}")
    
    # Initialize data loader
    data_loader = SynapseDataLoader(
        config.raw_base_dir,
        config.seg_base_dir,
        config.add_mask_base_dir
    )
    
    # Load synapse information
    syn_df = pd.read_excel(os.path.join(config.excel_dir, "synapse_info.xlsx"))
    
    # Initialize processor
    processor = Synapse3DProcessor()
    
    # Create dataset
    dataset = SynapseDataset(
        {bbox: data_loader.load_volumes(bbox) for bbox in bbox_names},
        syn_df,
        processor,
        segmentation_type,
        alpha=alpha
    )
    
    # Visualize all samples
    visualize_all_samples_from_bboxes(
        dataset,
        syn_df,
        bbox_names,
        save_gifs_dir,
        segmentation_type,
        limit=5  # Limit to 5 samples per bbox for testing
    )
    
    logger.info("Visualization complete!")

if __name__ == "__main__":
    main() 