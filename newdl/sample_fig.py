import os
import numpy as np
import pandas as pd
import torch
import imageio
import logging
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from newdl.dataloader3 import SynapseDataLoader
from synapse.utils.config import config

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


# Configuration - Global variables
USE_ALL_BBOXES = False  # Set to True to use all bboxes
BBOX_NAMES = ['bbox3'] if not USE_ALL_BBOXES else ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']

def compare_segmentation_types(num_samples_per_bbox=2, segmentation_types=[10, 11, 12]):
    """
    Visualize and compare segmentation types in a single image as GIFs
    using random samples from each bounding box in the config.
    
    Args:
        num_samples_per_bbox: Number of random samples to select from each bounding box
        segmentation_types: List of segmentation types to compare
    """
    # Initialize and parse configuration
    config.parse_args()
    
    # Create output directory for comparison images
    output_dir = os.path.join(config.save_gifs_dir, "segmentation_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting segmentation type comparison with random samples...")
    
    # Initialize data loader
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir,
        gray_color=config.gray_color
    )
    
    # Process each bounding box
    for bbox_name in config.bbox_name:
        logger.info(f"Processing bounding box: {bbox_name}")
        
        # Load volumes
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
        if raw_vol is None:
            logger.error(f"Could not load volumes for {bbox_name}")
            continue
        
        # Load synapse data for coordinates
        excel_path = os.path.join(config.excel_file, f"{bbox_name}.xlsx")
        if os.path.exists(excel_path):
            logger.info(f"Loading Excel data for {bbox_name}")
            syn_df = pd.read_excel(excel_path)
            
            # Get random samples
            if len(syn_df) <= num_samples_per_bbox:
                selected_samples = syn_df  # Use all samples if fewer than requested
                logger.info(f"Using all {len(syn_df)} samples from {bbox_name} (fewer than requested {num_samples_per_bbox})")
            else:
                selected_samples = syn_df.sample(n=num_samples_per_bbox, random_state=42)
                logger.info(f"Selected {num_samples_per_bbox} random samples from {bbox_name}")
            
            # Process each random sample
            for idx, sample_row in selected_samples.iterrows():
                var1 = sample_row['Var1']
                logger.info(f"Processing sample: {var1} from {bbox_name}")
                
                # Extract coordinates
                    # Check for different column naming conventions
                logger.info(f"Available columns: {list(sample_row.index)}")
            
                x_col, y_col, z_col = 'central_coord_1', 'central_coord_2', 'central_coord_3'
                x1_col, y1_col, z1_col = 'side_1_coord_1', 'side_1_coord_2', 'side_1_coord_3'
                x2_col, y2_col, z2_col = 'side_2_coord_1', 'side_2_coord_2', 'side_2_coord_3'
                
                central_coord = (
                    int(sample_row[x_col]), 
                    int(sample_row[y_col]), 
                    int(sample_row[z_col])
                )
                side1_coord = (
                    int(sample_row[x1_col]), 
                    int(sample_row[y1_col]), 
                    int(sample_row[z1_col])
                )
                side2_coord = (
                    int(sample_row[x2_col]), 
                    int(sample_row[y2_col]), 
                    int(sample_row[z2_col])
                )
                logger.info(f"Coordinates extracted for {var1}: central={central_coord}, side1={side1_coord}, side2={side2_coord}")

        # If we're here, we have actual synapse data to use
        for _, sample_row in selected_samples.iterrows():
            var1 = sample_row['Var1']
            
            # Create segmented cubes for each segmentation type
            cubes = {}
            
            for seg_type in segmentation_types:
                logger.info(f"Creating cube for segmentation type {seg_type}")
                cube = data_loader.create_segmented_cube(
                    raw_vol=raw_vol,
                    seg_vol=seg_vol,
                    add_mask_vol=add_mask_vol,
                    central_coord=central_coord,
                    side1_coord=side1_coord,
                    side2_coord=side2_coord,
                    segmentation_type=seg_type,
                    subvolume_size=config.subvol_size,
                    alpha=config.alpha,
                    bbox_name=bbox_name,
                    normalize_across_volume=True,

                )
                cubes[seg_type] = cube
            
            # Check if any segmentation types returned None (indicating they should be skipped)
            valid_seg_types = [seg_type for seg_type in segmentation_types if cubes[seg_type] is not None]
            
            if len(valid_seg_types) == 0:
                logger.warning(f"All segmentation types returned None for {bbox_name} - {var1}. Skipping sample.")
                continue
            
            # Create GIF frames with only valid cubes
            valid_cubes_shapes = [cubes[seg_type].shape[3] for seg_type in valid_seg_types]
            if not valid_cubes_shapes:
                logger.warning(f"No valid cubes available for {bbox_name} - {var1}. Skipping sample.")
                continue
                
            num_frames = min(valid_cubes_shapes)
            frames = []
            
            for frame_idx in range(num_frames):
                # Determine how many columns we need based on valid segmentation types
                num_columns = len(valid_seg_types)
                
                # Create figure with 1 row and columns for the valid segmentation types
                fig = plt.figure(figsize=(6.7 * num_columns, 6))
                gs = GridSpec(1, num_columns, figure=fig)
                
                # Define titles for each segmentation type
                titles = {
                    10: "Type 12: Vesicle Cloud (25×25×25)",
                }
                
                # Plot each valid segmentation type
                for i, seg_type in enumerate(valid_seg_types):
                    ax = fig.add_subplot(gs[0, i])
                    slice_data = cubes[seg_type][:, :, :, frame_idx]
                    ax.imshow(slice_data, vmin=0, vmax=1)  # Use consistent gray levels
                    ax.set_title(titles.get(seg_type, f"Type {seg_type}"))
                    ax.axis('off')
                
                # Add sample information as a suptitle
                plt.suptitle(f"Sample: {bbox_name} - {var1}", fontsize=14)
                plt.tight_layout()
                
                # Save the figure to a BytesIO object
                from io import BytesIO
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close(fig)
                
                # Convert BytesIO to numpy array
                buf.seek(0)
                import imageio.v3 as iio
                img = iio.imread(buf)
                frames.append(img)
            
            # Save as GIF
            sample_id = var1.replace(" ", "_").replace("/", "_")
            gif_path = os.path.join(output_dir, f"{bbox_name}_{sample_id}_comparison.gif")
            logger.info(f"Saving comparison GIF with {len(frames)} frames to {gif_path}")
            try:
                imageio.mimsave(gif_path, frames, fps=5)
                logger.info(f"GIF saved successfully at {gif_path}")
            except Exception as e:
                logger.error(f"Failed to save GIF: {e}")
    
    logger.info("Segmentation type comparison with random samples completed")

if __name__ == "__main__":

    compare_segmentation_types(num_samples_per_bbox=2, segmentation_types=[ 13])
