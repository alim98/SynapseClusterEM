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

from newdl.dataset2 import SynapseDataset
from newdl.dataloader2 import SynapseDataLoader, Synapse3DProcessor
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

# Predefined samples for visualization
fixed_samples = [
    {"id": 1, "bbox_name": "bbox1", "Var1": "non_spine_synapse_063", "slice_number": 25},
    # {"id": 2, "bbox_name": "bbox1", "Var1": "non_spine_synapse_050", "slice_number": 39},
    {"id": 2, "bbox_name": "bbox1", "Var1": "non_spine_synapse_062", "slice_number": 40},
    {"id": 3, "bbox_name": "bbox2", "Var1": "explorative_2024-10-17_Valentin_Pinkau_129", "slice_number": 33},
    {"id": 3, "bbox_name": "bbox2", "Var1": "explorative_2024-10-17_Valentin_Pinkau_130", "slice_number": 33},
    {"id": 3, "bbox_name": "bbox2", "Var1": "explorative_2024-10-17_Valentin_Pinkau_128", "slice_number": 33},
    {"id": 3, "bbox_name": "bbox2", "Var1": "explorative_2024-10-17_Valentin_Pinkau_131", "slice_number": 33},
    # {"id": 4, "bbox_name": "bbox2", "Var1": "explorative_2024-08-28_Cora_Wolter_031", "slice_number": 43},
    # # bbox3__5_193
    {"id": 5, "bbox_name": "bbox3", "Var1": "non_spine_synapse_041", "slice_number": 33},
    # # {"id": 5, "bbox_name": "bbox3", "Var1": "non_spine_synapse_035", "slice_number": 35},
    # {"id": 5, "bbox_name": "bbox3", "Var1": "non_spine_synapse_036", "slice_number": 41},
    # {"id": 6, "bbox_name": "bbox3", "Var1": "non_spine_synapse_018", "slice_number": 41},
    # {"id": 7, "bbox_name": "bbox4", "Var1": "explorative_2024-08-03_Ali_Karimi_022_5_238", "slice_number": 56},
    # {"id": 8, "bbox_name": "bbox5", "Var1": "non_spine_synapse_033", "slice_number": 48},
    # {"id": 9, "bbox_name": "bbox5", "Var1": "non_spine_synapse_045", "slice_number": 40},
    # {"id": 10, "bbox_name": "bbox6", "Var1": "spine_synapse_070", "slice_number": 37},
    # {"id": 11, "bbox_name": "bbox6", "Var1": "spine_synapse_021", "slice_number": 30},
    # {"id": 12, "bbox_name": "bbox7", "Var1": "non_spine_synapse_013", "slice_number": 25},
]

def visualize_specific_sample(dataset, syn_df, bbox_name, var1, save_gifs_dir, segmentation_type):
    """
    Visualize a specific sample identified by bbox_name and var1
    """
    logger.info(f"Starting visualization for {var1} from {bbox_name}")
    
    # Filter data based on var1 and bbox_name values
    specific_sample = syn_df[(syn_df['Var1'] == var1) & (syn_df['bbox_name'] == bbox_name)]
    logger.info(f"Found {len(specific_sample)} matching samples in syn_df")

    # Check if a sample was found
    if not specific_sample.empty:
        found = False
        for idx, (pixel_values, syn_info, sample_bbox_name) in enumerate(dataset):
            if sample_bbox_name == bbox_name and var1 == syn_info['Var1']:
                logger.info(f"Found matching sample at index {idx}")
                logger.info(f"Pixel values shape: {pixel_values.shape}")
                logger.info(f"Pixel values min: {pixel_values.min()}, max: {pixel_values.max()}, mean: {pixel_values.mean()}")
                
                # Debug: Show information about the frames
                if torch.isnan(pixel_values).any():
                    logger.warning("NaN values detected in pixel_values")
                if torch.all(pixel_values == 0):
                    logger.warning("All pixel values are zero!")

                # Denormalize the cube values
                logger.info("Denormalizing the cube values")
                denormalized_cube = pixel_values * torch.tensor([0.229]) + torch.tensor([0.485])
                logger.info(f"After denormalization - min: {denormalized_cube.min()}, max: {denormalized_cube.max()}, mean: {denormalized_cube.mean()}")
                
                denormalized_cube = torch.clamp(denormalized_cube, 0, 1)
                logger.info(f"After clamping - min: {denormalized_cube.min()}, max: {denormalized_cube.max()}, mean: {denormalized_cube.mean()}")
                
                frames = denormalized_cube.squeeze(1).numpy()
                logger.info(f"Frames shape after squeezing: {frames.shape}")
                logger.info(f"Frames min: {frames.min()}, max: {frames.max()}, mean: {frames.mean()}")
                
                # Save a sample frame as an image to check
                sample_frame_idx = frames.shape[0] // 2  # Middle frame
                sample_frame = frames[sample_frame_idx]
                logger.info(f"Sample frame shape: {sample_frame.shape}, min: {sample_frame.min()}, max: {sample_frame.max()}")
                
                # Create the output directory if it doesn't exist
                os.makedirs(save_gifs_dir, exist_ok=True)
                
                # Apply min-max normalization to each frame to enhance contrast
                enhanced_frames = []
                for frame in frames:
                    # Min-max normalization to stretch contrast
                    frame_min, frame_max = frame.min(), frame.max()
                    if frame_max > frame_min:  # Avoid division by zero
                        normalized = (frame - frame_min) / (frame_max - frame_min)
                    else:
                        normalized = frame
                    enhanced_frames.append((normalized * 255).astype(np.uint8))
                
                # Save as GIF
                Gif_Name = f"{bbox_name}_{var1}_{segmentation_type}_{idx}"
                output_gif_path = os.path.join(save_gifs_dir, f"{Gif_Name}.gif")
                
                try:
                    logger.info(f"Saving GIF with {len(enhanced_frames)} frames to {output_gif_path}")
                    imageio.mimsave(output_gif_path, enhanced_frames, fps=10)
                    logger.info(f"GIF saved successfully at {output_gif_path}")
                except Exception as e:
                    logger.error(f"Failed to save GIF: {e}")

                found = True
                break

        if not found:
            logger.warning(f"No matching sample found in dataset for Var1={var1} and bbox_name={bbox_name}")
    else:
        logger.warning(f"No sample found in syn_df with Var1={var1} and bbox_name={bbox_name}")

def visualize_all_samples_from_bboxes(dataset, syn_df, bbox_names, save_gifs_dir, segmentation_type, limit=None):
    """
    Visualize all samples from the given bboxes
    
    Args:
        dataset: SynapseDataset instance
        syn_df: DataFrame with synapse data
        bbox_names: List of bbox names to visualize
        save_gifs_dir: Directory to save GIF files
        segmentation_type: Type of segmentation overlay
        limit: Maximum number of samples to visualize per bbox (None for all)
    """
    os.makedirs(save_gifs_dir, exist_ok=True)
    
    for bbox in bbox_names:
        print(f"Processing samples from {bbox}...")
        bbox_samples = syn_df[syn_df['bbox_name'] == bbox]
        
        if bbox_samples.empty:
            print(f"No samples found for {bbox}")
            continue
            
        if limit:
            bbox_samples = bbox_samples.head(limit)
            
        for idx, sample in bbox_samples.iterrows():
            var1 = sample['Var1']
            print(f"  Visualizing {var1} from {bbox}...")
            
            visualize_specific_sample(
                dataset=dataset,
                syn_df=syn_df,
                bbox_name=bbox,
                var1=var1,
                save_gifs_dir=save_gifs_dir,
                segmentation_type=segmentation_type
            )

def main():
    # Initialize and parse configuration
    config.parse_args()
    
    # Override config with values set at the top
    config.bbox_name = BBOX_NAMES
    config.segmentation_type = config.segmentation_type
    
    logger.info(f"Starting with bbox_names: {config.bbox_name}")
    logger.info(f"Raw data directory: {config.raw_base_dir}")
    logger.info(f"Segmentation directory: {config.seg_base_dir}")
    logger.info(f"Additional mask directory: {config.add_mask_base_dir}")
    logger.info(f"Excel file directory: {config.excel_file}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(config.save_gifs_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    # Load volumes
    vol_data_dict = {}
    for bbox in config.bbox_name:
        logger.info(f"Loading volumes for {bbox}...")
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox)
        if raw_vol is not None:
            logger.info(f"Successfully loaded volumes for {bbox}")
            logger.info(f"Raw volume shape: {raw_vol.shape}, min: {raw_vol.min()}, max: {raw_vol.max()}")
            logger.info(f"Seg volume shape: {seg_vol.shape}, min: {seg_vol.min()}, max: {seg_vol.max()}")
            if add_mask_vol is not None:
                logger.info(f"Add mask volume shape: {add_mask_vol.shape}, min: {add_mask_vol.min()}, max: {add_mask_vol.max()}")
            vol_data_dict[bbox] = (raw_vol, seg_vol, add_mask_vol)
        else:
            logger.warning(f"Could not load volumes for {bbox}")

    # Load synapse data
    if config.excel_file:
        try:
            logger.info(f"Loading Excel files from {config.excel_file}")
            excel_files = [f"{bbox}.xlsx" for bbox in config.bbox_name]
            available_excel_files = [f for f in excel_files if os.path.exists(os.path.join(config.excel_file, f))]
            logger.info(f"Available Excel files: {available_excel_files}")
            
            syn_df = pd.concat([
                pd.read_excel(os.path.join(config.excel_file, f"{bbox}.xlsx")).assign(bbox_name=bbox)
                for bbox in config.bbox_name if os.path.exists(os.path.join(config.excel_file, f"{bbox}.xlsx"))
            ])
            logger.info(f"Loaded synapse data: {len(syn_df)} rows")
            if not syn_df.empty:
                logger.info(f"Sample columns: {syn_df.columns.tolist()}")
                logger.info(f"Sample data:\n{syn_df.head()}")
            
            if syn_df.empty:
                logger.warning(f"No data found in Excel files for the specified bboxes: {config.bbox_name}")
        except Exception as e:
            logger.error(f"Error loading Excel files: {e}")
            syn_df = pd.DataFrame()
    else:
        logger.warning("No excel_file specified. Using empty DataFrame.")
        syn_df = pd.DataFrame()

    # Initialize processor
    processor = Synapse3DProcessor(size=config.size)
    logger.info(f"Initialized processor with size {config.size}")

    # Create dataset
    logger.info("Creating SynapseDataset...")
    dataset = SynapseDataset(
        vol_data_dict=vol_data_dict,
        synapse_df=syn_df,
        processor=processor,
        segmentation_type=config.segmentation_type,
        subvol_size=config.subvol_size,
        num_frames=config.num_frames,
        alpha=config.alpha
    )
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Option 1: Visualize a single specific sample
    var1 = 'non_spine_synapse_018'
    bboxnumber = 'bbox3'
    logger.info(f"Visualizing single sample: {var1} from {bboxnumber}...")
    visualize_specific_sample(
        dataset=dataset,
        syn_df=syn_df,
        bbox_name=bboxnumber,
        var1=var1,
        save_gifs_dir=config.save_gifs_dir,
        segmentation_type=config.segmentation_type
    )
    
    # Option 2: Visualize all fixed samples
    logger.info("Visualizing all fixed samples...")
    for sample in fixed_samples:
        if sample['bbox_name'] in vol_data_dict:  # Only process if volumes are loaded
            logger.info(f"Processing {sample['Var1']} from {sample['bbox_name']}...")
            visualize_specific_sample(
                dataset=dataset,
                syn_df=syn_df,
                bbox_name=sample['bbox_name'],
                var1=sample['Var1'],
                save_gifs_dir=config.save_gifs_dir,
                segmentation_type=config.segmentation_type
            )
        else:
            logger.warning(f"Skipping {sample['Var1']} from {sample['bbox_name']} - volumes not loaded")
    
    logger.info("Visualization complete")

def visualize_with_consistent_gray():
    """
    Demonstration function showing how to visualize slices with consistent gray levels.
    This prevents matplotlib's automatic normalization from changing the appearance
    of the gray overlays.
    """
    # Initialize and parse configuration
    config.parse_args()
    
    # Initialize data loader
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir,
        gray_color=config.gray_color  # Explicitly set the gray color
    )
    
    # Pick a sample bbox to visualize
    bbox_name = "bbox1"
    
    # Load volumes
    raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
    if raw_vol is None:
        logger.error(f"Could not load volumes for {bbox_name}")
        return
    
    # Pick a central point for visualization
    center_z = raw_vol.shape[0] // 2
    center_y = raw_vol.shape[1] // 2
    center_x = raw_vol.shape[2] // 2
    
    central_coord = (center_x, center_y, center_z)
    
    # Define arbitrary side points for this example
    side1_coord = (center_x - 10, center_y - 10, center_z)
    side2_coord = (center_x + 10, center_y + 10, center_z)
    
    # Create output directory
    output_dir = os.path.join(config.save_gifs_dir, "consistent_gray_example")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comparison images with different segmentation types
    for seg_type in [0, 1, 3, 5, 10]:
        # Create segmented cube
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
            bbox_name=bbox_name
        )
        
        # Save with consistent gray (fixed vmin/vmax)
        consistent_path = os.path.join(output_dir, f"seg{seg_type}_consistent.png")
        data_loader.save_segmented_slice(
            cube, 
            consistent_path,
            consistent_gray=True
        )
        
        # For comparison, save with matplotlib's auto-scaling
        auto_scale_path = os.path.join(output_dir, f"seg{seg_type}_auto.png")
        data_loader.save_segmented_slice(
            cube, 
            auto_scale_path,
            consistent_gray=False
        )
        
        logger.info(f"Saved segmentation type {seg_type} comparisons")
    
    logger.info(f"All comparison images saved to {output_dir}")

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
                try:
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
                except Exception as e:
                    logger.error(f"Error extracting coordinates: {e}. Using default coordinates.")
                    # Fallback to center of volume if coordinates extraction fails
                    center_z = raw_vol.shape[0] // 2
                    center_y = raw_vol.shape[1] // 2
                    center_x = raw_vol.shape[2] // 2
                    central_coord = (center_x, center_y, center_z)
                    side1_coord = (center_x - 10, center_y - 10, center_z)
                    side2_coord = (center_x + 10, center_y + 10, center_z)
        else:
            logger.warning(f"No Excel file found for {bbox_name}. Using central samples.")
            # Without Excel data, just pick points in the center and a few random locations
            logger.info(f"Creating {num_samples_per_bbox} central samples for {bbox_name}")
            
            # Initialize empty list to store sample data
            random_samples = []
            
            # Generate random coordinates
            for i in range(num_samples_per_bbox):
                # For the center sample
                if i == 0:
                    center_z = raw_vol.shape[0] // 2
                    center_y = raw_vol.shape[1] // 2
                    center_x = raw_vol.shape[2] // 2
                    sample_name = f"center_sample"
                else:
                    # For random samples, add some random offset from center
                    # Stay within safe boundaries (10% to 90% of volume dimensions)
                    z_margin = int(raw_vol.shape[0] * 0.4)
                    y_margin = int(raw_vol.shape[1] * 0.4)
                    x_margin = int(raw_vol.shape[2] * 0.4)
                    
                    center_z = raw_vol.shape[0] // 2 + np.random.randint(-z_margin, z_margin)
                    center_y = raw_vol.shape[1] // 2 + np.random.randint(-y_margin, y_margin)
                    center_x = raw_vol.shape[2] // 2 + np.random.randint(-x_margin, x_margin)
                    sample_name = f"random_sample_{i}"
                
                # Create sample data
                central_coord = (center_x, center_y, center_z)
                side1_coord = (center_x - 10, center_y - 10, center_z)
                side2_coord = (center_x + 10, center_y + 10, center_z)
                
                # Store sample data
                random_samples.append({
                    "central_coord": central_coord,
                    "side1_coord": side1_coord,
                    "side2_coord": side2_coord,
                    "var1": sample_name
                })
            
            # Process each random sample
            for sample in random_samples:
                central_coord = sample["central_coord"]
                side1_coord = sample["side1_coord"]
                side2_coord = sample["side2_coord"]
                var1 = sample["var1"]
                
                logger.info(f"Processing random sample: {var1} in {bbox_name}")
                
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
                        12: "Type 12: Vesicle Cloud (25×25×25)",
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
            
            continue  # Skip to next bbox as we've manually created samples

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
                    12: "Type 12: Vesicle Cloud (25×25×25)",
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
    # Run the existing main function or the new comparison function
    # Uncomment one of these:
    # main()
    # visualize_with_consistent_gray()
    compare_segmentation_types(num_samples_per_bbox=2, segmentation_types=[ 13])
