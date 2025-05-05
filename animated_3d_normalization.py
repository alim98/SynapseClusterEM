import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path
import random
import imageio
from matplotlib.gridspec import GridSpec
import io
from PIL import Image

# Add the root directory to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the config directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'synapse')))
from synapse.utils.config import config

# Import SynapseDataLoader directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'newdl')))
from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor


def create_segmented_cube_with_z_score(data_loader, raw_vol, seg_vol, add_mask_vol, 
                                    central_coord, side1_coord, side2_coord, seg_type):
    """
    Recreate the segmented cube but use z-score normalization instead of min-max
    This simulates uncommenting the z-score normalization code in the dataloader
    """
    # Create the cube normally first
    cube = data_loader.create_segmented_cube(
        raw_vol=raw_vol,
        seg_vol=seg_vol,
        add_mask_vol=add_mask_vol,
        central_coord=central_coord,
        side1_coord=side1_coord,
        side2_coord=side2_coord,
        segmentation_type=seg_type,
        subvolume_size=80,
        alpha=config.alpha,
        normalize_across_volume=True
    )
    
    if cube is None:
        return None
    
    # Extract raw data from the RGB cube
    raw_data = cube[:, :, 0, :]  # Take first channel (R) as they're all the same in grayscale
    
    # Apply Z-score normalization
    mean = np.mean(raw_data)
    std = np.std(raw_data)
    if std > 0:
        normalized = (raw_data - mean) / std
        # Rescale to [0,1] range
        normalized = np.clip((normalized * 0.5) + 0.5, 0, 1)
    else:
        normalized = raw_data
    
    # Recreate the RGB cube
    z_score_cube = np.zeros_like(cube)
    for c in range(3):  # For all RGB channels
        z_score_cube[:, :, c, :] = normalized
    
    return z_score_cube


def create_segmented_cube_with_mean_only(data_loader, raw_vol, seg_vol, add_mask_vol, 
                                      central_coord, side1_coord, side2_coord, seg_type):
    """
    Recreate the segmented cube but use mean-only normalization
    This approach normalizes by simply dividing by the mean value
    """
    # Create the cube normally first
    cube = data_loader.create_segmented_cube(
        raw_vol=raw_vol,
        seg_vol=seg_vol,
        add_mask_vol=add_mask_vol,
        central_coord=central_coord,
        side1_coord=side1_coord,
        side2_coord=side2_coord,
        segmentation_type=seg_type,
        subvolume_size=80,
        alpha=config.alpha,
        normalize_across_volume=True
    )
    
    if cube is None:
        return None
    
    # Extract raw data from the RGB cube
    raw_data = cube[:, :, 0, :]  # Take first channel (R) as they're all the same in grayscale
    
    # Apply mean-only normalization
    mean = np.mean(raw_data)
    if mean > 0:
        # Normalize by dividing by the mean, then clip to [0,1]
        normalized = raw_data / (2 * mean)  # Divide by 2*mean to get values centered around 0.5
        normalized = np.clip(normalized, 0, 1)
    else:
        normalized = raw_data
    
    # Recreate the RGB cube
    mean_only_cube = np.zeros_like(cube)
    for c in range(3):  # For all RGB channels
        mean_only_cube[:, :, c, :] = normalized
    
    return mean_only_cube


def determine_id_column(df):
    """Determine which column contains synapse IDs"""
    possible_columns = ['Var1', 'ID', 'id', 'synapse_id', 'synapseID', 'synapse_ID']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # If none of the expected columns are found, try to find a column that might be the ID
    for col in df.columns:
        if 'id' in col.lower() or 'var' in col.lower() or 'syn' in col.lower():
            return col
    
    # Default to the first column if no suitable column is found
    if len(df.columns) > 0:
        return df.columns[0]
    
    return None


def extract_coordinates_from_row(row):
    """
    Extract coordinates from a row based on the specific column format in the Excel file
    
    Args:
        row: Pandas Series containing data
    
    Returns:
        Tuple of (central_coord, side1_coord, side2_coord) or (None, None, None) if not found
    """
    # Check for the specific format in your Excel file
    if 'central_coord_1' in row and 'central_coord_2' in row and 'central_coord_3' in row:
        central_coord = (row['central_coord_1'], row['central_coord_2'], row['central_coord_3'])
    else:
        central_coord = None
    
    if 'side_1_coord_1' in row and 'side_1_coord_2' in row and 'side_1_coord_3' in row:
        side1_coord = (row['side_1_coord_1'], row['side_1_coord_2'], row['side_1_coord_3'])
    else:
        side1_coord = None
    
    if 'side_2_coord_1' in row and 'side_2_coord_2' in row and 'side_2_coord_3' in row:
        side2_coord = (row['side_2_coord_1'], row['side_2_coord_2'], row['side_2_coord_3'])
    else:
        side2_coord = None
    
    # If we found all coordinates, return them
    if central_coord is not None and side1_coord is not None and side2_coord is not None:
        return central_coord, side1_coord, side2_coord
    
    # Try other common naming patterns as a fallback
    coord_patterns = get_coordinates(row)
    
    return coord_patterns


def get_coordinates(row):
    """
    Fallback method to extract coordinates from a row using various naming patterns
    
    Args:
        row: Pandas Series containing data
    
    Returns:
        Tuple of (central_coord, side1_coord, side2_coord) or (None, None, None) if not found
    """
    # Map possible coordinate column names
    coord_mappings = {
        'center': ['x', 'y', 'z', 'X', 'Y', 'Z', 'centerX', 'centerY', 'centerZ'],
        'side1': ['S1.x', 'S1.y', 'S1.z', 'side1X', 'side1Y', 'side1Z', 'S1X', 'S1Y', 'S1Z'],
        'side2': ['S2.x', 'S2.y', 'S2.z', 'side2X', 'side2Y', 'side2Z', 'S2X', 'S2Y', 'S2Z']
    }
    
    # Try to extract coordinates
    central_coord = extract_point_coordinates(row, coord_mappings['center'])
    side1_coord = extract_point_coordinates(row, coord_mappings['side1'])
    side2_coord = extract_point_coordinates(row, coord_mappings['side2'])
    
    # If we couldn't find the coordinates, return None
    if central_coord is None or side1_coord is None or side2_coord is None:
        return None, None, None
    
    return central_coord, side1_coord, side2_coord


def extract_point_coordinates(row, possible_columns):
    """
    Extract x, y, z coordinates from a row using possible column names
    
    Args:
        row: Pandas Series containing data
        possible_columns: List of possible column names for coordinates
        
    Returns:
        Tuple of (x, y, z) coordinates or None if not found
    """
    x, y, z = None, None, None
    
    # Try to find x coordinate
    for col in possible_columns:
        if col in row and col.lower().endswith('x'):
            x = row[col]
            break
    
    # Try to find y coordinate
    for col in possible_columns:
        if col in row and col.lower().endswith('y'):
            y = row[col]
            break
    
    # Try to find z coordinate
    for col in possible_columns:
        if col in row and col.lower().endswith('z'):
            z = row[col]
            break
    
    if x is not None and y is not None and z is not None:
        return (x, y, z)
    
    return None


def create_3d_slice_animation(
    bbox_name, 
    synapse_id=None, 
    output_dir='results/3d_normalization_gifs', 
    seg_type=10, 
    duration=0.2, 
    skip_every=1,
    create_combined_gif=True
):
    """
    Create animated GIFs showing different normalization methods on 3D synapse data.
    
    Args:
        bbox_name: Bounding box name to use
        synapse_id: Synapse ID to visualize (if None, will select a random one)
        output_dir: Directory to save results
        seg_type: Segmentation type to use
        duration: Duration of each frame in seconds
        skip_every: Only use every n-th slice to reduce size
        create_combined_gif: Whether to create a combined GIF comparing all methods
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    # Load volumes
    raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
    if raw_vol is None:
        print(f"Error: Could not load volumes for {bbox_name}")
        return
    
    # Load synapse data
    excel_path = os.path.join(config.excel_file, f"{bbox_name}.xlsx")
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found for {bbox_name}")
        return
    
    syn_df = pd.read_excel(excel_path).assign(bbox_name=bbox_name)
    
    # Debug: print available columns
    print(f"Available columns in the Excel file: {syn_df.columns.tolist()}")
    
    # Determine the ID column
    id_column = determine_id_column(syn_df)
    if id_column is None:
        print("Error: Could not determine ID column in the Excel file")
        return
    
    # Either select a specific synapse ID or choose a random one
    if synapse_id is not None:
        # Filter by the specified synapse ID
        syn_row = syn_df[syn_df[id_column] == synapse_id]
        if len(syn_row) == 0:
            print(f"Error: Synapse ID {synapse_id} not found in {bbox_name}")
            return
        syn_row = syn_row.iloc[0]
    else:
        # Select a random synapse
        syn_row = syn_df.sample(1).iloc[0]
        synapse_id = syn_row[id_column]
        print(f"Randomly selected synapse ID: {synapse_id}")
    
    # Extract coordinates
    central_coord, side1_coord, side2_coord = extract_coordinates_from_row(syn_row)
    if central_coord is None or side1_coord is None or side2_coord is None:
        print(f"Error: Could not extract coordinates for synapse ID {synapse_id}")
        return
    
    print(f"Using coordinates: central={central_coord}, side1={side1_coord}, side2={side2_coord}")
    
    # Define normalization options
    normalization_options = {
        "min_max": {
            "name": "Min-Max Normalization (Default)",
            "normalize_across_volume": True,
            # Default is min-max in the dataloader
        },
        "z_score": {
            "name": "Z-Score Normalization",
            "normalize_across_volume": True,
            "use_z_score": True
        },
        "mean_only": {
            "name": "Mean-Only Normalization",
            "normalize_across_volume": True,
            "use_mean_only": True
        }
    }
    
    # Dictionary to store cubes for each normalization option
    cubes = {}
    
    # Process with each normalization option
    for option_key, settings in normalization_options.items():
        print(f"Processing with {settings['name']}...")
        
        # Apply the appropriate normalization
        if "use_z_score" in settings and settings["use_z_score"]:
            cube = create_segmented_cube_with_z_score(
                data_loader,
                raw_vol, seg_vol, add_mask_vol,
                central_coord, side1_coord, side2_coord,
                seg_type
            )
        elif "use_mean_only" in settings and settings["use_mean_only"]:
            cube = create_segmented_cube_with_mean_only(
                data_loader,
                raw_vol, seg_vol, add_mask_vol,
                central_coord, side1_coord, side2_coord,
                seg_type
            )
        else:
            cube = data_loader.create_segmented_cube(
                raw_vol=raw_vol,
                seg_vol=seg_vol,
                add_mask_vol=add_mask_vol,
                central_coord=central_coord,
                side1_coord=side1_coord,
                side2_coord=side2_coord,
                segmentation_type=seg_type,
                subvolume_size=80,
                alpha=config.alpha,
                bbox_name=bbox_name,
                normalize_across_volume=settings.get("normalize_across_volume", True)
            )
        
        if cube is None:
            print(f"Warning: Could not create cube with {settings['name']}")
            continue
        
        # Store cube for combined GIF
        cubes[option_key] = {
            "cube": cube,
            "name": settings['name']
        }
        
        # Create animation for this normalization
        print(f"Creating animation for {option_key}...")
        
        # Determine which slices to use (skip some if needed)
        n_slices = cube.shape[3]
        slice_indices = np.arange(0, n_slices, skip_every)
        
        # Create frames for each slice
        frames = []
        
        for slice_idx in slice_indices:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(cube[:, :, :, slice_idx])
            ax.set_title(f"{settings['name']} - Slice {slice_idx}")
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save the figure to a temporary buffer using BytesIO instead
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            frame = np.array(img)
            plt.close(fig)
            buf.close()
            
            frames.append(frame)
        
        # Create forwards animation frames
        animation_frames = frames.copy()
        
        # Add backwards animation frames (in reverse order, excluding the last frame)
        # This creates a smooth loop effect
        animation_frames.extend(frames[-2::-1])
        
        # Save as GIF
        output_path = os.path.join(output_dir, f"{bbox_name}_{synapse_id}_{option_key}.gif")
        imageio.mimsave(output_path, animation_frames, duration=duration)
        print(f"Saved animation to {output_path}")
    
    # Create a combined visualization if requested
    if create_combined_gif and len(cubes) > 0:
        print("Creating combined comparison GIF...")
        
        # Determine which slices to use (skip some if needed)
        n_slices = next(iter(cubes.values()))["cube"].shape[3]
        slice_indices = np.arange(0, n_slices, skip_every)
        
        # Create frames for each slice
        combined_frames = []
        
        for slice_idx in slice_indices:
            # Create a figure with subplots for each normalization method
            n_methods = len(cubes)
            n_cols = min(3, n_methods)
            n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division
            
            fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
            plt.suptitle(f"Normalization Comparison - Slice {slice_idx}", fontsize=16)
            
            # Create a grid for plotting
            gs = GridSpec(n_rows, n_cols, figure=fig)
            
            # Plot each normalization method
            for i, (option_key, cube_data) in enumerate(cubes.items()):
                row = i // n_cols
                col = i % n_cols
                
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(cube_data["cube"][:, :, :, slice_idx])
                ax.set_title(cube_data["name"])
                ax.axis('off')
            
            plt.tight_layout()
            
            # Save the figure to a temporary buffer using BytesIO instead
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            frame = np.array(img)
            plt.close(fig)
            buf.close()
            
            combined_frames.append(frame)
        
        # Create forwards animation frames
        animation_frames = combined_frames.copy()
        
        # Add backwards animation frames (in reverse order, excluding the last frame)
        animation_frames.extend(combined_frames[-2::-1])
        
        # Save as GIF
        output_path = os.path.join(output_dir, f"{bbox_name}_{synapse_id}_comparison.gif")
        imageio.mimsave(output_path, animation_frames, duration=duration)
        print(f"Saved combined comparison animation to {output_path}")


def process_multiple_bboxes(
    bbox_names=None,
    samples_per_bbox=3,
    output_dir='results/3d_normalization_gifs',
    seg_type=10,
    duration=0.2,
    skip_every=1,
    create_combined_gif=True
):
    """
    Process multiple bounding boxes, creating animated GIFs for random samples from each
    
    Args:
        bbox_names: List of bounding box names to process (if None, will use default bounding boxes)
        samples_per_bbox: Number of random samples to process from each bounding box
        output_dir: Directory to save results
        seg_type: Segmentation type to use
        duration: Duration of each frame in seconds
        skip_every: Only use every n-th slice to reduce size
        create_combined_gif: Whether to create a combined GIF comparing all methods
    """
    # Use default bounding boxes if none provided
    if bbox_names is None:
        bbox_names = ["bbox1", "bbox2", "bbox3", "bbox4", "bbox5", "bbox6", "bbox7"]
    
    # Create the base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each bounding box
    for bbox_name in bbox_names:
        print(f"\n{'='*50}")
        print(f"Processing bounding box: {bbox_name}")
        print(f"{'='*50}\n")
        
        # Create a subdirectory for this bounding box
        bbox_output_dir = os.path.join(output_dir, bbox_name)
        os.makedirs(bbox_output_dir, exist_ok=True)
        
        try:
            # Load the excel file for this bounding box
            excel_path = os.path.join(config.excel_file, f"{bbox_name}.xlsx")
            if not os.path.exists(excel_path):
                print(f"Warning: Excel file not found for {bbox_name}, skipping")
                continue
                
            syn_df = pd.read_excel(excel_path).assign(bbox_name=bbox_name)
            
            # Find the ID column
            id_column = determine_id_column(syn_df)
            if id_column is None:
                print(f"Warning: Could not determine ID column for {bbox_name}, skipping")
                continue
            
            # Get available synapse IDs
            synapse_ids = syn_df[id_column].tolist()
            if not synapse_ids:
                print(f"Warning: No synapse IDs found for {bbox_name}, skipping")
                continue
            
            # Randomly select samples from this bounding box
            selected_ids = []
            if len(synapse_ids) <= samples_per_bbox:
                print(f"Only {len(synapse_ids)} samples available in {bbox_name}, using all")
                selected_ids = synapse_ids
            else:
                selected_ids = random.sample(synapse_ids, samples_per_bbox)
            
            print(f"Selected {len(selected_ids)} samples from {bbox_name}: {selected_ids}")
            
            # Process each selected sample
            for i, synapse_id in enumerate(selected_ids):
                print(f"\nProcessing sample {i+1}/{len(selected_ids)}: {synapse_id}")
                
                # Create visualizations for this sample
                create_3d_slice_animation(
                    bbox_name=bbox_name,
                    synapse_id=synapse_id,
                    output_dir=bbox_output_dir,
                    seg_type=seg_type,
                    duration=duration,
                    skip_every=skip_every,
                    create_combined_gif=create_combined_gif
                )
                
        except Exception as e:
            print(f"Error processing {bbox_name}: {str(e)}")
            continue
    
    print("\nAll bounding boxes processed")


def main():
    parser = argparse.ArgumentParser(description='Create animated GIFs for 3D normalization comparison')
    parser.add_argument('--bbox', default=None, help='Single bounding box name to process')
    parser.add_argument('--synapse_id', help='Specific synapse ID to visualize (only used with --bbox)')
    parser.add_argument('--output_dir', default='results/3d_normalization_gifs', 
                        help='Directory to save output visualizations')
    parser.add_argument('--seg_type', type=int, default=10, choices=range(0, 13),
                        help='Segmentation type to use (0-12)')
    parser.add_argument('--duration', type=float, default=0.2, 
                        help='Duration of each frame in seconds')
    parser.add_argument('--skip_every', type=int, default=1,
                        help='Only use every n-th slice to reduce size')
    parser.add_argument('--no_combined', action='store_true',
                        help='Disable creating a combined comparison GIF')
    parser.add_argument('--samples_per_bbox', type=int, default=3,
                        help='Number of random samples to process from each bounding box when processing all bboxes')
    parser.add_argument('--bboxes', nargs='+', help='List of bounding box names to process (default: all bboxes)')
    parser.add_argument('--all', action='store_true', help='Process all available bounding boxes')
    
    args = parser.parse_args()
    
    # Set the flag for combined GIF creation
    create_combined = not args.no_combined
    
    # If a specific bounding box and maybe a specific synapse ID is provided
    if args.bbox is not None:
        create_3d_slice_animation(
            bbox_name=args.bbox,
            synapse_id=args.synapse_id,
            output_dir=args.output_dir,
            seg_type=args.seg_type,
            duration=args.duration,
            skip_every=args.skip_every,
            create_combined_gif=create_combined
        )
    # If --all flag is set or specific bboxes are provided
    elif args.all or args.bboxes:
        process_multiple_bboxes(
            bbox_names=args.bboxes,  # If None and --all is set, will use defaults
            samples_per_bbox=args.samples_per_bbox,
            output_dir=args.output_dir,
            seg_type=args.seg_type,
            duration=args.duration,
            skip_every=args.skip_every,
            create_combined_gif=create_combined
        )
    # Default: just process bbox1
    else:
        create_3d_slice_animation(
            bbox_name="bbox1",
            synapse_id=args.synapse_id,
            output_dir=args.output_dir,
            seg_type=args.seg_type,
            duration=args.duration,
            skip_every=args.skip_every,
            create_combined_gif=create_combined
        )


if __name__ == "__main__":
    main() 