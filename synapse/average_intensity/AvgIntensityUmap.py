import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import random
import imageio
import umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageSequence, ImageDraw, ImageFont
import io
import seaborn as sns
import base64  # For image encoding
import json

# Add the parent directory to the path so we can import the synapse module
# This needs to be before any synapse imports
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)

# Now we can import from synapse
from synapse import config
from synapse.clustering import load_and_cluster_features, apply_tsne, save_tsne_plots, find_random_samples_in_clusters, save_cluster_samples

def create_average_intensity_projection(volume):
    """
    Create average intensity projections of a 3D volume along each axis.
    Crops the volume to 25x25x25 centered on the middle of the volume.
    
    Args:
        volume: 3D array representing volume data (z, y, x)
        
    Returns:
        tuple: (z_projection, y_projection, x_projection) where each is a 2D array
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().detach().numpy()
    
    # Ensure volume is a numpy array
    if not isinstance(volume, np.ndarray):
        raise TypeError(f"Volume must be a numpy array or PyTorch tensor, got {type(volume)}")
    
    # If volume has more than 3 dimensions, squeeze it
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    
    # Crop the volume to 25x25x25 centered on the middle
    z, y, x = volume.shape
    z_center, y_center, x_center = z // 2, y // 2, x // 2
    
    # Calculate crop boundaries ensuring they're within volume bounds
    z_min = max(0, z_center - 12)
    z_max = min(z, z_center + 13)
    y_min = max(0, y_center - 12)
    y_max = min(y, y_center + 13)
    x_min = max(0, x_center - 12)
    x_max = min(x, x_center + 13)
    
    # Crop the volume
    cropped_volume = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    print(f"Cropped volume from {volume.shape} to {cropped_volume.shape}")
    
    # If values are in 0-1 range, scale to 0-255 for processing
    if cropped_volume.max() <= 1.0:
        cropped_volume = cropped_volume * 255.0
    
    # Clip values to 0-255 range
    cropped_volume = np.clip(cropped_volume, 0.0, 255.0)
    
    # Create projections along each axis
    z_projection = np.mean(cropped_volume, axis=0).astype(np.uint8)  # Average along z-axis (x-y plane)
    y_projection = np.mean(cropped_volume, axis=1).astype(np.uint8)  # Average along y-axis (x-z plane)
    x_projection = np.mean(cropped_volume, axis=2).astype(np.uint8)  # Average along x-axis (y-z plane)
    
    return z_projection, y_projection, x_projection

def create_gif_from_volume(volume, output_path, fps=10, segmentation_type=None):
    """
    Create average intensity projections from a volume and return both individual and composite projections.
    The volume is cropped to 25x25x25 centered on the middle of the volume.
    
    Args:
        volume: 3D array representing volume data
        output_path: Path to save the projection image
        fps: Frames per second (kept for backward compatibility)
        segmentation_type: Type of segmentation used - if type 13, only show center 25 frames
        
    Returns:
        Tuple of (output_path, projection_data) where projection_data contains base64 encoded projections
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().detach().numpy()
    
    # Ensure volume is a numpy array
    if not isinstance(volume, np.ndarray):
        raise TypeError(f"Volume must be a numpy array or PyTorch tensor, got {type(volume)}")
    
    # If volume has more than 3 dimensions, squeeze it
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    
    # For segmentation type 13, only show the center 25 frames (27-53)
    # This is now handled by the cropping in create_average_intensity_projection
    # but we'll keep this for backward compatibility
    if segmentation_type == 13 and volume.shape[0] >= 54:
        print(f"Segmentation type 13 detected: Using only center frames 27-53 (25 frames)")
        volume = volume[27:54]  # Python indexing is 0-based, so 27-53 is 27:54
    
    # Create average intensity projections (function now handles cropping)
    z_proj, y_proj, x_proj = create_average_intensity_projection(volume)
    
    # Create a composite image with all three projections
    # Use a layout with z_proj at top left, y_proj at top right, x_proj at bottom left
    padding = 5  # pixels of padding between projections
    max_height = max(z_proj.shape[0], x_proj.shape[0])
    max_width = max(z_proj.shape[1], y_proj.shape[1])
    
    # Create a blank composite image
    composite_height = max_height * 2 + padding 
    composite_width = max_width * 2 + padding
    composite = np.zeros((composite_height, composite_width), dtype=np.uint8)
    
    # Add z projection (top left)
    z_h, z_w = z_proj.shape
    composite[:z_h, :z_w] = z_proj
    
    # Add y projection (top right)
    y_h, y_w = y_proj.shape
    composite[:y_h, z_w + padding:z_w + padding + y_w] = y_proj
    
    # Add x projection (bottom left)
    x_h, x_w = x_proj.shape
    composite[z_h + padding:z_h + padding + x_h, :x_w] = x_proj
    
    # Add labels to the projections
    composite_img = Image.fromarray(composite)
    draw = ImageDraw.Draw(composite_img)
    
    # Try to use a default font
    try:
        # Try to get a default font
        font = ImageFont.load_default()
        
        # Add labels
        draw.text((10, 10), "Z projection (top view)", fill=255, font=font)
        draw.text((z_w + padding + 10, 10), "Y projection (side view)", fill=255, font=font)
        draw.text((10, z_h + padding + 10), "X projection (front view)", fill=255, font=font)
    except Exception as e:
        print(f"Warning: Could not add text labels to projection image: {e}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Change the extension from .gif to .png
    output_path = output_path.replace('.gif', '.png')
    
    # Save the composite image
    composite_img.save(output_path)
    
    # Convert projections to base64-encoded PNGs for web display
    projection_data = {}
    
    # Convert z projection to base64
    with io.BytesIO() as output:
        Image.fromarray(z_proj).save(output, format="PNG")
        projection_data['z_proj'] = base64.b64encode(output.getvalue()).decode('utf-8')
    
    # Convert y projection to base64
    with io.BytesIO() as output:
        Image.fromarray(y_proj).save(output, format="PNG")
        projection_data['y_proj'] = base64.b64encode(output.getvalue()).decode('utf-8')
    
    # Convert x projection to base64
    with io.BytesIO() as output:
        Image.fromarray(x_proj).save(output, format="PNG")
        projection_data['x_proj'] = base64.b64encode(output.getvalue()).decode('utf-8')
    
    # Convert composite to base64
    with io.BytesIO() as output:
        composite_img.save(output, format="PNG")
        projection_data['composite'] = base64.b64encode(output.getvalue()).decode('utf-8')
    
    return output_path, projection_data

def create_projection_visualization(features_df, projection_paths, output_dir, dim_reduction='umap', projection_data=None, dataset=None):
    """
    Create a simple HTML page that displays intensity projections directly at their coordinates.
    The projections are embedded directly in the HTML as base64 data to avoid file:// protocol issues.
    Projections are made draggable so users can rearrange them.
    Includes a control to adjust how many projections are displayed at runtime.
    
    Args:
        features_df: DataFrame with features and coordinates
        projection_paths: Dictionary mapping sample indices to projection image paths
        output_dir: Directory to save the HTML file
        dim_reduction: Dimensionality reduction method ('umap' or 'tsne')
        projection_data: Dictionary mapping sample indices to projection data (z_proj, y_proj, x_proj)
        dataset: The dataset object to use for regenerating projections if needed
    
    Returns:
        Path to the HTML file
    """
    method_name = "UMAP" if dim_reduction == 'umap' else "t-SNE"
    import base64
    
    # Debug projection_data
    if projection_data:
        print(f"\nReceived projection_data with {len(projection_data)} entries")
        sample_key = next(iter(projection_data))
        print(f"Sample key: {sample_key}")
        print(f"Projection keys for this sample: {list(projection_data[sample_key].keys())}")
        print(f"Z projection available: {'z_proj' in projection_data[sample_key]}")
        print(f"Y projection available: {'y_proj' in projection_data[sample_key]}")
        print(f"X projection available: {'x_proj' in projection_data[sample_key]}")
    else:
        print("\nNo projection_data provided")
    
    # Define plot dimensions upfront
    plot_width = 1600  # From the CSS .plot-container width
    plot_height = 1200  # From the CSS .plot-container height
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Ensuring coordinates are available...")
    
    # Make sure we have the right coordinate columns
    coord_columns = {
        'umap': ['umap_x', 'umap_y'],
        'tsne': ['tsne_x', 'tsne_y'],
        'generic': ['x', 'y']
    }
    
    # First priority: check for generic columns
    if all(col in features_df.columns for col in coord_columns['generic']):
        x_col, y_col = coord_columns['generic']
        print(f"Using generic coordinate columns: {x_col}, {y_col}")
    # Second priority: check for method-specific columns
    elif all(col in features_df.columns for col in coord_columns[dim_reduction]):
        x_col, y_col = coord_columns[dim_reduction]
        print(f"Using {method_name}-specific coordinate columns: {x_col}, {y_col}")
    # Fall back to the other method if available
    elif dim_reduction == 'umap' and all(col in features_df.columns for col in coord_columns['tsne']):
        x_col, y_col = coord_columns['tsne']
        print(f"Using t-SNE coordinates as fallback: {x_col}, {y_col}")
    elif dim_reduction == 'tsne' and all(col in features_df.columns for col in coord_columns['umap']):
        x_col, y_col = coord_columns['umap']
        print(f"Using UMAP coordinates as fallback: {x_col}, {y_col}")
    else:
        raise ValueError(f"No suitable coordinate columns found in DataFrame. Available columns: {features_df.columns.tolist()}")
    
    # Extract coordinates and other info for samples with projections
    samples_with_projections = []
    
    # Track how many projections we have from each cluster for reporting
    cluster_counts = {}
    
    for idx in projection_paths:
        if idx in features_df.index:
            sample = features_df.loc[idx]
            
            # Use the determined coordinate columns
            if x_col in sample and y_col in sample:
                x, y = sample[x_col], sample[y_col]
                
                # Extract cluster and bbox information if available
                cluster = sample.get('cluster', 'N/A') if 'cluster' in sample else 'N/A'
                bbox = sample.get('bbox_name', 'unknown') if 'bbox_name' in sample else 'unknown'
                
                # Extract central coordinates if available
                central_coord_1 = sample.get('central_coord_1', 0) if 'central_coord_1' in sample else 0
                central_coord_2 = sample.get('central_coord_2', 0) if 'central_coord_2' in sample else 0
                central_coord_3 = sample.get('central_coord_3', 0) if 'central_coord_3' in sample else 0
                
                # Count samples per cluster
                if cluster != 'N/A':
                    cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
                
                # Convert numpy/pandas types to Python native types for JSON serialization
                if hasattr(idx, 'item'):
                    idx = idx.item()
                if hasattr(x, 'item'):
                    x = x.item()
                if hasattr(y, 'item'):
                    y = y.item()
                if hasattr(cluster, 'item'):
                    cluster = cluster.item()
                
                # Load the projection file and convert to base64
                try:
                    with open(projection_paths[idx], 'rb') as projection_file:
                        projection_file_data = projection_file.read()
                        encoded_projection = base64.b64encode(projection_file_data).decode('utf-8')
                        
                        # Get projection data for this sample index - should be in memory after create_gif_from_volume
                        sample_projections = {}
                        
                        # Use stored projection data if available (from function parameter)
                        if projection_data:
                            # Convert idx to string to match how it's stored in projection_data
                            str_idx = str(idx)
                            if str_idx in projection_data:
                                # The base64 data from projection_data is already encoded
                                sample_projections = projection_data[str_idx]
                                print(f"Using provided projection data for sample {idx}")
                            else:
                                print(f"No provided projection data found for sample {idx}, regenerating...")
                        else:
                            print(f"No projection_data dictionary provided, regenerating...")
                        
                        # Look up the sample in the dataset and recreate the projections if needed if not available from parameter
                        if not sample_projections and hasattr(dataset, '__getitem__') and idx < len(dataset):
                            sample_data = dataset[idx]
                            
                            # Extract volume
                            if isinstance(sample_data, tuple) and len(sample_data) > 0:
                                volume = sample_data[0]
                            elif isinstance(sample_data, dict):
                                volume = sample_data.get('pixel_values', sample_data.get('raw_volume'))
                            else:
                                volume = sample_data
                            
                            # Create projections
                            if volume is not None:
                                try:
                                    z_proj, y_proj, x_proj = create_average_intensity_projection(volume)
                                    
                                    # Convert projections to base64
                                    with io.BytesIO() as output:
                                        Image.fromarray(z_proj).save(output, format="PNG")
                                        sample_projections['z_proj'] = base64.b64encode(output.getvalue()).decode('utf-8')
                                    
                                    with io.BytesIO() as output:
                                        Image.fromarray(y_proj).save(output, format="PNG")
                                        sample_projections['y_proj'] = base64.b64encode(output.getvalue()).decode('utf-8')
                                    
                                    with io.BytesIO() as output:
                                        Image.fromarray(x_proj).save(output, format="PNG")
                                        sample_projections['x_proj'] = base64.b64encode(output.getvalue()).decode('utf-8')
                                except Exception as e:
                                    print(f"Error creating individual projections for sample {idx}: {e}")
                        
                        # Add the sample with all projections
                        sample_data = {
                            'id': idx,
                            'x': x,
                            'y': y,
                            'cluster': cluster,
                            'bbox': bbox,
                            'central_coord_1': central_coord_1,
                            'central_coord_2': central_coord_2, 
                            'central_coord_3': central_coord_3,
                            'projectionData': encoded_projection
                        }
                        
                        # Add individual projections if available
                        if sample_projections:
                            print(f"Adding projections for sample {idx} with keys: {list(sample_projections.keys())}")
                            for proj_key, proj_data in sample_projections.items():
                                try:
                                    # Don't try to encode the data, just use it directly
                                    # If it's already a string (base64 encoded), use it as is
                                    if isinstance(proj_data, str):
                                        sample_data[proj_key] = proj_data
                                        print(f"Added {proj_key} projection to sample {idx} (string data)")
                                    else:
                                        sample_data[proj_key] = proj_data
                                        print(f"Added {proj_key} projection to sample {idx} (direct data)")
                                except Exception as e:
                                    print(f"Error adding {proj_key} projection to sample {idx}: {e}")
                        else:
                            print(f"Warning: No projection data available for sample {idx}")
                        
                        samples_with_projections.append(sample_data)
                except Exception as e:
                    print(f"Error encoding projection for sample {idx}: {e}")
            else:
                print(f"Warning: Sample {idx} does not have required coordinates ({x_col}, {y_col}). Skipping.")
    
    # Print distribution of projections across clusters
    if cluster_counts:
        print("\nDistribution of projections across clusters:")
        for cluster, count in sorted(cluster_counts.items()):
            print(f"  Cluster {cluster}: {count} projections")
    
    if not samples_with_projections:
        raise ValueError("No samples with projections and valid coordinates found. Cannot create visualization.")
    
    print(f"\nTotal samples with projections and valid coordinates: {len(samples_with_projections)}")
    print(f"First sample for debugging: {json.dumps(samples_with_projections[0], default=str)[:200]}...")
    
    # Compute the bounds of the coordinate values
    all_x_values = features_df[x_col].values
    all_y_values = features_df[y_col].values
    
    print(f"X coordinate range: {min(all_x_values)} to {max(all_x_values)}")
    print(f"Y coordinate range: {min(all_y_values)} to {max(all_y_values)}")
    
    x_min, x_max = float(min(all_x_values)), float(max(all_x_values))
    y_min, y_max = float(min(all_y_values)), float(max(all_y_values))
    
    # Add padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding
    
    # Processing to create non-overlapping positions
    projection_size = 50  # Default size decreased from 100 to 50px
    shift_limit = 75  # Increased from 50 to 100px shift limit
    max_shift_x = shift_limit
    max_shift_y = shift_limit
    
    # Function to check if two rectangles overlap
    def do_rectangles_overlap(rect1, rect2):
        return not (rect1['right'] < rect2['left'] or 
                   rect1['left'] > rect2['right'] or 
                   rect1['bottom'] < rect2['top'] or 
                   rect1['top'] > rect2['bottom'])
    
    # Track placed rectangles to avoid overlap
    placed_rectangles = []
    
    # Function to find non-overlapping position
    def find_non_overlapping_position(baseX, baseY, existingRects):
        # Check if the original position works
        half_size = projection_size / 2
        rect = {
            'left': baseX - half_size,
            'right': baseX + half_size,
            'top': baseY - half_size,
            'bottom': baseY + half_size
        }
        
        # Check if original position has no overlap
        has_overlap = False
        overlap_rect = None
        
        for existing_rect in existingRects:
            if do_rectangles_overlap(rect, existing_rect):
                has_overlap = True
                overlap_rect = existing_rect
                break
                
        # If no overlap, use original position
        if not has_overlap:
            return (baseX, baseY, rect)
            
        # Calculate the minimum shift needed in each direction to avoid overlap
        if overlap_rect:
            # Calculate overlap amounts in each direction
            overlap_right = rect['right'] - overlap_rect['left']
            overlap_left = overlap_rect['right'] - rect['left']
            overlap_bottom = rect['bottom'] - overlap_rect['top']
            overlap_top = overlap_rect['bottom'] - rect['top']
            
            # Find the smallest shift needed
            shifts = [
                {'axis': 'x', 'amount': overlap_right, 'direction': 1},   # shift right
                {'axis': 'x', 'amount': -overlap_left, 'direction': -1},  # shift left
                {'axis': 'y', 'amount': overlap_bottom, 'direction': 1},  # shift down
                {'axis': 'y', 'amount': -overlap_top, 'direction': -1}    # shift up
            ]
            
            # Sort by absolute amount to find smallest shift
            shifts.sort(key=lambda s: abs(s['amount']))
            
            # Try each shift until we find one that works
            for shift in shifts:
                # Skip if shift is too large
                if abs(shift['amount']) > shift_limit:
                    continue
                
                shifted_x = baseX
                shifted_y = baseY
                
                if shift['axis'] == 'x':
                    shifted_x += shift['amount']
                else:
                    shifted_y += shift['amount']
                
                # Skip if this would move the projection out of bounds
                if (shifted_x - half_size < 0 or shifted_x + half_size > plot_width or
                    shifted_y - half_size < 0 or shifted_y + half_size > plot_height):
                    continue
                
                # Check if this position works with all existing rectangles
                shifted_rect = {
                    'left': shifted_x - half_size,
                    'right': shifted_x + half_size,
                    'top': shifted_y - half_size,
                    'bottom': shifted_y + half_size
                }
                
                shifted_overlap = False
                for existing_rect in existingRects:
                    if do_rectangles_overlap(shifted_rect, existing_rect):
                        shifted_overlap = True
                        break
                        
                if not shifted_overlap:
                    return (shifted_x, shifted_y, shifted_rect)
        
        # If the simple shifts didn't work, try a more general approach
        # Try cardinal and diagonal directions with increasing distances
        directions = [
            (1, 0),   # right
            (0, 1),   # down
            (-1, 0),  # left
            (0, -1),  # up
            (1, 1),   # down-right
            (1, -1),  # up-right
            (-1, 1),  # down-left
            (-1, -1)  # up-left
        ]
        
        # Try increasing distances with smaller steps
        for distance in range(1, int(shift_limit) + 1):
            for dir_x, dir_y in directions:
                shifted_x = baseX + (dir_x * distance)
                shifted_y = baseY + (dir_y * distance)
                
                # Skip if this would move the projection out of bounds
                if (shifted_x - half_size < 0 or shifted_x + half_size > plot_width or
                    shifted_y - half_size < 0 or shifted_y + half_size > plot_height):
                    continue
                
                # Check this position
                shifted_rect = {
                    'left': shifted_x - half_size,
                    'right': shifted_x + half_size,
                    'top': shifted_y - half_size,
                    'bottom': shifted_y + half_size
                }
                
                shifted_overlap = False
                for existing_rect in existingRects:
                    if do_rectangles_overlap(shifted_rect, existing_rect):
                        shifted_overlap = True
                        break
                        
                if not shifted_overlap:
                    return (shifted_x, shifted_y, shifted_rect)
        
        # Try with slightly smaller projection size as a last resort
        reduced_half_size = half_size * 0.8
        for distance in range(1, int(shift_limit) + 1, 2):
            for dir_x, dir_y in directions:
                shifted_x = baseX + (dir_x * distance)
                shifted_y = baseY + (dir_y * distance)
                
                # Skip if this would move the projection out of bounds
                if (shifted_x - reduced_half_size < 0 or shifted_x + reduced_half_size > plot_width or
                    shifted_y - reduced_half_size < 0 or shifted_y + reduced_half_size > plot_height):
                    continue
                
                # Check this position with reduced size
                shifted_rect = {
                    'left': shifted_x - reduced_half_size,
                    'right': shifted_x + reduced_half_size,
                    'top': shifted_y - reduced_half_size,
                    'bottom': shifted_y + reduced_half_size
                }
                
                shifted_overlap = False
                for existing_rect in existingRects:
                    if do_rectangles_overlap(shifted_rect, existing_rect):
                        shifted_overlap = True
                        break
                        
                if not shifted_overlap:
                    return (shifted_x, shifted_y, shifted_rect)
        
        # If we can't find a non-overlapping position, return null
        return None
    
    # Initialize originalPositions dictionary for storing initial projection positions
    # Note: we need to map the raw coordinates to plot coordinates
    # For this we use the same mapping logic as in the JavaScript mapToPlot function
    orig_positions_dict = {}
    samples_to_remove = []
    
    for i, sample in enumerate(samples_with_projections):
        id_val = sample['id']
        if hasattr(id_val, 'item'):
            id_val = id_val.item()  # Convert numpy types to native Python
        
        x_val = sample['x']
        y_val = sample['y']
        
        # Map data coordinates to plot coordinates (same as JavaScript mapToPlot function)
        plot_x = ((x_val - x_min) / (x_max - x_min)) * plot_width
        # Invert y-axis (data coordinates increase upward, plot coordinates increase downward)
        plot_y = plot_height - ((y_val - y_min) / (y_max - y_min)) * plot_height
        
        # Find non-overlapping position
        position = find_non_overlapping_position(plot_x, plot_y, placed_rectangles)
        
        # If no valid position found, skip this sample
        if position is None:
            print(f"Skipping sample {id_val} due to overlap that couldn't be resolved")
            samples_to_remove.append(i)
            continue
            
        # Unpack the position
        pos_x, pos_y, rect = position
        
        # Add to tracking for future samples
        placed_rectangles.append(rect)
        
        # Use string keys for the JavaScript object
        str_id = str(id_val)
        orig_positions_dict[str_id] = {"x": float(pos_x), "y": float(pos_y)}
        
        # If position was shifted, note it
        if pos_x != plot_x or pos_y != plot_y:
            print(f"Sample {id_val} shifted to avoid overlap")
    
    # Remove samples that couldn't be placed
    if samples_to_remove:
        for i in sorted(samples_to_remove, reverse=True):
            del samples_with_projections[i]
        print(f"Removed {len(samples_to_remove)} samples that couldn't be placed without overlap")
    
    # Convert to JSON string for embedding in JavaScript
    originalPositions = json.dumps(orig_positions_dict)
    print(f"originalPositions JSON string length: {len(originalPositions)}")
    print(f"Sample of originalPositions: {originalPositions[:100]}...")
    
    # Define colors for points based on clusters or bboxes
    point_colors = {}
    bbox_colors = {}
    
    if 'cluster' in features_df.columns:
        # Generate colors for each cluster
        clusters = features_df['cluster'].unique()
        import matplotlib.pyplot as plt
        
        # Use the new recommended approach to get colormaps
        try:
            cmap = plt.colormaps['tab10']
        except AttributeError:
            # Fallback for older matplotlib versions
            cmap = plt.cm.get_cmap('tab10')
        
        for i, cluster in enumerate(clusters):
            r, g, b, _ = cmap(i % 10)  # Use modulo to handle more than 10 clusters
            point_colors[cluster] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    
    # Generate colors for bboxes
    if 'bbox_name' in features_df.columns:
        bboxes = features_df['bbox_name'].unique()
        bbox_colors_list = [
            '#FF0000', '#00FFFF', '#FFA500', '#800080', 
            '#008000', '#0000FF', '#FF00FF', '#FFFF00', 
            '#808080', '#000000'
        ]
        
        for i, bbox in enumerate(bboxes):
            if i < len(bbox_colors_list):
                bbox_colors[bbox] = bbox_colors_list[i]
            else:
                # Generate a random color if we run out of predefined colors
                r = random.randint(0, 255)
                g = random.randint(0, 255) 
                b = random.randint(0, 255)
                bbox_colors[bbox] = f"rgb({r}, {g}, {b})"
    
    # Generate HTML content for data points
    points_content = ""
    for idx, row in features_df.iterrows():
        if x_col in row and y_col in row:
            x, y = row[x_col], row[y_col]
            
            # Convert to native Python types
            if hasattr(idx, 'item'):
                idx = idx.item()
            if hasattr(x, 'item'):
                x = x.item()
            if hasattr(y, 'item'):
                y = y.item()
            
            # Determine color based on cluster
            cluster_color = 'rgb(100, 100, 100)'
            cluster = None
            if 'cluster' in row:
                cluster = row['cluster']
                if hasattr(cluster, 'item'):
                    cluster = cluster.item()
                cluster_color = point_colors.get(cluster, 'rgb(100, 100, 100)')
            
            # Get bbox_name and color based on bbox
            bbox_name = row.get('bbox_name', 'unknown')
            if hasattr(bbox_name, 'item'):
                bbox_name = str(bbox_name.item())
            else:
                bbox_name = str(bbox_name)
            
            bbox_color = bbox_colors.get(bbox_name, 'rgb(100, 100, 100)')
            
            # Get Var1 for tooltip
            var1 = row.get('Var1', f'sample_{idx}')
            if hasattr(var1, 'item'):
                var1 = str(var1.item())
            else:
                var1 = str(var1)
            
            # Add this point to the samples array - make sure we have a valid number before adding
            if not (np.isnan(x) or np.isnan(y)):
                points_content += f"""
                {{
                    "id": {idx},
                    "x": {x},
                    "y": {y},
                    "color": "{cluster_color}",
                    "bbox_color": "{bbox_color}",
                    "cluster": "{str(cluster) if cluster is not None else 'unknown'}",
                    "hasProjection": {str(idx in projection_paths).lower()},
                    "bbox_name": "{bbox_name}",
                    "var1": "{var1}"
                }},"""
    
    # Count how many valid points we have
    print(f"Generated points_content with {points_content.count('id:')} points")
    print(f"Sample of points_content: {points_content[:200]}...")
    
    # Generate HTML content for projections
    projections_content = ""
    for sample in samples_with_projections:
        # Only include projection data if we have it
        has_projection = sample.get('projectionData') is not None and len(sample.get('projectionData', [])) > 0
        
        try:
            # Build the JSON for this sample with default projectionData first
            projection_json = f"""{{
                "id": {sample.get('id', 0)},
                "x": {sample.get('x', 0)},
                "y": {sample.get('y', 0)},
                "cluster": "{sample.get('cluster', 'N/A')}",
                "bbox": "{sample.get('bbox', 'unknown')}",
                "central_coord_1": {sample.get('central_coord_1', 0)},
                "central_coord_2": {sample.get('central_coord_2', 0)},
                "central_coord_3": {sample.get('central_coord_3', 0)}"""
            
            # Add projectionData if available 
            if has_projection:
                projection_json += f""",
                "projectionData": "{sample['projectionData']}" """
            
            # Add individual projection views if available
            for proj_type in ['z_proj', 'y_proj', 'x_proj']:
                if proj_type in sample and sample[proj_type]:
                    projection_json += f""",
                "{proj_type}": "{sample[proj_type]}" """
            
            # Add the hasProjection flag and close the JSON object
            projection_json += f""",
                "hasProjection": {str(has_projection).lower()}
            }},"""
            
            projections_content += projection_json
        except Exception as e:
            print(f"Error creating projection JSON for sample {sample.get('id', 0)}: {e}")
            continue
    
    # Count how many valid projections we have
    print(f"Generated projections_content with {projections_content.count('id:')} projections")
    print(f"Sample of projections_content (truncated): {projections_content[:100]}...")
    
    # Read the HTML template
    template_path = os.path.join(os.path.dirname(__file__), "average_intensity_template.html")
    try:
        with open(template_path, 'r', encoding='utf-8') as template_file:
            html_content = template_file.read()
            
        # Replace placeholders with actual data
        html_content = html_content.replace('{method_name}', method_name)
        html_content = html_content.replace('{x_min}', str(x_min))
        html_content = html_content.replace('{x_max}', str(x_max))
        html_content = html_content.replace('{y_min}', str(y_min))
        html_content = html_content.replace('{y_max}', str(y_max))
        html_content = html_content.replace('{originalPositions}', originalPositions)
        html_content = html_content.replace('{points_content}', points_content)
        html_content = html_content.replace('{projections_content}', projections_content)
        html_content = html_content.replace('{len(samples_with_projections)}', str(len(samples_with_projections)))
        
        print("Successfully loaded and processed HTML template")
    except Exception as e:
        print(f"Error loading or processing HTML template: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save the HTML file
    html_path = output_dir / f"projected_images_{dim_reduction}_visualization.html"
    try:
        # Add debug statements to narrow down the error
        print("Types of data:")
        print("- originalPositions type:", type(originalPositions))
        print("- points_content type:", type(points_content))
        print("- points_content length:", len(points_content))
        print("- projections_content type:", type(projections_content))
        print("- projections_content length:", len(projections_content))
        print("- First item in samples_with_projections:", samples_with_projections[0] if samples_with_projections else "No samples")
        
        # Check if the points_content and projections_content are empty 
        if not points_content.strip():
            print("WARNING: points_content is empty! No background points will be shown.")
        
        if not projections_content.strip():
            print("WARNING: projections_content is empty! No projections will be shown.")
            
        # Write the HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Created projected image visualization with embedded data: {html_path}")
        
        # Print a sample of the HTML content to verify it has data
        html_sample = html_content[0:1000]  # Get first 1000 chars
        print(f"Sample of HTML content: {html_sample}")
        
        # Check HTML file size
        html_size = os.path.getsize(html_path)
        print(f"HTML file size: {html_size} bytes")
        
        if html_size < 10000:  # If file is too small, something might be wrong
            print("WARNING: HTML file is very small! It might not contain all necessary data.")
            
    except Exception as e:
        print(f"Error creating projected image visualization: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
    
    return html_path

# Define function to initialize a dataset from the newdl folder
def initialize_dataset_from_newdl():
    """
    Initialize a SynapseDataset from the newdl folder if it's not available from the Clustering module.
    
    Returns:
        SynapseDataset instance or None if initialization fails
    """
    try:
        print("Initializing dataset from newdl...")
        # Import required classes from newdl

        from newdl.dataset3 import SynapseDataset
        from newdl.dataloader3 import SynapseDataLoader, Synapse3DProcessor

        # Initialize data loader
        data_loader = SynapseDataLoader(
            raw_base_dir=config.raw_base_dir,
            seg_base_dir=config.seg_base_dir,
            add_mask_base_dir=config.add_mask_base_dir
        )
        
        # Load volumes
        vol_data_dict = {}
        for bbox_name in config.bbox_name:
            print(f"Loading volumes for {bbox_name}...")
            raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
            if raw_vol is not None:
                vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
        
        # Check if we have any volumes
        if not vol_data_dict:
            print("No volumes could be loaded. Check the raw_base_dir, seg_base_dir, and add_mask_base_dir in config.")
            return None
            
        # Load synapse data
        syn_df = pd.DataFrame()
        if config.excel_file:
            try:
                syn_df = pd.concat([
                    pd.read_excel(os.path.join(config.excel_file, f"{bbox}.xlsx")).assign(bbox_name=bbox)
                    for bbox in config.bbox_name if os.path.exists(os.path.join(config.excel_file, f"{bbox}.xlsx"))
                ])
                print(f"Loaded synapse data: {len(syn_df)} rows")
            except Exception as e:
                print(f"Error loading Excel files: {e}")
                
        # Initialize processor
        processor = Synapse3DProcessor(size=config.size)
        processor.normalize_volume = False
        
        # Create dataset
        dataset = SynapseDataset(
            vol_data_dict=vol_data_dict,
            synapse_df=syn_df,
            processor=processor,
            segmentation_type=config.segmentation_type,
            subvol_size=config.subvol_size,
            num_frames=config.num_frames,
            alpha=config.alpha,
            normalize_across_volume=False  # Set to False for consistent gray values
        )
        
        print(f"Successfully created dataset with {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"Error initializing dataset from newdl: {e}")
        return None

# Define the perform_clustering_analysis function here instead of importing it
def perform_clustering_analysis(config, csv_path, output_path):
    """
    Perform clustering analysis on features from a CSV file using parameters from config.
    
    Args:
        config: Configuration object containing clustering parameters
        csv_path: Path to the CSV file containing features
        output_path: Directory to save clustering results
    
    Returns:
        features_df: DataFrame with cluster assignments
    """
    print(f"Starting clustering analysis on {csv_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and cluster features
    # Note: Only passing the parameters that the function accepts
    features_df, clusterer, feature_cols = load_and_cluster_features(
        csv_path, 
        n_clusters=config.n_clusters,
        random_state=42  # Using a fixed random state for reproducibility
    )
    
    # Save clustered features
    features_df.to_csv(output_dir / "clustered_features.csv", index=False)
    
    # Apply t-SNE for dimensionality reduction and visualization
    tsne_results_2d = apply_tsne(features_df, feature_cols, 2)
    tsne_results_3d = apply_tsne(features_df, feature_cols, 3)
    
    # Add t-SNE results to the features DataFrame
    features_df['tsne_x'] = tsne_results_2d[:, 0]
    features_df['tsne_y'] = tsne_results_2d[:, 1]
    features_df['tsne_z'] = tsne_results_3d[:, 2] if tsne_results_3d.shape[1] > 2 else 0
    
    # Define color mapping for different bounding boxes
    color_mapping = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    
    # Save t-SNE plots
    save_tsne_plots(features_df, tsne_results_2d, tsne_results_3d, clusterer, color_mapping, output_dir)
    # TODO: Maybe later - implement sample visualization
    # Find random samples from each cluster for visualization
    # random_samples_per_cluster = find_random_samples_in_clusters(features_df, feature_cols, 4)
    
    # Save cluster sample visualizations - skip this if dataset is not available
    # try:
    #     from Clustering import dataset
    #     save_cluster_samples(dataset, random_samples_per_cluster, output_dir)
    # except ImportError:
    #     print("Warning: 'dataset' not available, skipping sample visualization")
    
    print(f"Clustering analysis completed. Results saved to {output_dir}")
    
    return features_df

def extract_cleft_mask_and_max_slices(dataset, idx, bbox_name, features_df=None):
    """
    Extract the cleft mask for a given sample and calculate the max slices.
    
    Args:
        dataset: The synapse dataset
        idx: Sample index
        bbox_name: Name of the bounding box
        features_df: Features dataframe that may contain central coordinates
        
    Returns:
        dict: Dictionary with max slice information for x, y, z dimensions
              or None if cleft mask cannot be extracted
    """
    try:
        print(f"Extracting cleft mask for sample {idx}, bbox {bbox_name}")
        
        # Access the dataset's vol_data_dict to get the raw and cleft data
        if hasattr(dataset, 'vol_data_dict') and bbox_name in dataset.vol_data_dict:
            # Get the raw, seg, and add_mask volumes
            _, _, add_mask_vol = dataset.vol_data_dict[bbox_name]
            print(f"Got add_mask_vol for {bbox_name} with shape {add_mask_vol.shape}")
            
            # Default central coordinates to center of volume if not found
            cx, cy, cz = None, None, None
            
            # First try to get coordinates from synapse_df
            if hasattr(dataset, 'synapse_df'):
                syn_info = dataset.synapse_df[dataset.synapse_df['bbox_name'] == bbox_name]
                if not syn_info.empty:
                    row = syn_info.iloc[0]  # Get first matching row
                    if 'central_coord_1' in row and 'central_coord_2' in row and 'central_coord_3' in row:
                        cx = int(row['central_coord_1'])
                        cy = int(row['central_coord_2'])
                        cz = int(row['central_coord_3'])
                        print(f"Found coordinates in synapse_df: ({cx}, {cy}, {cz})")
            
            # If not found, try features_df
            if (cx is None or cy is None or cz is None) and features_df is not None and idx in features_df.index:
                row = features_df.loc[idx]
                if 'central_coord_1' in row and 'central_coord_2' in row and 'central_coord_3' in row:
                    cx = int(row['central_coord_1'])
                    cy = int(row['central_coord_2'])
                    cz = int(row['central_coord_3'])
                    print(f"Found coordinates in features_df: ({cx}, {cy}, {cz})")
            
            # Use center of volume as default if coordinates still not found
            if cx is None or cy is None or cz is None:
                cx = add_mask_vol.shape[2] // 2
                cy = add_mask_vol.shape[1] // 2
                cz = add_mask_vol.shape[0] // 2
                print(f"Using default center coordinates: ({cx}, {cy}, {cz})")
            
            # Get appropriate cleft label
            cleft_label, cleft_label2 = get_cleft_label(bbox_name)
            print(f"Using cleft labels {cleft_label}, {cleft_label2} for {bbox_name}")
            
            # Calculate subvolume bounds (using 80×80×80 as standard)
            half_size = 40
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
            
            # Check if the mask contains any positive pixels
            total_cleft_pixels = np.sum(cleft_mask)
            print(f"Cleft mask for {bbox_name} has {total_cleft_pixels} cleft pixels")
            
            if total_cleft_pixels > 0:
                # Find max slices
                max_slices = find_max_cleft_slices(cleft_mask)
                print(f"Found max slices for {bbox_name}: {max_slices}")
                return max_slices
            else:
                print(f"Warning: No cleft pixels found in mask for sample {idx}, bbox {bbox_name}")
                # Create a default max slices dictionary indicating the center of the volume
                default_max_slices = {
                    'max_z_slice': cleft_mask.shape[0] // 2,
                    'max_z_value': 0,
                    'max_y_slice': cleft_mask.shape[1] // 2,
                    'max_y_value': 0,
                    'max_x_slice': cleft_mask.shape[2] // 2,
                    'max_x_value': 0
                }
                return default_max_slices
        else:
            print(f"Error: Could not access vol_data_dict for sample {idx}, bbox {bbox_name}")
        
        return None
    except Exception as e:
        print(f"Error extracting cleft mask for sample {idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Parse configuration
    config.parse_args()
    
    # Add argument for dimensionality reduction method
    import argparse
    parser = argparse.ArgumentParser(description='UMAP and t-SNE Visualization with Intensity Projections')
    parser.add_argument('--dim-reduction', 
                       choices=['umap', 'tsne'], 
                       default='umap',
                       help='Dimensionality reduction method to use (umap or tsne)')
    parser.add_argument('--num-samples', 
                       type=int,
                       default=20,
                       help='Number of random samples to show with projections (default: 20)')
    parser.add_argument('--custom-clusters',
                       type=str,
                       default=r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\clusters\clustered_features.csv",
                       help='Path to custom cluster assignments CSV file')
    args, unknown = parser.parse_known_args()
    
    # Define paths 10
    # csv_path = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\10\features_layer20_seg10_alpha1.0\features_layer20_seg10_alpha1_0.csv"
    # output_path = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\10"
    # args.segtype = 10
    # Define paths 11
    # csv_path = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\11\features_layer20_seg11_alpha1.0\features_layer20_seg11_alpha1_0.csv"
    # output_path = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\11temp"
    # args.segtype = 11
    # Define paths 13
    csv_path = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\13\features_layer20_seg13_alpha1.0\features_layer20_seg13_alpha1_0.csv"
    output_path = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\13"
    # args.segtype = 13
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory created/verified at: {output_dir}")
    
    # Load existing features_df if it exists to avoid re-running clustering
    clustered_features_path = output_dir / "clustered_features.csv"
    
    # First check for custom cluster file at the specified path
    custom_cluster_path = args.custom_clusters
    
    if os.path.exists(custom_cluster_path):
        print(f"Loading custom cluster assignments from {custom_cluster_path}")
        custom_clusters_df = pd.read_csv(custom_cluster_path)
        
        # Check if it has the required columns
        if 'bbox' in custom_clusters_df.columns and 'Var1' in custom_clusters_df.columns and 'cluster' in custom_clusters_df.columns:
            print(f"Custom cluster file contains {len(custom_clusters_df)} cluster assignments")
            print(f"Sample of cluster data: {custom_clusters_df.head(3)}")
            
            # Load the feature file
            if os.path.exists(csv_path):
                print(f"Loading features from {csv_path}")
                features_df = pd.read_csv(csv_path)
                print(f"Features file contains {len(features_df)} samples")
                
                # Ensure index column is used as actual index
                if 'Unnamed: 0' in features_df.columns:
                    features_df.set_index('Unnamed: 0', inplace=True)
                elif 'index' in features_df.columns:
                    features_df.set_index('index', inplace=True)
                
                # Check for bbox column name variations
                bbox_col = None
                for col in ['bbox_name', 'bbox', 'bounding_box']:
                    if col in features_df.columns:
                        bbox_col = col
                        print(f"Found bbox column: {bbox_col}")
                        break
                
                if bbox_col is None:
                    print("Warning: Could not find bbox column in features file")
                    print(f"Available columns: {features_df.columns.tolist()}")
                
                # Check for Var1 column
                var1_col = None
                if 'Var1' in features_df.columns:
                    var1_col = 'Var1'
                    print(f"Found Var1 column: {var1_col}")
                
                if var1_col is None:
                    print("Warning: Could not find Var1 column in features file")
                    print(f"Available columns: {features_df.columns.tolist()}")
                
                # Create a mapping dictionary from (bbox, Var1) to cluster
                cluster_mapping = {}
                for _, row in custom_clusters_df.iterrows():
                    key = (row['bbox'], row['Var1'])
                    cluster_mapping[key] = row['cluster']
                
                print(f"Created cluster mapping with {len(cluster_mapping)} entries")
                
                # Apply the mapping to the features_df if we have the necessary columns
                if bbox_col is not None and var1_col is not None:
                    # Print a few samples before mapping
                    print(f"Sample data before mapping: {features_df[[bbox_col, var1_col]].head(3)}")
                    
                    # Apply the mapping
                    features_df['cluster'] = features_df.apply(
                        lambda row: cluster_mapping.get((row[bbox_col], row[var1_col]), -1), axis=1
                    )
                    
                    # Check if we successfully mapped clusters
                    mapped_count = sum(features_df['cluster'] != -1)
                    print(f"Successfully mapped {mapped_count} out of {len(features_df)} samples to clusters")
                    
                    if mapped_count == 0:
                        print("WARNING: No clusters were mapped! Check for data format issues:")
                        print(f"Cluster file bbox column sample: {custom_clusters_df['bbox'].head(3)}")
                        print(f"Features file {bbox_col} column sample: {features_df[bbox_col].head(3)}")
                        print(f"Cluster file Var1 column sample: {custom_clusters_df['Var1'].head(3)}")
                        print(f"Features file {var1_col} column sample: {features_df[var1_col].head(3)}")
                        
                        # Try to find matching entries manually for debugging
                        found_match = False
                        for _, feature_row in features_df.head(10).iterrows():
                            bbox_value = feature_row[bbox_col]
                            var1_value = feature_row[var1_col]
                            for _, cluster_row in custom_clusters_df.head(20).iterrows():
                                if cluster_row['bbox'] == bbox_value and cluster_row['Var1'] == var1_value:
                                    print(f"Found match: ({bbox_value}, {var1_value}) -> Cluster {cluster_row['cluster']}")
                                    found_match = True
                                    break
                        
                        if not found_match:
                            print("No matches found in the first 10 samples. Trying an alternative mapping approach...")
                            # Try a different mapping approach
                            # Strip any leading/trailing whitespace and handle case sensitivity
                            features_df['_bbox_clean'] = features_df[bbox_col].str.strip().str.lower()
                            features_df['_var1_clean'] = features_df[var1_col].str.strip().str.lower()
                            
                            custom_clusters_df['_bbox_clean'] = custom_clusters_df['bbox'].str.strip().str.lower()
                            custom_clusters_df['_var1_clean'] = custom_clusters_df['Var1'].str.strip().str.lower()
                            
                            # Create new mapping with cleaned values
                            clean_mapping = {}
                            for _, row in custom_clusters_df.iterrows():
                                clean_key = (row['_bbox_clean'], row['_var1_clean'])
                                clean_mapping[clean_key] = row['cluster']
                            
                            # Apply the cleaned mapping
                            features_df['cluster'] = features_df.apply(
                                lambda row: clean_mapping.get((row['_bbox_clean'], row['_var1_clean']), -1), axis=1
                            )
                            
                            # Check results of cleaned mapping
                            clean_mapped_count = sum(features_df['cluster'] != -1)
                            print(f"After cleaning: Mapped {clean_mapped_count} out of {len(features_df)} samples")
                            
                            # Remove temporary columns
                            features_df.drop(['_bbox_clean', '_var1_clean'], axis=1, inplace=True)
                    
                    # Save the clustered features
                    features_df.to_csv(clustered_features_path)
                    print(f"Saved clustered features to {clustered_features_path}")
                else:
                    print(f"Warning: Features file does not have necessary bbox and Var1 columns needed for mapping")
                    
                    # Fall back to regular clustering
                    if os.path.exists(clustered_features_path):
                        print(f"Loading existing clustered features from {clustered_features_path}")
                        features_df = pd.read_csv(clustered_features_path)
                        # Ensure index column is used as actual index
                        if 'Unnamed: 0' in features_df.columns:
                            features_df = features_df.set_index('Unnamed: 0')
                        elif 'index' in features_df.columns:
                            features_df = features_df.set_index('index')
                    else:
                        # Run clustering analysis
                        features_df = perform_clustering_analysis(config, csv_path, output_path)
            else:
                print(f"Error: Features file {csv_path} not found")
                sys.exit(1)
        else:
            print(f"Warning: Custom cluster file does not have required columns (bbox, Var1, cluster)")
            print(f"Columns in custom cluster file: {custom_clusters_df.columns.tolist()}")
            # Fall back to regular clustering path
            if os.path.exists(clustered_features_path):
                print(f"Loading existing clustered features from {clustered_features_path}")
                features_df = pd.read_csv(clustered_features_path)
                # Ensure index column is used as actual index
                if 'Unnamed: 0' in features_df.columns:
                    features_df = features_df.set_index('Unnamed: 0')
                elif 'index' in features_df.columns:
                    features_df = features_df.set_index('index')
            else:
                # Run clustering analysis
                features_df = perform_clustering_analysis(config, csv_path, output_path)
    elif os.path.exists(clustered_features_path):
        print(f"Loading existing clustered features from {clustered_features_path}")
        features_df = pd.read_csv(clustered_features_path)
        # Ensure index column is used as actual index
        if 'Unnamed: 0' in features_df.columns:
            features_df = features_df.set_index('Unnamed: 0')
        elif 'index' in features_df.columns:
            features_df = features_df.set_index('index')
    else:
        # Run clustering analysis
        features_df = perform_clustering_analysis(config, csv_path, output_path)
    
    print(f"Loaded features DataFrame with {len(features_df)} samples")
    print(f"Columns in features_df: {features_df.columns.tolist()}")
    print(f"Using dimensionality reduction method: {args.dim_reduction.upper()}")
    
    # Create UMAP or t-SNE visualization with GIFs
    # Try to get dataset from different sources
    dataset = None
    
    print("Dataset not available from Clustering module, trying to initialize from newdl...")
    dataset = initialize_dataset_from_newdl()
    
    # If we have a dataset, create the visualization
    if dataset is not None:
        # Create visualization with selected dim reduction method
        print(f"Creating {args.dim_reduction.upper()} visualization with projections for {args.num_samples} random samples...")
        
        # Check dataset length to ensure we only choose valid indices
        try:
            dataset_length = len(dataset)
            print(f"Dataset contains {dataset_length} samples")
            
            # Make sure the dataset has samples
            if dataset_length == 0:
                raise ValueError("Dataset is empty")
                
            # Get valid indices that are both in features_df and within dataset range
            valid_indices = [i for i in features_df.index if i < dataset_length]
            if len(valid_indices) == 0:
                print("Warning: No valid indices found that exist in both the dataset and features DataFrame.")
                print("Creating visualization without sample projections.")
                valid_indices = []
        except Exception as e:
            print(f"Warning: Could not determine dataset length: {e}")
            print("Assuming all feature indices are valid.")
            valid_indices = features_df.index.tolist()
        
        # Compute coordinates if not already in the DataFrame
        if args.dim_reduction == 'umap' and ('x' not in features_df.columns or 'y' not in features_df.columns):
            print("Computing UMAP...")
            feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
            if not feature_cols:
                feature_cols = [col for col in features_df.columns if 'layer' in col and 'feat_' in col]
                
            if feature_cols:
                features = features_df[feature_cols].values
                features_scaled = StandardScaler().fit_transform(features)
                
                # Use UMAP directly
                reducer = umap.UMAP(n_components=2, random_state=42)
                umap_results = reducer.fit_transform(features_scaled)
                
                features_df['x'] = umap_results[:, 0]
                features_df['y'] = umap_results[:, 1]
            else:
                print("ERROR: No feature columns found in the DataFrame")
                feature_cols = []
        elif args.dim_reduction == 'tsne' and ('tsne_x' not in features_df.columns or 'tsne_y' not in features_df.columns):
            print("Computing t-SNE...")
            feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
            if not feature_cols:
                feature_cols = [col for col in features_df.columns if 'layer' in col and 'feat_' in col]
                
            if feature_cols:
                features = features_df[feature_cols].values
                features_scaled = StandardScaler().fit_transform(features)
                
                # Use t-SNE
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42)
                tsne_results = tsne.fit_transform(features_scaled)
                
                features_df['tsne_x'] = tsne_results[:, 0]
                features_df['tsne_y'] = tsne_results[:, 1]
            else:
                print("ERROR: No feature columns found in the DataFrame")
                feature_cols = []
        
        # Select random samples for GIFs
        # If there are valid indices, select samples and create GIFs
        if valid_indices:
            # Create output directories
            projections_dir = output_dir / "sample_projections"
            projections_dir.mkdir(parents=True, exist_ok=True)
            
            # Select random samples, making sure they cover different clusters if possible
            np.random.seed(42)
            random_samples = []
            
            # Set number of samples to 40 as requested
            num_samples = 300  # User requested 40 samples
            
            if 'cluster' in features_df.columns:
                # Get all cluster IDs
                clusters = features_df['cluster'].unique()
                
                # Get approximately even samples from each cluster (limited to valid indices)
                samples_per_cluster = max(1, num_samples // len(clusters))
                remaining_samples = num_samples - (samples_per_cluster * len(clusters))
                
                for cluster in clusters:
                    # Get samples that are both in this cluster AND in valid_indices
                    cluster_df = features_df[features_df['cluster'] == cluster]
                    valid_cluster_indices = [i for i in cluster_df.index if i in valid_indices]
                    
                    if valid_cluster_indices:
                        # Select random indices from this cluster
                        sample_count = min(samples_per_cluster, len(valid_cluster_indices))
                        selected_indices = np.random.choice(valid_cluster_indices, size=sample_count, replace=False)
                        random_samples.extend(selected_indices)
                
                # Add any remaining samples from random clusters
                remaining_valid = [i for i in valid_indices if i not in random_samples]
                if remaining_samples > 0 and remaining_valid:
                    extra_samples = np.random.choice(remaining_valid, 
                                                   size=min(remaining_samples, len(remaining_valid)), 
                                                   replace=False)
                    random_samples.extend(extra_samples)
            else:
                # No clusters, just select random samples from valid indices
                sample_count = min(args.num_samples, len(valid_indices))
                random_samples = np.random.choice(valid_indices, size=sample_count, replace=False)
                
            # If we still don't have enough samples, try to add more from any valid indices
            if len(random_samples) < args.num_samples and len(valid_indices) > len(random_samples):
                additional_indices = [i for i in valid_indices if i not in random_samples]
                additional_count = min(args.num_samples - len(random_samples), len(additional_indices))
                if additional_count > 0:
                    additional_samples = np.random.choice(additional_indices, size=additional_count, replace=False)
                    random_samples = np.concatenate([random_samples, additional_samples])
            
            print(f"Selected {len(random_samples)} samples for projection creation")
            
            # Create projections for selected samples
            print(f"Creating projections for {len(random_samples)} samples...")
            projection_paths = {}
            
            for idx in random_samples:
                try:
                    # Validate index is within dataset range before accessing
                    if hasattr(dataset, '__len__') and idx >= len(dataset):
                        print(f"Skipping sample {idx} as it is out of bounds for dataset with length {len(dataset)}")
                        continue
                        
                    # Get the sample from the dataset
                    sample_data = dataset[idx]
                    
                    # Extract volume data (assuming dataset returns a tuple or has a standard format)
                    if isinstance(sample_data, tuple) and len(sample_data) > 0:
                        volume = sample_data[0]  # First element is typically the volume
                        if len(sample_data) > 2:
                            bbox_name = sample_data[2]  # Third element might be bbox_name
                        else:
                            bbox_name = None
                    elif isinstance(sample_data, dict):
                        volume = sample_data.get('pixel_values', sample_data.get('raw_volume'))
                        bbox_name = sample_data.get('bbox_name')
                    else:
                        volume = sample_data
                        bbox_name = None
                    
                    # If bbox_name not found in sample_data, try to get it from features_df
                    if not bbox_name and idx in features_df.index:
                        bbox_name = features_df.loc[idx].get('bbox_name', 'unknown')
                    
                    # Skip if no volume data found or it's None/empty
                    if volume is None or (hasattr(volume, 'numel') and volume.numel() == 0) or \
                       (hasattr(volume, 'size') and np.prod(volume.shape) == 0):
                        print(f"Skipping sample {idx}: No valid volume data")
                        continue
                    
                    # Create projection
                    sample_info = features_df.loc[idx]
                    bbox_name = sample_info.get('bbox_name', 'unknown')
                    var1 = sample_info.get('Var1', f'sample_{idx}')
                    
                    # Clean any problematic characters from filename
                    clean_var1 = str(var1).replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
                    
                    projection_filename = f"{bbox_name}_{clean_var1}_{idx}.png"
                    projection_path = projections_dir / projection_filename
                    
                    # Generate projection
                    projection_path, projections = create_gif_from_volume(volume, str(projection_path), fps=8, segmentation_type=config.segmentation_type)
                    
                    # Check if projection was successfully created
                    if os.path.exists(projection_path) and os.path.getsize(projection_path) > 0:
                        # Store full absolute path for HTML file - this is crucial for browser to find the projections
                        projection_paths[idx] = os.path.abspath(str(projection_path))
                        
                        # Store the individual projection data for later use when creating samples_with_projections
                        # Store in a dictionary with the sample index as the key
                        if 'projection_data_dict' not in locals():
                            projection_data_dict = {}
                        
                        # Ensure key is a string to avoid 'int' has no attribute 'items' error
                        str_idx = str(idx)
                        projection_data_dict[str_idx] = projections
                        
                        print(f"Created projection for sample {idx}")
                    else:
                        print(f"Failed to create projection for sample {idx} - file not created or empty")
                    
                except Exception as e:
                    print(f"Error creating projection for sample {idx}: {str(e)}")
            
            # Now create visualizations with our simpler methods if we have projections
            if projection_paths:                    
                print("\nCreating projected image visualization...")
                
                # Debug projection_data before passing it
                if 'projection_data_dict' in locals():
                    print(f"Debug: projection_data contains {len(projection_data_dict)} entries")
                    print(f"Debug: projection_data keys are of type: {type(next(iter(projection_data_dict)))}")
                    sample_key = next(iter(projection_data_dict))
                    print(f"Debug: Sample projection data for key {sample_key} has keys: {list(projection_data_dict[sample_key].keys())}")
                else:
                    print("Debug: projection_data variable not defined")
                
                try:
                    projected_path = create_projection_visualization(
                        features_df, projection_paths, output_dir, 
                        dim_reduction=args.dim_reduction, projection_data=projection_data_dict, dataset=dataset
                    )
                    print(f"Projected image visualization created at {projected_path}")
                    print(f"Open this in your browser to see projected images directly at their {args.dim_reduction.upper()} coordinates.")
                except Exception as e:
                    print(f"Error creating projected image visualization: {e}")
                
         
            else:
                print("No projections were created successfully. Skipping additional visualizations.")
        else:
            print("No valid indices found. Skipping projection creation and visualizations.")
    else:
        print("Warning: Could not initialize dataset. Skipping visualizations with projections.")
        print("If you want to create the visualization, please ensure your config has valid paths for:")
        print("- raw_base_dir: The directory containing raw volumes")
        print("- seg_base_dir: The directory containing segmentation volumes")
        print("- add_mask_base_dir: The directory containing additional mask volumes (optional)")
        print("- excel_file: The directory containing Excel files with synapse data")