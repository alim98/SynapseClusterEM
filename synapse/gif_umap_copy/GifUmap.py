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
from PIL import Image, ImageSequence
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

def ensure_gif_autoplay(gif_paths, loop=0):
    """
    Ensures all GIFs are set to autoplay by modifying their loop parameter.
    
    Args:
        gif_paths: Dictionary mapping sample indices to GIF paths
        loop: Loop parameter (0 = infinite, -1 = no loop, n = number of loops)
    
    Returns:
        Dictionary with paths to modified GIFs
    """
    from PIL import Image, ImageSequence
    import os
    
    modified_gif_paths = {}
    
    for idx, path in gif_paths.items():
        try:
            # Open the original GIF
            img = Image.open(path)
            
            # Create a new file path for the modified GIF
            dir_path = os.path.dirname(path)
            file_name = os.path.basename(path)
            new_path = os.path.join(dir_path, f"autoloop_{file_name}")
            
            # Extract frames
            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            
            # Save with the loop parameter
            frames[0].save(
                new_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                loop=loop,  # 0 means infinite loop
                duration=img.info.get('duration', 100)  # Use original duration or default to 100ms
            )
            
            # Store the new path
            modified_gif_paths[idx] = new_path
            print(f"Modified GIF for sample {idx} to auto-loop")
            
        except Exception as e:
            print(f"Error modifying GIF for sample {idx}: {e}")
            # Keep the original path if modification fails
            modified_gif_paths[idx] = path
            
    return modified_gif_paths

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

        from newdl.dataset2 import SynapseDataset
        from newdl.dataloader2 import SynapseDataLoader, Synapse3DProcessor

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
        
        # Create dataset
        dataset = SynapseDataset(
            vol_data_dict=vol_data_dict,
            synapse_df=syn_df,
            processor=processor,
            segmentation_type=config.segmentation_type,
            subvol_size=config.subvol_size,
            num_frames=config.num_frames,
            alpha=0.5
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

def create_gif_from_volume(volume, output_path, fps=10):
    """
    Create a GIF from a volume (3D array) and return the frames.
    
    Args:
        volume: 3D array representing volume data
        output_path: Path to save the GIF
        fps: Frames per second
        
    Returns:
        Tuple of (output_path, frames) where frames is a list of normalized frame data
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
    
    # Apply min-max normalization to enhance contrast
    enhanced_frames = []
    
    # Calculate global min/max for consistent scaling
    vol_min, vol_max = volume.min(), volume.max()
    scale_factor = vol_max - vol_min
    
    if scale_factor > 0:  # Avoid division by zero
        for i in range(volume.shape[0]):
            frame = volume[i]
            # Normalize using global min/max
            normalized = (frame - vol_min) / scale_factor
            enhanced_frames.append((normalized * 255).astype(np.uint8))
    else:
        # If all values are the same, create blank frames
        for i in range(volume.shape[0]):
            enhanced_frames.append(np.zeros_like(volume[i], dtype=np.uint8))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GIF with reduced size (lower fps and scale)
    imageio.mimsave(output_path, enhanced_frames, fps=fps)
    
    # Convert frames to base64-encoded PNGs for web display
    frame_data = []
    for frame in enhanced_frames:
        # Convert frame to PNG and then to base64
        with io.BytesIO() as output:
            Image.fromarray(frame).save(output, format="PNG")
            frame_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
            frame_data.append(frame_base64)
    
    return output_path, frame_data


def create_animated_gif_visualization(features_df, gif_paths, output_dir, dim_reduction='umap', frame_data=None):
    """
    Create a simple HTML page that displays animated GIFs directly at their coordinates.
    The GIFs are embedded directly in the HTML as base64 data to avoid file:// protocol issues.
    GIFs are made draggable so users can rearrange them.
    Includes a control to adjust how many GIFs are displayed at runtime.
    
    Args:
        features_df: DataFrame with features and coordinates
        gif_paths: Dictionary mapping sample indices to GIF paths
        output_dir: Directory to save the HTML file
        dim_reduction: Dimensionality reduction method ('umap' or 'tsne')
        frame_data: Dictionary mapping sample indices to lists of frame data (base64 encoded images)
    
    Returns:
        Path to the HTML file
    """
    method_name = "UMAP" if dim_reduction == 'umap' else "t-SNE"
    import base64
    
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
    
    # Extract coordinates and other info for samples with GIFs
    samples_with_gifs = []
    
    # Track how many GIFs we have from each cluster for reporting
    cluster_counts = {}
    
    for idx in gif_paths:
        if idx in features_df.index:
            sample = features_df.loc[idx]
            
            # Use the determined coordinate columns
            if x_col in sample and y_col in sample:
                x, y = sample[x_col], sample[y_col]
                
                # Extract cluster and bbox information if available
                cluster = sample.get('cluster', 'N/A') if 'cluster' in sample else 'N/A'
                bbox = sample.get('bbox_name', 'unknown') if 'bbox_name' in sample else 'unknown'
                
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
                
                # Load the GIF file and convert to base64
                try:
                    with open(gif_paths[idx], 'rb') as gif_file:
                        gif_data = gif_file.read()
                        encoded_gif = base64.b64encode(gif_data).decode('utf-8')
                        
                        # Add frame data if available
                        frames = []
                        if frame_data and idx in frame_data:
                            frames = frame_data[idx]
                        
                        samples_with_gifs.append({
                            'id': idx,
                            'x': x,
                            'y': y,
                            'cluster': cluster,
                            'bbox': bbox,
                            'gifData': encoded_gif,
                            'frames': frames
                        })
                except Exception as e:
                    print(f"Error encoding GIF for sample {idx}: {e}")
            else:
                print(f"Warning: Sample {idx} does not have required coordinates ({x_col}, {y_col}). Skipping.")
    
    # Print distribution of GIFs across clusters
    if cluster_counts:
        print("\nDistribution of GIFs across clusters:")
        for cluster, count in sorted(cluster_counts.items()):
            print(f"  Cluster {cluster}: {count} GIFs")
    
    if not samples_with_gifs:
        raise ValueError("No samples with GIFs and valid coordinates found. Cannot create visualization.")
    
    print(f"\nTotal samples with GIFs and valid coordinates: {len(samples_with_gifs)}")
    print(f"First sample for debugging: {json.dumps(samples_with_gifs[0], default=str)[:200]}...")
    
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
    gif_size = 50  # Default size decreased from 100 to 50px
    shift_limit = 50  # Fixed 50px shift limit (not percentage-based)
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
        half_size = gif_size / 2
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
                
                # Skip if this would move the GIF out of bounds
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
        
        # Try increasing distances
        for distance in range(1, int(shift_limit) + 1):
            for dir_x, dir_y in directions:
                shifted_x = baseX + (dir_x * distance)
                shifted_y = baseY + (dir_y * distance)
                
                # Skip if this would move the GIF out of bounds
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
        
        # If we can't find a non-overlapping position, return null
        return None
    
    # Initialize originalPositions dictionary for storing initial GIF positions
    # Note: we need to map the raw coordinates to plot coordinates
    # For this we use the same mapping logic as in the JavaScript mapToPlot function
    orig_positions_dict = {}
    samples_to_remove = []
    
    for i, sample in enumerate(samples_with_gifs):
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
            del samples_with_gifs[i]
        print(f"Removed {len(samples_to_remove)} samples that couldn't be placed without overlap")
    
    # Convert to JSON string for embedding in JavaScript
    originalPositions = json.dumps(orig_positions_dict)
    print(f"originalPositions JSON string length: {len(originalPositions)}")
    print(f"Sample of originalPositions: {originalPositions[:100]}...")
    
    # Determine colors for points based on clusters or bboxes
    point_colors = {}
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
            
            # Determine color based on cluster or bbox
            if 'cluster' in row:
                cluster = row['cluster']
                if hasattr(cluster, 'item'):
                    cluster = cluster.item()
                color = point_colors.get(cluster, 'rgb(100, 100, 100)')
            else:
                color = 'rgb(100, 100, 100)'
            
            # Add this point to the samples array - make sure we have a valid number before adding
            if not (np.isnan(x) or np.isnan(y)):
                points_content += f"""
                {{
                    "id": {idx},
                    "x": {x},
                    "y": {y},
                    "color": "{color}",
                    "hasGif": {str(idx in gif_paths).lower()}
                }},"""
    
    # Count how many valid points we have
    print(f"Generated points_content with {points_content.count('id:')} points")
    print(f"Sample of points_content: {points_content[:200]}...")
    
    # Generate HTML content for GIFs
    gifs_content = ""
    for sample in samples_with_gifs:
        # Ensure all values are properly stringified
        sample_id = sample['id']
        if hasattr(sample_id, 'item'):
            sample_id = sample_id.item()
            
        sample_x = float(sample['x'])
        sample_y = float(sample['y'])
        sample_cluster = str(sample['cluster'])
        sample_bbox = str(sample['bbox'])
        
        # Set a frames flag - store them separately to avoid huge HTML content
        has_frames = 'frames' in sample and len(sample['frames']) > 0
        
        gifs_content += f"""
                {{
                    "id": {sample_id},
                    "x": {sample_x},
                    "y": {sample_y},
                    "cluster": "{sample_cluster}",
                    "bbox": "{sample_bbox}",
                    "gifData": "{sample['gifData']}",
                    "hasFrames": {str(has_frames).lower()}
                }},"""
    
    # Count how many valid GIFs we have
    print(f"Generated gifs_content with {gifs_content.count('id:')} GIFs")
    print(f"Sample of gifs_content (without actual base64 data): {gifs_content[:200]}...")
    
    # Create a dedicated frames content structure
    frames_content = "{"
    has_any_frames = False
    
    # Check if we have frame data
    if frame_data:
        for idx, frames in frame_data.items():
            if frames:
                has_any_frames = True
                # Stringify the ID
                str_id = str(idx)
                if hasattr(idx, 'item'):
                    str_id = str(idx.item())
                
                # Add frames data for this sample as JSON array
                frames_content += f'"{str_id}": ['
                for frame in frames:
                    frames_content += f'"{frame}",'
                # Remove trailing comma if there are frames
                if frames:
                    frames_content = frames_content[:-1]
                frames_content += "],"
    
    # Remove trailing comma if any frames were added
    if frames_content.endswith(","):
        frames_content = frames_content[:-1]
    
    frames_content += "}"
    
    # If we have no frames, initialize a valid empty object
    if not has_any_frames:
        frames_content = "{}"
        
    print(f"Generated frames_content with data for {frame_data.keys() if frame_data else 0} GIFs")
    print(f"Has any frames: {has_any_frames}")
    print(f"frames_content length: {len(frames_content)}")
    
    # Create a simple HTML page with SVG for plotting points and GIFs
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Animated GIFs at {method_name} Coordinates</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1800px; /* Increased from 1200px */
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 20px;
                border-radius: 8px;
            }}
            h1 {{
                text-align: center;
                color: #333;
            }}
            .plot-container {{
                position: relative;
                margin: 20px auto;
                border: 1px solid #ddd;
                background-color: #fff;
                overflow: hidden;
                width: 1600px; /* Increased from 1000px */
                height: 1200px; /* Increased from 800px */
            }}
            .point {{
                position: absolute;
                width: 6px;
                height: 6px;
                border-radius: 50%;
                transform: translate(-50%, -50%);
            }}
            .gif-container {{
                position: absolute;
                border: 2px solid #333;
                background-color: white;
                border-radius: 4px;
                overflow: hidden;
                transform: translate(-50%, -50%);
                z-index: 10;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                cursor: move; /* Show move cursor to indicate draggability */
            }}
            .gif-container img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}
            .controls {{
                margin-top: 10px;
                text-align: center;
                padding: 15px;
                background-color: #f8f8f8;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .controls button {{
                padding: 8px 15px;
                margin: 0 5px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }}
            .controls button:hover {{
                background-color: #45a049;
            }}
            .gif-size-slider {{
                width: 200px;
                margin: 0 10px;
                vertical-align: middle;
            }}
            .control-group {{
                display: inline-block;
                margin: 0 15px;
                vertical-align: middle;
            }}
            .control-label {{
                font-weight: bold;
                margin-right: 10px;
            }}
            .dragging {{
                opacity: 0.8;
                z-index: 1000;
            }}
            .cluster-filter {{
                margin-top: 10px;
                text-align: center;
            }}
            .cluster-checkbox {{
                margin-right: 5px;
            }}
            .cluster-label {{
                margin-right: 15px;
                user-select: none;
            }}
            #debug-message {{
                padding: 10px;
                margin: 10px 0;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .frame-slider {{
                width: 300px;
                margin: 0 10px;
                vertical-align: middle;
            }}  
            .frame-control {{
                margin-top: 10px;
                padding: 8px;
                background-color: #e8f7e8;
                border-radius: 4px;
                border: 1px solid #4CAF50;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{method_name} Visualization with Animated GIFs</h1>
            
            <div id="debug-message"></div>
            
            <div class="controls">
                <div class="control-group">
                    <button id="toggle-gifs">Show/Hide GIFs</button>
                </div>
                
                <div class="control-group">
                    <span class="control-label">GIF Size:</span>
                    <input type="range" min="20" max="200" value="50" id="gif-size-slider" class="gif-size-slider">
                    <span id="size-value">50px</span>
                    <button id="resize-gifs">Apply Size</button>
                </div>
                
                <div class="control-group">
                    <span class="control-label">Number of GIFs:</span>
                    <input type="range" min="1" max="{len(samples_with_gifs)}" value="{len(samples_with_gifs)}" id="num-gifs-slider" class="gif-size-slider">
                    <span id="num-gifs-value">{len(samples_with_gifs)}</span>
                    <button id="apply-num-gifs">Apply</button>
                </div>
                
                <div class="control-group">
                    <button id="reset-positions">Reset Positions</button>
                </div>
                
                <div class="control-group frame-control">
                    <span class="control-label">Frame Control:</span>
                    <input type="range" min="0" max="80" value="0" id="frame-slider" class="frame-slider">
                    <span id="frame-value">0</span>
                    <button id="play-pause">Play/Pause</button>
                </div>
            </div>
            
            <div class="cluster-filter" id="cluster-filter">
                <span class="control-label">Filter by Cluster:</span>
                <!-- Cluster checkboxes will be added here by JavaScript -->
            </div>
            
            <div class="plot-container" id="plot">
                <!-- Background points will be added here -->
                <!-- GIFs will be added here -->
            </div>
        </div>
        
        <script>
            // Initialize debug element
            const debugMessage = document.getElementById('debug-message');
            function log(message) {{
                debugMessage.innerHTML += message + '<br>';
                console.log(message);
            }}
            
            log("Script started");
            
            // Define the UMAP bounds
            const xMin = {x_min};
            const xMax = {x_max};
            const yMin = {y_min};
            const yMax = {y_max};
            
            log(`Coordinate bounds: X: ${{xMin}} to ${{xMax}}, Y: ${{yMin}} to ${{yMax}}`);
            
            // Function to map UMAP coordinates to plot coordinates
            function mapToPlot(x, y, width, height) {{
                const plotX = ((x - xMin) / (xMax - xMin)) * width;
                // Invert y-axis (UMAP coordinates increase upward, plot coordinates increase downward)
                const plotY = height - ((y - yMin) / (yMax - yMin)) * height;
                return [plotX, plotY];
            }}
            
            // Get the plot container
            const plot = document.getElementById('plot');
            const plotWidth = plot.clientWidth;
            const plotHeight = plot.clientHeight;
            
            log(`Plot dimensions: ${{plotWidth}}x${{plotHeight}}`);
            
            // Store original positions of GIFs for reset functionality
            let originalPositions = {{}};
            try {{
                originalPositions = {originalPositions};
                log(`Loaded original positions data with ${{Object.keys(originalPositions).length}} items`);
            }} catch (e) {{
                log(`Error loading original positions: ${{e.message}}`);
            }}
            
            // Store all GIF data with cluster information
            const allGifData = [];
            
            // Store frames data for all GIFs
            const allFramesData = {{}};
            try {{
                // Parse frames data string to JavaScript object
                const framesStr = `{frames_content}`;
                log(`Loading frames data: ${{framesStr.substring(0, 100)}}...`);
                
                // Check if we have valid JSON
                if (framesStr && framesStr !== "{{}}" && framesStr.length > 2) {{
                    const framesData = JSON.parse(framesStr);
                    Object.assign(allFramesData, framesData);
                    log(`Loaded frames data for ${{Object.keys(allFramesData).length}} GIFs`);
                }} else {{
                    log("No frames data available");
                }}
            }} catch(e) {{
                log(`Error loading frames data: ${{e.message}}`);
            }}
            
            // Variable to track play/pause state
            let isPlaying = false;
            let animationInterval = null;
            let currentFrame = 0;
            
            // Function to update the frame display for all GIFs
            function updateFrameDisplay(frameIndex) {{
                // Update all visible gif containers
                document.querySelectorAll('.gif-container.has-frames').forEach(container => {{
                    const id = container.dataset.id;
                    if (id && allFramesData[id] && allFramesData[id].length > 0) {{
                        // Make sure we don't exceed the frame count for this GIF
                        const actualFrame = Math.min(frameIndex, allFramesData[id].length - 1);
                        const frameImg = container.querySelector('img');
                        if (frameImg) {{
                            frameImg.src = `data:image/png;base64,${{allFramesData[id][actualFrame]}}`;
                        }}
                    }}
                }});
                
                // Update the frame value display
                document.getElementById('frame-value').textContent = frameIndex;
            }}
            
            // Function to handle playing animation
            function playFrames() {{
                if (animationInterval) {{
                    clearInterval(animationInterval);
                }}
                
                animationInterval = setInterval(() => {{
                    // Get the max frame count among all GIFs
                    let maxFrames = 80; // Default value
                    Object.values(allFramesData).forEach(frames => {{
                        if (frames && frames.length > maxFrames) {{
                            maxFrames = frames.length;
                        }}
                    }});
                    
                    // Update frame slider max value
                    const frameSlider = document.getElementById('frame-slider');
                    if (frameSlider) {{
                        frameSlider.max = maxFrames - 1;
                    
                        // Increment current frame and loop if needed
                        currentFrame = (currentFrame + 1) % maxFrames;
                    
                        // Update frame slider position
                        frameSlider.value = currentFrame;
                    }}
                    
                    // Update all GIFs to show this frame
                    updateFrameDisplay(currentFrame);
                }}, 100); // 10 FPS
            }}
            
            // Function to stop playing animation
            function pauseFrames() {{
                if (animationInterval) {{
                    clearInterval(animationInterval);
                    animationInterval = null;
                }}
            }}
            
            // Add background points for all samples
            function addBackgroundPoints() {{
                log("Adding background points...");
                
                // Samples from features_df
                let samples = [];
                try {{
                    samples = [
                        {points_content}
                    ];
                    log(`Found ${{samples.length}} background points to add`);
                }} catch (e) {{
                    log(`Error loading background points: ${{e.message}}`);
                    return;
                }}
                
                // Add points to the plot
                let pointsAdded = 0;
                samples.forEach(sample => {{
                    try {{
                        const [plotX, plotY] = mapToPlot(sample.x, sample.y, plotWidth, plotHeight);
                        
                        const pointElem = document.createElement('div');
                        pointElem.className = 'point';
                        pointElem.style.left = `${{plotX}}px`;
                        pointElem.style.top = `${{plotY}}px`;
                        pointElem.style.backgroundColor = sample.color;
                        
                        // Make points with GIFs larger
                        if (sample.hasGif) {{
                            pointElem.style.width = '10px';
                            pointElem.style.height = '10px';
                            pointElem.style.border = '2px solid black';
                            pointElem.style.zIndex = '5';
                        }}
                        
                        plot.appendChild(pointElem);
                        pointsAdded++;
                    }} catch (e) {{
                        log(`Error adding point: ${{e.message}}`);
                    }}
                }});
                
                log(`Successfully added ${{pointsAdded}} background points`);
            }}
            
            // Add GIFs to the plot
            function addGifs() {{
                log("Adding GIFs...");
                
                // Samples with GIFs
                let samplesWithGifs = [];
                try {{
                    samplesWithGifs = [
                        {gifs_content}
                    ];
                    log(`Found ${{samplesWithGifs.length}} GIFs to add`);
                }} catch (e) {{
                    log(`Error loading GIFs: ${{e.message}}`);
                    return;
                }}
                
                // Store all GIF data for filtering
                allGifData.push(...samplesWithGifs);
                
                // Extract frames data for all GIFs
                samplesWithGifs.forEach(sample => {{
                    if (sample.hasFrames) {{
                        // The frames data is already loaded via the framesData object
                        log(`Found GIF ${{sample.id}} with frames`);
                    }}
                }});
                
                // Tracking placed rectangles to prevent overlap
                const placedRectangles = [];
                const gifSize = 50; // Default size decreased from 100 to 50px
                const shiftLimit = 50; // Fixed 50px shift limit (not percentage-based)
                const maxShiftX = shiftLimit;
                const maxShiftY = shiftLimit;
                
                // Function to check if two rectangles overlap
                function doRectanglesOverlap(rect1, rect2) {{
                    return !(rect1.right < rect2.left || 
                             rect1.left > rect2.right || 
                             rect1.bottom < rect2.top || 
                             rect1.top > rect2.bottom);
                }}
                
                // Function to find a position with no overlap
                function findNonOverlappingPosition(baseX, baseY, existingRects) {{
                    // Check if the original position works
                    const halfSize = gifSize / 2;
                    let rect = {{
                        left: baseX - halfSize,
                        right: baseX + halfSize,
                        top: baseY - halfSize,
                        bottom: baseY + halfSize
                    }};
                    
                    // Check if original position has no overlap
                    let hasOverlap = false;
                    let overlapRect = null;
                    
                    for (const existingRect of existingRects) {{
                        if (doRectanglesOverlap(rect, existingRect)) {{
                            hasOverlap = true;
                            overlapRect = existingRect;
                            break;
                        }}
                    }}
                    
                    // If no overlap, return original position
                    if (!hasOverlap) {{
                        return {{ x: baseX, y: baseY, rect: rect }};
                    }}
                    
                    // Calculate the minimum shift needed in each direction to avoid overlap
                    if (overlapRect) {{
                        // Calculate overlap amounts in each direction
                        const overlapRight = rect.right - overlapRect.left;
                        const overlapLeft = overlapRect.right - rect.left;
                        const overlapBottom = rect.bottom - overlapRect.top;
                        const overlapTop = overlapRect.bottom - rect.top;
                        
                        // Find the smallest shift needed
                        const shifts = [
                            {{ axis: 'x', amount: overlapRight, direction: 1 }},  // shift right
                            {{ axis: 'x', amount: -overlapLeft, direction: -1 }}, // shift left
                            {{ axis: 'y', amount: overlapBottom, direction: 1 }},  // shift down
                            {{ axis: 'y', amount: -overlapTop, direction: -1 }}   // shift up
                        ];
                        
                        // Sort by absolute amount to find smallest shift
                        shifts.sort((a, b) => Math.abs(a.amount) - Math.abs(b.amount));
                        
                        // Try each shift until we find one that works
                        for (const shift of shifts) {{
                            // Skip if shift is too large
                            if (Math.abs(shift.amount) > shiftLimit) {{
                                continue;
                            }}
                            
                            let shiftedX = baseX;
                            let shiftedY = baseY;
                            
                            if (shift.axis === 'x') {{
                                shiftedX += shift.amount;
                            }} else {{
                                shiftedY += shift.amount;
                            }}
                            
                            // Skip if this would move the GIF out of bounds
                            if (shiftedX - halfSize < 0 || shiftedX + halfSize > plotWidth ||
                                shiftedY - halfSize < 0 || shiftedY + halfSize > plotHeight) {{
                                continue;
                            }}
                            
                            // Check if this position works with all existing rectangles
                            const shiftedRect = {{
                                left: shiftedX - halfSize,
                                right: shiftedX + halfSize,
                                top: shiftedY - halfSize,
                                bottom: shiftedY + halfSize
                            }};
                            
                            let shiftedOverlap = false;
                            for (const existingRect of existingRects) {{
                                if (doRectanglesOverlap(shiftedRect, existingRect)) {{
                                    shiftedOverlap = true;
                                    break;
                                }}
                            }}
                            
                            if (!shiftedOverlap) {{
                                return {{ x: shiftedX, y: shiftedY, rect: shiftedRect }};
                            }}
                        }}
                    }}
                    
                    // If the simple shifts didn't work, try a more general approach
                    // Try cardinal and diagonal directions with increasing distances
                    const directions = [
                        {{ x: 1, y: 0 }},  // right
                        {{ x: 0, y: 1 }},  // down
                        {{ x: -1, y: 0 }}, // left
                        {{ x: 0, y: -1 }}, // up
                        {{ x: 1, y: 1 }},  // down-right
                        {{ x: 1, y: -1 }}, // up-right
                        {{ x: -1, y: 1 }}, // down-left
                        {{ x: -1, y: -1 }} // up-left
                    ];
                    
                    // Try increasing distances
                    for (let distance = 1; distance <= shiftLimit; distance += 1) {{
                        for (const dir of directions) {{
                            const shiftedX = baseX + (dir.x * distance);
                            const shiftedY = baseY + (dir.y * distance);
                            
                            // Skip if this would move the GIF out of bounds
                            if (shiftedX - halfSize < 0 || shiftedX + halfSize > plotWidth ||
                                shiftedY - halfSize < 0 || shiftedY + halfSize > plotHeight) {{
                                continue;
                            }}
                            
                            // Check this position
                            const shiftedRect = {{
                                left: shiftedX - halfSize,
                                right: shiftedX + halfSize,
                                top: shiftedY - halfSize,
                                bottom: shiftedY + halfSize
                            }};
                            
                            let shiftedOverlap = false;
                            for (const existingRect of existingRects) {{
                                if (doRectanglesOverlap(shiftedRect, existingRect)) {{
                                    shiftedOverlap = true;
                                    break;
                                }}
                            }}
                            
                            if (!shiftedOverlap) {{
                                return {{ x: shiftedX, y: shiftedY, rect: shiftedRect }};
                            }}
                        }}
                    }}
                    
                    // If we can't find a non-overlapping position, return null
                    return null;
                }}
                
                // Add GIFs to the plot
                let gifsAdded = 0;
                let gifsSkipped = 0;
                
                samplesWithGifs.forEach((sample, index) => {{
                    try {{
                        const [plotX, plotY] = mapToPlot(sample.x, sample.y, plotWidth, plotHeight);
                        
                        // Find a non-overlapping position
                        const position = findNonOverlappingPosition(plotX, plotY, placedRectangles);
                        
                        // Skip this GIF if no non-overlapping position found
                        if (!position) {{
                            log(`Skipping GIF ${{sample.id}} due to overlap that couldn't be resolved`);
                            gifsSkipped++;
                            return;
                        }}
                        
                        // Add this rectangle to our tracking
                        placedRectangles.push(position.rect);
                        
                        // Create GIF container
                        const gifContainer = document.createElement('div');
                        gifContainer.className = 'gif-container';
                        if (sample.hasFrames) {{
                            gifContainer.classList.add('has-frames');
                        }}
                        gifContainer.id = `gif-${{sample.id}}`;
                        gifContainer.dataset.id = sample.id;
                        gifContainer.style.left = `${{position.x}}px`;
                        gifContainer.style.top = `${{position.y}}px`;
                        gifContainer.style.width = `${{gifSize}}px`;
                        gifContainer.style.height = `${{gifSize}}px`;
                        gifContainer.dataset.index = index;
                        gifContainer.dataset.cluster = sample.cluster;
                        
                        // If this position was shifted, add a marker to indicate that
                        if (position.x !== plotX || position.y !== plotY) {{
                            // Add small visual indicator that this was shifted
                            gifContainer.style.border = '2px dashed #e74c3c';
                        }}
                        
                        // Create GIF image using base64 data
                        const gifImg = document.createElement('img');
                        gifImg.src = `data:image/gif;base64,${{sample.gifData}}`;
                        gifImg.alt = `Sample ${{sample.id}}`;
                        gifImg.setAttribute('loop', 'infinite');
                        
                        // Add to container
                        gifContainer.appendChild(gifImg);
                        
                        // Make the GIF container draggable
                        makeDraggable(gifContainer);
                        
                        // Add to plot
                        plot.appendChild(gifContainer);
                        gifsAdded++;
                    }} catch (e) {{
                        log(`Error adding GIF ${{index}}: ${{e.message}}`);
                    }}
                }});
                
                log(`Successfully added ${{gifsAdded}} GIFs, skipped ${{gifsSkipped}} due to overlaps`);
                
                // Store all gif containers for later use
                window.gifContainers = document.querySelectorAll('.gif-container');
                
                // Create cluster filter checkboxes
                createClusterFilter();
            }}
            
            // Create cluster filter checkboxes
            function createClusterFilter() {{
                const clusterFilter = document.getElementById('cluster-filter');
                const clusters = new Set();
                
                // Get unique clusters
                allGifData.forEach(sample => {{
                    if (sample['cluster'] !== 'N/A') {{
                        clusters.add(sample['cluster']);
                    }}
                }});
                
                // Create a checkbox for each cluster
                clusters.forEach(cluster => {{
                    const label = document.createElement('label');
                    label.className = 'cluster-label';
                    
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.className = 'cluster-checkbox';
                    checkbox.value = cluster;
                    checkbox.checked = true;
                    checkbox.id = `cluster-${{cluster}}`;
                    
                    checkbox.addEventListener('change', updateVisibleGifs);
                    
                    label.appendChild(checkbox);
                    label.appendChild(document.createTextNode(`Cluster ${{cluster}}`));
                    
                    clusterFilter.appendChild(label);
                }});
                
                // Add "Select All" and "Deselect All" buttons
                const selectAllBtn = document.createElement('button');
                selectAllBtn.textContent = 'Select All';
                selectAllBtn.addEventListener('click', () => {{
                    document.querySelectorAll('.cluster-checkbox').forEach(cb => {{
                        cb.checked = true;
                    }});
                    updateVisibleGifs();
                }});
                
                const deselectAllBtn = document.createElement('button');
                deselectAllBtn.textContent = 'Deselect All';
                deselectAllBtn.addEventListener('click', () => {{
                    document.querySelectorAll('.cluster-checkbox').forEach(cb => {{
                        cb.checked = false;
                    }});
                    updateVisibleGifs();
                }});
                
                clusterFilter.appendChild(document.createElement('br'));
                clusterFilter.appendChild(selectAllBtn);
                clusterFilter.appendChild(deselectAllBtn);
            }}
            
            // Update visible GIFs based on filters
            function updateVisibleGifs() {{
                // Get selected clusters
                const selectedClusters = [];
                document.querySelectorAll('.cluster-checkbox:checked').forEach(cb => {{
                    selectedClusters.push(cb.value);
                }});
                
                // Get max number of GIFs to show
                const maxGifs = parseInt(document.getElementById('num-gifs-slider').value);
                
                // Count how many GIFs we've shown
                let shownCount = 0;
                
                // Update visibility of GIF containers
                window.gifContainers.forEach(container => {{
                    const cluster = container.dataset.cluster;
                    const index = parseInt(container.dataset.index);
                    
                    // Show if cluster is selected and we haven't reached the limit
                    if (selectedClusters.includes(cluster) && shownCount < maxGifs) {{
                        container.style.display = 'block';
                        shownCount++;
                    }} else {{
                        container.style.display = 'none';
                    }}
                }});
            }}
            
            // Make an element draggable
            function makeDraggable(element) {{
                let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
                
                element.onmousedown = dragMouseDown;
                
                function dragMouseDown(e) {{
                    e = e || window.event;
                    e.preventDefault();
                    // Get the mouse cursor position at startup
                    pos3 = e.clientX;
                    pos4 = e.clientY;
                    
                    // Add dragging class
                    element.classList.add('dragging');
                    
                    // Bring the element to the front
                    element.style.zIndex = "1000";
                    
                    document.onmouseup = closeDragElement;
                    document.onmousemove = elementDrag;
                }}
                
                function elementDrag(e) {{
                    e = e || window.event;
                    e.preventDefault();
                    
                    // Calculate the new cursor position
                    pos1 = pos3 - e.clientX;
                    pos2 = pos4 - e.clientY;
                    pos3 = e.clientX;
                    pos4 = e.clientY;
                    
                    // Set the element's new position
                    const newTop = (element.offsetTop - pos2);
                    const newLeft = (element.offsetLeft - pos1);
                    
                    // Constrain to plot boundaries
                    const elemWidth = parseInt(element.style.width);
                    const elemHeight = parseInt(element.style.height);
                    
                    const boundedTop = Math.max(elemHeight/2, Math.min(newTop, plotHeight - elemHeight/2));
                    const boundedLeft = Math.max(elemWidth/2, Math.min(newLeft, plotWidth - elemWidth/2));
                    
                    element.style.top = boundedTop + "px";
                    element.style.left = boundedLeft + "px";
                }}
                
                function closeDragElement() {{
                    // Stop moving when mouse button is released
                    document.onmouseup = null;
                    document.onmousemove = null;
                    
                    // Remove dragging class
                    element.classList.remove('dragging');
                    
                    // Reset z-index to normal
                    setTimeout(() => {{
                        element.style.zIndex = "10";
                    }}, 200);
                }}
            }}
            
            // Initialize the visualization
            function init() {{
                // Add background points
                addBackgroundPoints();
                
                // Add GIFs
                addGifs();
                
                // Set up toggle button
                const toggleButton = document.getElementById('toggle-gifs');
                toggleButton.addEventListener('click', () => {{
                    window.gifContainers.forEach(container => {{
                        if (container.style.display === 'none') {{
                            container.style.display = 'block';
                        }} else {{
                            container.style.display = 'none';
                        }}
                    }});
                }});
                
                // Set up resize functionality
                const sizeSlider = document.getElementById('gif-size-slider');
                const sizeValue = document.getElementById('size-value');
                
                sizeSlider.addEventListener('input', () => {{
                    const size = sizeSlider.value;
                    sizeValue.textContent = size + 'px';
                }});
                
                // Set up resize button
                const resizeButton = document.getElementById('resize-gifs');
                resizeButton.addEventListener('click', () => {{
                    const size = sizeSlider.value;
                    
                    window.gifContainers.forEach(container => {{
                        container.style.width = size + 'px';
                        container.style.height = size + 'px';
                    }});
                }});
                
                // Set up number of GIFs slider
                const numGifsSlider = document.getElementById('num-gifs-slider');
                const numGifsValue = document.getElementById('num-gifs-value');
                
                numGifsSlider.addEventListener('input', () => {{
                    const num = numGifsSlider.value;
                    numGifsValue.textContent = num;
                }});
                
                // Set up apply number of GIFs button
                const applyNumGifsButton = document.getElementById('apply-num-gifs');
                applyNumGifsButton.addEventListener('click', updateVisibleGifs);
                
                // Set up reset positions button
                const resetButton = document.getElementById('reset-positions');
                resetButton.addEventListener('click', () => {{
                    window.gifContainers.forEach(container => {{
                        const id = container.id.replace('gif-', '');
                        // Use safe property access with proper syntax
                        if (originalPositions[id]) {{
                            const pos = originalPositions[id];
                            container.style.left = `${{pos.x}}px`;
                            container.style.top = `${{pos.y}}px`;
                        }}
                    }});
                }});
                
                // Setup frame slider
                const frameSlider = document.getElementById('frame-slider');
                const frameValue = document.getElementById('frame-value');
                
                // Find the maximum number of frames across all GIFs
                let maxFrames = 80; // Default value
                Object.values(allFramesData).forEach(frames => {{
                    if (frames && frames.length > maxFrames) {{
                        maxFrames = frames.length;
                    }}
                }});
                
                // Set the slider max value
                frameSlider.max = maxFrames - 1;
                
                frameSlider.addEventListener('input', () => {{
                    // Update the displayed value
                    currentFrame = parseInt(frameSlider.value);
                    frameValue.textContent = currentFrame;
                    
                    // Pause any animation that's playing
                    if (isPlaying) {{
                        pauseFrames();
                        isPlaying = false;
                    }}
                    
                    // Update all GIFs to display the selected frame
                    updateFrameDisplay(currentFrame);
                }});
                
                // Setup play/pause button
                const playPauseButton = document.getElementById('play-pause');
                playPauseButton.addEventListener('click', () => {{
                    isPlaying = !isPlaying;
                    
                    if (isPlaying) {{
                        playFrames();
                        playPauseButton.textContent = 'Pause';
                    }} else {{
                        pauseFrames();
                        playPauseButton.textContent = 'Play';
                    }}
                }});
                
                // Initial update of visible GIFs
                updateVisibleGifs();
            }}
            
            // Run initialization
            window.onload = init;
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    html_path = output_dir / f"animated_gifs_{dim_reduction}_visualization.html"
    try:
        # Add debug statements to narrow down the error
        print("Types of data:")
        print("- originalPositions type:", type(originalPositions))
        print("- frames_content type:", type(frames_content))
        print("- frames_content length:", len(frames_content))
        print("- First item in samples_with_gifs:", samples_with_gifs[0] if samples_with_gifs else "No samples")
        
        # Check if the points_content and gifs_content are empty 
        if not points_content.strip():
            print("WARNING: points_content is empty! No background points will be shown.")
        
        if not gifs_content.strip():
            print("WARNING: gifs_content is empty! No GIFs will be shown.")
            
        # Write the HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Created animated GIF visualization with embedded data: {html_path}")
        
        # Print a sample of the HTML content to verify it has data
        html_sample = html_content[0:1000]  # Get first 1000 chars
        print(f"Sample of HTML content: {html_sample}")
        
        # Check HTML file size
        html_size = os.path.getsize(html_path)
        print(f"HTML file size: {html_size} bytes")
        
        if html_size < 10000:  # If file is too small, something might be wrong
            print("WARNING: HTML file is very small! It might not contain all necessary data.")
            
    except Exception as e:
        print(f"Error creating animated GIF visualization: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
    
    return html_path

if __name__ == "__main__":
    # Parse configuration
    config.parse_args()
    
    # Add argument for dimensionality reduction method
    import argparse
    parser = argparse.ArgumentParser(description='UMAP and t-SNE Visualization with GIFs')
    parser.add_argument('--dim-reduction', 
                       choices=['umap', 'tsne'], 
                       default='umap',
                       help='Dimensionality reduction method to use (umap or tsne)')
    parser.add_argument('--num-samples', 
                       type=int,
                       default=20,
                       help='Number of random samples to show with GIFs (default: 20)')
    args, unknown = parser.parse_known_args()
    
    # Define paths
    csv_path = r"C:\Users\alim9\Documents\codes\synapse2\results\run_2025-03-14_15-50-53\features_layer20_seg10_alpha1.0\features_layer20_seg10_alpha1_0.csv"  # Replace with your actual CSV path
    output_path = "results/test"  # Replace with your desired output directory
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory created/verified at: {output_dir}")
    
    # Load existing features_df if it exists to avoid re-running clustering
    clustered_features_path = output_dir / "clustered_features.csv"
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
    
    print(f"Loaded features DataFrame with {len(features_df)} samples")
    print(f"Columns in features_df: {features_df.columns.tolist()}")
    print(f"Using dimensionality reduction method: {args.dim_reduction.upper()}")
    
    # Create UMAP or t-SNE visualization with GIFs
    # Try to get dataset from different sources
    dataset = None
    
    # First try to import from Clustering
    try:
        from Clustering import dataset
        print("Using dataset from Clustering module")
    except ImportError:
        print("Dataset not available from Clustering module, trying to initialize from newdl...")
        dataset = initialize_dataset_from_newdl()
    
    # If we have a dataset, create the visualization
    if dataset is not None:
        # Create visualization with selected dim reduction method
        print(f"Creating {args.dim_reduction.upper()} visualization with GIFs for {args.num_samples} random samples...")
        
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
                print("Creating visualization without sample GIFs.")
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
            gifs_dir = output_dir / "sample_gifs"
            gifs_dir.mkdir(parents=True, exist_ok=True)
            
            # Select random samples, making sure they cover different clusters if possible
            np.random.seed(42)
            random_samples = []
            
            # Set number of samples to 40 as requested
            num_samples = 500  # User requested 40 samples
            
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
            
            print(f"Selected {len(random_samples)} samples for GIF creation")
            
            # Create GIFs for selected samples
            print(f"Creating GIFs for {len(random_samples)} samples...")
            gif_paths = {}
            
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
                    elif isinstance(sample_data, dict):
                        volume = sample_data.get('pixel_values', sample_data.get('raw_volume'))
                    else:
                        volume = sample_data
                    
                    # Skip if no volume data found or it's None/empty
                    if volume is None or (hasattr(volume, 'numel') and volume.numel() == 0) or \
                       (hasattr(volume, 'size') and np.prod(volume.shape) == 0):
                        print(f"Skipping sample {idx}: No valid volume data")
                        continue
                        
                    # Create GIF
                    sample_info = features_df.loc[idx]
                    bbox_name = sample_info.get('bbox_name', 'unknown')
                    var1 = sample_info.get('Var1', f'sample_{idx}')
                    
                    # Clean any problematic characters from filename
                    clean_var1 = str(var1).replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
                    
                    gif_filename = f"{bbox_name}_{clean_var1}_{idx}.gif"
                    gif_path = gifs_dir / gif_filename
                    
                    # Generate GIF with reduced quality to save space
                    gif_path, frames = create_gif_from_volume(volume, str(gif_path), fps=8)
                    
                    # Check if GIF was successfully created
                    if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                        # Store full absolute path for HTML file - this is crucial for browser to find the GIFs
                        gif_paths[idx] = os.path.abspath(str(gif_path))
                        # Store frame data for the global slider
                        if 'all_frames' not in locals():
                            all_frames = {}
                        all_frames[idx] = frames
                        print(f"Created GIF for sample {idx} with {len(frames)} frames")
                    else:
                        print(f"Failed to create GIF for sample {idx} - file not created or empty")
                    
                except Exception as e:
                    print(f"Error creating GIF for sample {idx}: {str(e)}")
            
            # Skip the original visualization that takes longer
            # html_path = create_umap_with_gifs(features_df, dataset, output_path, num_samples=args.num_samples, random_seed=42, dim_reduction=args.dim_reduction)
            
            # Now create visualizations with our simpler methods if we have GIFs
            if gif_paths:                    
                print("\nCreating animated GIF visualization...")
                try:
                    animated_path = create_animated_gif_visualization(features_df, gif_paths, output_dir, dim_reduction=args.dim_reduction, frame_data=all_frames)
                    print(f"Animated GIF visualization created at {animated_path}")
                    print(f"Open this in your browser to see animated GIFs directly at their {args.dim_reduction.upper()} coordinates.")
                except Exception as e:
                    print(f"Error creating animated GIF visualization: {e}")
                
         
            else:
                print("No GIFs were created successfully. Skipping additional visualizations.")
        else:
            print("No valid indices found. Skipping GIF creation and visualizations.")
    else:
        print("Warning: Could not initialize dataset. Skipping visualizations with GIFs.")
        print("If you want to create the visualization, please ensure your config has valid paths for:")
        print("- raw_base_dir: The directory containing raw volumes")
        print("- seg_base_dir: The directory containing segmentation volumes")
        print("- add_mask_base_dir: The directory containing additional mask volumes (optional)")
        print("- excel_file: The directory containing Excel files with synapse data")