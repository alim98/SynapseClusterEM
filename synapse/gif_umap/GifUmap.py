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
            alpha=config.alpha
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
    Create a GIF from a volume (3D array).
    
    Args:
        volume: 3D array representing volume data
        output_path: Path to save the GIF
        fps: Frames per second
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
    
    return output_path

def create_animated_gif_visualization(features_df, gif_paths, output_dir, dim_reduction='umap'):
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
    
    Returns:
        Path to the HTML file
    """
    method_name = "UMAP" if dim_reduction == 'umap' else "t-SNE"
    import base64
    
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
                        
                        samples_with_gifs.append({
                            'id': idx,
                            'x': x,
                            'y': y,
                            'cluster': cluster,
                            'bbox': bbox,
                            'gif_data': encoded_gif
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
    
    # Compute the bounds of the coordinate values
    all_x_values = features_df[x_col].values
    all_y_values = features_df[y_col].values
    
    x_min, x_max = float(min(all_x_values)), float(max(all_x_values))
    y_min, y_max = float(min(all_y_values)), float(max(all_y_values))
    
    # Add padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding
    
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
            
            # Add this point to the samples array
            points_content += f"""
                {{
                    id: {idx},
                    x: {x},
                    y: {y},
                    color: "{color}",
                    hasGif: {str(idx in gif_paths).lower()}
                }},"""
    
    # Generate HTML content for GIFs
    gifs_content = ""
    for sample in samples_with_gifs:
        gifs_content += f"""
                {{
                    id: {sample['id']},
                    x: {sample['x']},
                    y: {sample['y']},
                    cluster: "{sample['cluster']}",
                    bbox: "{sample['bbox']}",
                    gifData: "{sample['gif_data']}"
                }},"""
    
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{method_name} Visualization with Animated GIFs</h1>
            
            <div class="controls">
                <div class="control-group">
                    <button id="toggle-gifs">Show/Hide GIFs</button>
                </div>
                
                <div class="control-group">
                    <span class="control-label">GIF Size:</span>
                    <input type="range" min="50" max="200" value="100" id="gif-size-slider" class="gif-size-slider">
                    <span id="size-value">100px</span>
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
            // Define the UMAP bounds
            const xMin = {x_min};
            const xMax = {x_max};
            const yMin = {y_min};
            const yMax = {y_max};
            
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
            
            // Store original positions of GIFs for reset functionality
            const originalPositions = {{}};
            
            // Store all GIF data with cluster information
            const allGifData = [];
            
            // Add background points for all samples
            function addBackgroundPoints() {{
                // Samples from features_df
                const samples = [{points_content}
                ];
                
                // Add points to the plot
                samples.forEach(sample => {{
                    const [plotX, plotY] = mapToPlot(sample.x, sample.y, plotWidth, plotHeight);
                    
                    const pointElem = document.createElement('div');
                    pointElem.className = 'point';
                    pointElem.style.left = ${{plotX}}px;
                    pointElem.style.top = ${{plotY}}px;
                    pointElem.style.backgroundColor = sample.color;
                    
                    // Make points with GIFs larger
                    if (sample.hasGif) {{
                        pointElem.style.width = '10px';
                        pointElem.style.height = '10px';
                        pointElem.style.border = '2px solid black';
                        pointElem.style.zIndex = '5';
                    }}
                    
                    plot.appendChild(pointElem);
                }});
            }}
            
            // Add GIFs to the plot
            function addGifs() {{
                // Samples with GIFs
                const samplesWithGifs = [{gifs_content}
                ];
                
                // Store all GIF data for filtering
                allGifData.push(...samplesWithGifs);
                
                // Add GIFs to the plot
                samplesWithGifs.forEach((sample, index) => {{
                    const [plotX, plotY] = mapToPlot(sample.x, sample.y, plotWidth, plotHeight);
                    
                    // Store original position for reset functionality
                    originalPositions[sample.id] = {{ x: plotX, y: plotY }};
                    
                    // Create GIF container
                    const gifContainer = document.createElement('div');
                    gifContainer.className = 'gif-container';
                    gifContainer.id = `gif-${sample.id}`;
                    gifContainer.style.left = `${plotX}px`;
                    gifContainer.style.top = `${plotY}px`;
                    gifContainer.style.width = '100px';
                    gifContainer.style.height = '100px';
                    gifContainer.dataset.index = index;
                    gifContainer.dataset.cluster = sample.cluster;
                    
                    // Create GIF image using base64 data
                    const gifImg = document.createElement('img');
                    gifImg.src = `data:image/gif;base64,${sample.gifData}`;
                    gifImg.alt = `Sample ${sample.id}`;
                    gifImg.setAttribute('loop', 'infinite');
                    
                    // Add to container
                    gifContainer.appendChild(gifImg);
                    
                    // Make the GIF container draggable
                    makeDraggable(gifContainer);
                    
                    // Add to plot
                    plot.appendChild(gifContainer);
                }});
                
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
                    if (sample.cluster !== 'N/A') {{
                        clusters.add(sample.cluster);
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
                    checkbox.id = `cluster-${cluster}`;
                    
                    checkbox.addEventListener('change', updateVisibleGifs);
                    
                    label.appendChild(checkbox);
                    label.appendChild(document.createTextNode(Cluster ${{cluster}}));
                    
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
                    
                    // Update active GIF container size if one is shown
                    if (activeGifContainer) {{
                        activeGifContainer.style.width = size + 'px';
                        activeGifContainer.style.height = size + 'px';
                    }}
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
                        if (originalPositions[id]) {{
                            container.style.left = `${originalPositions[id].x}px`;
                            container.style.top = `${originalPositions[id].y}px`;
                        }}
                    }});
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
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Created animated GIF visualization with embedded data: {html_path}")
    except UnicodeEncodeError:
        # If utf-8 fails, try with a more compatible encoding
        with open(html_path, 'w', encoding='ascii', errors='xmlcharrefreplace') as f:
            f.write(html_content)
        print(f"Created animated GIF visualization with embedded data (using ASCII encoding): {html_path}")
    
    return html_path

def create_clickable_gif_visualization(features_df, gif_paths, output_dir, dim_reduction='umap'):
    """
    Create an HTML visualization where GIFs only appear when you click on their corresponding points.
    
    Args:
        features_df: DataFrame with features and coordinates
        gif_paths: Dictionary mapping sample indices to GIF paths
        output_dir: Directory to save the HTML file
        dim_reduction: Dimensionality reduction method ('umap' or 'tsne')
    
    Returns:
        Path to the HTML file
    """
    method_name = "UMAP" if dim_reduction == 'umap' else "t-SNE"
    import base64
    
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
                        
                        samples_with_gifs.append({
                            'id': idx,
                            'x': x,
                            'y': y,
                            'cluster': cluster,
                            'bbox': bbox,
                            'gif_data': encoded_gif
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
    
    # Compute the bounds of the coordinate values
    all_x_values = features_df[x_col].values
    all_y_values = features_df[y_col].values
    
    x_min, x_max = float(min(all_x_values)), float(max(all_x_values))
    y_min, y_max = float(min(all_y_values)), float(max(all_y_values))
    
    # Add padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding
    
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
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Click-to-Show GIFs - {method_name} Visualization</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1800px;
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
                width: 1600px;
                height: 1200px;
            }}
            .point {{
                position: absolute;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                cursor: pointer;
                transition: transform 0.2s ease;
            }}
            .point:hover {{
                transform: translate(-50%, -50%) scale(1.5);
                z-index: 100;
            }}
            .point.has-gif {{
                width: 12px;
                height: 12px;
                border: 2px solid white;
                box-shadow: 0 0 4px rgba(0,0,0,0.5);
            }}
            .point.active {{
                transform: translate(-50%, -50%) scale(1.8);
                box-shadow: 0 0 8px rgba(0,0,0,0.8);
                z-index: 101;
            }}
            .gif-container {{
                position: absolute;
                border: 3px solid #333;
                background-color: white;
                border-radius: 8px;
                overflow: hidden;
                transform: translate(-50%, -50%);
                z-index: 200;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                display: none;
            }}
            .gif-container img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}
            .gif-info {{
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background-color: rgba(0,0,0,0.7);
                color: white;
                padding: 8px;
                font-size: 14px;
                text-align: center;
            }}
            .close-btn {{
                position: absolute;
                top: 5px;
                right: 5px;
                width: 24px;
                height: 24px;
                background-color: rgba(255,255,255,0.8);
                border: none;
                border-radius: 50%;
                font-weight: bold;
                cursor: pointer;
                z-index: 201;
            }}
            .close-btn:hover {{
                background-color: rgba(255,0,0,0.2);
            }}
            .controls {{
                margin: 20px auto;
                text-align: center;
                padding: 15px;
                background-color: #f8f8f8;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
            .gif-size-slider {{
                width: 200px;
                margin: 0 10px;
                vertical-align: middle;
            }}
            .legend {{
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: rgba(255,255,255,0.9);
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                max-width: 200px;
            }}
            .legend-item {{
                margin-bottom: 5px;
                display: flex;
                align-items: center;
            }}
            .legend-color {{
                width: 15px;
                height: 15px;
                border-radius: 50%;
                margin-right: 8px;
                display: inline-block;
            }}
            .instructions {{
                text-align: center;
                margin: 15px 0;
                color: #666;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Click-to-Show GIFs - {method_name} Visualization</h1>
            
            <div class="instructions">
                <p>Click on any highlighted point to view its GIF. Click elsewhere or on the close button to hide the GIF.</p>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <span class="control-label">GIF Size:</span>
                    <input type="range" min="100" max="400" value="200" id="gif-size-slider" class="gif-size-slider">
                    <span id="size-value">200px</span>
                </div>
                
                <div class="control-group">
                    <label>
                        <input type="checkbox" id="show-all-points" checked>
                        Show all points
                    </label>
                </div>
                
                <div class="control-group">
                    <label>
                        <input type="checkbox" id="highlight-gif-points" checked>
                        Highlight points with GIFs
                    </label>
                </div>
            </div>
            
            <div class="plot-container" id="plot">
                <!-- Points and GIFs will be added here by JavaScript -->
                
                <div class="legend" id="legend">
                    <h3>Legend</h3>
                    <!-- Legend items will be added here by JavaScript -->
                </div>
            </div>
        </div>
        
        <script>
            // Define the coordinate bounds
            const xMin = {x_min};
            const xMax = {x_max};
            const yMin = {y_min};
            const yMax = {y_max};
            
            // Function to map data coordinates to plot coordinates
            function mapToPlot(x, y, width, height) {{
                const plotX = ((x - xMin) / (xMax - xMin)) * width;
                // Invert y-axis (data coordinates increase upward, plot coordinates increase downward)
                const plotY = height - ((y - yMin) / (yMax - yMin)) * height;
                return [plotX, plotY];
            }}
            
            // Get the plot container
            const plot = document.getElementById('plot');
            const plotWidth = plot.clientWidth;
            const plotHeight = plot.clientHeight;
            
            // Store all points and GIF data
            const allPoints = [];
            const gifData = [];
            
            // Currently active GIF container
            let activeGifContainer = null;
            
            // Add all data points to the plot
            function addDataPoints() {{
                // Get all samples from the features DataFrame
                const samples = [];
                
                // Add points for all samples in the DataFrame
                {features_df.apply(lambda row: f"""
                samples.push({{
                    id: {row.name if hasattr(row.name, 'item') else row.name},
                    x: {row[x_col] if hasattr(row[x_col], 'item') else row[x_col]},
                    y: {row[y_col] if hasattr(row[y_col], 'item') else row[y_col]},
                    cluster: "{row.get('cluster', 'N/A') if hasattr(row.get('cluster', 'N/A'), 'item') else row.get('cluster', 'N/A')}",
                    hasGif: {str(row.name in gif_paths).lower()}
                }});
                """, axis=1).str.cat(sep='\n')}
                
                // Create a point element for each sample
                samples.forEach(sample => {{
                    const [plotX, plotY] = mapToPlot(sample.x, sample.y, plotWidth, plotHeight);
                    
                    // Determine color based on cluster
                    let color = "#666666";
                    if (sample.cluster !== "N/A") {{
                        // Use cluster-based coloring
                        const clusterColors = {{}};
                        {';'.join([f'clusterColors["{cluster}"] = "{point_colors.get(cluster, "#666666")}"' for cluster in features_df['cluster'].unique() if hasattr(cluster, 'item')])}
                        color = clusterColors[sample.cluster] || "#666666";
                    }}
                    
                    // Create the point element
                    const pointElem = document.createElement('div');
                    pointElem.className = 'point' + (sample.hasGif ? ' has-gif' : '');
                    pointElem.style.left = plotX + 'px';
                    pointElem.style.top = plotY + 'px';
                    pointElem.style.backgroundColor = color;
                    pointElem.dataset.id = sample.id;
                    pointElem.dataset.cluster = sample.cluster;
                    
                    // Store the point data
                    allPoints.push({{
                        id: sample.id,
                        element: pointElem,
                        x: plotX,
                        y: plotY,
                        cluster: sample.cluster,
                        hasGif: sample.hasGif
                    }});
                    
                    // Add click event for points with GIFs
                    if (sample.hasGif) {{
                        pointElem.addEventListener('click', (e) => {{
                            e.stopPropagation();
                            showGif(sample.id, plotX, plotY);
                            
                            // Mark this point as active
                            document.querySelectorAll('.point.active').forEach(p => {{
                                p.classList.remove('active');
                            }});
                            pointElem.classList.add('active');
                        }});
                    }}
                    
                    // Add to plot
                    plot.appendChild(pointElem);
                }});
                
                // Add click event to the plot to hide GIFs when clicking elsewhere
                plot.addEventListener('click', (e) => {{
                    if (e.target === plot) {{
                        hideAllGifs();
                        document.querySelectorAll('.point.active').forEach(p => {{
                            p.classList.remove('active');
                        }});
                    }}
                }});
            }}
            
            // Add GIF data
            function addGifData() {{
                // Samples with GIFs
                const samplesWithGifs = [];
                
                // Add GIF data for samples that have GIFs
                {';'.join([f"""
                samplesWithGifs.push({{
                    id: {sample['id']},
                    cluster: "{sample['cluster']}",
                    bbox: "{sample['bbox']}",
                    gifData: "{sample['gif_data']}"
                }});
                """ for sample in samples_with_gifs])}
                
                // Store GIF data for later use
                gifData.push(...samplesWithGifs);
            }}
            
            // Show GIF for a specific sample
            function showGif(sample_id, x, y) {{
                // Find the GIF data for this sample
                const sample = gifData.find(s => s.id === sample_id);
                if (!sample) return;
                
                // Check if this GIF is already visible
                let gifContainer = document.getElementById(`gif-${{sample_id}}`);
                if (gifContainer && gifContainer.style.display === 'block') {{
                    // GIF is already visible, no need to do anything
                    return;
                }}
                
                // Create GIF container if it doesn't exist
                if (!gifContainer) {{
                    gifContainer = document.createElement('div');
                    gifContainer.className = 'gif-container';
                    gifContainer.id = `gif-${{sample_id}}`;
                    
                    // Get the current size from the slider
                    const size = parseInt(document.getElementById('gif-size-slider').value);
                    gifContainer.style.width = size + 'px';
                    gifContainer.style.height = size + 'px';
                    
                    // Create GIF image using base64 data
                    const gifImg = document.createElement('img');
                    gifImg.src = 'data:image/gif;base64,' + sample.gifData;
                    gifImg.alt = 'Sample ' + sample_id;
                    
                    // Create info element
                    const infoElem = document.createElement('div');
                    infoElem.className = 'gif-info';
                    infoElem.textContent = 'ID: ' + sample_id + ', Cluster: ' + sample.cluster + ', BBox: ' + sample.bbox;
                    
                    // Create close button
                    const closeBtn = document.createElement('button');
                    closeBtn.className = 'close-btn';
                    closeBtn.textContent = '×';
                    closeBtn.addEventListener('click', (e) => {{
                        e.stopPropagation();
                        // Only hide this specific GIF, not all GIFs
                        gifContainer.style.display = 'none';
                        // Remove active state from the corresponding point
                        const point = document.querySelector(`.point[data-id="${{sample_id}}"]`);
                        if (point) {{
                            point.classList.remove('active');
                        }}
                    }});
                    
                    // Add elements to container
                    gifContainer.appendChild(gifImg);
                    gifContainer.appendChild(infoElem);
                    gifContainer.appendChild(closeBtn);
                    
                    // Add to plot
                    plot.appendChild(gifContainer);
                }}

                // Function to check if two rectangles overlap
                function checkOverlap(rect1, rect2) {{
                    return !(rect1.right < rect2.left || 
                            rect1.left > rect2.right || 
                            rect1.bottom < rect2.top || 
                            rect1.top > rect2.bottom);
                }}

                // Function to get rectangle dimensions for a position
                function getRectangle(posX, posY, size) {{
                    const halfSize = size / 2;
                    return {{
                        left: posX - halfSize,
                        right: posX + halfSize,
                        top: posY - halfSize,
                        bottom: posY + halfSize
                    }};
                }}

                // Function to find a non-overlapping position
                function findNonOverlappingPosition(startX, startY, size) {{
                    const visibleGifs = Array.from(document.querySelectorAll('.gif-container'))
                        .filter(container => container.id !== `gif-${{sample_id}}` && 
                                          container.style.display === 'block')
                        .map(container => {{
                            const rect = container.getBoundingClientRect();
                            const plotRect = plot.getBoundingClientRect();
                            return {{
                                left: rect.left - plotRect.left,
                                right: rect.right - plotRect.left,
                                top: rect.top - plotRect.top,
                                bottom: rect.bottom - plotRect.top
                            }};
                        }});

                    // If no visible GIFs, return original position
                    if (visibleGifs.length === 0) {{
                        return {{ x: startX, y: startY }};
                    }}

                    const halfSize = size / 2;
                    const spiralStep = size * 0.75; // Distance between spiral points
                    const maxAttempts = 50; // Maximum number of attempts to find position
                    let angle = 0;
                    let radius = spiralStep;

                    // Try positions in a spiral pattern
                    for (let i = 0; i < maxAttempts; i++) {{
                        // Calculate position on spiral
                        const testX = startX + radius * Math.cos(angle);
                        const testY = startY + radius * Math.sin(angle);

                        // Check if position is within plot bounds
                        if (testX - halfSize < 0 || testX + halfSize > plotWidth ||
                            testY - halfSize < 0 || testY + halfSize > plotHeight) {{
                            angle += Math.PI / 4;
                            radius += spiralStep;
                            continue;
                        }}

                        // Check if this position overlaps with any existing GIFs
                        const testRect = getRectangle(testX, testY, size);
                        const hasOverlap = visibleGifs.some(existingRect => 
                            checkOverlap(testRect, existingRect));

                        if (!hasOverlap) {{
                            return {{ x: testX, y: testY }};
                        }}

                        angle += Math.PI / 4;
                        radius += spiralStep;
                    }}

                    // If no non-overlapping position found, return original position
                    return {{ x: startX, y: startY }};
                }}

                // Get current size and find non-overlapping position
                const currentSize = parseInt(gifContainer.style.width);
                const newPosition = findNonOverlappingPosition(x, y, currentSize);

                // Position the GIF container
                gifContainer.style.left = newPosition.x + 'px';
                gifContainer.style.top = newPosition.y + 'px';

                // Show the GIF
                gifContainer.style.display = 'block';
            }}
            
            // Hide all GIFs
            function hideAllGifs() {{
                document.querySelectorAll('.gif-container').forEach(container => {{
                    container.style.display = 'none';
                }});
            }}
            
            // Create legend
            function createLegend() {{
                const legend = document.getElementById('legend');
                const clusters = new Set();
                const clusterColors = {{}};
                
                // Collect unique clusters and their colors
                allPoints.forEach(point => {{
                    if (point.cluster !== 'N/A') {{
                        clusters.add(point.cluster);
                        const color = point.element.style.backgroundColor;
                        clusterColors[point.cluster] = color;
                    }}
                }});
                
                // Create legend items
                clusters.forEach(cluster => {{
                    const item = document.createElement('div');
                    item.className = 'legend-item';
                    
                    const colorBox = document.createElement('div');
                    colorBox.className = 'legend-color';
                    colorBox.style.backgroundColor = clusterColors[cluster];
                    
                    const label = document.createElement('span');
                    label.textContent = 'Cluster ' + cluster;
                    
                    item.appendChild(colorBox);
                    item.appendChild(label);
                    legend.appendChild(item);
                }});
                
                // Add legend item for points with GIFs
                const gifItem = document.createElement('div');
                gifItem.className = 'legend-item';
                
                const gifMarker = document.createElement('div');
                gifMarker.className = 'legend-color';
                gifMarker.style.border = '2px solid white';
                gifMarker.style.boxShadow = '0 0 4px rgba(0,0,0,0.5)';
                gifMarker.style.backgroundColor = '#666';
                
                const gifLabel = document.createElement('span');
                gifLabel.textContent = 'Has GIF (clickable)';
                
                gifItem.appendChild(gifMarker);
                gifItem.appendChild(gifLabel);
                legend.appendChild(gifItem);
            }}
            
            // Initialize controls
            function initControls() {{
                // GIF size slider
                const sizeSlider = document.getElementById('gif-size-slider');
                const sizeValue = document.getElementById('size-value');
                
                sizeSlider.addEventListener('input', () => {{
                    const size = sizeSlider.value;
                    sizeValue.textContent = `${{size}}px`;
                    
                    // Update active GIF container size if one is shown
                    if (activeGifContainer) {{
                        activeGifContainer.style.width = `${{size}}px`;
                        activeGifContainer.style.height = `${{size}}px`;
                    }}
                }});
                
                // Show all points checkbox
                const showAllPointsCheckbox = document.getElementById('show-all-points');
                showAllPointsCheckbox.addEventListener('change', () => {{
                    const showAll = showAllPointsCheckbox.checked;
                    
                    allPoints.forEach(point => {{
                        if (!point.hasGif) {{
                            point.element.style.display = showAll ? 'block' : 'none';
                        }}
                    }});
                }});
                
                // Highlight points with GIFs checkbox
                const highlightGifPointsCheckbox = document.getElementById('highlight-gif-points');
                highlightGifPointsCheckbox.addEventListener('change', () => {{
                    const highlight = highlightGifPointsCheckbox.checked;
                    
                    allPoints.forEach(point => {{
                        if (point.hasGif) {{
                            if (highlight) {{
                                point.element.classList.add('has-gif');
                            }} else {{
                                point.element.classList.remove('has-gif');
                            }}
                        }}
                    }});
                }});
            }}
            
            // Initialize the visualization
            function init() {{
                // Add data points
                addDataPoints();
                
                // Add GIF data
                addGifData();
                
                // Create legend
                createLegend();
                
                // Initialize controls
                initControls();
                
                // Modify plot click handler to not close GIFs when clicking on empty space
                plot.removeEventListener('click', plot.onclick);
                plot.addEventListener('click', (e) => {{
                    // Only do something if we click directly on the plot (not on a point or GIF)
                    if (e.target === plot) {{
                        // We don't close GIFs anymore when clicking on empty space
                        // Just remove the active state from points
                        document.querySelectorAll('.point.active').forEach(p => {{
                            p.classList.remove('active');
                        }});
                    }}
                }});
                
                // Add a "Close All GIFs" button to the controls
                const controlsDiv = document.querySelector('.controls');
                const closeAllGroup = document.createElement('div');
                closeAllGroup.className = 'control-group';
                
                const closeAllBtn = document.createElement('button');
                closeAllBtn.id = 'close-all-gifs';
                closeAllBtn.textContent = 'Close All GIFs';
                closeAllBtn.addEventListener('click', () => {{
                    hideAllGifs();
                    document.querySelectorAll('.point.active').forEach(p => {{
                        p.classList.remove('active');
                    }});
                }});
                
                closeAllGroup.appendChild(closeAllBtn);
                controlsDiv.appendChild(closeAllGroup);
            }}
            
            // Run initialization when the page loads
            window.onload = init;
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    html_path = output_dir / f"clickable_{dim_reduction}_visualization.html"
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Created clickable {method_name} visualization: {html_path}")
    except UnicodeEncodeError:
        # If utf-8 fails, try with a more compatible encoding
        with open(html_path, 'w', encoding='ascii', errors='xmlcharrefreplace') as f:
            f.write(html_content)
        print(f"Created clickable {method_name} visualization (using ASCII encoding): {html_path}")
    
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
            num_samples = 80  # User requested 40 samples
            
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
                    create_gif_from_volume(volume, str(gif_path), fps=8)
                    
                    # Check if GIF was successfully created
                    if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                        # Store full absolute path for HTML file - this is crucial for browser to find the GIFs
                        gif_paths[idx] = os.path.abspath(str(gif_path))
                        print(f"Created GIF for sample {idx}")
                    else:
                        print(f"Failed to create GIF for sample {idx} - file not created or empty")
                    
                except Exception as e:
                    print(f"Error creating GIF for sample {idx}: {str(e)}")

            # Now create visualizations with our simpler methods if we have GIFs
            if gif_paths:
                print("\nCreating simpler matplotlib visualization with GIF thumbnails...")
             
                print("\nCreating clickable GIF visualization...")
                try:
                    import traceback
                    clickable_path = create_clickable_gif_visualization(features_df, gif_paths, output_dir, dim_reduction=args.dim_reduction)
                    print(f"Clickable GIF visualization created at {clickable_path}")
                    print(f"Open this in your browser to click on points and see their GIFs.")
                except Exception as e:
                    print(f"Error creating clickable GIF visualization: {e}")
                    print("Detailed error:")
                    traceback.print_exc()
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