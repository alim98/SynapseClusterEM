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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse import config
from synapse.clustering import load_and_cluster_features, apply_tsne, save_tsne_plots, find_random_samples_in_clusters, save_cluster_samples

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

def create_umap_with_embedded_gifs_at_coordinates(features_df, gif_paths, output_dir):
    """
    Create a custom HTML visualization that displays GIFs directly at their UMAP coordinates.
    
    Args:
        features_df: DataFrame with features, cluster assignments, and UMAP coordinates
        gif_paths: Dictionary mapping sample indices to GIF paths
        output_dir: Directory to save the HTML file
    
    Returns:
        Path to the HTML file
    """
    # Extract UMAP coordinates and other info for samples with GIFs
    samples_with_gifs = []
    for idx in gif_paths:
        if idx in features_df.index:
            sample = features_df.loc[idx]
            x, y = sample['umap_x'], sample['umap_y']
            
            # Extract cluster and bbox information if available
            cluster = sample.get('cluster', 'N/A') if 'cluster' in sample else 'N/A'
            bbox = sample.get('bbox_name', 'unknown') if 'bbox_name' in sample else 'unknown'
            
            # Convert numpy/pandas types to Python native types for JSON serialization
            if hasattr(idx, 'item'):  # Convert numpy types to Python native types
                idx = idx.item()
            if hasattr(x, 'item'):
                x = x.item()
            if hasattr(y, 'item'):
                y = y.item()
            if hasattr(cluster, 'item'):
                cluster = cluster.item()
            
            samples_with_gifs.append({
                'id': idx,
                'x': x,
                'y': y,
                'cluster': cluster,
                'bbox': bbox,
                'gif_path': gif_paths[idx]
            })
    
    # Compute the bounds of the UMAP coordinates to set the canvas size
    if samples_with_gifs:
        x_values = [s['x'] for s in samples_with_gifs]
        y_values = [s['y'] for s in samples_with_gifs]
        
        all_x_values = features_df['umap_x'].values
        all_y_values = features_df['umap_y'].values
        
        x_min, x_max = float(min(all_x_values)), float(max(all_x_values))
        y_min, y_max = float(min(all_y_values)), float(max(all_y_values))
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_min, x_max = x_min - x_padding, x_max + x_padding
        y_min, y_max = y_min - y_padding, y_max + y_padding
    else:
        # Default values if no samples with GIFs
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
    
    # Create canvas width and height
    canvas_width = 1200
    canvas_height = 900
    
    # Function to map UMAP coordinates to canvas positions
    def map_to_canvas(x, y):
        canvas_x = ((x - x_min) / (x_max - x_min)) * canvas_width
        # Invert y-axis (because in canvas, y increases downward)
        canvas_y = ((y_max - y) / (y_max - y_min)) * canvas_height
        return canvas_x, canvas_y
    
    # Generate points for all samples
    points = []
    for idx, row in features_df.iterrows():
        x, y = row['umap_x'], row['umap_y']
        canvas_x, canvas_y = map_to_canvas(x, y)
        
        # Convert numpy/pandas types to Python native types
        if hasattr(idx, 'item'):
            idx = idx.item()
        if hasattr(canvas_x, 'item'):
            canvas_x = canvas_x.item()
        if hasattr(canvas_y, 'item'):
            canvas_y = canvas_y.item()
        
        # Determine color based on cluster or bbox
        if 'cluster' in features_df.columns:
            cluster = row['cluster']
            if hasattr(cluster, 'item'):
                cluster = cluster.item()
            # Generate a color based on cluster (using a simple hash function)
            color = f"hsl({hash(str(cluster)) % 360}, 80%, 60%)"
            label = f"Cluster {cluster}"
        elif 'bbox_name' in features_df.columns:
            bbox = row['bbox_name']
            # Use predefined colors or generate
            color_mapping = {
                'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
                'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
            }
            color = color_mapping.get(bbox, f"hsl({hash(bbox) % 360}, 80%, 60%)")
            label = bbox
        else:
            color = "#666666"
            label = "Sample"
        
        points.append({
            'x': canvas_x,
            'y': canvas_y,
            'color': color,
            'label': label,
            'id': idx,
            'has_gif': idx in gif_paths
        })
    
    # Group points by label for the legend
    labels = {}
    for point in points:
        label = point['label']
        if label not in labels:
            labels[label] = {'color': point['color'], 'count': 0}
        labels[label]['count'] += 1
    
    # Convert points and samples_with_gifs to JSON strings for embedding in HTML
    import json
    try:
        points_json = json.dumps(points).replace('"', '\\"')
        samples_json = json.dumps(samples_with_gifs).replace('"', '\\"')
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}")
        # Try a more robust approach
        def safe_serialize(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [safe_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {safe_serialize(k): safe_serialize(v) for k, v in obj.items()}
            else:
                return obj
        
        # Apply safe serialization
        safe_points = [safe_serialize(p) for p in points]
        safe_samples = [safe_serialize(s) for s in samples_with_gifs]
        
        # Try again with safe data
        points_json = json.dumps(safe_points).replace('"', '\\"')
        samples_json = json.dumps(safe_samples).replace('"', '\\"')
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>UMAP Visualization with GIFs at Coordinates</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1300px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 20px;
                border-radius: 8px;
                position: relative;
            }}
            h1 {{
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }}
            .canvas-container {{
                position: relative;
                width: {canvas_width}px;
                height: {canvas_height}px;
                margin: 0 auto;
                border: 1px solid #ddd;
                overflow: hidden;
            }}
            .point {{
                position: absolute;
                width: 6px;
                height: 6px;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                cursor: pointer;
            }}
            .point.with-gif {{
                width: 10px;
                height: 10px;
                border: 2px solid black;
            }}
            .gif-container {{
                position: absolute;
                width: 150px;
                height: 150px;
                transform: translate(-50%, -50%);
                background-color: white;
                border: 2px solid #333;
                border-radius: 8px;
                overflow: hidden;
                z-index: 100;
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
                padding: 4px;
                font-size: 12px;
                text-align: center;
            }}
            .legend {{
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: rgba(255,255,255,0.8);
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
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
            }}
            .instructions {{
                text-align: center;
                margin: 15px 0;
                color: #666;
            }}
            /* Controls for zooming and panning */
            .controls {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                display: flex;
                gap: 10px;
            }}
            .control-btn {{
                background-color: rgba(255,255,255,0.8);
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px 10px;
                cursor: pointer;
            }}
            .control-btn:hover {{
                background-color: #f0f0f0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>UMAP Visualization with GIFs at Coordinates</h1>
            
            <div class="instructions">
                <p>Hover over points to see details. Click on highlighted points to view GIFs. Use controls to zoom and pan.</p>
            </div>
            
            <div class="canvas-container" id="canvas">
                <!-- Points will be added here by JavaScript -->
                
                <!-- Legend -->
                <div class="legend">
                    <h3>Legend</h3>
                    <div id="legend-content">
                        <!-- Legend items will be added here by JavaScript -->
                    </div>
                    <div style="margin-top: 10px;">
                        <small><b>•</b> Small points: regular samples</small><br>
                        <small><b>O</b> Large points: samples with GIFs</small>
                    </div>
                </div>
                
                <!-- Controls -->
                <div class="controls">
                    <button class="control-btn" id="zoom-in">Zoom In</button>
                    <button class="control-btn" id="zoom-out">Zoom Out</button>
                    <button class="control-btn" id="reset">Reset View</button>
                </div>
            </div>
        </div>
        
        <script>
            // Data for points
            const points = JSON.parse("{points_json}");
            
            // Data for GIFs
            const gifSamples = JSON.parse("{samples_json}");
            
            // Canvas dimensions
            const canvasWidth = {canvas_width};
            const canvasHeight = {canvas_height};
            
            // Setup the canvas
            const canvas = document.getElementById('canvas');
            
            // Zoom and pan variables
            let zoomLevel = 1;
            let panX = 0;
            let panY = 0;
            
            // Function to apply transformations
            function applyTransformation() {{
                const points = document.querySelectorAll('.point');
                const gifs = document.querySelectorAll('.gif-container');
                
                points.forEach(point => {{
                    const baseX = parseFloat(point.getAttribute('data-x'));
                    const baseY = parseFloat(point.getAttribute('data-y'));
                    
                    const transformedX = baseX * zoomLevel + panX;
                    const transformedY = baseY * zoomLevel + panY;
                    
                    point.style.left = transformedX + 'px';
                    point.style.top = transformedY + 'px';
                }});
                
                gifs.forEach(gif => {{
                    if (gif.style.display === 'block') {{
                        const baseX = parseFloat(gif.getAttribute('data-x'));
                        const baseY = parseFloat(gif.getAttribute('data-y'));
                        
                        const transformedX = baseX * zoomLevel + panX;
                        const transformedY = baseY * zoomLevel + panY;
                        
                        gif.style.left = transformedX + 'px';
                        gif.style.top = transformedY + 'px';
                    }}
                }});
            }}
            
            // Add event listeners for controls
            document.getElementById('zoom-in').addEventListener('click', () => {{
                zoomLevel *= 1.2;
                applyTransformation();
            }});
            
            document.getElementById('zoom-out').addEventListener('click', () => {{
                zoomLevel /= 1.2;
                applyTransformation();
            }});
            
            document.getElementById('reset').addEventListener('click', () => {{
                zoomLevel = 1;
                panX = 0;
                panY = 0;
                applyTransformation();
            }});
            
            // Add event listener for panning
            let isDragging = false;
            let lastX, lastY;
            
            canvas.addEventListener('mousedown', (e) => {{
                isDragging = true;
                lastX = e.clientX;
                lastY = e.clientY;
                canvas.style.cursor = 'grabbing';
            }});
            
            document.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    const deltaX = e.clientX - lastX;
                    const deltaY = e.clientY - lastY;
                    
                    panX += deltaX;
                    panY += deltaY;
                    
                    lastX = e.clientX;
                    lastY = e.clientY;
                    
                    applyTransformation();
                }}
            }});
            
            document.addEventListener('mouseup', () => {{
                isDragging = false;
                canvas.style.cursor = 'default';
            }});
            
            // Function to create a point element
            function createPoint(point) {{
                const pointElem = document.createElement('div');
                pointElem.className = 'point' + (point.has_gif ? ' with-gif' : '');
                pointElem.id = 'point-' + point.id;
                pointElem.style.left = point.x + 'px';
                pointElem.style.top = point.y + 'px';
                pointElem.style.backgroundColor = point.color;
                pointElem.setAttribute('data-x', point.x);
                pointElem.setAttribute('data-y', point.y);
                pointElem.setAttribute('data-id', point.id);
                
                // Add tooltip with basic info
                pointElem.title = `ID: ${{point.id}}, Label: ${{point.label}}`;
                
                // Add click event for points with GIFs
                if (point.has_gif) {{
                    pointElem.addEventListener('click', () => {{
                        // Find the GIF data
                        const gifData = gifSamples.find(g => g.id === point.id);
                        if (gifData) {{
                            // Check if the GIF container already exists
                            let gifContainer = document.getElementById('gif-' + point.id);
                            
                            // Toggle visibility
                            if (gifContainer) {{
                                if (gifContainer.style.display === 'none') {{
                                    // Hide all other GIFs first
                                    document.querySelectorAll('.gif-container').forEach(container => {{
                                        container.style.display = 'none';
                                    }});
                                    
                                    // Show this GIF
                                    gifContainer.style.display = 'block';
                                }} else {{
                                    gifContainer.style.display = 'none';
                                }}
                            }} else {{
                                // Create new GIF container
                                gifContainer = document.createElement('div');
                                gifContainer.className = 'gif-container';
                                gifContainer.id = 'gif-' + point.id;
                                gifContainer.style.left = point.x + 'px';
                                gifContainer.style.top = point.y + 'px';
                                gifContainer.setAttribute('data-x', point.x);
                                gifContainer.setAttribute('data-y', point.y);
                                
                                // Create image element
                                const img = document.createElement('img');
                                img.src = 'file://' + gifData.gif_path;
                                img.alt = 'Sample ' + point.id;
                                
                                // Create info element
                                const info = document.createElement('div');
                                info.className = 'gif-info';
                                
                                // Add appropriate info
                                let infoText = `ID: ${{point.id}}`;
                                if (gifData.cluster !== 'N/A') {{
                                    infoText += `, Cluster: ${{gifData.cluster}}`;
                                }}
                                if (gifData.bbox !== 'unknown') {{
                                    infoText += `, BBox: ${{gifData.bbox}}`;
                                }}
                                info.textContent = infoText;
                                
                                // Add close button
                                const closeBtn = document.createElement('button');
                                closeBtn.textContent = 'X';
                                closeBtn.style.position = 'absolute';
                                closeBtn.style.top = '5px';
                                closeBtn.style.right = '5px';
                                closeBtn.style.zIndex = '101';
                                closeBtn.style.background = 'rgba(255,255,255,0.7)';
                                closeBtn.style.border = 'none';
                                closeBtn.style.borderRadius = '50%';
                                closeBtn.style.width = '20px';
                                closeBtn.style.height = '20px';
                                closeBtn.style.cursor = 'pointer';
                                
                                closeBtn.addEventListener('click', (e) => {{
                                    e.stopPropagation();
                                    gifContainer.style.display = 'none';
                                }});
                                
                                // Add elements to container
                                gifContainer.appendChild(img);
                                gifContainer.appendChild(info);
                                gifContainer.appendChild(closeBtn);
                                
                                // Hide all other GIFs first
                                document.querySelectorAll('.gif-container').forEach(container => {{
                                    container.style.display = 'none';
                                }});
                                
                                // Add to canvas and show
                                canvas.appendChild(gifContainer);
                                gifContainer.style.display = 'block';
                            }}
                        }}
                    }});
                }}
                
                return pointElem;
            }}
            
            // Function to populate the legend
            function populateLegend() {{
                const legendContent = document.getElementById('legend-content');
                const labels = {{}};
                
                // Collect unique labels
                points.forEach(point => {{
                    if (!labels[point.label]) {{
                        labels[point.label] = point.color;
                    }}
                }});
                
                // Create legend items
                for (const label in labels) {{
                    const item = document.createElement('div');
                    item.className = 'legend-item';
                    
                    const colorBox = document.createElement('div');
                    colorBox.className = 'legend-color';
                    colorBox.style.backgroundColor = labels[label];
                    
                    const text = document.createElement('span');
                    text.textContent = label;
                    
                    item.appendChild(colorBox);
                    item.appendChild(text);
                    legendContent.appendChild(item);
                }}
            }}
            
            // Initialize the visualization
            function init() {{
                // Create all points
                points.forEach(point => {{
                    const pointElem = createPoint(point);
                    canvas.appendChild(pointElem);
                }});
                
                // Populate the legend
                populateLegend();
            }}
            
            // Run initialization
            window.onload = init;
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    html_path = Path(output_dir) / "umap_with_gifs_at_coordinates.html"
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Created custom UMAP visualization with GIFs at coordinates: {html_path}")
    except UnicodeEncodeError:
        # If utf-8 fails, try with a more compatible encoding
        with open(html_path, 'w', encoding='ascii', errors='xmlcharrefreplace') as f:
            f.write(html_content)
        print(f"Created custom UMAP visualization with GIFs at coordinates (using ASCII encoding): {html_path}")
    
    return html_path

def create_umap_with_gifs(features_df, dataset, output_path, num_samples=10, random_seed=42):
    """
    Create a UMAP visualization with GIFs for selected samples.
    
    Args:
        features_df: DataFrame containing features and cluster assignments
        dataset: SynapseDataset for generating GIFs
        output_path: Directory to save results
        num_samples: Number of random samples to show with GIFs
        random_seed: Random seed for reproducibility
    
    Returns:
        Path to the HTML file with the visualization
    """
    print(f"Creating UMAP visualization with GIFs for {num_samples} random samples...")
    
    # Create output directories
    output_dir = Path(output_path)
    gifs_dir = output_dir / "sample_gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get features for UMAP
    feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
    if not feature_cols:
        feature_cols = [col for col in features_df.columns if 'layer' in col and 'feat_' in col]
        
    if not feature_cols:
        raise ValueError("No feature columns found in the DataFrame")
    
    # Compute 2D UMAP
    print("Computing UMAP...")
    features = features_df[feature_cols].values
    features_scaled = StandardScaler().fit_transform(features)
    
    # Use UMAP directly instead of apply_tsne
    reducer = umap.UMAP(n_components=2, random_state=random_seed)
    umap_results = reducer.fit_transform(features_scaled)
    
    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]
    
    # Define color mapping for different bounding boxes
    color_mapping = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    
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
    
    # Select random samples, making sure they cover different clusters if possible
    np.random.seed(random_seed)
    random_samples = []
    
    if valid_indices:
        # If there are clusters, get samples from each
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
            sample_count = min(num_samples, len(valid_indices))
            random_samples = np.random.choice(valid_indices, size=sample_count, replace=False)
    
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
    
    # Create the plotly figure
    print("Creating Plotly visualization...")
    
    # Determine point colors based on cluster or bbox
    if 'cluster' in features_df.columns:
        color_column = 'cluster'
        title = "UMAP Visualization Colored by Cluster"
    elif 'bbox_name' in features_df.columns:
        color_column = 'bbox_name'
        title = "UMAP Visualization Colored by Bounding Box"
        color_discrete_map = color_mapping
    else:
        # No color grouping available
        color_column = None
        title = "UMAP Visualization"
        color_discrete_map = None
    
    # Basic scatter plot for all points
    if color_column:
        fig = px.scatter(
            features_df,
            x='umap_x',
            y='umap_y',
            color=color_column,
            title=title,
            color_discrete_map=color_discrete_map if color_column == 'bbox_name' else None,
            opacity=0.7
        )
    else:
        fig = px.scatter(
            features_df,
            x='umap_x',
            y='umap_y',
            title=title,
            opacity=0.7
        )
    
    # Add markers for samples with GIFs (only if we have any)
    if gif_paths:
        samples_with_gifs = list(gif_paths.keys())
        
        # Get dataframe subset for these samples
        try:
            gif_samples_df = features_df.loc[samples_with_gifs].copy()
            
            # Add the paths to the dataframe for reference in the hover
            gif_samples_df['gif_path'] = gif_samples_df.index.map(lambda x: gif_paths.get(x, ''))
            
            # Format hover text based on what columns are available
            hover_text_parts = []
            if 'cluster' in gif_samples_df.columns:
                hover_text_parts.append("'<b>Cluster:</b> ' + gif_samples_df['cluster'].astype(str)")
            if 'bbox_name' in gif_samples_df.columns:
                hover_text_parts.append("'<b>BBox:</b> ' + gif_samples_df['bbox_name']")
                
            hover_text = " + '<br>' + ".join(hover_text_parts) if hover_text_parts else "''"
            
            # Create a hover template with GIF images
            hover_template = (
                '<b>Sample ID:</b> %{text}<br>' +
                (eval(hover_text) if hover_text_parts else '') +
                ('<br>' if hover_text_parts else '') +
                '<img src="file://%{customdata}" width="150px"><extra></extra>'
            )
            
            # Add a trace for samples with GIFs (larger markers)
            fig.add_trace(
                go.Scatter(
                    x=gif_samples_df['umap_x'],
                    y=gif_samples_df['umap_y'],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='rgba(255, 255, 255, 0.8)',
                        line=dict(
                            color='rgba(0, 0, 0, 1)',
                            width=2
                        )
                    ),
                    text=gif_samples_df.index,
                    hovertemplate=hover_template,
                    customdata=gif_samples_df['gif_path'],
                    name='Samples with GIFs'
                )
            )
            
            # Also create a more reliable visualization method with embedded GIFs
            print("Creating enhanced HTML with embedded GIFs...")
            
            # Create a standalone HTML file with embedded GIFs
            enhanced_html_path = output_dir / "umap_with_embedded_gifs.html"
            
            # Create HTML with embedded GIFs
            create_enhanced_html_with_gifs(features_df, gif_paths, output_dir, enhanced_html_path)
            print(f"Enhanced HTML with embedded GIFs saved to {enhanced_html_path}")
            
            # Create a new visualization with GIFs positioned at UMAP coordinates
            print("Creating UMAP visualization with GIFs at coordinates...")
            coordinate_html_path = create_umap_with_embedded_gifs_at_coordinates(features_df, gif_paths, output_dir)
            print(f"UMAP with GIFs at coordinates saved to {coordinate_html_path}")
            
        except Exception as e:
            print(f"Error adding GIF sample markers: {e}")
    
    # Update layout for better appearance
    fig.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save the visualization (with optimized settings)
    html_path = output_dir / "umap_with_gifs.html"
    
    # Custom HTML to reduce file size - embed only necessary libraries and add custom CSS
    html_output = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>UMAP Visualization with GIF Samples</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .plot-container {
                background-color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 20px;
                border-radius: 8px;
            }
            .plotly-graph-div {
                margin: 0 auto;
            }
        </style>
    </head>
    <body>
        <div class="plot-container">
            <div id="plot"></div>
        </div>
        <script>
            var plotData = %s;
            Plotly.newPlot('plot', plotData.data, plotData.layout);
        </script>
        <div style="margin-top: 20px; text-align: center;">
            <p><b>Note:</b> If GIFs don't appear on hover, try opening the "umap_with_gifs_at_coordinates.html" file for a better visualization where GIFs are positioned directly at their coordinates.</p>
        </div>
    </body>
    </html>
    """ % fig.to_json()
    
    # Write the HTML file
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
    except UnicodeEncodeError:
        # If utf-8 fails, try with a more compatible encoding
        with open(html_path, 'w', encoding='ascii', errors='xmlcharrefreplace') as f:
            f.write(html_output)
    
    # Report results
    print(f"UMAP visualization with GIFs saved to {html_path}")
    print(f"Created {len(gif_paths)} GIFs in {gifs_dir}")
    
    return html_path

def create_enhanced_html_with_gifs(features_df, gif_paths, output_dir, html_path):
    """
    Create an enhanced HTML file with embedded GIFs for better reliability.
    This alternative approach doesn't rely on hover and directly shows the GIFs.
    
    Args:
        features_df: DataFrame with features and cluster assignments
        gif_paths: Dictionary mapping sample indices to GIF paths
        output_dir: Output directory
        html_path: Path to save the HTML file
    """
    # Get cluster information if available
    samples = []
    
    if 'cluster' in features_df.columns:
        clusters = features_df['cluster'].unique()
        cluster_samples = {cluster: [] for cluster in clusters}
        
        for idx in gif_paths:
            cluster = features_df.loc[idx, 'cluster']
            bbox_name = features_df.loc[idx].get('bbox_name', 'unknown')
            cluster_samples[cluster].append((idx, bbox_name, gif_paths[idx]))
    elif 'bbox_name' in features_df.columns:
        # No clusters, organize by bbox if available
        bboxes = features_df['bbox_name'].unique()
        bbox_samples = {bbox: [] for bbox in bboxes}
        
        for idx in gif_paths:
            bbox = features_df.loc[idx].get('bbox_name', 'unknown')
            bbox_samples[bbox].append((idx, gif_paths[idx]))
    else:
        # No organization, just list all
        samples = [(idx, gif_paths[idx]) for idx in gif_paths]
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>UMAP Visualization with Embedded GIFs</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 20px;
                border-radius: 8px;
            }
            h1 {
                text-align: center;
                color: #333;
            }
            h2 {
                margin-top: 30px;
                color: #555;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .gif-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .gif-item {
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .gif-item img {
                max-width: 100%;
                display: block;
            }
            .gif-info {
                padding: 10px;
                font-size: 14px;
            }
            .gif-info span {
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>UMAP Visualization - Sample GIFs</h1>
    """
    
    # Add samples organized by cluster or bbox
    if 'cluster' in features_df.columns:
        for cluster in clusters:
            if cluster_samples[cluster]:
                html_content += f"<h2>Cluster {cluster} ({len(cluster_samples[cluster])} samples)</h2>"
                html_content += '<div class="gif-grid">'
                
                for idx, bbox, gif_path in cluster_samples[cluster]:
                    html_content += f"""
                    <div class="gif-item">
                        <img src="file://{gif_path}" alt="Sample {idx}">
                        <div class="gif-info">
                            <span>Sample ID:</span> {idx}<br>
                            <span>BBox:</span> {bbox}
                        </div>
                    </div>
                    """
                
                html_content += '</div>'
    elif 'bbox_name' in features_df.columns:
        for bbox in bboxes:
            if bbox_samples[bbox]:
                html_content += f"<h2>BBox {bbox} ({len(bbox_samples[bbox])} samples)</h2>"
                html_content += '<div class="gif-grid">'
                
                for idx, gif_path in bbox_samples[bbox]:
                    html_content += f"""
                    <div class="gif-item">
                        <img src="file://{gif_path}" alt="Sample {idx}">
                        <div class="gif-info">
                            <span>Sample ID:</span> {idx}
                        </div>
                    </div>
                    """
                
                html_content += '</div>'
    else:
        html_content += "<h2>All Samples</h2>"
        html_content += '<div class="gif-grid">'
        
        for idx, gif_path in samples:
            html_content += f"""
            <div class="gif-item">
                <img src="file://{gif_path}" alt="Sample {idx}">
                <div class="gif-info">
                    <span>Sample ID:</span> {idx}
                </div>
            </div>
            """
        
        html_content += '</div>'
    
    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except UnicodeEncodeError:
        # If utf-8 fails, try with a more compatible encoding
        with open(html_path, 'w', encoding='ascii', errors='xmlcharrefreplace') as f:
            f.write(html_content)
    
    return html_path

def create_matplotlib_gif_visualization(features_df, gif_paths, output_dir):
    """
    Create a simple matplotlib-based visualization of GIFs at their UMAP coordinates.
    This is a much simpler alternative to the HTML-based visualization.
    
    Args:
        features_df: DataFrame with features and UMAP coordinates
        gif_paths: Dictionary mapping sample indices to GIF paths
        output_dir: Directory to save the output image
    
    Returns:
        Path to the output file
    """
    # Extract samples with GIFs
    samples_with_gifs = []
    for idx in gif_paths:
        if idx in features_df.index:
            sample = features_df.loc[idx]
            if 'umap_x' in sample and 'umap_y' in sample:
                x, y = sample['umap_x'], sample['umap_y']
                
                # Get cluster and bbox if available
                cluster = sample.get('cluster', 'N/A') if 'cluster' in sample else 'N/A'
                bbox = sample.get('bbox_name', 'unknown') if 'bbox_name' in sample else 'unknown'
                
                samples_with_gifs.append({
                    'id': idx,
                    'x': float(x),
                    'y': float(y),
                    'cluster': cluster,
                    'bbox': bbox,
                    'gif_path': gif_paths[idx]
                })
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Plot all points first
    if 'cluster' in features_df.columns:
        # Color by cluster
        scatter = plt.scatter(
            features_df['umap_x'], 
            features_df['umap_y'],
            c=features_df['cluster'], 
            cmap='tab10', 
            alpha=0.5,
            s=30
        )
        plt.colorbar(scatter, label='Cluster')
    elif 'bbox_name' in features_df.columns:
        # Color by bounding box
        scatter = plt.scatter(
            features_df['umap_x'], 
            features_df['umap_y'],
            c=features_df['bbox_name'].astype('category').cat.codes, 
            cmap='tab10', 
            alpha=0.5,
            s=30
        )
        plt.colorbar(scatter, label='Bounding Box')
    else:
        # Just plot points
        plt.scatter(
            features_df['umap_x'], 
            features_df['umap_y'],
            alpha=0.5,
            s=30
        )
    
    # Convert GIFs to static images (first frame)
    for sample in samples_with_gifs:
        try:
            # Open the GIF
            gif = Image.open(sample['gif_path'])
            
            # Get the first frame
            frame = gif.convert('RGB')
            
            # For each GIF, just use the first frame as a thumbnail
            thumb_size = 100  # Size of thumbnail
            frame.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            
            # Convert to array and create an OffsetImage
            img_box = OffsetImage(frame, zoom=0.5)
            
            # Create an annotation box
            ab = AnnotationBbox(
                img_box, 
                (sample['x'], sample['y']),
                pad=0.1,
                bboxprops=dict(edgecolor='black')
            )
            
            # Add to plot
            plt.gca().add_artist(ab)
            
            # Add a text label
            plt.text(
                sample['x'], sample['y'] - 0.2,
                f"ID: {sample['id']}",
                ha='center', 
                va='top',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, pad=1)
            )
        except Exception as e:
            print(f"Error adding GIF for sample {sample['id']}: {e}")
    
    # Add titles and labels
    plt.title("UMAP Visualization with GIF Thumbnails")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / "umap_with_gif_thumbnails.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    # Also save an interactive figure that shows sample IDs on hover
    try:
        import mpld3
        from mpld3 import plugins
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot all points with different colors for different clusters or bboxes
        if 'cluster' in features_df.columns:
            scatter = ax.scatter(
                features_df['umap_x'], 
                features_df['umap_y'],
                c=features_df['cluster'], 
                cmap='tab10', 
                alpha=0.7,
                s=50
            )
            
            # Create a label list
            labels = [f"ID: {i}, Cluster: {c}" for i, c in zip(features_df.index, features_df['cluster'])]
        elif 'bbox_name' in features_df.columns:
            scatter = ax.scatter(
                features_df['umap_x'], 
                features_df['umap_y'],
                c=features_df['bbox_name'].astype('category').cat.codes, 
                cmap='tab10', 
                alpha=0.7,
                s=50
            )
            
            # Create a label list
            labels = [f"ID: {i}, BBox: {b}" for i, b in zip(features_df.index, features_df['bbox_name'])]
        else:
            scatter = ax.scatter(
                features_df['umap_x'], 
                features_df['umap_y'],
                alpha=0.7,
                s=50
            )
            
            # Create a label list
            labels = [f"ID: {i}" for i in features_df.index]
        
        # Add tooltip plugin
        tooltip = plugins.PointLabelTooltip(scatter, labels=labels)
        plugins.connect(fig, tooltip)
        
        # Add titles and labels
        ax.set_title("Interactive UMAP Visualization (Hover for Sample IDs)")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        
        # Save as HTML
        interactive_path = output_dir / "interactive_umap.html"
        mpld3.save_html(fig, str(interactive_path))
        plt.close(fig)
        
        print(f"Interactive visualization saved to {interactive_path}")
    except Exception as e:
        print(f"Error creating interactive visualization: {e}")
        print("Continuing with static visualization only")
    
    print(f"Static GIF thumbnail visualization saved to {fig_path}")
    return fig_path

def create_seaborn_gif_visualization(features_df, gif_paths, output_dir):
    """
    Create a very simple visualization using Seaborn with thumbnails from GIFs.
    This is the simplest approach for visualizing GIFs at coordinates.
    
    Args:
        features_df: DataFrame with features and UMAP coordinates
        gif_paths: Dictionary mapping sample indices to GIF paths
        output_dir: Directory to save the output image
    
    Returns:
        Path to the output files
    """
    import seaborn as sns
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a figure
    plt.figure(figsize=(14, 12))
    
    # Use Seaborn for a prettier scatter plot
    if 'cluster' in features_df.columns:
        # Plot with cluster colors
        sns.scatterplot(
            data=features_df,
            x='umap_x',
            y='umap_y',
            hue='cluster',
            palette='tab10',
            s=50,
            alpha=0.7
        )
    elif 'bbox_name' in features_df.columns:
        # Plot with bounding box colors
        sns.scatterplot(
            data=features_df,
            x='umap_x',
            y='umap_y',
            hue='bbox_name',
            palette='Set1',
            s=50,
            alpha=0.7
        )
    else:
        # Simple scatter plot
        sns.scatterplot(
            data=features_df,
            x='umap_x',
            y='umap_y',
            s=50,
            alpha=0.7
        )
    
    # Add GIF first frames as thumbnails
    for idx, gif_path in gif_paths.items():
        if idx in features_df.index:
            try:
                # Get coordinates
                x = features_df.loc[idx, 'umap_x']
                y = features_df.loc[idx, 'umap_y']
                
                # Open the GIF and get first frame
                gif = Image.open(gif_path)
                frame = gif.convert('RGB')
                
                # Create a smaller thumbnail
                thumb_size = 80
                frame.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
                
                # Add the thumbnail to the plot
                img_box = OffsetImage(frame, zoom=0.5)
                ab = AnnotationBbox(
                    img_box,
                    (x, y),
                    frameon=True,
                    bboxprops=dict(edgecolor='black', linewidth=2)
                )
                plt.gca().add_artist(ab)
                
                # Add a marker to highlight this point
                plt.plot(x, y, 'ko', markersize=8, alpha=0.5)
                
                # Add a small text label
                plt.text(
                    x, y + 0.2,
                    f"{idx}",
                    ha='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )
            except Exception as e:
                print(f"Error adding GIF thumbnail for sample {idx}: {e}")
                plt.text(x, y, f"Error loading GIF for ID: {idx}",
                            ha='center', va='center')
    
    # Add titles and improve appearance
    plt.title("UMAP Visualization with GIF Thumbnails", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # Add a grid for better orientation
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Improve the legend
    plt.legend(title='Clusters' if 'cluster' in features_df.columns else 'Bounding Boxes',
               fontsize=10, title_fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    sns_path = output_dir / "seaborn_umap_with_gifs.png"
    plt.savefig(sns_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Seaborn visualization with GIF thumbnails saved to {sns_path}")
    
    # Create a grid of GIF thumbnails for better viewing
    try:
        # Calculate grid dimensions
        n_gifs = len(gif_paths)
        cols = min(5, n_gifs)  # Maximum 5 columns
        rows = (n_gifs + cols - 1) // cols  # Ceiling division
        
        # Create a new figure for the grid
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Add thumbnails to grid
        for i, (idx, gif_path) in enumerate(gif_paths.items()):
            if i < len(axes):
                try:
                    # Get cluster or bbox info if available
                    if idx in features_df.index:
                        info = features_df.loc[idx]
                        cluster = info.get('cluster', 'N/A') if 'cluster' in info else 'N/A'
                        bbox = info.get('bbox_name', 'unknown') if 'bbox_name' in info else 'unknown'
                        
                        # Open the GIF and get first frame
                        gif = Image.open(gif_path)
                        frame = gif.convert('RGB')
                        
                        # Display the frame
                        axes[i].imshow(frame)
                        axes[i].set_title(f"ID: {idx}" + 
                                        (f"\nCluster: {cluster}" if cluster != 'N/A' else "") +
                                        (f"\nBBox: {bbox}" if bbox != 'unknown' else ""))
                        axes[i].axis('off')
                except Exception as e:
                    print(f"Error adding GIF to grid for sample {idx}: {e}")
                    axes[i].text(0.5, 0.5, f"Error loading GIF for ID: {idx}",
                                ha='center', va='center')
                    axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the grid
        grid_path = output_dir / "gif_thumbnails_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Grid of GIF thumbnails saved to {grid_path}")
        
        return sns_path, grid_path
    except Exception as e:
        print(f"Error creating grid of GIF thumbnails: {e}")
        return sns_path

def create_plotly_express_visualization(features_df, gif_paths, output_dir):
    """
    Create a simple interactive visualization using Plotly Express that encodes
    the first frame of each GIF as a base64 string for hover display.
    
    Args:
        features_df: DataFrame with features and UMAP coordinates
        gif_paths: Dictionary mapping sample indices to GIF paths
        output_dir: Directory to save the HTML file
    
    Returns:
        Path to the HTML file
    """
    import plotly.express as px
    from PIL import Image
    import base64
    from io import BytesIO
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a copy of the DataFrame for visualization
    plot_df = features_df.copy()
    
    # Add a column for GIF thumbnails (first frame encoded as base64)
    plot_df['has_gif'] = plot_df.index.isin(gif_paths.keys())
    plot_df['gif_thumbnail'] = None
    plot_df['sample_id'] = plot_df.index
    
    # For points with GIFs, add the thumbnail data
    for idx, gif_path in gif_paths.items():
        if idx in plot_df.index:
            try:
                # Open the GIF and get first frame
                gif = Image.open(gif_path)
                frame = gif.convert('RGB')
                
                # Create a smaller thumbnail
                thumb_size = 100
                frame.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
                
                # Convert the image to base64 for HTML embedding
                buffered = BytesIO()
                frame.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Store base64 string in the DataFrame
                plot_df.loc[idx, 'gif_thumbnail'] = img_str
            except Exception as e:
                print(f"Error creating thumbnail for sample {idx}: {e}")
    
    # Determine color column for plotting
    if 'cluster' in plot_df.columns:
        color_column = 'cluster'
        title = "UMAP Visualization by Cluster"
    elif 'bbox_name' in plot_df.columns:
        color_column = 'bbox_name'
        title = "UMAP Visualization by Bounding Box"
    else:
        color_column = 'has_gif'
        title = "UMAP Visualization"
    
    # Create a custom hover template
    hover_template = """
    <b>Sample ID:</b> %{customdata[0]}<br>
    """
    
    if 'cluster' in plot_df.columns:
        hover_template += "<b>Cluster:</b> %{customdata[1]}<br>"
    if 'bbox_name' in plot_df.columns:
        hover_template += "<b>Bounding Box:</b> %{customdata[2]}<br>"
    
    hover_template += """
    <img src='data:image/png;base64,%{customdata[3]}' width='150px'><br>
    <extra></extra>
    """
    
    # Prepare custom data for hover
    if 'cluster' in plot_df.columns and 'bbox_name' in plot_df.columns:
        plot_df['customdata'] = plot_df.apply(
            lambda row: [
                row.name,  # Sample ID
                row.get('cluster', ''),  # Cluster
                row.get('bbox_name', ''),  # Bounding Box
                row.get('gif_thumbnail', '')  # Base64 image
            ], 
            axis=1
        )
    elif 'cluster' in plot_df.columns:
        plot_df['customdata'] = plot_df.apply(
            lambda row: [
                row.name,  # Sample ID
                row.get('cluster', ''),  # Cluster
                '',  # No Bounding Box
                row.get('gif_thumbnail', '')  # Base64 image
            ], 
            axis=1
        )
    elif 'bbox_name' in plot_df.columns:
        plot_df['customdata'] = plot_df.apply(
            lambda row: [
                row.name,  # Sample ID
                '',  # No Cluster
                row.get('bbox_name', ''),  # Bounding Box
                row.get('gif_thumbnail', '')  # Base64 image
            ], 
            axis=1
        )
    else:
        plot_df['customdata'] = plot_df.apply(
            lambda row: [
                row.name,  # Sample ID
                '',  # No Cluster
                '',  # No Bounding Box
                row.get('gif_thumbnail', '')  # Base64 image
            ], 
            axis=1
        )
    
    # Create the scatter plot
    fig = px.scatter(
        plot_df,
        x='umap_x',
        y='umap_y',
        color=color_column,
        title=title,
        custom_data='customdata'
    )
    
    # Update traces with hover template, but only show images for points with GIFs
    fig.update_traces(
        hovertemplate=hover_template,
        marker=dict(
            size=plot_df['has_gif'] * 10 + 5,  # Larger markers for points with GIFs
            line=dict(width=2, color='black')
        ),
        selector=dict(mode='markers')
    )
    
    # Update layout for better appearance
    fig.update_layout(
        width=1000,
        height=800,
        template='plotly_white',
        legend=dict(
            title=color_column.capitalize() if color_column else '',
            bordercolor='Black',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Save as HTML file
    html_path = output_dir / "plotly_express_umap.html"
    fig.write_html(str(html_path))
    
    print(f"Plotly Express visualization saved to {html_path}")
    return html_path

def create_animated_gif_visualization(features_df, gif_paths, output_dir):
    """
    Create a simple HTML page that displays animated GIFs directly at their UMAP coordinates.
    The GIFs are embedded directly in the HTML as base64 data to avoid file:// protocol issues.
    
    Args:
        features_df: DataFrame with features and UMAP coordinates
        gif_paths: Dictionary mapping sample indices to GIF paths
        output_dir: Directory to save the HTML file
    
    Returns:
        Path to the HTML file
    """
    import base64
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract UMAP coordinates and other info for samples with GIFs
    samples_with_gifs = []
    for idx in gif_paths:
        if idx in features_df.index:
            sample = features_df.loc[idx]
            if 'umap_x' in sample and 'umap_y' in sample:
                x, y = sample['umap_x'], sample['umap_y']
                
                # Extract cluster and bbox information if available
                cluster = sample.get('cluster', 'N/A') if 'cluster' in sample else 'N/A'
                bbox = sample.get('bbox_name', 'unknown') if 'bbox_name' in sample else 'unknown'
                
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
    
    # Compute the bounds of the UMAP coordinates
    all_x_values = features_df['umap_x'].values
    all_y_values = features_df['umap_y'].values
    
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
        cmap = plt.colormaps['tab10']
        
        for i, cluster in enumerate(clusters):
            r, g, b, _ = cmap(i % 10)  # Use modulo to handle more than 10 clusters
            point_colors[cluster] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    
    # Generate HTML content for data points
    points_content = ""
    for idx, row in features_df.iterrows():
        if 'umap_x' in row and 'umap_y' in row:
            x, y = row['umap_x'], row['umap_y']
            
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
        <title>Animated GIFs at UMAP Coordinates</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
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
                width: 1000px;
                height: 800px;
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
            }}
            .gif-container img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}
            .gif-label {{
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background-color: rgba(0,0,0,0.7);
                color: white;
                font-size: 10px;
                padding: 2px 5px;
                text-align: center;
            }}
            .controls {{
                margin-top: 10px;
                text-align: center;
            }}
            .controls button {{
                padding: 5px 10px;
                margin: 0 5px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            .controls button:hover {{
                background-color: #45a049;
            }}
            .gif-size-slider {{
                width: 200px;
                margin: 0 10px;
                vertical-align: middle;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>UMAP Visualization with Animated GIFs</h1>
            
            <div class="controls">
                <button id="toggle-gifs">Show/Hide GIFs</button>
                <button id="resize-gifs">Resize All GIFs</button>
                <input type="range" min="50" max="200" value="100" id="gif-size-slider" class="gif-size-slider">
                <span id="size-value">100px</span>
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
                }});
            }}
            
            // Add GIFs to the plot
            function addGifs() {{
                // Samples with GIFs
                const samplesWithGifs = [{gifs_content}
                ];
                
                // Add GIFs to the plot
                samplesWithGifs.forEach(sample => {{
                    const [plotX, plotY] = mapToPlot(sample.x, sample.y, plotWidth, plotHeight);
                    
                    // Create GIF container
                    const gifContainer = document.createElement('div');
                    gifContainer.className = 'gif-container';
                    gifContainer.style.left = `${{plotX}}px`;
                    gifContainer.style.top = `${{plotY}}px`;
                    gifContainer.style.width = '100px';
                    gifContainer.style.height = '100px';
                    
                    // Create GIF image using base64 data
                    const gifImg = document.createElement('img');
                    gifImg.src = `data:image/gif;base64,${{sample.gifData}}`;
                    gifImg.alt = `Sample ${{sample.id}}`;
                    
                    // Create label
                    const label = document.createElement('div');
                    label.className = 'gif-label';
                    label.textContent = `ID: ${{sample.id}}` + 
                                      (sample.cluster !== 'N/A' ? `, Cluster: ${{sample.cluster}}` : '') +
                                      (sample.bbox !== 'unknown' ? `, BBox: ${{sample.bbox}}` : '');
                    
                    // Add to container
                    gifContainer.appendChild(gifImg);
                    gifContainer.appendChild(label);
                    
                    // Add to plot
                    plot.appendChild(gifContainer);
                }});
                
                // Store all gif containers for later use
                window.gifContainers = document.querySelectorAll('.gif-container');
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
                    sizeValue.textContent = `${{size}}px`;
                    
                    window.gifContainers.forEach(container => {{
                        container.style.width = `${{size}}px`;
                        container.style.height = `${{size}}px`;
                    }});
                }});
                
                // Set up resize button
                const resizeButton = document.getElementById('resize-gifs');
                resizeButton.addEventListener('click', () => {{
                    const size = sizeSlider.value;
                    
                    window.gifContainers.forEach(container => {{
                        container.style.width = `${{size}}px`;
                        container.style.height = `${{size}}px`;
                    }});
                }});
            }}
            
            // Run initialization
            window.onload = init;
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    html_path = output_dir / "animated_gifs_visualization.html"
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

if __name__ == "__main__":
    # Parse configuration
    config.parse_args()
    
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
    
    # Create UMAP visualization with GIFs
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
        # First, create GIFs for samples and get paths
        print("Creating UMAP visualization with GIFs for a few random samples...")
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
        
        # Compute 2D UMAP if not already in the DataFrame
        if 'umap_x' not in features_df.columns or 'umap_y' not in features_df.columns:
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
                
                features_df['umap_x'] = umap_results[:, 0]
                features_df['umap_y'] = umap_results[:, 1]
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
            
            # If there are clusters, get samples from each
            num_samples = 10  # Reduced number of samples to show with GIFs
            
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
                sample_count = min(num_samples, len(valid_indices))
                sample_count = 40
                random_samples = np.random.choice(valid_indices, size=sample_count, replace=False)
            
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
            
            # Skip the original visualization that takes longer
            # html_path = create_umap_with_gifs(features_df, dataset, output_path, num_samples=25, random_seed=42)
            
            # Now create visualizations with our simpler methods if we have GIFs
            if gif_paths:
                print("\nCreating simpler matplotlib visualization with GIF thumbnails...")
                try:
                    matplotlib_path = create_matplotlib_gif_visualization(features_df, gif_paths, output_dir)
                    print(f"Matplotlib visualization created at {matplotlib_path}")
                except Exception as e:
                    print(f"Error creating matplotlib visualization: {e}")
                    
                print("\nCreating seaborn visualization with GIF thumbnails...")
                try:
                    seaborn_paths = create_seaborn_gif_visualization(features_df, gif_paths, output_dir)
                    if isinstance(seaborn_paths, tuple):
                        print(f"Seaborn visualization created with both scatter plot and grid view.")
                    else:
                        print(f"Seaborn visualization created at {seaborn_paths}")
                except Exception as e:
                    print(f"Error creating Seaborn visualization: {e}")
                    
                print("\nCreating simple Plotly Express visualization...")
                try:
                    plotly_path = create_plotly_express_visualization(features_df, gif_paths, output_dir)
                    print(f"Plotly Express visualization created at {plotly_path}")
                    print(f"This visualization is interactive - open in any browser to explore!")
                except Exception as e:
                    print(f"Error creating Plotly Express visualization: {e}")
                
                print(f"\nVisualizations created in {output_dir}")
                
                print("\nCreating animated GIF visualization...")
                try:
                    animated_path = create_animated_gif_visualization(features_df, gif_paths, output_dir)
                    print(f"Animated GIF visualization created at {animated_path}")
                    print(f"Open this in your browser to see animated GIFs directly at their UMAP coordinates.")
                except Exception as e:
                    print(f"Error creating animated GIF visualization: {e}")
            else:
                print("No GIFs were created successfully. Skipping additional visualizations.")
        else:
            print("No valid indices found. Skipping GIF creation and visualizations.")
    else:
        print("Warning: Could not initialize dataset. Skipping UMAP with GIFs visualization.")
        print("If you want to create the visualization, please ensure your config has valid paths for:")
        print("- raw_base_dir: The directory containing raw volumes")
        print("- seg_base_dir: The directory containing segmentation volumes")
        print("- add_mask_base_dir: The directory containing additional mask volumes (optional)")
        print("- excel_file: The directory containing Excel files with synapse data")
