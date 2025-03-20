import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import random
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
from PIL import Image
import json
import base64

def create_center_slice_from_volume(volume, output_path, cleft_mask=None):
    """
    Create a PNG image from the slice with most cleft mask pixels (or center slice if no mask provided).
    
    Args:
        volume: A 3D numpy array or torch tensor.
        output_path: Path to save the slice image.
        cleft_mask: Optional cleft mask tensor/array matching volume dimensions.
    
    Returns:
        The output_path after saving the image, or None if processing failed.
    """
    try:
        # Convert torch tensor to numpy if needed
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().detach().numpy()
        if not isinstance(volume, np.ndarray):
            raise TypeError(f"Volume must be a numpy array or torch tensor, got {type(volume)}")
            
        # Print full details of the volume for debugging
        print(f"Processing volume with shape {volume.shape}, dtype {volume.dtype}, min {volume.min()}, max {volume.max()}")
        
        # Handle various volume formats
        if volume.ndim > 3:
            volume = np.squeeze(volume)
            
        # Special case for (1, 1, N) format - this is the problematic format we need to handle
        if (volume.ndim == 3 and volume.shape[0] == 1 and volume.shape[1] == 1):
            print(f"Handling special case of shape {volume.shape}, dtype {volume.dtype}")
            # Extract the 1D array and reshape to 2D square
            linear_data = volume[0, 0, :]
            side_length = int(np.ceil(np.sqrt(linear_data.size)))
            
            # Create a square 2D array
            padded_size = side_length * side_length
            padded_data = np.zeros(padded_size, dtype=np.float32)  # Always use float32 for consistency
            padded_data[:linear_data.size] = linear_data.astype(np.float32)  # Convert to float32
            slice_img = padded_data.reshape(side_length, side_length)
            
            print(f"Reshaped {volume.shape} to 2D array of shape {slice_img.shape}, dtype {slice_img.dtype}")
        # If it's still not 3D, try to handle other formats
        elif volume.ndim != 3:
            if volume.ndim == 2:
                # If it's a 2D image already, keep it as is but ensure float type
                slice_img = volume.astype(np.float32)
                print(f"Using 2D image of shape {volume.shape}, converted to dtype {slice_img.dtype}")
            else:
                # For any other format, try to convert to 2D
                print(f"Warning: Unexpected volume format: {volume.shape}, ndim={volume.ndim}")
                try:
                    # Try to reshape to a square-ish 2D array
                    total_elements = volume.size
                    side_length = int(np.sqrt(total_elements))
                    slice_img = volume.reshape(side_length, -1).astype(np.float32)
                    print(f"Reshaped to 2D array of size {slice_img.shape}, dtype {slice_img.dtype}")
                except Exception as e:
                    print(f"Failed to reshape: {e}")
                    # Last resort: just flatten it to a single row
                    slice_img = volume.reshape(1, -1).astype(np.float32)
                    print(f"Flattened to 1D array of size {slice_img.shape}, dtype {slice_img.dtype}")
        else:
            # Normal 3D volume processing
            # Select the slice with most cleft mask pixels or center slice if no mask provided
            if cleft_mask is not None:
                # Convert cleft mask to numpy if needed
                if isinstance(cleft_mask, torch.Tensor):
                    cleft_mask = cleft_mask.cpu().detach().numpy()
                if cleft_mask.ndim > 3:
                    cleft_mask = np.squeeze(cleft_mask)
                    
                # Count cleft pixels in each slice
                cleft_pixels_per_slice = [np.sum(cleft_mask[i]) for i in range(cleft_mask.shape[0])]
                # Select slice with most cleft pixels
                if max(cleft_pixels_per_slice) > 0:
                    best_slice_idx = np.argmax(cleft_pixels_per_slice)
                    print(f"Selected slice {best_slice_idx} with {cleft_pixels_per_slice[best_slice_idx]} cleft pixels")
                else:
                    # If no cleft pixels found, fall back to center slice
                    best_slice_idx = volume.shape[0] // 2
                    print(f"No cleft pixels found, using center slice {best_slice_idx}")
            else:
                # If no cleft mask provided, use center slice
                best_slice_idx = volume.shape[0] // 2
                print(f"No cleft mask provided, using center slice {best_slice_idx}")
            
            # Extract the selected slice and ensure it's float type
            slice_img = volume[best_slice_idx].astype(np.float32)
        
        # Normalize the slice (using its min and max)
        vol_min, vol_max = slice_img.min(), slice_img.max()
        print(f"Global normalization: min={vol_min:.4f}, max={vol_max:.4f}, range={vol_max - vol_min:.4f}")
        if vol_max - vol_min > 0:
            norm_img = (slice_img - vol_min) / (vol_max - vol_min)
        else:
            norm_img = slice_img * 0
        
        # Convert to uint8 for image saving
        norm_img = (norm_img * 255).astype(np.uint8)
        
        # Save as PNG with robust error handling
        try:
            image = Image.fromarray(norm_img)
            image.save(output_path)
            print(f"Successfully saved image with shape {norm_img.shape} to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving image: {e}")
            # Try to convert to a compatible format
            try:
                # Mode 'L' (8-bit pixels, black and white)
                image = Image.fromarray(norm_img, mode='L')
                image.save(output_path)
                print(f"Saved image with explicit mode='L' to {output_path}")
                return output_path
            except Exception as e2:
                print(f"Failed with mode='L': {e2}")
                try:
                    # Try to convert to RGB
                    rgb_img = np.stack([norm_img, norm_img, norm_img], axis=2)
                    image = Image.fromarray(rgb_img, mode='RGB')
                    image.save(output_path)
                    print(f"Saved image with mode='RGB' to {output_path}")
                    return output_path
                except Exception as e3:
                    print(f"All image save attempts failed: {e3}")
                    raise
    except Exception as e:
        print(f"Error processing volume: {e}")
        return None

def create_clickable_image_visualization(features_df, image_paths, output_dir, dim_reduction='umap', canvas_width=1200, canvas_height=900, default_point_size=5, highlight_point_size=7):
    """
    Create an HTML visualization where clicking on a point displays its center slice image.
    
    Args:
        features_df: DataFrame with features and coordinates.
        image_paths: Dictionary mapping sample indices to image file paths.
        output_dir: Directory to save the HTML file.
        dim_reduction: 'umap' or 'tsne'; used to decide which coordinate columns to use.
        canvas_width: Width of the canvas for visualization.
        canvas_height: Height of the canvas for visualization.
        default_point_size: Default size for points in the visualization.
        highlight_point_size: Size for points with images in the visualization.
    
    Returns:
        Path to the generated HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert image paths to data URLs for browser compatibility
    image_data_urls = {}
    for idx, path in image_paths.items():
        try:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                image_data_urls[idx] = f"data:image/png;base64,{encoded_string}"
                print(f"Converted image {idx} to data URL")
        except Exception as e:
            print(f"Error converting image {idx} to data URL: {e}")
    
    # Use generic 'x' and 'y' if available; otherwise, use method-specific coordinates.
    if 'x' in features_df.columns and 'y' in features_df.columns:
        x_col, y_col = 'x', 'y'
    elif dim_reduction == 'umap' and 'umap_x' in features_df.columns and 'umap_y' in features_df.columns:
        x_col, y_col = 'umap_x', 'umap_y'
    elif dim_reduction == 'tsne' and 'tsne_x' in features_df.columns and 'tsne_y' in features_df.columns:
        x_col, y_col = 'tsne_x', 'tsne_y'
    else:
        raise ValueError("No suitable coordinate columns found in features_df.")
    
    # Print available columns for debugging
    print(f"Available columns in CSV: {features_df.columns.tolist()}")
    
    # Try to identify a good column to use for var1
    var1_column = None
    if 'Var1' in features_df.columns:
        var1_column = 'Var1'
        print(f"Found 'Var1' column in the CSV, using it for var1 values")
    elif 'var1' in features_df.columns:
        var1_column = 'var1'
        print(f"Found 'var1' column in the CSV, using it for var1 values")
    else:
        # Try alternative column names
        for col in ['var_1', 'variable1', 'param1', 'parameter1']:
            if col in features_df.columns:
                var1_column = col
                print(f"Using '{col}' column for var1 values")
                break
        
        # If still not found, look for feature columns
        if var1_column is None:
            feature_cols = [col for col in features_df.columns if col.startswith('feat_') or 'layer' in col]
            if feature_cols:
                var1_column = feature_cols[0]
                print(f"No var1 column found, using feature column '{var1_column}' for var1 values")
            # Or use the first numerical column that's not x, y, or cluster
            else:
                for col in features_df.columns:
                    if col not in [x_col, y_col, 'cluster'] and pd.api.types.is_numeric_dtype(features_df[col]):
                        var1_column = col
                        print(f"No var1 or feature columns found, using numerical column '{var1_column}' for var1 values")
                        break
    
    if var1_column is None:
        print("No suitable column found for var1 values, using index as var1")
        var1_column = 'index'  # We'll use row index as fallback
    else:
        print(f"Using column '{var1_column}' for var1 values")
    
    # Determine bounds from the coordinate columns and add some padding.
    all_x = features_df[x_col].values
    all_y = features_df[y_col].values
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad

    def map_to_canvas(x, y):
        canvas_x = ((x - x_min) / (x_max - x_min)) * canvas_width
        # Invert y-axis so higher data-y values appear toward the top of the canvas
        canvas_y = ((y_max - y) / (y_max - y_min)) * canvas_height
        return canvas_x, canvas_y

    # Create an array of point objects for visualization.
    points = []
    for idx, row in features_df.iterrows():
        x, y = row[x_col], row[y_col]
        canvas_x, canvas_y = map_to_canvas(x, y)
        
        # Default color and size
        color = "#888888"
        point_size = default_point_size
        
        # Assign color based on cluster if available
        if 'cluster' in features_df.columns:
            cluster = row['cluster']
            # Use a better color map for visibility and aesthetics
            cluster_colors = [
                "#4285F4", "#EA4335", "#FBBC05", "#34A853",  # Google colors
                "#9C27B0", "#3F51B5", "#03A9F4", "#009688", 
                "#8BC34A", "#FFEB3B", "#FF9800", "#795548"
            ]
            color_idx = int(cluster) % len(cluster_colors)
            color = cluster_colors[color_idx]
            
        # Adjust size - make points with images slightly larger
        if idx in image_data_urls:
            point_size = highlight_point_size
        
        # Get additional metadata for hovering
        bbox_name = row.get('bbox_name', 'Unknown')
        if isinstance(bbox_name, (int, float, np.number)):
            bbox_name = f"bbox_{bbox_name}"
                
        # Get var1 value if available 
        var1 = row.get(var1_column, None)
        
        # Special case for when we're using the index as var1
        if var1_column == 'index':
            var1 = f'sample_{idx}'
        
        # Convert to string and set fallback
        var1 = str(var1) if var1 is not None else f'sample_{idx}'
        
        points.append({
            'id': int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            'x': float(canvas_x),
            'y': float(canvas_y),
            'color': color,
            'size': point_size,
            'has_image': idx in image_data_urls,
            'cluster': str(row.get('cluster', 'none')),
            'bbox_name': str(bbox_name),
            'var1': var1
        })
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    # Don't replace quotes - this causes JSON parsing issues
    points_json = json.dumps(points, cls=NumpyEncoder)
    
    # Create image samples data from image_paths.
    image_samples = []
    for idx, url in image_data_urls.items():
        image_samples.append({"id": int(idx) if isinstance(idx, (int, np.integer)) else str(idx), "url": url})
    
    image_samples_json = json.dumps(image_samples, cls=NumpyEncoder)
    
    # Create HTML content.
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>UMAP Visualization with Hover Info</title>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                background-color: #f8f9fa;
                overflow: hidden;
            }}
            #visualization-container {{
                position: relative;
                width: 100vw;
                height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background-color: #f8f9fa;
            }}
            #title-bar {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                background-color: #343a40;
                color: white;
                padding: 10px 20px;
                font-size: 18px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                z-index: 100;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .title {{
                margin: 0;
            }}
            #info-statistics {{
                font-size: 14px;
                font-weight: normal;
                margin-left: 20px;
            }}
            #canvas-wrapper {{
                position: relative;
                margin-top: 60px;
                width: {canvas_width}px;
                height: {canvas_height}px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            #visualization-canvas {{
                position: absolute;
                top: 0;
                left: 0;
            }}
            .popup-image {{
                position: absolute;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);
                z-index: 1000;
                display: none;
                flex-direction: column;
                pointer-events: auto;
                width: 270px;
                cursor: move;
            }}
            .popup-image img {{
                max-width: 100%;
                height: auto;
                object-fit: contain;
                cursor: default;
            }}
            .popup-info {{
                margin-top: 8px;
                padding: 6px;
                background-color: #f1f3f5;
                border-radius: 4px;
                font-size: 12px;
                text-align: center;
                cursor: default;
            }}
            .resize-controls {{
                margin-top: 8px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 5px;
            }}
            .resize-controls label {{
                font-size: 12px;
                margin-right: 8px;
            }}
            .resize-slider {{
                flex-grow: 1;
                cursor: pointer;
            }}
            .global-controls {{
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background-color: white;
                border-radius: 6px;
                padding: 8px 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                z-index: 100;
                display: flex;
                align-items: center;
            }}
            .control-item {{
                display: flex;
                align-items: center;
                margin: 0 10px;
            }}
            .control-item label {{
                margin-right: 10px;
                font-size: 13px;
                white-space: nowrap;
            }}
            .control-item input[type="range"] {{
                width: 120px;
                margin: 0 10px;
            }}
            .control-item span {{
                font-size: 13px;
                min-width: 40px;
                text-align: right;
            }}
            .tooltip {{
                position: absolute;
                background-color: rgba(33, 37, 41, 0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                display: none;
                z-index: 1000;
                max-width: 200px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            .legend {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                background-color: white;
                border-radius: 6px;
                padding: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                z-index: 100;
                max-width: 300px;
            }}
            .legend-title {{
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 8px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 4px 0;
            }}
            .color-dot {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .instructions {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                background-color: white;
                border-radius: 6px;
                padding: 10px 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                z-index: 100;
                font-size: 13px;
                max-width: 250px;
                line-height: 1.4;
            }}
            .help-title {{
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .close-button {{
                position: absolute;
                top: 5px;
                right: 5px;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                line-height: 1;
                color: #555;
                pointer-events: auto;
            }}
        </style>
    </head>
    <body>
        <div id="visualization-container">
            <div id="title-bar">
                <div class="title">{dim_reduction.upper()} Visualization
                    <span id="info-statistics">
                        Points: {len(features_df)} | Images: {len(image_data_urls)}
                    </span>
                </div>
            </div>
            
            <div id="canvas-wrapper">
                <canvas id="visualization-canvas" width="{canvas_width}" height="{canvas_height}"></canvas>
            </div>
            
            <div id="popup-container"></div>
            
            <div id="tooltip" class="tooltip"></div>
            
            <div class="legend">
                <div class="legend-title">Clusters</div>
                <div id="legend-content"></div>
            </div>
            
            <div class="global-controls">
                <div class="control-item">
                    <label>Popup Size:</label>
                    <input type="range" id="global-size-slider" min="10" max="200" value="100" step="5" />
                    <span id="global-size-value">100%</span>
                </div>
                <div class="control-item">
                    <label>Point Size:</label>
                    <input type="range" id="point-size-slider" min="10" max="100" value="100" step="5" />
                    <span id="point-size-value">100%</span>
                </div>
            </div>
            
            <div class="instructions">
                <div class="help-title">How to use:</div>
                <ul style="padding-left: 15px; margin: 5px 0;">
                    <li>Hover over points to see details</li>
                    <li>Click on a point to see its image</li>
                    <li>Points with images have darker outlines</li>
                    <li>Multiple images can be viewed at once</li>
                    <li>Click the X to close an image</li>
                    <li>Drag images to reposition them</li>
                    <li>Use the size slider to resize all images</li>
                    <li>Adjust point sizes with the point slider</li>
                </ul>
            </div>
        </div>
        
        <script>
            // Define visualization data
            const points = {points_json};
            const imageSamples = {image_samples_json};
            
            // Setup canvas and context
            const canvas = document.getElementById('visualization-canvas');
            const ctx = canvas.getContext('2d');
            
            // Setup popup container
            const popupContainer = document.getElementById('popup-container');
            const tooltip = document.getElementById('tooltip');
            
            // Map of sample ID to image URL
            const imageUrls = {{}};
            imageSamples.forEach(sample => {{
                imageUrls[sample.id] = sample.url;
            }});
            
            // Track active popups
            const activePopups = new Set();
            
            // Global settings
            const globalSettings = {{
                popupSizePercent: 100,
                basePopupWidth: 270,
                baseImageWidth: 250,
                pointSizePercent: 100,
                basePointSize: 5,
                baseHighlightSize: 7
            }};
            
            // Create cluster map for legend
            const clusters = {{}};
            points.forEach(point => {{
                if (!clusters[point.cluster]) {{
                    clusters[point.cluster] = {{
                        color: point.color,
                        count: 1,
                        hasImages: point.has_image ? 1 : 0
                    }};
                }} else {{
                    clusters[point.cluster].count++;
                    if (point.has_image) clusters[point.cluster].hasImages++;
                }}
            }});
            
            // Create legend
            const legendContent = document.getElementById('legend-content');
            Object.keys(clusters).sort((a, b) => parseInt(a) - parseInt(b)).forEach(cluster => {{
                const item = document.createElement('div');
                item.className = 'legend-item';
                
                const colorDot = document.createElement('div');
                colorDot.className = 'color-dot';
                colorDot.style.backgroundColor = clusters[cluster].color;
                
                const label = document.createElement('div');
                label.textContent = "Cluster " + cluster + ": " + clusters[cluster].count + " points (" + clusters[cluster].hasImages + " images)";
                
                item.appendChild(colorDot);
                item.appendChild(label);
                legendContent.appendChild(item);
            }});
            
            // Draw all points
            function drawVisualization() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Add light grid for better visual structure
                ctx.strokeStyle = '#f0f0f0';
                ctx.lineWidth = 1;
                
                // Draw grid
                const gridSize = 50;
                for (let x = 0; x <= canvas.width; x += gridSize) {{
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, canvas.height);
                    ctx.stroke();
                }}
                
                for (let y = 0; y <= canvas.height; y += gridSize) {{
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(canvas.width, y);
                    ctx.stroke();
                }}
                
                // Draw points
                points.forEach(point => {{
                    // Calculate point size based on global settings
                    const scaledSize = point.size * (globalSettings.pointSizePercent / 100);
                    
                    // Draw point with shadow for depth
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, scaledSize, 0, 2 * Math.PI);
                    ctx.fillStyle = point.color;
                    ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
                    ctx.shadowBlur = 2;
                    ctx.shadowOffsetX = 1;
                    ctx.shadowOffsetY = 1;
                    ctx.fill();
                    ctx.shadowColor = 'transparent';
                    
                    // Add border to points with images
                    if (point.has_image) {{
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, scaledSize + 1.5, 0, 2 * Math.PI);
                        ctx.strokeStyle = '#333';
                        ctx.lineWidth = 1.5;
                        ctx.stroke();
                    }}
                }});
            }}
            
            // Initial draw
            drawVisualization();
            
            // Setup global size slider
            const globalSizeSlider = document.getElementById('global-size-slider');
            const globalSizeValue = document.getElementById('global-size-value');
            
            globalSizeSlider.addEventListener('input', function() {{
                // Update global settings
                globalSettings.popupSizePercent = parseInt(this.value);
                globalSizeValue.textContent = globalSettings.popupSizePercent + '%';
                
                // Update all existing popups
                resizeAllPopups();
            }});
            
            // Setup point size slider
            const pointSizeSlider = document.getElementById('point-size-slider');
            const pointSizeValue = document.getElementById('point-size-value');
            
            pointSizeSlider.addEventListener('input', function() {{
                // Update global settings
                globalSettings.pointSizePercent = parseInt(this.value);
                pointSizeValue.textContent = globalSettings.pointSizePercent + '%';
                
                // Redraw all points with new sizes
                drawVisualization();
            }});
            
            // Function to resize all popups
            function resizeAllPopups() {{
                const popups = document.querySelectorAll('.popup-image');
                
                popups.forEach(popup => {{
                    // Calculate new dimensions
                    const newWidth = (globalSettings.basePopupWidth * globalSettings.popupSizePercent / 100);
                    const img = popup.querySelector('img');
                    
                    // Resize the popup
                    popup.style.width = newWidth + 'px';
                    
                    // Resize the image if needed
                    if (img) {{
                        const newImgWidth = (globalSettings.baseImageWidth * globalSettings.popupSizePercent / 100);
                        img.style.width = newImgWidth + 'px';
                    }}
                }});
            }}
            
            // Handle hover
            canvas.addEventListener('mousemove', (event) => {{
                const rect = canvas.getBoundingClientRect();
                const mouseX = event.clientX - rect.left;
                const mouseY = event.clientY - rect.top;
                
                // Find closest point within hover range
                let hoverPoint = null;
                let closestDistance = Infinity;
                const baseHoverThreshold = 15; // Base hover detection area
                
                points.forEach(point => {{
                    const dx = point.x - mouseX;
                    const dy = point.y - mouseY;
                    const distance = Math.sqrt(dx*dx + dy*dy);
                    
                    // Scale the hover threshold based on point size
                    const scaledHoverThreshold = baseHoverThreshold * (globalSettings.pointSizePercent / 100);
                    
                    if (distance < scaledHoverThreshold && distance < closestDistance) {{
                        hoverPoint = point;
                        closestDistance = distance;
                    }}
                }});
                
                // Show or hide tooltip
                if (hoverPoint) {{
                    tooltip.style.display = 'block';
                    tooltip.style.left = `${{event.clientX + 12}}px`;
                    tooltip.style.top = `${{event.clientY + 12}}px`;
                    tooltip.innerHTML = `
                        <div><strong>Var1:</strong> ${{hoverPoint.var1 || 'N/A'}}</div>
                        <div><strong>BBox:</strong> ${{hoverPoint.bbox_name}}</div>
                        ${{hoverPoint.has_image ? 
                            '<div style="margin-top:4px;color:#7ff07f"><strong>Click to view image</strong></div>' : 
                            '<div style="margin-top:4px;color:#f0a07f">No image available</div>'}}
                    `;
                    
                    // Change cursor to pointer if it has an image
                    canvas.style.cursor = hoverPoint.has_image ? 'pointer' : 'default';
                }} else {{
                    tooltip.style.display = 'none';
                    canvas.style.cursor = 'default';
                }}
            }});
            
            // Hide tooltip when mouse leaves canvas
            canvas.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});
            
            // Function to check if two rectangles overlap
            function checkOverlap(rect1, rect2) {{
                return !(rect1.right < rect2.left || 
                        rect1.left > rect2.right || 
                        rect1.bottom < rect2.top || 
                        rect1.top > rect2.bottom);
            }}
            
            // Function to make an element draggable
            function makeDraggable(element) {{
                let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
                let isDragging = false;
                let zIndexCounter = 1000; // Base z-index value
                
                // Set initial z-index
                element.style.zIndex = zIndexCounter++;
                
                // Mouse down event starts the drag
                element.addEventListener('mousedown', dragMouseDown);
                
                function dragMouseDown(e) {{
                    // Only start drag if it's not the close button or img or popup-info or resize controls
                    if (e.target.classList.contains('close-button') || 
                        e.target.tagName === 'IMG' ||
                        e.target.classList.contains('popup-info') ||
                        e.target.classList.contains('resize-controls') ||
                        e.target.classList.contains('resize-slider') ||
                        e.target.parentElement?.classList.contains('resize-controls')) {{
                        return;
                    }}
                    
                    e.preventDefault();
                    
                    // Bring this popup to the front
                    element.style.zIndex = zIndexCounter++;
                    
                    // Get the current mouse position
                    pos3 = e.clientX;
                    pos4 = e.clientY;
                    
                    isDragging = true;
                    
                    // Set cursor to grabbing during drag
                    element.style.cursor = 'grabbing';
                    
                    // Add event listeners for move and release
                    document.addEventListener('mousemove', elementDrag);
                    document.addEventListener('mouseup', closeDragElement);
                }}
                
                function elementDrag(e) {{
                    if (!isDragging) return;
                    
                    e.preventDefault();
                    
                    // Calculate the new cursor position
                    pos1 = pos3 - e.clientX;
                    pos2 = pos4 - e.clientY;
                    pos3 = e.clientX;
                    pos4 = e.clientY;
                    
                    // Set the element's new position
                    const newTop = (element.offsetTop - pos2);
                    const newLeft = (element.offsetLeft - pos1);
                    
                    // Keep the element within the viewport
                    const maxTop = window.innerHeight - element.offsetHeight;
                    const maxLeft = window.innerWidth - element.offsetWidth;
                    
                    element.style.top = Math.max(0, Math.min(maxTop, newTop)) + "px";
                    element.style.left = Math.max(0, Math.min(maxLeft, newLeft)) + "px";
                }}
                
                function closeDragElement() {{
                    // Stop moving when mouse button is released
                    isDragging = false;
                    
                    // Reset cursor
                    element.style.cursor = 'move';
                    
                    // Remove event listeners
                    document.removeEventListener('mousemove', elementDrag);
                    document.removeEventListener('mouseup', closeDragElement);
                }}
            }}
            
            // Function to find a non-overlapping position
            function findNonOverlappingPosition(x, y, size) {{
                const rect = {{
                    left: x,
                    top: y,
                    right: x + size,
                    bottom: y + size
                }};
                
                // Get all existing popup elements
                const existingPopups = document.querySelectorAll('.popup-image[style*="display: flex"]');
                let overlap = false;
                
                // Check against each existing popup
                existingPopups.forEach(existingPopup => {{
                    const bounds = existingPopup.getBoundingClientRect();
                    const existingRect = {{
                        left: bounds.left,
                        top: bounds.top,
                        right: bounds.right,
                        bottom: bounds.bottom
                    }};
                    
                    if (checkOverlap(rect, existingRect)) {{
                        overlap = true;
                        // Try shifting position
                        rect.left += size + 20;
                        rect.right += size + 20;
                        
                        // If off screen, move down instead
                        if (rect.right > window.innerWidth) {{
                            rect.left = 10;
                            rect.right = rect.left + size;
                            rect.top += size + 20;
                            rect.bottom += size + 20;
                        }}
                    }}
                }});
                
                return {{ left: rect.left, top: rect.top }};
            }}
            
            // Modify canvas click handler to improve popup behavior
            // Store the original click handler
            const originalClickHandler = canvas.onclick;
            
            // Replace with our enhanced handler
            canvas.addEventListener('click', (event) => {{
                const rect = canvas.getBoundingClientRect();
                const mouseX = event.clientX - rect.left;
                const mouseY = event.clientY - rect.top;
                
                // Find closest point
                let clickedPoint = null;
                let closestDistance = Infinity;
                const baseClickThreshold = 20;
                
                points.forEach(point => {{
                    const dx = point.x - mouseX;
                    const dy = point.y - mouseY;
                    const distance = Math.sqrt(dx*dx + dy*dy);
                    
                    // Scale the click threshold based on point size
                    const scaledClickThreshold = baseClickThreshold * (globalSettings.pointSizePercent / 100);
                    
                    if (distance < scaledClickThreshold && distance < closestDistance) {{
                        clickedPoint = point;
                        closestDistance = distance;
                    }}
                }});
                
                // Handle the click
                if (clickedPoint && clickedPoint.has_image) {{
                    // Update popup content
                    const popupId = clickedPoint.id;
                    const popupContent = `
                        <div class="close-button">×</div>
                        <img src="${{imageUrls[popupId]}}" alt="Sample image">
                        <div class="popup-info">
                            <strong>Var1:</strong> ${{clickedPoint.var1 || 'N/A'}} |
                            <strong>BBox:</strong> ${{clickedPoint.bbox_name}}
                        </div>
                    `;
                    
                    // Create new popup element
                    const newPopup = document.createElement('div');
                    newPopup.className = 'popup-image';
                    newPopup.dataset.id = popupId;
                    newPopup.innerHTML = popupContent;
                    
                    // Position popup near the clicked point, but keep it fully visible
                    const popupWidth = globalSettings.basePopupWidth * (globalSettings.popupSizePercent / 100);
                    const popupHeight = 300 * (globalSettings.popupSizePercent / 100);
                    
                    // Calculate initial position relative to canvas
                    let popupX = clickedPoint.x + rect.left + 15;
                    let popupY = clickedPoint.y + rect.top - 10;
                    
                    // Check right edge
                    if (popupX + popupWidth > window.innerWidth) {{
                        popupX = clickedPoint.x + rect.left - popupWidth - 15;
                    }}
                    
                    // Check bottom edge
                    if (popupY + popupHeight > window.innerHeight) {{
                        popupY = window.innerHeight - popupHeight - 10;
                    }}
                    
                    // Check top edge
                    if (popupY < 0) {{
                        popupY = 10;
                    }}
                    
                    // Find non-overlapping position from existing popups
                    const nonOverlappingPos = findNonOverlappingPosition(popupX, popupY, popupWidth);
                    
                    // Set popup position and size
                    newPopup.style.left = `${{nonOverlappingPos.left}}px`;
                    newPopup.style.top = `${{nonOverlappingPos.top}}px`;
                    newPopup.style.width = `${{popupWidth}}px`;
                    newPopup.style.display = 'flex';
                    newPopup.style.pointerEvents = 'auto'; // Make the popup clickable
                    
                    // Set the image size
                    const img = newPopup.querySelector('img');
                    if (img) {{
                        const imgWidth = globalSettings.baseImageWidth * (globalSettings.popupSizePercent / 100);
                        img.style.width = `${{imgWidth}}px`;
                    }}
                    
                    // Add new popup to container
                    popupContainer.appendChild(newPopup);
                    
                    // Add close button event listener for this popup
                    const closeButton = newPopup.querySelector('.close-button');
                    closeButton.addEventListener('click', (e) => {{
                        e.stopPropagation(); // Prevent event bubbling
                        newPopup.remove(); // Remove the popup from DOM
                        activePopups.delete(popupId);
                    }});
                    
                    // Make the popup draggable
                    makeDraggable(newPopup);
                    
                    // Add active class to points if needed
                    document.querySelectorAll('.point.active').forEach(p => {{
                        p.classList.remove('active');
                    }});
                    
                    // Mark clicked point as active
                    if (clickedPoint.element) {{
                        clickedPoint.element.classList.add('active');
                    }}
                    
                    // Track active popups
                    activePopups.add(popupId);
                }} else if (event.target === canvas) {{
                    // Only remove active state from points when clicking on empty space
                    // Don't close popups when clicking on empty canvas areas
                    document.querySelectorAll('.point.active').forEach(p => {{
                        p.classList.remove('active');
                    }});
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Write HTML file
    html_file_path = output_dir / "clickable_umap_visualization_slice.html"
    with open(html_file_path, "w") as file:
        file.write(html_content)
    
    return str(html_file_path)

def find_cleft_mask(volume, threshold=0.7):
    """
    Generate a cleft mask from the volume by identifying high intensity regions.
    
    Args:
        volume: 3D volume as numpy array or torch tensor
        threshold: Value between 0-1, higher values mean more restrictive mask
    
    Returns:
        Binary mask of the same shape as the volume
    """
    # Convert torch tensor to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().detach().numpy()
    
    # Normalize volume to 0-1 range for consistent thresholding
    volume_norm = volume.copy()
    vol_min, vol_max = volume_norm.min(), volume_norm.max()
    if vol_max > vol_min:
        volume_norm = (volume_norm - vol_min) / (vol_max - vol_min)
    
    # Create a binary mask using the threshold
    cleft_mask = volume_norm > threshold
    
    # Count total mask pixels to check if we have a reasonable mask
    num_cleft_pixels = np.sum(cleft_mask)
    total_pixels = volume.size
    cleft_percentage = (num_cleft_pixels / total_pixels) * 100
    print(f"Cleft mask: {num_cleft_pixels} pixels ({cleft_percentage:.2f}% of volume)")
    
    # If the mask is too sparse or too dense, adjust threshold automatically
    if cleft_percentage < 0.5:
        # Too few pixels, try lower threshold
        adjusted_threshold = threshold * 0.8
        print(f"Adjusting threshold from {threshold} to {adjusted_threshold} (too few cleft pixels)")
        return find_cleft_mask(volume, adjusted_threshold)
    elif cleft_percentage > 30:
        # Too many pixels, try higher threshold
        adjusted_threshold = min(threshold * 1.2, 0.95)
        print(f"Adjusting threshold from {threshold} to {adjusted_threshold} (too many cleft pixels)")
        return find_cleft_mask(volume, adjusted_threshold)
    
    return cleft_mask

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Clickable Image Visualization from Center Slices')
    parser.add_argument('--dim-reduction', choices=['umap', 'tsne'], default='umap',
                        help='Dimensionality reduction method to use')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='Number of random samples to process')
    args = parser.parse_args()
    
    # Define file paths (customize these as needed)
    csv_path = r"C:\Users\alim9\Documents\codes\synapse2\results\run_2025-03-14_15-50-53\features_layer20_seg10_alpha1.0\features_layer20_seg10_alpha1_0.csv"
    output_path = "results/test"
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features from CSV (the CSV is assumed to contain precomputed features and possibly UMAP/t-SNE coordinates)
    if os.path.exists(csv_path):
        print(f"Loading features from {csv_path}")
        # Explicitly set index_col to avoid issues with index type
        features_df = pd.read_csv(csv_path)
        # Ensure the index is of string type for consistency
        if 'Unnamed: 0' in features_df.columns:
            features_df.set_index('Unnamed: 0', inplace=True)
        # Try to convert index to numeric if possible
        try:
            features_df.index = pd.to_numeric(features_df.index)
        except (ValueError, TypeError):
            print("Warning: Could not convert index to numeric type. Using string index.")
    else:
        print("CSV file not found. Exiting.")
        sys.exit(1)
    
    # Compute UMAP coordinates if needed and if dim_reduction is set to UMAP.
    if args.dim_reduction == 'umap' and ('x' not in features_df.columns or 'y' not in features_df.columns):
        print("Computing UMAP coordinates...")
        feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
        if not feature_cols:
            feature_cols = [col for col in features_df.columns if 'layer' in col and 'feat_' in col]
        if feature_cols:
            features = features_df[feature_cols].values
            features_scaled = StandardScaler().fit_transform(features)
            reducer = umap.UMAP(n_components=2, random_state=42)
            umap_results = reducer.fit_transform(features_scaled)
            features_df['x'] = umap_results[:, 0]
            features_df['y'] = umap_results[:, 1]
            
            # Add clustering for coloring
            n_clusters = min(10, len(features_df))
            print(f"Performing KMeans clustering with {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            features_df['cluster'] = kmeans.fit_predict(features_scaled)
            print(f"Clusters: {sorted(features_df['cluster'].unique().tolist())}")
        else:
            print("No feature columns found for UMAP computation.")
            sys.exit(1)
    
    # Load or initialize the dataset.
    # First, try importing from the Clustering module.
    try:
        # Add the project root to sys.path if needed
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from Clustering import dataset
        print("Using dataset from Clustering module.")
    except ImportError as e:
        print(f"Dataset not available from Clustering module. Error: {e}")
        print("Creating a dummy dataset for visualization purposes.")
        
        # Create a simple dummy dataset
        class DummyDataset:
            def __init__(self, size=100, volume_shape=(32, 64, 64)):
                self.size = size
                self.volume_shape = volume_shape
                print(f"Created dummy dataset with {size} samples")
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                if idx >= self.size:
                    raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
                
                # Create a random volume with a pattern
                volume = np.zeros(self.volume_shape, dtype=np.float32)
                
                # Add some random noise
                volume += np.random.randn(*self.volume_shape) * 0.1
                
                # Add a circle in the center slice with radius based on the idx
                center = np.array([d // 2 for d in self.volume_shape])
                radius = max(3, int(idx % 10) + 3)
                
                x, y, z = np.ogrid[:self.volume_shape[0], :self.volume_shape[1], :self.volume_shape[2]]
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                volume[dist <= radius] = 1.0
                
                # Convert to torch tensor
                volume_tensor = torch.tensor(volume, dtype=torch.float32)
                
                return volume_tensor
        
        dataset = DummyDataset(size=100)
    
    # Select valid indices (those within the dataset length).
    try:
        dataset_length = len(dataset)
    except Exception as e:
        print(f"Error determining dataset length: {e}")
        sys.exit(1)
    
    # Ensure indices are integers before comparison
    valid_indices = []
    for i in features_df.index:
        try:
            # Try to convert index to int if it's a string
            idx = int(i) if isinstance(i, str) else i
            if idx < dataset_length:
                valid_indices.append(i)
        except (ValueError, TypeError):
            # Skip indices that can't be converted to int
            continue
    
    if not valid_indices:
        print("No valid indices found. Exiting.")
        sys.exit(1)
    
    np.random.seed(42)
    if len(valid_indices) >= args.num_samples:
        random_samples = np.random.choice(valid_indices, size=args.num_samples, replace=False)
    else:
        random_samples = valid_indices
    
    # Create center-slice images for the selected samples.
    image_paths = {}
    images_dir = output_dir / "sample_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in random_samples:
        try:
            sample_data = dataset[idx]
            if isinstance(sample_data, tuple) and len(sample_data) > 0:
                volume = sample_data[0]
                
                # Try to extract cleft mask from the second element (if available)
                cleft_mask = None
                if len(sample_data) > 1 and isinstance(sample_data[1], dict):
                    # If the sample data has a cleft mask directly available, use it
                    cleft_mask = sample_data[1].get('cleft_mask')
                
                # If no cleft mask in the data, generate one
                if cleft_mask is None:
                    print(f"Generating cleft mask for sample {idx}")
                    cleft_mask = find_cleft_mask(volume)
            elif isinstance(sample_data, dict):
                volume = sample_data.get('pixel_values', sample_data.get('raw_volume'))
                # Try to get cleft mask from the dictionary
                cleft_mask = sample_data.get('cleft_mask')
                
                # If no cleft mask in the dictionary, generate one
                if cleft_mask is None:
                    print(f"Generating cleft mask for sample {idx}")
                    cleft_mask = find_cleft_mask(volume)
            else:
                volume = sample_data
                # Generate cleft mask 
                print(f"Generating cleft mask for sample {idx}")
                cleft_mask = find_cleft_mask(volume)
                
            if volume is None:
                print(f"Skipping sample {idx}: No volume data")
                continue
                
            image_filename = f"sample_{idx}.png"
            image_path = images_dir / image_filename
            
            # Save the slice with the most cleft mask, checking for success
            saved_path = create_center_slice_from_volume(volume, str(image_path), cleft_mask)
            
            # Only add to image_paths if save was successful
            if saved_path is not None and os.path.exists(saved_path) and os.path.getsize(saved_path) > 0:
                image_paths[idx] = saved_path
                print(f"Added image for sample {idx} to visualization")
            else:
                print(f"Failed to create valid image for sample {idx}")
            
            # Save a visualization of the cleft mask for debugging
            if cleft_mask is not None:
                mask_filename = f"mask_{idx}.png"
                mask_path = images_dir / mask_filename
                try:
                    # Find the slice with the most cleft pixels
                    if isinstance(cleft_mask, torch.Tensor):
                        cleft_mask = cleft_mask.cpu().detach().numpy()
                    
                    # Handle masks with shape issues
                    if cleft_mask.ndim > 3:
                        cleft_mask = np.squeeze(cleft_mask)
                    
                    # Special case for (1, 1, N) shape
                    if cleft_mask.ndim == 3 and cleft_mask.shape[0] == 1 and cleft_mask.shape[1] == 1:
                        mask_slice = np.reshape(cleft_mask[0, 0, :], (int(np.sqrt(cleft_mask.shape[2])), -1))
                    else:
                        cleft_pixels_per_slice = [np.sum(cleft_mask[i]) for i in range(cleft_mask.shape[0])]
                        if max(cleft_pixels_per_slice) > 0:
                            best_slice_idx = np.argmax(cleft_pixels_per_slice)
                            mask_slice = cleft_mask[best_slice_idx]
                        else:
                            mask_slice = cleft_mask[cleft_mask.shape[0] // 2]
                    
                    # Convert to appropriate format for saving
                    mask_slice = mask_slice.astype(np.uint8) * 255
                    mask_image = Image.fromarray(mask_slice, mode='L')
                    mask_image.save(str(mask_path))
                    print(f"Saved cleft mask visualization for sample {idx}")
                except Exception as e:
                    print(f"Error saving mask image for sample {idx}: {e}")
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
    
    # Create the clickable HTML visualization using the center-slice images.
    print(f"Creating visualization with {len(image_paths)} image paths")
    
    # Debug check for features data
    print(f"Features DF shape: {features_df.shape}")
    if 'x' in features_df.columns and 'y' in features_df.columns:
        print(f"x range: {features_df['x'].min()} to {features_df['x'].max()}")
        print(f"y range: {features_df['y'].min()} to {features_df['y'].max()}")
    else:
        print(f"Available columns: {features_df.columns.tolist()}")
    
    # Only create visualization if we have image paths
    if image_paths:
        clickable_html = create_clickable_image_visualization(features_df, image_paths, output_dir, dim_reduction=args.dim_reduction)
        
        # Verify the output HTML exists and check its size
        if os.path.exists(clickable_html):
            print(f"HTML file size: {os.path.getsize(clickable_html)} bytes")
            print(f"Visualization created: {clickable_html}")
        else:
            print(f"Warning: HTML file was not created at {clickable_html}")
    else:
        print("No images were successfully created, skipping HTML visualization creation.")
