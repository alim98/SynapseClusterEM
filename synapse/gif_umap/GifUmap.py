import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the synapse module
# This needs to be before any synapse imports
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)

"""
All other functions have been commented out as they are not needed for the clickable GIF visualization.
If you need any of the other functionality, please uncomment the relevant functions.
"""

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
                    pointElem.style.left = `${{plotX}}px`;
                    pointElem.style.top = `${{plotY}}px`;
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
                }})
                """ for sample in samples_with_gifs])}
                
                // Store GIF data for later use
                gifData.push(...samplesWithGifs);
            }}
            
            // Show GIF for a specific sample
            function showGif(sampleId, x, y) {{
                // Hide any currently shown GIF
                hideAllGifs();
                
                // Find the GIF data for this sample
                const sample = gifData.find(s => s.id === sampleId);
                if (!sample) return;
                
                // Create GIF container if it doesn't exist
                let gifContainer = document.getElementById(`gif-${{sampleId}}`);
                if (!gifContainer) {{
                    gifContainer = document.createElement('div');
                    gifContainer.className = 'gif-container';
                    gifContainer.id = `gif-${{sampleId}}`;
                    
                    // Get the current size from the slider
                    const size = document.getElementById('gif-size-slider').value;
                    gifContainer.style.width = `${{size}}px`;
                    gifContainer.style.height = `${{size}}px`;
                    
                    // Create GIF image using base64 data
                    const gifImg = document.createElement('img');
                    gifImg.src = `data:image/gif;base64,${{sample.gifData}}`;
                    gifImg.alt = `Sample ${{sampleId}}`;
                    
                    // Create info element
                    const infoElem = document.createElement('div');
                    infoElem.className = 'gif-info';
                    infoElem.textContent = `ID: ${{sampleId}}, Cluster: ${{sample.cluster}}, BBox: ${{sample.bbox}}`;
                    
                    // Create close button
                    const closeBtn = document.createElement('button');
                    closeBtn.className = 'close-btn';
                    closeBtn.textContent = '×';
                    closeBtn.addEventListener('click', (e) => {{
                        e.stopPropagation();
                        hideAllGifs();
                        document.querySelectorAll('.point.active').forEach(p => {{
                            p.classList.remove('active');
                        }});
                    }});
                    
                    // Add elements to container
                    gifContainer.appendChild(gifImg);
                    gifContainer.appendChild(infoElem);
                    gifContainer.appendChild(closeBtn);
                    
                    // Add to plot
                    plot.appendChild(gifContainer);
                }}
                
                // Position the GIF container
                gifContainer.style.left = `${{x}}px`;
                gifContainer.style.top = `${{y}}px`;
                
                // Show the GIF
                gifContainer.style.display = 'block';
                
                // Store the active GIF container
                activeGifContainer = gifContainer;
            }}
            
            // Hide all GIFs
            function hideAllGifs() {{
                document.querySelectorAll('.gif-container').forEach(container => {{
                    container.style.display = 'none';
                }});
                activeGifContainer = null;
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
                    label.textContent = `Cluster ${{cluster}}`;
                    
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
            
            # Skip the original visualization that takes longer
            # html_path = create_umap_with_gifs(features_df, dataset, output_path, num_samples=args.num_samples, random_seed=42, dim_reduction=args.dim_reduction)
            
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
                    animated_path = create_animated_gif_visualization(features_df, gif_paths, output_dir, dim_reduction=args.dim_reduction)
                    print(f"Animated GIF visualization created at {animated_path}")
                    print(f"Open this in your browser to see animated GIFs directly at their {args.dim_reduction.upper()} coordinates.")
                except Exception as e:
                    print(f"Error creating animated GIF visualization: {e}")
                
                print("\nCreating clickable GIF visualization...")
                try:
                    clickable_path = create_clickable_gif_visualization(features_df, gif_paths, output_dir, dim_reduction=args.dim_reduction)
                    print(f"Clickable GIF visualization created at {clickable_path}")
                    print(f"Open this in your browser to click on points and see their GIFs.")
                except Exception as e:
                    print(f"Error creating clickable GIF visualization: {e}")
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
