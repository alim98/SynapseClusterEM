"""
Synapse Preprocessing Comparison Tool

This script creates a comparative UMAP visualization from two feature CSV files,
showing how different preprocessing methods affect the feature representation.

Usage:
    python compare_csvs.py <csv_file1> <csv_file2> [--output_dir OUTPUT_DIR]

Example:
    python compare_csvs.py 
        "results/run_1/features_intelligent_crop.csv" 
        "results/run_2/features_normal_crop.csv" 
        --output_dir "results/comparison"
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import umap
import json
import random

def create_comparative_umap(csv_path1, csv_path2, output_dir, label1="Method 1", label2="Method 2", max_pairs=None):
    """
    Create a UMAP visualization comparing features extracted with different preprocessing methods.
    Points from the same sample are connected with lines to visualize the effect of preprocessing.
    
    Args:
        csv_path1: Path to first feature CSV file
        csv_path2: Path to second feature CSV file
        output_dir: Directory to save the visualization
        label1: Label for first dataset in the legend
        label2: Label for second dataset in the legend
        max_pairs: Maximum number of sample pairs to display with connections (None for all)
    """
    print("\n" + "="*80)
    print("Creating comparative UMAP visualization...")
    
    # Load feature data from both preprocessing methods
    dataframes = []
    feature_sets = []
    sample_ids = []
    
    for i, csv_path in enumerate([csv_path1, csv_path2]):
        print(f"Loading features from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Figure out the feature columns
        feature_cols = []
        # First try layer20_feat_ prefix (from stage-specific extraction)
        layer_cols = [col for col in df.columns if col.startswith('layer20_feat_')]
        if layer_cols:
            feature_cols = layer_cols
            print(f"Found {len(feature_cols)} feature columns with prefix 'layer20_feat_'")
        else:
            # Try feat_ prefix (from standard extraction)
            feat_cols = [col for col in df.columns if col.startswith('feat_')]
            if feat_cols:
                feature_cols = feat_cols
                print(f"Found {len(feature_cols)} feature columns with prefix 'feat_'")
            else:
                # Try other common prefixes
                for prefix in ['feature_', 'f_', 'layer']:
                    cols = [col for col in df.columns if col.startswith(prefix)]
                    if cols:
                        feature_cols = cols
                        print(f"Found {len(feature_cols)} feature columns with prefix '{prefix}'")
                        break
        
        # If still no feature columns, try to infer from numeric columns
        if not feature_cols:
            non_feature_cols = ['bbox', 'cluster', 'label', 'id', 'index', 'tsne', 'umap', 'var', 'Var']
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            feature_cols = [col for col in numeric_cols if not any(col.lower().startswith(x.lower()) for x in non_feature_cols)]
            
            if feature_cols:
                print(f"Detected {len(feature_cols)} potential feature columns based on numeric data type")
        
        if not feature_cols:
            print(f"No feature columns found in {csv_path}")
            return
        
        # Extract features
        features = df[feature_cols].values
        
        # Get sample identifiers
        # First check if Var1 column exists (synapse identifier)
        if 'Var1' in df.columns:
            ids = df['Var1'].tolist()
        elif 'id' in df.columns:
            print("var1 not found, error")
            exit()
        
        # Store data
        dataframes.append(df)
        feature_sets.append(features)
        sample_ids.append(ids)
        
        # Add preprocessing method tag
        df['preprocessing'] = label1 if i == 0 else label2
    
    # Check if both datasets have the same sample ids
    set1 = set(sample_ids[0])
    set2 = set(sample_ids[1])
    common_ids = set1.intersection(set2)
    
    if not common_ids:
        print("No common samples found between the two feature sets")
        # If no common IDs, we'll create a UMAP plot without connecting lines
    else:
        print(f"Found {len(common_ids)} common samples between the two feature sets")
        
        # If max_pairs is specified, select a subset of samples
        if max_pairs is not None and max_pairs < len(common_ids):
            # Randomly select max_pairs samples instead of using distance-based selection
            common_ids = random.sample(list(common_ids), max_pairs)
            print(f"Randomly selected {max_pairs} sample pairs for visualization")
    
    # Check if feature dimensions are the same
    if feature_sets[0].shape[1] != feature_sets[1].shape[1]:
        print(f"Feature dimensions don't match: {feature_sets[0].shape[1]} vs {feature_sets[1].shape[1]}")
        print("Using separate UMAP projections and aligning them to compare the feature spaces")
        
        # Scale each feature set separately
        scaled_sets = []
        for features in feature_sets:
            scaler = StandardScaler()
            scaled_sets.append(scaler.fit_transform(features))
        
        # Create separate UMAP projections with slightly different parameters to help separation
        # Use more neighbors for more global structure preservation
        reducer = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.3)
        embedding_1 = reducer.fit_transform(scaled_sets[0])
        
        # Use fewer neighbors for more local structure preservation
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_2 = reducer.fit_transform(scaled_sets[1])
        
        # Translate the second embedding to avoid direct overlap
        # Find centers of both embeddings
        center_1 = np.mean(embedding_1, axis=0)
        center_2 = np.mean(embedding_2, axis=0)
        
        # Shift second embedding to place its center at a slight offset from the first
        offset = [4.0, 0.0]  # Horizontal offset to separate the clusters visually
        embedding_2 = embedding_2 - center_2 + center_1 + offset
    else:
        # Combine features for UMAP if they have the same dimensions
        combined_features = np.vstack([feature_sets[0], feature_sets[1]])
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined_features)
        
        # Create UMAP projection with parameters that encourage separation
        print("Computing UMAP embedding...")
        reducer = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.3, repulsion_strength=1.5)
        embedding = reducer.fit_transform(scaled_features)
        
        # Split embedding back into the two sets
        n_samples_1 = feature_sets[0].shape[0]
        embedding_1 = embedding[:n_samples_1]
        embedding_2 = embedding[n_samples_1:]
    
    # Create plot with larger size
    plt.figure(figsize=(16, 14))
    
    # Define better color scheme for clearer contrast
    color1 = '#3366CC'  # Deeper blue
    color2 = '#DC3912'  # Brick red
    
    # Compute point sizes based on number of samples - make them larger
    point_size = max(70, min(250, 4000 / len(embedding_1)))
    
    # Add a background to help distinguish areas
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom colormaps for background density visualization
    cmap1 = LinearSegmentedColormap.from_list('custom_blue', ['#FFFFFF', '#CCDDFF'])
    cmap2 = LinearSegmentedColormap.from_list('custom_red', ['#FFFFFF', '#FFDCDC'])
    
    # Compute the bounds for the background
    all_points = np.vstack([embedding_1, embedding_2])
    x_min, x_max = all_points[:,0].min() - 1, all_points[:,0].max() + 1
    y_min, y_max = all_points[:,1].min() - 1, all_points[:,1].max() + 1
    
    # Create a meshgrid for background
    grid_step = 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute KDE background for method 1
    from scipy.spatial.distance import cdist
    from scipy.stats import gaussian_kde
    
    # Helper function for density visualization
    def kde_density(points, bandwidth=1.0):
        kde = gaussian_kde(points.T, bw_method=bandwidth)
        return kde(grid_points.T).reshape(xx.shape)
    
    # Get density estimates
    density1 = kde_density(embedding_1, 0.8)
    density2 = kde_density(embedding_2, 0.8)
    
    # Plot density backgrounds
    plt.contourf(xx, yy, density1, levels=15, cmap=cmap1, alpha=0.4)
    plt.contourf(xx, yy, density2, levels=15, cmap=cmap2, alpha=0.4)
    
    # Plot points with clearer visual styling
    stage_scatter = plt.scatter(embedding_1[:, 0], embedding_1[:, 1], 
                                c=color1, label=label1, 
                                alpha=0.7, s=point_size, 
                                edgecolors='navy', linewidths=0.7,
                                zorder=10)  # Higher zorder puts points on top
                                
    standard_scatter = plt.scatter(embedding_2[:, 0], embedding_2[:, 1], 
                                c=color2, label=label2, 
                                alpha=0.7, s=point_size, 
                                edgecolors='darkred', linewidths=0.7,
                                zorder=10)
    
    # If we have common samples, draw connecting lines
    if common_ids:
        # Create dictionaries to map sample IDs to row indices
        id_to_idx_1 = {id_val: idx for idx, id_val in enumerate(sample_ids[0])}
        id_to_idx_2 = {id_val: idx for idx, id_val in enumerate(sample_ids[1])}
        
        # Prepare for lines with varying opacity based on distance
        lines = []
        distances = []
        origins = []
        destinations = []
        
        # First pass: calculate all distances for normalization
        for sample_id in common_ids:
            idx1 = id_to_idx_1.get(sample_id)
            idx2 = id_to_idx_2.get(sample_id)
            
            if idx1 is not None and idx2 is not None:
                x1, y1 = embedding_1[idx1, 0], embedding_1[idx1, 1]
                x2, y2 = embedding_2[idx2, 0], embedding_2[idx2, 1]
                
                # Calculate Euclidean distance
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(distance)
                origins.append((x1, y1))
                destinations.append((x2, y2))
        
        # Normalize distances for line opacity
        max_dist = max(distances) if distances else 1.0
        min_dist = min(distances) if distances else 0.0
        dist_range = max_dist - min_dist
        
        # Calculate distance quartiles for coloring
        import matplotlib.cm as cm
        dist_colors = cm.coolwarm_r

        # Second pass: draw lines with varying opacity and colors based on distance
        for i, (origin, destination, distance) in enumerate(zip(origins, destinations, distances)):
            # Normalize distance to 0-1 range
            if dist_range > 0:
                normalized_distance = (distance - min_dist) / dist_range
                opacity = 0.9 - normalized_distance * 0.6  # Map to 0.3-0.9 range
                # Get color from colormap
                line_color = dist_colors(normalized_distance)
            else:
                opacity = 0.5
                line_color = 'gray'
                
            # Draw line with arrow
            x1, y1 = origin
            x2, y2 = destination
            
            # Use curved connections for better visualization and to avoid overlap
            # Vary the curvature slightly based on index
            curvature = 0.2 + (i % 5) * 0.02
            
            # Draw arrows from first method to second method
            arrow = plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle='->', 
                                         color=line_color, 
                                         alpha=opacity,
                                         lw=1.0,
                                         connectionstyle=f'arc3,rad={curvature}'),
                                         zorder=5)  # Place below points but above background
        
        # Add a legend for distance colors
        if dist_range > 0:
            # Create a colormap legend
            sm = plt.cm.ScalarMappable(cmap=dist_colors, norm=plt.Normalize(vmin=min_dist, vmax=max_dist))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), label='Distance between methods')
            cbar.set_label('Distance between paired samples', fontsize=12)
            
        # Add a note about the connections with better formatting
        note_text = f"Showing {len(common_ids)} sample pairs with connections. "
        if max_pairs is not None and max_pairs < len(set1.intersection(set2)):
            note_text += f"Randomly selected {max_pairs} pairs for visualization."
        note_text += " Color and opacity indicate distance (red = larger distance)."
        
        plt.figtext(0.5, 0.02, note_text,
                  ha='center', fontsize=11, 
                  bbox={"facecolor":"white", "edgecolor":"gray", "alpha":0.8, "pad":5})
    
    # Add better title and labels
    plt.title(f'UMAP Comparison: {label1} vs {label2}', fontsize=18, pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    
    # Create a more prominent legend
    legend = plt.legend(fontsize=13, framealpha=0.9, loc='upper right')
    legend.get_frame().set_edgecolor('gray')
    
    # Add grid but make it subtle
    plt.grid(alpha=0.2)
    
    # Add feature dimension text for clarity
    dim_text = f"Feature dimensions: {feature_sets[0].shape[1]} vs {feature_sets[1].shape[1]}"
    plt.text(0.01, 0.01, dim_text, transform=plt.gca().transAxes, 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the plot with high resolution
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'extraction_method_comparison_umap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced comparative UMAP visualization saved to {output_path}")
    
    # Create an improved interactive HTML version with plotly if available
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import matplotlib.colors as mcolors
        
        # Create a single figure instead of subplots
        fig = go.Figure()
        
        # Create a combined dataframe for plotly
        df1 = dataframes[0].copy()
        df2 = dataframes[1].copy()
        
        # Add UMAP coordinates
        df1['umap_x'] = embedding_1[:, 0]
        df1['umap_y'] = embedding_1[:, 1]
        df2['umap_x'] = embedding_2[:, 0]
        df2['umap_y'] = embedding_2[:, 1]
        
        # Add sample_id for hover info if available
        df1['sample_id'] = sample_ids[0]
        df2['sample_id'] = sample_ids[1]
        
        # Add distance information for points that have matches
        if common_ids:
            distance_map = {}
            for sample_id, distance in zip(common_ids, distances):
                distance_map[sample_id] = distance
                
            df1['pair_distance'] = df1['sample_id'].map(lambda x: distance_map.get(x, float('nan')))
            df2['pair_distance'] = df2['sample_id'].map(lambda x: distance_map.get(x, float('nan')))
            
            # Add boolean flag for whether the sample has a match
            df1['has_match'] = df1['sample_id'].isin(common_ids)
            df2['has_match'] = df2['sample_id'].isin(common_ids)
        
        combined_df = pd.concat([df1, df2])
        
        # Prepare hover data
        hover_data = ['sample_id']
        if 'Var1' in combined_df.columns:
            hover_data.append('Var1')
        if common_ids:
            hover_data.extend(['has_match', 'pair_distance'])
        
        # Main scatter plot for first dataset
        scatter1 = go.Scatter(
            x=df1['umap_x'], y=df1['umap_y'],
            mode='markers',
            marker=dict(
                size=10,
                color=color1,
                opacity=0.7,
                line=dict(width=1, color='navy')
            ),
            name=label1,
            hovertemplate=(
                f"<b>{label1}</b><br>" +
                "Sample: %{customdata[0]}<br>" +
                ("Var1: %{customdata[1]}<br>" if 'Var1' in df1.columns else "") +
                ("Has match: %{customdata[2]}<br>" if common_ids else "") +
                ("Distance: %{customdata[3]:.3f}" if common_ids else "")
            ),
            customdata=df1[hover_data].values,
        )
        
        # Main scatter plot for second dataset
        scatter2 = go.Scatter(
            x=df2['umap_x'], y=df2['umap_y'],
            mode='markers',
            marker=dict(
                size=10,
                color=color2,
                opacity=0.7,
                line=dict(width=1, color='darkred')
            ),
            name=label2,
            hovertemplate=(
                f"<b>{label2}</b><br>" +
                "Sample: %{customdata[0]}<br>" +
                ("Var1: %{customdata[1]}<br>" if 'Var1' in df2.columns else "") +
                ("Has match: %{customdata[2]}<br>" if common_ids else "") +
                ("Distance: %{customdata[3]:.3f}" if common_ids else "")
            ),
            customdata=df2[hover_data].values,
        )
        
        fig.add_trace(scatter1)
        fig.add_trace(scatter2)
        
        # Prepare for slider if we have common IDs
        steps = []
        all_connections = []
        
        if common_ids:
            # Sort connections by distance to display most significant ones first
            connections_with_distance = []
            
            for sample_id in common_ids:
                idx1 = id_to_idx_1.get(sample_id)
                idx2 = id_to_idx_2.get(sample_id)
                
                if idx1 is not None and idx2 is not None:
                    x1, y1 = embedding_1[idx1, 0], embedding_1[idx1, 1]
                    x2, y2 = embedding_2[idx2, 0], embedding_2[idx2, 1]
                    
                    # Calculate distance
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    connections_with_distance.append((sample_id, idx1, idx2, distance))
            
            # Sort by distance (largest first)
            connections_with_distance.sort(key=lambda x: x[3], reverse=True)
            
            # We'll take a different approach - create visible connection groups for each slider step
            num_steps = min(10, len(connections_with_distance))
            step_size = max(1, len(connections_with_distance) // num_steps)
            slider_steps = []
            
            for step in range(num_steps + 1):  # +1 to include zero connections
                # Calculate how many connections to show at this step
                num_connections = min(step * step_size, len(connections_with_distance))
                slider_steps.append(num_connections)
            
            # Add the final step with all connections if needed
            if len(connections_with_distance) not in slider_steps:
                slider_steps.append(len(connections_with_distance))
            
            print(f"Debug: Creating {len(slider_steps)} slider steps")
            
            # Create a separate trace for each slider step with increasing connections
            for step_idx, num_connections in enumerate(slider_steps):
                # Skip the first step (0 connections)
                if num_connections == 0:
                    continue
                
                # Prepare data for connections at this step
                x_data = []
                y_data = []
                hover_texts = []
                
                # Add all connections for this step
                for i in range(num_connections):
                    if i >= len(connections_with_distance):
                        break
                        
                    sample_id, idx1, idx2, distance = connections_with_distance[i]
                    x1, y1 = embedding_1[idx1, 0], embedding_1[idx1, 1]
                    x2, y2 = embedding_2[idx2, 0], embedding_2[idx2, 1]
                    
                    # Calculate normalized distance for color
                    if dist_range > 0:
                        normalized_distance = (distance - min_dist) / dist_range
                    else:
                        normalized_distance = 0.5
                    
                    # Add connection data with None to create breaks between lines
                    x_data.extend([x1, x2, None])
                    y_data.extend([y1, y2, None])
                    hover_texts.extend([
                        f"Sample: {sample_id}<br>Distance: {distance:.3f}",
                        f"Sample: {sample_id}<br>Distance: {distance:.3f}",
                        ""
                    ])
                
                # Create a single trace for all connections at this step
                connections_trace = go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    line=dict(
                        width=1.5,
                        color='rgba(100,100,100,0.7)',  # Use a fixed color for now
                        dash='solid'
                    ),
                    opacity=0.7,
                    hoverinfo='text',
                    hovertext=hover_texts,
                    name=f"{num_connections} Connections",
                    showlegend=(step_idx == len(slider_steps) - 1),  # Only show legend for max connections
                    visible=(step_idx == 1)  # Make the first step visible by default
                )
                
                # Add the trace to the figure
                fig.add_trace(connections_trace)
            
            # Create slider steps
            steps = []
            for i, num_connections in enumerate(slider_steps):
                # Create visibility array
                # First 2 traces (scatter plots) are always visible
                visible_array = [True, True]
                
                # For connection traces (indices 2 to 2+len(slider_steps)-1)
                # Make only the current step's trace visible
                for j in range(len(slider_steps) - 1):  # -1 because we skip the 0 connections step
                    visible_array.append(j == i - 1 if i > 0 else False)  # i-1 because step 0 has no trace
                
                step = dict(
                    method="update",
                    args=[
                        {"visible": visible_array},
                        {"title": f"Interactive UMAP Comparison: {label1} vs {label2} <br><sub>Showing {num_connections} of {len(connections_with_distance)} connections</sub>"}
                    ],
                    label=str(num_connections)
                )
                steps.append(step)
            
            # Create a colormap legend for distances
            if dist_range > 0:
                # Create a colormap legend manually
                colorscale = [
                    [0, 'rgb(220,220,220)'],  # Light gray for small distances
                    [0.5, 'rgb(100,100,100)'],  # Medium gray for medium distances
                    [1, 'rgb(50,50,50)']  # Dark gray for large distances
                ]
                
                # Add a colorbar
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(
                        colorscale=colorscale,
                        showscale=True,
                        cmin=min_dist,
                        cmax=max_dist,
                        colorbar=dict(
                            title='Distance',
                            thickness=15,
                            len=0.5,
                            y=0.5,
                            yanchor='middle'
                        )
                    ),
                    showlegend=False
                ))
        
        # Update layout
        layout_updates = dict(
            width=1200,
            height=800,
            template='plotly_white',
            title={
                'text': f"Interactive UMAP Comparison: {label1} vs {label2}",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            },
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=100, b=120),
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
        )
        
        # Add annotations if they exist
        if 'annotations' in locals():
            layout_updates['annotations'] = annotations
        
        # Add slider if we have common IDs
        if common_ids and steps:
            sliders = [dict(
                active=1,  # Start with the first non-zero step active
                currentvalue=dict(
                    prefix="Connections shown: ",
                    visible=True,
                    font=dict(size=14, color='#444'),
                    xanchor='left'
                ),
                pad=dict(t=60, b=10),
                steps=steps,
                len=0.9,
                x=0.1,
                xanchor='left',
                y=-0.15,  # Move it lower for better visibility
                yanchor='top',
                bgcolor='#F5F5F5',
                bordercolor='#DDDDDD',
                borderwidth=1,
                ticklen=5,
                tickwidth=1,
                tickcolor='#DDDDDD',
                font=dict(size=12)
            )]
            layout_updates['sliders'] = sliders
            
            # Add an instruction annotation about the slider
            if 'annotations' not in layout_updates:
                layout_updates['annotations'] = []
            
            layout_updates['annotations'].append(
                dict(
                    text="Use the slider below to control the number of connections shown",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.11,
                    font=dict(size=14, color='#444')
                )
            )
        
        fig.update_layout(**layout_updates)
        
        # Save as HTML
        html_path = os.path.join(output_dir, 'extraction_method_comparison_umap_interactive.html')
        
        # Save with config for better user experience
        config = {
            'displayModeBar': True,
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'umap_comparison',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
        
        # Use direct plotly HTML export instead of custom template
        fig.write_html(
            html_path,
            config=config,
            include_plotlyjs='cdn',
            full_html=True,
            include_mathjax=False,
            auto_open=False
        )
        
        print(f"Enhanced interactive UMAP visualization with slider control saved to {html_path}")
        
        # Add a direct message to instruct users
        print("Note: Use the slider at the bottom of the HTML to control the number of connections shown.")
        print("      The connections will appear as you increase the slider value.")
        
    except ImportError:
        print("Plotly not available, skipping interactive visualization")

def create_correlation_scatter_plot(csv_path1, csv_path2, output_dir, label1="Method 1", label2="Method 2"):
    """
    Create a correlation scatter plot between feature values from two CSV files.
    
    Args:
        csv_path1: Path to first feature CSV file
        csv_path2: Path to second feature CSV file
        output_dir: Directory to save the visualization
        label1: Label for first dataset
        label2: Label for second dataset
    """
    print("\n" + "="*80)
    print("Creating correlation scatter plot...")
    
    # Load feature data from both CSV files
    dataframes = []
    feature_sets = []
    sample_ids = []
    
    for i, csv_path in enumerate([csv_path1, csv_path2]):
        print(f"Loading features from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Figure out the feature columns using the same logic as in create_comparative_umap
        feature_cols = []
        layer_cols = [col for col in df.columns if col.startswith('layer20_feat_')]
        if layer_cols:
            feature_cols = layer_cols
            print(f"Found {len(feature_cols)} feature columns with prefix 'layer20_feat_'")
        else:
            feat_cols = [col for col in df.columns if col.startswith('feat_')]
            if feat_cols:
                feature_cols = feat_cols
                print(f"Found {len(feature_cols)} feature columns with prefix 'feat_'")
            else:
                for prefix in ['feature_', 'f_', 'layer']:
                    cols = [col for col in df.columns if col.startswith(prefix)]
                    if cols:
                        feature_cols = cols
                        print(f"Found {len(feature_cols)} feature columns with prefix '{prefix}'")
                        break
        
        if not feature_cols:
            non_feature_cols = ['bbox', 'cluster', 'label', 'id', 'index', 'tsne', 'umap', 'var', 'Var']
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            feature_cols = [col for col in numeric_cols if not any(col.lower().startswith(x.lower()) for x in non_feature_cols)]
            
            if feature_cols:
                print(f"Detected {len(feature_cols)} potential feature columns based on numeric data type")
        
        if not feature_cols:
            print(f"No feature columns found in {csv_path}")
            return
        
        # Extract features
        features = df[feature_cols].values
        
        # Get sample identifiers (same logic as in create_comparative_umap)
        if 'Var1' in df.columns:
            ids = df['Var1'].tolist()
        elif 'id' in df.columns:
            ids = df['id'].tolist()
        else:
            ids = [f"sample_{i}" for i in range(len(df))]
        
        # Store data
        dataframes.append(df)
        feature_sets.append(features)
        sample_ids.append(ids)
    
    # Check if both feature sets have the same dimensions
    if feature_sets[0].shape[1] != feature_sets[1].shape[1]:
        print(f"Feature dimensions don't match: {feature_sets[0].shape[1]} vs {feature_sets[1].shape[1]}")
        print("Reducing feature dimensions to a common space before correlation analysis")
        
        # Standardize the features
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        scaled_features1 = scaler1.fit_transform(feature_sets[0])
        scaled_features2 = scaler2.fit_transform(feature_sets[1])
        
        # Determine the target dimension - use the smaller of the two or a fixed value
        target_dim = min(feature_sets[0].shape[1], feature_sets[1].shape[1])
        target_dim = min(64, target_dim)  # Set a maximum dimension for computation efficiency
        
        print(f"Reducing both feature sets to {target_dim} dimensions")
        
        # Apply PCA or UMAP for dimensionality reduction
        try:
            # Try UMAP first for better preservation of relationships
            reducer1 = umap.UMAP(n_components=target_dim, random_state=42)
            reducer2 = umap.UMAP(n_components=target_dim, random_state=42)
            reduced_features1 = reducer1.fit_transform(scaled_features1)
            reduced_features2 = reducer2.fit_transform(scaled_features2)
            print(f"Successfully reduced features using UMAP to dimension {target_dim}")
        except Exception as e:
            print(f"UMAP reduction failed: {str(e)}")
            # Fall back to PCA if UMAP fails
            from sklearn.decomposition import PCA
            reducer1 = PCA(n_components=target_dim, random_state=42)
            reducer2 = PCA(n_components=target_dim, random_state=42)
            reduced_features1 = reducer1.fit_transform(scaled_features1)
            reduced_features2 = reducer2.fit_transform(scaled_features2)
            print(f"Successfully reduced features using PCA to dimension {target_dim}")
        
        # Update the feature sets with the reduced dimensions
        feature_sets[0] = reduced_features1
        feature_sets[1] = reduced_features2
    
    # Find common samples between the two datasets
    set1 = set(sample_ids[0])
    set2 = set(sample_ids[1])
    common_ids = list(set1.intersection(set2))
    
    if not common_ids:
        print("No common samples found between the two feature sets")
        return
    
    print(f"Found {len(common_ids)} common samples between the two feature sets")
    
    # Create dictionaries to map sample IDs to row indices
    id_to_idx_1 = {id_val: idx for idx, id_val in enumerate(sample_ids[0])}
    id_to_idx_2 = {id_val: idx for idx, id_val in enumerate(sample_ids[1])}
    
    # Calculate correlation coefficient for each feature
    feature_correlations = []
    for feature_idx in range(feature_sets[0].shape[1]):
        values_1 = []
        values_2 = []
        
        for sample_id in common_ids:
            idx1 = id_to_idx_1.get(sample_id)
            idx2 = id_to_idx_2.get(sample_id)
            
            if idx1 is not None and idx2 is not None:
                values_1.append(feature_sets[0][idx1, feature_idx])
                values_2.append(feature_sets[1][idx2, feature_idx])
        
        correlation = np.corrcoef(values_1, values_2)[0, 1]
        feature_correlations.append((feature_idx, correlation))
    
    # Sort features by correlation
    feature_correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Select 9 features for visualization, prioritizing strong correlations and some weaker ones
    num_features_to_plot = min(9, len(feature_correlations))
    selected_features = []
    
    # Select high correlation features
    high_corr = [fc for fc in feature_correlations if fc[1] > 0.7][:3]
    selected_features.extend(high_corr)
    
    # Select medium correlation features
    medium_corr = [fc for fc in feature_correlations if 0.3 <= fc[1] <= 0.7][:3]
    selected_features.extend(medium_corr)
    
    # Select low correlation features
    low_corr = [fc for fc in feature_correlations if fc[1] < 0.3][:3]
    selected_features.extend(low_corr)
    
    # If we don't have enough features for each category, take from the remaining features
    if len(selected_features) < num_features_to_plot:
        remaining = [fc for fc in feature_correlations if fc not in selected_features]
        remaining_needed = num_features_to_plot - len(selected_features)
        selected_features.extend(remaining[:remaining_needed])
    
    # Create a subplot grid
    num_plots = min(9, len(selected_features))
    rows = int(np.ceil(np.sqrt(num_plots)))
    cols = int(np.ceil(num_plots / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Generate scatter plots for each selected feature
    for i, (feature_idx, correlation) in enumerate(selected_features[:num_plots]):
        ax = axes[i]
        
        # Get feature values for the common samples
        values_1 = []
        values_2 = []
        
        for sample_id in common_ids:
            idx1 = id_to_idx_1.get(sample_id)
            idx2 = id_to_idx_2.get(sample_id)
            
            if idx1 is not None and idx2 is not None:
                values_1.append(feature_sets[0][idx1, feature_idx])
                values_2.append(feature_sets[1][idx2, feature_idx])
        
        # Create scatter plot
        ax.scatter(values_1, values_2, alpha=0.7, s=30, edgecolors='navy', linewidths=0.5)
        
        # Add reference line for y=x
        min_val = min(min(values_1), min(values_2))
        max_val = max(max(values_1), max(values_2))
        padding = (max_val - min_val) * 0.05
        ax.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 'r--', alpha=0.6)
        
        # Set axis limits with some padding
        ax.set_xlim(min_val-padding, max_val+padding)
        ax.set_ylim(min_val-padding, max_val+padding)
        
        # Add correlation coefficient to the plot
        ax.set_title(f"Feature {feature_idx}, r = {correlation:.3f}")
        
        # Set axis labels
        if i >= (rows-1) * cols:  # Only bottom row gets x-axis label
            ax.set_xlabel(f"{label1}")
        if i % cols == 0:  # Only leftmost column gets y-axis label
            ax.set_ylabel(f"{label2}")
    
    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, "correlation_scatter_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved correlation scatter plot to {output_path}")
    
    # Create a single summary correlation plot
    plt.figure(figsize=(10, 6))
    
    # Calculate overall correlation for each feature
    all_correlations = [corr for _, corr in feature_correlations]
    feature_indices = np.arange(len(all_correlations))
    
    plt.bar(feature_indices, all_correlations, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add horizontal lines for correlation reference
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=-0.5, color='g', linestyle='--', alpha=0.5)
    
    plt.title(f"Feature Correlation between {label1} and {label2}")
    plt.xlabel("Feature Index")
    plt.ylabel("Correlation Coefficient")
    plt.ylim(-1.05, 1.05)
    
    # Add mean correlation as text
    mean_corr = np.mean(all_correlations)
    plt.text(0.02, 0.95, f"Mean correlation: {mean_corr:.3f}", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the summary plot
    summary_path = os.path.join(output_dir, "correlation_summary.png")
    plt.savefig(summary_path, dpi=300)
    print(f"Saved correlation summary to {summary_path}")
    
    plt.close('all')
    return output_path

def create_2d_correlation_scatter_plot(csv_path1, csv_path2, output_dir, label1="Method 1", label2="Method 2"):
    """
    Create a 2D correlation scatter plot after reducing both feature sets to 2 dimensions.
    
    Args:
        csv_path1: Path to first feature CSV file
        csv_path2: Path to second feature CSV file
        output_dir: Directory to save the visualization
        label1: Label for first dataset
        label2: Label for second dataset
    """
    print("\n" + "="*80)
    print("Creating 2D correlation scatter plot...")
    
    # Load feature data from both CSV files using the same logic as other functions
    dataframes = []
    feature_sets = []
    sample_ids = []
    
    for i, csv_path in enumerate([csv_path1, csv_path2]):
        print(f"Loading features from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Figure out the feature columns using the same logic as in create_comparative_umap
        feature_cols = []
        layer_cols = [col for col in df.columns if col.startswith('layer20_feat_')]
        if layer_cols:
            feature_cols = layer_cols
            print(f"Found {len(feature_cols)} feature columns with prefix 'layer20_feat_'")
        else:
            feat_cols = [col for col in df.columns if col.startswith('feat_')]
            if feat_cols:
                feature_cols = feat_cols
                print(f"Found {len(feature_cols)} feature columns with prefix 'feat_'")
            else:
                for prefix in ['feature_', 'f_', 'layer']:
                    cols = [col for col in df.columns if col.startswith(prefix)]
                    if cols:
                        feature_cols = cols
                        print(f"Found {len(feature_cols)} feature columns with prefix '{prefix}'")
                        break
        
        if not feature_cols:
            non_feature_cols = ['bbox', 'cluster', 'label', 'id', 'index', 'tsne', 'umap', 'var', 'Var']
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            feature_cols = [col for col in numeric_cols if not any(col.lower().startswith(x.lower()) for x in non_feature_cols)]
            
            if feature_cols:
                print(f"Detected {len(feature_cols)} potential feature columns based on numeric data type")
        
        if not feature_cols:
            print(f"No feature columns found in {csv_path}")
            return
        
        # Extract features
        features = df[feature_cols].values
        
        # Get sample identifiers (same logic as in create_comparative_umap)
        if 'Var1' in df.columns:
            ids = df['Var1'].tolist()
        elif 'id' in df.columns:
            ids = df['id'].tolist()
        else:
            ids = [f"sample_{i}" for i in range(len(df))]
        
        # Store data
        dataframes.append(df)
        feature_sets.append(features)
        sample_ids.append(ids)
    
    # Standardize the features
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaled_features1 = scaler1.fit_transform(feature_sets[0])
    scaled_features2 = scaler2.fit_transform(feature_sets[1])
    
    # Find common samples between the two datasets
    set1 = set(sample_ids[0])
    set2 = set(sample_ids[1])
    common_ids = list(set1.intersection(set2))
    
    if not common_ids:
        print("No common samples found between the two feature sets")
        return
    
    print(f"Found {len(common_ids)} common samples between the two feature sets")
    
    # Create dictionaries to map sample IDs to row indices
    id_to_idx_1 = {id_val: idx for idx, id_val in enumerate(sample_ids[0])}
    id_to_idx_2 = {id_val: idx for idx, id_val in enumerate(sample_ids[1])}
    
    # Extract features for common samples only
    common_features1 = []
    common_features2 = []
    for sample_id in common_ids:
        idx1 = id_to_idx_1.get(sample_id)
        idx2 = id_to_idx_2.get(sample_id)
        
        if idx1 is not None and idx2 is not None:
            common_features1.append(scaled_features1[idx1])
            common_features2.append(scaled_features2[idx2])
    
    common_features1 = np.array(common_features1)
    common_features2 = np.array(common_features2)
    
    print(f"Reducing both feature sets to 2 dimensions for 2D correlation visualization")
    
    # Reduce to 2D using UMAP or PCA
    try:
        # Try UMAP first
        reducer1 = umap.UMAP(n_components=2, random_state=42)
        reducer2 = umap.UMAP(n_components=2, random_state=42)
        reduced_features1 = reducer1.fit_transform(common_features1)
        reduced_features2 = reducer2.fit_transform(common_features2)
        reduction_method = "UMAP"
        print("Successfully reduced features using UMAP to 2D")
    except Exception as e:
        print(f"UMAP reduction failed: {str(e)}")
        # Fall back to PCA
        from sklearn.decomposition import PCA
        reducer1 = PCA(n_components=2, random_state=42)
        reducer2 = PCA(n_components=2, random_state=42)
        reduced_features1 = reducer1.fit_transform(common_features1)
        reduced_features2 = reducer2.fit_transform(common_features2)
        reduction_method = "PCA"
        print("Successfully reduced features using PCA to 2D")
    
    # Calculate correlations between the two sets of 2D embeddings
    correlations = []
    for dim in range(2):
        corr = np.corrcoef(reduced_features1[:, dim], reduced_features2[:, dim])[0, 1]
        correlations.append(corr)
    
    # Create the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot X dimension correlation
    axes[0].scatter(reduced_features1[:, 0], reduced_features2[:, 0], alpha=0.7, s=40, 
                   edgecolors='navy', linewidths=0.5)
    
    # Add reference line for y=x
    dim0_min = min(reduced_features1[:, 0].min(), reduced_features2[:, 0].min())
    dim0_max = max(reduced_features1[:, 0].max(), reduced_features2[:, 0].max())
    padding = (dim0_max - dim0_min) * 0.05
    axes[0].plot([dim0_min-padding, dim0_max+padding], [dim0_min-padding, dim0_max+padding], 'r--', alpha=0.6)
    
    axes[0].set_title(f"Dimension 1 Correlation\nr = {correlations[0]:.3f}")
    axes[0].set_xlabel(f"{label1} Dimension 1")
    axes[0].set_ylabel(f"{label2} Dimension 1")
    
    # Plot Y dimension correlation
    axes[1].scatter(reduced_features1[:, 1], reduced_features2[:, 1], alpha=0.7, s=40, 
                   color='darkred', edgecolors='maroon', linewidths=0.5)
    
    # Add reference line for y=x
    dim1_min = min(reduced_features1[:, 1].min(), reduced_features2[:, 1].min())
    dim1_max = max(reduced_features1[:, 1].max(), reduced_features2[:, 1].max())
    padding = (dim1_max - dim1_min) * 0.05
    axes[1].plot([dim1_min-padding, dim1_max+padding], [dim1_min-padding, dim1_max+padding], 'r--', alpha=0.6)
    
    axes[1].set_title(f"Dimension 2 Correlation\nr = {correlations[1]:.3f}")
    axes[1].set_xlabel(f"{label1} Dimension 2")
    axes[1].set_ylabel(f"{label2} Dimension 2")
    
    # Create a side-by-side 2D scatter plot
    axes[2].scatter(reduced_features1[:, 0], reduced_features1[:, 1], alpha=0.7, s=40, 
                   label=label1, color='navy')
    axes[2].scatter(reduced_features2[:, 0], reduced_features2[:, 1], alpha=0.7, s=40, 
                   label=label2, color='darkred')
    
    # Connect the same samples with lines
    for i in range(len(common_ids)):
        axes[2].plot([reduced_features1[i, 0], reduced_features2[i, 0]], 
                    [reduced_features1[i, 1], reduced_features2[i, 1]], 
                    'gray', alpha=0.2, linewidth=0.5)
    
    axes[2].set_title(f"2D Embeddings of Both Feature Sets\n({reduction_method} Reduction)")
    axes[2].set_xlabel("Dimension 1")
    axes[2].set_ylabel("Dimension 2")
    axes[2].legend()
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, "2d_correlation_scatter_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved 2D correlation scatter plot to {output_path}")
    
    # Create a comprehensive correlation visualization
    plt.figure(figsize=(12, 12))
    
    # Create a 2x2 grid of scatter plots comparing both dimensions
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Plot 1: Method 1 dim1 vs Method 2 dim1
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(reduced_features1[:, 0], reduced_features2[:, 0], alpha=0.7, c='blue')
    ax1.set_title(f"Correlation: {correlations[0]:.3f}")
    ax1.set_xlabel(f"{label1} Dim 1")
    ax1.set_ylabel(f"{label2} Dim 1")
    
    # Plot 2: Method 1 dim1 vs Method 2 dim2
    ax2 = plt.subplot(gs[0, 1])
    cross_corr_1_2 = np.corrcoef(reduced_features1[:, 0], reduced_features2[:, 1])[0, 1]
    ax2.scatter(reduced_features1[:, 0], reduced_features2[:, 1], alpha=0.7, c='green')
    ax2.set_title(f"Correlation: {cross_corr_1_2:.3f}")
    ax2.set_xlabel(f"{label1} Dim 1")
    ax2.set_ylabel(f"{label2} Dim 2")
    
    # Plot 3: Method 1 dim2 vs Method 2 dim1
    ax3 = plt.subplot(gs[1, 0])
    cross_corr_2_1 = np.corrcoef(reduced_features1[:, 1], reduced_features2[:, 0])[0, 1]
    ax3.scatter(reduced_features1[:, 1], reduced_features2[:, 0], alpha=0.7, c='orange')
    ax3.set_title(f"Correlation: {cross_corr_2_1:.3f}")
    ax3.set_xlabel(f"{label1} Dim 2")
    ax3.set_ylabel(f"{label2} Dim 1")
    
    # Plot 4: Method 1 dim2 vs Method 2 dim2
    ax4 = plt.subplot(gs[1, 1])
    ax4.scatter(reduced_features1[:, 1], reduced_features2[:, 1], alpha=0.7, c='red')
    ax4.set_title(f"Correlation: {correlations[1]:.3f}")
    ax4.set_xlabel(f"{label1} Dim 2")
    ax4.set_ylabel(f"{label2} Dim 2")
    
    plt.suptitle(f"Cross-Dimensional Correlation Matrix ({reduction_method} Reduction)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the comprehensive correlation visualization
    matrix_path = os.path.join(output_dir, "2d_correlation_matrix.png")
    plt.savefig(matrix_path, dpi=300)
    print(f"Saved 2D correlation matrix to {matrix_path}")
    
    plt.close('all')
    return output_path

def create_distance_correlation_plot(csv_path1, csv_path2, output_dir, label1="Method 1", label2="Method 2"):
    """
    Create a correlation plot comparing pairwise sample distances between two methods.
    
    Args:
        csv_path1: Path to first feature CSV file
        csv_path2: Path to second feature CSV file
        output_dir: Directory to save the visualization
        label1: Label for first dataset
        label2: Label for second dataset
    """
    print("\n" + "="*80)
    print("Creating distance correlation plot...")
    
    # Load feature data from both CSV files using the same logic as other functions
    dataframes = []
    feature_sets = []
    sample_ids = []
    
    for i, csv_path in enumerate([csv_path1, csv_path2]):
        print(f"Loading features from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Figure out the feature columns using the same logic as in create_comparative_umap
        feature_cols = []
        layer_cols = [col for col in df.columns if col.startswith('layer20_feat_')]
        if layer_cols:
            feature_cols = layer_cols
            print(f"Found {len(feature_cols)} feature columns with prefix 'layer20_feat_'")
        else:
            feat_cols = [col for col in df.columns if col.startswith('feat_')]
            if feat_cols:
                feature_cols = feat_cols
                print(f"Found {len(feature_cols)} feature columns with prefix 'feat_'")
            else:
                for prefix in ['feature_', 'f_', 'layer']:
                    cols = [col for col in df.columns if col.startswith(prefix)]
                    if cols:
                        feature_cols = cols
                        print(f"Found {len(feature_cols)} feature columns with prefix '{prefix}'")
                        break
        
        if not feature_cols:
            non_feature_cols = ['bbox', 'cluster', 'label', 'id', 'index', 'tsne', 'umap', 'var', 'Var']
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            feature_cols = [col for col in numeric_cols if not any(col.lower().startswith(x.lower()) for x in non_feature_cols)]
            
            if feature_cols:
                print(f"Detected {len(feature_cols)} potential feature columns based on numeric data type")
        
        if not feature_cols:
            print(f"No feature columns found in {csv_path}")
            return
        
        # Extract features
        features = df[feature_cols].values
        
        # Get sample identifiers (same logic as in create_comparative_umap)
        if 'Var1' in df.columns:
            ids = df['Var1'].tolist()
        elif 'id' in df.columns:
            ids = df['id'].tolist()
        else:
            ids = [f"sample_{i}" for i in range(len(df))]
        
        # Store data
        dataframes.append(df)
        feature_sets.append(features)
        sample_ids.append(ids)
    
    # Standardize the features
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaled_features1 = scaler1.fit_transform(feature_sets[0])
    scaled_features2 = scaler2.fit_transform(feature_sets[1])
    
    # Handle different dimensions if necessary
    if feature_sets[0].shape[1] != feature_sets[1].shape[1]:
        print(f"Feature dimensions don't match: {feature_sets[0].shape[1]} vs {feature_sets[1].shape[1]}")
        print("Reducing feature dimensions to a common space before distance calculation")
        
        # Determine the target dimension - use the smaller of the two or a fixed value
        target_dim = min(feature_sets[0].shape[1], feature_sets[1].shape[1])
        target_dim = min(64, target_dim)  # Set a maximum dimension for computation efficiency
        
        print(f"Reducing both feature sets to {target_dim} dimensions")
        
        # Apply PCA or UMAP for dimensionality reduction
        try:
            # Try UMAP first for better preservation of relationships
            reducer1 = umap.UMAP(n_components=target_dim, random_state=42)
            reducer2 = umap.UMAP(n_components=target_dim, random_state=42)
            reduced_features1 = reducer1.fit_transform(scaled_features1)
            reduced_features2 = reducer2.fit_transform(scaled_features2)
            print(f"Successfully reduced features using UMAP to dimension {target_dim}")
        except Exception as e:
            print(f"UMAP reduction failed: {str(e)}")
            # Fall back to PCA if UMAP fails
            from sklearn.decomposition import PCA
            reducer1 = PCA(n_components=target_dim, random_state=42)
            reducer2 = PCA(n_components=target_dim, random_state=42)
            reduced_features1 = reducer1.fit_transform(scaled_features1)
            reduced_features2 = reducer2.fit_transform(scaled_features2)
            print(f"Successfully reduced features using PCA to dimension {target_dim}")
        
        # Update the feature sets with the reduced dimensions
        scaled_features1 = reduced_features1
        scaled_features2 = reduced_features2
    
    # Find common samples between the two datasets
    set1 = set(sample_ids[0])
    set2 = set(sample_ids[1])
    common_ids = list(set1.intersection(set2))
    
    if not common_ids:
        print("No common samples found between the two feature sets")
        return
    
    print(f"Found {len(common_ids)} common samples between the two feature sets")
    
    # Create dictionaries to map sample IDs to row indices
    id_to_idx_1 = {id_val: idx for idx, id_val in enumerate(sample_ids[0])}
    id_to_idx_2 = {id_val: idx for idx, id_val in enumerate(sample_ids[1])}
    
    # Extract features for common samples only and ensure they have same ordering
    common_features1 = []
    common_features2 = []
    ordered_common_ids = []
    
    for sample_id in common_ids:
        idx1 = id_to_idx_1.get(sample_id)
        idx2 = id_to_idx_2.get(sample_id)
        
        if idx1 is not None and idx2 is not None:
            common_features1.append(scaled_features1[idx1])
            common_features2.append(scaled_features2[idx2])
            ordered_common_ids.append(sample_id)
    
    common_features1 = np.array(common_features1)
    common_features2 = np.array(common_features2)
    
    # Compute pairwise distances within each method
    print("Computing pairwise distances...")
    from scipy.spatial.distance import pdist, squareform
    
    # Calculate full distance matrices
    dist_matrix1 = squareform(pdist(common_features1, metric='euclidean'))
    dist_matrix2 = squareform(pdist(common_features2, metric='euclidean'))
    
    # Flatten the distance matrices (excluding the diagonal)
    n_samples = len(common_features1)
    distances1 = []
    distances2 = []
    pair_indices = []
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            distances1.append(dist_matrix1[i, j])
            distances2.append(dist_matrix2[i, j])
            pair_indices.append((i, j))
    
    # Calculate correlation between the distances
    correlation = np.corrcoef(distances1, distances2)[0, 1]
    print(f"Correlation between pairwise distances: {correlation:.4f}")
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with point density coloring
    # The error occurs here with gaussian_kde, so let's modify this part
    try:
        from scipy.stats import gaussian_kde
        
        # Calculate point density for coloring
        xy = np.vstack([distances1, distances2])
        
        # Add small random noise to avoid singularity issues
        xy_with_noise = xy + np.random.normal(0, 0.0001, xy.shape)
        
        # Try to calculate density with noise-added data
        try:
            z = gaussian_kde(xy_with_noise)(xy)
            
            # Sort points by density for better visualization
            idx = z.argsort()
            distances1_sorted = np.array(distances1)[idx]
            distances2_sorted = np.array(distances2)[idx]
            z_sorted = z[idx]
            
            # Scatter plot with density coloring
            scatter = plt.scatter(distances1_sorted, distances2_sorted, 
                                c=z_sorted, s=30, alpha=0.7, 
                                cmap='viridis', edgecolor='w', linewidth=0.2)
            
            # Add color bar for density
            cbar = plt.colorbar(scatter)
            cbar.set_label('Point Density', fontsize=12)
        except Exception as e:
            print(f"Density estimation failed, using simpler coloring: {str(e)}")
            # Fall back to simpler coloring when density estimation fails
            plt.scatter(distances1, distances2, c='blue', alpha=0.5, s=20)
    except Exception as e:
        print(f"Density coloring failed: {str(e)}")
        # Basic scatter plot without density coloring
        plt.scatter(distances1, distances2, c='blue', alpha=0.5, s=20)
    
    # Add reference line y=x
    min_val = min(min(distances1), min(distances2))
    max_val = max(max(distances1), max(distances2))
    padding = (max_val - min_val) * 0.05
    plt.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 
             'r--', alpha=0.6, label='y=x')
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(distances1, distances2)
    plt.plot(np.array([min_val-padding, max_val+padding]), 
             intercept + slope*np.array([min_val-padding, max_val+padding]), 
             'g-', alpha=0.7, label=f'Regression line (slope={slope:.3f})')
    
    # Add axis labels and title
    plt.xlabel(f'Pairwise distances in {label1}', fontsize=14)
    plt.ylabel(f'Pairwise distances in {label2}', fontsize=14)
    plt.title(f'Correlation between Pairwise Sample Distances\nr = {correlation:.4f}', fontsize=16)
    
    # Add text with correlation and sample count
    plt.text(0.05, 0.95, 
             f"r = {correlation:.4f}\nSamples: {n_samples}\nPairs: {len(distances1)}", 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=12, verticalalignment='top')
    
    # Add legend
    plt.legend(loc='lower right')
    
    plt.grid(alpha=0.2)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, "distance_correlation_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved distance correlation plot to {output_path}")
    
    # Create plot showing the distribution of distance ratios
    plt.figure(figsize=(12, 6))
    
    # Calculate the ratio of distances (method2 / method1)
    ratios = np.array(distances2) / np.array(distances1)
    
    # Create histogram of ratios in the left subplot
    plt.subplot(1, 2, 1)
    plt.hist(ratios, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.8, label='Equal ratio (1.0)')
    plt.axvline(x=np.median(ratios), color='g', linestyle='-', alpha=0.8, 
                label=f'Median ratio ({np.median(ratios):.3f})')
    
    plt.xlabel('Distance Ratio (Method 2 / Method 1)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Distance Ratios', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.2)
    
    # Plot sorted ratios in the right subplot to show pattern
    plt.subplot(1, 2, 2)
    sorted_ratios = np.sort(ratios)
    plt.plot(np.arange(len(sorted_ratios)), sorted_ratios, 'b-', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.8, label='Equal ratio (1.0)')
    plt.axhline(y=np.median(ratios), color='g', linestyle='-', alpha=0.8, 
                label=f'Median ratio ({np.median(ratios):.3f})')
    
    plt.xlabel('Sorted Pair Index', fontsize=12)
    plt.ylabel('Distance Ratio (Method 2 / Method 1)', fontsize=12)
    plt.title('Sorted Distance Ratios', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.tight_layout()
    
    # Save the plot
    ratio_path = os.path.join(output_dir, "distance_ratio_analysis.png")
    plt.savefig(ratio_path, dpi=300)
    print(f"Saved distance ratio analysis to {ratio_path}")
    
    # Identify pairs with the largest distance differences
    distance_diffs = np.abs(np.array(distances1) - np.array(distances2))
    largest_diff_indices = np.argsort(distance_diffs)[-20:]  # Get indices of 20 largest differences
    
    # Create a table of the largest differences
    largest_diff_data = []
    for idx in largest_diff_indices[::-1]:  # Reverse to show largest first
        i, j = pair_indices[idx]
        largest_diff_data.append({
            'Sample 1': ordered_common_ids[i],
            'Sample 2': ordered_common_ids[j],
            f'Distance in {label1}': distances1[idx],
            f'Distance in {label2}': distances2[idx],
            'Absolute Difference': distance_diffs[idx],
            'Ratio (M2/M1)': distances2[idx] / distances1[idx]
        })
    
    # Create DataFrame for displaying the results
    diff_df = pd.DataFrame(largest_diff_data)
    
    # Save to CSV
    diff_csv_path = os.path.join(output_dir, "largest_distance_differences.csv")
    diff_df.to_csv(diff_csv_path, index=False)
    print(f"Saved list of largest distance differences to {diff_csv_path}")
    
    plt.close('all')
    return output_path

def analyze_distance_ratio_preservation(csv_path1, csv_path2, output_dir, label1="Method 1", label2="Method 2"):
    """
    Analyze how well pairwise distance ratios are preserved between two methods.
    
    This function compares the preservation of relative distances between sample pairs across two methods:
    - For each sample (s1), we analyze whether the distance ratios between pairs (s1,s2) and (s1,s3) are preserved
    - A high correlation of distance ratios indicates that the relative distances between samples are preserved,
      even if the absolute scaling of the distances changes.
      
    Args:
        csv_path1: Path to first feature CSV file
        csv_path2: Path to second feature CSV file
        output_dir: Directory to save the visualization
        label1: Label for first dataset
        label2: Label for second dataset
    """
    print("\n" + "="*80)
    print("Analyzing distance ratio preservation...")
    
    # Load feature data from both CSV files using the same logic as other functions
    dataframes = []
    feature_sets = []
    sample_ids = []
    
    for i, csv_path in enumerate([csv_path1, csv_path2]):
        print(f"Loading features from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Figure out the feature columns using the same logic as in create_comparative_umap
        feature_cols = []
        layer_cols = [col for col in df.columns if col.startswith('layer20_feat_')]
        if layer_cols:
            feature_cols = layer_cols
            print(f"Found {len(feature_cols)} feature columns with prefix 'layer20_feat_'")
        else:
            feat_cols = [col for col in df.columns if col.startswith('feat_')]
            if feat_cols:
                feature_cols = feat_cols
                print(f"Found {len(feature_cols)} feature columns with prefix 'feat_'")
            else:
                for prefix in ['feature_', 'f_', 'layer']:
                    cols = [col for col in df.columns if col.startswith(prefix)]
                    if cols:
                        feature_cols = cols
                        print(f"Found {len(feature_cols)} feature columns with prefix '{prefix}'")
                        break
        
        if not feature_cols:
            non_feature_cols = ['bbox', 'cluster', 'label', 'id', 'index', 'tsne', 'umap', 'var', 'Var']
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            feature_cols = [col for col in numeric_cols if not any(col.lower().startswith(x.lower()) for x in non_feature_cols)]
            
            if feature_cols:
                print(f"Detected {len(feature_cols)} potential feature columns based on numeric data type")
        
        if not feature_cols:
            print(f"No feature columns found in {csv_path}")
            return
        
        # Extract features
        features = df[feature_cols].values
        
        # Get sample identifiers (same logic as in create_comparative_umap)
        if 'Var1' in df.columns:
            ids = df['Var1'].tolist()
        elif 'id' in df.columns:
            ids = df['id'].tolist()
        else:
            ids = [f"sample_{i}" for i in range(len(df))]
        
        # Store data
        dataframes.append(df)
        feature_sets.append(features)
        sample_ids.append(ids)
    
    # Standardize the features
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaled_features1 = scaler1.fit_transform(feature_sets[0])
    scaled_features2 = scaler2.fit_transform(feature_sets[1])
    
    # Handle different dimensions if necessary
    if feature_sets[0].shape[1] != feature_sets[1].shape[1]:
        print(f"Feature dimensions don't match: {feature_sets[0].shape[1]} vs {feature_sets[1].shape[1]}")
        print("Reducing feature dimensions to a common space before distance calculation")
        
        # Determine the target dimension - use the smaller of the two or a fixed value
        target_dim = min(feature_sets[0].shape[1], feature_sets[1].shape[1])
        target_dim = min(64, target_dim)  # Set a maximum dimension for computation efficiency
        
        print(f"Reducing both feature sets to {target_dim} dimensions")
        
        # Apply PCA or UMAP for dimensionality reduction
        try:
            # Try UMAP first for better preservation of relationships
            reducer1 = umap.UMAP(n_components=target_dim, random_state=42)
            reducer2 = umap.UMAP(n_components=target_dim, random_state=42)
            reduced_features1 = reducer1.fit_transform(scaled_features1)
            reduced_features2 = reducer2.fit_transform(scaled_features2)
            print(f"Successfully reduced features using UMAP to dimension {target_dim}")
        except Exception as e:
            print(f"UMAP reduction failed: {str(e)}")
            # Fall back to PCA if UMAP fails
            from sklearn.decomposition import PCA
            reducer1 = PCA(n_components=target_dim, random_state=42)
            reducer2 = PCA(n_components=target_dim, random_state=42)
            reduced_features1 = reducer1.fit_transform(scaled_features1)
            reduced_features2 = reducer2.fit_transform(scaled_features2)
            print(f"Successfully reduced features using PCA to dimension {target_dim}")
        
        # Update the feature sets with the reduced dimensions
        scaled_features1 = reduced_features1
        scaled_features2 = reduced_features2
    
    # Find common samples between the two datasets
    set1 = set(sample_ids[0])
    set2 = set(sample_ids[1])
    common_ids = list(set1.intersection(set2))
    
    if not common_ids:
        print("No common samples found between the two feature sets")
        return
    
    print(f"Found {len(common_ids)} common samples between the two feature sets")
    
    # Create dictionaries to map sample IDs to row indices
    id_to_idx_1 = {id_val: idx for idx, id_val in enumerate(sample_ids[0])}
    id_to_idx_2 = {id_val: idx for idx, id_val in enumerate(sample_ids[1])}
    
    # Extract features for common samples only and ensure they have same ordering
    common_features1 = []
    common_features2 = []
    ordered_common_ids = []
    
    for sample_id in common_ids:
        idx1 = id_to_idx_1.get(sample_id)
        idx2 = id_to_idx_2.get(sample_id)
        
        if idx1 is not None and idx2 is not None:
            common_features1.append(scaled_features1[idx1])
            common_features2.append(scaled_features2[idx2])
            ordered_common_ids.append(sample_id)
    
    common_features1 = np.array(common_features1)
    common_features2 = np.array(common_features2)
    
    # Compute pairwise distances within each method
    print("Computing pairwise distances...")
    from scipy.spatial.distance import pdist, squareform
    
    # Calculate full distance matrices
    dist_matrix1 = squareform(pdist(common_features1, metric='euclidean'))
    dist_matrix2 = squareform(pdist(common_features2, metric='euclidean'))
    
    # Analysis of distance ratio preservation
    print("Analyzing distance ratio preservation...")
    
    n_samples = len(common_features1)
    
    # For each sample, calculate the ratio of distances to all other samples
    # and compare these ratios between the two methods
    ratio_correlations = []
    ratio_preservation_data = []
    
    # Create a figure for visualizing ratio preservation for individual samples
    n_display_samples = min(9, n_samples)  # Display max 9 samples
    display_samples = np.random.choice(range(n_samples), n_display_samples, replace=False)
    
    # Create subplot grid for sample-specific visualizations
    rows = int(np.ceil(np.sqrt(n_display_samples)))
    cols = int(np.ceil(n_display_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # For each sample, calculate the preservation of distance ratios
    for i in range(n_samples):
        # Get distances from sample i to all other samples in both methods
        distances1 = dist_matrix1[i, :]
        distances2 = dist_matrix2[i, :]
        
        # Skip the distance to itself
        distances1 = np.concatenate([distances1[:i], distances1[i+1:]])
        distances2 = np.concatenate([distances2[:i], distances2[i+1:]])
        
        # Check if we have enough distances to calculate meaningful ratios
        if len(distances1) < 2:
            continue
            
        # Calculate all pairwise ratios within distances1
        ratios1 = []
        ratios2 = []
        dist_pairs = []
        
        # For all pairs of distances, calculate the ratio
        for j in range(len(distances1)):
            for k in range(j+1, len(distances1)):
                # Skip if distances are zero or very small
                if distances1[j] < 1e-6 or distances1[k] < 1e-6 or distances2[j] < 1e-6 or distances2[k] < 1e-6:
                    continue
                    
                # Calculate ratios: distance to j / distance to k
                ratio1 = distances1[j] / distances1[k]
                ratio2 = distances2[j] / distances2[k]
                
                # Store ratios and corresponding distance pair indices
                ratios1.append(ratio1)
                ratios2.append(ratio2)
                dist_pairs.append((j, k))
        
        # Calculate correlation between the ratios
        if len(ratios1) > 1:
            correlation = np.corrcoef(ratios1, ratios2)[0, 1]
            ratio_correlations.append(correlation)
            
            # Store data for the sample
            sample_data = {
                'sample_id': ordered_common_ids[i],
                'ratio_correlation': correlation,
                'mean_ratio_method1': np.mean(ratios1),
                'mean_ratio_method2': np.mean(ratios2),
                'std_ratio_method1': np.std(ratios1),
                'std_ratio_method2': np.std(ratios2),
                'num_ratios': len(ratios1)
            }
            ratio_preservation_data.append(sample_data)
            
            # If this is one of the randomly selected samples, visualize its ratio preservation
            if i in display_samples:
                ax_idx = np.where(display_samples == i)[0][0]
                ax = axes[ax_idx]
                
                # Create scatter plot of ratios
                ax.scatter(ratios1, ratios2, alpha=0.7, s=30, edgecolors='navy', linewidths=0.5)
                
                # Add reference line for y=x
                min_val = min(min(ratios1), min(ratios2))
                max_val = max(max(ratios1), max(ratios2))
                padding = (max_val - min_val) * 0.05
                ax.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 'r--', alpha=0.6)
                
                # Add regression line
                from scipy import stats
                if len(ratios1) > 1:  # Need at least 2 points for regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(ratios1, ratios2)
                    ax.plot(np.array([min_val-padding, max_val+padding]), 
                            intercept + slope*np.array([min_val-padding, max_val+padding]), 
                            'g-', alpha=0.7)
                
                # Set axis limits with some padding
                ax.set_xlim(min_val-padding, max_val+padding)
                ax.set_ylim(min_val-padding, max_val+padding)
                
                # Add correlation coefficient to the plot
                ax.set_title(f"Sample: {ordered_common_ids[i]}\nr = {correlation:.3f}")
                
                # Only add axis labels to bottom and left plots
                if ax_idx >= (rows-1) * cols:  # Bottom row
                    ax.set_xlabel(f"Distance Ratios in {label1}")
                if ax_idx % cols == 0:  # Left column
                    ax.set_ylabel(f"Distance Ratios in {label2}")
    
    # Hide unused subplots
    for j in range(n_display_samples, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the sample-specific ratio preservation plot
    sample_ratio_path = os.path.join(output_dir, "sample_ratio_preservation.png")
    plt.savefig(sample_ratio_path, dpi=300)
    print(f"Saved sample-specific ratio preservation analysis to {sample_ratio_path}")
    
    # Create a correlation distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(ratio_correlations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(ratio_correlations), color='r', linestyle='-', alpha=0.7, 
                label=f'Mean: {np.mean(ratio_correlations):.3f}')
    plt.axvline(x=np.median(ratio_correlations), color='g', linestyle='--', alpha=0.7,
                label=f'Median: {np.median(ratio_correlations):.3f}')
    
    plt.xlabel('Correlation of Distance Ratios', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title('Distribution of Distance Ratio Correlation by Sample', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the correlation distribution plot
    corr_dist_path = os.path.join(output_dir, "ratio_correlation_distribution.png")
    plt.savefig(corr_dist_path, dpi=300)
    print(f"Saved ratio correlation distribution to {corr_dist_path}")
    
    # Create a DataFrame with the ratio preservation results
    ratio_df = pd.DataFrame(ratio_preservation_data)
    
    # Sort by correlation (descending)
    ratio_df = ratio_df.sort_values('ratio_correlation', ascending=False)
    
    # Save the results to CSV
    ratio_csv_path = os.path.join(output_dir, "ratio_preservation_by_sample.csv")
    ratio_df.to_csv(ratio_csv_path, index=False)
    print(f"Saved detailed ratio preservation data to {ratio_csv_path}")
    
    # Calculate the preservation of the rank order of distances
    print("Analyzing rank order preservation of distances...")
    
    # Flatten the distance matrices (excluding the diagonal)
    distances1 = []
    distances2 = []
    sample_pairs = []
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            distances1.append(dist_matrix1[i, j])
            distances2.append(dist_matrix2[i, j])
            sample_pairs.append((ordered_common_ids[i], ordered_common_ids[j]))
    
    # Create ranks
    rank1 = np.argsort(np.argsort(distances1))  # Ranks of distances in method 1
    rank2 = np.argsort(np.argsort(distances2))  # Ranks of distances in method 2
    
    # Calculate Spearman correlation between ranks
    spearman_corr = np.corrcoef(rank1, rank2)[0, 1]
    
    print(f"Spearman rank correlation of distances: {spearman_corr:.4f}")
    
    # Create a scatter plot of distance ranks
    plt.figure(figsize=(10, 8))
    plt.scatter(rank1, rank2, alpha=0.5, s=20)
    plt.plot([0, len(distances1)], [0, len(distances1)], 'r--', alpha=0.6)
    
    plt.xlabel(f'Distance Rank in {label1}', fontsize=14)
    plt.ylabel(f'Distance Rank in {label2}', fontsize=14)
    plt.title(f'Preservation of Distance Ranking\nSpearman Correlation: {spearman_corr:.4f}', fontsize=16)
    plt.grid(alpha=0.3)
    
    # Save the rank correlation plot
    rank_path = os.path.join(output_dir, "distance_rank_correlation.png")
    plt.savefig(rank_path, dpi=300)
    print(f"Saved distance rank correlation plot to {rank_path}")
    
    # Create a plot to visualize the overall pattern of distance ratio preservation
    plt.figure(figsize=(12, 10))
    
    # For each original distance in method 1, plot the corresponding distance in method 2
    # Color by the stability of ratios involving this distance
    
    # For each distance pair (i,j), calculate the consistency of its ratios
    # with all other distances
    ratio_consistency = np.zeros(len(distances1))
    
    for idx, (d1, d2) in enumerate(zip(distances1, distances2)):
        # Skip if distance is very small
        if d1 < 1e-6 or d2 < 1e-6:
            continue
            
        # For this distance pair, get ratios with all other distances
        curr_ratios1 = []
        curr_ratios2 = []
        
        for other_idx, (other_d1, other_d2) in enumerate(zip(distances1, distances2)):
            if idx == other_idx or other_d1 < 1e-6 or other_d2 < 1e-6:
                continue
                
            # Calculate ratio
            curr_ratios1.append(d1 / other_d1)
            curr_ratios2.append(d2 / other_d2)
        
        # Calculate correlation of ratios for this distance pair
        if len(curr_ratios1) > 1:
            ratio_consistency[idx] = np.corrcoef(curr_ratios1, curr_ratios2)[0, 1]
        
    # Use a colormap to visualize the ratio consistency
    plt.scatter(distances1, distances2, c=ratio_consistency, cmap='viridis', 
                s=30, alpha=0.7, edgecolors='gray', linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Ratio Preservation (Correlation)', fontsize=12)
    
    # Add reference line
    min_val = min(min(distances1), min(distances2))
    max_val = max(max(distances1), max(distances2))
    padding = (max_val - min_val) * 0.05
    plt.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 'r--', alpha=0.6)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(distances1, distances2)
    plt.plot(np.array([min_val-padding, max_val+padding]), 
             intercept + slope*np.array([min_val-padding, max_val+padding]), 
             'g-', alpha=0.7, label=f'Regression line (slope={slope:.3f})')
    
    plt.xlabel(f'Distances in {label1}', fontsize=14)
    plt.ylabel(f'Distances in {label2}', fontsize=14)
    plt.title(f'Distance Correlation with Ratio Preservation Coloring\nMean Ratio Correlation: {np.mean(ratio_correlations):.4f}', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the ratio preservation plot
    ratio_pres_path = os.path.join(output_dir, "distance_ratio_preservation.png")
    plt.savefig(ratio_pres_path, dpi=300)
    print(f"Saved distance ratio preservation analysis to {ratio_pres_path}")
    
    # Create a comprehensive report
    report = {
        'mean_ratio_correlation': float(np.mean(ratio_correlations)),
        'median_ratio_correlation': float(np.median(ratio_correlations)),
        'distance_spearman_correlation': float(spearman_corr),
        'num_samples': n_samples,
        'total_distance_pairs': len(distances1),
        'feature_dimensions': {'method1': feature_sets[0].shape[1], 'method2': feature_sets[1].shape[1]},
        'distance_correlation': float(np.corrcoef(distances1, distances2)[0, 1]),
        'distance_ratio': float(np.mean(np.array(distances2) / np.array(distances1)))
    }
    
    # Save report to JSON
    report_path = os.path.join(output_dir, "distance_ratio_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Saved comprehensive analysis report to {report_path}")
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Mean ratio correlation: {report['mean_ratio_correlation']:.4f}")
    print(f"Distance rank correlation: {report['distance_spearman_correlation']:.4f}")
    print(f"Distance correlation: {report['distance_correlation']:.4f}")
    print(f"Average distance scaling factor (Method2/Method1): {report['distance_ratio']:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if report['mean_ratio_correlation'] > 0.7:
        print("EXCELLENT ratio preservation: The methods preserve distance ratios very well")
    elif report['mean_ratio_correlation'] > 0.5:
        print("GOOD ratio preservation: The methods generally preserve distance ratios")
    elif report['mean_ratio_correlation'] > 0.3:
        print("MODERATE ratio preservation: The methods somewhat preserve distance ratios")
    else:
        print("POOR ratio preservation: The methods do not preserve distance ratios well")
        
    if report['distance_spearman_correlation'] > 0.7:
        print("EXCELLENT rank preservation: The relative ordering of distances is well preserved")
    elif report['distance_spearman_correlation'] > 0.5:
        print("GOOD rank preservation: The relative ordering of distances is largely preserved")
    elif report['distance_spearman_correlation'] > 0.3:
        print("MODERATE rank preservation: The relative ordering of distances is somewhat preserved")
    else:
        print("POOR rank preservation: The relative ordering of distances is not well preserved")
    
    # Return success
    plt.close('all')
    return True

def main():
    # for feature extraction stage specific layer20 and standard comparison
    # default_csv_file1 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\intelligent_cropping\layer20\features_extraction_stage_specific_layer20_segNone_alphaNone_intelligent_crop_w7\features_layer20_segNone_alphaNone.csv"
    # default_csv_file2 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\intelligent_cropping\standard\features_extraction_standard_segNone_alphaNone_intelligent_crop_w7\features_segNone_alphaNone.csv"
    # default_output_dir = r"C:\Users\alim9\Documents\codes\synapse2\results\features\intelligent_cropping\method_comparision"
    # method1_label = "layer20"
    # method2_label = "standard"
    # # for preprocessing method comparison
   
    default_csv_file2 = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\13\features_layer20_seg13_alpha1.0\features_layer20_seg13_alpha1_0.csv"
    default_csv_file1 = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\10\features_layer20_seg10_alpha1.0\features_layer20_seg10_alpha1_0.csv"
    
    default_output_dir = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\comparision13n"
    method1_label = "segmentation 10"
    method2_label = "segmentation 13"
    parser = argparse.ArgumentParser(description="Compare features from different CSV files and create visualizations")
    parser.add_argument("--csv1", default=default_csv_file1, help="Path to first feature CSV file")
    parser.add_argument("--csv2", default=default_csv_file2, help="Path to second feature CSV file")
    parser.add_argument("--output_dir", default=default_output_dir, help="Directory to save visualizations")
    parser.add_argument("--label1", default=method1_label, help="Label for first dataset")
    parser.add_argument("--label2", default=method2_label, help="Label for second dataset")
    parser.add_argument("--max_pairs", type=int, default=100, help="Maximum number of connected sample pairs to display")
    parser.add_argument("--visualization", choices=["umap", "correlation", "2d_correlation", "distance_correlation", "ratio_preservation", "all"], default="all", 
                        help="Type of visualization to create")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualizations based on the selected type
    if args.visualization in ["umap", "all"]:
        create_comparative_umap(
            args.csv1, 
            args.csv2, 
            args.output_dir,
            label1=args.label1,
            label2=args.label2,
            max_pairs=args.max_pairs
        )
    
    if args.visualization in ["correlation", "all"]:
        create_correlation_scatter_plot(
            args.csv1,
            args.csv2,
            args.output_dir,
            label1=args.label1,
            label2=args.label2
        )
    
    if args.visualization in ["2d_correlation", "all"]:
        create_2d_correlation_scatter_plot(
            args.csv1,
            args.csv2,
            args.output_dir,
            label1=args.label1,
            label2=args.label2
        )
        
    if args.visualization in ["distance_correlation", "all"]:
        create_distance_correlation_plot(
            args.csv1,
            args.csv2,
            args.output_dir,
            label1=args.label1,
            label2=args.label2
        )
        
    if args.visualization in ["ratio_preservation", "all"]:
        analyze_distance_ratio_preservation(
            args.csv1,
            args.csv2,
            args.output_dir,
            label1=args.label1,
            label2=args.label2
        )

if __name__ == "__main__":
    main() 