"""
Synapse Preprocessing Comparison Tool

This script creates a comparative UMAP visualization from two feature CSV files,
showing how different preprocessing methods affect the feature representation.

Usage:
    python compare_csvs.py [--csv_file1 CSV_FILE1] [--csv_file2 CSV_FILE2] [--output_dir OUTPUT_DIR]

Example:
    python compare_csvs.py 
        --csv_file1 "results/run_1/features_intelligent_crop.csv" 
        --csv_file2 "results/run_2/features_normal_crop.csv" 
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
            ids = df['id'].tolist()
        else:
            # Create arbitrary ids based on row number
            ids = [f"sample_{i}" for i in range(len(df))]
        
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
            # Sort common_ids by distance to select the most interesting pairs
            distances = []
            for sample_id in common_ids:
                idx1 = sample_ids[0].index(sample_id)
                idx2 = sample_ids[1].index(sample_id)
                x1, y1 = feature_sets[0][idx1, 0], feature_sets[0][idx1, 1]
                x2, y2 = feature_sets[1][idx2, 0], feature_sets[1][idx2, 1]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append((sample_id, distance))
            
            # Sort by distance and take the top max_pairs
            distances.sort(key=lambda x: x[1], reverse=True)
            common_ids = [x[0] for x in distances[:max_pairs]]
            print(f"Selected {max_pairs} sample pairs with the largest distances for visualization")
    
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
            note_text += f"Selected {max_pairs} pairs with largest distances."
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

def main():
    # Add debug print
    print("Debug: Starting main function")
    # # default paths to use if no CSV files are provided for 50 rows of data
    # default_csv_file1 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\50\run_2025-03-18_19-08-06\features_extraction_stage_specific_layer20_segNone_alphaNone\features_layer20_segNone_alphaNone.csv"
    # default_csv_file2 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\50\run_2025-03-18_18-12-54\features_extraction_stage_specific_layer20_segNone_alphaNone_intelligent_crop_w7\features_layer20_segNone_alphaNone.csv"
    # default paths to use if no CSV files are provided for 100 rows of data
    default_csv_file1 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\100\run_2025-03-18_19-08-06\features_extraction_stage_specific_layer20_segNone_alphaNone\features_layer20_segNone_alphaNone.csv"
    default_csv_file2 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\100\run_2025-03-18_18-12-54\features_extraction_stage_specific_layer20_segNone_alphaNone_intelligent_crop_w7\features_layer20_segNone_alphaNone.csv"
    default_output_dir = r"C:\Users\alim9\Documents\codes\synapse2\results\features\100\run_2025-03-18_19-08-06\features_extraction_stage_specific_layer20_segNone_alphaNone"
    #default paths to use if no CSV files are provided for 100 for feature extraction with 100 rows of data
    default_csv_file1 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\100\run_2025-03-19_00-13-14\features_extraction_stage_specific_layer20_segNone_alphaNone_intelligent_crop_w7\features_layer20_segNone_alphaNone.csv"
    default_csv_file2 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\100\run_2025-03-19_00-38-49\features_extraction_standard_segNone_alphaNone_intelligent_crop_w7\features_segNone_alphaNone.csv"
    default_output_dir = r"C:\Users\alim9\Documents\codes\synapse2\results\features\100\extraction_100_rows"
        #default paths to use if no CSV files are provided for 50 for feature extraction with 50 rows of data
    default_csv_file1 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\50\run_2025-03-19_00-13-14\features_extraction_stage_specific_layer20_segNone_alphaNone_intelligent_crop_w7\features_layer20_segNone_alphaNone.csv"
    default_csv_file2 = r"C:\Users\alim9\Documents\codes\synapse2\results\features\50\run_2025-03-19_00-38-49\features_extraction_standard_segNone_alphaNone_intelligent_crop_w7\features_segNone_alphaNone.csv"
    default_output_dir = r"C:\Users\alim9\Documents\codes\synapse2\results\features\50\extraction_50_rows"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare features from two CSV files using UMAP visualization")
    parser.add_argument("--csv_file1", default=default_csv_file1, help="Path to first feature CSV file")
    parser.add_argument("--csv_file2", default=default_csv_file2, help="Path to second feature CSV file")
    parser.add_argument("--output_dir", default=default_output_dir, help="Directory to save results")
    parser.add_argument("--label1", default="Intelligent Cropping", help="Label for first dataset")
    parser.add_argument("--label2", default="Normal Cropping", help="Label for second dataset")
    parser.add_argument("--max_pairs", type=int, help="Maximum number of sample pairs to display with connections")

    try:
        args = parser.parse_args()
        
        print(f"Debug: Command line arguments parsed successfully")
        print(f"Debug: csv_file1 = {args.csv_file1}")
        print(f"Debug: csv_file2 = {args.csv_file2}")
        print(f"Debug: output_dir = {args.output_dir}")
        
        # Check if files exist
        if not os.path.exists(args.csv_file1):
            print(f"Error: File not found: {args.csv_file1}")
            return
        
        if not os.path.exists(args.csv_file2):
            print(f"Error: File not found: {args.csv_file2}")
            return
        
        print(f"Debug: Both input files exist")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Debug: Output directory created/verified: {args.output_dir}")
        
        # Create comparative UMAP visualization
        try:
            create_comparative_umap(args.csv_file1, args.csv_file2, args.output_dir, args.label1, args.label2, args.max_pairs)
            print(f"Debug: create_comparative_umap function completed")
        except Exception as e:
            print(f"Error in create_comparative_umap: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("Comparison completed successfully!")
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 