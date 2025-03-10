import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import imageio
import shutil

from presynapse_distance_analysis import compare_intra_inter_presynapse_distances, add_distance_comparison_to_report

from synapse import (
    SynapseDataLoader,
    Synapse3DProcessor,
    SynapseDataset,
    config
)


def load_feature_data(feature_csv_path):
    print(f"Loading feature data from {feature_csv_path}")
    if os.path.exists(feature_csv_path):
        features_df = pd.read_csv(feature_csv_path)
        print(f"Loaded {len(features_df)} rows of feature data")
        return features_df
    else:
        print(f"Feature file {feature_csv_path} not found")
        return None


def identify_synapses_with_same_presynapse(seg_vol_dict, features_df):
    print("Identifying synapses with the same presynapse ID based on segmentation data and coordinates")
    
    presynapse_groups = {}
    
    if 'presynapse_id' not in features_df.columns:
        features_df['presynapse_id'] = -1
    if 'postsynapse_id' not in features_df.columns:
        features_df['postsynapse_id'] = -1
    
    presynapse_count = 0
    postsynapse_count = 0
    
    for idx, row in features_df.iterrows():
        bbox_name = row['bbox_name']
        
        if bbox_name not in seg_vol_dict:
            continue
        
        _, seg_vol, add_mask_vol = seg_vol_dict[bbox_name]
        
        required_cols = ['side_1_coord_1', 'side_1_coord_2', 'side_1_coord_3', 
                        'side_2_coord_1', 'side_2_coord_2', 'side_2_coord_3']
        if not all(col in row for col in required_cols):
            print(f"Warning: Missing coordinate columns for synapse at index {idx}")
            continue
        
        try:
            x1, y1, z1 = int(row['side_1_coord_1']), int(row['side_1_coord_2']), int(row['side_1_coord_3'])
            x2, y2, z2 = int(row['side_2_coord_1']), int(row['side_2_coord_2']), int(row['side_2_coord_3'])
            
            if (0 <= z1 < seg_vol.shape[0] and 0 <= y1 < seg_vol.shape[1] and 0 <= x1 < seg_vol.shape[2] and
                0 <= z2 < seg_vol.shape[0] and 0 <= y2 < seg_vol.shape[1] and 0 <= x2 < seg_vol.shape[2]):
                
                seg_id_1 = seg_vol[z1, y1, x1]
                seg_id_2 = seg_vol[z2, y2, x2]
                
                if seg_id_1 > 0 and features_df.at[idx, 'segmentation_type'] == 1:
                    features_df.at[idx, 'presynapse_id'] = int(seg_id_1)
                    if seg_id_2 > 0:
                        features_df.at[idx, 'postsynapse_id'] = int(seg_id_2)
                    presynapse_count += 1
                    
                elif seg_id_2 > 0 and features_df.at[idx, 'segmentation_type'] == 2:
                    features_df.at[idx, 'postsynapse_id'] = int(seg_id_1)
                    if seg_id_2 > 0:
                        features_df.at[idx, 'presynapse_id'] = int(seg_id_2)
                    postsynapse_count += 1
                    
                elif add_mask_vol is not None:
                    if add_mask_vol.ndim >= 4 and add_mask_vol.shape[3] > 0:
                        vesicle_mask = add_mask_vol[:, :, :, 0]
                        if 0 <= z1 < vesicle_mask.shape[0] and 0 <= y1 < vesicle_mask.shape[1] and 0 <= x1 < vesicle_mask.shape[2]:
                            vesicle_at_side1 = vesicle_mask[z1, y1, x1] > 0
                        else:
                            vesicle_at_side1 = False
                            
                        if 0 <= z2 < vesicle_mask.shape[0] and 0 <= y2 < vesicle_mask.shape[1] and 0 <= x2 < vesicle_mask.shape[2]:
                            vesicle_at_side2 = vesicle_mask[z2, y2, x2] > 0
                        else:
                            vesicle_at_side2 = False
                        
                        if vesicle_at_side1 and seg_id_1 > 0:
                            features_df.at[idx, 'presynapse_id'] = int(seg_id_1)
                            if seg_id_2 > 0:
                                features_df.at[idx, 'postsynapse_id'] = int(seg_id_2)
                            presynapse_count += 1
                        elif vesicle_at_side2 and seg_id_2 > 0:
                            features_df.at[idx, 'presynapse_id'] = int(seg_id_2)
                            if seg_id_1 > 0:
                                features_df.at[idx, 'postsynapse_id'] = int(seg_id_1)
                            presynapse_count += 1
                        elif seg_id_1 > 0:
                            features_df.at[idx, 'presynapse_id'] = int(seg_id_1)
                            if seg_id_2 > 0:
                                features_df.at[idx, 'postsynapse_id'] = int(seg_id_2)
                            presynapse_count += 1
                    else:
                        if seg_id_1 > 0:
                            features_df.at[idx, 'presynapse_id'] = int(seg_id_1)
                            if seg_id_2 > 0:
                                features_df.at[idx, 'postsynapse_id'] = int(seg_id_2)
                            presynapse_count += 1
            else:
                print(f"Warning: Coordinates out of bounds for synapse at index {idx}")
                
        except (ValueError, IndexError, KeyError) as e:
            print(f"Error processing synapse at index {idx}: {e}")
    
    print(f"Identified {presynapse_count} synapses with presynapse IDs and {postsynapse_count} with postsynapse IDs")
    
    for pre_id, group_df in features_df.groupby('presynapse_id'):
        if pre_id > 0 and len(group_df) > 1:
            presynapse_groups[pre_id] = group_df.index.tolist()
    
    print(f"Found {len(presynapse_groups)} presynapse IDs with multiple synapses")
    return presynapse_groups, features_df


def calculate_feature_distances(features_df, presynapse_groups):
    print("Calculating feature distances")
    
    distance_matrices = {}
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    
    for pre_id, synapse_indices in presynapse_groups.items():
        synapse_features = features_df.loc[synapse_indices, feature_cols].values
        
        distances = euclidean_distances(synapse_features)
        
        triu_indices = np.triu_indices_from(distances, k=1)
        
        triu_values = distances[triu_indices]
        
        if len(triu_values) > 0:
            mean_distance = np.mean(triu_values)
            max_distance = np.max(triu_values)
            min_distance = np.min(triu_values)
        else:
            mean_distance = 0
            max_distance = 0
            min_distance = 0
        
        distance_matrices[pre_id] = {
            'indices': synapse_indices,
            'distances': distances,
            'mean_distance': mean_distance,
            'max_distance': max_distance,
            'min_distance': min_distance
        }
    
    return distance_matrices


def analyze_cluster_membership(features_df, presynapse_groups):
    print("Analyzing cluster membership")
    
    cluster_info = {}
    
    if 'cluster' not in features_df.columns:
        print("No cluster information available in the feature data")
        return cluster_info
    
    for pre_id, synapse_indices in presynapse_groups.items():
        synapse_clusters = features_df.loc[synapse_indices, 'cluster'].values
        
        unique_clusters, counts = np.unique(synapse_clusters, return_counts=True)
        
        cluster_info[pre_id] = {
            'indices': synapse_indices,
            'clusters': synapse_clusters,
            'unique_clusters': unique_clusters,
            'cluster_counts': counts,
            'num_clusters': len(unique_clusters),
            'dominant_cluster': unique_clusters[np.argmax(counts)],
            'dominant_cluster_count': np.max(counts),
            'dominant_cluster_percentage': (np.max(counts) / len(synapse_indices)) * 100
        }
    
    return cluster_info


def create_distance_heatmaps(output_dir, distance_matrices, features_df):
    print("Creating distance heatmaps")
    
    heatmap_dir = os.path.join(output_dir, "distance_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    for pre_id, data in distance_matrices.items():
        indices = data['indices']
        distances = data['distances']
        
        if 'Var1' in features_df.columns:
            labels = features_df.loc[indices, 'Var1'].values
        else:
            labels = [f"Synapse {i}" for i in indices]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            distances, 
            annot=True, 
            fmt=".2f", 
            cmap="viridis", 
            xticklabels=labels, 
            yticklabels=labels
        )
        plt.title(f"Feature Distance Matrix for Presynapse ID {pre_id}")
        plt.tight_layout()
        
        plt.savefig(os.path.join(heatmap_dir, f"distance_heatmap_pre{pre_id}.png"), dpi=300)
        plt.close()
    
    print(f"Saved distance heatmaps to {heatmap_dir}")


def create_connected_umap(features_df, presynapse_groups, output_dir):
    print("Creating connected UMAP visualization for synapses sharing the same presynapse ID")
    
    if 'umap_x' not in features_df.columns or 'umap_y' not in features_df.columns:
        print("No UMAP coordinates found in features data")
        
        feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
        if len(feature_cols) > 0:
            print("Computing UMAP from feature data")
            features = features_df[feature_cols].values
            features_scaled = StandardScaler().fit_transform(features)
            
            reducer = umap.UMAP(random_state=42)
            umap_results = reducer.fit_transform(features_scaled)
            
            features_df['umap_x'] = umap_results[:, 0]
            features_df['umap_y'] = umap_results[:, 1]
            print("Added UMAP coordinates to features data")
        else:
            print("No feature columns found, cannot compute UMAP")
            return None
    
    # Create standard connected UMAP visualization (improved version)
    create_standard_connected_umap(features_df, presynapse_groups, output_dir)
    
    # Create a dedicated bbox-colored UMAP visualization
    create_bbox_colored_umap(features_df, output_dir)
    
    # Create a dedicated cluster-colored UMAP visualization (if cluster info available)
    if 'cluster' in features_df.columns:
        create_cluster_colored_umap(features_df, output_dir)
    
    # Create an interactive version with all information
    create_interactive_umap(features_df, presynapse_groups, output_dir)
    
    # Return the path to the main visualization
    return os.path.join(output_dir, "connected_umap_visualization.png")


def create_standard_connected_umap(features_df, presynapse_groups, output_dir):
    """Create the standard connected UMAP visualization with presynapse connections"""
    presynapse_ids = list(presynapse_groups.keys())
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    n_colors = len(presynapse_ids)
    if n_colors <= 10:
        colormap = plt.cm.get_cmap('tab10', n_colors)
    else:
        colormap = plt.cm.get_cmap('viridis', n_colors)
    
    color_dict = {pre_id: mcolors.rgb2hex(colormap(i)) 
                 for i, pre_id in enumerate(presynapse_ids)}
    
    plt.figure(figsize=(14, 12))
    
    plt.scatter(
        features_df['umap_x'], 
        features_df['umap_y'], 
        c='lightgray', 
        alpha=0.5, 
        s=20
    )
    
    for pre_id, indices in presynapse_groups.items():
        group_df = features_df.loc[indices]
        color = color_dict[pre_id]
        
        plt.scatter(
            group_df['umap_x'],
            group_df['umap_y'],
            c=color,
            s=100,
            alpha=0.8,
            edgecolors='black',
            label=f'Pre ID: {pre_id} ({len(indices)} synapses)'
        )
        
        if len(indices) > 1:
            for i, idx1 in enumerate(indices):
                row1 = features_df.loc[idx1]
                x1, y1 = row1['umap_x'], row1['umap_y']
                
                for idx2 in indices[i+1:]:
                    row2 = features_df.loc[idx2]
                    x2, y2 = row2['umap_x'], row2['umap_y']
                    
                    plt.plot([x1, x2], [y1, y2], color=color, alpha=0.4, linewidth=1.5)
        
        if 'Var1' in features_df.columns:
            for idx, row in group_df.iterrows():
                plt.annotate(
                    row['Var1'], 
                    (row['umap_x'], row['umap_y']),
                    fontsize=8,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
    
    if 'cluster' in features_df.columns:
        plt.title('UMAP Visualization with Connected Presynapse Groups and Cluster Information')
        
        unique_clusters = features_df['cluster'].unique()
        
        for cluster_id in unique_clusters:
            cluster_points = features_df[features_df['cluster'] == cluster_id]
            
            if len(cluster_points) < 5:
                continue
                
            centroid_x = cluster_points['umap_x'].mean()
            centroid_y = cluster_points['umap_y'].mean()
            
            plt.text(
                centroid_x, 
                centroid_y, 
                f'Cluster {cluster_id}', 
                fontsize=12, 
                weight='bold',
                alpha=0.7,
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
            )
    else:
        plt.title('UMAP Visualization with Connected Presynapse Groups')
    
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    if len(presynapse_ids) <= 15:
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    else:
        legend_ids = presynapse_ids[:15]
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[pre_id], 
                                    markersize=10, label=f'Pre ID: {pre_id}') 
                         for pre_id in legend_ids]
        plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1), 
                  title="Presynapse IDs (top 15)")
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "connected_umap_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Standard connected UMAP visualization saved to {output_path}")


def create_bbox_colored_umap(features_df, output_dir):
    """Create a UMAP visualization specifically colored by bounding box"""
    if 'bbox_name' not in features_df.columns:
        print("No bbox_name column in features data, skipping bbox-colored UMAP")
        return
    
    plt.figure(figsize=(14, 12))
    
    # Define a consistent color map for bounding boxes
    bbox_colors = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    
    # Get all unique bounding boxes
    unique_bboxes = features_df['bbox_name'].unique()
    
    # Plot points for each bounding box with its own color
    for bbox in unique_bboxes:
        bbox_df = features_df[features_df['bbox_name'] == bbox]
        color = bbox_colors.get(bbox, 'gray')  # Use defined color or default to gray
        
        plt.scatter(
            bbox_df['umap_x'],
            bbox_df['umap_y'],
            c=color,
            s=80,
            alpha=0.8,
            edgecolors='black',
            label=f'{bbox} ({len(bbox_df)} synapses)'
        )
    
    plt.title('UMAP Visualization Colored by Bounding Box')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    
    # Add legend
    if len(unique_bboxes) <= 10:
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "umap_bbox_colored.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Bounding box colored UMAP visualization saved to {output_path}")


def create_cluster_colored_umap(features_df, output_dir):
    """Create a UMAP visualization specifically colored by cluster"""
    plt.figure(figsize=(14, 12))
    
    # Get all unique clusters
    unique_clusters = features_df['cluster'].unique()
    
    # Add a categorical cluster column for coloring
    features_df = features_df.copy()  # Create a copy to avoid modifying the original
    features_df['cluster_str'] = 'Cluster ' + features_df['cluster'].astype(str)
    
    # Create a scatter plot with points colored by cluster - use a categorical color mapping
    # Instead of a continuous colormap like 'viridis', use a list of distinct colors
    from matplotlib.colors import ListedColormap
    import numpy as np
    
    # Create a colormap with distinct colors for each cluster
    num_clusters = len(unique_clusters)
    # Use a qualitative colormap with distinct colors 
    # We'll use tab10, tab20, or Set3 depending on the number of clusters
    if num_clusters <= 10:
        cmap_name = 'tab10'
    elif num_clusters <= 20:
        cmap_name = 'tab20'
    else:
        cmap_name = 'Set3'
        
    # Create a scatter plot with categorical colors
    scatter = plt.scatter(
        features_df['umap_x'],
        features_df['umap_y'],
        c=features_df['cluster'].astype('category').cat.codes,  # Use category codes for distinct coloring
        cmap=cmap_name,
        s=80,
        alpha=0.8,
        edgecolors='black'
    )
    
    # Add a legend with cluster names
    handles, labels = scatter.legend_elements()
    legend_labels = [f'Cluster {c}' for c in unique_clusters]
    plt.legend(handles, legend_labels, title="Clusters", loc='best')
    
    # Add annotations for cluster centers
    for cluster_id in unique_clusters:
        cluster_points = features_df[features_df['cluster'] == cluster_id]
        
        centroid_x = cluster_points['umap_x'].mean()
        centroid_y = cluster_points['umap_y'].mean()
        
        plt.text(
            centroid_x, 
            centroid_y, 
            f'Cluster {cluster_id}', 
            fontsize=12, 
            weight='bold',
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5')
        )
        
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.title('UMAP Visualization Colored by Cluster', fontsize=16)
    plt.tight_layout()
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'umap_cluster_colored.png'), dpi=300, bbox_inches='tight')
    
    # Also create an interactive plotly version
    from plotly.offline import plot
    import plotly.express as px
    
    # Create interactive plot with categorical colors for clusters
    features_df_for_plotly = features_df.copy()
    features_df_for_plotly['cluster_str'] = 'Cluster ' + features_df_for_plotly['cluster'].astype(str)
    
    fig = px.scatter(
        features_df_for_plotly, 
        x='umap_x', 
        y='umap_y',
        color='cluster_str',  # Use categorical string version for coloring
        hover_data=['bbox_name'],
        title='Interactive UMAP Visualization Colored by Cluster',
        color_discrete_sequence=px.colors.qualitative.Bold  # Use a qualitative color scale
    )
    
    # Set point size and opacity
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    
    # Add cluster center labels
    for cluster_id in unique_clusters:
        cluster_points = features_df[features_df['cluster'] == cluster_id]
        centroid_x = cluster_points['umap_x'].mean()
        centroid_y = cluster_points['umap_y'].mean()
        
        fig.add_annotation(
            x=centroid_x,
            y=centroid_y,
            text=f'Cluster {cluster_id}',
            showarrow=False,
            font=dict(size=14, color='black'),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1,
            borderpad=4
        )
    
    # Save as interactive HTML
    plot(fig, filename=os.path.join(output_dir, 'umap_cluster_colored.html'), auto_open=False)
    
    plt.close()
    
    return features_df


def create_interactive_umap(features_df, presynapse_groups, output_dir):
    """Create an interactive UMAP visualization with all information"""
    try:
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        import plotly.graph_objects as go
        
        presynapse_ids = list(presynapse_groups.keys())
        n_colors = len(presynapse_ids)
        if n_colors <= 10:
            colormap = plt.cm.get_cmap('tab10', n_colors)
        else:
            colormap = plt.cm.get_cmap('viridis', n_colors)
        
        color_dict = {pre_id: mcolors.rgb2hex(colormap(i)) 
                     for i, pre_id in enumerate(presynapse_ids)}
        
        plot_df = features_df.copy()
        plot_df['presynapse_id'] = -1
        
        for pre_id, indices in presynapse_groups.items():
            plot_df.loc[indices, 'presynapse_id'] = pre_id
        
        hover_text = []
        for idx, row in plot_df.iterrows():
            hover_info = f"Index: {idx}<br>"
            
            if 'Var1' in row:
                hover_info += f"Synapse: {row['Var1']}<br>"
                
            if 'bbox_name' in row:
                hover_info += f"BBox: {row['bbox_name']}<br>"
                
            if 'cluster' in row:
                hover_info += f"Cluster: {row['cluster']}<br>"
                
            hover_info += f"Presynapse ID: {int(row['presynapse_id'])}"
            hover_text.append(hover_info)
        
        fig = go.Figure()
        
        background_points = plot_df[plot_df['presynapse_id'] == -1]
        if len(background_points) > 0:
            fig.add_trace(go.Scatter(
                x=background_points['umap_x'],
                y=background_points['umap_y'],
                mode='markers',
                marker=dict(color='lightgray', size=5, opacity=0.5),
                text=[hover_text[i] for i in background_points.index],
                hoverinfo='text',
                name='Other Synapses'
            ))
        
        for pre_id, indices in presynapse_groups.items():
            group_df = plot_df.loc[indices]
            
            fig.add_trace(go.Scatter(
                x=group_df['umap_x'],
                y=group_df['umap_y'],
                mode='markers',
                marker=dict(color=color_dict[pre_id], size=10, line=dict(width=1, color='black')),
                text=[hover_text[i] for i in group_df.index],
                hoverinfo='text',
                name=f'Pre ID: {pre_id}'
            ))
            
            if len(indices) > 1:
                for i, idx1 in enumerate(indices):
                    row1 = plot_df.loc[idx1]
                    x1, y1 = row1['umap_x'], row1['umap_y']
                    
                    for idx2 in indices[i+1:]:
                        row2 = plot_df.loc[idx2]
                        x2, y2 = row2['umap_x'], row2['umap_y']
                        
                        fig.add_trace(go.Scatter(
                            x=[x1, x2, None],
                            y=[y1, y2, None],
                            mode='lines',
                            line=dict(color=color_dict[pre_id], width=1, dash='solid'),
                            opacity=0.4,
                            showlegend=False,
                            hoverinfo='none'
                        ))
        
        fig.update_layout(
            title='Interactive UMAP Visualization with Connected Presynapse Groups',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            hovermode='closest',
            width=1200,
            height=900
        )
        
        html_path = os.path.join(output_dir, "connected_umap_interactive.html")
        fig.write_html(html_path)
        print(f"Interactive UMAP visualization saved to {html_path}")
    
    except Exception as e:
        print(f"Error creating interactive UMAP visualization: {e}")


def create_cluster_visualizations(output_dir, cluster_info, features_df):
    print("Creating cluster visualizations")
    
    cluster_dir = os.path.join(output_dir, "cluster_visualizations")
    os.makedirs(cluster_dir, exist_ok=True)
    
    has_umap = 'umap_x' in features_df.columns and 'umap_y' in features_df.columns
    
    for pre_id, info in cluster_info.items():
        indices = info['indices']
        clusters = info['clusters']
        
        plt.figure(figsize=(10, 6))
        unique_clusters = info['unique_clusters']
        counts = info['cluster_counts']
        
        plt.bar(unique_clusters, counts)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Synapses')
        plt.title(f"Cluster Distribution for Presynapse ID {pre_id}")
        plt.xticks(unique_clusters)
        plt.tight_layout()
        
        plt.savefig(os.path.join(cluster_dir, f"cluster_dist_pre{pre_id}.png"), dpi=300)
        plt.close()
        
        if has_umap:
            plt.figure(figsize=(12, 10))
            
            plt.scatter(
                features_df['umap_x'], 
                features_df['umap_y'], 
                c='lightgray', 
                alpha=0.5, 
                s=30
            )
            
            pre_synapses = features_df.loc[indices]
            plt.scatter(
                pre_synapses['umap_x'], 
                pre_synapses['umap_y'], 
                c=clusters, 
                cmap='viridis', 
                s=100, 
                edgecolors='black'
            )
            
            if 'Var1' in features_df.columns:
                for idx, row in pre_synapses.iterrows():
                    plt.annotate(
                        row['Var1'], 
                        (row['umap_x'], row['umap_y']),
                        fontsize=8,
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
            
            plt.title(f"UMAP Visualization of Synapses with Presynapse ID {pre_id}")
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.colorbar(label='Cluster')
            plt.tight_layout()
            
            plt.savefig(os.path.join(cluster_dir, f"umap_pre{pre_id}.png"), dpi=300)
            plt.close()
    
    presynapse_groups = {pre_id: info['indices'] for pre_id, info in cluster_info.items()}
    if not presynapse_groups:
        if 'presynapse_id' in features_df.columns:
            for pre_id, group_df in features_df.groupby('presynapse_id'):
                if pre_id > 0 and len(group_df) > 1:
                    presynapse_groups[pre_id] = group_df.index.tolist()
    
    if presynapse_groups:
        create_connected_umap(features_df, presynapse_groups, cluster_dir)
    
    print(f"Saved cluster visualizations to {cluster_dir}")


def compare_intra_inter_presynapse_distances(features_df, presynapse_groups, output_dir):
    print("Comparing intra-presynapse and inter-presynapse distances")
    
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    if not feature_cols:
        print("No feature columns found, cannot compute distances")
        return None
    
    features = features_df[feature_cols].values
    
    index_to_presynapse = {}
    for pre_id, indices in presynapse_groups.items():
        for idx in indices:
            index_to_presynapse[idx] = pre_id
    
    intra_presynapse_distances = []
    inter_presynapse_distances = []
    
    print("Calculating pairwise distances between all synapses...")
    all_distances = euclidean_distances(features)
    
    for i in range(len(features_df)):
        for j in range(i+1, len(features_df)):
            if i not in index_to_presynapse or j not in index_to_presynapse:
                continue
            
            distance = all_distances[i, j]
            
            if index_to_presynapse[i] == index_to_presynapse[j]:
                intra_presynapse_distances.append(distance)
            else:
                inter_presynapse_distances.append(distance)
    
    if intra_presynapse_distances and inter_presynapse_distances:
        intra_mean = np.mean(intra_presynapse_distances)
        intra_std = np.std(intra_presynapse_distances)
        intra_min = np.min(intra_presynapse_distances)
        intra_max = np.max(intra_presynapse_distances)
        
        inter_mean = np.mean(inter_presynapse_distances)
        inter_std = np.std(inter_presynapse_distances)
        inter_min = np.min(inter_presynapse_distances)
        inter_max = np.max(inter_presynapse_distances)
        
        ratio = intra_mean / inter_mean if inter_mean > 0 else 0
        
        print(f"Intra-presynapse distance (mean ± std): {intra_mean:.4f} ± {intra_std:.4f}")
        print(f"Inter-presynapse distance (mean ± std): {inter_mean:.4f} ± {inter_std:.4f}")
        print(f"Ratio (intra/inter): {ratio:.4f}")
        
        plt.figure(figsize=(12, 8))
        
        bins = np.linspace(min(intra_min, inter_min), max(intra_max, inter_max), 30)
        plt.hist(intra_presynapse_distances, bins=bins, alpha=0.7, label=f'Same Presynapse (n={len(intra_presynapse_distances)})')
        plt.hist(inter_presynapse_distances, bins=bins, alpha=0.7, label=f'Different Presynapse (n={len(inter_presynapse_distances)})')
        
        plt.axvline(intra_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Same Presynapse Mean: {intra_mean:.4f}')
        plt.axvline(inter_mean, color='orange', linestyle='dashed', linewidth=2, label=f'Different Presynapse Mean: {inter_mean:.4f}')
        
        plt.title('Distribution of Distances: Same vs. Different Presynapse ID')
        plt.xlabel('Euclidean Distance in Feature Space')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        hist_path = os.path.join(output_dir, "distance_comparison_histogram.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        
        distance_df = pd.DataFrame({
            'Distance': intra_presynapse_distances + inter_presynapse_distances,
            'Group': ['Same Presynapse'] * len(intra_presynapse_distances) + 
                     ['Different Presynapse'] * len(inter_presynapse_distances)
        })
        
        sns.boxplot(x='Group', y='Distance', data=distance_df)
        sns.stripplot(x='Group', y='Distance', data=distance_df, 
                     size=4, color='.3', alpha=0.3)
        
        plt.title('Distance Distribution: Same vs. Different Presynapse ID')
        plt.ylabel('Euclidean Distance in Feature Space')
        plt.tight_layout()
        
        box_path = os.path.join(output_dir, "distance_comparison_boxplot.png")
        plt.savefig(box_path, dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.violinplot(x='Group', y='Distance', data=distance_df, inner='quartile')
        plt.title('Distance Distribution: Same vs. Different Presynapse ID')
        plt.ylabel('Euclidean Distance in Feature Space')
        plt.tight_layout()
        
        violin_path = os.path.join(output_dir, "distance_comparison_violinplot.png")
        plt.savefig(violin_path, dpi=300)
        plt.close()
        
        per_presynapse_stats = {}
        for pre_id, indices in presynapse_groups.items():
            if len(indices) > 1:
                distances = []
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i+1:]:
                        distances.append(all_distances[idx1, idx2])
                
                other_distances = []
                for idx1 in indices:
                    for pre_id2, indices2 in presynapse_groups.items():
                        if pre_id != pre_id2:
                            for idx2 in indices2:
                                other_distances.append(all_distances[idx1, idx2])
                
                per_presynapse_stats[pre_id] = {
                    'intra_mean': np.mean(distances) if distances else 0,
                    'intra_std': np.std(distances) if distances else 0,
                    'inter_mean': np.mean(other_distances) if other_distances else 0,
                    'inter_std': np.std(other_distances) if other_distances else 0,
                    'ratio': np.mean(distances) / np.mean(other_distances) if other_distances and np.mean(other_distances) > 0 else 0,
                    'num_synapses': len(indices)
                }
        
        plt.figure(figsize=(14, 10))
        pre_ids = list(per_presynapse_stats.keys())
        ratios = [per_presynapse_stats[pre_id]['ratio'] for pre_id in pre_ids]
        
        sorted_indices = np.argsort(ratios)
        sorted_pre_ids = [pre_ids[i] for i in sorted_indices]
        sorted_ratios = [ratios[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_ratios)))
        
        bars = plt.bar(range(len(sorted_pre_ids)), sorted_ratios, color=colors)
        
        plt.axhline(ratio, color='red', linestyle='--', label=f'Overall Average: {ratio:.4f}')
        
        for i, (pre_id, bar) in enumerate(zip(sorted_pre_ids, bars)):
            num_syn = per_presynapse_stats[pre_id]['num_synapses']
            plt.text(i, 0.02, f"n={num_syn}", ha='center', va='bottom', color='black')
        
        plt.title('Distance Ratio (Intra/Inter) by Presynapse ID\nLower is Better')
        plt.xlabel('Presynapse ID')
        plt.ylabel('Ratio of Intra-Presynapse Distance to Inter-Presynapse Distance')
        plt.xticks(range(len(sorted_pre_ids)), sorted_pre_ids, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        bar_path = os.path.join(output_dir, "distance_ratio_by_presynapse.png")
        plt.savefig(bar_path, dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        intra_means = [per_presynapse_stats[pre_id]['intra_mean'] for pre_id in pre_ids]
        inter_means = [per_presynapse_stats[pre_id]['inter_mean'] for pre_id in pre_ids]
        sizes = [per_presynapse_stats[pre_id]['num_synapses'] * 20 for pre_id in pre_ids]
        
        min_val = min(min(intra_means), min(inter_means))
        max_val = max(max(intra_means), max(inter_means))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        scatter = plt.scatter(intra_means, inter_means, c=ratios, cmap='viridis', 
                            s=sizes, alpha=0.7, edgecolors='black')
        
        for i, pre_id in enumerate(pre_ids):
            plt.annotate(str(pre_id), (intra_means[i], inter_means[i]), 
                        fontsize=8, ha='center', va='center')
        
        plt.colorbar(scatter, label='Intra/Inter Ratio (lower is better)')
        plt.title('Intra-Presynapse vs. Inter-Presynapse Distances')
        plt.xlabel('Average Distance Between Synapses with Same Presynapse ID')
        plt.ylabel('Average Distance to Synapses with Different Presynapse ID')
        plt.tight_layout()
        
        scatter_path = os.path.join(output_dir, "intra_vs_inter_distance_scatter.png")
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        
        summary = {
            'intra_mean': intra_mean,
            'intra_std': intra_std,
            'inter_mean': inter_mean,
            'inter_std': inter_std,
            'ratio': ratio,
            'n_intra': len(intra_presynapse_distances),
            'n_inter': len(inter_presynapse_distances),
            'per_presynapse': per_presynapse_stats,
            'plots': {
                'histogram': hist_path,
                'boxplot': box_path,
                'violinplot': violin_path,
                'barplot': bar_path,
                'scatterplot': scatter_path
            }
        }
        
        return summary
    else:
        print("Not enough data to compare distances")
        return None


def generate_report(output_dir, presynapse_groups, distance_matrices, cluster_info, features_df, distance_comparison=None):
    """
    Generate a comprehensive report on synapses with the same presynapse ID.
    
    Args:
        output_dir (str): Directory to save the report
        presynapse_groups (dict): Dictionary with presynapse IDs as keys and lists of synapse indices as values
        distance_matrices (dict): Dictionary with presynapse IDs as keys and distance matrices as values
        cluster_info (dict): Dictionary with presynapse IDs as keys and cluster info as values
        features_df (pd.DataFrame): DataFrame containing feature data
        distance_comparison (dict, optional): Results from distance comparison analysis
    """
    print("Generating report")
    
    report_path = os.path.join(output_dir, "presynapse_analysis_report.html")
    with open(report_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Presynapse Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                h3 { color: #2980b9; margin-top: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .summary { background-color: #eef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .visualization { margin-top: 20px; text-align: center; }
                img { max-width: 90%; height: auto; margin: 10px; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>Presynapse Analysis Report</h1>
            
            <div class="summary">
                <h2>Analysis Summary</h2>
                <p>Total number of presynapses with multiple connections: """ + str(len(presynapse_groups)) + """</p>
                <p>Total number of synapses analyzed: """ + str(len(features_df)) + """</p>
            """)
        
        if distance_comparison:
            f.write(f"""
                <h3>Distance Comparison</h3>
                <p>Average distance between synapses with the same presynapse ID: {distance_comparison['intra_mean']:.4f} ± {distance_comparison['intra_std']:.4f}</p>
                <p>Average distance between synapses with different presynapse IDs: {distance_comparison['inter_mean']:.4f} ± {distance_comparison['inter_std']:.4f}</p>
                <p>Ratio (intra/inter): {distance_comparison['ratio']:.4f} (lower is better - indicates synapses with same presynapse are more similar)</p>
                <p>Number of intra-presynapse comparisons: {distance_comparison['n_intra']}</p>
                <p>Number of inter-presynapse comparisons: {distance_comparison['n_inter']}</p>
            """)
        
        f.write("</div>")
            
        f.write("""
            <div class="visualization">
                <h2>Global Visualizations</h2>
        """)
        
        if distance_comparison and 'plots' in distance_comparison:
            plots = distance_comparison['plots']
            
            if 'histogram' in plots:
                hist_path = os.path.basename(plots['histogram'])
                # Copy the file to the output directory for easier access in the report
                shutil.copy2(plots['histogram'], os.path.join(output_dir, hist_path))
                f.write(f"""
                    <h3>Distance Distribution Comparison</h3>
                    <img src="{hist_path}" alt="Distance Distribution Histogram">
                    <p>This histogram compares the distribution of distances between synapses sharing the same presynapse ID versus distances between synapses with different presynapse IDs.</p>
                """)
            
            if 'boxplot' in plots:
                box_path = os.path.basename(plots['boxplot'])
                # Copy the file to the output directory
                shutil.copy2(plots['boxplot'], os.path.join(output_dir, box_path))
                f.write(f"""
                    <h3>Distance Box Plot</h3>
                    <img src="{box_path}" alt="Distance Box Plot">
                    <p>This box plot shows the distribution of distances with individual points overlaid.</p>
                """)
            
            if 'violinplot' in plots:
                violin_path = os.path.basename(plots['violinplot'])
                # Copy the file to the output directory
                shutil.copy2(plots['violinplot'], os.path.join(output_dir, violin_path))
                f.write(f"""
                    <h3>Distance Violin Plot</h3>
                    <img src="{violin_path}" alt="Distance Violin Plot">
                    <p>This violin plot shows the full distribution of distances.</p>
                """)
            
            if 'barplot' in plots:
                bar_path = os.path.basename(plots['barplot'])
                # Copy the file to the output directory
                shutil.copy2(plots['barplot'], os.path.join(output_dir, bar_path))
                f.write(f"""
                    <h3>Distance Ratio by Presynapse ID</h3>
                    <img src="{bar_path}" alt="Distance Ratio by Presynapse ID">
                    <p>This plot shows the ratio of intra-presynapse to inter-presynapse distance for each presynapse ID. Lower values indicate that synapses sharing this presynapse ID are more similar to each other than to other synapses.</p>
                """)
            
            if 'scatterplot' in plots:
                scatter_path = os.path.basename(plots['scatterplot'])
                # Copy the file to the output directory
                shutil.copy2(plots['scatterplot'], os.path.join(output_dir, scatter_path))
                f.write(f"""
                    <h3>Intra vs. Inter Presynapse Distances</h3>
                    <img src="{scatter_path}" alt="Intra vs. Inter Presynapse Distances">
                    <p>This scatter plot compares the average distance between synapses with the same presynapse ID (x-axis) to the average distance to synapses with different presynapse IDs (y-axis). Points below the diagonal line indicate presynapse IDs where synapses are more similar to each other than to other synapses.</p>
                """)
        
        connected_umap_path = "cluster_visualizations/connected_umap_visualization.png"
        if os.path.exists(os.path.join(output_dir, connected_umap_path)):
            # Copy the connected UMAP visualization to the output directory
            src_path = os.path.join(output_dir, connected_umap_path)
            dest_path = os.path.join(output_dir, "connected_umap_visualization.png")
            shutil.copy2(src_path, dest_path)
            f.write(f"""
                <h3>Connected UMAP Visualization of All Presynapse Groups</h3>
                <img src="connected_umap_visualization.png" alt="Connected UMAP Visualization">
                <p>This visualization shows all synapses in the UMAP space, with synapses sharing the same presynapse ID connected by lines.</p>
            """)
            
        interactive_umap_path = "cluster_visualizations/connected_umap_interactive.html"
        if os.path.exists(os.path.join(output_dir, interactive_umap_path)):
            # Copy the interactive UMAP visualization to the output directory
            src_path = os.path.join(output_dir, interactive_umap_path)
            dest_path = os.path.join(output_dir, "connected_umap_interactive.html")
            shutil.copy2(src_path, dest_path)
            f.write(f"""
                <p><a href="connected_umap_interactive.html" target="_blank">Interactive UMAP Visualization (Click to Open)</a></p>
            """)
            
        f.write("</div>")
        
        f.write("""
            <h2>Presynapse Summary</h2>
            <table>
                <tr>
                    <th>Presynapse ID</th>
                    <th>Number of Synapses</th>
                    <th>Average Feature Distance</th>
                    <th>Number of Clusters</th>
                    <th>Dominant Cluster</th>
                    <th>Dominant Cluster Percentage</th>
        """)
        
        if distance_comparison and 'per_presynapse' in distance_comparison:
            f.write("<th>Intra/Inter Distance Ratio</th>")
        
        f.write("</tr>")
        
        for pre_id in sorted(presynapse_groups.keys()):
            num_synapses = len(presynapse_groups[pre_id])
            
            if pre_id in distance_matrices:
                avg_distance = distance_matrices[pre_id]['mean_distance']
                if isinstance(avg_distance, float):
                    avg_distance_str = f"{avg_distance:.4f}"
                else:
                    avg_distance_str = str(avg_distance)
            else:
                avg_distance_str = "N/A"
            
            if pre_id in cluster_info:
                num_clusters = cluster_info[pre_id]['num_clusters']
                dominant_cluster = cluster_info[pre_id]['dominant_cluster']
                dominant_pct = cluster_info[pre_id]['dominant_cluster_percentage']
                if isinstance(dominant_pct, float):
                    dominant_pct_str = f"{dominant_pct:.2f}%"
                else:
                    dominant_pct_str = f"{dominant_pct}%"
            else:
                num_clusters = "N/A"
                dominant_cluster = "N/A"
                dominant_pct_str = "N/A"
            
            f.write(f"""
                <tr>
                    <td>{pre_id}</td>
                    <td>{num_synapses}</td>
                    <td>{avg_distance_str}</td>
                    <td>{num_clusters}</td>
                    <td>{dominant_cluster}</td>
                    <td>{dominant_pct_str}</td>
            """)
            
            if distance_comparison and 'per_presynapse' in distance_comparison:
                per_presynapse = distance_comparison['per_presynapse']
                if pre_id in per_presynapse:
                    ratio = per_presynapse[pre_id]['ratio']
                    f.write(f"<td>{ratio:.4f}</td>")
                else:
                    f.write("<td>N/A</td>")
            
            f.write("</tr>")
        
        f.write("</table>")
        
        for pre_id in sorted(presynapse_groups.keys()):
            indices = presynapse_groups[pre_id]
            synapses = features_df.loc[indices]
            
            f.write(f"""
                <h2>Presynapse ID: {pre_id}</h2>
                
                <h3>Synapses</h3>
                <table>
                    <tr>
                        <th>Index</th>
                        <th>Bounding Box</th>
            """)
            
            if 'Var1' in features_df.columns:
                f.write("<th>Var1</th>")
            if 'slice_number' in features_df.columns:
                f.write("<th>Slice Number</th>")
            if 'cluster' in features_df.columns:
                f.write("<th>Cluster</th>")
            
            f.write("</tr>")
            
            for idx, row in synapses.iterrows():
                f.write(f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{row.get('bbox_name', 'N/A')}</td>
                """)
                
                if 'Var1' in features_df.columns:
                    f.write(f"<td>{row['Var1']}</td>")
                if 'slice_number' in features_df.columns:
                    f.write(f"<td>{row.get('slice_number', 'N/A')}</td>")
                if 'cluster' in features_df.columns:
                    f.write(f"<td>{row.get('cluster', 'N/A')}</td>")
                
                f.write("</tr>")
            
            f.write("</table>")
            
            f.write("""
                <div class="visualization">
                    <h3>Visualizations</h3>
            """)
            
            heatmap_path = f"distance_heatmaps/distance_heatmap_pre{pre_id}.png"
            if os.path.exists(os.path.join(output_dir, heatmap_path)):
                # Copy the file to make it available in the same directory as the report
                original_path = os.path.join(output_dir, heatmap_path)
                dest_filename = f"distance_heatmap_pre{pre_id}.png"
                shutil.copy2(original_path, os.path.join(output_dir, dest_filename))
                
                f.write(f"""
                    <h4>Feature Distance Heatmap</h4>
                    <img src="{dest_filename}" alt="Feature Distance Heatmap">
                """)
            
            cluster_dist_path = f"cluster_visualizations/cluster_dist_pre{pre_id}.png"
            if os.path.exists(os.path.join(output_dir, cluster_dist_path)):
                # Copy the file to make it available in the same directory as the report
                original_path = os.path.join(output_dir, cluster_dist_path)
                dest_filename = f"cluster_dist_pre{pre_id}.png"
                shutil.copy2(original_path, os.path.join(output_dir, dest_filename))
                
                f.write(f"""
                    <h4>Cluster Distribution</h4>
                    <img src="{dest_filename}" alt="Cluster Distribution">
                """)
            
            umap_path = f"cluster_visualizations/umap_pre{pre_id}.png"
            if os.path.exists(os.path.join(output_dir, umap_path)):
                # Copy the file to make it available in the same directory as the report
                original_path = os.path.join(output_dir, umap_path)
                dest_filename = f"umap_pre{pre_id}.png"
                shutil.copy2(original_path, os.path.join(output_dir, dest_filename))
                
                f.write(f"""
                    <h4>UMAP Visualization</h4>
                    <img src="{dest_filename}" alt="UMAP Visualization">
                """)
            
            f.write("</div>")
        
        f.write("""
            </body>
            </html>
        """)
    
    print(f"Generated report saved to {report_path}")


def create_gif_from_volume(volume, output_path, fps=10, loop=0):
    volume = (volume - volume.min()) / (volume.max() - volume.min()) * 255
    volume = volume.astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with imageio.get_writer(output_path, mode='I', fps=fps, loop=loop) as writer:
        for i in range(volume.shape[0]):
            writer.append_data(volume[i])
    
    print(f"GIF saved to {output_path}")


def create_gifs_for_presynapse_groups(dataset, presynapse_groups, features_df, output_dir):
    print("Creating GIFs for synapses sharing the same presynapse ID")
    
    gifs_dir = os.path.join(output_dir, "synapse_gifs")
    os.makedirs(gifs_dir, exist_ok=True)
    
    gif_count = 0
    
    for pre_id, indices in presynapse_groups.items():
        pre_dir = os.path.join(gifs_dir, f"presynapse_{pre_id}")
        os.makedirs(pre_dir, exist_ok=True)
        
        group_synapses = features_df.loc[indices]
        
        print(f"Creating GIFs for {len(indices)} synapses with presynapse ID {pre_id}")
        
        for idx, row in group_synapses.iterrows():
            bbox_name = row.get('bbox_name', '')
            var1 = row.get('Var1', f"synapse_{idx}")
            
            try:
                if isinstance(dataset, dict):
                    if bbox_name in dataset:
                        pixel_values, syn_info, _ = dataset[bbox_name][idx]
                    else:
                        print(f"Warning: No data for bbox {bbox_name}")
                        continue
                else:
                    pixel_values, syn_info, _ = dataset[idx]
                
                if hasattr(pixel_values, 'squeeze'):
                    pixel_values = pixel_values.squeeze().numpy() if hasattr(pixel_values, 'numpy') else pixel_values.squeeze()
                
                if hasattr(var1, 'replace'):
                    clean_var1 = var1.replace('/', '_').replace('\\', '_').replace(':', '_')
                else:
                    clean_var1 = f"synapse_{idx}"
                
                output_path = os.path.join(pre_dir, f"{bbox_name}_{clean_var1}.gif")
                
                create_gif_from_volume(pixel_values, output_path)
                gif_count += 1
                
            except Exception as e:
                print(f"Error creating GIF for synapse at index {idx}: {e}")
    
    print(f"Created {gif_count} GIFs for synapses with shared presynapse IDs")
    return gifs_dir


def run_presynapse_analysis(config):
    print("Starting presynapse analysis")
    
    output_dir = os.path.join(config.clustering_output_dir, "presynapse_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading and preparing data")
    data_loader = SynapseDataLoader(
        raw_base_dir=config.raw_base_dir,
        seg_base_dir=config.seg_base_dir,
        add_mask_base_dir=config.add_mask_base_dir
    )
    
    seg_vol_dict = {}
    for bbox_name in config.bbox_name:
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
        if raw_vol is not None:
            seg_vol_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
    
    if isinstance(config.segmentation_type, list):
        segmentation_types = config.segmentation_type
    else:
        segmentation_types = [config.segmentation_type]

    if isinstance(config.alpha, list):
        alpha_values = config.alpha
    else:
        alpha_values = [config.alpha]
    
    print(f"Running presynapse analysis for segmentation types: {segmentation_types} and alpha values: {alpha_values}")
    
    for seg_type in segmentation_types:
        for alpha in alpha_values:
            print(f"\n{'='*80}\nAnalyzing presynapse relationships for segmentation type {seg_type} with alpha {alpha}\n{'='*80}")
            
            config.segmentation_type = seg_type
            config.alpha = alpha
            
            seg_output_dir = os.path.join(output_dir, f"seg{seg_type}_alpha{str(alpha).replace('.', '_')}")
            os.makedirs(seg_output_dir, exist_ok=True)
            
            clustered_path = os.path.join(config.csv_output_dir, "combined_analysis", "clustered_features.csv")
            features_df = None
            
            if os.path.exists(clustered_path):
                print(f"Loading clustered data from {clustered_path}")
                clustered_df = load_feature_data(clustered_path)
                
                if 'segmentation_type' in clustered_df.columns:
                    features_df = clustered_df[clustered_df['segmentation_type'] == seg_type]
                    if len(features_df) > 0:
                        print(f"Extracted {len(features_df)} rows for segmentation type {seg_type} from clustered data")
                    else:
                        features_df = None
                        print(f"No data for segmentation type {seg_type} found in clustered data")
                else:
                    print("No segmentation_type column found in clustered data")
            
            if features_df is None:
                features_path = os.path.join(
                    config.csv_output_dir, 
                    f"features_seg{seg_type}_alpha{str(alpha).replace('.', '_')}.csv"
                )
                
                features_df = load_feature_data(features_path)
            
            if features_df is None:
                print(f"No feature data found for segmentation type {seg_type} with alpha {alpha}. Skipping.")
                continue
            
            if 'cluster' not in features_df.columns:
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    print("No cluster information found in data. Performing clustering now...")
                    
                    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
                    if len(feature_cols) > 0:
                        features = features_df[feature_cols].values
                        features_scaled = StandardScaler().fit_transform(features)
                        
                        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(features_scaled)
                        
                        features_df['cluster'] = clusters
                        print(f"Added cluster assignments with {len(np.unique(clusters))} clusters")
                    else:
                        print("No feature columns found for clustering")
                except Exception as e:
                    print(f"Error performing clustering: {e}")
            
            presynapse_groups, updated_features_df = identify_synapses_with_same_presynapse(
                seg_vol_dict, features_df
            )
            
            if not presynapse_groups:
                print(f"No synapses with the same presynapse ID found for segmentation type {seg_type} with alpha {alpha}. Skipping.")
                continue
            
            distance_matrices = calculate_feature_distances(updated_features_df, presynapse_groups)
            
            cluster_info = analyze_cluster_membership(updated_features_df, presynapse_groups)
            
            create_distance_heatmaps(seg_output_dir, distance_matrices, updated_features_df)
            create_cluster_visualizations(seg_output_dir, cluster_info, updated_features_df)
            
            report_path = os.path.join(seg_output_dir, f"presynapse_analysis_report_seg{seg_type}_alpha{str(alpha).replace('.', '_')}.html")
            generate_report(
                seg_output_dir, 
                presynapse_groups, 
                distance_matrices, 
                cluster_info, 
                updated_features_df
            )
            
            updated_features_path = os.path.join(seg_output_dir, f"updated_features_seg{seg_type}_alpha{str(alpha).replace('.', '_')}.csv")
            updated_features_df.to_csv(updated_features_path, index=False)
            
            print("\nPerforming distance comparison analysis between same-presynapse and different-presynapse synapses...")
            distance_comparison = compare_intra_inter_presynapse_distances(
                updated_features_df, presynapse_groups, seg_output_dir
            )
            
            if distance_comparison:
                add_distance_comparison_to_report(report_path, distance_comparison, seg_output_dir)
            
            print(f"Analysis complete for segmentation type {seg_type} with alpha {alpha}")
    
    print(f"Presynapse analysis complete for all segmentation types and alpha values. Results saved to {output_dir}")


if __name__ == "__main__":
    config.parse_args()
    
    run_presynapse_analysis(config) 