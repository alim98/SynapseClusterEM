import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects

def load_data():
    """
    Load the manual annotations and VGG features
    """
    print("Loading data...")
    
    # Set up paths
    results_dir = 'manual/clustering_results'
    
    # Load manual clustering results
    manual_file = os.path.join(results_dir, "manual_clustered_samples.csv")
    if not os.path.exists(manual_file):
        raise FileNotFoundError(f"Manual clustering file not found: {manual_file}")
    manual_df = pd.read_csv(manual_file)
    print(f"Loaded manual clustering with {len(manual_df)} samples")
    
    # Load VGG features
    vgg_file = os.path.join(results_dir, "vgg_stage_specific_features.csv")
    if not os.path.exists(vgg_file):
        # Try alternative filenames if the stage-specific file isn't found
        vgg_file = os.path.join(results_dir, "vgg_features.csv")
        if not os.path.exists(vgg_file):
            vgg_file = os.path.join(results_dir, "vgg_clustered_features.csv")
            if not os.path.exists(vgg_file):
                raise FileNotFoundError(f"VGG features file not found in {results_dir}")
    
    vgg_df = pd.read_csv(vgg_file)
    print(f"Loaded VGG features with {len(vgg_df)} samples")
    
    # Merge DataFrames to get only manually annotated samples
    merged_df = manual_df[['bbox_name', 'Var1', 'Manual_Cluster']].merge(
        vgg_df, on=['bbox_name', 'Var1']
    )
    
    print(f"Merged data contains {len(merged_df)} manually annotated samples")
    return merged_df

def apply_dimension_reduction(df, method='umap'):
    """
    Apply dimensionality reduction to feature data
    
    Args:
        df: DataFrame with features
        method: 'umap' or 'pca' for dimensionality reduction method
        
    Returns:
        DataFrame with projected coordinates added
    """
    print(f"Applying {method.upper()} dimensionality reduction...")
    
    # Get feature columns
    feature_cols = [col for col in df.columns if 'feat_' in col]
    print(f"Using {len(feature_cols)} features for {method} projection")
    
    # Standardize features
    features = df[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply dimensionality reduction
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        projection = reducer.fit_transform(features_scaled)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
        projection = reducer.fit_transform(features_scaled)
    
    # Add projection coordinates to DataFrame
    df[f'{method}_x'] = projection[:, 0]
    df[f'{method}_y'] = projection[:, 1]
    
    return df

def visualize_cluster_connections(df, method='umap', output_dir='manual/connection_visualizations'):
    """
    Visualize cluster connections in 2D projection
    
    Args:
        df: DataFrame with projected coordinates and cluster information
        method: 'umap' or 'pca' for dimensionality reduction method
        output_dir: Directory to save visualizations
    """
    print(f"Visualizing cluster connections using {method.upper()}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get coordinate columns
    x_col = f'{method}_x'
    y_col = f'{method}_y'
    
    # Get unique manual clusters and assign colors
    clusters = sorted(df['Manual_Cluster'].unique())
    cluster_colors = {}
    for i, cluster in enumerate(clusters):
        cluster_colors[cluster] = plt.cm.tab10(i % 10)
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # First plot the connections (lines) between points in the same cluster
    for cluster in clusters:
        # Get points in this cluster
        cluster_points = df[df['Manual_Cluster'] == cluster]
        
        # Skip if only one point
        if len(cluster_points) <= 1:
            continue
            
        # Plot lines connecting all points in this cluster
        for i, row1 in cluster_points.iterrows():
            for j, row2 in cluster_points.iterrows():
                if i < j:  # Only draw each connection once
                    plt.plot(
                        [row1[x_col], row2[x_col]],
                        [row1[y_col], row2[y_col]],
                        color=cluster_colors[cluster],
                        alpha=0.3,
                        linestyle='-',
                        linewidth=1.0
                    )
    
    # Then plot the points over the lines
    for cluster, group in df.groupby('Manual_Cluster'):
        plt.scatter(
            group[x_col],
            group[y_col],
            color=cluster_colors[cluster],
            edgecolor='black',
            s=150,
            alpha=0.8,
            label=f"Cluster {cluster}"
        )
        
        # Add synapse labels
        for i, row in group.iterrows():
            # Get a short version of the synapse name
            if "synapse" in row['Var1']:
                parts = row['Var1'].split('_')
                if len(parts) > 1:
                    label = parts[-1]  # Use the last part (often a number)
                else:
                    label = row['Var1'][-5:]  # Use last 5 chars if no underscore
            else:
                label = row['Var1'][-5:]  # Use last 5 chars
            
            # Add text with black outline for better visibility
            txt = plt.text(
                row[x_col], row[y_col], 
                label,
                fontsize=9, 
                ha='center', 
                va='center',
                fontweight='bold',
                color='white'
            )
            
            # Add outline to text
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=2, foreground='black')
            ])
    
    # Add title and legend
    plt.title(f"Manual Cluster Connections using {method.upper()} Projection", fontsize=16)
    
    # Create a custom legend with larger markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[cluster],
              markeredgecolor='black', markersize=12, label=f'Cluster {cluster}')
        for cluster in clusters
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=12)
    
    # Add grid and set axis limits with some padding
    plt.grid(True, linestyle='--', alpha=0.5)
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    x_padding = 0.1 * (x_max - x_min)
    y_padding = 0.1 * (y_max - y_min)
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"manual_cluster_connections_{method}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"manual_cluster_connections_{method}.pdf"))
    
    # Create a more detailed version with bbox information
    plt.figure(figsize=(14, 12))
    
    # Use different marker styles for different bboxes
    bbox_markers = {
        'bbox1': 'o',  # circle
        'bbox2': 's',  # square
        'bbox3': '^',  # triangle up
        'bbox4': 'v',  # triangle down
        'bbox5': 'D',  # diamond
        'bbox6': 'p',  # pentagon
        'bbox7': 'h',  # hexagon
    }
    
    # First plot the connections
    for cluster in clusters:
        cluster_points = df[df['Manual_Cluster'] == cluster]
        if len(cluster_points) <= 1:
            continue
            
        for i, row1 in cluster_points.iterrows():
            for j, row2 in cluster_points.iterrows():
                if i < j:
                    plt.plot(
                        [row1[x_col], row2[x_col]],
                        [row1[y_col], row2[y_col]],
                        color=cluster_colors[cluster],
                        alpha=0.3,
                        linestyle='-',
                        linewidth=1.0
                    )
    
    # Then plot points with markers by bbox and colors by cluster
    for (cluster, bbox), group in df.groupby(['Manual_Cluster', 'bbox_name']):
        marker = bbox_markers.get(bbox, 'o')
        plt.scatter(
            group[x_col],
            group[y_col],
            color=cluster_colors[cluster],
            marker=marker,
            edgecolor='black',
            s=150,
            alpha=0.8,
            label=f"Cluster {cluster} - {bbox}"
        )
        
        # Add labels
        for i, row in group.iterrows():
            if "synapse" in row['Var1']:
                parts = row['Var1'].split('_')
                if len(parts) > 1:
                    label = parts[-1]
                else:
                    label = row['Var1'][-5:]
            else:
                label = row['Var1'][-5:]
            
            txt = plt.text(
                row[x_col], row[y_col], 
                label,
                fontsize=9, 
                ha='center', 
                va='center',
                fontweight='bold',
                color='white'
            )
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=2, foreground='black')
            ])
    
    # Create a custom legend
    legend_elements = []
    
    # Add cluster color legend
    for cluster in clusters:
        legend_elements.append(
            Patch(facecolor=cluster_colors[cluster], edgecolor='black', label=f'Cluster {cluster}')
        )
    
    # Add bbox marker legend
    for bbox, marker in bbox_markers.items():
        if bbox in df['bbox_name'].values:
            legend_elements.append(
                Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                      markeredgecolor='black', markersize=10, label=f'{bbox}')
            )
    
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    plt.title(f"Manual Cluster Connections with BBox Information\n({method.upper()} Projection)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"manual_cluster_connections_with_bbox_{method}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"manual_cluster_connections_with_bbox_{method}.pdf"))
    
    print(f"Visualizations saved to {output_dir}")

def main():
    # Set up
    output_dir = 'manual/connection_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Apply dimension reduction
    df = apply_dimension_reduction(df, method='umap')
    df = apply_dimension_reduction(df, method='pca')
    
    # Save processed data
    df.to_csv(os.path.join(output_dir, 'manual_data_with_projections.csv'), index=False)
    
    # Create visualizations
    visualize_cluster_connections(df, method='umap', output_dir=output_dir)
    visualize_cluster_connections(df, method='pca', output_dir=output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 