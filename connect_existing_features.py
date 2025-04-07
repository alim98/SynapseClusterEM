"""
Connect Clusters from Existing Features

This script loads existing VGG features and manual cluster annotations,
generates UMAP if needed, and connects points from the same manual cluster.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
import time
import datetime
import umap
from sklearn.preprocessing import StandardScaler

def load_features(features_path):
    """
    Load features from the specified CSV file
    
    Args:
        features_path: Path to the features CSV file
        
    Returns:
        pd.DataFrame: DataFrame with features
    """
    print(f"Loading features from: {features_path}")
    
    if not os.path.exists(features_path):
        print(f"Features file not found: {features_path}")
        return None
        
    features_df = pd.read_csv(features_path)
    print(f"Loaded features with shape: {features_df.shape}")
    
    # Check if the dataframe contains the required columns
    if 'bbox_name' not in features_df.columns:
        print("Features DataFrame is missing 'bbox_name' column")
        return None
        
    # Check if Var1 exists (synapse names)
    if 'Var1' not in features_df.columns:
        # Check for alternative column names that might contain synapse names
        alt_columns = ['synapse_name', 'name', 'syn_name', 'id']
        found = False
        for col in alt_columns:
            if col in features_df.columns:
                print(f"Using '{col}' as synapse name column")
                features_df['Var1'] = features_df[col]
                found = True
                break
        
        if not found:
            print("Features DataFrame is missing synapse name column")
            return None
    
    return features_df

def load_manual_clusters():
    """
    Load the manual cluster annotations
    
    Returns:
        pd.DataFrame: DataFrame with manual cluster annotations
    """
    print("Loading manual cluster annotations...")
    
    # Try multiple potential locations for manual clustering file
    potential_paths = [
        "manual_clustered_samples.csv",
        "manual/manual_clustered_samples.csv",
        "manual/clustering_results/manual_clustered_samples.csv",
        "clustered_samples.csv"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            manual_df = pd.read_csv(path)
            print(f"Loaded manual clusters from {path}")
            return manual_df
    
    print("Manual cluster annotations not found")
    return None

def generate_umap(features_df):
    """
    Generate UMAP coordinates if not already present
    
    Args:
        features_df: DataFrame with features
        
    Returns:
        pd.DataFrame: DataFrame with UMAP coordinates added
    """
    # If UMAP coordinates already exist, just return the dataframe
    if 'umap_x' in features_df.columns and 'umap_y' in features_df.columns:
        print("DataFrame already contains UMAP coordinates")
        return features_df
    
    # Check for alternative UMAP column names
    alt_umap_columns = [
        ('umap_1', 'umap_2'),
        ('umap1', 'umap2'),
        ('UMAP_1', 'UMAP_2'),
        ('UMAP1', 'UMAP2'),
        ('umap-1', 'umap-2')
    ]
    
    for x_col, y_col in alt_umap_columns:
        if x_col in features_df.columns and y_col in features_df.columns:
            print(f"Using existing UMAP coordinates: {x_col}, {y_col}")
            features_df['umap_x'] = features_df[x_col]
            features_df['umap_y'] = features_df[y_col]
            return features_df
    
    print("Generating UMAP projection...")
    
    # Identify feature columns (assuming they contain 'feat_' in their name)
    feature_cols = [col for col in features_df.columns if 'feat_' in col]
    
    if not feature_cols:
        # Try to identify feature columns by looking for numerical data
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude certain columns that are likely not features
        exclude_cols = ['cluster', 'x', 'y', 'z', 'Var1', 'id', 'index', 'Manual_Cluster']
        feature_cols = [col for col in numeric_cols if not any(exclude in col for exclude in exclude_cols)]
    
    if not feature_cols:
        print("Could not identify feature columns for UMAP projection")
        return None
    
    print(f"Using {len(feature_cols)} features for UMAP projection")
    
    # Extract features and standardize
    features = features_df[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(features_scaled)
    
    # Add UMAP coordinates to DataFrame
    features_df['umap_x'] = umap_result[:, 0]
    features_df['umap_y'] = umap_result[:, 1]
    
    return features_df

def merge_with_manual_clusters(features_df, manual_df):
    """
    Merge features with manual cluster annotations
    
    Args:
        features_df: DataFrame with feature and UMAP coordinates
        manual_df: DataFrame with manual cluster annotations
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    print("Merging features with manual clusters...")
    
    # Create a copy of the features DataFrame
    merged_df = features_df.copy()
    
    # Check if both DataFrames have the necessary columns
    if 'bbox_name' not in features_df.columns or 'Var1' not in features_df.columns:
        print("Features DataFrame is missing required columns (bbox_name, Var1)")
        return None
        
    if 'bbox_name' not in manual_df.columns or 'Var1' not in manual_df.columns or 'Manual_Cluster' not in manual_df.columns:
        print("Manual clusters DataFrame is missing required columns (bbox_name, Var1, Manual_Cluster)")
        return None
    
    # Create a merge key using bbox_name and Var1
    merged_df['merge_key'] = merged_df['bbox_name'] + ':' + merged_df['Var1'].astype(str)
    manual_df['merge_key'] = manual_df['bbox_name'] + ':' + manual_df['Var1'].astype(str)
    
    # Create a mapping from merge_key to Manual_Cluster
    manual_cluster_map = manual_df.set_index('merge_key')['Manual_Cluster'].to_dict()
    
    # Apply the mapping to the features DataFrame
    merged_df['Manual_Cluster'] = merged_df['merge_key'].map(manual_cluster_map)
    
    # Remove the merge_key column
    merged_df.drop(columns=['merge_key'], inplace=True)
    
    # Count the number of samples with manual clusters
    manual_count = merged_df['Manual_Cluster'].notna().sum()
    print(f"Found {manual_count} samples with manual cluster annotations")
    
    return merged_df

def visualize_cluster_connections(df, output_dir):
    """
    Visualize only points from clusters 1 and 2 with large colored markers
    
    Args:
        df: DataFrame with UMAP coordinates and manual cluster annotations
        output_dir: Directory to save visualizations
    """
    print("Visualizing clusters 1 and 2...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to only include samples with manual cluster annotations
    df_with_clusters = df.dropna(subset=['Manual_Cluster']).copy()
    df_with_clusters['Manual_Cluster'] = df_with_clusters['Manual_Cluster'].astype(int)
    
    # Filter to only include clusters 1 and 2
    df_with_clusters = df_with_clusters[df_with_clusters['Manual_Cluster'].isin([1, 2])]
    
    if len(df_with_clusters) == 0:
        print("No samples with manual cluster annotations in clusters 1 or 2")
        return
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Plot all points in gray
    plt.scatter(
        df['umap_x'],
        df['umap_y'],
        color='lightgray',
        s=20,
        alpha=0.3
    )
    
    # Define specific colors for clusters 1 and 2
    cluster_colors = {
        1: 'blue',
        2: 'red'
    }
    
    # Plot the points for each cluster
    for cluster, group in df_with_clusters.groupby('Manual_Cluster'):
        plt.scatter(
            group['umap_x'],
            group['umap_y'],
            color=cluster_colors[cluster],
            edgecolor='black',
            s=200,  # Larger points
            alpha=0.8,
            label=f"Cluster {cluster}"
        )
        
        # Add labels
        for i, row in group.iterrows():
            if 'Var1' in row:
                # Get a short version of the synapse name
                if isinstance(row['Var1'], str) and "synapse" in row['Var1']:
                    parts = row['Var1'].split('_')
                    if len(parts) > 1:
                        label = parts[-1]  # Use the last part (often a number)
                    else:
                        label = row['Var1'][-5:]  # Use last 5 chars if no underscore
                else:
                    label = str(row['Var1'])[-5:]  # Use last 5 chars
                
                # Add text with black outline for better visibility
                txt = plt.text(
                    row['umap_x'], row['umap_y'], 
                    label,
                    fontsize=8, 
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
    plt.title("Clusters 1 and 2 in UMAP Space", fontsize=16)
    
    # Create a custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[cluster],
              markeredgecolor='black', markersize=12, label=f'Cluster {cluster}')
        for cluster in cluster_colors.keys()
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=6)
    
    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clusters_1_and_2.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "clusters_1_and_2.pdf"))
    
    print(f"Saved visualizations to {output_dir}")

def create_interactive_visualization(df, output_dir):
    """
    Create an interactive HTML visualization using Plotly
    
    Args:
        df: DataFrame with UMAP coordinates and manual cluster annotations
        output_dir: Directory to save visualizations
    """
    print("Creating interactive visualization...")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Skipping interactive visualization.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to only include samples with manual cluster annotations
    df_with_clusters = df.dropna(subset=['Manual_Cluster']).copy()
    df_with_clusters['Manual_Cluster'] = df_with_clusters['Manual_Cluster'].astype(int)
    
    # Filter to only include clusters 1 and 2
    df_with_clusters = df_with_clusters[df_with_clusters['Manual_Cluster'].isin([1, 2])]
    df_with_clusters['Manual_Cluster_str'] = 'Cluster ' + df_with_clusters['Manual_Cluster'].astype(str)
    
    if len(df_with_clusters) == 0:
        print("No samples with manual cluster annotations in clusters 1 or 2")
        return
    
    # Create a color map for clusters 1 and 2
    color_map = {
        'Cluster 1': 'blue',
        'Cluster 2': 'red'
    }
    
    # Create scatter plot of all points
    fig = px.scatter(
        df, 
        x='umap_x', 
        y='umap_y',
        hover_data=['bbox_name', 'Var1'],
        opacity=0.3,
        color_discrete_sequence=['lightgray'],
        title='Clusters 1 and 2 in UMAP Space'
    )
    
    # Add points with manual clusters
    for cluster, group in df_with_clusters.groupby('Manual_Cluster_str'):
        fig.add_trace(
            go.Scatter(
                x=group['umap_x'],
                y=group['umap_y'],
                mode='markers',
                marker=dict(
                    color=color_map[cluster],
                    size=15,
                    line=dict(width=1, color='black')
                ),
                name=cluster,
                text=group['Var1'],
                hoverinfo='text+name',
                hovertext=[f"Bbox: {row['bbox_name']}<br>Synapse: {row['Var1']}" 
                          for _, row in group.iterrows()]
            )
        )
    
    # Improve layout
    fig.update_layout(
        width=1000,
        height=800,
        title_font_size=20,
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
    )
    
    # Save as HTML
    fig.write_html(os.path.join(output_dir, "clusters_1_and_2_interactive.html"))
    print(f"Saved interactive visualization to {output_dir}")

def main():
    # Set the features file path
    features_path = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\10\features_layer20_seg10_alpha1.0\features_layer20_seg10_alpha1_0.csv"
    
    # Create timestamp for output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"connected_clusters_{timestamp}")
    
    # Step 1: Load the existing features
    features_df = load_features(features_path)
    if features_df is None:
        print("Failed to load features. Exiting.")
        sys.exit(1)
    
    # Step 2: Generate UMAP if not already present
    features_df = generate_umap(features_df)
    if features_df is None:
        print("Failed to generate UMAP projection. Exiting.")
        sys.exit(1)
    
    # Step 3: Load manual cluster annotations
    manual_df = load_manual_clusters()
    if manual_df is None:
        print("Failed to load manual cluster annotations. Exiting.")
        sys.exit(1)
    
    # Step 4: Merge UMAP with manual clusters
    merged_df = merge_with_manual_clusters(features_df, manual_df)
    if merged_df is None:
        print("Failed to merge features with manual clusters. Exiting.")
        sys.exit(1)
    
    # Step 5: Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 6: Save merged data
    merged_csv = os.path.join(output_dir, "merged_features_with_manual_clusters.csv")
    merged_df.to_csv(merged_csv, index=False)
    print(f"Saved merged data to {merged_csv}")
    
    # Step 7: Visualize cluster connections
    visualize_cluster_connections(merged_df, output_dir)
    
    # Step 8: Create interactive visualization
    create_interactive_visualization(merged_df, output_dir)
    
    print(f"All done! Results saved to {output_dir}")
    
if __name__ == "__main__":
    main() 