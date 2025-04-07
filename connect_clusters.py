"""
Connect Clusters Script

This script runs the synapse pipeline to generate UMAP representations and then
connects points from the same clusters based on manual cluster annotations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import subprocess
import glob
from pathlib import Path
import time
import datetime

def run_synapse_pipeline():
    """
    Run the main synapse pipeline using subprocess
    
    Returns:
        str: Path to the results directory
    """
    print("Running synapse pipeline...")
    
    # Run the pipeline script as a subprocess
    cmd = ["python", "run_synapse_pipeline.py"]
    
    # Capture the timestamp from the start of the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    expected_results_dir = os.path.join("results", f"run_{timestamp}")
    
    # Run the pipeline script
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream output to console
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    if process.returncode != 0:
        print("Error running synapse pipeline. Check logs for details.")
        for line in process.stderr:
            print(line, end='')
        return None
    
    # Find the most recent results directory
    if not os.path.exists(expected_results_dir):
        results_dirs = sorted(glob.glob(os.path.join("results", "run_*")), 
                              key=os.path.getmtime, reverse=True)
        if results_dirs:
            results_dir = results_dirs[0]
            print(f"Using most recent results directory: {results_dir}")
        else:
            print("No results directory found")
            return None
    else:
        results_dir = expected_results_dir
        
    return results_dir

def find_umap_csv(results_dir):
    """
    Find the CSV file with UMAP coordinates in the results directory
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        str: Path to the CSV file with UMAP coordinates
    """
    print("Finding UMAP results...")
    
    # Search for clustered features CSV files with UMAP coordinates
    csv_pattern = os.path.join(results_dir, "**", "clustered_features.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    if not csv_files:
        print(f"No CSV files found with UMAP coordinates in {results_dir}")
        return None
        
    # Use the most recent file
    csv_file = sorted(csv_files, key=os.path.getmtime, reverse=True)[0]
    print(f"Found UMAP results: {csv_file}")
    
    # Verify the file contains UMAP coordinates
    df = pd.read_csv(csv_file)
    if 'umap_x' not in df.columns or 'umap_y' not in df.columns:
        print(f"CSV file does not contain UMAP coordinates: {csv_file}")
        return None
        
    return csv_file

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
        "manual/clustering_results/manual_clustered_samples.csv"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            manual_df = pd.read_csv(path)
            print(f"Loaded manual clusters from {path}")
            return manual_df
    
    print("Manual cluster annotations not found")
    return None

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
    features_df['merge_key'] = features_df['bbox_name'] + ':' + features_df['Var1'].astype(str)
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
    Visualize connections between points in the same manual cluster
    
    Args:
        df: DataFrame with UMAP coordinates and manual cluster annotations
        output_dir: Directory to save visualizations
    """
    print("Visualizing cluster connections...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to only include samples with manual cluster annotations
    df_with_clusters = df.dropna(subset=['Manual_Cluster']).copy()
    df_with_clusters['Manual_Cluster'] = df_with_clusters['Manual_Cluster'].astype(int)
    
    if len(df_with_clusters) == 0:
        print("No samples with manual cluster annotations")
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
    
    # Get unique manual clusters and assign colors
    clusters = sorted(df_with_clusters['Manual_Cluster'].unique())
    cluster_colors = {}
    for i, cluster in enumerate(clusters):
        cluster_colors[cluster] = plt.cm.tab10(i % 10)
    
    # First plot the connections (lines) between points in the same cluster
    for cluster in clusters:
        # Get points in this cluster
        cluster_points = df_with_clusters[df_with_clusters['Manual_Cluster'] == cluster]
        
        # Skip if only one point
        if len(cluster_points) <= 1:
            continue
            
        # Plot lines connecting all points in this cluster
        for i, row1 in cluster_points.iterrows():
            for j, row2 in cluster_points.iterrows():
                if i < j:  # Only draw each connection once
                    plt.plot(
                        [row1['umap_x'], row2['umap_x']],
                        [row1['umap_y'], row2['umap_y']],
                        color=cluster_colors[cluster],
                        alpha=0.5,
                        linestyle='-',
                        linewidth=1.5
                    )
    
    # Then plot the points over the lines
    for cluster, group in df_with_clusters.groupby('Manual_Cluster'):
        plt.scatter(
            group['umap_x'],
            group['umap_y'],
            color=cluster_colors[cluster],
            edgecolor='black',
            s=100,
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
    plt.title("Manual Cluster Connections in UMAP Space", fontsize=16)
    
    # Create a custom legend with larger markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[cluster],
              markeredgecolor='black', markersize=10, label=f'Cluster {cluster}')
        for cluster in clusters
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "manual_cluster_connections.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "manual_cluster_connections.pdf"))
    
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
    df_with_clusters['Manual_Cluster_str'] = 'Cluster ' + df_with_clusters['Manual_Cluster'].astype(str)
    
    if len(df_with_clusters) == 0:
        print("No samples with manual cluster annotations")
        return
    
    # Create scatter plot of all points
    fig = px.scatter(
        df, 
        x='umap_x', 
        y='umap_y',
        hover_data=['bbox_name', 'Var1'],
        opacity=0.3,
        color_discrete_sequence=['lightgray'],
        title='Manual Cluster Connections in UMAP Space'
    )
    
    # Add points with manual clusters
    cluster_fig = px.scatter(
        df_with_clusters, 
        x='umap_x', 
        y='umap_y',
        color='Manual_Cluster_str',
        hover_data=['bbox_name', 'Var1'],
        opacity=0.8,
        title='Manual Cluster Connections in UMAP Space'
    )
    
    for trace in cluster_fig.data:
        fig.add_trace(trace)
    
    # Add connections between points in the same cluster
    clusters = sorted(df_with_clusters['Manual_Cluster'].unique())
    
    for cluster in clusters:
        cluster_points = df_with_clusters[df_with_clusters['Manual_Cluster'] == cluster]
        
        # Skip if only one point
        if len(cluster_points) <= 1:
            continue
            
        # Add lines connecting all points in this cluster
        for i, row1 in cluster_points.iterrows():
            for j, row2 in cluster_points.iterrows():
                if i < j:  # Only draw each connection once
                    fig.add_trace(
                        go.Scatter(
                            x=[row1['umap_x'], row2['umap_x']],
                            y=[row1['umap_y'], row2['umap_y']],
                            mode='lines',
                            line=dict(color=f'rgba({np.random.randint(0,256)},{np.random.randint(0,256)},{np.random.randint(0,256)},0.3)'),
                            showlegend=False,
                            hoverinfo='none'
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
    fig.write_html(os.path.join(output_dir, "manual_cluster_connections_interactive.html"))
    print(f"Saved interactive visualization to {output_dir}")

def main():
    # Create timestamp for output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"connected_clusters_{timestamp}")
    
    # Step 1: Run the synapse pipeline
    results_dir = run_synapse_pipeline()
    if not results_dir:
        print("Failed to run synapse pipeline. Exiting.")
        sys.exit(1)
    
    # Step 2: Find the UMAP CSV file
    umap_csv = find_umap_csv(results_dir)
    if not umap_csv:
        print("Failed to find UMAP results. Exiting.")
        sys.exit(1)
    
    # Step 3: Load the UMAP results
    features_df = pd.read_csv(umap_csv)
    print(f"Loaded features with shape: {features_df.shape}")
    
    # Step 4: Load manual cluster annotations
    manual_df = load_manual_clusters()
    if manual_df is None:
        print("Failed to load manual cluster annotations. Exiting.")
        sys.exit(1)
    
    # Step 5: Merge UMAP with manual clusters
    merged_df = merge_with_manual_clusters(features_df, manual_df)
    if merged_df is None:
        print("Failed to merge features with manual clusters. Exiting.")
        sys.exit(1)
    
    # Step 6: Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 7: Save merged data
    merged_csv = os.path.join(output_dir, "merged_features_with_manual_clusters.csv")
    merged_df.to_csv(merged_csv, index=False)
    print(f"Saved merged data to {merged_csv}")
    
    # Step 8: Visualize cluster connections
    visualize_cluster_connections(merged_df, output_dir)
    
    # Step 9: Create interactive visualization
    create_interactive_visualization(merged_df, output_dir)
    
    print(f"All done! Results saved to {output_dir}")
    
if __name__ == "__main__":
    main() 