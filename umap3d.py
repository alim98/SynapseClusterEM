import os
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans
import umap

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from tqdm import tqdm  # For progress bars

def compute_umap(features_scaled, n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """Compute UMAP projection with progress feedback"""
    print(f"Computing {n_components}D UMAP projection (this may take a while)...")
    start_time = time.time()
    
    # Create and fit UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=True  # Enable verbose output
    )
    
    umap_results = reducer.fit_transform(features_scaled)
    
    elapsed_time = time.time() - start_time
    print(f"UMAP computation completed in {elapsed_time:.2f} seconds")
    
    return umap_results, reducer

def create_bbox_colored_umap(features_df, output_dir, reuse_umap_results=None):
    """Create a 3D UMAP visualization specifically colored by bounding box"""
    if 'bbox_name' not in features_df.columns:
        print("No bbox_name column in features data, skipping bbox-colored UMAP")
        return
    
    # Define a consistent color map for bounding boxes
    bbox_colors = {
        'bbox1': '#FF0000', 'bbox2': '#00FFFF', 'bbox3': '#FFA500',
        'bbox4': '#800080', 'bbox5': '#808080', 'bbox6': '#0000FF', 'bbox7': '#000000'
    }
    
    # Debug: Print column names to understand their structure
    print("Available columns:", features_df.columns.tolist())
    
    # Simple feature column detection - check if 'feat_' appears anywhere in the column name
    feature_cols = [col for col in features_df.columns if 'feat_' in col]
    print(f"Found {len(feature_cols)} feature columns")
    
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame")
    
    print(f"Using {len(feature_cols)} feature columns for UMAP")
    print("First few feature columns:", feature_cols[:5])
    
    # Extract features
    features = features_df[feature_cols].values
    
    # Scale features
    print("Scaling features...")
    features_scaled = StandardScaler().fit_transform(features)
    
    # Use provided UMAP results or compute new ones
    if reuse_umap_results is not None:
        print("Using pre-computed UMAP projection")
        umap_results = reuse_umap_results
    else:
        # Compute 3D UMAP
        umap_results, _ = compute_umap(features_scaled, n_components=3)
    
    # Add UMAP coordinates to dataframe
    features_df = features_df.copy()
    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]
    features_df['umap_z'] = umap_results[:, 2]
    
    print("Creating 3D visualization...")
    
    # Create 3D plotly figure using px.scatter_3d
    fig = px.scatter_3d(
        features_df,
        x='umap_x',
        y='umap_y',
        z='umap_z',
        color='bbox_name',
        color_discrete_map=bbox_colors,
        hover_data=['Var1'],  # Display synapse ID in hover
        title='3D UMAP Visualization Colored by Bounding Box',
        opacity=0.8
    )
    
    # Update marker size and appearance
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='black')))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        ),
        legend=dict(
            title="Bounding Box",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode='closest'
    )
    
    # Save as interactive HTML with error handling
    print("Saving visualization...")
    output_path_html = os.path.join(output_dir, "umap3d_bbox_colored.html")
    try:
        print(f"Attempting to save HTML to {output_path_html}")
        fig.write_html(output_path_html)
        print(f"Successfully saved HTML")
    except Exception as e:
        print(f"Error saving HTML: {str(e)}")
        print("Trying alternative save method...")
        
        try:
            # Alternative: Save as a JSON file
            import json
            fig_json = fig.to_json()
            json_path = os.path.join(output_dir, "umap3d_bbox_colored.json")
            with open(json_path, 'w') as f:
                f.write(fig_json)
            print(f"Saved figure as JSON to {json_path}")
        except Exception as e2:
            print(f"Error saving JSON: {str(e2)}")
    
    # Also save a static image for reference with error handling
    output_path_png = os.path.join(output_dir, "umap3d_bbox_colored.png")
    try:
        print(f"Attempting to save PNG to {output_path_png}")
        # fig.write_image(output_path_png, width=1200, height=1000)
        print(f"Successfully saved PNG")
    except Exception as e:
        print(f"Error saving PNG: {str(e)}")
        print("Trying alternative save method...")
        
        try:
            # Alternative: Save as SVG which often has fewer dependencies
            svg_path = os.path.join(output_dir, "umap3d_bbox_colored.svg")
            fig.write_image(svg_path, format="svg")
            print(f"Saved figure as SVG to {svg_path}")
        except Exception as e2:
            print(f"Error saving SVG: {str(e2)}")
    
    print(f"3D Bounding box colored UMAP visualization processing completed")
    
    # Save UMAP coordinates as CSV for easy reuse
    umap_coords_df = features_df[['bbox_name', 'Var1', 'umap_x', 'umap_y', 'umap_z']]
    umap_coords_path = os.path.join(output_dir, "umap3d_bbox_coordinates.csv")
    umap_coords_df.to_csv(umap_coords_path, index=False)
    print(f"Saved UMAP coordinates to {umap_coords_path}")
    
    return features_df, umap_results


def create_cluster_colored_umap(features_df, output_dir, reuse_umap_results=None):
    """Create a 3D UMAP visualization specifically colored by cluster"""
    if 'cluster' not in features_df.columns:
        print("No cluster column in features data, skipping cluster-colored UMAP")
        return
    
    # Debug: Print column names to understand their structure
    print("Available columns:", features_df.columns.tolist())
    
    # Simple feature column detection - check if 'feat_' appears anywhere in the column name
    feature_cols = [col for col in features_df.columns if 'feat_' in col]
    print(f"Found {len(feature_cols)} feature columns")
    
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame")
    
    print(f"Using {len(feature_cols)} feature columns for UMAP")
    print("First few feature columns:", feature_cols[:5])
    
    # Extract features
    features = features_df[feature_cols].values
    
    # Scale features
    print("Scaling features...")
    features_scaled = StandardScaler().fit_transform(features)
    
    # Use provided UMAP results or compute new ones
    if reuse_umap_results is not None and 'umap_x' in features_df.columns:
        print("Using pre-computed UMAP projection")
        # If the dataframe already has UMAP coordinates, we'll use those
        umap_results = features_df[['umap_x', 'umap_y', 'umap_z']].values
    elif reuse_umap_results is not None:
        print("Using provided UMAP projection")
        umap_results = reuse_umap_results
    else:
        # Compute 3D UMAP
        umap_results, _ = compute_umap(features_scaled, n_components=3)
    
    # Add UMAP coordinates to dataframe if not already present
    features_df = features_df.copy()  # Create a copy to avoid modifying the original
    if 'umap_x' not in features_df.columns:
        features_df['umap_x'] = umap_results[:, 0]
        features_df['umap_y'] = umap_results[:, 1]
        features_df['umap_z'] = umap_results[:, 2]
    
    features_df['cluster_str'] = 'Cluster ' + features_df['cluster'].astype(str)
    
    # Get all unique clusters
    unique_clusters = features_df['cluster'].unique()
    
    print("Creating 3D visualization...")
    # Create 3D plotly figure
    fig = px.scatter_3d(
        features_df,
        x='umap_x',
        y='umap_y',
        z='umap_z',
        color='cluster_str',
        hover_data=['bbox_name'],
        title='3D UMAP Visualization Colored by Cluster',
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.8
    )
    
    # Update marker size
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='black')))
    
    # Add cluster center annotations
    for cluster_id in unique_clusters:
        cluster_points = features_df[features_df['cluster'] == cluster_id]
        
        centroid_x = cluster_points['umap_x'].mean()
        centroid_y = cluster_points['umap_y'].mean()
        centroid_z = cluster_points['umap_z'].mean()
        
        # Add a marker for the centroid
        fig.add_trace(go.Scatter3d(
            x=[centroid_x],
            y=[centroid_y],
            z=[centroid_z],
            mode='markers+text',
            marker=dict(
                size=10,
                color='black',
                symbol='diamond'
            ),
            text=[f'Cluster {cluster_id}'],
            textposition='top center',
            name=f'Centroid {cluster_id}',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        ),
        legend=dict(
            title="Clusters",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode='closest'
    )
    
    # Save as interactive HTML with error handling
    print("Saving visualization...")
    output_path_html = os.path.join(output_dir, "umap3d_cluster_colored.html")
    try:
        print(f"Attempting to save HTML to {output_path_html}")
        fig.write_html(output_path_html)
        print(f"Successfully saved HTML")
    except Exception as e:
        print(f"Error saving HTML: {str(e)}")
        print("Trying alternative save method...")
        
        try:
            # Alternative: Save as a JSON file
            import json
            fig_json = fig.to_json()
            json_path = os.path.join(output_dir, "umap3d_cluster_colored.json")
            with open(json_path, 'w') as f:
                f.write(fig_json)
            print(f"Saved figure as JSON to {json_path}")
        except Exception as e2:
            print(f"Error saving JSON: {str(e2)}")
    
    # Also save a static image for reference with error handling
    output_path_png = os.path.join(output_dir, "umap3d_cluster_colored.png")
    try:
        print(f"Attempting to save PNG to {output_path_png}")
        # fig.write_image(output_path_png, width=1200, height=1000)
        print(f"Successfully saved PNG")
    except Exception as e:
        print(f"Error saving PNG: {str(e)}")
        print("Trying alternative save method...")
        
        try:
            # Alternative: Save as SVG which often has fewer dependencies
            svg_path = os.path.join(output_dir, "umap3d_cluster_colored.svg")
            # fig.write_image(svg_path, format="svg")
            print(f"Saved figure as SVG to {svg_path}")
        except Exception as e2:
            print(f"Error saving SVG: {str(e2)}")
    
    print(f"3D Cluster colored UMAP visualization processing completed")
    
    # Save cluster assignment with UMAP coordinates
    cluster_coords_df = features_df[['bbox_name', 'Var1', 'cluster', 'umap_x', 'umap_y', 'umap_z']]
    cluster_coords_path = os.path.join(output_dir, "cluster_umap_coordinates.csv")
    cluster_coords_df.to_csv(cluster_coords_path, index=False)
    print(f"Saved cluster assignments with UMAP coordinates to {cluster_coords_path}")
    
    return features_df


def cluster_features(features_df, n_clusters=10, reuse_umap_results=None):
    """Perform clustering on feature data"""
    print(f"Clustering features into {n_clusters} clusters...")
    
    # Debug: Print column names to understand their structure
    print("Available columns:", features_df.columns.tolist())
    
    # Simple feature column detection - check if 'feat_' appears anywhere in the column name
    feature_cols = [col for col in features_df.columns if 'feat_' in col]
    print(f"Found {len(feature_cols)} feature columns")
    
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame")
    
    print(f"Using {len(feature_cols)} feature columns for clustering")
    print("First few feature columns:", feature_cols[:5])
    
    # Extract features
    features = features_df[feature_cols].values
    
    # Scale features
    print("Scaling features...")
    features_scaled = StandardScaler().fit_transform(features)
    
    # Apply KMeans clustering
    print(f"Performing KMeans clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Add cluster assignments to dataframe
    features_df = features_df.copy()
    features_df['cluster'] = clusters
    
    # Print cluster distribution
    cluster_counts = features_df['cluster'].value_counts().sort_index()
    print("Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples")
    
    # Add UMAP coordinates to dataframe if available
    if reuse_umap_results is not None and 'umap_x' not in features_df.columns:
        features_df['umap_x'] = reuse_umap_results[:, 0]
        features_df['umap_y'] = reuse_umap_results[:, 1]
        features_df['umap_z'] = reuse_umap_results[:, 2]
    
    return features_df


print("Starting UMAP visualization and clustering pipeline...")
print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load feature data
print("Loading feature data...")
df = pd.read_csv(r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\11_concat_avg_max\features_layer20_concat_avg_max_seg11_alpha1.0\features_layer20_concat_avg_max_seg11_alpha1_0.csv")
print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

# Create output directory
output_dir = r"C:\Users\alim9\Documents\codes\synapse2\results\extracted\stable\11_concat_avg_max"
os.makedirs(output_dir, exist_ok=True)

# Create the bounding box colored UMAP
print("\n=== Creating bounding box colored UMAP ===")
df_with_umap, umap_results = create_bbox_colored_umap(df, output_dir)

# Cluster the data
print("\n=== Clustering data ===")
df_clustered = cluster_features(df_with_umap, n_clusters=10, reuse_umap_results=umap_results)

# Save the clustered data
print("\n=== Saving clustered data ===")
clustered_data_path = os.path.join(output_dir, "clustered_features.csv")
df_clustered.to_csv(clustered_data_path, index=False)
print(f"Saved clustered data to {clustered_data_path}")

# Create the cluster colored UMAP
print("\n=== Creating cluster colored UMAP ===")
create_cluster_colored_umap(df_clustered, output_dir, reuse_umap_results=umap_results)

print("\nAll processing completed successfully!")
print(f"Final time: {time.strftime('%Y-%m-%d %H:%M:%S')}")