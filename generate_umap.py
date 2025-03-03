import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import logging
import umap
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UMAPGenerator")

def generate_umap_coordinates(input_path, output_path, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Generate UMAP coordinates for the input data and save to the output path.
    
    Args:
        input_path: Path to the input CSV file (clustered data without UMAP coords)
        output_path: Path to save the output CSV file (with UMAP coords)
        n_components: Number of UMAP dimensions to generate
        n_neighbors: Parameter for UMAP algorithm
        min_dist: Parameter for UMAP algorithm
    """
    # Load clustered data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Extract feature columns
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    
    if not feature_cols:
        logger.error("No feature columns found in the data")
        return
    
    logger.info(f"Found {len(feature_cols)} feature columns")
    logger.info(f"Data shape: {df.shape}")
    
    # Extract and scale features
    features = df[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute UMAP embeddings
    logger.info(f"Computing UMAP embeddings with {n_components} components...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    embeddings = reducer.fit_transform(features_scaled)
    
    # Add UMAP coordinates to DataFrame
    logger.info("Adding UMAP coordinates to DataFrame")
    for i in range(n_components):
        df[f'umap_{chr(120+i)}'] = embeddings[:, i]  # umap_x, umap_y, umap_z
    
    # Save the DataFrame with UMAP coordinates
    logger.info(f"Saving data with UMAP coordinates to {output_path}")
    df.to_csv(output_path, index=False)
    
    # Create and save a preview plot
    preview_dir = os.path.dirname(output_path)
    preview_path = os.path.join(preview_dir, 'umap_preview.png')
    
    plt.figure(figsize=(10, 8))
    
    # Color by cluster if available
    if 'cluster' in df.columns:
        for cluster in df['cluster'].unique():
            subset = df[df['cluster'] == cluster]
            plt.scatter(subset['umap_x'], subset['umap_y'], 
                        label=f'Cluster {cluster}', alpha=0.7)
        plt.legend(title='Cluster')
    else:
        plt.scatter(df['umap_x'], df['umap_y'], alpha=0.7)
    
    plt.title('UMAP Preview')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(preview_path, dpi=300)
    plt.close()
    
    logger.info(f"Preview saved to {preview_path}")
    logger.info("UMAP generation complete!")
    
    return df

def main():
    # Setup command line arguments (use hardcoded defaults for simplicity)
    input_path = 'outputs/main_results/clustering/clustered_data.csv'
    output_path = 'outputs/main_results/clustering/clustered_data_with_umap.csv'
    
    # Generate UMAP coordinates
    df_with_umap = generate_umap_coordinates(input_path, output_path)
    
    # Check if visualization directory exists, if not create it
    viz_dir = 'outputs/main_results/visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create basic visualizations 
    if df_with_umap is not None:
        # Create and save cluster visualization
        if 'cluster' in df_with_umap.columns:
            fig_cluster = px.scatter(
                df_with_umap, x='umap_x', y='umap_y', color='cluster',
                title='UMAP 2D Projection by Cluster',
                labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2', 'cluster': 'Cluster'}
            )
            cluster_viz_path = os.path.join(viz_dir, 'umap_by_cluster.html')
            fig_cluster.write_html(cluster_viz_path)
            logger.info(f"Cluster visualization saved to {cluster_viz_path}")
        
        # Create and save bbox visualization
        bbox_col = 'bbox_name' if 'bbox_name' in df_with_umap.columns else 'bbox'
        if bbox_col in df_with_umap.columns:
            fig_bbox = px.scatter(
                df_with_umap, x='umap_x', y='umap_y', color=bbox_col,
                title=f'UMAP 2D Projection by {bbox_col}',
                labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2', bbox_col: 'Bounding Box'}
            )
            bbox_viz_path = os.path.join(viz_dir, 'umap_by_bbox.html')
            fig_bbox.write_html(bbox_viz_path)
            logger.info(f"Bounding box visualization saved to {bbox_viz_path}")

if __name__ == "__main__":
    sys.exit(main()) 