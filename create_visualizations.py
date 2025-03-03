#!/usr/bin/env python
"""
Create Visualizations Script

This script creates the visualizations requested by the user:
1. UMAP in 2D colored by bbox number for feature extraction
2. Comparison of 2 UMAP projections (one colored by bbox number and one colored by cluster)

The script loads the clustered data from the output directory and generates the visualizations.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VisualizationScript')

def parse_args():
    parser = argparse.ArgumentParser(description='Create visualizations from clustered data')
    parser.add_argument('--clustered_data', type=str, default='outputs/main_results/clustering/clustered_data.csv',
                      help='Path to the clustered data CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs/main_results/visualizations',
                      help='Directory to save the visualizations')
    return parser.parse_args()

def create_visualizations(clustered_data_path, output_dir):
    """Create visualizations from the clustered data"""
    logger.info("Creating visualizations...")

    # Load clustered data
    logger.info(f"Loading clustered data from {clustered_data_path}")
    if os.path.exists(clustered_data_path):
        clustered_df = pd.read_csv(clustered_data_path)
        logger.info(f"Loaded clustered data with {len(clustered_df)} samples")
        
        # Print DataFrame info for debugging
        logger.info("DataFrame columns: %s", clustered_df.columns.tolist())
        logger.info("DataFrame shape: %s", str(clustered_df.shape))
        logger.info("DataFrame head:\n%s", clustered_df.head().to_string())
        
        # Check if required columns exist
        required_columns = ['umap_x', 'umap_y', 'bbox_name', 'cluster']
        missing_columns = [col for col in required_columns if col not in clustered_df.columns]
        if missing_columns:
            logger.error("Missing required columns: %s", missing_columns)
            
            # If 'bbox_name' is missing but 'bbox' exists, rename it
            if 'bbox_name' in missing_columns and 'bbox' in clustered_df.columns:
                logger.info("Found 'bbox' column, renaming to 'bbox_name'")
                clustered_df = clustered_df.rename(columns={'bbox': 'bbox_name'})
                missing_columns.remove('bbox_name')
            
            # Check again after potential renaming
            if missing_columns:
                logger.error("Still missing required columns: %s", missing_columns)
                return
    else:
        logger.error(f"Clustered data file not found at {clustered_data_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    # Check if the DataFrame has UMAP coordinates
    if 'umap_x' in clustered_df.columns and 'umap_y' in clustered_df.columns:
        logger.info("Found UMAP coordinates in the DataFrame")
        
        # Check the values of UMAP coordinates
        logger.info("UMAP x range: %s to %s", 
                    clustered_df['umap_x'].min(), clustered_df['umap_x'].max())
        logger.info("UMAP y range: %s to %s", 
                    clustered_df['umap_y'].min(), clustered_df['umap_y'].max())
        
        # 1. Create feature visualization: UMAP in 2D colored by bbox number
        logger.info("Creating feature visualization: UMAP 2D colored by bbox number...")
        
        # Get unique bboxes
        unique_bboxes = clustered_df['bbox_name'].unique()
        logger.info("Found %d unique bounding boxes: %s", 
                    len(unique_bboxes), unique_bboxes)
        
        try:
            # Create matplotlib figure for feature visualization
            plt.figure(figsize=(12, 10))
            for bbox in unique_bboxes:
                subset = clustered_df[clustered_df['bbox_name'] == bbox]
                logger.info("Plotting %d points for bbox %s", len(subset), bbox)
                plt.scatter(subset['umap_x'], subset['umap_y'], label=bbox, alpha=0.7)
            
            plt.title('UMAP 2D Projection - Features by Bounding Box', fontsize=16)
            plt.xlabel('UMAP 1', fontsize=14)
            plt.ylabel('UMAP 2', fontsize=14)
            plt.legend(title='Bounding Box')
            plt.tight_layout()
            
            # Save the feature visualization
            feature_viz_path = os.path.join(output_dir, 'umap_features_by_bbox.png')
            plt.savefig(feature_viz_path, dpi=300)
            plt.close()
            logger.info(f"Feature visualization saved to {feature_viz_path}")
        except Exception as e:
            logger.error("Error creating feature visualization: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())

        # 2. Create a comparison visualization with two UMAP projections side by side
        logger.info("Creating comparison of UMAP projections (bbox colored vs cluster colored)...")
        
        try:
            # 2c. Create a combined static image for comparison
            plt.figure(figsize=(20, 10))
            
            # Left subplot: colored by bbox
            plt.subplot(1, 2, 1)
            for bbox in unique_bboxes:
                subset = clustered_df[clustered_df['bbox_name'] == bbox]
                plt.scatter(subset['umap_x'], subset['umap_y'], label=bbox, alpha=0.7)
            
            plt.title('UMAP by Bounding Box', fontsize=16)
            plt.xlabel('UMAP 1', fontsize=14)
            plt.ylabel('UMAP 2', fontsize=14)
            plt.legend(title='Bounding Box')
            
            # Right subplot: colored by cluster
            plt.subplot(1, 2, 2)
            unique_clusters = clustered_df['cluster'].unique()
            logger.info("Found %d unique clusters: %s", 
                        len(unique_clusters), unique_clusters)
            for cluster in unique_clusters:
                subset = clustered_df[clustered_df['cluster'] == cluster]
                logger.info("Plotting %d points for cluster %s", len(subset), cluster)
                plt.scatter(subset['umap_x'], subset['umap_y'], label=f'Cluster {cluster}', alpha=0.7)
            
            plt.title('UMAP by Cluster', fontsize=16)
            plt.xlabel('UMAP 1', fontsize=14)
            plt.ylabel('UMAP 2', fontsize=14)
            plt.legend(title='Cluster')
            
            plt.tight_layout()
            
            # Save the comparison visualization
            comparison_viz_path = os.path.join(output_dir, 'umap_comparison.png')
            plt.savefig(comparison_viz_path, dpi=300)
            plt.close()
            logger.info(f"Comparison visualization saved to {comparison_viz_path}")
        except Exception as e:
            logger.error("Error creating comparison visualization: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
        
        try:
            # Create interactive visualizations with Plotly
            # 2a. Create bbox-colored visualization
            fig = px.scatter(
                clustered_df, x='umap_x', y='umap_y', color='bbox_name',
                title='UMAP 2D Projection by Bounding Box',
                labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2', 'bbox_name': 'Bounding Box'}
            )
            
            # Save the bbox-colored visualization
            bbox_viz_path = os.path.join(output_dir, 'umap_by_bbox.html')
            fig.write_html(bbox_viz_path)
            logger.info(f"UMAP by bbox visualization saved to {bbox_viz_path}")
            
            # 2b. Create the cluster-colored visualization
            fig2 = px.scatter(
                clustered_df, x='umap_x', y='umap_y', color='cluster',
                title='UMAP 2D Projection by Cluster',
                labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2', 'cluster': 'Cluster'}
            )
            
            # Save the cluster-colored visualization
            cluster_viz_path = os.path.join(output_dir, 'umap_by_cluster.html')
            fig2.write_html(cluster_viz_path)
            logger.info(f"UMAP by cluster visualization saved to {cluster_viz_path}")
        except Exception as e:
            logger.error("Error creating interactive visualizations: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("All visualizations created successfully!")
    else:
        logger.error("UMAP embeddings not found in clustered DataFrame. Cannot create visualizations.")
        # Print the available columns for debugging
        logger.error(f"Available columns in the DataFrame: {clustered_df.columns.tolist()}")

def main():
    """Main function to run the script"""
    args = parse_args()
    create_visualizations(args.clustered_data, args.output_dir)

if __name__ == "__main__":
    main() 