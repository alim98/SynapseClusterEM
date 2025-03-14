#!/usr/bin/env python
"""
Generate t-SNE visualizations for an existing run.
This script loads feature data from a specified run folder and generates
t-SNE visualizations colored by bounding box and cluster.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
import glob
from synapse import config

def find_feature_file(run_folder):
    """Find the feature file in the run folder."""
    # Look for cluster feature files first (they should already have cluster assignments)
    cluster_files = glob.glob(os.path.join(run_folder, "clustering_results", "*clustered*.csv"))
    
    if cluster_files:
        print(f"Found clustered feature file: {cluster_files[0]}")
        return cluster_files[0]
    
    # If no cluster files, look for feature files
    feature_files = glob.glob(os.path.join(run_folder, "features_*", "*.csv"))
    feature_files.extend(glob.glob(os.path.join(run_folder, "features", "*.csv")))
    
    if feature_files:
        print(f"Found feature file: {feature_files[0]}")
        return feature_files[0]
    
    print("No feature files found in run folder.")
    return None

def apply_tsne(data, perplexity=30, n_components=2, random_state=42):
    """Apply t-SNE to reduce dimensionality of data."""
    print(f"Applying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    reduced_data = tsne.fit_transform(data)
    return reduced_data

def generate_tsne_visualizations(features_df, output_dir):
    """
    Generate t-SNE visualizations colored by bbox and cluster.
    
    Args:
        features_df: DataFrame with features and cluster/bbox assignments
        output_dir: Directory to save visualization results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract feature columns (they typically start with 'feat_')
    feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
    
    if not feature_cols:
        print("No feature columns found in DataFrame. Looking for alternative column names...")
        # Try other common naming patterns for feature columns
        for pattern in ['feature_', 'f_', 'embedding_', 'pc_']:
            feature_cols = [col for col in features_df.columns if col.startswith(pattern)]
            if feature_cols:
                print(f"Found feature columns with pattern '{pattern}'")
                break
    
    if not feature_cols:
        # If still no feature columns found, try numerical columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        non_feature_cols = ['bbox_number', 'cluster', 'x', 'y', 'z', 'index', 'umap_1', 'umap_2', 'tsne_1', 'tsne_2']
        feature_cols = [col for col in numeric_cols if col not in non_feature_cols]
        print(f"Using {len(feature_cols)} numeric columns as features")
    
    print(f"Using {len(feature_cols)} feature columns")
    
    # Apply t-SNE
    tsne_result = apply_tsne(features_df[feature_cols].values)
    
    # Add t-SNE coordinates to DataFrame
    features_df['tsne_1'] = tsne_result[:, 0]
    features_df['tsne_2'] = tsne_result[:, 1]
    
    # Create t-SNE plot colored by bbox
    if 'bbox_name' in features_df.columns:
        print("Creating t-SNE plot colored by bounding box...")
        plt.figure(figsize=(10, 8))
        bbox_names = features_df['bbox_name'].unique()
        cmap = plt.cm.get_cmap('tab10', len(bbox_names))
        
        # Create dictionary mapping bbox names to colors
        bbox_to_color = {bbox: cmap(i) for i, bbox in enumerate(bbox_names)}
        
        # Create scatter plot with colors based on bbox
        for bbox in bbox_names:
            subset = features_df[features_df['bbox_name'] == bbox]
            plt.scatter(subset['tsne_1'], subset['tsne_2'], color=bbox_to_color[bbox], 
                        alpha=0.7, label=bbox)
        
        plt.title('t-SNE Projection Colored by Bounding Box')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_names, title="Bounding Box", loc='best')
        plt.tight_layout()
        
        tsne_bbox_path = os.path.join(output_dir, "tsne_bbox_colored.png")
        plt.savefig(tsne_bbox_path)
        plt.close()
        print(f"Saved t-SNE bbox plot to {tsne_bbox_path}")
    else:
        print("Warning: 'bbox_name' column not found in DataFrame, skipping bbox colored plot")
    
    # Create t-SNE plot colored by cluster
    if 'cluster' in features_df.columns:
        print("Creating t-SNE plot colored by cluster...")
        plt.figure(figsize=(10, 8))
        clusters = sorted(features_df['cluster'].unique())
        cluster_cmap = plt.cm.get_cmap('tab10', len(clusters))
        
        # Create scatter plot with colors based on cluster
        for i, cluster in enumerate(clusters):
            subset = features_df[features_df['cluster'] == cluster]
            plt.scatter(subset['tsne_1'], subset['tsne_2'], color=cluster_cmap(i), 
                        alpha=0.7, label=f'Cluster {cluster}')
        
        plt.title('t-SNE Projection Colored by Cluster')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(title="Cluster", loc='best')
        plt.tight_layout()
        
        tsne_cluster_path = os.path.join(output_dir, "tsne_cluster_colored.png")
        plt.savefig(tsne_cluster_path)
        plt.close()
        print(f"Saved t-SNE cluster plot to {tsne_cluster_path}")
    else:
        print("Warning: 'cluster' column not found in DataFrame, skipping cluster colored plot")
    
    return {
        'tsne_bbox': tsne_bbox_path if 'bbox_name' in features_df.columns else None,
        'tsne_cluster': tsne_cluster_path if 'cluster' in features_df.columns else None
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate t-SNE visualizations for an existing run")
    parser.add_argument('run_folder', help='Run folder to analyze (e.g., run_2025-03-14_12-36-59)')
    parser.add_argument('--results-dir', type=str, default="results", 
                      help='Base directory containing run folders')
    parser.add_argument('--perplexity', type=int, default=30,
                      help='Perplexity parameter for t-SNE (default: 30)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for t-SNE visualizations (default: run_folder/structured_visualizations/1_dimension_reduction)')
    
    args = parser.parse_args()
    
    # Construct the full path to the run folder
    run_folder = args.run_folder
    if not os.path.isabs(run_folder):
        if run_folder.startswith("run_"):
            run_folder = os.path.join(args.results_dir, run_folder)
        else:
            run_folder = os.path.join(args.results_dir, f"run_{run_folder}")
    
    if not os.path.exists(run_folder):
        print(f"Error: Run folder '{run_folder}' does not exist.")
        return 1
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(run_folder, "structured_visualizations", "1_dimension_reduction")
    
    # Find feature file
    feature_file = find_feature_file(run_folder)
    if feature_file is None:
        print("Error: No feature file found.")
        return 1
    
    # Load feature data
    print(f"Loading feature data from {feature_file}...")
    features_df = pd.read_csv(feature_file)
    print(f"Loaded feature data with shape: {features_df.shape}")
    
    # Generate t-SNE visualizations
    tsne_paths = generate_tsne_visualizations(features_df, args.output_dir)
    
    # Print summary
    print("\nSuccessfully generated t-SNE visualizations:")
    if tsne_paths['tsne_bbox']:
        print(f"t-SNE colored by bounding box: {tsne_paths['tsne_bbox']}")
    if tsne_paths['tsne_cluster']:
        print(f"t-SNE colored by cluster: {tsne_paths['tsne_cluster']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 