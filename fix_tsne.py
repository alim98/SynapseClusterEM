"""
Simple script to generate t-SNE plots for a specific run.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set paths
run_folder = "results/run_2025-03-14_12-36-59"
output_dir = os.path.join(run_folder, "structured_visualizations", "1_dimension_reduction")
feature_file = os.path.join(run_folder, "clustering_results", "clustered_features.csv")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load features data
print(f"Loading data from {feature_file}")
features_df = pd.read_csv(feature_file)
print(f"Loaded data with shape: {features_df.shape}")

# Extract feature columns
feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
print(f"Found {len(feature_cols)} feature columns")

# Apply t-SNE with low perplexity (since we have few samples)
print("Applying t-SNE...")
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(features_df[feature_cols].values)

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

print("Done! t-SNE visualizations created in:", output_dir) 