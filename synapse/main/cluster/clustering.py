"""
Synapse Analysis Pipeline - Clustering Module

This module provides functionality for clustering feature data extracted from synapse images.
It can be used independently to process feature CSV files and add cluster assignments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import umap
import seaborn as sns
from pathlib import Path

def perform_clustering(input_csv_path, output_dir=None, n_clusters=10, algorithm='KMeans', 
                       feature_prefix='feat_', perplexity=30, random_state=42, 
                       dbscan_eps=0.5, dbscan_min_samples=5):
    """
    Read a CSV file with extracted features, perform clustering, and save the results.
        
        Args:
        input_csv_path: Path to the input CSV file with extracted features
        output_dir: Directory to save output files (defaults to same directory as input)
        n_clusters: Number of clusters for KMeans
        algorithm: Clustering algorithm to use ('KMeans' or 'DBSCAN')
        feature_prefix: Prefix of feature column names
        perplexity: Perplexity parameter for t-SNE
        random_state: Random state for reproducibility
        dbscan_eps: Epsilon parameter for DBSCAN
        dbscan_min_samples: Minimum samples parameter for DBSCAN
        
    Returns:
        Path to the saved clustered features CSV file
    """
    print(f"Reading features from {input_csv_path}")
    
    # Read features
    features_df = pd.read_csv(input_csv_path)
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_csv_path)
        os.makedirs(output_dir, exist_ok=True)
        
    # Extract feature columns
    feature_cols = [col for col in features_df.columns if col.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(f"No feature columns found with prefix '{feature_prefix}'")
    
    print(f"Found {len(feature_cols)} feature columns")
    features = features_df[feature_cols].values
    
    # Check if we have enough samples for clustering
    if len(features) < n_clusters:
        print(f"Warning: n_samples ({len(features)}) < n_clusters ({n_clusters}), reducing n_clusters to {len(features)//2}")
        n_clusters = max(2, len(features)//2)
    
    # Scale features
    print("Scaling features...")
    features_scaled = StandardScaler().fit_transform(features)
    
    # Apply clustering
    print(f"Applying {algorithm} clustering...")
    if algorithm == 'KMeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        features_df['cluster'] = clusterer.fit_predict(features_scaled)
    else:  # DBSCAN
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        features_df['cluster'] = clusterer.fit_predict(features_scaled)
        
        # DBSCAN assigns -1 to noise points, remap for better visualization
        if -1 in features_df['cluster'].values:
            print(f"Found {(features_df['cluster'] == -1).sum()} noise points (cluster -1)")
            features_df.loc[features_df['cluster'] == -1, 'cluster'] = 999
    
    # Apply dimensionality reduction for visualization
    
    # 1. UMAP
    print("Computing UMAP embedding...")
    reducer = umap.UMAP(random_state=random_state)
    umap_results = reducer.fit_transform(features_scaled)
    features_df['umap_x'] = umap_results[:, 0]
    features_df['umap_y'] = umap_results[:, 1]
    
    # 2. t-SNE (2D)
    print("Computing t-SNE 2D embedding...")
    # Adjust perplexity if needed
    effective_perplexity = min(perplexity, len(features) - 1)
    if effective_perplexity != perplexity:
        print(f"Warning: n_samples ({len(features)}) <= perplexity ({perplexity}), reducing perplexity to {effective_perplexity}")
    
    tsne_2d = TSNE(n_components=2, perplexity=effective_perplexity, random_state=random_state)
    tsne_results_2d = tsne_2d.fit_transform(features_scaled)
    features_df['tsne_2d_x'] = tsne_results_2d[:, 0]
    features_df['tsne_2d_y'] = tsne_results_2d[:, 1]
    
    # 3. t-SNE (3D) if we have enough samples
    if len(features) > 4:
        print("Computing t-SNE 3D embedding...")
        tsne_3d = TSNE(n_components=3, perplexity=effective_perplexity, random_state=random_state)
        tsne_results_3d = tsne_3d.fit_transform(features_scaled)
        features_df['tsne_3d_x'] = tsne_results_3d[:, 0]
        features_df['tsne_3d_y'] = tsne_results_3d[:, 1]
        features_df['tsne_3d_z'] = tsne_results_3d[:, 2]
    
    # Save results
    output_csv_path = os.path.join(output_dir, "clustered_features.csv")
    features_df.to_csv(output_csv_path, index=False)
    print(f"Clustered features saved to {output_csv_path}")
    
    # Create and save visualizations
    create_clustering_visualizations(features_df, output_dir)
    
    # Find and save cluster samples
    cluster_samples = find_samples_in_clusters(features_df)
    
    # Print cluster sample info
    for cluster_id, indices in cluster_samples.items():
        print(f"Saving {len(indices)} sample visualizations for cluster {cluster_id} (indices: {indices})")
    
    return output_csv_path

def create_clustering_visualizations(features_df, output_dir):
    """
    Create and save visualizations of clustering results.
        
        Args:
        features_df: DataFrame with features and clustering results
        output_dir: Directory to save visualizations
        """
    os.makedirs(output_dir, exist_ok=True)
        
    # 1. UMAP visualization colored by cluster
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_df['umap_x'], 
        features_df['umap_y'],
        c=features_df['cluster'], 
        cmap='tab10', 
        alpha=0.8, 
        s=50
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title('UMAP Visualization of Clusters')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_clusters.png'), dpi=300)
    plt.close()
    
    # 2. t-SNE visualization colored by cluster
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_df['tsne_2d_x'], 
        features_df['tsne_2d_y'],
        c=features_df['cluster'], 
        cmap='tab10', 
        alpha=0.8, 
        s=50
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_clusters.png'), dpi=300)
    plt.close()
    
    # 3. If we have bounding box information, create a visualization by bbox
    if 'bbox_name' in features_df.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='umap_x', 
            y='umap_y',
            hue='bbox_name', 
            palette='Set1',
            alpha=0.8, 
            s=50,
            data=features_df
        )
        plt.title('UMAP Visualization by Bounding Box')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend(title='Bounding Box')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_bbox.png'), dpi=300)
        plt.close()
    
    # 4. Create cluster distribution analysis
    if 'bbox_name' in features_df.columns:
        cluster_bbox_counts = pd.crosstab(
            features_df['cluster'], 
            features_df['bbox_name']
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cluster_bbox_counts, 
            annot=True, 
            cmap='YlGnBu',
            fmt='d',
            cbar_kws={'label': 'Count'}
        )
        plt.title('Distribution of Bounding Boxes in Each Cluster')
        plt.xlabel('Bounding Box')
        plt.ylabel('Cluster')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_bbox_distribution.png'), dpi=300)
        plt.close()

def find_samples_in_clusters(features_df, n_samples=4):
    """
    Find sample indices in each cluster for visualization.
    
    Args:
        features_df: DataFrame with features and cluster assignments
        n_samples: Number of samples to select per cluster
        
    Returns:
        dict: Dictionary mapping cluster IDs to lists of sample indices
    """
    if 'cluster' not in features_df.columns:
        print("No cluster information found in features DataFrame")
        return {}
    
    sample_indices = {}
    for cluster_id in features_df['cluster'].unique():
        cluster_samples = features_df[features_df['cluster'] == cluster_id]
        if len(cluster_samples) > 0:
            if len(cluster_samples) <= n_samples:
                # Take all samples
                indices = cluster_samples.index.tolist()
        else:
                # Randomly select n_samples
                indices = np.random.choice(cluster_samples.index, n_samples, replace=False).tolist()
            
                sample_indices[cluster_id] = indices
    
    return sample_indices

def main():
    """
    Main function to use as a command-line tool.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Cluster features from a CSV file')
    parser.add_argument('input_csv', help='Path to input CSV file with features')
    parser.add_argument('--output_dir', help='Directory to save output files')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for KMeans')
    parser.add_argument('--algorithm', choices=['KMeans', 'DBSCAN'], default='KMeans', 
                        help='Clustering algorithm to use')
    parser.add_argument('--feature_prefix', default='feat_', help='Prefix of feature column names')
    parser.add_argument('--perplexity', type=int, default=30, help='Perplexity parameter for t-SNE')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--dbscan_eps', type=float, default=0.5, help='Epsilon parameter for DBSCAN')
    parser.add_argument('--dbscan_min_samples', type=int, default=5, 
                        help='Minimum samples parameter for DBSCAN')
    
    args = parser.parse_args()
    
    perform_clustering(
        args.input_csv,
        args.output_dir,
        args.n_clusters,
        args.algorithm,
        args.feature_prefix,
        args.perplexity,
        args.random_state,
        args.dbscan_eps,
        args.dbscan_min_samples
    )

if __name__ == "__main__":
    main() 