"""
Script for visualizing synapse features using UMAP.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import logging
from pathlib import Path
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from contrastive modules
from contrastive.utils.config import config as contrastive_config


def setup_logging():
    """Set up logging for the visualization script."""
    # Create log directory if it doesn't exist
    os.makedirs(contrastive_config.log_dir, exist_ok=True)
    
    # Set up logging to file and console
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(contrastive_config.log_dir, f"umap_visualization.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_features(features_path):
    """
    Load features from CSV file.
    
    Args:
        features_path (str): Path to the features CSV file
        
    Returns:
        tuple: (features_df, feature_cols, metadata_cols)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading features from {features_path}")
    
    # Load features
    features_df = pd.read_csv(features_path)
    
    # Identify feature columns and metadata columns
    feature_cols = [col for col in features_df.columns if col.startswith('feature_')]
    metadata_cols = [col for col in features_df.columns if not col.startswith('feature_')]
    
    logger.info(f"Loaded {len(features_df)} samples with {len(feature_cols)} features")
    
    return features_df, feature_cols, metadata_cols


def preprocess_features(features_df, feature_cols):
    """
    Preprocess features for UMAP.
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
        feature_cols (list): List of feature column names
        
    Returns:
        tuple: (X, scaler)
    """
    logger = logging.getLogger(__name__)
    
    # Extract features
    X = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Preprocessed features: shape={X_scaled.shape}, range=[{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    
    return X_scaled, scaler


def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    Apply UMAP to the features.
    
    Args:
        X (np.ndarray): Preprocessed features
        n_components (int): Number of components for UMAP
        n_neighbors (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance for UMAP
        metric (str): Metric for UMAP
        
    Returns:
        tuple: (umap_embeddings, umap_model)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Applying UMAP with n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    # Apply UMAP
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    
    umap_embeddings = umap_model.fit_transform(X)
    
    logger.info(f"UMAP embeddings: shape={umap_embeddings.shape}, range=[{umap_embeddings.min():.3f}, {umap_embeddings.max():.3f}]")
    
    return umap_embeddings, umap_model


def cluster_features(X, method='kmeans', n_clusters=5):
    """
    Cluster features using KMeans or DBSCAN.
    
    Args:
        X (np.ndarray): Preprocessed features
        method (str): Clustering method ('kmeans' or 'dbscan')
        n_clusters (int): Number of clusters for KMeans
        
    Returns:
        tuple: (cluster_labels, cluster_model)
    """
    logger = logging.getLogger(__name__)
    
    if method == 'kmeans':
        logger.info(f"Applying KMeans clustering with n_clusters={n_clusters}")
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = cluster_model.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        logger.info(f"KMeans silhouette score: {silhouette_avg:.3f}")
        
    elif method == 'dbscan':
        logger.info("Applying DBSCAN clustering")
        cluster_model = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = cluster_model.fit_predict(X)
        
        # Count number of clusters (excluding noise)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(f"DBSCAN found {n_clusters} clusters")
        
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    return cluster_labels, cluster_model


def visualize_umap(umap_embeddings, features_df, metadata_cols, output_dir, title="UMAP Visualization"):
    """
    Visualize UMAP embeddings.
    
    Args:
        umap_embeddings (np.ndarray): UMAP embeddings
        features_df (pd.DataFrame): DataFrame with features and metadata
        metadata_cols (list): List of metadata column names
        output_dir (str): Output directory
        title (str): Plot title
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating UMAP visualizations in {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame with UMAP embeddings and metadata
    umap_df = pd.DataFrame(
        umap_embeddings,
        columns=[f'UMAP{i+1}' for i in range(umap_embeddings.shape[1])]
    )
    
    # Add metadata columns
    for col in metadata_cols:
        umap_df[col] = features_df[col].values
    
    # Save UMAP embeddings
    umap_df.to_csv(os.path.join(output_dir, 'umap_embeddings.csv'), index=False)
    
    # Create 2D visualization
    if umap_embeddings.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        
        # Plot UMAP embeddings
        scatter = plt.scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            alpha=0.7,
            s=50
        )
        
        plt.title(title)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        
        # Add colorbar if there are clusters
        if 'cluster' in umap_df.columns:
            scatter = plt.scatter(
                umap_embeddings[:, 0],
                umap_embeddings[:, 1],
                c=umap_df['cluster'],
                cmap='viridis',
                alpha=0.7,
                s=50
            )
            plt.colorbar(scatter, label='Cluster')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_2d.png'), dpi=300)
        plt.close()
    
    # Create 3D visualization if available
    if umap_embeddings.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot UMAP embeddings
        scatter = ax.scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            umap_embeddings[:, 2],
            alpha=0.7,
            s=50
        )
        
        # Add colorbar if there are clusters
        if 'cluster' in umap_df.columns:
            scatter = ax.scatter(
                umap_embeddings[:, 0],
                umap_embeddings[:, 1],
                umap_embeddings[:, 2],
                c=umap_df['cluster'],
                cmap='viridis',
                alpha=0.7,
                s=50
            )
            plt.colorbar(scatter, label='Cluster')
        
        ax.set_title(title)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_3d.png'), dpi=300)
        plt.close()
    
    # Create visualizations for each metadata column
    for col in metadata_cols:
        if col == 'cluster':
            continue  # Skip cluster column as it's already used for coloring
        
        # Skip non-categorical columns
        if not pd.api.types.is_categorical_dtype(features_df[col]) and not pd.api.types.is_object_dtype(features_df[col]):
            continue
        
        plt.figure(figsize=(10, 8))
        
        # Plot UMAP embeddings colored by metadata
        scatter = plt.scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            c=pd.Categorical(features_df[col]).codes,
            cmap='tab20',
            alpha=0.7,
            s=50
        )
        
        plt.title(f"{title} - Colored by {col}")
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i/len(pd.unique(features_df[col]))), 
                      label=cat, markersize=10)
            for i, cat in enumerate(pd.unique(features_df[col]))
        ]
        plt.legend(handles=legend_elements, title=col, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'umap_{col}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"UMAP visualizations saved to {output_dir}")


def main():
    """Main function for UMAP visualization."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="UMAP visualization for synapse features")
    parser.add_argument("--features", type=str, required=True, help="Path to features CSV file")
    parser.add_argument("--output", type=str, default="contrastive/visualization/umap", help="Output directory")
    parser.add_argument("--n_components", type=int, default=2, help="Number of UMAP components")
    parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance for UMAP")
    parser.add_argument("--cluster", action="store_true", help="Apply clustering")
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "dbscan"], help="Clustering method")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters for KMeans")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting UMAP visualization")
    
    # Load features
    features_df, feature_cols, metadata_cols = load_features(args.features)
    
    # Preprocess features
    X, scaler = preprocess_features(features_df, feature_cols)
    
    # Apply UMAP
    umap_embeddings, umap_model = apply_umap(
        X,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )
    
    # Apply clustering if requested
    if args.cluster:
        cluster_labels, cluster_model = cluster_features(
            X,
            method=args.cluster_method,
            n_clusters=args.n_clusters
        )
        
        # Add cluster labels to features DataFrame
        features_df['cluster'] = cluster_labels
    
    # Visualize UMAP
    visualize_umap(
        umap_embeddings,
        features_df,
        metadata_cols,
        args.output,
        title="Synapse Features UMAP Visualization"
    )
    
    logger.info("UMAP visualization completed")


if __name__ == "__main__":
    main() 