import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from typing import Tuple, Dict, Any
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

def perform_clustering(features_df: pd.DataFrame, 
                      method: str = 'kmeans',
                      n_clusters: int = 10,
                      **kwargs) -> Tuple[pd.DataFrame, Any]:
    """
    Perform clustering on the feature data.
    
    Args:
        features_df: DataFrame containing features
        method: Clustering method ('kmeans' or 'dbscan')
        n_clusters: Number of clusters for KMeans
        **kwargs: Additional arguments for clustering algorithms
        
    Returns:
        Tuple of (DataFrame with cluster assignments, clustering model)
    """
    feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
    features = features_df[feature_cols].values
    features_scaled = StandardScaler().fit_transform(features)
    
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
    elif method == 'dbscan':
        model = DBSCAN(**kwargs)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
        
    features_df['cluster'] = model.fit_predict(features_scaled)
    return features_df, model

def compute_embeddings(features_df: pd.DataFrame,
                      method: str = 'umap',
                      n_components: int = 2,
                      **kwargs) -> np.ndarray:
    """
    Compute low-dimensional embeddings of the features.
    
    Args:
        features_df: DataFrame containing features
        method: Dimensionality reduction method ('umap' or 'tsne')
        n_components: Number of dimensions in the embedding
        **kwargs: Additional arguments for the embedding method
        
    Returns:
        Array containing the embeddings
    """
    feature_cols = [col for col in features_df.columns if col.startswith('feat_')]
    features = features_df[feature_cols].values
    features_scaled = StandardScaler().fit_transform(features)
    
    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, **kwargs)
    else:
        raise ValueError(f"Unsupported embedding method: {method}")
        
    return reducer.fit_transform(features_scaled)

def analyze_clusters(features_df: pd.DataFrame,
                    model: Any,
                    feature_cols: list,
                    output_dir: Path) -> None:
    """
    Analyze clusters and save results.
    
    Args:
        features_df: DataFrame containing features and cluster assignments
        model: Fitted clustering model
        feature_cols: List of feature column names
        output_dir: Directory to save analysis results
    """
    # Compute centroid distances if applicable
    if hasattr(model, 'cluster_centers_'):
        centroids = model.cluster_centers_
    else:
        # For DBSCAN, compute mean of each cluster
        unique_clusters = np.unique(features_df['cluster'])
        centroids = []
        for cluster in unique_clusters:
            if cluster == -1:  # Skip noise points
                continue
            cluster_points = features_df[features_df['cluster'] == cluster][feature_cols].values
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)

    # Compute and save centroid distances
    centroid_distances = squareform(pdist(centroids, 'euclidean'))
    pd.DataFrame(centroid_distances).to_csv(output_dir / 'centroid_distances.csv')

def create_cluster_visualization(features_df: pd.DataFrame,
                               embeddings: np.ndarray,
                               color_by: str = 'cluster',
                               title: str = '',
                               color_mapping: Dict = None) -> go.Figure:
    """
    Create visualization of clusters using plotly.
    
    Args:
        features_df: DataFrame containing features and metadata
        embeddings: Low-dimensional embeddings of the features
        color_by: Column to use for coloring points
        title: Plot title
        color_mapping: Optional mapping of values to colors
        
    Returns:
        Plotly figure object
    """
    df_plot = features_df.copy()
    df_plot['x'] = embeddings[:, 0]
    df_plot['y'] = embeddings[:, 1]
    
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color=color_by,
        color_discrete_map=color_mapping,
        title=title
    )
    
    fig.update_traces(marker=dict(size=4))
    return fig

def save_cluster_visualizations(features_df: pd.DataFrame,
                               embeddings: np.ndarray,
                               output_dir: Path,
                               color_mapping: Dict = None) -> None:
    """
    Create and save cluster visualizations.
    
    Args:
        features_df: DataFrame containing features and metadata
        embeddings: Low-dimensional embeddings of the features
        output_dir: Directory to save visualizations
        color_mapping: Optional mapping of values to colors
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    cluster_fig = create_cluster_visualization(
        features_df,
        embeddings,
        color_by='cluster',
        title='Clusters',
        color_mapping=None
    )
    
    bbox_fig = create_cluster_visualization(
        features_df,
        embeddings,
        color_by='bbox_name',
        title='Bounding Boxes',
        color_mapping=color_mapping
    )
    
    # Save visualizations
    cluster_fig.write_image(output_dir / 'clusters.png')
    bbox_fig.write_image(output_dir / 'bboxes.png')
    
    if embeddings.shape[1] == 3:
        # Create 3D output directory
        threed_dir = output_dir / '3d'
        threed_dir.mkdir(parents=True, exist_ok=True)
        
        fig_3d = px.scatter_3d(
            features_df,
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            color='cluster',
            title='3D Visualization'
        )
        fig_3d.write_html(threed_dir / 'visualization_3d.html') 