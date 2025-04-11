import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import umap
from typing import Optional, Tuple, List, Dict
import logging

class FeatureVisualizer:
    """Class for visualizing features using UMAP and clustering."""
    
    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            logger: Optional logger instance
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_feature_columns(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract feature columns from DataFrame."""
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        features = df[feature_cols].values
        return features, feature_cols
    
    def perform_umap(self, 
                    features: np.ndarray, 
                    n_components: int = 2,
                    n_neighbors: int = 15,
                    min_dist: float = 0.1) -> np.ndarray:
        """
        Perform UMAP dimensionality reduction.
        
        Args:
            features: Input features array
            n_components: Number of components (2 or 3)
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            
        Returns:
            UMAP embeddings
        """
        self.logger.info(f"Performing UMAP with {n_components} components...")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        embeddings = reducer.fit_transform(features)
        return embeddings
    
    def perform_clustering(self, 
                         features: np.ndarray,
                         method: str = 'kmeans',
                         n_clusters: int = 5) -> np.ndarray:
        """
        Perform clustering on features.
        
        Args:
            features: Input features array
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters for kmeans
            
        Returns:
            Cluster labels
        """
        self.logger.info(f"Performing {method} clustering...")
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        labels = clusterer.fit_predict(features)
        return labels
    
    def plot_umap(self,
                 embeddings: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 bbox_names: Optional[List[str]] = None,
                 title: str = "UMAP Visualization",
                 save_path: Optional[str] = None) -> None:
        """
        Plot UMAP embeddings.
        
        Args:
            embeddings: UMAP embeddings
            labels: Optional cluster labels
            bbox_names: Optional list of bounding box names for coloring
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        if embeddings.shape[1] == 2:
            if bbox_names is not None:
                # Create a color map based on unique bbox names
                unique_bboxes = sorted(set(bbox_names))
                color_map = {bbox: plt.cm.tab20(i/len(unique_bboxes)) 
                           for i, bbox in enumerate(unique_bboxes)}
                colors = [color_map[bbox] for bbox in bbox_names]
                
                scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                                   c=colors, cmap='tab20')
                
                # Add legend
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=color, label=bbox, markersize=10)
                                 for bbox, color in color_map.items()]
                plt.legend(handles=legend_elements, title="Bounding Boxes",
                          bbox_to_anchor=(1.05, 1), loc='upper left')
            elif labels is not None:
                scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                                   c=labels, cmap='tab20')
                plt.colorbar(scatter)
            else:
                plt.scatter(embeddings[:, 0], embeddings[:, 1])
        else:  # 3D
            ax = plt.axes(projection='3d')
            if bbox_names is not None:
                # Create a color map based on unique bbox names
                unique_bboxes = sorted(set(bbox_names))
                color_map = {bbox: plt.cm.tab20(i/len(unique_bboxes)) 
                           for i, bbox in enumerate(unique_bboxes)}
                colors = [color_map[bbox] for bbox in bbox_names]
                
                scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                                   c=colors, cmap='tab20')
                
                # Add legend
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=color, label=bbox, markersize=10)
                                 for bbox, color in color_map.items()]
                plt.legend(handles=legend_elements, title="Bounding Boxes",
                          bbox_to_anchor=(1.05, 1), loc='upper left')
            elif labels is not None:
                scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                                   c=labels, cmap='tab20')
                plt.colorbar(scatter)
            else:
                ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2])
        
        plt.title(title)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            self.logger.info(f"Saved UMAP plot to {save_path}")
        plt.close()
    
    def compare_models(self,
                      base_features: np.ndarray,
                      contrastive_features: np.ndarray,
                      labels: Optional[np.ndarray] = None,
                      bbox_names: Optional[List[str]] = None,
                      save_dir: Optional[str] = None) -> None:
        """
        Compare UMAP visualizations between base and contrastive models.
        
        Args:
            base_features: Features from base model
            contrastive_features: Features from contrastive model
            labels: Optional cluster labels
            bbox_names: Optional list of bounding box names for coloring
            save_dir: Directory to save comparison plots
        """
        # Perform UMAP on both feature sets
        base_embeddings = self.perform_umap(base_features)
        contrastive_embeddings = self.perform_umap(contrastive_features)
        
        # Create comparison plots
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Plot base model
            self.plot_umap(
                base_embeddings,
                labels,
                bbox_names,
                "Base Model UMAP",
                os.path.join(save_dir, "base_model_umap.png")
            )
            
            # Plot contrastive model
            self.plot_umap(
                contrastive_embeddings,
                labels,
                bbox_names,
                "Contrastive Model UMAP",
                os.path.join(save_dir, "contrastive_model_umap.png")
            )
            
            # Plot side by side
            plt.figure(figsize=(20, 8))
            
            # Create color map based on bbox names if available
            if bbox_names is not None:
                unique_bboxes = sorted(set(bbox_names))
                color_map = {bbox: plt.cm.tab20(i/len(unique_bboxes)) 
                           for i, bbox in enumerate(unique_bboxes)}
                colors = [color_map[bbox] for bbox in bbox_names]
            
            plt.subplot(121)
            if bbox_names is not None:
                scatter = plt.scatter(base_embeddings[:, 0], base_embeddings[:, 1],
                                   c=colors, cmap='tab20')
            elif labels is not None:
                scatter = plt.scatter(base_embeddings[:, 0], base_embeddings[:, 1],
                                   c=labels, cmap='tab20')
                plt.colorbar(scatter)
            else:
                plt.scatter(base_embeddings[:, 0], base_embeddings[:, 1])
            plt.title("Base Model")
            
            plt.subplot(122)
            if bbox_names is not None:
                scatter = plt.scatter(contrastive_embeddings[:, 0], contrastive_embeddings[:, 1],
                                   c=colors, cmap='tab20')
            elif labels is not None:
                scatter = plt.scatter(contrastive_embeddings[:, 0], contrastive_embeddings[:, 1],
                                   c=labels, cmap='tab20')
                plt.colorbar(scatter)
            else:
                plt.scatter(contrastive_embeddings[:, 0], contrastive_embeddings[:, 1])
            plt.title("Contrastive Model")
            
            # Add legend if using bbox names
            if bbox_names is not None:
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=color, label=bbox, markersize=10)
                                 for bbox, color in color_map.items()]
                plt.legend(handles=legend_elements, title="Bounding Boxes",
                          bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.savefig(os.path.join(save_dir, "model_comparison.png"), bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved comparison plots to {save_dir}")
    
    def analyze_features(self,
                        df: pd.DataFrame,
                        labels: Optional[np.ndarray] = None,
                        save_dir: Optional[str] = None) -> Dict:
        """
        Perform comprehensive feature analysis.
        
        Args:
            df: DataFrame containing features
            labels: Optional cluster labels
            save_dir: Directory to save analysis results
            
        Returns:
            Dictionary containing analysis results
        """
        features, feature_cols = self.extract_feature_columns(df)
        
        # Get bbox names if available
        bbox_names = df['bbox_name'].tolist() if 'bbox_name' in df.columns else None
        
        # Perform UMAP
        embeddings_2d = self.perform_umap(features, n_components=2)
        embeddings_3d = self.perform_umap(features, n_components=3)
        
        # Perform clustering if labels not provided
        if labels is None:
            labels = self.perform_clustering(features)
        
        results = {
            'embeddings_2d': embeddings_2d,
            'embeddings_3d': embeddings_3d,
            'labels': labels,
            'feature_cols': feature_cols,
            'bbox_names': bbox_names
        }
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save 2D UMAP
            self.plot_umap(
                embeddings_2d,
                labels,
                bbox_names,
                "2D UMAP Visualization",
                os.path.join(save_dir, "umap_2d.png")
            )
            
            # Save 3D UMAP
            self.plot_umap(
                embeddings_3d,
                labels,
                bbox_names,
                "3D UMAP Visualization",
                os.path.join(save_dir, "umap_3d.png")
            )
            
            # Save cluster statistics
            if labels is not None:
                # Get unique clusters (excluding noise points -1)
                unique_clusters = np.unique(labels[labels >= 0])
                cluster_counts = np.bincount(labels[labels >= 0])
                
                # Create DataFrame with proper alignment
                cluster_stats = pd.DataFrame({
                    'cluster': unique_clusters,
                    'count': cluster_counts[unique_clusters]
                })
                
                # Add total count
                total_count = len(labels)
                noise_count = np.sum(labels == -1) if -1 in labels else 0
                
                cluster_stats = pd.concat([
                    pd.DataFrame({
                        'cluster': ['total', 'noise'],
                        'count': [total_count, noise_count]
                    }),
                    cluster_stats
                ])
                
                cluster_stats.to_csv(os.path.join(save_dir, "cluster_statistics.csv"), index=False)
            
            self.logger.info(f"Saved analysis results to {save_dir}")
        
        return results 