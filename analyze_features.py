#!/usr/bin/env python3
"""
Run clustering analysis on the specified layer20 features.
"""

from temp import ClusteringAnalyzer

# Path to the features directory
features_path = r"C:\Users\alim9\Documents\codes\synapse2\results\run_20250314_133152\run_2025-03-14_13-31-52\features_layer20_seg10_alpha1.0"

def main():
    # Create analyzer
    analyzer = ClusteringAnalyzer(
        features_path=features_path,
        output_dir=None  # Use default timestamped directory
    )
    
    # Apply UMAP for visualization
    analyzer.apply_umap(
        n_neighbors=15,  # Standard parameter
        min_dist=0.1,    # Standard parameter
        n_components=2   # 2D projection
    )
    
    # Apply t-SNE for visualization
    analyzer.apply_tsne(
        perplexity=5,    # Lower perplexity works better for small datasets
        n_components=2   # 2D projection
    )
    
    # Apply PCA for dimensionality reduction
    analyzer.apply_pca()
    
    # Try different clustering methods
    
    # 1. KMeans with 3 clusters
    analyzer.apply_kmeans(n_clusters=3)
    
    # Plot the results
    analyzer.plot_clusters(
        use_tsne=True,
        use_umap=True,
        use_pca=True,
        projection='2d'
    )
    
    # Save the results
    analyzer.save_results()
    
    print("\nTrying DBSCAN clustering...")
    
    # 2. DBSCAN with different parameters
    analyzer.apply_dbscan(eps=0.5, min_samples=2)
    
    # Plot the results again
    analyzer.plot_clusters(
        use_tsne=True,
        use_umap=True,
        use_pca=False,
        projection='2d'
    )
    
    # Save the results
    analyzer.save_results()
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 