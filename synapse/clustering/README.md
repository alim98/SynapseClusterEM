# Synapse Clustering Module

This module contains the clustering-related functionality for the SynapseClusterEM system.

## Overview

The clustering module provides tools for:

- Loading and clustering feature data extracted from synapse images
- Dimensionality reduction using t-SNE 
- Finding representative samples in clusters
- Visualization of clustering results

## Key Components

- **clusterhelper.py**: Contains the core clustering and visualization functions
- **__init__.py**: Exports the key functions for easy import

## Usage

```python
# Import clustering functions directly from the synapse package
from synapse import (
    load_and_cluster_features,
    apply_tsne,
    save_tsne_plots
)

# Or import from the clustering module
from synapse.clustering import (
    load_and_cluster_features,
    find_random_samples_in_clusters,
    find_closest_samples_in_clusters,
    apply_tsne,
    save_tsne_plots,
    save_cluster_samples
)

# Example usage
features_df, clusterer, feature_cols = load_and_cluster_features(
    csv_path="results/features.csv", 
    n_clusters=10
)

# Apply t-SNE dimensionality reduction
tsne_results_2d = apply_tsne(features_df, feature_cols, 2)
tsne_results_3d = apply_tsne(features_df, feature_cols, 3)

# Generate and save visualizations
save_tsne_plots(
    features_df, 
    tsne_results_2d, 
    tsne_results_3d, 
    clusterer, 
    color_mapping, 
    output_dir
)
``` 