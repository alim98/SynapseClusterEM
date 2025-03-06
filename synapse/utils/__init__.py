"""
Utility functions and helpers.
"""

from synapse.utils.config import SynapseConfig
from synapse.utils.clusterhelper import (
    load_and_cluster_features,
    find_random_samples_in_clusters,
    get_center_slice,
    plot_synapse_samples,
    find_closest_samples_in_clusters,
    apply_tsne,
    plot_tsne,
    save_tsne_plots,
    save_cluster_samples
)

# Create a singleton config instance for easy access
config = SynapseConfig()

__all__ = [
    'config', 
    'SynapseConfig',
    'load_and_cluster_features',
    'find_random_samples_in_clusters',
    'get_center_slice',
    'plot_synapse_samples',
    'find_closest_samples_in_clusters',
    'apply_tsne',
    'plot_tsne',
    'save_tsne_plots',
    'save_cluster_samples'
] 