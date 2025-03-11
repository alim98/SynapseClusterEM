"""
Synapse Analysis Package

A framework for 3D synapse data analysis, feature extraction and clustering.
"""

__version__ = '0.1.0'

# Import commonly used components
from synapse.models import Vgg3D, load_model_from_checkpoint
from synapse.data import SynapseDataset, SynapseDataset2, Synapse3DProcessor, SynapseDataLoader
from synapse.utils import config
from synapse.visualization import create_gif_from_volume, visualize_specific_sample, visualize_all_samples_from_bboxes
from synapse.clustering import (
    load_and_cluster_features,
    find_random_samples_in_clusters,
    find_closest_samples_in_clusters,
    apply_tsne,
    save_tsne_plots,
    save_cluster_samples
)

# Export the most commonly used components
__all__ = [
    'Vgg3D', 
    'load_model_from_checkpoint',
    'SynapseDataset',
    'SynapseDataset2',
    'Synapse3DProcessor',
    'SynapseDataLoader',
    'config',
    'create_gif_from_volume',
    'visualize_specific_sample',
    'visualize_all_samples_from_bboxes',
    # Clustering exports
    'load_and_cluster_features',
    'find_random_samples_in_clusters',
    'find_closest_samples_in_clusters',
    'apply_tsne',
    'save_tsne_plots',
    'save_cluster_samples'
] 